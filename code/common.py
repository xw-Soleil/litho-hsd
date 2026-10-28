"""Shared data / model / metric code for both schemes."""
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("DTCO_CACHE", ROOT / "cache"))
RUNS = Path(os.environ.get("DTCO_RUNS", ROOT / "runs"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# bf16 autocast + channels_last on CUDA only (2.8x faster on GB10); output heads stay fp32. DTCO_AMP=0 disables.
AMP = DEV.type == "cuda" and os.environ.get("DTCO_AMP", "1") == "1"


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP)


def fp32():
    return torch.autocast("cuda", enabled=False) if DEV.type == "cuda" else contextlib.nullcontext()


def prep_model(net):
    return net.to(DEV, memory_format=torch.channels_last) if AMP else net.to(DEV)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    elif DEV.type == "mps":
        torch.mps.synchronize()
CLASSES = ["NHS", "HS", "NNHS"]  # label ids 0 / 1 / 2
CID = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------------------- data
def load_meta():
    """meta.csv + a fixed split column: train / val / calib / eval."""
    m = pd.read_csv(CACHE / "meta.csv")
    m["y"] = m.cls.map(CID)
    rng = np.random.RandomState(0)
    split = np.where(m.split == "train", "train", "eval").astype(object)
    # val: 10% of official train, stratified by benchmark x class
    for _, idx in m[m.split == "train"].groupby(["bench", "cls"]).groups.items():
        idx = np.array(idx)
        split[rng.choice(idx, max(1, round(0.1 * len(idx))), replace=False)] = "val"
    # calib: 20% of official-test neighbourhood groups, stratified by benchmark x center type
    te = m[m.split == "test"]
    gtype = te[te.is_center == 1].set_index("group").cls  # HS-group or NHS-group
    for (_, _), g in gtype.groupby([gtype.index.str.split("/").str[0], gtype.values]):
        groups = g.index.values
        pick = set(rng.choice(groups, round(0.2 * len(groups)), replace=False))
        split[(m.split == "test").values & m.group.isin(pick).values] = "calib"
    m["part"] = split
    return m


class Arrays:
    """Memory-mapped cached representations."""

    def __init__(self):
        self.dens = np.load(CACHE / "dens120.npy", mmap_mode="r")
        self.densm = np.load(CACHE / "dens120m.npy", mmap_mode="r") if (CACHE / "dens120m.npy").exists() else None
        self.topo = np.load(CACHE / "sq_topo.npy", mmap_mode="r")
        self.dx = np.load(CACHE / "sq_dx.npy", mmap_mode="r")
        self.dy = np.load(CACHE / "sq_dy.npy", mmap_mode="r")

    def raw(self, rep, idx):
        """uint tensors on device for rows idx (sorted access is much faster on memmaps)."""
        idx = np.asarray(idx)
        if rep.startswith("dens"):
            src = self.densm if rep == "densm" else self.dens
            return (torch.from_numpy(np.ascontiguousarray(src[idx])).to(DEV),)
        return tuple(torch.from_numpy(np.ascontiguousarray(a[idx]).astype(np.int32 if a.dtype == np.uint16 else a.dtype)).to(DEV)
                     for a in (self.topo, self.dx, self.dy))


def d4(raw, rep, mode=None):
    """Dihedral (flip / transpose) augmentation. mode=None -> random per sample, int 0..7 -> fixed."""
    x = list(raw)
    B = x[0].shape[0]
    if mode is None:
        bits = torch.randint(0, 2, (3, B), device=DEV).bool()
    else:
        bits = torch.tensor([(mode >> k) & 1 for k in range(3)], device=DEV).bool()[:, None].expand(3, B)
    hf, vf, tr = bits
    w = lambda m, a, b: torch.where(m.view(-1, *([1] * (a.dim() - 1))), a, b)
    if rep.startswith("dens"):
        d = x[0]
        d = w(hf, d.flip(-1), d)
        d = w(vf, d.flip(-2), d)
        d = w(tr, d.transpose(-1, -2), d)
        return (d,)
    t, dx, dy = x
    t, dx = w(hf, t.flip(-1), t), w(hf, dx.flip(-1), dx)
    t, dy = w(vf, t.flip(-2), t), w(vf, dy.flip(-1), dy)
    t2 = w(tr, t.transpose(-1, -2), t)
    dx2, dy2 = w(tr, dy, dx), w(tr, dx, dy)
    return t2, dx2, dy2


def to_input(raw, rep):
    """uint tensors -> float network input."""
    if rep.startswith("dens"):
        x = raw[0].float().unsqueeze(1) / 255.0
        return x.contiguous(memory_format=torch.channels_last) if AMP else x
    t, dx, dy = raw
    n = t.shape[-1]
    dxf, dyf = dx.float(), dy.float()
    unit = 1200.0 / n
    fx = torch.log2(dxf / unit) / 4  # interval widths, log-scaled
    fy = torch.log2(dyf / unit) / 4
    cx = (torch.cumsum(dxf, -1) - dxf / 2) / 1200 - 0.5  # physical cell-center coordinates
    cy = (torch.cumsum(dyf, -1) - dyf / 2) / 1200 - 0.5
    B = t.shape[0]
    x = torch.stack([t.float(),
                     fx[:, None, :].expand(B, n, n), fy[:, :, None].expand(B, n, n),
                     cx[:, None, :].expand(B, n, n), cy[:, :, None].expand(B, n, n)], 1)
    return x.contiguous(memory_format=torch.channels_last) if AMP else x


IN_CH = {"dens": 1, "densm": 1, "squish": 5}


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                   if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class Backbone(nn.Module):
    """Small ResNet. dens: 120x120 -> stem stride 2; squish: 64x64 -> stem stride 1. Output 256-d."""

    def __init__(self, rep, width=32):
        super().__init__()
        s = 2 if rep.startswith("dens") else 1
        self.stem = nn.Sequential(nn.Conv2d(IN_CH[rep], width, 3, s, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        chans = [width, width * 2, width * 4, width * 8]
        layers, cin = [], width
        for i, c in enumerate(chans):
            layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]
            cin = c
        self.layers = nn.Sequential(*layers)
        self.dim = cin

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1)


class Classifier(nn.Module):
    def __init__(self, rep, n_out, drop=0.2):
        super().__init__()
        self.f = Backbone(rep)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.f.dim, n_out))

    def forward(self, x):
        h = self.f(x)
        with fp32():
            return self.head(h.float())


class Embedder(nn.Module):
    def __init__(self, rep, dim=128):
        super().__init__()
        self.f = Backbone(rep)
        self.proj = nn.Linear(self.f.dim, dim)
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward(self, x):
        h = self.f(x)
        with fp32():
            return F.normalize(self.proj(h.float()), dim=-1)


@torch.no_grad()
def predict(fn, arrays, rep, idx, tta=8, bs=1024):
    """Run fn(input)->tensor over rows idx, averaging over `tta` dihedral views. Returns numpy, aligned to idx."""
    idx = np.asarray(idx)
    order = np.argsort(idx)
    out = [None] * len(idx)
    res = []
    for s in range(0, len(idx), bs):
        raw = arrays.raw(rep, idx[order[s:s + bs]])
        acc = 0
        for mode in range(tta):
            with autocast():
                acc = acc + fn(to_input(d4(raw, rep, mode), rep)).float()
        res.append((acc / tta).float().cpu().numpy())
    res = np.concatenate(res)
    out = np.empty_like(res)
    out[order] = res
    return out


# ----------------------------------------------------------------------------- metrics
def auc(y, s):
    """ROC-AUC via rank statistic (y in {0,1})."""
    y, s = np.asarray(y), np.asarray(s)
    r = pd.Series(s).rank().values
    npos = y.sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def multiclass_report(y, pred, n=3):
    cm = np.zeros((n, n), int)
    np.add.at(cm, (y, pred), 1)
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return dict(macro_f1=float(f1.mean()), bal_acc=float(rec.mean()),
                recall={CLASSES[i]: float(rec[i]) for i in range(n)},
                precision={CLASSES[i]: float(prec[i]) for i in range(n)},
                f1={CLASSES[i]: float(f1[i]) for i in range(n)}, cm=cm.tolist())


def hotspot_report(is_hs, pred_hs):
    """ICCAD-style: Accuracy = hotspot recall, FA = #non-hotspots flagged."""
    is_hs, pred_hs = np.asarray(is_hs).astype(bool), np.asarray(pred_hs).astype(bool)
    return dict(accuracy=float((pred_hs & is_hs).sum() / is_hs.sum()), fa=int((pred_hs & ~is_hs).sum()),
                n_hs=int(is_hs.sum()), n_non=int((~is_hs).sum()))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
