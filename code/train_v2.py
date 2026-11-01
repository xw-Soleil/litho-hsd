"""Train frozen-protocol v2 models on a Slurm-allocated CUDA device.

This script never selects a model or epoch using tune/test labels. Binary and
episodic models use only official-training rows assigned to ``train``. The
supervised three-class reference additionally uses ``support``. All active rows
are exported for a separate evaluator; exporting is inference, not adaptation.

Example:
  python train_v2.py --mode binary --rep densm --seed 0 --protocol experiments/v2

Use --validate-only for CPU-side protocol checks without training or inference.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (AMP, CACHE, CID, DEV, ROOT, Arrays, Classifier, Embedder,
                    auc, autocast, d4, fp32, predict, prep_model, to_input)


PARTS = {"train", "val", "support", "tune", "test", "excluded"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def idx_sha256(idx):
    return hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temp, path)


def atomic_torch(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(temp, path)


def load_protocol(path, cache=CACHE):
    path = Path(path)
    path = path / "manifest.csv" if path.is_dir() else path
    m = pd.read_csv(path)
    base = pd.read_csv(Path(cache) / "meta.csv")
    required = {"row_id", "path", "bench", "cls", "y", "part", "split"}
    if not required.issubset(m.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(m.columns))}")
    m = m.sort_values("row_id").reset_index(drop=True)
    if len(m) != len(base) or not np.array_equal(m.row_id.to_numpy(), np.arange(len(base))):
        raise ValueError("Manifest must contain every cache row exactly once with its original row_id")
    for col in ("path", "bench", "cls", "split"):
        if not np.array_equal(m[col].to_numpy(), base[col].to_numpy()):
            raise ValueError(f"Manifest/cache mismatch in {col}; do not reuse stale cache indices")
    if not np.array_equal(m.y.to_numpy(), m.cls.map(CID).to_numpy()):
        raise ValueError("Manifest label IDs disagree with NHS=0, HS=1, NNHS=2")
    if m.part.isna().any() or not set(m.part).issubset(PARTS):
        raise ValueError("Manifest has missing/unknown partition names")
    if (m.loc[m.part.isin(["train", "val"]), "split"] != "train").any():
        raise ValueError("train/val must come from the official training set")
    if (m.loc[m.part.isin(["support", "tune", "test"]), "split"] != "test").any():
        raise ValueError("support/tune/test must come from the official test set")
    # group/is_center are structural metadata, never model input.
    for col in ("group", "is_center"):
        if col not in m:
            m[col] = base[col]
    return m, path


def select_rows(m, mode, holdout):
    parts = ["train", "support"] if mode == "sup3" else ["train"]
    tr = m.index[m.part.isin(parts) & (m.bench != holdout)].to_numpy(dtype=np.int64)
    va = m.index[(m.part == "val") & (m.bench != holdout)].to_numpy(dtype=np.int64)
    export = m.index[m.part != "excluded"].to_numpy(dtype=np.int64)
    if not len(tr) or not len(export):
        raise ValueError("Empty training/export split")
    expected = {0, 1, 2} if mode == "sup3" else {0, 1}
    if set(m.y.iloc[tr]) != expected:
        raise ValueError(f"Training labels must be exactly {expected}")
    if holdout and (m.bench.iloc[tr].eq(holdout).any() or m.bench.iloc[va].eq(holdout).any()):
        raise AssertionError("Holdout design entered optimization/validation")
    return tr, va, export


def episode_pools(m, tr):
    frame = m.iloc[tr].copy()
    frame["position"] = np.arange(len(tr))
    # Avoid choosing identical canonical layouts as both support and query.
    # This is a training-episode check in addition to the between-split audit.
    if "dup_key" in frame:
        frame = frame.drop_duplicates(["bench", "cls", "dup_key"])
    members = {f"{b}/{c}": g.position.to_numpy()
               for (b, c), g in frame.groupby(["bench", "cls"])}
    members = {key: val for key, val in members.items() if len(val) >= 2}
    benches = sorted(b for b in set(frame.bench)
                     if f"{b}/NHS" in members and f"{b}/HS" in members)
    if not benches:
        raise ValueError("No design has at least two distinct training layouts in both classes")
    return members, benches


def sample_episode(rng, members, benches, shot, query):
    names = sorted(members)
    if rng.rand() < 0.5:
        b = benches[rng.randint(len(benches))]
        classes = [f"{b}/HS", f"{b}/NHS"]
    else:
        classes = list(rng.choice(names, min(4, len(names)), replace=False))
    sup, qry = [], []
    for c in classes:
        pool = members[c]
        ns = min(shot, max(1, len(pool) // 2))
        nq = min(query, len(pool) - ns)
        picked = rng.choice(pool, ns + nq, replace=False)
        sup.append(picked[:ns])
        qry.append(picked[ns:])
    return sup, qry


def rng_state(rng):
    state = {"numpy": rng.get_state(), "torch": torch.get_rng_state()}
    if DEV.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, rng):
    rng.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if DEV.type == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def validate_model(net, arrays, rep, va, labels, mode, inference_bs):
    if not len(va) or mode == "proto":
        return {}
    net.eval()
    p = predict(lambda x: F.softmax(net(x), -1), arrays, rep, va,
                tta=1, bs=inference_bs)
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite validation predictions")
    y = labels[va]
    result = {"val_nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)).mean())}
    if len(np.unique(y)) == 2:
        result["val_hs_auc"] = auc((y == 1).astype(int), p[:, 1])
    net.train()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["binary", "proto", "sup3"])
    ap.add_argument("--rep", default="densm", choices=["dens", "densm", "squish"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--protocol", type=Path, default=ROOT / "experiments" / "v2" / "manifest.csv")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("DTCO_RUNS", ROOT / "runs" / "v2")))
    ap.add_argument("--out", type=Path, help="Override this single run's output directory")
    ap.add_argument("--steps", type=int, default=None,
                    help="Fixed optimization budget: default binary=12 negative-class passes, proto=1500, sup3=3000")
    ap.add_argument("--epochs", type=int, default=12, help="Used only to resolve binary default steps")
    ap.add_argument("--bs", type=int, default=128, help="Balanced classifier batch size, rounded down to a class multiple")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tta", type=int, choices=[1, 4, 8], default=4)
    ap.add_argument("--inference-bs", type=int, default=512)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    for key in ("epochs", "bs", "episodes", "shot", "query", "inference_bs", "checkpoint_every"):
        if getattr(args, key) <= 0:
            ap.error(f"--{key.replace('_', '-')} must be positive")
    if args.lr <= 0 or (args.steps is not None and args.steps < 2):
        ap.error("lr must be positive and steps must be >=2")
    m, protocol_path = load_protocol(args.protocol)
    if args.holdout and args.holdout not in set(m.bench):
        ap.error("holdout benchmark does not exist")
    tr, va, export = select_rows(m, args.mode, args.holdout)
    labels = m.y.to_numpy(dtype=np.int64)
    nclasses = 3 if args.mode == "sup3" else 2
    actual_bs = args.bs // nclasses * nclasses
    if actual_bs < nclasses:
        ap.error("batch size must be at least the number of classes")
    if args.steps is None:
        args.steps = (args.epochs * max(1, math.ceil(int((labels[tr] == 0).sum()) / (actual_bs // 2)))
                      if args.mode == "binary" else 1500 if args.mode == "proto" else 3000)
    members, benches = episode_pools(m, tr) if args.mode == "proto" else ({}, [])
    name = f"{args.mode}_{args.rep}_s{args.seed}" + (f"_ho{args.holdout}" if args.holdout else "")
    out = args.out or args.runs / name
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["actual_classifier_batch_size"] = actual_bs
    meta = dict(args=config, protocol_sha256=sha256_file(protocol_path),
                train_idx_sha256=idx_sha256(tr), val_idx_sha256=idx_sha256(va),
                export_idx_sha256=idx_sha256(export), cache_meta_sha256=sha256_file(CACHE / "meta.csv"),
                code_sha256={p.name: sha256_file(p) for p in [Path(__file__), Path(__file__).with_name("common.py")]},
                train_counts=m.iloc[tr].groupby(["bench", "cls"]).size().rename("n").reset_index().to_dict("records"),
                n_train=len(tr), n_val=len(va), n_export=len(export),
                episode_pool_sizes={k: len(v) for k, v in members.items()},
                model_selection="fixed final step, no tune/test label-based selection",
                classification_sampling="class-balanced with replacement",
                features="L2-normalized 128-d projection" if args.mode == "proto" else "L2-normalized 256-d backbone",
                torch_version=torch.__version__, numpy_version=np.__version__,
                device=str(DEV), amp_bfloat16=AMP)
    print(json.dumps({"run": name, "steps": args.steps, "n_train": len(tr), "n_val": len(va),
                      "n_export": len(export), "episode_pool_sizes": meta["episode_pool_sizes"],
                      "protocol_sha256": meta["protocol_sha256"]}), flush=True)
    if args.validate_only:
        return
    if DEV.type != "cuda":
        raise RuntimeError("v2 training requires an allocated CUDA GPU; use xlab01 Slurm or --validate-only")
    out.mkdir(parents=True, exist_ok=True)
    # Output paths and runtime-device details may change when resuming on another node.
    fingerprint_config = {k: v for k, v in config.items() if k not in {"runs", "out", "protocol", "validate_only"}}
    fingerprint = hashlib.sha256(json.dumps(dict(config=fingerprint_config,
        protocol=meta["protocol_sha256"], train=meta["train_idx_sha256"],
        cache=meta["cache_meta_sha256"], code=meta["code_sha256"]), sort_keys=True).encode()).hexdigest()
    meta["run_fingerprint"] = fingerprint
    marker = out / "complete.json"
    if marker.exists():
        completed = json.loads(marker.read_text())
        if completed.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Existing completed run has different configuration/protocol/code: {out}")
        for filename, expected in completed["artifacts_sha256"].items():
            if sha256_file(out / filename) != expected:
                raise ValueError(f"Completed artifact was changed/corrupted: {filename}")
        print("SKIP verified completed run", out, flush=True)
        return

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    arrays = Arrays()
    if args.rep == "densm" and arrays.densm is None:
        raise ValueError("dens120m.npy missing; run prep_masked.py before training")
    raw_train = arrays.raw(args.rep, tr)
    y_train = torch.as_tensor(labels[tr], device=DEV)
    class_positions = [torch.nonzero(y_train == c).flatten() for c in range(nclasses)]
    net = prep_model(Embedder(args.rep) if args.mode == "proto" else Classifier(args.rep, nclasses))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
                 if args.steps >= 20 else torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.01, 1 - s / args.steps)))
    history, start, previous_seconds, images_seen = [], 0, 0.0, 0
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location=DEV, weights_only=False)
        if cp["run_fingerprint"] != fingerprint:
            raise ValueError(f"Checkpoint configuration/protocol/code mismatch: {checkpoint}")
        net.load_state_dict(cp["model"])
        opt.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        history, start, previous_seconds, images_seen = cp["history"], cp["step"], cp["train_seconds"], cp["images_seen"]
        restore_rng(cp["rng"], rng)
        print("RESUME", start, "of", args.steps, flush=True)
        del cp
    elif (out / "log.json").exists():
        existing = json.loads((out / "log.json").read_text())
        if existing.get("run_fingerprint") != fingerprint:
            raise ValueError("Existing log belongs to a different run; choose another --out directory")

    t0 = time.monotonic()
    running_loss, running_acc, running_steps = 0.0, 0.0, 0
    net.train()
    for step in range(start, args.steps):
        if args.mode == "proto":
            episodes = [sample_episode(rng, members, benches, args.shot, args.query) for _ in range(args.episodes)]
            flat = np.concatenate([np.concatenate(s + q) for s, q in episodes])
            raw = tuple(r[torch.as_tensor(flat, device=DEV)] for r in raw_train)
            with autocast():
                z = net(to_input(d4(raw, args.rep), args.rep))
            loss, episode_acc, offset = 0.0, 0.0, 0
            for sup, qry in episodes:
                protos = []
                for support_rows in sup:
                    protos.append(z[offset:offset + len(support_rows)].mean(0))
                    offset += len(support_rows)
                protos = F.normalize(torch.stack(protos), dim=-1)
                nq = sum(map(len, qry))
                zq = z[offset:offset + nq]
                offset += nq
                targets = torch.cat([torch.full((len(q),), c, device=DEV, dtype=torch.long) for c, q in enumerate(qry)])
                logits = net.log_scale.exp() * zq @ protos.T
                loss = loss + F.cross_entropy(logits, targets)
                episode_acc += (logits.argmax(1) == targets).float().mean().item()
            loss = loss / len(episodes)
            running_acc += episode_acc / len(episodes)
            images_seen += len(flat)
        else:
            k = actual_bs // nclasses
            batch = torch.cat([c[torch.randint(len(c), (k,), device=DEV)] for c in class_positions])
            raw = tuple(r[batch] for r in raw_train)
            with autocast():
                loss = F.cross_entropy(net(to_input(d4(raw, args.rep), args.rep)), y_train[batch], label_smoothing=0.05)
            images_seen += len(batch)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ also checks gradient finiteness before parameters change.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        running_loss += loss.item()
        running_steps += 1
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            seconds = previous_seconds + time.monotonic() - t0
            entry = dict(step=step + 1, train_loss=running_loss / running_steps,
                         train_seconds=seconds, images_seen=images_seen)
            if args.mode == "proto":
                entry.update(episode_accuracy=running_acc / running_steps, scale=float(net.log_scale.exp().detach()))
            entry.update(validate_model(net, arrays, args.rep, va, labels, args.mode, args.inference_bs))
            history.append(entry)
            print(json.dumps(entry), flush=True)
            atomic_torch(dict(model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                              step=step + 1, history=history, train_seconds=seconds, images_seen=images_seen,
                              rng=rng_state(rng), run_fingerprint=fingerprint), checkpoint)
            atomic_json(meta | dict(log=history, status="training"), out / "log.json")
            running_loss, running_acc, running_steps = 0.0, 0.0, 0

    atomic_torch(net.state_dict(), out / "model.pt")
    net.eval()
    ti = time.monotonic()
    if args.mode == "proto":
        z = predict(net, arrays, args.rep, export, tta=args.tta, bs=args.inference_bs)
    else:
        # One backbone pass per view produces both logits and baseline features.
        def combined(x):
            h = net.f(x).float()
            with fp32():
                prob = F.softmax(net.head(h), -1)
                return torch.cat([prob, F.normalize(h, dim=-1)], dim=1)
        values = predict(combined, arrays, args.rep, export, tta=args.tta, bs=args.inference_bs)
        probs, z = values[:, :nclasses], values[:, nclasses:]
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1, atol=1e-4):
            raise FloatingPointError("Invalid exported class probabilities")
        atomic_npz(out / "pred.npz", idx=export, prob=probs.astype(np.float32))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if not np.isfinite(z).all() or (norms <= 1e-12).any():
        raise FloatingPointError("Invalid/zero embedding vectors")
    z = z / norms
    atomic_npz(out / "emb.npz", idx=export, z=z.astype(np.float32))
    atomic_json(meta | dict(log=history, status="complete", infer_seconds=time.monotonic() - ti,
                           images_seen=images_seen, final_step=args.steps), out / "log.json")
    artifacts = ["model.pt", "emb.npz", "log.json"] + ([] if args.mode == "proto" else ["pred.npz"])
    atomic_json(dict(run_fingerprint=fingerprint, protocol_sha256=meta["protocol_sha256"],
                     artifacts_sha256={p: sha256_file(out / p) for p in artifacts}), marker)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
