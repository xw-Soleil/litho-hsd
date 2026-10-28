"""Scheme 2 base model: binary hotspot classifier (HS vs NHS) trained on the official train set only.

Class imbalance (1:13) is handled with class-balanced batches; D4 augmentation in training,
D4 test-time augmentation at inference. Saves p(HS) for val + all official-test images.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import Arrays, Classifier, DEV, RUNS, auc, autocast, d4, load_meta, predict, prep_model, save_json, to_input

ap = argparse.ArgumentParser()
ap.add_argument("--rep", default="dens", choices=["dens", "densm", "squish"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--holdout", default="", help="benchmark excluded from training (cross-design test)")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--tta", type=int, default=4)
args = ap.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
out = RUNS / (f"binary_{args.rep}_s{args.seed}" + (f"_ho{args.holdout}" if args.holdout else ""))
out.mkdir(parents=True, exist_ok=True)

m = load_meta()
A = Arrays()
tr = m.index[(m.part == "train") & (m.bench != args.holdout)].values
va = m.index[m.part == "val"].values
te = m.index[m.split == "test"].values
raw_tr = A.raw(args.rep, tr)  # whole train set lives on the GPU
y_tr = torch.tensor((m.y.values[tr] == 1).astype(np.int64), device=DEV)
pos = torch.nonzero(y_tr == 1).flatten()
neg = torch.nonzero(y_tr == 0).flatten()

net = prep_model(Classifier(args.rep, 2))
steps_per_epoch = len(neg) // (args.bs // 2)
total = steps_per_epoch * args.epochs
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)


def val_metrics():
    net.eval()
    p = predict(lambda x: F.softmax(net(x), -1)[:, 1], A, args.rep, va, tta=1)
    net.train()
    yv = (m.y.values[va] == 1).astype(int)
    return auc(yv, p), float(-np.mean(np.where(yv == 1, np.log(p + 1e-7) / yv.mean(), np.log(1 - p + 1e-7) / (1 - yv.mean())) / 2))


log = []
t0 = time.time()
for ep in range(args.epochs):
    perm = neg[torch.randperm(len(neg), device=DEV)]
    tot = 0.0
    for s in range(steps_per_epoch):
        b = torch.cat([perm[s * args.bs // 2:(s + 1) * args.bs // 2],
                       pos[torch.randint(0, len(pos), (args.bs // 2,), device=DEV)]])
        raw = tuple(r[b] for r in raw_tr)
        with autocast():
            loss = F.cross_entropy(net(to_input(d4(raw, args.rep), args.rep)), y_tr[b], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    va_auc, va_bce = val_metrics()
    log.append(dict(epoch=ep + 1, loss=tot / steps_per_epoch, val_auc=va_auc, val_bal_logloss=va_bce, sec=time.time() - t0))
    print(log[-1], flush=True)

torch.save(net.state_dict(), out / "model.pt")
net.eval()
fn = lambda x: F.softmax(net(x), -1)[:, 1]
t1 = time.time()
p_va = predict(fn, A, args.rep, va, tta=args.tta)
p_te = predict(fn, A, args.rep, te, tta=args.tta)
np.savez(out / "pred.npz", idx_val=va, p_val=p_va, idx_test=te, p_test=p_te)
save_json(dict(args=vars(args), log=log, infer_sec=time.time() - t1), out / "log.json")
print("saved", out, f"inference {time.time() - t1:.0f}s")
