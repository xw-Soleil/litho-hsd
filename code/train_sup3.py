"""Reference (Option A): fully supervised 3-class CNN trained on official train + CALIB (which contains NNHS).
Upper bound on how well NNHS can be told apart from a single clip. Evaluated on EVAL only.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import Arrays, Classifier, DEV, RUNS, autocast, d4, load_meta, predict, prep_model, save_json, to_input

ap = argparse.ArgumentParser()
ap.add_argument("--rep", default="dens", choices=["dens", "densm", "squish"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=129)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--tta", type=int, default=4)
args = ap.parse_args()
torch.manual_seed(args.seed)
out = RUNS / f"sup3_{args.rep}_s{args.seed}"
out.mkdir(parents=True, exist_ok=True)

m = load_meta()
A = Arrays()
tr = m.index[m.part.isin(["train", "calib"])].values
ev = m.index[m.part == "eval"].values
raw_tr = A.raw(args.rep, tr)
y_tr = torch.tensor(m.y.values[tr], device=DEV)
per_cls = [torch.nonzero(y_tr == c).flatten() for c in range(3)]

net = prep_model(Classifier(args.rep, 3))
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
k = args.bs // 3
log, t0, run = [], time.time(), 0.0
for step in range(args.steps):
    b = torch.cat([c[torch.randint(0, len(c), (k,), device=DEV)] for c in per_cls])
    raw = tuple(r[b] for r in raw_tr)
    with autocast():
        loss = F.cross_entropy(net(to_input(d4(raw, args.rep), args.rep)), y_tr[b], label_smoothing=0.05)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    run += loss.item()
    if (step + 1) % 250 == 0:
        log.append(dict(step=step + 1, loss=run / 250, sec=time.time() - t0))
        print(log[-1], flush=True)
        run = 0.0

torch.save(net.state_dict(), out / "model.pt")
net.eval()
P = predict(lambda x: F.softmax(net(x), -1), A, args.rep, ev, tta=args.tta)
np.savez(out / "pred.npz", idx_eval=ev, prob=P)
save_json(dict(args=vars(args), log=log), out / "log.json")
print("saved", out)
