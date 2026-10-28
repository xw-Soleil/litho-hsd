"""Scheme 1: meta-learning with a Prototypical Network (Snell et al., 2017).

Meta-training uses only the official train set (HS / NHS). Pseudo-classes = benchmark x label
(10 classes), so each episode is a small few-shot task:
  * 50%: 2-way "HS vs NHS within one design"  (the real task)
  * 50%: 4-way among random pseudo-classes     (task diversity -> transferable embedding)
At test time a new class (NNHS) is added from K labelled examples without any retraining.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import Arrays, DEV, Embedder, RUNS, autocast, d4, load_meta, predict, prep_model, save_json, to_input

ap = argparse.ArgumentParser()
ap.add_argument("--rep", default="dens", choices=["dens", "densm", "squish"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--holdout", default="", help="benchmark excluded from training (cross-design test)")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--episodes", type=int, default=8, help="episodes per optimisation step")
ap.add_argument("--shot", type=int, default=5)
ap.add_argument("--query", type=int, default=10)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--tta", type=int, default=4)
args = ap.parse_args()
torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
out = RUNS / (f"proto_{args.rep}_s{args.seed}" + (f"_ho{args.holdout}" if args.holdout else ""))
out.mkdir(parents=True, exist_ok=True)

m = load_meta()
A = Arrays()
tr = m.index[(m.part == "train") & (m.bench != args.holdout)].values
raw_tr = A.raw(args.rep, tr)
pseudo = (m.bench.values[tr] + "/" + m.cls.values[tr])
names = sorted(set(pseudo))
members = {n: np.nonzero(pseudo == n)[0] for n in names}  # positions into tr
benches = sorted(set(m.bench.values[tr]))


def sample_episode():
    if rng.rand() < 0.5:
        b = benches[rng.randint(len(benches))]
        cls = [f"{b}/HS", f"{b}/NHS"]
    else:
        cls = list(rng.choice(names, 4, replace=False))
    sup, qry = [], []
    for c in cls:
        pick = rng.choice(members[c], args.shot + args.query, replace=False)
        sup.append(pick[:args.shot])
        qry.append(pick[args.shot:])
    return sup, qry


net = prep_model(Embedder(args.rep))
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

log, t0, run_loss, run_acc = [], time.time(), 0.0, 0.0
for step in range(args.steps):
    eps = [sample_episode() for _ in range(args.episodes)]
    flat = np.concatenate([np.concatenate(s + q) for s, q in eps])
    raw = tuple(r[torch.as_tensor(flat, device=DEV)] for r in raw_tr)
    with autocast():
        z = net(to_input(d4(raw, args.rep), args.rep))
    loss, acc, off = 0.0, 0.0, 0
    for sup, qry in eps:
        n_way = len(sup)
        zs = z[off:off + n_way * args.shot].view(n_way, args.shot, -1)
        off += n_way * args.shot
        zq = z[off:off + n_way * args.query]
        off += n_way * args.query
        protos = F.normalize(zs.mean(1), dim=-1)
        logits = net.log_scale.exp() * zq @ protos.T
        yq = torch.arange(n_way, device=DEV).repeat_interleave(args.query)
        loss = loss + F.cross_entropy(logits, yq)
        acc += (logits.argmax(1) == yq).float().mean().item()
    loss = loss / len(eps)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    run_loss += loss.item()
    run_acc += acc / len(eps)
    if (step + 1) % 100 == 0:
        log.append(dict(step=step + 1, loss=run_loss / 100, episode_acc=run_acc / 100,
                        scale=float(net.log_scale.exp()), sec=time.time() - t0))
        print(log[-1], flush=True)
        run_loss = run_acc = 0.0

torch.save(net.state_dict(), out / "model.pt")
net.eval()
t1 = time.time()
emb = lambda x: net(x)  # TTA averages unit vectors; renormalised below
allidx = m.index[m.part.isin(["train", "val", "calib", "eval"])].values
Z = predict(emb, A, args.rep, allidx, tta=args.tta)
Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
np.savez(out / "emb.npz", idx=allidx, z=Z.astype(np.float16), scale=float(net.log_scale.exp()))
save_json(dict(args=vars(args), log=log, infer_sec=time.time() - t1), out / "log.json")
print("saved", out, f"inference {time.time() - t1:.0f}s")
