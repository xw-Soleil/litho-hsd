"""Which few-shot classifier on the frozen meta-learned embedding works best? (CALIB support, EVAL query)"""
import numpy as np
import torch
import torch.nn.functional as F

from common import DEV, RUNS, load_meta, multiclass_report

m = load_meta()
y, bench, part = m.y.values, m.bench.values, m.part.values
CAL, EV = np.nonzero(part == "calib")[0], np.nonzero(part == "eval")[0]
d = np.load(RUNS / "proto_dens_s0" / "emb.npz")
Z = np.zeros((len(m), d["z"].shape[1]), np.float32)
Z[d["idx"]] = d["z"].astype(np.float32)
benches = sorted(set(bench))
rng = np.random.RandomState(0)


def support(K):
    s = []
    for b in benches:
        for c in range(3):
            pool = CAL[(bench[CAL] == b) & (y[CAL] == c)]
            s.append(rng.choice(pool, min(K, len(pool)), replace=False))
    return np.concatenate(s)


def per_bench(fn, sup):
    pred = np.zeros(len(EV), int)
    for b in benches:
        q = EV[bench[EV] == b]
        sb = sup[bench[sup] == b]
        pred[bench[EV] == b] = fn(Z[sb], y[sb], Z[q])
    return pred


def proto(zs, ys, zq):
    P = np.stack([zs[ys == c].mean(0) for c in range(3)])
    return (zq @ (P / np.linalg.norm(P, axis=1, keepdims=True)).T).argmax(1)


def knn(k):
    def f(zs, ys, zq):
        S = torch.as_tensor(zq, device=DEV) @ torch.as_tensor(zs, device=DEV).T
        idx = S.topk(min(k, len(zs)), dim=1).indices.cpu().numpy()
        votes = np.stack([(ys[idx] == c).sum(1) for c in range(3)], 1)
        return votes.argmax(1)
    return f


def multiproto(n):
    def f(zs, ys, zq):
        P, lab = [], []
        for c in range(3):
            zc = zs[ys == c]
            k = min(n, len(zc))
            cent = zc[rng.choice(len(zc), k, replace=False)]
            for _ in range(10):  # spherical k-means
                a = (zc @ cent.T).argmax(1)
                cent = np.stack([zc[a == j].mean(0) if (a == j).any() else cent[j] for j in range(k)])
                cent /= np.linalg.norm(cent, axis=1, keepdims=True)
            P.append(cent); lab += [c] * k
        return np.array(lab)[(zq @ np.concatenate(P).T).argmax(1)]
    return f


def linear(zs, ys, zq):
    """logistic-regression head on the frozen embedding (class-balanced, like the support set)."""
    X = torch.as_tensor(zs, device=DEV)
    t = torch.as_tensor(ys, device=DEV)
    W = torch.zeros(X.shape[1], 3, device=DEV, requires_grad=True)
    b = torch.zeros(3, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(10 * X @ W + b, t) + 1e-3 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (torch.as_tensor(zq, device=DEV) @ W + b).argmax(1).cpu().numpy()


for K in (20, 100, 400):
    for name, fn in [("prototype", proto), ("kNN k=5", knn(5)), ("kNN k=15", knn(15)),
                     ("multi-proto x4", multiproto(4)), ("multi-proto x8", multiproto(8)), ("linear head", linear)]:
        f1 = [multiclass_report(y[EV], per_bench(fn, support(K)))["macro_f1"] for _ in range(5)]
        print(f"K={K:4d}  {name:15s} macro-F1 {np.mean(f1):.4f} ± {np.std(f1):.4f}", flush=True)
