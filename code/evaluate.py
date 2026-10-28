"""Evaluate both schemes on the same EVAL split and draw figures.

  A. binary hotspot detection on the standard ICCAD12 test (centre clips, iccad2-5 merged) vs. Geng et al. ICCAD'20
  B. Scheme 2: binary probability + two thresholds  -> HS / NNHS / NHS
  C. Scheme 1: prototypical network, 0-shot open-set and K-shot NNHS prototypes
  D. Option-A supervised 3-class reference
"""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import CLASSES, DEV, ROOT, RUNS, auc, hotspot_report, load_meta, multiclass_report, save_json

OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PAPER = dict(accuracy=0.9842, fa=2481)  # Geng et al., ICCAD'20, Table 2, ICCAD12 (merged 28nm)
C_HS, C_NHS, C_NN = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": ["PingFang SC", "Arial Unicode MS", "sans-serif"], "font.size": 10,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.titlecolor": INK, "axes.titlesize": 11, "legend.frameon": False})

m = load_meta()
N = len(m)
part, y, bench = m.part.values, m.y.values, m.bench.values
is_center, is_test = m.is_center.values == 1, (m.split == "test").values
iccad12 = is_test & is_center & (bench != "iccad1") & (m.cls.values != "NNHS")  # literature test set
CAL, EV, VA, TR = (np.nonzero(part == p)[0] for p in ("calib", "eval", "val", "train"))
R = {}  # all metrics
ROLES = ["HS", "NHS(中心窗)", "NHS(热点的邻窗)", "NNHS(非热点的邻窗)"]
ROLE = np.where(y == 1, ROLES[0], np.where(y == 2, ROLES[3], np.where(is_center, ROLES[1], ROLES[2])))


def by_role(idx, pred):
    """For each window role: share predicted as NHS / HS / NNHS."""
    out = {}
    for r in ROLES:
        sel = ROLE[idx] == r
        out[r] = {CLASSES[c]: round(float((pred[sel] == c).mean()), 4) for c in range(3)} | {"n": int(sel.sum())}
    return out


def full(idx, vals, fill=np.nan):
    a = np.full((N,) + np.shape(vals)[1:], fill, dtype=np.float32)
    a[idx] = vals
    return a


def binary_block(score, thr):
    """ICCAD-style metrics on the literature test set, at a fixed threshold and at the paper's accuracy."""
    s, hs = score[iccad12], y[iccad12] == 1
    t_match = np.sort(s[hs])[int(np.floor((1 - PAPER["accuracy"]) * hs.sum()))]
    return dict(auc=auc(hs.astype(int), s), at_default=hotspot_report(hs, s >= thr),
                fa_at_paper_accuracy=hotspot_report(hs, s >= t_match),
                per_bench={b: hotspot_report(y[is_test & is_center & (bench == b) & (y < 2)] == 1,
                                             score[is_test & is_center & (bench == b) & (y < 2)] >= thr)
                           for b in sorted(set(bench))})


# ============================================================ A + B: Scheme 2 (binary probability)
P, ENS = {}, {}
for f in sorted(glob.glob(str(RUNS / "binary_*_s*/pred.npz"))):
    if "_ho" in f:
        continue
    d = np.load(f)
    name = Path(f).parent.name
    P[name] = full(np.r_[d["idx_val"], d["idx_test"]], np.r_[d["p_val"], d["p_test"]])
for rep in ("dens", "densm", "squish"):
    runs = [v for k, v in P.items() if k.startswith(f"binary_{rep}_")]
    if runs:
        ENS[rep] = dict(p=np.mean(runs, 0), u=np.std(runs, 0) if len(runs) > 1 else None, n=len(runs))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))

R["A_binary"] = {k: binary_block(v, 0.5) for k, v in P.items()}
R["A_binary"].update({f"ensemble_{rep}(x{e['n']})": binary_block(e["p"], 0.5) for rep, e in ENS.items()})


def three_way(s, lo, hi):
    return np.where(s < lo, 0, np.where(s > hi, 1, 2))


def tune_band(s, yy, grid):
    """Grid-search (lo, hi) maximising macro-F1. cdf trick: counts of each class below each grid value."""
    cdf = np.stack([np.searchsorted(np.sort(s[yy == c]), grid, side="left") for c in range(3)])  # [3,G]
    tot = np.array([(yy == c).sum() for c in range(3)])
    best = (-1, None)
    F1 = np.full((len(grid), len(grid)), np.nan)
    for i, lo in enumerate(grid):
        j = np.arange(i, len(grid))
        below = cdf[:, i][:, None]                                     # pred NHS
        above = tot[:, None] - np.stack([np.searchsorted(np.sort(s[yy == c]), grid[j], side="right") for c in range(3)])
        mid = tot[:, None] - below - above                             # pred NNHS
        pred = np.stack([below.repeat(len(j), 1), above, mid])         # [pred, true, J]
        tp = np.stack([pred[0, 0], pred[1, 1], pred[2, 2]])
        prec = tp / np.maximum(pred.sum(1), 1)
        rec = tp / tot[:, None]
        f1 = (2 * prec * rec / np.maximum(prec + rec, 1e-12)).mean(0)
        F1[i, j] = f1
        k = int(np.argmax(f1))
        if f1[k] > best[0]:
            best = (float(f1[k]), (float(lo), float(grid[j][k])))
    return best, F1


GRID = np.arange(-12, 12.01, 0.25)
R["B_scheme2"] = {}
best_rep = max(ENS, key=lambda r: R["A_binary"][f"ensemble_{r}(x{ENS[r]['n']})"]["auc"]) if ENS else None
for rep, e in ENS.items():
    s = logit(e["p"])
    (f1c, (lo, hi)), F1 = tune_band(s[CAL], y[CAL], GRID)
    res = dict(tuned_on_calib=dict(lo_p=float(1 / (1 + np.exp(-lo))), hi_p=float(1 / (1 + np.exp(-hi))), calib_macro_f1=f1c,
                                   eval=multiclass_report(y[EV], three_way(s[EV], lo, hi)),
                                   by_role=by_role(EV, three_way(s[EV], lo, hi))),
               naive_band_02_08=multiclass_report(y[EV], three_way(s[EV], logit(np.array(0.2)), logit(np.array(0.8)))))
    if e["u"] is not None:  # uncertainty-aware: ensemble disagreement also votes for NNHS
        u = e["u"]
        best = (-1, None)
        for q in np.linspace(0.5, 0.99, 15):
            tau = np.quantile(u[CAL], q)
            keep = u[CAL] <= tau
            s_c = np.where(keep, s[CAL], 0.0)  # forced into the middle band when uncertain
            (f1q, (lo2, hi2)), _ = tune_band(s_c, y[CAL], np.arange(-12, 12.01, 0.5))
            if f1q > best[0]:
                best = (f1q, (tau, lo2, hi2, q))
        tau, lo2, hi2, q = best[1]
        pred = three_way(np.where(u[EV] <= tau, s[EV], 0.0), min(lo2, -1e-3), max(hi2, 1e-3))
        res["uncertainty_aware"] = dict(u_quantile=q, calib_macro_f1=best[0], eval=multiclass_report(y[EV], pred))
    triv = np.where(s[EV] >= 0, 1, 2)  # binary decision, every non-hotspot called NNHS (majority class)
    res["trivial_all_nonHS_as_NNHS"] = multiclass_report(y[EV], triv)
    R["B_scheme2"][rep] = res
    if rep == best_rep:
        S2 = dict(s=s, lo=lo, hi=hi, F1=F1, pred=three_way(s[EV], lo, hi))

# ============================================================ C: Scheme 1 (prototypical network)
R["C_scheme1"] = {}
S1 = None
for f in sorted(glob.glob(str(RUNS / "proto_*_s*/emb.npz"))):
    if "_ho" in f:
        continue
    name = Path(f).parent.name
    d = np.load(f)
    Z = torch.zeros((N, d["z"].shape[1]), device=DEV)
    Z[torch.as_tensor(d["idx"], device=DEV)] = torch.as_tensor(d["z"].astype(np.float32), device=DEV)
    benches = sorted(set(bench))
    proto = {}
    for b in benches:
        for c in (0, 1):
            sel = TR[(bench[TR] == b) & (y[TR] == c)]
            proto[(b, c)] = torch.nn.functional.normalize(Z[sel].mean(0), dim=0)
    sim_nhs, sim_hs, knn = (torch.zeros(N, device=DEV) for _ in range(3))
    for b in benches:
        rows = torch.as_tensor(np.nonzero(bench == b)[0], device=DEV)
        sim_nhs[rows] = Z[rows] @ proto[(b, 0)]
        sim_hs[rows] = Z[rows] @ proto[(b, 1)]
        sup = Z[torch.as_tensor(TR[bench[TR] == b], device=DEV)]
        for s0 in range(0, len(rows), 8192):  # novelty = mean cosine to 10 nearest training clips (same design)
            r = rows[s0:s0 + 8192]
            knn[r] = (Z[r] @ sup.T).topk(10, dim=1).values.mean(1)
    sim_nhs, sim_hs, knn = (t.cpu().numpy() for t in (sim_nhs, sim_hs, knn))
    hs_score = sim_hs - sim_nhs
    res = dict(binary=binary_block(hs_score, 0.0))

    # 0-shot open set: NNHS = "never seen" = far from every training clip; tau from VAL only (no NNHS labels)
    tau = float(np.quantile(knn[VA], 0.05))
    pred0 = np.where(knn[EV] < tau, 2, (hs_score[EV] > 0).astype(int))
    res["zero_shot_openset"] = dict(tau_from_val_q05=tau, eval=multiclass_report(y[EV], pred0),
                                    auroc_nnhs_vs_known=auc((y[EV] == 2).astype(int), -knn[EV]))
    taus = np.quantile(knn[CAL], np.linspace(0.01, 0.99, 99))
    f1s = [multiclass_report(y[CAL], np.where(knn[CAL] < t, 2, (hs_score[CAL] > 0).astype(int)))["macro_f1"] for t in taus]
    t_best = taus[int(np.argmax(f1s))]
    res["zero_shot_openset_tau_tuned_on_calib"] = multiclass_report(
        y[EV], np.where(knn[EV] < t_best, 2, (hs_score[EV] > 0).astype(int)))

    # K-shot: add an NNHS prototype per design from K labelled CALIB examples, no retraining
    rng = np.random.RandomState(0)
    Zc = Z.cpu().numpy()
    cal_nn = {b: CAL[(bench[CAL] == b) & (y[CAL] == 2)] for b in benches}
    ev_b = bench[EV]
    kshot = {}
    for K in (1, 5, 10, 20, 50, 100, 200):
        runs, roles_acc = [], []
        for rep_i in range(30):
            sim_nn = np.zeros(len(EV), np.float32)
            for b in benches:
                sup = rng.choice(cal_nn[b], min(K, len(cal_nn[b])), replace=False)
                pz = Zc[sup].mean(0)
                pz /= np.linalg.norm(pz)
                rows = ev_b == b
                sim_nn[rows] = Zc[EV[rows]] @ pz
            pred = np.argmax(np.stack([sim_nhs[EV], sim_hs[EV], sim_nn]), 0)
            runs.append(multiclass_report(y[EV], pred))
            roles_acc.append(by_role(EV, pred))
        f1 = np.array([r["macro_f1"] for r in runs])
        kshot[K] = dict(macro_f1_mean=float(f1.mean()), macro_f1_std=float(f1.std()),
                        recall_mean={c: float(np.mean([r["recall"][c] for r in runs])) for c in CLASSES},
                        cm_mean=np.mean([r["cm"] for r in runs], 0).round().astype(int).tolist(),
                        by_role_mean={r: {c: round(float(np.mean([ra[r][c] for ra in roles_acc])), 4) for c in CLASSES}
                                      for r in ROLES})
    res["k_shot"] = kshot
    # fair few-shot: K labelled CALIB examples for EVERY class (all prototypes from the test domain)
    cal_c = {(b, c): CAL[(bench[CAL] == b) & (y[CAL] == c)] for b in benches for c in range(3)}
    kshot3 = {}
    for K in (1, 5, 10, 20, 50, 100):
        runs, roles_acc = [], []
        for rep_i in range(30):
            sims = np.zeros((3, len(EV)), np.float32)
            for b in benches:
                rows = ev_b == b
                for c in range(3):
                    pool_ = cal_c[(b, c)]
                    sup = rng.choice(pool_, min(K, len(pool_)), replace=False)
                    pz = Zc[sup].mean(0)
                    sims[c, rows] = Zc[EV[rows]] @ (pz / np.linalg.norm(pz))
            pred = sims.argmax(0)
            runs.append(multiclass_report(y[EV], pred))
            roles_acc.append(by_role(EV, pred))
        f1 = np.array([r["macro_f1"] for r in runs])
        kshot3[K] = dict(macro_f1_mean=float(f1.mean()), macro_f1_std=float(f1.std()),
                         recall_mean={c: float(np.mean([r["recall"][c] for r in runs])) for c in CLASSES},
                         cm_mean=np.mean([r["cm"] for r in runs], 0).round().astype(int).tolist(),
                         by_role_mean={r: {c: round(float(np.mean([ra[r][c] for ra in roles_acc])), 4) for c in CLASSES}
                                       for r in ROLES})
    res["k_shot_all_classes_from_test_domain"] = kshot3

    # same support sets, but kNN instead of one prototype per class (the test-domain NHS class is bimodal:
    # NNHS-like centre clips + dense hotspot neighbours, which a single mean vector blurs)
    Zt = torch.as_tensor(Zc, device=DEV)
    knn_res = {}
    for K in (20, 100, 400):
        runs, roles_acc = [], []
        for rep_i in range(5):
            pred = np.zeros(len(EV), int)
            for b in benches:
                sup = np.concatenate([rng.choice(cal_c[(b, c)], min(K, len(cal_c[(b, c)])), replace=False) for c in range(3)])
                rows = np.nonzero(ev_b == b)[0]
                S = Zt[torch.as_tensor(EV[rows], device=DEV)] @ Zt[torch.as_tensor(sup, device=DEV)].T
                nn_idx = S.topk(min(15, len(sup)), dim=1).indices.cpu().numpy()
                votes = np.stack([(y[sup][nn_idx] == c).sum(1) for c in range(3)], 1)
                pred[rows] = votes.argmax(1)
            runs.append(multiclass_report(y[EV], pred))
            roles_acc.append(by_role(EV, pred))
        f1 = np.array([r["macro_f1"] for r in runs])
        knn_res[K] = dict(macro_f1_mean=float(f1.mean()), macro_f1_std=float(f1.std()),
                          recall_mean={c: float(np.mean([r["recall"][c] for r in runs])) for c in CLASSES},
                          cm_mean=np.mean([r["cm"] for r in runs], 0).round().astype(int).tolist(),
                          by_role_mean={r: {c: round(float(np.mean([ra[r][c] for ra in roles_acc])), 4) for c in CLASSES}
                                        for r in ROLES})
    res["k_shot_knn15"] = knn_res
    R["C_scheme1"][name] = res
    if S1 is None or name == "proto_dens_s0":  # figures use the main model
        S1 = dict(name=name, Z=Zc, kshot=kshot, knn=knn, tau=tau, hs_score=hs_score)

# ============================================================ D: Option-A supervised reference
R["D_optionA"] = {}
for f in sorted(glob.glob(str(RUNS / "sup3_*_s*/pred.npz"))):
    d = np.load(f)
    R["D_optionA"][Path(f).parent.name] = multiclass_report(y[d["idx_eval"]], d["prob"].argmax(1)) | dict(
        by_role=by_role(d["idx_eval"], d["prob"].argmax(1)))

# ============================================================ E: label-conflict ceiling for NNHS
nh_hash = set(m.hash[m.cls == "NHS"])
ev_nn = EV[y[EV] == 2]
conf = m.hash.values[ev_nn]
R["E_ceiling"] = dict(eval_nnhs=int(len(ev_nn)),
                      identical_to_an_NHS_image=int(np.isin(conf, list(nh_hash)).sum()),
                      blank_clip=int((m.n_edges.values[ev_nn] == 0).sum()))
# ============================================================ G: context analysis (NOT pure image classification)
# Uses the neighbourhood structure encoded in file names (.pngK = neighbour K of centre clip <id>)
# to test the hypothesis that NHS/NNHS of a neighbour window is decided by its centre clip, not by its pixels.
if ENS:
    p_ctx = ENS[best_rep]["p"]
    center_of = dict(zip(m.group[is_center], np.nonzero(is_center)[0]))
    cidx = np.array([center_of.get(g, -1) for g in m.group.values[EV]])
    center_hs = p_ctx[cidx] >= 0.5
    pred_ctx = np.where(is_center[EV], (p_ctx[EV] >= 0.5).astype(int), np.where(center_hs, 0, 2))
    R["G_context_analysis"] = multiclass_report(y[EV], pred_ctx) | dict(by_role=by_role(EV, pred_ctx))

# ============================================================ F: cross-design few-shot transfer
# Train without design `ho` (binary CNN and ProtoNet), then adapt with K labelled clips per class from that design.
def k_shot_eval(Zf, pool, target, hs_t, Ks, reps=50):
    rng = np.random.RandomState(0)
    res = {}
    for K in Ks:
        if K > min(len(pool[0]), len(pool[1])):
            continue
        accs, fas, aucs = [], [], []
        for _ in range(reps):
            pr = [Zf[rng.choice(pool[c], K, replace=False)].mean(0) for c in (0, 1)]
            pr = [v / np.linalg.norm(v) for v in pr]
            sc = Zf[target] @ pr[1] - Zf[target] @ pr[0]
            rep_ = hotspot_report(hs_t, sc > 0)
            accs.append(rep_["accuracy"]); fas.append(rep_["fa"]); aucs.append(auc(hs_t.astype(int), sc))
        res[K] = dict(accuracy_mean=float(np.mean(accs)), accuracy_std=float(np.std(accs)), fa_mean=float(np.mean(fas)),
                      fa_std=float(np.std(fas)), auc_mean=float(np.mean(aucs)))
    return res


def cross_design(ho, Ks=(1, 2, 5, 10, 20, 50), seed=0):
    out = {}
    target = (bench == ho) & is_test & is_center & (y < 2)
    hs_t = y[target] == 1
    pool = {c: TR[(bench[TR] == ho) & (y[TR] == c)] for c in (0, 1)}
    out["n_target"] = dict(hs=int(hs_t.sum()), nhs=int((~hs_t).sum()))
    out["n_pool"] = dict(hs=int(len(pool[1])), nhs=int(len(pool[0])))
    f_bin = RUNS / f"binary_dens_s{seed}_ho{ho}" / "pred.npz"
    if f_bin.exists():
        d = np.load(f_bin)
        pb = full(d["idx_test"], d["p_test"])
        out["binary_trained_without_it"] = dict(auc=auc(hs_t.astype(int), pb[target]), **hotspot_report(hs_t, pb[target] >= 0.5))
    f_pro = RUNS / f"proto_dens_s{seed}_ho{ho}" / "emb.npz"
    if f_pro.exists():
        d = np.load(f_pro)
        Zh = np.zeros((N, d["z"].shape[1]), np.float32)
        Zh[d["idx"]] = d["z"].astype(np.float32)
        out["protonet_k_shot_per_class"] = k_shot_eval(Zh, pool, target, hs_t, Ks)
    # ablation: identical K-shot prototype procedure on the conventional CNN's features (no meta-training)
    f_bm = RUNS / f"binary_dens_s{seed}_ho{ho}" / "model.pt"
    if f_bm.exists():
        from common import Arrays, Classifier, predict
        net = Classifier("dens", 2).to(DEV)
        net.load_state_dict(torch.load(f_bm, map_location=DEV))
        net.eval()
        rows_ = np.r_[pool[0], pool[1], np.nonzero(target)[0]]
        feat = predict(lambda x: torch.nn.functional.normalize(net.f(x), dim=-1), Arrays(), "dens", rows_, tta=4)
        Fb = np.zeros((N, feat.shape[1]), np.float32)
        Fb[rows_] = feat / np.linalg.norm(feat, axis=1, keepdims=True)
        out["cnn_features_k_shot_per_class"] = k_shot_eval(Fb, pool, target, hs_t, Ks)
    if "binary_dens_s0" in R["A_binary"]:
        out["reference_in_domain_binary"] = R["A_binary"]["binary_dens_s0"]["per_bench"][ho]
    return out


R["F_cross_node"] = cross_design("iccad1")
R["F_lodo"] = {b: cross_design(b) for b in ("iccad2", "iccad3", "iccad4", "iccad5")
               if (RUNS / f"proto_dens_s0_ho{b}" / "emb.npz").exists()}
R["F_cross_node"]["binary_trained_on_28nm_only"] = R["F_cross_node"].pop("binary_trained_without_it", None)

# training-seed robustness of the 28nm -> 32nm result (each seed: its own CNN and its own ProtoNet)
seeds = [sd for sd in range(5) if (RUNS / f"proto_dens_s{sd}_hoiccad1" / "emb.npz").exists()
         and (RUNS / f"binary_dens_s{sd}_hoiccad1" / "model.pt").exists()]
if len(seeds) > 1:
    per = {sd: (R["F_cross_node"] if sd == 0 else cross_design("iccad1", Ks=(2, 5, 10), seed=sd)) for sd in seeds}
    agg = {}
    for key, getter in [("cnn_direct_accuracy", lambda v: (v.get("binary_trained_on_28nm_only") or v["binary_trained_without_it"])["accuracy"]),
                        ("cnn_direct_auc", lambda v: (v.get("binary_trained_on_28nm_only") or v["binary_trained_without_it"])["auc"]),
                        ("proto_5shot_accuracy", lambda v: v["protonet_k_shot_per_class"][5]["accuracy_mean"]),
                        ("proto_5shot_fa", lambda v: v["protonet_k_shot_per_class"][5]["fa_mean"]),
                        ("proto_5shot_auc", lambda v: v["protonet_k_shot_per_class"][5]["auc_mean"]),
                        ("cnnfeat_5shot_accuracy", lambda v: v["cnn_features_k_shot_per_class"][5]["accuracy_mean"]),
                        ("cnnfeat_5shot_fa", lambda v: v["cnn_features_k_shot_per_class"][5]["fa_mean"]),
                        ("cnnfeat_5shot_auc", lambda v: v["cnn_features_k_shot_per_class"][5]["auc_mean"])]:
        vals = [getter(per[sd]) for sd in seeds]
        agg[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), per_seed=[float(v) for v in vals])
    R["F_cross_node_seeds"] = dict(seeds=seeds, **agg)

save_json(R, OUT / "metrics.json")

# ============================================================ figures
if ENS:
    s = S2["s"]
    roles = [("热点 HS（中心窗）", (y == 1), C_HS), ("非热点 NHS（中心窗）", (y == 0) & is_center, C_NHS),
             ("NHS（热点的 8 邻窗）", (y == 0) & ~is_center, C_NHS), ("疑似 NNHS（非热点的 8 邻窗）", (y == 2), C_NN)]
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 7), sharex=True)
    bins = np.linspace(-14, 14, 113)
    for ax, (title, mask, col) in zip(axes, roles):
        v = s[EV][mask[EV]]
        ax.hist(v, bins=bins, color=col, alpha=0.9, weights=np.full(len(v), 100 / len(v)), edgecolor=SURF, linewidth=0.3)
        ax.axvspan(S2["lo"], S2["hi"], color=C_NN, alpha=0.10, linewidth=0)
        ax.set_title(f"{title}   n={len(v):,}", loc="left", fontsize=10)
        ax.set_ylabel("% 样本")
    axes[0].text(S2["lo"] + 0.2, axes[0].get_ylim()[1] * 0.8, "判为 NNHS 的区间", color=INK2, fontsize=9)
    axes[-1].set_xlabel("logit p(HS)   ← 越像非热点 | 越像热点 →")
    fig.suptitle("方案二：二分类模型输出的热点概率在四类窗口上的分布（EVAL）", x=0.02, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "s2_prob_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    F1 = S2["F1"]
    im = ax.imshow(F1.T, origin="lower", extent=[GRID[0], GRID[-1], GRID[0], GRID[-1]], cmap="Blues", vmin=np.nanmin(F1), vmax=np.nanmax(F1))
    ax.plot(S2["lo"], S2["hi"], marker="o", ms=8, mfc="none", mec=C_HS, mew=2)
    ax.annotate(f"最优 macro-F1 = {np.nanmax(F1):.3f}", (S2["lo"], S2["hi"]), xytext=(8, -14), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("下阈值 lo（logit）"); ax.set_ylabel("上阈值 hi（logit）"); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="CALIB macro-F1")
    ax.set_title("方案二：双阈值搜索（CALIB）", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "s2_threshold_grid.png", dpi=200); plt.close(fig)

if S1 is not None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    curves = [(R["C_scheme1"][S1["name"]]["k_shot_all_classes_from_test_domain"], "方案一：三类各取 K 个（全部来自测试域）", C_NHS, "-"),
              (S1["kshot"], "方案一：只取 K 个 NNHS（HS/NHS 原型来自训练集）", "#86b6ef", "--"),
              (R["C_scheme1"][S1["name"]]["k_shot_knn15"], "方案一：三类各取 K 个 + kNN(15)", "#6250d6", "-")]
    for ks, lab, col, ls in curves:
        Ks = sorted(ks, key=int)
        mu = np.array([ks[k]["macro_f1_mean"] for k in Ks])
        sd = np.array([ks[k]["macro_f1_std"] for k in Ks])
        Kn = [int(k) for k in Ks]
        ax.fill_between(Kn, mu - sd, mu + sd, color=col, alpha=0.15, linewidth=0)
        ax.plot(Kn, mu, color=col, lw=2, ls=ls, marker="o", ms=6, mec=SURF, mew=1.5, label=lab)
    refs = []
    if ENS:
        refs.append((R["B_scheme2"][best_rep]["tuned_on_calib"]["eval"]["macro_f1"], "方案二：概率双阈值", C_HS, "--"))
        refs.append((R["B_scheme2"][best_rep]["trivial_all_nonHS_as_NNHS"]["macro_f1"], "平凡基线：非热点全判 NNHS", MUTED, ":"))
    for k, v in R["D_optionA"].items():
        refs.append((v["macro_f1"], "参考：全监督三分类（CALIB 全部标签）", INK2, "-."))
    for v, lab, col, ls in refs:
        ax.axhline(v, color=col, ls=ls, lw=1.6, label=f"{lab}  {v:.3f}")
    ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20, 50, 100, 200, 400]); ax.set_xticklabels([1, 5, 10, 20, 50, 100, 200, 400])
    ax.set_xlabel("每个设计、每类提供的样本数 K"); ax.set_ylabel("EVAL macro-F1（三分类）")
    ax.set_title("方案一：K-shot 原型 vs 方案二 vs 参考线", loc="left")
    ax.set_ylim(0.42, 0.93)
    ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "s1_kshot_curve.png", dpi=200); plt.close(fig)

    # 2-D PCA of the embedding, subsample per role
    rng = np.random.RandomState(0)
    Zc = S1["Z"]
    pick = np.concatenate([rng.choice(EV[(y[EV] == c)], 1500, replace=False) for c in (0, 1, 2)])
    Zs = Zc[pick] - Zc[pick].mean(0)
    _, _, Vt = np.linalg.svd(Zs, full_matrices=False)
    XY = Zs @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, col in ((2, C_NN), (0, C_NHS), (1, C_HS)):
        sel = y[pick] == c
        ax.scatter(XY[sel, 0], XY[sel, 1], s=6, color=col, alpha=0.55, linewidths=0, label=CLASSES[c])
    ax.legend(markerscale=3, loc="best"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("方案一：度量空间嵌入（PCA 2D，EVAL 各类 1500 个）", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "s1_embedding_pca.png", dpi=200); plt.close(fig)


# confusion matrices side by side
def cm_plot(ax, cm, title):
    cm = np.asarray(cm, float)
    rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{rn[i, j]*100:.1f}%\n{int(cm[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="white" if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.grid(False); ax.set_title(title, fontsize=10, loc="left")


panels = []
if ENS:
    panels.append((R["B_scheme2"][best_rep]["tuned_on_calib"]["eval"]["cm"], "方案二 概率双阈值"))
if S1 is not None:
    ksn = R["C_scheme1"][S1["name"]]["k_shot_knn15"]
    kbest = max(ksn, key=lambda k: ksn[k]["macro_f1_mean"])
    panels.append((ksn[kbest]["cm_mean"], f"方案一 {kbest}-shot kNN"))
for k, v in R["D_optionA"].items():
    panels.append((v["cm"], "参考 全监督三分类"))
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    for ax, (cm, t) in zip(np.atleast_1d(axes), panels):
        cm_plot(ax, cm, t)
    fig.tight_layout(); fig.savefig(FIG / "confusion.png", dpi=200); plt.close(fig)

# ROC on the literature test set with the paper's operating point
fig, ax = plt.subplots(figsize=(6, 4.5))
hs = y[iccad12] == 1
curves = []
if ENS:
    curves.append((S2["s"][iccad12], f"方案二 二分类 CNN 集成（{best_rep}）", C_HS))
if S1 is not None:
    curves.append((S1["hs_score"][iccad12], "方案一 原型网络", C_NHS))
for sc, lab, col in curves:
    o = np.argsort(-sc)
    tpr = np.cumsum(hs[o]) / hs.sum()
    fpr = np.cumsum(~hs[o]) / (~hs).sum()
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{lab}  AUC={auc(hs.astype(int), sc):.4f}")
ax.plot(PAPER["fa"] / (~hs).sum(), PAPER["accuracy"], marker="D", ms=8, color=INK, linestyle="none",
        label=f"Geng et al. ICCAD'20：Acc {PAPER['accuracy']*100:.2f}%, FA {PAPER['fa']}")
ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0.9, 1.002)
ax.set_xlabel("误报率 FA / #NHS（对数）"); ax.set_ylabel("热点检出率 Accuracy")
ax.set_title("二分类热点检测：ICCAD12 标准测试集（iccad2-5 合并）", loc="left")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "binary_roc.png", dpi=200); plt.close(fig)

print(json.dumps({k: R[k] for k in ("E_ceiling",)}, indent=1))
print("figures ->", FIG)

# cross-node few-shot figure
F = R["F_cross_node"]
if "protonet_k_shot_per_class" in F and "cnn_features_k_shot_per_class" in F:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, lab, col in (("protonet_k_shot_per_class", "方案一 原型网络（meta 训练）", C_NHS),
                          ("cnn_features_k_shot_per_class", "消融：普通 CNN 特征 + 原型", MUTED)):
        Ks = sorted(F[key], key=int)
        mu = np.array([F[key][k]["accuracy_mean"] for k in Ks]) * 100
        sd = np.array([F[key][k]["accuracy_std"] for k in Ks]) * 100
        Kn = [int(k) for k in Ks]
        ax.fill_between(Kn, np.clip(mu - sd, 0, 100), np.clip(mu + sd, 0, 100), color=col, alpha=0.15, linewidth=0)
        ax.plot(Kn, mu, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=lab)
        ax.annotate(f"K=5: {mu[2]:.1f}%", (Kn[2], mu[2]), xytext=(0, -16), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ref = F["reference_in_domain_binary"]["accuracy"] * 100
    ax.axhline(ref, color=INK2, ls="-.", lw=1.6, label=f"参考：用全部 439 个 32nm 标签训练  {ref:.1f}%")
    b0 = F["binary_trained_on_28nm_only"]["accuracy"] * 100
    ax.axhline(b0, color=C_HS, ls="--", lw=1.6, label=f"普通 CNN 只用 28nm 训练、直接用于 32nm  {b0:.1f}%")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels([1, 2, 5, 10, 20, 50])
    ax.set_ylim(-3, 103)
    ax.set_xlabel("32nm 新设计中每类给出的标注样本数 K"); ax.set_ylabel("热点检出率 Accuracy（%）")
    ax.set_title("跨工艺节点：28nm 训练 → 32nm 测试（iccad1，226 HS / 319 NHS）", loc="left")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "cross_node.png", dpi=200); plt.close(fig)
    print("cross-node figure saved")
