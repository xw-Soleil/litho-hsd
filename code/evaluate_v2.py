"""Evaluate a frozen, deduplicated protocol without selecting methods on test.

Only saved predictions/embeddings are read; this script never trains a model.
Threshold fitting uses tune, open-set calibration uses val, and support sampling
uses support (or target-node train in the cross-node experiment). All evaluated
predictions and exact support row IDs are retained for independent recomputation.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["NHS", "HS", "NNHS"]
SCALARS = ["macro_f1", "balanced_accuracy", "overall_accuracy", "hs_recall",
           "false_alarms", "false_positive_rate", "roc_auc"]
K_VALUES = (1, 5, 10, 20)
KNN_VALUES = (20, 100)
REPEATS = 30
SUPPORT_SEED = 20261031
RUN_PATTERN = re.compile(r"^(binary|proto|sup3)_(densm|dens|squish)_s(\d+)(?:_ho(iccad\d+))?$")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def roc_auc(y, score):
    y = np.asarray(y, bool)
    if not y.any() or y.all():
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    n = int(y.sum())
    return float((ranks[y].sum() - n * (n + 1) / 2) / (n * (~y).sum()))


def report(y, pred, nclasses=3, score=None):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    if not len(y):
        raise ValueError("Cannot evaluate an empty target")
    if np.any((pred < 0) | (pred >= nclasses)):
        raise ValueError("Prediction label outside the task label set")
    cm = np.bincount(nclasses * y + pred, minlength=nclasses ** 2).reshape(nclasses, nclasses)
    count = cm.sum(1)
    recall = np.divide(cm.diagonal(), count, out=np.zeros(nclasses), where=count > 0)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(nclasses), where=cm.sum(0) > 0)
    f1 = np.divide(2 * cm.diagonal(), count + cm.sum(0), out=np.zeros(nclasses), where=count + cm.sum(0) > 0)
    false_alarms = int(((y != 1) & (pred == 1)).sum())
    return dict(n=len(y), n_per_class=dict(zip(CLASSES[:nclasses], map(int, count))),
                confusion=cm.tolist(), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), overall_accuracy=float(np.trace(cm) / len(y)),
                hs_recall=float(recall[1]), false_alarms=false_alarms,
                false_positive_rate=float(false_alarms / (y != 1).sum()) if (y != 1).any() else None,
                roc_auc=roc_auc(y == 1, score) if score is not None else None,
                per_class={c: dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]))
                           for i, c in enumerate(CLASSES[:nclasses])})


def summarize(reports):
    mean, sd = {}, {}
    for key in SCALARS:
        values = [r[key] for r in reports if r[key] is not None]
        mean[key] = float(np.mean(values)) if values else None
        sd[key] = float(np.std(values, ddof=0)) if values else None
    return dict(n=reports[0]["n"], n_per_class=reports[0]["n_per_class"], mean=mean,
                support_sampling_std=sd, trials=len(reports),
                confusion_mean=np.mean([r["confusion"] for r in reports], axis=0).tolist(),
                per_class_mean={c: {k: float(np.mean([r["per_class"][c][k] for r in reports]))
                                     for k in ("precision", "recall", "f1")} for c in reports[0]["per_class"]})


def three_way(score, lo, hi):
    return np.where(score < lo, 0, np.where(score > hi, 1, 2)).astype(np.uint8)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def tune_band(score, y):
    """Fixed logit grid; ties resolve to the first ascending (lo, hi) pair."""
    grid = np.arange(-12.0, 12.001, 0.25)
    counts = np.bincount(y, minlength=3)
    if np.any(counts == 0):
        raise ValueError("tune must contain all three classes")
    ordered = [np.sort(score[y == c]) for c in range(3)]
    below = np.stack([np.searchsorted(x, grid, side="left") for x in ordered])
    above = counts[:, None] - np.stack([np.searchsorted(x, grid, side="right") for x in ordered])
    best = (-1.0, None, None)
    for i, lo in enumerate(grid):
        left = np.repeat(below[:, i, None], len(grid) - i, axis=1)
        right = above[:, i:]
        middle = counts[:, None] - left - right
        cm = np.stack([left, right, middle], axis=1)  # true, predicted, threshold
        tp = np.stack([cm[c, c] for c in range(3)])
        f1 = np.mean(2 * tp / np.maximum(counts[:, None] + cm.sum(axis=0), 1), axis=0)
        j = int(np.argmax(f1))
        if f1[j] > best[0]:
            best = (float(f1[j]), float(lo), float(grid[i + j]))
    return dict(tune_macro_f1=best[0], lo_logit=best[1], hi_logit=best[2],
                lo_score=float(1 / (1 + np.exp(-best[1]))), hi_score=float(1 / (1 + np.exp(-best[2]))))


def home_relative(args):
    """Replace the home-directory prefix in recorded paths with ~ (keeps account names out of published metrics)."""
    if not isinstance(args, dict):
        return args
    home = str(Path.home())
    return {k: "~" + v[len(home):] if isinstance(v, str) and v.startswith(home + "/") else v for k, v in args.items()}


def read_artifact(path, key, required, nrows):
    with np.load(path, allow_pickle=False) as data:
        idx = data["idx"].astype(np.int64)
        values = data[key].astype(np.float32)
    if len(idx) != len(set(idx)) or len(idx) != len(values) or np.any(idx < 0) or np.any(idx >= nrows):
        raise ValueError(f"Invalid index alignment: {path}")
    if not np.isfinite(values).all() or not np.isin(required, idx).all():
        raise ValueError(f"Missing/nonfinite required values: {path}")
    full = np.full((nrows,) + values.shape[1:], np.nan, np.float32)
    full[idx] = values
    if key == "z":
        norm = np.linalg.norm(full[idx], axis=1, keepdims=True)
        if np.any(norm < 1e-10):
            raise ValueError(f"Zero embeddings: {path}")
        full[idx] /= norm
    elif key == "prob":
        if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(1), 1, atol=1e-4):
            raise ValueError(f"Invalid probabilities: {path}")
    return full


def support_plan(meta, pool, target, nclasses, K, repeats=REPEATS):
    """Seed independent of model, training seed and the order of evaluated Ks."""
    groups = sorted(meta.loc[target, "bench"].unique())
    pools = {(b, c): pool[(meta.loc[pool, "bench"].to_numpy() == b) & (meta.loc[pool, "y"].to_numpy() == c)]
             for b in groups for c in range(nclasses)}
    # Sample equivalent images once per class, rather than counting repeated files
    # as independent labels. Keep the first stable row as each key's representative.
    if "dup_key" in meta:
        pools = {key: meta.loc[rows].drop_duplicates("dup_key").row_id.to_numpy()
                 for key, rows in pools.items()}
    missing = [str(k) for k, v in pools.items() if not len(v)]
    if missing:
        raise ValueError("Empty support classes: " + ", ".join(missing))
    counts = {f"{b}/{CLASSES[c]}": min(K, len(v)) for (b, c), v in pools.items()}
    rng = np.random.default_rng(np.random.SeedSequence([SUPPORT_SEED, K, nclasses]))
    picks = np.stack([np.concatenate([rng.choice(v, min(K, len(v)), replace=False) for v in pools.values()])
                      for _ in range(repeats)])
    return picks, counts


def feature_predict(z, meta, target, support, nclasses, knn=None):
    pred = np.empty(len(target), np.uint8)
    score = np.empty(len(target), np.float32)
    target_bench = meta.loc[target, "bench"].to_numpy()
    for b in sorted(set(target_bench)):
        where = np.flatnonzero(target_bench == b)
        sup = support[meta.loc[support, "bench"].to_numpy() == b]
        ys = meta.loc[sup, "y"].to_numpy()
        if knn is None:
            prototypes = np.stack([z[sup[ys == c]].mean(0) for c in range(nclasses)])
            prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
            logits = z[target[where]] @ prototypes.T
            pred[where] = logits.argmax(1)
            score[where] = logits[:, 1] - logits[:, 0]
        else:
            # Stable sort makes ties deterministic, including exact duplicates in a partition.
            for start in range(0, len(where), 1024):
                loc = where[start:start + 1024]
                sim = z[target[loc]] @ z[sup].T
                nearest = np.argsort(-sim, axis=1, kind="stable")[:, :min(knn, len(sup))]
                votes = np.stack([(ys[nearest] == c).sum(1) for c in range(nclasses)], axis=1)
                pred[loc] = votes.argmax(1)
                score[loc] = votes[:, 1] / nearest.shape[1]
    return pred, score


def prototype_novelty(z, meta, target, train):
    similarity = np.empty((len(target), 2), np.float32)
    for b in sorted(meta.loc[target, "bench"].unique()):
        loc = np.flatnonzero(meta.loc[target, "bench"].to_numpy() == b)
        prototypes = []
        for c in (0, 1):
            rows = train[(meta.loc[train, "bench"].to_numpy() == b) & (meta.loc[train, "y"].to_numpy() == c)]
            if not len(rows):
                raise ValueError(f"Missing training prototype {b}/{c}")
            p = z[rows].mean(0)
            prototypes.append(p / max(np.linalg.norm(p), 1e-12))
        similarity[loc] = z[target[loc]] @ np.stack(prototypes).T
    return similarity


def cross_component_interval(meta, items, out, repeats=1000):
    """Cluster bootstrap conditional on the trained models and support draws.

    Average each query's predicted-HS indicator over draws and training seeds,
    then resample whole connected components, retaining their internal rows.
    This estimates target-sampling variation, not the uncertainty of retraining.
    """
    expected_hs, target = None, None
    for item in items:
        with np.load(out / item["prediction_file"], allow_pickle=False) as data:
            idx = data["idx"]
            if target is None:
                target = idx
                expected_hs = np.zeros(len(idx), float)
            elif not np.array_equal(target, idx):
                raise ValueError("Cross-node query rows differ across seeds")
            expected_hs += (data["pred"] == 1).mean(axis=0) / len(items)
    _, group = np.unique(meta.loc[target, "component_id"].to_numpy(), return_inverse=True)
    n_groups = int(group.max()) + 1
    hs = meta.loc[target, "y"].to_numpy() == 1
    count_hs = np.bincount(group, weights=hs, minlength=n_groups)
    count_nhs = np.bincount(group, weights=~hs, minlength=n_groups)
    tp = np.bincount(group, weights=expected_hs * hs, minlength=n_groups)
    fp = np.bincount(group, weights=expected_hs * (~hs), minlength=n_groups)
    rng = np.random.default_rng(2026103197)
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=repeats)
    result = dict(unit="component_id", components=n_groups, resamples=repeats, seed=2026103197,
                  conditional_on="fixed trained models and fixed support draws")
    for key, numerator, denominator in (("hs_recall", tp, count_hs), ("false_positive_rate", fp, count_nhs)):
        den = weights @ denominator
        valid = den > 0
        values = (weights @ numerator)[valid] / den[valid]
        result[key] = dict(low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                           valid_resamples=int(valid.sum()))
    return result


def evaluate(args):
    manifest = Path(args.protocol)
    if manifest.is_dir():
        manifest /= "manifest.csv"
    meta = pd.read_csv(manifest, low_memory=False).sort_values("row_id").set_index("row_id", drop=False)
    if not np.array_equal(meta.index, np.arange(len(meta))):
        raise ValueError("Manifest must cover every cache row exactly once")
    original = pd.read_csv(Path(args.cache) / "meta.csv")
    if len(meta) != len(original) or not np.array_equal(meta.path, original.path):
        raise ValueError("Manifest rows do not match cache metadata")
    for col in ("is_center",):
        if col not in meta:
            meta[col] = original[col].to_numpy()
    if not np.array_equal(meta.y, meta.cls.map({c: i for i, c in enumerate(CLASSES)})):
        raise ValueError("Manifest label mapping mismatch")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "support_plans").mkdir(exist_ok=True)
    protocol_hash = sha256(manifest)
    audit_path = manifest.parent / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if audit and audit.get("manifest_sha256") != protocol_hash:
        raise ValueError("Data audit refers to a different manifest")
    plan_path = manifest.parent / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    expected_runs = set()
    if plan:
        if plan["primary_fewshot"]["support_draws"] != REPEATS or plan["primary_fewshot"]["shots_per_design_per_class"] != 5:
            raise ValueError("Frozen plan disagrees with evaluator support settings")
        for seed in plan["primary_seeds"]:
            for mode in ("binary", "proto", "sup3"):
                expected_runs.add(f"{mode}_{plan['primary_representation']}_s{seed}")
        for rep in plan["representation_ablation"]["representations"]:
            for seed in plan["representation_ablation"]["seeds"]:
                for mode in plan["representation_ablation"]["modes"]:
                    expected_runs.add(f"{mode}_{rep}_s{seed}")
        for seed in plan["cross_node"]["seeds"]:
            for mode in ("binary", "proto"):
                expected_runs.add(f"{mode}_{plan['primary_representation']}_s{seed}_ho{plan['cross_node']['holdout']}")
        missing = [name for name in sorted(expected_runs) if not (Path(args.runs) / name / "complete.json").exists()]
        if missing:
            raise ValueError("Frozen experiment list is incomplete: " + ", ".join(missing))
    active = meta.index[meta.part != "excluded"].to_numpy()
    parts = {p: meta.index[meta.part == p].to_numpy() for p in ("train", "val", "support", "tune", "test")}
    if any(not len(rows) for rows in parts.values()):
        raise ValueError("Every active partition must be nonempty")
    for key in ("component_id", "dup_key", "hash"):
        if key not in meta:
            continue
        seen = set()
        for name, rows in parts.items():
            keys = set(meta.loc[rows, key])
            if keys & seen:
                raise ValueError(f"Cross-partition {key} overlap at {name}")
            seen.update(keys)
    y = meta.y.to_numpy()
    entries, plans, sources = [], {}, {}
    roles = np.where(y == 1, "HS center", np.where(y == 2, "NNHS neighbor",
                     np.where(meta.is_center.to_numpy() == 1, "NHS center", "NHS neighbor")))

    def get_plan(task, pool, target, nclasses, K):
        key = f"{task}_k{K}"
        if key not in plans:
            support, counts = support_plan(meta, pool, target, nclasses, K)
            if np.intersect1d(support, target).size:
                raise ValueError("Support and query indices overlap")
            if "dup_key" in meta and set(meta.loc[support.ravel(), "dup_key"]) & set(meta.loc[target, "dup_key"]):
                raise ValueError("Support and query duplicate keys overlap")
            path = out / "support_plans" / f"{key}.npz"
            np.savez_compressed(path, support=support, target=target)
            unique_inputs = [int(meta.loc[rows, "dup_key"].nunique()) for rows in support] if "dup_key" in meta else [len(rows) for rows in support]
            conflicting_keys = []
            for rows in support:
                groups = meta.loc[rows].groupby("dup_key").y.nunique() if "dup_key" in meta else pd.Series(dtype=int)
                conflicting_keys.append(int((groups > 1).sum()))
            plans[key] = (support, counts, str(path.relative_to(out)), sha256(path), unique_inputs, conflicting_keys)
        return plans[key]

    def add(run, mode, rep, seed, task, method, target, predictions, scores=None, nclasses=3, **details):
        predictions = np.atleast_2d(predictions).astype(np.uint8)
        scores = np.atleast_2d(scores) if scores is not None else None
        reports = [report(y[target], p, nclasses, None if scores is None else scores[i]) for i, p in enumerate(predictions)]
        name = f"{run}_{task}_{method}" + (f"_k{details['requested_k']}" if "requested_k" in details else "")
        path = out / "predictions" / f"{name}.npz"
        arrays = dict(idx=target, pred=predictions)
        if scores is not None:
            arrays["hs_score"] = scores.astype(np.float32)
        np.savez_compressed(path, **arrays)
        subgroups = {"bench:" + b: meta.loc[target, "bench"].to_numpy() == b for b in sorted(meta.loc[target, "bench"].unique())}
        if nclasses == 3:
            subgroups.update({"role:" + r: roles[target] == r for r in sorted(set(roles[target]))})
        subreports = {}
        for label, mask in subgroups.items():
            subreports[label] = summarize([report(y[target][mask], p[mask], nclasses,
                                                  None if scores is None else scores[i][mask])
                                           for i, p in enumerate(predictions)])
        entries.append(dict(run=run, mode=mode, representation=rep, training_seed=seed, task=task, method=method,
                            summary=summarize(reports), subgroups=subreports, trial_metrics=reports,
                            prediction_file=str(path.relative_to(out)), prediction_sha256=sha256(path), **details))

    for folder in sorted(Path(args.runs).iterdir()):
        match = RUN_PATTERN.fullmatch(folder.name)
        if not match:
            continue
        if expected_runs and folder.name not in expected_runs:
            raise ValueError(f"Unplanned run must not enter the report: {folder.name}")
        if not (folder / "complete.json").exists():
            raise ValueError(f"Incomplete run: {folder}")
        mode, rep, seed_text, holdout = match.groups()
        seed = int(seed_text)
        log = json.loads((folder / "log.json").read_text())
        marker = json.loads((folder / "complete.json").read_text())
        if log.get("status") != "complete" or marker.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"Invalid completion marker: {folder}")
        required_files = {"log.json", "model.pt", "emb.npz"} | ({"pred.npz"} if mode != "proto" else set())
        hashes = marker.get("artifacts_sha256", {})
        if not required_files.issubset(hashes):
            raise ValueError(f"Completion marker does not cover every artifact: {folder}")
        for filename, expected in hashes.items():
            if Path(filename).name != filename or sha256(folder / filename) != expected:
                raise ValueError(f"Artifact checksum mismatch: {folder}/{filename}")
        run_args = log.get("args", {})
        if (run_args.get("mode"), run_args.get("rep"), run_args.get("seed"), run_args.get("holdout") or None) != (mode, rep, seed, holdout):
            raise ValueError(f"Run name and training arguments disagree: {folder}")
        if log.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"Protocol mismatch in {folder}")
        expected_train = np.r_[parts["train"], parts["support"]] if mode == "sup3" else parts["train"]
        if holdout:
            expected_train = expected_train[meta.loc[expected_train, "bench"].to_numpy() != holdout]
        expected_hash = hashlib.sha256(np.sort(expected_train).astype("<i8").tobytes()).hexdigest()
        if log.get("train_idx_sha256") != expected_hash:
            raise ValueError(f"Training rows disagree with the frozen protocol: {folder}")
        sources[folder.name] = dict(log_sha256=sha256(folder / "log.json"), args=home_relative(log.get("args")),
                                    train_idx_sha256=log.get("train_idx_sha256"))
        print("evaluating", folder.name, flush=True)
        prob = read_artifact(folder / "pred.npz", "prob", active, len(meta)) if mode != "proto" else None
        if holdout:
            if mode == "sup3":
                raise ValueError("Cross-node supervised runs are not in the frozen plan")
            task = "cross_" + holdout
            if holdout == "iccad1" and "cross_target" in meta:
                target = meta.index[meta.cross_target.astype(str).str.lower().eq("true")].to_numpy()
                pool = meta.index[meta.cross_support.astype(str).str.lower().eq("true")].to_numpy()
            else:
                target = parts["test"][(meta.loc[parts["test"], "bench"].to_numpy() == holdout) &
                                       (meta.loc[parts["test"], "is_center"].to_numpy() == 1) & (y[parts["test"]] < 2)]
                pool = parts["train"][meta.loc[parts["train"], "bench"].to_numpy() == holdout]
            nclasses = 2
            if mode == "binary":
                add(folder.name, mode, rep, seed, task, "direct_no_target_labels", target,
                    prob[target, 1] >= 0.5, prob[target, 1], nclasses, target_label_budget=0)
        else:
            task, target, pool, nclasses = "three_class", parts["test"], parts["support"], 3
            if mode == "sup3":
                add(folder.name, mode, rep, seed, task, "supervised", target, prob[target].argmax(1),
                    target_label_budget=len(pool), training_partitions=["train", "support"])
                continue
            if mode == "binary":
                scores = logit(prob[:, 1])
                fitted = tune_band(scores[parts["tune"]], y[parts["tune"]])
                add(folder.name, mode, rep, seed, task, "two_threshold", target,
                    three_way(scores[target], fitted["lo_logit"], fitted["hi_logit"]),
                    threshold_fit=fitted, threshold_partition="tune", target_label_budget=len(parts["tune"]))
                add(folder.name, mode, rep, seed, task, "fixed_band_02_08", target,
                    three_way(prob[target, 1], 0.2, 0.8), target_label_budget=0)
                add(folder.name, mode, rep, seed, task, "nonhs_as_nnhs", target,
                    np.where(prob[target, 1] >= 0.5, 1, 2), target_label_budget=0)
                center = target[(meta.loc[target, "is_center"].to_numpy() == 1) & (y[target] < 2)]
                add(folder.name, mode, rep, seed, "binary_centers", "fixed_threshold_05", center,
                    prob[center, 1] >= 0.5, prob[center, 1], 2, target_label_budget=0)
        z = read_artifact(folder / "emb.npz", "z", active, len(meta))
        if not holdout and mode == "proto":
            va_sim = prototype_novelty(z, meta, parts["val"], parts["train"])
            threshold = float(np.quantile(va_sim.max(1), 0.05))
            sim = prototype_novelty(z, meta, target, parts["train"])
            add(folder.name, mode, rep, seed, task, "zero_label_prototype_rejection", target,
                np.where(sim.max(1) < threshold, 2, sim.argmax(1)), target_label_budget=0,
                calibration_partition="val", similarity_threshold=threshold,
                novelty_definition="maximum cosine similarity to the two same-design training class prototypes")
        methods = [("prototype", K, None) for K in K_VALUES]
        if not holdout:
            methods += [("knn15", K, 15) for K in KNN_VALUES]
        for method, K, knn in methods:
            support, counts, filename, filehash, unique_inputs, conflicts = get_plan(task, pool, target, nclasses, K)
            predictions, scores = [], []
            for rows in support:
                pred, score = feature_predict(z, meta, target, rows, nclasses, knn)
                predictions.append(pred)
                scores.append(score)
            add(folder.name, mode, rep, seed, task, method, target, np.stack(predictions),
                np.stack(scores) if nclasses == 2 else None, nclasses,
                requested_k=K, actual_support_per_design_class=counts, target_label_budget=sum(counts.values()),
                unique_support_inputs_per_trial=unique_inputs, conflicting_support_keys_per_trial=conflicts,
                support_plan=filename, support_plan_sha256=filehash, primary=(method == "prototype" and K == 5))
    if not entries:
        raise ValueError("No completed v2 runs found")
    grouped = {}
    for entry in entries:
        key = (entry["task"], entry["representation"], entry["mode"], entry["method"], entry.get("requested_k"))
        grouped.setdefault(key, []).append(entry)
    aggregates = []
    for key, items in grouped.items():
        if len({e["training_seed"] for e in items}) != len(items):
            raise ValueError("Duplicate training seeds in an aggregate")
        means, stds = {}, {}
        for metric in SCALARS:
            vals = [e["summary"]["mean"][metric] for e in items if e["summary"]["mean"][metric] is not None]
            means[metric] = float(np.mean(vals)) if vals else None
            stds[metric] = float(np.std(vals, ddof=0)) if vals else None
        aggregates.append(dict(task=key[0], representation=key[1], mode=key[2], method=key[3], requested_k=key[4],
                               training_seeds=[e["training_seed"] for e in items], mean=means, training_seed_std=stds,
                               per_class_mean={c: {metric: float(np.mean([e["summary"]["per_class_mean"][c][metric] for e in items]))
                                                    for metric in ("precision", "recall", "f1")}
                                               for c in items[0]["summary"]["per_class_mean"]},
                               support_sampling_std_mean={metric: float(np.mean([e["summary"]["support_sampling_std"][metric] for e in items]))
                                                          if items[0]["summary"]["support_sampling_std"][metric] is not None else None
                                                          for metric in SCALARS},
                               target_label_budget=items[0].get("target_label_budget"),
                               unique_support_inputs_per_trial=items[0].get("unique_support_inputs_per_trial"),
                               conflicting_support_keys_per_trial=items[0].get("conflicting_support_keys_per_trial"),
                               actual_support_per_design_class=items[0].get("actual_support_per_design_class"),
                               n=items[0]["summary"]["n"], n_per_class=items[0]["summary"]["n_per_class"]))
        if key[0].startswith("cross_") and (key[3] == "direct_no_target_labels" or key[4] == 5):
            aggregates[-1]["component_bootstrap_ci95"] = cross_component_interval(meta, items, out)
    result = dict(schema_version=2, protocol_sha256=protocol_hash,
                  frozen_plan_sha256=sha256(plan_path) if plan else None,
                  data_audit_sha256=sha256(audit_path) if audit else None,
                  test_structure=audit.get("test_structure") if audit else None,
                  source_test_count=int((meta.split == "test").sum()),
                  excluded_source_test_count=int(((meta.split == "test") & (meta.part == "excluded")).sum()),
                  evaluation_policy=dict(primary_representation="densm", primary_k=5, prototype_k=list(K_VALUES),
                                         secondary_knn15_k=list(KNN_VALUES), support_repeats=REPEATS,
                                         support_seed=SUPPORT_SEED, threshold_grid_logit=[-12, 12, 0.25],
                                         cross_component_bootstrap=dict(resamples=1000, seed=2026103197),
                                         seed_std_ddof=0, test_used_for_selection=False,
                                         caveat="The dataset was explored in the previous project. This is a frozen audited rerun, not an untouched external test dataset."),
                  counts=meta.groupby(["part", "cls"]).size().unstack(fill_value=0).to_dict(orient="index"),
                  run_sources=sources, aggregates=aggregates, entries=entries)
    save_json(result, out / "metrics.json")
    print("saved", out / "metrics.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--protocol", default=str(root / "experiments/v2/manifest.csv"))
    parser.add_argument("--cache", default=str(root / "cache"))
    parser.add_argument("--runs", default=str(root / "runs/v2"))
    parser.add_argument("--out", default=str(root / "results/v2"))
    evaluate(parser.parse_args())
