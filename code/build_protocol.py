"""Freeze a component-disjoint protocol before any v2 model is trained.

Equivalence joins every official-test neighbourhood, identical original images,
and identical D4 views of the masked 120x120 density input. Any test component
touching official train is excluded rather than allowed to cross that boundary.
The remaining components are allocated using labels/counts only, never predictions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VERSION = "component-disjoint-v2.1"
SEED = 20261031
LABELS = {"NHS": 0, "HS": 1, "NNHS": 2}
PARTS = ["train", "val", "support", "tune", "test"]


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_key(x):
    """128-bit digest invariant to all eight square symmetries."""
    views = (x, x[:, ::-1], x[::-1, :], x[::-1, ::-1],
             x.T, x.T[:, ::-1], x.T[::-1, :], x.T[::-1, ::-1])
    return min(hashlib.blake2b(v.tobytes(), digest_size=16).hexdigest() for v in views)


def fingerprints(meta, cache, workers):
    source = {"meta_sha256": file_sha256(cache / "meta.csv"),
              "dens120m_sha256": file_sha256(cache / "dens120m.npy"),
              "algorithm": "minimum blake2b-128 digest across all 8 D4 views"}
    folder = cache / "v2"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = folder / "fingerprints.csv", folder / "fingerprints.json"
    if csv_path.exists() and json_path.exists():
        saved = json.loads(json_path.read_text())
        if saved.get("source") == source and saved.get("csv_sha256") == file_sha256(csv_path):
            keys = pd.read_csv(csv_path, dtype={"dup_key": str})
            if keys.path.tolist() == meta.path.tolist():
                return keys.dup_key.to_numpy(), source
    arr = np.load(cache / "dens120m.npy", mmap_mode="r")
    if arr.shape != (len(meta), 120, 120) or arr.dtype != np.uint8:
        raise ValueError("Unexpected density cache shape/dtype")

    def chunk(start):
        return [canonical_key(x) for x in arr[start:start + 512]]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        keys = list(itertools.chain.from_iterable(executor.map(chunk, range(0, len(meta), 512))))
    pd.DataFrame({"path": meta.path, "dup_key": keys}).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"source": source, "csv_sha256": file_sha256(csv_path)}, indent=2))
    return np.asarray(keys), source


def components(meta):
    """Connected components, including test neighbours even when their labels differ."""
    parent = np.arange(len(meta))
    size = np.ones(len(meta), dtype=int)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]

    for series in (meta.dup_key, meta["hash"],
                   meta.group.where(meta.split == "test", meta.path)):
        first = {}
        for i, key in enumerate(series):
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    roots = np.array([root(i) for i in range(len(meta))])
    minimum = {}
    for i, r in enumerate(roots):
        minimum.setdefault(r, i)
    return np.array([minimum[r] for r in roots])


def allocate_components(meta, names, fractions, minimum_per_class, seed):
    """Deterministic weighted allocation; repair count minima without consulting a model.

    Components spanning several benchmarks/classes stay indivisible. Therefore
    the requested fractions are targets, not exact quotas. Minima refer to rows;
    the audit separately reports unique equivalence keys available for few-shot.
    """
    counts = pd.crosstab(meta.component_id, meta.bench + "/" + meta.cls)
    values = counts.to_numpy()
    total = values.sum(axis=0)
    minima = np.broadcast_to(np.asarray(minimum_per_class)[:, None], (len(names), len(total))).copy()
    if np.any(total < minima.sum(axis=0)):
        raise ValueError("Not enough rows to satisfy predetermined split minima")
    targets = np.asarray(fractions)[:, None] * total
    for j in range(len(names) - 1):
        targets[j] = np.maximum(targets[j], minima[j])
    targets[-1] = total - targets[:-1].sum(axis=0)
    if np.any(targets[-1] < minima[-1]):
        raise ValueError("Split targets conflict with minimum counts")
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(values)), -np.max(values / np.maximum(total, 1), axis=1)))
    current = np.zeros_like(targets)
    assignment = np.empty(len(values), dtype=int)
    denom = np.maximum(targets, 1)
    for i in order:
        costs = (((current + values[i] - targets) ** 2 - (current - targets) ** 2) / denom).sum(axis=1)
        j = int(np.argmin(costs))
        current[j] += values[i]
        assignment[i] = j

    # Greedy counts can be off by one for rare HS classes. Move an entire
    # component only if no donor minimum is violated; never split its images.
    repairs = []
    while np.any(current < minima):
        recipient, col = np.argwhere(current < minima)[0]
        choices = []
        for i in np.flatnonzero(values[:, col] > 0):
            donor = assignment[i]
            if donor == recipient or np.any(current[donor] - values[i] < minima[donor]):
                continue
            before = ((current[[donor, recipient]] - targets[[donor, recipient]]) ** 2 / denom[[donor, recipient]]).sum()
            after = current[[donor, recipient]].copy()
            after[0] -= values[i]
            after[1] += values[i]
            cost = ((after - targets[[donor, recipient]]) ** 2 / denom[[donor, recipient]]).sum() - before
            choices.append((float(cost), int(counts.index[i]), int(i), int(donor)))
        if not choices:
            raise ValueError(f"Indivisible components cannot meet minimum for {names[recipient]}/{counts.columns[col]}")
        _, component, i, donor = min(choices)
        assignment[i] = recipient
        current[donor] -= values[i]
        current[recipient] += values[i]
        repairs.append({"component_id": component, "from": names[donor], "to": names[recipient]})
    mapping = dict(zip(counts.index, np.asarray(names)[assignment]))
    detail = {"requested_fractions": dict(zip(names, fractions)),
              "minimum_rows_per_benchmark_class": dict(zip(names, minimum_per_class)),
              "target_counts": {n: dict(zip(counts.columns, targets[j].tolist())) for j, n in enumerate(names)},
              "count_only_repairs": repairs}
    return mapping, detail


def assert_protocol(manifest):
    """Fail closed on a split leak, wrong origin, or unsafe cross-node pool."""
    m = manifest
    assert np.array_equal(m.row_id.to_numpy(), np.arange(len(m))), "row alignment"
    assert m.path.is_unique, "duplicate paths"
    assert m.part.isin(PARTS + ["excluded"]).all(), "unknown split"
    active = m[m.part != "excluded"]
    for col in ("component_id", "dup_key", "hash"):
        assert active.groupby(col).part.nunique().max() == 1, f"{col} leaks between parts"
    test_origin = active[active.split == "test"]
    assert test_origin.groupby("group").part.nunique().max() == 1, "neighbourhood leak"
    assert (active[active.part.isin(["train", "val"])].split == "train").all()
    assert (active[active.part.isin(["support", "tune", "test"])].split == "test").all()
    assert not m.loc[m.split == "train", "component_id"].isin(test_origin.component_id).any()
    for part in PARTS:
        for bench in sorted(m.bench.unique()):
            labels = ("HS", "NHS") if part in ("train", "val") else ("HS", "NHS", "NNHS")
            for cls in labels:
                count = int(((m.part == part) & (m.bench == bench) & (m.cls == cls)).sum())
                minimum = 15 if part == "train" else 5 if part in ("support", "tune") else 1
                assert count >= minimum, f"insufficient {part}/{bench}/{cls}: {count}"
    cross = m[m.cross_target.astype(bool)]
    pool = m[m.cross_support.astype(bool)]
    assert (cross.bench == "iccad1").all() and (cross.is_center == 1).all()
    assert (cross.split == "test").all() and (cross.part != "excluded").all()
    assert (pool.bench == "iccad1").all() and (pool.part == "train").all()
    assert not cross.component_id.isin(m.loc[m.split == "train", "component_id"]).any()


def build_manifest(meta, seed=SEED):
    m = meta.copy().reset_index(drop=True)
    m.insert(0, "row_id", np.arange(len(m)))
    m["y"] = m.cls.map(LABELS)
    m["component_id"] = components(m)
    train = m[m.split == "train"]
    train_components = set(train.component_id)
    eligible = m[(m.split == "test") & ~m.component_id.isin(train_components)]
    training, train_details = allocate_components(train, ["train", "val"], [.9, .1], [15, 1], seed)
    testing, test_details = allocate_components(eligible, ["support", "tune", "test"], [.2, .2, .6], [5, 5, 1], seed + 1)
    m["part"] = "excluded"
    m.loc[train.index, "part"] = train.component_id.map(training)
    m.loc[eligible.index, "part"] = eligible.component_id.map(testing)
    m["exclude_reason"] = np.where(m.part == "excluded", "component_contains_official_train", "")
    m["cross_support"] = (m.bench == "iccad1") & (m.part == "train")
    m["cross_target"] = (m.bench == "iccad1") & (m.split == "test") & (m.part != "excluded") & (m.is_center == 1)
    assert_protocol(m)
    return m, {"training_allocation": train_details, "target_allocation": test_details}


def label_conflicts(frame, keys):
    """Count mutually incompatible labels, without claiming an F1 ceiling.

    A deterministic classifier returning one label per identical input can
    satisfy at most the most frequent label in each key group. The other rows
    are the minimum number of errors, not all the NNHS rows in that group.
    """
    table = frame.groupby(keys + ["cls"]).size().unstack(fill_value=0)
    conflict = table[table.gt(0).sum(axis=1) > 1]
    return {"grouping_keys": keys,
            "groups": len(conflict),
            "rows": int(conflict.sum().sum()),
            "rows_by_class": {c: int(conflict[c].sum()) if c in conflict else 0 for c in LABELS},
            "minimum_unavoidable_errors": int((conflict.sum(axis=1) - conflict.max(axis=1)).sum())}


def test_structure(m):
    """Structural facts from the frozen labels/inputs only; no prediction access."""
    test = m[m.part == "test"].copy()
    test["role"] = np.where(test.cls == "HS", "HS-center",
                            np.where(test.cls == "NNHS", "NNHS-neighbour",
                                     np.where(test.is_center == 1, "NHS-center", "NHS-neighbour")))
    nhs_hashes = set(test.loc[test.cls == "NHS", "hash"])
    nnhs = test[test.cls == "NNHS"]
    matched = int(nnhs["hash"].isin(nhs_hashes).sum())
    return {"rows": len(test),
            "counts_by_role": test.groupby("role").size().to_dict(),
            "counts_by_benchmark_class": test.groupby(["bench", "cls"]).size().rename("rows").reset_index().to_dict("records"),
            "counts_by_benchmark_role": test.groupby(["bench", "role"]).size().rename("rows").reset_index().to_dict("records"),
            "unique_raw_hashes": int(test["hash"].nunique()),
            "unique_masked_d4_keys": int(test.dup_key.nunique()),
            "neighbourhoods": int(test.group.nunique()),
            "components": int(test.component_id.nunique()),
            "neighbourhood_size_histogram": {str(k): int(v) for k, v in test.groupby("group").size().value_counts().sort_index().items()},
            "blank_rows_n_edges_zero": int((test.n_edges == 0).sum()),
            "blank_rows_by_benchmark_class": test[test.n_edges == 0].groupby(["bench", "cls"]).size().rename("rows").reset_index().to_dict("records"),
            "raw_hash_conflicts": label_conflicts(test, ["hash"]),
            "benchmark_raw_hash_conflicts": label_conflicts(test, ["bench", "hash"]),
            "masked_d4_conflicts": label_conflicts(test, ["dup_key"]),
            "nnhs_matching_test_nhs_raw_image": {"rows": matched, "nnhs_rows": len(nnhs), "fraction": matched / len(nnhs)},
            "interpretation": [
                "The minimum unavoidable error count applies to a deterministic classifier that must output one label per grouping key. It is not a macro-F1 ceiling.",
                "Raw-image conflicts assume image-only prediction; benchmark+raw-image conflicts allow the benchmark identity as input.",
                "Masked-D4 conflicts require invariance to all eight D4 views. Four-view TTA does not guarantee that invariance.",
                "Blank clips were excluded through the component policy; this changes the evaluated population relative to the original test set."]}


def audit_manifest(m, source, allocation):
    counts = m.groupby(["part", "bench", "cls"]).size().rename("rows").reset_index()
    unique = m.groupby(["part", "bench", "cls"]).dup_key.nunique().rename("unique_dup_keys").reset_index()
    by_class = counts.merge(unique).to_dict(orient="records")
    overlap = []
    for a, b in itertools.combinations(PARTS, 2):
        ma, mb = m[m.part == a], m[m.part == b]
        overlap.append({"parts": [a, b], **{col: len(set(ma[col]) & set(mb[col]))
                                            for col in ("component_id", "dup_key", "hash")}})
    sizes = m.groupby("component_id").size().sort_values(ascending=False)
    cross_counts = {name: m[m[col]].groupby("cls").size().to_dict()
                    for name, col in (("support", "cross_support"), ("target", "cross_target"))}
    return {"protocol_version": VERSION, "seed": SEED, "source": source,
            "row_count": len(m), "counts": by_class, "part_counts": m.part.value_counts().to_dict(),
            "components": {"n": len(sizes), "largest_sizes": sizes.head(15).tolist()},
            "pairwise_overlap": overlap, "cross_node": cross_counts,
            "test_structure": test_structure(m), **allocation,
            "limitations": [
                "This new protocol cannot undo historical exploration of the original official test set.",
                "Only D4-equivalent masked density inputs, original pixel hashes and named neighbourhoods are joined; arbitrary spatial overlap or approximate similarity is not certified absent.",
                "Test components connected to official train are excluded wholesale. Results describe this restricted subset, not the original ICCAD benchmark population.",
                "Support and tune target 20% each, with a predetermined minimum of five rows per benchmark/class; component constraints make fractions approximate.",
                "Duplicate images within a part and conflicting labels are retained. Few-shot evaluators must sample unique dup_keys and report actual budgets.",
                "Cross-node target consists of all eligible iccad1 centres across support/tune/test. It is distinct from the main three-class test; holdout models must exclude all iccad1 labels during training and model selection."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "v2")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_csv(args.cache / "meta.csv", dtype={"hash": str})
    meta["dup_key"], source = fingerprints(meta, args.cache, args.workers)
    manifest, allocation = build_manifest(meta)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    if manifest_path.exists():
        # Re-running is permitted only if the frozen manifest is byte-identical.
        expected = manifest.to_csv(index=False).encode()
        if manifest_path.read_bytes() != expected:
            raise RuntimeError("Existing frozen manifest differs; use a new protocol directory")
    else:
        manifest.to_csv(manifest_path, index=False)
    audit = audit_manifest(manifest, source, allocation)
    audit["manifest_sha256"] = file_sha256(manifest_path)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sha256": audit["manifest_sha256"],
                      "parts": audit["part_counts"], "cross_node": audit["cross_node"]}, indent=2))


if __name__ == "__main__":
    main()
