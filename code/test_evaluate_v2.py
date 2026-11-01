"""Small, synthetic checks for evaluation correctness; no real test scores used.

Run: python -m unittest discover -s code -p test_evaluate_v2.py
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import pandas as pd

import evaluate_v2 as ev


def fixture_meta():
    rows = []
    for part in ("train", "val", "support", "tune", "test"):
        for bench in ("iccad1", "iccad2"):
            for cls in range(2 if part in ("train", "val") else 3):
                for k in range(6):
                    # Six files but only five distinct support inputs per class.
                    key = f"{part}_{bench}_{cls}_{0 if part == 'support' and k == 5 else k}"
                    rows.append(dict(row_id=len(rows), path=f"{bench}/{part}/{cls}_{k}.png",
                                     part=part, split="train" if part in ("train", "val") else "test",
                                     bench=bench, y=cls, cls=ev.CLASSES[cls], is_center=int(cls < 2),
                                     hash=key, dup_key=key, component_id=key,
                                     cross_target=part in ("support", "tune", "test") and bench == "iccad1" and cls < 2,
                                     cross_support=part == "train" and bench == "iccad1"))
    return pd.DataFrame(rows).set_index("row_id", drop=False)


class EvaluationChecks(unittest.TestCase):
    def test_threshold_search_matches_direct_counting_with_ties(self):
        # Includes observations exactly on the grid: <lo and >hi must agree
        # between threshold selection and deployment, including lo == hi.
        y = np.tile(np.arange(3), 6)
        score = np.array([-2, 2, 0, -2, 0, 0, -1, 1, -1, 0, 1, 1,
                          -.25, .25, -.25, 0, .25, .25])
        chosen = ev.tune_band(score, y)
        best = -1.0
        grid = np.arange(-12, 12.001, .25)
        for i, lo in enumerate(grid):
            for hi in grid[i:]:
                pred = ev.three_way(score, lo, hi)
                # Independent, direct definition of class F1, without evaluator
                # reports or its CDF implementation.
                f1 = []
                for c in range(3):
                    tp = np.count_nonzero((pred == c) & (y == c))
                    f1.append(2 * tp / ((pred == c).sum() + (y == c).sum()))
                best = max(best, np.mean(f1))
        self.assertAlmostEqual(chosen["tune_macro_f1"], best)

    def test_support_draws_are_unique_paired_and_order_independent(self):
        m = fixture_meta()
        pool, target = (m.index[m.part == p].to_numpy() for p in ("support", "test"))
        first, counts = ev.support_plan(m, pool, target, 3, 5)
        ev.support_plan(m, pool, target, 3, 1)
        second, _ = ev.support_plan(m, pool, target, 3, 5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(counts.values()), {5})
        large, capped = ev.support_plan(m, pool, target, 3, 20)
        self.assertEqual(sum(capped.values()), 30)
        for draw in large:
            self.assertEqual(len(draw), m.loc[draw, "dup_key"].nunique())
            self.assertFalse(set(draw) & set(target))

    def test_saved_predictions_cross_target_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="dtco_eval_check_") as tmp:
            root = Path(tmp)
            cache, runs, out = (root / name for name in ("cache", "runs", "out"))
            cache.mkdir()
            runs.mkdir()
            m = fixture_meta()
            manifest = root / "manifest.csv"
            m.to_csv(manifest, index=False)
            m.to_csv(cache / "meta.csv", index=False)
            rng = np.random.default_rng(2)
            z = np.eye(4)[m.y] + rng.normal(0, .03, (len(m), 4))
            z = (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)
            for mode, holdout in (("binary", ""), ("proto", ""), ("sup3", ""),
                                  ("binary", "iccad1"), ("proto", "iccad1")):
                name = f"{mode}_densm_s0" + (f"_ho{holdout}" if holdout else "")
                folder = runs / name
                folder.mkdir()
                np.savez(folder / "emb.npz", idx=np.arange(len(m)), z=z)
                if mode != "proto":
                    nc = 3 if mode == "sup3" else 2
                    prob = np.full((len(m), nc), .03, np.float32)
                    prob[np.arange(len(m)), m.y if nc == 3 else (m.y == 1).astype(int)] = 1 - .03 * (nc - 1)
                    np.savez(folder / "pred.npz", idx=np.arange(len(m)), prob=prob)
                train = m.index[m.part.isin(["train", "support"] if mode == "sup3" else ["train"]) & (m.bench != holdout)].to_numpy()
                log = dict(protocol_sha256=ev.sha256(manifest), status="complete",
                           train_idx_sha256=hashlib.sha256(np.sort(train).astype("<i8").tobytes()).hexdigest(),
                           args=dict(mode=mode, rep="densm", seed=0, holdout=holdout))
                ev.save_json(log, folder / "log.json")
                (folder / "model.pt").write_bytes(b"synthetic fixture; no model inference")
                ev.save_json(dict(protocol_sha256=ev.sha256(manifest),
                                  artifacts_sha256={p.name: ev.sha256(p) for p in folder.iterdir()}),
                             folder / "complete.json")
            args = Namespace(protocol=str(manifest), cache=str(cache), runs=str(runs), out=str(out))
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(args)
            result = json.loads((out / "metrics.json").read_text())
            self.assertEqual(len(result["entries"]), 27)
            paired = [r for r in result["entries"] if r["task"] == "three_class" and r["method"] == "prototype" and r["requested_k"] == 5]
            self.assertEqual(len(paired), 2)
            self.assertEqual(paired[0]["support_plan_sha256"], paired[1]["support_plan_sha256"])
            for row in result["entries"]:
                with np.load(out / row["prediction_file"]) as data:
                    expected_idx = m.index[m.cross_target] if row["task"].startswith("cross_") else m.index[m.part == "test"]
                    if row["task"] == "binary_centers":
                        expected_idx = expected_idx[m.loc[expected_idx, "is_center"] == 1]
                    np.testing.assert_array_equal(data["idx"], expected_idx)
                    self.assertEqual(data["pred"].shape[1], len(expected_idx))
            # The cross task deliberately includes the main support/tune clips;
            # it must not silently shrink to only main test clips.
            self.assertEqual(int(m.cross_target.sum()), 36)
            self.assertEqual(int((m.cross_target & (m.part == "test")).sum()), 12)
            (runs / "binary_densm_s0" / "model.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                ev.evaluate(args)


if __name__ == "__main__":
    unittest.main()
