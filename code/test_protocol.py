"""Targeted regression checks for the v2 partition boundary."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_protocol import (ROOT, allocate_components, assert_protocol, canonical_key,
                            components, file_sha256, label_conflicts)


class ProtocolTests(unittest.TestCase):
    def test_d4_canonicalization(self):
        x = np.arange(49, dtype=np.uint8).reshape(7, 7)
        key = canonical_key(x)
        for v in (x, x[:, ::-1], x[::-1], x.T, x.T[::-1, ::-1]):
            self.assertEqual(canonical_key(v), key)
        y = x.copy()
        y[0, 1] += 1
        self.assertNotEqual(canonical_key(y), key)

    def test_transitive_neighbour_and_duplicate_link(self):
        # Train A == test B; test B neighbours test C; test C == test D.
        m = pd.DataFrame({"dup_key": ["a", "a", "c", "c", "e"],
                          "hash": ["raw0", "raw1", "raw2", "raw3", "raw4"],
                          "split": ["train", "test", "test", "test", "train"],
                          "group": ["ignored", "g1", "g1", "g2", "ignored"],
                          "path": ["p0", "p1", "p2", "p3", "p4"]})
        ids = components(m)
        self.assertEqual(len(set(ids[:4])), 1)
        self.assertNotEqual(ids[0], ids[4])

    def test_allocation_is_deterministic_and_retains_rare_classes(self):
        m = pd.DataFrame({"component_id": np.arange(50), "bench": "iccad5",
                          "cls": ["HS"] * 22 + ["NHS"] * 28})
        a, _ = allocate_components(m, ["support", "tune", "test"], [.2, .2, .6], [5, 5, 1], 17)
        b, _ = allocate_components(m.sample(frac=1, random_state=3), ["support", "tune", "test"], [.2, .2, .6], [5, 5, 1], 17)
        self.assertEqual(a, b)
        m["part"] = m.component_id.map(a)
        self.assertGreaterEqual(len(m[(m.part == "support") & (m.cls == "HS")]), 5)
        self.assertGreaterEqual(len(m[(m.part == "tune") & (m.cls == "HS")]), 5)

    def test_real_manifest_boundaries_and_source_alignment(self):
        path = ROOT / "experiments" / "v2" / "manifest.csv"
        if not path.exists():
            self.skipTest("Build the frozen manifest first")
        m = pd.read_csv(path, dtype={"hash": str, "dup_key": str}, low_memory=False)
        assert_protocol(m)
        meta = pd.read_csv(ROOT / "cache" / "meta.csv")
        self.assertEqual(m.path.tolist(), meta.path.tolist())
        self.assertEqual(m["hash"].tolist(), meta["hash"].tolist())
        import json
        audit = json.loads((path.parent / "audit.json").read_text())
        self.assertEqual(file_sha256(path), audit["manifest_sha256"])
        # Negative test: a shared component assigned to two parts must be rejected.
        copies = m.copy()
        repeated = copies[copies.part == "support"].component_id.value_counts()
        component = repeated[repeated > 1].index[0]
        row = copies.index[copies.component_id == component][0]
        copies.loc[row, "part"] = "test"
        with self.assertRaisesRegex(AssertionError, "component_id leaks"):
            assert_protocol(copies)

    def test_conflicts_count_minority_labels_not_all_nnhs(self):
        frame = pd.DataFrame({"hash": ["a"] * 4 + ["b"] * 4 + ["c"],
                              "cls": ["NHS", "NNHS", "NNHS", "NNHS",
                                      "NHS", "NHS", "NHS", "NNHS", "HS"]})
        conflict = label_conflicts(frame, ["hash"])
        self.assertEqual(conflict["groups"], 2)
        self.assertEqual(conflict["rows"], 8)
        self.assertEqual(conflict["minimum_unavoidable_errors"], 2)
        self.assertEqual(conflict["rows_by_class"], {"NHS": 4, "HS": 0, "NNHS": 4})


if __name__ == "__main__":
    unittest.main()
