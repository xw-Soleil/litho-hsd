"""Data audit: exact duplicates, train/test leakage, label conflicts, squish losslessness."""
import sys
from pathlib import Path

import pandas as pd

cache, out = Path(sys.argv[1]), Path(sys.argv[2])
m = pd.read_csv(cache / "meta.csv")
m["role"] = m.cls.where(m.is_center == 1, m.cls + "(nbr)")  # nbr = one of the 8 neighbour windows
lines = []
p = lambda *a: lines.append(" ".join(str(x) for x in a))

p("## squish-64 lossless fraction:", f"{m.lossless.mean():.5f}", "max edges:", m.n_edges.max())
p("\n## counts by split/role")
p(m.groupby(["split", "role"]).size().to_string())

p("\n## exact-duplicate structure")
p("unique hashes:", m.hash.nunique(), "of", len(m))
labels_per_hash = m.groupby("hash").cls.nunique()
p("hashes carrying >1 class label:", int((labels_per_hash > 1).sum()))

tr = m[m.split == "train"]
te = m[m.split == "test"]
tr_lab = tr.groupby("hash").cls.agg(lambda s: "/".join(sorted(set(s))))
te = te.assign(train_label=te.hash.map(tr_lab))
p("\n## test images whose exact layout also appears in TRAIN (leakage), by role -> train label")
p(te.groupby(["role", te.train_label.fillna("-")]).size().unstack(fill_value=0).to_string())

p("\n## within TEST: NNHS images identical to a test HS-center image")
hs_hash = set(te[(te.cls == "HS")].hash)
p("NNHS == some HS:", int(te[te.cls == "NNHS"].hash.isin(hs_hash).sum()),
  "| NHS(nbr) == some HS:", int(te[(te.cls == "NHS") & (te.is_center == 0)].hash.isin(hs_hash).sum()),
  "| NHS(center) == some HS:", int(te[(te.cls == "NHS") & (te.is_center == 1)].hash.isin(hs_hash).sum()))
p("empty (all-black) clips by role:")
empty_hash = m.loc[m.n_edges == 0, "hash"].unique()
p(m[m.hash.isin(empty_hash)].groupby(["split", "role"]).size().to_string())

txt = "\n".join(lines)
print(txt)
(out / "audit.txt").write_text(txt)
