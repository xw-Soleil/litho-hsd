"""Inspect the file-naming structure of the ICCAD-2012 png dataset."""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
pat = re.compile(r"^(NNHS|NHS|HS)(CAD)?(\d+)\.png(\d)?(?:\.png)?$")

stats = defaultdict(Counter)
ids = defaultdict(set)
variants_per_id = defaultdict(Counter)
bad = []
for p in root.rglob("*.png"):
    bench, split = p.parts[-3], p.parts[-2]
    m = pat.match(p.name)
    if not m:
        bad.append(p.name)
        continue
    cls, _, idx, var = m.groups()
    key = (bench, split, cls)
    stats[key]["files"] += 1
    stats[key]["variant" if var else "plain"] += 1
    ids[key].add(int(idx))
    variants_per_id[key][(int(idx))] += 1

print("unmatched:", len(bad), bad[:5])
print(f"{'bench':7}{'split':6}{'cls':5}{'files':>7}{'plain':>7}{'var':>7}{'ids':>6}  id-range  files/id")
for key in sorted(stats):
    s = stats[key]
    i = ids[key]
    per = Counter(variants_per_id[key].values())
    print(f"{key[0]:7}{key[1]:6}{key[2]:5}{s['files']:7}{s['plain']:7}{s['variant']:7}{len(i):6}  {min(i)}-{max(i)}  {dict(per)}")

# Do NNHS ids overlap with HS / NHS ids of the same benchmark & split?
print("\nid overlap in test split:")
for b in sorted({k[0] for k in stats}):
    nn = ids.get((b, "test", "NNHS"), set())
    hs = ids.get((b, "test", "HS"), set())
    nh = ids.get((b, "test", "NHS"), set())
    print(b, "NNHS&HS", len(nn & hs), "NNHS&NHS", len(nn & nh), "HS&NHS", len(hs & nh))
