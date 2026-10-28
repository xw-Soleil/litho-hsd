"""Stats on a random sample: decode time, content bbox, number of distinct edge coordinates."""
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
files = sorted(root.rglob("*.png"))
random.seed(0)
sample = random.sample(files, 1500)

t0 = time.time()
mins, maxs, nx, ny, gray = [], [], [], [], 0
for p in sample:
    a = np.asarray(Image.open(p))
    gray += np.count_nonzero((a > 0) & (a < 255))
    b = a > 127
    ys, xs = np.nonzero(b)
    if len(xs):
        mins.append(min(xs.min(), ys.min()))
        maxs.append(max(xs.max(), ys.max()))
    # distinct x where some row changes value / distinct y where some column changes
    nx.append(np.count_nonzero((b[:, 1:] != b[:, :-1]).any(axis=0)))
    ny.append(np.count_nonzero((b[1:, :] != b[:-1, :]).any(axis=1)))
dt = (time.time() - t0) / len(sample)
nx, ny = np.array(nx), np.array(ny)
print(f"{len(files)} files, {dt*1000:.1f} ms/img incl. stats")
print("non-binary pixels:", gray)
print("content min coord:", min(mins), "max coord:", max(maxs))
for name, v in [("x-edges", nx), ("y-edges", ny), ("max(x,y)", np.maximum(nx, ny))]:
    print(f"{name}: mean {v.mean():.1f}  p50 {np.percentile(v,50):.0f}  p90 {np.percentile(v,90):.0f}  "
          f"p99 {np.percentile(v,99):.0f}  max {v.max()}  <=32: {(v<=32).mean():.3f}  <=64: {(v<=64).mean():.3f}")
