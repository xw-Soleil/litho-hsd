"""One pass over all 165k layout clips (1200x1200 binary PNG) -> compact cached representations.

Outputs in <cache>/:
  meta.csv        one row per image: path, bench, split, cls, id, variant, group, is_center, hash, n_edges, lossless
  dens120.npy     uint8 [N,120,120]   area-averaged density map (10 nm/px, exact 10x10 box mean)
  sq_topo.npy     uint8 [N,64,64]     adaptive squish pattern: polygon occupancy on a non-uniform grid
  sq_dx.npy       uint16 [N,64]       column widths of the squish grid (px)
  sq_dy.npy       uint16 [N,64]       row heights of the squish grid (px)
"""
import csv
import hashlib
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

SIZE, DS, NSQ = 1200, 10, 64
PAT = re.compile(r"^(NNHS|NHS|HS)(?:CAD)?(\d+)\.png(\d)?(?:\.png)?$")


def squish_lines(edges, n):
    """Grid lines {0, edges, SIZE}; split widest interval (or merge narrowest) until exactly n intervals."""
    lines = [0, *edges.tolist(), SIZE]
    while len(lines) - 1 < n:
        w = np.diff(lines)
        k = int(np.argmax(w))
        lines.insert(k + 1, lines[k] + w[k] // 2)
    while len(lines) - 1 > n:  # lossy fallback, never hit on this dataset (max ~50 edges)
        w = np.diff(lines)
        k = int(np.argmin(w[1:-1])) + 1
        lines.pop(k)
    return np.array(lines)


def process(args):
    i, path = args
    b = np.asarray(Image.open(path)) > 127
    dens = b.reshape(SIZE // DS, DS, SIZE // DS, DS).mean(axis=(1, 3))
    ex = np.nonzero((b[:, 1:] != b[:, :-1]).any(axis=0))[0] + 1
    ey = np.nonzero((b[1:, :] != b[:-1, :]).any(axis=1))[0] + 1
    X, Y = squish_lines(ex, NSQ), squish_lines(ey, NSQ)
    cx, cy = (X[:-1] + X[1:]) // 2, (Y[:-1] + Y[1:]) // 2
    topo = b[np.ix_(cy, cx)]
    h = hashlib.md5(np.packbits(b).tobytes()).hexdigest()[:16]
    return i, np.round(dens * 255).astype(np.uint8), topo.astype(np.uint8), np.diff(X), np.diff(Y), h, max(len(ex), len(ey))


def main():
    root, cache = Path(sys.argv[1]), Path(sys.argv[2])
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.png"))
    rows = []
    for p in files:
        bench, split = p.parts[-3], p.parts[-2]
        cls, idx, var = PAT.match(p.name).groups()
        var = int(var) if var else 5
        rows.append(dict(path=str(p.relative_to(root)), bench=bench, split=split, cls=cls, id=int(idx),
                         variant=var, group=f"{bench}/{split}/{idx}", is_center=int(var == 5)))
    n = len(rows)
    dens = np.lib.format.open_memmap(cache / "dens120.npy", "w+", np.uint8, (n, SIZE // DS, SIZE // DS))
    topo = np.lib.format.open_memmap(cache / "sq_topo.npy", "w+", np.uint8, (n, NSQ, NSQ))
    dx = np.lib.format.open_memmap(cache / "sq_dx.npy", "w+", np.uint16, (n, NSQ))
    dy = np.lib.format.open_memmap(cache / "sq_dy.npy", "w+", np.uint16, (n, NSQ))
    with Pool(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()) as pool:
        for k, (i, d, t, wx, wy, h, ne) in enumerate(
                pool.imap_unordered(process, enumerate(files), chunksize=64)):
            dens[i], topo[i], dx[i], dy[i] = d, t, wx, wy
            rows[i]["hash"], rows[i]["n_edges"], rows[i]["lossless"] = h, ne, int(ne + 1 <= NSQ)
            if k % 20000 == 0:
                print(f"{k}/{n}", flush=True)
    for a in (dens, topo, dx, dy):
        a.flush()
    with open(cache / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("done", n)


if __name__ == "__main__":
    main()
