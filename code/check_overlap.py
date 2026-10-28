"""Find the offset between a center clip and its left/right/up/down neighbour windows."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
W0, W1 = 88, 1112  # drawable window inside the 1200x1200 canvas


def load(p):
    return np.asarray(Image.open(p)) > 127


def scan(center, nb, axis, sign):
    """Return offsets s (px) where nb's window equals center's window moved by s along axis."""
    c = center[W0:W1, W0:W1]
    n = nb[W0:W1, W0:W1]
    hits = []
    for s in range(8, 1016, 4):
        if axis == 1:
            a, b = (c[:, s:], n[:, :-s]) if sign > 0 else (c[:, :-s], n[:, s:])
        else:
            a, b = (c[s:, :], n[:-s, :]) if sign > 0 else (c[:-s, :], n[s:, :])
        if a.any() and (a == b).mean() > 0.999:
            hits.append(s)
    return hits[:6]


bench, ccls, vcls, idx = sys.argv[2].split(":")
d = root / bench / "test"
pre = "" if bench == "iccad1" else "CAD"
center = load(d / f"{ccls}{pre}{idx}.png")
# slot 6 = right neighbour, 4 = left, 2 = up, 8 = down (row-major 3x3, 5 = center)
for k, axis, sign in [(6, 1, +1), (4, 1, -1), (8, 0, +1), (2, 0, -1)]:
    nb = load(d / f"{vcls}{pre}{idx}.png{k}.png")
    print(f"png{k}: matching offsets {scan(center, nb, axis, sign)}")
