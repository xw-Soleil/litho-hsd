"""Check how the .png1-.png9 variants relate to their plain center clip (shift? flip?)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])


def load(p):
    return (np.asarray(Image.open(p)) > 127).astype(np.float32)


def bbox(a):
    ys, xs = np.nonzero(a)
    return (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else None


def best_shift(a, b, maxs=400):
    """Integer (dy, dx) such that b ~= a shifted by (dy, dx), via FFT cross-correlation."""
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    c = np.fft.irfft2(np.conj(fa) * fb, s=a.shape)
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    dy = dy - a.shape[0] if dy > a.shape[0] // 2 else dy
    dx = dx - a.shape[1] if dx > a.shape[1] // 2 else dx
    return int(dy), int(dx)


cases = [
    ("iccad1", "HS", 60, "NHS"),
    ("iccad1", "NHS", 473, "NNHS"),
    ("iccad3", "HS", 100, "NHS"),
    ("iccad2", "NHS", 3000, "NNHS"),
]
for bench, ccls, idx, vcls in cases:
    d = root / bench / "test"
    pre = "" if bench == "iccad1" else "CAD"
    center = load(d / f"{ccls}{pre}{idx}.png")
    print(f"\n{bench} {ccls}{idx}: bbox={bbox(center)} white={center.mean():.3f}")
    for k in [1, 2, 3, 4, 6, 7, 8, 9]:
        v = load(d / f"{vcls}{pre}{idx}.png{k}.png")
        dy, dx = best_shift(center, v)
        # overlap agreement after undoing the shift, inside the common window
        sh = np.roll(center, (dy, dx), axis=(0, 1))
        win = (slice(88 + abs(dy), 1112 - abs(dy)), slice(88 + abs(dx), 1112 - abs(dx)))
        agree = (sh[win] == v[win]).mean()
        print(f"  png{k}: shift(dy,dx)=({dy:4d},{dx:4d})  bbox={bbox(v)}  agree={agree:.4f}  white={v.mean():.3f}")
