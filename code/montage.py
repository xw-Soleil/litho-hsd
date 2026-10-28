"""3x3 montage: center clip at slot 5, variants .png1-.png9 around it (slot k = variant k)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root, out = Path(sys.argv[1]), Path(sys.argv[2])
cases = [tuple(c.split(":")) for c in sys.argv[3:]]  # bench:centerCls:variantCls:id

S = 300
for bench, ccls, vcls, idx in cases:
    d = root / bench / "test"
    pre = "" if bench == "iccad1" else "CAD"
    canvas = Image.new("L", (3 * S + 4, 3 * S + 4), 128)
    for k in range(1, 10):
        p = d / (f"{ccls}{pre}{idx}.png" if k == 5 else f"{vcls}{pre}{idx}.png{k}.png")
        im = Image.open(p).convert("L").resize((S, S), Image.BOX)
        r, c = divmod(k - 1, 3)
        canvas.paste(im, (c * (S + 2), r * (S + 2)))
        ImageDraw.Draw(canvas).text((c * (S + 2) + 5, r * (S + 2) + 5), str(k), fill=128)
    canvas.save(out / f"montage_{bench}_{ccls}{idx}.png")
    print("saved", out / f"montage_{bench}_{ccls}{idx}.png")
