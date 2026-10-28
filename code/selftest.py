"""Sanity checks: split sizes, squish<->density consistency under D4 augmentation, throughput."""
import time

import numpy as np
import torch

from common import DEV, Arrays, Classifier, d4, load_meta, sync, to_input

m = load_meta()
print(m.groupby(["part", "cls"]).size().unstack(fill_value=0))

A = Arrays()
idx = np.sort(np.random.RandomState(1).choice(len(m), 16, replace=False))


def render(t, dx, dy):
    """squish -> 1200x1200 binary -> 120x120 box-mean, on CPU."""
    t, dx, dy = t.cpu().numpy(), dx.cpu().numpy(), dy.cpu().numpy()
    out = []
    for k in range(len(t)):
        img = np.repeat(np.repeat(t[k], dy[k], axis=0), dx[k], axis=1)
        out.append(np.round(img.reshape(120, 10, 120, 10).mean((1, 3)) * 255))
    return np.stack(out)


sq, de = A.raw("squish", idx), A.raw("dens", idx)
for mode in range(8):
    a = render(*d4(sq, "squish", mode))
    b = d4(de, "dens", mode)[0].cpu().numpy()
    print(f"D4 mode {mode}: max |squish-render - dens| = {np.abs(a - b).max():.0f}")

for rep, bs in [("dens", 512), ("squish", 512)]:
    net = Classifier(rep, 2).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    ridx = np.sort(np.random.RandomState(2).choice(len(m), bs, replace=False))
    raw = A.raw(rep, ridx)
    y = torch.zeros(bs, dtype=torch.long, device=DEV)
    for it in range(13):
        if it == 3:
            sync(); t0 = time.time()
        loss = torch.nn.functional.cross_entropy(net(to_input(d4(raw, rep), rep)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    sync(); tr = 10 * bs / (time.time() - t0)
    net.eval()
    with torch.no_grad():
        for it in range(13):
            if it == 3:
                sync(); t0 = time.time()
            net(to_input(raw, rep))
    sync(); inf = 10 * bs / (time.time() - t0)
    print(f"{rep}: train {tr:.0f} img/s, inference {inf:.0f} img/s, params {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
