"""dens120m.npy: density maps where every clip is rendered like the official test clips
(content only inside [88, 1112), black border outside). Test rows are unchanged; train rows are re-rendered."""
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

W0, W1 = 88, 1112


def render(path):
    b = np.asarray(Image.open(path)) > 127
    b[:W0], b[W1:], b[:, :W0], b[:, W1:] = False, False, False, False
    return np.round(b.reshape(120, 10, 120, 10).mean(axis=(1, 3)) * 255).astype(np.uint8)


if __name__ == "__main__":
    root, cache = Path(sys.argv[1]), Path(sys.argv[2])
    m = pd.read_csv(cache / "meta.csv")
    shutil.copyfile(cache / "dens120.npy", cache / "dens120m.npy")
    out = np.load(cache / "dens120m.npy", mmap_mode="r+")
    rows = np.nonzero((m.split == "train").values)[0]
    with Pool(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()) as pool:
        for i, d in zip(rows, pool.imap(render, [root / m.path[i] for i in rows], chunksize=64)):
            out[i] = d
    out.flush()
    # test rows must be untouched by the mask
    te = np.nonzero((m.split == "test").values)[0][::97]
    same = all(np.array_equal(render(root / m.path[i]), out[i]) for i in te)
    print("train rows re-rendered:", len(rows), "| test rows unchanged by mask:", same)
