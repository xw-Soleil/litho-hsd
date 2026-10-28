"""Explanatory figures: (1) high-resolution handling: raw vs density vs squish, (2) dataset neighbourhood structure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import ROOT, load_meta

DATA = ROOT / "data" / "iccad-official"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
C_HS, C_NHS, C_NN, INK, INK2, SURF = "#eb6834", "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": ["PingFang SC", "sans-serif"], "axes.unicode_minus": False,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10})
m = load_meta()

# ---------------------------------------------------------------- (1) representation
row = m[(m.split == "test") & (m.bench == "iccad3") & (m.cls == "HS")].iloc[7]
i = int(row.name)
b = np.asarray(Image.open(DATA / row.path)) > 127
dens = np.load(ROOT / "cache" / "dens120.npy", mmap_mode="r")[i]
topo = np.load(ROOT / "cache" / "sq_topo.npy", mmap_mode="r")[i]
dx = np.load(ROOT / "cache" / "sq_dx.npy", mmap_mode="r")[i].astype(int)
dy = np.load(ROOT / "cache" / "sq_dy.npy", mmap_mode="r")[i].astype(int)
X, Y = np.r_[0, np.cumsum(dx)], np.r_[0, np.cumsum(dy)]

fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
ax[0].imshow(b, cmap="gray", interpolation="nearest")
for x in X[1:-1]:
    ax[0].axvline(x - 0.5, color=C_HS, lw=0.5, alpha=0.8)
for yy in Y[1:-1]:
    ax[0].axhline(yy - 0.5, color=C_HS, lw=0.5, alpha=0.8)
ax[0].set_title("原始版图 1200×1200 = 1,440,000 像素\n橙线 = squish 网格（多边形边坐标，再补齐到 64 格）", fontsize=10, loc="left")
ax[1].imshow(dens, cmap="gray", interpolation="nearest")
ax[1].set_title("表示 A：面积平均密度图 120×120\n10 nm/像素，灰度保留亚像素边位置", fontsize=10, loc="left")
ax[2].imshow(topo, cmap="gray", interpolation="nearest")
ax[2].set_title(f"表示 B：Squish 拓扑 64×64 + 行高/列宽\n无损（100% 样本可完全还原），压缩 ≈ 350×", fontsize=10, loc="left")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("如何处理 1200×1200 高分辨率版图", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "representation.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- (2) neighbourhood structure
def grid(bench, center_cls, nb_cls, idx):
    pre = "" if bench == "iccad1" else "CAD"
    d = DATA / bench / "test"
    out = []
    for k in range(1, 10):
        p = d / (f"{center_cls}{pre}{idx}.png" if k == 5 else f"{nb_cls}{pre}{idx}.png{k}.png")
        out.append(np.asarray(Image.open(p).convert("L").resize((200, 200), Image.BOX)))
    return out


hs_id = int(m[(m.split == "test") & (m.bench == "iccad3") & (m.cls == "HS")].iloc[7].id)
nhs_id = int(m[(m.split == "test") & (m.bench == "iccad3") & (m.cls == "NHS") & (m.is_center == 1)
               & m.id.isin(m[m.cls == "NNHS"].id)].iloc[11].id)
fig, axes = plt.subplots(3, 7, figsize=(13, 6), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.25, 1, 1, 1]))
for blk, (cc, nc, idx, ccol, ncol, title) in enumerate([
        ("HS", "NHS", hs_id, C_HS, C_NHS, f"热点 HS{hs_id} 的邻域：中心=HS，8 个邻窗被标成 NHS"),
        ("NHS", "NNHS", nhs_id, C_NHS, C_NN, f"非热点 NHS{nhs_id} 的邻域：中心=NHS，8 个邻窗被标成 NNHS")]):
    ims = grid("iccad3", cc, nc, idx)
    for k in range(9):
        a = axes[k // 3, blk * 4 + k % 3]
        a.imshow(ims[k], cmap="gray", vmin=0, vmax=255)
        col = ccol if k == 4 else ncol
        for s in a.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3 if k == 4 else 1.5)
        a.set_xticks([]); a.set_yticks([])
        a.text(6, 24, (cc if k == 4 else nc) + (f"{idx}.png" if k == 4 else f".png{k+1}"), color="white", fontsize=7,
               bbox=dict(facecolor=col, edgecolor="none", pad=1.5))
    axes[0, blk * 4 + 1].set_title(title, fontsize=10, color=INK)
for r in range(3):
    axes[r, 3].axis("off")
fig.suptitle("数据集真实结构：测试集中每个中心窗口都配有 8 个相邻窗口（.png1–.png9，5 为中心）", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(FIG / "dataset_structure.png", dpi=200)
plt.close(fig)
print("ok")
