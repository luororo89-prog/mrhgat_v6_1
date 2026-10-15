"""绘制各社区需求人口（联通人口）的空间分布。a 为 maup 归属，b 为现有需求人口（网格中心点归属）。"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/maup_demand"
SRC = HERE.parent / "input"
COMM = SRC / "深圳社区objectid.shp"
DIST = SRC / "shenzhen_district_4547.shp"

SURFACE, INK, INK2, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
BINS = [0, 1500, 3000, 6000, 12000, 24000, np.inf]
LABELS = ["< 1,500", "1,500 至 3,000", "3,000 至 6,000", "6,000 至 12,000", "12,000 至 24,000", "≥ 24,000"]

t = pd.read_csv(OUT / "community_demand_population_maup.csv", dtype={"SQCODE": str}, encoding="utf-8-sig")
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(4547).merge(t, on="SQCODE", validate="one_to_one")
dist = gpd.read_file(DIST).to_crs(4547)

panels = [("需求人口_联通_maup", "a  maup 归属（最大重叠面积）"),
          ("需求人口_联通_现有供给表", "b  现有需求人口（网格中心点归属）")]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), facecolor=SURFACE)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.17, wspace=0.03)
for ax, (col, title) in zip(axes, panels):
    cls = pd.cut(comm[col], BINS, labels=False, right=False).astype(int)
    counts = cls.value_counts().reindex(range(len(SEQ)), fill_value=0)
    ax.set_facecolor(SURFACE)
    comm.plot(ax=ax, color=[SEQ[c] for c in cls], edgecolor=SURFACE, linewidth=0.15)
    dist.boundary.plot(ax=ax, color=EDGE, linewidth=0.6)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=6)
    v = comm[col]
    ax.text(0.0, 0.97, f"合计 {v.sum():,.0f} 人\n中位数 {v.median():,.0f} 人", transform=ax.transAxes,
            va="top", fontsize=9.5, color=INK2, linespacing=1.5)
    handles = [Patch(facecolor=c, edgecolor=EDGE, linewidth=0.4, label=f"{l}（{n} 个）")
               for c, l, n in zip(SEQ, LABELS, counts)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=9.5, frameon=False,
              labelcolor=INK2, title="社区需求人口（联通人口，人）", title_fontsize=9.5, alignment="left")

fig.savefig(OUT / "fig_maup_demand.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(OUT / "fig_maup_demand.png")
