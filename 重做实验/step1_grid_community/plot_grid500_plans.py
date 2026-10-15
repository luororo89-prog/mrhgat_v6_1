"""绘制 500 m 网格两种方案下的需求人口空间分布。a 为方案一（小社区认领），b 为方案二（小社区并入）。"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/grid500_nest"
SRC = HERE.parent / "input"
COMM = SRC / "深圳社区objectid.shp"
DIST = SRC / "shenzhen_district_4547.shp"

SURFACE, INK, INK2, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
BINS = [0, 1500, 3000, 6000, 12000, 24000, np.inf]
LABELS = ["< 1,500", "1,500 至 3,000", "3,000 至 6,000", "6,000 至 12,000", "12,000 至 24,000", "≥ 24,000"]

comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(4547)
dist = gpd.read_file(DIST).to_crs(4547)

s1 = pd.read_csv(OUT / "plan1_community_demand.csv", dtype={"SQCODE": str}, encoding="utf-8-sig")
g1 = comm.merge(s1[["SQCODE", "需求人口_500m"]], on="SQCODE", validate="one_to_one")

m2 = pd.read_csv(OUT / "plan2_community_to_node.csv", dtype=str)
s2 = pd.read_csv(OUT / "plan2_node_demand.csv", dtype={"SQCODE": str}, encoding="utf-8-sig")
g2 = comm.merge(m2, on="SQCODE", validate="one_to_one").dissolve(by="node_SQCODE").reset_index()
g2 = g2.merge(s2[["SQCODE", "需求人口_500m"]].rename(columns={"SQCODE": "node_SQCODE"}), on="node_SQCODE",
              validate="one_to_one")

panels = [(g1, "a  方案一  小社区认领 500m 网格（666 个节点）"),
          (g2, "b  方案二  小社区并入大社区（634 个节点）")]
fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), facecolor=SURFACE)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.17, wspace=0.03)
for ax, (g, title) in zip(axes, panels):
    cls = pd.cut(g.需求人口_500m, BINS, labels=False, right=False).astype(int)
    counts = cls.value_counts().reindex(range(len(SEQ)), fill_value=0)
    ax.set_facecolor(SURFACE)
    g.plot(ax=ax, color=[SEQ[c] for c in cls], edgecolor=SURFACE, linewidth=0.15)
    dist.boundary.plot(ax=ax, color=EDGE, linewidth=0.6)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=6)
    v = g.需求人口_500m
    ax.text(0.68, 0.97, f"合计 {v.sum():,.0f} 人\n中位数 {v.median():,.0f} 人", transform=ax.transAxes,
            va="top", fontsize=9.5, color=INK2, linespacing=1.5)
    handles = [Patch(facecolor=c, edgecolor=EDGE, linewidth=0.4, label=f"{l}（{n} 个）")
               for c, l, n in zip(SEQ, LABELS, counts)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=9.5, frameon=False,
              labelcolor=INK2, title="节点需求人口（联通人口，人）", title_fontsize=9.5, alignment="left")

fig.savefig(OUT / "fig_plans_demand.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(OUT / "fig_plans_demand.png")
