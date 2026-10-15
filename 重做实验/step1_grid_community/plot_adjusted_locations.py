"""微调中受影响社区（景田、松岗、洪浪、沙埔头、梅都、龙井、锦龙）的位置图，附医院面域与改动的 500 m 网格。"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "input"
CRS = 4547
SURFACE, INK, INK2, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
GAIN, LOSE, SMALL, HOSP = "#eb6834", "#2a78d6", "#4a3aa7", "#d03b3b"

comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "SQNAME", "geometry"]].to_crs(CRS)
dist = gpd.read_file(SRC / "shenzhen_district_4547.shp").to_crs(CRS)
grid = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
aoi = gpd.read_file(HERE.parent / "input"
                    / "hospital_aoi_35.shp").to_crs(CRS)
log = pd.read_csv(RES / "grid500_nest/plan1_hospital_change_log.csv", dtype={"SQCODE": str, "SQCODE_plan1": str})
changed = grid.merge(log[["Tid", "原社区", "新社区"]], on="Tid")

groups = {"拿到医院网格后需求人口偏多": ["景田", "松岗", "洪浪"],
          "失去全部门诊流入": ["沙埔头", "梅都", "龙井"],
          "补回后偏多 72%": ["锦龙"]}
color = {n: c for (g, ns), c in zip(groups.items(), [GAIN, LOSE, SMALL]) for n in ns}
focus = comm[comm.SQNAME.isin(color)].copy()
focus["c"] = focus.SQNAME.map(color)

zooms = [("b  福田北（景田、梅都）", ["景田", "梅都"], 700),
         ("c  福田南（锦龙、沙埔头）", ["锦龙", "沙埔头"], 600),
         ("d  宝安南部（洪浪、龙井）", ["洪浪", "龙井"], 700),
         ("e  宝安松岗（松岗）", ["松岗"], 700)]
OFFSET = {"松岗": (0, 16), "洪浪": (-34, 14), "龙井": (-34, -16), "景田": (-44, -6), "梅都": (-10, 22),
          "锦龙": (10, -24), "沙埔头": (44, 10)}

fig = plt.figure(figsize=(18, 11.5), facecolor=SURFACE)
gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.1, wspace=0.05)
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor(SURFACE)
comm.plot(ax=ax0, color="#f0efec", edgecolor=SURFACE, linewidth=0.2)
dist.boundary.plot(ax=ax0, color=EDGE, linewidth=0.6)
focus.plot(ax=ax0, color=focus.c, edgecolor=INK, linewidth=0.4)
for r in dist.itertuples():
    p = r.geometry.representative_point()
    ax0.text(p.x, p.y, r.dt_name, fontsize=10, color=INK2, ha="center", va="center", alpha=0.8)
for r in focus.itertuples():
    p = r.geometry.representative_point()
    ax0.annotate(r.SQNAME, (p.x, p.y), xytext=OFFSET[r.SQNAME], textcoords="offset points", fontsize=10, color=INK, ha="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor=EDGE, linewidth=0.4))
for (title, names, _) in zooms:
    b = comm[comm.SQNAME.isin(names)].total_bounds
    ax0.add_patch(plt.Rectangle((b[0] - 1500, b[1] - 1500), b[2] - b[0] + 3000, b[3] - b[1] + 3000, fill=False,
                                edgecolor=INK2, linewidth=0.8, linestyle=(0, (3, 2))))
    ax0.text(b[0] - 1500, b[3] + 1700, title[0], fontsize=11, color=INK2)
ax0.set_axis_off()
ax0.set_title("a  受影响社区的位置", loc="left", fontsize=13, color=INK)
legend = [Patch(facecolor=c, edgecolor=INK, linewidth=0.4, label=f"{g}（{'、'.join(ns)}）")
          for (g, ns), c in zip(groups.items(), [GAIN, LOSE, SMALL])]
ax0.legend(handles=legend, loc="lower right", fontsize=10, frameon=False, labelcolor=INK2)

for k, (title, names, pad) in enumerate(zooms):
    ax = fig.add_subplot(gs[1, k])
    ax.set_facecolor(SURFACE)
    b = comm[comm.SQNAME.isin(names)].total_bounds
    ext = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    comm.plot(ax=ax, color="#f0efec", edgecolor="#c9c8c3", linewidth=0.5)
    focus.plot(ax=ax, color=focus.c, alpha=0.55, edgecolor=INK, linewidth=0.6)
    grid.boundary.plot(ax=ax, color="#d9d8d3", linewidth=0.3)
    changed.boundary.plot(ax=ax, color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
    aoi.plot(ax=ax, facecolor="none", edgecolor=HOSP, linewidth=1.4)
    for r in comm[comm.geometry.representative_point().within(
            gpd.GeoSeries.from_xy([ext[0]], [ext[1]]).buffer(1).envelope.iloc[0].union(
                gpd.GeoSeries.from_xy([ext[2]], [ext[3]]).iloc[0]).envelope)].itertuples():
        p = r.geometry.representative_point()
        bold = r.SQNAME in color
        ax.text(p.x, p.y, r.SQNAME, fontsize=11 if bold else 7.5, color=INK if bold else "#8a8984", ha="center", va="center",
                fontweight="bold" if bold else "normal")
    for r in aoi.itertuples():
        p = r.geometry.representative_point()
        if ext[0] < p.x < ext[2] and ext[1] < p.y < ext[3]:
            ax.annotate(r.fac_name.replace("深圳市", "").replace("（深圳福田）", "").replace("（福田）", ""), (p.x, p.y),
                        xytext=(0, -14), textcoords="offset points", fontsize=8, color=HOSP, ha="center",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor=SURFACE, edgecolor="none", alpha=0.8))
    ax.set_xlim(ext[0], ext[2]); ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=12, color=INK)
fig.legend(handles=[Line2D([], [], color=HOSP, lw=1.4, label="医院面域"),
                    Line2D([], [], color=INK, lw=1.2, linestyle=(0, (4, 2)), label="微调中改动的 500m 网格"),
                    Line2D([], [], color="#d9d8d3", lw=1, label="500m 网格")],
           loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
out = RES / "grid500_nest/fig_adjusted_locations.png"
fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(out)
