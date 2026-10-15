"""有设施但到达人次为 0 的社区位置图。

上图为全市位置，下图为各社区放大：社区边界、社区内社康点位、有到达人次的 500 m 网格（清洗后 OD）。
社区名单取 results/discount_0.5/community_destination_demand.csv 中 located_facilities>0 且 arrivals=0 的社区（三档折扣相同）。
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "supply input"
STEP1 = HERE.parent / "step1_grid_community/results"
STEP2 = HERE.parent / "step2_supply/results"
CRS = 4547
SURFACE, INK, INK2, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
C_COMM, C_ARR = "#2a78d6", "#eb6834"

t = pd.read_csv(RES / "discount_0.5/community_destination_demand.csv", dtype={"SQCODE": str})
zero = t[(t.located_facilities > 0) & (t.arrivals == 0)].reset_index(drop=True)
zero["no"] = np.arange(1, len(zero) + 1)
comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
dist = gpd.read_file(SRC / "shenzhen_district_4547.shp").to_crs(CRS)
zc = comm.merge(zero, on="SQCODE")
od = pd.read_csv(STEP1 / "grid500_origin_250m/outpatient_grid_OD_home_tid_clean_used.csv", encoding="utf-8-sig")
arrive = od.groupby("end_tid")["count"].sum()
g500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
arr = g500[g500.Tid.isin(arrive.index)].assign(arrivals=lambda d: d.Tid.map(arrive))
fac = pd.read_csv(STEP2 / "facility_community.csv", dtype={"SQCODE": str}).merge(
    pd.read_csv(STEP2 / "supply_indicators/facility_supply_audit_2018_mobile_users.csv")[
        ["facility_id", "longitude", "latitude"]], on="facility_id")
fac = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.longitude, fac.latitude), crs=4326).to_crs(CRS)
zf = fac[fac.SQCODE.isin(zero.SQCODE)]
xmin, ymin, xmax, ymax = comm.total_bounds

ncol = 5
nrow = int(np.ceil(len(zero) / ncol))
fig = plt.figure(figsize=(20, 9 + 4.2 * nrow), facecolor=SURFACE)
top = fig.add_axes([0.01, 1 - 8.6 / (9 + 4.2 * nrow), 0.98, 8.2 / (9 + 4.2 * nrow)])
top.set_facecolor(SURFACE)
comm.plot(ax=top, color="#f0efec", edgecolor="#d6d5d0", linewidth=0.2)
arr.plot(ax=top, color=C_ARR, alpha=0.35, linewidth=0)
dist.boundary.plot(ax=top, color=EDGE, linewidth=0.6)
zc.plot(ax=top, color=C_COMM, edgecolor=INK, linewidth=0.8, zorder=3)
zf.plot(ax=top, color="#ffffff", edgecolor=INK, markersize=22, linewidth=0.8, zorder=4)
for r in zc.itertuples():
    p = r.geometry.representative_point()
    top.annotate(f"{r.no} {r.SQNAME}", (p.x, p.y), xytext=(6, 6), textcoords="offset points", fontsize=10, color=INK,
                 zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
top.set_xlim(xmin - 500, xmax + 500)
top.set_ylim(ymin - 500, ymax + 500)
top.set_aspect("equal")
top.set_axis_off()
top.set_title(f"有设施但到达人次为 0 的社区（{len(zero)} 个）", loc="left", fontsize=14, color=INK)
top.legend(handles=[Patch(facecolor=C_COMM, edgecolor=INK, label="有设施但到达人次为 0 的社区"),
                    Patch(facecolor=C_ARR, alpha=0.35, label="有到达人次的 500 m 网格"),
                    Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor="#ffffff", markeredgecolor=INK,
                           label="这些社区的社康")],
           loc="lower left", fontsize=10, frameon=False)

h = 4.0 / (9 + 4.2 * nrow)
for i, r in enumerate(zc.sort_values("no").itertuples()):
    row, col = divmod(i, ncol)
    ax = fig.add_axes([0.01 + col * 0.198, 1 - (8.9 + 4.2 * (row + 1)) / (9 + 4.2 * nrow), 0.19, h])
    x0, y0, x1, y1 = r.geometry.bounds
    pad = max(x1 - x0, y1 - y0) * 0.15 + 1200
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2 + pad
    ax.set_facecolor(SURFACE)
    comm.plot(ax=ax, color="#f0efec", edgecolor="#c9c8c3", linewidth=0.4)
    arr.plot(ax=ax, color=C_ARR, alpha=0.35, edgecolor=C_ARR, linewidth=0.4)
    gpd.GeoSeries([r.geometry], crs=CRS).plot(ax=ax, color=C_COMM, alpha=0.35, edgecolor=INK, linewidth=1.4, zorder=3)
    zf[zf.SQCODE == r.SQCODE].plot(ax=ax, color="#ffffff", edgecolor=INK, markersize=40, linewidth=1.0, zorder=4)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    near = arr.cx[cx - half:cx + half, cy - half:cy + half]
    ax.set_title(f"{r.no} {r.SQNAME}（{len(zf[zf.SQCODE == r.SQCODE])} 个社康，图内有到达网格 {len(near)} 个）",
                 loc="left", fontsize=10, color=INK, pad=3)
fig.savefig(RES / "fig_zero_arrival_communities.png", dpi=150, facecolor=SURFACE)
print(zero[["no", "SQNAME", "SQCODE"]].to_string(index=False))
