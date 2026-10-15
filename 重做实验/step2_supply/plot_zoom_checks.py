"""社康生活圈归属与需求人口的放大检查图。

图 a：社区内有社康、但社康 RA 为 0 的 30 个社区。底图为 ESA WorldCover 2020 地类，叠加社区边界、社区生活圈
（15 分钟步行等时圈）、有人口的 250 m 居住网格、社区人口重心和社康点位。
图 b：社康 SA 最高的东涌、西涌、东山、内伶仃岛四个社区，网格内标注归属该社区的联通人口。
表：每个社康周边 250 m 范围内的地类构成，按是否落在任一社区生活圈内分组汇总。
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "zoom_checks"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
COMM = SRC / "深圳社区objectid.shp"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
ALLOC = HERE.parent / "step1_grid_community/results/landuse_demand/grid250_demand_allocation.csv"
WC = HERE.parent / "step1_grid_community/results/landuse_demand/worldcover_2020_shenzhen.tif"
INPUT_SOURCES = HERE.parents[1] / "1input_sources/等时圈/等时圈wgs84拆分"
ISO = INPUT_SOURCES / "等时圈wgs84拆分/foot等时圈.shp"
CENTER = INPUT_SOURCES / "Shenzhen_人口重心/深圳社区_联通人口重心result.shp"
CRS = 4547
BUFFER_M = 250

WC_CLASS = {10: ("林地", "#006400"), 20: ("灌木", "#ffbb22"), 30: ("草地", "#ffff4c"), 40: ("耕地", "#f096ff"),
            50: ("建设用地", "#fa0000"), 60: ("裸地", "#b4b4b4"), 80: ("水体", "#0064c8"),
            90: ("草本湿地", "#0096a0"), 95: ("红树林", "#00cf75")}
CODES = sorted(WC_CLASS)
CMAP = ListedColormap([WC_CLASS[c][1] for c in CODES])
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

comm = gpd.read_file(COMM)[["SQCODE", "SQNAME", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.make_valid()
iso = gpd.read_file(ISO).to_crs(CRS)
iso["SQCODE"] = iso.num.astype(str).str.replace(r"\.0$", "", regex=True)
iso = pd.concat([iso[["SQCODE", "geometry"]], comm[~comm.SQCODE.isin(iso.SQCODE)][["SQCODE", "geometry"]]])
iso = gpd.GeoDataFrame(iso, geometry="geometry", crs=CRS).set_index("SQCODE")
cen = gpd.read_file(CENTER).to_crs(CRS)
cen["SQCODE"] = cen.SQCODE.astype(str)
cen = cen.set_index("SQCODE")

grid = gpd.read_file(GRID250)[["Tid", "联通人", "geometry"]].to_crs(CRS)
alloc = pd.read_csv(ALLOC, dtype={"SQCODE": str})
alloc = alloc[alloc.联通人_分配 > 0]
gpiece = grid[["Tid", "geometry"]].merge(alloc[["Tid", "SQCODE", "联通人_分配"]], on="Tid")

ind = pd.read_csv(RES / "supply_indicators/community_supply_indicators_2018_mobile_users.csv", dtype={"SQCODE": str})
ind = ind.set_index("SQCODE")
loc = pd.read_csv(RES / "community_located_facilities.csv", dtype={"SQCODE": str}).set_index("SQCODE")
fac = pd.read_csv(RES / "facility_community.csv", dtype={"SQCODE": str})
aud = pd.read_csv(RES / "supply_indicators/facility_supply_audit_2018_mobile_users.csv")
fac = fac.merge(aud[["facility_id", "longitude", "latitude", "candidate_community_count"]], on="facility_id")
fac = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.longitude, fac.latitude), crs=4326).to_crs(CRS)
chc = fac[(fac.tier == "CHC") & fac.SQCODE.notna()].copy()

src = rasterio.open(WC)
vrt = WarpedVRT(src, crs=f"EPSG:{CRS}", resampling=rasterio.enums.Resampling.nearest)


def read_window(bounds):
    win = from_bounds(*bounds, transform=vrt.transform).round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, vrt.width, vrt.height))
    return vrt.read(1, window=win), win


def landcover(ax, bounds):
    arr, win = read_window(bounds)
    bounds = window_bounds(win, vrt.transform)
    idx = np.full(arr.shape, np.nan)
    for i, c in enumerate(CODES):
        idx[arr == c] = i
    ax.imshow(idx, cmap=CMAP, vmin=-0.5, vmax=len(CODES) - 0.5, alpha=0.55, interpolation="nearest",
              extent=(bounds[0], bounds[2], bounds[1], bounds[3]), zorder=0)


def panel(ax, sq, label_pop=False):
    geom = comm.set_index("SQCODE").geometry[sq]
    lc = iso.geometry[sq]
    x0, y0, x1, y1 = np.array(gpd.GeoSeries([geom, lc]).total_bounds) + np.array([-400, -400, 400, 400])
    landcover(ax, (x0, y0, x1, y1))
    comm.boundary.plot(ax=ax, color="#ffffff", linewidth=0.9, zorder=1)
    gp = gpiece.cx[x0:x1, y0:y1]
    other = gp[gp.SQCODE != sq]
    own = gp[gp.SQCODE == sq]
    other.boundary.plot(ax=ax, color="#6b6a66", linewidth=0.4, zorder=2)
    own.boundary.plot(ax=ax, color=INK, linewidth=0.9, zorder=3)
    gpd.GeoSeries([geom], crs=CRS).boundary.plot(ax=ax, color=INK, linewidth=2.0, zorder=4)
    gpd.GeoSeries([lc], crs=CRS).boundary.plot(ax=ax, color="#1c5cab", linewidth=1.6, linestyle="--", zorder=4)
    c = cen.geometry[sq]
    ax.scatter([c.x], [c.y], marker="P", s=70, color="#ffffff", edgecolor=INK, linewidth=0.8, zorder=6)
    cs = chc.cx[x0:x1, y0:y1]
    inc, exc = cs[cs.candidate_community_count > 0], cs[cs.candidate_community_count == 0]
    ax.scatter(inc.geometry.x, inc.geometry.y, s=45, marker="o", color="#ffffff", edgecolor=INK, linewidth=1.0, zorder=6)
    ax.scatter(exc.geometry.x, exc.geometry.y, s=110, marker="*", color="#ffffff", edgecolor=INK, linewidth=0.9, zorder=6)
    if label_pop:
        for _, g in own.iterrows():
            p = g.geometry.centroid
            ax.text(p.x, p.y, f"{g.联通人_分配:.0f}", ha="center", va="center", fontsize=7, color=INK, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    r = ind.loc[sq]
    ax.set_title(f"{r.community_name}  人口 {r.population_demand:,.0f}  RA {r.RA_CHC:.2f}  SA {r.SA_CHC_per_10000:.1f}",
                 loc="left", fontsize=9, color=INK, pad=3)


def legend(fig, y, extra=""):
    handles = [Patch(facecolor=WC_CLASS[c][1], alpha=0.55, label=WC_CLASS[c][0]) for c in CODES]
    handles += [
        Line2D([], [], color=INK, lw=2.0, label="社区边界"),
        Line2D([], [], color="#1c5cab", lw=1.6, ls="--", label="社区生活圈（15 分钟步行等时圈）"),
        Patch(facecolor="none", edgecolor=INK, lw=0.9, label="归属本社区的有人口 250 m 网格" + extra),
        Patch(facecolor="none", edgecolor="#6b6a66", lw=0.4, label="归属其他社区的有人口网格"),
        Line2D([], [], marker="P", ls="", markersize=8, markerfacecolor="#ffffff", markeredgecolor=INK, label="社区人口重心"),
        Line2D([], [], marker="o", ls="", markersize=6.5, markerfacecolor="#ffffff", markeredgecolor=INK,
               label="社康，落在某个社区生活圈内"),
        Line2D([], [], marker="*", ls="", markersize=11, markerfacecolor="#ffffff", markeredgecolor=INK,
               label="社康，不在任何社区生活圈内"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, y), ncol=8, fontsize=9, frameon=False)


# ---------- 图 a：社区内有社康但 RA 为 0 的社区 ----------
zero = loc[(loc.located_count_CHC > 0)].index.intersection(ind[ind.facility_count_CHC == 0].index)
zero = sorted(zero, key=lambda s: s)
ncol = 6
nrow = int(np.ceil(len(zero) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(24, 4.3 * nrow + 1.2), facecolor=SURFACE)
fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.07, wspace=0.04, hspace=0.12)
fig.suptitle(f"社区内有社康但社康 RA 为 0 的社区（{len(zero)} 个）", x=0.005, ha="left", fontsize=14, color=INK)
for ax, sq in zip(axes.flat, zero):
    panel(ax, sq)
for ax in axes.flat[len(zero):]:
    ax.set_axis_off()
legend(fig, 0.005)
fig.savefig(OUT / "fig_zero_ra_communities.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# ---------- 图 b：社康 SA 最高的四个社区 ----------
names = ["东涌", "西涌", "东山", "内伶仃岛"]
top = [ind.index[ind.community_name == n][0] for n in names]
fig, axes = plt.subplots(2, 2, figsize=(16, 15), facecolor=SURFACE)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.08, wspace=0.04, hspace=0.1)
fig.suptitle("社康 SA 最高的四个社区（网格内数字为归属本社区的联通人口）", x=0.01, ha="left", fontsize=14, color=INK)
for ax, sq in zip(axes.flat, top):
    panel(ax, sq, label_pop=True)
legend(fig, 0.01, "，数字为人口")
fig.savefig(OUT / "fig_top_sa_communities.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# 四个社区的需求人口与生活圈人口
grid_c = grid.copy()
grid_c["geometry"] = grid_c.geometry.centroid
rows = []
for sq in top:
    lc = iso.geometry[sq]
    in_lc = grid_c[grid_c.within(lc) & (grid_c.联通人 > 0)]
    own = alloc[alloc.SQCODE == sq]
    r = ind.loc[sq]
    rows.append({"社区": r.community_name, "SQCODE": sq, "社区需求人口": r.population_demand,
                 "归属本社区的有人口网格数": len(own),
                 "生活圈内有人口网格数": len(in_lc), "生活圈人口": float(in_lc.联通人.sum()),
                 "生活圈社康数": int(r.facility_count_CHC), "社康服务能力合计": r.capacity_sum_CHC,
                 "SA_每万人": r.SA_CHC_per_10000,
                 "社区面积_km2": comm.set_index("SQCODE").geometry[sq].area / 1e6})
pd.DataFrame(rows).to_csv(OUT / "top_sa_communities.csv", index=False, encoding="utf-8-sig")

# ---------- 表：社康周边地类 ----------
recs = []
for _, f in chc.iterrows():
    buf = f.geometry.buffer(BUFFER_M)
    b = buf.bounds
    arr, win = read_window(b)
    tr = vrt.window_transform(win)
    m = ~geometry_mask([buf], out_shape=arr.shape, transform=tr)
    vals = arr[m]
    vals = vals[vals > 0]
    rec = {"facility_id": f.facility_id, "facility_name": f.facility_name, "SQCODE": f.SQCODE,
           "in_any_life_circle": bool(f.candidate_community_count > 0)}
    for c in CODES:
        rec[f"share_{WC_CLASS[c][0]}"] = float((vals == c).mean()) if len(vals) else np.nan
    recs.append(rec)
lcov = pd.DataFrame(recs)
gc = grid[grid.联通人 > 0].copy()
gc["geometry"] = gc.geometry.centroid
near = gpd.sjoin_nearest(chc[["facility_id", "geometry"]], gc[["geometry"]], distance_col="d")
lcov["dist_to_populated_grid_centre_m"] = lcov.facility_id.map(near.groupby("facility_id").d.min())
lcov["dist_to_own_community_centre_m"] = [g.distance(cen.geometry[s]) for g, s in zip(chc.geometry, chc.SQCODE)]
lcov.to_csv(OUT / "chc_landcover_250m.csv", index=False, encoding="utf-8-sig")
cols = [f"share_{WC_CLASS[c][0]}" for c in CODES]
summary = lcov.groupby("in_any_life_circle")[cols + ["dist_to_populated_grid_centre_m", "dist_to_own_community_centre_m"]
                                            ].median().round(3)
summary.index = summary.index.map({True: "在某个生活圈内", False: "不在任何生活圈内"})
summary.insert(0, "社康数", lcov.groupby("in_any_life_circle").size().rename(index={True: "在某个生活圈内",
                                                                                   False: "不在任何生活圈内"}))
summary.to_csv(OUT / "chc_landcover_250m_summary.csv", encoding="utf-8-sig")
print(summary.T.to_string())
print(pd.DataFrame(rows).round(2).to_string(index=False))
