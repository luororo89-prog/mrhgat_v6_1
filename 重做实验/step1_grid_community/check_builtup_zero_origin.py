"""出发人次为 0 的社区及其相邻社区的建设用地情况。

建设用地取 ESA WorldCover 2020 v100（10 m）Built-up 类（值 50），与 landuse_demand_population.py 相同。
统计社区（及多块边界的各块）面积、各地类占比、建设用地面积，以及 250 m 网格分配到社区的联通、七普需求人口。
相邻社区为与目标社区边界相接（5 m 缓冲相交）的社区。
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_origin_maup"
SRC = HERE.parent / "supply input"
WC = RES / "landuse_demand/worldcover_2020_shenzhen.tif"
CRS = 4547
FOCUS = ["水贝", "金鹏", "坳下", "粤桂", "志盛", "东渔", "南澳"]
CLASSES = {10: "林地", 20: "灌木", 30: "草地", 40: "耕地", 50: "建设用地", 60: "裸地", 80: "水体", 90: "湿地",
           95: "红树林"}
plt.rcParams["font.family"] = "Hiragino Sans GB"
INK, MUTED = "#1f2328", "#6e7781"
TARGET_EDGE, BUILT, WATER = "#1f5fa8", "#d9a066", "#cfe3f3"


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


demand = pd.read_csv(RES / "landuse_demand/community_demand_population.csv", encoding="utf-8-sig",
                     dtype={"SQCODE": str})
names = demand.set_index("SQCODE").SQNAME
comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.apply(polygonal)
comm["SQNAME"] = comm.SQCODE.map(names)
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "联通人", "geometry"]].to_crs(CRS)
d250 = pd.read_csv(RES / "landuse_demand/grid250_demand_allocation.csv", encoding="utf-8-sig",
                   dtype={"SQCODE": str})
src = rasterio.open(WC)


def landcover(geom):
    """几何范围内各地类占比（按 10 m 像元计数）。"""
    g = gpd.GeoSeries([geom], crs=CRS).to_crs(4326).iloc[0]
    w = from_bounds(*g.bounds, src.transform).round_offsets(op="floor").round_lengths(op="ceil")
    w = Window(w.col_off, w.row_off, w.width + 1, w.height + 1)
    lc = src.read(1, window=w)
    inside = ~geometry_mask([g], out_shape=lc.shape, transform=src.window_transform(w))
    v = lc[inside]
    return {k: float((v == k).mean()) if v.size else 0.0 for k in CLASSES}


def row(sq, geom, part, role, target, pop_l, pop_q):
    lc = landcover(geom)
    area = geom.area
    built = lc[50] * area
    r = {"目标社区": target, "类型": role, "SQCODE": sq, "SQNAME": names.get(sq), "部分": part,
         "面积_公顷": round(area / 1e4, 2), "建设用地_公顷": round(built / 1e4, 2), "建设用地占比": round(lc[50], 3)}
    for k, n in CLASSES.items():
        if k != 50:
            r[f"{n}占比"] = round(lc[k], 3)
    r["需求人口_联通"] = round(pop_l, 1)
    r["需求人口_七普"] = round(pop_q, 1)
    r["联通人口_每公顷建设用地"] = round(pop_l / (built / 1e4), 1) if built > 0 else np.nan
    r["七普人口_每公顷建设用地"] = round(pop_q / (built / 1e4), 1) if built > 0 else np.nan
    return r


rows = []
focus_sq = demand.loc[demand.SQNAME.isin(FOCUS), "SQCODE"].tolist()
for sq in focus_sq:
    tname = names[sq]
    geom = comm.loc[comm.SQCODE == sq, "geometry"].iloc[0]
    dem = demand.set_index("SQCODE").loc[sq]
    rows.append(row(sq, geom, "整体", "目标社区", tname, dem.需求人口_联通, dem.需求人口_七普))
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    if len(parts) > 1:
        a = d250[(d250.SQCODE == sq) & (d250.weight > 0)].merge(g250[["Tid", "geometry"]], on="Tid")
        a = gpd.GeoDataFrame(a, geometry="geometry", crs=CRS)
        which = a.geometry.apply(lambda g: int(np.argmax([g.intersection(p).area for p in parts])))
        for i, p in enumerate(parts):
            sel = a[which == i]
            rows.append(row(sq, p, f"第{i + 1}块", "目标社区分块", tname, sel["联通人_分配"].sum(), sel["七普加_分配"].sum()))
    for _, nb in comm[(comm.SQCODE != sq) & comm.intersects(geom.buffer(5))].iterrows():
        dn = demand.set_index("SQCODE").loc[nb.SQCODE]
        rows.append(row(nb.SQCODE, nb.geometry, "整体", "相邻社区", tname, dn.需求人口_联通, dn.需求人口_七普))

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "zero_origin_builtup_check.csv", index=False, encoding="utf-8-sig")

# ---------- 分布图 ----------
cmap = ListedColormap(["#ffffff", WATER, BUILT])
fig, axes = plt.subplots(3, 3, figsize=(21, 21), facecolor="white")
axes = axes.ravel()
for ax, sq in zip(axes, focus_sq):
    geom = comm.loc[comm.SQCODE == sq, "geometry"].iloc[0]
    x0, y0, x1, y1 = geom.buffer(350).bounds
    view = shapely.box(x0, y0, x1, y1)
    vw = gpd.GeoSeries([view], crs=CRS).to_crs(4326).total_bounds
    w = from_bounds(*vw, src.transform).round_offsets(op="floor").round_lengths(op="ceil")
    lc = src.read(1, window=w)
    cls = np.where(lc == 50, 2, np.where(lc == 80, 1, 0))
    l, b, r, t = rasterio.windows.bounds(w, src.transform)
    ax.imshow(cls, cmap=cmap, vmin=0, vmax=2, extent=(l, r, b, t), interpolation="nearest", zorder=1)
    ax.set_xlim(vw[0], vw[2])
    ax.set_ylim(vw[1], vw[3])
    ax.set_aspect(1 / np.cos(np.radians((vw[1] + vw[3]) / 2)))

    gv = g250[g250.intersects(view)].to_crs(4326)
    gv.boundary.plot(ax=ax, color="#9aa3ad", linewidth=0.5, zorder=2)
    for _, c in gv.iterrows():
        pt = c.geometry.centroid
        ax.text(pt.x, pt.y, f"{int(c['联通人'])}", fontsize=7, color=INK, ha="center", va="center", zorder=6, clip_on=True,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    near = comm[comm.intersects(view)].to_crs(4326)
    near.boundary.plot(ax=ax, color=INK, linewidth=0.9, linestyle=(0, (4, 2)), alpha=0.7, zorder=3)
    tw = gpd.GeoSeries([view], crs=CRS).to_crs(4326).iloc[0]
    sub = tab[tab.目标社区 == names[sq]]
    for _, c in near[near.SQCODE != sq].iterrows():
        part = c.geometry.intersection(tw)
        if part.area > 0:
            pt = part.representative_point()
            s = sub.loc[(sub.SQCODE == c.SQCODE) & (sub.类型 == "相邻社区"), "建设用地占比"]
            lab = c.SQNAME + (f"\n建设用地{s.iloc[0]:.0%}" if len(s) else "")
            ax.text(pt.x, pt.y, lab, fontsize=9, color=MUTED, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    tg = gpd.GeoSeries([geom], crs=CRS).to_crs(4326)
    tg.boundary.plot(ax=ax, color=TARGET_EDGE, linewidth=2.6, zorder=5)
    pts = list(tg.iloc[0].geoms) if tg.iloc[0].geom_type == "MultiPolygon" else [tg.iloc[0]]
    for i, p in enumerate(pts):
        pt = p.representative_point()
        lab = names[sq] if len(pts) == 1 else f"{names[sq]} 第{i + 1}块"
        ax.text(pt.x, pt.y, lab, fontsize=11, fontweight="bold", color=TARGET_EDGE, ha="center", va="center",
                zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=TARGET_EDGE, lw=0.8, alpha=0.85))
    me = sub[sub.类型 == "目标社区"].iloc[0]
    ax.set_title(f"{names[sq]}  面积{me.面积_公顷:.1f}公顷  建设用地{me.建设用地占比:.0%}（{me.建设用地_公顷:.1f}公顷）\n"
                 f"需求人口 联通{me.需求人口_联通:.0f} 七普{me.需求人口_七普:.0f}", fontsize=12, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
for ax in axes[len(focus_sq):]:
    ax.axis("off")
handles = [Patch(facecolor=BUILT, label="建设用地（WorldCover 2020）"), Patch(facecolor=WATER, label="水体"),
           Patch(facecolor="white", edgecolor="#9aa3ad", label="250m 网格（数字为联通人口）"),
           Line2D([], [], color=TARGET_EDGE, linewidth=2.6, label="目标社区"),
           Line2D([], [], color=INK, linewidth=0.9, linestyle=(0, (4, 2)), label="其他社区边界")]
fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.95, 0.1), fontsize=13, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_zero_origin_builtup.png", dpi=150)

cols = ["目标社区", "类型", "SQNAME", "部分", "面积_公顷", "建设用地_公顷", "建设用地占比", "林地占比", "水体占比",
        "需求人口_联通", "需求人口_七普", "联通人口_每公顷建设用地", "七普人口_每公顷建设用地"]
pd.set_option("display.width", 250)
print(tab[cols].to_string(index=False))
