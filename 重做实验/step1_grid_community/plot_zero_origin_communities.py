"""出发人次为 0 的社区：社区边界、与其相交的 500 m 网格及内部 250 m 网格的分布图与明细表。

500 m 网格标注 Tid、maup 归属社区、本社区片段占该网格社区覆盖面积比例、出发人次，有争议网格用虚线。
250 m 网格标注联通人口，需求人口分配给本社区的 250 m 网格着色。
"""
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_origin_maup"
SRC = HERE.parent / "supply input"
CRS = 4547
plt.rcParams["font.family"] = "Hiragino Sans GB"

INK, MUTED, GRID_LINE = "#1f2328", "#6e7781", "#b8bec6"
TARGET_EDGE, TARGET_FILL = "#1f5fa8", "#dbe7f6"
ASSIGNED = "#e08a2c"


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


t = pd.read_csv(OUT / "community_origin_demand.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
zero = t[t.origin_trips == 0].reset_index(drop=True)
names = t.set_index("SQCODE").SQNAME

comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.apply(polygonal)
comm["SQNAME"] = comm.SQCODE.map(names)
g500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
g500 = g500.merge(pd.read_csv(OUT / "grid500_community_maup.csv", encoding="utf-8-sig",
                              dtype={"SQCODE_maup": str}), on="Tid")
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "联通人", "geometry"]].to_crs(CRS)
d250 = pd.read_csv(RES / "landuse_demand/grid250_demand_allocation.csv", encoding="utf-8-sig",
                   dtype={"SQCODE": str})
od = pd.read_csv(RES / "od_clean_validation/outpatient_grid_OD_home_tid_clean.csv", encoding="utf-8-sig")
g500["origin_trips"] = g500.Tid.map(od.groupby("home_tid")["count"].sum()).fillna(0).astype(int)

c250 = g250.assign(geometry=g250.geometry.centroid)
nest = gpd.sjoin(c250, g500[["Tid", "geometry"]], predicate="within", lsuffix="250", rsuffix="500")
g250 = g250.merge(nest[["Tid_250", "Tid_500"]].rename(columns={"Tid_250": "Tid"}), on="Tid", how="left")

pieces = gpd.overlay(g500[["Tid", "geometry"]], comm[["SQCODE", "geometry"]], how="intersection",
                     keep_geom_type=True)
pieces["piece_m2"] = pieces.area
pieces = pieces[pieces.piece_m2 > 0]
pieces["area_share"] = pieces.piece_m2 / pieces.groupby("Tid").piece_m2.transform("sum")

# ---------- 明细表 ----------
rows = []
for _, z in zero.iterrows():
    sq = z.SQCODE
    for _, p in pieces[pieces.SQCODE == sq].sort_values("area_share", ascending=False).iterrows():
        g = g500.set_index("Tid").loc[p.Tid]
        sub = d250[d250.Tid.isin(g250.loc[g250.Tid_500 == p.Tid, "Tid"])]
        rows.append({
            "SQCODE": sq, "SQNAME": z.SQNAME, "Tid_500": p.Tid,
            "本社区片段面积_m2": round(p.piece_m2), "本社区占网格社区覆盖面积": round(p.area_share, 4),
            "maup归属社区": names.get(g.SQCODE_maup), "maup社区占比": round(g.max_area_share, 4),
            "有争议": bool(g.disputed), "出发人次": int(g.origin_trips),
            "网格内250m联通人口": int(g250.loc[g250.Tid_500 == p.Tid, "联通人"].sum()),
            "网格内分配给本社区的250m需求人口": round(sub.loc[sub.SQCODE == sq, "联通人_分配"].sum(), 1),
            "网格内250m需求人口最多的社区": (names.get(sub.groupby("SQCODE")["联通人_分配"].sum().idxmax())
                                  if sub["联通人_分配"].sum() > 0 else None),
        })
detail = pd.DataFrame(rows)
detail.to_csv(OUT / "zero_origin_communities_grid500_detail.csv", index=False, encoding="utf-8-sig")

d250_target = d250[d250.SQCODE.isin(zero.SQCODE)][["Tid", "SQCODE", "weight", "rule", "联通人", "联通人_分配"]]
d250_target.assign(SQNAME=d250_target.SQCODE.map(names)).to_csv(
    OUT / "zero_origin_communities_grid250_detail.csv", index=False, encoding="utf-8-sig")

# ---------- 分布图 ----------
n = len(zero)
ncol = 3
nrow = math.ceil(n / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 7 * nrow), facecolor="white")
axes = axes.ravel()
for ax, (_, z) in zip(axes, zero.iterrows()):
    sq = z.SQCODE
    tids = pieces.loc[pieces.SQCODE == sq, "Tid"]
    gs = g500[g500.Tid.isin(tids)]
    x0, y0, x1, y1 = gs.total_bounds
    pad = 120
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    view = shapely.box(x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    tgt = comm[comm.SQCODE == sq]
    tgt.plot(ax=ax, color=TARGET_FILL, edgecolor="none", zorder=1)
    a250 = d250_target[(d250_target.SQCODE == sq) & (d250_target["联通人_分配"] > 0)]
    sub250 = g250[g250.Tid_500.isin(tids)]
    hit = sub250.merge(a250[["Tid", "联通人_分配"]], on="Tid")
    if len(hit):
        hit.plot(ax=ax, color=ASSIGNED, alpha=0.45, edgecolor="none", zorder=2)
    sub250.boundary.plot(ax=ax, color=GRID_LINE, linewidth=0.6, zorder=3)
    for _, c in sub250.iterrows():
        pt = c.geometry.centroid
        txt = f"{int(c['联通人'])}"
        a = hit.loc[hit.Tid == c.Tid, "联通人_分配"]
        if len(a) and abs(a.iloc[0] - c["联通人"]) > 0.5:
            txt += f"\n本社区{a.iloc[0]:.0f}"
        ax.text(pt.x, pt.y - 45, txt, fontsize=7, color=MUTED, ha="center", va="center", zorder=6)

    near = comm[comm.intersects(view)]
    near.boundary.plot(ax=ax, color=INK, linewidth=0.8, linestyle=(0, (4, 2)), alpha=0.6, zorder=4)
    for _, c in near[near.SQCODE != sq].iterrows():
        part = c.geometry.intersection(view)
        if part.area > 0:
            pt = part.representative_point()
            ax.text(pt.x, pt.y, c.SQNAME, fontsize=9, color=MUTED, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    tgt.boundary.plot(ax=ax, color=TARGET_EDGE, linewidth=2.4, zorder=5)
    pt = tgt.geometry.iloc[0].representative_point()
    ax.text(pt.x, pt.y, z.SQNAME, fontsize=12, fontweight="bold", color=TARGET_EDGE, ha="center", va="center",
            zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=TARGET_EDGE, lw=0.8, alpha=0.85))

    for _, g in gs.iterrows():
        ls = (0, (3, 2)) if g.disputed else "-"
        ax.plot(*g.geometry.exterior.xy, color=INK, linewidth=1.6, linestyle=ls, zorder=6)
        share = pieces.loc[(pieces.Tid == g.Tid) & (pieces.SQCODE == sq), "area_share"].iloc[0]
        gx0, _, _, gy1 = g.geometry.bounds
        ax.text(gx0 + 15, gy1 - 15,
                f"500m {g.Tid}\n归属 {names.get(g.SQCODE_maup)} {g.max_area_share:.0%}\n"
                f"{z.SQNAME}占 {share:.0%}\n出发 {g.origin_trips}",
                fontsize=8, color=INK, ha="left", va="top", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_title(f"{z.SQNAME}（{sq}）  250m需求人口 联通 {d250_target.loc[d250_target.SQCODE == sq, '联通人_分配'].sum():.0f}",
                 fontsize=12, color=INK, loc="left")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
for ax in axes[n:]:
    ax.axis("off")

handles = [
    Patch(facecolor=TARGET_FILL, edgecolor=TARGET_EDGE, linewidth=2, label="目标社区"),
    Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (4, 2)), alpha=0.6, label="其他社区边界"),
    Line2D([], [], color=INK, linewidth=1.6, label="500m 网格（无争议）"),
    Line2D([], [], color=INK, linewidth=1.6, linestyle=(0, (3, 2)), label="500m 网格（有争议）"),
    Patch(facecolor="white", edgecolor=GRID_LINE, label="250m 网格（数字为联通人口）"),
    Patch(facecolor=ASSIGNED, alpha=0.45, label="250m 需求人口分配给目标社区"),
]
fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.97, 0.06), fontsize=11, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_zero_origin_communities.png", dpi=160)
print(detail.to_string(index=False))
