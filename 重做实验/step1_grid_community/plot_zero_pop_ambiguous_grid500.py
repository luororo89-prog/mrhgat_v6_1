"""内部 250 m 网格联通人口全为 0、且最大社区面积占比不足 80% 的出发 500 m 网格的位置。

网格取 grid500_origin_250m/grid250_origin_allocation.csv 中 rule 为 maup_250m需求人口为0 的出发网格，
最大社区占比、maup 归属取 grid500_origin_maup/grid500_community_maup.csv。区按网格中心点所在区界。
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from matplotlib.patches import Patch
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_origin_250m"
SRC = HERE.parent / "supply input"
CRS = 4547
plt.rcParams["font.family"] = "Hiragino Sans GB"
INK, MUTED = "#1f2328", "#6e7781"
C_MID, C_LOW = "#e08a2c", "#7b3294"


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


a = pd.read_csv(OUT / "grid250_origin_allocation.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
m5 = pd.read_csv(RES / "grid500_origin_maup/grid500_community_maup.csv", encoding="utf-8-sig",
                 dtype={"SQCODE_maup": str})
names = pd.read_csv(RES / "landuse_demand/community_demand_population.csv", encoding="utf-8-sig",
                    dtype={"SQCODE": str}).set_index("SQCODE").SQNAME
dist = gpd.read_file(SRC / "shenzhen_district_4547.shp")[["dt_adcode", "dt_name", "geometry"]].to_crs(CRS)

z = a[a.rule == "maup_250m需求人口为0"].drop_duplicates("home_tid")[["home_tid", "origin_trips"]]
z = z.merge(m5, left_on="home_tid", right_on="Tid").drop(columns="home_tid")
z = z[z.max_area_share < 0.8].copy()
z["类别"] = z.max_area_share.map(lambda s: "最大社区50%-80%" if s >= 0.5 else "最大社区不足50%")

g500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
g = g500.merge(z, on="Tid")
comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.apply(polygonal)
pieces = gpd.overlay(g[["Tid", "geometry"]], comm, how="intersection", keep_geom_type=True)
pieces["piece_m2"] = pieces.area
pieces["share"] = pieces.piece_m2 / pieces.groupby("Tid").piece_m2.transform("sum")
pieces = pieces.sort_values(["Tid", "share"], ascending=[True, False])
pieces["rank"] = pieces.groupby("Tid").cumcount()
first = pieces[pieces["rank"] == 0].set_index("Tid")
second = pieces[pieces["rank"] == 1].set_index("Tid")

tab = pd.DataFrame({"Tid": g.Tid, "类别": g.类别, "出发人次": g.origin_trips})
cen = gpd.sjoin(g[["Tid", "geometry"]].assign(geometry=g.centroid), dist, predicate="within", how="left")
tab["区"] = tab.Tid.map(cen.drop_duplicates("Tid").set_index("Tid").dt_name)
tab["最大社区"] = tab.Tid.map(first.SQCODE).map(names)
tab["最大社区占比"] = tab.Tid.map(first.share).round(3)
tab["第二社区"] = tab.Tid.map(second.SQCODE).map(names)
tab["第二社区占比"] = tab.Tid.map(second.share).round(3)
tab["相交社区数"] = g.n_communities.to_numpy()
tab = tab.sort_values("出发人次", ascending=False)
tab.to_csv(OUT / "zero_pop_ambiguous_grid500.csv", index=False, encoding="utf-8-sig")

# ---------- 位置图 ----------
fig, ax = plt.subplots(figsize=(16, 9.5), facecolor="white")
comm.boundary.plot(ax=ax, color="#d0d5db", linewidth=0.3, zorder=1)
dist.boundary.plot(ax=ax, color=MUTED, linewidth=1.0, zorder=2)
for _, d in dist.iterrows():
    pt = d.geometry.representative_point()
    ax.text(pt.x, pt.y, d.dt_name, fontsize=13, color=MUTED, ha="center", va="center", zorder=3)
for cat, col in [("最大社区50%-80%", C_MID), ("最大社区不足50%", C_LOW)]:
    s = g[g.类别 == cat]
    s.plot(ax=ax, color=col, edgecolor=col, linewidth=1.2, zorder=4 if col == C_MID else 5)
ax.set_aspect("equal")
ax.set_axis_off()
n_mid, n_low = (g.类别 == "最大社区50%-80%").sum(), (g.类别 == "最大社区不足50%").sum()
t_mid = g.loc[g.类别 == "最大社区50%-80%", "origin_trips"].sum()
t_low = g.loc[g.类别 == "最大社区不足50%", "origin_trips"].sum()
ax.legend(handles=[Patch(color=C_MID, label=f"最大社区占 50%-80%（{n_mid} 格，{t_mid} 人次）"),
                   Patch(color=C_LOW, label=f"最大社区不足 50%（{n_low} 格，{t_low} 人次）")],
          loc="lower left", fontsize=12, frameon=False)
ax.set_title("250m 格子都没人住、且社区归属不清的出发 500m 网格", fontsize=15, color=INK, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig_zero_pop_ambiguous_grid500.png", dpi=150)

print(tab.groupby("区").agg(网格数=("Tid", "size"), 出发人次=("出发人次", "sum")).sort_values("网格数", ascending=False)
      .to_string())
print(tab.head(15).to_string(index=False))
