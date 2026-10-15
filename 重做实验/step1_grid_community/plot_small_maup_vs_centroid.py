"""小社区 250 m 网格在 maup 规则与中心点规则下归属不同的局部放大图。

maup：网格整个在社区内归该社区，否则归最大重叠面积社区（grid250_community_1to1.csv 的 SQCODE）。
中心点：网格中心所在社区（同文件 centroid_SQCODE）。小社区为面积小于 0.3 km² 的社区。
两种规则归属不同、且其中一方为小社区的网格，按 400 m 内相邻聚成片区，每个片区一幅放大图：
涉及的社区填色，归属不同的网格为黑色虚线框，黑点为网格中心，红框为医院 AOI。
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "maup_demand"
SRC = HERE.parent / "input"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM = SRC / "深圳社区objectid.shp"
DIST = SRC / "shenzhen_district_4547.shp"
AOI = (HERE.parent / "input"
       / "hospital_aoi_35.shp")
CRS = 4547
FILL = ["#8fabd9", "#e4a98b", "#9d93c9", "#9cc5a1", "#e8cf8a", "#d99ab0", "#86c0c7", "#c4b08f", "#b5c97f", "#c9a0dc"]

g = pd.read_csv(RES / "grid250_community_1to1.csv", dtype={"SQCODE": str, "centroid_SQCODE": str})
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})
small = set(cs.loc[cs.comm_area_m2 < 0.3e6, "SQCODE"])
name = cs.set_index("SQCODE").SQNAME
mark = lambda c: name[c] + ("(小)" if c in small else "")

grid = gpd.read_file(GRID250)[["Tid", "geometry"]].to_crs(CRS)
comm = gpd.read_file(COMM)[["SQCODE", "SQNAME", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.make_valid()
dist = gpd.read_file(DIST).to_crs(CRS)
aoi = gpd.read_file(AOI)[["fac_name", "geometry"]].to_crs(CRS)
aoi["label"] = aoi.fac_name.str.replace(r"（.*?）", "", regex=True).str.replace("深圳市", "")

d = g[(g.SQCODE != g.centroid_SQCODE) & (g.SQCODE.isin(small) | g.centroid_SQCODE.isin(small))]
dg = grid.merge(d[["Tid", "SQCODE", "centroid_SQCODE", "联通人"]], on="Tid").reset_index(drop=True)
dg[["Tid", "SQCODE", "centroid_SQCODE", "联通人"]].assign(
    maup社区=dg.SQCODE.map(mark), 中心点社区=dg.centroid_SQCODE.map(mark)).to_csv(
    OUT / "small_maup_vs_centroid_grids.csv", index=False, encoding="utf-8-sig")

# 400 m 内相邻的网格聚成片区
j = gpd.sjoin(dg[["geometry"]].assign(geometry=dg.buffer(400)), dg[["geometry"]], predicate="intersects")
par = list(range(len(dg)))
def root(i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i
for a, b in zip(j.index, j.index_right):
    par[root(a)] = root(b)
# 涉及同一小社区的网格并入同一片区
for c in small:
    idx = dg.index[(dg.SQCODE == c) | (dg.centroid_SQCODE == c)]
    for i in idx[1:]:
        par[root(i)] = root(idx[0])
dg["cl"] = [root(i) for i in range(len(dg))]

areas = []
for _, p in dg.groupby("cl"):
    codes = list(dict.fromkeys(list(p.SQCODE) + list(p.centroid_SQCODE)))
    focus = comm[comm.SQCODE.isin([c for c in codes if c in small])]
    x0, y0, x1, y1 = pd.concat([focus.geometry, p.geometry]).total_bounds
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2 + 300
    dn = dist.loc[dist.contains(p.geometry.union_all().centroid), "dt_name"]
    pairs = p.groupby(["SQCODE", "centroid_SQCODE"])["联通人"].agg(["size", "sum"]).reset_index()
    areas.append({"p": p, "codes": codes, "box": (cx - half, cy - half, cx + half, cy + half),
                  "dist": dn.iloc[0] if len(dn) else "", "pop": int(p["联通人"].sum()),
                  "title": "、".join([name[c] for c in codes if c in small][:4]) + ("等" if sum(c in small for c in codes) > 4 else ""),
                  "sub": "\n".join(f"maup {mark(a)} → 中心点 {mark(b)}　{n}格 {s}人"
                                   for a, b, n, s in pairs.itertuples(index=False))})
areas.sort(key=lambda a: (a["dist"], -a["pop"]))

ncol = 5
nrow = -(-len(areas) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 5.6), dpi=200)
for k, (ax, a) in enumerate(zip(axes.flat, areas)):
    x0, y0, x1, y1 = a["box"]
    ax.set_facecolor("#efefed")
    view = comm.cx[x0:x1, y0:y1]
    view.plot(ax=ax, facecolor="#efefed", edgecolor="#c9c9c6", linewidth=0.7)
    grid.cx[x0:x1, y0:y1].boundary.plot(ax=ax, color="#dcdcd8", linewidth=0.4)
    for i, c in enumerate(a["codes"]):
        comm[comm.SQCODE == c].plot(ax=ax, facecolor=FILL[i % len(FILL)], edgecolor="#5c5c5c", linewidth=0.8, alpha=0.9)
    frame = box(x0, y0, x1, y1)
    for r in view.itertuples():
        vis = r.geometry.intersection(frame)
        if vis.is_empty or vis.area < 5000:
            continue
        pt = vis.representative_point()
        if True:
            hit = r.SQCODE in a["codes"]
            ax.text(pt.x, pt.y, r.SQNAME, ha="center", va="center", fontsize=10 if hit else 7.5,
                    color="#1f1f1f" if hit else "#8a8a8a", zorder=6, clip_on=True)
    av = aoi.cx[x0:x1, y0:y1]
    if len(av):
        av.boundary.plot(ax=ax, color="#c8312f", linewidth=1.2, zorder=4)
    for r in av.itertuples():
        pt = r.geometry.representative_point()
        if not (x0 < pt.x < x1 and y0 + 300 < pt.y < y1):
            continue
        ax.text(pt.x, pt.y - 60, r.label, ha="center", va="top", fontsize=7.5, color="#c8312f", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5), clip_on=True)
    a["p"].boundary.plot(ax=ax, color="#111111", linewidth=1.4, linestyle=(0, (4, 2.5)), zorder=5)
    ce = a["p"].geometry.centroid
    ax.scatter(ce.x, ce.y, s=10, color="#111111", zorder=7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{chr(97 + k)}  {a['dist']}（{a['title']}）", loc="left", fontsize=12, pad=6)
    ax.text(0, -0.015, a["sub"], transform=ax.transAxes, fontsize=8, color="#555555", va="top", linespacing=1.35)
for ax in list(axes.flat)[len(areas):]:
    ax.set_visible(False)
fig.text(0.01, 0.995, "小社区 250 m 网格：maup 与中心点规则归属不同的网格（黑色虚线框，黑点为网格中心；红框为医院 AOI）",
         fontsize=13, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.0)
fig.savefig(OUT / "fig_small_maup_vs_centroid.png", facecolor="white")
print(len(areas), "片区", int(dg["联通人"].sum()), "人", len(dg), "网格")
