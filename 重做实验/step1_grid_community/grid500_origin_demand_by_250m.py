"""500 m 网格门诊出发需求（O）全部按内部 250 m 网格各社区需求人口占比分配到社区（SQCODE）。

每个 500 m 网格的出发人次按其内部 250 m 网格中各社区需求人口（landuse_demand 的 联通人_分配）占比分配；
内部 250 m 网格需求人口为 0 或没有 250 m 网格时，按 maup 整格归属（grid500_origin_maup/grid500_community_maup.csv）；
其中最大社区面积占比不足 80% 的网格，按内部 250 m 网格各社区片段的建设用地面积占比分配并挂接到对应 250 m 网格，
建设用地取 ESA WorldCover 2020 v100 Built-up 类（与 landuse_demand_population.py 相同），网格内无建设用地时删除该网格出发的 OD 记录
（删除记录 od_dropped_zero_pop_no_builtup.csv，实际使用的 OD outpatient_grid_OD_home_tid_clean_used.csv）。
门诊 OD 取 od_clean_validation 清洗后的 outpatient_grid_OD_home_tid_clean.csv。
出发人次同时挂接到 250 m 网格（grid250_origin_allocation.csv、grid250_origin.shp），挂接后的社区合计与社区结果一致。
结果与 grid500_origin_maup 的现行口径（最大片段<50% 为争议网格）及 maup 整格归属对比。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from scipy.stats import spearmanr
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_origin_250m"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
OD = RES / "od_clean_validation/outpatient_grid_OD_home_tid_clean.csv"
DEMAND250 = RES / "landuse_demand/grid250_demand_allocation.csv"
DEMAND = RES / "landuse_demand/community_demand_population.csv"
MAUP500 = RES / "grid500_origin_maup/grid500_community_maup.csv"
CURRENT = RES / "grid500_origin_maup/community_origin_demand.csv"
COMM = SRC / "深圳社区objectid.shp"
WC = RES / "landuse_demand/worldcover_2020_shenzhen.tif"
BUILT = 50  # WorldCover Built-up
AMBIG = 0.8  # 250 m 需求人口为 0 且最大社区面积占比低于此值的网格按建设用地比例分配
CRS = 4547
FOCUS = ["水贝", "金鹏", "坳下", "粤桂", "志盛", "东渔", "南澳"]


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])



node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
od = pd.read_csv(OD, encoding="utf-8-sig")[["home_tid", "end_tid", "count"]]
d250 = pd.read_csv(DEMAND250, encoding="utf-8-sig", dtype={"SQCODE": str})[["Tid", "SQCODE", "联通人_分配"]]
g500 = pd.read_csv(MAUP500, encoding="utf-8-sig", dtype={"SQCODE_maup": str})[["Tid", "SQCODE_maup", "max_area_share"]]
cur = pd.read_csv(CURRENT, encoding="utf-8-sig", dtype={"SQCODE": str})
dem = pd.read_csv(DEMAND, encoding="utf-8-sig", dtype={"SQCODE": str})[["SQCODE", "需求人口_联通"]]

# ---------- 250 m 网格嵌套于 500 m 网格（中心点落入） ----------
grid500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]]
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "联通人", "geometry"]]
c250 = g250[["Tid", "geometry"]].to_crs(4547)
c250 = c250.assign(geometry=c250.geometry.centroid).to_crs(grid500.crs)
nest = gpd.sjoin(c250, grid500, predicate="within", lsuffix="250", rsuffix="500")[["Tid_250", "Tid_500"]]
assert nest.Tid_250.is_unique
home = od.groupby("home_tid")["count"].sum().rename("origin_trips")

# ---------- 250 m 需求人口为 0、最大社区面积占比不足 80% 的出发网格：建设用地比例 ----------
pop500 = nest.merge(d250.rename(columns={"Tid": "Tid_250"}), on="Tid_250").groupby("Tid_500")["联通人_分配"].sum()
zero_pop = set(pop500[pop500 <= 0].index) & set(home.index)
amb = sorted(zero_pop & set(g500.loc[g500.max_area_share < AMBIG, "Tid"]))
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.apply(polygonal)
sub250 = g250[g250.Tid.isin(nest.loc[nest.Tid_500.isin(amb), "Tid_250"])][["Tid", "geometry"]].to_crs(CRS)
bp = gpd.overlay(sub250.rename(columns={"Tid": "Tid_250"}), comm, how="intersection", keep_geom_type=True)
bp["piece_m2"] = bp.area
bp = bp[bp.piece_m2 > 0].merge(nest, on="Tid_250")
with rasterio.open(WC) as src:
    frac = []
    for geom in bp.to_crs(4326).geometry:
        win = from_bounds(*geom.bounds, src.transform).round_offsets(op="floor").round_lengths(op="ceil")
        win = Window(win.col_off, win.row_off, win.width + 1, win.height + 1)
        lc = src.read(1, window=win)
        inside = ~geometry_mask([geom], out_shape=lc.shape, transform=src.window_transform(win))
        frac.append(float((lc[inside] == BUILT).mean()) if inside.any() else 0.0)
bp["builtup_frac"] = frac
bp["builtup_m2"] = bp.builtup_frac * bp.piece_m2
bp["grid_builtup_m2"] = bp.groupby("Tid_500").builtup_m2.transform("sum")
bp = bp[bp.grid_builtup_m2 > 0].copy()
bp["weight"] = bp.builtup_m2 / bp.grid_builtup_m2
built_tid = set(bp.Tid_500)

# 无建设用地的这类出发网格：删除其 OD 记录
drop_tid = sorted(set(amb) - built_tid)
od[od.home_tid.isin(drop_tid)].to_csv(OUT / "od_dropped_zero_pop_no_builtup.csv", index=False, encoding="utf-8-sig")
od = od[~od.home_tid.isin(drop_tid)]
od.to_csv(OUT / "outpatient_grid_OD_home_tid_clean_used.csv", index=False, encoding="utf-8-sig")
home_all = home
home = od.groupby("home_tid")["count"].sum().rename("origin_trips")
bp_out = bp.drop(columns="geometry").merge(home, left_on="Tid_500", right_index=True)
bp_out["O_trips"] = bp_out.origin_trips * bp_out.weight
bp_out["SQCODE_maup"] = bp_out.Tid_500.map(g500.set_index("Tid").SQCODE_maup)
bp_out[["Tid_500", "Tid_250", "SQCODE", "SQCODE_maup", "piece_m2", "builtup_frac", "builtup_m2", "weight",
        "origin_trips", "O_trips"]].sort_values(["Tid_500", "Tid_250", "SQCODE"]).to_csv(
    OUT / "zero_pop_ambiguous_builtup_pieces.csv", index=False, encoding="utf-8-sig")

# ---------- 每个 500 m 网格的社区权重 ----------
w = (nest.merge(d250.rename(columns={"Tid": "Tid_250"}), on="Tid_250")
     .groupby(["Tid_500", "SQCODE"], as_index=False)["联通人_分配"].sum()
     .rename(columns={"Tid_500": "Tid", "联通人_分配": "demand_pop"}))
w["grid_demand_pop"] = w.groupby("Tid").demand_pop.transform("sum")
w = w[(w.grid_demand_pop > 0) & (w.demand_pop > 0)].copy()
w["weight"] = w.demand_pop / w.grid_demand_pop
w["rule"] = "250m需求人口占比"
wb = bp.groupby(["Tid_500", "SQCODE"], as_index=False).weight.sum().rename(columns={"Tid_500": "Tid"})
wb["rule"] = "建设用地比例_250m需求人口为0"
fb = g500[~g500.Tid.isin(w.Tid) & ~g500.Tid.isin(built_tid)].rename(columns={"SQCODE_maup": "SQCODE"})
fb["weight"] = 1.0
fb["rule"] = "maup_250m需求人口为0"
w = pd.concat([w[["Tid", "SQCODE", "weight", "rule"]], wb, fb[["Tid", "SQCODE", "weight", "rule"]]], ignore_index=True)
assert np.allclose(w.groupby("Tid").weight.sum(), 1)
w = w.merge(node[["OBJECTID", "SQCODE"]], on="SQCODE", how="left", validate="many_to_one")
assert w.OBJECTID.notna().all()

# ---------- 出发需求分配 ----------
wo =w[w.Tid.isin(home.index)].rename(columns={"Tid": "home_tid"}).merge(home, left_on="home_tid", right_index=True)
wo["origin_trips_alloc"] = wo.origin_trips * wo.weight
assert set(wo.home_tid) == set(home.index)
wo.sort_values(["home_tid", "weight"], ascending=[True, False]).to_csv(
    OUT / "origin_grid500_community_weights.csv", index=False, encoding="utf-8-sig")

flow = od.merge(wo[["home_tid", "SQCODE", "OBJECTID", "weight"]], on="home_tid")
flow["trips"] = flow["count"] * flow.weight
flow.groupby(["OBJECTID", "SQCODE", "end_tid"], as_index=False).trips.sum().to_csv(
    OUT / "community_origin_end_tid.csv", index=False, encoding="utf-8-sig")

# ---------- 社区出发需求与对比 ----------
t = cur[["OBJECTID", "SQCODE", "SQNAME", "origin_trips_maup", "origin_trips"]].rename(
    columns={"origin_trips": "origin_trips_现行50"})
t["origin_trips"] = t.SQCODE.map(wo.groupby("SQCODE").origin_trips_alloc.sum()).fillna(0)
t = t.merge(dem, on="SQCODE", how="left")
t["差值_相对现行50"] = t.origin_trips - t.origin_trips_现行50
for c in ["origin_trips", "origin_trips_现行50", "origin_trips_maup"]:
    t[c.replace("origin_trips", "每千需求人口出发人次")] = np.where(t.需求人口_联通 > 0, t[c] / t.需求人口_联通 * 1000, np.nan)
t = t[["OBJECTID", "SQCODE", "SQNAME", "需求人口_联通", "origin_trips", "origin_trips_现行50", "origin_trips_maup",
       "差值_相对现行50", "每千需求人口出发人次", "每千需求人口出发人次_现行50", "每千需求人口出发人次_maup"]]
t.to_csv(OUT / "community_origin_demand.csv", index=False, encoding="utf-8-sig")


def summary(col):
    o = t[col]
    rate = t[col.replace("origin_trips", "每千需求人口出发人次")]
    q = rate.quantile([0, .05, .25, .5, .75, .95, 1])
    rho = spearmanr(t.需求人口_联通, o).statistic
    return {"出发人次为0的社区": int((o == 0).sum()),
            "出发人次为0的社区名单": "、".join(t.loc[o == 0, "SQNAME"]),
            "社区出发人次与需求人口Spearman": round(float(rho), 4),
            "每千需求人口出发人次_分位数": {f"{k:.0%}": round(float(v), 1) for k, v in q.items()},
            "每千需求人口出发人次_超过1000的社区": int((rate > 1000).sum()),
            "每千需求人口出发人次_低于50的社区": int((rate < 50).sum())}


# ---------- 出发需求挂接到 250 m 网格 ----------
# 250 m 需求人口为正：按 250 m 网格（及其社区片段）需求人口占 500 m 网格需求人口的比例挂接；
# 250 m 需求人口为 0 且最大社区面积占比不足 80%：按 250 m 网格各社区片段建设用地面积占比挂接；
# 其余 250 m 需求人口为 0：出发人次归 maup 社区，平均挂到内部归属该社区的 250 m 网格；
# 没有这类网格时，挂到中心点距离最近的有出发需求的 250 m 网格。
a = (nest.merge(d250.rename(columns={"Tid": "Tid_250"}), on="Tid_250")
     .rename(columns={"Tid_500": "home_tid"}))
a = a[a.home_tid.isin(home.index)]
a["grid_demand_pop"] = a.groupby("home_tid")["联通人_分配"].transform("sum")
pa = a[(a.grid_demand_pop > 0) & (a["联通人_分配"] > 0)].copy()
pa["o_share"] = pa["联通人_分配"] / pa.grid_demand_pop
pa["rule"] = "250m需求人口占比"
pbb = bp[["Tid_500", "Tid_250", "SQCODE", "weight"]].rename(columns={"Tid_500": "home_tid", "weight": "o_share"})
pbb["联通人_分配"] = 0.0
pbb["rule"] = "建设用地比例_250m需求人口为0"
maup_of = g500.set_index("Tid").SQCODE_maup
rest = home.index.difference(pa.home_tid).difference(pbb.home_tid)
pb = a[a.home_tid.isin(rest)]
pb = pb[pb.SQCODE == pb.home_tid.map(maup_of)].copy()
pb["o_share"] = 1 / pb.groupby("home_tid").Tid_250.transform("size")
pb["rule"] = "maup_250m需求人口为0"
rest = rest.difference(pb.home_tid)
cand = g250[g250.Tid.isin(pd.concat([pa, pbb, pb]).Tid_250)][["Tid", "geometry"]].to_crs(CRS)
cand = cand.assign(geometry=cand.geometry.centroid).rename(columns={"Tid": "Tid_250"})
src = grid500[grid500.Tid.isin(rest)].to_crs(CRS)
src = src.assign(geometry=src.geometry.centroid).rename(columns={"Tid": "home_tid"})
near = (gpd.sjoin_nearest(src, cand, distance_col="dist_m").sort_values(["home_tid", "dist_m", "Tid_250"])
        .drop_duplicates("home_tid"))
pc = pd.DataFrame({"home_tid": near.home_tid.to_numpy(), "Tid_250": near.Tid_250.to_numpy(),
                   "SQCODE": maup_of.reindex(near.home_tid).to_numpy(), "联通人_分配": 0.0, "o_share": 1.0,
                   "rule": "maup_无250m网格_挂最近有出发需求250m"})
assert set(pc.home_tid) == set(rest)
a250 = pd.concat([pa, pbb, pb, pc], ignore_index=True).drop(columns="grid_demand_pop")
a250 = a250.merge(home, left_on="home_tid", right_index=True)
a250["O_trips"] = a250.origin_trips * a250.o_share
a250 = a250.merge(node[["OBJECTID", "SQCODE"]], on="SQCODE", how="left", validate="many_to_one")
assert np.allclose(a250.groupby("home_tid").o_share.sum(), 1)
assert np.allclose(a250.groupby("SQCODE").O_trips.sum().reindex(t.SQCODE).fillna(0).to_numpy(), t.origin_trips.to_numpy())
a250["Tid_250"] = a250.Tid_250.astype("Int64")
a250 = a250[["home_tid", "Tid_250", "SQCODE", "OBJECTID", "rule", "联通人_分配", "o_share", "origin_trips", "O_trips"]]
a250.sort_values(["home_tid", "Tid_250", "SQCODE"]).to_csv(
    OUT / "grid250_origin_allocation.csv", index=False, encoding="utf-8-sig")

by250 = a250.dropna(subset=["Tid_250"]).groupby("Tid_250")
lay = g250.merge(nest.rename(columns={"Tid_250": "Tid", "Tid_500": "Tid500"}), on="Tid", how="left")
own = a250.dropna(subset=["Tid_250"])
own = own[own.home_tid.to_numpy() == own.Tid_250.map(nest.set_index("Tid_250").Tid_500).to_numpy()]
lay["O_share"] = lay.Tid.map(own.groupby("Tid_250").o_share.sum()).fillna(0)  # 占所在 500 m 网格出发人次的比例
lay["O_trips"] = lay.Tid.map(by250.O_trips.sum()).fillna(0)
lay["n_comm"] = lay.Tid.map(by250.SQCODE.nunique()).fillna(0).astype(int)
main = a250.dropna(subset=["Tid_250"]).sort_values("O_trips", ascending=False).drop_duplicates("Tid_250")
lay["SQCODE"] = lay.Tid.map(main.set_index("Tid_250").SQCODE).astype(object)
lay.rename(columns={"联通人": "pop_lt"}).to_file(OUT / "grid250_origin.shp", encoding="utf-8")

og = wo.drop_duplicates("home_tid").set_index("home_tid")
audit = {
    "出发网格": int(len(home)), "出发人次": int(home.sum()),
    "出发网格_按规则": og.rule.value_counts().to_dict(),
    "出发人次_按规则": {k: float(v) for k, v in og.groupby("rule").origin_trips.sum().items()},
    "出发网格_拆分到多个社区": int((wo.groupby("home_tid").size() > 1).sum()),
    "分配后出发人次合计": float(t.origin_trips.sum()),
    "相对现行50_出发人次变化的社区": int((t.差值_相对现行50.abs() > 0.5).sum()),
    "相对现行50_转移人次": round(float(t.差值_相对现行50.abs().sum() / 2), 1),
    "全部按250m需求人口占比": summary("origin_trips"),
    "现行50": summary("origin_trips_现行50"),
    "maup整格": summary("origin_trips_maup"),
    "挂接250m网格": {
        "出发网格_按规则": a250.drop_duplicates("home_tid").rule.value_counts().to_dict(),
        "出发人次_按规则": {k: float(v) for k, v in a250.groupby("rule").O_trips.sum().items()},
        "有出发人次的250m网格": int((lay.O_trips > 0).sum()),
        "挂在250m网格上的出发人次": float(lay.O_trips.sum()),
        "未挂250m网格的出发人次": float(a250.loc[a250.Tid_250.isna(), "O_trips"].sum()),
        "无250m网格的出发网格_挂最近250m": near.assign(dist_m=near.dist_m.round(1))[["home_tid", "Tid_250", "dist_m"]]
                                    .to_dict("records"),
        "拆到多个社区的250m网格": int((lay.n_comm > 1).sum()),
    },
    "250m需求人口为0且最大社区不足80%": {
        "出发网格": len(amb), "出发人次": int(home_all[amb].sum()),
        "按建设用地比例": len(built_tid), "按建设用地比例_人次": int(home_all[sorted(built_tid)].sum()),
        "无建设用地_删除OD的出发网格": len(drop_tid), "删除OD行": int(len(pd.read_csv(
            OUT / "od_dropped_zero_pop_no_builtup.csv", encoding="utf-8-sig"))),
        "删除出发人次": int(home_all[drop_tid].sum()),
        "分给maup以外社区的人次": round(float(bp_out.loc[bp_out.SQCODE != bp_out.SQCODE_maup, "O_trips"].sum()), 1),
        "片段建设用地占比中位数": round(float(bp.builtup_frac.median()), 3),
    },
    "重点社区": t.set_index("SQNAME").loc[FOCUS, ["需求人口_联通", "origin_trips", "origin_trips_现行50"]]
                .round(1).to_dict("index"),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
