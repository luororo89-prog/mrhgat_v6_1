"""500 m 网格门诊出发需求（O）按 maup 规则分配到社区（SQCODE）。

500 m 网格与社区边界叠加，maup.assign 归属（网格整个在社区内则归该社区，否则归最大重叠面积社区）。
最大社区片段面积不足该网格社区覆盖面积 50% 的网格为有争议网格，按其内部 250 m 网格中各社区需求人口
（landuse_demand 的 联通人_分配）占比分配该网格的出发人次；内部 250 m 网格需求人口为 0 时按 maup 整格归属。
门诊 OD 取 od_clean_validation 清洗后的 outpatient_grid_OD_home_tid_clean.csv（时间筛选后的门诊 OD），250 m 需求人口取 facility_grid250.py 修订后的 250 m 网格社区归属。
"""
import json
from pathlib import Path

import geopandas as gpd
import maup
import numpy as np
import pandas as pd
import shapely
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_origin_maup"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
GRID500 = SRC / "SZ_grid_500m.shp"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM = SRC / "深圳社区objectid.shp"
COMM_POP = SRC / "深圳各社区人口.shp"
NODE = SRC / "node_order.csv"
OD = RES / "od_clean_validation/outpatient_grid_OD_home_tid_clean.csv"
DEMAND250 = HERE / "results/facility_grid250/grid250_demand_allocation_revised.csv"
CRS = 4547
DISPUTE = 0.5  # 最大社区片段占社区覆盖面积的比例低于此值即为有争议网格


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


# ---------- 输入 ----------
node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
names = gpd.read_file(COMM_POP)[["SQCODE", "SQNAME"]]
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(CRS).reset_index(drop=True)
comm["geometry"] = comm.geometry.apply(polygonal)
g500 = gpd.read_file(GRID500)[["Tid", "geometry"]].to_crs(CRS).reset_index(drop=True)
g250 = gpd.read_file(GRID250)[["Tid", "geometry"]].to_crs(CRS)
od = pd.read_csv(OD, encoding="utf-8-sig")[["home_tid", "end_tid", "count"]]
d250 = pd.read_csv(DEMAND250, encoding="utf-8-sig", dtype={"SQCODE": str})[["Tid", "SQCODE", "联通人_分配"]]

# ---------- 500 m 网格 maup 归属与争议判定 ----------
idx = maup.assign(g500, comm)
g500["SQCODE_maup"] = comm.SQCODE.reindex(idx.to_numpy()).to_numpy()

pieces = gpd.overlay(g500[["Tid", "geometry"]], comm, how="intersection", keep_geom_type=True)
pieces["piece_m2"] = pieces.area
pieces = pieces[pieces.piece_m2 > 0]
pieces["area_share"] = pieces.piece_m2 / pieces.groupby("Tid").piece_m2.transform("sum")
top = pieces.sort_values("area_share", ascending=False).drop_duplicates("Tid").set_index("Tid")
g500["max_area_share"] = g500.Tid.map(top.area_share)
g500["n_communities"] = g500.Tid.map(pieces.groupby("Tid").size()).fillna(0).astype(int)
g500["disputed"] = g500.max_area_share < DISPUTE
has_comm = g500.SQCODE_maup.notna()
maup_same_as_max_area = bool((g500.loc[has_comm, "Tid"].map(top.SQCODE) == g500.loc[has_comm, "SQCODE_maup"]).all())

# ---------- 250 m 网格嵌套于 500 m 网格（中心点落入） ----------
c250 = g250.assign(geometry=g250.geometry.centroid)
nest = gpd.sjoin(c250, g500[["Tid", "geometry"]], predicate="within", lsuffix="250", rsuffix="500")
nest = nest[["Tid_250", "Tid_500"]]
assert nest.Tid_250.is_unique
g500["n_grid250"] = g500.Tid.map(nest.groupby("Tid_500").size()).fillna(0).astype(int)

# ---------- 有争议网格：内部 250 m 网格各社区需求人口占比 ----------
dis_tid = set(g500.loc[g500.disputed, "Tid"])
dp = (nest[nest.Tid_500.isin(dis_tid)]
      .merge(d250.rename(columns={"Tid": "Tid_250"}), on="Tid_250", how="inner")
      .groupby(["Tid_500", "SQCODE"], as_index=False)["联通人_分配"].sum()
      .rename(columns={"Tid_500": "Tid", "联通人_分配": "demand_pop"}))
dp["grid_demand_pop"] = dp.groupby("Tid").demand_pop.transform("sum")
dp = dp[(dp.grid_demand_pop > 0) & (dp.demand_pop > 0)].copy()
dp["weight"] = dp.demand_pop / dp.grid_demand_pop
dp["rule"] = "250m需求人口占比"
fallback_tid = dis_tid - set(dp.Tid)

w = g500.loc[has_comm & ~g500.Tid.isin(dp.Tid), ["Tid", "SQCODE_maup"]].rename(columns={"SQCODE_maup": "SQCODE"})
w["weight"] = 1.0
w["rule"] = np.where(w.Tid.isin(fallback_tid), "maup_250m需求人口为0", "maup")
w = pd.concat([w, dp[["Tid", "SQCODE", "weight", "rule"]]], ignore_index=True)
assert np.allclose(w.groupby("Tid").weight.sum(), 1)
w = w.merge(node[["OBJECTID", "SQCODE"]], on="SQCODE", how="left", validate="many_to_one")
assert w.OBJECTID.notna().all()

g500.drop(columns="geometry").merge(
    dp.groupby("Tid").SQCODE.nunique().rename("n_communities_demand"), on="Tid", how="left"
).to_csv(OUT / "grid500_community_maup.csv", index=False, encoding="utf-8-sig")

# ---------- 出发需求分配 ----------
home = od.groupby("home_tid")["count"].sum().rename("origin_trips")
wo = w[w.Tid.isin(home.index)].rename(columns={"Tid": "home_tid"})
wo = wo.merge(home, left_on="home_tid", right_index=True)
wo["origin_trips_alloc"] = wo.origin_trips * wo.weight
wo.sort_values(["home_tid", "weight"], ascending=[True, False]).to_csv(
    OUT / "origin_grid500_community_weights.csv", index=False, encoding="utf-8-sig")
unassigned = home[~home.index.isin(w.Tid)]

flow = od.merge(wo[["home_tid", "SQCODE", "OBJECTID", "weight"]], on="home_tid")
flow["trips"] = flow["count"] * flow.weight
flow = flow.groupby(["OBJECTID", "SQCODE", "end_tid"], as_index=False).trips.sum()
flow.to_csv(OUT / "community_origin_end_tid.csv", index=False, encoding="utf-8-sig")

# ---------- 社区出发需求 ----------
t = node[["OBJECTID", "SQCODE"]].merge(names, on="SQCODE", how="left", validate="one_to_one")
maup_trips = home.rename_axis("Tid").reset_index().merge(g500[["Tid", "SQCODE_maup"]], on="Tid")
t["n_grid500_maup"] = t.SQCODE.map(g500.groupby("SQCODE_maup").size()).fillna(0).astype(int)
t["n_disputed_grid500"] = t.SQCODE.map(dp.groupby("SQCODE").Tid.nunique()).fillna(0).astype(int)
t["origin_trips"] = t.SQCODE.map(wo.groupby("SQCODE").origin_trips_alloc.sum()).fillna(0)
t["origin_trips_maup"] = t.SQCODE.map(maup_trips.groupby("SQCODE_maup")["origin_trips"].sum()).fillna(0)
t["差值_相对maup"] = t.origin_trips - t.origin_trips_maup
t.to_csv(OUT / "community_origin_demand.csv", index=False, encoding="utf-8-sig")

# ---------- 审计 ----------
dg = g500[g500.disputed]
og = wo.drop_duplicates("home_tid").set_index("home_tid")
audit = {
    "输入": {"500m网格": GRID500.name, "250m人口网格": GRID250.name, "社区边界": COMM.name,
           "门诊OD": str(OD.relative_to(HERE)), "250m需求人口": str(DEMAND250.relative_to(HERE))},
    "争议门槛_最大片段占社区覆盖面积": DISPUTE,
    "maup与最大面积归属一致": maup_same_as_max_area,
    "500m网格数": len(g500), "与社区相交的500m网格": int(has_comm.sum()),
    "跨社区500m网格": int((g500.n_communities > 1).sum()),
    "有争议500m网格": len(dg), "有争议500m网格_按250m需求人口占比": int(dp.Tid.nunique()),
    "有争议500m网格_250m需求人口为0按maup": len(fallback_tid),
    "250m网格嵌套于500m网格": len(nest), "250m网格不在500m网格内": int(len(g250) - len(nest)),
    "OD行": len(od), "出发人次": int(od["count"].sum()), "出发网格": int(len(home)),
    "出发网格_按规则": og.rule.value_counts().to_dict(),
    "出发人次_按规则": {k: float(v) for k, v in og.groupby("rule").origin_trips.sum().items()},
    "出发网格_不与社区相交": int(len(unassigned)), "出发人次_不与社区相交": int(unassigned.sum()),
    "分配后出发人次合计": float(t.origin_trips.sum()),
    "出发人次为0的社区": t.loc[t.origin_trips == 0, ["SQCODE", "SQNAME"]].to_dict("records"),
    "maup整格归属下出发人次为0的社区": int((t.origin_trips_maup == 0).sum()),
    "无maup归属500m网格的社区": int((t.n_grid500_maup == 0).sum()),
    "相对maup_出发人次变化的社区": int((t.差值_相对maup.abs() > 0.5).sum()),
    "相对maup_转移人次": float(t.差值_相对maup.abs().sum() / 2),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2, default=int))
