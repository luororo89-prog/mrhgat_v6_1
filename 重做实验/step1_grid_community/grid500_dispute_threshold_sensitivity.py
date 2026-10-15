"""500 m 网格争议判定口径的敏感性：比较不同口径下有争议网格数、出发人次为 0 的社区和相对 maup 的转移人次。

口径：
  最大片段<50%：最大社区片段不足该网格社区覆盖面积 50%（grid500_origin_demand_maup.py 现行口径）
  次大片段≥x：除 maup 社区外，另有社区片段占该网格社区覆盖面积不低于 x
有争议网格按内部 250 m 网格各社区需求人口（联通人_分配）占比分配出发人次，需求人口为 0 时按 maup。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_origin_maup"
SRC = HERE.parent / "supply input"
CRS = 4547
FOCUS = ["水贝", "金鹏", "坳下", "粤桂", "志盛", "东渔", "南澳"]


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


t = pd.read_csv(OUT / "community_origin_demand.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
g500 = pd.read_csv(OUT / "grid500_community_maup.csv", encoding="utf-8-sig", dtype={"SQCODE_maup": str})
comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.apply(polygonal)
grid = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
d250 = pd.read_csv(RES / "landuse_demand/grid250_demand_allocation.csv", encoding="utf-8-sig",
                   dtype={"SQCODE": str})[["Tid", "SQCODE", "联通人_分配"]]
od = pd.read_csv(RES / "od_clean_validation/outpatient_grid_OD_home_tid_clean.csv", encoding="utf-8-sig")
home = od.groupby("home_tid")["count"].sum()

pieces = gpd.overlay(grid, comm, how="intersection", keep_geom_type=True)
pieces["piece_m2"] = pieces.area
pieces = pieces[pieces.piece_m2 > 0]
pieces["area_share"] = pieces.piece_m2 / pieces.groupby("Tid").piece_m2.transform("sum")
ranked = pieces.sort_values("area_share", ascending=False)
ranked = ranked[ranked.groupby("Tid").cumcount() == 1]
second = ranked.set_index("Tid").area_share
g500["second_area_share"] = g500.Tid.map(second).fillna(0)

c250 = g250.assign(geometry=g250.geometry.centroid)
nest = gpd.sjoin(c250, grid, predicate="within", lsuffix="250", rsuffix="500")[["Tid_250", "Tid_500"]]
dem = (nest.merge(d250.rename(columns={"Tid": "Tid_250"}), on="Tid_250")
       .groupby(["Tid_500", "SQCODE"])["联通人_分配"].sum().rename("demand_pop").reset_index())
dem["grid_pop"] = dem.groupby("Tid_500").demand_pop.transform("sum")
dem = dem[(dem.grid_pop > 0) & (dem.demand_pop > 0)]
dem["weight"] = dem.demand_pop / dem.grid_pop

maup_o = t.set_index("SQCODE").origin_trips_maup
names = t.set_index("SQCODE").SQNAME


def allocate(disputed):
    dtid = set(g500.loc[disputed, "Tid"]) & set(home.index)
    dp = dem[dem.Tid_500.isin(dtid)]
    whole = g500[g500.Tid.isin(home.index) & ~g500.Tid.isin(dp.Tid_500)]
    o = whole.groupby("SQCODE_maup").Tid.apply(lambda s: home[s].sum())
    o = o.add((dp.weight * dp.Tid_500.map(home)).groupby(dp.SQCODE).sum(), fill_value=0)
    return o.reindex(t.SQCODE).fillna(0), len(dtid), dp.Tid_500.nunique()


rules = {"最大片段<50%（现行）": g500.max_area_share < 0.5}
for x in (0.20, 0.15, 0.10):
    rules[f"次大片段≥{x:.0%}"] = g500.second_area_share >= x

rows, focus = [], {}
for name, disputed in rules.items():
    o, n_dis_home, n_dis_used = allocate(disputed)
    assert np.isclose(o.sum(), home.sum())
    diff = o - maup_o.reindex(o.index)
    zero = o[o == 0]
    rows.append({
        "口径": name, "有争议500m网格": int(disputed.sum()), "有出发人次的有争议网格": n_dis_home,
        "其中按250m需求人口占比": n_dis_used,
        "有争议网格出发人次": int(home[home.index.isin(g500.loc[disputed, "Tid"])].sum()),
        "相对maup转移人次": round(float(diff.abs().sum() / 2), 1),
        "出发人次变化的社区": int((diff.abs() > 0.5).sum()),
        "出发人次为0的社区数": len(zero), "出发人次为0的社区": "、".join(names[zero.index]),
    })
    focus[name] = o.rename(index=names)[FOCUS].round(1)

res = pd.DataFrame(rows)
res.to_csv(OUT / "dispute_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
fc = pd.DataFrame(focus)
fc.to_csv(OUT / "dispute_threshold_focus_communities.csv", encoding="utf-8-sig")
print(res.to_string(index=False))
print(fc.to_string())
