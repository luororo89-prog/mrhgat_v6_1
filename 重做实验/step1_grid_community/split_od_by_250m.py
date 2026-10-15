"""500 m 网格 OD 经 250 m 网格拆到社区。

250 m 网格按最大重叠面积一对一归属社区（match_grid_community.py 的 grid250_community_1to1.csv）。
出发端 home_tid 按其内部 250 m 网格的联通人口拆分，人口为 0 时按 250 m 网格等面积拆分。
到达端 end_tid 按其内部 250 m 网格等面积拆分。
内部没有 250 m 网格的 500 m 网格，按 500 m 网格一对一归属的社区处理。
社区矩阵先按行总量、再在行内用最大余数法取整，总人次不变。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "od_250m"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "input"
GRID500 = SRC / "SZ_grid_500m.shp"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
NET = HERE.parent / "input"
OD = NET / "outpatient_grid_OD_home_tid.csv"
NODE = NET / "node_order.csv"

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
code_to_id = dict(zip(node.SQCODE, node.OBJECTID))
ids = node.OBJECTID.to_numpy()
pos = {int(c): i for i, c in enumerate(ids)}

# 250 m 网格一对一归属
a250 = pd.read_csv(RES / "grid250_community_1to1.csv", dtype={"SQCODE": str})[["Tid", "SQCODE", "联通人"]]
assert a250.Tid.is_unique and a250.SQCODE.notna().all()
a250["OBJECTID"] = a250.SQCODE.map(code_to_id)
assert a250.OBJECTID.notna().all()

# 250 m 网格嵌套于 500 m 网格
g500 = gpd.read_file(GRID500)[["Tid", "geometry"]]
g250 = gpd.read_file(GRID250)[["Tid", "geometry"]]
g250["geometry"] = g250.geometry.centroid
nest = gpd.sjoin(g250, g500, predicate="within", lsuffix="250", rsuffix="500")[["Tid_250", "Tid_500"]]
assert nest.Tid_250.is_unique
nest = nest.merge(a250.rename(columns={"Tid": "Tid_250"}), on="Tid_250")
nest.to_csv(OUT / "grid250_in_grid500.csv", index=False)

a500 = pd.read_csv(RES / "grid500_community_1to1.csv", dtype={"SQCODE": str})[["Tid", "SQCODE"]]
a500["OBJECTID"] = a500.SQCODE.map(code_to_id)

od = pd.read_csv(OD, encoding="utf-8-sig")[["home_tid", "end_tid", "count"]]
total = int(od["count"].sum())
audit = {"od_records": len(od), "od_trips": total,
         "grid250_cells": len(a250), "grid250_outside_grid500": int(len(a250) - len(nest)),
         "grid500_children_count": nest.groupby("Tid_500").size().value_counts().sort_index().to_dict()}


def weights(tids, by_population):
    sub = nest[nest.Tid_500.isin(tids)].copy()
    sub["n"] = 1.0
    basis = {}
    if by_population:
        popsum = sub.groupby("Tid_500")["联通人"].transform("sum")
        sub["w"] = np.where(popsum > 0, sub["联通人"], sub["n"])
        basis = sub.groupby("Tid_500")["联通人"].sum().gt(0).map({True: "250m_population", False: "250m_area"})
    else:
        sub["w"] = sub["n"]
        basis = sub.groupby("Tid_500").size().map(lambda _: "250m_area")
    w = sub.groupby(["Tid_500", "OBJECTID"]).w.sum().reset_index()
    w["weight"] = w.w / w.groupby("Tid_500").w.transform("sum")
    w["basis"] = w.Tid_500.map(basis)
    rest = sorted(set(tids) - set(w.Tid_500))
    fb = a500[a500.Tid.isin(rest)].rename(columns={"Tid": "Tid_500"})[["Tid_500", "OBJECTID"]]
    fb["weight"] = 1.0
    fb["basis"] = "500m_one_to_one"
    w = pd.concat([w[["Tid_500", "OBJECTID", "weight", "basis"]], fb], ignore_index=True)
    assert set(w.Tid_500) == set(tids) and np.allclose(w.groupby("Tid_500").weight.sum(), 1)
    w["OBJECTID"] = w.OBJECTID.astype(int)
    return w


wo = weights(set(od.home_tid), True).rename(columns={"Tid_500": "home_tid", "OBJECTID": "source"})
wd = weights(set(od.end_tid), False).rename(columns={"Tid_500": "end_tid", "OBJECTID": "target"})
wo.to_csv(OUT / "origin_grid500_community_weights.csv", index=False)
wd.to_csv(OUT / "destination_grid500_community_weights.csv", index=False)

home = od.groupby("home_tid")["count"].sum()
end = od.groupby("end_tid")["count"].sum()
for tag, w, mass, key in [("origin", wo, home, "home_tid"), ("destination", wd, end, "end_tid")]:
    b = w.drop_duplicates(key).set_index(key).basis
    audit[f"{tag}_grids"] = int(len(b))
    audit[f"{tag}_grids_by_basis"] = b.value_counts().to_dict()
    audit[f"{tag}_trips_by_basis"] = {k: int(mass[b.index[b == k]].sum()) for k in b.unique()}
    audit[f"{tag}_grids_split_to_multi_community"] = int((w.groupby(key).size() > 1).sum())

# 拆分
flow = od.merge(wo[["home_tid", "source", "weight"]], on="home_tid").rename(columns={"weight": "wo"})
flow = flow.merge(wd[["end_tid", "target", "weight"]], on="end_tid").rename(columns={"weight": "wd"})
flow["value"] = flow["count"] * flow.wo * flow.wd
frac = np.zeros((len(ids), len(ids)))
np.add.at(frac, (flow.source.map(pos).to_numpy(), flow.target.map(pos).to_numpy()), flow.value.to_numpy())
assert np.isclose(frac.sum(), total)


def largest_remainder(values, n):
    values = np.asarray(values, dtype=float)
    scaled = values * (n / values.sum()) if values.sum() > 0 else values
    base = np.floor(scaled).astype(np.int64)
    rest = int(n - base.sum())
    if rest > 0:
        base[np.argsort(-(scaled - base), kind="stable")[:rest]] += 1
    return base


rows = largest_remainder(frac.sum(1), total)
matrix = np.zeros_like(frac, dtype=np.int64)
for i in range(len(ids)):
    if rows[i]:
        matrix[i] = largest_remainder(frac[i], int(rows[i]))
assert int(matrix.sum()) == total

pd.DataFrame(matrix, index=ids, columns=ids).to_csv(OUT / "community_od_matrix_int.csv", index_label="OBJECTID")
long = pd.DataFrame(frac, index=ids, columns=ids).stack().rename("trips").reset_index()
long.columns = ["source", "target", "trips"]
long[long.trips > 0].to_csv(OUT / "community_od_long_fraction.csv", index=False)

summary = node.copy()
summary["n_grid250"] = summary.OBJECTID.map(a250.groupby("OBJECTID").size()).fillna(0).astype(int)
summary["pop_grid250"] = summary.OBJECTID.map(a250.groupby("OBJECTID")["联通人"].sum()).fillna(0).astype(int)
summary["out_trips"] = matrix.sum(1)
summary["in_trips"] = matrix.sum(0)
summary["self_trips"] = np.diag(matrix)
summary["out_trips_fraction"] = frac.sum(1)
summary["in_trips_fraction"] = frac.sum(0)
summary.to_csv(OUT / "community_od_summary.csv", index=False)

zero500 = pd.read_csv(RES / "community_grid500_summary.csv", dtype={"SQCODE": str})
zero500 = set(zero500.loc[zero500.n_grids == 0, "SQCODE"])
z = summary[summary.SQCODE.isin(zero500)]
audit["assigned_trips"] = int(matrix.sum())
audit["self_trips"] = int(np.trace(matrix))
audit["self_share"] = float(np.trace(matrix) / total)
audit["nonzero_od_pairs"] = int((matrix > 0).sum())
audit["community_zero_out"] = int((summary.out_trips == 0).sum())
audit["community_zero_in"] = int((summary.in_trips == 0).sum())
audit["community_out_lt1_fraction"] = int((summary.out_trips_fraction < 1).sum())
audit["community_in_lt1_fraction"] = int((summary.in_trips_fraction < 1).sum())
audit["community_out_quantiles"] = {str(k): float(v) for k, v in summary.out_trips.quantile([0, .05, .25, .5, .75, .95, 1]).items()}
audit["community_in_quantiles"] = {str(k): float(v) for k, v in summary.in_trips.quantile([0, .05, .25, .5, .75, .95, 1]).items()}
audit["former_zero_grid500_communities"] = len(z)
audit["former_zero_grid500_out_trips_min_median"] = [int(z.out_trips.min()), float(z.out_trips.median())]
audit["former_zero_grid500_in_trips_min_median"] = [int(z.in_trips.min()), float(z.in_trips.median())]
audit["community_n_grid250_quantiles"] = {str(k): float(v) for k, v in summary.n_grid250.quantile([0, .05, .25, .5, .75, .95, 1]).items()}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2, default=int))
