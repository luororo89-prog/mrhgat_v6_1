"""250 m 网格与社区一对一归属，社区面积不过大也不过小，同时空间偏移尽量少。

每个网格整格归入一个与它重叠的社区。社区的应得面积为各网格面积按重叠比例摊到该社区的总和，
海岸和市界网格超出社区边界的部分按同样比例摊给与其重叠的社区。
每个社区分得面积与应得面积的偏差比例不超过允许范围，允许范围取 BAND 与整格分配可达到的最小偏差两者中的较大值。
在此前提下，整数规划使全部网格的错位面积最小。错位面积指网格中不属于其归入社区的部分。
重叠比例低于 MIN_SHARE 的社区不作为该网格的候选。
网格改归非最大重叠社区时，其上下左右至少一个相邻网格须归入同一社区，避免出现孤立的单格飞地。
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GRID250 = (HERE.parent / "input"
           / "深圳250m网格_人口_所在社区.shp")
MIN_SHARE = 0.10   # 候选社区在网格内的最低重叠比例
OVER = 1000.0      # 超出允许范围的惩罚，相对偏差比例的单位
EPS = 0.001        # 允许范围内偏差的次要惩罚
CORE = ("440303", "440304", "440305")  # 罗湖、福田、南山

ap = argparse.ArgumentParser()
ap.add_argument("--band", type=float, default=0.2, help="允许的面积偏差比例")
args = ap.parse_args()
BAND = args.band
tag = f"b{int(round(BAND * 100)):02d}"

p = pd.read_csv(RES / "grid250_community_pieces.csv", dtype={"SQCODE": str})
base = pd.read_csv(RES / "grid250_community_1to1.csv", dtype={"SQCODE": str})
comm = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})

p["covered_m2"] = p.groupby("Tid").piece_m2.transform("sum")
p["nshare"] = p.piece_m2 / p.covered_m2
p["target_part"] = p.grid_area_m2 * p.nshare
p["misplaced_m2"] = p.covered_m2 - p.piece_m2
target = p.groupby("SQCODE").target_part.sum()
codes = comm.SQCODE.to_numpy()
cpos = {c: i for i, c in enumerate(codes)}
T = target.reindex(codes).to_numpy()
cell = float(p.grid_area_m2.median())
Tc = T / cell
k = np.maximum(np.rint(Tc), 1)
limit = np.abs(k - Tc) / Tc                      # 整格分配可达到的最小偏差
allow = np.maximum(BAND, limit + 0.005)

# 候选：重叠比例不低于 MIN_SHARE，各网格的最大重叠社区始终保留
p = p.sort_values(["Tid", "nshare"], ascending=[True, False])
p["rank"] = p.groupby("Tid").cumcount()
cand = p[(p.nshare >= MIN_SHARE) | (p["rank"] == 0)].reset_index(drop=True)
tids = cand.Tid.unique()
tpos = {t: i for i, t in enumerate(tids)}
nx, nc, nt = len(cand), len(codes), len(tids)

# 网格行列号（上下左右相邻）
g = gpd.read_file(GRID250)[["Tid", "geometry"]]
cen = g.geometry.centroid
g["col"] = np.rint((cen.x - cen.x.min()) / 0.002).astype(int)
g["row"] = np.rint((cen.y - cen.y.min()) / 0.002).astype(int)
assert not g.duplicated(["col", "row"]).any()
rc = g.set_index("Tid")[["col", "row"]]
cand = cand.join(rc, on="Tid")

# 变量 [x (nx), dplus (nc), dminus (nc), over (nc)]，面积以网格面积为单位
nv = nx + 3 * nc
ci = cand.SQCODE.map(cpos).to_numpy()
ti = cand.Tid.map(tpos).to_numpy()
area = cand.grid_area_m2.to_numpy() / cell
cost = np.concatenate([cand.misplaced_m2.to_numpy() / cell, EPS / Tc, EPS / Tc, OVER / Tc])
# 每个网格恰好归入一个社区
A1 = coo_matrix((np.ones(nx), (ti, np.arange(nx))), shape=(nt, nv))
# 分得面积 - dplus + dminus = 应得面积
A2 = coo_matrix((np.concatenate([area, -np.ones(nc), np.ones(nc)]),
                 (np.concatenate([ci, np.arange(nc), np.arange(nc)]),
                  np.concatenate([np.arange(nx), nx + np.arange(nc), nx + nc + np.arange(nc)]))), shape=(nc, nv))
# dplus + dminus - over <= 允许偏差
A3 = coo_matrix((np.concatenate([np.ones(nc), np.ones(nc), -np.ones(nc)]),
                 (np.tile(np.arange(nc), 3), nx + np.arange(3 * nc))), shape=(nc, nv))
# 每个社区至少一个网格
A4 = coo_matrix((np.ones(nx), (ci, np.arange(nx))), shape=(nc, nv))
# 改归的网格须与同社区网格相邻
var = {(c, r, q): i for i, (c, r, q) in enumerate(zip(cand.col, cand.row, cand.SQCODE))}
ra, rb, rv, n5 = [], [], [], 0
for i in np.flatnonzero(cand["rank"].to_numpy() > 0):
    c, r, q = cand.col[i], cand.row[i], cand.SQCODE[i]
    ra.append(n5); rb.append(i); rv.append(1.0)
    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        j = var.get((c + dc, r + dr, q))
        if j is not None:
            ra.append(n5); rb.append(j); rv.append(-1.0)
    n5 += 1
A5 = coo_matrix((rv, (ra, rb)), shape=(n5, nv))
cons = [LinearConstraint(A1, 1, 1), LinearConstraint(A2, Tc, Tc), LinearConstraint(A3, -np.inf, allow * Tc),
        LinearConstraint(A4, 1, np.inf), LinearConstraint(A5, -np.inf, 0)]
integrality = np.concatenate([np.ones(nx), np.zeros(3 * nc)])
bounds = Bounds(np.zeros(nv), np.concatenate([np.ones(nx), np.full(3 * nc, np.inf)]))
sol = milp(cost, constraints=cons, integrality=integrality, bounds=bounds,
           options={"time_limit": 1800, "mip_rel_gap": 1e-4, "disp": False})
assert sol.x is not None, sol.message
x = sol.x[:nx] > 0.5
chosen = cand.loc[x, ["Tid", "SQCODE", "nshare", "rank"]]
assert chosen.Tid.is_unique and len(chosen) == base.Tid.nunique()

out = base[["Tid", "grid_area_m2", "SQCODE", "max_share", "cover_share", "n_communities"]].rename(
    columns={"SQCODE": "SQCODE_max_area"})
out = out.merge(chosen.rename(columns={"nshare": "assigned_share_of_covered"}), on="Tid")
out["changed"] = out.SQCODE != out.SQCODE_max_area
out = out.merge(comm[["SQCODE", "SQNAME"]], on="SQCODE", how="left")


def components(assign):
    df = assign.join(rc, on="Tid")
    key = {(c, r): i for i, (c, r) in enumerate(zip(df.col, df.row))}
    code = df.SQCODE.to_numpy()
    a, b = [], []
    for i, (c, r) in enumerate(zip(df.col, df.row)):
        for dc, dr in ((1, 0), (0, 1)):
            j = key.get((c + dc, r + dr))
            if j is not None and code[i] == code[j]:
                a.append(i); b.append(j)
    n = len(df)
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    df["comp"] = lab
    return df.groupby("SQCODE").comp.nunique()


def summary(assign):
    s = comm[["SQCODE", "SQNAME", "comm_area_m2", "touched_m2"]].copy()
    s["target_m2"] = T
    s["allowed_dev"] = allow
    m = assign.merge(p[["Tid", "SQCODE", "piece_m2", "covered_m2", "grid_area_m2"]].drop_duplicates(["Tid", "SQCODE"]),
                     on=["Tid", "SQCODE"], how="left")
    got = m.groupby("SQCODE").agg(n_grids=("Tid", "size"), assigned_m2=("grid_area_m2", "sum"),
                                  inside_m2=("piece_m2", "sum"), assigned_cov_m2=("covered_m2", "sum"))
    s = s.join(got, on="SQCODE")
    s["n_grids"] = s.n_grids.fillna(0).astype(int)
    s["ratio_to_target"] = s.assigned_m2 / s.target_m2
    s["ratio_to_area"] = s.assigned_m2 / s.comm_area_m2
    # 错位比例：分到网格中不属于本社区的面积与本社区落在其他社区网格中的面积之和，除以本社区面积
    s["offset_ratio"] = ((s.assigned_cov_m2 - s.inside_m2) + (s.touched_m2 - s.inside_m2)) / s.touched_m2
    s["own_area_covered"] = s.inside_m2 / s.touched_m2
    s["n_parts"] = s.SQCODE.map(components(assign[["Tid", "SQCODE"]])).fillna(0).astype(int)
    r = s.ratio_to_target
    core = s.SQCODE.str[:6].isin(CORE) & (s.comm_area_m2 < 0.3e6)
    q = lambda v: {str(k): round(float(x), 3) for k, x in v.quantile([0, .05, .5, .95, 1]).items()}
    a = {"misplaced_km2": round(float((m.covered_m2 - m.piece_m2).sum() / 1e6), 3),
         "ratio_quantiles": q(r), "ratio_lt_0.8": int((r < 0.8).sum()), "ratio_gt_1.2": int((r > 1.2).sum()),
         "outside_allowed": int(((r - 1).abs() > s.allowed_dev + 1e-6).sum()),
         "mean_abs_dev": round(float((r - 1).abs().mean()), 4),
         "offset_ratio_quantiles": q(s.offset_ratio),
         "one_grid": int((s.n_grids == 1).sum()), "multi_part_communities": int((s.n_parts > 1).sum()),
         "core_small_n": int(core.sum()), "core_small_ratio_range": [round(float(r[core].min()), 3), round(float(r[core].max()), 3)],
         "core_small_mean_abs_dev": round(float((r[core] - 1).abs().mean()), 4),
         "core_small_offset_ratio_median": round(float(s.offset_ratio[core].median()), 3),
         "core_small_own_area_covered_median": round(float(s.own_area_covered[core].median()), 3)}
    return s, a


s0, a0 = summary(base[["Tid", "SQCODE"]])
s1, a1 = summary(out[["Tid", "SQCODE"]])
s1 = s1.merge(s0[["SQCODE", "n_grids", "ratio_to_target", "offset_ratio", "n_parts"]].rename(
    columns={"n_grids": "n_grids_max_area", "ratio_to_target": "ratio_max_area",
             "offset_ratio": "offset_ratio_max_area", "n_parts": "n_parts_max_area"}), on="SQCODE")

ch = out[out.changed]
audit = {"band": BAND, "min_share": MIN_SHARE, "solver_status": sol.message, "mip_gap": getattr(sol, "mip_gap", None),
         "cells": int(len(out)), "cells_changed": int(len(ch)),
         "changed_assigned_share_quantiles": {str(k): round(float(v), 3) for k, v in
                                              ch.assigned_share_of_covered.quantile([0, .1, .5, .9, 1]).items()} if len(ch) else {},
         "changed_share_lt_0.3": int((ch.assigned_share_of_covered < 0.3).sum()),
         "max_area": a0, "assigned": a1}
out.to_csv(RES / f"grid250_community_{tag}.csv", index=False)
s1.to_csv(RES / f"community_grid250_{tag}_summary.csv", index=False)
(RES / f"audit_{tag}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
