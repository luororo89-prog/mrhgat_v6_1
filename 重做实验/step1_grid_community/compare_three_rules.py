"""maup（最大重叠面积）、中心点、混合规则三种 250 m 网格归属的整体空间准确度对比。

几何指标基于网格与社区边界的重叠面积。人口指标以面积加权人口为参照，
即每个网格的联通人口按各社区在网格内的面积比例分摊后汇总。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "hybrid"
GRID250 = (HERE.parent / "input"
           / "深圳250m网格_人口_所在社区.shp")

h = pd.read_csv(OUT / "grid250_community_hybrid.csv", dtype={"SQCODE": str, "SQCODE_maup": str, "centroid_SQCODE": str})
rules = {"maup": h.set_index("Tid").SQCODE_maup, "中心点": h.set_index("Tid").centroid_SQCODE,
         "混合": h.set_index("Tid").SQCODE}

p = pd.read_csv(RES / "grid250_community_pieces.csv", dtype={"SQCODE": str})
p["covered_m2"] = p.groupby("Tid").piece_m2.transform("sum")
p["w"] = p.piece_m2 / p.covered_m2
g = gpd.read_file(GRID250)[["Tid", "联通人", "geometry"]]
p = p.merge(g[["Tid", "联通人"]], on="Tid")
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})[["SQCODE", "comm_area_m2", "touched_m2"]]
cs["target_m2"] = cs.SQCODE.map((p.grid_area_m2 * p.w).groupby(p.SQCODE).sum())
cs["ref_pop"] = cs.SQCODE.map((p["联通人"] * p.w).groupby(p.SQCODE).sum()).fillna(0)
small = cs.comm_area_m2 < 0.3e6
cell = p.drop_duplicates("Tid").set_index("Tid")[["grid_area_m2", "covered_m2"]]
piece = p.set_index(["Tid", "SQCODE"])[["piece_m2", "w"]]
pop = g.set_index("Tid")["联通人"]

c = g.geometry.centroid
rc = pd.DataFrame({"col": np.rint((c.x - c.x.min()) / 0.002).astype(int).to_numpy(),
                   "row": np.rint((c.y - c.y.min()) / 0.002).astype(int).to_numpy()}, index=g.Tid)


def n_parts(a):
    df = a.dropna().rename("SQCODE").to_frame().join(rc)
    key = {(x, y): i for i, (x, y) in enumerate(zip(df.col, df.row))}
    code = df.SQCODE.to_numpy()
    u, v = [], []
    for i, (x, y) in enumerate(zip(df.col, df.row)):
        for dx, dy in ((1, 0), (0, 1)):
            j = key.get((x + dx, y + dy))
            if j is not None and code[i] == code[j]:
                u.append(i); v.append(j)
    n = len(df)
    _, lab = connected_components(coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)), directed=False)
    return df.assign(comp=lab).groupby("SQCODE").comp.nunique()


res = {}
for name, a in rules.items():
    df = a.rename("SQCODE").to_frame().join(cell).join(pop)
    idx = pd.MultiIndex.from_arrays([df.index, df.SQCODE])
    df["inside_m2"] = piece.piece_m2.reindex(idx).fillna(0).to_numpy()
    df["share"] = piece.w.reindex(idx).fillna(0).to_numpy()
    got = df.dropna(subset=["SQCODE"]).groupby("SQCODE").agg(
        n=("SQCODE", "size"), area=("grid_area_m2", "sum"), inside=("inside_m2", "sum"),
        cov=("covered_m2", "sum"), dpop=("联通人", "sum"))
    s = cs.join(got, on="SQCODE").fillna({"n": 0, "area": 0, "inside": 0, "cov": 0, "dpop": 0})
    s["ratio"] = s.area / s.target_m2
    s["offset"] = ((s["cov"] - s.inside) + (s.touched_m2 - s.inside)) / s.touched_m2
    s["pop_err"] = (s.dpop - s.ref_pop).abs() / s.ref_pop
    parts = n_parts(a)
    total_pop = float(pop.sum())
    res[name] = {
        "未归属网格": int(a.isna().sum()),
        "全市错位面积_km2": round(float((df.covered_m2 - df.inside_m2).sum() / 1e6), 3),
        "网格层面错配人口": int(round(float((df["联通人"] * (1 - df.share)).sum()))),
        "社区层面错配人口": int(round(float((s.dpop - s.ref_pop).abs().sum() / 2))),
        "社区层面错配人口占比": round(float((s.dpop - s.ref_pop).abs().sum() / 2 / total_pop), 4),
        "人口偏差中位数": round(float(s.pop_err.median()), 4),
        "人口偏差均值": round(float(s.pop_err.mean()), 4),
        "人口偏差超过20%社区": int((s.pop_err > 0.2).sum()),
        "人口偏差超过50%社区": int((s.pop_err > 0.5).sum()),
        "小社区人口偏差中位数": round(float(s.pop_err[small].median()), 4),
        "大社区人口偏差中位数": round(float(s.pop_err[~small].median()), 4),
        "面积偏差均值": round(float((s.ratio - 1).abs().mean()), 4),
        "面积比低于0.8或高于1.2": int(((s.ratio < 0.8) | (s.ratio > 1.2)).sum()),
        "错位比例中位数": round(float(s.offset.median()), 3),
        "单网格社区": int((s.n == 1).sum()),
        "分成多块的社区": int((s.SQCODE.map(parts).fillna(0) > 1).sum()),
    }
(OUT / "compare_three_rules.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(pd.DataFrame(res).to_string())
