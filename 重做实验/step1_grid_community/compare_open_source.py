"""用 maup 与 TransBigData 对 250 m 网格做社区一对一归属，并与本目录结果对比。

maup.assign：网格被某社区完全覆盖则归该社区，否则归重叠面积最大的社区。
TransBigData.grid_to_area：按网格中心点落入的社区归属。
对比对象：grid250_community_1to1.csv（最大重叠面积）与 grid250_community_b20.csv（面积偏差不超过 20%）。
"""
import json
import time
from pathlib import Path

import geopandas as gpd
import maup
import numpy as np
import pandas as pd
import shapely
import transbigdata as tbd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "compare_open_source"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "input"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM = SRC / "深圳社区objectid.shp"
CRS = 4547
CORE = ("440303", "440304", "440305")


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


comm = gpd.read_file(COMM)[["SQCODE", "SQNAME", "geometry"]]
comm["geometry"] = comm.geometry.apply(polygonal)
grid = gpd.read_file(GRID250)[["Tid", "geometry"]]
timing = {}

# ---------- maup ----------
t0 = time.time()
g_p = grid.to_crs(CRS).reset_index(drop=True)
c_p = comm.to_crs(CRS).reset_index(drop=True)
idx = maup.assign(g_p, c_p)
maup_res = pd.Series(c_p.SQCODE.reindex(idx.to_numpy()).to_numpy(), index=g_p.Tid, name="SQCODE")
timing["maup_s"] = round(time.time() - t0, 1)

# ---------- TransBigData ----------
t0 = time.time()
cen = grid.geometry.centroid  # 经纬度网格的几何中心
ref = int(np.argmin(cen.x + cen.y))
params = {"slon": float(cen.x[ref]), "slat": float(cen.y[ref]), "deltalon": 0.002, "deltalat": 0.002,
          "theta": 0, "method": "rect"}
loncol, latcol = tbd.GPS_to_grid(cen.x, cen.y, params)
cells = pd.DataFrame({"Tid": grid.Tid, "LONCOL": np.asarray(loncol), "LATCOL": np.asarray(latcol)})
hlon, hlat = tbd.grid_to_centre([cells.LONCOL, cells.LATCOL], params)
centre_err = float(np.hypot(np.asarray(hlon) - cen.x.to_numpy(), np.asarray(hlat) - cen.y.to_numpy()).max())
c_w = comm.to_crs(4326)
tb = tbd.grid_to_area(cells, c_w, params, col=["LONCOL", "LATCOL"])
tb_dup = int(tb.Tid.duplicated().sum())
tbd_res = tb.drop_duplicates("Tid").set_index("Tid").SQCODE.reindex(grid.Tid)
timing["transbigdata_s"] = round(time.time() - t0, 1)

# ---------- 本目录结果 ----------
own_max = pd.read_csv(RES / "grid250_community_1to1.csv", dtype={"SQCODE": str}).set_index("Tid").SQCODE.reindex(grid.Tid)
own_b20 = pd.read_csv(RES / "grid250_community_b20.csv", dtype={"SQCODE": str}).set_index("Tid").SQCODE.reindex(grid.Tid)
methods = {"maup": maup_res.reindex(grid.Tid), "transbigdata": tbd_res, "本目录_最大面积": own_max, "本目录_b20": own_b20}

# ---------- 评价指标 ----------
p = pd.read_csv(RES / "grid250_community_pieces.csv", dtype={"SQCODE": str})
p["covered_m2"] = p.groupby("Tid").piece_m2.transform("sum")
p["target_part"] = p.grid_area_m2 * p.piece_m2 / p.covered_m2
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})[["SQCODE", "SQNAME", "comm_area_m2", "touched_m2"]]
cs["target_m2"] = cs.SQCODE.map(p.groupby("SQCODE").target_part.sum())
gi = p.drop_duplicates("Tid").set_index("Tid")[["grid_area_m2", "covered_m2"]].reindex(grid.Tid)
piece = p.set_index(["Tid", "SQCODE"]).piece_m2
core_small = cs.SQCODE.str[:6].isin(CORE) & (cs.comm_area_m2 < 0.3e6)

g = grid.copy()
c = g.geometry.centroid
g["col"] = np.rint((c.x - c.x.min()) / 0.002).astype(int)
g["row"] = np.rint((c.y - c.y.min()) / 0.002).astype(int)
rc = g.set_index("Tid")[["col", "row"]]


def n_parts(assign):
    df = assign.dropna().rename("SQCODE").reset_index().join(rc, on="Tid")
    key = {(a, b): i for i, (a, b) in enumerate(zip(df.col, df.row))}
    code = df.SQCODE.to_numpy()
    a, b = [], []
    for i, (x, y) in enumerate(zip(df.col, df.row)):
        for dx, dy in ((1, 0), (0, 1)):
            j = key.get((x + dx, y + dy))
            if j is not None and code[i] == code[j]:
                a.append(i); b.append(j)
    n = len(df)
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    return df.assign(comp=lab).groupby("SQCODE").comp.nunique()


def evaluate(assign):
    a = assign.rename("SQCODE")
    ok = a.notna()
    df = pd.DataFrame({"Tid": a.index, "SQCODE": a.to_numpy()}).join(gi, on="Tid")
    df["inside_m2"] = [piece.get((t, s), 0.0) if isinstance(s, str) else 0.0 for t, s in zip(df.Tid, df.SQCODE)]
    m = df[ok.to_numpy()]
    got = m.groupby("SQCODE").agg(n=("Tid", "size"), area=("grid_area_m2", "sum"),
                                  inside=("inside_m2", "sum"), cov=("covered_m2", "sum"))
    s = cs.join(got, on="SQCODE")
    s[["n", "area", "inside", "cov"]] = s[["n", "area", "inside", "cov"]].fillna(0)
    s["ratio"] = s.area / s.target_m2
    s["offset"] = ((s["cov"] - s.inside) + (s.touched_m2 - s.inside)) / s.touched_m2
    s["parts"] = s.SQCODE.map(n_parts(a)).fillna(0).astype(int)
    r = s.ratio
    return s, {
        "已归属网格": int(ok.sum()), "未归属网格": int((~ok).sum()),
        "无网格社区": int((s.n == 0).sum()), "单网格社区": int((s.n == 1).sum()),
        "面积比最小": round(float(r.min()), 3), "面积比最大": round(float(r.max()), 3),
        "面积比低于0.8": int((r < 0.8).sum()), "面积比高于1.2": int((r > 1.2).sum()),
        "平均面积偏差": round(float((r - 1).abs().mean()), 4),
        "全市错位面积_km2": round(float((m.covered_m2 - m.inside_m2).sum() / 1e6), 3),
        "错位比例中位数": round(float(s.offset.median()), 3),
        "市中心小社区错位比例中位数": round(float(s.offset[core_small].median()), 3),
        "市中心小社区平均面积偏差": round(float((r[core_small] - 1).abs().mean()), 4),
        "分成多块的社区": int((s.parts > 1).sum()),
    }


table, per = {}, {}
for name, a in methods.items():
    per[name], table[name] = evaluate(a)

agree = {}
names = list(methods)
for i, x in enumerate(names):
    for y in names[i + 1:]:
        both = methods[x].notna() & methods[y].notna()
        agree[f"{x} vs {y}"] = {"一致网格": int((methods[x][both] == methods[y][both]).sum()),
                                "不一致网格": int((methods[x][both] != methods[y][both]).sum()),
                                "任一方未归属": int((~both).sum())}

assign_table = pd.DataFrame({k: v.to_numpy() for k, v in methods.items()}, index=grid.Tid)
assign_table.to_csv(OUT / "grid250_assignment_by_method.csv")
pd.concat({k: v.set_index("SQCODE")[["SQNAME", "n", "ratio", "offset", "parts"]] for k, v in per.items()},
          axis=1).to_csv(OUT / "community_metrics_by_method.csv")
audit = {"timing": timing, "transbigdata_params": params, "transbigdata_centre_error_deg": centre_err,
         "transbigdata_duplicate_matches": tb_dup, "metrics": table, "agreement": agree}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
