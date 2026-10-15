"""以 250 m 网格人口为总体需求人口，确定各社区（SQCODE）的需求人口。

无争议网格按 maup 结果整格归属社区；有争议网格（最大社区片段面积不足该网格社区覆盖面积的 50%）
按各社区片段内城市建设用地面积占该网格建设用地面积的比例，把网格人口分配给各社区。
城市建设用地取 ESA WorldCover 2020 v100（10 m）的 Built-up 类（值 50），有争议网格内无建设用地时按 maup 整格归属。
联通人口取网格字段 联通人，七普扩样人口取字段 七普加。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "landuse_demand"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM = SRC / "深圳社区objectid.shp"
COMM_POP = SRC / "深圳各社区人口.shp"
NODE = SRC / "node_order.csv"
MAUP = RES / "compare_open_source/grid250_assignment_by_method.csv"
CRS = 4547
DISPUTE = 0.5  # 最大社区片段占社区覆盖面积的比例低于此值即为有争议网格
BUILT = 50  # WorldCover Built-up
WC_URL = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/"
          "ESA_WorldCover_10m_2020_v100_{}_Map.tif")
WC_TILES = {"N21E111": 111, "N21E114": 114}
WC_LOCAL = OUT / "worldcover_2020_shenzhen.tif"
RESO = 1 / 12000

# ---------- 网格、社区、maup 归属 ----------
grid = gpd.read_file(GRID250)[["Tid", "联通人", "七普加", "geometry"]].to_crs(CRS)
grid["grid_area_m2"] = grid.area
maup_a = pd.read_csv(MAUP, dtype=str)[["Tid", "maup"]].rename(columns={"maup": "SQCODE_maup"})
maup_a["Tid"] = maup_a.Tid.astype(int)
grid = grid.merge(maup_a, on="Tid", how="left", validate="one_to_one")
assert grid.SQCODE_maup.notna().all()

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]]


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


comm["geometry"] = comm.geometry.apply(polygonal)

pieces = gpd.overlay(grid[["Tid", "geometry"]], comm, how="intersection", keep_geom_type=True)
pieces["piece_m2"] = pieces.area
pieces = pieces[pieces.piece_m2 > 0]
pieces["area_share"] = pieces.piece_m2 / pieces.groupby("Tid").piece_m2.transform("sum")
top = pieces.groupby("Tid").area_share.max()
grid["max_area_share"] = grid.Tid.map(top)
grid["n_communities"] = grid.Tid.map(pieces.groupby("Tid").size())
grid["disputed"] = grid.max_area_share < DISPUTE
top_sq = pieces.sort_values("area_share", ascending=False).drop_duplicates("Tid").set_index("Tid").SQCODE
maup_same_as_max_area = bool((grid.Tid.map(top_sq) == grid.SQCODE_maup).all())

# ---------- WorldCover 深圳范围 ----------
if not WC_LOCAL.exists():
    x0, y0, x1, y1 = comm.to_crs(4326).total_bounds + np.array([-0.01, -0.01, 0.01, 0.01])
    x0, x1 = np.floor(x0 / RESO) * RESO, np.ceil(x1 / RESO) * RESO
    y0, y1 = np.floor(y0 / RESO) * RESO, np.ceil(y1 / RESO) * RESO
    parts = []
    for tile, lon0 in WC_TILES.items():
        a, b = max(x0, lon0), min(x1, lon0 + 3)
        if a >= b:
            continue
        with rasterio.open(WC_URL.format(tile)) as s:
            w = from_bounds(a, y0, b, y1, s.transform).round_offsets().round_lengths()
            parts.append(s.read(1, window=w))
    arr = np.hstack(parts)
    with rasterio.open(WC_LOCAL, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(x0, y1, RESO, RESO),
                       compress="deflate") as d:
        d.write(arr, 1)

# ---------- 有争议网格：各社区片段内的建设用地面积 ----------
dp = pieces[pieces.Tid.isin(grid.loc[grid.disputed, "Tid"])].copy()
with rasterio.open(WC_LOCAL) as s:
    built_frac = []
    for g in dp.to_crs(4326).geometry:
        w = from_bounds(*g.bounds, s.transform).round_offsets(op="floor").round_lengths(op="ceil")
        w = Window(w.col_off, w.row_off, w.width + 1, w.height + 1)
        lc = s.read(1, window=w)
        inside = ~geometry_mask([g], out_shape=lc.shape, transform=s.window_transform(w))
        built_frac.append((lc[inside] == BUILT).mean() if inside.any() else 0.0)
dp["builtup_frac"] = built_frac
dp["builtup_m2"] = dp.builtup_frac * dp.piece_m2
dp["grid_builtup_m2"] = dp.groupby("Tid").builtup_m2.transform("sum")
no_built = dp.grid_builtup_m2 == 0
dp["weight"] = np.where(no_built, 0.0, dp.builtup_m2 / dp.grid_builtup_m2.where(~no_built, 1))
fallback_tid = set(dp.loc[no_built, "Tid"])
dp.loc[no_built, "weight"] = (dp.loc[no_built, "SQCODE"].values
                              == dp.loc[no_built, "Tid"].map(grid.set_index("Tid").SQCODE_maup).values).astype(float)
dp["rule"] = np.where(no_built, "maup_无建设用地", "建设用地比例")

# ---------- 网格人口分配到社区 ----------
und = grid.loc[~grid.disputed, ["Tid", "SQCODE_maup"]].rename(columns={"SQCODE_maup": "SQCODE"})
und["weight"] = 1.0
und["rule"] = "maup"
alloc = pd.concat([und, dp[["Tid", "SQCODE", "weight", "rule"]]], ignore_index=True)
alloc = alloc[alloc.weight > 0].merge(grid[["Tid", "联通人", "七普加"]], on="Tid", how="left")
alloc["联通人_分配"] = alloc.联通人 * alloc.weight
alloc["七普加_分配"] = alloc.七普加 * alloc.weight
alloc.sort_values(["Tid", "weight"], ascending=[True, False]).to_csv(
    OUT / "grid250_demand_allocation.csv", index=False, encoding="utf-8-sig")

dp_out = dp.drop(columns="geometry").merge(grid[["Tid", "联通人", "七普加", "SQCODE_maup"]], on="Tid")
dp_out["联通人_分配"] = dp_out.联通人 * dp_out.weight
dp_out["七普加_分配"] = dp_out.七普加 * dp_out.weight
dp_out.sort_values(["Tid", "area_share"], ascending=[True, False]).to_csv(
    OUT / "grid250_disputed_pieces.csv", index=False, encoding="utf-8-sig")

# ---------- 社区需求人口 ----------
names = gpd.read_file(COMM_POP)[["SQCODE", "SQNAME"]]
t = node[["OBJECTID", "SQCODE"]].merge(names, on="SQCODE", how="left", validate="one_to_one")
by = alloc.groupby("SQCODE").agg(需求人口_联通=("联通人_分配", "sum"), 需求人口_七普=("七普加_分配", "sum"))
by_maup = grid.groupby("SQCODE_maup").agg(需求人口_联通_maup=("联通人", "sum"), 需求人口_七普_maup=("七普加", "sum"))
n_dis = dp.groupby("SQCODE").Tid.nunique().rename("n_disputed_grids")
t = t.join(by, on="SQCODE").join(by_maup, on="SQCODE").join(n_dis, on="SQCODE")
t = t.fillna({c: 0 for c in ["需求人口_联通", "需求人口_七普", "需求人口_联通_maup", "需求人口_七普_maup",
                             "n_disputed_grids"]})
t["n_disputed_grids"] = t.n_disputed_grids.astype(int)
t["联通_差值_相对maup"] = t.需求人口_联通 - t.需求人口_联通_maup
t["联通_相对差_相对maup"] = np.where(t.需求人口_联通_maup > 0, t.联通_差值_相对maup / t.需求人口_联通_maup, np.nan)
t.to_csv(OUT / "community_demand_population.csv", index=False, encoding="utf-8-sig")

dg = grid[grid.disputed]
audit = {
    "争议门槛_最大片段占社区覆盖面积": DISPUTE,
    "建设用地数据": "ESA WorldCover 2020 v100 10 m, Built-up (50)",
    "maup与最大面积归属一致": maup_same_as_max_area,
    "网格数": len(grid), "跨社区网格": int((grid.n_communities > 1).sum()),
    "有争议网格": len(dg), "有争议网格_有人口": int((dg.联通人 > 0).sum()),
    "有争议网格_联通人口": int(dg.联通人.sum()), "有争议网格_七普人口": float(dg.七普加.sum()),
    "有争议网格_无建设用地按maup": len(fallback_tid),
    "有争议网格_无建设用地_联通人口": int(grid.loc[grid.Tid.isin(fallback_tid), "联通人"].sum()),
    "有争议网格_片段数": len(dp), "有争议网格_片段建设用地占比中位数": float(dp.builtup_frac.median()),
    "涉及有争议网格的社区": int((t.n_disputed_grids > 0).sum()),
    "联通人口合计": {"网格": int(grid.联通人.sum()), "分配后": float(t.需求人口_联通.sum()),
                 "maup": float(t.需求人口_联通_maup.sum())},
    "七普人口合计": {"网格": float(grid.七普加.sum()), "分配后": float(t.需求人口_七普.sum())},
    "联通需求为0的社区": t.loc[t.需求人口_联通 == 0, ["SQCODE", "SQNAME"]].to_dict("records"),
    "相对maup_联通变化的社区": int((t.联通_差值_相对maup.abs() > 0.5).sum()),
    "相对maup_联通变化人口_绝对值合计的一半": float(t.联通_差值_相对maup.abs().sum() / 2),
    "相对maup_联通相对差超过5%": int((t.联通_相对差_相对maup.abs() > 0.05).sum()),
    "相对maup_联通相对差超过20%": int((t.联通_相对差_相对maup.abs() > 0.2).sum()),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
cols = ["SQCODE", "SQNAME", "n_disputed_grids", "需求人口_联通_maup", "需求人口_联通", "联通_差值_相对maup",
        "联通_相对差_相对maup"]
print(t.reindex(t.联通_差值_相对maup.abs().sort_values(ascending=False).index)[cols].head(15).round(3)
      .to_string(index=False))
