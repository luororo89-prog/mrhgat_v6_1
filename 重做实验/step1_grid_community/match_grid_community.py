"""网格与社区一对一归属：每个网格归入与其重叠面积最大的社区，社区以 SQCODE 为唯一键。"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.validation import make_valid

SRC = Path(__file__).resolve().parent.parent / "input"
GRID500 = SRC / "SZ_grid_500m.shp"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM = SRC / "深圳社区objectid.shp"
COMM_POP = SRC / "深圳各社区人口.shp"
NET = Path(__file__).resolve().parent.parent / "input"
OD = NET / "outpatient_grid_OD_home_tid.csv"
NODE = NET / "node_order.csv"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
CRS = 4547  # 社区边界原始投影 CGCS2000 3度带 114E

audit = {}

# ---------- 社区边界与属性核对 ----------
comm = gpd.read_file(COMM)
audit["community_n"] = len(comm)
audit["community_sqcode_unique"] = bool(comm.SQCODE.is_unique)
audit["community_objectid_unique"] = bool(comm.OBJECTID.is_unique)
audit["community_objectid_nunique"] = int(comm.OBJECTID.nunique())
bad_attr = comm.loc[comm.SQCODE != comm.SQCODE_1,
                    ["SQCODE", "SQNAME", "OBJECTID", "FID_1", "SQCODE_1", "联通人"]]
audit["community_rows_sqcode_ne_sqcode1"] = len(bad_attr)
bad_attr.sort_values(["OBJECTID", "SQCODE"]).to_csv(OUT / "community_attribute_mismatch.csv", index=False)

pop = gpd.read_file(COMM_POP)[["SQCODE", "联通人"]].rename(columns={"联通人": "联通人_社区人口表"})
comm = comm.merge(pop, on="SQCODE", how="left")
audit["community_pop_mismatch_rows"] = int((comm["联通人"] != comm["联通人_社区人口表"]).sum())
audit["community_pop_sum_objectid_shp"] = int(comm["联通人"].sum())
audit["community_pop_sum_pop_shp"] = int(comm["联通人_社区人口表"].sum())

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
audit["node_order_n"] = len(node)
audit["node_order_objectid_nunique"] = int(node.OBJECTID.nunique())
audit["node_order_sqcode_same_as_shp"] = bool(set(node.SQCODE) == set(comm.SQCODE))
# 边界文件的 OBJECTID 有重复，节点编号改用 node_order 中按 SQCODE 对应的 OBJECTID
comm = comm.drop(columns="OBJECTID").merge(node[["SQCODE", "OBJECTID"]], on="SQCODE", how="left")

invalid = comm.loc[~comm.is_valid, ["SQCODE", "SQNAME"]]
audit["community_invalid"] = invalid.to_dict("records")


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


comm["geometry"] = comm.geometry.apply(polygonal)
comm["comm_area_m2"] = comm.area
comm = comm[["SQCODE", "SQNAME", "OBJECTID", "联通人", "联通人_社区人口表", "comm_area_m2", "geometry"]]

# 社区之间的重叠
pairs = gpd.sjoin(comm[["SQCODE", "geometry"]], comm[["SQCODE", "geometry"]], predicate="intersects")
pairs = pairs[pairs.SQCODE_left < pairs.SQCODE_right]
geom = comm.set_index("SQCODE").geometry
pairs["overlap_m2"] = [geom[a].intersection(geom[b]).area
                       for a, b in zip(pairs.SQCODE_left, pairs.SQCODE_right)]
ov = pairs.loc[pairs.overlap_m2 > 1, ["SQCODE_left", "SQCODE_right", "overlap_m2"]]
ov.to_csv(OUT / "community_overlaps.csv", index=False)
audit["community_overlap_pairs_gt1m2"] = len(ov)
audit["community_overlap_pairs_gt1000m2"] = int((ov.overlap_m2 > 1000).sum())
audit["community_overlap_total_m2"] = float(ov.overlap_m2.sum())

# 社区并集内部空洞（相邻社区之间的缝隙）
union = shapely.union_all(comm.geometry.values)
holes = [shapely.Polygon(h).area for poly in getattr(union, "geoms", [union]) for h in poly.interiors]
audit["community_union_holes_n"] = len(holes)
audit["community_union_holes_gt1000m2"] = sum(a > 1000 for a in holes)
audit["community_union_holes_total_m2"] = float(sum(holes))

# ---------- OD ----------
od = pd.read_csv(OD, encoding="utf-8-sig")
od_home = od.groupby("home_tid")["count"].sum()
od_end = od.groupby("end_tid")["count"].sum()
audit["od_records"] = len(od)
audit["od_trips"] = int(od["count"].sum())


def match(grid_path, tag, keep, rename=None):
    grid = gpd.read_file(grid_path).to_crs(CRS).rename(columns=rename or {})
    a = {"grid_n": len(grid), "tid_unique": bool(grid.Tid.is_unique)}
    grid["grid_area_m2"] = grid.area
    b = grid.geometry.bounds
    a["cell_width_m_median"] = float((b.maxx - b.minx).median())
    a["cell_height_m_median"] = float((b.maxy - b.miny).median())
    a["cell_area_m2_median"] = float(grid.grid_area_m2.median())

    pieces = gpd.overlay(grid[["Tid", "grid_area_m2", "geometry"]], comm[["SQCODE", "geometry"]],
                         how="intersection", keep_geom_type=True)
    pieces["piece_m2"] = pieces.area
    pieces = pieces[pieces.piece_m2 > 0]
    pieces["share"] = pieces.piece_m2 / pieces.grid_area_m2
    pieces = pieces.sort_values(["Tid", "piece_m2"], ascending=[True, False])
    pieces.drop(columns="geometry").to_csv(OUT / f"{tag}_community_pieces.csv", index=False)

    top = pieces.groupby("Tid").nth(0).set_index("Tid")
    second = pieces.groupby("Tid").nth(1).set_index("Tid")
    res = grid[["Tid", "grid_area_m2"] + keep].set_index("Tid")
    res["SQCODE"] = top.SQCODE
    res["max_share"] = top.share
    res["cover_share"] = pieces.groupby("Tid").share.sum()
    res["n_communities"] = pieces.groupby("Tid").size()
    res["second_SQCODE"] = second.SQCODE
    res["second_share"] = second.share
    res[["cover_share", "max_share"]] = res[["cover_share", "max_share"]].fillna(0)
    res["n_communities"] = res.n_communities.fillna(0).astype(int)
    res["tie"] = (res.max_share - res.second_share).abs() < 1e-6

    cen = grid[["Tid", "geometry"]].copy()
    cen["geometry"] = cen.geometry.centroid
    hit = gpd.sjoin(cen, comm[["SQCODE", "geometry"]], predicate="within")
    a["centroid_multi_hit"] = int(hit.Tid.duplicated().sum())
    res["centroid_SQCODE"] = hit.drop_duplicates("Tid").set_index("Tid").SQCODE
    res["centroid_agree"] = res.centroid_SQCODE == res.SQCODE
    res = res.reset_index().merge(comm[["SQCODE", "SQNAME"]], on="SQCODE", how="left")

    has = res.SQCODE.notna()
    a["no_community"] = int((~has).sum())
    a["single_community"] = int((res.n_communities == 1).sum())
    a["multi_community"] = int((res.n_communities > 1).sum())
    a["max_share_lt_half_of_covered"] = int((has & (res.max_share / res.cover_share < 0.5)).sum())
    a["ties"] = int(res.tie.sum())
    a["centroid_outside_all"] = int((has & res.centroid_SQCODE.isna()).sum())
    a["centroid_disagree"] = int((has & res.centroid_SQCODE.notna() & ~res.centroid_agree).sum())
    a["partial_cover_lt_0.5"] = int(((res.cover_share > 0) & (res.cover_share < 0.5)).sum())
    a["partial_cover_lt_0.99"] = int(((res.cover_share > 0) & (res.cover_share < 0.99)).sum())

    per = res[has].groupby("SQCODE").agg(n_grids=("Tid", "size"), assigned_grid_m2=("grid_area_m2", "sum"))
    touch = pieces.groupby("SQCODE").agg(n_touch=("Tid", "nunique"), touched_m2=("piece_m2", "sum"),
                                         max_share_in_any_grid=("share", "max"))
    cs = comm.drop(columns="geometry").set_index("SQCODE").join(per).join(touch).reset_index()
    cs[["n_grids", "n_touch"]] = cs[["n_grids", "n_touch"]].fillna(0).astype(int)
    cs["assigned_grid_m2"] = cs.assigned_grid_m2.fillna(0)
    cs["assigned_area_ratio"] = cs.assigned_grid_m2 / cs.comm_area_m2
    a["community_zero_grid"] = int((cs.n_grids == 0).sum())
    a["community_one_grid"] = int((cs.n_grids == 1).sum())
    a["community_n_grids_quantiles"] = {str(k): float(v) for k, v in
                                        cs.n_grids.quantile([0, .1, .25, .5, .75, .9, 1]).items()}
    return res, cs, a


# ---------- 500 m 网格（门诊 OD 所用网格） ----------
res500, cs500, a500 = match(GRID500, "grid500", [])
res500.insert(res500.columns.get_loc("SQCODE") + 1, "OBJECTID",
              res500.SQCODE.map(comm.set_index("SQCODE").OBJECTID).astype("Int64"))
res500["od_home_trips"] = res500.Tid.map(od_home).fillna(0).astype(int)
res500["od_end_trips"] = res500.Tid.map(od_end).fillna(0).astype(int)
a500["od_tid_not_in_grid"] = len((set(od.home_tid) | set(od.end_tid)) - set(res500.Tid))
nc = res500.SQCODE.isna()
a500["od_grids_no_community"] = int((nc & ((res500.od_home_trips > 0) | (res500.od_end_trips > 0))).sum())
a500["od_home_trips_in_no_community_grid"] = int(res500.loc[nc, "od_home_trips"].sum())
a500["od_end_trips_in_no_community_grid"] = int(res500.loc[nc, "od_end_trips"].sum())
multi = res500.n_communities > 1
a500["od_home_trips_in_multi_grid"] = int(res500.loc[multi, "od_home_trips"].sum())
a500["od_end_trips_in_multi_grid"] = int(res500.loc[multi, "od_end_trips"].sum())
cs500 = cs500.merge(res500.groupby("SQCODE")[["od_home_trips", "od_end_trips"]].sum(),
                    left_on="SQCODE", right_index=True, how="left")
cs500[["od_home_trips", "od_end_trips"]] = cs500[["od_home_trips", "od_end_trips"]].fillna(0).astype(int)
z = cs500.n_grids == 0
a500["zero_grid_community_pop"] = int(cs500.loc[z, "联通人_社区人口表"].sum())
a500["community_zero_home_trips"] = int((cs500.od_home_trips == 0).sum())
a500["community_zero_end_trips"] = int((cs500.od_end_trips == 0).sum())
a500["community_zero_home_and_end_trips"] = int(((cs500.od_home_trips == 0) & (cs500.od_end_trips == 0)).sum())
res500.to_csv(OUT / "grid500_community_1to1.csv", index=False)
cs500.to_csv(OUT / "community_grid500_summary.csv", index=False)

# ---------- 250 m 居住人口网格 ----------
res250, cs250, a250 = match(GRID250, "grid250", ["SQCODE_in_file", "联通人"], {"SQCODE": "SQCODE_in_file"})
a250["file_sqcode_not_in_community"] = int((~res250.SQCODE_in_file.astype(str).isin(comm.SQCODE)).sum())
has = res250.SQCODE.notna()
dis = has & (res250.SQCODE_in_file.astype(str) != res250.SQCODE)
a250["file_sqcode_disagree_max_area"] = int(dis.sum())
a250["file_sqcode_disagree_pop"] = int(res250.loc[dis, "联通人"].sum())
a250["no_community_grids"] = int((~has).sum())
a250["no_community_pop"] = int(res250.loc[~has, "联通人"].sum())
a250["pop_total"] = int(res250["联通人"].sum())
z = cs250.n_grids == 0
a250["zero_grid_community_pop"] = int(cs250.loc[z, "联通人_社区人口表"].sum())
res250.to_csv(OUT / "grid250_community_1to1.csv", index=False)
cs250.to_csv(OUT / "community_grid250_summary.csv", index=False)

audit["grid500"] = a500
audit["grid250"] = a250
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2, default=float))
