"""社康与医院 AOI 的所在社区归属。

社康取 facility_supply_audit_2018_mobile_users.csv 中 tier 为 CHC 的 912 个点位，点位落在哪个社区边界内即归属该社区；
落在全部社区边界之外的点位不归属社区，记录其到最近社区的距离。
医院取 hospital_aoi_35.shp 的 40 个 AOI（23 家三级、17 家二级），AOI 与社区边界做面相交，归属重叠面积最大的社区，
同时记录 AOI 覆盖的全部社区及面积比例。社区以 SQCODE 识别，OBJECTID 取 node_order.csv。
另记录设施是否落在所在社区的生活圈内（社康按点位，医院按 AOI 相交），生活圈与供给评价一致，
取社区 15 分钟步行等时圈，无等时圈的社区取社区边界。
facility_corrections.csv 中的社康按核对结果修正：站点以相邻社区命名、点位在边界外 500 m 内的，归属改为站点命名的社区（点位不变）；
地址与点位相差较远的（东乐站），点位改为修正坐标。点位所在社区另存为 SQCODE_point。
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
FAC = SRC / "facility_supply_audit_2018_mobile_users.csv"
AOI = SRC / "hospital_aoi_35.shp"
COMM = SRC / "深圳社区objectid.shp"
NODE = SRC / "node_order.csv"
CORR = HERE / "facility_corrections.csv"
ISO = HERE.parents[1] / "1input_sources/等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp"
CRS = 4547


def polygonal(g):
    g = g if g.is_valid else make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])


node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("SQCODE").OBJECTID
comm = gpd.read_file(COMM)[["SQCODE", "SQNAME", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.apply(polygonal)
assert comm.SQCODE.is_unique and comm.SQCODE.isin(node.index).all()
name = comm.set_index("SQCODE").SQNAME
iso = gpd.read_file(ISO).to_crs(CRS)
iso["SQCODE"] = iso.num.astype(str).str.replace(r"\.0$", "", regex=True)
iso["geometry"] = iso.geometry.apply(polygonal)
iso = iso[iso.SQCODE.isin(node.index)][["SQCODE", "geometry"]]
no_iso = sorted(set(node.index) - set(iso.SQCODE))
iso = pd.concat([iso, comm[comm.SQCODE.isin(no_iso)][["SQCODE", "geometry"]]], ignore_index=True)

fac = pd.read_csv(FAC)
audit = {}

# ---------- 社康：点位所在社区 ----------
corr = pd.read_csv(CORR, encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("facility_id")
chc = fac[fac.tier == "CHC"][["facility_id", "facility_name", "tier", "capacity", "longitude", "latitude"]].copy()
moved = corr.longitude.dropna().index
chc.loc[chc.facility_id.isin(moved), "longitude"] = chc.facility_id.map(corr.longitude)
chc.loc[chc.facility_id.isin(moved), "latitude"] = chc.facility_id.map(corr.latitude)
chc = gpd.GeoDataFrame(chc, geometry=gpd.points_from_xy(chc.longitude, chc.latitude), crs=4326).to_crs(CRS)
inside = gpd.sjoin(chc, comm[["SQCODE", "geometry"]], how="left", predicate="within")
assert not inside.index.duplicated().any()
chc["SQCODE"] = inside.SQCODE
chc["rule"] = "点位在社区内"
chc["dist_to_community_m"] = 0.0
out_idx = chc.SQCODE.isna()
if out_idx.any():
    near = gpd.sjoin_nearest(chc[out_idx].drop(columns=["SQCODE"]), comm[["SQCODE", "geometry"]],
                             how="left", distance_col="d")
    near = near[~near.index.duplicated()]
    chc.loc[out_idx, "dist_to_community_m"] = near.d
    chc.loc[out_idx, "rule"] = "点位在全部社区边界外，不归属"
chc["SQCODE_point"] = chc.SQCODE
fix = chc.facility_id.isin(corr.index)
chc.loc[fix, "SQCODE"] = chc.loc[fix, "facility_id"].map(corr.SQCODE)
chc.loc[fix, "rule"] = chc.loc[fix, "facility_id"].map(corr.依据)
chc["n_communities"] = 1
chc["area_share"] = 1.0
chc["other_communities"] = ""
audit["社康"] = {"点位": len(chc), "点位在社区内": int((~out_idx).sum()), "点位在社区外": int(out_idx.sum()),
               "社区外点位": [f"{n}（距最近社区 {d:.0f} m）" for n, d in
                         zip(chc.loc[out_idx, "facility_name"], chc.loc[out_idx, "dist_to_community_m"])],
               "按核对修正归属": int(fix.sum()), "修正点位": len(moved),
               "修正后与点位所在社区不同": int((chc.SQCODE != chc.SQCODE_point).sum())}

# ---------- 医院：AOI 所在社区 ----------
aoi = gpd.read_file(AOI).to_crs(CRS)
aoi["geometry"] = aoi.geometry.apply(polygonal)
hosp = fac[fac.tier != "CHC"][["facility_id", "facility_name", "tier", "capacity", "longitude", "latitude"]]
assert set(hosp.facility_name) == set(aoi.fac_name) and aoi.fac_name.is_unique
aoi = aoi.merge(hosp, left_on="fac_name", right_on="facility_name", how="left", validate="one_to_one")
aoi = aoi.rename(columns={"tier_x": "tier"}).drop(columns=["tier_y"])
ac = gpd.overlay(aoi[["facility_id", "geometry"]], comm[["SQCODE", "geometry"]], how="intersection",
                 keep_geom_type=True)
ac["m2"] = ac.area
ac = ac[ac.m2 > 0]
ac["area_share"] = ac.m2 / ac.groupby("facility_id").m2.transform("sum")
ac = ac.sort_values(["facility_id", "m2"], ascending=[True, False])
top = ac.drop_duplicates("facility_id").set_index("facility_id")
others = ac[ac.duplicated("facility_id")].groupby("facility_id")[["SQCODE", "area_share"]].apply(
    lambda d: "、".join(f"{name[s]}({s}) {a:.1%}" for s, a in zip(d.SQCODE, d.area_share)))
aoi["SQCODE"] = aoi.facility_id.map(top.SQCODE)
aoi["area_share"] = aoi.facility_id.map(top.area_share)
aoi["n_communities"] = aoi.facility_id.map(ac.groupby("facility_id").size())
aoi["other_communities"] = aoi.facility_id.map(others).fillna("")
aoi["rule"] = "AOI 与社区重叠面积最大"
aoi["dist_to_community_m"] = 0.0
assert aoi.SQCODE.notna().all()
pt = gpd.GeoDataFrame(aoi[["facility_id"]], geometry=gpd.points_from_xy(aoi.longitude, aoi.latitude),
                      crs=4326).to_crs(CRS)
pt_sq = gpd.sjoin(pt, comm[["SQCODE", "geometry"]], how="left", predicate="within").set_index("facility_id").SQCODE
aoi["SQCODE_supply_point"] = aoi.facility_id.map(pt_sq)
aoi["SQCODE_aoi_field"] = aoi.sqcode.astype(str)
audit["医院"] = {
    "AOI": len(aoi), "三级": int((aoi.tier == "TH").sum()), "二级": int((aoi.tier == "SH").sum()),
    "AOI 跨多个社区": int((aoi.n_communities > 1).sum()),
    "最大社区面积比例最小值": round(float(aoi.area_share.min()), 4),
    "所在社区与 AOI 表 sqcode 字段一致": int((aoi.SQCODE == aoi.SQCODE_aoi_field).sum()),
    "所在社区与供给表点位所在社区一致": int((aoi.SQCODE == aoi.SQCODE_supply_point).sum()),
}

# ---------- 设施是否在所在社区的 15 分钟步行等时圈内 ----------
iso_by = iso.set_index("SQCODE").geometry
chc["in_own_life_circle"] = [bool(iso_by[s].intersects(g)) if pd.notna(s) and s in iso_by.index else None
                             for s, g in zip(chc.SQCODE, chc.geometry)]
aoi["in_own_life_circle"] = [bool(iso_by[s].intersects(g)) if s in iso_by.index else None
                             for s, g in zip(aoi.SQCODE, aoi.geometry)]
chc["n_life_circles"] = chc.index.map(gpd.sjoin(chc[["geometry"]], iso, predicate="intersects").groupby(level=0).size()
                                      ).fillna(0).astype(int)
aoi["n_life_circles"] = aoi.index.map(gpd.sjoin(aoi[["geometry"]], iso, predicate="intersects").groupby(level=0).size()
                                      ).fillna(0).astype(int)
audit["生活圈"] = {
    "生活圈社区数": int(iso.SQCODE.nunique()),
    "无等时圈、取社区边界的社区": no_iso,
    "社康不在所在社区生活圈内": int((chc.in_own_life_circle == False).sum()),
    "社康不在任何生活圈内": int((chc.n_life_circles == 0).sum()),
    "医院 AOI 不与所在社区生活圈相交": int((aoi.in_own_life_circle == False).sum()),
    "医院 AOI 不与任何生活圈相交": int((aoi.n_life_circles == 0).sum()),
    "社康落入生活圈个数中位数": float(chc.n_life_circles.median()),
    "医院 AOI 相交生活圈个数中位数": float(aoi.n_life_circles.median()),
}

# ---------- 输出 ----------
cols = ["facility_id", "facility_name", "tier", "capacity", "SQCODE", "SQNAME", "OBJECTID", "rule", "area_share",
        "n_communities", "other_communities", "dist_to_community_m", "in_own_life_circle", "n_life_circles"]
allf = pd.concat([chc.drop(columns="geometry"), aoi.drop(columns="geometry")], ignore_index=True)
allf["SQNAME"] = allf.SQCODE.map(name)
allf["OBJECTID"] = allf.SQCODE.map(node).astype("Int64")
allf[cols + ["SQCODE_point", "SQCODE_supply_point", "SQCODE_aoi_field"]].to_csv(OUT / "facility_community.csv", index=False,
                                                                 encoding="utf-8-sig")

cnt = allf.pivot_table(index="SQCODE", columns="tier", values="facility_id", aggfunc="count").reindex(
    columns=["CHC", "SH", "TH"]).fillna(0).astype(int)
cap = allf.pivot_table(index="SQCODE", columns="tier", values="capacity", aggfunc="sum").reindex(
    columns=["CHC", "SH", "TH"]).fillna(0)
tab = pd.DataFrame({"OBJECTID": node, "SQCODE": node.index}).reset_index(drop=True)
tab["SQNAME"] = tab.SQCODE.map(name)
for t in ["CHC", "SH", "TH"]:
    tab[f"located_count_{t}"] = tab.SQCODE.map(cnt[t]).fillna(0).astype(int)
    tab[f"located_capacity_{t}"] = tab.SQCODE.map(cap[t]).fillna(0)
tab.sort_values("OBJECTID").to_csv(OUT / "community_located_facilities.csv", index=False, encoding="utf-8-sig")
audit["社区"] = {f"有{t}的社区": int((tab[f"located_count_{t}"] > 0).sum()) for t in ["CHC", "SH", "TH"]}
audit["社区"]["社区数"] = len(tab)
audit["社区"]["单社区社康数最大"] = int(tab.located_count_CHC.max())
(OUT / "facility_community_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
