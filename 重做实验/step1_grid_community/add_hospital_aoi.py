"""为供给名单中缺 AOI 的 5 家二级医院补充 AOI，追加写入 supply input/hospital_aoi_35.shp。

AOI 来源（医院点位均落在所取面内）：
  龙岗区第二人民医院、光明新区人民医院、光明新区中心医院取 1input_sources/poi/poi/medical/医院_床位.shp，
  后两家对应 中国科学院大学深圳医院(西院区) 与 (东院区)；
  沙井人民医院、龙岗区第五人民医院取 OSM 现行数据（2026-10-15 获取，results/hospital_aoi_added/osm_261015_two_hospitals.gpkg），
  前者为 深圳市中西医结合医院（OSM 标签 old_name=沙井人民医院）。
字段口径与原 35 行一致：beds 取供给表 capacity，org_code 与 outpat 取外部门诊表（2018 年，按机构名称匹配），
pt_dist 与 area_m2 在 EPSG:4547 下计算，sqcode 为与 AOI 重叠面积最大的社区，OBJECTID 取 node_order。
id_visits 无对应数据，留空。原文件备份在 results/hospital_aoi_added/backup_hospital_aoi_35/。
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "supply input"
OUT = HERE / "results/hospital_aoi_added"
AOI = SRC / "hospital_aoi_35.shp"
BEDS = HERE.parents[1] / "1input_sources/poi/poi/medical/医院_床位.shp"
OSM = OUT / "osm_261015_two_hospitals.gpkg"
EXT = OUT / "hospital_external_validation.csv"
FAC = SRC / "facility_supply_audit_2018_mobile_users.csv"
COMM = SRC / "深圳社区objectid.shp"
COMM_POP = SRC / "深圳各社区人口.shp"
NODE = SRC / "node_order.csv"
CRS = 4547

ADD = {  # 医院: (来源, AOI 名称)
    "深圳市龙岗区第二人民医院": ("医院_床位", "龙岗区第二人民医院"),
    "深圳市光明新区人民医院": ("医院_床位", "中国科学院大学深圳医院(西院区)"),
    "深圳市光明新区中心医院": ("医院_床位", "中国科学院大学深圳医院(东院区)"),
    "深圳市宝安区沙井人民医院": ("OSM_261015", "深圳市中西医结合医院"),
    "深圳市龙岗区第五人民医院": ("OSM_261015", "龙岗区第五人民医院"),
}

aoi = gpd.read_file(AOI)
assert not aoi.fac_name.isin(list(ADD)).any(), "已追加过"
fac = pd.read_csv(FAC).set_index("facility_name")
ext = pd.read_csv(EXT, dtype={"组织机构代码": str}).set_index("机构名称")
beds = gpd.read_file(BEDS)
osm = gpd.read_file(OSM)
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.make_valid()
names = gpd.read_file(COMM_POP).set_index("SQCODE").SQNAME
node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("SQCODE").OBJECTID
assert (aoi.sqcode.map(names) == aoi.sqname).all()

rows = []
for fac_name, (src, aoi_name) in ADD.items():
    if src == "医院_床位":
        g = beds.loc[beds.Name == aoi_name]
        osm_id = None
    else:
        g = osm.loc[osm.name == aoi_name]
        osm_id = g.osm_id.iloc[0].lstrip("wr")
    assert len(g) == 1, fac_name
    rows.append({"fac_name": fac_name, "aoi_src": src, "aoi_name": aoi_name, "osm_id": osm_id,
                 "tier": fac.at[fac_name, "tier"], "org_code": ext.at[fac_name, "组织机构代码"],
                 "beds": float(fac.at[fac_name, "capacity"]), "outpat": int(ext.at[fac_name, "门诊人次"]),
                 "id_visits": float("nan"), "geometry": g.to_crs(4326).geometry.iloc[0]})
new = gpd.GeoDataFrame(rows, crs=4326)
m = new.to_crs(CRS)
pt = gpd.GeoSeries(gpd.points_from_xy(fac.loc[new.fac_name, "longitude"], fac.loc[new.fac_name, "latitude"]),
                   crs=4326).to_crs(CRS)
new["pt_dist"] = m.distance(pt.set_axis(m.index)).round().astype(int)
new["area_m2"] = m.area.round()
ov = gpd.overlay(m[["fac_name", "geometry"]], comm, how="intersection", keep_geom_type=True)
ov["m2"] = ov.area
new["sqcode"] = new.fac_name.map(ov.sort_values("m2", ascending=False).drop_duplicates("fac_name")
                                 .set_index("fac_name").SQCODE)
new["OBJECTID"] = new.sqcode.map(node).astype(int)
new["sqname"] = new.sqcode.map(names)

out = pd.concat([aoi, new[aoi.columns]], ignore_index=True)
out.to_file(AOI, driver="ESRI Shapefile", encoding="utf-8")
new.drop(columns="geometry").to_csv(OUT / "added_rows.csv", index=False, encoding="utf-8-sig")
print(new.drop(columns="geometry").to_string())
print(len(out))
