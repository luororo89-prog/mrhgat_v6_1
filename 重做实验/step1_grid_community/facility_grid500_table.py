"""医疗设施与 500 m 网格的归属关系表（方案一：小社区认领 500 m 网格）。

设施取当前供给计算的设施表 facility_supply_audit_2018_mobile_users.csv（社康 CHC、二级 SH、三级 TH，经纬度为 WGS84）。
每个设施按点位找到所在 500 m 网格与 250 m 网格，并给出：
  设施点位所在社区（社区边界）、500 m 网格在方案一下归属的社区、250 m 网格按 maup 归属的社区，
  以及该 500 m 网格作为门诊到达网格的人次。
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "facility_grid"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "input"
GRID500 = SRC / "SZ_grid_500m.shp"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM = SRC / "深圳社区objectid.shp"
FAC = (HERE.parent / "input"
       / "facility_supply_audit_2018_mobile_users.csv")
NODE = HERE.parent / "input/node_order.csv"

fac = pd.read_csv(FAC)[["facility_id", "facility_name", "tier", "source_grade", "capacity", "longitude", "latitude"]]
pts = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.longitude, fac.latitude), crs=4326)
g5 = gpd.read_file(GRID500)[["Tid", "geometry"]].rename(columns={"Tid": "Tid_500"})
g2 = gpd.read_file(GRID250)[["Tid", "geometry"]].rename(columns={"Tid": "Tid_250"})
comm = gpd.read_file(COMM)[["SQCODE", "SQNAME", "geometry"]].to_crs(4326)
comm["geometry"] = comm.geometry.make_valid()
node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
name = comm.set_index("SQCODE").SQNAME

t = gpd.sjoin(pts, g5, how="left", predicate="within").drop(columns="index_right")
t = gpd.sjoin(t, g2, how="left", predicate="within").drop(columns="index_right")
t = gpd.sjoin(t, comm.rename(columns={"SQCODE": "设施所在社区SQCODE", "SQNAME": "设施所在社区"}), how="left",
              predicate="within").drop(columns="index_right")
t = t.drop_duplicates("facility_id")

plan1 = pd.read_csv(RES / "grid500_nest/plan1_grid500_community.csv", dtype={"SQCODE": str}).set_index("Tid")
own250 = pd.read_csv(RES / "grid250_community_1to1.csv", dtype={"SQCODE": str}).set_index("Tid").SQCODE
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")
arrive = od.groupby("end_tid")["count"].sum()

t["500m网格归属社区SQCODE"] = t.Tid_500.map(plan1.SQCODE)
t["500m网格归属社区"] = t["500m网格归属社区SQCODE"].map(name)
t["500m网格归属OBJECTID"] = t["500m网格归属社区SQCODE"].map(node.set_index("SQCODE").OBJECTID).astype("Int64")
t["500m网格归属依据"] = t.Tid_500.map(plan1.basis)
t["250m网格归属社区SQCODE"] = t.Tid_250.map(own250)
t["250m网格归属社区"] = t["250m网格归属社区SQCODE"].map(name)
t["500m归属与设施所在社区一致"] = t["500m网格归属社区SQCODE"] == t["设施所在社区SQCODE"]
t["250m归属与设施所在社区一致"] = t["250m网格归属社区SQCODE"] == t["设施所在社区SQCODE"]
t["该500m网格门诊到达人次"] = t.Tid_500.map(arrive).fillna(0).astype(int)
t["该500m网格设施数"] = t.Tid_500.map(t.groupby("Tid_500").size()).astype("Int64")
t["Tid_500"] = t.Tid_500.astype("Int64")
t["Tid_250"] = t.Tid_250.astype("Int64")
cols = ["facility_id", "facility_name", "tier", "source_grade", "capacity", "longitude", "latitude",
        "Tid_500", "500m网格归属社区SQCODE", "500m网格归属社区", "500m网格归属OBJECTID", "500m网格归属依据",
        "设施所在社区SQCODE", "设施所在社区", "500m归属与设施所在社区一致",
        "Tid_250", "250m网格归属社区SQCODE", "250m网格归属社区", "250m归属与设施所在社区一致",
        "该500m网格门诊到达人次", "该500m网格设施数"]
t = pd.DataFrame(t[cols])
t.to_csv(OUT / "facility_grid500_community_plan1.csv", index=False, encoding="utf-8-sig")
t.to_excel(OUT / "facility_grid500_community_plan1.xlsx", index=False)

by = t.groupby("tier")
audit = {
    "设施数": by.size().to_dict(),
    "不在500m网格内": int(t.Tid_500.isna().sum()),
    "不在任何社区边界内": int(t["设施所在社区SQCODE"].isna().sum()),
    "500m归属与所在社区不一致": by["500m归属与设施所在社区一致"].apply(lambda x: int((~x).sum())).to_dict(),
    "250m归属与所在社区不一致": by["250m归属与设施所在社区一致"].apply(lambda x: int((~x).sum())).to_dict(),
    "设施所在500m网格无门诊到达": by["该500m网格门诊到达人次"].apply(lambda x: int((x == 0).sum())).to_dict(),
    "含设施的500m网格数": int(t.Tid_500.nunique()),
    "门诊到达网格中含设施的": int(len(set(arrive.index) & set(t.Tid_500.dropna().astype(int)))),
    "门诊到达网格总数": int(len(arrive)),
    "门诊到达人次落在含设施网格的比例": round(float(arrive[arrive.index.isin(t.Tid_500.dropna().astype(int))].sum() / arrive.sum()), 4),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
