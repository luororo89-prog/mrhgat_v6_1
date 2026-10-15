"""500 m 网格—设施—社区三者关系表（方案一：小社区认领 500 m 网格）。

网格归属设施：社康按点位落入的 500 m 网格；医院有 AOI 的（hospital_aoi_35.shp）按 AOI 与网格相交，
  无 AOI 的按点位。设施取 facility_supply_audit_2018_mobile_users.csv（CHC、SH、TH）。
设施归属社区：点位所在社区；有 AOI 的医院取与社区边界重叠面积最大的社区。
网格归属社区：results/grid500_nest/plan1_grid500_community.csv。
对每个含设施的 500 m 网格，比较其设施所在社区与方案一归属社区，并给出门诊到达人次。
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
COMM = SRC / "深圳社区objectid.shp"
VAL = HERE.parent / "input"
FAC = VAL / "facility_supply_audit_2018_mobile_users.csv"
AOI = VAL / "hospital_aoi_35.shp"
NODE = HERE.parent / "input/node_order.csv"
CRS = 4547

g5 = gpd.read_file(GRID500)[["Tid", "geometry"]].to_crs(CRS)
comm = gpd.read_file(COMM)[["SQCODE", "SQNAME", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.make_valid()
name = comm.set_index("SQCODE").SQNAME
node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("SQCODE").OBJECTID
plan1 = pd.read_csv(RES / "grid500_nest/plan1_grid500_community.csv", dtype={"SQCODE": str}).set_index("Tid")
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")
arrive = od.groupby("end_tid")["count"].sum()

fac = pd.read_csv(FAC)[["facility_id", "facility_name", "tier", "longitude", "latitude"]]
pts = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.longitude, fac.latitude), crs=4326).to_crs(CRS)
aoi = gpd.read_file(AOI).to_crs(CRS)
aoi = aoi.merge(fac[["facility_id", "facility_name"]], left_on="fac_name", right_on="facility_name")[
    ["facility_id", "geometry"]]
has_aoi = set(aoi.facility_id)

# 设施归属社区
pc = gpd.sjoin(pts, comm[["SQCODE", "geometry"]], how="left", predicate="within").drop_duplicates("facility_id")
fac_comm = pc.set_index("facility_id").SQCODE
ac = gpd.overlay(aoi, comm[["SQCODE", "geometry"]], how="intersection", keep_geom_type=True)
ac["m2"] = ac.area
fac_comm.update(ac.sort_values("m2", ascending=False).drop_duplicates("facility_id").set_index("facility_id").SQCODE)

# 网格归属设施
gp = gpd.sjoin(pts[~pts.facility_id.isin(has_aoi)], g5, how="inner", predicate="within")[["facility_id", "Tid"]]
gp["依据"] = "点位"
gp["占设施面积比例"] = 1.0
ga = gpd.overlay(aoi.assign(a=aoi.area), g5, how="intersection", keep_geom_type=True)
ga["占设施面积比例"] = ga.area / ga.a
ga = pd.DataFrame(ga[ga["占设施面积比例"] > 0][["facility_id", "Tid", "占设施面积比例"]])
ga["依据"] = "AOI"
link = pd.concat([gp, ga], ignore_index=True).merge(fac[["facility_id", "facility_name", "tier"]], on="facility_id")
link["设施所在社区SQCODE"] = link.facility_id.map(fac_comm)
link["网格归属社区SQCODE"] = link.Tid.map(plan1.SQCODE)
link["一致"] = link["设施所在社区SQCODE"] == link["网格归属社区SQCODE"]

# 每个含设施网格一行
rows = []
for tid, p in link.groupby("Tid"):
    g = plan1.SQCODE.get(tid)
    fc = sorted(set(p["设施所在社区SQCODE"].dropna()))
    if fc == [g]:
        rel = "设施全在网格归属社区"
    elif len(fc) == 1:
        rel = "设施全在另一社区"
    elif g in fc:
        rel = "设施分属多社区_含网格归属社区"
    else:
        rel = "设施分属多社区_不含网格归属社区"
    rows.append({
        "Tid_500": tid, "网格归属社区SQCODE": g, "网格归属社区": name.get(g), "网格归属OBJECTID": node.get(g),
        "网格归属依据": plan1.basis.get(tid),
        "设施数": len(p), "TH": int((p.tier == "TH").sum()), "SH": int((p.tier == "SH").sum()),
        "CHC": int((p.tier == "CHC").sum()),
        "设施": "、".join(p.facility_name), "设施所在社区SQCODE": "、".join(fc),
        "设施所在社区": "、".join(name.get(c, "") for c in fc), "关系": rel,
        "门诊到达人次": int(arrive.get(tid, 0)),
    })
grid = pd.DataFrame(rows)
link["网格归属社区"] = link["网格归属社区SQCODE"].map(name)
link["设施所在社区"] = link["设施所在社区SQCODE"].map(name)
link["门诊到达人次"] = link.Tid.map(arrive).fillna(0).astype(int)
link = link.rename(columns={"Tid": "Tid_500"})[["facility_id", "facility_name", "tier", "依据", "Tid_500", "占设施面积比例",
    "设施所在社区SQCODE", "设施所在社区", "网格归属社区SQCODE", "网格归属社区", "一致", "门诊到达人次"]]
link.to_csv(OUT / "grid500_facility_link_plan1.csv", index=False, encoding="utf-8-sig")
grid.to_csv(OUT / "grid500_facility_relation_plan1.csv", index=False, encoding="utf-8-sig")

tot = arrive.sum()
by = grid.groupby("关系")
audit = {
    "设施数": fac.tier.value_counts().to_dict(), "有AOI的医院": len(has_aoi),
    "未落入500m网格的设施": sorted(set(fac.facility_id) - set(link.facility_id)),
    "未落入社区边界的设施": fac_comm[fac_comm.isna()].index.tolist(),
    "含设施的500m网格数": len(grid),
    "网格关系": {k: {"网格数": int(len(v)), "门诊到达人次": int(v["门诊到达人次"].sum()),
                 "占全部门诊到达": round(float(v["门诊到达人次"].sum() / tot), 4)} for k, v in by},
    "不含设施的门诊到达网格": int(len(set(arrive.index) - set(grid.Tid_500))),
    "不含设施网格的门诊到达占比": round(float(arrive[~arrive.index.isin(grid.Tid_500)].sum() / tot), 4),
    "跨多个网格的设施": int((link.groupby("facility_id").size() > 1).sum()),
    "设施网格与设施所在社区不一致的设施": int(link.groupby("facility_id")["一致"].all().eq(False).sum()),
}
(OUT / "audit_grid500_facility_relation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
