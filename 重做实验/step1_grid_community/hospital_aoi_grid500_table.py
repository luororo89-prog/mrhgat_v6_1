"""医院 AOI 与 500 m 网格的归属关系表（方案一：小社区认领 500 m 网格）。

医院 AOI 取 重做实验/input/hospital_aoi_35.shp。
每个 AOI 与 500 m 网格做面相交，得到 AOI 落在各网格的面积和比例，并给出：
  网格在方案一下归属的社区、AOI 自身所在社区（与社区边界重叠面积最大者）、网格的门诊到达人次。
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
AOI = (HERE.parent / "input"
       / "hospital_aoi_35.shp")
NODE = HERE.parent / "input/node_order.csv"
CRS = 4547

aoi = gpd.read_file(AOI).to_crs(CRS).reset_index(drop=True)
aoi["aoi_id"] = [f"AOI_{i + 1:02d}" for i in range(len(aoi))]
aoi["aoi_area_m2"] = aoi.area
g5 = gpd.read_file(GRID500)[["Tid", "geometry"]].to_crs(CRS)
comm = gpd.read_file(COMM)[["SQCODE", "SQNAME", "geometry"]].to_crs(CRS)
comm["geometry"] = comm.geometry.make_valid()
name = comm.set_index("SQCODE").SQNAME
node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("SQCODE").OBJECTID
plan1 = pd.read_csv(RES / "grid500_nest/plan1_grid500_community.csv", dtype={"SQCODE": str}).set_index("Tid")
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")
arrive = od.groupby("end_tid")["count"].sum()

# AOI 所在社区：与社区边界重叠面积最大者
ac = gpd.overlay(aoi[["aoi_id", "geometry"]], comm[["SQCODE", "geometry"]], how="intersection", keep_geom_type=True)
ac["m2"] = ac.area
ac = ac.sort_values(["aoi_id", "m2"], ascending=[True, False])
aoi_comm = ac.drop_duplicates("aoi_id").set_index("aoi_id").SQCODE
aoi_comm_n = ac.groupby("aoi_id").size()

# AOI 与 500 m 网格相交
pc = gpd.overlay(aoi[["aoi_id", "fac_name", "tier", "aoi_area_m2", "geometry"]], g5, how="intersection", keep_geom_type=True)
pc["AOI在网格内面积_m2"] = pc.area
pc = pc[pc["AOI在网格内面积_m2"] > 0]
pc["占AOI面积比例"] = pc["AOI在网格内面积_m2"] / pc.aoi_area_m2
pc = pc.sort_values(["aoi_id", "AOI在网格内面积_m2"], ascending=[True, False])
pc["网格归属社区SQCODE"] = pc.Tid.map(plan1.SQCODE)
pc["网格归属社区"] = pc["网格归属社区SQCODE"].map(name)
pc["网格归属OBJECTID"] = pc["网格归属社区SQCODE"].map(node)
pc["网格归属依据"] = pc.Tid.map(plan1.basis)
pc["AOI所在社区SQCODE"] = pc.aoi_id.map(aoi_comm)
pc["AOI所在社区"] = pc["AOI所在社区SQCODE"].map(name)
pc["网格归属与AOI所在社区一致"] = pc["网格归属社区SQCODE"] == pc["AOI所在社区SQCODE"]
pc["网格门诊到达人次"] = pc.Tid.map(arrive).fillna(0).astype(int)
pieces = pd.DataFrame(pc.drop(columns=["geometry", "aoi_area_m2"])).rename(columns={"Tid": "Tid_500"})

# 每家医院一行
rows = []
for r in aoi.itertuples():
    p = pieces[pieces.aoi_id == r.aoi_id]
    main = p.iloc[0]
    share_by_comm = p.groupby("网格归属社区")["占AOI面积比例"].sum().sort_values(ascending=False)
    same = p.loc[p["网格归属与AOI所在社区一致"], "占AOI面积比例"].sum()
    rows.append({
        "aoi_id": r.aoi_id, "医院": r.fac_name, "tier": r.tier, "beds": r.beds, "统计门诊人次": r.outpat,
        "AOI识别到访": r.id_visits, "AOI面积_m2": round(r.aoi_area_m2),
        "AOI所在社区SQCODE": aoi_comm[r.aoi_id], "AOI所在社区": name[aoi_comm[r.aoi_id]],
        "AOI跨社区数": int(aoi_comm_n[r.aoi_id]),
        "AOI字段sqcode": r.sqcode, "AOI字段sqcode与计算一致": r.sqcode == aoi_comm[r.aoi_id],
        "相交500m网格数": len(p), "主网格Tid_500": int(main.Tid_500), "主网格占AOI比例": round(main["占AOI面积比例"], 3),
        "主网格归属社区": main["网格归属社区"], "主网格归属OBJECTID": int(main["网格归属OBJECTID"]),
        "主网格归属与AOI所在社区一致": bool(main["网格归属与AOI所在社区一致"]),
        "AOI面积归入本社区网格的比例": round(same, 3),
        "AOI覆盖网格的归属社区及比例": "、".join(f"{k} {v:.0%}" for k, v in share_by_comm.items()),
        "相交网格门诊到达人次合计": int(p["网格门诊到达人次"].sum()),
    })
summary = pd.DataFrame(rows)

pieces.to_csv(OUT / "hospital_aoi_grid500_pieces.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUT / "hospital_aoi_grid500_summary.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(OUT / "hospital_aoi_grid500_plan1.xlsx") as w:
    summary.to_excel(w, sheet_name="医院汇总", index=False)
    pieces.to_excel(w, sheet_name="AOI与网格相交", index=False)

aoi_grids = set(pieces.Tid_500)
audit = {
    "医院AOI数": len(aoi), "按等级": aoi.tier.value_counts().to_dict(),
    "AOI跨多个社区": int((summary.AOI跨社区数 > 1).sum()),
    "AOI字段sqcode与计算不一致": summary.loc[~summary["AOI字段sqcode与计算一致"], "医院"].tolist(),
    "AOI跨多个500m网格": int((summary["相交500m网格数"] > 1).sum()),
    "主网格归属与AOI所在社区不一致": summary.loc[~summary["主网格归属与AOI所在社区一致"], "医院"].tolist(),
    "AOI面积全部归入本社区网格的医院": int((summary["AOI面积归入本社区网格的比例"] > 0.999).sum()),
    "AOI面积全部不在本社区网格的医院": summary.loc[summary["AOI面积归入本社区网格的比例"] < 0.001, "医院"].tolist(),
    "AOI相交网格数": len(aoi_grids),
    "AOI相交网格中有门诊到达的": int(len(aoi_grids & set(arrive.index))),
    "门诊到达人次落在AOI相交网格的比例": round(float(arrive[arrive.index.isin(aoi_grids)].sum() / arrive.sum()), 4),
    "AOI识别到访合计": float(aoi.id_visits.sum()), "门诊OD总人次": int(od["count"].sum()),
}
(OUT / "audit_hospital_aoi.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
pd.set_option("display.width", 250)
print(summary[["医院", "tier", "AOI所在社区", "相交500m网格数", "主网格占AOI比例", "主网格归属社区", "AOI面积归入本社区网格的比例",
               "AOI覆盖网格的归属社区及比例", "相交网格门诊到达人次合计"]].to_string(index=False))
