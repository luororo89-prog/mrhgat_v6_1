"""医院 AOI 缓冲区大小比较：缓冲区内的 500 m 网格归医院所在社区时，各缓冲距离的收益与代价。

网格进入缓冲区：网格与 AOI 相交，或网格中心到 AOI 的距离不超过缓冲距离（与 run_validation 的 r200/r300/r500 同口径）。
医院 AOI 取 hospital_aoi_35.shp，无 AOI 的 5 家医院用点位。医院所在社区取 grid500_facility_link_plan1.csv。
网格原归属取方案一 plan1_grid500_community.csv；小社区为面积小于 0.3 km² 的社区。
门诊到达人次取清洗前的 outpatient_grid_OD_home_tid.csv。
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "facility_grid"
SRC = HERE.parent / "input"
GRID500 = SRC / "SZ_grid_500m.shp"
VAL = HERE.parent / "input"
FAC = VAL / "facility_supply_audit_2018_mobile_users.csv"
AOI = VAL / "hospital_aoi_35.shp"
CRS = 4547
BUFFERS = [0, 100, 200, 300, 400, 500, 600, 800, 1000]

g5 = gpd.read_file(GRID500)[["Tid", "x", "y", "geometry"]].to_crs(CRS)
cen = gpd.GeoDataFrame(g5[["Tid"]], geometry=gpd.points_from_xy(g5.x, g5.y), crs=4326).to_crs(CRS)
plan1 = pd.read_csv(RES / "grid500_nest/plan1_grid500_community.csv", dtype={"SQCODE": str}).set_index("Tid")
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})
small = set(cs.loc[cs.comm_area_m2 < 0.3e6, "SQCODE"])
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")
arrive = od.groupby("end_tid")["count"].sum()
link = pd.read_csv(OUT / "grid500_facility_link_plan1.csv", dtype={"设施所在社区SQCODE": str})
chc_grid = link[link.tier == "CHC"][["Tid_500", "设施所在社区SQCODE"]]

fac = pd.read_csv(FAC)
hosp = fac[fac.tier != "CHC"][["facility_name", "tier", "longitude", "latitude"]]
hosp_comm = link[link.tier != "CHC"].drop_duplicates("facility_name").set_index("facility_name")["设施所在社区SQCODE"]
aoi = gpd.read_file(AOI)[["fac_name", "geometry"]].rename(columns={"fac_name": "facility_name"}).to_crs(CRS)
pt = hosp[~hosp.facility_name.isin(aoi.facility_name)]
pt = gpd.GeoDataFrame(pt[["facility_name"]], geometry=gpd.points_from_xy(pt.longitude, pt.latitude), crs=4326).to_crs(CRS)
shape = pd.concat([aoi, pt], ignore_index=True)
shape["comm"] = shape.facility_name.map(hosp_comm)
shape["tier"] = shape.facility_name.map(hosp.set_index("facility_name").tier)

inter = gpd.sjoin(shape, g5[["Tid", "geometry"]], how="inner", predicate="intersects")[["facility_name", "Tid"]]
base_cnt = plan1.SQCODE.value_counts()
rows, detail = [], []
for b in BUFFERS:
    near = gpd.sjoin(cen, shape.assign(geometry=shape.buffer(b)) if b else shape, how="inner",
                     predicate="within" if b else "within")[["facility_name", "Tid"]]
    m = pd.concat([inter, near]).drop_duplicates()
    m["comm"] = m.facility_name.map(hosp_comm)
    m["plan1"] = m.Tid.map(plan1.SQCODE)
    m["basis"] = m.Tid.map(plan1.basis)
    m["arr"] = m.Tid.map(arrive).fillna(0)
    multi = m.groupby("Tid").comm.nunique()
    conflict = set(multi[multi > 1].index)
    # 缓冲区网格改归医院社区（多家医院冲突的网格不改）
    new = plan1.SQCODE.copy()
    one = m[~m.Tid.isin(conflict)].drop_duplicates("Tid").set_index("Tid").comm
    new.update(one)
    moved = new[new != plan1.SQCODE]
    cnt = new.value_counts()
    lost = [c for c in base_cnt.index if cnt.get(c, 0) == 0]
    per = m.groupby("facility_name").agg(n=("Tid", "size"), arr=("arr", "sum"))
    other_chc = chc_grid[chc_grid.Tid_500.isin(one.index)]
    other_chc = other_chc[other_chc["设施所在社区SQCODE"] != other_chc.Tid_500.map(one)]
    rows.append({
        "缓冲距离_m": b, "每家医院网格数_平均": round(per.n.mean(), 1), "每家医院网格数_最多": int(per.n.max()),
        "缓冲区网格": int(m.Tid.nunique()), "缓冲区到达人次": int(arrive[arrive.index.isin(m.Tid)].sum()),
        "到达人次为0的医院": int((per.arr == 0).sum() + (len(shape) - len(per))),
        "多家不同社区医院共享的网格": len(conflict),
        "改变归属的网格": int(len(moved)), "改变归属的到达人次": int(arrive[arrive.index.isin(moved.index)].sum()),
        "其中原属小社区的网格": int(plan1.SQCODE[moved.index].isin(small).sum()),
        "其中锚定网格": int(plan1.basis[moved.index].str.startswith("锚定").sum()),
        "因此失去全部网格的社区": len(lost), "其中小社区": sum(c in small for c in lost),
        "并入医院社区的其他社区社康": int(len(other_chc)),
        "失去全部网格的社区名单": "、".join(cs.set_index("SQCODE").SQNAME[c] + ("(小)" if c in small else "") for c in lost),
    })
    per = per.reindex(shape.facility_name).fillna(0)
    detail.append(per.assign(缓冲距离_m=b).reset_index())
res = pd.DataFrame(rows)
res.to_csv(OUT / "aoi_buffer_sensitivity.csv", index=False, encoding="utf-8-sig")
pd.concat(detail).to_csv(OUT / "aoi_buffer_sensitivity_by_hospital.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(res.to_string(index=False))
