"""设施所在 250 m 网格、设施归属社区与 250 m 网格社区归属修订，供门诊 OD 清洗、出发与到达分配共同读取。

顺序：先定 250 m 网格与社区的归属（step1 landuse_demand 的 grid250_demand_allocation，出发端 250 m 挂接据此），
再定设施与社区的归属，由此确定设施所在 250 m 网格；与 250 m 网格社区归属冲突时修订该网格的社区归属。
1. 社康点位取 博后三胎/outputs/chc_point_merge_20261018/2018年深圳CHC_点位合并.xlsx 的“点位”表（912 个社康的 WGS84 坐标，原表行号 对应 facility_id 的 CHC_<行号>），
   社康归属新点位所在社区（不在任何社区内的不归属）；医院沿用供给评价 facility_community.csv 的 AOI 主社区。
   到达分配后点位核查（check_zero_arrival_points.py）判定不准确的社康，在 chc_point_updates.csv 中更新点位，覆盖新点位表坐标。
2. 设施所在 250 m 网格：社康取新点位所在网格，医院取 AOI 与 250 m 网格相交的网格，按相交面积拆分。
3. 设施归属社区不在所在网格的社区归属内、但与该网格相交时，修订该网格的社区归属：在原归属社区与设施归属社区之间，
   按各社区片段内建设用地面积（ESA WorldCover 2020 Built-up，同 landuse_demand）占比分配网格人口；
   设施归属社区片段无建设用地时按片段面积占比。网格人口总数不变，出发人次随网格人口在社区间重新分配，OD 人次不变。
4. 设施归属社区与所在网格不相交时（社康点位在命名社区以外的网格），社康移到中心点距点位最近、社区归属含其归属社区的 250 m 网格
   （记入 chc_relocated.csv，是否影响供给评价、社区 OD 与拓扑由 check_relocated_facilities.py 判定）；
   医院的这类 AOI 网格去掉，面积占比在其余网格中重新归一。
5. 设施吸引力 = resource_weight × 面积占比 × 层级倍数；层级倍数 = 该层级医院 2018 年平均门诊量（hospital_aoi_35 的 outpat）
   ÷ 社康平均年诊疗人次，社康平均年诊疗人次 = 2018 年全市总诊疗人次（2017、2019 年中值）× 社康诊疗量占比 33.4% ÷ 社康数。
结果 results/facility_grid250/：facility_grid250_attraction.csv、grid250_demand_allocation_revised.csv（修订后的 250 m 网格社区归属，
列同 grid250_demand_allocation）、community_demand_population_revised.csv、grid250_revised_cells.csv、chc_relocated.csv、
hospital_aoi_cells_dropped.csv、facility_location_final.csv（社康最终坐标，供重算供给评价）、audit.json。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/facility_grid250"
OUT.mkdir(parents=True, exist_ok=True)
SRC = HERE.parent / "supply input"
STEP1 = HERE.parent / "step1_grid_community/results"
STEP2 = HERE.parent / "step2_supply/results"
DEMAND250 = STEP1 / "landuse_demand/grid250_demand_allocation.csv"
DEMAND = STEP1 / "landuse_demand/community_demand_population.csv"
WC = STEP1 / "landuse_demand/worldcover_2020_shenzhen.tif"
FAC_COMM = STEP2 / "facility_community.csv"
FAC_AUDIT = STEP2 / "supply_indicators/facility_supply_audit_2018_mobile_users.csv"
AOI = SRC / "hospital_aoi_35.shp"
COMM = SRC / "深圳社区objectid.shp"
CHC_XY = Path("/Users/luoxuan/Documents/博后三胎/outputs/chc_point_merge_20261018/2018年深圳CHC_点位合并.xlsx")
UPDATE = HERE / "chc_point_updates.csv"
CRS = 4547
BUILT = 50  # WorldCover Built-up
TOTAL_VISITS_2018 = (8705.44 + 10811.72) / 2 * 1e4  # 深圳统计公报 2017、2019 年总诊疗人次的中值
CHC_VISIT_SHARE = 0.334  # 社康诊疗量占全市诊疗量比例（深圳市卫健委）

node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
sqname = node.set_index("SQCODE").SQNAME
d250 = pd.read_csv(DEMAND250, encoding="utf-8-sig", dtype={"SQCODE": str})
assert np.allclose(d250["联通人_分配"], d250["联通人"] * d250.weight)
base = set(zip(d250.loc[d250.weight > 0, "Tid"], d250.loc[d250.weight > 0, "SQCODE"]))

grid500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
c250 = g250.assign(geometry=g250.geometry.centroid)
nest = gpd.sjoin(c250, grid500, predicate="within", lsuffix="250", rsuffix="500")[["Tid_250", "Tid_500"]]
assert nest.Tid_250.is_unique
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.make_valid()

# ---------- 设施点位、AOI 与 250 m 网格 ----------
fac = pd.read_csv(FAC_COMM, dtype={"SQCODE": str})[["facility_id", "facility_name", "tier", "SQCODE"]].merge(
    pd.read_csv(FAC_AUDIT)[["facility_id", "resource_weight", "longitude", "latitude"]], on="facility_id")
chc_xy = pd.read_excel(CHC_XY, sheet_name="点位")[["原表行号", "经度", "纬度"]].rename(
    columns={"原表行号": "row", "经度": "longitude", "纬度": "latitude"})
chc_xy = chc_xy.assign(facility_id="CHC_" + chc_xy.row.astype(int).astype(str)).set_index("facility_id")
is_chc = fac.tier == "CHC"
assert set(fac.loc[is_chc, "facility_id"]) == set(chc_xy.index), "新点位表须覆盖全部社康"
upd = pd.read_csv(UPDATE, encoding="utf-8-sig").set_index("facility_id")
assert set(upd.index) <= set(chc_xy.index)
for c in ["longitude", "latitude"]:
    fac.loc[is_chc, c] = fac.loc[is_chc, "facility_id"].map(chc_xy[c]).astype(float)
    fac.loc[fac.facility_id.isin(upd.index), c] = fac.facility_id.map(upd[c])
chc = fac[is_chc].copy()
chc = gpd.GeoDataFrame(chc, geometry=gpd.points_from_xy(chc.longitude, chc.latitude), crs=4326).to_crs(CRS)
# 社康归属新点位所在社区
chc["SQCODE"] = gpd.sjoin(chc[["geometry"]], comm, predicate="within").groupby(level=0).SQCODE.first()
no_comm = sorted(chc.loc[chc.SQCODE.isna(), "facility_name"])
chc = chc[chc.SQCODE.notna()]
fc = gpd.sjoin(chc, g250.rename(columns={"Tid": "Tid_250"}), predicate="within")
fc = fc[["facility_id", "facility_name", "tier", "SQCODE", "Tid_250", "resource_weight", "geometry"]].assign(
    footprint_share=1.0)
no_cell = sorted(set(chc.facility_name) - set(fc.facility_name))
aoi = gpd.read_file(AOI).to_crs(CRS)[["fac_name", "geometry"]]
aoi["geometry"] = aoi.geometry.make_valid()
hosp = fac[fac.tier != "CHC"].set_index("facility_name")
fh = gpd.overlay(aoi, g250.rename(columns={"Tid": "Tid_250"}), how="intersection", keep_geom_type=True)
fh["m2"] = fh.area
fh = fh[fh.m2 > 0].copy()
for c in ["facility_id", "tier", "SQCODE", "resource_weight"]:
    fh[c] = fh.fac_name.map(hosp[c])
fh["facility_name"] = fh.fac_name

# 设施所在网格与社区的片段
cells = g250[g250.Tid.isin(set(fc.Tid_250) | set(fh.Tid_250))].rename(columns={"Tid": "Tid_250"})
pieces = gpd.overlay(cells, comm, how="intersection", keep_geom_type=True)
pieces["piece_m2"] = pieces.area
pieces = pieces[pieces.piece_m2 > 0].reset_index(drop=True)
touch = set(zip(pieces.Tid_250, pieces.SQCODE))
fc["SQCODE_located"] = fc.SQCODE

# ---------- 冲突：设施归属社区不在所在网格的社区归属内 ----------
fc["Tid_250_point"] = fc.Tid_250
fc["conflict"] = [(t_, s) not in base for t_, s in zip(fc.Tid_250, fc.SQCODE)]
fc["touch"] = [(t_, s) in touch for t_, s in zip(fc.Tid_250, fc.SQCODE)]
fh["conflict"] = [(t_, s) not in base for t_, s in zip(fh.Tid_250, fh.SQCODE)]
fh["touch"] = [(t_, s) in touch for t_, s in zip(fh.Tid_250, fh.SQCODE)]
revise = pd.concat([fc.loc[fc.conflict & fc.touch, ["Tid_250", "SQCODE", "facility_name"]],
                    fh.loc[fh.conflict & fh.touch, ["Tid_250", "SQCODE", "facility_name"]]], ignore_index=True)

# 设施归属社区与所在网格不相交的社康：移到最近的、社区归属含其归属社区的 250 m 网格
cen = c250.set_index("Tid").geometry
moved = []
for i, r in fc[fc.conflict & ~fc.touch].iterrows():
    cand = cen.reindex(sorted({t_ for t_, s in base if s == r.SQCODE}))
    dist = cand.distance(r.geometry)
    fc.at[i, "Tid_250"] = int(dist.idxmin())
    moved.append({"facility_id": r.facility_id, "facility_name": r.facility_name, "SQCODE": r.SQCODE,
                  "SQNAME": sqname[r.SQCODE], "Tid_250_point": r.Tid_250_point,
                  "点位网格社区归属": "、".join(sqname.reindex(d250.loc[(d250.Tid == r.Tid_250_point) & (d250.weight > 0),
                                                                 "SQCODE"]).fillna("?")),
                  "Tid_250": int(dist.idxmin()), "move_m": round(float(dist.min()), 1),
                  "x_point": r.geometry.x, "y_point": r.geometry.y})
moved = pd.DataFrame(moved, columns=["facility_id", "facility_name", "SQCODE", "SQNAME", "Tid_250_point", "点位网格社区归属",
                                     "Tid_250", "move_m", "x_point", "y_point"])
fc["move_m"] = fc.facility_id.map(moved.set_index("facility_id").move_m).fillna(0.0)
moved.sort_values("move_m", ascending=False).to_csv(OUT / "chc_relocated.csv", index=False, encoding="utf-8-sig")

# 社康最终坐标：新点位，点位核查更新的点位，移到的 250 m 网格中心点
cen84 = cen.to_crs(4326)
loc = fac.loc[is_chc, ["facility_id", "facility_name", "longitude", "latitude"]]
loc = loc.assign(依据=np.where(loc.facility_id.isin(upd.index), "chc_point_updates.csv 点位核查更新", CHC_XY.name))
loc = pd.concat([loc[~loc.facility_id.isin(moved.facility_id)], pd.DataFrame({
    "facility_id": moved.facility_id, "facility_name": moved.facility_name,
    "longitude": cen84.reindex(moved.Tid_250).x.to_numpy(), "latitude": cen84.reindex(moved.Tid_250).y.to_numpy(),
    "依据": "归属社区与点位所在 250 m 网格不相交，移到最近的归属含该社区的 250 m 网格中心点"})], ignore_index=True)
loc.to_csv(OUT / "facility_location_final.csv", index=False, encoding="utf-8-sig")

# 医院与所在网格不相交的 AOI 网格去掉
fh["footprint_share_all"] = fh.m2 / fh.groupby("fac_name").m2.transform("sum")
drop = fh[fh.conflict & ~fh.touch]
drop.assign(SQNAME=drop.SQCODE.map(sqname))[["facility_name", "SQNAME", "Tid_250", "m2", "footprint_share_all"]].to_csv(
    OUT / "hospital_aoi_cells_dropped.csv", index=False, encoding="utf-8-sig")
fh = fh[~(fh.conflict & ~fh.touch)].copy()
fh["footprint_share"] = fh.m2 / fh.groupby("fac_name").m2.transform("sum")

# ---------- 修订 250 m 网格社区归属：原归属社区与设施归属社区之间按建设用地面积占比 ----------
rp = pieces[pieces.Tid_250.isin(revise.Tid_250)].copy()
with rasterio.open(WC) as src:
    frac = []
    for geom in rp.to_crs(4326).geometry:
        win = from_bounds(*geom.bounds, src.transform).round_offsets(op="floor").round_lengths(op="ceil")
        win = Window(win.col_off, win.row_off, win.width + 1, win.height + 1)
        lc = src.read(1, window=win)
        inside = ~geometry_mask([geom], out_shape=lc.shape, transform=src.window_transform(win))
        frac.append(float((lc[inside] == BUILT).mean()) if inside.any() else 0.0)
rp["builtup_m2"] = np.array(frac) * rp.piece_m2
keep = set(base) | set(zip(revise.Tid_250, revise.SQCODE))
rp = rp[[(t_, s) in keep for t_, s in zip(rp.Tid_250, rp.SQCODE)]]
rp = rp.groupby(["Tid_250", "SQCODE"], as_index=False)[["piece_m2", "builtup_m2"]].sum()
need = set(zip(revise.Tid_250, revise.SQCODE))
no_built = {t_ for t_, s in need if rp.loc[(rp.Tid_250 == t_) & (rp.SQCODE == s), "builtup_m2"].sum() <= 0}
rp["by_area"] = rp.Tid_250.isin(no_built)
rp["v"] = np.where(rp.by_area, rp.piece_m2, rp.builtup_m2)
rp["weight"] = rp.v / rp.groupby("Tid_250").v.transform("sum")
rp = rp[rp.weight > 0]
cell_pop = d250.drop_duplicates("Tid").set_index("Tid")[["联通人", "七普加"]]
new = rp[["Tid_250", "SQCODE", "weight"]].rename(columns={"Tid_250": "Tid"})
new["rule"] = np.where(new.Tid.isin(no_built), "设施归属修订_面积比例", "设施归属修订_建设用地比例")
new = new.join(cell_pop, on="Tid")
new["联通人_分配"] = new["联通人"] * new.weight
new["七普加_分配"] = new["七普加"] * new.weight
d250r = pd.concat([d250[~d250.Tid.isin(new.Tid)], new[d250.columns]], ignore_index=True).sort_values(["Tid", "SQCODE"])
assert np.allclose(d250r.groupby("Tid").weight.sum(), 1)
assert np.isclose(d250r["联通人_分配"].sum(), d250["联通人_分配"].sum())
d250r.to_csv(OUT / "grid250_demand_allocation_revised.csv", index=False, encoding="utf-8-sig")
base_r = set(zip(d250r.loc[d250r.weight > 0, "Tid"], d250r.loc[d250r.weight > 0, "SQCODE"]))

before = d250[d250.Tid.isin(new.Tid) & (d250.weight > 0)].groupby("Tid").apply(
    lambda g: "、".join(f"{sqname[s]}{w:.2f}" for s, w in zip(g.SQCODE, g.weight)), include_groups=False)
after = new.groupby("Tid").apply(lambda g: "、".join(f"{sqname[s]}{w:.2f}" for s, w in zip(g.SQCODE, g.weight)),
                                 include_groups=False)
rc = pd.DataFrame({"修订前": before, "修订后": after})
rc["设施"] = rc.index.map(revise.groupby("Tid_250").facility_name.apply(lambda s: "、".join(sorted(set(s)))))
rc["联通人"] = rc.index.map(cell_pop["联通人"])
moved_pop = (d250r.set_index(["Tid", "SQCODE"])["联通人_分配"].reindex(
    pd.MultiIndex.from_frame(new[["Tid", "SQCODE"]])).fillna(0) - d250.set_index(["Tid", "SQCODE"])["联通人_分配"]
             .reindex(pd.MultiIndex.from_frame(new[["Tid", "SQCODE"]])).fillna(0))
rc["转给设施归属社区的联通人"] = rc.index.map(moved_pop.clip(lower=0).groupby(level=0).sum())
rc.rename_axis("Tid_250").reset_index().to_csv(OUT / "grid250_revised_cells.csv", index=False, encoding="utf-8-sig")

dem = pd.read_csv(DEMAND, encoding="utf-8-sig", dtype={"SQCODE": str})[["OBJECTID", "SQCODE", "SQNAME", "需求人口_联通",
                                                                         "需求人口_七普"]]
dem = dem.rename(columns={"需求人口_联通": "需求人口_联通_修订前", "需求人口_七普": "需求人口_七普_修订前"})
dem["需求人口_联通"] = dem.SQCODE.map(d250r.groupby("SQCODE")["联通人_分配"].sum()).fillna(0)
dem["需求人口_七普"] = dem.SQCODE.map(d250r.groupby("SQCODE")["七普加_分配"].sum()).fillna(0)
dem.to_csv(OUT / "community_demand_population_revised.csv", index=False, encoding="utf-8-sig")

# ---------- 吸引力 ----------
fh["SQCODE_located"] = fh.SQCODE
fh["Tid_250_point"] = pd.NA
fh["move_m"] = 0.0
cols = ["facility_id", "facility_name", "tier", "SQCODE", "SQCODE_located", "Tid_250", "Tid_250_point", "move_m",
        "resource_weight", "footprint_share"]
fp = pd.concat([fc[cols], fh[cols]], ignore_index=True)
aoi_out = gpd.read_file(AOI)[["tier", "outpat"]]
chc_visits = TOTAL_VISITS_2018 * CHC_VISIT_SHARE / int((fac.tier == "CHC").sum())
TIER_MULT = (aoi_out.groupby("tier").outpat.mean() / chc_visits).to_dict()
TIER_MULT["CHC"] = 1.0
fp["tier_mult"] = fp.tier.map(TIER_MULT)
fp["attraction"] = fp.resource_weight * fp.footprint_share * fp.tier_mult
fp = fp.merge(nest, on="Tid_250", how="left")
assert all((t_, s) in base_r for t_, s in zip(fp.Tid_250, fp.SQCODE)), "设施归属社区须在所在 250 m 网格的社区归属内"
fp.to_csv(OUT / "facility_grid250_attraction.csv", index=False, encoding="utf-8-sig")

pop_change = (dem.需求人口_联通 - dem.需求人口_联通_修订前)
audit = {
    "250m网格社区归属": str(DEMAND250.relative_to(HERE.parent)),
    "医院吸引力倍数（相对社康）": {k: round(float(v), 2) for k, v in TIER_MULT.items()},
    "社康平均年诊疗人次": round(float(chc_visits)),
    "社康": int(len(fc)), "点位不在任何250m网格的社康": no_cell,
    "社康点位": str(CHC_XY), "新点位不在任何社区的社康": no_comm,
    "点位核查更新的社康": upd.facility_name.tolist(),
    "归属冲突_社康": int(fc.conflict.sum()), "归属冲突_社康_修订网格归属": int((fc.conflict & fc.touch).sum()),
    "归属冲突_社康_与所在网格不相交需移动": int(len(moved)),
    "移动的社康": moved[["facility_name", "SQNAME", "move_m"]].to_dict("records"),
    "归属冲突_医院AOI网格_修订网格归属": int((fh.conflict & fh.touch).sum()),
    "去掉的医院AOI网格": int(len(drop)),
    "去掉的AOI面积占比_各医院": {k: round(float(v), 3) for k, v in drop.groupby("fac_name").footprint_share_all.sum().items()},
    "修订归属的250m网格": int(new.Tid.nunique()),
    "修订归属_按面积比例的网格": len(no_built),
    "修订网格联通人": round(float(cell_pop.loc[sorted(new.Tid.unique()), "联通人"].sum()), 1),
    "转给设施归属社区的联通人": round(float(rc["转给设施归属社区的联通人"].sum()), 1),
    "需求人口变化的社区": int((pop_change.abs() > 0.5).sum()),
    "需求人口变化_绝对值合计/2": round(float(pop_change.abs().sum() / 2), 1),
    "设施250m网格": int(fp.Tid_250.nunique()), "设施500m网格": int(fp.Tid_500.nunique()),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
