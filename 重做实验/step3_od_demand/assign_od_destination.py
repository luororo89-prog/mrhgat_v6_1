"""500 m 网格门诊到达需求（D）按内部 250 m 网格的设施吸引力分配到 250 m 网格和社区（SQCODE），并与出发端合成社区 OD。

分配思路与出发端 step1_grid_community/grid500_origin_demand_by_250m.py 相同，出发端用 250 m 需求人口，到达端用 250 m 设施吸引力：
1. 设施吸引力取供给评价的设施资源权重 resource_weight（社区 RA 即生活圈内设施 resource_weight 之和），
   医院再乘以层级倍数，换算到社康尺度：倍数 = 该层级医院 2018 年平均门诊量（hospital_aoi_35 的 outpat）÷ 社康平均年诊疗人次；
   社康平均年诊疗人次 = 2018 年全市总诊疗人次（取 2017 年 8705.44 万与 2019 年 10811.72 万的中值）× 社康诊疗量占比 33.4% ÷ 社康数，
   社康按点位落入的 250 m 网格计入，医院按 AOI 与 250 m 网格相交面积比例拆分计入；吸引力归设施所在社区
   （step2_supply/results/facility_community.csv，医院为 AOI 主社区）。
2. 250 m 网格按中心点嵌套于 500 m 网格。每个 500 m 到达网格的到达人次按候选 250 m 网格的设施吸引力占比分配到 250 m 网格，
   候选为本格内设施（×1）；本格没有到达人次的设施参与周围第一圈到达网格的分配（×0.5），
   第一圈也没有到达人次的再参与第二圈（×0.25）；本格没有设施的到达网格，周围第一圈设施都参与（×0.5）。
3. 社康先与社区名字对齐：取站点名称中“社区健康服务”之前、举办医院名之后的部分（“本部”站点取举办医院名去掉“人民医院”），
   以其所在 250 m 网格所跨社区的社区名结尾时归该社区（多个匹配取名称最长者），否则归设施所在社区；
   step2_supply/facility_corrections.csv 中已人工核对的社康直接用核对结果（facility_community.csv）；医院归 AOI 主社区。
   250 m 网格到社区：网格内设施都归同一社区（无竞争）时，人次直接给该社区；
   网格内有多个社区的设施（有竞争）时，按各社区设施吸引力 × 该社区占网格面积比例分配。
   两圈内都没有候选设施的到达网格按 maup 整格归属（grid500_origin_maup/grid500_community_maup.csv）。
4. 分配逐条 OD 记录进行，起点按出发端权重（origin_grid500_community_weights.csv）归社区，得到有方向的社区 OD；
   同时按出发端 250 m 挂接（grid250_origin_allocation.csv）保留 250 m 网格层级的 OD（od_250m.csv.gz）。
OD 与出发端相同，取 grid500_origin_250m/outpatient_grid_OD_home_tid_clean_used.csv。
对比口径：maup 整格；现行50（仅最大社区片段<50% 的争议网格按吸引力分配，其余 maup）。
周围圈层折扣做敏感性：命令行参数为第一圈折扣（默认 0.5），第二圈取其一半；结果写入 results/discount_<第一圈折扣>/。
第二个参数为 hospring 时，医院不论本格有无到达人次都参与周围第一圈的分配（×第一圈折扣），结果写入 results/discount_<折扣>_hospring/。
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
D1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
HOSP_RING = len(sys.argv) > 2 and sys.argv[2] == "hospring"  # 医院始终参与周围第一圈的分配
OUT = HERE / "results" / (f"discount_{D1:g}" + ("_hospring" if HOSP_RING else ""))
OUT.mkdir(parents=True, exist_ok=True)
SRC = HERE.parent / "supply input"
STEP1 = HERE.parent / "step1_grid_community/results"
STEP2 = HERE.parent / "step2_supply/results"
OD = STEP1 / "grid500_origin_250m/outpatient_grid_OD_home_tid_clean_used.csv"
ORIGIN_W = STEP1 / "grid500_origin_250m/origin_grid500_community_weights.csv"
ORIGIN_250 = STEP1 / "grid500_origin_250m/grid250_origin_allocation.csv"
MAUP500 = STEP1 / "grid500_origin_maup/grid500_community_maup.csv"
DEMAND250 = STEP1 / "landuse_demand/grid250_demand_allocation.csv"
FAC_COMM = STEP2 / "facility_community.csv"
FAC_AUDIT = STEP2 / "supply_indicators/facility_supply_audit_2018_mobile_users.csv"
AOI = SRC / "hospital_aoi_35.shp"
COMM = SRC / "深圳社区objectid.shp"
CORR = HERE.parent / "step2_supply/facility_corrections.csv"
CRS = 4547
DISPUTE = 0.5
DISCOUNT = (1.0, D1, D1 / 2)  # 本格、周围第一圈、周围第二圈
TOTAL_VISITS_2018 = (8705.44 + 10811.72) / 2 * 1e4  # 深圳统计公报 2017、2019 年总诊疗人次的中值
CHC_VISIT_SHARE = 0.334  # 社康诊疗量占全市诊疗量比例（深圳市卫健委）

node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
od = pd.read_csv(OD, encoding="utf-8-sig")[["home_tid", "end_tid", "count"]]
g500 = pd.read_csv(MAUP500, encoding="utf-8-sig", dtype={"SQCODE_maup": str})[
    ["Tid", "SQCODE_maup", "max_area_share", "disputed"]]
d250 = pd.read_csv(DEMAND250, encoding="utf-8-sig", dtype={"SQCODE": str})[["Tid", "SQCODE", "rule"]]
d250 = d250[["Tid", "SQCODE"]]

# ---------- 250 m 网格嵌套于 500 m 网格（中心点落入） ----------
grid500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "x", "y", "geometry"]].to_crs(CRS)
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
c250 = g250.assign(geometry=g250.geometry.centroid)
nest = gpd.sjoin(c250, grid500[["Tid", "geometry"]], predicate="within", lsuffix="250", rsuffix="500")[
    ["Tid_250", "Tid_500"]]
assert nest.Tid_250.is_unique

# 500 m 网格周围第一圈、第二圈（0.004° 规则格网，按行列号取邻域）
xy = grid500[["Tid", "x", "y"]].copy()
xy["col"] = ((xy.x - xy.x.min()) / 0.004).round().astype(int)
xy["row"] = ((xy.y - xy.y.min()) / 0.004).round().astype(int)
cell = xy.set_index(["col", "row"]).Tid


def ring(k):
    r = pd.concat([pd.DataFrame({"end_tid": xy.Tid, "key": list(zip(xy.col + dc, xy.row + dr))})
                   for dc in range(-k, k + 1) for dr in range(-k, k + 1) if max(abs(dc), abs(dr)) == k])
    r["Tid_500"] = r.key.map(cell)
    return r.dropna(subset=["Tid_500"]).astype({"Tid_500": int})[["end_tid", "Tid_500"]]


nb1, nb2 = ring(1), ring(2)

# ---------- 250 m 网格设施吸引力 ----------
fac = pd.read_csv(FAC_COMM, dtype={"SQCODE": str})[["facility_id", "facility_name", "tier", "SQCODE"]].merge(
    pd.read_csv(FAC_AUDIT)[["facility_id", "resource_weight", "longitude", "latitude"]], on="facility_id")
chc = fac[(fac.tier == "CHC") & fac.SQCODE.notna()]
chc = gpd.GeoDataFrame(chc, geometry=gpd.points_from_xy(chc.longitude, chc.latitude), crs=4326).to_crs(CRS)
fc = gpd.sjoin(chc, g250.rename(columns={"Tid": "Tid_250"}), predicate="within")
fc = fc[["facility_id", "facility_name", "tier", "SQCODE", "Tid_250", "resource_weight"]].assign(footprint_share=1.0)
aoi = gpd.read_file(AOI).to_crs(CRS)[["fac_name", "geometry"]]
aoi["geometry"] = aoi.geometry.make_valid()
hosp = fac[fac.tier != "CHC"].set_index("facility_name")
fh = gpd.overlay(aoi, g250.rename(columns={"Tid": "Tid_250"}), how="intersection", keep_geom_type=True)
fh["m2"] = fh.area
fh = fh[fh.m2 > 0]
fh["footprint_share"] = fh.m2 / fh.groupby("fac_name").m2.transform("sum")
for c in ["facility_id", "tier", "SQCODE", "resource_weight"]:
    fh[c] = fh.fac_name.map(hosp[c])
fh["facility_name"] = fh.fac_name
fp = pd.concat([fc, fh[["facility_id", "facility_name", "tier", "SQCODE", "Tid_250", "resource_weight",
                        "footprint_share"]]], ignore_index=True)
aoi_out = gpd.read_file(AOI)[["tier", "outpat"]]
chc_visits = TOTAL_VISITS_2018 * CHC_VISIT_SHARE / int((fac.tier == "CHC").sum())
TIER_MULT = (aoi_out.groupby("tier").outpat.mean() / chc_visits).to_dict()
TIER_MULT["CHC"] = 1.0
fp["tier_mult"] = fp.tier.map(TIER_MULT)
fp["attraction"] = fp.resource_weight * fp.footprint_share * fp.tier_mult
fp = fp.merge(nest, on="Tid_250", how="left")

# 社康与社区名字对齐：在社康所在 250 m 网格所跨的社区中按站点名称匹配
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.make_valid()
cells = g250[g250.Tid.isin(fp.Tid_250)].rename(columns={"Tid": "Tid_250"})
pieces_fac = gpd.overlay(cells, comm, how="intersection", keep_geom_type=True)
pieces_fac["m2"] = pieces_fac.area
pieces_fac = pieces_fac[pieces_fac.m2 > 0][["Tid_250", "SQCODE", "m2"]]
pieces_fac["area_share"] = pieces_fac.m2 / pieces_fac.groupby("Tid_250").m2.transform("sum")
area_share = pieces_fac.set_index(["Tid_250", "SQCODE"]).area_share
sqname = node.set_index("SQCODE").SQNAME


def station_stem(name):
    i = name.rfind("医院")
    own = name[i + 2:] if i >= 0 else name
    if own.startswith("本部"):
        return name[:i + 2].removesuffix("人民医院").removesuffix("医院")
    k = own.find("社区健康服务")
    return own[:k] if k >= 0 else own


def by_name(name, tid):
    stem = station_stem(name)
    hit = [s for s in pieces_fac.loc[pieces_fac.Tid_250 == tid, "SQCODE"] if stem.endswith(sqname[s])]
    return max(hit, key=lambda s: len(sqname[s])) if hit else None


checked = set(pd.read_csv(CORR, encoding="utf-8-sig").facility_id)
fp["SQCODE_located"] = fp.SQCODE
named = [by_name(n, t_) if tier == "CHC" and f not in checked else None
         for f, n, t_, tier in zip(fp.facility_id, fp.facility_name, fp.Tid_250, fp.tier)]
fp["SQCODE"] = pd.Series(named, index=fp.index).fillna(fp.SQCODE_located)
fp.to_csv(OUT / "facility_grid250_attraction.csv", index=False, encoding="utf-8-sig")

# ---------- 每个 500 m 网格的到达权重 ----------
arrive = od.groupby("end_tid")["count"].sum().rename("arrivals")
fpa = fp.dropna(subset=["Tid_500"]).astype({"Tid_500": int})
cols = ["Tid_500", "Tid_250", "facility_id", "facility_name", "tier", "SQCODE", "attraction"]
# 500 m → 250 m：本格设施 ×1；本格无到达人次的设施参与第一圈 ×0.5，第一圈也无到达人次的参与第二圈 ×0.25；
# 本格没有设施的到达网格由第一圈全部设施分配（×0.5）
near = set(arrive.index) | set(nb1.loc[nb1.Tid_500.isin(arrive.index), "end_tid"])
far = fpa[~fpa.Tid_500.isin(near)]
r1 = nb1.merge(fpa[cols], on="Tid_500")
r1 = r1[~r1.Tid_500.isin(arrive.index) | ~r1.end_tid.isin(fpa.Tid_500) | (HOSP_RING & (r1.tier != "CHC"))]
cand = pd.concat([
    fpa[cols].assign(end_tid=fpa.Tid_500, discount=DISCOUNT[0], ring="本格"),
    r1.assign(discount=DISCOUNT[1], ring="周围第一圈"),
    nb2.merge(far[cols], on="Tid_500").assign(discount=DISCOUNT[2], ring="周围第二圈"),
], ignore_index=True)
a = cand[cand.end_tid.isin(arrive.index)].copy()
a["cell_attraction"] = a.groupby(["end_tid", "Tid_250"]).attraction.transform("sum")
cellw = a.drop_duplicates(["end_tid", "Tid_250"]).assign(w=lambda d: d.cell_attraction * d.discount)
a["share_250"] = a.set_index(["end_tid", "Tid_250"]).index.map(
    (cellw.set_index(["end_tid", "Tid_250"]).w / cellw.groupby("end_tid").w.sum().reindex(cellw.end_tid).to_numpy()))
# 250 m → 社区：网格内设施都归同一社区（无竞争）时直接给该社区；有竞争时按各社区设施吸引力 × 社区占网格面积比例
a["n_comm"] = a.groupby(["end_tid", "Tid_250"]).SQCODE.transform("nunique")
direct = a[a.n_comm == 1].drop_duplicates(["end_tid", "Tid_250"]).copy()
direct["attraction"] = direct.cell_attraction
direct["comm_share"] = 1.0
direct["rule"] = direct.ring + "_无竞争给设施所在社区"
comp = (a[a.n_comm > 1].groupby(["end_tid", "Tid_250", "SQCODE"], as_index=False)
        .agg(attraction=("attraction", "sum"), share_250=("share_250", "first"), ring=("ring", "first")))
comp["area_share"] = [area_share.get((t_, s_), 0.0) for t_, s_ in zip(comp.Tid_250, comp.SQCODE)]
comp["w"] = comp.attraction * comp.area_share
comp["comm_share"] = comp.w / comp.groupby(["end_tid", "Tid_250"]).w.transform("sum")
comp["rule"] = comp.ring + "_有竞争按吸引力×社区面积占比"
comp.assign(SQNAME=comp.SQCODE.map(sqname)).drop_duplicates(["Tid_250", "SQCODE"]).to_csv(
    OUT / "grid250_competition.csv", index=False, encoding="utf-8-sig")
assert comp.comm_share.notna().all(), "有竞争的 250 m 网格中设施所属社区与网格不相交"
a = pd.concat([direct, comp], ignore_index=True)
a["d_share"] = a.share_250 * a.comm_share
a = a[["end_tid", "Tid_250", "SQCODE", "attraction", "d_share", "rule"]]
assert np.allclose(a.groupby("end_tid").d_share.sum(), 1)
maup_of = g500.set_index("Tid").SQCODE_maup
rest = arrive.index.difference(a.end_tid)
pc = pd.DataFrame({"end_tid": rest, "SQCODE": maup_of.reindex(rest).to_numpy(), "d_share": 1.0, "attraction": 0.0,
                   "rule": "maup_两圈内都无设施"})
assert pc.SQCODE.notna().all(), "到达网格必须与社区相交"
a250 = pd.concat([a, pc], ignore_index=True)
a250 = a250.merge(arrive, left_on="end_tid", right_index=True)
a250["D_trips"] = a250.arrivals * a250.d_share
a250["Tid_250"] = a250.Tid_250.astype("Int64")
a250 = a250.merge(node[["OBJECTID", "SQCODE"]], on="SQCODE", how="left", validate="many_to_one")
assert a250.OBJECTID.notna().all() and np.allclose(a250.groupby("end_tid").d_share.sum(), 1)
a250[["end_tid", "Tid_250", "SQCODE", "OBJECTID", "rule", "attraction", "d_share", "arrivals", "D_trips"]].sort_values(
    ["end_tid", "Tid_250", "SQCODE"]).to_csv(OUT / "grid250_destination_allocation.csv", index=False,
                                             encoding="utf-8-sig")
wd = a250.groupby(["end_tid", "SQCODE", "OBJECTID", "rule"], as_index=False).d_share.sum().rename(
    columns={"d_share": "weight"})
wd.to_csv(OUT / "destination_grid500_community_weights.csv", index=False, encoding="utf-8-sig")

# ---------- 有方向的社区 OD：逐条 OD 记录分配起点社区与到达社区 ----------
wo = pd.read_csv(ORIGIN_W, encoding="utf-8-sig", dtype={"SQCODE": str})[["home_tid", "SQCODE", "OBJECTID", "weight"]]
assert set(od.home_tid) <= set(wo.home_tid)
f = (od.merge(wo, on="home_tid").rename(columns={"SQCODE": "SQCODE_o", "OBJECTID": "OBJECTID_o", "weight": "w_o"})
     .merge(wd[["end_tid", "SQCODE", "OBJECTID", "weight"]], on="end_tid")
     .rename(columns={"SQCODE": "SQCODE_d", "OBJECTID": "OBJECTID_d", "weight": "w_d"}))
f["trips"] = f["count"] * f.w_o * f.w_d
cod = f.groupby(["OBJECTID_o", "SQCODE_o", "OBJECTID_d", "SQCODE_d"], as_index=False).trips.sum()
assert np.isclose(cod.trips.sum(), od["count"].sum())
cod.to_csv(OUT / "community_od.csv", index=False, encoding="utf-8-sig")

# ---------- 250 m 网格 OD：起点 250 m 网格（社区）→ 到达 250 m 网格（社区），逐条 OD 记录展开 ----------
w250o = pd.read_csv(ORIGIN_250, encoding="utf-8-sig", dtype={"SQCODE": str})[["home_tid", "Tid_250", "SQCODE", "o_share"]]
f250 = (od.merge(w250o.rename(columns={"Tid_250": "Tid_250_o", "SQCODE": "SQCODE_o"}), on="home_tid")
        .merge(a250[["end_tid", "Tid_250", "SQCODE", "d_share"]].rename(
            columns={"Tid_250": "Tid_250_d", "SQCODE": "SQCODE_d"}), on="end_tid"))
f250["trips"] = f250["count"] * f250.o_share * f250.d_share
od250 = f250.groupby(["Tid_250_o", "SQCODE_o", "Tid_250_d", "SQCODE_d"], as_index=False, dropna=False).trips.sum()
assert np.isclose(od250.trips.sum(), od["count"].sum())
chk = od250.groupby(["SQCODE_o", "SQCODE_d"]).trips.sum()
assert np.allclose(chk.reindex(cod.set_index(["SQCODE_o", "SQCODE_d"]).index).to_numpy(), cod.trips.to_numpy()), \
    "250 m OD 汇总到社区须与社区 OD 一致"
od250.to_csv(OUT / "od_250m.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
a250.groupby(["Tid_250", "SQCODE", "OBJECTID"], as_index=False, dropna=False).D_trips.sum().to_csv(
    OUT / "grid250_destination.csv", index=False, encoding="utf-8-sig")

# ---------- 社区到达需求与对比 ----------
t = node[["OBJECTID", "SQCODE", "SQNAME"]].copy()
t["arrivals"] = t.SQCODE.map(a250.groupby("SQCODE").D_trips.sum()).fillna(0)
end = g500.set_index("Tid").loc[arrive.index].assign(arrivals=arrive)
t["arrivals_maup"] = t.SQCODE.map(end.groupby("SQCODE_maup").arrivals.sum()).fillna(0)
dis = end[end.max_area_share < DISPUTE].index
by_att = a250.rule.str.startswith("250m设施吸引力占比")
v50 = pd.concat([a250[a250.end_tid.isin(dis) & by_att][["SQCODE", "D_trips"]],
                 end[~end.index.isin(a250.loc[a250.end_tid.isin(dis) & by_att, "end_tid"])]
                 .rename(columns={"SQCODE_maup": "SQCODE", "arrivals": "D_trips"})[["SQCODE", "D_trips"]]])
t["arrivals_现行50"] = t.SQCODE.map(v50.groupby("SQCODE").D_trips.sum()).fillna(0)
fac_n = fac.dropna(subset=["SQCODE"]).groupby("SQCODE").size()
t["located_facilities"] = t.SQCODE.map(fac_n).fillna(0).astype(int)
t["internal_trips"] = t.SQCODE.map(cod[cod.SQCODE_o == cod.SQCODE_d].set_index("SQCODE_o").trips).fillna(0)
t.to_csv(OUT / "community_destination_demand.csv", index=False, encoding="utf-8-sig")

# ---------- 医院 AOI 网格的到达人次留在医院所在社区的比例 ----------
haoi = fh.merge(nest, on="Tid_250").drop_duplicates(["fac_name", "Tid_500"])
haoi = haoi[haoi.Tid_500.isin(arrive.index)]
wdi = wd.groupby(["end_tid", "SQCODE"]).weight.sum()


def hospital_share(get_w):
    tot = own = 0.0
    for _, r in haoi.iterrows():
        n = arrive[r.Tid_500]
        tot += n
        own += n * get_w(r.Tid_500, r.SQCODE)
    return round(own / tot, 3)


maup_w = lambda t_, s: float(maup_of[t_] == s)
og = a250.drop_duplicates("end_tid")


def summary(col):
    o = t[col]
    return {"到达人次为0的社区": int((o == 0).sum()),
            "有设施但到达人次为0的社区": int(((o == 0) & (t.located_facilities > 0)).sum()),
            "无设施但到达人次大于0的社区": int(((o > 0) & (t.located_facilities == 0)).sum()),
            "社区到达人次与社区设施数Spearman": round(float(spearmanr(t.located_facilities, o).statistic), 4)}


audit = {
    "折扣_本格_第一圈_第二圈": list(DISCOUNT),
    "医院始终参与周围第一圈": bool(HOSP_RING),
    "医院吸引力倍数（相对社康）": {k: round(float(v), 2) for k, v in TIER_MULT.items()},
    "社康平均年诊疗人次": round(float(chc_visits)),
    "OD": str(OD), "到达网格": int(len(arrive)), "到达人次": int(arrive.sum()),
    "设施吸引力": "resource_weight，社康落点、医院 AOI 按面积拆分到 250 m 网格，归设施所在社区",
    "有吸引力的250m网格": int(fp.Tid_250.nunique()),
    "到达网格_按规则": og.rule.str.replace("_250m.*", "", regex=True).value_counts().to_dict(),
    "到达人次_按规则": {k: float(v) for k, v in og.assign(rule=og.rule.str.replace("_250m.*", "", regex=True))
                  .groupby("rule").arrivals.sum().items()},
    "250m网格到社区_按规则人次": {k: round(float(v), 1) for k, v in a250.groupby("rule").D_trips.sum().items()},
    "无竞争的250m网格": int(a250.loc[a250.rule.str.contains("无竞争"), "Tid_250"].nunique()),
    "有竞争的250m网格": int(comp.Tid_250.nunique()),
    "按名字对齐改归的社康": sorted({f"{r.facility_name}：{sqname[r.SQCODE_located]} → {sqname[r.SQCODE]}"
                           for r in fp[fp.SQCODE != fp.SQCODE_located].itertuples()}),
    "到达人次_按候选圈层": {k: round(float(v), 1) for k, v in a250.assign(r=a250.rule.str.split("_").str[0])
                       .groupby("r").D_trips.sum().items()},
    "参与第一圈分配的本格无人次设施": int(r1.loc[r1.end_tid.isin(fpa.Tid_500), "facility_id"].nunique()),
    "参与第二圈分配的设施": int(far.facility_id.nunique()),
    "两圈内都无设施_按maup的到达网格": int(len(pc)),
    "到达网格_拆分到多个社区": int((wd.groupby("end_tid").size() > 1).sum()),
    "分配后到达人次合计": float(t.arrivals.sum()),
    "社区OD人次合计": float(cod.trips.sum()), "社区OD非零对": int((cod.trips > 0).sum()),
    "250m网格OD对": int(len(od250)), "250m网格OD起点网格": int(od250.Tid_250_o.nunique()),
    "250m网格OD到达网格": int(od250.Tid_250_d.nunique()),
    "社区内部人次占比": round(float(t.internal_trips.sum() / cod.trips.sum()), 4),
    "医院AOI网格人次留在医院所在社区": {"本方法": hospital_share(lambda t_, s: float(wdi.get((t_, s), 0.0))),
                             "maup整格": hospital_share(maup_w)},
    "本方法": summary("arrivals"), "现行50": summary("arrivals_现行50"), "maup整格": summary("arrivals_maup"),
    "相对maup_转移人次": round(float((t.arrivals - t.arrivals_maup).abs().sum() / 2), 1),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
