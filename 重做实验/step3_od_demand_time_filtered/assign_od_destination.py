"""500 m 网格门诊到达需求（D）按内部 250 m 网格的设施吸引力分配到 250 m 网格和社区（SQCODE），并与出发端合成社区 OD。

分配思路与出发端 step1_grid_community/grid500_origin_demand_by_250m.py 相同，出发端用 250 m 需求人口，到达端用 250 m 设施吸引力：
1. 设施吸引力取供给评价的设施资源权重 resource_weight（社区 RA 即生活圈内设施 resource_weight 之和），
   医院再乘以层级倍数，换算到社康尺度：倍数 = 该层级医院 2018 年平均门诊量（hospital_aoi_35 的 outpat）÷ 社康平均年诊疗人次；
   社康平均年诊疗人次 = 2018 年全市总诊疗人次（取 2017 年 8705.44 万与 2019 年 10811.72 万的中值）× 社康诊疗量占比 33.4% ÷ 社康数，
   社康按所在 250 m 网格计入，医院按 AOI 与 250 m 网格相交面积比例拆分计入；吸引力归设施归属社区
   （设施网格与吸引力由 facility_grid250.py 计算，见第 2、3 条）。
2. 250 m 网格按中心点嵌套于 500 m 网格。每个 500 m 到达网格的到达人次按候选 250 m 网格的设施吸引力占比分配到 250 m 网格：
   本格有设施的到达网格由本格设施分配（×1）；本格没有设施的到达网格由周围第一圈设施分配（×d）。
   按此分配后没有到达人次的设施，先核查点位（check_zero_arrival_points.py，不准确的在 chc_point_updates.csv 中更新点位，
   重算设施、250 m 网格与社区归属后再分配）；点位准确仍没有到达人次的，若为医院（二级、三级）或所属社区到达人次为 0，
   作为补给参与周围第二圈含该设施网格的到达网格的吸引力分配（×d/2）。
   周围第一圈为 500 m 网格四周相邻的一圈 250 m 网格（以该 500 m 网格为中心 4×4 个 250 m 网格去掉内部 2×2，共 12 个，
   网格中心距 500 m 网格中心不超过约 453 m），第二圈为再外一圈（6×6 去掉内部 4×4，共 20 个）。
   设施所在 250 m 网格、吸引力与 250 m 网格社区归属取 facility_grid250.py 结果（设施归属社区与所在网格冲突时修订网格社区归属，
   出发端与到达端共用修订后的 grid250_demand_allocation_revised.csv）。
3. 设施归属社区（facility_grid250.py）：社康归点位所在社区（2018年深圳CHC_912个点位_XY_WGS84.xlsx，点位核查更新的用
   chc_point_updates.csv）；医院归 AOI 主社区。
   250 m 网格到社区：网格内设施都归同一社区（无竞争）时，人次直接给该社区；
   网格内有多个社区的设施（有竞争）时，按各社区设施吸引力 × 该社区在网格社区归属中的权重分配。
   本格与周围第一圈都没有候选设施的到达网格按 maup 整格归属（grid500_origin_maup/grid500_community_maup.csv）。
4. 分配逐条 OD 记录进行，起点按出发端权重（origin_grid500_community_weights.csv）归社区，得到有方向的社区 OD；
   同时按出发端 250 m 挂接（grid250_origin_allocation.csv）保留 250 m 网格层级的 OD（od_250m.csv.gz）。
OD 与出发端相同，取本目录 results/grid500_origin_250m/outpatient_grid_OD_home_tid_clean_used.csv（时间筛选后的门诊 OD）。
对比口径：maup 整格；现行50（仅最大社区片段<50% 的争议网格按吸引力分配，其余 maup）。
补给的圈层折扣做敏感性：命令行参数为第一圈折扣（默认 0.5），第二圈取其一半；本格无设施且没有补给设施的到达网格只分给第一圈，
折扣不影响其分配；
结果写入 results/discount_<折扣>/。
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
OUT = HERE / "results" / f"discount_{D1:g}"
OUT.mkdir(parents=True, exist_ok=True)
SRC = HERE.parent / "supply input"
STEP1 = HERE.parent / "step1_grid_community/results"
NEW = HERE / "results"
OD = NEW / "grid500_origin_250m/outpatient_grid_OD_home_tid_clean_used.csv"
ORIGIN_W = NEW / "grid500_origin_250m/origin_grid500_community_weights.csv"
ORIGIN_250 = NEW / "grid500_origin_250m/grid250_origin_allocation.csv"
MAUP500 = STEP1 / "grid500_origin_maup/grid500_community_maup.csv"
DEMAND250 = NEW / "facility_grid250/grid250_demand_allocation_revised.csv"
FACG = NEW / "facility_grid250/facility_grid250_attraction.csv"
CRS = 4547
DISPUTE = 0.5
DISCOUNT = (1.0, D1, D1 / 2)  # 本格、周围第一圈、周围第二圈

node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
od = pd.read_csv(OD, encoding="utf-8-sig")[["home_tid", "end_tid", "count"]]
g500 = pd.read_csv(MAUP500, encoding="utf-8-sig", dtype={"SQCODE_maup": str})[
    ["Tid", "SQCODE_maup", "max_area_share", "disputed"]]
d250 = pd.read_csv(DEMAND250, encoding="utf-8-sig", dtype={"SQCODE": str})[["Tid", "SQCODE", "weight"]]
d250 = d250[d250.weight > 0]

# ---------- 250 m 网格嵌套于 500 m 网格（中心点落入） ----------
grid500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "x", "y", "geometry"]].to_crs(CRS)
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
c250 = g250.assign(geometry=g250.geometry.centroid)
nest = gpd.sjoin(c250, grid500[["Tid", "geometry"]], predicate="within", lsuffix="250", rsuffix="500")[
    ["Tid_250", "Tid_500"]]
assert nest.Tid_250.is_unique

# 500 m 网格周围第一圈 250 m 网格（500 m 为 0.004° 规则格网，250 m 为 0.002°，与 500 m 网格对齐，按行列号取邻域）：
# 500 m 网格（col, row）内部为 250 m 网格（2col..2col+1, 2row..2row+1），第 k 圈为其外第 k 圈 250 m 网格
xy = grid500[["Tid", "x", "y"]].copy()
xy["col"] = ((xy.x - xy.x.min()) / 0.004).round().astype(int)
xy["row"] = ((xy.y - xy.y.min()) / 0.004).round().astype(int)
b25 = g250.to_crs(4326).bounds
xy250 = pd.DataFrame({"Tid": g250.Tid,
                      "col": ((b25.minx - (xy.x.min() - 0.002)) / 0.002).round().astype(int),
                      "row": ((b25.miny - (xy.y.min() - 0.002)) / 0.002).round().astype(int)})
cell250 = xy250.set_index(["col", "row"]).Tid
assert cell250.index.is_unique
chk = nest.merge(xy250.rename(columns={"Tid": "Tid_250"}), on="Tid_250").merge(
    xy.rename(columns={"Tid": "Tid_500"}), on="Tid_500", suffixes=("_250", "_500"))
assert ((chk.col_250 // 2 == chk.col_500) & (chk.row_250 // 2 == chk.row_500)).all(), "250 m 与 500 m 行列号不对齐"


def lev(o):
    return max(-o, o - 1, 0)


def ring(k):
    offs = range(-k, k + 2)
    r = pd.concat([pd.DataFrame({"end_tid": xy.Tid, "key": list(zip(2 * xy.col + dc, 2 * xy.row + dr))})
                   for dc in offs for dr in offs if max(lev(dc), lev(dr)) == k])
    r["Tid_250"] = r.key.map(cell250)
    return r.dropna(subset=["Tid_250"]).astype({"Tid_250": int})[["end_tid", "Tid_250"]]


nb1, nb2 = ring(1), ring(2)

# ---------- 250 m 网格设施吸引力（facility_grid250.py：社康网格跟随归属社区，医院只含归属社区的 AOI 网格） ----------
fp = pd.read_csv(FACG, encoding="utf-8-sig", dtype={"SQCODE": str, "SQCODE_located": str})
fp.to_csv(OUT / "facility_grid250_attraction.csv", index=False, encoding="utf-8-sig")
TIER_MULT = fp.groupby("tier").tier_mult.first().to_dict()
sqname = node.set_index("SQCODE").SQNAME
base_w = d250.set_index(["Tid", "SQCODE"]).weight  # 250 m 网格社区归属权重（与出发端一致）

# ---------- 每个 500 m 网格的到达权重 ----------
arrive = od.groupby("end_tid")["count"].sum().rename("arrivals")
fpa = fp.dropna(subset=["Tid_500"]).astype({"Tid_500": int})
cols = ["Tid_500", "Tid_250", "facility_id", "facility_name", "tier", "SQCODE", "attraction"]
fac500 = set(fpa.Tid_500)


def candidates(extra):
    # 500 m → 250 m：本格有设施的到达网格由本格设施分配（×1）；本格没有设施的由周围第一圈 250 m 网格的设施分配（×d）；
    # extra 为补给设施（点位准确仍无到达人次，且为医院或所在社区到达人次为 0），参与周围第二圈含该设施网格的到达网格（×d/2）
    r1 = nb1[~nb1.end_tid.isin(fac500)].merge(fpa[cols], on="Tid_250")
    parts = [fpa[cols].assign(end_tid=fpa.Tid_500, discount=DISCOUNT[0], ring="本格"),
             r1.assign(discount=DISCOUNT[1], ring="周围第一圈")]
    if len(extra):
        e2 = nb2.merge(extra[cols], on="Tid_250")
        parts.append(e2.assign(discount=DISCOUNT[2], ring="补给第二圈"))
    c = pd.concat(parts, ignore_index=True)
    c = c[c.end_tid.isin(arrive.index)].copy()
    assert not c.duplicated(["end_tid", "facility_id", "Tid_250"]).any()
    c["cell_attraction"] = c.groupby(["end_tid", "Tid_250"]).attraction.transform("sum")
    cellw = c.drop_duplicates(["end_tid", "Tid_250"]).assign(w=lambda d: d.cell_attraction * d.discount)
    c["share_250"] = c.set_index(["end_tid", "Tid_250"]).index.map(
        (cellw.set_index(["end_tid", "Tid_250"]).w / cellw.groupby("end_tid").w.sum().reindex(cellw.end_tid).to_numpy()))
    return c


a0 = candidates(fpa.iloc[:0])
zero = sorted(set(fpa.SQCODE) - set(a0.SQCODE))  # 有设施但必要分配后到达人次为 0 的社区
zero_fac = fpa[~fpa.facility_id.isin(a0.facility_id)]  # 必要分配后没有到达人次的设施
extra = zero_fac[(zero_fac.tier != "CHC") | zero_fac.SQCODE.isin(zero)]
a = candidates(extra)
still_zero = sorted(set(zero) - set(a.SQCODE))
extra_unreached = sorted(set(extra.facility_name) - set(a.facility_name))
# 250 m → 社区：网格内设施都归同一社区（无竞争）时直接给该社区；有竞争时按各社区设施吸引力 × 该社区在网格社区归属中的权重
a["n_comm"] = a.groupby(["end_tid", "Tid_250"]).SQCODE.transform("nunique")
direct = a[a.n_comm == 1].drop_duplicates(["end_tid", "Tid_250"]).copy()
direct["attraction"] = direct.cell_attraction
direct["comm_share"] = 1.0
direct["rule"] = direct.ring + "_无竞争给设施所在社区"
comp = (a[a.n_comm > 1].groupby(["end_tid", "Tid_250", "SQCODE"], as_index=False)
        .agg(attraction=("attraction", "sum"), share_250=("share_250", "first"), ring=("ring", "first")))
comp["base_w"] = [base_w.get((t_, s_), 0.0) for t_, s_ in zip(comp.Tid_250, comp.SQCODE)]
comp["w"] = comp.attraction * comp.base_w
comp["comm_share"] = comp.w / comp.groupby(["end_tid", "Tid_250"]).w.transform("sum")
comp["rule"] = comp.ring + "_有竞争按吸引力×网格社区归属权重"
comp.assign(SQNAME=comp.SQCODE.map(sqname)).drop_duplicates(["Tid_250", "SQCODE"]).to_csv(
    OUT / "grid250_competition.csv", index=False, encoding="utf-8-sig")
assert (comp.base_w > 0).all() and comp.comm_share.notna().all(), "设施归属社区须在所在 250 m 网格的社区归属内"
a = pd.concat([direct, comp], ignore_index=True)
a["d_share"] = a.share_250 * a.comm_share
a = a[["end_tid", "Tid_250", "SQCODE", "attraction", "d_share", "rule"]]
assert np.allclose(a.groupby("end_tid").d_share.sum(), 1)
maup_of = g500.set_index("Tid").SQCODE_maup
rest = arrive.index.difference(a.end_tid)
pc = pd.DataFrame({"end_tid": rest, "SQCODE": maup_of.reindex(rest).to_numpy(), "d_share": 1.0, "attraction": 0.0,
                   "rule": "maup_本格与周围一圈都无设施"})
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
fac_n = fp.drop_duplicates("facility_id").groupby("SQCODE").size()  # 设施归属社区取 facility_grid250.py 结果
t["located_facilities"] = t.SQCODE.map(fac_n).fillna(0).astype(int)
t["internal_trips"] = t.SQCODE.map(cod[cod.SQCODE_o == cod.SQCODE_d].set_index("SQCODE_o").trips).fillna(0)
t.to_csv(OUT / "community_destination_demand.csv", index=False, encoding="utf-8-sig")

# ---------- 医院 AOI 网格的到达人次留在医院所在社区的比例 ----------
haoi = fpa[fpa.tier != "CHC"].drop_duplicates(["facility_name", "Tid_500"])
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
    "医院吸引力倍数（相对社康）": {k: round(float(v), 2) for k, v in TIER_MULT.items()},
    "OD": str(OD), "到达网格": int(len(arrive)), "到达人次": int(arrive.sum()),
    "设施吸引力": "facility_grid250.py：resource_weight × 层级倍数，社康网格跟随归属社区，医院 AOI 按面积拆分到归属社区的 250 m 网格",
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
    "补给社区（必要分配后有设施但到达人次为0）": [sqname[s_] for s_ in zero],
    "必要分配后无到达人次的设施": int(zero_fac.facility_id.nunique()),
    "补给设施": int(extra.facility_id.nunique()),
    "补给医院": sorted(extra.loc[extra.tier != "CHC", "facility_name"].unique().tolist()),
    "补给后仍无到达人次的补给设施": extra_unreached,
    "补给后仍为0的补给社区": [sqname[s_] for s_ in still_zero],
    "本格与周围一圈都无设施_按maup的到达网格": int(len(pc)),
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
