"""方案一（小社区认领 500 m 网格）按门诊 OD 与医院面域微调。

1. 医院网格先定：与医院 AOI 相交、AOI 落在该格的面积不少于 AOI 面积 MIN_AOI_SHARE、且该格有门诊到达人次的 500 m 网格，
   归医院 AOI 所在社区（与社区边界重叠面积最大者）。
2. 其余网格保持方案一的归属。
3. 以 250 m 口径的需求人口为参照补回网格。参照为社区所辖 250 m 网格（maup 归属）的联通人口之和，
   社区需求人口为所辖 500 m 网格内的联通人口之和，偏差为两者之差的绝对值除以参照。
   与医院面域相交的网格不参与补回。
   (a) 因医院网格改归而没有网格的社区，从与其重叠的网格中补一格；该格原归属社区若因此没有网格，
       再为它补一格，最多传递三层。在所有补法中取相关社区偏差之和增加最少者，相同时取改动少者。
   (b) 在医院网格改归和 (a) 中失去网格、且需求人口低于参照的社区，逐格尝试从与其重叠的网格中补入。
       补入须使本社区偏差减小、双方偏差之和下降，原归属社区须保留至少一格；每次取双方偏差之和下降最多的一格，
       直到没有可补的网格。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_nest"
NODE = HERE.parent / "input/node_order.csv"
SMALL_AREA = 0.3e6
MIN_AOI_SHARE = 0.3

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})[["SQCODE", "SQNAME", "comm_area_m2"]]
small = set(cs.loc[cs.comm_area_m2 < SMALL_AREA, "SQCODE"])
name = cs.set_index("SQCODE").SQNAME
nest = pd.read_csv(RES / "od_250m/grid250_in_grid500.csv", dtype={"SQCODE": str})
g5 = pd.read_csv(RES / "grid500_community_1to1.csv", dtype={"SQCODE": str})[["Tid", "SQCODE"]].rename(
    columns={"SQCODE": "SQCODE_maup"})
p5 = pd.read_csv(RES / "grid500_community_pieces.csv", dtype={"SQCODE": str})[["Tid", "SQCODE", "piece_m2"]]
plan1 = pd.read_csv(OUT / "plan1_grid500_community.csv", dtype={"SQCODE": str}).set_index("Tid")
aoi = pd.read_csv(RES / "facility_grid/hospital_aoi_grid500_pieces.csv", dtype={"AOI所在社区SQCODE": str})
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")
arrive = od.groupby("end_tid")["count"].sum()

# ---------- 1. 医院网格 ----------
hosp = aoi[(aoi["占AOI面积比例"] >= MIN_AOI_SHARE) & (aoi.Tid_500.map(arrive).fillna(0) > 0)]
hosp = hosp.sort_values("AOI在网格内面积_m2", ascending=False).drop_duplicates("Tid_500")
fixed = {int(t): (c, "医院面域") for t, c in zip(hosp.Tid_500, hosp["AOI所在社区SQCODE"])}

# ---------- 2. 其余网格保持方案一，3. 以 250 m 需求人口为参照补回 ----------
R = nest.groupby("SQCODE")["联通人"].sum().reindex(cs.SQCODE).fillna(0)  # 250 m 口径需求人口（参照）
gpop = nest.groupby("Tid_500")["联通人"].sum()
owner = plan1.SQCODE.copy()
basis = plan1.basis.copy()
for t, (c, b) in fixed.items():
    owner[t], basis[t] = c, b
D = pd.Series(owner.index.map(gpop).fillna(0).to_numpy(), index=owner.to_numpy()).groupby(level=0).sum() \
    .reindex(cs.SQCODE).fillna(0)
count = owner.value_counts()
lost = {plan1.SQCODE[t] for t, (c, _) in fixed.items() if plan1.SQCODE[t] != c}
aoi_grids = set(aoi.Tid_500)  # 与医院面域相交的网格不参与补回
by_comm = p5[~p5.Tid.isin(aoi_grids)].groupby("SQCODE").Tid.apply(list).to_dict()


def err(c, d):
    return abs(d - R[c]) / max(R[c], 1.0)


def delta(path):
    """补法 path（[(网格, 新社区), ...]）使相关社区偏差之和的变化量。"""
    d = {}
    for t, newc in path:
        old, g = owner[t], gpop.get(t, 0)
        d[old] = d.get(old, D[old]) - g
        d[newc] = d.get(newc, D[newc]) + g
    return sum(err(c, v) - err(c, D[c]) for c, v in d.items())


def paths(c, seen, depth):
    for t in by_comm.get(c, []):
        o = owner[t]
        if o == c or t in seen:
            continue
        if count.get(o, 0) >= 2:
            yield [(t, c)]
        elif depth > 1:
            for rest in paths(o, seen | {t}, depth - 1):
                yield [(t, c)] + rest


def apply(path, tag, track):
    for t, newc in path:
        old, g = owner[t], gpop.get(t, 0)
        D[old] -= g
        D[newc] += g
        count[old] -= 1
        count[newc] = count.get(newc, 0) + 1
        owner[t], basis[t] = newc, tag
        if track:
            lost.add(old)


# (a) 没有网格的社区必须补一格
missing = [c for c in cs.SQCODE if count.get(c, 0) == 0]
unresolved = []
for c in missing:
    options = list(paths(c, set(), 3))
    if not options:
        unresolved.append(c)
        continue
    best = min(options, key=lambda pth: (round(delta(pth), 9), len(pth)))
    apply(best, "补回_无网格", True)

# (b) 失去网格的社区，补入能降低偏差的网格
moves_b = 0
while True:
    best, best_d = None, -1e-9
    for c in sorted(lost):
        if D[c] >= R[c]:  # 需求人口已不低于参照，不再补
            continue
        for t in by_comm.get(c, []):
            o = owner[t]
            if o == c or count.get(o, 0) < 2:
                continue
            if err(c, D[c] + gpop.get(t, 0)) >= err(c, D[c]):  # 补入后本社区偏差须减小
                continue
            dd = delta([(t, c)])
            if dd < best_d:
                best, best_d = [(t, c)], dd
    if best is None:
        break
    apply(best, "补回_降低偏差", False)
    moves_b += 1
a = g5.copy()
a["SQCODE"] = a.Tid.map(owner)
a["basis"] = a.Tid.map(basis)
a["SQCODE_plan1"] = a.Tid.map(plan1.SQCODE)
a["changed_vs_plan1"] = a.SQCODE != a.SQCODE_plan1
a = a.merge(node[["SQCODE", "OBJECTID"]], on="SQCODE", how="left")
a.to_csv(OUT / "plan1_hospital_grid500_community.csv", index=False)

# ---------- 评价 ----------
owner_new = a.set_index("Tid").SQCODE
owner_old = plan1.SQCODE


def flows(owner):
    o = od.home_tid.map(owner)
    d = od.end_tid.map(owner)
    return od.groupby(o)["count"].sum(), od.groupby(d)["count"].sum()


out_old, in_old = flows(owner_old)
out_new, in_new = flows(owner_new)
d250 = nest.groupby("SQCODE")["联通人"].sum()
dem_old = nest.groupby(nest.Tid_500.map(owner_old))["联通人"].sum()
dem_new = nest.groupby(nest.Tid_500.map(owner_new))["联通人"].sum()
t = node[["OBJECTID", "SQCODE"]].merge(cs, on="SQCODE")
for k, s in {"流出_方案一": out_old, "流出_微调": out_new, "流入_方案一": in_old, "流入_微调": in_new,
             "需求人口_方案一": dem_old, "需求人口_微调": dem_new, "需求人口_250m": d250}.items():
    t[k] = t.SQCODE.map(s).fillna(0).astype(int)
t["网格数_方案一"] = t.SQCODE.map(owner_old.value_counts()).fillna(0).astype(int)
t["网格数_微调"] = t.SQCODE.map(owner_new.value_counts()).fillna(0).astype(int)
t.to_csv(OUT / "plan1_hospital_community_summary.csv", index=False, encoding="utf-8-sig")

ch = a[a.changed_vs_plan1]
log = ch[["Tid", "SQCODE_plan1", "SQCODE", "basis"]].copy()
log["原社区"] = log.SQCODE_plan1.map(name)
log["新社区"] = log.SQCODE.map(name)
log["门诊到达人次"] = log.Tid.map(arrive).fillna(0).astype(int)
log["门诊出发人次"] = log.Tid.map(od.groupby("home_tid")["count"].sum()).fillna(0).astype(int)
log["联通人口"] = log.Tid.map(nest.groupby("Tid_500")["联通人"].sum()).fillna(0).astype(int)
log["医院"] = log.Tid.map(hosp.set_index("Tid_500").fac_name)
log.to_csv(OUT / "plan1_hospital_change_log.csv", index=False, encoding="utf-8-sig")

aoi_main = aoi.sort_values("占AOI面积比例", ascending=False).drop_duplicates("aoi_id")
main_ok_old = (aoi_main.Tid_500.map(owner_old) == aoi_main["AOI所在社区SQCODE"])
main_ok_new = (aoi_main.Tid_500.map(owner_new) == aoi_main["AOI所在社区SQCODE"])
share_own = lambda owner: aoi.assign(ok=aoi.Tid_500.map(owner) == aoi["AOI所在社区SQCODE"]).groupby("aoi_id").apply(
    lambda x: x.loc[x.ok, "占AOI面积比例"].sum())
r = lambda s: (s.需求人口_微调 - s.需求人口_250m).abs() / s.需求人口_250m
audit = {
    "min_aoi_share": MIN_AOI_SHARE, "医院网格": len(fixed), "失去全部网格的社区": [name[c] for c in missing], "降低偏差补入次数": moves_b,
    "医院网格明细": [[int(t_), f_, name[c_], round(s_, 3), int(arrive.get(t_, 0))]
                for t_, f_, c_, s_ in zip(hosp.Tid_500, hosp.fac_name, hosp["AOI所在社区SQCODE"], hosp["占AOI面积比例"])],
    "与方案一不同的网格": int(len(ch)), "未能指派的社区": [name[c] for c in unresolved],
    "无网格社区": int((t.网格数_微调 == 0).sum()), "单网格社区": int((t.网格数_微调 == 1).sum()),
    "主网格归医院所在社区的医院": {"方案一": int(main_ok_old.sum()), "微调": int(main_ok_new.sum()), "总数": len(aoi_main)},
    "面域全部不在本社区网格的医院": {"方案一": int((share_own(owner_old) < 0.001).sum()),
                         "微调": int((share_own(owner_new) < 0.001).sum())},
    "流入为0的社区": {"方案一": int((t.流入_方案一 == 0).sum()), "微调": int((t.流入_微调 == 0).sum())},
    "流出为0的社区": {"方案一": int((t.流出_方案一 == 0).sum()), "微调": int((t.流出_微调 == 0).sum())},
    "改归网格的到达人次": int(log.门诊到达人次.sum()), "改归网格的出发人次": int(log.门诊出发人次.sum()),
    "改归网格的联通人口": int(log.联通人口.sum()),
    "需求人口相对差中位数_对250m": {"方案一": round(float(((t.需求人口_方案一 - t.需求人口_250m).abs() / t.需求人口_250m).median()), 4),
                           "微调": round(float(r(t).median()), 4)},
}
(OUT / "audit_plan1_hospital.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
pd.set_option("display.width", 250)
print(log[["Tid", "原社区", "新社区", "basis", "医院", "门诊到达人次", "门诊出发人次", "联通人口"]].to_string(index=False))
