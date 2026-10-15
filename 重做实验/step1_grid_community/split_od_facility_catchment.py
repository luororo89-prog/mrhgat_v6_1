"""门诊 OD 归到社区：出发端按 250 m 需求人口拆分，到达端按设施—社区从属关系归属。

出发端 home_tid：沿用 split_od_by_250m.py 的 origin_grid500_community_weights.csv，
  即按 500 m 网格内部 250 m 网格（maup 归属社区）的联通人口比例拆到社区。
到达端 end_tid：设施的到达范围为设施所在网格（医院为 AOI 相交的网格，社康与无 AOI 的医院为点位所在网格）
  加周围一圈网格（0.004° 规则格网的 3×3 邻域）。到达网格按以下顺序找候选设施：
  网格本身有设施时取这些设施，否则取以该网格为周围一圈的设施；候选设施等分该网格的到达人次，
  再按设施所在社区（grid500_facility_link_plan1.csv）汇总到社区。没有候选设施的到达网格不进入结果。
设施取 facility_supply_audit_2018_mobile_users.csv，医院 AOI 取 hospital_aoi_35.shp。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "od_facility"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "input"
GRID500 = SRC / "SZ_grid_500m.shp"
NODE = HERE.parent / "input/node_order.csv"

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
code_to_id = dict(zip(node.SQCODE, node.OBJECTID))
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")
wo = pd.read_csv(RES / "od_250m/origin_grid500_community_weights.csv")
link = pd.read_csv(RES / "facility_grid/grid500_facility_link_plan1.csv", dtype={"设施所在社区SQCODE": str})
plan1 = pd.read_csv(RES / "grid500_nest/plan1_grid500_community.csv", dtype={"SQCODE": str}).set_index("Tid").SQCODE

# 设施的到达范围：本格（ring 0）与周围一圈（ring 1）
xy = gpd.read_file(GRID500, ignore_geometry=True)[["Tid", "x", "y"]]
xy["col"] = ((xy.x - xy.x.min()) / 0.004).round().astype(int)
xy["row"] = ((xy.y - xy.y.min()) / 0.004).round().astype(int)
cell = xy.set_index(["col", "row"]).Tid
pos = xy.set_index("Tid")[["col", "row"]]
f0 = link[["facility_id", "tier", "Tid_500", "设施所在社区SQCODE"]].rename(columns={"Tid_500": "Tid"}).drop_duplicates()
f0 = f0[f0["设施所在社区SQCODE"].notna()]
f0["ring"] = 0
ring1 = []
for dc in (-1, 0, 1):
    for dr in (-1, 0, 1):
        if (dc, dr) == (0, 0):
            continue
        k = list(zip(f0.Tid.map(pos.col) + dc, f0.Tid.map(pos.row) + dr))
        ring1.append(f0.assign(Tid=pd.Series(k, index=f0.index).map(cell), ring=1))
f1 = pd.concat(ring1).dropna(subset=["Tid"]).astype({"Tid": int})
catch = pd.concat([f0, f1]).sort_values("ring").drop_duplicates(["facility_id", "Tid"])

# 到达网格的候选设施：本格有设施取 ring 0，否则取 ring 1；候选等分
end = pd.DataFrame({"end_tid": od.end_tid.unique()})
cand = end.merge(catch.rename(columns={"Tid": "end_tid"}), on="end_tid")
cand = cand[cand.ring == cand.groupby("end_tid").ring.transform("min")]
cand["w"] = 1 / cand.groupby("end_tid").facility_id.transform("size")
wd = cand.groupby(["end_tid", "设施所在社区SQCODE"]).agg(weight=("w", "sum"), ring=("ring", "first"),
                                                     n_fac=("facility_id", "size")).reset_index()
wd["target"] = wd["设施所在社区SQCODE"].map(code_to_id)
wd.to_csv(OUT / "destination_grid500_facility_community_weights.csv", index=False, encoding="utf-8-sig")

flow = od.merge(wo[["home_tid", "source", "weight"]], on="home_tid").rename(columns={"weight": "wo"})
flow = flow.merge(wd[["end_tid", "target", "weight"]], on="end_tid").rename(columns={"weight": "wd"})
flow["trips"] = flow["count"] * flow.wo * flow.wd
long = flow.groupby(["source", "target"]).trips.sum().reset_index()
long.to_csv(OUT / "community_od_long_fraction.csv", index=False, encoding="utf-8-sig")
mat = long.pivot(index="source", columns="target", values="trips").reindex(
    index=node.OBJECTID, columns=node.OBJECTID, fill_value=0).fillna(0)
mat.to_csv(OUT / "community_od_matrix_fraction.csv", index_label="OBJECTID")

# 社区流出流入
t = node[["OBJECTID", "SQCODE", "SQNAME"]].copy()
inner = long[long.source == long.target].set_index("source").trips
t["流出人次"] = t.OBJECTID.map(long.groupby("source").trips.sum()).fillna(0)
t["流入人次"] = t.OBJECTID.map(long.groupby("target").trips.sum()).fillna(0)
t["内部人次"] = t.OBJECTID.map(inner).fillna(0)
fac_cnt = link.drop_duplicates("facility_id").groupby("设施所在社区SQCODE").tier.value_counts().unstack(fill_value=0)
for c in ["TH", "SH", "CHC"]:
    t[c] = t.SQCODE.map(fac_cnt.get(c, pd.Series(dtype=int))).fillna(0).astype(int)
t.round(3).to_csv(OUT / "community_outpatient_flow.csv", index=False, encoding="utf-8-sig")

# 与方案一（到达端按网格归属）比较
tot = int(od["count"].sum())
d_plan1 = flow.end_tid.map(plan1).map(code_to_id)
diff = float(flow.loc[flow.target != d_plan1, "trips"].sum())
multi = wd.groupby("end_tid").target.nunique()
by_ring = wd.drop_duplicates("end_tid").set_index("end_tid").ring
arr = od.groupby("end_tid")["count"].sum()
has_fac = set(t.loc[t[["TH", "SH", "CHC"]].sum(axis=1) > 0, "OBJECTID"])
audit = {
    "门诊人次_原": tot, "进入结果的人次": round(float(long.trips.sum()), 1),
    "进入比例": round(float(long.trips.sum() / tot), 4),
    "到达网格_原": int(len(end)), "到达网格_有候选设施": int(wd.end_tid.nunique()),
    "到达网格_按本格设施": int((by_ring == 0).sum()), "到达网格_按周围一圈设施": int((by_ring == 1).sum()),
    "人次_按本格设施": int(arr[by_ring.index[by_ring == 0]].sum()),
    "人次_按周围一圈设施": int(arr[by_ring.index[by_ring == 1]].sum()),
    "到达网格_候选设施分属多个社区": int((multi > 1).sum()),
    "人次_候选设施分属多个社区": int(arr[multi.index[multi > 1]].sum()),
    "设施所在社区数": len(has_fac),
    "有流入的社区": int((t["流入人次"] > 0).sum()),
    "有流入但无设施的社区": int(((t["流入人次"] > 0) & ~t.OBJECTID.isin(has_fac)).sum()),
    "有流出的社区": int((t["流出人次"] > 0).sum()),
    "社区内部人次占比": round(float(t["内部人次"].sum() / long.trips.sum()), 4),
    "到达端社区与方案一网格归属不同的人次": round(diff, 1),
    "到达端社区与方案一网格归属不同的比例": round(diff / float(long.trips.sum()), 4),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
