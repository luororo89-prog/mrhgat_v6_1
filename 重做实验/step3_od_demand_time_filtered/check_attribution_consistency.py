"""250 m 网格社区归属的一致性检查（本目录 results，命令行参数为折扣档，默认 0.5；归属规则三档相同）。

1. 同一 250 m 网格出发（O）与到达（D）的社区归属：O 取 grid250_origin_allocation，D 取 grid250_destination_allocation，
   比较各社区占比；D 另与该网格的需求人口社区归属（facility_grid250.py 修订后的 grid250_demand_allocation_revised）比较。
2. 到达 250 m 网格与设施：D 是否只落在有设施的 250 m 网格，设施所在 250 m 网格是否都得到到达人次。
3. 设施与社区归属：D 所归社区是否为该网格内设施的归属社区；医院归属与供给评价 facility_community.csv 是否一致；
   设施归属社区与设施点位（社康，取 facility_grid250.py 的新点位所在社区）或 AOI（医院）所在社区是否一致。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
D = sys.argv[1] if len(sys.argv) > 1 else "0.5"
OUT = RES / "check_attribution_consistency"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
STEP1 = HERE.parent / "step1_grid_community/results"
STEP2 = HERE.parent / "step2_supply/results"

sqname = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("SQCODE").SQNAME
o = pd.read_csv(RES / "grid500_origin_250m/grid250_origin_allocation.csv", encoding="utf-8-sig",
                dtype={"SQCODE": str}).dropna(subset=["Tid_250"]).astype({"Tid_250": int})
d = pd.read_csv(RES / f"discount_{D}/grid250_destination_allocation.csv", encoding="utf-8-sig",
                dtype={"SQCODE": str}).dropna(subset=["Tid_250"]).astype({"Tid_250": int})
fp = pd.read_csv(RES / f"discount_{D}/facility_grid250_attraction.csv", encoding="utf-8-sig",
                 dtype={"SQCODE": str, "SQCODE_located": str})
fc = pd.read_csv(STEP2 / "facility_community.csv", dtype={"SQCODE": str, "SQCODE_point": str})
dem = pd.read_csv(RES / "facility_grid250/grid250_demand_allocation_revised.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
pc250 = set(map(tuple, pd.read_csv(STEP1 / "grid250_community_pieces.csv", dtype={"SQCODE": str})[["Tid", "SQCODE"]]
                .to_numpy()))
tot_d = float(d.D_trips.sum())


def r(x, k=4):
    return round(float(x), k)


def shares(df, val):
    s = df.groupby(["Tid_250", "SQCODE"])[val].sum()
    s = s[s > 0]
    return s / s.groupby(level=0).transform("sum")


# ---------- 1. 同一 250 m 网格 O 与 D 的社区归属 ----------
so, sd = shares(o, "O_trips"), shares(d, "D_trips")
sp = shares(dem.rename(columns={"Tid": "Tid_250"}), "weight")  # 需求人口社区归属（含人口为 0 的 maup 网格）
cells_o, cells_d = set(so.index.get_level_values(0)), set(sd.index.get_level_values(0))
both = sorted(cells_o & cells_d)
dcell = d.groupby("Tid_250").D_trips.sum()


def compare(ref, ref_name):
    cells = sorted(set(ref.index.get_level_values(0)) & cells_d)
    j = pd.concat([ref.rename("ref"), sd.rename("d")], axis=1).fillna(0)
    j = j[j.index.get_level_values(0).isin(cells)]
    ov = j.min(axis=1).groupby(level=0).sum()  # 两边社区占比重合度，1 为完全一致
    set_eq = j.groupby(level=0).apply(lambda g: set(g.index[g.ref > 0].get_level_values(1)) ==
                                      set(g.index[g.d > 0].get_level_values(1)))
    main_eq = (j.ref.groupby(level=0).idxmax().str[1] == j.d.groupby(level=0).idxmax().str[1])
    w = dcell.reindex(cells)
    d_not_in_ref = j[(j.d > 0) & (j.ref == 0)]
    trips_not_in_ref = float((d_not_in_ref.d * dcell.reindex(d_not_in_ref.index.get_level_values(0)).to_numpy()).sum())
    return {f"同时有{ref_name}与D的250m网格": len(cells),
            "社区集合相同的网格": int(set_eq.sum()), "主社区相同的网格": int(main_eq.sum()),
            "社区占比完全一致的网格": int((ov > 1 - 1e-6).sum()),
            "社区占比重合度_D人次加权均值": r(np.average(ov, weights=w)),
            f"D分给{ref_name}未归属社区的人次": r(trips_not_in_ref, 1),
            f"D分给{ref_name}未归属社区的人次占全部D": r(trips_not_in_ref / tot_d)}, j, ov


res1, j1, ov1 = compare(so, "O")
res1p, j1p, ovp = compare(sp, "需求人口")
audit = {"折扣档": D, "1_同一250m网格O与D社区归属": {
    "有O的250m网格": len(cells_o), "有D的250m网格": len(cells_d), "D网格中没有O的网格": len(cells_d - cells_o),
    "D网格中没有O的网格D人次占比": r(dcell.reindex(sorted(cells_d - cells_o)).sum() / tot_d),
    "O与D比较": res1, "D与需求人口社区归属比较（覆盖全部D网格）": res1p}}
bad = j1p[(j1p.d > 0) & (j1p.ref == 0)].reset_index()
bad.columns = ["Tid_250", "SQCODE_D", "需求人口占比", "D占比"]
bad["D人次"] = bad.D占比 * bad.Tid_250.map(dcell)
pop_main = sp.groupby(level=0).idxmax().str[1]
bad["需求人口主社区"] = bad.Tid_250.map(pop_main).map(sqname)
bad["D社区"] = bad.SQCODE_D.map(sqname)
fac_in = fp.groupby("Tid_250").apply(lambda g: "、".join(sorted(set(g.facility_name.str.replace("深圳市", "")))))
bad["网格内设施"] = bad.Tid_250.map(fac_in)
bad["D社区与网格相交"] = [(t, s) in pc250 for t, s in zip(bad.Tid_250, bad.SQCODE_D)]
bad.sort_values("D人次", ascending=False).to_csv(OUT / f"d_community_not_in_demand_community_d{D}.csv", index=False,
                                               encoding="utf-8-sig")

# ---------- 2. 到达 250 m 网格与设施 ----------
fac_cells = set(fp.Tid_250.dropna().astype(int))
no_fac = d[~d.Tid_250.isin(fac_cells)]
fac_no_d = fp[~fp.Tid_250.isin(cells_d)]
audit["2_到达250m网格与设施"] = {
    "D落在无设施250m网格的人次": r(no_fac.D_trips.sum(), 1),
    "有设施的250m网格": len(fac_cells), "其中得到D的": len(fac_cells & cells_d),
    "未得到D的设施250m网格": len(fac_cells - cells_d),
    "未得到D的设施": {k: int(v) for k, v in fac_no_d.drop_duplicates("facility_id").groupby("tier").size().items()},
    "部分AOI网格未得到D的医院": int(fac_no_d[fac_no_d.tier != "CHC"].facility_id.nunique()),
    "全部网格都未得到D的设施": {k: int(v) for k, v in fp[~fp.facility_id.isin(fp[fp.Tid_250.isin(cells_d)].facility_id)]
                         .drop_duplicates("facility_id").groupby("tier").size().items()},
}

# ---------- 3. 设施与社区归属 ----------
fac_comm = set(map(tuple, fp[["Tid_250", "SQCODE"]].dropna().to_numpy()))
d_not_fac = d[[(t, s) not in fac_comm for t, s in zip(d.Tid_250, d.SQCODE)]]
f1 = fp.drop_duplicates("facility_id").merge(fc[["facility_id", "SQCODE", "SQCODE_point", "rule", "area_share"]],
                                             on="facility_id", how="left", suffixes=("", "_supply"))
# 社康点位所在社区取 facility_grid250.py 的新点位结果（SQCODE_located）
f1.loc[f1.tier == "CHC", "SQCODE_point"] = f1.loc[f1.tier == "CHC", "SQCODE_located"]
chc = f1[f1.tier == "CHC"]
hosp = f1[f1.tier != "CHC"]
chc_diff = chc[chc.SQCODE != chc.SQCODE_point]
d_fac = d.merge(fp[["Tid_250", "SQCODE", "facility_id", "tier"]].drop_duplicates(["Tid_250", "SQCODE", "facility_id"]),
                on=["Tid_250", "SQCODE"])
d_fac["n"] = d_fac.groupby(["end_tid", "Tid_250", "SQCODE"]).facility_id.transform("size")
d_fac["trips"] = d_fac.D_trips / d_fac.n  # 同一网格同一社区多个设施时平分，只用于统计
hosp_cells = fp[fp.tier != "CHC"][["Tid_250", "SQCODE", "facility_name"]]
hosp_cells = hosp_cells.assign(touch=[(t, s) in pc250 for t, s in zip(hosp_cells.Tid_250, hosp_cells.SQCODE)])
d_h = d.merge(hosp_cells.drop_duplicates(["Tid_250", "SQCODE"]), on=["Tid_250", "SQCODE"])
audit["3_设施与社区归属"] = {
    "D所归社区不是网格内设施归属社区的人次": r(d_not_fac.D_trips.sum(), 1),
    "医院归属与供给评价facility_community不同": int((hosp.SQCODE != hosp.SQCODE_supply).sum()),
    "社康": int(len(chc)), "社康归属社区与点位所在社区不同": int(len(chc_diff)),
    "这些社康所在网格的D人次占全部D": r(d_fac[d_fac.facility_id.isin(chc_diff.facility_id)].trips.sum() / tot_d),
    "医院": int(len(hosp)), "医院AOI在归属社区内的面积占比_中位数": r(hosp.area_share.median(), 3),
    "医院AOI在归属社区内的面积占比_最小": r(hosp.area_share.min(), 3),
    "医院AOI网格与归属社区不相交的D人次": r(d_h.loc[~d_h.touch, "D_trips"].sum(), 1),
    "医院AOI网格D人次": r(d_h.D_trips.sum(), 1),
}
chc_diff.assign(归属社区=chc_diff.SQCODE.map(sqname), 点位所在社区=chc_diff.SQCODE_point.map(sqname),
                D人次=chc_diff.facility_id.map(d_fac.groupby("facility_id").trips.sum()).fillna(0))[
    ["facility_id", "facility_name", "Tid_250", "归属社区", "点位所在社区", "rule", "D人次"]].sort_values(
    "D人次", ascending=False).to_csv(OUT / f"chc_community_vs_point_d{D}.csv", index=False, encoding="utf-8-sig")

(OUT / f"audit_d{D}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2, default=float))
