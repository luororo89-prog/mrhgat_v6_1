"""500 m 网格一一归属：按其内部 250 m 网格的需求人口比例确定归属社区。

250 m 网格按 maup 规则（最大重叠面积）归属社区，见 grid250_community_1to1.csv。
每个 500 m 网格内部 250 m 网格的联通人口按所属社区汇总，500 m 网格归入需求人口最多的社区，
人口相同时取与 500 m 网格重叠面积较大的社区。
内部 250 m 网格联通人口全为 0 或内部没有 250 m 网格的 500 m 网格，按 maup 规则（最大重叠面积）归属。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_pop"
OUT.mkdir(exist_ok=True)
NODE = HERE.parent / "input/node_order.csv"

nest = pd.read_csv(RES / "od_250m/grid250_in_grid500.csv", dtype={"SQCODE": str})  # Tid_250, Tid_500, 250m maup 社区, 联通人
g5 = pd.read_csv(RES / "grid500_community_1to1.csv", dtype={"SQCODE": str})[["Tid", "SQCODE"]].rename(
    columns={"Tid": "Tid_500", "SQCODE": "SQCODE_area"})
p5 = pd.read_csv(RES / "grid500_community_pieces.csv", dtype={"SQCODE": str})[["Tid", "SQCODE", "piece_m2"]].rename(
    columns={"Tid": "Tid_500"})
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})[["SQCODE", "SQNAME", "comm_area_m2"]]
node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})

pop = nest.groupby(["Tid_500", "SQCODE"])["联通人"].sum().rename("pop").reset_index()
pop = pop.merge(p5, on=["Tid_500", "SQCODE"], how="left").fillna({"piece_m2": 0})
pop = pop[pop["pop"] > 0].sort_values(["Tid_500", "pop", "piece_m2"], ascending=[True, False, False])
pop["pop_share"] = pop["pop"] / pop.groupby("Tid_500")["pop"].transform("sum")
top = pop.drop_duplicates("Tid_500").set_index("Tid_500")

a = g5.copy()
a["SQCODE"] = a.Tid_500.map(top.SQCODE).fillna(a.SQCODE_area)
a["basis"] = np.where(a.Tid_500.isin(top.index), "250m需求人口", "最大重叠面积")
a["pop_share"] = a.Tid_500.map(top.pop_share)
a["grid_pop"] = a.Tid_500.map(nest.groupby("Tid_500")["联通人"].sum()).fillna(0).astype(int)
a = a.merge(node[["SQCODE", "OBJECTID"]], on="SQCODE", how="left")
assert a.OBJECTID.notna().all()
a.to_csv(OUT / "grid500_community_by_population.csv", index=False)

# 250 m 层面的需求人口（参照）与 500 m 一一归属后的需求人口
ref = nest.groupby("SQCODE")["联通人"].sum()
out_pop = nest[~nest.Tid_500.isin(a.Tid_500)]
assert out_pop.empty
n = nest.merge(a[["Tid_500", "SQCODE"]].rename(columns={"SQCODE": "SQ_pop"}), on="Tid_500").merge(
    g5.rename(columns={"SQCODE_area": "SQ_area"}), on="Tid_500")


def summarize(col, tag):
    d = n.groupby(col)["联通人"].sum()
    cnt = a.groupby("SQCODE").size() if col == "SQ_pop" else g5.groupby("SQCODE_area").size()
    s = cs.copy()
    s[f"需求人口_250m"] = s.SQCODE.map(ref).fillna(0).astype(int)
    s[f"n_grid500"] = s.SQCODE.map(cnt).fillna(0).astype(int)
    s[f"需求人口_500m"] = s.SQCODE.map(d).fillna(0).astype(int)
    s["相对差"] = np.where(s.需求人口_250m > 0, (s.需求人口_500m - s.需求人口_250m) / s.需求人口_250m, np.nan)
    moved = int(n.loc[n[col] != n.SQCODE, "联通人"].sum())
    r = s.相对差.abs()
    small = s.comm_area_m2 < 0.3e6
    return s, {
        "无500m网格社区": int((s.n_grid500 == 0).sum()), "其中小社区": int(((s.n_grid500 == 0) & small).sum()),
        "单网格社区": int((s.n_grid500 == 1).sum()),
        "需求人口为0的社区": int((s.需求人口_500m == 0).sum()),
        "与250m归属不一致的人口": moved, "占比": round(moved / float(n["联通人"].sum()), 4),
        "需求人口相对差中位数": round(float(r.median()), 4), "相对差超过20%": int((r > 0.2).sum()),
        "相对差超过50%": int((r > 0.5).sum()),
        "小社区相对差中位数": round(float(r[small].median()), 4), "大社区相对差中位数": round(float(r[~small].median()), 4),
    }


s_pop, m_pop = summarize("SQ_pop", "pop")
s_area, m_area = summarize("SQ_area", "area")
s_pop.to_csv(OUT / "community_demand_grid500_by_population.csv", index=False, encoding="utf-8-sig")
audit = {"500m网格": len(a), "归属依据": a.basis.value_counts().to_dict(),
         "与500m最大面积归属不同的网格": int((a.SQCODE != a.SQCODE_area).sum()),
         "需求人口比例中位数": round(float(a.pop_share.median()), 3),
         "最大社区人口占比低于50%的网格": int((a.pop_share < 0.5).sum()),
         "按需求人口": m_pop, "按最大面积": m_area}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
z = s_pop[s_pop.n_grid500 == 0]
print(z.groupby(pd.cut(z.comm_area_m2 / 1e6, [0, 0.15, 0.3, 0.6, 99], right=False)).size().to_dict())
print(z.sort_values("需求人口_250m", ascending=False)[["SQNAME", "comm_area_m2", "需求人口_250m"]].head(8).to_string(index=False))
print("zero-grid pop", int(z.需求人口_250m.sum()))
