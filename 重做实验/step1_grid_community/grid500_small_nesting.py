"""500 m 网格一一归属：大社区按 maup，小社区按 250 m 与 500 m 的嵌套关系处理，两种方案。

小社区为面积小于 SMALL_AREA 的社区。250 m 网格按 maup 规则归属社区（grid250_community_1to1.csv）。
每个 500 m 网格内部的 250 m 网格按所属社区汇总联通人口与网格数，作为嵌套关系。

方案一 小社区认领 500 m 网格，保留全部社区
  1. 先保证每个社区各有一个 500 m 网格。候选为与社区重叠的 500 m 网格，
     以社区经嵌套关系在候选格内的联通人口为得分（其次 250 m 网格数、重叠面积），
     用指派算法为 666 个社区各配一个互不相同的网格，使总得分最大。
  2. 其余 500 m 网格中，内部人口最多的社区（人口相同时比 250 m 网格数）若为小社区，该格归此小社区。
  3. 剩下的 500 m 网格在与其重叠的大社区中取重叠面积最大者，不与任何大社区重叠时取全部社区中重叠面积最大者。

方案二 小社区并入大社区，节点数减少
  1. 500 m 网格全部按 maup 规则（最大重叠面积）归属。
  2. 没有 500 m 网格的小社区，并入其人口最多（其次 250 m 网格数、重叠面积）的那个 500 m 网格所属的社区。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "grid500_nest"
OUT.mkdir(exist_ok=True)
NODE = HERE.parent / "input/node_order.csv"
SMALL_AREA = 0.3e6

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})[["SQCODE", "SQNAME", "comm_area_m2"]]
small = set(cs.loc[cs.comm_area_m2 < SMALL_AREA, "SQCODE"])
name = cs.set_index("SQCODE").SQNAME
nest = pd.read_csv(RES / "od_250m/grid250_in_grid500.csv", dtype={"SQCODE": str})
g5 = pd.read_csv(RES / "grid500_community_1to1.csv", dtype={"SQCODE": str})[["Tid", "SQCODE"]].rename(
    columns={"SQCODE": "SQCODE_maup"})
p5 = pd.read_csv(RES / "grid500_community_pieces.csv", dtype={"SQCODE": str})[["Tid", "SQCODE", "piece_m2"]]

# 嵌套关系：每个 500 m 网格内各社区的人口、250 m 网格数、重叠面积
inner = nest.groupby(["Tid_500", "SQCODE"]).agg(pop=("联通人", "sum"), n250=("Tid_250", "size")).reset_index()
inner = inner.merge(p5.rename(columns={"Tid": "Tid_500"}), on=["Tid_500", "SQCODE"], how="left").fillna({"piece_m2": 0})
inner = inner.sort_values(["Tid_500", "pop", "n250", "piece_m2"], ascending=[True, False, False, False])
ref = nest.groupby("SQCODE")["联通人"].sum()  # 250 m 层面的需求人口

# ---------- 方案一 ----------
# 1. 每个社区一个锚定网格（指派问题）
codes = cs.SQCODE.tolist()
cpos = {c: i for i, c in enumerate(codes)}
cand = p5.rename(columns={"Tid": "Tid_500"}).merge(
    inner[["Tid_500", "SQCODE", "pop", "n250"]], on=["Tid_500", "SQCODE"], how="left").fillna({"pop": 0, "n250": 0})
tids = np.sort(cand.Tid_500.unique())
tpos = {t: j for j, t in enumerate(tids)}
BIG = 1e12
cost = np.full((len(codes), len(tids)), BIG)
score = cand["pop"] + 1e-3 * cand.n250 + 1e-9 * cand.piece_m2
cost[cand.SQCODE.map(cpos).to_numpy(), cand.Tid_500.map(tpos).to_numpy()] = -score.to_numpy()
rows, cols = linear_sum_assignment(cost)
anchor_ok = cost[rows, cols] < BIG
assign = {int(tids[j]): (codes[i], "锚定_小社区" if codes[i] in small else "锚定_大社区")
          for i, j, ok in zip(rows, cols, anchor_ok) if ok}
unresolved = [codes[i] for i, ok in zip(rows, anchor_ok) if not ok]
# 2. 其余网格中小社区人口最多的归小社区
dominant = inner.drop_duplicates("Tid_500")
for t, c in zip(dominant.Tid_500, dominant.SQCODE):
    if c in small and int(t) not in assign:
        assign[int(t)] = (c, "小社区_人口最多")
large_top = p5[~p5.SQCODE.isin(small)].sort_values(["Tid", "piece_m2"], ascending=[True, False]).drop_duplicates("Tid").set_index("Tid").SQCODE
a1 = g5.copy()
a1["SQCODE"] = [assign[t][0] if t in assign else large_top.get(t, m) for t, m in zip(a1.Tid, a1.SQCODE_maup)]
a1["basis"] = [assign[t][1] if t in assign else ("大社区_maup" if t in large_top.index else "maup") for t in a1.Tid]
a1 = a1.merge(node[["SQCODE", "OBJECTID"]], on="SQCODE", how="left")
a1.to_csv(OUT / "plan1_grid500_community.csv", index=False)


def evaluate(cell_owner, member_to_node):
    """cell_owner: 500 m 网格 -> 节点 SQCODE；member_to_node: 社区 SQCODE -> 节点 SQCODE。"""
    n = nest.assign(node_250=nest.SQCODE.map(member_to_node), node_500=nest.Tid_500.map(cell_owner))
    d500 = n.groupby("node_500")["联通人"].sum()
    d250 = n.groupby("node_250")["联通人"].sum()
    nodes = sorted(set(member_to_node.values()))
    s = pd.DataFrame({"node": nodes})
    s["n_grid500"] = s.node.map(pd.Series(cell_owner).value_counts()).fillna(0).astype(int)
    s["需求人口_250m"] = s.node.map(d250).fillna(0).astype(int)
    s["需求人口_500m"] = s.node.map(d500).fillna(0).astype(int)
    s["相对差"] = np.where(s.需求人口_250m > 0, (s.需求人口_500m - s.需求人口_250m) / s.需求人口_250m, np.nan)
    sm = s.node.isin(small)
    r = s.相对差.abs()
    moved = int(n.loc[n.node_250 != n.node_500, "联通人"].sum())
    return s, {"节点数": len(s), "无500m网格节点": int((s.n_grid500 == 0).sum()), "单网格节点": int((s.n_grid500 == 1).sum()),
               "与250m归属不一致的人口": moved, "占比": round(moved / float(n["联通人"].sum()), 4),
               "需求人口相对差中位数": round(float(r.median()), 4), "相对差超过20%": int((r > 0.2).sum()),
               "相对差超过50%": int((r > 0.5).sum()),
               "小社区节点相对差中位数": round(float(r[sm].median()), 4) if sm.any() else None,
               "大社区节点相对差中位数": round(float(r[~sm].median()), 4)}


identity = dict(zip(cs.SQCODE, cs.SQCODE))
s1, m1 = evaluate(dict(zip(a1.Tid, a1.SQCODE)), identity)
s1 = s1.rename(columns={"node": "SQCODE"}).merge(cs, on="SQCODE")
s1.to_csv(OUT / "plan1_community_demand.csv", index=False, encoding="utf-8-sig")
lost_large = s1[(~s1.SQCODE.isin(small))]
base_cnt = g5.groupby("SQCODE_maup").size()
m1.update({"归属依据": a1.basis.value_counts().to_dict(), "与500m_maup不同的网格": int((a1.SQCODE != a1.SQCODE_maup).sum()),
           "未能认领的小社区": [name[c] for c in unresolved],
           "网格被小社区拿走的大社区": int((lost_large.n_grid500 < lost_large.SQCODE.map(base_cnt).fillna(0)).sum()),
           "因此失去全部网格的大社区": int(((lost_large.n_grid500 == 0)).sum())})

# ---------- 方案二 ----------
owner = dict(zip(g5.Tid, g5.SQCODE_maup))
has_cell = set(g5.SQCODE_maup)
no_cell = sorted(set(cs.SQCODE) - has_cell)
merge_to, merge_rows = {}, []
for c in no_cell:
    best = inner[inner.SQCODE == c].sort_values(["pop", "n250", "piece_m2"], ascending=False).iloc[0]
    host = owner[int(best.Tid_500)]
    merge_to[c] = host
    merge_rows.append({"SQCODE": c, "SQNAME": name[c], "小社区": c in small, "需求人口_250m": int(ref.get(c, 0)),
                       "所在500m网格": int(best.Tid_500), "并入SQCODE": host, "并入SQNAME": name[host],
                       "并入社区为小社区": host in small})
member = {c: merge_to.get(c, c) for c in cs.SQCODE}
s2, m2 = evaluate(owner, member)
s2 = s2.rename(columns={"node": "SQCODE"}).merge(cs, on="SQCODE")
s2["成员社区"] = s2.SQCODE.map(pd.Series(list(member.keys()), index=list(member.values())).groupby(level=0)
                           .apply(lambda x: "、".join(name[v] for v in x)))
mr = pd.DataFrame(merge_rows)
mr.to_csv(OUT / "plan2_merge_table.csv", index=False, encoding="utf-8-sig")
s2.to_csv(OUT / "plan2_node_demand.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"SQCODE": list(member), "node_SQCODE": list(member.values())}).to_csv(
    OUT / "plan2_community_to_node.csv", index=False)
m2.update({"并入的社区": len(mr), "接收并入的社区": int(mr.并入SQCODE.nunique()),
           "接收方本身是小社区": int(mr.并入社区为小社区.sum()),
           "并入社区合计需求人口": int(mr.需求人口_250m.sum())})

audit = {"small_area_m2": SMALL_AREA, "小社区数": len(small), "方案一_小社区认领": m1, "方案二_小社区并入": m2}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
print(mr.sort_values("需求人口_250m", ascending=False)[["SQNAME", "需求人口_250m", "并入SQNAME"]].head(12).to_string(index=False))
