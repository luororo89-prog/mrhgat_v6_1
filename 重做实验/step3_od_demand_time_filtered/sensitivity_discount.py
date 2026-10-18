"""到达需求分配的周围圈层折扣敏感性：第一圈 ×0.5 / ×0.25 / ×0.1（第二圈取其一半）三档结果对比。

先运行 assign_od_destination.py 0.5、0.25、0.1，结果分别在 results/discount_<折扣>/。
对比社区到达人次、社区 OD 与各档核对指标，输出 results/sensitivity/。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "sensitivity"
OUT.mkdir(exist_ok=True)
LEVELS = ["0.5", "0.25", "0.1"]

tabs = {d: pd.read_csv(RES / f"discount_{d}/community_destination_demand.csv", dtype={"SQCODE": str}) for d in LEVELS}
audits = {d: json.loads((RES / f"discount_{d}/audit.json").read_text(encoding="utf-8")) for d in LEVELS}
cods = {d: pd.read_csv(RES / f"discount_{d}/community_od.csv", dtype={"SQCODE_o": str, "SQCODE_d": str})
        .set_index(["SQCODE_o", "SQCODE_d"]).trips for d in LEVELS}

t = tabs[LEVELS[0]][["OBJECTID", "SQCODE", "SQNAME", "located_facilities"]].copy()
for d in LEVELS:
    t[f"arrivals_{d}"] = tabs[d].set_index("SQCODE").arrivals.reindex(t.SQCODE).to_numpy()
cols = [f"arrivals_{d}" for d in LEVELS]
t["arrivals_min"] = t[cols].min(axis=1)
t["arrivals_max"] = t[cols].max(axis=1)
t["max_relative_change"] = np.where(t.arrivals_max > 0, (t.arrivals_max - t.arrivals_min) / t.arrivals_max, 0.0)
t.to_csv(OUT / "community_destination_by_discount.csv", index=False, encoding="utf-8-sig")

pairs = [(LEVELS[i], LEVELS[j]) for i in range(len(LEVELS)) for j in range(i + 1, len(LEVELS))]
od_all = pd.concat(cods, axis=1).fillna(0)
summary = {
    "各档": {d: {"周围圈层人次": audits[d]["到达人次_按候选圈层"],
                "医院AOI网格人次留在医院所在社区": audits[d]["医院AOI网格人次留在医院所在社区"]["本方法"],
                "有设施但到达人次为0的社区": audits[d]["本方法"]["有设施但到达人次为0的社区"],
                "无设施但到达人次大于0的社区": audits[d]["本方法"]["无设施但到达人次大于0的社区"],
                "社区内部人次占比": audits[d]["社区内部人次占比"],
                "社区OD非零对": audits[d]["社区OD非零对"]} for d in LEVELS},
    "社区到达人次Spearman": {f"{a}_vs_{b}": round(float(spearmanr(t[f"arrivals_{a}"], t[f"arrivals_{b}"]).statistic), 4)
                         for a, b in pairs},
    "社区OD人次Spearman（全部非零对）": {f"{a}_vs_{b}": round(float(spearmanr(od_all[a], od_all[b]).statistic), 4)
                                for a, b in pairs},
    "社区到达人次三档最大相对变化": {
        "中位数": round(float(t.loc[t.arrivals_max > 0, "max_relative_change"].median()), 4),
        "超过20%的社区": int((t.max_relative_change > 0.2).sum()),
        "超过50%的社区": int((t.max_relative_change > 0.5).sum()),
    },
    "相对变化最大的社区": t.sort_values("max_relative_change", ascending=False).head(10)[
        ["SQNAME", *cols]].round(0).to_dict("records"),
}
(OUT / "sensitivity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
