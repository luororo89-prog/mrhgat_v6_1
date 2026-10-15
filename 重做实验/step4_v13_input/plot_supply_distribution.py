"""新供给指标与供给类型的分布（版式参照 3node_typing/code/plot_node_types.py 的 a 图）。

(a) 九项供给指标按类型的累计分布：行 RA、SA（每万人）、TC，列 CHC、SH、TH；
(b) 各类型社区数与占比（4 类与 3 类）。
供给取 step2_supply 联通人口版社区供给指标，类型取 node_typing（k=4）与 node_typing_k3（k=3）。
输出 node_typing/fig_supply_distribution.{png,pdf} 与 supply_by_type_summary.csv。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
SUPPLY = HERE.parent / "step2_supply/results/supply_indicators/community_supply_indicators_2018_mobile_users.csv"
TYPE_EN = {"稀缺型（三甲低·二级低·社康低）": "Scarcity", "社康主导型（三甲低·二级低·社康高）": "CHC-based",
           "中等型（三甲低·二级低·社康中）": "Moderate", "三甲二级型（三甲高·二级高·社康中）": "SH-based",
           "三甲社康型（三甲高·二级低·社康高）": "Abundance"}
ORDER = ["Scarcity", "Moderate", "CHC-based", "SH-based", "Abundance"]
COLOR = {"Scarcity": "#3C78B4", "Moderate": "#7FB3AE", "CHC-based": "#5AA84F", "SH-based": "#F28E2B",
         "Abundance": "#D9473F"}
TIERS = ["CHC", "SH", "TH"]
ROWS = [("RA", "RA_{}"), ("SA", "SA_{}_per_10000"), ("TC", "TC_{}")]
MM = 1 / 25.4

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "font.size": 7})
s = pd.read_csv(SUPPLY)
t4 = pd.read_csv(HERE / "node_typing/node_types.csv", encoding="utf-8-sig")
t3 = pd.read_csv(HERE / "node_typing_k3/node_types.csv", encoding="utf-8-sig")
assert (s.community_id.to_numpy() == t4.community_id.to_numpy()).all()
typ4 = t4.type_name.map(TYPE_EN).to_numpy()
typ3 = t3.type_name.map(TYPE_EN).to_numpy()
en4 = [t for t in ORDER if t in set(typ4)]
en3 = [t for t in ORDER if t in set(typ3)]

fig = plt.figure(figsize=(183 * MM, 150 * MM))
gs = fig.add_gridspec(3, 3, left=0.07, right=0.98, top=0.95, bottom=0.36, hspace=0.6, wspace=0.12)
fig.text(0.01, 0.985, "a", fontweight="bold", fontsize=8, va="top")
fig.text(0.01, 0.29, "b", fontweight="bold", fontsize=8, va="top")
for r, (key, col) in enumerate(ROWS):
    for c, tier in enumerate(TIERS):
        ax = fig.add_subplot(gs[r, c])
        v_all = s[col.format(tier)].to_numpy()
        for t in en4:
            v = np.sort(v_all[typ4 == t])
            ax.step(np.r_[v[0], v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where="post", color=COLOR[t], lw=0.9,
                    label=t)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        if c:
            ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(direction="in", labelsize=6.5, length=3, width=0.6)
        ax.set_title(f"{key}_{tier}  (zero: {np.mean(v_all == 0):.0%})", fontsize=7.5, fontweight="bold", pad=3)
        if r == 0 and c == 0:
            ax.legend(fontsize=6, frameon=False, loc="lower right", handlelength=1.2)
fig.text(0.025, 0.655, "Cumulative probability", rotation=90, ha="center", va="center", fontsize=7.5)

for i, (typ, en, k) in enumerate([(typ4, en4, 4), (typ3, en3, 3)]):
    ax = fig.add_axes([0.11 + i * 0.47, 0.05, 0.38, 0.2])
    n = pd.Series(typ).value_counts().reindex(en)
    ax.barh(range(len(en)), n.to_numpy(), color=[COLOR[t] for t in en], height=0.65)
    for j, (t, v) in enumerate(n.items()):
        ax.text(v + 6, j, f"{v} ({v / n.sum():.1%})", va="center", fontsize=6.5)
    ax.set_yticks(range(len(en)))
    ax.set_yticklabels(en, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, n.max() * 1.35)
    ax.set_title(f"Supply configuration types, k = {k}", fontsize=7.5, fontweight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6.5)
fig.savefig(HERE / "node_typing/fig_supply_distribution.png", dpi=400)
fig.savefig(HERE / "node_typing/fig_supply_distribution.pdf")

cols = [c.format(t) for _, c in ROWS for t in TIERS]
rows = []
for name, typ in [("全部", np.array(["全部"] * len(s)))] + [(t, None) for t in en4]:
    m = np.ones(len(s), bool) if name == "全部" else typ4 == name
    d = s.loc[m, cols]
    rows.append({"类型": name, "社区数": int(m.sum()), **{f"{c}_中位数": round(float(d[c].median()), 3) for c in cols},
                 **{f"{c}_为0比例": round(float((d[c] == 0).mean()), 3) for c in cols}})
out = pd.DataFrame(rows)
out.to_csv(HERE / "node_typing/supply_by_type_summary.csv", index=False, encoding="utf-8-sig")
print(out[["类型", "社区数"] + [f"{c}_中位数" for c in cols]].to_string(index=False))
print(out[["类型"] + [f"{c}_为0比例" for c in ["RA_CHC", "RA_SH", "RA_TH"]]].to_string(index=False))
