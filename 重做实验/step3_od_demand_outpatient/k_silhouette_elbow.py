"""按轮廓系数和肘部法选供给分类数 k。

特征空间同 fit_node_types.py 的 composite 模式（九项指标 Yeo-Johnson 变换、标准化后投影到平行分析主成分），
KMeans 设置同 ruler.select_k（n_init = 50，random_state = 0）。k = 1 至 10 计算组内平方和（SSE），k = 2 至 10 计算轮廓系数；
肘部取 SSE 曲线（k 与 SSE 归一化到 0 至 1）离首尾连线最远的 k。
输出 results/node_typing/k_silhouette_elbow.csv 与 fig_k_silhouette_elbow.png。
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, "/Users/luoxuan/Documents/mrhgat_v6_1_destpred/node_typing/code")
from fit_node_types import COLUMNS  # noqa: E402
from ruler import fit_ruler, standardize  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/node_typing"
OUT.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(HERE / "results/supply_indicators_revised/community_supply_indicators_2018_mobile_users.csv",
                encoding="utf-8-sig").sort_values("community_id")
raw = s[COLUMNS].to_numpy(float)
r, rep = fit_ruler(raw, COLUMNS, None, force_mode="composite", k_override=3, pa_reps=1000, boots=200,
                   transform_mode="yeo-johnson", log_form=[])
F = standardize(raw, r) @ r["U"]

rows = []
for k in range(1, 11):
    km = KMeans(k, n_init=50, random_state=0).fit(F)
    rows.append({"k": k, "SSE": km.inertia_, "silhouette": silhouette_score(F, km.labels_) if k > 1 else np.nan})
t = pd.DataFrame(rows)
x = (t.k - t.k.min()) / (t.k.max() - t.k.min())
y = (t.SSE - t.SSE.min()) / (t.SSE.max() - t.SSE.min())
t["elbow_distance"] = np.abs(x + y - 1) / np.sqrt(2)  # 到 (0,1)-(1,0) 连线的距离
t["SSE_drop_pct"] = (-t.SSE.diff() / t.SSE.shift() * 100).round(1)
k_elbow = int(t.loc[t.elbow_distance.idxmax(), "k"])
k_sil = int(t.loc[t.silhouette.idxmax(), "k"])
t.to_csv(OUT / "k_silhouette_elbow.csv", index=False, encoding="utf-8-sig")

INK, MUTED, GRID, SERIES, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), facecolor=SURF)
for ax, col, kk, title, ylab in [(axes[0], "silhouette", k_sil, "Silhouette coefficient", "Silhouette"),
                                 (axes[1], "SSE", k_elbow, "Within-cluster SSE (elbow)", "SSE")]:
    d = t.dropna(subset=[col])
    ax.set_facecolor(SURF)
    ax.plot(d.k, d[col], color=SERIES, lw=2, marker="o", ms=5, zorder=3)
    sel = d[d.k == kk]
    ax.plot(sel.k, sel[col], marker="o", ms=9, mfc="none", mec=INK, mew=1.2, zorder=4)
    ax.annotate(f"k = {kk}", (kk, float(sel[col].iloc[0])), xytext=(9, -3) if col == "silhouette" else (8, 16), textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.set_xlabel("Number of supply types k")
    ax.set_ylabel(ylab)
    ax.set_xticks(range(1 if col == "SSE" else 2, 11))
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_k_silhouette_elbow.png", dpi=300, facecolor=SURF)
print(t.round(4).to_string(index=False))
print("轮廓系数最大 k", k_sil, " 肘部 k", k_elbow)
