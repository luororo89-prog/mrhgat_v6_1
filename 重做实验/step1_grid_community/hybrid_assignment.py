"""250 m 网格混合归属：小社区按网格中心点，大社区按最大重叠面积（maup 规则），并汇总需求人口。

小社区为面积小于 SMALL_AREA 的社区。
网格中心点落在小社区内的，归该小社区。
其余网格在与其重叠的大社区中取重叠面积最大者，网格不与任何大社区重叠时取全部社区中重叠面积最大者。
需求人口为社区所辖 250 m 网格联通人口（字段 联通人）之和。
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "hybrid"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "input"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM = SRC / "深圳社区objectid.shp"
DIST = SRC / "shenzhen_district_4547.shp"
NODE = HERE.parent / "input/node_order.csv"
SMALL_AREA = 0.3e6  # 平方米

cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})
small = set(cs.loc[cs.comm_area_m2 < SMALL_AREA, "SQCODE"])
base = pd.read_csv(RES / "grid250_community_1to1.csv", dtype={"SQCODE": str, "centroid_SQCODE": str})
p = pd.read_csv(RES / "grid250_community_pieces.csv", dtype={"SQCODE": str}).sort_values(
    ["Tid", "piece_m2"], ascending=[True, False])

large_top = p[~p.SQCODE.isin(small)].drop_duplicates("Tid").set_index("Tid").SQCODE
any_top = p.drop_duplicates("Tid").set_index("Tid").SQCODE
a = base[["Tid", "centroid_SQCODE"]].rename(columns={"SQCODE": "SQCODE_maup"}).copy()
a["SQCODE_maup"] = base.SQCODE
by_centroid = a.centroid_SQCODE.isin(small)
a["SQCODE"] = np.where(by_centroid, a.centroid_SQCODE, a.Tid.map(large_top).fillna(a.Tid.map(any_top)))
a["rule"] = np.where(by_centroid, "小社区_中心点", np.where(a.Tid.map(large_top).notna(), "大社区_最大面积", "无大社区重叠_最大面积"))
assert a.SQCODE.notna().all()

pop = gpd.read_file(GRID250)[["Tid", "联通人"]]
a = a.merge(pop, on="Tid", validate="one_to_one")
a.to_csv(OUT / "grid250_community_hybrid.csv", index=False)

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
dem = pd.read_csv(RES / "maup_demand/community_demand_population_maup.csv", dtype={"SQCODE": str}, encoding="utf-8-sig")
g = a.groupby("SQCODE").agg(n_grids_hybrid=("Tid", "size"), 需求人口_联通_混合=("联通人", "sum"))
t = node[["OBJECTID", "SQCODE"]].merge(cs[["SQCODE", "SQNAME", "comm_area_m2"]], on="SQCODE").join(g, on="SQCODE")
t[["n_grids_hybrid", "需求人口_联通_混合"]] = t[["n_grids_hybrid", "需求人口_联通_混合"]].fillna(0).astype(int)
t["小社区"] = t.SQCODE.isin(small)
t = t.merge(dem[["SQCODE", "需求人口_联通_maup", "需求人口_联通_现有供给表"]], on="SQCODE")

# 面积加权人口作参照
pp = p.merge(pop, on="Tid")
pp["w"] = pp.piece_m2 / pp.groupby("Tid").piece_m2.transform("sum")
t["面积加权人口"] = t.SQCODE.map((pp["联通人"] * pp.w).groupby(pp.SQCODE).sum())
for k, c in [("混合", "需求人口_联通_混合"), ("maup", "需求人口_联通_maup"), ("现有", "需求人口_联通_现有供给表")]:
    t[f"误差_{k}"] = (t[c] - t.面积加权人口).abs() / t.面积加权人口
t.to_csv(OUT / "community_demand_population_hybrid.csv", index=False, encoding="utf-8-sig")

audit = {"small_area_m2": SMALL_AREA, "小社区数": len(small), "网格数": len(a),
         "网格规则": a.rule.value_counts().to_dict(),
         "与maup不同的网格": int((a.SQCODE != a.SQCODE_maup).sum()),
         "与中心点不同的网格": int((a.SQCODE != a.centroid_SQCODE).sum()),
         "联通人口合计": int(t.需求人口_联通_混合.sum()), "无网格社区": int((t.n_grids_hybrid == 0).sum()),
         "单网格社区": int((t.n_grids_hybrid == 1).sum()), "需求人口为0的社区": int((t.需求人口_联通_混合 == 0).sum()),
         "需求人口分位数": {str(k): float(v) for k, v in t.需求人口_联通_混合.quantile([0, .1, .25, .5, .75, .9, 1]).items()}}
for name, m in {"小社区": t.小社区, "大社区": ~t.小社区}.items():
    audit[f"{name}_人口偏差中位数"] = {k: round(float(t.loc[m, f"误差_{k}"].median()), 4) for k in ("混合", "maup", "现有")}
    audit[f"{name}_人口偏差均值"] = {k: round(float(t.loc[m, f"误差_{k}"].mean()), 4) for k in ("混合", "maup", "现有")}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))

# 空间分布图
SURFACE, INK, INK2, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
BINS = [0, 1500, 3000, 6000, 12000, 24000, np.inf]
LABELS = ["< 1,500", "1,500 至 3,000", "3,000 至 6,000", "6,000 至 12,000", "12,000 至 24,000", "≥ 24,000"]
shp = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(4547).merge(t, on="SQCODE", validate="one_to_one")
dist = gpd.read_file(DIST).to_crs(4547)
cls = pd.cut(shp.需求人口_联通_混合, BINS, labels=False, right=False).astype(int)
counts = cls.value_counts().reindex(range(len(SEQ)), fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
shp.plot(ax=ax, color=[SEQ[c] for c in cls], edgecolor=SURFACE, linewidth=0.15)
dist.boundary.plot(ax=ax, color=EDGE, linewidth=0.6)
ax.set_axis_off()
ax.set_title("各社区需求人口（联通人口）  小社区按中心点，大社区按最大重叠面积", loc="left", fontsize=13, color=INK, pad=6)
v = shp.需求人口_联通_混合
ax.text(0.68, 0.97, f"合计 {v.sum():,.0f} 人\n中位数 {v.median():,.0f} 人\n小社区为面积小于 {SMALL_AREA / 1e6:g} 平方千米的 {len(small)} 个社区",
        transform=ax.transAxes, va="top", fontsize=9.5, color=INK2, linespacing=1.5)
handles = [Patch(facecolor=c, edgecolor=EDGE, linewidth=0.4, label=f"{l}（{n} 个）") for c, l, n in zip(SEQ, LABELS, counts)]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=9.5, frameon=False,
          labelcolor=INK2, title="社区需求人口（人）", title_fontsize=9.5, alignment="left")
fig.savefig(OUT / "fig_hybrid_demand.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(OUT / "fig_hybrid_demand.png", counts.to_dict())
