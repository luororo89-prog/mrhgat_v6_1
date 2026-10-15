"""方案一（小社区认领 500 m 网格）归属下，门诊 OD 的社区流出与流入分布。

门诊 OD 取自 GitHub 仓库 mrhgat_v6_1 分支 codex/upload-medical-outpatient-od-20261015 的
outputs/manuscript_revision_20261007/_source/input_sources/26上实验/medical_identification_validation/results/outpatient_grid_OD_home_tid.csv，
字段 home_tid、end_tid、count 均为 500 m 网格编号与人次。
流出为居住网格（home_tid）所属社区的出行人次，流入为到达网格（end_tid）所属社区的到达人次，均含社区内部出行。
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
OUT = RES / "outpatient_od"
SRC = HERE.parent / "input"
COMM = SRC / "深圳社区objectid.shp"
DIST = SRC / "shenzhen_district_4547.shp"
NODE = HERE.parent / "input/node_order.csv"

owner = pd.read_csv(RES / "grid500_nest/plan1_grid500_community.csv", dtype={"SQCODE": str}).set_index("Tid").SQCODE
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")
od["o"] = od.home_tid.map(owner)
od["d"] = od.end_tid.map(owner)
assert od[["o", "d"]].notna().all().all()

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
cs = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str})[["SQCODE", "SQNAME"]]
t = node[["OBJECTID", "SQCODE"]].merge(cs, on="SQCODE")
inner = od[od.o == od.d]
t["流出人次"] = t.SQCODE.map(od.groupby("o")["count"].sum()).fillna(0).astype(int)
t["流入人次"] = t.SQCODE.map(od.groupby("d")["count"].sum()).fillna(0).astype(int)
t["内部人次"] = t.SQCODE.map(inner.groupby("o")["count"].sum()).fillna(0).astype(int)
t["流出_不含内部"] = t.流出人次 - t.内部人次
t["流入_不含内部"] = t.流入人次 - t.内部人次
t["到达网格数"] = t.SQCODE.map(od.drop_duplicates("end_tid").groupby("d").size()).fillna(0).astype(int)
t.to_csv(OUT / "community_outpatient_flow_plan1.csv", index=False, encoding="utf-8-sig")
mat = od.groupby(["o", "d"])["count"].sum().unstack(fill_value=0).reindex(index=node.SQCODE, columns=node.SQCODE, fill_value=0)
mat.index = node.OBJECTID.to_numpy()
mat.columns = node.OBJECTID.to_numpy()
mat.to_csv(OUT / "community_od_matrix_plan1.csv", index_label="OBJECTID")

q = lambda v: {str(k): float(x) for k, x in v.quantile([0, .1, .25, .5, .75, .9, 1]).items()}
audit = {"OD记录": len(od), "总人次": int(od["count"].sum()), "社区内部人次": int(inner["count"].sum()),
         "内部占比": round(float(inner["count"].sum() / od["count"].sum()), 4),
         "到达网格数": int(od.end_tid.nunique()), "居住网格数": int(od.home_tid.nunique()),
         "流出为0的社区": t.loc[t.流出人次 == 0, "SQNAME"].tolist(), "流入为0的社区数": int((t.流入人次 == 0).sum()),
         "流出分位数": q(t.流出人次), "流入分位数": q(t.流入人次),
         "流出最多": t.nlargest(5, "流出人次")[["SQNAME", "流出人次"]].values.tolist(),
         "流入最多": t.nlargest(5, "流入人次")[["SQNAME", "流入人次"]].values.tolist()}
(OUT / "audit_plan1.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))

# ---------- 图 ----------
SURFACE, INK, INK2, EDGE, ZERO = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
ORANGE = ["#fde0d3", "#f9bfa6", "#f39b78", "#eb6834", "#c4521f", "#8f3a14"]
BINS = [1, 500, 1000, 2000, 4000, 8000, np.inf]
LABELS = ["1 至 500", "500 至 1,000", "1,000 至 2,000", "2,000 至 4,000", "4,000 至 8,000", "≥ 8,000"]
shp = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(4547).merge(t, on="SQCODE", validate="one_to_one")
dist = gpd.read_file(DIST).to_crs(4547)

fig, axes = plt.subplots(1, 2, figsize=(16, 7.0), facecolor=SURFACE)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.18, wspace=0.03)
for ax, col, ramp, title, ltitle in [
        (axes[0], "流出人次", BLUE, "a  社区门诊流出（居住地所在社区）", "流出人次"),
        (axes[1], "流入人次", ORANGE, "b  社区门诊流入（就诊地所在社区）", "流入人次")]:
    v = shp[col]
    cls = pd.cut(v, BINS, labels=False, right=False)
    colors = [ZERO if pd.isna(c) else ramp[int(c)] for c in cls]
    counts = cls.value_counts().reindex(range(len(ramp)), fill_value=0)
    ax.set_facecolor(SURFACE)
    shp.plot(ax=ax, color=colors, edgecolor=SURFACE, linewidth=0.15)
    dist.boundary.plot(ax=ax, color=EDGE, linewidth=0.6)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=6)
    ax.text(0.68, 0.97, f"合计 {v.sum():,.0f} 人次\n中位数 {v.median():,.0f} 人次\n含社区内部 {audit['社区内部人次']:,} 人次",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK2, linespacing=1.5)
    handles = [Patch(facecolor=ZERO, edgecolor=EDGE, linewidth=0.4, label=f"0（{int((v == 0).sum())} 个）")]
    handles += [Patch(facecolor=c, edgecolor=EDGE, linewidth=0.4, label=f"{l}（{n} 个）")
                for c, l, n in zip(ramp, LABELS, counts)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=9.5, frameon=False,
              labelcolor=INK2, title=f"{ltitle}（2018 年 11 月门诊出行）", title_fontsize=9.5, alignment="left")
fig.savefig(OUT / "fig_outpatient_flow_plan1.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(OUT / "fig_outpatient_flow_plan1.png")
