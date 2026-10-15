"""小社区需求人口对比：maup 归属（最大重叠面积）与现有需求人口（网格中心点归属）。

参照值为面积加权人口，即每个 250 m 网格的联通人口按各社区在网格内的面积比例分摊后汇总。
小社区指面积小于 0.3 平方千米的社区，市中心指罗湖、福田、南山。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "maup_demand"
GRID250 = (HERE.parent / "input"
           / "深圳250m网格_人口_所在社区.shp")
CORE = ("440303", "440304", "440305")

pop = gpd.read_file(GRID250)[["Tid", "联通人"]]
p = pd.read_csv(RES / "grid250_community_pieces.csv", dtype={"SQCODE": str}).merge(pop, on="Tid")
p["w"] = p.piece_m2 / p.groupby("Tid").piece_m2.transform("sum")
ref = (p["联通人"] * p.w).groupby(p.SQCODE).sum().rename("面积加权人口")

t = pd.read_csv(OUT / "community_demand_population_maup.csv", dtype={"SQCODE": str}, encoding="utf-8-sig")
geo = pd.read_csv(RES / "compare_open_source/community_metrics_by_method.csv", header=[0, 1], index_col=0)
geo.index = geo.index.astype(str)
area = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str}).set_index("SQCODE").comm_area_m2
t = t.join(ref, on="SQCODE")
t["面积"] = t.SQCODE.map(area)
t["maup误差"] = (t.需求人口_联通_maup - t.面积加权人口).abs() / t.面积加权人口
t["现有误差"] = (t.需求人口_联通_现有供给表 - t.面积加权人口).abs() / t.面积加权人口
t["maup错位比例"] = t.SQCODE.map(geo[("maup", "offset")])
t["现有错位比例"] = t.SQCODE.map(geo[("transbigdata", "offset")])
t["maup网格数"] = t.SQCODE.map(geo[("maup", "n")]).astype(int)
t["现有网格数"] = t.SQCODE.map(geo[("transbigdata", "n")]).astype(int)

groups = {"市中心小社区": t.SQCODE.str[:6].isin(CORE) & (t.面积 < 0.3e6),
          "全部小社区": t.面积 < 0.3e6, "其他社区": t.面积 >= 0.3e6}
audit = {}
for name, m in groups.items():
    s = t[m]
    diff = s.maup误差 - s.现有误差
    audit[name] = {
        "社区数": int(m.sum()),
        "人口误差中位数_maup": round(float(s.maup误差.median()), 4),
        "人口误差中位数_现有": round(float(s.现有误差.median()), 4),
        "人口误差均值_maup": round(float(s.maup误差.mean()), 4),
        "人口误差均值_现有": round(float(s.现有误差.mean()), 4),
        "误差超过20%_maup": int((s.maup误差 > 0.2).sum()), "误差超过20%_现有": int((s.现有误差 > 0.2).sum()),
        "误差超过50%_maup": int((s.maup误差 > 0.5).sum()), "误差超过50%_现有": int((s.现有误差 > 0.5).sum()),
        "maup更接近": int((diff < -1e-9).sum()), "现有更接近": int((diff > 1e-9).sum()), "相同": int((diff.abs() <= 1e-9).sum()),
        "错位比例中位数_maup": round(float(s.maup错位比例.median()), 3),
        "错位比例中位数_现有": round(float(s.现有错位比例.median()), 3),
        "单网格社区_maup": int((s.maup网格数 == 1).sum()), "单网格社区_现有": int((s.现有网格数 == 1).sum()),
    }
cols = ["SQCODE", "SQNAME", "面积", "面积加权人口", "需求人口_联通_maup", "需求人口_联通_现有供给表", "maup误差", "现有误差",
        "maup网格数", "现有网格数", "maup错位比例", "现有错位比例"]
t.loc[groups["全部小社区"], cols].sort_values("面积").to_csv(OUT / "small_communities_maup_vs_existing.csv",
                                                           index=False, encoding="utf-8-sig")
(OUT / "small_communities_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
s = t[groups["市中心小社区"]].copy()
s["差"] = s.maup误差 - s.现有误差
print(s.sort_values("差")[["SQNAME", "面积加权人口", "需求人口_联通_maup", "需求人口_联通_现有供给表", "maup误差", "现有误差"]].round(2).head(6).to_string(index=False))
print(s.sort_values("差")[["SQNAME", "面积加权人口", "需求人口_联通_maup", "需求人口_联通_现有供给表", "maup误差", "现有误差"]].round(2).tail(6).to_string(index=False))
