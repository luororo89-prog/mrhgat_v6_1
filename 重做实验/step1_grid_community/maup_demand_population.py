"""按 maup 归属结果汇总各社区（SQCODE）的需求人口，并与现有供给计算所用的需求人口核对。

需求人口为社区所辖 250 m 居住网格的人口之和，联通人口取字段 联通人，七普扩样人口取字段 七普加。
现有供给计算（2supply_indicators/code/calculate_supply_indicators_2018.py）按网格文件自带的 SQCODE 汇总，
其联通人口版本的结果为 community_supply_indicators_2018_mobile_users.csv 的 population_demand。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "maup_demand"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "input"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
COMM_POP = SRC / "深圳各社区人口.shp"
NODE = HERE.parent / "input/node_order.csv"
SUPPLY = (HERE.parent / "input"
          / "community_supply_indicators_2018_mobile_users.csv")
CORE = ("440303", "440304", "440305")

grid = gpd.read_file(GRID250)[["Tid", "SQCODE", "联通人", "七普加"]].rename(columns={"SQCODE": "SQCODE_file"})
grid["SQCODE_file"] = grid.SQCODE_file.astype("string")
maup_a = pd.read_csv(RES / "compare_open_source/grid250_assignment_by_method.csv", dtype=str)[["Tid", "maup"]]
maup_a["Tid"] = maup_a.Tid.astype(int)
grid = grid.merge(maup_a.rename(columns={"maup": "SQCODE_maup"}), on="Tid", how="left", validate="one_to_one")
assert grid.SQCODE_maup.notna().all()

node = pd.read_csv(NODE, encoding="utf-8-sig", dtype={"SQCODE": str})
comm = gpd.read_file(COMM_POP)[["SQCODE", "SQNAME", "联通人", "七普扩"]].rename(
    columns={"联通人": "联通人_社区人口表", "七普扩": "七普扩_社区人口表"})
comm = node[["OBJECTID", "SQCODE"]].merge(comm, on="SQCODE", how="left", validate="one_to_one")

by_maup = grid.groupby("SQCODE_maup").agg(n_grids_maup=("Tid", "size"),
                                         需求人口_联通_maup=("联通人", "sum"), 需求人口_七普_maup=("七普加", "sum"))
by_file = grid.groupby("SQCODE_file").agg(需求人口_联通_网格字段=("联通人", "sum"), 需求人口_七普_网格字段=("七普加", "sum"))
by_file.index = by_file.index.astype(str)
sup = pd.read_csv(SUPPLY, dtype={"SQCODE": str})[["SQCODE", "population_demand"]].rename(
    columns={"population_demand": "需求人口_联通_现有供给表"})

t = comm.join(by_maup, on="SQCODE").join(by_file, on="SQCODE").merge(sup, on="SQCODE", how="left")
num = [c for c in t.columns if c.startswith("需求人口") or c == "n_grids_maup"]
t[num] = t[num].fillna(0)
t["n_grids_maup"] = t.n_grids_maup.astype(int)
t["联通_差值"] = t.需求人口_联通_maup - t.需求人口_联通_现有供给表
t["联通_相对差"] = np.where(t.需求人口_联通_现有供给表 > 0, t.联通_差值 / t.需求人口_联通_现有供给表, np.nan)
t["七普_差值"] = t.需求人口_七普_maup - t.需求人口_七普_网格字段
t["七普_相对差"] = np.where(t.需求人口_七普_网格字段 > 0, t.七普_差值 / t.需求人口_七普_网格字段, np.nan)
t["市中心小社区"] = False  # 罗湖、福田、南山面积小于 0.3 平方千米
area = pd.read_csv(RES / "community_grid250_summary.csv", dtype={"SQCODE": str}).set_index("SQCODE").comm_area_m2
t["市中心小社区"] = t.SQCODE.str[:6].isin(CORE) & (t.SQCODE.map(area) < 0.3e6)
t.to_csv(OUT / "community_demand_population_maup.csv", index=False, encoding="utf-8-sig")

d = t.联通_差值
r = t.联通_相对差
q = lambda v: {str(k): round(float(x), 4) for k, x in v.quantile([0, .05, .25, .5, .75, .95, 1]).items()}
audit = {
    "社区数": len(t), "SQCODE唯一": bool(t.SQCODE.is_unique),
    "联通人口合计": {"网格": int(grid.联通人.sum()), "maup": int(t.需求人口_联通_maup.sum()),
                 "现有供给表": float(t.需求人口_联通_现有供给表.sum()), "社区人口表": int(t.联通人_社区人口表.sum())},
    "七普人口合计": {"网格": float(grid.七普加.sum()), "maup": float(t.需求人口_七普_maup.sum()),
                 "网格字段": float(t.需求人口_七普_网格字段.sum())},
    "现有供给表_与网格字段汇总一致的社区": int(np.isclose(t.需求人口_联通_现有供给表, t.需求人口_联通_网格字段).sum()),
    "社区人口表_与网格字段汇总一致的社区": int(np.isclose(t.联通人_社区人口表, t.需求人口_联通_网格字段, atol=1).sum()),
    "maup联通需求为0的社区": t.loc[t.需求人口_联通_maup == 0, ["SQCODE", "SQNAME"]].to_dict("records"),
    "maup七普需求为0的社区": t.loc[t.需求人口_七普_maup == 0, ["SQCODE", "SQNAME"]].to_dict("records"),
    "联通_变化的社区": int((d.abs() > 0.5).sum()),
    "联通_变化人口_绝对值合计的一半": float(d.abs().sum() / 2),
    "联通_相对差分位数": q(r.dropna()),
    "联通_相对差超过5%": int((r.abs() > 0.05).sum()), "联通_相对差超过20%": int((r.abs() > 0.2).sum()),
    "联通_相对差超过50%": int((r.abs() > 0.5).sum()),
    "市中心小社区_联通相对差超过20%": int((r.abs() > 0.2)[t.市中心小社区].sum()),
    "七普_相对差超过20%": int((t.七普_相对差.abs() > 0.2).sum()),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
cols = ["SQCODE", "SQNAME", "n_grids_maup", "需求人口_联通_现有供给表", "需求人口_联通_maup", "联通_差值", "联通_相对差"]
print(t.reindex(t.联通_相对差.abs().sort_values(ascending=False).index)[cols].head(15).round(3).to_string(index=False))
print(t.reindex(t.联通_差值.abs().sort_values(ascending=False).index)[cols].head(10).round(3).to_string(index=False))
