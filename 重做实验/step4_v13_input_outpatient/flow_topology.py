"""（门诊 OD 版本：复制 step3_od_demand_time_filtered/flow_topology.py，只改输入输出路径，流量取 step3_od_demand_outpatient/results，结果写入本目录 flow_topology/。）
按起点社区与到达 250 m 网格的拓扑关系，把门诊流分为 within、adj、cross 三类并统计比例。

within：到达 250 m 网格在起点社区的生活圈内；
adj：不属于 within，到达网格在起点社区、起点社区的邻接社区或邻接社区的生活圈内；
cross：其余。
生活圈与供给评价一致，取社区 15 分钟步行等时圈，无等时圈的社区取社区边界；到达网格按中心点判定；
within 与 adj 的判定范围（生活圈、起点社区、邻接社区、邻接社区生活圈）都向外加 BUFFER_M 米缓冲（现为 0 m，与主模型配置的 topology_pairs 一致），
结果写入 results/flow_topology_buf500/（BUFFER_M 为 0 时写入 results/flow_topology/）；
邻接社区为边界相接（共边或共点）的社区。流量取 results/discount_<折扣>/od_250m.csv.gz，三档折扣分别统计。
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "step3_od_demand_outpatient" / "results"
BUFFER_M = 0
OUT = HERE / ("flow_topology" + (f"_buf{BUFFER_M}" if BUFFER_M else ""))
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
ISO = Path("/Users/luoxuan/Documents/mrhgat_v6_1_destpred/input_sources") / "等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp"
LEVELS = sys.argv[1:] or ["0.5", "0.25", "0.1"]  # 也可传入 0.5_hospring 等结果目录后缀
CRS = 4547

node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.make_valid()
iso = gpd.read_file(ISO).to_crs(CRS)
iso["SQCODE"] = iso.num.astype(str).str.replace(r"\.0$", "", regex=True)
iso = iso[iso.SQCODE.isin(comm.SQCODE)][["SQCODE", "geometry"]]
life = pd.concat([iso, comm[~comm.SQCODE.isin(iso.SQCODE)]], ignore_index=True)
life = gpd.GeoDataFrame(life, geometry="geometry", crs=CRS)
life["geometry"] = life.geometry.make_valid()

# 邻接社区：边界相接
adj = gpd.sjoin(comm, comm.assign(geometry=comm.buffer(1.0)), predicate="intersects", lsuffix="o", rsuffix="nb")
adj = adj[adj.SQCODE_o != adj.SQCODE_nb][["SQCODE_o", "SQCODE_nb"]].drop_duplicates()

# 到达 250 m 网格中心点所在的生活圈与社区
cells = set()
flows = {}
for d in LEVELS:
    f = pd.read_csv(RES / f"discount_{d}/od_250m.csv.gz", dtype={"SQCODE_o": str, "SQCODE_d": str})
    flows[d] = f.groupby(["SQCODE_o", "Tid_250_d"], as_index=False).trips.sum()
    cells |= set(flows[d].Tid_250_d)
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
cen = g250[g250.Tid.isin(cells)].assign(geometry=lambda x: x.geometry.centroid)
in_life = gpd.sjoin(cen, life.assign(geometry=life.buffer(BUFFER_M)), predicate="within")[["SQCODE", "Tid"]].drop_duplicates()
in_comm = gpd.sjoin(cen, comm.assign(geometry=comm.buffer(BUFFER_M)), predicate="within")[["SQCODE", "Tid"]].drop_duplicates()

within_pairs = set(zip(in_life.SQCODE, in_life.Tid))
adj_pairs = set(zip(in_comm.SQCODE, in_comm.Tid))
nb_comm = adj.merge(in_comm.rename(columns={"SQCODE": "SQCODE_nb"}), on="SQCODE_nb")
nb_life = adj.merge(in_life.rename(columns={"SQCODE": "SQCODE_nb"}), on="SQCODE_nb")
adj_pairs |= set(zip(nb_comm.SQCODE_o, nb_comm.Tid)) | set(zip(nb_life.SQCODE_o, nb_life.Tid))

summary = {"缓冲_m": BUFFER_M, "规则": {"within": "到达250m网格中心点在起点社区生活圈内",
                  "adj": "非within，且在起点社区、邻接社区或邻接社区生活圈内",
                  "cross": "其余"},
           "邻接社区对": int(len(adj)), "无邻接社区的社区": sorted(set(comm.SQCODE) - set(adj.SQCODE_o))}
name = node.set_index("SQCODE").SQNAME
for d in LEVELS:
    f = flows[d]
    key = list(zip(f.SQCODE_o, f.Tid_250_d))
    f["topology"] = ["within" if k in within_pairs else "adj" if k in adj_pairs else "cross" for k in key]
    f.to_csv(OUT / f"flow_topology_{d}.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    tot = f.trips.sum()
    share = f.groupby("topology").trips.sum().reindex(["within", "adj", "cross"]).fillna(0)
    pairs = f.groupby("topology").size().reindex(["within", "adj", "cross"]).fillna(0)
    by = f.pivot_table(index="SQCODE_o", columns="topology", values="trips", aggfunc="sum").reindex(
        columns=["within", "adj", "cross"]).fillna(0)
    by["total"] = by.sum(axis=1)
    for c in ["within", "adj", "cross"]:
        by[f"{c}_share"] = by[c] / by.total
    by.insert(0, "SQNAME", by.index.map(name))
    by.reset_index().to_csv(OUT / f"community_topology_share_{d}.csv", index=False, encoding="utf-8-sig")
    summary[d] = {"总人次": round(float(tot), 1),
                  "人次比例": {k: round(float(v / tot), 4) for k, v in share.items()},
                  "各类人次": {k: round(float(v), 1) for k, v in share.items()},
                  "起点社区-到达网格对": {k: int(v) for k, v in pairs.items()},
                  "起点社区within比例中位数": round(float(by.within_share.median()), 4),
                  "起点社区adj比例中位数": round(float(by.adj_share.median()), 4),
                  "起点社区cross比例中位数": round(float(by.cross_share.median()), 4)}
(OUT / ("summary.json" if len(sys.argv) == 1 else f"summary_{'_'.join(LEVELS)}.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
