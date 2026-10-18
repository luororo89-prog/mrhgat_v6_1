"""生成 v13 模型输入：供给表、社区网络（OD 矩阵）、社区对拓扑关系（时间筛选门诊 OD 与新社康点位版本）。

复制 step4_v13_input/build_v13_inputs.py，只改输入：供给取 step3_od_demand_time_filtered/results/supply_indicators_revised，
OD、设施吸引力与流量拓扑取 step3_od_demand_time_filtered/results，250 m 网格社区归属取修订后的 grid250_demand_allocation_revised.csv，
缓冲 BUFFER_M 取 0（与主模型配置使用的 topology_pairs_d<折扣>.csv 一致）。

1. 供给表 supply_v13.xlsx：supply_indicators_revised 联通人口版社区供给指标，列与原 联通人口_九项医疗供给指标_2018.xlsx 相同。
2. 社区网络 network_d<折扣>.csv：step3_od_demand/results/discount_<折扣>/community_od.csv 转为 666×666 整数人次矩阵
   （逐行取整：各起点社区先向下取整，余下人次按小数部分从大到小补 1，起点社区总人次为原值四舍五入），
   行列为 community_id（node_order 的 OBJECTID），格式与 4inter_community_network/results/medical_network_15min_walk.csv 相同。
3. 社区对拓扑 topology_pairs.csv：起点社区 O 到目的社区 D 的拓扑关系，由 D 的到达 250 m 网格相对 O 的位置确定：
   网格中心点在 O 的生活圈内为 within；否则在 O 社区、O 的邻接社区或邻接社区生活圈内为 adj；其余为 cross。
   D 的到达网格取 D 的设施所在 250 m 网格，按设施吸引力加权（与到达需求分配相同）；D 没有设施时取归属 D 的全部 250 m 网格，等权。
   取权重最大的一类（并列时取 within、adj、cross 中靠前者）。
   有流量的社区对改用逐条流量的拓扑（step3_od_demand/results/flow_topology，起点社区到到达 250 m 网格逐条分类），
   取该社区对人次最多的一类，三档折扣分别生成 topology_pairs_d<折扣>.csv；没有流量的社区对用上面的空间规则。
   编码 within=0、adj=1、cross=2，对角线为 0（社区内部出行不进入网络）。
   within 与 adj 的判定范围都向外加 BUFFER_M 米缓冲（现为 500 m），流量拓扑取 flow_topology_buf500，
   拓扑文件为 topology_pairs_d<折扣>_buf500.csv（BUFFER_M 为 0 时不加后缀）。
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REDO = HERE.parent
SRC = REDO / "supply input"
SUPPLY = REDO / "step3_od_demand_time_filtered/results/supply_indicators_revised/community_supply_indicators_2018_mobile_users.csv"
OD_DIR = REDO / "step3_od_demand_time_filtered/results"
ATTR = OD_DIR / "discount_0.5/facility_grid250_attraction.csv"  # 设施吸引力与折扣无关
DEMAND250 = REDO / "step3_od_demand_time_filtered/results/facility_grid250/grid250_demand_allocation_revised.csv"
ISO = REDO.parent / "1input_sources/等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp"
BUFFER_M = 0
SUF = f"_buf{BUFFER_M}" if BUFFER_M else ""
LEVELS = sys.argv[1:] or ["0.5", "0.25", "0.1"]  # 也可传入 0.5_hospring 等结果目录后缀
CRS = 4547
CODE = {"within": 0, "adj": 1, "cross": 2}

node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
ids = node.sort_values("OBJECTID").OBJECTID.to_numpy()
code_of = node.set_index("SQCODE").OBJECTID
audit = {}

# ---------- 1. 供给表 ----------
s = pd.read_csv(SUPPLY, dtype={"SQCODE": str}).sort_values("community_id")
cols = ["community_id", "SQCODE", "community_name", "RA_CHC", "RA_SH", "RA_TH", "SA_CHC_per_10000", "SA_SH_per_10000",
        "SA_TH_per_10000", "TC_CHC", "TC_SH", "TC_TH"]
assert (s.community_id.to_numpy() == ids).all()
s[cols].to_excel(HERE / "supply_v13.xlsx", index=False)

# ---------- 2. 社区网络 ----------
for d in LEVELS:
    c = pd.read_csv(OD_DIR / f"discount_{d}/community_od.csv")
    W = c.pivot_table(index="OBJECTID_o", columns="OBJECTID_d", values="trips", aggfunc="sum").reindex(
        index=ids, columns=ids).fillna(0.0)
    F = np.floor(W.to_numpy())
    rem = np.rint(W.to_numpy().sum(1) - F.sum(1)).astype(int)
    frac = W.to_numpy() - F
    for i in np.flatnonzero(rem > 0):
        F[i, np.argsort(-frac[i], kind="stable")[:rem[i]]] += 1
    W = pd.DataFrame(F.astype(np.int64), index=W.index, columns=W.columns)
    W.index.name = "source_community_id"
    W.to_csv(HERE / f"network_d{d}.csv", encoding="utf-8-sig")
    off = ~np.eye(len(ids), dtype=bool)
    audit[f"network_d{d}"] = {"人次": round(float(W.to_numpy().sum()), 1), "社区内部人次": round(float(np.trace(W.to_numpy())), 1),
                              "非零社区对（不含对角线）": int(((W.to_numpy() > 0) & off).sum())}

# ---------- 3. 社区对拓扑 ----------
comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.make_valid()
iso = gpd.read_file(ISO).to_crs(CRS)
iso["SQCODE"] = iso.num.astype(str).str.replace(r"\.0$", "", regex=True)
iso = iso[iso.SQCODE.isin(comm.SQCODE)][["SQCODE", "geometry"]]
life = gpd.GeoDataFrame(pd.concat([iso, comm[~comm.SQCODE.isin(iso.SQCODE)]], ignore_index=True), crs=CRS)
life["geometry"] = life.geometry.make_valid()
adj = gpd.sjoin(comm, comm.assign(geometry=comm.buffer(1.0)), predicate="intersects", lsuffix="o", rsuffix="nb")
adj = adj[adj.SQCODE_o != adj.SQCODE_nb][["SQCODE_o", "SQCODE_nb"]].drop_duplicates()

fp = pd.read_csv(ATTR, dtype={"SQCODE": str})
dest = fp.groupby(["SQCODE", "Tid_250"], as_index=False).attraction.sum().rename(columns={"attraction": "w"})
d250 = pd.read_csv(DEMAND250, dtype={"SQCODE": str})[["Tid", "SQCODE"]].drop_duplicates()
nofac = d250[~d250.SQCODE.isin(dest.SQCODE)].rename(columns={"Tid": "Tid_250"}).assign(w=1.0)
dest = pd.concat([dest, nofac], ignore_index=True)
assert set(dest.SQCODE) == set(node.SQCODE)

g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
cen = g250[g250.Tid.isin(dest.Tid_250)].assign(geometry=lambda x: x.geometry.centroid)
in_life = gpd.sjoin(cen, life.assign(geometry=life.buffer(BUFFER_M)), predicate="within")[["SQCODE", "Tid"]].drop_duplicates()
in_comm = gpd.sjoin(cen, comm.assign(geometry=comm.buffer(BUFFER_M)), predicate="within")[["SQCODE", "Tid"]].drop_duplicates()
within = in_life.groupby("SQCODE").Tid.apply(set).to_dict()
own = in_comm.groupby("SQCODE").Tid.apply(set).to_dict()
near = {o: set(v) for o, v in own.items()}
for o, nb in adj.groupby("SQCODE_o").SQCODE_nb:
    for b in nb:
        near[o] = near.get(o, set()) | own.get(b, set()) | within.get(b, set())

dest_cells = {k: (g.Tid_250.to_numpy(), g.w.to_numpy()) for k, g in dest.groupby("SQCODE")}
rows = []
for o in node.SQCODE:
    wi, ne = within.get(o, set()), near.get(o, set())
    for dq, (cells, w) in dest_cells.items():
        if dq == o:
            continue
        cls = np.array([0 if c in wi else 1 if c in ne else 2 for c in cells])
        tot = np.bincount(cls, weights=w, minlength=3)
        rows.append((code_of[o], code_of[dq], int(np.argmax(tot))))
topo = pd.DataFrame(rows, columns=["source_community_id", "target_community_id", "topology_code"])
inv = {v: k for k, v in CODE.items()}
topo["topology"] = topo.topology_code.map(inv)
name = node.set_index("OBJECTID").SQNAME
topo.insert(1, "source_name", topo.source_community_id.map(name))
topo.insert(3, "target_name", topo.target_community_id.map(name))
topo.to_csv(HERE / f"topology_pairs{SUF}.csv", index=False, encoding="utf-8-sig")
audit["社区对拓扑"] = {"社区对": int(len(topo)), "分布": topo.topology.value_counts().to_dict(),
                     "邻接社区对": int(len(adj)), "无设施社区按全部归属网格等权": int(nofac.SQCODE.nunique())}
for d in LEVELS:
    od = pd.read_csv(OD_DIR / f"discount_{d}/od_250m.csv.gz", dtype={"SQCODE_o": str, "SQCODE_d": str})
    ft = pd.read_csv(OD_DIR / f"flow_topology{SUF}/flow_topology_{d}.csv.gz", dtype={"SQCODE_o": str}).set_index(
        ["SQCODE_o", "Tid_250_d"]).topology
    od["topology"] = ft.reindex(pd.MultiIndex.from_arrays([od.SQCODE_o, od.Tid_250_d])).to_numpy()
    assert od.topology.notna().all()
    od = od[od.SQCODE_o != od.SQCODE_d]
    pt = od.groupby(["SQCODE_o", "SQCODE_d", "topology"]).trips.sum().unstack(fill_value=0).reindex(
        columns=["within", "adj", "cross"], fill_value=0)
    flow_cls = pt.idxmax(axis=1)
    td = topo.copy()
    key = pd.MultiIndex.from_arrays([td.source_community_id.map(dict(zip(node.OBJECTID, node.SQCODE))),
                                     td.target_community_id.map(dict(zip(node.OBJECTID, node.SQCODE)))])
    fc = flow_cls.reindex(key).to_numpy()
    td["rule"] = np.where(pd.notna(fc), "流量多数类", "空间规则")
    td["topology"] = np.where(pd.notna(fc), fc, td.topology)
    td["topology_code"] = td.topology.map(CODE)
    td.to_csv(HERE / f"topology_pairs_d{d}{SUF}.csv", index=False, encoding="utf-8-sig")
    audit[f"topology_pairs_d{d}{SUF}"] = {"按流量多数类的社区对": int((td.rule == "流量多数类").sum()),
                                     "分布": td.topology.value_counts().to_dict()}
    W = pd.read_csv(HERE / f"network_d{d}.csv", index_col=0, encoding="utf-8-sig")
    W.columns = W.columns.astype(int)
    t = td.set_index(["source_community_id", "target_community_id"]).topology
    f = W.stack()
    f = f[(f > 0) & (f.index.get_level_values(0) != f.index.get_level_values(1))]
    share = f.groupby(t.reindex(f.index).to_numpy()).sum() / f.sum()
    audit[f"network_d{d}"]["各拓扑人次比例（不含对角线）"] = {k: round(float(v), 4) for k, v in share.items()}
(HERE / (f"build_audit{SUF}.json" if len(sys.argv) == 1 else f"build_audit_{'_'.join(LEVELS)}{SUF}.json")).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
