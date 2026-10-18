"""移动的社康（facility_grid250.py 中归属社区与点位所在 250 m 网格不相交、移到归属社区网格的社康）是否影响供给评价、社区 OD 与拓扑。

对每个移动的社康比较原点位与移动后网格中心点：
1. 供给评价：社康所属生活圈（社区 15 分钟步行等时圈，无等时圈的社区取社区边界，与供给评价一致）是否改变。
2. 社区 OD：所在 500 m 网格是否改变（改变则参与分配的到达网格改变），以及该社康网格的到达人次。
3. 拓扑：到达该网格的流（起点社区 → 到达网格）按 step3_od_demand/flow_topology.py 的规则（生活圈、社区与邻接社区向外缓冲 500 m，
   网格按中心点）分为 within、adj、cross，比较原点位网格与移动后网格的分类，统计分类改变的人次。
命令行参数为折扣档（默认 0.5），结果写入 results/check_relocated_facilities/。
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
D = sys.argv[1] if len(sys.argv) > 1 else "0.5"
OUT = RES / "check_relocated_facilities"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
ISO = HERE.parents[1] / "1input_sources/等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp"
CRS = 4547
BUFFER_M = 500

mv = pd.read_csv(RES / "facility_grid250/chc_relocated.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
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
sqname = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("SQCODE").SQNAME

grid500 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
cen = g250.set_index("Tid").centroid
up = gpd.sjoin(gpd.GeoDataFrame(geometry=cen, crs=CRS).reset_index(), grid500, predicate="within").set_index(
    "Tid_left").Tid_right


def members(geoms, lay):
    pts = gpd.GeoDataFrame({"k": range(len(geoms))}, geometry=list(geoms), crs=CRS)
    j = gpd.sjoin(pts, lay, predicate="within")
    return j.groupby("k").SQCODE.apply(lambda s: frozenset(s)).reindex(range(len(geoms))).apply(
        lambda v: v if isinstance(v, frozenset) else frozenset())


pt = gpd.points_from_xy(mv.x_point, mv.y_point)
new = cen.reindex(mv.Tid_250).to_numpy()
mv["生活圈_原点位"] = members(pt, life).to_numpy()
mv["生活圈_移动后"] = members(new, life).to_numpy()
mv["供给评价_生活圈改变"] = mv["生活圈_原点位"] != mv["生活圈_移动后"]
mv["500m网格_原点位"] = mv.Tid_250_point.map(up)
mv["500m网格_移动后"] = mv.Tid_250.map(up)
mv["社区OD_500m网格改变"] = mv["500m网格_原点位"] != mv["500m网格_移动后"]

# 拓扑：到达移动后网格（归属社区）的流按原点位网格与移动后网格分别分类
od250 = pd.read_csv(RES / f"discount_{D}/od_250m.csv.gz", dtype={"SQCODE_o": str, "SQCODE_d": str})
fl = od250.merge(mv[["facility_id", "Tid_250", "SQCODE", "Tid_250_point"]],
                 left_on=["Tid_250_d", "SQCODE_d"], right_on=["Tid_250", "SQCODE"])
fl = fl.groupby(["facility_id", "SQCODE_o", "Tid_250", "Tid_250_point"], as_index=False).trips.sum()
lay_life = life.assign(geometry=life.buffer(BUFFER_M))
lay_comm = comm.assign(geometry=comm.buffer(BUFFER_M))
nbr = adj.groupby("SQCODE_o").SQCODE_nb.apply(set)


def classify(o, cell_life, cell_comm):
    if o in cell_life:
        return "within"
    nb = nbr.get(o, set())
    if o in cell_comm or nb & cell_comm or nb & cell_life:
        return "adj"
    return "cross"


cells = sorted(set(fl.Tid_250) | set(fl.Tid_250_point))
cl = dict(zip(cells, members(cen.reindex(cells).to_numpy(), lay_life)))
cc = dict(zip(cells, members(cen.reindex(cells).to_numpy(), lay_comm)))
fl["拓扑_移动后"] = [classify(o, cl[t_], cc[t_]) for o, t_ in zip(fl.SQCODE_o, fl.Tid_250)]
fl["拓扑_原点位"] = [classify(o, cl[t_], cc[t_]) for o, t_ in zip(fl.SQCODE_o, fl.Tid_250_point)]
fl.to_csv(OUT / f"relocated_flow_topology_d{D}.csv", index=False, encoding="utf-8-sig")
mv["到达人次"] = mv.facility_id.map(fl.groupby("facility_id").trips.sum()).fillna(0)
chg = fl[fl["拓扑_移动后"] != fl["拓扑_原点位"]]
mv["拓扑_分类改变的人次"] = mv.facility_id.map(chg.groupby("facility_id").trips.sum()).fillna(0)
mv["社区OD_影响"] = mv["社区OD_500m网格改变"] & (mv["到达人次"] > 0)
mv["拓扑_影响"] = mv["拓扑_分类改变的人次"] > 0
mv["有影响"] = mv["供给评价_生活圈改变"] | mv["社区OD_影响"] | mv["拓扑_影响"]
for c in ["生活圈_原点位", "生活圈_移动后"]:
    mv[c] = mv[c].apply(lambda s: "、".join(sorted(sqname.get(x, x) for x in s)))
mv.drop(columns=["x_point", "y_point"]).sort_values(["有影响", "到达人次"], ascending=False).to_csv(
    OUT / f"relocated_effect_d{D}.csv", index=False, encoding="utf-8-sig")

audit = {"折扣档": D, "移动的社康": int(len(mv)),
         "供给评价_生活圈改变": int(mv["供给评价_生活圈改变"].sum()),
         "社区OD_500m网格改变且有到达人次": int(mv["社区OD_影响"].sum()),
         "拓扑_分类改变": int(mv["拓扑_影响"].sum()),
         "有影响": int(mv["有影响"].sum()),
         "有影响的社康": mv.loc[mv["有影响"], "facility_name"].tolist(),
         "无影响的社康": mv.loc[~mv["有影响"], "facility_name"].tolist()}
(OUT / f"audit_d{D}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
