"""清洗门诊 OD：到达网格本身及周围一圈网格（共边或共角的 500 m 网格）内都没有社康和医院 AOI 的记录删除。

社康取 facility_supply_audit_2018_mobile_users.csv 中 tier 为 CHC 的点位；
医院 AOI 取 hospital_aoi_35.shp，与网格面相交即算网格含 AOI。
OD 取 input/outpatient_grid_OD_home_tid.csv。
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "outpatient_od"
SRC = HERE.parent / "input"
GRID500 = SRC / "SZ_grid_500m.shp"
VAL = HERE.parent / "input"
FAC = VAL / "facility_supply_audit_2018_mobile_users.csv"
AOI = VAL / "hospital_aoi_35.shp"
CRS = 4547

g5 = gpd.read_file(GRID500)[["Tid", "geometry"]].to_crs(CRS)
fac = pd.read_csv(FAC)
chc = fac[fac.tier == "CHC"]
chc = gpd.GeoDataFrame(chc, geometry=gpd.points_from_xy(chc.longitude, chc.latitude), crs=4326).to_crs(CRS)
aoi = gpd.read_file(AOI).to_crs(CRS)
od = pd.read_csv(HERE.parent / "input/outpatient_grid_OD_home_tid.csv", encoding="utf-8-sig")

n_chc = gpd.sjoin(chc, g5, how="inner", predicate="within").groupby("Tid").size()
n_aoi = gpd.sjoin(aoi[["fac_name", "geometry"]], g5, how="inner", predicate="intersects")
n_aoi = n_aoi[n_aoi.apply(lambda r: r.geometry.intersection(g5.geometry.iloc[r.index_right]).area > 0, axis=1)]
n_aoi = n_aoi.groupby("Tid").size()
has = set(n_chc.index) | set(n_aoi.index)

# 周围一圈：与该网格共边或共角的网格。网格为 0.004° 规则格网，按行列号取 3×3 邻域
xy = gpd.read_file(GRID500)[["Tid", "x", "y"]]
xy["col"] = ((xy.x - xy.x.min()) / 0.004).round().astype(int)
xy["row"] = ((xy.y - xy.y.min()) / 0.004).round().astype(int)
cell = xy.set_index(["col", "row"]).Tid
nb = pd.concat([pd.DataFrame({"Tid": xy.Tid, "key": list(zip(xy.col + dc, xy.row + dr))})
                for dc in (-1, 0, 1) for dr in (-1, 0, 1) if (dc, dr) != (0, 0)])
nb["Tid_nb"] = nb.key.map(cell)
nb = nb.dropna(subset=["Tid_nb"]).astype({"Tid_nb": int})[["Tid", "Tid_nb"]]
nb_has = nb[nb.Tid_nb.isin(has)].groupby("Tid").Tid_nb.apply(lambda x: "、".join(map(str, sorted(x))))
n_nb = nb.groupby("Tid").size()

end = pd.DataFrame({"end_tid": sorted(od.end_tid.unique())})
end["网格内社康数"] = end.end_tid.map(n_chc).fillna(0).astype(int)
end["网格内医院AOI数"] = end.end_tid.map(n_aoi).fillna(0).astype(int)
end["周围网格数"] = end.end_tid.map(n_nb).fillna(0).astype(int)
end["周围含设施网格"] = end.end_tid.map(nb_has).fillna("")
end["在网格图层内"] = end.end_tid.isin(g5.Tid)
end["保留"] = end.end_tid.isin(has) | end["周围含设施网格"].ne("")
end["到达人次"] = end.end_tid.map(od.groupby("end_tid")["count"].sum())
end.to_csv(OUT / "outpatient_end_grid_facility_flag.csv", index=False, encoding="utf-8-sig")

keep = od[od.end_tid.isin(end.loc[end["保留"], "end_tid"])]
keep.to_csv(OUT / "outpatient_grid_OD_home_tid_clean.csv", index=False, encoding="utf-8-sig")

audit = {
    "社康点位": int(len(chc)), "医院AOI": int(len(aoi)),
    "含社康或医院AOI的网格": len(has),
    "周围网格数分布": n_nb.value_counts().sort_index().to_dict(),
    "到达网格_原": int(len(end)), "到达网格_保留": int(end["保留"].sum()),
    "到达网格_本格有设施": int(end.end_tid.isin(has).sum()),
    "到达网格_仅周围有设施": int((~end.end_tid.isin(has) & end["保留"]).sum()),
    "到达网格不在网格图层内": int((~end["在网格图层内"]).sum()),
    "OD行_原": int(len(od)), "OD行_保留": int(len(keep)),
    "人次_原": int(od["count"].sum()), "人次_保留": int(keep["count"].sum()),
    "人次保留比例": round(float(keep["count"].sum() / od["count"].sum()), 4),
    "居住网格_原": int(od.home_tid.nunique()), "居住网格_保留": int(keep.home_tid.nunique()),
}
(OUT / "audit_clean.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
