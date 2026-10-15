"""门诊 OD 清洗与验证。

清洗：到达网格（d）本身及周围一圈网格（共边或共角的 500 m 网格）内都没有社康和医院 AOI 的记录删除。
社康取 facility_supply_audit_2018_mobile_users.csv 中 tier 为 CHC 的 912 个点位；医院 AOI 取 hospital_aoi_35.shp，
与网格面相交（相交面积大于 0）即算网格含 AOI。

验证（清洗后的 OD）：
1. 医院门诊量相关：到达网格与医院 AOI 挂接，网格与 AOI 相交或网格中心到 AOI 不超过 300 m 即挂接到该医院，
   同一网格挂接多家医院时到达人次平均分给这些医院。各医院到达人次与 AOI 表 outpat 字段（年度门诊量）计算相关。
2. 缓冲距离敏感性（Yu et al., 2026）：0、100、200、300、500 m 下的挂接网格、到达人次与相关系数。
3. 起点代表性（Wang et al., 2021）：各居住网格出发人次与网格内联通居住人口的相关。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/od_clean_validation"
OUT.mkdir(parents=True, exist_ok=True)
SRC = HERE.parent / "supply input"
GRID500 = SRC / "SZ_grid_500m.shp"
GRID250 = SRC / "深圳250m网格_人口_所在社区.shp"
FAC = SRC / "facility_supply_audit_2018_mobile_users.csv"
AOI = SRC / "hospital_aoi_35.shp"
OD = SRC / "outpatient_grid_OD_home_tid.csv"
CRS = 4547
BUFFER = 300
BUFFERS = [0, 100, 200, 300, 500]

g5 = gpd.read_file(GRID500)[["Tid", "x", "y", "geometry"]].to_crs(CRS)
cen = gpd.GeoDataFrame(g5[["Tid"]], geometry=gpd.points_from_xy(g5.x, g5.y), crs=4326).to_crs(CRS)
fac = pd.read_csv(FAC)
chc = fac[fac.tier == "CHC"]
chc = gpd.GeoDataFrame(chc, geometry=gpd.points_from_xy(chc.longitude, chc.latitude), crs=4326).to_crs(CRS)
aoi = gpd.read_file(AOI).to_crs(CRS)
od = pd.read_csv(OD, encoding="utf-8-sig")

# ---------- 清洗 ----------
n_chc = gpd.sjoin(chc, g5[["Tid", "geometry"]], how="inner", predicate="within").groupby("Tid").size()
ag = gpd.overlay(aoi[["fac_name", "geometry"]], g5[["Tid", "geometry"]], how="intersection", keep_geom_type=True)
ag = ag[ag.area > 0]
n_aoi = ag.groupby("Tid").fac_name.nunique()
has = set(n_chc.index) | set(n_aoi.index)

# 周围一圈：网格为 0.004° 规则格网，按行列号取 3×3 邻域
xy = g5[["Tid", "x", "y"]].copy()
xy["col"] = ((xy.x - xy.x.min()) / 0.004).round().astype(int)
xy["row"] = ((xy.y - xy.y.min()) / 0.004).round().astype(int)
cell = xy.set_index(["col", "row"]).Tid
nb = pd.concat([pd.DataFrame({"Tid": xy.Tid, "key": list(zip(xy.col + dc, xy.row + dr))})
                for dc in (-1, 0, 1) for dr in (-1, 0, 1) if (dc, dr) != (0, 0)])
nb["Tid_nb"] = nb.key.map(cell)
nb = nb.dropna(subset=["Tid_nb"]).astype({"Tid_nb": int})[["Tid", "Tid_nb"]]
nb_has = nb[nb.Tid_nb.isin(has)].groupby("Tid").Tid_nb.apply(lambda x: "、".join(map(str, sorted(x))))

end = pd.DataFrame({"end_tid": sorted(od.end_tid.unique())})
end["网格内社康数"] = end.end_tid.map(n_chc).fillna(0).astype(int)
end["网格内医院AOI数"] = end.end_tid.map(n_aoi).fillna(0).astype(int)
end["周围含设施网格"] = end.end_tid.map(nb_has).fillna("")
end["保留"] = end.end_tid.isin(has) | end["周围含设施网格"].ne("")
end["到达人次"] = end.end_tid.map(od.groupby("end_tid")["count"].sum())
end.to_csv(OUT / "end_grid_facility_flag.csv", index=False, encoding="utf-8-sig")

keep = od[od.end_tid.isin(end.loc[end["保留"], "end_tid"])]
keep.to_csv(OUT / "outpatient_grid_OD_home_tid_clean.csv", index=False, encoding="utf-8-sig")
arrive = keep.groupby("end_tid")["count"].sum()

# 无 AOI 的医院点位所在网格是否保留
hosp = fac[fac.tier != "CHC"]
no_aoi = hosp[~hosp.facility_name.isin(aoi.fac_name)]
no_aoi = gpd.GeoDataFrame(no_aoi[["facility_name", "tier"]],
                          geometry=gpd.points_from_xy(no_aoi.longitude, no_aoi.latitude), crs=4326).to_crs(CRS)
no_aoi = gpd.sjoin(no_aoi, g5[["Tid", "geometry"]], how="left", predicate="within")
no_aoi["网格保留"] = no_aoi.Tid.isin(end.loc[end["保留"], "end_tid"]) | ~no_aoi.Tid.isin(end.end_tid)
no_aoi["网格到达人次_清洗前"] = no_aoi.Tid.map(od.groupby("end_tid")["count"].sum()).fillna(0).astype(int)


# ---------- 医院挂接与门诊量相关 ----------
def link(b):
    m = ag[["fac_name", "Tid"]]
    if b > 0:
        near = gpd.sjoin(cen, aoi[["fac_name", "geometry"]].assign(geometry=aoi.buffer(b)),
                         how="inner", predicate="within")[["fac_name", "Tid"]]
        m = pd.concat([m, near])
    m = m.drop_duplicates().copy()
    m["n_hosp"] = m.groupby("Tid").fac_name.transform("nunique")
    m["arr"] = m.Tid.map(arrive).fillna(0) / m.n_hosp
    return m


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pr, pp = stats.pearsonr(x, y)
    pos = (x > 0) & (y > 0)
    lr, lp = stats.pearsonr(np.log10(x[pos]), np.log10(y[pos]))
    sr, sp = stats.spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": pr, "pearson_p": pp, "R2": pr ** 2,
            "log_pearson_r": lr, "log_pearson_p": lp, "n_log": int(pos.sum()), "spearman_rho": sr, "spearman_p": sp}


base = aoi[["fac_name", "tier", "beds", "outpat"]].set_index("fac_name")
rows, by_hosp = [], {}
for b in BUFFERS:
    m = link(b)
    per = base.join(m.groupby("fac_name").agg(n_grids=("Tid", "nunique"), arrivals=("arr", "sum")))
    per[["n_grids", "arrivals"]] = per[["n_grids", "arrivals"]].fillna(0)
    by_hosp[b] = per
    r = {"缓冲距离_m": b, "挂接网格": int(m.Tid.nunique()),
         "多家医院共享网格": int((m.drop_duplicates("Tid").n_hosp > 1).sum()),
         "挂接网格到达人次": int(arrive[arrive.index.isin(m.Tid)].sum()),
         "占清洗后到达人次": float(arrive[arrive.index.isin(m.Tid)].sum() / arrive.sum()),
         "到达人次为0的医院": int((per.arrivals == 0).sum())}
    r.update({f"全部_{k}": v for k, v in corr(per.arrivals, per.outpat).items()})
    for tier in ["TH", "SH"]:
        s = per[per.tier == tier]
        r.update({f"{tier}_{k}": v for k, v in corr(s.arrivals, s.outpat).items()})
    rows.append(r)
sens = pd.DataFrame(rows)
sens.to_csv(OUT / "buffer_sensitivity.csv", index=False, encoding="utf-8-sig")

h = by_hosp[BUFFER].reset_index().rename(columns={"fac_name": "医院", "outpat": "年度门诊量_统计",
                                                  "arrivals": f"到达人次_{BUFFER}m", "n_grids": "挂接网格数"})
h["到达人次占比"] = h[f"到达人次_{BUFFER}m"] / h[f"到达人次_{BUFFER}m"].sum()
h["门诊量占比"] = h.年度门诊量_统计 / h.年度门诊量_统计.sum()
h["占比比值"] = h.到达人次占比 / h.门诊量占比
h.sort_values("年度门诊量_统计", ascending=False).to_csv(OUT / "hospital_outpatient_validation.csv",
                                                     index=False, encoding="utf-8-sig")

# ---------- 起点代表性 ----------
g25 = gpd.read_file(GRID250)[["Tid", "联通人", "geometry"]].to_crs(CRS)
g25["geometry"] = g25.geometry.centroid
pop5 = gpd.sjoin(g25, g5[["Tid", "geometry"]], how="inner", predicate="within", lsuffix="250", rsuffix="500")
pop5 = pop5.groupby("Tid_500").联通人.sum()
home = keep.groupby("home_tid")["count"].sum()
o = pd.DataFrame({"pop": pop5, "trips": home}).fillna(0)
o = o[(o["pop"] > 0) | (o.trips > 0)]
rate = o.trips[o["pop"] > 0].sum() / o["pop"][o["pop"] > 0].sum()

audit = {
    "清洗": {
        "社康点位": int(len(chc)), "医院AOI": int(len(aoi)), "含社康或医院AOI的网格": len(has),
        "到达网格_原": int(len(end)), "到达网格_保留": int(end["保留"].sum()),
        "到达网格_本格有设施": int(end.end_tid.isin(has).sum()),
        "到达网格_仅周围有设施": int((~end.end_tid.isin(has) & end["保留"]).sum()),
        "OD行_原": int(len(od)), "OD行_保留": int(len(keep)),
        "人次_原": int(od["count"].sum()), "人次_保留": int(keep["count"].sum()),
        "人次保留比例": float(keep["count"].sum() / od["count"].sum()),
        "居住网格_原": int(od.home_tid.nunique()), "居住网格_保留": int(keep.home_tid.nunique()),
        "无AOI的医院": no_aoi[["facility_name", "tier", "Tid", "网格保留", "网格到达人次_清洗前"]].to_dict("records"),
    },
    f"医院门诊量相关_{BUFFER}m": sens.set_index("缓冲距离_m").loc[BUFFER].to_dict(),
    "起点代表性": {
        "居住网格数": int(len(o)), "有出发人次但无联通居住人口的网格": int(((o["pop"] == 0) & (o.trips > 0)).sum()),
        "有居住人口但无出发人次的网格": int(((o["pop"] > 0) & (o.trips == 0)).sum()),
        "每千联通居住人口出发人次": float(rate * 1000),
        **{k: v for k, v in corr(o["pop"], o.trips).items()},
    },
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2, default=float))
pd.set_option("display.width", 250)
cols = ["缓冲距离_m", "挂接网格", "多家医院共享网格", "占清洗后到达人次", "到达人次为0的医院",
        "全部_pearson_r", "全部_log_pearson_r", "全部_spearman_rho", "TH_spearman_rho", "SH_spearman_rho"]
print(sens[cols].round(3).to_string(index=False))
print(h.sort_values("年度门诊量_统计", ascending=False)[["医院", "tier", "年度门诊量_统计", "挂接网格数",
                                                    f"到达人次_{BUFFER}m", "占比比值"]].round(2).to_string(index=False))
