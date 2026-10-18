"""500 m 门诊 OD 拆到 250 m 网格并对齐社区后的错位与平滑检查（时间筛选门诊 OD，本目录 results）。

错位
1. 网格嵌套：250 m 网格落在所嵌套 500 m 网格内的面积比例。
2. 出发端：出发人次挂到所在 500 m 网格以外 250 m 网格的人次与距离。
3. 到达端：到达人次离开原 500 m 到达网格的人次（按候选圈层）与位移距离。
4. 社区对齐：人次所归社区与所挂 250 m 网格、原 500 m 网格是否相交（step1 的 grid250/grid500_community_pieces）。
平滑
5. OD 对：500 m OD 对拆成 250 m OD 对后的对数，不足 1 人次的对数与人次。
6. 250 m OD 汇总回 500 m 后与原 500 m OD 的差异，到达网格集中度。
7. 出行距离分布：500 m 中心点距离与 250 m 中心点距离（人次加权）。
8. 社区 OD：本方法与 maup 整格（出发、到达 500 m 网格各整格归一个社区）的非零对、出发社区的到达社区熵与集中度。
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "check_alignment_smoothing"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
STEP1 = HERE.parent / "step1_grid_community/results"
LEVELS = ["0.5", "0.25", "0.1"]
CRS = 4547
QS = (.1, .25, .5, .75, .9, .99)


def wq(x, w):
    i = np.argsort(np.asarray(x, float))
    x, w = np.asarray(x, float)[i], np.asarray(w, float)[i]
    c = np.cumsum(w) / w.sum()
    return {f"P{q * 100:g}": round(float(x[min(np.searchsorted(c, q), len(x) - 1)]), 1) for q in QS}


def wmean(x, w):
    return round(float(np.average(x, weights=w)), 1)


def gini(v):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    return round(float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum())), 4)


def top_share(v, frac=0.1):
    v = np.sort(np.asarray(v, float))[::-1]
    return round(float(v[:max(1, int(len(v) * frac))].sum() / v.sum()), 4)


def r(x, k=4):
    return round(float(x), k)


# ---------- 输入 ----------
od = pd.read_csv(RES / "grid500_origin_250m/outpatient_grid_OD_home_tid_clean_used.csv", encoding="utf-8-sig")
total = float(od["count"].sum())
o250 = pd.read_csv(RES / "grid500_origin_250m/grid250_origin_allocation.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
maup = pd.read_csv(STEP1 / "grid500_origin_maup/grid500_community_maup.csv", encoding="utf-8-sig",
                   dtype={"SQCODE_maup": str}).set_index("Tid").SQCODE_maup
pc500 = set(map(tuple, pd.read_csv(STEP1 / "grid500_community_pieces.csv", dtype={"SQCODE": str})[["Tid", "SQCODE"]]
                .to_numpy()))
pc250 = set(map(tuple, pd.read_csv(STEP1 / "grid250_community_pieces.csv", dtype={"SQCODE": str})[["Tid", "SQCODE"]]
                .to_numpy()))

g5 = gpd.read_file(SRC / "SZ_grid_500m.shp")[["Tid", "geometry"]].to_crs(CRS)
g25 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
c5, c25 = g5.set_index("Tid").centroid, g25.set_index("Tid").centroid
x5, y5, x25, y25 = c5.x, c5.y, c25.x, c25.y
nest = gpd.sjoin(g25.assign(geometry=g25.centroid), g5, predicate="within", lsuffix="250", rsuffix="500")[
    ["Tid_250", "Tid_500"]]
up = nest.set_index("Tid_250").Tid_500


def dist(xa, ya, xb, yb):
    return np.hypot(np.asarray(xa) - np.asarray(xb), np.asarray(ya) - np.asarray(yb))


audit = {"OD人次": total, "OD行": int(len(od))}

# ---------- 1. 网格嵌套 ----------
a = nest.merge(g25.rename(columns={"Tid": "Tid_250"}), on="Tid_250").merge(
    g5.rename(columns={"Tid": "Tid_500", "geometry": "g500"}), on="Tid_500")
s250 = gpd.GeoSeries(a.geometry.to_numpy(), crs=CRS)
inside = s250.intersection(gpd.GeoSeries(a.g500.to_numpy(), crs=CRS)).area / s250.area
used = set(o250.Tid_250.dropna().astype(int))
audit["1_网格嵌套"] = {
    "250m网格": int(len(g25)), "嵌套于500m网格": int(len(nest)),
    "落在所嵌套500m网格内的面积比例_最小值": r(inside.min()),
    "面积比例不足99%的250m网格": int((inside < 0.99).sum()),
    "500m网格内250m网格数": nest.groupby("Tid_500").size().value_counts().sort_index().to_dict(),
}

# ---------- 2. 出发端 ----------
o = o250.dropna(subset=["Tid_250"]).astype({"Tid_250": int})
o["Tid_500_of_250"] = o.Tid_250.map(up)
o["d_m"] = dist(x5.reindex(o.home_tid), y5.reindex(o.home_tid), x25.reindex(o.Tid_250), y25.reindex(o.Tid_250))
out_o = o[o.Tid_500_of_250 != o.home_tid]
o_comm_ok = np.array([(t, s) in pc250 for t, s in zip(o.Tid_250, o.SQCODE)])
o_comm500_ok = np.array([(t, s) in pc500 for t, s in zip(o.home_tid, o.SQCODE)])
audit["2_出发端"] = {
    "挂到所在500m网格以外的人次": r(out_o.O_trips.sum(), 1),
    "挂到所在500m网格以外的人次占比": r(out_o.O_trips.sum() / total, 6),
    "挂到所在500m网格以外的距离_m": out_o.d_m.round(1).tolist(),
    "500m中心到所挂250m中心距离_m": wq(o.d_m, o.O_trips),
    "拆到多个社区的500m出发网格人次占比": r(o[o.groupby("home_tid").SQCODE.transform("nunique") > 1].O_trips.sum() / total),
}
audit["4_社区对齐_出发端"] = {
    "所归社区与所挂250m网格不相交的人次": r(o.loc[~o_comm_ok, "O_trips"].sum(), 1),
    "所归社区与原500m出发网格不相交的人次": r(o.loc[~o_comm500_ok, "O_trips"].sum(), 1),
}

# ---------- 7. 500 m 距离 ----------
od["d_m"] = dist(x5.reindex(od.home_tid), y5.reindex(od.home_tid), x5.reindex(od.end_tid), y5.reindex(od.end_tid))
dist500 = {"均值": wmean(od.d_m, od["count"]), **wq(od.d_m, od["count"]),
           "不足1km人次占比": r(od.loc[od.d_m < 1000, "count"].sum() / total),
           "同一500m网格人次占比": r(od.loc[od.home_tid == od.end_tid, "count"].sum() / total)}
arr500 = od.groupby("end_tid")["count"].sum()
pairs500 = od.set_index(["home_tid", "end_tid"])["count"]

# maup 整格社区 OD
mo = od.assign(SQCODE_o=od.home_tid.map(maup), SQCODE_d=od.end_tid.map(maup))
cod_maup = mo.groupby(["SQCODE_o", "SQCODE_d"])["count"].sum()


def comm_stats(cod):
    cod = cod[cod > 0]
    p = cod / cod.groupby(level=0).transform("sum")
    h = -(p * np.log(p)).groupby(level=0).sum()
    w = cod.groupby(level=0).sum()
    dom = p.groupby(level=0).max()
    internal = cod[cod.index.get_level_values(0) == cod.index.get_level_values(1)].sum()
    return {"非零对": int(len(cod)), "不足1人次的对": int((cod < 1).sum()),
            "不足1人次的对的人次占比": r(cod[cod < 1].sum() / cod.sum()),
            "出发社区数": int(len(w)),
            "每个出发社区的到达社区数_中位数": float(cod.groupby(level=0).size().median()),
            "出发社区到达有效社区数exp(熵)_人次加权中位数": wq(np.exp(h.reindex(w.index)), w)["P50"],
            "出发社区最大到达社区占比_人次加权中位数": round(wq(dom.reindex(w.index), w)["P50"], 3),
            "社区OD对Gini": gini(cod), "社区内部人次占比": r(internal / cod.sum())}


audit["7_出行距离_500m中心点"] = dist500
audit["8_社区OD_maup整格"] = comm_stats(cod_maup)

# ---------- 逐档到达端 ----------
for d in LEVELS:
    R = RES / f"discount_{d}"
    da = pd.read_csv(R / "grid250_destination_allocation.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
    da = da.dropna(subset=["Tid_250"]).astype({"Tid_250": int})
    da["ring"] = da.rule.str.split("_").str[0]
    da["Tid_500_of_250"] = da.Tid_250.map(up)
    da["moved"] = da.Tid_500_of_250 != da.end_tid
    da["d_m"] = dist(x5.reindex(da.end_tid), y5.reindex(da.end_tid), x25.reindex(da.Tid_250), y25.reindex(da.Tid_250))
    c250_ok = np.array([(t, s) in pc250 for t, s in zip(da.Tid_250, da.SQCODE)])
    c500_ok = np.array([(t, s) in pc500 for t, s in zip(da.end_tid, da.SQCODE)])
    mv = da[da.moved]
    per_end = da.groupby(["end_tid", "SQCODE"]).d_share.sum()
    dom = per_end.groupby(level=0).max()
    n_comm_end = per_end.groupby(level=0).size()
    n_comm_geo = pd.Series([t for t, _ in pc500]).value_counts()
    res = {
        "3_到达端": {
            "离开原500m到达网格的人次": r(mv.D_trips.sum(), 1),
            "离开原500m到达网格的人次占比": r(mv.D_trips.sum() / total),
            "按候选圈层人次占比": {k: r(v / total) for k, v in da.groupby("ring").D_trips.sum().items()},
            "位移距离_m_全部": {"均值": wmean(da.d_m, da.D_trips), **wq(da.d_m, da.D_trips)},
            "位移距离_m_离开原网格": {"均值": wmean(mv.d_m, mv.D_trips), **wq(mv.d_m, mv.D_trips)},
            "位移超过1km的人次占比": r(da.loc[da.d_m > 1000, "D_trips"].sum() / total),
            "到达网格拆到多个社区": int((n_comm_end > 1).sum()), "到达网格": int(len(n_comm_end)),
            "到达网格最大社区占比_人次加权中位数": round(wq(dom, arr500.reindex(dom.index))["P50"], 3),
            "到达网格分到的社区数多于其相交社区数": int((n_comm_end > n_comm_geo.reindex(n_comm_end.index).fillna(0)).sum()),
        },
        "4_社区对齐_到达端": {
            "所归社区与所挂250m网格不相交的人次": r(da.loc[~c250_ok, "D_trips"].sum(), 1),
            "所归社区与所挂250m网格不相交的人次占比": r(da.loc[~c250_ok, "D_trips"].sum() / total),
            "所归社区与原500m到达网格不相交的人次": r(da.loc[~c500_ok, "D_trips"].sum(), 1),
            "所归社区与原500m到达网格不相交的人次占比": r(da.loc[~c500_ok, "D_trips"].sum() / total),
        },
    }
    if d == LEVELS[0]:
        bad = da[~c250_ok].groupby(["Tid_250", "SQCODE", "rule"], as_index=False).D_trips.sum()
        fga = pd.read_csv(R / "facility_grid250_attraction.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
        bad = bad.merge(fga.groupby(["Tid_250", "SQCODE"]).agg(
            facility=("facility_name", lambda x: "、".join(sorted(set(x)))), tier=("tier", "first")).reset_index(),
            on=["Tid_250", "SQCODE"], how="left")
        bad.sort_values("D_trips", ascending=False).to_csv(OUT / f"dest_comm_not_touch_250m_d{d}.csv", index=False,
                                                          encoding="utf-8-sig")

    od250 = pd.read_csv(R / "od_250m.csv.gz", dtype={"SQCODE_o": str, "SQCODE_d": str}).dropna(
        subset=["Tid_250_o", "Tid_250_d"]).astype({"Tid_250_o": int, "Tid_250_d": int})
    assert np.isclose(od250.trips.sum(), total)
    # 5. OD 对
    res["5_OD对"] = {
        "500mOD对": int(len(od)), "250mOD对": int(len(od250)), "对数倍数": r(len(od250) / len(od), 2),
        "250mOD对_不足0.1人次": int((od250.trips < 0.1).sum()),
        "250mOD对_不足1人次": int((od250.trips < 1).sum()),
        "不足1人次的对的人次占比": r(od250.loc[od250.trips < 1, "trips"].sum() / total),
        "500mOD对人次中位数": float(od["count"].median()), "250mOD对人次中位数": r(od250.trips.median()),
    }
    # 6. 汇总回 500 m
    back = od250.assign(h=od250.Tid_250_o.map(up), e=od250.Tid_250_d.map(up)).groupby(["h", "e"]).trips.sum()
    j = pd.concat([pairs500.rename("orig"), back.rename("back")], axis=1).fillna(0)
    arr_back = back.groupby(level=1).sum()
    ja = pd.concat([arr500.rename("orig"), arr_back.rename("back")], axis=1).fillna(0)
    res["6_汇总回500m"] = {
        "OD对_原": int((j.orig > 0).sum()), "OD对_汇总回": int((j.back > 0).sum()),
        "OD人次变动比例(0.5×Σ|差|/总)": r(0.5 * (j.orig - j.back).abs().sum() / total),
        "到达网格_原": int((ja.orig > 0).sum()), "到达网格_汇总回": int((ja.back > 0).sum()),
        "到达人次变动比例": r(0.5 * (ja.orig - ja.back).abs().sum() / total),
        "到达网格Gini_原": gini(ja.orig[ja.orig > 0]), "到达网格Gini_汇总回": gini(ja.back[ja.back > 0]),
        "到达前10%网格人次占比_原": top_share(ja.orig[ja.orig > 0]),
        "到达前10%网格人次占比_汇总回": top_share(ja.back[ja.back > 0]),
        "到达人次Spearman_原vs汇总回(两者并集网格)": r(spearmanr(ja.orig, ja.back).statistic),
        "到达人次Spearman_原vs汇总回(两者都有人次的网格)": r(spearmanr(*ja[(ja.orig > 0) & (ja.back > 0)].T.to_numpy()).statistic),
        "两者都有人次的网格": int(((ja.orig > 0) & (ja.back > 0)).sum()),
        "原有到达人次_汇总回为0的网格": int(((ja.orig > 0) & (ja.back == 0)).sum()),
        "原无到达人次_汇总回有人次的网格": int(((ja.orig == 0) & (ja.back > 0)).sum()),
    }
    # 7. 距离
    od250["d_m"] = dist(x25.reindex(od250.Tid_250_o), y25.reindex(od250.Tid_250_o),
                        x25.reindex(od250.Tid_250_d), y25.reindex(od250.Tid_250_d))
    res["7_出行距离_250m中心点"] = {
        "均值": wmean(od250.d_m, od250.trips), **wq(od250.d_m, od250.trips),
        "不足1km人次占比": r(od250.loc[od250.d_m < 1000, "trips"].sum() / total)}
    # 8. 社区 OD
    cod = pd.read_csv(R / "community_od.csv", dtype={"SQCODE_o": str, "SQCODE_d": str}).groupby(
        ["SQCODE_o", "SQCODE_d"]).trips.sum()
    jc = pd.concat([cod.rename("method"), cod_maup.rename("maup")], axis=1).fillna(0)
    res["8_社区OD_本方法"] = {**comm_stats(cod),
                         "与maup整格Spearman(并集对)": r(spearmanr(jc.method, jc.maup).statistic),
                         "与maup整格人次变动比例": r(0.5 * (jc.method - jc.maup).abs().sum() / total)}
    audit[f"d={d}"] = res
    if d == LEVELS[0]:
        bins = [0, 500, 1000, 2000, 3000, 5000, 10000, 20000, np.inf]
        h500 = od.groupby(pd.cut(od.d_m, bins, right=False), observed=False)["count"].sum() / total
        h250 = od250.groupby(pd.cut(od250.d_m, bins, right=False), observed=False).trips.sum() / total
        pd.DataFrame({"人次占比_500m": h500, "人次占比_250m": h250}).round(4).to_csv(
            OUT / f"distance_distribution_d{d}.csv", encoding="utf-8-sig")
        pd.concat([da[["end_tid", "Tid_250", "SQCODE", "ring", "moved", "d_m", "D_trips"]]]).to_csv(
            OUT / f"destination_displacement_d{d}.csv", index=False, encoding="utf-8-sig")

(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2, default=float))
