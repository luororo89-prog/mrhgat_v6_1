"""社区供给配置类型地图（各区统计标注），版式参照参考图：类型着色、区界、各区类型数量与占比引线标注、
图例附全市占比、经纬网、指北针与比例尺。

类型取 --types 目录下 node_types.csv（3node_typing/code/fit_node_types.py 用本目录 supply_indicators_revised 联通人口版供给指标拟合），
社区边界与区界同 3node_typing/code/plot_node_types.py。社区所在区按社区代表点落入的区界判定。
程序命名的“三甲二级型（三甲高·二级高·社康中）”中仅约 5% 社区有三甲医院，英文名记为 SH-based。
输出：<types>/fig_types_district.{pdf,svg,png} 与 fig_types_district_stats.csv。

python 重做实验/step3_od_demand_outpatient/plot_types_district_map.py
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE.parent / "supply input"
DISTRICT_FILE = SRC / "shenzhen_district_4547.shp"  # 原 1input_sources 已不在本机
DISTRICT_EN = {"440303": "Luohu", "440304": "Futian", "440305": "Nanshan", "440306": "Bao'an", "440307": "Longgang",
               "440308": "Yantian", "440309": "Longhua", "440310": "Pingshan", "440311": "Guangming"}
TYPE_EN = {"稀缺型（三甲低·二级低·社康低）": "Scarcity", "社康主导型（三甲低·二级低·社康高）": "CHC-based",
           "中等型（三甲低·二级低·社康中）": "Moderate", "三甲二级型（三甲高·二级高·社康中）": "SH-based",
           "二级主导型（三甲低·二级高·社康中）": "SH-based", "三甲社康型（三甲高·二级低·社康高）": "Abundance"}
ORDER = ["Scarcity", "Moderate", "CHC-based", "SH-based", "Abundance"]
COLOR = {"Scarcity": "#3C78B4", "Moderate": "#7FB3AE", "CHC-based": "#5AA84F", "SH-based": "#F28E2B",
         "Abundance": "#D9473F"}
# 各区标注位置（经度、纬度）：上方、左侧、下方三组
LABEL_AT = {"Guangming": (113.83, 22.93, "top"), "Longhua": (114.05, 22.93, "top"), "Longgang": (114.29, 22.93, "top"),
            "Pingshan": (114.53, 22.93, "top"), "Bao'an": (113.52, 22.74, "side"), "Nanshan": (113.53, 22.42, "side"),
            "Futian": (113.95, 22.355, "bottom"), "Luohu": (114.17, 22.355, "bottom"),
            "Yantian": (114.39, 22.355, "bottom")}
MM = 1 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default=str(HERE / "results/node_typing"))
    out = Path(ap.parse_args().types)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "svg.fonttype": "none",
                         "pdf.fonttype": 42, "font.size": 7})

    nt = pd.read_csv(out / "node_types.csv", encoding="utf-8-sig")
    nt["ctype"] = nt.type_name.map(TYPE_EN)
    assert nt.ctype.notna().all(), "类型名缺少英文对照"
    types = [t for t in ORDER if t in set(nt.ctype)]
    letter = {t: "abcde"[i] for i, t in enumerate(types)}
    node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
    comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(4547)
    comm["SQCODE"] = comm.SQCODE.astype(str)
    comm["geometry"] = comm.geometry.make_valid()
    comm = comm.merge(node[["SQCODE", "OBJECTID"]], on="SQCODE").merge(
        nt[["community_id", "ctype"]], left_on="OBJECTID", right_on="community_id", how="left")
    dist = gpd.read_file(DISTRICT_FILE).to_crs(4547)
    dist["district"] = dist.dt_adcode.astype(str).map(DISTRICT_EN)
    pts = comm.assign(geometry=comm.geometry.representative_point())
    comm["district"] = gpd.sjoin(pts, dist[["district", "geometry"]], predicate="within", how="left") \
        .groupby(level=0).district.first()
    land = comm.geometry.buffer(50).union_all().buffer(-50)
    dist["geometry"] = dist.geometry.intersection(land)

    stats = comm.pivot_table(index="district", columns="ctype", values="SQCODE", aggfunc="count").reindex(
        columns=types).fillna(0).astype(int)
    stats["n"] = stats.sum(axis=1)
    share = stats[types].div(stats.n, axis=0)
    stats.join(share.add_suffix("_share")).to_csv(out / "fig_types_district_stats.csv", encoding="utf-8-sig")
    total = comm["ctype"].value_counts().reindex(types)

    c4 = comm.to_crs(4326)
    d4 = dist.to_crs(4326)
    fig = plt.figure(figsize=(183 * MM, 125 * MM))
    ax = fig.add_axes([0.06, 0.06, 0.70, 0.90])
    lat0 = 22.65
    ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
    c4.plot(ax=ax, color=[COLOR.get(t, "#D9D9D9") for t in c4["ctype"]], edgecolor="white", linewidth=0.12)
    d4.boundary.plot(ax=ax, color="#333333", linewidth=0.5)
    x0, x1, y0, y1 = 113.42, 114.80, 22.17, 23.07
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for v in np.arange(113.6, 114.8, 0.2):
        ax.axvline(v, color="#9A9A9A", lw=0.4, ls=(0, (4, 3)), zorder=0)
    for v in np.arange(22.2, 23.1, 0.1):
        ax.axhline(v, color="#9A9A9A", lw=0.4, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(np.arange(113.6, 114.8, 0.2))
    ax.set_xticklabels([f"{v:.1f}° E" for v in np.arange(113.6, 114.8, 0.2)])
    ax.set_yticks(np.arange(22.2, 23.1, 0.1))
    ax.set_yticklabels([f"{v:.1f}° N" for v in np.arange(22.2, 23.1, 0.1)])
    ax.tick_params(labelsize=6, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

    fs, lh = 6.0, 6.0 * 1.3
    for name, geom in d4.set_index("district").geometry.items():
        if name not in LABEL_AT or name not in stats.index:
            continue
        p = geom.representative_point()
        lx, ly, pos = LABEL_AT[name]
        ax.plot(p.x, p.y, "o", ms=3, color="black", zorder=5)
        ax.annotate("", xy=(p.x, p.y), xytext=(lx, ly), zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="black", shrinkA=0, shrinkB=0))
        rows = "\n".join(f"{letter[t]} = {stats.at[name, t]} ({share.at[name, t]:.1%})" for t in types)
        kw = dict(textcoords="offset points", ha="center", zorder=6, annotation_clip=False,
                  bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        n = len(types)
        if pos == "top":
            ax.annotate(rows, (lx, ly), xytext=(0, 2), va="bottom", fontsize=fs, linespacing=1.3, **kw)
            ax.annotate(name, (lx, ly), xytext=(0, 5 + lh * n), va="bottom", fontsize=7, fontweight="bold", **kw)
        elif pos == "bottom":
            ax.annotate(name, (lx, ly), xytext=(0, -2), va="top", fontsize=7, fontweight="bold", **kw)
            ax.annotate(rows, (lx, ly), xytext=(0, -12), va="top", fontsize=fs, linespacing=1.3, **kw)
        else:
            ax.annotate(name, (lx, ly), xytext=(0, lh * n / 2 + 3), va="bottom", fontsize=7, fontweight="bold", **kw)
            ax.annotate(rows, (lx, ly), xytext=(0, 0), va="center", fontsize=fs, linespacing=1.3, **kw)

    # 指北针与比例尺（地图坐标）
    ax.annotate("N", xy=(114.74, 23.03), xytext=(114.74, 22.95), ha="center", va="top", fontsize=7, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
    deg_km = 1 / (111.32 * np.cos(np.deg2rad(lat0)))
    sx, sy = 114.55, 22.22
    for k_, (a, b) in enumerate([(0, 5), (5, 10), (10, 20)]):
        ax.add_patch(Rectangle((sx + a * deg_km, sy), (b - a) * deg_km, 0.008, fc="black" if k_ % 2 == 0 else "white",
                               ec="black", lw=0.5, zorder=6))
    for v in (0, 5, 10, 20):
        ax.text(sx + v * deg_km, sy + 0.012, f"{v} km" if v == 20 else str(v), ha="center", va="bottom", fontsize=5.8, zorder=6)

    lg = fig.add_axes([0.765, 0.40, 0.23, 0.40])
    lg.set_axis_off()
    lg.text(0, 1, f"Supply configuration types (k = {len(types)})", fontsize=7.5, fontweight="bold", va="top")
    lg.text(0, 0.92, "Community level, n = 666", fontsize=6.5, va="top")
    for i, t in enumerate(types):
        y = 0.80 - i * 0.11
        lg.add_patch(Rectangle((0, y - 0.035), 0.08, 0.07, fc=COLOR[t], ec="none", transform=lg.transAxes))
        lg.text(0.12, y, f"{letter[t]}: {t} ({total[t] / total.sum():.1%})", fontsize=6.5, va="center")

    for ext in ("pdf", "svg"):
        fig.savefig(out / f"fig_types_district.{ext}")
    fig.savefig(out / "fig_types_district.png", dpi=400)
    print(stats.join(share.round(3).add_suffix("_share")).to_string())
    print("saved", out / "fig_types_district.png")


if __name__ == "__main__":
    main()
