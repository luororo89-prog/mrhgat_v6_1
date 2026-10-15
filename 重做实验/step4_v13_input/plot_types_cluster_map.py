"""社区 OD 有向社区发现（组团）与组团内供给类型构成地图。

组团：network_d0.5.csv（不含社区内部出行）构成有向加权网络，用 Leiden 算法最大化有向模块度（RBConfiguration，
分辨率 1.0，10 个随机种子取模块度最高者）。
组团中心：组团内接收其他成员流入人次最多的社区，地图上以黑点标出，引线连到组团标注。
标注：组团内各供给类型的社区数与占比（字母对应图例中的类型），5 个社区以下的组团不标注。
类型取 --types 目录 node_types.csv；输出 <types>/fig_types_cluster.{pdf,svg,png}、types_cluster_stats.csv、
cluster_membership.csv。

python 重做实验/step4_v13_input/plot_types_cluster_map.py --types 重做实验/step4_v13_input/node_typing_k3
"""
import argparse
from pathlib import Path

import geopandas as gpd
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "supply input"
NETWORK = HERE / "network_d0.5.csv"
RESOLUTION = 1.0
MIN_LABEL = 5
TYPE_EN = {"稀缺型（三甲低·二级低·社康低）": "Scarcity", "社康主导型（三甲低·二级低·社康高）": "CHC-based",
           "中等型（三甲低·二级低·社康中）": "Moderate", "三甲二级型（三甲高·二级高·社康中）": "SH-based",
           "二级主导型（三甲低·二级高·社康中）": "SH-based", "三甲社康型（三甲高·二级低·社康高）": "Abundance"}
ORDER = ["Scarcity", "Moderate", "CHC-based", "SH-based", "Abundance"]
COLOR = {"Scarcity": "#3C78B4", "Moderate": "#7FB3AE", "CHC-based": "#5AA84F", "SH-based": "#F28E2B",
         "Abundance": "#D9473F"}
MM = 1 / 25.4


def clusters():
    W = pd.read_csv(NETWORK, index_col=0, encoding="utf-8-sig")
    W.columns = W.columns.astype(int)
    ids = W.index.to_numpy()
    A = W.to_numpy(float)
    np.fill_diagonal(A, 0)
    src, dst = np.nonzero(A)
    g = ig.Graph(n=len(ids), edges=list(zip(src, dst)), directed=True)
    g.es["weight"] = A[src, dst]
    best = None
    for seed in range(10):
        p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                              resolution_parameter=RESOLUTION, seed=seed, n_iterations=-1)
        q = g.modularity(p.membership, weights="weight", directed=True, resolution=RESOLUTION)
        if best is None or q > best[0]:
            best = (q, p.membership)
    memb = np.asarray(best[1])
    # 组团按规模从大到小编号
    order = pd.Series(memb).value_counts().index
    relabel = {c: i + 1 for i, c in enumerate(order)}
    memb = np.array([relabel[c] for c in memb])
    hub = {}
    for c in np.unique(memb):
        m = memb == c
        hub[c] = int(ids[m][np.argmax(A[np.ix_(m, m)].sum(0))])
    inflow_in = A.sum(0)
    return pd.DataFrame({"community_id": ids, "cluster": memb, "inflow_total": inflow_in}), hub, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default=str(HERE / "node_typing_k3"))
    out = Path(ap.parse_args().types)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Arial Unicode MS"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7})
    memb, hub, q = clusters()
    nt = pd.read_csv(out / "node_types.csv", encoding="utf-8-sig")
    nt["ctype"] = nt.type_name.map(TYPE_EN)
    types = [t for t in ORDER if t in set(nt.ctype)]
    letter = {t: "abcde"[i] for i, t in enumerate(types)}
    node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
    comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(4547)
    comm["SQCODE"] = comm.SQCODE.astype(str)
    comm["geometry"] = comm.geometry.make_valid()
    comm = comm.merge(node[["SQCODE", "OBJECTID", "SQNAME"]], on="SQCODE").rename(columns={"OBJECTID": "community_id"})
    comm = comm.merge(nt[["community_id", "ctype"]], on="community_id").merge(memb, on="community_id")
    comm[["community_id", "SQCODE", "SQNAME", "cluster", "ctype", "inflow_total"]].assign(
        is_hub=comm.community_id.isin(hub.values())).to_csv(out / "cluster_membership.csv", index=False,
                                                            encoding="utf-8-sig")
    stats = comm.pivot_table(index="cluster", columns="ctype", values="SQCODE", aggfunc="count").reindex(
        columns=types).fillna(0).astype(int)
    stats["n"] = stats.sum(axis=1)
    share = stats[types].div(stats.n, axis=0)
    name = comm.set_index("community_id").SQNAME
    stats.insert(0, "hub", [name[hub[c]] for c in stats.index])
    stats.join(share.add_suffix("_share")).to_csv(out / "types_cluster_stats.csv", encoding="utf-8-sig")
    total = comm["ctype"].value_counts().reindex(types)

    c4 = comm.to_crs(4326)
    cl = c4.dissolve("cluster")
    fig = plt.figure(figsize=(183 * MM, 125 * MM))
    ax = fig.add_axes([0.06, 0.06, 0.70, 0.90])
    lat0 = 22.65
    ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
    c4.plot(ax=ax, color=[COLOR[t] for t in c4["ctype"]], edgecolor="white", linewidth=0.12)
    cl.boundary.plot(ax=ax, color="#1A1A1A", linewidth=1.0)
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
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

    # 标注位置：中心在 22.62°N 以北的组团放上方，以南的放下方，各行按中心经度排开
    pts = c4.set_index("community_id").geometry.representative_point()
    lab = [c for c in stats.index if stats.at[c, "n"] >= MIN_LABEL]
    top = sorted([c for c in lab if pts[hub[c]].y >= 22.62], key=lambda c: pts[hub[c]].x)
    bot = sorted([c for c in lab if pts[hub[c]].y < 22.62], key=lambda c: pts[hub[c]].x)
    fs, lh = 6.0, 6.0 * 1.3
    for row, ly, pos in [(top, 22.93, "top"), (bot, 22.355, "bottom")]:
        xs = np.linspace(113.62, 114.62, len(row) + 2)[1:-1] if len(row) > 1 else [pts[hub[row[0]]].x] if row else []
        for c, lx in zip(row, xs):
            p = pts[hub[c]]
            ax.plot(p.x, p.y, "o", ms=4, color="black", mec="white", mew=0.6, zorder=5)
            ax.annotate("", xy=(p.x, p.y), xytext=(lx, ly), zorder=4,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="black", shrinkA=0, shrinkB=0))
            head = f"Cluster {c} · {name[hub[c]]} (n = {stats.at[c, 'n']})"
            rows = "\n".join(f"{letter[t]} = {stats.at[c, t]} ({share.at[c, t]:.1%})" for t in types)
            kw = dict(textcoords="offset points", ha="center", zorder=6, annotation_clip=False,
                      bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
            if pos == "top":
                ax.annotate(rows, (lx, ly), xytext=(0, 2), va="bottom", fontsize=fs, linespacing=1.3, **kw)
                ax.annotate(head, (lx, ly), xytext=(0, 5 + lh * len(types)), va="bottom", fontsize=6.5,
                            fontweight="bold", **kw)
            else:
                ax.annotate(head, (lx, ly), xytext=(0, -2), va="top", fontsize=6.5, fontweight="bold", **kw)
                ax.annotate(rows, (lx, ly), xytext=(0, -12), va="top", fontsize=fs, linespacing=1.3, **kw)

    ax.annotate("N", xy=(114.74, 23.03), xytext=(114.74, 22.95), ha="center", va="top", fontsize=7, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
    deg_km = 1 / (111.32 * np.cos(np.deg2rad(lat0)))
    sx, sy = 114.55, 22.22
    for k_, (a, b) in enumerate([(0, 5), (5, 10), (10, 20)]):
        ax.add_patch(Rectangle((sx + a * deg_km, sy), (b - a) * deg_km, 0.008, fc="black" if k_ % 2 == 0 else "white",
                               ec="black", lw=0.5, zorder=6))
    for v in (0, 5, 10, 20):
        ax.text(sx + v * deg_km, sy + 0.012, str(v), ha="center", va="bottom", fontsize=5.8, zorder=6)
    ax.text(sx + 20 * deg_km + 0.008, sy, "km", ha="left", va="bottom", fontsize=5.8, zorder=6)

    lg = fig.add_axes([0.765, 0.36, 0.23, 0.46])
    lg.set_axis_off()
    lg.text(0, 1, f"Supply configuration types (k = {len(types)})", fontsize=7.5, fontweight="bold", va="top")
    lg.text(0, 0.93, "Community level, n = 666", fontsize=6.5, va="top")
    for i, t in enumerate(types):
        y = 0.82 - i * 0.095
        lg.add_patch(Rectangle((0, y - 0.03), 0.08, 0.06, fc=COLOR[t], ec="none", transform=lg.transAxes))
        lg.text(0.12, y, f"{letter[t]}: {t} ({total[t] / total.sum():.1%})", fontsize=6.5, va="center")
    y = 0.82 - len(types) * 0.095
    lg.plot([0, 0.08], [y, y], color="#1A1A1A", lw=1.0, transform=lg.transAxes)
    lg.text(0.12, y, f"OD clusters (directed Leiden, Q = {q:.3f})", fontsize=6.5, va="center")
    lg.plot([0.04], [y - 0.095], "o", ms=4, color="black", mec="white", mew=0.6, transform=lg.transAxes)
    lg.text(0.12, y - 0.095, "Cluster centre (largest inflow\nfrom cluster members)", fontsize=6.5, va="center")

    for ext in ("pdf", "svg"):
        fig.savefig(out / f"fig_types_cluster.{ext}")
    fig.savefig(out / "fig_types_cluster.png", dpi=400)
    print(f"modularity {q:.3f}")
    print(stats.join(share.round(3).add_suffix("_share")).to_string())


if __name__ == "__main__":
    main()
