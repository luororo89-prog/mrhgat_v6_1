"""Four-panel relative supply-shift figure reconstructed from ``mech_no_pair``.

Panel a shows the signs of shifts in the three frozen PCs relative to typical
external aggregation.  Panel b compares the nine-indicator shift magnitude
across baseline supply configurations.  Panel c maps the living-circle shift and summarizes it by within-cluster destination
configuration.  Panel d relates cluster configuration diversity to mean shift.

The plotting script only consumes reconstruction outputs and fixed descriptive
inputs.  It does not refit the decoder or alter model results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

TYPE_ORDER = (0, 1, 2, 3)
TYPE_NAMES = ("scarcity", "chc-based", "sh-based", "abundance")
TYPE_COLORS = ("#3B82B4", "#58A847", "#F28E1C", "#E34234")
CLUSTER_ORDER = (4, 3, 5, 2, 1, 6)


def nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6,
        "axes.titlesize": 6.5,
        "axes.labelsize": 6,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.fontsize": 5.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def holm(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def significance(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def weighted_destination_summary(
    change: np.ndarray,
    node_type: np.ndarray,
    cluster: np.ndarray,
    W: np.ndarray,
    cluster_ids: tuple[int, ...] = CLUSTER_ORDER,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    n = len(change)
    off = ~np.eye(n, dtype=bool)
    for group in cluster_ids:
        member = cluster == group
        within = W * (member[:, None] & member[None, :] & off)
        denominator = within.sum(axis=1)
        for destination_type in TYPE_ORDER:
            flow = within[:, node_type == destination_type].sum(axis=1)
            share = np.divide(flow, denominator, out=np.zeros(n), where=denominator > 0)
            valid = member & (share > 0)
            value = float(np.average(change[valid], weights=share[valid])) if valid.any() else np.nan
            rows.append({"cluster": group, "destination_type": destination_type,
                         "weighted_change": value, "weight_sum": float(share[valid].sum()),
                         "origin_n": int(valid.sum())})
    return pd.DataFrame(rows)


def alluvial(ax: plt.Axes, data: pd.DataFrame) -> None:
    pc = [f"delta_PC{i}" for i in (1, 2, 3)]
    states = np.column_stack([
        data.baseline_type.to_numpy(dtype=int),
        *[np.where(data[column].to_numpy() < -1e-12, 0,
                   np.where(data[column].to_numpy() > 1e-12, 2, 1)) for column in pc],
    ])
    n_total = len(data)
    gap = max(8, int(round(n_total * 0.035)))
    xs, width = [0.0, 1.18, 2.36, 3.54], 0.16
    positions: list[dict[int, tuple[float, int]]] = []
    for layer in range(4):
        values = TYPE_ORDER if layer == 0 else (0, 1, 2)
        heights = {value: int((states[:, layer] == value).sum()) for value in values}
        total = sum(heights.values()) + gap * (len(values) - 1)
        cursor = (n_total + 3 * gap - total) / 2
        layer_positions: dict[int, tuple[float, int]] = {}
        for value in values:
            layer_positions[value] = (cursor, heights[value])
            cursor += heights[value] + gap
        positions.append(layer_positions)
    for layer in range(3):
        left = {key: value[0] for key, value in positions[layer].items()}
        right = {key: value[0] for key, value in positions[layer + 1].items()}
        for supply_type in TYPE_ORDER:
            for before_state in positions[layer]:
                for after_state in positions[layer + 1]:
                    count = int(((states[:, 0] == supply_type) & (states[:, layer] == before_state) &
                                 (states[:, layer + 1] == after_state)).sum())
                    if not count:
                        continue
                    y0, y1 = left[before_state], right[after_state]
                    left[before_state] += count
                    right[after_state] += count
                    x0, x1 = xs[layer] + width, xs[layer + 1]
                    handle = (x1 - x0) * 0.46
                    vertices = [(x0, y0), (x0 + handle, y0), (x1 - handle, y1), (x1, y1),
                                (x1, y1 + count), (x1 - handle, y1 + count),
                                (x0 + handle, y0 + count), (x0, y0 + count), (x0, y0)]
                    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
                    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=TYPE_COLORS[supply_type],
                                           alpha=0.42, edgecolor="none"))
    for layer in range(4):
        for value, (bottom, height) in positions[layer].items():
            color = TYPE_COLORS[value] if layer == 0 else ("#D8D8D8", "#B7B7B7", "#777777")[value]
            ax.add_patch(Rectangle((xs[layer], bottom), width, height, facecolor=color, edgecolor="none"))
            if layer == 0:
                ax.text(xs[layer] - 0.06, bottom + height / 2, TYPE_NAMES[value], ha="right", va="center")
            elif height >= max(20, int(n_total * 0.06)):
                ax.text(xs[layer] + width / 2, bottom + height / 2, ("Decline", "Stable", "Rise")[value],
                        rotation=90, ha="center", va="center", rotation_mode="anchor", fontsize=5,
                        color="#3F3F3F" if value < 2 else "white")
    for layer, label in enumerate(("Baseline config.", "PC1", "PC2", "PC3")):
        ax.text(xs[layer] + width / 2, n_total + 3 * gap + 16, label, ha="center", va="top")
    ax.set_xlim(-1.25, 3.85)
    ax.set_ylim(n_total + 3 * gap + 50, -55)
    ax.axis("off")


def boxplot(ax: plt.Axes, data: pd.DataFrame) -> None:
    groups = [data.loc[data.baseline_type == supply_type, "change_9"].to_numpy() for supply_type in TYPE_ORDER]
    box = ax.boxplot(groups, patch_artist=True, widths=0.5, showfliers=False,
                     medianprops={"color": "#F7B32B", "linewidth": 1.0},
                     whiskerprops={"color": "#666666", "linewidth": 0.65},
                     capprops={"color": "#666666", "linewidth": 0.65},
                     boxprops={"color": "#555555", "linewidth": 0.65})
    for patch, color in zip(box["boxes"], TYPE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    means = [group.mean() for group in groups]
    ax.scatter(np.arange(1, 5), means, s=14, facecolor="white", edgecolor="#666666", linewidth=0.65, zorder=4)
    omnibus = kruskal(*groups)
    ax.text(0.02, 0.98, f"Kruskal–Wallis p = {omnibus.pvalue:.2g}", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.2)
    pairs, raw = [], []
    for left in range(4):
        for right in range(left + 1, 4):
            pairs.append((left, right))
            raw.append(mannwhitneyu(groups[left], groups[right], alternative="two-sided").pvalue)
    adjusted = holm(raw)
    selected = [(pair, value) for pair, value in zip(pairs, adjusted) if value < 0.05]
    whiskers = []
    for group in groups:
        q1, q3 = np.quantile(group, (0.25, 0.75))
        iqr = q3 - q1
        inside = group[(group >= q1 - 1.5 * iqr) & (group <= q3 + 1.5 * iqr)]
        whiskers.append((inside.min(), inside.max()))
    low = min(bound[0] for bound in whiskers)
    high = max(bound[1] for bound in whiskers)
    span = max(high - low, 1e-6)
    step = span * 0.18
    for level, ((left, right), value) in enumerate(selected):
        y = high + step * (level + 1)
        ax.plot([left + 1, left + 1, right + 1, right + 1], [y, y + step * 0.2, y + step * 0.2, y],
                color="#555555", linewidth=0.55)
        ax.text(right + 1.06, y + step * 0.20, significance(value), ha="left", va="center",
                fontsize=5, fontweight="bold")
    ax.set_ylim(max(0, low - span * 0.12), high + step * (len(selected) + 2.0))
    ax.set_xticks(range(1, 5), TYPE_NAMES)
    ax.set_ylabel("Nine-indicator relative-shift magnitude")
    ax.text(0.5, -0.18, "White circles indicate means · Outliers omitted visually\n"
            "Pairwise tests use Holm correction",
            transform=ax.transAxes, ha="center", va="top", fontsize=5, color="#444444")


def polygon_parts(geometry: dict) -> list[np.ndarray]:
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    return [np.asarray(polygon[0], dtype=float) for polygon in polygons if polygon and polygon[0]]


def polygon_centroid(points: np.ndarray) -> tuple[float, float]:
    x, y = points[:, 0], points[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = cross.sum() / 2
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x[:-1] + x[1:]) * cross).sum() / (6 * area)), float(((y[:-1] + y[1:]) * cross).sum() / (6 * area))


def map_panel(ax: plt.Axes, geojson_path: Path, values: dict[int, float], clusters: dict[int, int]) -> tuple[mpl.cm.ScalarMappable, dict[int, tuple[float, float]]]:
    geo = json.loads(geojson_path.read_text(encoding="utf-8"))
    valid_values = np.asarray(list(values.values()), dtype=float)
    norm = Normalize(vmin=float(valid_values.min()), vmax=float(valid_values.max()))
    cmap = mpl.colormaps["YlOrRd"]
    polygons, colors = [], []
    centroid_rows: dict[int, list[tuple[float, float]]] = {}
    for feature in geo["features"]:
        community = int(feature["properties"]["community_id"])
        parts = polygon_parts(feature["geometry"])
        polygons.extend(parts)
        colors.extend([cmap(norm(values[community]))] * len(parts))
        if community in clusters:
            centroid_rows.setdefault(clusters[community], []).extend(polygon_centroid(part) for part in parts)
    collection = PolyCollection(
        polygons, facecolors=colors, edgecolors="#333333", linewidths=0.18,
        rasterized=True, zorder=2,
    )
    ax.add_collection(collection)
    all_points = np.concatenate(polygons)
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    ax.set_aspect(1 / np.cos(np.deg2rad(all_points[:, 1].mean())))
    ax.set_axis_off()
    centres = {group: tuple(np.mean(points, axis=0)) for group, points in centroid_rows.items()}
    for group in range(1, 7):
        if group in centres:
            ax.text(*centres[group], str(group), ha="center", va="center", fontsize=5.5, fontweight="bold",
                    color="#333333", zorder=4)
    ax.annotate("N", xy=(0.965, 0.92), xytext=(0.965, 0.77), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=5.5, arrowprops={"arrowstyle": "-|>", "lw": 0.65})
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap), centres


def cluster_bars(ax: plt.Axes, group: int, summary: pd.DataFrame, y_max: float) -> None:
    table = summary.loc[summary.cluster == group].set_index("destination_type").reindex(TYPE_ORDER)
    values = table.weighted_change.to_numpy(dtype=float)
    for index, (value, color) in enumerate(zip(values, TYPE_COLORS)):
        if np.isfinite(value):
            ax.bar(index, value, color=color, width=0.68)
            ax.text(index, value + y_max * 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=5)
        else:
            ax.text(index, y_max * 0.20, "No flow", ha="center", va="bottom", rotation=90,
                    rotation_mode="anchor", fontsize=5, color="#777777")
    ax.set_title(f"Cluster {group}", loc="left", fontweight="bold", pad=1.5)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(4), ("scarcity", "CHC", "SH", "abund."), rotation=35,
                  ha="right", rotation_mode="anchor")


def cluster_relation(ax: plt.Axes, data: pd.DataFrame, cluster: np.ndarray, W: np.ndarray) -> None:
    rows = []
    origin_volume = W.sum(axis=1)
    for group in range(1, 7):
        member = cluster == group
        weight = origin_volume[member]
        proportions = np.array([weight[data.loc[member, "baseline_type"].to_numpy(dtype=int) == supply_type].sum()
                                for supply_type in TYPE_ORDER], dtype=float)
        proportions /= proportions.sum()
        rows.append({"cluster": group, "diversity": 1 - np.sum(proportions ** 2),
                     "change": float(data.loc[member, "change_9"].mean())})
    table = pd.DataFrame(rows)
    pearson = pearsonr(table.diversity, table.change)
    spearman = spearmanr(table.diversity, table.change)
    fit = np.polyfit(table.diversity, table.change, 1)
    x = np.linspace(table.diversity.min(), table.diversity.max(), 100)
    ax.plot(x, np.polyval(fit, x), color="#3E88B8", linewidth=0.8)
    ax.scatter(table.diversity, table.change, s=18, color="#3E88B8", edgecolor="white", linewidth=0.45, zorder=3)
    ax.margins(x=0.10, y=0.12)
    label_offsets = {1: (-12, 6), 2: (-12, -8), 4: (5, 7), 6: (10, -18)}
    for row in table.itertuples():
        offset = label_offsets.get(row.cluster, (3, 2))
        ax.annotate(str(row.cluster), (row.diversity, row.change), xytext=offset,
                    textcoords="offset points", fontsize=5)
    ax.text(0.03, 0.03, f"r = {pearson.statistic:.3f}, p = {pearson.pvalue:.3f}\n"
            f"ρ = {spearman.statistic:.3f}, p = {spearman.pvalue:.3f}\nn = 6 clusters",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5)
    ax.set_xlabel("Cluster type diversity")
    ax.set_ylabel("Cluster mean relative-shift magnitude")
    ax.text(0.5, -0.22, "Diversity uses four configuration shares\nweighted by origin OD volume.",
            transform=ax.transAxes, ha="center", va="top", fontsize=5, color="#677384")


def run_alignment_audit(fig: plt.Figure, axes: list[plt.Axes], panel_ids: list[str], output: Path) -> None:
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as exc:
        raise RuntimeError("audit_panel_alignment.py must be available on PYTHONPATH for final export") from exc
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=panel_ids,
        row_groups=[panel_ids[:3], panel_ids[3:]],
        json_out=output.with_suffix(".alignment.json"),
        overlay_svg=output.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--fixed-split", type=Path, required=True)
    parser.add_argument("--cluster-membership", type=Path, required=True)
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(args.reconstruction / "node_supply_change.csv")
    nodes = nodes.loc[nodes.seed == args.seed].copy()
    with np.load(args.fixed_split) as split:
        ids = split["ids"].astype(int)
        W = split["W"].astype(float)
    nodes = nodes.set_index("community_id").reindex(ids).reset_index()
    if nodes.isna().any().any():
        raise ValueError("reconstruction does not match fixed node order")
    membership = pd.read_csv(args.cluster_membership).set_index("community_id").reindex(ids)
    if membership.cluster.isna().any():
        raise ValueError("cluster membership does not cover fixed node order")
    cluster = membership.cluster.to_numpy(dtype=int)
    data = nodes.loc[nodes.eligible.astype(bool)].copy()
    summary = weighted_destination_summary(nodes.change_9.to_numpy(), nodes.baseline_type.to_numpy(dtype=int),
                                           cluster, W)
    summary.to_csv(args.output.with_name(args.output.name + "_cluster_destination.csv"), index=False)

    nature_style()
    fig = plt.figure(figsize=(7.2047, 6.7717), facecolor="white")
    ax_a = fig.add_axes([0.055, 0.705, 0.43, 0.255])
    ax_b = fig.add_axes([0.57, 0.735, 0.375, 0.22])
    ax_map = fig.add_axes([0.17, 0.24, 0.43, 0.24])
    ax_d = fig.add_axes([0.755, 0.255, 0.19, 0.235])
    bar_positions = {
        4: [0.08, 0.535, 0.16, 0.095],
        3: [0.29, 0.535, 0.16, 0.095],
        5: [0.50, 0.535, 0.16, 0.095],
        2: [0.08, 0.09, 0.16, 0.095],
        1: [0.29, 0.09, 0.16, 0.095],
        6: [0.50, 0.09, 0.16, 0.095],
    }
    bar_axes = {group: fig.add_axes(position) for group, position in bar_positions.items()}

    alluvial(ax_a, data)
    boxplot(ax_b, data)
    values = dict(zip(nodes.community_id.astype(int), nodes.change_9.astype(float)))
    groups = dict(zip(nodes.community_id.astype(int), cluster))
    scalar, _centres = map_panel(ax_map, args.geojson, values, groups)
    colorbar_axis = fig.add_axes([0.145, 0.225, 0.48, 0.012])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=5, length=1.8, width=0.4)
    y_max = max(summary.weighted_change.max() * 1.18, 0.1)
    for index, group in enumerate(CLUSTER_ORDER):
        cluster_bars(bar_axes[group], group, summary, y_max)
        if index % 3:
            bar_axes[group].set_yticklabels([])
        else:
            bar_axes[group].set_ylabel("Weighted mean shift", labelpad=1)
    cluster_relation(ax_d, nodes, cluster, W)

    fig.text(0.055, 0.975, "a. PC shift relative to type-specific range", fontsize=7.5,
             fontweight="bold", va="top")
    fig.text(0.57, 0.975, "b. Relative-shift magnitude by configuration", fontsize=7.5, fontweight="bold", va="top")
    fig.text(0.29, 0.699, "Rise = positive · Stable = zero · Decline = negative\n"
             "Band width = number of living circles", ha="center", va="top", fontsize=5)
    fig.text(0.055, 0.665, "c. Living-circle shift and OD destination configuration",
             fontsize=7.5, fontweight="bold", va="top")
    fig.text(0.755, 0.505, "d. Cluster-level relationship", fontsize=7.5, fontweight="bold", va="top")
    fig.text(0.5, 0.006, f"Seed {args.seed} · Spatially cross-fitted shift relative to the type-specific hospital-range reference\n"
             "Bars weight each origin’s shift by its within-cluster OD share to the destination configuration; self flows excluded.",
             ha="center", va="bottom", fontsize=5, color="#64748B")

    ordered_bar_axes = [bar_axes[group] for group in CLUSTER_ORDER]
    panel_ids = [f"cluster{group}" for group in CLUSTER_ORDER]
    run_alignment_audit(fig, ordered_bar_axes, panel_ids, args.output)
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {args.output.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
