"""Four-panel relative supply-shift figure reconstructed from ``mech_no_pair``.

Panel a shows the signs of shifts in the three frozen PCs relative to typical
external aggregation. Panel b compares the nine-indicator shift magnitude
across baseline supply configurations. Panel c maps living-circle shifts with
continuous cluster envelopes. Panel d relates baseline-configuration diversity
to mean shift, with outgoing OD volume encoded by bubble area.

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
from shapely.geometry import shape
from shapely.ops import polylabel, unary_union

TYPE_ORDER = (0, 1, 2, 3)
TYPE_NAMES = ("scarcity", "chc-based", "sh-based", "abundance")
TYPE_COLORS = ("#3B82B4", "#58A847", "#F28E1C", "#E34234")
CLUSTER_ORDER = (4, 3, 5, 2, 1, 6)
CLUSTER_IDS = (1, 2, 3, 4, 5, 6)
CLUSTER_COLORS = {
    1: "#0072B2",
    2: "#009E73",
    3: "#CC79A7",
    4: "#56B4E9",
    5: "#E69F00",
    6: "#D55E00",
}


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
        "axes.spines.top": False,
        "axes.spines.right": False,
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


def cluster_relation_summary(
    data: pd.DataFrame,
    cluster: np.ndarray,
    W: np.ndarray,
    cluster_ids: tuple[int, ...] = CLUSTER_IDS,
) -> pd.DataFrame:
    """Compute cluster diversity, mean shift, and outgoing OD volume."""
    if len(data) != len(cluster) or W.shape != (len(data), len(data)):
        raise ValueError("node rows, cluster labels, and OD matrix must align")
    off_diagonal = W.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    origin_outflow = off_diagonal.sum(axis=1)
    baseline_type = data.baseline_type.to_numpy(dtype=int)
    rows: list[dict[str, float | int]] = []
    for group in cluster_ids:
        member = cluster == group
        weights = origin_outflow[member]
        outgoing_flow = float(weights.sum())
        if outgoing_flow <= 0:
            raise ValueError(f"cluster {group} has no outgoing OD flow")
        group_types = baseline_type[member]
        proportions = np.array([
            weights[group_types == supply_type].sum() for supply_type in TYPE_ORDER
        ], dtype=float) / outgoing_flow
        rows.append({
            "cluster": group,
            "configuration_diversity": float(1.0 - np.sum(proportions ** 2)),
            "mean_change": float(data.loc[member, "change_9"].mean()),
            "outgoing_flow": outgoing_flow,
            "community_n": int(member.sum()),
        })
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


def cluster_outline(geometries: list[object]) -> object:
    """Return the largest connected polygon as one continuous cluster outline."""
    if not geometries:
        raise ValueError("cluster outline requires at least one community geometry")
    merged = unary_union(geometries)
    polygons = [merged] if merged.geom_type == "Polygon" else [
        item for item in getattr(merged, "geoms", []) if item.geom_type == "Polygon"
    ]
    outline = max(polygons, key=lambda item: item.area) if polygons else merged.convex_hull
    if outline.geom_type != "Polygon" or outline.is_empty:
        raise ValueError("cluster outline could not be represented by one polygon")
    min_x, min_y, max_x, max_y = outline.bounds
    span = max(max_x - min_x, max_y - min_y)
    return outline.simplify(span * 0.0005, preserve_topology=True)


def map_panel(
    ax: plt.Axes,
    geojson_path: Path,
    values: dict[int, float],
    clusters: dict[int, int],
) -> mpl.cm.ScalarMappable:
    geo = json.loads(geojson_path.read_text(encoding="utf-8"))
    valid_values = np.asarray(list(values.values()), dtype=float)
    norm = Normalize(vmin=float(valid_values.min()), vmax=float(valid_values.max()))
    cmap = mpl.colormaps["YlOrRd"]
    polygons, colors = [], []
    cluster_parts: dict[int, list[object]] = {group: [] for group in CLUSTER_IDS}
    for feature in geo["features"]:
        community = int(feature["properties"]["community_id"])
        parts = polygon_parts(feature["geometry"])
        polygons.extend(parts)
        colors.extend([cmap(norm(values[community]))] * len(parts))
        if community in clusters:
            geometry = shape(feature["geometry"])
            group = clusters[community]
            if group in cluster_parts:
                cluster_parts[group].append(geometry if geometry.is_valid else geometry.buffer(0))
    collection = PolyCollection(
        polygons, facecolors=colors, edgecolors="#A7A7A7", linewidths=0.14,
        rasterized=True, zorder=2,
    )
    ax.add_collection(collection)
    all_points = np.concatenate(polygons)
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    ax.set_aspect(1 / np.cos(np.deg2rad(all_points[:, 1].mean())), adjustable="datalim")
    ax.set_axis_off()
    for group in CLUSTER_IDS:
        outline = cluster_outline(cluster_parts[group])
        points = np.asarray(outline.exterior.coords, dtype=float)
        ax.plot(points[:, 0], points[:, 1], color="white", linewidth=2.4,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot(points[:, 0], points[:, 1], color=CLUSTER_COLORS[group], linewidth=1.25,
                solid_capstyle="round", solid_joinstyle="round", zorder=4)
        min_x, min_y, max_x, max_y = outline.bounds
        label_point = polylabel(outline, tolerance=max(max_x - min_x, max_y - min_y) * 0.002)
        ax.scatter(label_point.x, label_point.y, s=32, color=CLUSTER_COLORS[group],
                   edgecolor="white", linewidth=0.55, zorder=5)
        ax.text(label_point.x, label_point.y, str(group), ha="center", va="center",
                fontsize=5, fontweight="bold", color="white", zorder=6)
    ax.annotate("N", xy=(0.965, 0.92), xytext=(0.965, 0.77), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=5.5, arrowprops={"arrowstyle": "-|>", "lw": 0.65})
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def bubble_sizes(values: np.ndarray, low: float = 42.0, high: float = 210.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span <= 0:
        return np.full_like(values, (low + high) / 2)
    return low + (values - values.min()) / span * (high - low)


def cluster_relation(ax: plt.Axes, table: pd.DataFrame) -> None:
    diversity = table.configuration_diversity.to_numpy(dtype=float)
    change = table.mean_change.to_numpy(dtype=float)
    outgoing_flow = table.outgoing_flow.to_numpy(dtype=float)
    sizes = bubble_sizes(outgoing_flow)
    pearson = pearsonr(diversity, change)
    spearman = spearmanr(diversity, change)
    fit = np.polyfit(diversity, change, 1)
    x = np.linspace(diversity.min(), diversity.max(), 100)
    ax.plot(x, np.polyval(fit, x), color="#64748B", linewidth=0.75, zorder=1)
    label_offsets = {
        1: (0.007, 0.007),
        2: (-0.015, 0.015),
        3: (0.000, -0.010),
        4: (0.006, -0.012),
        5: (0.007, -0.009),
        6: (-0.006, -0.010),
    }
    for row, size in zip(table.itertuples(), sizes):
        color = CLUSTER_COLORS[int(row.cluster)]
        ax.scatter(row.configuration_diversity, row.mean_change, s=size, color=color,
                   edgecolor="white", linewidth=0.8, alpha=0.92, zorder=3)
        dx, dy = label_offsets[int(row.cluster)]
        ax.text(row.configuration_diversity + dx, row.mean_change + dy, str(int(row.cluster)),
                ha="right" if dx < 0 else "left", va="center", fontsize=5,
                fontweight="bold", color=color, zorder=4)
    ax.margins(x=0.16, y=0.18)
    reference_flow = np.quantile(outgoing_flow, (0.0, 0.5, 1.0))
    reference_size = bubble_sizes(reference_flow, low=42.0, high=210.0)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#94A3B8",
               markeredgecolor="white", markeredgewidth=0.7, markersize=np.sqrt(size))
        for size in reference_size
    ]
    legend = ax.legend(handles, [f"{value / 1000:.0f}k" for value in reference_flow],
                       title="Outgoing flow", loc="upper left", ncol=3,
                       handletextpad=0.45, columnspacing=0.8, borderaxespad=0.2,
                       fontsize=5, title_fontsize=5)
    legend._legend_box.align = "left"
    ax.text(0.02, 0.03, f"r = {pearson.statistic:.3f}, p = {pearson.pvalue:.3f}\n"
            f"ρ = {spearman.statistic:.3f}, p = {spearman.pvalue:.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5, color="#404040")
    ax.set_xlabel("Baseline configuration diversity")
    ax.set_ylabel("Cluster mean relative-shift magnitude")


def run_alignment_audit(fig: plt.Figure, axes: list[plt.Axes], panel_ids: list[str], output: Path) -> None:
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as exc:
        raise RuntimeError("audit_panel_alignment.py must be available on PYTHONPATH for final export") from exc
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=panel_ids,
        json_out=output.with_suffix(".alignment.json"),
        overlay_svg=output.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )


def panel_heading(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=8.2,
            fontweight="bold", ha="left", va="bottom")
    ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=5)


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
    relation = cluster_relation_summary(nodes, cluster, W)
    relation.to_csv(args.output.with_name(args.output.name + "_cluster_relation.csv"), index=False)

    nature_style()
    fig = plt.figure(figsize=(7.2047, 5.50), facecolor="white")
    grid = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.97, top=0.945, bottom=0.12,
        width_ratios=(1.58, 1.0), height_ratios=(1.0, 1.25),
        hspace=0.48, wspace=0.24,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    alluvial(ax_a, data)
    boxplot(ax_b, data)
    values = dict(zip(nodes.community_id.astype(int), nodes.change_9.astype(float)))
    groups = dict(zip(nodes.community_id.astype(int), cluster))
    scalar = map_panel(ax_map, args.geojson, values, groups)
    colorbar_axis = ax_map.inset_axes([0.12, -0.095, 0.76, 0.030])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=5, length=1.8, width=0.4)
    colorbar.ax.set_title("Nine-indicator relative-shift magnitude", fontsize=5, pad=2)
    cluster_relation(ax_d, relation)

    panel_heading(ax_a, "a", "PC shift relative to typical aggregation")
    panel_heading(ax_b, "b", "Relative-shift magnitude by configuration")
    panel_heading(ax_map, "c", "Community supply shift and cluster envelopes")
    panel_heading(ax_d, "d", "Cluster diversity and mean supply shift")
    ax_a.text(0.5, -0.10, "Rise = positive · Stable = zero · Decline = negative\n"
              "Band width = number of living circles", transform=ax_a.transAxes,
              ha="center", va="top", fontsize=5)
    fig.text(0.5, 0.012, f"Seed {args.seed} · Spatially cross-fitted shift relative to typical external aggregation\n"
             "Diversity = 1 − Σp² across four baseline configurations, weighted by origin OD volume\n"
             "Cluster outlines and bubbles share colors · Bubble area is outgoing OD volume with self flows excluded",
             ha="center", va="bottom", fontsize=5, color="#64748B")

    axes = [ax_a, ax_b, ax_map, ax_d]
    panel_ids = ["a", "b", "c", "d"]
    run_alignment_audit(fig, axes, panel_ids, args.output)
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {args.output.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
