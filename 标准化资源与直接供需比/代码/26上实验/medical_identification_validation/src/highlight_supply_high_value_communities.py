"""Mark the highest-value administrative communities on the nine supply maps."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

import visualize_supply_spatial_distribution as source


ROOT = Path(__file__).resolve().parents[3]
DATA = source.DATA_DIR
OUT = source.OUTPUT_DIR
sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
    }
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = gpd.read_file(DATA / f"community_supply_indicators_2018{source.SUFFIX}.gpkg").to_crs(32650)
    values = pd.read_csv(DATA / f"community_supply_indicators_2018{source.SUFFIX}.csv")
    frame = geometry[["community_id", "geometry"]].merge(values, on="community_id", validate="one_to_one")
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=geometry.crs).sort_values("community_id")

    fig, axes = plt.subplots(3, 3, figsize=(183 / 25.4, 150 / 25.4))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.06, top=0.91, wspace=0.06, hspace=0.10)
    cmap = ListedColormap(source.COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)
    boundary = frame.dissolve().boundary
    bounds = frame.total_bounds
    colorbar_axes = []
    highlighted = []

    for row, (metric, english, chinese) in enumerate(source.METRICS):
        for col, tier in enumerate(source.TIERS):
            ax = axes[row, col]
            letter = "abcdefghi"[row * 3 + col]
            column = source.metric_column(metric, tier)
            classes, labels, _ = source.classify_quartiles(frame[column])
            frame.assign(_class=classes).plot(
                column="_class", ax=ax, cmap=cmap, norm=norm,
                edgecolor="#FFFFFF", linewidth=0.12, antialiased=True,
            )
            boundary.plot(ax=ax, color="#334155", linewidth=0.45, zorder=3)

            maximum = float(frame[column].max())
            top = frame.loc[np.isclose(frame[column], maximum, rtol=0, atol=1e-9)]
            top.boundary.plot(ax=ax, color="#E85D04", linewidth=1.25, zorder=6)
            points = top.geometry.representative_point()
            ax.scatter(points.x, points.y, s=19, facecolor="#E85D04",
                       edgecolor="white", linewidth=0.55, zorder=7)
            for _, item in top.iterrows():
                point = item.geometry.representative_point()
                highlighted.append({
                    "panel": letter, "tier": tier, "indicator": metric,
                    "community_id": int(item.community_id),
                    "community_name": item.community_name,
                    "SQCODE": str(item.SQCODE), "value": maximum,
                    "longitude": float(gpd.GeoSeries([point], crs=frame.crs).to_crs(4326).x.iloc[0]),
                    "latitude": float(gpd.GeoSeries([point], crs=frame.crs).to_crs(4326).y.iloc[0]),
                    "population_demand": float(item.population_demand),
                    "facility_count": int(item[f"facility_count_{tier}"]),
                    "capacity_sum": float(item[f"capacity_sum_{tier}"]),
                    "within_15min_population_pct": float(item[f"within_15min_population_pct_{tier}"]),
                })

            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_axis_off()
            ax.set_gid(letter)
            ax.text(-0.035, 1.01, letter, transform=ax.transAxes, fontsize=8,
                    fontweight="bold", va="bottom", ha="left")
            if row == 0:
                ax.set_title(source.TIER_LABELS[tier], fontsize=7.2, fontweight="bold", pad=3.5)
            if col == 0:
                ax.text(-0.12, 0.5, f"{english}\n{chinese}", transform=ax.transAxes,
                        rotation=90, rotation_mode="anchor", ha="center", va="center",
                        fontsize=6.4, color="#1F2937")
            name = "、".join(top.community_name.tolist())
            ax.text(0.98, 0.98, f"最高：{name} {maximum:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=5.2, color="#B54708",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.88})

            cax = ax.inset_axes([0.08, 0.035, 0.78, 0.032])
            colorbar_axes.append(cax)
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                              orientation="horizontal", ticks=np.arange(0, 5))
            cb.ax.set_xticklabels(labels, fontsize=5.2)
            cb.ax.tick_params(length=0, pad=1)
            cb.outline.set_linewidth(0.35)

    source.add_scale_and_north(axes[2, 2], bounds)
    fig.suptitle(f"2018年行政社区医疗设施供给指标：最高值定位（{source.DEMAND_LABEL}）", fontsize=9,
                 fontweight="bold", y=0.972)
    fig.text(0.52, 0.019,
             "道路：© OpenStreetMap contributors（Geofabrik，2019-01-01，ODbL 1.0）；橙色=最高值行政社区。",
             ha="center", va="center", fontsize=5.5, color="#475569")
    fig.canvas.draw()
    base = OUT / "supply_spatial_distribution_highlights"
    require_matplotlib_panel_alignment(
        fig, axes=list(axes.flat), panel_ids=list("abcdefghi"),
        row_groups=[list("abc"), list("def"), list("ghi")],
        column_groups=[list("adg"), list("beh"), list("cfi")],
        exclude_axes=colorbar_axes,
        json_out=base.with_suffix(".alignment.json"),
        overlay_svg=base.with_suffix(".alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True,
    )
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(highlighted).to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
