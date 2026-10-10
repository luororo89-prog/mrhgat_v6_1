"""绘制社区15分钟邻里医疗设施供给指标的空间分布并执行一致性检查。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEMAND_SOURCE = os.environ.get("SUPPLY_DEMAND_SOURCE", "census_expanded")
if DEMAND_SOURCE not in {"census_expanded", "mobile_users"}:
    raise ValueError(f"Unknown population demand source: {DEMAND_SOURCE}")
SUFFIX = "_mobile_users" if DEMAND_SOURCE == "mobile_users" else ""
DEMAND_LABEL = "联通用户数" if DEMAND_SOURCE == "mobile_users" else "七普扩样人口"
DATA_DIR = ROOT / "26上实验" / "medical_identification_validation" / "results" / "supply_indicators_2018_osm_190101"
INPUT = DATA_DIR / f"community_supply_indicators_2018{SUFFIX}.gpkg"
OUTPUT_DIR = DATA_DIR / f"spatial_visualization{SUFFIX}"
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


TIERS = ["CHC", "SH", "TH"]
TIER_LABELS = {
    "CHC": "CHC（基层）",
    "SH": "SH（二级医院）",
    "TH": "TH（三级医院）",
}
METRICS = [
    ("RA", "Resource availability", "资源可用性"),
    ("SA", "Spatial accessibility", "空间可达性（每万人）"),
    ("TC", "Travel convenience", "出行便利性"),
]
COLORS = ["#E5E7EB", "#D7E9F5", "#9ECAE1", "#4292C6", "#084594"]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def metric_column(metric: str, tier: str) -> str:
    if metric == "SA":
        return f"SA_{tier}_per_10000"
    return f"{metric}_{tier}"


def classify_quartiles(values: pd.Series) -> tuple[np.ndarray, list[str], list[float]]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    positive = x[np.isfinite(x) & (x > 0)]
    classes = np.zeros(len(x), dtype=float)
    classes[~np.isfinite(x)] = np.nan
    if len(positive) == 0:
        return classes, ["0", "—", "—", "—", "—"], [0.0, 0.0, 0.0, 0.0]
    q25, q50, q75, vmax = np.quantile(positive, [0.25, 0.50, 0.75, 1.0])
    classes[(x > 0) & (x <= q25)] = 1
    classes[(x > q25) & (x <= q50)] = 2
    classes[(x > q50) & (x <= q75)] = 3
    classes[x > q75] = 4

    def fmt(value: float) -> str:
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        if value >= 1:
            return f"{value:.2f}"
        if value >= 0.1:
            return f"{value:.2f}"
        return f"{value:.3f}"

    labels = ["0", f"≤{fmt(q25)}", f"≤{fmt(q50)}", f"≤{fmt(q75)}", f">{fmt(q75)}"]
    return classes, labels, [float(q25), float(q50), float(q75), float(vmax)]


def queen_edges(frame: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    src: list[int] = []
    dst: list[int] = []
    index = frame.sindex
    for i, geom in enumerate(frame.geometry):
        for j in index.query(geom, predicate="touches"):
            j = int(j)
            if i < j:
                src.extend([i, j])
                dst.extend([j, i])
    return np.asarray(src, dtype=int), np.asarray(dst, dtype=int)


def moran_i(values: np.ndarray, src: np.ndarray, dst: np.ndarray, permutations: int = 499) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    edge_valid = valid[src] & valid[dst]
    src, dst = src[edge_valid], dst[edge_valid]
    centered = np.zeros(len(x), dtype=float)
    centered[valid] = x[valid] - x[valid].mean()
    denominator = float(np.square(centered[valid]).sum())
    if valid.sum() < 3 or denominator == 0 or len(src) == 0:
        return float("nan"), float("nan")
    observed = int(valid.sum()) / len(src) * float(np.sum(centered[src] * centered[dst])) / denominator
    rng = np.random.Generator(np.random.PCG64(20260930))
    permuted = np.empty(permutations, dtype=float)
    for k in range(permutations):
        xp = np.zeros(len(x), dtype=float)
        xp[valid] = rng.permutation(centered[valid])
        permuted[k] = int(valid.sum()) / len(src) * float(np.sum(xp[src] * xp[dst])) / denominator
    p_value = (1 + int(np.sum(np.abs(permuted) >= abs(observed)))) / (permutations + 1)
    return float(observed), float(p_value)


def spearman(a: pd.Series, b: pd.Series) -> float:
    ra = pd.to_numeric(a, errors="coerce").rank(method="average")
    rb = pd.to_numeric(b, errors="coerce").rank(method="average")
    return float(ra.corr(rb))


def add_scale_and_north(ax, bounds) -> None:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    bar = 20000.0
    x0 = minx + width * 0.63
    y0 = miny + height * 0.055
    ax.plot([x0, x0 + bar], [y0, y0], color="#222222", lw=1.2, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0], [y0 - 650, y0 + 650], color="#222222", lw=0.8, zorder=5)
    ax.plot([x0 + bar, x0 + bar], [y0 - 650, y0 + 650], color="#222222", lw=0.8, zorder=5)
    ax.text(x0 + bar / 2, y0 + 1100, "20 km", ha="center", va="bottom", fontsize=5.5)
    ax.annotate(
        "N",
        xy=(0.93, 0.87),
        xytext=(0.93, 0.72),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#222222"),
    )


def make_figure(frame: gpd.GeoDataFrame) -> dict:
    width_in = 183.0 / 25.4
    height_in = 150.0 / 25.4
    fig, axes = plt.subplots(3, 3, figsize=(width_in, height_in), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.91, wspace=0.06, hspace=0.10)
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)
    panel_info: list[dict] = []
    panel_letters = iter("abcdefghi")
    colorbar_axes = []
    bounds = frame.total_bounds

    for row, (metric, english, chinese) in enumerate(METRICS):
        for col, tier in enumerate(TIERS):
            ax = axes[row, col]
            letter = next(panel_letters)
            column = metric_column(metric, tier)
            classes, labels, breaks = classify_quartiles(frame[column])
            plot_frame = frame.assign(_class=classes)
            plot_frame.plot(
                column="_class",
                ax=ax,
                cmap=cmap,
                norm=norm,
                edgecolor="#FFFFFF",
                linewidth=0.12,
                antialiased=True,
                missing_kwds={"color": "#F8FAFC", "edgecolor": "#CBD5E1", "hatch": "///"},
            )
            frame.dissolve().boundary.plot(ax=ax, color="#334155", linewidth=0.45, zorder=3)
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_axis_off()
            ax.set_gid(letter)
            ax.text(-0.035, 1.01, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
            if row == 0:
                ax.set_title(TIER_LABELS[tier], pad=3.5, fontweight="bold", fontsize=7.2)
            if col == 0:
                ax.text(
                    -0.12,
                    0.50,
                    f"{english}\n{chinese}",
                    transform=ax.transAxes,
                    rotation=90,
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="#1F2937",
                )
            numeric = pd.to_numeric(frame[column], errors="coerce")
            zero_count = int(numeric.eq(0).sum())
            missing_count = int(numeric.isna().sum())
            ax.text(0.98, 0.98, f"零值 n={zero_count}  无需求 n={missing_count}", transform=ax.transAxes, ha="right", va="top", fontsize=5.2, color="#475569")

            cax = ax.inset_axes([0.08, 0.035, 0.78, 0.032])
            colorbar_axes.append(cax)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=np.arange(0, 5))
            cb.ax.set_xticklabels(labels, fontsize=5.2)
            cb.ax.tick_params(length=0, pad=1)
            cb.outline.set_linewidth(0.35)
            panel_info.append(
                {
                    "panel": letter,
                    "metric": metric,
                    "tier": tier,
                    "column": column,
                    "zero_count": zero_count,
                    "missing_count": missing_count,
                    "positive_quartile_breaks": breaks,
                }
            )

    add_scale_and_north(axes[2, 2], bounds)
    fig.suptitle(f"2018年行政社区分层医疗设施供给指标空间分布（{DEMAND_LABEL}）", fontsize=9, fontweight="bold", y=0.972)
    fig.text(
        0.52,
        0.018,
        "道路：© OpenStreetMap contributors（Geofabrik，2019-01-01，ODbL 1.0）；灰色=0，斜线=无需求。",
        ha="center",
        va="center",
        fontsize=5.5,
        color="#475569",
    )
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig,
        axes=list(axes.flat),
        panel_ids=list("abcdefghi"),
        row_groups=[list("abc"), list("def"), list("ghi")],
        column_groups=[list("adg"), list("beh"), list("cfi")],
        exclude_axes=colorbar_axes,
        json_out=OUTPUT_DIR / "supply_spatial_distribution.alignment.json",
        overlay_svg=OUTPUT_DIR / "supply_spatial_distribution.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    base = OUTPUT_DIR / "supply_spatial_distribution"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {"panels": panel_info, "alignment_verdict": alignment.get("verdict")}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = gpd.read_file(INPUT).to_crs(32650).sort_values("community_id").reset_index(drop=True)
    audit = json.loads((INPUT.parent / f"calculation_audit_2018{SUFFIX}.json").read_text(encoding="utf-8"))
    src, dst = queen_edges(frame)
    figure_info = make_figure(frame)

    diagnostic_rows: list[dict] = []
    extreme_rows: list[dict] = []
    tier_summary: list[dict] = []
    for tier in TIERS:
        ra = frame[f"RA_{tier}"]
        sa = frame[f"SA_{tier}_per_10000"]
        tc = frame[f"TC_{tier}"]
        count = frame[f"facility_count_{tier}"]
        tier_summary.append(
            {
                "tier": tier,
                "source_facility_count": audit["facility_definition"]["counts"][tier],
                "communities_with_facility": int((count > 0).sum()),
                "communities_RA_positive": int((ra > 0).sum()),
                "communities_SA_positive": int((sa > 0).sum()),
                "communities_TC_positive": int((tc > 0).sum()),
                "RA_positive_but_SA_zero": int(((ra > 0) & (sa <= 0)).sum()),
                "RA_positive_but_TC_zero": int(((ra > 0) & (tc <= 0)).sum()),
                "spearman_count_RA": spearman(count, ra),
                "spearman_RA_SA": spearman(ra, sa),
                "spearman_RA_TC": spearman(ra, tc),
                "spearman_TC_coverage": spearman(tc, frame[f"within_15min_population_pct_{tier}"]),
            }
        )
        for metric, _, _ in METRICS:
            column = metric_column(metric, tier)
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            moran, p_value = moran_i(values, src, dst)
            diagnostic_rows.append(
                {
                    "tier": tier,
                    "indicator": metric,
                    "column": column,
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                    "maximum": float(np.max(finite)),
                    "zero_count": int((values == 0).sum()),
                    "missing_count": int((~np.isfinite(values)).sum()),
                    "moran_i_queen": moran,
                    "permutation_p_two_sided": p_value,
                    "permutations": 499,
                }
            )
            top = frame.nlargest(5, column)
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                extreme_rows.append(
                    {
                        "tier": tier,
                        "indicator": metric,
                        "rank": rank,
                        "community_id": int(row["community_id"]),
                        "community_name": row["community_name"],
                        "value": float(row[column]),
                        "population_demand": float(row["population_demand"]),
                        "facility_count": int(row[f"facility_count_{tier}"]),
                    }
                )

    diagnostic = pd.DataFrame(diagnostic_rows)
    summary = pd.DataFrame(tier_summary)
    extremes = pd.DataFrame(extreme_rows)
    diagnostic.to_csv(OUTPUT_DIR / "spatial_autocorrelation_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "indicator_consistency_diagnostics.csv", index=False, encoding="utf-8-sig")
    extremes.to_csv(OUTPUT_DIR / "indicator_top5_communities.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# 行政社区医疗设施供给指标空间分布合理性检查",
        "",
        "## 数据口径",
        "",
        "道路采用Geofabrik提供的OSM中国2019年1月1日快照，并按深圳行政社区范围外扩5公里提取。",
        f"250米居住网格{DEMAND_LABEL}合计{frame['population_demand'].sum():,.0f}人；有正人口的行政社区为{int(frame['population_demand'].gt(0).sum())}个，无正人口的行政社区为{int(frame['population_demand'].eq(0).sum())}个。",
        "",
        "## 分布检查",
        "",
    ]
    for tier in TIERS:
        positive = frame.loc[frame[f"SA_{tier}_per_10000"] > 0, f"SA_{tier}_per_10000"]
        top = frame.loc[frame[f"SA_{tier}_per_10000"].idxmax()]
        diag = diagnostic.loc[(diagnostic["tier"] == tier) & (diagnostic["indicator"] == "SA")].iloc[0]
        report_lines.append(
            f"{tier}层级的SA有{int(diag['zero_count'])}个零值和{int(diag['missing_count'])}个无需求缺失值。"
            f"正值中位数为{positive.median():.2f}，第95百分位为{positive.quantile(0.95):.2f}，"
            f"最大值为{top[f'SA_{tier}_per_10000']:.2f}。最大值所在行政社区编号为{int(top['community_id'])}，"
            f"扩样人口为{top['population_demand']:.0f}人。"
        )
    report_lines.extend(["", "## 零值归因", "", "666个行政社区均有正人口，指标零值不是行政社区缺少人口网格。"])
    for tier in TIERS:
        no_facility = frame[f"facility_count_{tier}"].eq(0)
        sa_zero = frame[f"SA_{tier}"].eq(0)
        tc_zero = frame[f"TC_{tier}"].eq(0)
        report_lines.append(
            f"{tier}：OSM道路15分钟步行范围内无纳入设施的行政社区有{int(no_facility.sum())}个；"
            f"另有{int((sa_zero & ~no_facility).sum())}个行政社区SA为零、"
            f"{int((tc_zero & ~no_facility).sum())}个行政社区TC为零。"
        )
    report_lines.append("设施归属由行政社区人口重心到设施实际点位的OSM道路15分钟步行时间决定；该条件也作用于医院层级的RA、SA和TC。")
    report_lines.extend([
        "",
        "## 解释边界",
        "",
        "SA的每万单位指每万人。CHC的供给量为服务能力评分，SH和TH的供给量为床位数。",
        "服务范围内可达的人口网格较少会放大供需比，极端高值应结合道路连通性和设施服务范围内需求人口检查。",
        "医院SA高值还应核对相应设施的可达人口网格数、竞争需求、设施接入距离和道路连通性。",
        "地图按各面板正值四分位独立分级。灰色表示零值，斜线表示无需求，面板间颜色不表示相同的绝对数值。",
        "SH和TH的道路时间按OSM道路类别及有效限速估算，机动车可达时间仍需其他数据核验。",
    ])
    (OUTPUT_DIR / "空间分布合理性检查.md").write_text("\n".join(report_lines), encoding="utf-8")
    (OUTPUT_DIR / "figure_metadata.json").write_text(
        json.dumps(
            {
                "core_conclusion": "基层设施覆盖的行政社区更多，医院SA高值需要结合路网连通性和人口分母解释。",
                "archetype": "quantitative grid",
                "source_community_count": int(len(frame)),
                "road_snapshot_date": audit["inputs"]["road_snapshot_date"],
                "demand_source": DEMAND_SOURCE,
                "queen_directed_edges": int(len(src)),
                "figure": figure_info,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(diagnostic[["tier", "indicator", "moran_i_queen", "permutation_p_two_sided", "zero_count", "maximum"]].to_string(index=False))


if __name__ == "__main__":
    main()
