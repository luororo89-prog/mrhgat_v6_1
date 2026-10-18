"""分层供给水平地图与基本供给配置图，输出两张图。

图一 fig_supply_level_map：(a)(b)(c) 为 CHC、SH、TH 供给水平（supply level），横排一行，排版参照分面可达性地图
（无边框，供给水平图例与指北针置于地图右上空白处，设施与边界图例置于下方海域，图名在下方）。
按 15 分钟步行生活圈着色（无等时圈的社区用社区边界）。供给水平取 RA、SA（每万人口）、TC 三分量按
results/node_typing/ruler.json 冻结参数做 Yeo-Johnson 变换并标准化后的均值，与节点分型 composite 模式的层级分量一致；
非零生活圈按五分位分为 Low 至 High 五级，三分量均为 0 的生活圈为 No facility。
各图叠加该层级评价设施（社康绿点、二级医院橙色十字、三级医院蓝色十字），分级配色避开设施颜色。
图二 fig_basic_supply_configuration：(a) 九项指标按配置类型的累积分布，一行一种设施（CHC、SH、TH），列为 RA、SA（每万人口）、TC；
(b) 基本供给配置地图，按行政社区着色（results/node_typing/node_types.csv，类型英文名与配色同 plot_types_district_map.py），
版式参照类型分布底图：右上四向指北针，左下类型图例，右下类型占比环图与比例尺。
地图统一标注区名，区名置于市界外侧，不压在地图上。
输出：results/supply_indicators_revised/ 下两张图的 {pdf,svg,png} 与 community_supply_composite.csv。

python 重做实验/step3_od_demand_time_filtered/plot_supply_composite_map.py [--isochrone foot等时圈.shp]
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, Wedge
from scipy.stats import yeojohnson
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

from plot_types_district_map import COLOR, ORDER, TYPE_EN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE.parent / "supply input"
RESULT_DIR = HERE / "results" / "supply_indicators_revised"
DISTRICT_FILE = SRC / "shenzhen_district_4547.shp"
ISOCHRONE_FILE = ROOT / "input_sources/等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp"
RULER = HERE / "results" / "node_typing" / "ruler.json"
TYPES_FILE = HERE / "results" / "node_typing" / "node_types.csv"
CRS = 4547
MM = 1 / 25.4

TIERS = ["CHC", "SH", "TH"]
COMPONENTS = {t: [f"RA_{t}", f"SA_{t}_per_10000", f"TC_{t}"] for t in TIERS}
# 五级配色（低到高，紫红色系），与绿色社康、橙色二级、蓝色三级标记区分
CLASS_COLORS = ["#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]
LAND, ZERO = "#F7F7F7", "#D9D9D9"
DISTRICT_EDGE, COMMUNITY_EDGE, CIRCLE_EDGE = "#333333", "#CFCFCF", "#6E6E6E"
FACILITY_STYLE = {"CHC": dict(marker="o", ms=1.0, color="#1E9E3A", mec="white", mew=0.12, label="CHC"),
                  "SH": dict(marker="+", ms=3.4, color="#F28E2B", mew=0.9, label="Secondary hospital"),
                  "TH": dict(marker="+", ms=3.4, color="#1F5FD6", mew=0.9, label="Tertiary hospital")}
CAPTION = {"CHC": "(a) Supply level of CHC", "SH": "(b) Supply level of SH", "TH": "(c) Supply level of TH"}
# 区名位置（经度、纬度、水平对齐、垂直对齐），贴市界外侧；龙岗区北部与大鹏半岛各标一次
DISTRICT_LABELS = [("Guangming", 113.978, 22.852, "center", "bottom"), ("Longhua", 114.05, 22.781, "center", "bottom"),
                   ("Longgang", 114.221, 22.822, "center", "bottom"), ("Pingshan", 114.447, 22.69, "left", "center"),
                   ("Bao'an", 113.778, 22.565, "center", "center"), ("Nanshan", 113.905, 22.44, "center", "top"),
                   ("Futian", 114.04, 22.497, "center", "top"), ("Luohu", 114.13, 22.524, "center", "top"),
                   ("Yantian", 114.30, 22.556, "center", "top"), ("Longgang", 114.62, 22.57, "center", "center")]
LEGEND_AT = (114.462, 22.872)      # 图一右上，供给水平图例左上角
COMPASS_AT = (114.676, 22.79)      # 图一图例右侧，指北针底部中点
SCALEBAR_AT = (114.43, 22.375)    # 图一右下
LINES_AT = (114.03, 22.462)       # 图一下方海域，设施与边界图例左上角
# 图二地图版面位置
ROSE_AT = (114.615, 22.80)
TYPE_LEGEND_AT = (113.965, 22.428)
DONUT_AT = (114.345, 22.478)
SCALEBAR2_X = 114.405            # 图二比例尺起点经度，纵向与 Abundance 图例行对齐


def to_xy(lon, lat):
    p = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(CRS).iloc[0]
    return p.x, p.y


def tier_composite(ind: pd.DataFrame) -> pd.DataFrame:
    frozen = json.loads(RULER.read_text(encoding="utf-8"))["main"]["frozen"]
    z = {}
    for i, col in enumerate(frozen["columns"]):
        x = ind[col].astype(float).to_numpy()
        if col in frozen["lambdas"]:
            x = yeojohnson(x, lmbda=frozen["lambdas"][col])
        z[col] = (x - frozen["mean"][i]) / frozen["std"][i]
    out = ind[["community_id", "SQCODE", "community_name"]].copy()
    for t in TIERS:
        out[f"composite_{t}"] = np.mean([z[c] for c in COMPONENTS[t]], axis=0)
        out[f"all_zero_{t}"] = (ind[COMPONENTS[t]] <= 0).all(axis=1)
    return out


def quantile_classes(values: np.ndarray, k: int = 5):
    """五分位分级；重复断点合并。返回类别号与颜色（低到高）。"""
    breaks = np.unique(np.nanquantile(values, np.linspace(0, 1, k + 1)))
    idx = np.clip(np.searchsorted(breaks[1:-1], values, side="right"), 0, len(breaks) - 2)
    colors = [CLASS_COLORS[i] for i in np.round(np.linspace(0, len(CLASS_COLORS) - 1, len(breaks) - 1)).astype(int)]
    return idx, colors


def compass(ax, x, y, r, fs):
    """指北箭头（左黑右白），x、y 为箭头底部中点。"""
    w = r * 0.32
    return [ax.add_patch(Polygon([(x, y + r), (x - w, y), (x, y + r * 0.28)], closed=True, fc="black", ec="black",
                                 lw=0.4, zorder=8)),
            ax.add_patch(Polygon([(x, y + r), (x + w, y), (x, y + r * 0.28)], closed=True, fc="white", ec="black",
                                 lw=0.4, zorder=8)),
            ax.text(x, y + r * 1.15, "N", ha="center", va="bottom", fontsize=fs, zorder=8)]


def compass_rose(ax, x, y, r, fs):
    """四向指北针（类型分布底图样式）：四个主向尖角与四个斜向小尖角半黑半白，中心为黑白相间圆。"""
    out = []

    def point(ang, length, width):
        a = np.deg2rad(ang)
        tip = (x + length * np.cos(a), y + length * np.sin(a))
        nx, ny = -np.sin(a) * width, np.cos(a) * width
        out.append(ax.add_patch(Polygon([(x, y), tip, (x + nx, y + ny)], closed=True, fc="black", ec="black", lw=0.5,
                                        zorder=8)))
        out.append(ax.add_patch(Polygon([(x, y), tip, (x - nx, y - ny)], closed=True, fc="white", ec="black", lw=0.5,
                                        zorder=8)))
    for ang in (45, 135, 225, 315):
        point(ang, r * 0.55, r * 0.11)
    for ang in (90, 0, 270, 180):
        point(ang, r, r * 0.17)
    ax.add_patch(Circle((x, y), r * 0.40, fc="white", ec="black", lw=0.8, zorder=9))
    for k, start in enumerate((0, 90, 180, 270)):
        ax.add_patch(Wedge((x, y), r * 0.30, start, start + 90, fc="black" if k % 2 == 0 else "white",
                           ec="black", lw=0.5, zorder=10))
    for ang, lab in ((90, "N"), (0, "E"), (270, "S"), (180, "W")):
        a = np.deg2rad(ang)
        out.append(ax.text(x + r * 1.25 * np.cos(a), y + r * 1.25 * np.sin(a), lab, ha="center", va="center",
                           zorder=10, fontsize=fs * (1.15 if lab == "N" else 1.0),
                           fontweight="bold" if lab == "N" else "normal", fontfamily="Times New Roman"))
    return out


def scalebar(ax, x, y, fs, pt_m):
    """比例尺；pt_m 为 1 pt 对应的地图米数，km 标注与 20 同行。"""
    km = 1000.0
    h = 2.4 * pt_m
    out = [ax.add_patch(Rectangle((x + a * km, y), (b - a) * km, h, fc="black" if i % 2 == 0 else "white",
                                  ec="black", lw=0.4, zorder=8)) for i, (a, b) in enumerate([(0, 5), (5, 10), (10, 20)])]
    for v in (0, 5, 10, 20):
        out.append(ax.text(x + v * km, y + h + 1.2 * pt_m, str(v), ha="center", va="bottom", fontsize=fs, zorder=8))
    out.append(ax.text(x + 20 * km + 1.0 * fs * pt_m, y + h + 1.2 * pt_m, "km", ha="left", va="bottom", fontsize=fs,
                       zorder=8))
    return out


def level_legend(ax, colors, k, fs):
    """供给水平图例：五级色块自上而下由高到低，仅标 High 与 Low，下方为 No facility。
    k 为版面缩放系数（以 89.5 mm 宽地图为 1），行距与色块随地图比例换算。"""
    x, ly = to_xy(*LEGEND_AT)
    out = [ax.text(x, ly, "Legend", ha="left", va="top", fontsize=fs + 1.2, fontweight="bold", zorder=9),
           ax.text(x, ly - 2400 * k, "Supply level", ha="left", va="top", fontsize=fs + 0.5, fontweight="bold",
                   fontstyle="italic", zorder=9)]
    top = ly - 4900 * k
    bw, bh, gap = 2300 * k, 1050 * k, 800 * k
    for i, c in enumerate(reversed(colors)):
        out.append(ax.add_patch(Rectangle((x, top - (i + 1) * bh), bw, bh, fc=c, ec="none", zorder=9)))
    n = len(colors)
    out.append(ax.text(x + bw + gap, top - 0.5 * bh, "High", ha="left", va="center", fontsize=fs, zorder=9))
    out.append(ax.text(x + bw + gap, top - (n - 0.5) * bh, "Low", ha="left", va="center", fontsize=fs, zorder=9))
    y = top - n * bh - gap
    out.append(ax.add_patch(Rectangle((x, y - bh), bw, bh, fc=ZERO, ec="none", zorder=9)))
    out.append(ax.text(x + bw + gap, y - 0.5 * bh, "No facility", ha="left", va="center", fontsize=fs, zorder=9))
    return out


def district_labels(ax, fs):
    return [ax.text(*to_xy(lon, lat), name, ha=ha, va=va, fontsize=fs, fontweight="bold", fontstyle="italic",
                    color="#333333", zorder=7) for name, lon, lat, ha, va in DISTRICT_LABELS]


def data_boxes(fig, ax, artists, pad=0.0):
    """图上元素在地图坐标中的外接框之并。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    boxes = []
    for a in artists:
        bb = a.get_tightbbox(renderer) if hasattr(a, "get_tightbbox") and not isinstance(a, plt.Text) \
            else a.get_window_extent(renderer)
        (bx0, by0), (bx1, by1) = inv.transform(bb.get_points())
        boxes.append(box(bx0, by0, bx1, by1).buffer(pad))
    return unary_union(boxes)


def place_labels(fig, ax, texts, land, obstacles, extent, clear=400.0, step=300.0, radius=15000.0):
    """区名就近放到市界外：文字框与市域保持 clear 米以上间距，不压图例、指北针、比例尺和其他区名，且在图幅内。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    blocked = prep(unary_union([land.buffer(clear), obstacles]))
    frame = prep(box(extent[0], extent[2], extent[1], extent[3]))
    grid = np.arange(-radius, radius + step, step)
    offsets = sorted(((dx, dy) for dx in grid for dy in grid), key=lambda o: o[0] ** 2 + o[1] ** 2)
    placed = []
    for t in texts:
        (bx0, by0), (bx1, by1) = inv.transform(t.get_window_extent(renderer).get_points())
        for dx, dy in offsets:
            b = box(bx0 + dx, by0 + dy, bx1 + dx, by1 + dy)
            if frame.contains(b) and not blocked.intersects(b) and not any(b.intersects(p) for p in placed):
                px, py = t.get_position()
                t.set_position((px + dx, py + dy))
                placed.append(b.buffer(400))
                break
        else:
            print(f"label {t.get_text()} has no clear position")


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(RESULT_DIR / f"{name}.{ext}")
    fig.savefig(RESULT_DIR / f"{name}.png", dpi=400)
    print("saved", RESULT_DIR / f"{name}.png")


def ecdf(ax, values, color):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.r_[x[0], x], np.r_[0, y], where="post", color=color, lw=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isochrone", default=str(ISOCHRONE_FILE), help="15 分钟步行等时圈 shp")
    isochrone_file = Path(ap.parse_args().isochrone)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "svg.fonttype": "none",
                         "pdf.fonttype": 42, "font.size": 6})

    ind = pd.read_csv(RESULT_DIR / "community_supply_indicators_2018_mobile_users.csv", encoding="utf-8-sig",
                      dtype={"SQCODE": str})
    comp = tier_composite(ind)
    comp.to_csv(RESULT_DIR / "community_supply_composite.csv", index=False, encoding="utf-8-sig")
    fac = pd.read_csv(RESULT_DIR / "facility_supply_audit_2018_mobile_users.csv", encoding="utf-8-sig")
    fac = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.longitude, fac.latitude), crs=4326).to_crs(CRS)

    comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
    comm["SQCODE"] = comm.SQCODE.astype(str).str.replace(r"\.0$", "", regex=True)
    comm["geometry"] = comm.geometry.make_valid()
    dist = gpd.read_file(DISTRICT_FILE).to_crs(CRS)
    land = comm.geometry.buffer(50).union_all().buffer(-50)
    dist["geometry"] = dist.geometry.intersection(land)

    # 生活圈：15 分钟步行等时圈，无等时圈的社区用社区边界
    iso = gpd.read_file(isochrone_file).to_crs(CRS)
    iso["SQCODE"] = iso["num"].astype(str)
    iso = iso.loc[iso.SQCODE.isin(comm.SQCODE), ["SQCODE", "geometry"]]
    circles = pd.concat([comm.loc[~comm.SQCODE.isin(iso.SQCODE), ["SQCODE", "geometry"]], iso], ignore_index=True)
    circles = gpd.GeoDataFrame(circles, geometry="geometry", crs=CRS)
    circles["geometry"] = circles.geometry.make_valid()
    circles = circles.merge(comp, on="SQCODE", how="left")
    assert circles["composite_CHC"].notna().all(), "生活圈缺少供给水平"

    # 基本供给配置：社区经 node_order 的 OBJECTID 对应 node_types 的 community_id
    node = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
    nt = pd.read_csv(TYPES_FILE, encoding="utf-8-sig")
    nt["ctype"] = nt.type_name.map(TYPE_EN)
    assert nt.ctype.notna().all(), "类型名缺少英文对照"
    typed = comm.merge(node[["SQCODE", "OBJECTID"]], on="SQCODE", how="left") \
        .merge(nt[["community_id", "ctype"]], left_on="OBJECTID", right_on="community_id", how="left")
    assert typed["ctype"].notna().all(), "社区缺少类型"
    types = [t for t in ORDER if t in set(typed.ctype)]
    share = typed.ctype.value_counts().reindex(types) / len(typed)
    ind_typed = ind.merge(nt[["community_id", "ctype"]], on="community_id", how="inner", validate="one_to_one")
    assert len(ind_typed) == len(ind), "指标与类型社区不一致"

    x0, y0, x1, y1 = land.bounds

    # 图一：供给水平，横排一行（下方多留海域放设施与边界图例）
    extent = (x0 - 1500, x1 + 8500, y0 - 5000, y1 + 1500)
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    W, side_mm, gap_mm, top_mm, caption_mm, legend_mm = 183.0, 1.0, 2.0, 1.0, 5.5, 0.5
    map_w_mm = (W - 2 * side_mm - 2 * gap_mm) / 3
    map_h_mm = map_w_mm * aspect
    k1 = 89.5 / map_w_mm
    pt1_m = (extent[1] - extent[0]) / (map_w_mm * 72 / 25.4)
    H = top_mm + map_h_mm + caption_mm + legend_mm
    fig = plt.figure(figsize=(W * MM, H * MM))
    for c, tier in enumerate(TIERS):
        ax = fig.add_axes([(side_mm + c * (map_w_mm + gap_mm)) / W, (H - top_mm - map_h_mm) / H,
                           map_w_mm / W, map_h_mm / H])
        ax.set_aspect("equal")
        comm.plot(ax=ax, color=LAND, edgecolor=COMMUNITY_EDGE, linewidth=0.1, zorder=1)
        zero_mask = circles[f"all_zero_{tier}"].astype(bool).to_numpy()
        circles.loc[zero_mask].plot(ax=ax, color=ZERO, edgecolor="none", zorder=2)
        circles.boundary.plot(ax=ax, color="#A6A6A6", linewidth=0.04, alpha=0.6, zorder=3)
        filled = circles.loc[~zero_mask].copy()
        cls, colors = quantile_classes(filled[f"composite_{tier}"].to_numpy(dtype=float))
        filled["cls"] = cls
        filled = filled.sort_values(f"composite_{tier}")
        filled.plot(ax=ax, color=[colors[i] for i in filled["cls"]], edgecolor=CIRCLE_EDGE, linewidth=0.06,
                    zorder=4)
        dist.boundary.plot(ax=ax, color=DISTRICT_EDGE, linewidth=0.45, zorder=5)
        pts = fac.loc[fac.tier == tier]
        st = {k_: v for k_, v in FACILITY_STYLE[tier].items() if k_ != "label"}
        ax.plot(pts.geometry.x, pts.geometry.y, ls="none", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")] if tier != "CHC" else None, **st)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_axis_off()
        lx, ly = to_xy(*LINES_AT)
        lines = ax.legend(handles=[Line2D([], [], ls="none", **{**FACILITY_STYLE[tier],
                                                                "ms": FACILITY_STYLE[tier]["ms"] * 1.4}),
                                   Line2D([], [], color=DISTRICT_EDGE, lw=0.7, label="District boundary"),
                                   Line2D([], [], color=COMMUNITY_EDGE, lw=0.7, label="Community boundary"),
                                   Line2D([], [], color=CIRCLE_EDGE, lw=0.5, label="Life circle boundary")],
                          loc="upper left", bbox_to_anchor=(lx, ly), bbox_transform=ax.transData, frameon=False,
                          fontsize=4.3, handlelength=1.5, labelspacing=0.12, handletextpad=0.4, borderpad=0,
                          borderaxespad=0)
        furniture = level_legend(ax, colors, k1, 4.8) + compass(ax, *to_xy(*COMPASS_AT), 3600 * k1, 5.5) \
            + scalebar(ax, *to_xy(*SCALEBAR_AT), 4.5, pt1_m) + [lines]
        if c == 0:
            touch = data_boxes(fig, ax, furniture).intersection(land).area
            if touch > 0:
                print(f"map furniture overlaps land by {touch / 1e6:.2f} km2")
        place_labels(fig, ax, district_labels(ax, 5), land, data_boxes(fig, ax, furniture, 300), extent)
        ax.text(0.5, -0.02, CAPTION[tier], transform=ax.transAxes, ha="center", va="top", fontsize=8,
                fontfamily="Times New Roman")
    save(fig, "fig_supply_level_map")
    plt.close(fig)

    # 图二：(a) 九项指标按类型的累积分布（一行一种设施），(b) 基本供给配置地图
    W = 183.0
    left_mm, right_mm, col_gap_mm = 20.0, 3.0, 9.0
    cell_w_mm = (W - left_mm - right_mm - 2 * col_gap_mm) / 3
    cell_h_mm, row_step_mm, top_mm = 17.0, 27.0, 6.0
    grid_h_mm = top_mm + 2 * row_step_mm + cell_h_mm + 6.0
    extent2 = (x0 - 1500, x1 + 8500, y0 - 5000, y1 + 1500)
    map_w2_mm = 150.0
    map_h2_mm = map_w2_mm * (extent2[3] - extent2[2]) / (extent2[1] - extent2[0])
    k2 = 89.5 / map_w2_mm
    pt2_m = (extent2[1] - extent2[0]) / (map_w2_mm * 72 / 25.4)
    H = grid_h_mm + 4.0 + map_h2_mm + 1.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    metrics = [("RA", "RA_{}"), ("SA", "SA_{}_per_10000"), ("TC", "TC_{}")]
    for r, tier in enumerate(TIERS):
        for c, (key, fmt) in enumerate(metrics):
            col = fmt.format(tier)
            ax = fig.add_axes([(left_mm + c * (cell_w_mm + col_gap_mm)) / W,
                               (H - top_mm - r * row_step_mm - cell_h_mm) / H, cell_w_mm / W, cell_h_mm / H])
            for t in types:
                ecdf(ax, ind_typed.loc[ind_typed.ctype == t, col].to_numpy(dtype=float), COLOR[t])
            ax.set_title(f"{key}_{tier}", fontsize=8, fontweight="bold", pad=3)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(direction="in", labelsize=7, length=2.5, width=0.6, pad=2, top=False, right=False)
            ax.locator_params(axis="x", nbins=5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.7)
    grid_mid = (H - top_mm - row_step_mm - cell_h_mm / 2) / H
    fig.text(9.5 / W, grid_mid, "Cumulative probability", rotation=90, ha="center", va="center", fontsize=8)
    fig.text(1.5 / W, grid_mid, "(a)", ha="left", va="center", fontsize=10, fontfamily="Times New Roman")

    ax = fig.add_axes([left_mm / W, 1.0 / H, map_w2_mm / W, map_h2_mm / H])
    ax.set_aspect("equal")
    typed.plot(ax=ax, color=[COLOR[t] for t in typed.ctype], edgecolor="white", linewidth=0.15, zorder=1)
    dist.boundary.plot(ax=ax, color=DISTRICT_EDGE, linewidth=0.6, zorder=5)
    ax.set_xlim(extent2[0], extent2[1])
    ax.set_ylim(extent2[2], extent2[3])
    ax.set_axis_off()
    furniture = compass_rose(ax, *to_xy(*ROSE_AT), 9000 * k2, 8)
    lx, ly = to_xy(*TYPE_LEGEND_AT)
    leg = ax.legend(handles=[Patch(fc=COLOR[t], ec="none", label=t) for t in types],
                    loc="upper left", bbox_to_anchor=(lx, ly), bbox_transform=ax.transData, ncol=2, frameon=False,
                    fontsize=7.5, handlelength=2.0, handleheight=1.0, labelspacing=0.5, columnspacing=1.2,
                    handletextpad=0.6, borderpad=0, borderaxespad=0)
    # 比例尺色条中线与 Abundance 图例行中线对齐
    fig.canvas.draw()
    abundance = next(t for t in leg.get_texts() if t.get_text() == "Abundance")
    (_, ay0), (_, ay1) = ax.transData.inverted().transform(abundance.get_window_extent(fig.canvas.get_renderer())
                                                            .get_points())
    bar = scalebar(ax, to_xy(SCALEBAR2_X, 22.4)[0], (ay0 + ay1) / 2 - 1.2 * pt2_m, 7, pt2_m)
    touch = data_boxes(fig, ax, bar).intersection(land).area
    if touch > 0:
        print(f"scale bar overlaps land by {touch / 1e6:.2f} km2")
    furniture += [leg] + bar
    # 类型占比环图
    cx, cy = to_xy(*DONUT_AT)
    r_m = 10500 * k2
    to_fig = lambda x, y: fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    (fx0, fy0), (fx1, fy1) = to_fig(cx - r_m, cy - r_m), to_fig(cx + r_m, cy + r_m)
    pax = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
    pax.pie(share.values, colors=[COLOR[t] for t in types], startangle=90, counterclock=False,
            wedgeprops=dict(width=0.32, edgecolor="white", linewidth=0.8),
            labels=[f"{v:.1%}" for v in share.values], labeldistance=1.2, textprops=dict(fontsize=7.5))
    pax.set_aspect("equal")
    place_labels(fig, ax, district_labels(ax, 8), land,
                 unary_union([data_boxes(fig, ax, furniture, 400), data_boxes(fig, ax, [pax], 400)]), extent2)
    fig.text(1.5 / W, (1.0 + map_h2_mm / 2) / H, "(b)", ha="left", va="center", fontsize=10,
             fontfamily="Times New Roman")
    save(fig, "fig_basic_supply_configuration")


if __name__ == "__main__":
    main()
