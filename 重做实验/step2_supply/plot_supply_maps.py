"""绘制社区生活圈供给评价结果（联通人口版）与社康、医院的空间分布，供检查。
图 1 为 3×3 指标图（另出一版在各行叠加对应层级的设施）：行为社康、二级医院、三级医院，列为 RA、SA（每万人）、TC；无设施的社区为 0，单独着灰色，其余社区按非零值五分位分级。
图 2 为设施分布：a 为社康点位，按是否落在任一社区生活圈内区分；b 为医院 AOI 代表点，并给出所在社区，福田、罗湖另附放大图。
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
IND = RES / "supply_indicators"
SRC = HERE.parent / "supply input"
COMM = SRC / "深圳社区objectid.shp"
DIST = SRC / "shenzhen_district_4547.shp"
AOI = SRC / "hospital_aoi_35.shp"
CRS = 4547

SURFACE, INK, INK2, EDGE, ZERO = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
C_IN, C_OUT, C_SH, C_TH = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

ind = pd.read_csv(IND / "community_supply_indicators_2018_mobile_users.csv", dtype={"SQCODE": str})
comm = gpd.read_file(COMM)[["SQCODE", "geometry"]].to_crs(CRS).merge(ind, on="SQCODE", validate="one_to_one")
dist = gpd.read_file(DIST).to_crs(CRS)
xmin, ymin, xmax, ymax = comm.total_bounds


def frame(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(xmin - 500, xmax + 500)
    ax.set_ylim(ymin - 500, ymax + 500)
    ax.set_aspect("equal")
    ax.set_axis_off()


def place_labels(ax, xs, ys, texts, fontsize):
    """逐个放置编号，在右上、左上、右下、左下、正上、正下中取第一个不与已放编号和符号重叠的位置。"""
    ax.apply_aspect()
    k = ax.figure.dpi / 72
    px = ax.transData.transform(np.column_stack([xs, ys]))
    taken = [(x - 5 * k, y - 5 * k, x + 5 * k, y + 5 * k) for x, y in px]
    for (x, y), t in zip(px, texts):
        w, h = 0.62 * fontsize * k * len(t), 0.9 * fontsize * k
        for dx, dy in [(4, 3), (-4 - w / k, 3), (4, -3 - h / k), (-4 - w / k, -3 - h / k),
                       (-w / 2 / k, 6), (-w / 2 / k, -6 - h / k), (8, -h / 2 / k), (-8 - w / k, -h / 2 / k)]:
            box = (x + dx * k, y + dy * k, x + dx * k + w, y + dy * k + h)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken):
                break
        taken.append(box)
        ax.annotate(t, (ax.transData.inverted().transform((x, y))), xytext=(dx, dy), textcoords="offset points",
                    ha="left", va="bottom", fontsize=fontsize, color=INK, zorder=5)


def fmt(v):
    return f"{v:.3g}" if v < 1000 else f"{v:,.0f}"


# ---------- 设施 ----------
fac = pd.read_csv(RES / "facility_community.csv", dtype={"SQCODE": str})
aud = pd.read_csv(IND / "facility_supply_audit_2018_mobile_users.csv")
fac = fac.merge(aud[["facility_id", "longitude", "latitude", "candidate_community_count"]], on="facility_id")
pts = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.longitude, fac.latitude), crs=4326).to_crs(CRS)
inside_sz = pts.SQCODE.notna()
chc = pts[(pts.tier == "CHC") & inside_sz]
hosp = pts[pts.tier != "CHC"].copy()
hosp["short"] = hosp.facility_name.str.replace("深圳市", "", regex=False)
hosp = hosp.sort_values(["tier", "capacity"], ascending=[False, False]).reset_index(drop=True)
hosp["no"] = np.arange(1, len(hosp) + 1)
aoi = gpd.read_file(AOI).to_crs(CRS)

# ---------- 图 1：3×3 指标；带设施的版本在各行叠加该层级设施 ----------
tiers = [("CHC", "社康"), ("SH", "二级医院"), ("TH", "三级医院")]
kinds = [("RA_{}", "RA 资源可得性"), ("SA_{}_per_10000", "SA 空间可达性（每万人）"), ("TC_{}", "TC 出行便利性")]


def draw_3x3(overlay, path):
    fig, axes = plt.subplots(3, 3, figsize=(19, 14.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.03, hspace=0.5)
    fig.suptitle("社区生活圈供给评价（联通人口，666 个社区）", x=0.01, y=0.985, ha="left", fontsize=15, color=INK)
    for r, (t, tname) in enumerate(tiers):
        for c, (k, kname) in enumerate(kinds):
            ax = axes[r, c]
            frame(ax)
            v = comm[k.format(t)]
            pos = v[v > 0]
            edges = np.unique(pos.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy())
            cls = pd.Series(-1, index=v.index)
            cls[v > 0] = np.clip(np.searchsorted(edges, pos, side="right") - 1, 0, len(edges) - 2)
            seq = SEQ if len(edges) - 1 == len(SEQ) else [SEQ[i] for i in np.linspace(0, 4, len(edges) - 1).round().astype(int)]
            colors = [ZERO if x < 0 else seq[x] for x in cls]
            comm.plot(ax=ax, color=colors, edgecolor=SURFACE, linewidth=0.12)
            dist.boundary.plot(ax=ax, color=EDGE, linewidth=0.5)
            ax.set_title(f"{tname}  {kname}", loc="left", fontsize=12.5, color=INK, pad=4)
            if overlay:
                if t == "CHC":
                    inc = chc[chc.candidate_community_count > 0]
                    exc = chc[chc.candidate_community_count == 0]
                    inc.plot(ax=ax, color=INK, markersize=2.5, zorder=3)
                    exc.plot(ax=ax, color=C_OUT, markersize=7, edgecolor=INK, linewidth=0.3, zorder=4)
                    note = f"黑点 社康在生活圈内（{len(inc)}），橙点 不在任何生活圈内（{len(exc)}）"
                else:
                    hs = hosp[hosp.tier == t]
                    ax.scatter(hs.geometry.x, hs.geometry.y, s=34, marker="^" if t == "SH" else "s",
                               color=C_SH if t == "SH" else C_TH, edgecolor=INK, linewidth=0.6, zorder=4)
                    ms = hs[hs.candidate_community_count == 0]
                    ax.scatter(ms.geometry.x, ms.geometry.y, s=130, facecolor="none", edgecolor=C_OUT, linewidth=1.3,
                               zorder=4)
                    place_labels(ax, hs.geometry.x, hs.geometry.y, hs.no.astype(str).tolist(), 7)
                    note = f"符号为{tname} AOI 代表点（{len(hs)}），编号见设施分布图" + (
                        "，橙圈 AOI 不与任何生活圈相交" if len(ms) else "")
            handles = [Patch(facecolor=ZERO, edgecolor=EDGE, linewidth=0.4, label=f"0（{int((cls < 0).sum())}）")]
            for i in range(len(edges) - 1):
                n = int((cls == i).sum())
                handles.append(Patch(facecolor=seq[i], edgecolor=EDGE, linewidth=0.4,
                                     label=f"{fmt(edges[i])} 至 {fmt(edges[i + 1])}（{n}）"))
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=8.5,
                      frameon=False, labelcolor=INK2, handlelength=1.2, columnspacing=1.0, alignment="left",
                      title=f"有设施社区 {len(pos)} 个，无设施 {int((v == 0).sum())} 个，非零中位数 {fmt(pos.median())}"
                            + (f"\n{note}" if overlay else ""),
                      title_fontsize=9)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


draw_3x3(False, RES / "fig_supply_3x3.png")
draw_3x3(True, RES / "fig_supply_3x3_facilities.png")

# ---------- 图 2：设施分布 ----------

fig = plt.figure(figsize=(19, 7.6), facecolor=SURFACE)
ax1 = fig.add_axes([0.005, 0.17, 0.40, 0.76])
ax2 = fig.add_axes([0.415, 0.17, 0.40, 0.76])
axl = fig.add_axes([0.825, 0.03, 0.17, 0.92])
axl.set_axis_off()

frame(ax1)
comm.plot(ax=ax1, color="#f0efec", edgecolor="#d6d5d0", linewidth=0.2)
dist.boundary.plot(ax=ax1, color=EDGE, linewidth=0.5)
inc = chc[chc.candidate_community_count > 0]
exc = chc[chc.candidate_community_count == 0]
inc.plot(ax=ax1, color=C_IN, markersize=7, edgecolor=SURFACE, linewidth=0.3, zorder=3)
exc.plot(ax=ax1, color=C_OUT, markersize=11, edgecolor=INK, linewidth=0.35, zorder=4)
ax1.set_title("a  社康点位", loc="left", fontsize=13, color=INK, pad=6)
ax1.legend(handles=[
    Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=C_IN, markeredgecolor=SURFACE,
           label=f"落在至少一个社区生活圈内（{len(inc)}）"),
    Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=C_OUT, markeredgecolor=INK,
           label=f"不在任何社区生活圈内（{len(exc)}）"),
], loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=9.5, frameon=False, labelcolor=INK2, alignment="left",
   title=f"深圳境内 {len(chc)} 个，所在社区 {chc.SQCODE.nunique()} 个；西九龙站内地口岸医务室位于香港，未绘出",
   title_fontsize=9.5)

frame(ax2)
located = comm.SQCODE.map(hosp.drop_duplicates("SQCODE").set_index("SQCODE").tier)
comm.plot(ax=ax2, color=["#c5ecdc" if t == "SH" else "#fbe3ad" if t == "TH" else "#f0efec" for t in located],
          edgecolor="#d6d5d0", linewidth=0.2)
dist.boundary.plot(ax=ax2, color=EDGE, linewidth=0.5)
aoi.boundary.plot(ax=ax2, color=INK, linewidth=0.6, zorder=3)
for t, color, marker in [("SH", C_SH, "^"), ("TH", C_TH, "s")]:
    s = hosp[hosp.tier == t]
    ax2.scatter(s.geometry.x, s.geometry.y, s=46, marker=marker, color=color, edgecolor=INK, linewidth=0.6, zorder=4)
core = hosp[hosp.SQCODE.str[:6].isin(["440303", "440304"])]
zx0, zy0, zx1, zy1 = core.total_bounds + np.array([-1500, -1200, 1500, 1200])
miss = hosp[hosp.candidate_community_count == 0]
ax2.scatter(miss.geometry.x, miss.geometry.y, s=170, facecolor="none", edgecolor=C_OUT, linewidth=1.4, zorder=4)

axz = ax2.inset_axes([0.60, 0.0, 0.40, 0.46])
axz.set_facecolor(SURFACE)
comm.plot(ax=axz, color=["#c5ecdc" if t == "SH" else "#fbe3ad" if t == "TH" else "#f0efec" for t in located],
          edgecolor="#d6d5d0", linewidth=0.3)
dist.boundary.plot(ax=axz, color=EDGE, linewidth=0.6)
aoi.boundary.plot(ax=axz, color=INK, linewidth=0.8, zorder=3)
for t, color, marker in [("SH", C_SH, "^"), ("TH", C_TH, "s")]:
    z = hosp[hosp.tier == t]
    axz.scatter(z.geometry.x, z.geometry.y, s=46, marker=marker, color=color, edgecolor=INK, linewidth=0.6, zorder=4)
axz.set_xlim(zx0, zx1)
axz.set_ylim(zy0, zy1)
axz.set_aspect("equal")
axz.set_xticks([])
axz.set_yticks([])
for sp in axz.spines.values():
    sp.set_edgecolor(EDGE)
ax2.indicate_inset_zoom(axz, edgecolor=EDGE, alpha=1)
inz = hosp.geometry.x.between(zx0, zx1) & hosp.geometry.y.between(zy0, zy1)
place_labels(ax2, hosp[~inz].geometry.x, hosp[~inz].geometry.y, hosp[~inz].no.astype(str).tolist(), 7.5)
place_labels(axz, hosp[inz].geometry.x, hosp[inz].geometry.y, hosp[inz].no.astype(str).tolist(), 8)
axz.set_title("福田、罗湖放大", loc="left", fontsize=9.5, color=INK2, pad=3)
ax2.set_title("b  医院 AOI 与所在社区", loc="left", fontsize=13, color=INK, pad=6)
ax2.legend(handles=[
    Line2D([], [], marker="^", ls="", markersize=7, markerfacecolor=C_SH, markeredgecolor=INK,
           label=f"二级医院（{int((hosp.tier == 'SH').sum())}）"),
    Line2D([], [], marker="s", ls="", markersize=6.5, markerfacecolor=C_TH, markeredgecolor=INK,
           label=f"三级医院（{int((hosp.tier == 'TH').sum())}）"),
    Patch(facecolor="#c5ecdc", edgecolor=EDGE, linewidth=0.4, label="二级医院所在社区"),
    Patch(facecolor="#fbe3ad", edgecolor=EDGE, linewidth=0.4, label="三级医院所在社区"),
    Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor="none", markeredgecolor=C_OUT,
           markeredgewidth=1.4, label="AOI 不与任何社区生活圈相交"),
], loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=9.5, frameon=False, labelcolor=INK2, alignment="left",
   title="符号为 AOI 代表点，黑色轮廓为 AOI，底色为医院所在社区", title_fontsize=9.5)

lines = ["医院编号（所在社区，床位）"]
for t, tname in [("TH", "三级医院"), ("SH", "二级医院")]:
    lines.append("")
    lines.append(tname)
    for _, h in hosp[hosp.tier == t].iterrows():
        lines.append(f"{h.no:>2}  {h.short}（{h.SQNAME}，{h.capacity:,.0f}）")
axl.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=7.6, color=INK2, linespacing=1.45, transform=axl.transAxes)
fig.savefig(RES / "fig_facility_distribution.png", dpi=200, facecolor=SURFACE)
plt.close(fig)
print("saved", RES / "fig_supply_3x3.png", RES / "fig_facility_distribution.png")
