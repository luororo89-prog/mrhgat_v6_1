"""From basic supply configuration to the subgroups of supply change (layout of the manuscript's Fig. 7), Nature style.

S_base = composite score of the nominal indicators x under the fixed ruler; S_eff = composite score of x + dx, with dx
the mean supply change over the seeds of a model_v11 run (consensus_community.csv); Delta S = S_eff - S_base.
Delta S > 0 is active rise, Delta S < 0 passive decline.
Tendency of a type: rise when its share of active rise exceeds the citywide share, decline otherwise; the subgroup in
the tendency direction is named type-direction (for example Scarcity-rise) and the other subgroup keeps the type name.
(a) Alluvial: types ordered by supply level -> passive decline / active rise -> subgroups; the percentage on a type is
    the share of its communities in the tendency direction.
(b) Count and column share of every type in passive decline and active rise; the tendency cell is labelled.
(c) S_eff of the subgroups (violins and points, ordered by median); two-sided Mann-Whitney U test between the two
    subgroups of a type.
(d) Map of the subgroups; district outlines from 1input_sources/shenzhen_district (nine districts, clipped to the
    community polygons), or dissolved from the community codes (pre-2017 districts) when that file is absent.
Outputs in results_lineb/<name>/figures: fig_supply_change.pdf, .svg, .png and the table behind the figure.

python 5lineb/v11_nodeedge/plot_supply_change_v11.py --config 5lineb/v11_nodeedge/configs/walk_v4_pca5_v11_k3_s10.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch, Polygon, Rectangle
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from graph import build_graph  # noqa: E402
from ruler import standardize, measure  # noqa: E402
from run_v2 import frozen  # noqa: E402

TYPE_EN = {'稀缺型（三甲低·二级低·社康低）': 'Scarcity', '中等型（三甲中·二级低·社康中）': 'SH-scarcity',
           '中等型（三甲中·二级中·社康低）': 'CHC-scarcity', '二级主导型（三甲低·二级高·社康中）': 'SH-based',
           '三甲社康型（三甲高·二级中·社康高）': 'Abundance'}
TYPE_COLOR = {'Scarcity': '#3C78B4', 'SH-scarcity': '#5AA84F', 'CHC-scarcity': '#F28E2B', 'SH-based': '#9B6CB5', 'Abundance': '#D9473F'}
LABEL_SHIFT = {'Nanshan': (-5500, -6500), 'Futian': (-1500, -7000), 'Luohu': (5500, -5000), 'Yantian': (4000, -6000),
               "Bao'an": (1500, -3500), 'Longgang': (-7000, 3000), 'Longhua': (0, 0), 'Guangming': (0, 2000),
               'Pingshan': (2500, 2500)}
DISTRICT_FILE = '1input_sources/shenzhen_district/shenzhen_district_4547.shp'
DISTRICT_EN = {'440303': 'Luohu', '440304': 'Futian', '440305': 'Nanshan', '440306': "Bao'an", '440307': 'Longgang',
               '440308': 'Yantian', '440309': 'Longhua', '440310': 'Pingshan', '440311': 'Guangming'}
DISTRICT = {'440303': 'Luohu', '440304': 'Futian', '440305': 'Nanshan', '440306': "Bao'an", '440307': 'Longgang',
            '440308': 'Yantian', '440322': "Bao'an"}
NODES = '1input_sources/26上实验/medical_identification_validation/results/node_order.csv'
POLYGONS = '1input_sources/shp/深圳社区objectid.shp'
MM = 1 / 25.4


def nature_style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.right': False,
                         'axes.spines.top': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'legend.frameon': False})


def ribbon(ax, x0, y0, x1, y1, color, alpha=0.45, n=50):
    """Band from the interval y0 = (top, bottom) at x0 to y1 at x1 with smoothstep edges."""
    t = np.linspace(0, 1, n)
    s = t * t * (3 - 2 * t)
    xs = x0 + (x1 - x0) * t
    top = y0[0] + (y1[0] - y0[0]) * s
    bot = y0[1] + (y1[1] - y0[1]) * s
    ax.add_patch(Polygon(np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]], closed=True, fc=color, ec='none', alpha=alpha))


def stack(sizes, total, gap):
    """Top and bottom of stacked nodes (top to bottom) on a 0..1 axis."""
    scale = (1 - gap * (len(sizes) - 1)) / total
    out, y = {}, 1.0
    for k, v in sizes.items():
        h = v * scale
        out[k] = (y, y - h)
        y -= h + gap
    return out, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    root = HERE.parents[1]
    out = root / 'results_lineb' / cfg['name']
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)
    R = frozen(root / cfg['ruler_from'])['main']
    g = build_graph(cfg)
    x = standardize(g['raw'], R)
    com = pd.read_csv(out / 'consensus_community.csv', encoding='utf-8-sig').set_index('community_id').loc[g['ids']]
    dx = com[[f'dx_{c}' for c in g['columns']]].to_numpy()
    s_base, s_eff = measure(x, R)['composite'], measure(x + dx, R)['composite']
    d = pd.DataFrame({'community_id': g['ids'], 'type': com.type_nominal.map(TYPE_EN).to_numpy(), 'S_base': s_base,
                      'S_eff': s_eff, 'dS': s_eff - s_base})
    d['direction'] = np.where(d.dS > 0, 'rise', 'decline')
    order = d.groupby('type').S_base.mean().sort_values().index.tolist()
    rise_all = (d.direction == 'rise').mean()
    tendency = {t: 'rise' if (d[d.type == t].direction == 'rise').mean() > rise_all else 'decline' for t in order}
    d['subgroup'] = [f'{t}-{r}' if tendency[t] == r else t for t, r in zip(d.type, d.direction)]
    med = d.groupby('subgroup').S_eff.median().sort_values()
    cmap = mpl.colormaps['RdYlGn_r']
    sub_color = {s: cmap(0.08 + 0.84 * i / max(len(med) - 1, 1)) for i, s in enumerate(med.index)}

    tab = []
    for t in order:
        for r in ('decline', 'rise'):
            m = (d.type == t) & (d.direction == r)
            tab.append({'type': t, 'direction': r, 'n': int(m.sum()), 'column_share': m.sum() / (d.direction == r).sum(),
                        'type_share': m.sum() / (d.type == t).sum(), 'tendency': tendency[t] == r,
                        'subgroup': f'{t}-{r}' if tendency[t] == r else t})
    tab = pd.DataFrame(tab)
    tests = []
    for t in order:
        a_, b_ = d[(d.type == t) & (d.direction == 'rise')].S_eff, d[(d.type == t) & (d.direction == 'decline')].S_eff
        p = mannwhitneyu(a_, b_).pvalue if len(a_) > 1 and len(b_) > 1 else np.nan
        tests.append({'type': t, 'n_rise': len(a_), 'n_decline': len(b_), 'p_mannwhitney': p})
    tests = pd.DataFrame(tests)
    tab.merge(tests, on='type').to_csv(fig_dir / 'fig_supply_change_table.csv', index=False)
    d.to_csv(fig_dir / 'fig_supply_change_communities.csv', index=False)

    nature_style()
    fig = plt.figure(figsize=(183 * MM, 160 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.2, left=0.075, right=0.99, top=0.97, bottom=0.11)
    top_gs = outer[0].subgridspec(1, 2, width_ratios=[1.75, 1], wspace=0.12)
    bot_gs = outer[1].subgridspec(1, 2, width_ratios=[1, 1.25], wspace=0.08)

    # (a) alluvial
    ax = fig.add_subplot(top_gs[0])
    ax.set_axis_off()
    n_tot = len(d)
    left, _ = stack({t: (d.type == t).sum() for t in order}, n_tot, 0.035)
    mid, _ = stack({r: (d.direction == r).sum() for r in ('decline', 'rise')}, n_tot, 0.08)
    right_keys = [(t, 'decline') for t in order] + [(t, 'rise') for t in order]
    right, _ = stack({k: ((d.type == k[0]) & (d.direction == k[1])).sum() for k in right_keys}, n_tot, 0.012)
    xl, xm, xr, w = 0.22, 0.58, 0.93, 0.035
    lo_cur = {t: left[t][0] for t in order}
    mi_in = {r: mid[r][0] for r in mid}
    mi_out = {r: mid[r][0] for r in mid}
    span = lambda v, bounds_top, total_h: (bounds_top, bounds_top - v)
    h_unit = (left[order[0]][0] - left[order[0]][1]) / max((d.type == order[0]).sum(), 1)
    for t in order:
        for r in ('decline', 'rise'):
            v = ((d.type == t) & (d.direction == r)).sum() * h_unit
            if v <= 0:
                continue
            y0 = (lo_cur[t], lo_cur[t] - v)
            y1 = (mi_in[r], mi_in[r] - v)
            ribbon(ax, xl + w, y0, xm, y1, TYPE_COLOR[t])
            lo_cur[t] -= v
            mi_in[r] -= v
    for k in right_keys:
        v = right[k][0] - right[k][1]
        if v <= 0:
            continue
        ribbon(ax, xm + w, (mi_out[k[1]], mi_out[k[1]] - v), xr, right[k], TYPE_COLOR[k[0]])
        mi_out[k[1]] -= v
    for t in order:
        top, bot = left[t]
        ax.add_patch(Rectangle((xl, bot), w, top - bot, fc=TYPE_COLOR[t], ec='none'))
        ax.text(xl - 0.012, (top + bot) / 2, t, ha='right', va='center')
        share = tab[(tab.type == t) & tab.tendency].type_share.iloc[0]
        ax.text(xl + w + 0.008, (top + bot) / 2, f'{share * 100:.1f}%', ha='left', va='center', fontsize=6, fontweight='bold')
    for r, label in (('decline', 'Passive\ndecline'), ('rise', 'Active\nrise')):
        top, bot = mid[r]
        ax.add_patch(Rectangle((xm, bot), w, top - bot, fc='#BDBDBD' if r == 'decline' else '#4D4D4D', ec='none'))
        ax.text(xm + w / 2, (top + bot) / 2, label, ha='center', va='center', fontsize=6, rotation=90,
                color='black' if r == 'decline' else 'white')
    for k in right_keys:
        top, bot = right[k]
        if top - bot > 0:
            sub = f'{k[0]}-{k[1]}' if tendency[k[0]] == k[1] else k[0]
            ax.add_patch(Rectangle((xr, bot), w, top - bot, fc=sub_color[sub], ec='none'))
    ax.annotate('', xy=(0.012, 0.0), xytext=(0.012, 1.0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color='black'))
    ax.text(0.0, 1.0, 'Low', ha='left', va='bottom', fontsize=6)
    ax.text(0.0, -0.02, 'High', ha='left', va='top', fontsize=6)
    ax.text(-0.005, 0.5, 'Supply level', rotation=90, ha='right', va='center', fontweight='bold')
    ax.text(xl + w / 2, 1.05, 'Basic supply\nconfiguration', ha='center', va='bottom', fontsize=6)
    ax.text(xm + w / 2, 1.05, 'Supply change', ha='center', va='bottom', fontsize=6)
    ax.text(xr + w / 2, 1.05, 'Subgroup', ha='center', va='bottom', fontsize=6)
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.06, 1.12)
    ax.text(-0.03, 1.12, 'a', fontweight='bold', fontsize=8, va='top')

    # (b) matrix
    ax = fig.add_subplot(top_gs[1])
    ax.set_axis_off()
    for i, t in enumerate(order):
        for j, r in enumerate(('decline', 'rise')):
            row = tab[(tab.type == t) & (tab.direction == r)].iloc[0]
            yy = len(order) - 1 - i
            ax.add_patch(Rectangle((j * 1.05, yy * 1.05), 1, 1, fc=sub_color[row.subgroup], ec='white', lw=0.8))
            label = r if row.tendency else '--'
            ax.text(j * 1.05 + 0.5, yy * 1.05 + 0.5, f'{label}\nN={row.n}\n({row.column_share * 100:.2f}%)', ha='center',
                    va='center', fontsize=6, linespacing=1.25)
        ax.text(-0.1, (len(order) - 1 - i) * 1.05 + 0.5, t, ha='right', va='center')
    ax.text(0.5, -0.12, 'Passive\ndecline', ha='center', va='top', fontsize=6)
    ax.text(1.55, -0.12, 'Active\nrise', ha='center', va='top', fontsize=6)
    ax.set_xlim(-1.25, 2.15)
    ax.set_ylim(-0.75, len(order) * 1.05 + 0.15)
    ax.text(-1.25, len(order) * 1.05 + 0.15, 'b', fontweight='bold', fontsize=8, va='top')

    # (c) violins
    ax = fig.add_subplot(bot_gs[0])
    subs = med.sort_values(ascending=False).index.tolist()
    data = [d[d.subgroup == s].S_eff.to_numpy() for s in subs]
    pos = np.arange(len(subs))
    for p_, s, v in zip(pos, subs, data):
        if len(v) > 1:
            vp = ax.violinplot(v, positions=[p_], widths=0.8, showextrema=False)
            for b in vp['bodies']:
                b.set_facecolor(sub_color[s])
                b.set_edgecolor('none')
                b.set_alpha(0.55)
        jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
        ax.scatter(p_ + jit, v, s=1.2, color=sub_color[s], edgecolor='none', alpha=0.9, zorder=3)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([p_, p_], [q1, q3], color='black', lw=0.8, zorder=4)
        ax.plot([p_ - 0.18, p_ + 0.18], [q2, q2], color='black', lw=0.8, zorder=4)
    ax.set_xticks(pos)
    ax.set_xticklabels(subs, rotation=45, ha='right', rotation_mode='anchor', fontsize=5.5)
    ax.set_ylabel('Effective supply (composite score)')
    ymax, ymin = d.S_eff.max(), d.S_eff.min()
    step = (ymax - ymin) * 0.075
    pairs = []
    for row in tests.dropna().itertuples():
        a1 = f'{row.type}-rise' if tendency[row.type] == 'rise' else row.type
        b1 = f'{row.type}-decline' if tendency[row.type] == 'decline' else row.type
        if a1 in subs and b1 in subs:
            pairs.append((*sorted([subs.index(a1), subs.index(b1)]), row))
    pairs.sort(key=lambda q: q[1] - q[0])
    for i, (xa, xb, row) in enumerate(pairs):
        yb = ymax + step * (1.3 * i + 0.6)
        ax.plot([xa, xa, xb, xb], [yb - step * 0.25, yb, yb, yb - step * 0.25], color='black', lw=0.5)
        ptxt = 'P < 0.001' if row.p_mannwhitney < 0.001 else f'P = {row.p_mannwhitney:.3f}'
        ax.text((xa + xb) / 2, yb + step * 0.08, ptxt, ha='center', va='bottom', fontsize=5.5)
    ax.set_ylim(ymin - step * 0.5, ymax + step * (1.3 * len(pairs) + 0.8))
    ax.set_xlim(-0.6, len(subs) - 0.4)
    ax.text(-0.09, 1.02, 'c', transform=ax.transAxes, fontweight='bold', fontsize=8, va='bottom')

    # (d) map
    ax = fig.add_subplot(bot_gs[1])
    nodes = pd.read_csv(root / NODES, encoding='utf-8-sig').assign(SQCODE=lambda f: f.SQCODE.astype(str))
    poly = gpd.read_file(root / POLYGONS)[['SQCODE', 'geometry']].assign(SQCODE=lambda f: f.SQCODE.astype(str))
    poly = poly.merge(nodes[['SQCODE', 'OBJECTID']], on='SQCODE').rename(columns={'OBJECTID': 'community_id'})
    poly = poly.merge(d[['community_id', 'subgroup']], on='community_id', how='left')
    poly['color'] = poly.subgroup.map(lambda s: sub_color.get(s, '#EEEEEE') if isinstance(s, str) else '#EEEEEE')
    poly.plot(ax=ax, color=poly.color, edgecolor='white', linewidth=0.08)
    if (root / DISTRICT_FILE).exists():
        land = poly.geometry.buffer(50).union_all().buffer(-50)
        dist = gpd.read_file(root / DISTRICT_FILE).to_crs(poly.crs)
        dist = dist.assign(district=dist.dt_adcode.astype(str).map(DISTRICT_EN)).set_index('district')
        dist['geometry'] = dist.geometry.intersection(land)
    else:
        poly['district'] = poly.SQCODE.str[:6].map(DISTRICT)
        dist = poly.dissolve('district')
        dist['geometry'] = dist.geometry.buffer(50).buffer(-50)
    dist.boundary.plot(ax=ax, color='#333333', linewidth=0.5)
    for name, geom in dist.geometry.items():
        pt = geom.representative_point()
        dx_, dy_ = LABEL_SHIFT.get(name, (0, 0))
        ax.text(pt.x + dx_, pt.y + dy_, name, ha='center', va='center', fontsize=6, fontstyle='italic', fontweight='bold')
    ax.set_axis_off()
    xmin, ymin_, xmax, ymax_ = poly.total_bounds
    sx, sy = xmin + (xmax - xmin) * 0.62, ymin_ + (ymax_ - ymin_) * 0.04
    for k_, (a0, a1) in enumerate([(0, 5000), (5000, 10000), (10000, 20000)]):
        ax.add_patch(Rectangle((sx + a0, sy), a1 - a0, (ymax_ - ymin_) * 0.012, fc='black' if k_ % 2 == 0 else 'white',
                               ec='black', lw=0.4))
    for v in (0, 5, 10, 20):
        ax.text(sx + v * 1000, sy - (ymax_ - ymin_) * 0.012, str(v), ha='center', va='top', fontsize=5.5)
    ax.text(sx + 20000 + (xmax - xmin) * 0.012, sy, 'km', ha='left', va='bottom', fontsize=5.5)
    nx, ny = xmax - (xmax - xmin) * 0.05, ymax_ - (ymax_ - ymin_) * 0.10
    ax.annotate('N', xy=(nx, ny + (ymax_ - ymin_) * 0.08), xytext=(nx, ny), ha='center', va='top', fontsize=7,
                fontweight='bold', arrowprops=dict(arrowstyle='-|>', lw=0.8, color='black'))
    legend_order = [f'{t}-{tendency[t]}' if r == tendency[t] else t for t in order for r in ('decline', 'rise')]
    handles = [Patch(fc=sub_color[s], ec='none', label=s) for s in legend_order if s in sub_color]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.02), ncol=2, handlelength=1.0, handleheight=0.8,
              columnspacing=0.8, borderaxespad=0.0)
    ax.text(0.0, 1.02, 'd', transform=ax.transAxes, fontweight='bold', fontsize=8, va='bottom')

    for ext in ('pdf', 'svg'):
        fig.savefig(fig_dir / f'fig_supply_change.{ext}')
    fig.savefig(fig_dir / 'fig_supply_change.png', dpi=600)
    print(tab.merge(tests, on='type').round(4).to_string(index=False))
    print('saved to', fig_dir)


if __name__ == '__main__':
    main()
