"""From basic supply configuration to the change of the three ruler components, in the four panels of the manuscript's
Fig. 7 with every panel carrying the three components (Nature style).

dPC_u = (dx_u) U: change of the component scores of community u under the fixed ruler, with dx the mean supply change
over the seeds of a model_v11 run (consensus_community.csv). PC1 is the overall level led by TH, PC2 SH against CHC,
PC3 CHC against TH (loadings of the ruler). Types are ordered by the mean composite score of the nominal indicators.
(a) Parallel sets: type -> PC1 decline / rise -> PC2 decline / rise -> PC3 decline / rise, ribbons coloured by type.
(b) Count and column share of every type in the decline and rise of each component; the tendency cell (column where
    the type's share exceeds its share of all communities) is labelled, as in the manuscript's Fig. 7(b); cells shaded by
    the share of the type in that direction.
(c) Change of the three component scores per type (three violins per type); two-sided Wilcoxon signed-rank test
    against 0 (*** P < 0.001, ** P < 0.01, * P < 0.05, ns).
(d) Map of the eight direction combinations; the legend is two 2 x 2 blocks (PC1 rise, PC1 decline; rows PC2,
    columns PC3) with the number of communities.
Outputs in results_lineb/<name>/figures: fig_supply_change_pcs.pdf, .svg, .png and fig_supply_change_pcs_table.csv.

python 5lineb/v11_nodeedge/plot_supply_change_pcs_v11.py --config 5lineb/v11_nodeedge/configs/walk_v4_pca5_v11_k3_s10.yaml
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from graph import build_graph  # noqa: E402
from ruler import standardize, measure  # noqa: E402
from run_v2 import frozen  # noqa: E402
from plot_supply_change_v11 import (TYPE_EN, TYPE_COLOR, DISTRICT, DISTRICT_EN, DISTRICT_FILE, LABEL_SHIFT, NODES,  # noqa: E402
                                    POLYGONS, MM, nature_style, ribbon)

PCS = ('PC1', 'PC2', 'PC3')
PC_LABEL = {'PC1': 'PC1\nTH-led level', 'PC2': 'PC2\nSH vs CHC', 'PC3': 'PC3\nCHC vs TH'}
PC_COLOR = {'PC1': '#C0504D', 'PC2': '#8064A2', 'PC3': '#4F9A8E'}
PC_CMAP = {'PC1': 'Reds', 'PC2': 'Purples', 'PC3': 'BuGn'}
COMBO_COLOR = {'+++': '#99000D', '++-': '#E6550D', '+-+': '#C51B7D', '+--': '#FDAE6B',
               '-++': '#08519C', '-+-': '#6BAED6', '--+': '#238B45', '---': '#A1D99B'}


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


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
    dpc = dx @ R['U']
    d = pd.DataFrame({'community_id': g['ids'], 'type': com.type_nominal.map(TYPE_EN).to_numpy(),
                      'S_base': measure(x, R)['composite']})
    for k, pc in enumerate(PCS):
        d[f'd{pc}'] = dpc[:, k]
        d[pc] = np.where(dpc[:, k] > 0, 'rise', 'decline')
    d['combo'] = [''.join('+' if r == 'rise' else '-' for r in row) for row in d[list(PCS)].to_numpy()]
    order = d.groupby('type').S_base.mean().sort_values().index.tolist()
    tidx = {t: i for i, t in enumerate(order)}

    tab, tests = [], []
    for pc in PCS:
        rise_all = (d[pc] == 'rise').mean()
        for t in order:
            m_t = d.type == t
            tend = 'rise' if (d[m_t][pc] == 'rise').mean() > rise_all else 'decline'
            for r in ('decline', 'rise'):
                m = m_t & (d[pc] == r)
                tab.append({'component': pc, 'type': t, 'direction': r, 'n': int(m.sum()),
                            'column_share': m.sum() / (d[pc] == r).sum(), 'type_share': m.sum() / m_t.sum(),
                            'tendency': tend == r})
            v = d.loc[m_t, f'd{pc}']
            tests.append({'component': pc, 'type': t, 'n': len(v), 'mean_change': v.mean(), 'median_change': v.median(),
                          'rise_share': (v > 0).mean(), 'p_wilcoxon': wilcoxon(v).pvalue})
    tab, tests = pd.DataFrame(tab), pd.DataFrame(tests)
    tab.merge(tests, on=['component', 'type']).to_csv(fig_dir / 'fig_supply_change_pcs_table.csv', index=False)
    d.to_csv(fig_dir / 'fig_supply_change_pcs_communities.csv', index=False)

    nature_style()
    fig = plt.figure(figsize=(183 * MM, 165 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.2, left=0.075, right=0.99, top=0.97, bottom=0.07)
    top_gs = outer[0].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.1)
    bot_gs = outer[1].subgridspec(1, 2, width_ratios=[1, 1.2], wspace=0.08)

    # (a) parallel sets over the type and the three components
    ax = fig.add_subplot(top_gs[0])
    ax.set_axis_off()
    n_tot = len(d)
    paths = d.groupby(['type'] + list(PCS)).size().reset_index(name='n')
    paths['key'] = [(tidx[t],) + tuple(0 if v == 'decline' else 1 for v in row)
                    for t, row in zip(paths.type, paths[list(PCS)].to_numpy())]
    paths = paths.sort_values('key').reset_index(drop=True)
    xs = [0.18, 0.45, 0.70, 0.95]
    w, gap = 0.028, 0.03
    unit = (1 - gap * (len(order) - 1)) / n_tot
    seg = {}
    y = 1.0
    nodes0 = {}
    for t in order:
        h = (d.type == t).sum() * unit
        nodes0[t] = (y, y - h)
        cur = y
        for i, p in paths[paths.type == t].iterrows():
            seg[(0, i)] = (cur, cur - p.n * unit)
            cur -= p.n * unit
        y -= h + gap
    unit_pc = (1 - 0.08) / n_tot
    nodes = {}
    for j, pc in enumerate(PCS, start=1):
        y = 1.0
        for r in ('decline', 'rise'):
            m = paths[pc] == r
            h = paths.loc[m, 'n'].sum() * unit_pc
            nodes[(pc, r)] = (y, y - h)
            cur = y
            sub = paths[m].copy()
            sub['k2'] = [(tuple(k[1:j]), k[0], tuple(k[j + 1:])) for k in sub.key]
            for i, p in sub.sort_values('k2').iterrows():
                seg[(j, i)] = (cur, cur - p.n * unit_pc)
                cur -= p.n * unit_pc
            y -= h + 0.08
    for i, p in paths.iterrows():
        for j in range(3):
            ribbon(ax, xs[j] + w, seg[(j, i)], xs[j + 1], seg[(j + 1, i)], TYPE_COLOR[p.type], alpha=0.42)
    for t in order:
        top, bot = nodes0[t]
        ax.add_patch(Rectangle((xs[0], bot), w, top - bot, fc=TYPE_COLOR[t], ec='none'))
        ax.text(xs[0] - 0.012, (top + bot) / 2, t, ha='right', va='center')
    for j, pc in enumerate(PCS, start=1):
        for r in ('decline', 'rise'):
            top, bot = nodes[(pc, r)]
            ax.add_patch(Rectangle((xs[j], bot), w, top - bot, fc='#BDBDBD' if r == 'decline' else '#4D4D4D', ec='none'))
            ax.text(xs[j] + w / 2, (top + bot) / 2, 'Decline' if r == 'decline' else 'Rise', ha='center', va='center',
                    rotation=90, fontsize=5.5, color='black' if r == 'decline' else 'white')
        ax.text(xs[j] + w / 2, 1.03, PC_LABEL[pc], ha='center', va='bottom', fontsize=6, color=PC_COLOR[pc], fontweight='bold')
    ax.text(xs[0] + w / 2, 1.03, 'Basic supply\nconfiguration', ha='center', va='bottom', fontsize=6)
    ax.annotate('', xy=(0.012, 0.0), xytext=(0.012, 1.0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color='black'))
    ax.text(0.0, 1.0, 'Low', ha='left', va='bottom', fontsize=6)
    ax.text(0.0, -0.02, 'High', ha='left', va='top', fontsize=6)
    ax.text(-0.005, 0.5, 'Supply level', rotation=90, ha='right', va='center', fontweight='bold')
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.06, 1.13)
    ax.text(-0.03, 1.13, 'a', fontweight='bold', fontsize=8, va='top')

    # (b) matrix: rows types, three blocks of decline / rise
    ax = fig.add_subplot(top_gs[1])
    ax.set_axis_off()
    cw, ch, bg = 0.9, 1.0, 0.25
    for b_, pc in enumerate(PCS):
        cmap = mpl.colormaps[PC_CMAP[pc]]
        x0 = b_ * (2 * cw + bg)
        for i, t in enumerate(order):
            yy = (len(order) - 1 - i) * (ch + 0.06)
            for j, r in enumerate(('decline', 'rise')):
                row = tab[(tab.component == pc) & (tab.type == t) & (tab.direction == r)].iloc[0]
                ax.add_patch(Rectangle((x0 + j * cw, yy), cw - 0.04, ch, fc=cmap(0.12 + 0.75 * row.type_share), ec='none'))
                label = r if row.tendency else '--'
                dark = row.type_share > 0.6
                ax.text(x0 + j * cw + (cw - 0.04) / 2, yy + ch / 2, f'{label}\nN={row.n}\n({row.column_share * 100:.1f}%)',
                        ha='center', va='center', fontsize=4.8, linespacing=1.15, color='white' if dark else 'black')
        ax.text(x0 + cw - 0.02, len(order) * (ch + 0.06) + 0.05, pc, ha='center', va='bottom', fontsize=6.5,
                color=PC_COLOR[pc], fontweight='bold')
        for j, r in enumerate(('Decline', 'Rise')):
            ax.text(x0 + j * cw + (cw - 0.04) / 2, -0.1, r, ha='center', va='top', fontsize=5.5)
    for i, t in enumerate(order):
        ax.text(-0.08, (len(order) - 1 - i) * (ch + 0.06) + ch / 2, t, ha='right', va='center', fontsize=6)
    ax.set_xlim(-1.45, 3 * (2 * cw + bg))
    ax.set_ylim(-0.6, len(order) * (ch + 0.06) + 0.55)
    ax.text(-1.45, len(order) * (ch + 0.06) + 0.55, 'b', fontweight='bold', fontsize=8, va='top')

    # (c) three violins per type
    ax = fig.add_subplot(bot_gs[0])
    off = {'PC1': -0.27, 'PC2': 0.0, 'PC3': 0.27}
    rng = np.random.default_rng(0)
    ymax = max(d[f'd{pc}'].max() for pc in PCS)
    ymin = min(d[f'd{pc}'].min() for pc in PCS)
    for i, t in enumerate(order):
        for pc in PCS:
            v = d.loc[d.type == t, f'd{pc}'].to_numpy()
            p_ = i + off[pc]
            vp = ax.violinplot(v, positions=[p_], widths=0.25, showextrema=False)
            for b in vp['bodies']:
                b.set_facecolor(PC_COLOR[pc])
                b.set_edgecolor('none')
                b.set_alpha(0.45)
            ax.scatter(p_ + rng.uniform(-0.05, 0.05, len(v)), v, s=0.6, color=PC_COLOR[pc], edgecolor='none', alpha=0.8, zorder=3)
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([p_, p_], [q1, q3], color='black', lw=0.7, zorder=4)
            ax.plot([p_ - 0.06, p_ + 0.06], [q2, q2], color='black', lw=0.7, zorder=4)
            pv = tests[(tests.component == pc) & (tests.type == t)].p_wilcoxon.iloc[0]
            ax.text(p_, ymax + (ymax - ymin) * 0.03, stars(pv), ha='center', va='bottom', fontsize=5, color=PC_COLOR[pc])
    ax.axhline(0, color='#555555', lw=0.5, ls='--', zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=6)
    ax.set_ylabel('Change in component score')
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(ymin - (ymax - ymin) * 0.03, ymax + (ymax - ymin) * 0.12)
    handles = [plt.Line2D([], [], marker='s', ls='', ms=4, color=PC_COLOR[pc], alpha=0.7, label=PC_LABEL[pc].replace('\n', ' '))
               for pc in PCS]
    ax.legend(handles=handles, loc='lower left', fontsize=5.5, handletextpad=0.3, borderaxespad=0.2)
    ax.text(-0.09, 1.02, 'c', transform=ax.transAxes, fontweight='bold', fontsize=8, va='bottom')

    # (d) map of the eight direction combinations
    ax = fig.add_subplot(bot_gs[1])
    nodes_df = pd.read_csv(root / NODES, encoding='utf-8-sig').assign(SQCODE=lambda f: f.SQCODE.astype(str))
    poly = gpd.read_file(root / POLYGONS)[['SQCODE', 'geometry']].assign(SQCODE=lambda f: f.SQCODE.astype(str))
    poly = poly.merge(nodes_df[['SQCODE', 'OBJECTID']], on='SQCODE').rename(columns={'OBJECTID': 'community_id'})
    poly = poly.merge(d[['community_id', 'combo']], on='community_id', how='left')
    poly.plot(ax=ax, color=poly.combo.map(COMBO_COLOR).fillna('#EEEEEE'), edgecolor='white', linewidth=0.08)
    if (root / DISTRICT_FILE).exists():
        land = poly.geometry.buffer(50).union_all().buffer(-50)
        dist = gpd.read_file(root / DISTRICT_FILE).to_crs(poly.crs)
        dist = dist.assign(district=dist.dt_adcode.astype(str).map(DISTRICT_EN)).set_index('district')
        dist['geometry'] = dist.geometry.intersection(land)
    else:
        poly['district'] = poly.SQCODE.str[:6].map(DISTRICT)
        dist = poly.dissolve('district')
    dist.boundary.plot(ax=ax, color='#333333', linewidth=0.5)
    for name, geom in dist.geometry.items():
        pt = geom.representative_point()
        dx_, dy_ = LABEL_SHIFT.get(name, (0, 0))
        ax.text(pt.x + dx_, pt.y + dy_, name, ha='center', va='center', fontsize=5.5, fontstyle='italic', fontweight='bold')
    ax.set_axis_off()
    xmin, ymin_, xmax, ymax_ = poly.total_bounds
    sx, sy = xmin + (xmax - xmin) * 0.66, ymin_ - (ymax_ - ymin_) * 0.02
    for k_, (a0, a1) in enumerate([(0, 5000), (5000, 10000), (10000, 20000)]):
        ax.add_patch(Rectangle((sx + a0, sy), a1 - a0, (ymax_ - ymin_) * 0.012, fc='black' if k_ % 2 == 0 else 'white',
                               ec='black', lw=0.4))
    for v in (0, 5, 10, 20):
        ax.text(sx + v * 1000, sy - (ymax_ - ymin_) * 0.012, str(v), ha='center', va='top', fontsize=5)
    ax.text(sx + 20000 + (xmax - xmin) * 0.012, sy, 'km', ha='left', va='bottom', fontsize=5)
    nx, ny = xmax - (xmax - xmin) * 0.04, ymax_ - (ymax_ - ymin_) * 0.08
    ax.annotate('N', xy=(nx, ny + (ymax_ - ymin_) * 0.08), xytext=(nx, ny), ha='center', va='top', fontsize=6.5,
                fontweight='bold', arrowprops=dict(arrowstyle='-|>', lw=0.8, color='black'))
    ax.set_ylim(ymin_ - (ymax_ - ymin_) * 0.75, ymax_ + (ymax_ - ymin_) * 0.02)
    counts = d.combo.value_counts()
    lg = ax.inset_axes([0.02, 0.0, 0.62, 0.36])
    lg.set_axis_off()
    for bi, s1 in enumerate(('+', '-')):
        bx = bi * 2.5
        lg.text(bx + 1.0, 2.25, 'PC1 rise' if s1 == '+' else 'PC1 decline', ha='center', va='bottom', fontsize=5.5,
                fontweight='bold', color=PC_COLOR['PC1'])
        for ri, s2 in enumerate(('+', '-')):
            for ci, s3 in enumerate(('+', '-')):
                combo = s1 + s2 + s3
                lg.add_patch(Rectangle((bx + ci, 1 - ri), 0.96, 0.96, fc=COMBO_COLOR[combo], ec='white'))
                n_ = int(counts.get(combo, 0))
                lg.text(bx + ci + 0.48, 1 - ri + 0.48, f'{n_}', ha='center', va='center', fontsize=5,
                        color='white' if combo in ('+++', '+-+', '-++', '--+') else 'black')
            if bi == 0:
                lg.text(bx - 0.06, 1 - ri + 0.48, 'PC2 ' + ('rise' if s2 == '+' else 'decline'), ha='right', va='center',
                        fontsize=4.8, color=PC_COLOR['PC2'])
        for ci, s3 in enumerate(('+', '-')):
            lg.text(bx + ci + 0.48, -0.06, 'PC3\n' + ('rise' if s3 == '+' else 'decline'), ha='center', va='top',
                    fontsize=4.8, color=PC_COLOR['PC3'])
    lg.set_xlim(-1.1, 4.55)
    lg.set_ylim(-0.75, 2.6)
    ax.text(0.0, 1.0, 'd', transform=ax.transAxes, fontweight='bold', fontsize=8, va='bottom')

    for ext in ('pdf', 'svg'):
        fig.savefig(fig_dir / f'fig_supply_change_pcs.{ext}')
    fig.savefig(fig_dir / 'fig_supply_change_pcs.png', dpi=600)
    pd.set_option('display.width', 250)
    print(tests.round(4).to_string(index=False))
    print(counts.to_string())
    print('saved to', fig_dir)


if __name__ == '__main__':
    main()
