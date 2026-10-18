"""Type change from basic supply configuration to effective supply (panels a and b), Nature style.

Types are read from consensus_community.csv of a model_v11 run: type_nominal is the type of the nominal indicators x
under the fixed ruler, type_effective the type of x + dx with dx the mean supply change over the seeds. No composite
score is used; the types keep the order of their names (Scarcity, SH-scarcity, CHC-scarcity, SH-based, Abundance).
(a) Alluvial: basic supply configuration -> effective supply; ribbons in the colour of the basic type, ribbons of
    communities that change type drawn darker; the percentage on a type is the share of its communities that change type.
(b) Transition matrix: count and row share of every basic type in every effective type; cell colour of the effective
    type, shaded by the row share.
Outputs in results_lineb/<name>/figures: fig_type_change.pdf, .svg, .png and fig_type_change_table.csv (with the
number of robust type changes, at least 80% of the seeds giving the same changed type).
With --ruler (for example results_lineb/walk_v4_pca5_rot from rotate_ruler_v11.py) both types are measured with that
ruler from x and x + mean dx (dx from consensus_community.csv), in the type order of the ruler; the outputs carry the
ruler suffix (fig_type_change_rot.*) and the table has no robust column. When the ruler has rotated components
(loadings_rotated in its ruler.json), the panels show the change of every component score from S_base = x B to
S_eff = (x + dx) B by basic type, rise when the score increases and decline when it decreases:
(a) parallel sets basic type -> TH -> SH -> CHC (decline or rise on each axis);
(b) grid of basic type (x) by the change patterns of the three components (y, drawn as an UpSet combination matrix,
    ordered by size, patterns with fewer than MIN_PATTERN communities pooled); every cell is a bubble on one area
    scale, the light disc the number of communities and the dark disc inside the number whose effective type (nearest
    fixed centre of S_eff) differs from the basic type.
The state shares by basic type go to fig_type_change<suffix>_components.csv, the pattern counts to
fig_type_change<suffix>_patterns.csv.

python 5lineb/v11_nodeedge/plot_type_change_v11.py --config 5lineb/v11_nodeedge/configs/walk_v4_pca5_v11_k3_s10.yaml
    [--ruler results_lineb/walk_v4_pca5_rot]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from plot_supply_change_v11 import TYPE_EN, TYPE_COLOR, MM, nature_style, ribbon, stack  # noqa: E402
from graph import build_graph  # noqa: E402
from ruler import standardize, measure  # noqa: E402
from run_v2 import frozen  # noqa: E402

ORDER = ['Scarcity', 'SH-scarcity', 'CHC-scarcity', 'SH-based', 'Abundance']
TYPE_EN_ROT = {**TYPE_EN, '社康主导型（三甲低·二级低·社康高）': 'CHC-based', '中等型（三甲低·二级低·社康中）': 'Hospital-scarcity'}
DIRS = ['Decline', 'Rise']
DIR_COLOR = {'Decline': '#BDBDBD', 'Rise': '#4D4D4D'}
MIN_PATTERN = 10
FS_TITLE, FS_TICK, FS_LEG = 7, 6, 6
ARROW = {'Rise': dict(marker='^', s=34, color='#1A1A1A', edgecolor='none'),
         'Decline': dict(marker='v', s=34, color='#B3B3B3', edgecolor='none')}
STATE_SHADE = {'Retained': 0.25, 'Changed': 1.0}
COLOR = {**TYPE_COLOR, 'CHC-based': '#5AA84F', 'Hospital-scarcity': '#3C78B4', 'CHC-scarcity': '#3E9B57'}


def tint(color, share):
    """Colour mixed with white, from 15% colour at share 0 to full colour at share 1."""
    c = np.array(to_rgb(color))
    f = 0.15 + 0.85 * share
    return tuple(1 - f * (1 - c))


def alluvial(ax, d, order, n, share, color):
    """Basic type -> effective type."""
    left, scale = stack({t: n.loc[t].sum() for t in order}, len(d), 0.035)
    right, _ = stack({u: n[u].sum() for u in order}, len(d), 0.035)
    xl, xr, w = 0.27, 0.70, 0.035
    lo = {t: left[t][0] for t in order}
    ri = {u: right[u][0] for u in order}
    for t in order:
        for u in order:
            v = n.loc[t, u] * scale
            if v <= 0:
                continue
            ribbon(ax, xl + w, (lo[t], lo[t] - v), xr, (ri[u], ri[u] - v), color[t], alpha=0.3 if t == u else 0.75)
            lo[t] -= v
            ri[u] -= v
    for t in order:
        top, bot = left[t]
        ax.add_patch(Rectangle((xl, bot), w, top - bot, fc=color[t], ec='none'))
        ax.text(xl - 0.015, (top + bot) / 2, f'{t} ({n.loc[t].sum()})', ha='right', va='center')
        ax.text(xl + w + 0.012, (top + bot) / 2, f'{(1 - share.loc[t, t]) * 100:.1f}%', ha='left', va='center',
                fontsize=6, fontweight='bold')
        top, bot = right[t]
        ax.add_patch(Rectangle((xr, bot), w, top - bot, fc=color[t], ec='none'))
        ax.text(xr + w + 0.015, (top + bot) / 2, f'{t} ({n[t].sum()})', ha='left', va='center')
    ax.text(xl + w, 1.04, 'Basic supply configuration', ha='right', va='bottom', fontsize=6)
    ax.text(xr, 1.04, 'Effective supply', ha='left', va='bottom', fontsize=6)
    ax.text(xl + w + 0.012, 1.04, 'Changed', ha='left', va='bottom', fontsize=6, fontstyle='italic')
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.1)


def parallel_sets(ax, d, order, comps, color):
    """Basic type -> decline or rise of every component (parallel sets). Between two neighbouring axes the ribbons group
    the communities by basic type and the two categories; inside every node they are stacked by basic type, so the
    bundle of a basic type keeps its place on both sides of a node. One height per community on all axes; ribbons in
    the colour of the basic type."""
    p = d.assign(**{c: d[f's_{c}'] for c in comps})
    keys = ['base', *comps]
    cats = [order, *[DIRS] * len(comps)]
    rank = [{c: i for i, c in enumerate(cs)} for cs in cats]
    gaps = [0.035 if k == 'base' else 0.06 for k in keys]
    sizes = [{c: int((p[k] == c).sum()) for c in cs if (p[k] == c).any()} for k, cs in zip(keys, cats)]
    unit = min((1 - g * (len(sz) - 1)) / len(d) for g, sz in zip(gaps, sizes))
    nodes = []
    for g, sz in zip(gaps, sizes):
        y = 1 - (1 - unit * len(d) - g * (len(sz) - 1)) / 2
        nd = {}
        for c, v in sz.items():
            nd[c] = (y, y - v * unit)
            y -= v * unit + g
        nodes.append(nd)
    xs = [0.3, 0.52, 0.72, 0.92]
    w = 0.035
    for a_ in range(len(keys) - 1):
        ka, kb = keys[a_], keys[a_ + 1]
        seg = p.groupby(list(dict.fromkeys(['base', ka, kb]))).size().reset_index(name='n')
        seg['ra'], seg['rb'], seg['rt'] = seg[ka].map(rank[a_]), seg[kb].map(rank[a_ + 1]), seg['base'].map(rank[0])
        out = {c: nodes[a_][c][0] for c in nodes[a_]}
        inn = {c: nodes[a_ + 1][c][0] for c in nodes[a_ + 1]}
        y0, y1 = {}, {}
        for r in seg.sort_values(['ra', 'rt', 'rb']).index:
            c, h = seg.loc[r, ka], seg.n[r] * unit
            y0[r] = (out[c], out[c] - h)
            out[c] -= h
        for r in seg.sort_values(['rb', 'rt', 'ra']).index:
            c, h = seg.loc[r, kb], seg.n[r] * unit
            y1[r] = (inn[c], inn[c] - h)
            inn[c] -= h
        for r in seg.index:
            ribbon(ax, xs[a_] + w, y0[r], xs[a_ + 1], y1[r], color[seg.base[r]], alpha=0.45)
    for a_, (k, nd) in enumerate(zip(keys, nodes)):
        for c, (top, bot) in nd.items():
            if k == 'base':
                ax.add_patch(Rectangle((xs[a_], bot), w, top - bot, fc=color[c], ec='none'))
                ax.text(xs[a_] - 0.012, (top + bot) / 2, c, ha='right', va='center',
                        fontsize=FS_TICK)
            else:
                ax.add_patch(Rectangle((xs[a_], bot), w, top - bot, fc=DIR_COLOR[c], ec='none'))
                lab = c
                if top - bot > 0.12:
                    ax.text(xs[a_] + w / 2, (top + bot) / 2, lab, ha='center', va='center', rotation=90, fontsize=5.5,
                            color='black' if c == 'Decline' else 'white')
                else:
                    ax.text(xs[a_] + w + 0.008, (top + bot) / 2, lab, ha='left', va='center', fontsize=5)
    ax.text(xs[0] + w / 2, -0.035, 'Basic supply\nconfiguration', ha='center', va='top', fontsize=FS_TICK,
            linespacing=1.1)
    for a_, c in enumerate(comps, start=1):
        ax.text(xs[a_] + w / 2, -0.035, c, ha='center', va='top', fontsize=FS_TICK)
    x0_, x1_ = xs[1], xs[-1] + w
    ax.plot([x0_, x0_, x1_, x1_], [1.02, 1.03, 1.03, 1.02], color='black', lw=0.5)
    ax.text((x0_ + x1_) / 2, 1.045, 'Component change', ha='center', va='bottom', fontsize=FS_TITLE)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.12, 1.1)


def pattern_key(d, comps):
    """Change pattern of the components of every community, e.g. 'TH_Rise SH_Decline CHC_Rise'; patterns with
    fewer than MIN_PATTERN communities are pooled as 'Other'."""
    key = pd.Series([' '.join(f'{c}_{v}' for c, v in zip(comps, r)) for r in d[[f's_{c}' for c in comps]].to_numpy()],
                    index=d.index)
    n = key.value_counts()
    return key.where(key.map(n) >= MIN_PATTERN, 'Other')


def patterns(d, order, comps):
    """Counts and shares within the basic type of the change patterns of the components, ordered by size, 'Other'
    last."""
    cnt = pd.crosstab(pattern_key(d, comps), d.base).reindex(columns=order, fill_value=0)
    idx = [i for i in cnt.sum(1).sort_values(ascending=False, kind='stable').index if i != 'Other']
    cnt = cnt.loc[idx + (['Other'] if 'Other' in cnt.index else [])]
    return cnt, cnt / cnt.sum(0)


def pattern_matrix(ax, d, order, comps, color):
    """Dot plot in a framed grid (layout of the CellChat-style dot plots in Nature Communications): rows the change
    patterns of the components (rise or decline of TH, SH and CHC, as arrows in three aligned columns), ordered by the
    number of communities; columns the basic types. Every cell is a bubble on one area scale for all cells: the light
    disc the number of communities, the dark disc inside the number whose effective type (nearest fixed centre of S_eff)
    differs from the basic type. Legends on the right."""
    cnt, _ = patterns(d, order, comps)
    eff = pd.crosstab([pattern_key(d, comps), d.base], d.eff).reindex(columns=order, fill_value=0)
    m, k = len(cnt), len(order)
    nmax, smax = int(cnt.to_numpy().max()), 160
    ax.set_axis_on()
    for i, pat in enumerate(cnt.index):
        y = m - 1 - i
        for j, t in enumerate(order):
            c = int(cnt.loc[pat, t])
            if c == 0:
                continue
            n_ch = int(c - eff.loc[(pat, t), t])
            ax.scatter(j, y, s=smax * c / nmax, color=tint(color[t], 0.45), edgecolor='none', zorder=3)
            if n_ch:
                ax.scatter(j, y, s=smax * n_ch / nmax, color=tint(color[t], 1.0), edgecolor='none', zorder=4)
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(-0.5, m - 0.5)
    ax.set_xticks(range(k))
    ax.set_xticklabels(order, rotation=30, ha='right', rotation_mode='anchor', fontsize=FS_TICK)
    ax.set_yticks(range(m))
    ax.set_yticklabels([''] * m)
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, m), minor=True)
    ax.grid(which='minor', color='#BDBDBD', lw=0.5)
    ax.tick_params(which='minor', length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    tr = ax.get_yaxis_transform()
    xs_ = [-0.32, -0.21, -0.1]
    for x, c in zip(xs_, comps):
        ax.text(x, m - 0.5 + 0.15, c, transform=tr, ha='center', va='bottom', fontsize=FS_TICK)
    for i, pat in enumerate(cnt.index):
        for x, v in zip(xs_, (x_.split('_')[1] for x_ in pat.split())):
            ax.scatter(x, m - 1 - i, transform=tr, clip_on=False, **ARROW[v])
    ax.set_ylabel('Component change', fontsize=FS_TITLE, labelpad=40)
    ax.set_xlabel('Basic supply configuration', fontsize=FS_TITLE, labelpad=2)
    h1 = [plt.Line2D([], [], ls='', marker='o', markersize=(smax * v / nmax) ** 0.5, markerfacecolor='#C8C8C8',
                     markeredgecolor='none', label=str(v)) for v in (10, 40, 80)]
    h2 = [plt.Line2D([], [], ls='', marker='o', markersize=6.5, markerfacecolor='#C8C8C8', markeredgecolor='none',
                     label='Retained'),
          plt.Line2D([], [], ls='', marker='o', markersize=6.5, markerfacecolor='#6E6E6E', markeredgecolor='none',
                     label='Changed')]
    kw = dict(frameon=False, fontsize=FS_LEG, title_fontsize=FS_LEG, alignment='left', handletextpad=0.6,
              borderaxespad=0.0)
    lg = ax.legend(handles=h1, title='Communities', loc='upper left', bbox_to_anchor=(1.06, 1.0), labelspacing=1.3,
                   handleheight=1.5, **kw)
    ax.add_artist(lg)
    lg = ax.legend(handles=h2, title='Effective type', loc='upper left', bbox_to_anchor=(1.06, 0.55), **kw)
    ax.add_artist(lg)
    h3 = [plt.Line2D([], [], ls='', marker=ARROW[v]['marker'], markersize=ARROW[v]['s'] ** 0.5,
                     markerfacecolor=ARROW[v]['color'], markeredgecolor='none', label=v) for v in DIRS[::-1]]
    ax.legend(handles=h3, title='Component', loc='upper left', bbox_to_anchor=(1.06, 0.27), **kw)


def retained_bars(ax, d, order, color):
    """Paired bars above the dot plot, aligned with its columns: share of the communities of every basic type whose
    effective type is the basic type (light, left) and whose effective type differs (dark, right), values written
    above the bars; no y axis."""
    for j, t in enumerate(order):
        ch_ = float((d[d.base == t].eff != t).mean())
        for dxj, v, fc in ((-0.2, 1 - ch_, tint(color[t], 0.45)), (0.2, ch_, tint(color[t], 1.0))):
            ax.bar(j + dxj, v * 100, width=0.32, color=fc, edgecolor="none")
            ax.text(j + dxj, v * 100 + 4, f'{v * 100:.0f}%', ha='center', va='bottom', fontsize=5)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, 125)
    ax.set_axis_off()


def transition_matrix(ax, order, n, share, color):
    """Count and row share of every basic type in every effective type."""
    k = len(order)
    for i, t in enumerate(order):
        y = (k - 1 - i) * 1.05
        for j, u in enumerate(order):
            c, s = int(n.loc[t, u]), share.loc[t, u]
            fc = tint(color[u], s) if c > 0 else '#F7F7F7'
            ax.add_patch(Rectangle((j * 1.05, y), 1, 1, fc=fc, ec='white', lw=0.8))
            txt = f'{c}\n({s * 100:.1f}%)' if c > 0 else '0'
            dark = c > 0 and s > 0.6
            ax.text(j * 1.05 + 0.5, y + 0.5, txt, ha='center', va='center', fontsize=5.5, linespacing=1.2,
                    color='white' if dark else ('#999999' if c == 0 else 'black'), fontweight='bold' if i == j else 'normal')
        ax.text(-0.1, y + 0.5, t, ha='right', va='center')
    for j, u in enumerate(order):
        ax.text(j * 1.05 + 0.5, -0.12, u, ha='right', va='top', rotation=35, rotation_mode='anchor')
    ax.text(k * 1.05 / 2, k * 1.05 + 0.12, 'Effective supply', ha='center', va='bottom', fontsize=6)
    ax.text(-2.75, k * 1.05 / 2, 'Basic supply configuration', ha='center', va='center', rotation=90, fontsize=6)
    ax.set_xlim(-2.9, k * 1.05)
    ax.set_ylim(-1.2, k * 1.05 + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ruler', default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    out = HERE.parents[1] / 'results_lineb' / cfg['name']
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)
    com = pd.read_csv(out / 'consensus_community.csv', encoding='utf-8-sig')
    color = COLOR
    if a.ruler:
        root = HERE.parents[1]
        R = frozen(root / a.ruler)['main']
        g = build_graph(cfg)
        x = standardize(g['raw'], R)
        dx = com.set_index('community_id').loc[g['ids'], [f'dx_{c}' for c in g['columns']]].to_numpy()
        en = [TYPE_EN_ROT[t] for t in R['type_names']]
        order = en
        d = pd.DataFrame({'community_id': g['ids'], 'base': [en[c] for c in measure(x, R)['class']],
                          'eff': [en[c] for c in measure(x + dx, R)['class']]})
        tag = '_' + Path(a.ruler).name.replace(cfg['ruler_from'].split('/')[-1] + '_', '')
        rep = json.loads((root / a.ruler / 'ruler.json').read_text(encoding='utf-8'))['main']
        comps = list(rep['loadings_rotated']) if 'loadings_rotated' in rep else None
        if comps:
            Sb, Se = x @ R['U'], (x + dx) @ R['U']
            for j, c in enumerate(comps):
                d[f'd_{c}'] = Se[:, j] - Sb[:, j]
                d[f's_{c}'] = np.where(Se[:, j] > Sb[:, j], 'Rise', 'Decline')
    else:
        comps = None
        order = ORDER
        d = pd.DataFrame({'community_id': com.community_id, 'base': com.type_nominal.map(TYPE_EN),
                          'eff': com.type_effective.map(TYPE_EN), 'robust': com.robust_type_change})
        tag = ''
    n = pd.crosstab(d.base, d.eff).reindex(index=order, columns=order, fill_value=0)
    share = n.div(n.sum(axis=1), axis=0)
    tab = pd.DataFrame([{'basic_type': t, 'effective_type': u, 'n': int(n.loc[t, u]), 'row_share': share.loc[t, u]}
                        for t in order for u in order])
    if not a.ruler:
        nr = pd.crosstab(d.base[d.robust], d.eff[d.robust]).reindex(index=order, columns=order, fill_value=0)
        tab['robust_changes'] = [int(nr.loc[t, u]) if t != u else 0 for t in order for u in order]
    tab.to_csv(fig_dir / f'fig_type_change{tag}_table.csv', index=False)
    if comps:
        rise = pd.DataFrame([{'basic_type': t, 'component': c, 'n': int((d.base == t).sum()),
                              **{st: float((d.loc[d.base == t, f's_{c}'] == st).mean()) for st in DIRS}}
                             for t in order for c in comps])
        rise.to_csv(fig_dir / f'fig_type_change{tag}_components.csv', index=False)

    nature_style()
    fig = plt.figure(figsize=(183 * MM, (88 if comps else 72) * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05 if comps else 1.15, 1], wspace=0.08 if comps else 0.22, left=0.02,
                          right=0.99, top=0.9 if comps else 0.88, bottom=0.05 if comps else 0.17)
    fig.text(0.01, 0.98, 'a', fontweight='bold', fontsize=8, va='top')
    fig.text(gs[1].get_position(fig).x0 - 0.01, 0.98, 'b', fontweight='bold', fontsize=8, va='top')
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax_a.set_axis_off()
    ax_b.set_axis_off()
    if comps:
        pb = gs[1].get_position(fig)
        ax_b.set_position([pb.x0 + 0.1, 0.22, pb.width - 0.23, 0.53])
        ax_t = fig.add_axes([pb.x0 + 0.1, 0.75, pb.width - 0.23, 0.11])
    if comps:
        parallel_sets(ax_a, d, order, comps, color)
        pattern_matrix(ax_b, d, order, comps, color)
        retained_bars(ax_t, d, order, color)
        cnt, shr = patterns(d, order, comps)
        pd.concat({'n': cnt, 'share': shr}, axis=1).to_csv(fig_dir / f'fig_type_change{tag}_patterns.csv',
                                                           index_label='pattern')
    else:
        alluvial(ax_a, d, order, n, share, color)
        transition_matrix(ax_b, order, n, share, color)

    for ext in ('pdf', 'svg'):
        fig.savefig(fig_dir / f'fig_type_change{tag}.{ext}')
    fig.savefig(fig_dir / f'fig_type_change{tag}.png', dpi=600)
    pd.set_option('display.width', 200)
    print(n.to_string())
    print('changed', int((d.base != d.eff).sum()), 'of', len(d))
    if comps:
        print('share of communities by component state\n', rise.round(3).to_string(index=False))
        print(pd.concat({'n': cnt, 'share': shr.round(3)}, axis=1).to_string())
    print('saved to', fig_dir)


if __name__ == '__main__':
    main()
