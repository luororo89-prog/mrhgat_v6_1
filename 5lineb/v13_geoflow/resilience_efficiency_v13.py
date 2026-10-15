"""Resilience efficiency of the basic configuration types (resilience_community.csv of resilience_city_v13.py): resilience
gained over the Scarcity type per additional unit of resource availability.

Resource of a community: RA_CHC of the supply table of the run (resource availability of the CHC tier, facilities in the
15-minute circle weighted by (1 - lambda) + lambda C_j / median of the tier). RA_TH and RA_SH are not added: every
facility counts as a median facility of its own tier, so the tiers are not on one scale. Efficiency of type t:
       E_t = (mean R_t - mean R_Scarcity) / (mean RA_CHC,t - mean RA_CHC,Scarcity).
95% intervals and two-sided p of the differences between types from 5000 bootstrap resamples of the communities within
every type (p = twice the smaller share of resampled differences on either side of 0).
Outputs in results_lineb/<name>/resilience: efficiency.csv, efficiency_differences.csv,
fig_resilience_efficiency.pdf/.svg/.png and fig_resilience_efficiency_joint.pdf/.svg/.png (one figure: communities by
RA and resilience coloured by type, type means with 95% intervals joined in order of RA with the slope of every step
in the legend, marginal boxes).

fig_resilience_by_resource.pdf/.svg/.png: communities by RA and resilience with the least-squares line, 95% band and
Pearson r of every type. resilience_by_resource.csv: resilience by RA unit bins and type.

python 5lineb/v13_geoflow/resilience_efficiency_v13.py --config <config>
"""
import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'common'))
from graph import build_graph  # noqa: E402

TYPES = ['CHC-based', 'SH-based', 'Abundance']
COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}
B = 5000
RES = 'RA_CHC'


def joint_figure(d, E, D, out):
    """One figure: communities by resource (RA) and resilience as circles with a darker rim, type means with 95%
    intervals joined in order of their mean RA by a grey dashed line (the slope of every
    step in the legend; boxed main axes, inward ticks on the left and bottom), ridge densities of RA (top), slim boxes
    of resilience (right: 5-95% line, interquartile bar, median white dot) by type, and in the corner the resilience
    gain per unit RA over Scarcity with 95% intervals. Washes for the densities and a marker shape per type in the
    corner as second identity channel (the type colours of the paper fail the protan separation of orange and green)."""
    from matplotlib.colors import to_rgb
    from scipy.stats import gaussian_kde
    order = ['Scarcity', 'CHC-based', 'SH-based', 'Abundance']
    shape = {'Scarcity': 's', 'CHC-based': 'o', 'SH-based': '^', 'Abundance': 'D'}
    ink, muted, axis_c = '#1F1F1F', '#6B6B6B', '#B5B5B5'
    e = E.set_index('type')
    rng = np.random.default_rng(1)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.edgecolor': axis_c, 'axes.linewidth': 0.8,
                         'xtick.color': axis_c, 'ytick.color': axis_c, 'xtick.labelcolor': muted, 'ytick.labelcolor': muted,
                         'axes.labelcolor': ink, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8})
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.35], height_ratios=[1.35, 4], wspace=0.08, hspace=0.12)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)
    corner = fig.add_subplot(gs[0, 1])
    rim = {t: tuple(0.6 * np.array(to_rgb(COLOR[t]))) for t in order}
    for t in ['Abundance', 'SH-based', 'Scarcity', 'CHC-based']:
        s = d[d.type == t]
        ax.scatter(s[RES] + rng.uniform(-0.07, 0.07, len(s)), s.resilience + rng.uniform(-0.005, 0.005, len(s)),
                   s=16, marker='o', facecolor=COLOR[t], edgecolor=rim[t], alpha=0.75, linewidths=0.5, zorder=1)
    slope = {}
    chain = sorted(order, key=lambda t: e[f'mean_{RES}'][t])
    for t0, t in zip(chain[:-1], chain[1:]):
        xa, ya, xb, yb = e[f'mean_{RES}'][t0], e.mean_resilience[t0], e[f'mean_{RES}'][t], e.mean_resilience[t]
        ax.plot([xa, xb], [ya, yb], color=muted, lw=1.3, ls=(0, (4, 2.5)), dash_capstyle='round', zorder=2.5)
        slope[t] = (yb - ya) / (xb - xa)
    for t in order:
        s = d[d.type == t]
        mx, my = e[f'mean_{RES}'][t], e.mean_resilience[t]
        cx, cy = 1.96 * s[RES].std() / np.sqrt(len(s)), 1.96 * s.resilience.std() / np.sqrt(len(s))
        ax.plot([mx - cx, mx + cx], [my, my], color=COLOR[t], lw=1.2, zorder=3)
        ax.plot([mx, mx], [my - cy, my + cy], color=COLOR[t], lw=1.2, zorder=3)
        ax.scatter(mx, my, s=90, marker='o', facecolor=COLOR[t], edgecolor=rim[t], linewidths=0.9, zorder=4)
        ax.scatter([], [], s=60, marker='o', facecolor=COLOR[t], edgecolor=rim[t], linewidths=0.9,
                   label=t if t == chain[0] else f'{t}  ({slope[t]:.3f})')
    leg = ax.legend(loc='lower right', frameon=False, fontsize=8.5, handletextpad=0.4, labelspacing=0.5,
                    title='Type  (slope)', title_fontsize=8.5)
    leg._legend_box.align = 'left'
    for txt in leg.get_texts():
        txt.set_color(ink)
    ax.set_xlabel('CHC resource availability in the 15-minute circle (RA$_{CHC}$)', fontsize=10)
    ax.set_ylabel('Resilience $R^F$', fontsize=10)
    ax.set_ylim(d.resilience.min() - 0.03, d.resilience.max() + 0.025)
    ax.set_xlim(-0.3, min(d[RES].max() + 0.4, 16))
    for sp in ax.spines.values():
        sp.set_color('#000000')
    ax.tick_params(direction='in', top=False, right=False, color='#000000', labelcolor='#000000')
    # ridge densities
    for axm, key in ((top, RES),):
        lo, hi = (ax.get_xlim() if axm is top else (d.resilience.min(), 1.0))
        grid = np.linspace(lo, hi, 300)
        for k, t in enumerate(order):
            v = d[d.type == t][key].to_numpy()
            dens = gaussian_kde(v, bw_method=0.25 if axm is top else 0.18)(grid)
            dens = 0.85 * dens / dens.max()
            hl = t == 'CHC-based'
            if axm is top:
                axm.fill_between(grid, k, k + dens, color=COLOR[t], alpha=0.35 if hl else 0.18, linewidth=0)
                axm.plot(grid, k + dens, color=COLOR[t], lw=1.4 if hl else 0.9)
                axm.plot([np.median(v)] * 2, [k, k + 0.85], color=COLOR[t], lw=1.2)
            else:
                axm.fill_betweenx(grid, k, k + dens, color=COLOR[t], alpha=0.35 if hl else 0.18, linewidth=0)
                axm.plot(k + dens, grid, color=COLOR[t], lw=1.4 if hl else 0.9)
                axm.plot([k, k + 0.85], [np.median(v)] * 2, color=COLOR[t], lw=1.2)
        axm.tick_params(length=0)
        for sp in axm.spines.values():
            sp.set_visible(False)
    for k, t in enumerate(order):
        v = d[d.type == t].resilience.to_numpy()
        q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        hl = t == 'CHC-based'
        right.plot([k, k], [q05, q95], color=COLOR[t], lw=1.0, solid_capstyle='round')
        right.plot([k, k], [q25, q75], color=COLOR[t], lw=9 if hl else 7, alpha=0.85 if hl else 0.55, solid_capstyle='butt')
        right.scatter(k, q50, s=34, color='white', edgecolor=COLOR[t], linewidths=1.4, zorder=3)
    right.tick_params(length=0)
    for sp in right.spines.values():
        sp.set_visible(False)
    top.set_yticks(np.arange(len(order)) + 0.3)
    top.set_yticklabels(order, fontsize=8)
    top.tick_params(axis='x', labelbottom=False)
    top.set_ylim(-0.1, len(order))
    right.set_xticks(np.arange(len(order)))
    right.set_xticklabels(order, fontsize=8, rotation=90)
    right.tick_params(axis='y', labelleft=False)
    right.set_xlim(-0.6, len(order) - 0.4)
    # efficiency
    y_ = np.arange(len(TYPES))[::-1]
    for k, t in zip(y_, TYPES):
        hl = t == 'CHC-based'
        corner.plot([e.ci_low[t], e.ci_high[t]], [k, k], color=COLOR[t], lw=2 if hl else 1.2, solid_capstyle='round')
        corner.scatter(e.efficiency[t], k, s=80 if hl else 45, marker=shape[t], color=COLOR[t], edgecolor='white',
                       linewidths=1.4, zorder=3)
    corner.set_yticks([])
    p_ab = D.set_index('pair').p['CHC-based - Abundance']
    xb = max(e.ci_high['CHC-based'], e.ci_high['Abundance']) + 0.002
    corner.plot([xb, xb + 0.0015, xb + 0.0015, xb], [y_[0], y_[0], y_[2], y_[2]], color=muted, lw=0.8)
    corner.text(xb + 0.0025, (y_[0] + y_[2]) / 2, f'p = {p_ab:.3f}', va='center', fontsize=7.5, color=muted, rotation=90)
    corner.set_xlim(e.ci_low.min() - 0.003, xb + 0.008)
    corner.set_ylim(-0.6, len(TYPES) - 0.4)
    corner.set_title('Resilience gain per unit RA$_{CHC}$', fontsize=8.5, color=ink, pad=4)
    corner.tick_params(axis='x', labelsize=7.5)
    for sp in ('top', 'right', 'left'):
        corner.spines[sp].set_visible(False)
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_resilience_efficiency_joint.{ext}', bbox_inches='tight')
    fig.savefig(out / 'fig_resilience_efficiency_joint.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

def binned_figure(d, out, min_n=5):
    """Resilience by resource level (layout of the access-HALE scatter figures): every community as a translucent dot
    at its RA and resilience coloured by type, for every type the least-squares line over its RA range with the 95% band
    of the fitted mean, and the Pearson r, two-sided P and 95% interval of r (Fisher z) in the type colour.
    resilience_by_resource.csv: RA in unit bins (0-1, ..., 8 and more), for every type with at least min_n communities
    in a bin the mean and interquartile range of resilience."""
    from scipy.stats import pearsonr, t as t_dist
    order = ['Scarcity', 'CHC-based', 'SH-based', 'Abundance']
    shape = {'Scarcity': 's', 'CHC-based': 'o', 'SH-based': '^', 'Abundance': 'D'}
    ink = '#1F1F1F'
    d = d.assign(bin=np.minimum(np.floor(d[RES]), 8).astype(int))
    rows = []
    for t in order:
        for b in range(9):
            v = d[(d.type == t) & (d.bin == b)].resilience.to_numpy()
            if len(v) < min_n:
                continue
            q25, q75 = np.percentile(v, [25, 75])
            rows.append({'type': t, 'RA_bin': b, 'n': len(v), 'mean': v.mean(), 'q25': q25, 'q75': q75})
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.edgecolor': ink, 'axes.linewidth': 0.8,
                         'xtick.color': ink, 'ytick.color': ink, 'xtick.labelcolor': ink, 'ytick.labelcolor': ink,
                         'axes.labelcolor': ink, 'mathtext.fontset': 'stixsans'})
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for t in ['CHC-based', 'Scarcity', 'SH-based', 'Abundance']:
        s = d[d.type == t]
        ax.scatter(s[RES], s.resilience, s=26, marker=shape[t], color=COLOR[t], alpha=0.55, linewidths=0, zorder=1)
    for k, t in enumerate(order):
        s = d[d.type == t]
        x, y = s[RES].to_numpy(), s.resilience.to_numpy()
        n = len(x)
        slope, icpt = np.polyfit(x, y, 1)
        gx = np.linspace(x.min(), x.max(), 200)
        fit = icpt + slope * gx
        h = t_dist.ppf(0.975, n - 2) * np.sqrt(((y - icpt - slope * x) ** 2).sum() / (n - 2)
                                               * (1 / n + (gx - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()))
        ax.fill_between(gx, fit - h, fit + h, color=COLOR[t], alpha=0.2, linewidth=0, zorder=2)
        ax.plot(gx, fit, color=COLOR[t], lw=2, zorder=3)
        r = pearsonr(x, y)
        ci = r.confidence_interval()
        p = '$P$ < 0.001' if r.pvalue < 0.001 else f'$P$ = {r.pvalue:.3f}'
        yk = 0.97 - 0.16 * k
        ax.scatter(1.045, yk - 0.016, s=30, marker=shape[t], color=COLOR[t], transform=ax.transAxes, clip_on=False, zorder=4)
        ax.text(1.075, yk, f'{t}\n$r$ = {r.statistic:.2f}, {p}\n95% CI, ({ci.low:.2f}, {ci.high:.2f})',
                transform=ax.transAxes, color=COLOR[t], fontsize=8.5, linespacing=1.2, va='top')
    ax.set_xlim(-0.4, np.ceil(d[RES].max()) + 0.4)
    ax.set_ylim(d.resilience.min() - 0.04, d.resilience.max() + 0.03)
    ax.set_xlabel('CHC resource availability (RA$_{CHC}$)', fontsize=11)
    ax.set_ylabel('Resilience $R^F$', fontsize=11)
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_resilience_by_resource.{ext}', bbox_inches='tight')
    fig.savefig(out / 'fig_resilience_by_resource.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / 'resilience_by_resource.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    out = ROOT / 'results_lineb' / cfg['name'] / 'resilience'
    g = build_graph(cfg)
    cols = list(g['columns'])
    ra = pd.DataFrame({'community_id': g['ids'], **{f'RA_{t}': g['raw'][:, cols.index(f'RA_{t}')] for t in ('TH', 'SH', 'CHC')}})
    ra['RA_total'] = ra[['RA_TH', 'RA_SH', 'RA_CHC']].sum(1)
    d = pd.read_csv(out.parent / 'resilience_city' / 'resilience_community.csv', encoding='utf-8-sig',
                    float_precision='round_trip').merge(ra, on='community_id')
    d['resilience'] = d.resilience.round(10)
    groups = {t: d[d.type == t] for t in ['Scarcity'] + TYPES}

    def eff(sample):
        s = sample['Scarcity']
        return {t: (sample[t].resilience.mean() - s.resilience.mean()) / (sample[t][RES].mean() - s[RES].mean())
                for t in TYPES}

    point = eff(groups)
    rng = np.random.default_rng(0)
    boot = pd.DataFrame([eff({t: g_.iloc[rng.integers(0, len(g_), len(g_))] for t, g_ in groups.items()}) for _ in range(B)])
    rows = []
    for t in ['Scarcity'] + TYPES:
        g_ = groups[t]
        rows.append({'type': t, 'n': len(g_), 'mean_resilience': g_.resilience.mean(), 'mean_RA_total': g_.RA_total.mean(),
                     **{f'mean_RA_{c}': g_[f'RA_{c}'].mean() for c in ('TH', 'SH', 'CHC')},
                     'gain_over_scarcity': g_.resilience.mean() - groups['Scarcity'].resilience.mean() if t != 'Scarcity' else np.nan,
                     'extra_RA_over_scarcity': g_[RES].mean() - groups['Scarcity'][RES].mean() if t != 'Scarcity' else np.nan,
                     'efficiency': point.get(t, np.nan),
                     'ci_low': boot[t].quantile(0.025) if t in TYPES else np.nan,
                     'ci_high': boot[t].quantile(0.975) if t in TYPES else np.nan})
    E = pd.DataFrame(rows)
    E.to_csv(out / 'efficiency.csv', index=False)
    diffs = []
    for t1, t2 in itertools.combinations(TYPES, 2):
        dd = boot[t1] - boot[t2]
        diffs.append({'pair': f'{t1} - {t2}', 'difference': point[t1] - point[t2], 'ci_low': dd.quantile(0.025),
                      'ci_high': dd.quantile(0.975), 'p': min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))})
    D = pd.DataFrame(diffs)
    D.to_csv(out / 'efficiency_differences.csv', index=False)

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 12})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax_, sp_ in ((ax, None), (bx, None)):
        for sp in ax_.spines.values():
            sp.set_linewidth(1.4)
    e = E.set_index('type').loc[TYPES]
    ax.scatter(e.extra_RA_over_scarcity, e.gain_over_scarcity, s=140, c=[COLOR[t] for t in TYPES], edgecolor='black', zorder=3)
    for t in TYPES:
        right = t == TYPES[-1]
        ax.annotate(t, (e.extra_RA_over_scarcity[t], e.gain_over_scarcity[t]), textcoords='offset points',
                    xytext=(0, 12) if right else (8, -14), ha='center' if right else 'left', fontweight='bold')
    ax.plot([0, e.extra_RA_over_scarcity.max() * 1.1], [0, e.extra_RA_over_scarcity.max() * 1.1 * point['CHC-based']], ls='--',
            color=COLOR['CHC-based'], lw=1)
    ax.scatter([0], [0], s=140, c=COLOR['Scarcity'], edgecolor='black', zorder=3)
    ax.annotate('Scarcity', (0, 0), textcoords='offset points', xytext=(8, 4), fontweight='bold')
    ax.set_xlabel('Additional CHC resource over Scarcity ($\\Delta$RA$_{CHC}$)', fontweight='bold')
    ax.set_ylabel('Resilience gain over Scarcity ($\\Delta R^F$)', fontweight='bold')
    ax.set_title('Resilience gain and resource', fontweight='bold')
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes)
    x_ = np.arange(len(TYPES))
    bx.bar(x_, e.efficiency, width=0.5, color=['#FFFFE0' if t != 'CHC-based' else '#FA6F5E' for t in TYPES], edgecolor='black')
    bx.errorbar(x_, e.efficiency, yerr=[e.efficiency - e.ci_low, e.ci_high - e.efficiency], fmt='none', ecolor='black', capsize=6)
    bx.set_ylim(0, e.ci_high.max() * 1.35)
    bx.set_xticks(x_)
    bx.set_xticklabels([f'{t}\n(n={int(e.n[t])})' for t in TYPES], fontweight='bold')
    bx.set_ylabel('Resilience gain per unit RA$_{CHC}$', fontweight='bold')
    bx.set_title('Resilience efficiency', fontweight='bold')
    bx.text(-0.12, 1.05, '(b)', transform=bx.transAxes)
    txt = '\n'.join(f"{r.pair}: p = {r.p:.3g}" for r in D.itertuples() if r.pair.startswith('CHC-based'))
    bx.text(0.97, 0.95, txt, transform=bx.transAxes, ha='right', va='top', fontsize=10)
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_resilience_efficiency.{ext}')
    fig.savefig(out / 'fig_resilience_efficiency.png', dpi=300)
    joint_figure(d, E, D, out)
    binned_figure(d, out)
    pd.set_option('display.width', 220)
    print(E.round(4).to_string(index=False))
    print(D.round(4).to_string(index=False))
    print('saved to', out)


if __name__ == '__main__':
    main()
