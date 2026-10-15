"""Resilience of communities that keep their basic type under the interaction against those whose type changes, with the
direction of the change (resilience_community.csv of resilience_v13.py).

A Scarcity community whose type changes is raised by compensation (Upgraded); a CHC-based, SH-based or Abundance
community whose type changes is lowered (Downgraded: it does not withstand the external demand pressure). Panels a-d
compare Retained with the changed communities of each type (two-sided Mann-Whitney U); panel e pools the three types
exposed to pressure with resilience centred on the mean of its type, tested with a stratified permutation test (labels
permuted within type, 10000 permutations, statistic the inverse-variance weighted mean of the within-type differences
Downgraded minus Retained).
Outputs in results_lineb/<name>/resilience: resilience_pressure.csv and fig_resilience_pressure.pdf/.svg/.png.

python 5lineb/v13_geoflow/resilience_pressure_v13.py --config <config>
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}
PRESSURE = ['CHC-based', 'SH-based', 'Abundance']


def tint(color, f):
    c = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(1 - f * (1 - c))


def strat_diff(df, lab):
    tot = w = 0.0
    for _, s in df.groupby('type'):
        a, b = s.resilience[lab[s.index] == 1], s.resilience[lab[s.index] == 0]
        if len(a) and len(b):
            k = 1 / (1 / len(a) + 1 / len(b))
            tot += k * (a.mean() - b.mean())
            w += k
    return tot / w


RETAINED, CHANGED = '#FFFFE0', '#FA6F5E'


def box(ax, data, labels, title, letter, ptxt):
    """Box of the reference style: full frame, light yellow and salmon boxes, orange median, white mean circle."""
    bp = ax.boxplot(data, positions=[0, 1], widths=0.3, patch_artist=True, showfliers=False,
                    medianprops=dict(color='#FFA500', lw=1.4), whiskerprops=dict(color='black', lw=1.1),
                    capprops=dict(color='black', lw=1.1), boxprops=dict(lw=1.1))
    for patch, c in zip(bp['boxes'], (RETAINED, CHANGED)):
        patch.set_facecolor(c)
        patch.set_edgecolor('black')
    for i, v in enumerate(data):
        ax.scatter(i, np.mean(v), s=70, facecolor='white', edgecolor='black', lw=1.1, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.4)
    ax.tick_params(direction='out', length=4, width=1.1)
    ax.set_title(title, fontweight='bold', fontsize=15)
    ax.text(-0.12, 1.06, f'({letter})', transform=ax.transAxes, fontsize=12)
    ax.text(0.03, 0.95, ptxt, transform=ax.transAxes, fontsize=10, va='top')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    out = ROOT / 'results_lineb' / cfg['name'] / 'resilience'
    d = pd.read_csv(out / 'resilience_community.csv', encoding='utf-8-sig', float_precision='round_trip').dropna(subset=['resilience'])
    d['resilience'] = d.resilience.round(10)
    d['direction'] = np.where(d.state == 'Retained', 'Retained', np.where(d.type == 'Scarcity', 'Upgraded', 'Downgraded'))
    rows = []
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 12})
    fig, grid = plt.subplots(3, 2, figsize=(10.5, 13.5))
    axes = grid.ravel()
    for ax, lab, t in zip(axes, 'abcd', ['Scarcity'] + PRESSURE):
        s = d[d.type == t]
        other = 'Upgraded' if t == 'Scarcity' else 'Downgraded'
        r_, c_ = s[s.direction == 'Retained'].resilience, s[s.direction == other].resilience
        p = mannwhitneyu(c_, r_, alternative='two-sided').pvalue
        rows.append({'panel': t, 'compare': f'{other} vs Retained', 'n_retained': len(r_), 'n_changed': len(c_),
                     'mean_retained': r_.mean(), 'mean_changed': c_.mean(), 'median_retained': r_.median(),
                     'median_changed': c_.median(), 'p': p, 'test': 'Mann-Whitney U'})
        box(ax, [r_, c_], [f'Retained (n={len(r_)})', f'{other} (n={len(c_)})'], t, lab, f'Mann-Whitney p = {p:.3g}')
        ax.set_ylim(0.45, 1.06)
    adv = d[d.type.isin(PRESSURE)].copy()
    adv['down'] = (adv.direction == 'Downgraded').astype(int)
    adv['centred'] = adv.resilience - adv.groupby('type').resilience.transform('mean')
    obs = strat_diff(adv, adv.down)
    rng = np.random.default_rng(0)
    cnt = sum(abs(strat_diff(adv, adv.groupby('type').down.transform(lambda s: rng.permutation(s.to_numpy())))) >= abs(obs)
              for _ in range(10000))
    p = (cnt + 1) / 10001
    r_, c_ = adv[adv.down == 0].centred, adv[adv.down == 1].centred
    rows.append({'panel': 'Pressure-exposed types pooled', 'compare': 'Downgraded vs Retained (within type)',
                 'n_retained': len(r_), 'n_changed': len(c_), 'mean_retained': r_.mean(), 'mean_changed': c_.mean(),
                 'median_retained': r_.median(), 'median_changed': c_.median(), 'p': p,
                 'test': f'stratified permutation, difference {obs:.3f}'})
    box(axes[4], [r_, c_], [f'Retained (n={len(r_)})', f'Downgraded (n={len(c_)})'], 'Three types pooled (within type)', 'e',
        f'Stratified permutation p = {p:.3g}')
    axes[4].axhline(0, color='#999999', lw=0.8, ls='--')
    axes[4].set_ylim(-0.42, 0.2)
    axes[5].set_axis_off()
    for i, ax in enumerate(axes[:5]):
        if i % 2 == 0:
            ax.set_ylabel('Resilience $R^F$' if i < 4 else '$R^F$ minus type mean', fontweight='bold', fontsize=15)
    for ax in (axes[3], axes[4]):
        ax.set_xlabel('Communities', fontweight='bold', fontsize=15)
    fig.tight_layout(h_pad=2.5)
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_resilience_pressure.{ext}')
    fig.savefig(out / 'fig_resilience_pressure.png', dpi=300)
    T = pd.DataFrame(rows)
    T.to_csv(out / 'resilience_pressure.csv', index=False)
    pd.set_option('display.width', 220)
    print(T.round(4).to_string(index=False))
    print('saved to', out)


if __name__ == '__main__':
    main()
