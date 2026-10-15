"""Box plots of the comparable flow-based resilience (resilience_city_v13.py) of Retained and Changed communities by
basic configuration type in the type colours of the type change figure (Retained light tint 0.45, Changed full colour), black median, white mean marker, closed frame, Arial, 2 x 2 panels.
Outputs in results_lineb/<name>/resilience_city: fig_resilience_retained_changed_box.pdf/.svg/.png.

python 5lineb/v13_geoflow/plot_resilience_city_v13.py --config <config>
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORDER = ['Scarcity', 'CHC-based', 'SH-based', 'Abundance']
COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}
FILL = {'Retained': 0.45, 'Changed': 1.0}


def tint(color, f):
    c = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(1 - f * (1 - c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    out = ROOT / 'results_lineb' / cfg['name'] / 'resilience_city'
    d = pd.read_csv(out / 'resilience_community.csv', encoding='utf-8-sig')
    S = pd.read_csv(out / 'resilience_by_type.csv')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'axes.linewidth': 1.4})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.8), sharey=True)
    for ax, lab, t in zip(axes.ravel(), 'abcd', ORDER):
        data = [d[(d.type == t) & (d.state == s)].resilience.dropna().to_numpy() for s in FILL]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.3, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', lw=1.2), whiskerprops=dict(color='black', lw=1.1),
                        capprops=dict(color='black', lw=1.1), boxprops=dict(edgecolor='black', lw=1.1))
        for patch, s in zip(bp['boxes'], FILL):
            patch.set_facecolor(tint(COLOR[t], FILL[s]))
        for i, v in enumerate(data):
            ax.scatter(i, v.mean(), s=80, facecolor='white', edgecolor='black', linewidths=1.1, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(FILL), fontweight='bold')
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(t, fontweight='bold', fontsize=14)
        ax.text(-0.02, 1.03, f'({lab})', transform=ax.transAxes, ha='right', va='bottom', fontsize=11)
        p = S[S.type == t].mannwhitney_p.iloc[0]
        ax.set_xlabel(f'Mann-Whitney p = {p:.3f}' if p >= 0.001 else 'Mann-Whitney p < 0.001', fontsize=11)
        for tl in ax.get_yticklabels():
            tl.set_fontweight('bold')
        ax.tick_params(width=1.2, length=4)
    for ax in axes[:, 0]:
        ax.set_ylabel('Flow-based resilience', fontweight='bold', fontsize=15)
    fig.tight_layout(w_pad=1.5, h_pad=2.0)
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_resilience_retained_changed_box.{ext}')
    fig.savefig(out / 'fig_resilience_retained_changed_box.png', dpi=300)
    print('saved to', out)


if __name__ == '__main__':
    main()
