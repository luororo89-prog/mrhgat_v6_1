"""Channel contributions to the component changes of a model_v13 run by basic configuration type: for every origin the
change of tier component j (d_tier_j of consensus_community.csv, mean over the seeds) is the sum of the contribution of
the compensation channel and of the competition channel (d_tier_j_C1 + d_tier_j_C2, channel names of
comparison/mechanism_origin.csv). Bars: mean contribution of each channel over the type's origins (hollow
compensation, filled competition), diamond: mean change, above the bars the origins whose component rises.
Outputs in results_lineb/<name>/explain: fig_channel_contrib.pdf/.svg/.png.

python 5lineb/v13_geoflow/plot_channel_contrib_v13.py --run results_lineb/<name>
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TYPE_EN = {'稀缺型（三甲低·二级低·社康低）': 'Scarcity', '二级主导型（三甲低·二级高·社康中）': 'SH-based',
           '社康主导型（三甲低·二级低·社康高）': 'CHC-based', '三甲社康型（三甲高·二级低·社康高）': 'Abundance'}
COMPS = ('TH', 'SH', 'CHC')
COLOR = {'TH': '#0072B2', 'SH': '#E69F00', 'CHC': '#009E73'}


def plot(run, order):
    com = pd.read_csv(run / 'consensus_community.csv', encoding='utf-8-sig')
    com['type'] = com.type_nominal.map(TYPE_EN)
    names = pd.read_csv(run / 'comparison' / 'mechanism_origin.csv').set_index('name').channel.to_dict()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 5.2))
    for ax, t in zip(axes, order):
        d = com[com.type == t]
        x = np.arange(len(COMPS))
        for i, c in enumerate(COMPS):
            comp, compet = d[f"d_tier_{c}_{names['compensation']}"].mean(), d[f"d_tier_{c}_{names['competition']}"].mean()
            tot = d[f'd_tier_{c}'].mean()
            ax.bar(i - 0.18, comp, 0.34, facecolor='white', edgecolor=COLOR[c], linewidth=1.8, zorder=2)
            ax.bar(i + 0.18, compet, 0.34, facecolor=COLOR[c], edgecolor=COLOR[c], linewidth=1.8, zorder=2)
            ax.scatter(i, tot, marker='D', s=60, color='#333333', zorder=3)
            ax.annotate(f'{tot:+.2f}', (i, tot), xytext=(8, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=10)
            ax.text(i, 1.0, f"rise {int((d[f'd_tier_{c}'] > 0).sum())}/{len(d)}", transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=10)
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(COMPS)
        ax.set_xlim(-0.6, len(COMPS) - 0.4)
        ax.grid(axis='y', color='#E5E5E5', zorder=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_title(f'{t} (n={len(d)})', fontweight='bold', pad=22)
    axes[0].set_ylabel('Change of the tier component', fontweight='bold')
    hd = [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#777777', linewidth=1.8, label='Compensation (hollow)'),
          plt.Rectangle((0, 0), 1, 1, facecolor='#777777', edgecolor='#777777', linewidth=1.8, label='Competition (filled)'),
          plt.Line2D([], [], ls='', marker='D', markersize=8, color='#333333', label='Total change')]
    fig.legend(handles=hd, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = run / 'explain'
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_channel_contrib.{ext}')
    fig.savefig(out / 'fig_channel_contrib.png', dpi=300)
    plt.close(fig)
    print('saved to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--order', nargs='*', default=['Scarcity', 'CHC-based', 'SH-based', 'Abundance'])
    a = ap.parse_args()
    plot(Path(a.run), a.order)


if __name__ == '__main__':
    main()
