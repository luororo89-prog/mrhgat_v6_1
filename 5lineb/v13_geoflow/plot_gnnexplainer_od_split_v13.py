"""Panels (a) Cleveland dot plot and (b) heatmap of the masks of fig_gnnexplainer_od, one figure per basic configuration
type, drawn from the aggregated gnnexplainer_od.csv of gnnexplainer_od_v13 (--aggregate).
Outputs in results_lineb/<name>/explain: fig_gnnexplainer_od_<type>.pdf/.svg/.png.

python 5lineb/v13_geoflow/plot_gnnexplainer_od_split_v13.py --explain results_lineb/<name>/explain
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATS = ['Own TH', 'Own SH', 'Own CHC', 'Dest TH', 'Dest SH', 'Dest CHC', 'ADJ', 'CRS']
TYPE_COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}
CHANNELS = ('compensation', 'competition')


def plot(out, order):
    agg = pd.read_csv(out / 'gnnexplainer_od.csv')
    get = lambda t, w, f: agg[(agg.type == t) & (agg.what == w) & (agg.feature == f)].value.iloc[0]
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    for t in order:
        M = np.array([[get(t, c, f) for c in CHANNELS] for f in FEATS])
        n = int(agg[agg.type == t].n_origins.iloc[0])
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 7.4), gridspec_kw={'width_ratios': [1.0, 0.75]})
        for i in range(len(FEATS)):
            for j, c in enumerate(CHANNELS):
                ax.scatter(M[i, j], len(FEATS) - 1 - i, marker='o', s=80, zorder=3, linewidth=1.8,
                           edgecolor=TYPE_COLOR[t], facecolor=TYPE_COLOR[t] if c == 'competition' else 'white')
        ax.set_yticks(range(len(FEATS)))
        ax.set_yticklabels(FEATS[::-1])
        for y_ in (1.5, 4.5):
            ax.axhline(y_, ls='--', lw=0.8, color='#999999')
        ax.set_xlim(-2, 102)
        ax.grid(axis='y', color='#E5E5E5')
        ax.set_xlabel('Mask kept by the model (%)')
        ax.set_ylabel('Supply components and topology types', fontweight='bold')
        ax.set_title('(a) Per-Component Cleveland Dot Plot', fontweight='bold')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        hd = [plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor=TYPE_COLOR[t],
                         markeredgecolor=TYPE_COLOR[t], label=f'{t} Competition (filled)'),
              plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='white',
                         markeredgecolor=TYPE_COLOR[t], label=f'{t} Compensation (hollow)')]
        ax.legend(handles=hd, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=10)
        im = bx.imshow(M, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                bx.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=10,
                        color='white' if M[i, j] > 60 else '#333333')
        bx.set_xticks(range(len(CHANNELS)))
        bx.set_xticklabels([f'{t}\n{c.capitalize()}\nn={n}' for c in CHANNELS], fontsize=10)
        bx.set_yticks(range(len(FEATS)))
        bx.set_yticklabels(FEATS)
        for y_ in (2.5, 5.5):
            bx.axhline(y_, color='white', lw=3)
        bx.axvline(0.5, color='white', lw=3)
        bx.set_title('(b) Importance Heatmap (%)', fontweight='bold')
        fig.colorbar(im, ax=bx, fraction=0.05, pad=0.04, label='Importance (%)')
        fig.tight_layout()
        for ext in ('pdf', 'svg'):
            fig.savefig(out / f'fig_gnnexplainer_od_{t}.{ext}', bbox_inches='tight')
        fig.savefig(out / f'fig_gnnexplainer_od_{t}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print('saved to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--explain', required=True)
    ap.add_argument('--order', nargs='*', default=['Scarcity', 'CHC-based', 'SH-based', 'Abundance'])
    a = ap.parse_args()
    plot(Path(a.explain), a.order)


if __name__ == '__main__':
    main()
