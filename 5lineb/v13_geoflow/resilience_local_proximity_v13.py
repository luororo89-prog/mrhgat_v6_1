"""Local network robustness under the proximity assumption, compared with the network-based robustness of
resilience_local_v13.py (robustness_community.csv), in the layout of Fan et al. (2026, Nature Cities, Fig. 1b).

The two measures share the robustness formula, the service nodes, the demand of every community (all trips of its
residents), gamma, the capacity rule (1.25 times the baseline load), the redistribution and the B attack sequences
(seed b). They differ only in where the residents seek care:
- network-based: the observed trips w_nm, local network G_n = {m in {n} and the walking-adjacent communities: w_nm > 0};
- proximity-based: the demand D_n0 of community n is placed on the service nodes of its 15-minute walking neighbourhood
  (n and the communities adjacent to n, positive observed load) by the Huff model,
  w^P_nm = D_n0 exp(-gamma d_nm) / sum_l exp(-gamma d_nl), and the local network is that neighbourhood.
Fidelity of the proximity placement to the observed trips: common part of the trips CPC_n = sum_m min(w^P_nm, w_nm) / D_n0
and the share of the observed trips inside the neighbourhood.
Gap of a community: (R^prox - R^net) / R^net. Comparison of the two measures per type: OLS of R^net on R^prox and
Spearman rank correlation; rank shift of a community between the measures (percentile of R^prox minus percentile of
R^net). Retained and Changed compared within type under both measures (two-sided Mann-Whitney U).
Outputs in results_lineb/<name>/resilience_local: proximity_community.csv, proximity_by_type.csv,
proximity_changed_retained.csv and fig_robustness_proximity_vs_network.pdf/.svg/.png.

python 5lineb/v13_geoflow/resilience_local_proximity_v13.py --config <config>
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import linregress, mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import resilience_local_v13 as rl  # noqa: E402

ORDER = ['Scarcity', 'CHC-based', 'SH-based', 'Abundance']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    out = ROOT / 'results_lineb' / cfg['name'] / 'resilience_local'
    net = pd.read_csv(cfg['flow_file'], index_col=0, encoding='utf-8-sig')
    ids = net.index.to_numpy(int)
    W = net.to_numpy(float)
    road = np.load(ROOT / cfg['road_impedance'])
    assert (road['ids'] == ids).all()
    D = road['km'].astype(float)
    gamma, _ = rl.estimate_gamma(W, D)
    L, D0 = W.sum(0), W.sum(1)
    adj = pd.read_csv(cfg['adjacency_file'], encoding='utf-8-sig')
    adj = adj[adj[cfg['adjacency_column']] == 1]
    pos = {c: i for i, c in enumerate(ids)}
    hood = np.eye(len(ids), dtype=bool)
    hood[adj.source_community_id.map(pos).to_numpy(), adj.target_community_id.map(pos).to_numpy()] = True
    hood &= (L > 0)[None]
    K = np.exp(-gamma * D) * hood
    s = K.sum(1)
    WP = D0[:, None] * K / np.where(s > 0, s, 1.0)[:, None]
    LP = WP.sum(0)
    state = {'W': WP, 'E': K, 'L': LP, 'nodes': np.nonzero(L > 0)[0], 'D0': np.where(D0 > 0, D0, 1.0)}
    with Pool(a.workers, initializer=rl.init, initargs=(state,)) as pool:
        Rb = np.array(pool.map(rl.simulate, range(rl.B)))
    P = pd.DataFrame({'community_id': ids, 'robustness_proximity': Rb.mean(0), 'robustness_proximity_sd': Rb.std(0),
                      'cpc': np.minimum(WP, W).sum(1) / np.where(D0 > 0, D0, 1.0),
                      'observed_share_in_hood': (W * hood).sum(1) / np.where(D0 > 0, D0, 1.0),
                      'n_hood_nodes': hood.sum(1)})
    P.loc[(D0 <= 0) | (s <= 0), ['robustness_proximity', 'robustness_proximity_sd', 'cpc', 'observed_share_in_hood']] = np.nan
    d = pd.read_csv(out / 'robustness_community.csv', encoding='utf-8-sig').merge(P, on='community_id')
    d['gap'] = (d.robustness_proximity - d.robustness) / d.robustness
    ok = d.robustness.notna() & d.robustness_proximity.notna()
    d.loc[ok, 'rank_shift'] = (d[ok].robustness_proximity.rank(pct=True) - d[ok].robustness.rank(pct=True)) * 100
    d.to_csv(out / 'proximity_community.csv', index=False, encoding='utf-8-sig')
    report(d[ok], out)


def report(d, out):
    rows = []
    for t in ['All'] + ORDER:
        g = d if t == 'All' else d[d.type == t]
        f = linregress(g.robustness_proximity, g.robustness)
        rows.append({'type': t, 'n': len(g), 'mean_network': g.robustness.mean(), 'mean_proximity': g.robustness_proximity.mean(),
                     'mean_gap_pct': 100 * g.gap.mean(), 'share_overestimated': (g.gap > 0).mean(),
                     'spearman': spearmanr(g.robustness_proximity, g.robustness)[0], 'ols_slope': f.slope,
                     'ols_r2': f.rvalue ** 2, 'mean_rank_shift': g.rank_shift.mean(), 'mean_cpc': g.cpc.mean(),
                     'mean_observed_share_in_hood': g.observed_share_in_hood.mean()})
    S = pd.DataFrame(rows)
    S.to_csv(out / 'proximity_by_type.csv', index=False)
    rows = []
    for t in ORDER:
        for col, lab in (('robustness', 'network'), ('robustness_proximity', 'proximity')):
            r_, c_ = (d[(d.type == t) & (d.state == s)][col] for s in ('Retained', 'Changed'))
            rows.append({'type': t, 'measure': lab, 'n_retained': len(r_), 'n_changed': len(c_), 'mean_retained': r_.mean(),
                         'mean_changed': c_.mean(), 'mannwhitney_p': mannwhitneyu(c_, r_, alternative='two-sided').pvalue})
    C = pd.DataFrame(rows)
    C.to_csv(out / 'proximity_changed_retained.csv', index=False)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={'width_ratios': [1.15, 1]})
    lo = min(d.robustness.min(), d.robustness_proximity.min()) - 0.01
    hi = max(d.robustness.max(), d.robustness_proximity.max()) + 0.01
    ax.scatter(d.robustness_proximity, d.robustness, s=6, color='#9A9A9A', alpha=0.45, linewidths=0)
    ax.plot([lo, hi], [lo, hi], ls='--', color='black', lw=0.9)
    xs = np.linspace(lo, hi, 50)
    for t in ORDER:
        g = d[d.type == t]
        f = linregress(g.robustness_proximity, g.robustness)
        ax.plot(xs, f.intercept + f.slope * xs, color=rl.COLOR[t], lw=1.6, label=f'{t} ($R^2$ = {f.rvalue ** 2:.2f})')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Proximity-based local network robustness')
    ax.set_ylabel('Network-based local network robustness $R^L$')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    ax.set_title('a', loc='left', fontweight='bold')
    for i, t in enumerate(ORDER):
        v = 100 * d[d.type == t].gap.to_numpy()
        ax_ = bx.boxplot([v], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                         medianprops=dict(color='#333333', lw=1.2))
        ax_['boxes'][0].set_facecolor(rl.tint(rl.COLOR[t], 0.6))
        bx.scatter(i, v.mean(), s=36, facecolor='white', edgecolor='#333333', zorder=3)
    bx.axhline(0, ls='--', color='black', lw=0.9)
    bx.set_xticks(range(len(ORDER)))
    bx.set_xticklabels([f'{t}\n(n={int((d.type == t).sum())})' for t in ORDER])
    bx.set_ylabel('Gap of the proximity-based measure (%)')
    bx.set_title('b', loc='left', fontweight='bold')
    for a_ in (ax, bx):
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_robustness_proximity_vs_network.{ext}')
    fig.savefig(out / 'fig_robustness_proximity_vs_network.png', dpi=300)
    pd.set_option('display.width', 220)
    print(S.round(4).to_string(index=False))
    print(C.round(4).to_string(index=False))
    print('saved to', out)


if __name__ == '__main__':
    main()
