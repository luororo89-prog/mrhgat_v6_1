"""Local network robustness of every community and its comparison between communities whose type changes and
communities whose type is retained, by basic configuration type of a model_v13 run.

Network: the inter-community medical network of the run (flow_file, all trips, self loops included). w_nm is the number
of trips of the residents of community n to community m. The local network of n is the part of its destination set in
its 15-minute walking neighbourhood, G_n = {m in {n} and the communities adjacent to n (adjacency_column of
adjacency_file): w_nm > 0}. Service nodes are the communities with positive baseline load L_m = sum_n w_nm; the capacity
of a node is 1.25 L_m for every node, so its spare capacity is 0.25 L_m.

Attack: random removal of service nodes over the whole city, one node per step without replacement, the same sequence
for all communities of a replicate, q = k / N with N the number of service nodes of the city. Baseline demand stays
fixed (all trips of the residents); the demand served at a removed node (baseline and redistributed) is displaced and
redistributed to the surviving nodes of the local network of its origin with spare capacity by the distance-weighted
Huff model
x_nm = min(b_m, u_n exp(-gamma d_nm) / sum_l exp(-gamma d_nl)), repeated over the nodes with spare capacity left until no
capacity or demand remains; the rest is unmet. All origins compete for the same spare capacity, which is updated after
every allocation. d_nm is the road walking distance between the communities (road_impedance of the config, km), gamma
the maximum likelihood estimate of the Huff model with equal attractiveness on the baseline trips of the network.

Robustness: P_n(q) = satisfied demand of the residents of n / baseline demand of n, and
R_n = 1/B sum_b integral_0^1 P_n(b)(q) dq (trapezoidal rule, B = 100 sequences, seed b). The same attack sequences,
q scale, capacity rule and gamma for every community keep the values comparable over the city.
Types: type_nominal and type_effective of consensus_community.csv (English names of plot_type_change_v13); changed when
they differ. Comparison per type: Mann-Whitney U test (two-sided).
Outputs in results_lineb/<name>/resilience_local: robustness_community.csv, robustness_by_type.csv, gamma.json and
fig_robustness_changed_retained.pdf/.svg/.png.

python 5lineb/v13_geoflow/resilience_local_v13.py \
    --config 5lineb/v13_geoflow/configs/walk_sup1013_net2_v13_k2_s100_hold_iters7_disencode_routescore_topo_tau05_ruleorigin_road.yaml
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from plot_type_change_v13 import TYPE_EN  # noqa: E402

B = 100
COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}
G = {}


def tint(color, f):
    c = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(1 - f * (1 - c))


def estimate_gamma(C, D):
    """Maximum likelihood gamma of exp(-gamma d) with equal attractiveness over all destination communities."""
    rows = C.sum(1) > 0
    C, D = C[rows], D[rows]
    tot = C.sum(1)

    def nll(gm):
        z = -gm * D
        m = z.max(1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(z - m).sum(1)))
        return -((C * z).sum() - (tot * lse).sum()) / tot.sum()
    res = minimize_scalar(nll, bounds=(0.0, 5.0), method='bounded')
    return float(res.x), float(res.fun)


def simulate(b):
    """Satisfied share of every community along one random attack sequence and its area under the curve."""
    W, E, L, nodes, D0 = G['W'], G['E'], G['L'], G['nodes'], G['D0']
    order = np.random.default_rng(b).permutation(nodes)
    N = len(order)
    A = W.copy()
    spare = 0.25 * L
    alive = L > 0
    curve = np.empty((N + 1, len(D0)))
    curve[0] = A.sum(1) / D0
    for k, m in enumerate(order):
        u = A[:, m].copy()
        A[:, m] = 0.0
        alive[m] = False
        spare[m] = 0.0
        rows = np.nonzero(u > 1e-12)[0]
        ur, Er = u[rows], E[rows]
        for _ in range(100):
            if not len(rows):
                break
            Wt = Er * (alive & (spare > 1e-9))[None]
            s = Wt.sum(1)
            x = Wt * (ur / np.where(s > 0, s, 1.0))[:, None]
            X = x.sum(0)
            ratio = np.where(X > 0, np.minimum(1.0, spare / np.where(X > 0, X, 1.0)), 0.0)
            a = x * ratio[None]
            got = a.sum(1)
            A[rows] += a
            spare = spare - a.sum(0)
            ur = ur - got
            if got.sum() < 1e-9 or (ur * (s > 0)).sum() < 1e-9:
                break
        curve[k + 1] = A.sum(1) / D0
    return np.trapezoid(curve, dx=1.0 / N, axis=0)


def init(state):
    G.update(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--report_only', action='store_true')
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    run = ROOT / 'results_lineb' / cfg['name']
    out = run / 'resilience_local'
    out.mkdir(exist_ok=True)
    if a.report_only:
        report(pd.read_csv(out / 'robustness_community.csv', encoding='utf-8-sig', float_precision='round_trip'), out)
        return
    net = pd.read_csv(cfg['flow_file'], index_col=0, encoding='utf-8-sig')
    ids = net.index.to_numpy(int)
    assert (net.columns.astype(int).to_numpy() == ids).all()
    W = net.to_numpy(float)
    road = np.load(ROOT / cfg['road_impedance'])
    assert (road['ids'] == ids).all()
    D = road['km'].astype(float)
    gamma, nll = estimate_gamma(W, D)
    L = W.sum(0)
    D0 = W.sum(1)
    (out / 'gamma.json').write_text(json.dumps({'gamma_per_km': gamma, 'mean_neg_loglik': nll, 'distance': cfg['road_impedance'],
                                                'service_nodes': int((L > 0).sum()), 'origins': int((D0 > 0).sum()),
                                                'trips': float(W.sum())}, indent=1))
    print(f'gamma = {gamma:.4f} per km', flush=True)
    adj = pd.read_csv(cfg['adjacency_file'], encoding='utf-8-sig')
    adj = adj[adj[cfg['adjacency_column']] == 1]
    pos = {c: i for i, c in enumerate(ids)}
    local = np.eye(len(ids), dtype=bool)
    local[adj.source_community_id.map(pos).to_numpy(), adj.target_community_id.map(pos).to_numpy()] = True
    local &= W > 0
    state = {'W': W, 'E': np.exp(-gamma * D) * local, 'L': L, 'nodes': np.nonzero(L > 0)[0],
             'D0': np.where(D0 > 0, D0, 1.0)}
    with Pool(a.workers, initializer=init, initargs=(state,)) as pool:
        Rb = np.array(pool.map(simulate, range(B)))
    R = pd.DataFrame({'community_id': ids, 'robustness': Rb.mean(0), 'robustness_sd': Rb.std(0), 'baseline_demand': D0,
                      'local_demand_share': (W * local).sum(1) / np.where(D0 > 0, D0, 1.0), 'n_local_nodes': local.sum(1)})
    R.loc[R.baseline_demand <= 0, ['robustness', 'robustness_sd']] = np.nan
    com = pd.read_csv(run / 'consensus_community.csv', encoding='utf-8-sig')
    d = com[['community_id', 'name', 'type_nominal', 'type_effective']].merge(R, on='community_id', how='left')
    d['type'] = d.type_nominal.map(TYPE_EN)
    d['type_effective_en'] = d.type_effective.map(TYPE_EN)
    d['state'] = np.where(d.type_nominal != d.type_effective, 'Changed', 'Retained')
    d.to_csv(out / 'robustness_community.csv', index=False, encoding='utf-8-sig')
    report(d, out)


def report(d, out):
    order = ['Scarcity', 'CHC-based', 'SH-based', 'Abundance']
    rows = []
    for t in order:
        for s in ('Retained', 'Changed'):
            v = d[(d.type == t) & (d.state == s)].robustness.dropna()
            rows.append({'type': t, 'state': s, 'n': len(v), 'mean': v.mean(), 'median': v.median(), 'q1': v.quantile(.25),
                         'q3': v.quantile(.75)})
        r_, c_ = (d[(d.type == t) & (d.state == s)].robustness.dropna() for s in ('Retained', 'Changed'))
        p = mannwhitneyu(c_, r_, alternative='two-sided').pvalue if len(r_) >= 3 and len(c_) >= 3 else np.nan
        rows[-1]['mannwhitney_p'] = rows[-2]['mannwhitney_p'] = p
    S = pd.DataFrame(rows)
    S.to_csv(out / 'robustness_by_type.csv', index=False)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    for ax, (lab, t) in zip(axes, zip('abcd', order)):
        data = [d[(d.type == t) & (d.state == s)].robustness.dropna().to_numpy() for s in ('Retained', 'Changed')]
        ok = [i for i, v in enumerate(data) if len(v)]
        bp = ax.boxplot([data[i] for i in ok], positions=[i for i in ok], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color='#FF8C00', lw=1.6), whiskerprops=dict(color='#333333'),
                        capprops=dict(color='#333333'))
        for patch, i in zip(bp['boxes'], ok):
            patch.set_facecolor(tint(COLOR[t], 0.3 if i == 0 else 0.85))
            patch.set_edgecolor('#333333')
        for i in ok:
            ax.scatter(i, data[i].mean(), s=60, facecolor='white', edgecolor='#333333', zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Retained\n(n={len(data[0])})', f'Changed\n(n={len(data[1])})'], fontweight='bold')
        p = S[S.type == t].mannwhitney_p.iloc[0]
        ax.set_title(f'({lab}) {t}', fontweight='bold')
        ax.set_xlabel(f'Mann-Whitney p = {p:.3g}' if np.isfinite(p) else '', fontsize=10, color='#555555')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('Local network robustness $R^L$', fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_robustness_changed_retained.{ext}')
    fig.savefig(out / 'fig_robustness_changed_retained.png', dpi=300)
    pd.set_option('display.width', 200)
    print(S.round(4).to_string(index=False))
    print('robustness summary', d.robustness.describe().round(4).to_dict())
    print('saved to', out)


if __name__ == '__main__':
    main()
