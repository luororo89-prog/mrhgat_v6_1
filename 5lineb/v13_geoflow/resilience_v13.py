"""Flow-based healthcare resilience of every community (manuscript section 3.5) and its comparison between communities
whose type changes and communities whose type is retained, by basic configuration type of a model_v13 run.

Service nodes are the destination grids of the outpatient trips (500 m grids, outpatient_grid_OD_home_tid.csv), the
trips of a residential grid are split over communities with origin_grid_population_weights.csv. For community n the
eligible nodes J_n_out are the destination grids outside its 15-minute walking circle (grid centroid not within the
circle, foot isochrone) with positive baseline trips of its residents; they are removed in random order without
replacement, q = k / N_n_out. Baseline demand stays fixed, surviving nodes keep their loads, the capacity of a node is
1.25 times its baseline load (all communities), so its spare capacity is 0.25 times that load. The demand displaced by a
removal is redistributed to the surviving nodes inside the circle of n with spare capacity by the distance-weighted
Huff model x_rj = min(b_j, u_r exp(-gamma d_rj) / sum_l exp(-gamma d_rl)), repeated over the nodes with spare capacity
left until no capacity or demand remains; the rest is unmet, spare capacity is updated after every allocation. d_rj is
the straight-line distance between the grid centroids in km, gamma the maximum likelihood estimate of the Huff model
with equal attractiveness on the baseline trips over all destination grids of the city. Resilience
R_n = 1/B sum_b integral_0^1 D_n(b)(q) / D_n0 dq (trapezoidal rule, B = 100 sequences, seed fixed per community),
D_n0 all baseline trips of the residents of n (local services included). The same gamma, capacity rule, B and q scale
for every community keep the values comparable over the city. A community without eligible nodes keeps R = 1
(no_external = True); a community whose circle contains no destination grid (has_circle False) has no node to
redistribute to. R is rounded to 10 decimals so that 1 - 1e-16 from the trapezoidal sums ties with 1.
Types: type_nominal and type_effective of consensus_community.csv (ruler_from of the run, English names of
plot_type_change_v13); changed when they differ. Comparison per type: Mann-Whitney U test (two-sided).
Outputs in results_lineb/<name>/resilience: resilience_community.csv, resilience_by_type.csv, gamma.json and
fig_resilience_changed_retained.pdf/.svg/.png.

SUPPLY_ROOT=<data root> python 5lineb/v13_geoflow/resilience_v13.py \
    --config 5lineb/v13_geoflow/configs/walk_sup1013_net2_v13_k2_s100_hold_iters7_disencode_routescore_topo_tau05_ruleorigin_road.yaml
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
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

NET = ROOT / '4inter_community_network' / 'results'
CRS = 32650
B = 100
COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}
G = {}


def tint(color, f):
    c = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(1 - f * (1 - c))


def estimate_gamma(C, D):
    """Maximum likelihood gamma of exp(-gamma d) with equal attractiveness over all destination grids."""
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


def simulate(n):
    """Resilience of community n (mean and sd over the B sequences)."""
    Arn, outside, inside, Lin, E = G['S'][n]
    D0 = float(Arn.sum())
    if D0 <= 0:
        return n, np.nan, np.nan, 0.0, 0, 0
    N = len(outside)
    if N == 0:
        return n, 1.0, 0.0, D0, 0, len(inside)
    rng = np.random.default_rng(int(n))
    perm = np.argsort(rng.random((B, N)), 1)
    spare = np.tile(0.25 * Lin, (B, 1)).astype(float)
    unmet = np.zeros(B)
    curve = np.ones((B, N + 1))
    for k in range(N):
        u = Arn[:, outside[perm[:, k]]].T.copy()
        for _ in range(20):
            if not len(inside):
                break
            avail = spare > 1e-9
            W = E[None] * avail[:, None, :]
            s = W.sum(2)
            P = W / np.where(s > 0, s, 1.0)[:, :, None]
            x = u[:, :, None] * P
            X = x.sum(1)
            ratio = np.where(X > 0, np.minimum(1.0, spare / np.where(X > 0, X, 1.0)), 0.0)
            a = x * ratio[:, None, :]
            got = a.sum(2)
            spare = spare - a.sum(1)
            u = u - got
            if got.sum() < 1e-9 or (u * (s > 0)).sum() < 1e-9:
                break
        unmet += u.sum(1)
        curve[:, k + 1] = 1 - unmet / D0
    R = np.trapezoid(curve, dx=1.0 / N, axis=1)
    return n, float(R.mean()), float(R.std()), float(D0), N, len(inside)


def init(state):
    G.update(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--report_only', action='store_true')
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    run = ROOT / 'results_lineb' / cfg['name']
    out = run / 'resilience'
    out.mkdir(exist_ok=True)
    if a.report_only:
        d = pd.read_csv(out / 'resilience_community.csv', encoding='utf-8-sig', float_precision='round_trip')
        d['resilience'] = d.resilience.round(10)
        d.to_csv(out / 'resilience_community.csv', index=False, encoding='utf-8-sig')
        report(d, out)
        return
    sroot = Path(os.environ['SUPPLY_ROOT'])
    od = pd.read_csv(NET / 'outpatient_grid_OD_home_tid.csv', encoding='utf-8-sig')
    wts = pd.read_csv(NET / 'origin_grid_population_weights.csv', encoding='utf-8-sig')
    grid = gpd.read_file(sroot / 'SZ_grid_500m/SZ_grid_500m/SZ_grid_500m.shp').to_crs(CRS)[['Tid', 'geometry']]
    grid['Tid'] = grid.Tid.astype(int)
    cent = grid.set_index('Tid').geometry.centroid
    homes = np.sort(od.home_tid.unique())
    dests = np.sort(od.end_tid.unique())
    hi, di = {t: i for i, t in enumerate(homes)}, {t: i for i, t in enumerate(dests)}
    C = np.zeros((len(homes), len(dests)), dtype=np.float32)
    np.add.at(C, (od.home_tid.map(hi).to_numpy(), od.end_tid.map(di).to_numpy()), od['count'].to_numpy(float))
    hx, hy = cent.loc[homes].x.to_numpy(), cent.loc[homes].y.to_numpy()
    dxy = cent.loc[dests]
    D = (np.sqrt((hx[:, None] - dxy.x.to_numpy()[None]) ** 2 + (hy[:, None] - dxy.y.to_numpy()[None]) ** 2) / 1000.0).astype(np.float32)
    gamma, nll = estimate_gamma(C, D)
    (out / 'gamma.json').write_text(json.dumps({'gamma_per_km': gamma, 'mean_neg_loglik': nll, 'destination_grids': len(dests),
                                                'residential_grids': len(homes), 'trips': float(C.sum())}, indent=1))
    print(f'gamma = {gamma:.4f} per km', flush=True)
    L = C.sum(0)
    circles = gpd.read_file(sroot / '等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp').to_crs(CRS)
    circles['objectid'] = pd.to_numeric(circles.objectid).astype(int)
    pts = gpd.GeoDataFrame({'Tid': dests}, geometry=cent.loc[dests].values, crs=CRS)
    hits = gpd.sjoin(pts, circles[['objectid', 'geometry']], how='inner', predicate='within')
    inside = {int(o): np.array(sorted(di[t] for t in g_.Tid)) for o, g_ in hits.groupby('objectid')}
    com = pd.read_csv(run / 'consensus_community.csv', encoding='utf-8-sig')
    S = {}
    for cid, g_ in wts[wts.weight > 0].groupby('community_id'):
        r = np.array([hi[t] for t in g_.home_tid if t in hi])
        if not len(r):
            continue
        wv = g_.set_index('home_tid').weight.loc[homes[r]].to_numpy()
        Arn = C[r].astype(float) * wv[:, None]
        ins = inside.get(int(cid), np.array([], int))
        out_mask = Arn.sum(0) > 0
        out_mask[ins] = False
        ins = ins[L[ins] > 0]
        keep = np.nonzero((Arn.sum(0) > 0) | np.isin(np.arange(len(dests)), ins))[0]
        pos = {j: i for i, j in enumerate(keep)}
        S[int(cid)] = (Arn[:, keep], np.array([pos[j] for j in np.nonzero(out_mask)[0]], int),
                       np.array([pos[j] for j in ins], int), L[ins].astype(float),
                       np.exp(-gamma * D[np.ix_(r, ins)].astype(float)) if len(ins) else np.zeros((len(r), 0)))
    del D
    ids = [int(c) for c in com.community_id if int(c) in S]
    state = {'S': S}
    with Pool(a.workers, initializer=init, initargs=(state,)) as pool:
        res = pool.map(simulate, ids, chunksize=4)
    R = pd.DataFrame(res, columns=['community_id', 'resilience', 'resilience_sd', 'baseline_demand', 'n_external_nodes',
                                   'n_inside_nodes'])
    R['resilience'] = R.resilience.round(10)  # 1 - 1e-16 from the trapezoidal sums counts as 1 (ties in the tests)
    R['no_external'] = R.n_external_nodes == 0
    R['has_circle'] = R.community_id.isin(inside.keys())
    d = com[['community_id', 'name', 'type_nominal', 'type_effective']].merge(R, on='community_id', how='left')
    d['type'] = d.type_nominal.map(TYPE_EN)
    d['type_effective_en'] = d.type_effective.map(TYPE_EN)
    d['state'] = np.where(d.type_nominal != d.type_effective, 'Changed', 'Retained')
    d.to_csv(out / 'resilience_community.csv', index=False, encoding='utf-8-sig')
    report(d, out)


def report(d, out):
    order = ['Scarcity', 'CHC-based', 'SH-based', 'Abundance']
    rows = []
    for t in order:
        for s in ('Retained', 'Changed'):
            v = d[(d.type == t) & (d.state == s)].resilience.dropna()
            rows.append({'type': t, 'state': s, 'n': len(v), 'mean': v.mean(), 'median': v.median(), 'q1': v.quantile(.25),
                         'q3': v.quantile(.75)})
        r_, c_ = (d[(d.type == t) & (d.state == s)].resilience.dropna() for s in ('Retained', 'Changed'))
        p = mannwhitneyu(c_, r_, alternative='two-sided').pvalue if len(r_) >= 3 and len(c_) >= 3 else np.nan
        rows[-1]['mannwhitney_p'] = rows[-2]['mannwhitney_p'] = p
    S = pd.DataFrame(rows)
    S.to_csv(out / 'resilience_by_type.csv', index=False)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    for ax, (lab, t) in zip(axes, zip('abcd', order)):
        data = [d[(d.type == t) & (d.state == s)].resilience.dropna().to_numpy() for s in ('Retained', 'Changed')]
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
    axes[0].set_ylabel('Flow-based resilience $R^F$', fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_resilience_changed_retained.{ext}')
    fig.savefig(out / 'fig_resilience_changed_retained.png', dpi=300)
    pd.set_option('display.width', 200)
    print(S.round(4).to_string(index=False))
    print('no external nodes:', int(d.no_external.sum()), '| no destination grid in the circle:', int((~d.has_circle).sum()),
          '| resilience summary', d.resilience.describe().round(4).to_dict())
    print('saved to', out)


if __name__ == '__main__':
    main()
