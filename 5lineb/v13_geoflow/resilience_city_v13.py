"""Flow-based healthcare resilience of resilience_v13.py made comparable over the city, and its comparison between
communities whose type changes and communities whose type is retained, by basic configuration type of a model_v13 run.

Kept from resilience_v13.py: service nodes are the destination grids of the outpatient trips (500 m grids,
outpatient_grid_OD_home_tid.csv), the trips of a residential grid are split over communities with
origin_grid_population_weights.csv, the capacity of a grid is 1.25 times its baseline load (spare 0.25 times), displaced
demand goes to the surviving grids inside the 15-minute walking circle of the community of its origin (grid centroid
within the foot isochrone, positive load) with spare capacity by the distance-weighted Huff model
x_rj = min(b_j, u_r exp(-gamma d_rj) / sum_l exp(-gamma d_rl)), repeated over the grids with spare capacity left; the rest
is unmet. d_rj is the straight-line distance between the grid centroids in km, gamma the maximum likelihood estimate of
the Huff model with equal attractiveness on the baseline trips. D_n0 is all baseline trips of the residents of n.

Comparable over the city:
- one simulation for the whole city: all communities redistribute at the same time and compete for the same spare
  capacity, updated after every allocation;
- one attack for the whole city: random removal of all service grids (inside and outside the circles), one grid per
  step without replacement, the same sequence for all communities, q = k / N with N the number of service grids;
- demand placed on a grid that is removed later (baseline or redistributed) is displaced again.
Resilience R_n = 1/B sum_b integral_0^1 D_n(b)(q) / D_n0 dq (trapezoidal rule, B = 100 sequences, seed b).
Types: type_nominal and type_effective of consensus_community.csv; changed when they differ. Comparison per type:
Mann-Whitney U test (two-sided), report() of resilience_v13.py.
Outputs in results_lineb/<name>/resilience_city: resilience_community.csv, resilience_by_type.csv, gamma.json and
fig_resilience_changed_retained.pdf/.svg/.png.

SUPPLY_ROOT=<data root> python 5lineb/v13_geoflow/resilience_city_v13.py --config <config>
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from plot_type_change_v13 import TYPE_EN  # noqa: E402
from resilience_v13 import CRS, NET, estimate_gamma, report  # noqa: E402

B = 100
G = {}


def simulate(b):
    """Satisfied share of every community along one random attack sequence and its area under the curve."""
    base, rows, cols, e, ent, unit_com, D0, L = (G[k] for k in ('base', 'rows', 'cols', 'e', 'ent', 'unit_com', 'D0', 'L'))
    nu, nd, nc = base.shape[0], base.shape[1], len(D0)
    order = np.random.default_rng(b).permutation(np.nonzero(L > 0)[0])
    N = len(order)
    alive = L > 0
    spare = 0.25 * L
    Rs = np.zeros(len(rows))
    served_base = np.asarray(base.sum(1)).ravel()
    curve = np.empty((N + 1, nc))
    curve[0] = np.bincount(unit_com, served_base, nc) / D0
    for k, m in enumerate(order):
        lo, hi = base.indptr[m], base.indptr[m + 1]
        u = np.zeros(nu)
        np.add.at(u, base.indices[lo:hi], base.data[lo:hi])
        served_base[base.indices[lo:hi]] -= base.data[lo:hi]
        em = ent[m]
        np.add.at(u, rows[em], Rs[em])
        Rs[em] = 0.0
        alive[m] = False
        spare[m] = 0.0
        act = np.nonzero(u[rows] > 1e-12)[0]
        r_, c_, e_ = rows[act], cols[act], e[act]
        for _ in range(100):
            if not len(act):
                break
            wv = e_ * (alive & (spare > 1e-9))[c_]
            s = np.bincount(r_, wv, nu)
            x = np.where(s[r_] > 0, u[r_] * wv / np.where(s[r_] > 0, s[r_], 1.0), 0.0)
            X = np.bincount(c_, x, nd)
            ratio = np.where(X > 0, np.minimum(1.0, spare / np.where(X > 0, X, 1.0)), 0.0)
            a = x * ratio[c_]
            Rs[act] += a
            spare -= np.bincount(c_, a, nd)
            got = np.bincount(r_, a, nu)
            u -= got
            if got.sum() < 1e-9 or (u * (s > 0)).sum() < 1e-9:
                break
        served = served_base + np.bincount(rows, Rs, nu)
        curve[k + 1] = np.bincount(unit_com, served, nc) / D0
    return np.trapezoid(curve, dx=1.0 / N, axis=0)


def init(state):
    G.update(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    run = ROOT / 'results_lineb' / cfg['name']
    out = run / 'resilience_city'
    out.mkdir(exist_ok=True)
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
    L = C.sum(0).astype(float)
    circles = gpd.read_file(sroot / '等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp').to_crs(CRS)
    circles['objectid'] = pd.to_numeric(circles.objectid).astype(int)
    pts = gpd.GeoDataFrame({'Tid': dests}, geometry=cent.loc[dests].values, crs=CRS)
    hits = gpd.sjoin(pts, circles[['objectid', 'geometry']], how='inner', predicate='within')
    inside = {int(o): np.array(sorted(di[t] for t in g_.Tid)) for o, g_ in hits.groupby('objectid')}
    inside = {o: v[L[v] > 0] for o, v in inside.items()}
    com = pd.read_csv(run / 'consensus_community.csv', encoding='utf-8-sig')
    cpos = {int(c): i for i, c in enumerate(com.community_id)}
    units = wts[(wts.weight > 0) & wts.home_tid.isin(hi) & wts.community_id.isin(cpos)].reset_index(drop=True)
    r = units.home_tid.map(hi).to_numpy()
    base = sparse.csc_matrix(sparse.diags(units.weight.to_numpy(float)) @ sparse.csr_matrix(C[r].astype(float)))
    rows, cols = [], []
    for k, cid in enumerate(units.community_id.astype(int)):
        v = inside.get(cid, np.array([], int))
        rows.append(np.full(len(v), k))
        cols.append(v)
    rows, cols = np.concatenate(rows).astype(int), np.concatenate(cols).astype(int)
    e = np.exp(-gamma * D[r[rows], cols].astype(float))
    del D
    srt = np.argsort(cols, kind='stable')
    ent = np.split(srt, np.searchsorted(cols[srt], np.arange(1, len(dests))))
    unit_com = units.community_id.map(cpos).to_numpy()
    D0 = np.bincount(unit_com, np.asarray(base.sum(1)).ravel(), len(com))
    state = {'base': base, 'rows': rows, 'cols': cols, 'e': e, 'ent': ent, 'unit_com': unit_com,
             'D0': np.where(D0 > 0, D0, 1.0), 'L': L}
    with Pool(a.workers, initializer=init, initargs=(state,)) as pool:
        Rb = np.array(pool.map(simulate, range(B)))
    d = com[['community_id', 'name', 'type_nominal', 'type_effective']].copy()
    d['resilience'] = np.where(D0 > 0, Rb.mean(0), np.nan)
    d['resilience_sd'] = np.where(D0 > 0, Rb.std(0), np.nan)
    d['baseline_demand'] = D0
    d['n_inside_nodes'] = [len(inside.get(int(c), [])) for c in com.community_id]
    used = (sparse.csr_matrix((np.ones(len(units)), (unit_com, np.arange(len(units)))), shape=(len(com), len(units))) @ base).tocsr()
    d['n_external_nodes'] = [len(np.setdiff1d(used[i].indices, inside.get(int(c), []))) for i, c in enumerate(com.community_id)]
    d['no_external'] = d.n_external_nodes == 0
    d['has_circle'] = d.n_inside_nodes > 0
    d['type'] = d.type_nominal.map(TYPE_EN)
    d['type_effective_en'] = d.type_effective.map(TYPE_EN)
    d['state'] = np.where(d.type_nominal != d.type_effective, 'Changed', 'Retained')
    d.to_csv(out / 'resilience_community.csv', index=False, encoding='utf-8-sig')
    report(d, out)


if __name__ == '__main__':
    main()
