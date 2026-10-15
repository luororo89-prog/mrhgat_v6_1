"""Effective supply of model_v13 against a simple flow-weighted model and a flow-driven 2SFCA (修订思路 section 6).

Inputs: a finished model_v13 run with its consensus (consensus_v13.py), the supply table, the flows and the adjacency
of its config; facilities and community polygons of the supply calculation (SUPPLY_ROOT, calculate_supply_indicators_
2018.py: CHC with their service capacity score, secondary and tertiary hospitals with their beds).
Methods (standardised indicators x of the frozen ruler of the run; pi_AB = W_AB / sum_B W_AB with the internal trips):
  basic          x_A;
  flow-weighted  x~_A = sum_B pi_AB x_B (B = A included), no mechanism split;
  model_v13      x~_A = x_A + dx_A (consensus mean over the seeds), split into the two outflow channels;
  2SFCA          flow-driven two-step floating catchment area on the observed trips (the visit-based catchments of
                 Kazazi & Li 2026 at the community level): supply-to-demand ratio of the destination B and tier l,
                     R_B,l = 10000 S_B,l / D_B,   S_B,l = capacity of tier l located in B,   D_B = sum_C W_CB (all trips
                     arriving in B, its own residents included),
                 and accessibility A_A,l = sum_B pi_AB R_B,l. The trips are not split by tier, so D_B is the demand of
                 all tiers. The insurance and age/sex need adjustments of SDA-2SFCA (Shao & Luo 2022) have no data here.
Tier scores of the flow-weighted model and model_v13: mean of the three standardised indicators of the tier
(ruler.tier_components); effective types with the rotated ruler --ruler.
Mechanism content (only model_v13 splits the outflow): per origin the outflow share of each channel; the channel whose
share has the higher mean Spearman correlation with the inflow pressure of the origin (model_v13.pressure_features) is
named competition, the other compensation (names from the run itself, no sign set in advance). They are compared with
measures of the 2SFCA that the model does not read: the external share of the trips arriving in A,
1 - W_AA / D_A, the own supply-to-demand ratio R_A,l (lower is more crowded), and, per outflow edge, the gap of the ratio
R_B,l - R_A,l (flow-weighted correlation with the routing share).
Outputs in results_lineb/<name>/comparison: community_methods.csv, tier_rank_correlation.csv, change_agreement.csv,
type_changes.csv, mechanism_origin.csv, mechanism_edges.csv, facilities_by_community.csv.

SUPPLY_ROOT=<data root> python 5lineb/v13_geoflow/compare_v13.py --config 5lineb/v13_geoflow/configs/walk_v1_net2_v13_k2_s100.yaml \
    --ruler results_lineb/walk_v1_pca5_rot
"""
import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(ROOT / 'src'))
from model_v13 import pressure_features  # noqa: E402
from graph import build_graph  # noqa: E402
from ruler import standardize, measure, tier_components, TIERS  # noqa: E402
from run_v2 import frozen  # noqa: E402

SUPPLY_SRC = '/Users/luoxuan/Documents/博后三胎/供给评价代码与数据/26上实验/medical_identification_validation/src'


def wcorr(a, b, w):
    a = a - (w * a).sum() / w.sum()
    b = b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def located_supply(ids):
    """[n, 3] capacity of the facilities located in every community (columns TH, SH, CHC)."""
    sys.path.insert(0, SUPPLY_SRC)
    import calculate_supply_indicators_2018 as base
    fac = base.build_facilities().to_crs(base.PROJECTED_CRS)
    poly = gpd.read_file(base.COMMUNITY_PATH).to_crs(base.PROJECTED_CRS)
    centers = gpd.read_file(base.CENTER_PATH)
    code_to_id = dict(zip(base.normalize_code(centers['SQCODE']), pd.to_numeric(centers['OBJECTID']).astype(int)))
    poly['community_id'] = base.normalize_code(poly['SQCODE']).map(code_to_id)
    hit = gpd.sjoin(fac[['facility_id', 'tier', 'capacity', 'geometry']], poly[['community_id', 'geometry']],
                    how='left', predicate='within')
    table = hit.pivot_table(index='community_id', columns='tier', values='capacity', aggfunc='sum').reindex(ids).fillna(0)
    counts = hit.pivot_table(index='community_id', columns='tier', values='facility_id', aggfunc='count').reindex(ids).fillna(0)
    unplaced = int(hit['community_id'].isna().sum())
    return table.reindex(columns=list(TIERS)).fillna(0).to_numpy(float), counts.reindex(columns=list(TIERS)).fillna(0), unplaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ruler', required=True)
    a = ap.parse_args()
    if 'SUPPLY_ROOT' not in os.environ:
        raise SystemExit('SUPPLY_ROOT is needed for the facilities and community polygons')
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    run = ROOT / 'results_lineb' / cfg['name']
    out = run / 'comparison'
    out.mkdir(exist_ok=True)
    g = build_graph(cfg)
    ids = np.asarray(g['ids'])
    n = len(ids)
    x = standardize(g['raw'], frozen(ROOT / cfg['ruler_from'])['main'])
    Rrot = frozen(ROOT / a.ruler)['main']
    W = g['W']
    pi = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    com = pd.read_csv(run / 'consensus_community.csv', encoding='utf-8-sig').set_index('community_id').loc[ids]
    x_v13 = x + com[[f'dx_{c}' for c in g['columns']]].to_numpy(float)
    x_fw = pi @ x
    T0, Tv, Tf = tier_components(x), tier_components(x_v13), tier_components(x_fw)

    S, counts, unplaced = located_supply(ids)
    D = W.sum(0)
    R = np.divide(1e4 * S, D[:, None], out=np.zeros_like(S), where=D[:, None] > 0)
    A2 = pi @ R
    ext = np.divide(D - np.diag(W), D, out=np.full(n, np.nan), where=D > 0)
    counts.assign(trips_arriving=D).to_csv(out / 'facilities_by_community.csv', encoding='utf-8-sig')

    # channels of the run: outflow share per origin, competition named by the pressure association
    e = pd.read_csv(run / 'consensus_edges.csv')
    chans = sorted(c[2:] for c in e.columns if c.startswith('p_C'))
    idx = {v: k for k, v in enumerate(ids.tolist())}
    oi, di = e.origin_id.map(idx).to_numpy(), e.destination_id.map(idx).to_numpy()
    f = e.flow.to_numpy(float)
    tot = np.bincount(oi, weights=f, minlength=n)
    share = {ch: np.divide(np.bincount(oi, weights=f * e[f'p_{ch}'].to_numpy(), minlength=n), tot,
                           out=np.full(n, np.nan), where=tot > 0) for ch in chans}
    pf = pressure_features(g)
    assoc = {}
    for ch in chans:
        rho = []
        for j in range(3):
            has = (pf[:, 3 + j] > 0) & np.isfinite(share[ch])
            rho.append(spearmanr(share[ch][has], pf[has, j]).statistic)
        assoc[ch] = float(np.mean(rho))
    compet = max(chans, key=lambda c: assoc[c])
    names = {ch: ('competition' if ch == compet else 'compensation') for ch in chans}

    table = pd.DataFrame({'community_id': ids, 'name': g['names'], 'external_share_arriving': ext, 'trips_arriving': D})
    for j, t in enumerate(TIERS):
        table[f'basic_{t}'] = T0[:, j]
        table[f'flow_weighted_{t}'] = Tf[:, j]
        table[f'v13_{t}'] = Tv[:, j]
        for ch in chans:
            table[f'v13_{names[ch]}_{t}'] = com[f'd_tier_{t}_{ch}'].to_numpy()
        table[f'located_capacity_{t}'] = S[:, j]
        table[f'sdr_own_{t}'] = R[:, j]
        table[f'2sfca_{t}'] = A2[:, j]
    for ch in chans:
        table[f'outflow_share_{names[ch]}'] = share[ch]
    table.to_csv(out / 'community_methods.csv', index=False, encoding='utf-8-sig')

    rank = []
    for j, t in enumerate(TIERS):
        M = pd.DataFrame({'basic': T0[:, j], 'flow-weighted': Tf[:, j], 'model_v13': Tv[:, j], '2SFCA': A2[:, j]})
        c = M.corr(method='spearman')
        for i_, r_ in enumerate(c.index):
            for c_ in c.columns[i_ + 1:]:
                rank.append({'tier': t, 'method_a': r_, 'method_b': c_, 'spearman': c.loc[r_, c_]})
    pd.DataFrame(rank).to_csv(out / 'tier_rank_correlation.csv', index=False)

    chg = []
    for j, t in enumerate(TIERS):
        dv, df_ = Tv[:, j] - T0[:, j], Tf[:, j] - T0[:, j]
        row = {'tier': t, 'spearman_v13_vs_flow_weighted': spearmanr(dv, df_).statistic,
               'same_sign_share': float(np.mean(np.sign(dv) == np.sign(df_))),
               'v13_mean_abs': float(np.abs(dv).mean()), 'flow_weighted_mean_abs': float(np.abs(df_).mean())}
        for ch in chans:
            row[f'spearman_{names[ch]}_part_vs_flow_weighted'] = spearmanr(com[f'd_tier_{t}_{ch}'], df_).statistic
        chg.append(row)
    pd.DataFrame(chg).to_csv(out / 'change_agreement.csv', index=False)

    c0 = measure(x, Rrot)['class']
    cv, cf = measure(x_v13, Rrot)['class'], measure(x_fw, Rrot)['class']
    tn = list(Rrot['type_names'])
    types = pd.DataFrame({'type': tn, 'communities': np.bincount(c0, minlength=len(tn)),
                          'changed_v13': np.bincount(c0[cv != c0], minlength=len(tn)),
                          'changed_flow_weighted': np.bincount(c0[cf != c0], minlength=len(tn))})
    types.loc[len(types)] = ['all', n, int((cv != c0).sum()), int((cf != c0).sum())]
    types['v13_and_flow_weighted_same_effective_type'] = np.nan
    types.loc[types.type == 'all', 'v13_and_flow_weighted_same_effective_type'] = float((cv == cf).mean())
    types.to_csv(out / 'type_changes.csv', index=False, encoding='utf-8-sig')

    mech = []
    for ch in chans:
        s_ = share[ch]
        ok = np.isfinite(s_) & np.isfinite(ext)
        row = {'channel': ch, 'name': names[ch], 'pressure_association': assoc[ch],
               'flow_share': float((f * e[f'p_{ch}']).sum() / f.sum()),
               'rho_external_share_arriving': spearmanr(s_[ok], ext[ok]).statistic}
        for j, t in enumerate(TIERS):
            has = ok & (S[:, j] > 0)
            row[f'n_with_{t}'] = int(has.sum())
            row[f'rho_own_sdr_{t}'] = spearmanr(s_[has], R[has, j]).statistic if has.sum() > 5 else np.nan
        mech.append(row)
    pd.DataFrame(mech).to_csv(out / 'mechanism_origin.csv', index=False)
    medge = []
    for ch in chans:
        p_ = e[f'p_{ch}'].to_numpy()
        row = {'channel': ch, 'name': names[ch]}
        for j, t in enumerate(TIERS):
            row[f'wcorr_sdr_gap_{t}'] = wcorr(p_, R[di, j] - R[oi, j], f)
            row[f'wcorr_basic_gap_{t}'] = wcorr(p_, T0[di, j] - T0[oi, j], f)
        medge.append(row)
    pd.DataFrame(medge).to_csv(out / 'mechanism_edges.csv', index=False)

    pd.set_option('display.width', 250)
    print('facilities not inside a modelled community:', unplaced)
    for fname in ('tier_rank_correlation', 'change_agreement', 'type_changes', 'mechanism_origin', 'mechanism_edges'):
        print('\n' + fname)
        print(pd.read_csv(out / f'{fname}.csv').round(3).to_string(index=False))


if __name__ == '__main__':
    main()
