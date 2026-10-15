"""Consensus edge assignment over the seeds of a model_v10 run and the supply change it implies.

1  Load every trained seed of the run (channels aligned to its reference seed by run_v9, channel_order in summary.csv)
   and its routing shares p_uv,k with all observed outflow edges as messages.
2  Consensus channel of every observed non-self outflow edge: the channel with the highest routing share averaged over
   the seeds. Agreement = share of seeds whose dominant channel is the consensus channel.
3  Per seed, the aggregated vectors use the consensus assignment (one-hot) in place of the seed's own routing shares,
       c_u,k = l2(a_u s_u,k + sum_v pi_uv 1[k = consensus(u, v)] h_uv,k)
   with a_u the internal trip share when out_weight is share (1 otherwise, and 1 for a community without trips), and
   are decoded by the seed's joint decoder fitted on the self loop vectors: dx_u = D(c_u) - D(s_u); channel parts by
   replacing one block at a time.
4  Per community: mean change over the seeds with the 2.5% and 97.5% seed quantiles; a change is significant when that
   interval excludes 0. Effective indicators x~ = x + mean dx are measured with the fixed PCA ruler; a type change is
   robust when at least 80% of the seeds, each measured with its own dx, give the same changed type.
Outputs in results_lineb/<name>: consensus_edges.csv, consensus_channel_profiles.csv, consensus_community.csv,
consensus_by_type.csv, consensus_channel_by_type.csv, consensus_transition.csv, consensus_status.json.

With model: node in the config the run is a model_v11 run: the aggregated vectors mix node vectors,
c_u,k = l2(a_u zr_u,k + sum_v pi_uv 1[k = consensus(u, v)] zr_v,k), and the decoder is fitted on the node vectors zr.

python 5lineb/v10_edgedisen/consensus_v10.py --config 5lineb/v10_edgedisen/configs/walk_v4_pca5_v10_k3_s20.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
sys.path.insert(0, str(HERE.parent / 'v11_nodeedge'))
from graph import build_graph, ADJ  # noqa: E402
from ruler import standardize, measure, tier_components, TIERS  # noqa: E402
from decode import ALPHAS  # noqa: E402
from run_v2 import frozen  # noqa: E402
from model_v10 import EdgeDisen, l2n  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    root = HERE.parents[1]
    out = root / 'results_lineb' / cfg['name']
    K = cfg.get('n_channels', 3)
    CH = [f'C{k + 1}' for k in range(K)]
    R = frozen(root / cfg['ruler_from'])['main']
    type_names = json.loads((root / cfg['ruler_from'] / 'ruler.json').read_text(encoding='utf-8'))['main']['type_names']
    g = build_graph(cfg)
    x = standardize(g['raw'], R)
    W = g['W']
    n = len(W)
    idx = np.arange(n)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    oi, di = np.nonzero(obs)
    w = W[obs]
    xt = torch.as_tensor(x, dtype=torch.float32)
    T = [xt, torch.as_tensor(g['t'], dtype=torch.float32), torch.as_tensor(g['K'], dtype=torch.long),
         torch.as_tensor(g['candidates'])]
    share_np = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    share = torch.as_tensor(share_np, dtype=torch.float32)
    tau = cfg['tau_final']
    order = {int(r.seed): json.loads(str(r.channel_order)) for r in pd.read_csv(out / 'summary.csv').itertuples()}
    seeds = list(order)

    node = cfg.get('model') == 'node'

    def load(seed):
        if node:
            from model_v11 import NodeDisen, node_extras
            m = NodeDisen(iters=cfg.get('iters', 5), n_channels=K, out_weight=cfg.get('out_weight', 'one'),
                          **node_extras(g, cfg.get('inflow'), cfg.get('edge_topology', False)))
        else:
            m = EdgeDisen(iters=cfg.get('iters', 5), n_channels=K, self_weight=cfg.get('self_weight', 'one'),
                          out_weight=cfg.get('out_weight'))
        m.load_state_dict(torch.load(out / f'seed{seed}' / 'model.pt'))
        m.eval()
        return m

    E = []
    for seed in seeds:
        with torch.no_grad():
            p = load(seed)(*T, share, torch.as_tensor(obs), tau)['p'].numpy()[:, order[seed]]
        E.append(p.transpose(0, 2, 1)[obs].astype(float))
    E = np.stack(E)
    Pbar = E.mean(0)
    assign = Pbar.argmax(1)
    agreement = (E.argmax(2) == assign[None]).mean(0)

    onehot = torch.zeros(n, K, n)
    onehot[oi, assign, di] = torch.as_tensor(share_np[oi, di], dtype=torch.float32)
    if cfg.get('out_weight', cfg.get('self_weight', 'one')) == 'share':
        a_self = np.diag(share_np).copy()
        a_self[W.sum(1) <= 0] = 1.0
    else:
        a_self = np.ones(n)
    a_self = torch.as_tensor(a_self, dtype=torch.float32)
    DX, PARTS = [], []
    for seed in seeds:
        m = load(seed)
        with torch.no_grad():
            if node:
                s = l2n(m.embed(xt))[:, order[seed]]
                c = l2n(a_self[:, None, None] * s + torch.einsum('ukv,vkd->ukd', onehot, s))
            else:
                _, h = m.edges(xt)
                h = h[:, :, order[seed]]
                s = h[idx, idx]
                c = l2n(a_self[:, None, None] * s + torch.einsum('ukv,uvkd->ukd', onehot, h))
        S, C = s.numpy().reshape(n, -1).astype(float), c.numpy().reshape(n, -1).astype(float)
        dec = RidgeCV(alphas=ALPHAS).fit(S, x)
        base = dec.predict(S)
        d = s.shape[2]
        parts = []
        for k in range(K):
            Sk = S.copy()
            Sk[:, k * d:(k + 1) * d] = C[:, k * d:(k + 1) * d]
            parts.append(dec.predict(Sk) - base)
        DX.append(dec.predict(C) - base)
        PARTS.append(parts)
        print(f'seed {seed} decoded', flush=True)
    DX, PARTS = np.stack(DX), np.array(PARTS)
    DT = np.stack([tier_components(v) for v in DX])
    PT = np.stack([[tier_components(v) for v in ps] for ps in PARTS])

    T0 = tier_components(x)
    Tz = (T0 - T0.mean(0)) / T0.std(0)
    dTz = Tz[di] - Tz[oi]
    km = g['distance_km'][oi, di]
    adj = g['K'][oi, di] == ADJ
    edge = pd.DataFrame({'origin_id': g['ids'][oi], 'destination_id': g['ids'][di], 'topology': np.where(adj, 'ADJ', 'CRS'),
                         'flow': w, 'distance_km': km, 'channel': [CH[k] for k in assign], 'agreement': agreement,
                         **{f'p_{ch}': Pbar[:, k] for k, ch in enumerate(CH)}})
    edge.to_csv(out / 'consensus_edges.csv', index=False)
    prof = []
    for k, ch in enumerate(CH):
        m_ = assign == k
        wk = w[m_]
        prof.append({'channel': ch, 'flow_share': wk.sum() / w.sum(), 'edges': int(m_.sum()),
                     'mean_km': float((wk * km[m_]).sum() / wk.sum()), 'adjacent_share': float(wk[adj[m_]].sum() / wk.sum()),
                     **{f'd_{t}': float((wk * dTz[m_, j]).sum() / wk.sum()) for j, t in enumerate(TIERS)},
                     'mean_agreement': float((wk * agreement[m_]).sum() / wk.sum())})
    pd.DataFrame(prof).to_csv(out / 'consensus_channel_profiles.csv', index=False)

    lo, hi = np.percentile(DX, 2.5, axis=0), np.percentile(DX, 97.5, axis=0)
    tlo, thi = np.percentile(DT, 2.5, axis=0), np.percentile(DT, 97.5, axis=0)
    nominal = measure(x, R)
    eff = measure(x + DX.mean(0), R)
    per_cls = np.stack([measure(x + v, R)['class'] for v in DX])
    robust = (eff['class'] != nominal['class']) & ((per_cls == eff['class'][None]).mean(0) >= 0.8)
    com = pd.DataFrame({'community_id': g['ids'], 'name': g['names'], 'type_nominal': [type_names[c] for c in nominal['class']],
                        'type_effective': [type_names[c] for c in eff['class']],
                        'type_seed_share': (per_cls == eff['class'][None]).mean(0), 'robust_type_change': robust})
    for j, c in enumerate(g['columns']):
        com[f'dx_{c}'], com[f'dx_{c}_lo'], com[f'dx_{c}_hi'] = DX[:, :, j].mean(0), lo[:, j], hi[:, j]
    for j, t in enumerate(TIERS):
        com[f'd_tier_{t}'], com[f'd_tier_{t}_lo'], com[f'd_tier_{t}_hi'] = DT[:, :, j].mean(0), tlo[:, j], thi[:, j]
        com[f'sig_tier_{t}'] = np.where(tlo[:, j] > 0, 'up', np.where(thi[:, j] < 0, 'down', ''))
        for k, ch in enumerate(CH):
            com[f'd_tier_{t}_{ch}'] = PT[:, k, :, j].mean(0)
    com.to_csv(out / 'consensus_community.csv', index=False, encoding='utf-8-sig')
    pd.crosstab(com.type_nominal, com.type_effective).to_csv(out / 'consensus_transition.csv', encoding='utf-8-sig')
    by, byc = [], []
    for ci, nm in enumerate(type_names):
        f_ = com[nominal['class'] == ci]
        row = {'type': nm, 'n': len(f_), 'robust_type_changes': int(f_.robust_type_change.sum())}
        for t in TIERS:
            row.update({f'd_{t}': f_[f'd_tier_{t}'].mean(), f'{t}_sig_up': int((f_[f'sig_tier_{t}'] == 'up').sum()),
                        f'{t}_sig_down': int((f_[f'sig_tier_{t}'] == 'down').sum())})
            byc.append({'type': nm, 'tier': t, **{ch: f_[f'd_tier_{t}_{ch}'].mean() for ch in CH}})
        by.append(row)
    pd.DataFrame(by).to_csv(out / 'consensus_by_type.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(byc).to_csv(out / 'consensus_channel_by_type.csv', index=False, encoding='utf-8-sig')
    status = {'seeds': seeds, 'flow_share_agreement_ge_0.8': float(w[agreement >= 0.8].sum() / w.sum()),
              'flow_share_agreement_ge_0.5': float(w[agreement >= 0.5].sum() / w.sum()),
              'consensus_flow_share': {p_['channel']: round(p_['flow_share'], 4) for p_ in prof},
              'robust_type_changes': int(robust.sum()), 'type_changes_mean_dx': int((eff['class'] != nominal['class']).sum()),
              **{f'sig_communities_{t}': {'up': int((com[f'sig_tier_{t}'] == 'up').sum()), 'down': int((com[f'sig_tier_{t}'] == 'down').sum())}
                 for t in TIERS}}
    (out / 'consensus_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding='utf-8')
    pd.set_option('display.width', 250)
    print(json.dumps(status, ensure_ascii=False))
    print(pd.DataFrame(prof).round(3).to_string(index=False))
    print(pd.DataFrame(by).round(3).to_string(index=False))
    print(pd.crosstab(com.type_nominal, com.type_effective).to_string())


if __name__ == '__main__':
    main()
