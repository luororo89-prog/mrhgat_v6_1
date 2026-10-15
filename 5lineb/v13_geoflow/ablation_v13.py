"""Channel ablations of model_v13 on the fixed held-out outflow edges (修订思路 section 6; 修订方案 7.2).

Retrained ablations ("删掉通道并重新训练", the predictive value that cannot be replaced): every config given is trained
on the seeds of the config with its held-out edges (holdout_share, holdout_seed), for example
  walk_v1_net2_v13_k2_s10_hold            full model: Base, two outflow channels, inflow pressure, channel attention
  walk_v1_net2_v13_k1_s10_hold            one outflow channel (no compensation/competition split)
  walk_v1_net2_v13_k2_s10_hold_onlybase   only Base (the outflow carries no message)
  walk_v1_net2_v13_k2_s10_hold_nobase     no real internal trips (A keeps weight 1)
  walk_v1_net2_v13_k2_s10_hold_noatt      no channel attention
  walk_v1_net2_v13_k2_s10_hold_nopressure no inflow pressure
Frozen ablations ("冻结模型后将通道关闭", the dependence of the trained model on a channel): the seeds of the first
config are evaluated with one outflow channel switched off (its outflow mass set to zero) and with Base switched off
(A keeps weight 0). The two kinds are reported apart.
Held-out metrics of model_v13.evaluate: CE (origins equal), CE weighted by held-out trips, CPC, NRMSE, internal share
error. Stability of the outflow assignment across seeds (two or more channels): flow-weighted Fleiss kappa after
aligning the channels to the seed with the lowest training CE (Hungarian method on the flow-weighted correlation of the
routing shares), mean pairwise ARI. Pressure association: per seed and channel, Spearman correlation of the origin's
outflow share in the channel with its inflow pressure of every tier (communities with that tier); the channel with the
higher mean association is the pressure channel, and the share of seeds where it is the same aligned channel is given.
Outputs in results_lineb/v13_ablation: retrained_per_seed.csv, retrained.csv, frozen_per_seed.csv, frozen.csv,
channels.csv; trained models in results_lineb/v13_ablation/<config name>/seed<s>/model.pt.

python 5lineb/v13_geoflow/ablation_v13.py walk_v1_net2_v13_k2_s10_hold walk_v1_net2_v13_k1_s10_hold ...
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(ROOT / 'src'))
from model_v13 import train_v13, evaluate, holdout_mask, pressure_features, road_matrix  # noqa: E402
from graph import build_graph  # noqa: E402
from ruler import standardize  # noqa: E402
from run_v2 import frozen  # noqa: E402

KEYS = ('n_channels', 'iters', 'epochs', 'lr', 'decay', 'rec_weight', 'tau0', 'tau_final', 'n_stages', 'stage_epochs',
        'stop_window', 'stop_tol', 'out_weight', 'pressure', 'attention', 'outflow', 'base', 'road_impedance', 'dest_fe',
        'holdout_share', 'holdout_seed')
OUT = ROOT / 'results_lineb' / 'v13_ablation'


def wcorr(a, b, w):
    a = a - (w * a).sum() / w.sum()
    b = b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def stability(P, ce, w):
    """Fleiss kappa (flow weighted) and mean pairwise ARI of the dominant channels; P [seeds, edges, K] aligned in place."""
    S_, _, K = P.shape
    ref = int(np.argmin(ce))
    for s in range(S_):
        C = np.array([[wcorr(P[ref][:, i], P[s][:, j], w) for j in range(K)] for i in range(K)])
        P[s] = P[s][:, linear_sum_assignment(-C)[1]]
    A = P.argmax(2)
    cnt = np.stack([(A == k).sum(0) for k in range(K)], 1)
    Pi = (cnt * (cnt - 1)).sum(1) / (S_ * (S_ - 1))
    pj = (cnt * w[:, None]).sum(0) / (S_ * w.sum())
    kappa = ((Pi * w).sum() / w.sum() - (pj ** 2).sum()) / (1 - (pj ** 2).sum())
    ari = np.mean([adjusted_rand_score(A[i], A[j]) for i, j in itertools.combinations(range(S_), 2)])
    return float(kappa), float(ari)


def main():
    names = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rows, frozen_rows, chan_rows, summary = [], [], [], []
    g, flow = None, None
    for name in names:
        cfg = yaml.safe_load((HERE / 'configs' / f'{name}.yaml').read_text(encoding='utf-8'))
        if g is None or cfg.get('flow_file') != flow:
            flow = cfg.get('flow_file')
            g = build_graph(cfg)
            g['x'] = standardize(g['raw'], frozen(ROOT / cfg['ruler_from'])['main'])
            W = g['W']
            n = len(W)
            obs = (W > 0) & ~np.eye(n, dtype=bool)
            oi, di = np.nonzero(obs)
            w = W[obs]
            pi = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
            share = torch.as_tensor(pi, dtype=torch.float32)
            T = [torch.as_tensor(g['x'], dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
                 torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
            pf = pressure_features(g)
            out_share = np.zeros(n)
            np.add.at(out_share, oi, pi[oi, di])
        H = holdout_mask(W, cfg['holdout_share'], cfg['holdout_seed'])
        Tcfg = list(T)
        if cfg.get('road_impedance'):
            Tcfg[1] = torch.as_tensor(road_matrix(g, cfg['road_impedance']), dtype=torch.float32)
        P, ce = [], []
        for seed in cfg['seeds']:
            m, h, tau = train_v13(g, seed, **{k: cfg[k] for k in KEYS if k in cfg})
            sdir = OUT / name / f'seed{seed}'
            sdir.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), sdir / 'model.pt')
            met = {k: v for k, v in h[-1].items() if k.startswith('hold_')}
            rows.append({'config': name, 'seed': seed, 'epochs': h[-1]['epoch'], 'train_ce': h[-1]['ce'], **met})
            print(name, seed, met, flush=True)
            ce.append(h[-1]['ce'])
            with torch.no_grad():
                P.append(m(*Tcfg, share, torch.as_tensor(obs), tau)['p'].numpy().transpose(0, 2, 1)[obs])
            if name == names[0]:
                for label, off, base_off in [(f'C{k + 1} off', {k}, False) for k in range(m.K)] + [('Base off', set(), True)]:
                    m.off, m.base_off = off, base_off
                    frozen_rows.append({'seed': seed, 'switched_off': label, **evaluate(m, Tcfg, share, W, H, tau)})
                    m.off, m.base_off = set(), False
                frozen_rows.append({'seed': seed, 'switched_off': 'none', **met})
        P = np.stack(P)
        K = P.shape[2]
        kappa, ari = stability(P, ce, w) if K > 1 and cfg.get('outflow', True) else (np.nan, np.nan)
        assoc = []
        if K > 1 and cfg.get('outflow', True):
            for s in range(len(P)):
                for k in range(K):
                    sk = np.zeros(n)
                    np.add.at(sk, oi, pi[oi, di] * P[s][:, k])
                    sk = np.divide(sk, out_share, out=np.full(n, np.nan), where=out_share > 0)
                    r = {'config': name, 'seed': s, 'channel': f'C{k + 1}', 'flow_share': float((w * P[s][:, k]).sum() / w.sum())}
                    for j, tier in enumerate(('CHC', 'SH', 'TH')):
                        has = (pf[:, 3 + j] > 0) & np.isfinite(sk)
                        r[f'rho_pressure_{tier}'] = float(spearmanr(sk[has], pf[has, j]).statistic)
                    assoc.append(r)
            a = pd.DataFrame(assoc)
            a['rho_mean'] = a[[c for c in a.columns if c.startswith('rho_pressure')]].mean(1)
            pressure_slot = a.loc[a.groupby('seed').rho_mean.idxmax(), 'channel']
            same = float(pressure_slot.value_counts().iloc[0] / len(P))
            chan_rows.append(a)
        else:
            same = np.nan
        r = pd.DataFrame([x for x in rows if x['config'] == name])
        summary.append({'config': name, 'seeds': len(r), **r.drop(columns=['config', 'seed']).mean().to_dict(),
                        **{f'{c}_sd': r[c].std() for c in ('hold_ce', 'hold_cpc')},
                        'kappa': kappa, 'ARI': ari, 'pressure_channel_same_slot': same})
    pd.DataFrame(rows).to_csv(OUT / 'retrained_per_seed.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'retrained.csv', index=False)
    fr = pd.DataFrame(frozen_rows)
    fr.to_csv(OUT / 'frozen_per_seed.csv', index=False)
    fr.groupby('switched_off').mean(numeric_only=True).drop(columns='seed').to_csv(OUT / 'frozen.csv')
    if chan_rows:
        pd.concat(chan_rows).to_csv(OUT / 'channels.csv', index=False)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(fr.groupby('switched_off').mean(numeric_only=True).drop(columns='seed').round(4).to_string())
    (OUT / 'configs.json').write_text(json.dumps(names, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
