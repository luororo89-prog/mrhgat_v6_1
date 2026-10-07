"""Mixture of three outflow mechanisms with feedback (model_v7): train to convergence, classify, permutation importance.

Per seed: feedback rounds until stable, edge-level mechanism shares, permutation importance of the own,
inflow and outflow feature groups. Mechanisms are named C1-C3 by flow-weighted mean distance (C1 shortest);
interpretation is left to the saved outputs.

python lineb/run_v7.py --config lineb/configs/walk_v4_pca5_v7.yaml
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from graph import build_graph  # noqa: E402
from ruler import standardize  # noqa: E402
from model_v7 import fit_v7, permutation_importance, GROUPS  # noqa: E402
from run_v2 import frozen  # noqa: E402

FIT_KEYS = ('d', 'hidden', 'lr', 'lr_decay', 'decay', 'epochs_init', 'epochs_round', 'damping', 'damping_min', 'osc_corr', 'tol_dp', 'tol_switch', 'tol_ll',
            'patience', 'max_rounds', 'batch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', type=int, nargs='*', help='rerun these seeds only; rows of the other seeds are kept')
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    root = HERE.parent
    out = root / 'results_lineb' / cfg['name']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    logf = (out / 'progress.log').open('a', encoding='utf-8')

    def log(msg):
        line = f'{datetime.now():%H:%M:%S} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    g = build_graph(cfg)
    g['x'] = standardize(g['raw'], frozen(root / cfg['ruler_from'])['main'])
    ids = np.asarray(g['ids'])
    seeds = a.seeds if a.seeds else cfg['seeds']
    summary, imp_rows = [], []
    if a.seeds and (out / 'summary.csv').exists():
        summary = [r for r in pd.read_csv(out / 'summary.csv').to_dict('records') if r['seed'] not in seeds]
        imp_rows = [r for r in pd.read_csv(out / 'importance_summary.csv').to_dict('records') if r['seed'] not in seeds]
    for seed in seeds:
        model, data, p, hist, converged = fit_v7(g, seed, log=log, **{k: cfg[k] for k in FIT_KEYS if k in cfg})
        ll0, p0, rows = permutation_importance(model, data, p, reps=cfg['reps'], seed=seed, batch=cfg.get('batch', 8192))
        km = np.array([(data.w * p0[:, k] * data.km).sum() / (data.w * p0[:, k]).sum() for k in range(3)])
        order = np.argsort(km)
        names = [f'C{j + 1}' for j in range(3)]
        rdir = out / f'seed{seed}'
        rdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), rdir / 'model.pt')
        H = pd.DataFrame(hist).rename(columns={f'share_{k}': f'share_{names[j]}' for j, k in enumerate(order)})
        H[['round', 'll', 'mean_dp', 'switch', 'rel_dll', 'residual', 'step_corr', 'damping'] + [f'share_{c}' for c in names]].to_csv(rdir / 'convergence.csv', index=False)
        np.savez(rdir / 'edge_mechanism.npz', origin_id=ids[data.eu], destination_id=ids[data.ev], flow=data.w,
                 distance_km=data.km, p=p0[:, order], p_feedback=p[:, order], channel_order=order)
        R = pd.DataFrame(rows).rename(columns={f'dP_{k}': f'dP_{names[j]}' for j, k in enumerate(order)})
        R.insert(0, 'seed', seed)
        R = R[['seed', 'group', 'rep', 'dll', 'switch'] + [f'dP_{c}' for c in names]]
        R.to_csv(rdir / 'importance_perm.csv', index=False)
        M = R.groupby('group', sort=False)[['dll', 'switch'] + [f'dP_{c}' for c in names]].mean()
        M['share'] = M['dll'] / M['dll'].sum()
        for grp in GROUPS:
            imp_rows.append({'seed': seed, 'group': grp, **M.loc[grp].to_dict()})
        shares = (data.w[:, None] * p0[:, order]).sum(0) / data.w.sum()
        summary.append({'seed': seed, 'converged': converged, 'rounds': int(hist[-1]['round']), 'll': ll0,
                        **{f'share_{c}': shares[j] for j, c in enumerate(names)},
                        **{f'mean_km_{c}': km[order[j]] for j, c in enumerate(names)},
                        'max_share_flow_weighted': float((data.w * p0.max(1)).sum() / data.w.sum())})
        log(f'seed {seed} converged {converged} rounds {hist[-1]["round"]} ll {ll0:.5f} shares {np.round(shares, 3).tolist()} '
            f'km {np.round(km[order], 2).tolist()}')
        summary.sort(key=lambda r: r['seed'])
        imp_rows.sort(key=lambda r: (r['seed'], GROUPS.index(r['group'])))
        pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
        pd.DataFrame(imp_rows).to_csv(out / 'importance_summary.csv', index=False)
    pd.set_option('display.width', 300)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    I = pd.DataFrame(imp_rows)
    print(I.groupby('group', sort=False)[['share', 'dll', 'switch', 'dP_C1', 'dP_C2', 'dP_C3']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
