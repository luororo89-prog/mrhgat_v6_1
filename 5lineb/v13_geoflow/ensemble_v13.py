"""Multi-seed ensemble destination prediction of a model_v13 run on its fixed held-out edges.

Only the destination prediction is averaged: for every held-out edge the predicted probability is the mean over the
seeds of the seed's candidate probabilities (bagging), and the held-out metrics of model_v13.evaluate are computed on
the averaged distribution. The routing shares and every other model output keep their per-seed values. Per-seed metrics
are written alongside the ensemble row for comparison.

python 5lineb/v13_geoflow/ensemble_v13.py --config 5lineb/v13_geoflow/configs/walk_v1_net2_v13_k2_s10_hold.yaml \
    --run results_lineb/v13_ablation/walk_v1_net2_v13_k2_s10_hold
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(ROOT / 'src'))
from model_v13 import MechNet, evaluate, holdout_mask, pressure_features, road_matrix  # noqa: E402
from graph import build_graph  # noqa: E402
from ruler import standardize  # noqa: E402
from run_v2 import frozen  # noqa: E402


def ensemble_metrics(models, T, share, W, H, tau):
    """Per-seed metrics of model_v13.evaluate and the metrics of the seed-averaged candidate probabilities."""
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    msg = torch.as_tensor(obs & ~H)
    ev_t = T[3] & ~msg & ~torch.eye(n, dtype=torch.bool)
    per_seed, probs, self_p = [], [], []
    for seed, model in models:
        met = evaluate(model, T, share, W, H, tau)
        met['seed'] = seed
        per_seed.append(met)
        with torch.no_grad():
            logpi_full = model(*T, share, msg, tau)['logpi']
            logpi = torch.log_softmax(logpi_full.masked_fill(~ev_t, -1e9), 1).numpy()
        probs.append(np.exp(logpi))
        self_p.append(np.exp(np.diagonal(logpi_full.numpy())))
    p = np.mean(probs, 0)
    ev = ev_t.numpy()
    F = W * H
    Fu = F.sum(1)
    rows = Fu > 0
    q = F[rows] / Fu[rows, None]
    ce_o = -(q * np.where(H[rows], np.log(np.clip(p[rows], 1e-30, None)), 0.0)).sum(1)
    Fh = Fu[rows, None] * p[rows] * ev[rows]
    m = ev[rows]
    self_pred = np.mean(self_p, 0)[rows]
    self_true = np.diagonal(W)[rows] / (np.diagonal(W)[rows] + Fu[rows])
    ens = {'seed': 'ensemble',
           'hold_ce': float(ce_o.mean()), 'hold_ce_trips': float((ce_o * Fu[rows]).sum() / Fu[rows].sum()),
           'hold_cpc': float(np.minimum(Fh, F[rows])[m].sum() / F[rows][m].sum()),
           'hold_nrmse': float(np.sqrt(np.mean((Fh[m] - F[rows][m]) ** 2)) / np.mean(F[rows][m])),
           'hold_internal_err': float(np.abs(self_pred - self_true).mean())}
    return per_seed, ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    torch.set_num_threads(cfg.get('threads', 8))
    g = build_graph(cfg)
    g['x'] = standardize(g['raw'], frozen(ROOT / cfg['ruler_from'])['main'])
    n = len(g['ids'])
    W = g['W']
    H = holdout_mask(W, cfg['holdout_share'], cfg['holdout_seed'])
    T = [torch.as_tensor(g['x'], dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
         torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
    if cfg.get('road_impedance'):
        T[1] = torch.as_tensor(road_matrix(g, cfg['road_impedance']), dtype=torch.float32)
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    run = ROOT / (a.run or f'results_lineb/{cfg["name"]}')
    n2v = cfg.get('n2v')
    models = []
    for seed in cfg['seeds']:
        m = MechNet(iters=cfg['iters'], n_channels=cfg['n_channels'], out_weight=cfg['out_weight'],
                    attention=cfg.get('attention', True), outflow=cfg.get('outflow', True),
                    base=cfg.get('base', True), emb=np.zeros((n, 16)) if n2v else None, n2v=n2v,
                    pressure=pressure_features(g) if cfg.get('pressure', True) else None,
                    etype=g['K'] if cfg.get('edge_topology') else None)
        m.load_state_dict(torch.load(run / f'seed{seed}' / 'model.pt'))
        m.eval()
        models.append((seed, m))
    per_seed, ens = ensemble_metrics(models, T, share, W, H, cfg['tau_final'])
    out = run / 'ensemble.csv'
    frame = pd.DataFrame(per_seed + [ens])
    frame.to_csv(out, index=False, encoding='utf-8-sig')
    cols = ['seed', 'hold_ce', 'hold_ce_trips', 'hold_cpc', 'hold_nrmse', 'hold_internal_err']
    print(frame[cols].round(4).to_string(index=False))
    print(f'saved {out}')


if __name__ == '__main__':
    main()
