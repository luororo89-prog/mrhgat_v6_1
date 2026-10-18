"""Gravity-type destination choice benchmark on the node representations of the mechanism model (M4, diagnostic).

python 5lineb/v13_geoflow/benchmark_gravity_mechanism_v13.py --config \
    5lineb/v13_geoflow/configs/outpatient_d0.25_od_mechanism_v13.yaml [--variant mech_full]

Same fixed test and validation splits and the same candidate destinations as the main model (non-message non-self
destinations; training targets are the flows of the available training edges, validation and test of the held-out
edges). Score of every model, softmax over the candidate destinations,
       z_ij = A(u_j) - softplus(beta) d_ij + B0[r_ij],   A(u_j) = w . u_j,
  gravity_distance  no attraction term (distance and topology only);
  gravity_supply    u_j = x_j, the nine standardised supply indicators;
  gravity_repr      u_j = h_j, the three-channel node representation [base || comp || compete] of the trained model of
                    the same seed (encoder frozen; training and validation read the representation of the available
                    training context, the test the representation of the test context).
Parameters are fitted on the training edges only (Adam, lr and weight decay of the config), early stopping on the
validation CE. Outputs: results_od_mechanism/<name>/gravity_benchmark.csv and gravity_benchmark_summary.csv.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_od_relation_v13 as reference
from run_od_ablation_v13 import setup
from run_od_mechanism_v13 import predict
from model_od_mechanism_v13 import make_mechanism_model


class Gravity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False) if dim else None
        self.beta = nn.Parameter(torch.zeros(()))
        self.B0 = nn.Parameter(torch.zeros(3))

    def forward(self, u, t, R):
        score = -F.softplus(self.beta) * t + self.B0[R]
        return score if self.w is None else score + self.w(u).squeeze(1)[None, :]


def fit(kind, u_train, u_test, cfg, g, tensors, H, Hv, seed):
    torch.manual_seed(seed)
    W = g['W']
    off = ~np.eye(len(W), dtype=bool)
    edge = (W > 0) & off
    visible = edge & ~H & ~Hv
    t, R = tensors[1], tensors[2]
    model = Gravity(0 if u_train is None else u_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    target = torch.as_tensor(W * visible, dtype=torch.float32)
    cand = torch.as_tensor(off & ~H & ~Hv)
    best, best_ce, since = None, np.inf, 0
    for _ in range(cfg['epochs']):
        model.train()
        loss = reference.destination_ce(model(u_train, t, R), target, cand)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = reference.evaluate_scores(model(u_train, t, R), W, Hv, visible, excluded=H)
        if val['ce'] < best_ce:
            best_ce, since = val['ce'], 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= cfg['early']:
            break
    model.load_state_dict(best)
    with torch.no_grad():
        test = reference.evaluate_scores(model(u_test, t, R), W, H, edge & ~H)
    return dict(model=kind, seed=seed, ce=test['ce'], cpc=test['cpc'], nrmse=test['nrmse'], best_val_ce=best_ce)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', default='mech_full')
    args = parser.parse_args()
    cfg, g, tensors, H, Hv = setup(args.config)
    torch.set_num_threads(cfg['threads'])
    output = ROOT / 'results_od_mechanism' / cfg['name']
    W = g['W']
    n = len(W)
    edge = (W > 0) & ~np.eye(n, dtype=bool)
    rows = []
    for seed in cfg['seeds']:
        directory = output / args.variant / f'seed{seed}'
        model = make_mechanism_model(args.variant, cfg, n, tensors[2])
        model.load_state_dict(torch.load(directory / 'model.pt', weights_only=True))
        h_train = predict(model, g, tensors, W * (edge & ~H & ~Hv), cfg['tau_final'])['c']
        h_test = predict(model, g, tensors, W * (edge & ~H), cfg['tau_final'])['c']
        x = tensors[0]
        for kind, ut, us in (('gravity_distance', None, None), ('gravity_supply', x, x),
                             ('gravity_repr', h_train, h_test)):
            rows.append(fit(kind, ut, us, cfg, g, tensors, H, Hv, seed))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(output / 'gravity_benchmark.csv', index=False)
    S = R.groupby('model')[['ce', 'cpc', 'nrmse']].agg(['mean', 'std'])
    S.to_csv(output / 'gravity_benchmark_summary.csv')
    print(S.round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()
