"""Graph model benchmark for the outflow representation of model_v13 (reviewer comments 4 and 16: GCN, GraphSAGE,
Node2Vec and flow-aware / OD-specific architectures against the attention model).

Same task, inputs and evaluation for every model: destination prediction on the fixed held-out outflow edges of the
config (model_v13.holdout_mask, evaluate), each epoch half of the training edges carry messages and the rest and the
self loop are targets, the same candidate destinations, stopping rule, optimiser, learning rate and weight decay.
Node inputs: origin side o_A = [x_A, f_A] (nine standardised indicators and the inflow pressure of
model_v13.pressure_features), destination side x_B. Score of every model,
       s_AB = h_A . z_B / tau_s + x_B . w_m - beta t_AB + B0[topology_AB],   z_B = W_d x_B (d = 16),
softmax over the candidate destinations; models differ in the origin representation h_A (one propagation step on the
message edges, d = 16):
  gravity   no h (origin-constrained gravity/Huff type: destination attraction, distance decay, topology);
  gravity_fe  gravity with a free attraction constant c_B for every destination (destination fixed effects, n
            parameters): a control for what the node identity alone predicts;
  mlp       h_A = LeakyReLU(W o_A) (no graph; DeepGravity type);
  gcn       h = LeakyReLU(A^ O W), A^ = D^-1/2 (A + I) D^-1/2, A = message flow shares made symmetric
            (Kipf & Welling 2017);
  sage      h_A = LeakyReLU(W_1 o_A + W_2 mean_{B in N(A)} o_B), N(A) the message destinations of A, mean aggregator
            (Hamilton et al. 2017);
  gat       h_A = LeakyReLU(sum_{B in N(A) + A} alpha_AB W o_B), alpha = softmax(LeakyReLU(a . [W o_A, W o_B])), two
            heads of 8 (Velickovic et al. 2018);
  node2vec  h_A = LeakyReLU(W [e_A, o_A]), z_B = W_d [e_B, x_B], e the node2vec embedding (d = 16, walks of length
            40, 10 per node, window 5, p = 1, q = 0.5, skip-gram with 5 negative samples, 5 epochs) of the undirected
            graph of the training edges weighted by log1p of the flows (Grover & Leskovec 2016), fixed per seed.
The flow-aware OD-specific models are model_v13 itself (Base with the internal trip share, inflow pressure in the
routing, two outflow channels with attention) and its one-channel ablation, read from results_lineb/v13_ablation.
Outputs in results_lineb/v13_benchmark: per_seed.csv, summary.csv (with the number of trainable parameters).

python 5lineb/v13_geoflow/benchmark_v13.py --config 5lineb/v13_geoflow/configs/walk_v1_net2_v13_k2_s10_hold.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as fn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(ROOT / 'src'))
from model_v13 import evaluate, holdout_mask, pressure_features  # noqa: E402
from node2vec_v13 import node2vec  # noqa: E402
from graph import build_graph  # noqa: E402
from ruler import standardize  # noqa: E402
from run_v2 import frozen  # noqa: E402

MODELS = ('gravity', 'gravity_fe', 'mlp', 'gcn', 'sage', 'gat', 'node2vec')
OUT = ROOT / 'results_lineb' / 'v13_benchmark'


class Baseline(nn.Module):
    def __init__(self, kind, x_dim, o_dim, emb=None, d=16, heads=2, n=None):
        super().__init__()
        self.kind, self.d, self.heads = kind, d, heads
        if kind == 'gravity_fe':
            self.c = nn.Parameter(torch.zeros(n))
        self.log_tau_s = nn.Parameter(torch.zeros(()))
        self.w_m = nn.Parameter(torch.zeros(x_dim))
        self.beta = nn.Parameter(torch.zeros(()))
        self.B0 = nn.Parameter(torch.zeros(3))
        self.emb = None if emb is None else torch.as_tensor(emb, dtype=torch.float32)
        e_dim = 0 if emb is None else emb.shape[1]
        if kind not in ('gravity', 'gravity_fe'):
            self.Wd = nn.Linear(x_dim + e_dim, d)
        if kind in ('mlp', 'gcn'):
            self.W = nn.Linear(o_dim, d)
        elif kind == 'node2vec':
            self.W = nn.Linear(o_dim + e_dim, d)
        elif kind == 'sage':
            self.W1, self.W2 = nn.Linear(o_dim, d), nn.Linear(o_dim, d, bias=False)
        elif kind == 'gat':
            self.W = nn.Linear(o_dim, d, bias=False)
            self.a = nn.Parameter(torch.randn(heads, 2 * d // heads) * 0.1)

    def decay(self):
        return sum(m.weight.square().sum() for m in self.modules() if isinstance(m, nn.Linear)) + self.w_m.square().sum()

    def represent(self, o, msg_w):
        n = len(o)
        if self.kind == 'mlp':
            return fn.leaky_relu(self.W(o), 0.2)
        if self.kind == 'node2vec':
            return fn.leaky_relu(self.W(torch.cat([self.emb, o], 1)), 0.2)
        if self.kind == 'gcn':
            A = msg_w + msg_w.T + torch.eye(n)
            dinv = A.sum(1).rsqrt()
            return fn.leaky_relu((dinv[:, None] * A * dinv[None, :]) @ self.W(o), 0.2)
        if self.kind == 'sage':
            M = (msg_w > 0).float()
            mean = (M @ o) / M.sum(1, keepdim=True).clamp_min(1.0)
            return fn.leaky_relu(self.W1(o) + self.W2(mean), 0.2)
        if self.kind == 'gat':
            h = self.W(o).view(n, self.heads, -1)
            src = (h * self.a[None, :, :h.shape[2]]).sum(2)
            dst = (h * self.a[None, :, h.shape[2]:]).sum(2)
            e = fn.leaky_relu(src.T[:, :, None] + dst.T[:, None, :], 0.2)
            mask = (msg_w > 0) | torch.eye(n, dtype=torch.bool)
            alpha = torch.softmax(e.masked_fill(~mask[None], -1e9), 2)
            return fn.leaky_relu(torch.einsum('huv,vhd->uhd', alpha, h).reshape(n, -1), 0.2)

    def forward(self, x, t, R, cand, share, msg, tau, o=None):
        o = self.o if o is None else o
        score = (x @ self.w_m)[None, :] - fn.softplus(self.beta) * t + self.B0[R]
        if self.kind == 'gravity_fe':
            score = score + self.c[None, :]
        if self.kind not in ('gravity', 'gravity_fe'):
            msg_w = share * msg.float()
            h = self.represent(o, msg_w)
            z = self.Wd(x if self.emb is None else torch.cat([self.emb, x], 1))
            score = score + h @ z.T / torch.exp(self.log_tau_s)
        return {'logpi': torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1)}


def train(kind, g, seed, cfg, H):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    x = torch.as_tensor(g['x'], dtype=torch.float32)
    o = torch.cat([x, torch.as_tensor(pressure_features(g), dtype=torch.float32)], 1)
    T = [x, torch.as_tensor(g['t'], dtype=torch.float32), torch.as_tensor(g['K'], dtype=torch.long),
         torch.as_tensor(g['candidates'])]
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    train_obs = obs & ~H
    emb = None
    if kind == 'node2vec':
        Wt = np.log1p(W * train_obs)
        emb = node2vec(Wt + Wt.T, seed)
    model = Baseline(kind, x.shape[1], o.shape[1], emb, n=n)
    model.o = o
    rows = torch.as_tensor(g['origins'])
    cand_train = T[3] & ~torch.as_tensor(H)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    hist = []
    for ep in range(cfg['epochs']):
        msg_np = train_obs & (rng.random(W.shape) < 0.5)
        out = model(T[0], T[1], T[2], cand_train, share, torch.as_tensor(msg_np), cfg['tau_final'])
        tgt = torch.as_tensor(W * ~msg_np * ~H, dtype=torch.float32)
        qd = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)
        ce = -(qd[rows] * out['logpi'][rows]).sum(1).mean()
        loss = ce + cfg['decay'] * model.decay()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(ce.detach()))
        sw = cfg['stop_window']
        if (ep + 1) % sw == 0 and ep + 1 >= 2 * sw and np.mean(hist[-2 * sw:-sw]) - np.mean(hist[-sw:]) < cfg['stop_tol']:
            break
    model.eval()
    met = evaluate(model, T, share, W, H, cfg['tau_final'])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'epochs': ep + 1, 'train_ce': float(np.mean(hist[-50:])), 'params': params, **met}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    torch.set_num_threads(cfg.get('threads', 8))
    g = build_graph(cfg)
    g['x'] = standardize(g['raw'], frozen(ROOT / cfg['ruler_from'])['main'])
    H = holdout_mask(g['W'], cfg['holdout_share'], cfg['holdout_seed'])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in a.models:
        for seed in cfg['seeds']:
            r = {'model': kind, 'seed': seed, **train(kind, g, seed, cfg, H)}
            rows.append(r)
            print(r, flush=True)
    per = pd.DataFrame(rows)
    old = OUT / 'per_seed.csv'
    if old.exists():
        prev = pd.read_csv(old)
        per = pd.concat([prev[~prev.model.isin(a.models)], per], ignore_index=True)
    per.to_csv(old, index=False)
    abl = pd.read_csv(ROOT / 'results_lineb' / 'v13_ablation' / 'retrained_per_seed.csv')
    own = abl[abl.config.isin(['walk_v1_net2_v13_k2_s10_hold', 'walk_v1_net2_v13_k1_s10_hold'])].copy()
    own['model'] = own.config.map({'walk_v1_net2_v13_k2_s10_hold': 'v13 (2 channels)', 'walk_v1_net2_v13_k1_s10_hold': 'v13 (1 channel)'})
    from model_v13 import MechNet
    for K, name in ((2, 'v13 (2 channels)'), (1, 'v13 (1 channel)')):
        m = MechNet(iters=cfg['iters'], n_channels=K, out_weight=cfg['out_weight'], pressure=pressure_features(g))
        own.loc[own.model == name, 'params'] = sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)
    allr = pd.concat([per, own.drop(columns='config')], ignore_index=True)
    cols = ['hold_ce', 'hold_ce_trips', 'hold_cpc', 'hold_nrmse', 'hold_internal_err']
    summ = allr.groupby('model')[cols + ['params']].mean()
    for c in ('hold_ce', 'hold_cpc'):
        summ[f'{c}_sd'] = allr.groupby('model')[c].std()
    summ = summ.sort_values('hold_ce')
    summ.to_csv(OUT / 'summary.csv')
    pd.set_option('display.width', 250)
    print(summ.round(4).to_string())


if __name__ == '__main__':
    main()
