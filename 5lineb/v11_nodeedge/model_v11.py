"""Channels split the edges, encoding, aggregation and decoding work on the nodes (model_v11, from model_v9).

1  Encoding of every community from its own nine indicators only (model_v9, DisenGCN variant V1):
       z_i,k = LeakyReLU(W_k^T (omega_k * x_i) + b_k),  zr_i,k = z_i,k / ||z_i,k||
2  Routing splits the outflow edges: each edge u -> v is compared with the channel centres of its origin,
       p_uv,k = softmax_k(zr_v,k . c_u,k / tau),   c_u,k <- l2(zr_u,k + sum_v pi_uv p_uv,k zr_v,k)   (iters times)
   so the channel of an edge depends on both ends (destination vector against the origin's centre).
3  Aggregation for the decoder (out_weight 'share'): the community's own vector weighted by its internal trip share plus
   the destination vectors weighted by trip share times routing share,
       c_out_u,k = l2(pi_uu zr_u,k + sum_v pi_uv p_uv,k zr_v,k)   (a community without trips keeps weight 1);
   with out_weight 'one' the routing centre c_u,k is used.
4  Decoding (run_v9, decoder: joint): one ridge decoder is fitted on the communities' own vectors [zr_1, ..., zr_K]
   and reads the aggregated vectors [c_out_1, ..., c_out_K]; both are mixtures of the same kind of node vectors.
Training as model_v9 at a fixed temperature (each epoch half of the observed non-self outflow edges carry messages,
the other half are predicted; per-channel reconstruction of x from z_k; weight decay), with the stopping rule and
load_from of model_v10.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v9_disengcn'))
from model_v9 import DisenGCNFlow, l2n  # noqa: E402


class NodeDisen(DisenGCNFlow):
    def __init__(self, d=8, iters=5, n_channels=3, out_weight='one'):
        super().__init__(d=d, iters=iters, n_channels=n_channels)
        self.out_weight = out_weight

    def forward(self, x, t, R, cand, share, msg, tau):
        out = super().forward(x, t, R, cand, share, msg, tau)
        out['c_route'] = out['c']
        if self.out_weight == 'share':
            a = torch.diagonal(share).clone()
            a[share.sum(1) <= 0] = 1.0
            w_edge = share * msg.float()
            out['c'] = l2n(a[:, None, None] * out['zr'] + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * out['p'], out['zr']))
        return out


def train_v11(g, seed, n_channels=3, iters=5, epochs=1000, lr=0.02, decay=1e-3, rec_weight=0.1, tau0=0.05, tau_final=0.05,
              n_stages=1, stage_epochs=30, perturb_scale=0.0, stop_window=50, stop_tol=None, self_weight='one',
              out_weight='one', load_from=None, log=None):
    """Same interface as model_v10.train_v10; fixed temperature tau_final, routing with self weight 1."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    T = [torch.as_tensor(g['x'], dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
         torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    rows = torch.as_tensor(g['origins'])
    model = NodeDisen(iters=iters, n_channels=n_channels, out_weight=out_weight)
    tau = tau_final
    src = Path(__file__).resolve().parents[2] / load_from / f'seed{seed}' if load_from else None
    if src is not None and (src / 'model.pt').exists():
        model.load_state_dict(torch.load(src / 'model.pt'))
        model.eval()
        if log:
            log(f'seed {seed} parameters loaded from {load_from}')
        return model, pd.read_csv(src / 'history.csv').to_dict('records'), tau
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        msg_np = obs & (rng.random(W.shape) < 0.5)
        msg = torch.as_tensor(msg_np)
        out = model(*T, share, msg, tau)
        tgt = torch.as_tensor(W * ~msg_np, dtype=torch.float32)
        qd = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)
        ce = -(qd[rows] * out['logpi'][rows]).sum(1).mean()
        rec = sum(fn.mse_loss(model.rec[k](out['z'][:, k]), T[0]) for k in range(model.K)) / model.K
        loss = ce + rec_weight * rec + decay * model.decay()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({'epoch': ep + 1, 'tau': tau, 'ce': float(ce.detach()), 'rec': float(rec.detach())})
        if log and (ep + 1) % 50 == 0:
            log(f'seed {seed} epoch {ep + 1} tau {tau:.3f} ce {float(ce):.4f} rec {float(rec):.4f}')
        if stop_tol is not None and (ep + 1) % stop_window == 0 and ep + 1 >= 2 * stop_window:
            ces = [h_['ce'] for h_ in hist]
            gain = np.mean(ces[-2 * stop_window:-stop_window]) - np.mean(ces[-stop_window:])
            if gain < stop_tol:
                if log:
                    log(f'seed {seed} stopped at epoch {ep + 1}: window gain {gain:.5f}')
                break
    model.eval()
    return model, hist, tau
