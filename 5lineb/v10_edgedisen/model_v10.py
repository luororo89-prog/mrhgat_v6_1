"""Disentangled routing over edge properties (model_v10): DisenGCN with edge vectors in place of node vectors.

DisenGCN splits a neighbour by what the neighbour node is like; here an outflow u -> v is split by what the edge is
like, that is, by the tiered vectors of both ends.
1  Edge vectors per channel k: h_uv,k = l2(LeakyReLU(W_k^T (omega_k * [x_u ; x_v]) + b_k)), omega_k = 18 softmax(g_k)
   over the nine indicators of the origin and the nine of the destination. Each edge is normalised to unit length in
   every channel. The self loop u -> u (both ends u) gives the vector of community u itself, s_u,k = h_uu,k.
2  Routing (DisenGCN, Algorithm 1, with edges as the routed items): c_u,k <- s_u,k; iters times
       p_uv,k = softmax_k(h_uv,k . c_u,k / tau),   c_u,k <- l2(s_u,k + sum_v pi_uv p_uv,k h_uv,k)
   pi_uv = W_uv / sum_v' W_uv' on the outflow edges that carry messages (the self loop counts in the denominator).
   self_weight: 'one' adds s_u,k with weight 1 as in DisenGCN; 'share' weights it by the internal trip share pi_uu, so
   that the centre mixes the community and its destinations by trips (a community without any trip keeps weight 1).
   out_weight ('one' or 'share', default: as self_weight): self weight of the aggregated vector handed to the decoder,
   c_out_u,k = l2(a_u s_u,k + sum_v pi_uv p_uv,k h_uv,k) with the final routing shares; the routing and the scores keep
   self_weight. With self_weight 'one' and out_weight 'share' the edges are routed against the community itself and the
   decoded vector mixes the community and its destinations by trips.
   load_from (train_v10): load the trained parameters of an earlier run instead of training; seeds without a saved
   model in that run are trained.
3  Training as model_v9: each epoch half of the observed non-self outflow edges carry messages, the other half are
   predicted from score(u, v) = sum_k h_uv,k . c_u,k / tau_s + x_v . w_m - beta t_uv + B0[topology_uv] (softmax over
   candidate destinations, message edges excluded), plus a per-channel linear reconstruction of x_u from the self loop
   vector before normalisation, and weight decay. The temperature is fixed. With stop_tol set, training stops at the
   end of a window of stop_window epochs when the mean training cross-entropy of the window is less than stop_tol below
   that of the previous window (epochs is then the upper limit).
Output keys follow model_v9: zr = self loop vectors s, c = aggregated channel vectors, p = routing shares [u, K, v].
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as fn

l2n = lambda v: fn.normalize(v, dim=-1)


class EdgeDisen(nn.Module):
    def __init__(self, d=8, iters=5, n_channels=3, self_weight='one', out_weight=None):
        super().__init__()
        self.d, self.iters, self.K, self.self_weight = d, int(iters), int(n_channels), self_weight
        self.out_weight = out_weight or self_weight
        self.g = nn.Parameter(torch.zeros(self.K, 18))
        self.W = nn.Parameter(torch.empty(self.K, 18, d))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(self.K, d))
        self.log_tau_s = nn.Parameter(torch.zeros(()))
        self.w_m = nn.Parameter(torch.zeros(9))
        self.beta = nn.Parameter(torch.zeros(()))
        self.B0 = nn.Parameter(torch.zeros(3))
        self.rec = nn.ModuleList(nn.Linear(d, 9) for _ in range(self.K))

    @staticmethod
    def self_weights(share, mode):
        if mode != 'share':
            return torch.ones(share.shape[0])
        a = torch.diagonal(share).clone()
        a[share.sum(1) <= 0] = 1.0
        return a

    def weights(self):
        return 18 * torch.softmax(self.g, 1)

    def edges(self, x):
        """Edge vectors before and after normalisation, [u, v, K, d]."""
        om = self.weights()
        a = torch.einsum('nj,kjd->nkd', x, om[:, :9, None] * self.W[:, :9])
        q = torch.einsum('nj,kjd->nkd', x, om[:, 9:, None] * self.W[:, 9:])
        z = fn.leaky_relu(a[:, None] + q[None, :] + self.b, 0.2)
        return z, l2n(z)

    def forward(self, x, t, R, cand, share, msg, tau):
        n = x.shape[0]
        z, h = self.edges(x)
        idx = torch.arange(n)
        s = h[idx, idx]
        w_edge = share * msg.float()
        a_self = self.self_weights(share, self.self_weight)
        c = s
        for _ in range(self.iters):
            p = torch.softmax(torch.einsum('uvkd,ukd->ukv', h, c) / tau, 1)
            c = l2n(a_self[:, None, None] * s + torch.einsum('ukv,uvkd->ukd', w_edge[:, None, :] * p, h))
        sim = torch.einsum('uvkd,ukd->ukv', h, c)
        p = torch.softmax(sim / tau, 1)
        c_out = c if self.out_weight == self.self_weight else l2n(
            self.self_weights(share, self.out_weight)[:, None, None] * s + torch.einsum('ukv,uvkd->ukd', w_edge[:, None, :] * p, h))
        score = sim.sum(1) / torch.exp(self.log_tau_s) + (x @ self.w_m)[None, :] - fn.softplus(self.beta) * t + self.B0[R]
        logpi = torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1)
        return {'z': z[idx, idx], 'zr': s, 'c': c_out, 'c_route': c, 'p': p, 'logpi': logpi}

    def decay(self):
        return self.W.square().sum() + self.w_m.square().sum()


def train_v10(g, seed, n_channels=3, iters=5, epochs=300, lr=0.02, decay=1e-3, rec_weight=0.1, tau0=0.05, tau_final=0.05,
              n_stages=1, stage_epochs=30, perturb_scale=0.0, stop_window=50, stop_tol=None, self_weight='one',
              out_weight=None, load_from=None, log=None):
    """Same interface as model_v9.train_v9; the temperature is fixed at tau_final (no annealing)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    T = [torch.as_tensor(g['x'], dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
         torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    rows = torch.as_tensor(g['origins'])
    model = EdgeDisen(iters=iters, n_channels=n_channels, self_weight=self_weight, out_weight=out_weight)
    src = Path(__file__).resolve().parents[2] / load_from / f'seed{seed}' if load_from else None
    if src is not None and (src / 'model.pt').exists():
        model.load_state_dict(torch.load(src / 'model.pt'))
        model.eval()
        if log:
            log(f'seed {seed} parameters loaded from {load_from}')
        return model, pd.read_csv(src / 'history.csv').to_dict('records'), tau_final
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tau = tau_final
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
