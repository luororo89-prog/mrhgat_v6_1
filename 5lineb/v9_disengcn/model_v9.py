"""DisenGCN routing on the outflow graph, aggregation straight to the decoders (model_v9, from model_v5 variant V1).

1  Channels (n_channels of them): z_i,k = LeakyReLU(W_k^T (omega_k * x_i) + b_k), omega_k = 9 softmax(g_k); zr_i,k = z_i,k / ||z_i,k||.
2  Routing (DisenGCN, Algorithm 1): c_u,k <- zr_u,k; iters times
       p_uv,k = softmax_k(zr_v,k . c_u,k / tau),   c_u,k <- l2(zr_u,k + sum_v pi_uv p_uv,k zr_v,k)
   pi_uv = W_uv / sum_v' W_uv' on the outflow edges that carry messages (the self loop counts in the denominator).
3  The aggregated channel vectors c_u,k are the output of the layer; no attention layer follows the routing.
4  Training: each epoch half of the observed non-self outflow edges carry messages, the other half are predicted from
       score(u, v) = sum_k zr_v,k . c_u,k / tau_s + x_v . w_m - beta t_uv + B0[topology_uv]
   (softmax over candidate destinations, message edges excluded), plus a per-channel linear reconstruction of x from
   z_k and weight decay, as in model_v5.
5  Temperature annealing: tau starts at tau0, where every neighbour is shared almost equally by the three channels, and
   is lowered geometrically over n_stages stages of stage_epochs epochs to tau_final, which is kept for the remaining
   epochs. At the start of every later stage the parameters receive a small random perturbation, so the channels
   separate where the data make the shared solution unstable and not where the random start puts them. With
   n_stages = 1 and tau0 = tau_final the temperature is fixed (no annealing, no perturbation).
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn

l2n = lambda v: fn.normalize(v, dim=-1)


class DisenGCNFlow(nn.Module):
    def __init__(self, d=8, iters=5, n_channels=3):
        super().__init__()
        self.d, self.iters, self.K = d, int(iters), int(n_channels)
        self.g = nn.Parameter(torch.zeros(self.K, 9))
        self.W = nn.Parameter(torch.empty(self.K, 9, d))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(self.K, d))
        self.log_tau_s = nn.Parameter(torch.zeros(()))
        self.w_m = nn.Parameter(torch.zeros(9))
        self.beta = nn.Parameter(torch.zeros(()))
        self.B0 = nn.Parameter(torch.zeros(3))
        self.rec = nn.ModuleList(nn.Linear(d, 9) for _ in range(self.K))

    def weights(self):
        return 9 * torch.softmax(self.g, 1)

    def embed(self, x):
        om = self.weights()
        return fn.leaky_relu(torch.stack([(x * om[k]) @ self.W[k] + self.b[k] for k in range(self.K)], 1), 0.2)

    @staticmethod
    def sim(zv, c):
        """zv [v, K, d] destination vectors, c [u, K, d] origin centres -> [u, K, v] dot products."""
        return torch.einsum('ukd,vkd->ukv', c, zv)

    def forward(self, x, t, R, cand, share, msg, tau):
        z = self.embed(x)
        zr = l2n(z)
        w_edge = share * msg.float()
        c = zr
        for _ in range(self.iters):
            p = torch.softmax(self.sim(zr, c) / tau, 1)
            c = l2n(zr + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * p, zr))
        p = torch.softmax(self.sim(zr, c) / tau, 1)
        score = self.sim(zr, c).sum(1) / torch.exp(self.log_tau_s) + (x @ self.w_m)[None, :] \
            - fn.softplus(self.beta) * t + self.B0[R]
        logpi = torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1)
        return {'z': z, 'zr': zr, 'c': c, 'p': p, 'logpi': logpi}

    def decay(self):
        return self.W.square().sum() + self.w_m.square().sum()


def schedule(tau0, tau_final, n_stages):
    return [float(v) for v in np.geomspace(tau0, tau_final, n_stages)]


def train_v9(g, seed, n_channels=3, iters=5, epochs=300, lr=0.02, decay=1e-3, rec_weight=0.1, tau0=1.0, tau_final=0.05,
             n_stages=6, stage_epochs=30, perturb_scale=0.01, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    T = [torch.as_tensor(g['x'], dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
         torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    rows = torch.as_tensor(g['origins'])
    model = DisenGCNFlow(iters=iters, n_channels=n_channels)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    temps = schedule(tau0, tau_final, n_stages)
    hist = []
    for ep in range(epochs):
        stage = min(ep // stage_epochs, n_stages - 1)
        tau = temps[stage]
        if ep and ep % stage_epochs == 0 and ep // stage_epochs < n_stages:
            with torch.no_grad():
                for prm in model.parameters():
                    size = prm.std(unbiased=False) if prm.numel() > 1 else prm.abs().reshape(())
                    prm.add_(perturb_scale * size.clamp_min(1e-3) * torch.randn(prm.shape, generator=gen))
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
        hist.append({'epoch': ep + 1, 'tau': tau, 'ce': float(ce), 'rec': float(rec)})
        if log and (ep + 1) % 50 == 0:
            log(f'seed {seed} epoch {ep + 1} tau {tau:.3f} ce {float(ce):.4f} rec {float(rec):.4f}')
    model.eval()
    return model, hist, temps[-1]
