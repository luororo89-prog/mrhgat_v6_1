"""DisenGAT following the model figure (线B, figure version).

1  Disentangled channels: channel k projects every community's nine indicators with its own weights,
   z_j,k = f((x_j * omega_k) W_k); neighbours j enter channel k through z_j,k.
2  Routing: each candidate destination j of origin i is softly assigned to the channels by the mixture
   posterior p_ij,k (edge scores against channel centres, topology and impedance terms, origin gates;
   only the competition gate reads the inflow context), as in the original line B model.
3  Graph attention aggregation per channel:
       w_ij,k = flow_ij * p_ij,k (j != i),  a_i,k = sum_j w_ij,k
       alpha_ij,k = w_ij,k exp(e_ij,k) / sum_j w_ij,k exp(e_ij,k)
       e_ij,k = LeakyReLU(q_k . h_i,k + r_k . z_j,k + g_k t_ij + b_k[topology_ij])
       h_i,k = (1 - a_i,k) z_i,k + a_i,k sum_j alpha_ij,k z_j,k
   During training flow_ij is the model's own predicted flow (no observed OD enters the representation);
   for the effective supply the observed flow shares are used.
4  Feedback: routing is recomputed from h and the aggregation repeated (routing_iters rounds).
5  Decoding: the three channel vectors are concatenated and decoded by one MLP into the nine indicators.
   The MLP is trained to reproduce the nominal indicators from the self vectors [z_B; z_C; z_T]; applied to
   the aggregated vectors [h_B; h_C; h_T] it gives the effective supply.
"""
import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn

from model import DisenGAT, margin_loss, split_labels, typed, as_tensors


class DisenGATv2(DisenGAT):
    def __init__(self, d=8, routing_iters=3, mlp_hidden=32, att_shared=False, channel_rec=False, **kwargs):
        """att_shared: one set of attention parameters used by all three channels.
        channel_rec: one linear decoder per channel for the joint reconstruction term (方案 §2.7)."""
        super().__init__(d=d, reconstruct=False, **kwargs)
        self.chan_dec = nn.ModuleList(nn.Linear(d, 9) for _ in range(3)) if channel_rec else None
        self.routing_iters = int(routing_iters)
        self.att_shared = att_shared
        c = 1 if att_shared else 3
        self.att_q = nn.Parameter(torch.zeros(c, d))
        self.att_r = nn.Parameter(torch.zeros(c, d))
        self.att_g = nn.Parameter(torch.zeros(c))
        self.att_b = nn.Parameter(torch.zeros(c, 3))
        nn.init.normal_(self.att_q, std=0.1)
        nn.init.normal_(self.att_r, std=0.1)
        self.mlp = nn.Sequential(nn.Linear(3 * d, mlp_hidden), nn.LeakyReLU(0.2), nn.Linear(mlp_hidden, 9))

    def attention(self, h, z, t, R):
        """e[i, k, j] for every origin i, channel k and destination j."""
        q, r_, g_, b_ = (p.expand(3, *p.shape[1:]) for p in (self.att_q, self.att_r, self.att_g, self.att_b))
        A = (h * q[None]).sum(-1)                              # [n, 3]
        B = (z * r_[None]).sum(-1)                             # [n, 3]
        topo = b_[:, R].permute(1, 0, 2)                       # [n, 3, n]
        e = A[:, :, None] + B.T[None, :, :] + g_[None, :, None] * t[:, None, :] + topo
        return fn.leaky_relu(e, 0.2)

    def aggregate(self, z, h, w, t, R):
        """h_i,k = (1 - a) z_i,k + a sum_j alpha_ij,k z_j,k with attention inside each channel."""
        a = w.sum(2)                                            # [n, 3]
        logits = torch.log(w.clamp_min(1e-30)) + self.attention(h, z, t, R)
        logits = logits.masked_fill(w <= 0, -1e9)
        alpha = torch.softmax(logits, 2) * (a > 0).float()[:, :, None]
        msg = torch.einsum('ikj,jkd->ikd', alpha, z)
        return (1 - a)[:, :, None] * z + a[:, :, None] * msg, alpha

    def forward(self, x, t, R, u, candidates, centre_labels, flow=None):
        """flow: optional observed flow shares [n, n]; without it the predicted flow is used."""
        n = x.shape[0]
        off = 1.0 - torch.eye(n)
        z = self.embed(x)
        h = z
        out = self.route(x, h, t, R, u, candidates, centre_labels)
        for _ in range(self.routing_iters):
            w = out['logpi'].exp()[:, None, :] * out['posterior'] * off[:, None, :]
            h, _ = self.aggregate(z, h, w, t, R)
            out = self.route(x, h, t, R, u, candidates, centre_labels)
        weight = flow if flow is not None else out['logpi'].exp()
        w = weight[:, None, :] * out['posterior'] * off[:, None, :]
        h_final, alpha = self.aggregate(z, h, w, t, R)
        out.update(z=z, h=h_final, w=w, alpha=alpha)
        return out

    def decode(self, v):
        """v: [n, 3, d] channel vectors -> nine standardized indicators."""
        return self.mlp(v.reshape(v.shape[0], -1))

    def decay(self):
        return super().decay() + sum(p.square().sum() for p in self.mlp.parameters() if p.dim() > 1)

    def attention_penalty(self):
        return sum(p.square().sum() for p in (self.att_q, self.att_r, self.att_g, self.att_b))


def train_v2(g, labels, seed, epochs=300, lr=0.02, decay=1e-3, margin=0.2, weight=1.0, rec_weight=0.5,
             log=None, model_kwargs=None, att_l2=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.as_tensor(g['x'], dtype=torch.float32)
    t = torch.as_tensor(g['t'], dtype=torch.float32)
    R = torch.as_tensor(g['K'], dtype=torch.long)
    u = torch.as_tensor(g['u'], dtype=torch.float32)
    cand = torch.as_tensor(g['candidates'])
    W = torch.as_tensor(g['W'], dtype=torch.float32)
    rows = torch.as_tensor(g['origins'])
    share = W / W.sum(1, keepdim=True).clamp_min(1e-12)
    model = DisenGATv2(**(model_kwargs or {}))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    halves = [typed(h) for h in split_labels(labels, seed)]
    history = []
    for epoch in range(epochs):
        centre, lossset = halves[epoch % 2], halves[1 - epoch % 2]
        out = model(x, t, R, u, cand, centre)
        ce = -(share[rows] * out['logpi'][rows]).sum(1).mean()
        la = margin_loss(out['posterior'], lossset, margin)
        if model.chan_dec is not None:
            rec = sum(fn.mse_loss(model.chan_dec[k](out['z'][:, k]), x) for k in range(3)) / 3
        else:
            rec = fn.mse_loss(model.decode(out['z']), x)
        loss = ce + weight * la + rec_weight * rec + decay * model.decay() + att_l2 * model.attention_penalty()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append({'epoch': epoch + 1, 'ce': float(ce), 'anchor': float(la), 'rec': float(rec), 'loss': float(loss)})
        if log and (epoch + 1) % 50 == 0:
            log(f'seed {seed} epoch {epoch + 1} ce {float(ce):.4f} anchor {float(la):.4f} rec {float(rec):.4f}')
    model.eval()
    return model, history


def effective(model, g, labels, share_obs, mask_topology=None):
    """Nominal reproduction, effective supply and channel attributions with the observed flow shares.
    Returns x_hat = MLP([z]), x_eff_dec = MLP([h]), Shapley contributions per channel."""
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    x, t, R, u = T(g['x']), T(g['t']), T(g['K'], torch.long), T(g['u'])
    cand = torch.as_tensor(g['candidates'])
    flow = T(share_obs if mask_topology is None else share_obs * mask_topology)
    with torch.no_grad():
        out = model(x, t, R, u, cand, labels, flow=flow)
        z, h = out['z'], out['h']
        base = model.decode(z)
        f = {}
        for S in itertools.chain.from_iterable(itertools.combinations(range(3), r) for r in range(4)):
            v = z.clone()
            for k in S:
                v[:, k] = h[:, k]
            f[S] = (model.decode(v) - base).numpy()
        shap = []
        for k in range(3):
            others = [c for c in range(3) if c != k]
            val = 0
            for r in range(3):
                for S in itertools.combinations(others, r):
                    wgt = math.factorial(len(S)) * math.factorial(3 - len(S) - 1) / math.factorial(3)
                    val = val + wgt * (f[tuple(sorted(S + (k,)))] - f[S])
            shap.append(val)
    return {'x_hat': base.numpy(), 'delta': f[(0, 1, 2)], 'shapley': np.stack(shap, 1), 'posterior': out['posterior'].numpy(),
            'logpi': out['logpi'].numpy(), 'z': z.numpy(), 'h': h.numpy(), 'alpha': out['alpha'].numpy()}
