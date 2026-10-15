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
load_from of model_v10. With indep_weight > 0 an independence penalty between the channels is added (IPGDN, Liu et al.
2020): normalised HSIC with Gaussian kernels (bandwidth the median pairwise distance) between the unit vectors zr_k and
zr_j of all communities, averaged over the channel pairs,
       loss = ce + rec_weight rec + decay ||W||^2 + indep_weight mean_{k<j} nHSIC(zr_k, zr_j).
With inflow: own the routing reads the inflow of the origin u: its own vector in the routing (the start and the self
term of its centres) carries its inflow context u_u (graph.u: log1p of the adjacent and of the cross inflow from other
communities per trip of its residents, standardised),
       zo_u,k = l2(z_u,k + V_k^T u_u),   c_u,k <- l2(zo_u,k + sum_v pi_uv p_uv,k zr_v,k),
while the destination vectors zr_v, the aggregation for the decoder and the decoder keep the supply vectors only.
V_k starts at zero (the model without inflow) and is included in the weight decay. With inflow: topology the origin
reads the topology of its inflow instead, the self loop as a topology type of its own: f_u = log1p of the self-loop
trips, of the inflow from adjacent and of the inflow from cross communities, each per trip of its residents,
standardised (topology_features).
With edge_topology the routing reads the topology of every outflow edge (adjacent or cross; the self loop carries no
message): a channel bias by edge type is added to the routing logits,
       p_uv,k = softmax_k(zr_v,k . c_u,k / tau + b_k,type(u,v)),
with b starting at zero and included in the weight decay.
With labels_file the routing is supervised on labelled edges (dominant anchors, graph.load_labels): each mechanism has
a fixed channel (label_channels, default compet C1, comp C2, base C3), and the flow-weighted ranking margin of
v1_disengat (margin_loss) on the routing shares of the labelled edges is added with label_weight,
       label = sum_e f_e sum_{k != y_e} relu(m - (p_e,y_e - p_e,k)) / sum_e f_e,   m = label_margin.
Labels of the outer fold label_holdout are left out of training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v9_disengcn'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from model_v9 import DisenGCNFlow, l2n  # noqa: E402
from graph import ADJ, CRS  # noqa: E402


def topology_features(g):
    """[n, 3] log1p of self-loop trips, adjacent inflow and cross inflow per trip of the residents, standardised."""
    W, K = g['W'], g['K']
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    O = W.sum(1)
    f = np.column_stack([np.diag(W), (W * off * (K == ADJ)).sum(0), (W * off * (K == CRS)).sum(0)])
    f = np.log1p(np.divide(f, O[:, None], out=np.zeros_like(f), where=O[:, None] > 0))
    return (f - f.mean(0)) / f.std(0)


def node_extras(g, inflow=None, edge_topology=False):
    """Keyword arguments of NodeDisen for the inflow and edge topology options of a config."""
    feats = {'own': g['u'], 'topology': topology_features(g) if inflow == 'topology' else None}.get(inflow) if inflow else None
    return {'inflow': feats, 'etype': g['K'] if edge_topology else None}


class NodeDisen(DisenGCNFlow):
    def __init__(self, d=8, iters=5, n_channels=3, out_weight='one', inflow=None, etype=None):
        super().__init__(d=d, iters=iters, n_channels=n_channels)
        self.out_weight = out_weight
        if inflow is None:
            self.inflow = None
        else:
            self.register_buffer('inflow', torch.as_tensor(np.asarray(inflow), dtype=torch.float32))
            self.V = torch.nn.Parameter(torch.zeros(self.K, self.inflow.shape[1], d))
        if etype is None:
            self.etype = None
        else:
            self.register_buffer('etype', torch.as_tensor(np.asarray(etype), dtype=torch.long))
            self.Bt = torch.nn.Parameter(torch.zeros(self.K, 3))

    def decay(self):
        return (super().decay() + (self.V.square().sum() if self.inflow is not None else 0.0)
                + (self.Bt.square().sum() if self.etype is not None else 0.0))

    def forward(self, x, t, R, cand, share, msg, tau):
        if self.inflow is None and self.etype is None:
            out = super().forward(x, t, R, cand, share, msg, tau)
        else:
            z = self.embed(x)
            zr = l2n(z)
            zo = zr if self.inflow is None else l2n(z + torch.einsum('nf,kfd->nkd', self.inflow, self.V))
            bias = 0.0 if self.etype is None else self.Bt[:, self.etype].permute(1, 0, 2)
            w_edge = share * msg.float()
            c = zo
            for _ in range(self.iters):
                p = torch.softmax(self.sim(zr, c) / tau + bias, 1)
                c = l2n(zo + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * p, zr))
            p = torch.softmax(self.sim(zr, c) / tau + bias, 1)
            score = self.sim(zr, c).sum(1) / torch.exp(self.log_tau_s) + (x @ self.w_m)[None, :] \
                - fn.softplus(self.beta) * t + self.B0[R]
            logpi = torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1)
            out = {'z': z, 'zr': zr, 'c': c, 'p': p, 'logpi': logpi}
        out['c_route'] = out['c']
        if self.out_weight == 'share':
            a = torch.diagonal(share).clone()
            a[share.sum(1) <= 0] = 1.0
            w_edge = share * msg.float()
            out['c'] = l2n(a[:, None, None] * out['zr'] + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * out['p'], out['zr']))
        return out


def nhsic(a, b):
    """Normalised HSIC between two sets of vectors [n, d] with Gaussian kernels of median bandwidth."""
    def centred(z):
        d2 = torch.cdist(z, z).square()
        k = torch.exp(-d2 / (2 * d2.detach()[d2.detach() > 0].median()))
        return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()
    ka, kb = centred(a), centred(b)
    return (ka * kb).sum() / torch.sqrt((ka * ka).sum() * (kb * kb).sum()).clamp_min(1e-12)


def independence(zr):
    K = zr.shape[1]
    return torch.stack([nhsic(zr[:, k], zr[:, j]) for k in range(K) for j in range(k + 1, K)]).mean()


def margin_loss(p, lab, m):
    """Flow-weighted ranking margin of the routing shares p [u, K, v] on labelled edges (v1_disengat margin_loss)."""
    pe = p[lab['origin'], :, lab['dest']]
    py = pe.gather(1, lab['channel'][:, None])
    gap = torch.relu(m - (py - pe)).scatter(1, lab['channel'][:, None], 0.0).sum(1)
    return (lab['flow'] * gap).sum() / lab['flow'].sum()


def labelled(g, labels_file, label_channels=None, label_holdout=None):
    """Labelled edges with the channel of their mechanism; train and held-out parts (outer fold label_holdout)."""
    from graph import load_labels, MECHANISMS
    lab = load_labels(labels_file, g)
    slots = label_channels or {'compet': 'C1', 'comp': 'C2', 'base': 'C3'}
    lab['channel'] = np.array([int(slots[MECHANISMS[k]][1:]) - 1 for k in lab['mech']])
    hold = (lab['fold'] == label_holdout) if label_holdout is not None else np.zeros(len(lab['mech']), dtype=bool)
    pick = lambda m: {k: torch.as_tensor(lab[k][m]) for k in ('origin', 'dest', 'channel', 'flow')}
    return pick(~hold), pick(hold)


def train_v11(g, seed, n_channels=3, iters=5, epochs=1000, lr=0.02, decay=1e-3, rec_weight=0.1, tau0=0.05, tau_final=0.05,
              n_stages=1, stage_epochs=30, perturb_scale=0.0, stop_window=50, stop_tol=None, self_weight='one',
              out_weight='one', load_from=None, indep_weight=0.0, inflow=None, edge_topology=False, labels_file=None, label_weight=1.0,
              label_margin=0.2, label_channels=None, label_holdout=None, log=None):
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
    model = NodeDisen(iters=iters, n_channels=n_channels, out_weight=out_weight, **node_extras(g, inflow, edge_topology))
    tau = tau_final
    src = Path(__file__).resolve().parents[2] / load_from / f'seed{seed}' if load_from else None
    if src is not None and (src / 'model.pt').exists():
        model.load_state_dict(torch.load(src / 'model.pt'))
        model.eval()
        if log:
            log(f'seed {seed} parameters loaded from {load_from}')
        return model, pd.read_csv(src / 'history.csv').to_dict('records'), tau
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lab = labelled(g, labels_file, label_channels, label_holdout)[0] if labels_file else None
    if lab is not None:
        lab['flow'] = lab['flow'].float()
    hist = []
    for ep in range(epochs):
        msg_np = obs & (rng.random(W.shape) < 0.5)
        msg = torch.as_tensor(msg_np)
        out = model(*T, share, msg, tau)
        tgt = torch.as_tensor(W * ~msg_np, dtype=torch.float32)
        qd = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)
        ce = -(qd[rows] * out['logpi'][rows]).sum(1).mean()
        rec = sum(fn.mse_loss(model.rec[k](out['z'][:, k]), T[0]) for k in range(model.K)) / model.K
        ind = independence(out['zr']) if indep_weight > 0 else torch.zeros(())
        lab_loss = margin_loss(out['p'], lab, label_margin) if lab is not None else torch.zeros(())
        loss = ce + rec_weight * rec + decay * model.decay() + indep_weight * ind + label_weight * lab_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({'epoch': ep + 1, 'tau': tau, 'ce': float(ce.detach()), 'rec': float(rec.detach()), 'indep': float(ind.detach()),
                     'label': float(lab_loss.detach())})
        if log and (ep + 1) % 50 == 0:
            log(f'seed {seed} epoch {ep + 1} tau {tau:.3f} ce {float(ce.detach()):.4f} rec {float(rec.detach()):.4f} '
                f'indep {float(ind.detach()):.4f} label {float(lab_loss.detach()):.4f}')
        if stop_tol is not None and (ep + 1) % stop_window == 0 and ep + 1 >= 2 * stop_window:
            ces = [h_['ce'] for h_ in hist]
            gain = np.mean(ces[-2 * stop_window:-stop_window]) - np.mean(ces[-stop_window:])
            if gain < stop_tol:
                if log:
                    log(f'seed {seed} stopped at epoch {ep + 1}: window gain {gain:.5f}')
                break
    model.eval()
    return model, hist, tau
