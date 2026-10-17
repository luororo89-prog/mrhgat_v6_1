"""Base-Compensation-Competition network (修订思路: 自环基线与补偿竞争机制), built on model_v11.

Nodes are the 15-minute communities with the nine basic supply indicators x (CHC, SH, TH); edges are the directed
outflows of the medical trips. With community A:
  Base          A -> A, the real internal trips: in the new representation A keeps its own supply with the weight of
                its internal trip share pi_AA (a community without trips keeps weight 1);
  Compensation  and
  Competition   the two channels of the outflow edges A -> B (B != A): every outflow edge is softly assigned to them.
Steps:
1  Own supply (technical self loop): channel vectors z_A,k = LeakyReLU(W_k^T (omega_k * x_A) + b_k), zr = z / ||z||
   (model_v9). The technical self loop keeps A in its routing centre with weight 1; it is recorded apart from the real
   internal trips pi_AA, which weight A in the new representation (step 4).
2  Inflow pressure on A (read before the outflow is assigned): for every tier l with service in A (RA_A,l > 0),
       P_A,l = log1p(I_A / RA_A,l),   I_A = sum_{C != A} W_CA,
   standardised over the communities with that tier, 0 where A has no service of the tier, together with the three
   service indicators 1[RA_A,l > 0]. The pressure joins only A's own vector in the routing,
       zo_A,k = l2(z_A,k + V_k^T f_A)   (model_v11 inflow option; V starts at zero);
   the supply of the inflow neighbours C is never aggregated into A.
3  Soft assignment of the outflow (DisenGCN routing, model_v11):
       p_AB,k = softmax_k(zr_B,k . c_A,k / tau),   c_A,k <- l2(zo_A,k + sum_B pi_AB p_AB,k zr_B,k)   (iters times).
4  Channel attention over the destinations (attention: true): within channel k the destinations of A are weighted by
       alpha_AB,k = softmax_B(LeakyReLU(a_k . [zr_A,k, zr_B,k]) + log(pi_AB p_AB,k)),
   and the new representation is
       h_A,k = l2(pi_AA zr_A,k + m_A,k sum_B alpha_AB,k zr_B,k),   m_A,k = sum_B pi_AB p_AB,k (outflow share of k);
   a starts at zero, where alpha_AB,k = pi_AB p_AB,k / m_A,k (flow weights). The attention coefficients are weights of
   the aggregation and not shares of service capacity.
5  Training: destination prediction as auxiliary task from the new representation (each epoch half of the training
   outflow edges carry messages, the rest and the self loop are targets),
       score_AB = sum_k zr_B,k . h_A,k / tau_s + x_B . w_m - beta t_AB + B0[topology_AB],   softmax over destinations,
   plus the per-channel reconstruction of x and weight decay. With labels_file the labelled outflow edges (anchors
   comp or compet; base anchors on A -> B edges are not used, Base being A -> A) add the flow-weighted cross-entropy
   of the soft assignment p_AB with label_weight (channel C1 compensation, C2 competition); labels of the outer fold
   label_holdout are left out. Rule-generated labels are weak labels.
Ablations: n_channels 1 (one outflow channel, no compensation/competition split); outflow: false (only Base, the
outflow carries no message into the new representation); base: false (the real internal trips are not used, A keeps
weight 1 in the new representation); attention: false; pressure: false. A frozen model can switch channels off at
evaluation (attribute off: channel indices whose outflow mass is set to zero; base_off: A keeps weight 0).
Node2vec context (n2v, optional): the node2vec embedding e (node2vec_v13.py, d = 16) of the undirected graph of the
training outflow edges weighted by log1p of the trips, fixed per seed and stored with the model, joins
  score    the destination prediction only, s_AB += (E_o e_A) . (E_d e_B) + e_B . w_e (network position of both ends and
           of the destination), while the outflow assignment keeps reading the supply and the inflow pressure;
  routing  the soft assignment only, p_AB,k = softmax_k(zr_B,k . c_A,k / tau + e_A^T M_k e_B), M_k starting at zero;
  both     both places.
Gravity score (gravity_mlp > 0, optional, Deep Gravity type, Simini et al. 2021): every community keeps the channel
vectors aggregated with the attention of step 4, concatenated H_A = [h_A,1 || ... || h_A,K], and the score of step 5
is replaced by a feed-forward network of the origin vector, the destination vector and the distance,
       score_AB = MLP([H_A, H_B, t_AB])   (one hidden layer of gravity_mlp units, LeakyReLU),
without the terms x_B . w_m and B0[topology_AB]. With freeze_from (a finished run without gravity_mlp) the parameters
of its seed are loaded and frozen, so the outflow assignment, the attention and the vectors stay those of that run, and
only the gravity network is trained.
DisenGCN prediction head (fc_head, optional, Ma et al. 2019 section 2.3): the destinations are the classes; the
concatenated aggregated vectors of the origin, with dropout (rate dropout, training only), go through one fully
connected layer to the scores of all destinations,
       score_A. = W^T dropout(H_A) + b,   W of size (K d) x n,
without the similarity, x_B . w_m, distance and B0[topology_AB] terms; L2 regularisation through decay.
DisenGCN-pytorch settings (optional; code/DisenGCN-pytorch, src/models): proj 'linear' projects the nine indicators
with one linear layer (Xavier weights, zero bias) to hid_dim, split into the K channels, ReLU, z = view(ReLU(W x + b));
routing_layers stacks parameter-free routing layers, the next layer routing l2(dropout(ReLU(c))) of the previous one
(the inflow pressure joins the first layer); dropout (rate dropout, training only) also on the vectors entering the
score; weight_decay is the L2 rate of Adam; val_share takes a validation share of the training outflow edges
(seed holdout_seed + 1), the held-out CPC on it is evaluated every epoch, the parameters with the best validation CPC
are kept and training stops after early epochs without improvement.
Channel scoring rules (channel_score, optional): every community has its aggregated channel vectors h_X,k (same
aggregation for origins and destinations); for the candidate A -> B the channel that B resembles is
       q_AB,k = softmax_k(h_B,k . h_A,k / tau_q)   (tau_q learned, starting at 1),
and every channel scores with its own rule shared by all destinations (no parameter per destination),
       score_AB = sum_k q_AB,k [h_B,k . h_A,k / tau_k + x_B . w_k - softplus(beta_k) t_AB] + B0[topology_AB],
so an edge in the wrong channel is scored with the wrong rule.
Routing shares as the channel of the candidate (route_score, optional, the DisenGCN classifier reading the channels
of the routing): the channel of every candidate A -> B is the routing share p_AB,k of the edge split itself (the
destination vector against A's routing centres, which carry A's inflow pressure), and with route_distance the
routing logits get a distance term per channel, zr_B,k . c_A,k / tau + delta_k t_AB (delta_k starting at 0, routing
layers > 1); score_AB = sum_k p_AB,k [h_B,k . h_A,k / tau_k + x_B . w_k - softplus(beta_k) t_AB] + B0[topology_AB].
With rule_topology the distance term of every channel rule is replaced by a per-channel offset of the topology of
A -> B (internal, adjacent, crossing; starting at 0): rule_k = h_B,k . h_A,k / tau_k + x_B . w_k + C_k[topology_AB].
With rule_origin the destination supply term is replaced by the origin's supply and inflow pressure features,
rule_k = h_B,k . h_A,k / tau_k + [x_A, f_A] . u_k (+ C_k[topology_AB]); as the channel shares p_AB,k differ between
the destinations, sum_k p_AB,k [x_A, f_A] . u_k does not cancel in the softmax over B (only u_1 - u_2 matters).
With consistency_score the destination score is the agreement of the origin's tendency and the channel of the edge:
the tendency of A for the edge A -> B from its own features and the topology of the edge,
q_AB,k = softmax_k([x_A, f_A] . u_k + D_k[topology_AB]), the channel of the edge from the routing share p_AB,k, and
score_AB = log(sum_k q_AB,k p_AB,k) - softplus(beta) t_AB + B0[topology_AB] (similarity and supply terms left out);
with consistency_sim the destination similarity of the channels is added back, + sum_k p_AB,k h_B,k . h_A,k / tau_k.
With rule_distance the distance term - softplus(beta_k) t_AB of every channel rule is kept next to the topology offset
(t the road impedance when road_impedance is given).
With od_pair_head, a candidate's supply pair is compared separately with the origin's aggregated channels,
       e_AB,k = l2(zr_A,k W_o,k + zr_B,k W_d,k),   a_AB,k = e_AB,k . l2(h_A,k) / exp(tau_k) + f_A . u_k,
       score_AB = sum_k p_AB,k a_AB,k + B0[topology_AB] - softplus(beta) t_AB.
The pair uses the initial supply embeddings, not the destination's aggregated representation. The pair maps are
shared with OD routing when present. Pressure is channel-specific inside the routing-weighted score; topology and
distance correct the OD score. The candidate softmax includes the internal option. This head requires route_score
and rule_origin and is used by the CE plus Adam L2 comparison runner without auxiliary losses.
Held-out edges (holdout_share > 0): a fixed random share of the observed outflow edges (holdout_seed) never carries
messages, is no target and is left out of the candidate destinations during training; evaluated with all other
observed edges as messages over the candidate destinations without messages and self loop:
       CE_hold = mean over origins of -sum_{B in H_A} q_AB log pi^_AB,   CPC = sum min(F^, F) / sum F,
       NRMSE = RMSE(F^, F) / mean F,   F^_AB = F_A(H) pi^_AB,
CE_hold_trips (origins weighted by their held-out trips) and the internal share error, mean over origins of
|pi^_AA - W_AA / (W_AA + F_A(H))| with pi^_AA the predicted self share over the candidates without messages.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v11_nodeedge'))
sys.path.insert(0, str(HERE.parent / 'v9_disengcn'))
sys.path.insert(0, str(HERE.parent / 'common'))
from model_v11 import NodeDisen  # noqa: E402
from model_v9 import l2n, schedule  # noqa: E402
from node2vec_v13 import node2vec  # noqa: E402
from graph import load_labels, MECHANISMS  # noqa: E402

TIERS = ('CHC', 'SH', 'TH')


def pressure_features(g):
    """[n, 6] inflow pressure per tier with service (standardised) and the three service indicators."""
    W, raw, cols = g['W'], g['raw'], g['columns']
    n = len(W)
    inflow = (W * ~np.eye(n, dtype=bool)).sum(0)
    feats, has_ = [], []
    for tier in TIERS:
        S = raw[:, cols.index(f'RA_{tier}')]
        has = S > 0
        p = np.zeros(n)
        p[has] = np.log1p(inflow[has] / S[has])
        p[has] = (p[has] - p[has].mean()) / p[has].std()
        feats.append(p)
        has_.append(has.astype(float))
    return np.column_stack(feats + has_)


def road_matrix(g, path):
    """[n, n] road distance of road_impedance_v12.py aligned to g, standardised over the pairs off the diagonal."""
    z = np.load(HERE.parents[1] / path)
    index = {v: k for k, v in enumerate(z['ids'].tolist())}
    order = np.array([index[v] for v in g['ids'].tolist()])
    D = z['km'][np.ix_(order, order)]
    off = ~np.eye(len(D), dtype=bool)
    return (D - D[off].mean()) / D[off].std()


class MechNet(NodeDisen):
    def __init__(self, d=8, iters=5, n_channels=2, out_weight='share', inflow=None, etype=None, pressure=None,
                 attention=True, outflow=True, base=True, emb=None, n2v=None, dest_fe=False, n=None, gravity_mlp=0,
                 fc_head=False, dropout=0.0, proj='omega', routing_layers=1, channel_score=False, route_score=False,
                 route_distance=False, rule_topology=False, rule_origin=False, rule_distance=False,
                 consistency_score=False, consistency_sim=False, od_pair_head=False):
        super().__init__(d=d, iters=iters, n_channels=n_channels, out_weight=out_weight, inflow=pressure, etype=etype)
        self.attention, self.outflow, self.base = attention, outflow, base
        self.off, self.base_off = set(), False
        self.dest_fe = dest_fe
        if dest_fe:
            self.c_B = nn.Parameter(torch.zeros(n))
        self.gravity_mlp = gravity_mlp
        if gravity_mlp:
            self.G_o = nn.Linear(self.K * d, gravity_mlp)
            self.G_d = nn.Linear(self.K * d, gravity_mlp, bias=False)
            self.g_t = nn.Parameter(torch.zeros(gravity_mlp))
            self.G_out = nn.Linear(gravity_mlp, 1)
        self.fc_head, self.dropout = fc_head, dropout
        self.proj, self.routing_layers = proj, routing_layers
        self.channel_score, self.route_score, self.route_distance = channel_score, route_score, route_distance
        if route_distance:
            assert routing_layers > 1, 'route_distance needs the stacked routing (routing_layers > 1)'
            self.rd = nn.Parameter(torch.zeros(self.K))
        self.rule_topology, self.rule_origin, self.rule_distance = rule_topology, rule_origin, rule_distance
        if rule_origin:
            self.cs_u = nn.Parameter(torch.zeros(self.K, 9 + (self.inflow.shape[1] if self.inflow is not None else 0)))
        if rule_topology:
            self.cs_topo = nn.Parameter(torch.zeros(self.K, 3))
        self.consistency_score, self.consistency_sim = consistency_score, consistency_sim
        if consistency_sim:
            assert consistency_score and (channel_score or route_score), 'consistency_sim needs consistency_score and cs_tau'
        if consistency_score:
            self.cq_u = nn.Parameter(torch.zeros(self.K, 9 + (self.inflow.shape[1] if self.inflow is not None else 0)))
            self.cq_topo = nn.Parameter(torch.zeros(self.K, 3))
            self.cq_beta = nn.Parameter(torch.zeros(()))
        if channel_score or route_score:
            self.cs_q = nn.Parameter(torch.zeros(()))
            self.cs_tau = nn.Parameter(torch.zeros(self.K))
            self.cs_w = nn.Parameter(torch.zeros(self.K, 9))
            self.cs_beta = nn.Parameter(torch.zeros(self.K))
        if proj == 'linear':
            self.lin = nn.Linear(9, self.K * d)
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
        if fc_head:
            self.fc = nn.Linear(self.K * d, n)
        if attention:
            self.att = nn.Parameter(torch.zeros(self.K, 2 * d))
        self.od_pair_head = od_pair_head
        if od_pair_head:
            assert route_score and rule_origin, 'od_pair_head needs route_score and rule_origin'
            identity = torch.eye(d).repeat(self.K, 1, 1)
            self.pair_origin = nn.Parameter(0.5 * identity.clone())
            self.pair_destination = nn.Parameter(0.5 * identity.clone())
        self.n2v = n2v if emb is not None else None
        if self.n2v:
            self.register_buffer('emb', torch.as_tensor(np.asarray(emb), dtype=torch.float32))
            de = self.emb.shape[1]
            if self.n2v in ('score', 'both'):
                self.Eo, self.Ed = nn.Linear(de, de, bias=False), nn.Linear(de, de, bias=False)
                nn.init.zeros_(self.Eo.weight)
                self.w_e = nn.Parameter(torch.zeros(de))
            if self.n2v in ('routing', 'both'):
                self.Mr = nn.Parameter(torch.zeros(self.K, de, de))

    def decay(self):
        out = super().decay() + (self.att.square().sum() if self.attention else 0.0)
        if self.n2v in ('score', 'both'):
            out = out + self.Eo.weight.square().sum() + self.Ed.weight.square().sum() + self.w_e.square().sum()
        if self.n2v in ('routing', 'both'):
            out = out + self.Mr.square().sum()
        if self.gravity_mlp:
            out = out + sum(m.weight.square().sum() for m in (self.G_o, self.G_d, self.G_out)) + self.g_t.square().sum()
        if self.fc_head:
            out = out + self.fc.weight.square().sum()
        if self.channel_score:
            out = out + self.cs_w.square().sum()
        return out

    def embed(self, x):
        if self.proj == 'linear':
            return torch.relu(self.lin(x)).view(len(x), self.K, -1)
        return super().embed(x)

    def od_pair_scores(self, z, h, p, t, R):
        """Candidate supply pair against the origin's two aggregated channels."""
        supply = l2n(z)
        origin = torch.einsum('nkd,kde->nke', supply, self.pair_origin)
        destination = torch.einsum('nkd,kde->nke', supply, self.pair_destination)
        pair = l2n(origin[:, None] + destination[None, :])
        similarity = torch.einsum('uvkd,ukd->ukv', pair, l2n(h))
        rule = similarity / torch.exp(self.cs_tau)[None, :, None]
        if self.inflow is not None:
            rule = rule + (self.inflow @ self.cs_u[:, 9:].T)[:, :, None]
        return (p * rule).sum(1) + self.B0[R] - fn.softplus(self.beta) * t

    def route_layers(self, x, share, msg, tau, t=None):
        """Stacked parameter-free routing layers (DisenGCN-pytorch), the inflow pressure in the first layer."""
        z = self.embed(x)
        zr = l2n(z)
        zo = zr if self.inflow is None else l2n(z + torch.einsum('nf,kfd->nkd', self.inflow, self.V))
        bias = 0.0 if self.etype is None else self.Bt[:, self.etype].permute(1, 0, 2)
        if self.route_distance:
            bias = bias + self.rd[None, :, None] * t[:, None, :]
        w_edge = share * msg.float()
        for layer in range(self.routing_layers):
            if layer:
                zr = l2n(fn.dropout(torch.relu(c), self.dropout, self.training))
                zo = zr
            c = zo
            for _ in range(self.iters):
                p = torch.softmax(self.sim(zr, c) / tau + bias, 1)
                c = l2n(zo + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * p, zr))
            p = torch.softmax(self.sim(zr, c) / tau + bias, 1)
        return {'z': z, 'zr': zr, 'c': c, 'c_route': c, 'p': p}

    def route(self, x, share, msg, tau):
        """Routing of model_v11 with the node2vec bias e_A^T M_k e_B on the logits."""
        z = self.embed(x)
        zr = l2n(z)
        zo = zr if self.inflow is None else l2n(z + torch.einsum('nf,kfd->nkd', self.inflow, self.V))
        bias = torch.einsum('ud,kde,ve->ukv', self.emb, self.Mr, self.emb)
        if self.etype is not None:
            bias = bias + self.Bt[:, self.etype].permute(1, 0, 2)
        w_edge = share * msg.float()
        c = zo
        for _ in range(self.iters):
            p = torch.softmax(self.sim(zr, c) / tau + bias, 1)
            c = l2n(zo + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * p, zr))
        p = torch.softmax(self.sim(zr, c) / tau + bias, 1)
        return {'z': z, 'zr': zr, 'c': c, 'c_route': c, 'p': p}

    def forward(self, x, t, R, cand, share, msg, tau):
        if self.n2v in ('routing', 'both'):
            out = self.route(x, share, msg, tau)
        elif self.routing_layers > 1:
            out = self.route_layers(x, share, msg, tau, t)
        else:
            out = super().forward(x, t, R, cand, share, msg, tau)
        zr, p = out['zr'], out['p']
        a = torch.diagonal(share).clone()
        a[share.sum(1) <= 0] = 1.0
        if not self.base:
            a = torch.ones_like(a)
        if self.base_off:
            a = torch.zeros_like(a)
        mass = (share * msg.float())[:, None, :] * p * float(self.outflow)
        if self.off:
            keep = torch.ones(self.K)
            keep[list(self.off)] = 0.0
            mass = mass * keep[None, :, None]
        if self.attention:
            d = zr.shape[2]
            e = fn.leaky_relu(torch.einsum('ukd,kd->uk', zr, self.att[:, :d])[:, :, None]
                              + torch.einsum('vkd,kd->kv', zr, self.att[:, d:])[None], 0.2)
            logit = (e + torch.log(mass.clamp_min(1e-30))).masked_fill(mass <= 0, -1e9)
            alpha = torch.softmax(logit, 2) * (mass.sum(2, keepdim=True) > 0)
            out['alpha'] = alpha
            agg = mass.sum(2)[:, :, None] * torch.einsum('ukv,vkd->ukd', alpha, zr)
        else:
            agg = torch.einsum('ukv,vkd->ukd', mass, zr)
        out['c'] = l2n(a[:, None, None] * zr + agg)
        hc = fn.dropout(out['c'], self.dropout, self.training) if self.dropout else out['c']
        if self.od_pair_head:
            score = self.od_pair_scores(out['z'], hc, p, t, R)
            out['q'] = p
        elif self.consistency_score:
            o = x if self.inflow is None else torch.cat([x, self.inflow], 1)
            qa = torch.softmax((o @ self.cq_u.T)[:, :, None] + self.cq_topo[:, R].permute(1, 0, 2), 1)
            agree = (qa * p).sum(1)
            score = torch.log(agree.clamp_min(1e-12)) - fn.softplus(self.cq_beta) * t + self.B0[R]
            if self.consistency_sim:
                S = torch.einsum('ukd,vkd->ukv', hc, hc)
                score = score + (p * S / torch.exp(self.cs_tau)[None, :, None]).sum(1)
            out['q'] = qa
            out['agree'] = agree
        elif self.fc_head:
            score = self.fc(hc.reshape(len(x), -1))
        elif self.channel_score or self.route_score:
            S = torch.einsum('ukd,vkd->ukv', hc, hc)
            q = p if self.route_score else torch.softmax(S / torch.exp(self.cs_q), 1)
            if self.rule_origin:
                o = x if self.inflow is None else torch.cat([x, self.inflow], 1)
                rule = S / torch.exp(self.cs_tau)[None, :, None] + (o @ self.cs_u.T)[:, :, None]
            else:
                rule = S / torch.exp(self.cs_tau)[None, :, None] + (x @ self.cs_w.T).T[None]
            if self.rule_topology:
                rule = rule + self.cs_topo[:, R].permute(1, 0, 2)
            if self.rule_distance or not self.rule_topology:
                rule = rule - fn.softplus(self.cs_beta)[None, :, None] * t[:, None, :]
            score = (q * rule).sum(1) + self.B0[R]
            out['q'] = q
        elif self.gravity_mlp:
            H = hc.reshape(len(x), -1)
            hid = self.G_o(H)[:, None, :] + self.G_d(H)[None, :, :] + t[:, :, None] * self.g_t
            score = self.G_out(fn.leaky_relu(hid, 0.2)).squeeze(2)
        else:
            score = self.sim(zr, hc).sum(1) / torch.exp(self.log_tau_s) + (x @ self.w_m)[None, :] \
                - fn.softplus(self.beta) * t + self.B0[R]
        if self.dest_fe:
            score = score + self.c_B[None, :]
        if self.n2v in ('score', 'both'):
            score = score + self.Eo(self.emb) @ self.Ed(self.emb).T + (self.emb @ self.w_e)[None, :]
        out['score'] = score
        out['logpi'] = torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1)
        return out


def holdout_mask(W, share_, seed):
    """Fixed random share of the observed non-self outflow edges."""
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    oi, di = np.nonzero(obs)
    pick = np.random.default_rng(seed).random(len(oi)) < share_
    H = np.zeros_like(obs)
    H[oi[pick], di[pick]] = True
    return H


def evaluate(model, T, share, W, H, tau):
    """Held-out CE, CPC and NRMSE of the auxiliary destination prediction (module docstring)."""
    n = len(W)
    msg = (W > 0) & ~np.eye(n, dtype=bool) & ~H
    was = model.training
    model.eval()
    with torch.no_grad():
        logpi_full = model(*T, share, torch.as_tensor(msg), tau)['logpi']
        ev = (T[3] & ~torch.as_tensor(msg) & ~torch.eye(n, dtype=torch.bool))
        logpi = torch.log_softmax(logpi_full.masked_fill(~ev, -1e9), 1).numpy()
    model.train(was)
    ev = ev.numpy()
    F = W * H
    Fu = F.sum(1)
    rows = Fu > 0
    q = F[rows] / Fu[rows, None]
    ce_o = -(q * np.where(H[rows], logpi[rows], 0.0)).sum(1)
    Fh = Fu[rows, None] * np.exp(logpi[rows]) * ev[rows]
    m = ev[rows]
    self_pred = np.exp(np.diagonal(logpi_full.numpy()))[rows]
    self_true = np.diagonal(W)[rows] / (np.diagonal(W)[rows] + Fu[rows])
    return {'hold_ce': float(ce_o.mean()), 'hold_ce_trips': float((ce_o * Fu[rows]).sum() / Fu[rows].sum()),
            'hold_cpc': float(np.minimum(Fh, F[rows])[m].sum() / F[rows][m].sum()),
            'hold_nrmse': float(np.sqrt(np.mean((Fh[m] - F[rows][m]) ** 2)) / np.mean(F[rows][m])),
            'hold_internal_err': float(np.abs(self_pred - self_true).mean())}


def mechanism_labels(g, labels_file, label_holdout=None):
    """Labelled outflow edges with comp (C1) or compet (C2); base anchors and the outer fold label_holdout dropped."""
    lab = load_labels(labels_file, g)
    keep = np.isin(lab['mech'], [MECHANISMS.index('comp'), MECHANISMS.index('compet')])
    if label_holdout is not None and 'fold' in lab:
        keep &= lab['fold'] != label_holdout
    ch = np.where(lab['mech'] == MECHANISMS.index('comp'), 0, 1)
    return {'origin': torch.as_tensor(lab['origin'][keep]), 'dest': torch.as_tensor(lab['dest'][keep]),
            'channel': torch.as_tensor(ch[keep]),
            'weight': torch.as_tensor((lab['flow'] * lab['confidence'])[keep], dtype=torch.float32)}


def train_v13(g, seed, n_channels=2, iters=5, epochs=1000, lr=0.02, decay=1e-3, rec_weight=0.1, tau0=0.05,
              tau_final=0.05, n_stages=1, stage_epochs=30, stop_window=50, stop_tol=None, out_weight='share',
              pressure=True, attention=True, outflow=True, base=True,
              edge_topology=False, road_impedance=None, dest_fe=False,
              labels_file=None, label_weight=1.0, label_holdout=None, holdout_share=0.0, holdout_seed=0, n2v=None,
              load_from=None, gravity_mlp=0, freeze_from=None, fc_head=False, dropout=0.0, proj='omega', hid_dim=None,
              routing_layers=1, weight_decay=0.0, val_share=0.0, early=None, channel_score=False, route_score=False,
              route_distance=False, rule_topology=False, rule_origin=False, rule_distance=False,
              consistency_score=False, consistency_sim=False, log=None, **_):
    """Interface of model_v11.train_v11 (run_v9 pipeline); held-out metrics in the last history record."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    H = holdout_mask(W, holdout_share, holdout_seed) if holdout_share > 0 else np.zeros_like(obs)
    T = [torch.as_tensor(g['x'], dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
         torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
    if road_impedance:
        T[1] = torch.as_tensor(road_matrix(g, road_impedance), dtype=torch.float32)
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    rows = torch.as_tensor(g['origins'])
    emb = None
    if n2v:
        Wt = np.log1p(W * (obs & ~H))
        emb = node2vec(Wt + Wt.T, seed)
    model = MechNet(d=hid_dim // n_channels if hid_dim else 8, proj=proj, routing_layers=routing_layers, channel_score=channel_score,
                    route_score=route_score, route_distance=route_distance, rule_topology=rule_topology,
                    rule_origin=rule_origin, rule_distance=rule_distance, consistency_score=consistency_score,
                    consistency_sim=consistency_sim,
                    iters=iters, n_channels=n_channels, out_weight=out_weight, attention=attention, outflow=outflow, base=base,
                    emb=emb, n2v=n2v, dest_fe=dest_fe, n=n, gravity_mlp=gravity_mlp, fc_head=fc_head, dropout=dropout,
                    pressure=pressure_features(g) if pressure else None, etype=g['K'] if edge_topology else None)
    tau = tau_final
    temps = schedule(tau0, tau_final, n_stages)
    src = HERE.parents[1] / load_from / f'seed{seed}' if load_from else None
    if src is not None and (src / 'model.pt').exists():
        model.load_state_dict(torch.load(src / 'model.pt'))
        model.eval()
        if log:
            log(f'seed {seed} parameters loaded from {load_from}')
        return model, pd.read_csv(src / 'history.csv').to_dict('records'), tau
    if freeze_from:
        model.load_state_dict(torch.load(HERE.parents[1] / freeze_from / f'seed{seed}' / 'model.pt'), strict=False)
        for name_, p_ in model.named_parameters():
            p_.requires_grad = name_.startswith(('G_', 'g_t'))
    lab = mechanism_labels(g, labels_file, label_holdout) if labels_file else None
    opt = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=lr, weight_decay=weight_decay)
    Hv = holdout_mask(W * ~H, val_share, holdout_seed + 1) if val_share > 0 else np.zeros_like(obs)
    train_obs = obs & ~H & ~Hv
    cand_train = T[3] & ~torch.as_tensor(H | Hv)
    hist = []
    best, best_cpc, since = None, -np.inf, 0
    for ep in range(epochs):
        tau = temps[min(ep // stage_epochs, n_stages - 1)]
        msg_np = train_obs & (rng.random(W.shape) < 0.5)
        msg = torch.as_tensor(msg_np)
        out = model(T[0], T[1], T[2], cand_train, share, msg, tau)
        tgt = torch.as_tensor(W * ~msg_np * ~H * ~Hv, dtype=torch.float32)
        qd = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)
        ce = -(qd[rows] * out['logpi'][rows]).sum(1).mean()
        rec = sum(fn.mse_loss(model.rec[k](out['z'][:, k]), T[0]) for k in range(model.K)) / model.K
        loss = ce + rec_weight * rec + decay * model.decay()
        lab_loss = torch.zeros(())
        if lab is not None:
            full = model(T[0], T[1], T[2], cand_train, share, torch.as_tensor(train_obs), tau)['p']
            pe = full[lab['origin'], :, lab['dest']]
            lab_loss = (lab['weight'] * -torch.log(pe.gather(1, lab['channel'][:, None]).squeeze(1).clamp_min(1e-12))).sum() \
                / lab['weight'].sum()
            loss = loss + label_weight * lab_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({'epoch': ep + 1, 'tau': tau, 'ce': float(ce.detach()), 'rec': float(rec.detach()),
                     'label': float(lab_loss.detach())})
        if Hv.any():
            val = evaluate(model, T[:3] + [T[3] & ~torch.as_tensor(H)], share, W * ~H, Hv, tau)
            hist[-1].update({'val_ce': val['hold_ce'], 'val_cpc': val['hold_cpc']})
            if val['hold_cpc'] > best_cpc:
                best_cpc, since = val['hold_cpc'], 0
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
            if early is not None and since >= early:
                if log:
                    log(f'seed {seed} early stop at epoch {ep + 1}, best validation CPC {best_cpc:.4f}')
                break
        if log and (ep + 1) % 50 == 0:
            extra = ' ' + ' '.join(f'{k} {v:.4f}' for k, v in evaluate(model, T, share, W, H, tau).items()) if H.any() else ''
            log(f'seed {seed} epoch {ep + 1} ce {float(ce.detach()):.4f} rec {float(rec.detach()):.4f} '
                f'label {float(lab_loss.detach()):.4f}{extra}')
        if stop_tol is not None and (ep + 1) % stop_window == 0 and ep + 1 >= 2 * stop_window:
            ces = [h_['ce'] for h_ in hist]
            gain = np.mean(ces[-2 * stop_window:-stop_window]) - np.mean(ces[-stop_window:])
            if gain < stop_tol:
                if log:
                    log(f'seed {seed} stopped at epoch {ep + 1}: window gain {gain:.5f}')
                break
    if best is not None:
        model.load_state_dict(best)
        hist[-1]['best_epoch'] = len(hist) - since
    model.eval()
    if H.any():
        hist[-1].update(evaluate(model, T, share, W, H, tau))
    return model, hist, tau
