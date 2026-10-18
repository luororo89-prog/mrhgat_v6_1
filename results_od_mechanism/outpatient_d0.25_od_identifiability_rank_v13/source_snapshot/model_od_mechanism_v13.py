"""Non-exclusive mechanism routing with three-channel node representations and destination prediction (M1).

Node i: supply x_i (nine standardised indicators), context flows phi_i = standardised [log1p F_ii, log1p O_i]
(F_ii the within-community trips, O_i the available outgoing message trips), inflow pressure P_i (pressure_context,
set by forward_context into model.inflow).
1  Local baseline (own channel, no neighbours): h_i^base = ReLU(W_base [z_i, phi_i] + b), z_i = ReLU(W x_i + b) the
   existing linear supply embedding.
2  Node mechanism propensities, two independent sigmoids (no softmax, no sum constraint):
       s_i^c = sigmoid(w_c . [x_i, phi_i, P_i] + b_c),   c in {comp, compete}.
3  OD gates on the available outgoing messages i -> j, two independent sigmoids with separate parameters:
       q_ij^c = sigmoid(MLP_c([s_i^c, x_i, x_j, log1p F_ij, d_ij, onehot r_ij, P_i])),
   so one OD can take part in both channels and the outflows of one origin can be routed differently.
4  Channel aggregation over the available outgoing neighbours, independent GATv2 attention per channel
   (Brody et al. 2022), normalised over the outgoing message neighbours of i,
       a_ij^c = softmax_j(att_c . LeakyReLU(U_s,c x_i + U_t,c x_j + U_e,c e_ij)),   e_ij = [log1p F_ij, d_ij, onehot r_ij],
       h_i^comp    = sum_j a_ij^comp q_ij^comp W_comp x_j,
       h_i^compete = sum_j a_ij^compete q_ij^compete W_compete v_ij,   v_ij = [x_j, d_ij, onehot r_ij];
   a node without outgoing messages gets zero external channels. Masked edges carry no message, enter no attention
   denominator and no pressure.
5  Node representation h_i = [h_i^base || h_i^comp || h_i^compete] (kept separately in out['channels']).
Destination head on every candidate i -> j:
       z_ij = MLP([h_i, h_j, x_i, x_j, d_ij]) + B0[r_ij],   softmax over the candidates (run_od_relation_v13),
the supply pair x_i, x_j entering directly (pair_path; switched off in mech_no_pair). The destination CE reaches the
gates, the attention and the channel maps through h_i and h_j. No balance, orthogonality or label term.
Variants: mech_full; mech_base_only, mech_base_comp, mech_base_compete (external channels removed and retrained);
mech_no_pair (no direct supply pair path). out['p'] [n, 2, n] holds the gates q^comp, q^compete of all pairs (the flow
input is 0 where i -> j carries no message), out['s'] the node propensities.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model_v13 import MechNet


MECH_VARIANTS = {
    'mech_full': {'channels': ('base', 'comp', 'compete'), 'pair_path': True},
    'mech_base_only': {'channels': ('base',), 'pair_path': True},
    'mech_base_comp': {'channels': ('base', 'comp'), 'pair_path': True},
    'mech_base_compete': {'channels': ('base', 'compete'), 'pair_path': True},
    'mech_no_pair': {'channels': ('base', 'comp', 'compete'), 'pair_path': False},
}
MECHANISMS = ('comp', 'compete')


def group_softmax(logit, index, n):
    """Softmax of edge logits [m, ...] over the edges sharing the same origin index."""
    shape = (n,) + logit.shape[1:]
    top = torch.full(shape, -torch.inf).scatter_reduce(
        0, index.view(-1, *[1] * (logit.dim() - 1)).expand_as(logit), logit.detach(), 'amax')
    e = torch.exp(logit - top[index])
    return e / torch.zeros(shape).index_add(0, index, e)[index]


class MechanismNet(MechNet):
    def __init__(self, *args, channels=('base', 'comp', 'compete'), pair_path=True, gate_hidden=16, att_dim=16, **kwargs):
        super().__init__(*args, **kwargs)
        d, f = self.d, self.inflow.shape[1]
        self.channels, self.pair_path = tuple(channels), pair_path
        self.register_buffer('phi', torch.zeros(self.inflow.shape[0], 2))
        self.register_buffer('context_flow', torch.zeros(self.inflow.shape[0], self.inflow.shape[0]))
        self.base = nn.Linear(self.K * d + 2, d)
        self.propensity = nn.ModuleDict({c: nn.Linear(9 + 2 + f, 1) for c in MECHANISMS})
        gate_in = 1 + 9 + 9 + 1 + 1 + 3 + f
        self.gate = nn.ModuleDict({c: nn.Sequential(nn.Linear(gate_in, gate_hidden), nn.LeakyReLU(0.2),
                                                    nn.Linear(gate_hidden, 1)) for c in MECHANISMS})
        self.att_s = nn.ModuleDict({c: nn.Linear(9, att_dim) for c in MECHANISMS})
        self.att_t = nn.ModuleDict({c: nn.Linear(9, att_dim, bias=False) for c in MECHANISMS})
        self.att_e = nn.ModuleDict({c: nn.Linear(5, att_dim, bias=False) for c in MECHANISMS})
        self.att_a = nn.ParameterDict({c: nn.Parameter(torch.empty(att_dim)) for c in MECHANISMS})
        for c in MECHANISMS:
            nn.init.normal_(self.att_a[c], std=1 / np.sqrt(att_dim))
        self.value = nn.ModuleDict({'comp': nn.Linear(9, d), 'compete': nn.Linear(9 + 1 + 3, d)})
        width = 3 * d
        self.M_o, self.M_d = nn.Linear(width, self.gravity_mlp), nn.Linear(width, self.gravity_mlp, bias=False)
        self.M_xo, self.M_xd = nn.Linear(9, self.gravity_mlp, bias=False), nn.Linear(9, self.gravity_mlp, bias=False)
        self.M_t = nn.Parameter(torch.zeros(self.gravity_mlp))
        self.M_out = nn.Linear(self.gravity_mlp, 1)
        self.zero_external = False

    def set_context(self, context, internal):
        """Context flows of the available messages and the standardised log1p of F_ii and O_i (forward_context)."""
        off = ~np.eye(len(context), dtype=bool)
        phi = np.log1p(np.column_stack([internal, (context * off).sum(1)]))
        phi = (phi - phi.mean(0)) / np.maximum(phi.std(0), 1e-12)
        self.phi.copy_(torch.as_tensor(phi, dtype=torch.float32))
        self.context_flow.copy_(torch.as_tensor(context * off, dtype=torch.float32))

    def gate_inputs(self, x, t, R, oi, di, s):
        flow = torch.log1p(self.context_flow[oi, di])[:, None]
        topo = F.one_hot(R[oi, di], 3).float()
        return flow, topo, torch.cat([s[oi][:, None], x[oi], x[di], flow, t[oi, di][:, None], topo, self.inflow[oi]], 1)

    def mechanisms(self, x, t, R, msg):
        n = len(x)
        node = torch.cat([x, self.phi, self.inflow], 1)
        s = {c: torch.sigmoid(self.propensity[c](node)).squeeze(1) for c in MECHANISMS}
        oi, di = torch.nonzero(msg & ~torch.eye(n, dtype=torch.bool), as_tuple=True)
        q, a, v = {}, {}, {}
        for c in MECHANISMS:
            flow, topo, u = self.gate_inputs(x, t, R, oi, di, s[c])
            q[c] = torch.sigmoid(self.gate[c](u)).squeeze(1)
            edge = torch.cat([flow, t[oi, di][:, None], topo], 1)
            e = (F.leaky_relu(self.att_s[c](x)[oi] + self.att_t[c](x)[di] + self.att_e[c](edge), 0.2)
                 * self.att_a[c]).sum(1)
            a[c] = group_softmax(e, oi, n)
            v[c] = self.value[c](x[di] if c == 'comp' else torch.cat([x[di], t[oi, di][:, None], topo], 1))
        return s, q, a, v, oi, di

    def all_gates(self, x, t, R, s):
        n = len(x)
        oi, di = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        oi, di = oi.reshape(-1), di.reshape(-1)
        return torch.stack([torch.sigmoid(self.gate[c](self.gate_inputs(x, t, R, oi, di, s[c])[2])).view(n, n)
                            for c in MECHANISMS], 1)

    def forward(self, x, t, R, cand, share, msg, tau):
        n = len(x)
        z = self.embed(x).reshape(n, -1)
        base = torch.relu(self.base(torch.cat([z, self.phi], 1)))
        s, q, a, v, oi, di = self.mechanisms(x, t, R, msg)
        external = {c: torch.zeros(n, self.d).index_add(0, oi, (a[c] * q[c])[:, None] * v[c]) for c in MECHANISMS}
        channels = {'base': base, **external}
        for c in MECHANISMS:
            if c not in self.channels or self.zero_external:
                channels[c] = torch.zeros_like(channels[c])
        h = torch.cat([channels['base'], channels['comp'], channels['compete']], 1)
        hd = F.dropout(h, self.dropout, self.training)
        hidden = self.M_o(hd)[:, None, :] + self.M_d(hd)[None, :, :] + t[:, :, None] * self.M_t
        if self.pair_path:
            hidden = hidden + self.M_xo(x)[:, None, :] + self.M_xd(x)[None, :, :]
        score = self.M_out(F.leaky_relu(hidden, 0.2)).squeeze(2) + self.B0[R]
        out = {'score': score, 'c': h, 'channels': channels, 's': torch.stack([s[c] for c in MECHANISMS], 1),
               'q_edge': torch.stack([q[c] for c in MECHANISMS], 1), 'a_edge': torch.stack([a[c] for c in MECHANISMS], 1),
               'edge_index': torch.stack([oi, di])}
        out['p'] = self.all_gates(x, t, R, s) if not self.training else torch.zeros(n, 2, n)
        out['logpi'] = torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1)
        return out


def make_mechanism_model(variant, cfg, n, topology=None):
    return MechanismNet(
        d=cfg['hid_dim'] // 2, iters=cfg['iters'], n_channels=2, out_weight='share', attention=False, base=False,
        pressure=np.zeros((n, 6)), etype=None, n=n, proj='linear', routing_layers=1, dropout=cfg['dropout'],
        gravity_mlp=cfg['gravity_mlp'], **MECH_VARIANTS[variant])

