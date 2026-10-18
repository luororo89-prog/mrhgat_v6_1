"""DisenGCN-official routing with GATv2 neighbour weights and the gravity destination head.

Following DisenGCN-py3/notebook_single-label-tasks/main.py (CapsuleNet, RoutingLayer):
  messages        the routing layers have no parameters of their own; the message of the edge A -> B is the
                  destination's vector of the layer input v_B (no OD-pair projection);
  last iteration  the centre of the last routing iteration is not normalised,
                      u_A,k = s_A,k + sum_B w_AB p_AB,k v_B,k;
  head input      the destination head reads the output of the last routing layer, H = dropout(relu(u)),
                      score_AB = MLP([H_A, H_B, t_AB])   (gravity_mlp),
                  with no attention re-aggregation after the routing.
Neighbour weights w_AB (GATv2, Brody et al. 2022, dynamic attention), one attention per routing layer, softmax over
the message destinations of A,
       w_AB = softmax_B(a^T LeakyReLU(Theta_s v_A + Theta_t v_B [+ theta_e log pi_AB])),
a starting at zero (equal weights at the start). disen_gatv2 reads the node vectors only; disen_gatv2_flow adds the
log flow share pi_AB of the message edge as edge feature.
Kept from fix1: linear projection with ReLU, inflow pressure in the self vector of the first layer, routing starting
from the own vector, tau, topology bias on the routing logits, dropout and l2 between the layers.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model_v13 import MechNet
from model_v9 import l2n


DISEN_VARIANTS = {
    'disen_gatv2': {'gat_flow': False},
    'disen_gatv2_flow': {'gat_flow': True},
}


class DisenGATv2Net(MechNet):
    def __init__(self, *args, gat_flow=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.gravity_mlp, 'the destination head is gravity_mlp'
        width = self.K * self.d
        self.gat_flow = gat_flow
        self.gat_source = nn.ModuleList(nn.Linear(width, width) for _ in range(self.routing_layers))
        self.gat_target = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(self.routing_layers))
        self.gat_a = nn.Parameter(torch.zeros(self.routing_layers, width))
        if gat_flow:
            self.gat_edge = nn.Parameter(torch.zeros(self.routing_layers, width))

    def neighbour_weights(self, layer, v, share, oi, di):
        """GATv2 weights of the message edges, softmax over the message destinations of every origin."""
        n = len(v)
        h = v.reshape(n, -1)
        e = self.gat_source[layer](h)[oi] + self.gat_target[layer](h)[di]
        if self.gat_flow:
            e = e + torch.log(share[oi, di])[:, None] * self.gat_edge[layer]
        logit = (F.leaky_relu(e, 0.2) * self.gat_a[layer]).sum(-1)
        top = torch.full((n,), -torch.inf).scatter_reduce(0, oi, logit.detach(), 'amax')
        weight = torch.exp(logit - top[oi])
        return weight / torch.zeros(n).index_add(0, oi, weight)[oi]

    def _centres(self, message, anchor, mass, origin, bias, tau):
        """Routing iterations of RoutingLayer; the last centre keeps its length."""
        centre = anchor
        for it in range(self.iters):
            logits = (message * centre[origin]).sum(-1) / tau + bias
            probability = torch.softmax(logits, dim=-1)
            u = anchor + torch.zeros_like(anchor).index_add(
                0, origin, mass[:, None, None] * probability[:, :, None] * message)
            centre = l2n(u) if it < self.iters - 1 else u
        return centre

    def route_layers(self, x, share, msg, tau, t=None):
        z = self.embed(x)
        v = l2n(z)
        s = v if self.inflow is None else l2n(z + torch.einsum('nf,kfd->nkd', self.inflow, self.V))
        oi, di = torch.nonzero(msg & (share > 0), as_tuple=True)
        bias = 0.0 if self.etype is None else self.Bt[:, self.etype].permute(1, 0, 2)
        edge_bias = bias if isinstance(bias, float) else bias[oi, :, di]
        for layer in range(self.routing_layers):
            if layer:
                v = l2n(F.dropout(torch.relu(u), self.dropout, self.training))
                s = v
            w = self.neighbour_weights(layer, v, share, oi, di)
            u = self._centres(v[di], s, w, oi, edge_bias, tau)
        logits = torch.einsum('ukd,vkd->ukv', l2n(u), v) / tau + bias
        return {'z': z, 'zr': v, 'u': u, 'p': torch.softmax(logits, dim=1)}

    def forward(self, x, t, R, cand, share, msg, tau):
        out = self.route_layers(x, share, msg, tau, t)
        h = torch.relu(out['u'])
        H = F.dropout(h, self.dropout, self.training).reshape(len(x), -1)
        hidden = self.G_o(H)[:, None, :] + self.G_d(H)[None, :, :] + t[:, :, None] * self.g_t
        score = self.G_out(F.leaky_relu(hidden, 0.2)).squeeze(2)
        out.update(c=h, c_route=l2n(out['u']), score=score,
                   logpi=torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1))
        return out


def make_disen_model(variant, cfg, n, topology=None):
    return DisenGATv2Net(
        d=cfg['hid_dim'] // 2, iters=cfg['iters'], n_channels=2,
        out_weight='share', attention=False, base=False,
        pressure=np.zeros((n, 6)), etype=topology, n=n,
        proj='linear', routing_layers=cfg['routing_layers'], dropout=cfg['dropout'],
        gravity_mlp=cfg['gravity_mlp'], **DISEN_VARIANTS[variant])
