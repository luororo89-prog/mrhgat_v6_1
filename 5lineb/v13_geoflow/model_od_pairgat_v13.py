"""OD-pair routing, GATv2 aggregation inside every channel, gravity destination head and channel independence.

Every routing layer l (DisenGCN-official stacking, five layers):
1  OD-pair projections of fix1 (independent per layer): the pair of the origin with itself initialises the centre,
       e_ii,k = l2(zo_i,k W_o,k + v_i,k W_d,k),   e_ij,k = l2(zo_i,k W_o,k + v_j,k W_d,k),
   and the routing compares pair with pair (DisenGCN routing on pair vectors, flow-share mass pi_ij, tau, topology
   bias), c_i,k <- l2(e_ii,k + sum_j pi_ij p_ij,k e_ij,k),   p_ij,k = softmax_k(e_ij,k . c_i,k / tau + b_k[topology]).
2  Inside channel k the neighbours are aggregated with dynamic attention (GATv2, Brody et al. 2022, gatv2_conv_PyG:
   lin_l for the neighbour, lin_r for the target, self loop, Glorot initialisation), the routing mass of the edge
   entering the attention as prior; the values are the channel vectors themselves (parameter-free aggregation of
   DisenGCN),
       a_ij,k = softmax_{j in {i} + N(i)}(att_k . LeakyReLU(W_l,k v_j,k + W_r,k zo_i,k) + log(pi_ij p_ij,k)),
       h_i,k = sum_j a_ij,k v_j,k   (self loop with prior 1; the output keeps its length).
3  The next layer reads v = l2(dropout(relu(h))); the last layer gives H_i = [relu(h_i,1) || ... || relu(h_i,K)].
Destination head (Deep Gravity type, gravity_mlp): score_ij = MLP([H_i, H_j, t_ij]); the destination CE reaches the
routing through the attention prior and the pair projections. Channel independence (IPGDN, Liu et al. 2020): the
mean normalised HSIC between the channels of the last layer, l2(h_k), over the communities (out['indep']), added to
the loss with indep_weight by the runner. Pressure enters zo of the first layer as in fix1.
Options against the channel collapse (variants):
  pair_gatv2_dec     DEC target of every layer (Xie et al. 2016, ResiliNet), q the routing shares of the message
                     edges, f_k = sum_e pi_e q_e,k, t_e,k = (q_e,k^2 / f_k) / sum_k' (q_e,k'^2 / f_k') (no gradient),
                     out['dec'] = mean over layers of sum_e pi_e KL(t_e || q_e) / sum_e pi_e, weighted by dec_weight;
  pair_gatv2_center  the routing similarity of every channel is centred on its pi-weighted mean over the message
                     neighbours of the origin, before tau and the topology bias (iterations, edge shares, output p);
  pair_gatv2_center_notopo  centring as pair_gatv2_center and no topology bias on the routing logits;
  pair_gatv2_center_notopo_dec  pair_gatv2_center_notopo with the DEC target of pair_gatv2_dec;
  pair_gatv2_center_notopo_dec_supply  as pair_gatv2_center_notopo_dec, the projections read the supply features only
                     (no inflow pressure in zo of the first layer);
  pair_gatv2_center_notopo_dec_state  as pair_gatv2_center_notopo_dec, every layer adds the origin state against the
                     signed supply difference to the routing logits after the centring, o_i^T M_l,k d_ij, with
                     o_i = [x_i, inflow pressure f_i, retention r_i, 1], d_ij = [x_j - x_i, 1], M starting at zero;
                     r_i = W_ii / (W_ii + available outgoing messages), standardised (set by forward_context); the
                     internal trips leave the training target (exclude_internal_target in the config);
  pair_gatv2_center_notopo_dec_diff  as pair_gatv2_center_notopo_dec, every layer adds the signed supply difference of
                     the pair to the routing logits after the centring, u_l,k . (x_j - x_i), x the nine standardised
                     supply indicators, u starting at zero;
  pair_gatv2_sample20  DisenGCN-official neighbour sampling: per forward at most 20 message neighbours per origin,
                     drawn uniformly with replacement where there are more, pi renormalised over the draws, shared by
                     the five layers; fixed draws (generator seed 0) in evaluation mode.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v11_nodeedge'))
from model_od_ablation_v13 import ODAblationNet  # noqa: E402
from model_v11 import independence  # noqa: E402
from model_v9 import l2n  # noqa: E402


PAIRGAT_VARIANTS = {
    'pair_gatv2': {},
    'pair_gatv2_dec': {'dec': True},
    'pair_gatv2_center': {'route_centering': True},
    'pair_gatv2_center_notopo': {'route_centering': True, 'route_topology': False},
    'pair_gatv2_center_notopo_dec': {'route_centering': True, 'route_topology': False, 'dec': True},
    'pair_gatv2_center_notopo_dec_supply': {'route_centering': True, 'route_topology': False, 'dec': True,
                                            'supply_only': True},
    'pair_gatv2_center_notopo_dec_diff': {'route_centering': True, 'route_topology': False, 'dec': True,
                                          'supply_difference': True},
    'pair_gatv2_center_notopo_dec_state': {'route_centering': True, 'route_topology': False, 'dec': True,
                                           'origin_state': True},
    'pair_gatv2_sample20': {'neighbour_sample': 20},
}


class PairGATv2Net(ODAblationNet):
    def __init__(self, *args, dec=False, route_centering=False, neighbour_sample=0, supply_only=False,
                 supply_difference=False, origin_state=False, **kwargs):
        super().__init__(*args, independent_layers=True, **kwargs)
        self.dec, self.route_centering, self.neighbour_sample = dec, route_centering, neighbour_sample
        self.supply_only, self.supply_difference = supply_only, supply_difference
        if supply_difference:
            self.diff_u = nn.Parameter(torch.zeros(self.routing_layers, self.K, 9))
        self.origin_state = origin_state
        if origin_state:
            self.register_buffer('retention', torch.zeros(self.inflow.shape[0]))
            self.state_M = nn.Parameter(torch.zeros(self.routing_layers, self.K, 9 + self.inflow.shape[1] + 2, 10))

        assert self.gravity_mlp, 'the destination head is gravity_mlp'
        L, K, d = self.routing_layers, self.K, self.d
        self.gat_l = nn.Parameter(torch.empty(L, K, d, d))
        self.gat_r = nn.Parameter(torch.empty(L, K, d, d))
        self.gat_att = nn.Parameter(torch.empty(L, K, d))
        for layer in range(L):
            nn.init.xavier_uniform_(self.gat_att.data[layer])
            for k in range(K):
                nn.init.xavier_uniform_(self.gat_l.data[layer, k])
                nn.init.xavier_uniform_(self.gat_r.data[layer, k])

    def state_terms(self, x):
        """Origin state [x_i, f_i, r_i, 1] of the routing term."""
        return torch.cat([x, self.inflow, self.retention[:, None], torch.ones(len(x), 1)], 1)

    def aggregate(self, layer, v, zo, mass, oi, di):
        """GATv2 inside every channel over the self loop and the message neighbours, routing mass as prior."""
        n = len(v)
        xl = torch.einsum('nkd,kde->nke', v, self.gat_l[layer])
        xr = torch.einsum('nkd,kde->nke', zo, self.gat_r[layer])
        att = self.gat_att[layer]
        own = (F.leaky_relu(xl + xr, 0.2) * att).sum(-1)
        edge = (F.leaky_relu(xl[di] + xr[oi], 0.2) * att).sum(-1) + torch.log(mass.clamp_min(1e-30))
        top = own.detach().clone().scatter_reduce(0, oi[:, None].expand_as(edge), edge.detach(), 'amax')
        e_own, e_edge = torch.exp(own - top), torch.exp(edge - top[oi])
        total = e_own + torch.zeros_like(own).index_add(0, oi, e_edge)
        return (e_own / total)[:, :, None] * v + torch.zeros_like(v).index_add(
            0, oi, (e_edge / total[oi])[:, :, None] * v[di])

    @staticmethod
    def origin_mean(sim, mass, origin, n):
        """pi-weighted mean of the edge similarities [m, K] over the message neighbours of every origin."""
        total = torch.zeros(n, sim.shape[1]).index_add(0, origin, mass[:, None] * sim)
        return total / torch.zeros(n).index_add(0, origin, mass).clamp_min(1e-12)[:, None]

    def _centres(self, message, anchor, mass, origin, bias, tau):
        if not self.route_centering:
            return super()._centres(message, anchor, mass, origin, bias, tau)
        centre = anchor
        for _ in range(self.iters):
            sim = (message * centre[origin]).sum(-1)
            sim = sim - self.origin_mean(sim, mass, origin, len(anchor))[origin]
            probability = torch.softmax(sim / tau + bias, dim=-1)
            update = torch.zeros_like(anchor).index_add(
                0, origin, mass[:, None, None] * probability[:, :, None] * message)
            centre = l2n(anchor + update)
        return centre

    def sample_neighbours(self, oi, di, w, n):
        """At most neighbour_sample draws per origin (uniform, with replacement), pi renormalised over the draws."""
        size = self.neighbour_sample
        degree = torch.bincount(oi, minlength=n)
        start = torch.cumsum(degree, 0) - degree
        big = torch.nonzero(degree > size).squeeze(1)
        generator = None if self.training else torch.Generator().manual_seed(0)
        draws = start[big][:, None] + (torch.rand(len(big), size, generator=generator)
                                       * degree[big][:, None]).long()
        keep = torch.cat([torch.nonzero(degree[oi] <= size).squeeze(1), draws.reshape(-1)])
        oi, di, w = oi[keep], di[keep], w[keep]
        return oi, di, w / torch.zeros(n).index_add(0, oi, w)[oi]

    def route_layers(self, x, share, msg, tau, t=None):
        z = self.embed(x)
        v = l2n(z)
        zo = v if self.inflow is None or self.supply_only else l2n(z + torch.einsum('nf,kfd->nkd', self.inflow, self.V))
        weight = share * msg.float()
        oi, di = torch.nonzero(weight > 0, as_tuple=True)
        w = weight[oi, di]
        if self.neighbour_sample:
            oi, di, w = self.sample_neighbours(oi, di, w, len(x))
        bias = 0.0 if self.etype is None else self.Bt[:, self.etype].permute(1, 0, 2)
        edge_bias = bias if isinstance(bias, float) else bias[oi, :, di]
        shares = []
        for layer in range(self.routing_layers):
            if layer:
                v = l2n(F.dropout(torch.relu(h), self.dropout, self.training))
                zo = v
            wo, wd = self.pair_projections(layer)
            origin = torch.einsum('nkd,kde->nke', zo, wo)
            destination = torch.einsum('nkd,kde->nke', v, wd)
            anchor = l2n(origin + destination)
            message = l2n(origin[oi] + destination[di])
            layer_bias = edge_bias
            if self.supply_difference:
                layer_bias = edge_bias + (x[di] - x[oi]) @ self.diff_u[layer].T
            if self.origin_state:
                state = self.state_terms(x)
                diff = torch.cat([x[di] - x[oi], torch.ones(len(oi), 1)], 1)
                layer_bias = layer_bias + torch.einsum('mo,kod,md->mk', state[oi], self.state_M[layer], diff)
            c = self._centres(message, anchor, w, oi, layer_bias, tau)
            sim = (message * c[oi]).sum(-1)
            if self.route_centering:
                mean = self.origin_mean(sim, w, oi, len(x))
                sim = sim - mean[oi]
            p_edge = torch.softmax(sim / tau + layer_bias, dim=-1)
            shares.append(p_edge)
            h = self.aggregate(layer, v, zo, w[:, None] * p_edge, oi, di)
        relation = l2n(origin[:, None] + destination[None, :])
        sim = torch.einsum('uvkd,ukd->ukv', relation, c)
        if self.route_centering:
            sim = sim - mean[:, :, None]
        if self.supply_difference:
            bias = bias + torch.einsum('uvf,kf->ukv', x[None, :, :] - x[:, None, :], self.diff_u[-1])
        if self.origin_state:
            diff = torch.cat([x[None, :, :] - x[:, None, :], torch.ones(len(x), len(x), 1)], 2)
            bias = bias + torch.einsum('uo,kod,uvd->ukv', self.state_terms(x), self.state_M[-1], diff)
        p = torch.softmax(sim / tau + bias, dim=1)
        return {'z': z, 'zr': v, 'h': h, 'c_route': c, 'p': p, 'edge_shares': shares, 'edge_mass': w}

    def forward(self, x, t, R, cand, share, msg, tau):
        out = self.route_layers(x, share, msg, tau, t)
        hc = torch.relu(out['h'])
        H = F.dropout(hc, self.dropout, self.training).reshape(len(x), -1)
        hidden = self.G_o(H)[:, None, :] + self.G_d(H)[None, :, :] + t[:, :, None] * self.g_t
        score = self.G_out(F.leaky_relu(hidden, 0.2)).squeeze(2)
        out.update(c=hc, score=score, indep=independence(l2n(out['h'])),
                   logpi=torch.log_softmax(score.masked_fill(~(cand & ~msg), -1e9), 1))
        if self.dec:
            out['dec'] = torch.stack([self.dec_loss(q, out['edge_mass']) for q in out['edge_shares']]).mean()
        return out

    @staticmethod
    def dec_loss(q, mass):
        """pi-weighted KL(t || q) with the DEC target t (squared shares over the channel frequencies), t fixed."""
        f = (mass[:, None] * q).sum(0)
        t = q.square() / f
        t = (t / t.sum(1, keepdim=True)).detach()
        kl = (t * (torch.log(t.clamp_min(1e-30)) - torch.log(q.clamp_min(1e-30)))).sum(1)
        return (mass * kl).sum() / mass.sum()


def make_pairgat_model(variant, cfg, n, topology=None):
    options = dict(PAIRGAT_VARIANTS[variant])
    if not options.pop('route_topology', True):
        topology = None
    channels = cfg.get('n_channels', 2)
    return PairGATv2Net(
        d=cfg.get('channel_dim', cfg['hid_dim'] // 2), iters=cfg['iters'], n_channels=channels,
        out_weight='share', attention=False, base=False,
        pressure=np.zeros((n, 6)), etype=topology, n=n,
        proj='linear', routing_layers=cfg['routing_layers'], dropout=cfg['dropout'],
        gravity_mlp=cfg['gravity_mlp'], **options)
