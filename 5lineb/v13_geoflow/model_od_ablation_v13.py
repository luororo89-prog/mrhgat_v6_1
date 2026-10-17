"""Independent pair projections for the existing OD-pair routing model.

fix1: independent pair projections in routing layers 2--5. Layer 1 and the
prediction head retain the existing pair projections. Every layer starts
from the same half-identity initialization; routing within a layer is intact.
The OD-pair prediction head, pressure terms and attention are inherited
unchanged. No initial residual or extra objective is introduced.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model_od_relation_v13 import ODRelationNet
from model_v9 import l2n


VARIANTS = {
    'original': False,
    'fix1': True,
}


class ODAblationNet(ODRelationNet):
    def __init__(self, *args, independent_layers=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.independent_layers = independent_layers
        if independent_layers:
            self.layer_pair_origin = nn.ParameterList([
                nn.Parameter(self.pair_origin.detach().clone())
                for _ in range(self.routing_layers - 1)])
            self.layer_pair_destination = nn.ParameterList([
                nn.Parameter(self.pair_destination.detach().clone())
                for _ in range(self.routing_layers - 1)])

    def pair_projections(self, layer):
        if self.independent_layers and layer > 0:
            return self.layer_pair_origin[layer - 1], self.layer_pair_destination[layer - 1]
        return self.pair_origin, self.pair_destination

    def route_layers(self, x, share, msg, tau, t=None):
        if not self.independent_layers:
            return super().route_layers(x, share, msg, tau, t)
        z = self.embed(x)
        zr = l2n(z)
        zo = zr if self.inflow is None else l2n(
            z + torch.einsum('nf,kfd->nkd', self.inflow, self.V))
        weight = share * msg.float()
        oi, di = torch.nonzero(weight > 0, as_tuple=True)
        bias = 0.0 if self.etype is None else self.Bt[:, self.etype].permute(1, 0, 2)
        if self.route_distance:
            bias = bias + self.rd[None, :, None] * t[:, None, :]
        edge_bias = bias if isinstance(bias, float) else bias[oi, :, di]
        for layer in range(self.routing_layers):
            if layer:
                zr = l2n(F.dropout(torch.relu(c), self.dropout, self.training))
                zo = zr
            wo, wd = self.pair_projections(layer)
            origin = torch.einsum('nkd,kde->nke', zo, wo)
            destination = torch.einsum('nkd,kde->nke', zr, wd)
            anchor = l2n(origin + destination)
            message = l2n(origin[oi] + destination[di])
            c = self._centres(message, anchor, weight[oi, di], oi, edge_bias, tau)
        relation = l2n(origin[:, None] + destination[None, :])
        logits = torch.einsum('uvkd,ukd->ukv', relation, c) / tau + bias
        p = torch.softmax(logits, dim=1)
        return {'z': z, 'zr': zr, 'c': c, 'c_route': c, 'p': p}

def make_ablation_model(variant, cfg, n, topology=None):
    import numpy as np
    independent = VARIANTS[variant]
    return ODAblationNet(
        d=cfg['hid_dim'] // 2, iters=cfg['iters'], n_channels=2,
        out_weight='share', attention=cfg['attention'], base=False,
        pressure=np.zeros((n, 6)), etype=topology, n=n,
        proj='linear', routing_layers=cfg['routing_layers'], dropout=cfg['dropout'],
        route_score=True, rule_origin=True, od_pair_head=True,
        independent_layers=independent)
