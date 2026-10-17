"""OD-pair routing with the reference MechNet aggregation and prediction head.

Only route_layers is overridden. Both models use five routing layers, seven
iterations per layer, two 32-dimensional channels, and the inherited forward.
The pair projections are shared across routing layers and the OD-pair head.
Both variants have identical parameter sets and initial values when that
common prediction head is enabled.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model_v13 import MechNet
from model_v9 import l2n


class ODRelationNet(MechNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'pair_origin'):
            identity = torch.eye(self.d).repeat(self.K, 1, 1)
            self.pair_origin = nn.Parameter(0.5 * identity.clone())
            self.pair_destination = nn.Parameter(0.5 * identity.clone())

    def _centres(self, message, anchor, mass, origin, bias, tau):
        """Sparse evaluation of the same weighted routing update."""
        centre = anchor
        for _ in range(self.iters):
            logits = (message * centre[origin]).sum(-1) / tau + bias
            probability = torch.softmax(logits, dim=-1)
            update = torch.zeros_like(anchor).index_add(
                0, origin, mass[:, None, None] * probability[:, :, None] * message)
            centre = l2n(anchor + update)
        return centre

    def route_layers(self, x, share, msg, tau, t=None):
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
            origin = torch.einsum('nkd,kde->nke', zo, self.pair_origin)
            destination = torch.einsum('nkd,kde->nke', zr, self.pair_destination)
            anchor = l2n(origin + destination)
            message = l2n(origin[oi] + destination[di])
            c = self._centres(message, anchor, weight[oi, di], oi, edge_bias, tau)
        relation = l2n(origin[:, None] + destination[None, :])
        logits = torch.einsum('uvkd,ukd->ukv', relation, c) / tau + bias
        p = torch.softmax(logits, dim=1)
        return {'z': z, 'zr': zr, 'c': c, 'c_route': c, 'p': p}
