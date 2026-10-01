"""Aggregate MR-HGAT: supply attributes and directed flow; no capacity/time."""
import torch
from torch import nn
from torch.nn import functional as fn

class MRHGAT(nn.Module):
    def __init__(self, dim=16, int_channel='nested', base_beta=None, base_topo=None,
                 compet_own_supply_gate=False, compet_own_supply_bias=True, compet_outflow=False):
        super().__init__()
        if int_channel not in ('nested', 'independent', 'none'):
            raise ValueError(int_channel)
        self.int_channel = int_channel
        # Toggles for the 2026-09-30 ablation grid: which of compet's three
        # added information channels (own-supply gate term, own-supply
        # destination-attention bias, leave-one-out outflow term) are active.
        # Defaults are the chosen configuration (gate=0,bias=1,outflow=0): best
        # necessary-gate result found (3/4 metrics passed, smallest shortfall
        # on the remaining one) — see reports/D6_model_selection.md.
        self.compet_own_supply_gate = compet_own_supply_gate
        self.compet_own_supply_bias = compet_own_supply_bias
        self.compet_outflow = compet_outflow
        self.projection = nn.ModuleDict({c: nn.Linear(9, dim) for c in ('comp','in','int')})
        self.A_s = nn.Parameter(torch.empty(dim, dim))
        self.A_t = nn.Parameter(torch.empty(dim, dim))
        self.V = nn.Parameter(torch.empty(dim, dim))
        self.U = nn.Parameter(torch.empty(dim, dim))
        self.E_topo = nn.Parameter(torch.empty(3, dim))
        self.a_in = nn.Parameter(torch.empty(dim))
        self.g = nn.Parameter(torch.empty(dim))
        self.bu = nn.Parameter(torch.zeros(dim))
        # Leave-one-out outflow representation: mirrors A_s/A_t/V/U/a_in/g/bu
        # above but for the opposite direction (i's real outgoing flow, excluding
        # whichever destination is currently being scored), feeding comp's
        # destination attention alongside its existing local-supply projection.
        self.A_s_out = nn.Parameter(torch.empty(dim, dim))
        self.A_t_out = nn.Parameter(torch.empty(dim, dim))
        self.V_out = nn.Parameter(torch.empty(dim, dim))
        self.U_out = nn.Parameter(torch.empty(dim, dim))
        self.a_out = nn.Parameter(torch.empty(dim))
        self.g_out = nn.Parameter(torch.empty(dim))
        self.bu_out = nn.Parameter(torch.zeros(dim))
        self.r_comp = nn.Parameter(torch.empty(dim))
        # compet gets a strictly larger information budget than comp (inflow +
        # inflow-source supply already inside hid; plus outflow leave-one-out and
        # own supply added below) — matching that competition's defining condition
        # is a compound function of the graph, not a single local property, while
        # compensation's is; this keeps the two channels structurally asymmetric
        # rather than reintroducing the identical-twin non-identifiability problem.
        self.u2b = nn.Parameter(torch.empty(dim))
        self.r_compet = nn.Parameter(torch.empty(dim))
        self.s_compet = nn.Parameter(torch.empty(dim))
        self.w1 = nn.Parameter(torch.empty(dim))
        self.u2 = nn.Parameter(torch.empty(dim))
        self.w3 = nn.Parameter(torch.empty(dim))
        self.u3 = nn.Parameter(torch.empty(dim))
        self.c1 = nn.Parameter(torch.tensor(-1.))
        self.c2 = nn.Parameter(torch.tensor(0.))
        self.c3 = nn.Parameter(torch.tensor(0.))
        self.cd = nn.Parameter(torch.tensor(-1.))
        self.P = nn.ParameterDict({c: nn.Parameter(torch.empty(dim, dim)) for c in ('comp','compet')})
        self.Q = nn.ParameterDict({c: nn.Parameter(torch.empty(dim, dim)) for c in ('comp','compet')})
        self.a = nn.ParameterDict({c: nn.Parameter(torch.empty(dim)) for c in ('comp','compet')})
        self.e = nn.ParameterDict({c: nn.Parameter(torch.empty(dim)) for c in ('comp','compet')})
        self.B = nn.ParameterDict({c: nn.Parameter(torch.zeros(3)) for c in ('comp','compet')})
        self.base_beta = nn.Parameter(torch.zeros(9))
        self.base_topo = nn.Parameter(torch.zeros(3))
        for name, p in self.named_parameters():
            if p.ndim == 2:
                nn.init.xavier_uniform_(p)
            elif name in ('a_in','g','w1','u2','w3','u3','a_out','g_out','r_comp',
                          'u2b','r_compet','s_compet') or name.startswith(('a.', 'e.')):
                nn.init.normal_(p, std=.1)
            elif name.startswith('projection.') and name.endswith('bias'):
                nn.init.zeros_(p)
        nn.init.normal_(self.E_topo, std=.1)
        with torch.no_grad():
            if base_beta is not None:
                self.base_beta.copy_(torch.as_tensor(base_beta))
            if base_topo is not None:
                self.base_topo.copy_(torch.as_tensor(base_topo))

    def encode(self, x, K):
        return {c: fn.normalize(fn.relu(layer(x)), dim=1) for c, layer in self.projection.items()}

    def forward(self, x, F, K):
        z = self.encode(x, K)
        self_mask = K == 0
        score = fn.leaky_relu((z['in'] @ self.A_s)[:,None,:] +
            (z['in'] @ self.A_t)[None,:,:] + self.E_topo[K], .2) @ self.a_in
        alpha = torch.sigmoid(score).masked_fill(self_mask, 0)
        # One citywide numerical scale; never a destination-specific capacity.
        scale = F.masked_fill(self_mask, 0).sum(0).mean().clamp_min(1e-8)
        w = alpha * F / scale
        sigma = w.sum(0)
        h_in = (w.T @ (z['in'] @ self.V)) / sigma[:,None].clamp_min(1e-8)
        hid = fn.leaky_relu(h_in @ self.U + sigma.clamp_min(1e-8).log()[:,None] * self.g + self.bu, .2)
        # Symmetric outflow counterpart to hid, but leave-one-out per edge: for
        # scoring edge (i,j), aggregate i's real outflow to every OTHER destination
        # (excluding j itself), so this cannot just reproduce the F[i,:] row being
        # predicted. Built via a running-sum-minus-one-term trick, not by looping.
        score_out = fn.leaky_relu((z['in'] @ self.A_s_out)[:,None,:] +
            (z['in'] @ self.A_t_out)[None,:,:] + self.E_topo[K], .2) @ self.a_out
        alpha_out = torch.sigmoid(score_out).masked_fill(self_mask, 0)
        w_out = alpha_out * F / scale
        value_out = z['in'] @ self.V_out
        raw_out = w_out @ value_out
        sigma_out = w_out.sum(1)
        sigma_out_loo = (sigma_out[:,None] - w_out).clamp_min(1e-8)
        h_out_loo = (raw_out[:,None,:] - w_out[:,:,None] * value_out[None,:,:]) / sigma_out_loo[:,:,None]
        hid_out_loo = fn.leaky_relu(h_out_loo @ self.U_out +
            sigma_out_loo.clamp_min(1e-8).log()[:,:,None] * self.g_out + self.bu_out, .2)
        comp = z['comp'] @ self.w1 + self.c1
        compet = hid @ self.u2 + self.c2
        if self.compet_own_supply_gate:
            compet = compet + z['comp'] @ self.u2b
        base = torch.zeros_like(comp)
        if self.int_channel == 'none':
            log_rho = torch.log_softmax(torch.stack([base, comp, compet], 1), 1)
            pressed = log_rho[:,2].exp()
        elif self.int_channel == 'independent':
            internal = z['int'] @ self.w3 + hid @ self.u3 + self.c3
            log_rho = torch.log_softmax(torch.stack([base, comp, compet, internal], 1), 1)
            pressed = log_rho[:,2].exp()
        else:
            intent = z['int'] @ self.w3 + self.c3
            lr = torch.log_softmax(torch.stack([base, comp, intent], 1), 1)
            displaced = hid @ self.u3 + self.cd
            log_rho = torch.stack([lr[:,0], lr[:,1], lr[:,2] + fn.logsigmoid(displaced),
                lr[:,2] + fn.logsigmoid(-displaced)], 1)
            pressed = displaced.sigmoid()
        base_logits = (x @ self.base_beta)[None,:] + self.base_topo[K]
        if self.int_channel != 'none':
            base_logits = base_logits.masked_fill(self_mask, -torch.inf)
        logp = [torch.log_softmax(base_logits, 1)]
        # Reverted 2026-09-30: using z['comp'] (own supply) as compet's query
        # tested worse and less stable than hid (inflow) when combined with the
        # chosen gate=0/bias=1/outflow=0 toggles (see reports/D6_model_selection.md).
        # compet's query is hid (inflow context); comp's query is its own local
        # supply. Both channels' key is destination supply (candidate j's own
        # z['comp'][j]) — 方案 234-235 行.
        destination_query = {'comp': z['comp'], 'compet': hid}
        destination_key = {'comp': z['comp'], 'compet': z['comp']}
        for c in ('comp','compet'):
            zq, zk = destination_query[c], destination_key[c]
            logits = fn.leaky_relu((zq @ self.P[c])[:,None,:] +
                (zk @ self.Q[c])[None,:,:] + self.e[c], .2) @ self.a[c] + self.B[c][K]
            if c == 'comp':
                logits = logits + hid_out_loo @ self.r_comp
            else:
                if self.compet_outflow:
                    logits = logits + hid_out_loo @ self.r_compet
                if self.compet_own_supply_bias:
                    logits = logits + (z['comp'] @ self.s_compet)[:,None]
            logp.append(torch.log_softmax(logits.masked_fill(self_mask, -torch.inf), 1))
        if self.int_channel != 'none':
            logp.append(torch.zeros_like(F).masked_fill(~self_mask, -torch.inf))
        logp = torch.stack(logp, 1)
        logjoint = log_rho[:,:,None] + logp
        logpi = torch.logsumexp(logjoint, 1)
        posterior = (logjoint - logpi[:,None,:]).exp()
        return {'logpi': logpi, 'pi': logpi.exp(), 'rho': log_rho.exp(), 'logp': logp,
            'r': posterior, 'alpha': alpha, 'pressed': pressed, 'sigma': sigma,
            'factor_embeddings': torch.stack([z['comp'], hid], 1)}

    def regularization(self):
        return sum(p.square().sum() for p in self.parameters() if p.ndim == 2)

class SeparatedMRHGAT(MRHGAT):
    """MD forward model plus FactorGCN-style auxiliary factor-graph classifier.

    The shared classifier predicts anonymous factor IDs, never mechanism labels.
    Original gates and destination probabilities are inherited unchanged.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor_layers=nn.ModuleList([nn.Linear(9,16),nn.Linear(16,8),nn.Linear(8,16)])
        self.factor_classifier=nn.Linear(32,2)

    def forward(self, x, F, K, loss_rows=None):
        out=super().forward(x,F,K)
        rows=(F.sum(1)>0) if loss_rows is None else loss_rows
        edges=((F>0)&(K!=0)&rows[:,None]).to(x.dtype)
        # Graph degree normalization, not facility capacity correction.
        norm=(edges.sum(1)[:,None]*edges.sum(0)[None,:]).sqrt().clamp_min(1)
        reps=[]
        for channel in (1,2):
            adjacency=edges*out['r'][:,channel,:]/norm
            outgoing=incoming=x
            for layer in self.factor_layers:
                outgoing=torch.tanh(adjacency@layer(outgoing))
                incoming=torch.tanh(adjacency.T@layer(incoming))
            reps.append(torch.cat([outgoing.mean(0),incoming.mean(0)]))
        logits=self.factor_classifier(torch.stack(reps))
        labels=torch.arange(2,device=x.device)
        out['factor_discrimination']=fn.cross_entropy(logits,labels)
        return out

def cross_entropy(logpi, share, rows):
    if not rows.any():
        raise ValueError('No loss rows')
    return -(share[rows] * logpi[rows]).sum(1).mean()

class ChoiceBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(9))
        self.topo = nn.Parameter(torch.zeros(3))

    def forward(self, x, K):
        return torch.log_softmax((x @ self.beta)[None,:] + self.topo[K], 1)
