"""DisenGAT routing for line B (线B §2.5–2.6).

Three channels (base, comp, compet), per-channel nine-indicator weights,
edge representation [z_v ; z_v - z_i] compared with citywide label centres by
negative squared Euclidean distance, shared destination score, topology and
impedance terms, node gates (only compet reads inflow), mixture posterior.
PyTorch is used for automatic differentiation.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn

from graph import INT, CRS


class DisenGAT(nn.Module):
    def __init__(self, d=8, norm='none', reconstruct=False, channels=(0, 1, 2), fixed_weights=False,
                 use_impedance=True, free_centres=False, symmetric=False, routing_iters=0):
        """Ablation switches (线B §4): channels kept (B6), equal indicator and topology weights (B6),
        impedance terms off (B9), centres learned freely without labels (B15).
        symmetric: the three channels are exchangeable (each has its own topology and impedance terms,
        each gate reads its embedding and the inflow context, and every channel may route to the self
        loop), used for the unsupervised check.
        routing_iters: rounds of routing feedback; in each round every channel vector absorbs the vectors
        of the destinations the model itself routes to that channel (predicted flow times posterior, self
        loop excluded), and routing is recomputed from the updated vectors. 0 keeps the original model."""
        super().__init__()
        self.symmetric = symmetric
        self.routing_iters = int(routing_iters)
        self.d, self.norm = d, norm
        self.channels = tuple(channels)
        self.use_impedance = use_impedance
        self.free_centres = free_centres
        self.g = nn.Parameter(torch.zeros(3, 9))
        self.W = nn.Parameter(torch.empty(3, 9, d))
        self.b = nn.Parameter(torch.zeros(3, d))
        nn.init.xavier_uniform_(self.W)
        self.w_m = nn.Parameter(torch.zeros(9))
        self.beta = nn.Parameter(torch.zeros(3))         # beta_0, beta_comp, beta_compet before softplus
        self.B0 = nn.Parameter(torch.zeros(3))           # INT, ADJ, CRS
        self.Bk = nn.Parameter(torch.zeros(2, 3))        # comp, compet x topology
        self.log_tau = nn.Parameter(torch.zeros(3))
        self.a_C = nn.Parameter(torch.zeros(d))
        self.c_C = nn.Parameter(torch.zeros(()))
        self.a_D = nn.Parameter(torch.zeros(d + 2))
        self.c_D = nn.Parameter(torch.zeros(()))
        self.decoder = nn.ModuleList(nn.Linear(d, 9) for _ in range(3)) if reconstruct else None
        self.centre_param = nn.Parameter(0.1 * torch.randn(3, 2 * d)) if free_centres else None
        if symmetric:
            self.Bs = nn.Parameter(torch.zeros(3, 3))
            self.beta_s = nn.Parameter(torch.zeros(3))
            self.a_S = nn.Parameter(torch.zeros(3, d + 2))
            self.c_S = nn.Parameter(torch.zeros(3))
        if fixed_weights:
            for p in (self.g, self.B0, self.Bk):
                p.requires_grad_(False)

    def weights(self):
        return 9 * torch.softmax(self.g, 1)

    def embed(self, x):
        omega = self.weights()
        pre = torch.stack([(x * omega[k]) @ self.W[k] + self.b[k] for k in range(3)], 1)
        if self.norm == 'l2':
            return fn.normalize(torch.relu(pre), dim=-1)
        return fn.leaky_relu(pre, 0.2)

    @staticmethod
    def edge_parts(z, k):
        zk = z[:, k]
        return zk[None, :, :], zk[None, :, :] - zk[:, None, :]   # [i, v, d] halves of h_iv

    def centres(self, z, labels):
        out = []
        for k in range(3):
            sel = labels['mech'] == k
            if not sel.any():
                out.append(None)
                continue
            o, v, f = labels['origin'][sel], labels['dest'][sel], labels['flow'][sel]
            h = torch.cat([z[v, k], z[v, k] - z[o, k]], 1)
            out.append((f[:, None] * h).sum(0) / f.sum())
        return out

    def private(self, z, k, centre):
        if centre is None:
            return torch.zeros(z.shape[0], z.shape[0])
        a, b = self.edge_parts(z, k)
        c1, c2 = centre[:self.d], centre[self.d:]
        if self.norm == 'l2':
            return (a * c1).sum(-1) + (b * c2).sum(-1)
        return -(((a - c1) ** 2).sum(-1) + ((b - c2) ** 2).sum(-1)) / (2 * self.d)

    def forward(self, x, t, R, u, candidates, centre_labels):
        z0 = self.embed(x)
        out = self.route(x, z0, t, R, u, candidates, centre_labels)
        z = z0
        if self.routing_iters:
            off = 1.0 - torch.eye(x.shape[0])
        for _ in range(self.routing_iters):
            w = out['logpi'].exp()[:, None, :] * out['posterior'] * off[:, None, :]
            a = w.sum(2)
            z = (1 - a)[:, :, None] * z0 + torch.einsum('ikv,vkd->ikd', w, z)
            out = self.route(x, z, t, R, u, candidates, centre_labels)
        out['z'] = z0
        out['z_routed'] = z
        return out

    def route(self, x, z, t, R, u, candidates, centre_labels):
        n = x.shape[0]
        centres = list(self.centre_param) if self.free_centres else self.centres(z, centre_labels)
        beta = fn.softplus(self.beta) * (1.0 if self.use_impedance else 0.0)
        tau = torch.exp(self.log_tau)
        s0 = (x @ self.w_m)[None, :] - beta[0] * t + self.B0[R]
        if self.symmetric:
            beta_s = fn.softplus(self.beta_s) * (1.0 if self.use_impedance else 0.0)
            scores = [s0 + self.private(z, k, centres[k]) / tau[k] + self.Bs[k][R] - beta_s[k] * t for k in range(3)]
        else:
            scores = [s0 + self.private(z, 0, centres[0]) / tau[0]]
            for j, k in enumerate((1, 2)):
                scores.append(s0 + self.private(z, k, centres[k]) / tau[k] + self.Bk[j][R] - beta[k] * t)
        eye = torch.eye(n, dtype=torch.bool)
        logp = []
        for k, s in enumerate(scores):
            mask = candidates & (~eye if (k and not self.symmetric) else torch.ones_like(eye))
            # A large finite penalty keeps every gradient finite; such entries carry no flow.
            logp.append(torch.log_softmax(s.masked_fill(~mask, -1e9), 1))
        logp = torch.stack(logp, 1)
        if self.symmetric:
            log_omega = torch.stack([torch.cat([z[:, k], u], 1) @ self.a_S[k] + self.c_S[k] for k in range(3)], 1)
        else:
            log_omega = torch.stack([torch.zeros(n), z[:, 1] @ self.a_C + self.c_C,
                                     torch.cat([z[:, 2], u], 1) @ self.a_D + self.c_D], 1)
        absent = torch.tensor([k not in self.channels for k in range(3)])
        log_omega = log_omega.masked_fill(absent[None, :], -1e9)
        log_lambda = torch.log_softmax(log_omega, 1)
        joint = log_lambda[:, :, None] + logp
        logpi = torch.logsumexp(joint, 1)
        post = torch.exp(joint - logpi[:, None, :])
        return {'z': z, 'logpi': logpi, 'logp': logp, 'posterior': post, 'lambda': log_lambda.exp(),
                'centres': centres}

    def decay(self):
        extra = self.a_S.square().sum() if self.symmetric else 0.0
        return self.W.square().sum() + self.a_C.square().sum() + self.a_D.square().sum() + self.w_m.square().sum() + extra

    def topology_weights(self):
        beta = fn.softplus(self.beta).detach().numpy()
        return {'B0': self.B0.detach().numpy().tolist(), 'B_comp': self.Bk[0].detach().numpy().tolist(),
                'B_compet': self.Bk[1].detach().numpy().tolist(), 'beta': beta.tolist(),
                'tau': torch.exp(self.log_tau).detach().numpy().tolist()}


def margin_loss(post, labels, m=0.2):
    """Flow-weighted ranking margin on labelled edges (loss half)."""
    p = post[labels['origin'], :, labels['dest']]
    y = labels['mech']
    py = p.gather(1, y[:, None])
    gap = torch.relu(m - (py - p)).scatter(1, y[:, None], 0.0).sum(1)
    return (labels['flow'] * gap).sum() / labels['flow'].sum()


def split_labels(labels, seed):
    """Random half split of labelled origins: one half defines centres, the other enters the loss."""
    origins = np.unique(labels['origin'])
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(origins)[:len(origins) // 2].tolist())
    a = np.array([o in half for o in labels['origin']])
    pick = lambda mask: {k: (torch.as_tensor(v[mask]) if isinstance(v, np.ndarray) else v) for k, v in labels.items()
                         if isinstance(v, np.ndarray)}
    return pick(a), pick(~a)


def as_tensors(labels):
    return {k: torch.as_tensor(v) for k, v in labels.items() if isinstance(v, np.ndarray)}


def typed(h):
    return {k: (v.long() if k in ('origin', 'dest', 'mech') else v.float()) for k, v in h.items()}


def train(g, labels, seed, epochs=300, lr=0.02, decay=1e-3, margin=0.2, weight=1.0, norm='none',
          reconstruct=0.0, log=None, model_kwargs=None):
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
    model = DisenGAT(norm=norm, reconstruct=reconstruct > 0, **(model_kwargs or {}))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    halves = [typed(h) for h in split_labels(labels, seed)]
    history = []
    for epoch in range(epochs):
        centre, lossset = halves[epoch % 2], halves[1 - epoch % 2]
        out = model(x, t, R, u, cand, centre)
        ce = -(share[rows] * out['logpi'][rows]).sum(1).mean()
        la = margin_loss(out['posterior'], lossset, margin)
        loss = ce + weight * la + decay * model.decay()
        if reconstruct:
            rec = sum(fn.mse_loss(model.decoder[k](out['z'][:, k]), x) for k in range(3)) / 3
            loss = loss + reconstruct * rec
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append({'epoch': epoch + 1, 'ce': float(ce), 'anchor': float(la), 'loss': float(loss)})
        if log and (epoch + 1) % 50 == 0:
            log(f'seed {seed} epoch {epoch + 1} ce {float(ce):.4f} anchor {float(la):.4f}')
    model.eval()
    full = typed(as_tensors(labels))
    with torch.no_grad():
        out = model(x, t, R, u, cand, full)
        ce = float(-(share[rows] * out['logpi'][rows]).sum(1).mean())
    return model, out, history, ce
