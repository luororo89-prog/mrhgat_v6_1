"""Mixture of K = 3 outflow mechanisms with feedback (model_v7).

Each observed non-self outflow u -> v is classified into K mechanisms.
1  Node descriptor per mechanism, for the outflow u -> v being classified:
       h_u,k^-v = [x_u, a_u,k^in, a_u,k^out,-v]
   x_u: nine standardized supply indicators of the origin.
   a_u,k^in: inflow summary of u weighted by w_iu p_iu,k: log1p(volume), weighted mean of the origins'
             indicators, weighted mean distance (the reverse flow v -> u is kept), plus log1p(self-loop trips)
             of u (standardized citywide, not split by mechanism).
   a_u,k^out,-v: outflow summary of u weighted by w_uj p_uj,k over j != v (the edge u -> v itself removed):
             log1p(volume), weighted mean of the destinations' indicators, weighted mean distance.
2  Origin and destination projections (separate parameters):
       s_u,k = normalize(LeakyReLU(W_k^O h_u,k^-v + b_k^O)),  t_v,k = normalize(LeakyReLU(W_k^D x_v + b_k^D))
3  Mechanism prior of the origin: pi_u,k = softmax_k(MLP([h_u,1^-v, h_u,2^-v, h_u,3^-v]))
4  Destination choice within mechanism k (softmax over candidate destinations, self excluded):
       f_k(v | u) = softmax_v(s_u,k^T M_k t_v,k - beta_k d_uv + b_k[topology_uv])
5  Classification (E step): p_uv,k = pi_u,k f_k(v|u) / sum_k' pi_u,k' f_k'(v|u)
6  Training (M step): maximize sum_uv w_uv log sum_k pi_u,k f_k(v|u) / sum_uv w_uv.
Feedback: round 0 uses p = 1/3 (summaries weighted by w only). Each later round recomputes the summaries
with the previous p, trains from the previous parameters with learning rate lr * lr_decay^round (so that the
mini-batch noise of the parameters vanishes), and updates p <- lam p_new + (1 - lam) p_old. lam starts at 0.2 and
is halved (down to 0.025) whenever two successive steps p_new - p_old point in opposite directions (correlation
< -0.5), which marks an oscillating feedback. Stop when
for two consecutive rounds: flow-weighted mean |dp| < 1e-3, flow-weighted share of edges whose dominant
mechanism changes < 0.5 %, relative change of the log likelihood < 1e-4.
Importance (after convergence, parameters and p frozen): one feature group (own x_u, inflow a^in, outflow
a^out) is permuted across origin nodes, the other inputs are kept, one E step without feedback is run.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn

GROUPS = ('own', 'inflow', 'outflow')
N_AGG = 11
l2n = lambda v: fn.normalize(v, dim=-1)


class FlowData:
    """Observed non-self outflow edges and the summaries that feed the descriptors."""

    def __init__(self, g):
        W = g['W']
        n = len(W)
        obs = (W > 0) & ~np.eye(n, dtype=bool)
        self.n = n
        self.eu, self.ev = np.nonzero(obs)
        self.w = W[self.eu, self.ev].astype(float)
        self.E = len(self.w)
        self.eid = -np.ones((n, n), dtype=np.int64)
        self.eid[self.eu, self.ev] = np.arange(self.E)
        self.x = np.asarray(g['x'], dtype=float)
        sl = np.log1p(np.diag(W).astype(float))
        self.self_loop = (sl - sl.mean()) / sl.std()
        t = np.asarray(g['t'], dtype=float)
        te = t[self.eu, self.ev][:, None]
        self.phi_out = np.hstack([self.x[self.ev], te])
        self.phi_in = np.hstack([self.x[self.eu], te])
        self.nodes = np.unique(self.eu)
        self.km = np.asarray(g['distance_km'])[self.eu, self.ev]
        self.scale = None
        self.xt = torch.as_tensor(self.x, dtype=torch.float32)
        K = np.asarray(g['K'])
        self.edge_terms = torch.as_tensor(np.stack([t, K == 1, K == 2], -1), dtype=torch.float32)
        self.mask = torch.as_tensor(np.asarray(g['candidates']) & ~np.eye(n, dtype=bool))
        self.eut = torch.as_tensor(self.eu)
        self.evt = torch.as_tensor(self.ev)
        self.wt = torch.as_tensor(self.w, dtype=torch.float32)

    def _sums(self, p):
        wp = self.w[:, None] * p
        N_out, N_in = np.zeros((self.n, 3)), np.zeros((self.n, 3))
        S_out, S_in = np.zeros((self.n, 3, N_AGG - 1)), np.zeros((self.n, 3, N_AGG - 1))
        np.add.at(N_out, self.eu, wp)
        np.add.at(N_in, self.ev, wp)
        np.add.at(S_out, self.eu, wp[:, :, None] * self.phi_out[:, None, :])
        np.add.at(S_in, self.ev, wp[:, :, None] * self.phi_in[:, None, :])
        return wp, N_out, S_out, N_in, S_in

    @staticmethod
    def _summary(N, S):
        ok = N > 1e-9
        mean = np.where(ok[..., None], S / np.where(ok, N, 1.0)[..., None], 0.0)
        return np.concatenate([np.log1p(np.maximum(N, 0.0))[..., None], mean], -1), ok

    def fit_scale(self, p):
        _, N_out, S_out, N_in, S_in = self._sums(p)
        a_out, ok_out = self._summary(N_out, S_out)
        a_in, ok_in = self._summary(N_in, S_in)
        self.scale = {'out': (a_out[ok_out].mean(0), a_out[ok_out].std(0) + 1e-9),
                      'in': (a_in[ok_in].mean(0), a_in[ok_in].std(0) + 1e-9)}

    def features(self, p, perm=None):
        """[E, 3, 9 + 12 + 11] descriptors h_u,k^-v; perm = (group, sigma) replaces that group of origin u by node sigma[u]."""
        wp, N_out, S_out, N_in, S_in = self._sums(p)
        so = si = sx = self.eu
        if perm is not None:
            group, sigma = perm
            if group == 'own':
                sx = sigma[self.eu]
            elif group == 'inflow':
                si = sigma[self.eu]
            else:
                so = sigma[self.eu]
        a_in, ok_in = self._summary(N_in[si], S_in[si])
        a_in = (a_in - self.scale['in'][0]) / self.scale['in'][1]
        a_in[..., 1:] *= ok_in[..., None]
        a_in = np.concatenate([a_in, np.broadcast_to(self.self_loop[si][:, None, None], (self.E, 3, 1))], -1)
        e2 = self.eid[so, self.ev]
        has = (e2 >= 0)[:, None]
        e2c = np.where(e2 >= 0, e2, 0)
        Nm = N_out[so] - has * wp[e2c]
        Sm = S_out[so] - has[..., None] * wp[e2c][:, :, None] * self.phi_out[e2c][:, None, :]
        a_out, ok_out = self._summary(Nm, Sm)
        a_out = (a_out - self.scale['out'][0]) / self.scale['out'][1]
        a_out[..., 1:] *= ok_out[..., None]
        own = np.broadcast_to(self.x[sx][:, None, :], (self.E, 3, self.x.shape[1]))
        return torch.as_tensor(np.concatenate([own, a_in, a_out], -1), dtype=torch.float32)


class FlowMixture(nn.Module):
    def __init__(self, n_x=9, d=8, hidden=16):
        super().__init__()
        n_h = n_x + 2 * N_AGG + 1
        self.W_O = nn.Parameter(torch.empty(3, n_h, d))
        self.W_D = nn.Parameter(torch.empty(3, n_x, d))
        self.M = nn.Parameter(torch.empty(3, d, d))
        for prm in (self.W_O, self.W_D, self.M):
            for k in range(3):
                nn.init.xavier_uniform_(prm.data[k])
        self.b_O = nn.Parameter(torch.zeros(3, d))
        self.b_D = nn.Parameter(torch.zeros(3, d))
        self.beta = nn.Parameter(torch.zeros(3))
        self.b_R = nn.Parameter(torch.zeros(3, 3))
        self.prior = nn.Sequential(nn.Linear(3 * n_h, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 3))

    def forward(self, h, u, v, data):
        """h [B, 3, n_h] descriptors of edges u -> v; returns log likelihood per edge, posterior p, log prior."""
        s = l2n(fn.leaky_relu(torch.einsum('bkf,kfd->bkd', h, self.W_O) + self.b_O, 0.2))
        td = l2n(fn.leaky_relu(torch.einsum('nf,kfd->nkd', data.xt, self.W_D) + self.b_D, 0.2))
        q = torch.einsum('bkd,kde->bke', s, self.M)
        coef = torch.stack([-fn.softplus(self.beta), self.b_R[:, 1], self.b_R[:, 2]], 1)
        sc = torch.einsum('bkd,nkd->bkn', q, td) + torch.einsum('bnj,kj->bkn', data.edge_terms[u], coef)
        sc = sc.masked_fill(~data.mask[u][:, None, :], -1e9)
        logf = sc.gather(2, v[:, None, None].expand(-1, 3, 1)).squeeze(2) - torch.logsumexp(sc, 2)
        logpi = torch.log_softmax(self.prior(h.reshape(len(h), -1)), 1)
        joint = logpi + logf
        return torch.logsumexp(joint, 1), torch.softmax(joint, 1), logpi

    def decay(self):
        return self.W_O.square().sum() + self.W_D.square().sum() + self.prior[0].weight.square().sum()


def e_step(model, data, feats, batch=8192):
    ll, post = torch.empty(data.E), torch.empty(data.E, 3)
    with torch.no_grad():
        for i in range(0, data.E, batch):
            sl = slice(i, i + batch)
            ll[sl], post[sl], _ = model(feats[sl], data.eut[sl], data.evt[sl], data)
    return float((data.wt * ll).sum() / data.wt.sum()), post.numpy().astype(float)


def m_step(model, opt, data, feats, epochs, decay, gen, batch=8192):
    for _ in range(epochs):
        for idx in torch.randperm(data.E, generator=gen).split(batch):
            ll, _, _ = model(feats[idx], data.eut[idx], data.evt[idx], data)
            w = data.wt[idx]
            loss = -(w * ll).sum() / w.sum() + decay * model.decay()
            opt.zero_grad()
            loss.backward()
            opt.step()


def fit_v7(g, seed, d=8, hidden=16, lr=0.01, lr_decay=0.8, decay=1e-3, epochs_init=60, epochs_round=15, damping=0.2,
           damping_min=0.025, osc_corr=-0.5, tol_dp=1e-3, tol_switch=5e-3, tol_ll=1e-4, patience=2, max_rounds=30, batch=8192, log=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    data = FlowData(g)
    model = FlowMixture(n_x=data.x.shape[1], d=d, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    p = np.full((data.E, 3), 1 / 3)
    data.fit_scale(p)
    m_step(model, opt, data, data.features(p), epochs_init, decay, gen, batch)
    ll_prev, p = e_step(model, data, data.features(p), batch)
    hist = [{'round': 0, 'll': ll_prev, 'mean_dp': np.nan, 'switch': np.nan, 'rel_dll': np.nan, 'residual': np.nan,
             'step_corr': np.nan, 'damping': np.nan,
             **{f'share_{k}': float((data.w * p[:, k]).sum() / data.w.sum()) for k in range(3)}}]
    if log:
        log(f'seed {seed} round 0 ll {ll_prev:.5f} shares {np.round([hist[0][f"share_{k}"] for k in range(3)], 3).tolist()}')
    streak, converged, lam, prev_step = 0, False, damping, None
    for r in range(1, max_rounds + 1):
        feats = data.features(p)
        for grp in opt.param_groups:
            grp['lr'] = lr * lr_decay ** r
        m_step(model, opt, data, feats, epochs_round, decay, gen, batch)
        ll, p_new = e_step(model, data, feats, batch)
        step = p_new - p
        corr = np.nan if prev_step is None else float(np.corrcoef(step.ravel(), prev_step.ravel())[0, 1])
        if corr < osc_corr:
            lam = max(lam / 2, damping_min)
        prev_step = step
        p_next = p + lam * step
        residual = float((data.w * np.abs(step).mean(1)).sum() / data.w.sum())
        mean_dp = float((data.w * np.abs(p_next - p).mean(1)).sum() / data.w.sum())
        switch = float((data.w * (p_next.argmax(1) != p.argmax(1))).sum() / data.w.sum())
        rel = abs(ll - ll_prev) / abs(ll_prev)
        streak = streak + 1 if (mean_dp < tol_dp and switch < tol_switch and rel < tol_ll) else 0
        p, ll_prev = p_next, ll
        hist.append({'round': r, 'll': ll, 'mean_dp': mean_dp, 'switch': switch, 'rel_dll': rel, 'residual': residual,
                     'step_corr': corr, 'damping': lam,
                     **{f'share_{k}': float((data.w * p[:, k]).sum() / data.w.sum()) for k in range(3)}})
        if log:
            log(f'seed {seed} round {r} ll {ll:.5f} mean|dp| {mean_dp:.2e} switch {switch:.4f} rel_dll {rel:.2e} '
                f'step_corr {corr:.3f} damping {lam:g}')
        if streak >= patience:
            converged = True
            break
    model.eval()
    return model, data, p, hist, converged


def permutation_importance(model, data, p, reps=30, seed=0, batch=8192):
    """Permute each feature group across origin nodes; returns base E step and one row per group and repetition.

    dll = LL* - LL_perm; dP_k = flow-weighted mean over edges whose dominant mechanism is k of (p*_k - p_perm,k);
    switch = flow-weighted share of edges whose dominant mechanism changes."""
    rng = np.random.default_rng(seed)
    ll0, p0 = e_step(model, data, data.features(p), batch)
    dom = p0.argmax(1)
    rows = []
    for group in GROUPS:
        for rep in range(reps):
            sigma = np.arange(data.n)
            sigma[data.nodes] = rng.permutation(data.nodes)
            ll1, p1 = e_step(model, data, data.features(p, (group, sigma)), batch)
            row = {'group': group, 'rep': rep, 'dll': ll0 - ll1,
                   'switch': float((data.w * (p1.argmax(1) != dom)).sum() / data.w.sum())}
            for k in range(3):
                m = dom == k
                row[f'dP_{k}'] = float((data.w[m] * (p0[m, k] - p1[m, k])).sum() / max(data.w[m].sum(), 1e-12))
            rows.append(row)
    return ll0, p0, rows
