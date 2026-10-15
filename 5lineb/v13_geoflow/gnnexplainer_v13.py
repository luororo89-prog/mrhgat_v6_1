"""GNNExplainer (Ying et al. 2019) feature masks on the edge channels of a model_v13 run: which own supply component (TH,
SH, CHC) of the origin and which edge topology the trained model needs to keep the channel of the outflow edges, by
basic configuration type, component change of the origin and channel.

For a trained seed the routing of route_layers (all observed outflow edges as messages, the run's tau, dropout off) gives
the channel of every edge, argmax_k p_AB,k, named by the consensus names (summary.csv channel_order, compare_v13
names). For component j of TH, SH, CHC the origins are grouped by basic type and direction of change of j (as in
explain_channel_change_v13); every group g has a mask m_g = sigmoid(theta_g) over four features of its origin rows:
the three indicators of TH, of SH and of CHC of the origin (x of the origin multiplied by the mask, 0 being the city
mean of the standardised indicators) and the topology bias of its routing (B_t[topology_AB] of the origin's edges
multiplied by the mask). For channel c the masks of all groups are fitted together on the edges whose channel is c,
model objective of GNNExplainer (PyG defaults: 100 epochs, Adam lr 0.01, masks starting at sigmoid(N(0, 0.1)),
size coefficient 1.0 on the mean mask, entropy coefficient 0.1):
       loss = sum_g [ - sum_{(A,B) in g, channel c} F_AB log p_AB,c / sum F_AB ] + 1.0 mean(m) + 0.1 mean(H(m)).
Importance = the fitted mask value x 100 (relative within a cell), mean over the seeds; groups with fewer than
MIN_ORIGINS origins are left empty. Seeds: the first --n_seeds seeds whose edge split correlates with the consensus at
|r| >= 0.8 (the majority mode).
With --by type the origins are grouped by basic type only (one mask per type and channel), and the signed effect of
every feature is added: the change of the trip-weighted competition share of the type's outflow edges (soft routing
shares) when that feature of the type's origins is removed (indicators set to the city mean, topology bias set to 0),
negative when the feature pushes the edges toward competition. Outputs gnnexplainer_type_seed<s>.csv,
gnnexplainer_type_masks.csv and fig_gnnexplainer_type.pdf/.svg/.png.
Outputs in results_lineb/<name>/explain: gnnexplainer_seed<s>.csv, gnnexplainer_masks.csv and
fig_gnnexplainer_<type>.pdf/.svg/.png (layout of explain_channel_change_v13).

python 5lineb/v13_geoflow/gnnexplainer_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --seeds 0 1
python 5lineb/v13_geoflow/gnnexplainer_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --aggregate
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from model_v13 import MechNet, pressure_features, l2n, road_matrix  # noqa: E402
from explain_channel_change_v13 import COMPS, FEATURES, MIN_ORIGINS, plot  # noqa: E402
from plot_type_change_v13 import TYPE_EN  # noqa: E402
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
from graph import build_graph  # noqa: E402
from ruler import standardize, measure  # noqa: E402
from run_v2 import frozen  # noqa: E402

TIER_COLS = {'TH': [0, 1, 2], 'SH': [3, 4, 5], 'CHC': [6, 7, 8]}


def route_masked(m, x, share, msg, tau, scale):
    """route_layers of MechNet with the topology bias of origin u multiplied by scale[u]."""
    z = m.embed(x)
    zr = l2n(z)
    zo = l2n(z + torch.einsum('nf,kfd->nkd', m.inflow, m.V))
    bias = m.Bt[:, m.etype].permute(1, 0, 2) * scale[:, None, None]
    w_edge = share * msg.float()
    for layer in range(m.routing_layers):
        if layer:
            zr = l2n(torch.relu(c))
            zo = zr
        c = zo
        for _ in range(m.iters):
            p = torch.softmax(m.sim(zr, c) / tau + bias, 1)
            c = l2n(zo + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * p, zr))
        p = torch.softmax(m.sim(zr, c) / tau + bias, 1)
    return p


def setup(cfg, ruler):
    g = build_graph(cfg)
    ids = np.asarray(g['ids'])
    T = frozen(ROOT / cfg['ruler_from'])['main']
    U = frozen(ROOT / ruler)['main']['U']
    x = standardize(g['raw'], T)
    run = ROOT / 'results_lineb' / cfg['name']
    com = pd.read_csv(run / 'consensus_community.csv', encoding='utf-8-sig').set_index('community_id').loc[ids]
    dx = com[[f'dx_{c}' for c in g['columns']]].to_numpy(float)
    Sb, Se = x @ U, (x + dx) @ U
    node = pd.DataFrame({'type': [TYPE_EN[T['type_names'][c]] for c in measure(x, T)['class']],
                         **{f'dir_{c}': np.where(Se[:, j] > Sb[:, j], 'rise', 'decline') for j, c in enumerate(COMPS)}})
    return g, x, node, run, [TYPE_EN[t] for t in T['type_names']]


def make_model(cfg, g, n):
    K = cfg['n_channels']
    return MechNet(d=cfg['hid_dim'] // K, iters=cfg['iters'], n_channels=K, out_weight=cfg['out_weight'],
                   pressure=pressure_features(g), proj=cfg['proj'], routing_layers=cfg['routing_layers'],
                   dropout=cfg.get('dropout', 0.0), route_score=cfg.get('route_score', False),
                   route_distance=cfg.get('route_distance', False), rule_topology=cfg.get('rule_topology', False),
                   rule_origin=cfg.get('rule_origin', False), rule_distance=cfg.get('rule_distance', False),
                   etype=g['K'] if cfg.get('edge_topology') else None, n=n)


def explain_seed(cfg, g, x, node, run, seed, by='change'):
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    oi, di = np.nonzero(obs)
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    msg = torch.as_tensor(obs)
    tau = cfg['tau_final']
    m = make_model(cfg, g, n)
    m.load_state_dict(torch.load(run / f'seed{seed}' / 'model.pt'))
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    xt = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        p0 = route_masked(m, xt, share, msg, tau, torch.ones(n))
    order = json.loads(pd.read_csv(run / 'summary.csv').set_index('seed').channel_order.loc[seed])
    names = pd.read_csv(run / 'comparison' / 'mechanism_origin.csv').set_index('channel').name.to_dict()
    raw_name = {order[k]: names[f'C{k + 1}'] for k in range(len(order))}
    ch = p0[oi, :, di].argmax(1).numpy()
    flow = torch.as_tensor(W[obs], dtype=torch.float32)
    rows = []
    torch.manual_seed(seed)
    for comp in (COMPS if by == 'change' else ('ALL',)):
        key = node.type + ' ' + node[f'dir_{comp}'] if by == 'change' else node.type + ' all'
        groups = sorted(key.unique())
        gidx = key.map({k: i for i, k in enumerate(groups)}).to_numpy()
        n_or = key.value_counts()
        for raw, cname in raw_name.items():
            sel = ch == raw
            e_o, e_d, e_w, e_g = torch.as_tensor(oi[sel]), torch.as_tensor(di[sel]), flow[sel], torch.as_tensor(gidx[oi[sel]])
            theta = torch.nn.Parameter(torch.randn(len(groups), 4) * 0.1)
            opt = torch.optim.Adam([theta], lr=0.01)
            G = torch.as_tensor(gidx)
            for _ in range(100):
                mk = torch.sigmoid(theta)
                xm = xt.clone()
                for t_, cols in enumerate(TIER_COLS.values()):
                    xm[:, cols] = xt[:, cols] * mk[G, t_][:, None]
                p = route_masked(m, xm, share, msg, tau, mk[G, 3])
                lp = torch.log(p[e_o, raw, e_d].clamp_min(1e-12))
                num = torch.zeros(len(groups)).index_add(0, e_g, e_w * lp)
                den = torch.zeros(len(groups)).index_add(0, e_g, e_w)
                pred = -(num[den > 0] / den[den > 0]).sum()
                ent = -(mk * torch.log(mk + 1e-15) + (1 - mk) * torch.log(1 - mk + 1e-15))
                loss = pred + 1.0 * mk.mean() + 0.1 * ent.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            mk = torch.sigmoid(theta).detach().numpy()
            for i, k in enumerate(groups):
                tname, d_ = k.rsplit(' ', 1)
                for f, v in zip(FEATURES, mk[i]):
                    lab = f"{'Compet.' if cname == 'competition' else 'Comp.'} {comp} {d_}" if by == 'change' else cname
                    rows.append({'seed': seed, 'type': tname, 'class': lab,
                                 'feature': f, 'importance': 100 * float(v), 'n_origins': int(n_or[k]),
                                 'n_edges': int((gidx[oi[sel]] == i).sum())})
        print(f'seed {seed} {comp} done', flush=True)
    if by == 'type':
        compet_raw = next(r for r, c in raw_name.items() if c == 'competition')
        w = W[obs]
        with torch.no_grad():
            for tname in node.type.unique():
                rows_t = torch.as_tensor((node.type == tname).to_numpy())
                eo = (node.type == tname).to_numpy()[oi]
                base = float((w[eo] * p0[oi, compet_raw, di].numpy()[eo]).sum() / w[eo].sum())
                for f, cols in list(zip(FEATURES[:3], TIER_COLS.values())) + [(FEATURES[3], None)]:
                    xm, sc = xt.clone(), torch.ones(n)
                    if cols is None:
                        sc[rows_t] = 0.0
                    else:
                        xm[rows_t.nonzero()[:, 0][:, None], torch.as_tensor(cols)[None]] = 0.0
                    pr = route_masked(m, xm, share, msg, tau, sc)[oi, compet_raw, di].numpy()
                    rows.append({'seed': seed, 'type': tname, 'class': 'signed', 'feature': f,
                                 'importance': 100 * (float((w[eo] * pr[eo]).sum() / w[eo].sum()) - base),
                                 'n_origins': int((node.type == tname).sum()), 'n_edges': int(eo.sum()),
                                 'competition_share': 100 * base})
    return pd.DataFrame(rows)


def majority_seeds(cfg, g, run, n_seeds):
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    e = pd.read_csv(run / 'consensus_edges.csv')
    w = e.flow.to_numpy(float)
    ref = e.p_C1.to_numpy()
    s = pd.read_csv(run / 'summary.csv')
    out = []
    for r in s.itertuples():
        order = json.loads(r.channel_order)
        tr = run / f'seed{r.seed}' / 'history.csv'
        if not tr.exists():
            continue
        m = make_model(cfg, g, n)
        m.load_state_dict(torch.load(run / f'seed{r.seed}' / 'model.pt'))
        m.eval()
        x = standardize(g['raw'], frozen(ROOT / cfg['ruler_from'])['main'])
        share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
        with torch.no_grad():
            p = route_masked(m, torch.as_tensor(x, dtype=torch.float32), share, torch.as_tensor(obs), cfg['tau_final'],
                             torch.ones(n))[:, order].numpy().transpose(0, 2, 1)[obs][:, 0]
        a, b = p - (w * p).sum() / w.sum(), ref - (w * ref).sum() / w.sum()
        if abs((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum())) >= 0.8:
            out.append(int(r.seed))
        if len(out) == n_seeds:
            break
    return out


def plot_type(out, order):
    """By type: GNNExplainer masks of the compensation and competition edges and the signed removal effect."""
    tab = pd.concat([pd.read_csv(f) for f in sorted(out.glob('gnnexplainer_type_seed*.csv'))])
    agg = tab.groupby(['type', 'class', 'feature'], as_index=False).agg(value=('importance', 'mean'), sd=('importance', 'std'),
                                                                        n_origins=('n_origins', 'first'))
    agg.to_csv(out / 'gnnexplainer_type_masks.csv', index=False)
    cols = [(t, c) for t in order for c in ('compensation', 'competition')]
    M = np.array([[agg.query('type == @t and `class` == @c and feature == @f').value.iloc[0] for t, c in cols] for f in FEATURES])
    Sg = np.array([[agg.query('type == @t and `class` == "signed" and feature == @f').value.iloc[0] for t in order] for f in FEATURES])
    n = {t: int(agg[agg.type == t].n_origins.iloc[0]) for t in order}
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    fig, (bx, sx) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={'width_ratios': [1.6, 1]})
    im = bx.imshow(M, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            bx.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=10, color='white' if M[i, j] > 60 else '#333333')
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels([f'{t}\n{c.capitalize()}\nn={n[t]}' for t, c in cols], fontsize=10)
    bx.set_yticks(range(len(FEATURES)))
    bx.set_yticklabels(FEATURES)
    bx.axhline(2.5, color='white', lw=3)
    for x_ in (1.5, 3.5, 5.5):
        bx.axvline(x_, color='white', lw=3)
    bx.set_title('(a) GNNExplainer mask: needed to keep the channel (%)', fontweight='bold')
    fig.colorbar(im, ax=bx, fraction=0.03, pad=0.02)
    lim = np.nanmax(np.abs(Sg)) or 1
    im2 = sx.imshow(Sg, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(Sg.shape[0]):
        for j in range(Sg.shape[1]):
            sx.text(j, i, f'{Sg[i, j]:+.1f}', ha='center', va='center', fontsize=10)
    sx.set_xticks(range(len(order)))
    sx.set_xticklabels([f'{t}\nn={n[t]}' for t in order], fontsize=10)
    sx.set_yticks(range(len(FEATURES)))
    sx.set_yticklabels(FEATURES)
    sx.axhline(2.5, color='white', lw=3)
    sx.set_title('(b) Removal: change of competition share (pp)', fontweight='bold')
    fig.colorbar(im2, ax=sx, fraction=0.04, pad=0.02, label='< 0: feature pushes toward competition')
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_gnnexplainer_type.{ext}')
    fig.savefig(out / 'fig_gnnexplainer_type.png', dpi=300)
    pd.set_option('display.width', 250)
    print(agg.pivot_table(index='feature', columns=['type', 'class'], values='value').round(1).to_string())
    print('saved to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ruler', required=True)
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--n_seeds', type=int, default=10)
    ap.add_argument('--list_seeds', action='store_true')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--by', choices=('change', 'type'), default='change')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    g, x, node, run, order = setup(cfg, a.ruler)
    out = run / 'explain'
    out.mkdir(exist_ok=True)
    if a.list_seeds:
        print(' '.join(map(str, majority_seeds(cfg, g, run, a.n_seeds))))
        return
    if not a.aggregate:
        for s in a.seeds:
            pre = 'gnnexplainer_seed' if a.by == 'change' else 'gnnexplainer_type_seed'
            explain_seed(cfg, g, x, node, run, s, a.by).to_csv(out / f'{pre}{s}.csv', index=False)
        return
    if a.by == 'type':
        plot_type(out, order)
        return
    tab = pd.concat([pd.read_csv(f) for f in sorted(out.glob('gnnexplainer_seed*.csv'))])
    seeds = sorted(tab.seed.unique())
    agg = tab.groupby(['type', 'class', 'feature'], as_index=False).agg(importance=('importance', 'mean'),
                                                                        importance_sd=('importance', 'std'),
                                                                        n_origins=('n_origins', 'first'),
                                                                        n_edges=('n_edges', 'mean'))
    agg.loc[agg.n_origins < MIN_ORIGINS, 'importance'] = np.nan
    agg.to_csv(out / 'gnnexplainer_masks.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    for tname in order:
        plot(agg[agg.type == tname], f'{tname} (GNNExplainer, {len(seeds)} seeds)', out / f'fig_gnnexplainer_{tname}')
    pd.set_option('display.width', 250)
    print('seeds', seeds)
    print(agg.pivot_table(index=['type', 'feature'], columns='class', values='importance').round(1).to_string())
    print('saved to', out)


if __name__ == '__main__':
    main()
