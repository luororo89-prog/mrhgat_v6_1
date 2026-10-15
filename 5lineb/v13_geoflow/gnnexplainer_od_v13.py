"""GNNExplainer (Ying et al. 2019) on the edge channels of a model_v13 run with origin and destination roles: which
supply component (TH, SH, CHC) of the origin, which of the destination and which edge topology (ADJ, CRS) the trained
model needs to put the outflow edges of a basic configuration type into compensation or into competition.

Roles are kept apart with two routing tracks of route_layers (all observed outflow edges as messages, the run's tau,
dropout off): the destination track routes every community with its destination mask applied (indicators of a tier
multiplied by the mask, 0 being the city mean of the standardised indicators) and gives the destination vectors of
every layer; the origin track routes the communities of the type with their own mask applied to their own indicators,
against the destination vectors of the destination track, with the topology bias B_t of their ADJ and CRS edges
multiplied by the topology masks. Without masks both tracks reproduce the model's routing. Internal trips (INT) are
not assigned to a channel and have no mask.
For a type and channel c one mask m = sigmoid(theta) over Own TH, Own SH, Own CHC, Dest TH, Dest SH, Dest CHC, ADJ, CRS
is fitted on the type's edges whose channel is c (GNNExplainer model objective, PyG defaults: 100 epochs, Adam lr
0.01, theta ~ N(0, 0.1), size coefficient 1.0, entropy coefficient 0.1):
       loss = - sum F_AB log p_AB,c / sum F_AB + 1.0 mean(m) + 0.1 mean(H(m)).
Signed effect: change of the trip-weighted competition share (soft routing shares, percentage points) of the type's
edges when one feature is removed (its mask set to 0, the others 1); negative when the feature pushes toward
competition. Mean over the majority seeds of gnnexplainer_v13 (--seeds), channel names as in gnnexplainer_v13.
Outputs in results_lineb/<name>/explain: gnnexplainer_od_seed<s>.csv, gnnexplainer_od.csv and
fig_gnnexplainer_od.pdf/.svg/.png ((a) Cleveland dot plot and (b) heatmap of the masks, (c) signed effects).

python 5lineb/v13_geoflow/gnnexplainer_od_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --seeds 0 1
python 5lineb/v13_geoflow/gnnexplainer_od_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --aggregate
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
sys.path.insert(0, str(HERE))
from model_v13 import l2n  # noqa: E402
from gnnexplainer_v13 import setup, make_model, TIER_COLS  # noqa: E402

FEATS = ['Own TH', 'Own SH', 'Own CHC', 'Dest TH', 'Dest SH', 'Dest CHC', 'ADJ', 'CRS']
TYPE_COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}
ADJ, CRS = 1, 2


def mask_x(x, mk, rows=None):
    """Indicators of each tier multiplied by the tier mask (rows None: every community)."""
    out = x.clone()
    for t_, cols in enumerate(TIER_COLS.values()):
        if rows is None:
            out[:, cols] = x[:, cols] * mk[t_]
        else:
            out[rows[:, None], torch.as_tensor(cols)[None]] = x[rows][:, cols] * mk[t_]
    return out


def routing(m, xo, xd, share, msg, tau, rows, s_adj, s_crs):
    """Origin track of the type's communities (own x xo, topology scales) against the destination track (x xd)."""
    w_edge = share * msg.float()
    bias = m.Bt[:, m.etype].permute(1, 0, 2)
    zd_layers = []
    z = m.embed(xd)
    zr = l2n(z)
    zo = l2n(z + torch.einsum('nf,kfd->nkd', m.inflow, m.V))
    for layer in range(m.routing_layers):
        if layer:
            zr = l2n(torch.relu(c))
            zo = zr
        zd_layers.append(zr)
        c = zo
        for _ in range(m.iters):
            p = torch.softmax(m.sim(zr, c) / tau + bias, 1)
            c = l2n(zo + torch.einsum('ukv,vkd->ukd', w_edge[:, None, :] * p, zr))
    et = m.etype[rows]
    scale = torch.ones(et.shape)
    scale = torch.where(et == ADJ, s_adj, scale)
    scale = torch.where(et == CRS, s_crs, scale)
    bias_o = bias[rows] * scale[:, None, :]
    zo_all = m.embed(xo)
    zo = l2n(zo_all + torch.einsum('nf,kfd->nkd', m.inflow, m.V))[rows]
    we = w_edge[rows]
    for layer in range(m.routing_layers):
        if layer:
            zo = l2n(torch.relu(c))
        Z = zd_layers[layer]
        c = zo
        for _ in range(m.iters):
            p = torch.softmax(m.sim(Z, c) / tau + bias_o, 1)
            c = l2n(zo + torch.einsum('ukv,vkd->ukd', we[:, None, :] * p, Z))
        p = torch.softmax(m.sim(Z, c) / tau + bias_o, 1)
    return p


def apply(m, xt, share, msg, tau, rows, mk):
    return routing(m, mask_x(xt, mk[:3], rows), mask_x(xt, mk[3:6]), share, msg, tau, rows, mk[6], mk[7])


def explain_seed(cfg, g, x, node, run, seed):
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    msg = torch.as_tensor(obs)
    tau = cfg['tau_final']
    m = make_model(cfg, g, n)
    m.load_state_dict(torch.load(run / f'seed{seed}' / 'model.pt'))
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    xt = torch.as_tensor(x, dtype=torch.float32)
    order = json.loads(pd.read_csv(run / 'summary.csv').set_index('seed').channel_order.loc[seed])
    names = pd.read_csv(run / 'comparison' / 'mechanism_origin.csv').set_index('channel').name.to_dict()
    raw_name = {order[k]: names[f'C{k + 1}'] for k in range(len(order))}
    compet = next(r for r, c in raw_name.items() if c == 'competition')
    torch.manual_seed(seed)
    rows_out = []
    for tname in node.type.unique():
        rows = torch.as_tensor(np.nonzero((node.type == tname).to_numpy())[0])
        Wt = torch.as_tensor(W[rows.numpy()], dtype=torch.float32)
        E = torch.as_tensor(obs[rows.numpy()])
        with torch.no_grad():
            p0 = apply(m, xt, share, msg, tau, rows, torch.ones(8))
        ch = p0.argmax(1)
        base = float((Wt * p0[:, compet] * E).sum() / (Wt * E).sum())
        for raw, cname in raw_name.items():
            sel = (E & (ch == raw)).float() * Wt
            theta = torch.nn.Parameter(torch.randn(8) * 0.1)
            opt = torch.optim.Adam([theta], lr=0.01)
            for _ in range(100):
                mk = torch.sigmoid(theta)
                p = apply(m, xt, share, msg, tau, rows, mk)
                pred = -(sel * torch.log(p[:, raw].clamp_min(1e-12))).sum() / sel.sum()
                ent = -(mk * torch.log(mk + 1e-15) + (1 - mk) * torch.log(1 - mk + 1e-15))
                loss = pred + 1.0 * mk.mean() + 0.1 * ent.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            for f, v in zip(FEATS, torch.sigmoid(theta).detach().numpy()):
                rows_out.append({'seed': seed, 'type': tname, 'what': cname, 'feature': f, 'value': 100 * float(v),
                                 'n_origins': len(rows), 'n_edges': int((sel > 0).sum())})
        with torch.no_grad():
            for j, f in enumerate(FEATS):
                mk = torch.ones(8)
                mk[j] = 0.0
                p = apply(m, xt, share, msg, tau, rows, mk)
                eff = float((Wt * p[:, compet] * E).sum() / (Wt * E).sum()) - base
                rows_out.append({'seed': seed, 'type': tname, 'what': 'signed', 'feature': f, 'value': 100 * eff,
                                 'n_origins': len(rows), 'n_edges': int(E.sum()), 'competition_share': 100 * base})
        print(f'seed {seed} {tname} done', flush=True)
    return pd.DataFrame(rows_out)


def plot(out, order):
    tab = pd.concat([pd.read_csv(f) for f in sorted(out.glob('gnnexplainer_od_seed*.csv'))])
    agg = tab.groupby(['type', 'what', 'feature'], as_index=False).agg(value=('value', 'mean'), sd=('value', 'std'),
                                                                       n_origins=('n_origins', 'first'))
    agg['seeds'] = tab.seed.nunique()
    agg.to_csv(out / 'gnnexplainer_od.csv', index=False)
    get = lambda t, w, f: agg[(agg.type == t) & (agg.what == w) & (agg.feature == f)].value.iloc[0]
    cols = [(t, c) for t in order for c in ('compensation', 'competition')]
    M = np.array([[get(t, c, f) for t, c in cols] for f in FEATS])
    Sg = np.array([[get(t, 'signed', f) for t in order] for f in FEATS])
    n = {t: int(agg[agg.type == t].n_origins.iloc[0]) for t in order}
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    fig, (ax, bx, sx) = plt.subplots(1, 3, figsize=(22, 7.4), gridspec_kw={'width_ratios': [1.0, 1.5, 0.9]})
    for i, f in enumerate(FEATS):
        for j, (t, c) in enumerate(cols):
            ax.scatter(M[i, j], len(FEATS) - 1 - i, marker='o', s=80, zorder=3, linewidth=1.8, edgecolor=TYPE_COLOR[t],
                       facecolor=TYPE_COLOR[t] if c == 'competition' else 'white')
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels(FEATS[::-1])
    for y_ in (1.5, 4.5):
        ax.axhline(y_, ls='--', lw=0.8, color='#999999')
    ax.set_xlim(-2, 102)
    ax.grid(axis='y', color='#E5E5E5')
    ax.set_xlabel('Mask kept by the model (%)')
    ax.set_ylabel('Supply components and topology types', fontweight='bold')
    ax.set_title('(a) Per-Component Cleveland Dot Plot', fontweight='bold')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    hd = [plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor=TYPE_COLOR[t], markeredgecolor=TYPE_COLOR[t],
                     label=t) for t in order]
    hd += [plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='#777777', markeredgecolor='#777777',
                      label='Competition (filled)'),
           plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='white', markeredgecolor='#777777',
                      label='Compensation (hollow)')]
    ax.legend(handles=hd, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=10)
    im = bx.imshow(M, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            bx.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=10, color='white' if M[i, j] > 60 else '#333333')
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels([f'{t}\n{c.capitalize()}\nn={n[t]}' for t, c in cols], fontsize=10)
    bx.set_yticks(range(len(FEATS)))
    bx.set_yticklabels(FEATS)
    for y_ in (2.5, 5.5):
        bx.axhline(y_, color='white', lw=3)
    for x_ in (1.5, 3.5, 5.5):
        bx.axvline(x_, color='white', lw=3)
    bx.set_title('(b) Importance Heatmap (%)', fontweight='bold')
    fig.colorbar(im, ax=bx, fraction=0.03, pad=0.02, label='Importance (%)')
    lim = float(np.nanmax(np.abs(Sg))) or 1.0
    im2 = sx.imshow(Sg, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(Sg.shape[0]):
        for j in range(Sg.shape[1]):
            sx.text(j, i, f'{Sg[i, j]:+.1f}', ha='center', va='center', fontsize=10)
    sx.set_xticks(range(len(order)))
    sx.set_xticklabels([f'{t}\nn={n[t]}' for t in order], fontsize=10)
    sx.set_yticks(range(len(FEATS)))
    sx.set_yticklabels(FEATS)
    for y_ in (2.5, 5.5):
        sx.axhline(y_, color='white', lw=3)
    sx.set_title('(c) Removal: competition share change (pp)', fontweight='bold')
    fig.colorbar(im2, ax=sx, fraction=0.05, pad=0.03, label='< 0: the feature pushes toward competition')
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(out / f'fig_gnnexplainer_od.{ext}')
    fig.savefig(out / 'fig_gnnexplainer_od.png', dpi=300)
    pd.set_option('display.width', 250)
    print(agg.pivot_table(index='feature', columns=['type', 'what'], values='value').reindex(FEATS).round(1).to_string())
    print('saved to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ruler', required=True)
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    g, x, node, run, order = setup(cfg, a.ruler)
    out = run / 'explain'
    out.mkdir(exist_ok=True)
    if a.aggregate:
        plot(out, order)
        return
    for s in a.seeds:
        explain_seed(cfg, g, x, node, run, s).to_csv(out / f'gnnexplainer_od_seed{s}.csv', index=False)


if __name__ == '__main__':
    main()
