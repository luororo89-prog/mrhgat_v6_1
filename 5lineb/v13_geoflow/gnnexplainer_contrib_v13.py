"""GNNExplainer (Ying et al. 2019) on the channel contributions to the component changes of a model_v13 run: which supply
component of the origin (Own TH, SH, CHC), of the destination (Dest TH, SH, CHC) and which edge topology (ADJ, CRS) the
trained model needs for the contribution of channel c to the change of component j of the origins of a basic
configuration type, and in which direction each of them moves that contribution.

Contribution (consensus_v10, model: node): with the seed's vectors s = l2(embed(x)) and its joint ridge decoder D fitted
on them (x on s, alphas of decode), origin A of channel k has the aggregated vector
       c_A,k = l2(a_A s_A,k + sum_B pi_AB p_AB,k s_B,k)
(a_A the internal trip share, 1 without trips; p the soft routing shares of the two routing tracks of
gnnexplainer_od_v13 in place of the consensus one-hot), and the contribution of k to component j is the tier component
(mean of the three indicators of j) of D(s with block k replaced by c_k) - D(s). Masks as in gnnexplainer_od_v13: the
own mask multiplies the indicators of the type's origins (in s_A and in the origin track), the destination mask the
indicators of every community as destination (in s_B and in the destination track), 0 being the city mean, the
topology masks the bias B_t of the ADJ and CRS edges of the type's origins.
For a type, component j and channel c one mask m = sigmoid(theta) over the eight features is fitted on the
contributions P_A of the type's origins, GNNExplainer regression objective (PyG defaults: 100 epochs, Adam lr 0.01,
theta ~ N(0, 0.1), size coefficient 1.0, entropy coefficient 0.1), the squared error relative to the mean square of the
unmasked contributions:
       loss = mean_A (P_A(m) - P_A(1))^2 / mean_A P_A(1)^2 + 1.0 mean(m) + 0.1 mean(H(m)).
Signed effect: mean contribution over the type's origins minus the mean contribution when one feature is removed (its
mask set to 0, the others 1); positive when the feature raises component j through channel c. Mean over the seeds of
gnnexplainer_od_v13 (--seeds), channel names as in gnnexplainer_v13.
Outputs in results_lineb/<name>/explain: gnnexplainer_contrib_seed<s>.csv, gnnexplainer_contrib.csv and
fig_gnnexplainer_contrib_<type>.pdf/.svg/.png ((a) Cleveland dot plot and (b) heatmap of the masks, (c) signed effects).

python 5lineb/v13_geoflow/gnnexplainer_contrib_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --seeds 0 2
python 5lineb/v13_geoflow/gnnexplainer_contrib_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --aggregate
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import RidgeCV

if importlib.util.find_spec('geopandas') is None:
    sys.modules['geopandas'] = types.ModuleType('geopandas')  # only the maps of plot_supply_change_v11 use it

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_v13 import l2n  # noqa: E402
from gnnexplainer_od_v13 import FEATS, mask_x, routing  # noqa: E402
from gnnexplainer_v13 import setup, make_model  # noqa: E402
from explain_channel_change_v13 import COMPS, COLOR  # noqa: E402
from decode import ALPHAS  # noqa: E402

CHANNELS = ('compensation', 'competition')


def contribution(m, xt, share, msg, tau, rows, mk, a_self, Wd):
    """Contributions of every channel to the tier components of the change of the origins rows, (rows, K, 3)."""
    xo, xd = mask_x(xt, mk[:3], rows), mask_x(xt, mk[3:6])
    p = routing(m, xo, xd, share, msg, tau, rows, mk[6], mk[7])
    so, sd = l2n(m.embed(xo))[rows], l2n(m.embed(xd))
    w = (share * msg.float())[rows][:, None, :] * p
    c = l2n(a_self[rows][:, None, None] * so + torch.einsum('ukv,vkd->ukd', w, sd))
    part = torch.einsum('ukd,jkd->ukj', c - so, Wd)
    return part.reshape(len(rows), part.shape[1], 3, 3).mean(3)


def explain_seed(cfg, g, x, node, run, seed):
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    share_np = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    share = torch.as_tensor(share_np, dtype=torch.float32)
    msg = torch.as_tensor(obs)
    tau = cfg['tau_final']
    a_self = np.diag(share_np).copy() if cfg.get('out_weight', cfg.get('self_weight', 'one')) == 'share' else np.ones(n)
    a_self[W.sum(1) <= 0] = 1.0
    a_self = torch.as_tensor(a_self, dtype=torch.float32)
    m = make_model(cfg, g, n)
    m.load_state_dict(torch.load(run / f'seed{seed}' / 'model.pt'))
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    xt = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        s = l2n(m.embed(xt))
    dec = RidgeCV(alphas=ALPHAS).fit(s.numpy().reshape(n, -1).astype(float), x)
    Wd = torch.as_tensor(dec.coef_, dtype=torch.float32).reshape(x.shape[1], s.shape[1], s.shape[2])
    order = json.loads(pd.read_csv(run / 'summary.csv').set_index('seed').channel_order.loc[seed])
    names = pd.read_csv(run / 'comparison' / 'mechanism_origin.csv').set_index('channel').name.to_dict()
    raw_name = {order[k]: names[f'C{k + 1}'] for k in range(len(order))}
    torch.manual_seed(seed)
    rows_out = []
    for tname in node.type.unique():
        rows = torch.as_tensor(np.nonzero((node.type == tname).to_numpy())[0])
        with torch.no_grad():
            P0 = contribution(m, xt, share, msg, tau, rows, torch.ones(8), a_self, Wd)
        rec = {'seed': seed, 'type': tname, 'n_origins': len(rows)}
        for raw, cname in raw_name.items():
            for j, comp in enumerate(COMPS):
                rows_out.append({**rec, 'component': comp, 'channel': cname, 'what': 'base', 'feature': '',
                                 'value': float(P0[:, raw, j].mean())})
                y0 = P0[:, raw, j]
                theta = torch.nn.Parameter(torch.randn(8) * 0.1)
                opt = torch.optim.Adam([theta], lr=0.01)
                for _ in range(100):
                    mk = torch.sigmoid(theta)
                    y = contribution(m, xt, share, msg, tau, rows, mk, a_self, Wd)[:, raw, j]
                    pred = ((y - y0) ** 2).mean() / (y0 ** 2).mean().clamp_min(1e-12)
                    ent = -(mk * torch.log(mk + 1e-15) + (1 - mk) * torch.log(1 - mk + 1e-15))
                    loss = pred + 1.0 * mk.mean() + 0.1 * ent.mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                for f, v in zip(FEATS, torch.sigmoid(theta).detach().numpy()):
                    rows_out.append({**rec, 'component': comp, 'channel': cname, 'what': 'mask', 'feature': f,
                                     'value': 100 * float(v)})
        with torch.no_grad():
            for i, f in enumerate(FEATS):
                mk = torch.ones(8)
                mk[i] = 0.0
                P = contribution(m, xt, share, msg, tau, rows, mk, a_self, Wd)
                for raw, cname in raw_name.items():
                    for j, comp in enumerate(COMPS):
                        rows_out.append({**rec, 'component': comp, 'channel': cname, 'what': 'effect', 'feature': f,
                                         'value': float(P0[:, raw, j].mean() - P[:, raw, j].mean())})
        print(f'seed {seed} {tname} done', flush=True)
    return pd.DataFrame(rows_out)


def plot(agg, tname, path):
    cols = [(j, c) for j in COMPS for c in CHANNELS]
    get = lambda w, j, c, f: agg[(agg.what == w) & (agg.component == j) & (agg.channel == c) & (agg.feature == f)].value.iloc[0]
    M = np.array([[get('mask', j, c, f) for j, c in cols] for f in FEATS])
    Sg = np.array([[get('effect', j, c, f) for j, c in cols] for f in FEATS])
    P = [agg[(agg.what == 'base') & (agg.component == j) & (agg.channel == c)].value.iloc[0] for j, c in cols]
    labels = [f'{j}\n{c.capitalize()}\n{p:+.2f}' for (j, c), p in zip(cols, P)]
    fig, (ax, bx, sx) = plt.subplots(1, 3, figsize=(22, 7.4), gridspec_kw={'width_ratios': [1.0, 1.0, 1.0]})
    for i in range(len(FEATS)):
        for k, (j, c) in enumerate(cols):
            ax.scatter(M[i, k], len(FEATS) - 1 - i, marker='o', s=80, zorder=3, linewidth=1.8, edgecolor=COLOR[j],
                       facecolor=COLOR[j] if c == 'competition' else 'white')
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
    hd = [plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor=COLOR[j], markeredgecolor=COLOR[j],
                     label=j) for j in COMPS]
    hd += [plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='#777777', markeredgecolor='#777777',
                      label='Competition (filled)'),
           plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='white', markeredgecolor='#777777',
                      label='Compensation (hollow)')]
    ax.legend(handles=hd, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=10)
    im = bx.imshow(M, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            bx.text(k, i, f'{M[i, k]:.1f}', ha='center', va='center', fontsize=10, color='white' if M[i, k] > 60 else '#333333')
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels(labels, fontsize=10)
    bx.set_yticks(range(len(FEATS)))
    bx.set_yticklabels(FEATS)
    for y_ in (2.5, 5.5):
        bx.axhline(y_, color='white', lw=3)
    for x_ in (1.5, 3.5):
        bx.axvline(x_, color='white', lw=3)
    bx.set_xlabel('Component x channel (mean contribution)')
    bx.set_title('(b) Importance Heatmap (%)', fontweight='bold')
    fig.colorbar(im, ax=bx, fraction=0.04, pad=0.03, label='Importance (%)')
    lim = float(np.nanmax(np.abs(Sg))) or 1.0
    im2 = sx.imshow(Sg, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(Sg.shape[0]):
        for k in range(Sg.shape[1]):
            sx.text(k, i, f'{Sg[i, k]:+.2f}', ha='center', va='center', fontsize=10)
    sx.set_xticks(range(len(cols)))
    sx.set_xticklabels(labels, fontsize=10)
    sx.set_yticks(range(len(FEATS)))
    sx.set_yticklabels(FEATS)
    for y_ in (2.5, 5.5):
        sx.axhline(y_, color='white', lw=3)
    for x_ in (1.5, 3.5):
        sx.axvline(x_, color='white', lw=3)
    sx.set_xlabel('Component x channel (mean contribution)')
    sx.set_title('(c) Removal: change of the contribution', fontweight='bold')
    fig.colorbar(im2, ax=sx, fraction=0.04, pad=0.03, label='> 0: the feature raises the component')
    fig.suptitle(tname, fontsize=16, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(path.with_suffix(f'.{ext}'))
    fig.savefig(path.with_suffix('.png'), dpi=300)
    plt.close(fig)


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
    if not a.aggregate:
        for s in a.seeds:
            explain_seed(cfg, g, x, node, run, s).to_csv(out / f'gnnexplainer_contrib_seed{s}.csv', index=False)
        return
    tab = pd.concat([pd.read_csv(f) for f in sorted(out.glob('gnnexplainer_contrib_seed*.csv'))])
    tab['feature'] = tab.feature.fillna('')
    seeds = sorted(tab.seed.unique())
    agg = tab.groupby(['type', 'component', 'channel', 'what', 'feature'], as_index=False).agg(
        value=('value', 'mean'), sd=('value', 'std'), n_origins=('n_origins', 'first'))
    agg.to_csv(out / 'gnnexplainer_contrib.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    for tname in order:
        a_t = agg[agg.type == tname]
        plot(a_t, f'{tname} (GNNExplainer on the channel contributions, {len(seeds)} seeds, n={int(a_t.n_origins.iloc[0])})',
             out / f'fig_gnnexplainer_contrib_{tname}')
    pd.set_option('display.width', 250)
    print('seeds', seeds)
    print(agg[agg.what != 'mask'].pivot_table(index=['type', 'what', 'feature'], columns=['component', 'channel'],
                                              values='value').round(3).to_string())
    print('saved to', out)


if __name__ == '__main__':
    main()
