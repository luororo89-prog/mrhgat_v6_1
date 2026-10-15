"""GNNExplainer masks of gnnexplainer_od_v13 (Own TH, Own SH, Own CHC, Dest TH, Dest SH, Dest CHC, ADJ, CRS) by component
change of the origin: for a basic configuration type, component j of TH, SH, CHC and direction of change of j of the
origin (rise or decline, as in explain_channel_change_v13; a direction with fewer than MIN_ORIGINS origins in the type is
left out) and channel c one mask is fitted on the outflow edges of the group's origins whose channel is c, with the two
routing tracks, the objective and the seeds of gnnexplainer_od_v13. Mean over the seeds.
Outputs in results_lineb/<name>/explain: gnnexplainer_od_change_seed<s>.csv, gnnexplainer_od_change.csv and
fig_gnnexplainer_od_change_<type>.pdf/.svg/.png ((a) Cleveland dot plot and (b) heatmap, layout of
explain_channel_change_v13).

python 5lineb/v13_geoflow/gnnexplainer_od_change_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --seeds 0 2
python 5lineb/v13_geoflow/gnnexplainer_od_change_v13.py --config <config> --ruler results_lineb/walk_sup1013_rot --aggregate
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

if importlib.util.find_spec('geopandas') is None:
    sys.modules['geopandas'] = types.ModuleType('geopandas')  # only the maps of plot_supply_change_v11 use it

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gnnexplainer_od_v13 import FEATS, apply  # noqa: E402
from gnnexplainer_v13 import setup, make_model  # noqa: E402
from explain_channel_change_v13 import COMPS, CHANNELS, COLOR, MIN_ORIGINS  # noqa: E402


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
    torch.manual_seed(seed)
    rows_out = []
    for tname in node.type.unique():
        for comp in COMPS:
            for d_ in ('rise', 'decline'):
                idx = np.nonzero(((node.type == tname) & (node[f'dir_{comp}'] == d_)).to_numpy())[0]
                if len(idx) < MIN_ORIGINS:
                    continue
                rows = torch.as_tensor(idx)
                Wt = torch.as_tensor(W[idx], dtype=torch.float32)
                E = torch.as_tensor(obs[idx])
                with torch.no_grad():
                    p0 = apply(m, xt, share, msg, tau, rows, torch.ones(8))
                ch = p0.argmax(1)
                for raw, cname in raw_name.items():
                    sel = (E & (ch == raw)).float() * Wt
                    if sel.sum() == 0:
                        continue
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
                    lab = f"{'Compet.' if cname == 'competition' else 'Comp.'} {comp} {d_}"
                    for f, v in zip(FEATS, torch.sigmoid(theta).detach().numpy()):
                        rows_out.append({'seed': seed, 'type': tname, 'class': lab, 'feature': f,
                                         'importance': 100 * float(v), 'n_origins': len(idx),
                                         'n_edges': int((sel > 0).sum())})
                print(f'seed {seed} {tname} {comp} {d_} done', flush=True)
    return pd.DataFrame(rows_out)


def plot(tab, tname, path):
    cols = [f'{ch} {c} {s}' for c in COMPS for ch in CHANNELS for s in ('rise', 'decline')]
    M = tab.pivot(index='feature', columns='class', values='importance').reindex(index=FEATS, columns=cols)
    n = tab.drop_duplicates('class').set_index('class').n_origins.reindex(cols)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(19, 7.4), gridspec_kw={'width_ratios': [0.9, 1.4]})
    for i, f in enumerate(FEATS):
        for c in cols:
            v = M.loc[f, c]
            if np.isfinite(v):
                ch, comp, s = c.split()
                ax.scatter(v, len(FEATS) - 1 - i, marker='^' if s == 'rise' else 'v', s=90, zorder=3,
                           facecolor=COLOR[comp] if ch == 'Compet.' else 'white', edgecolor=COLOR[comp], linewidth=1.6)
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
    hd = [plt.Line2D([], [], ls='', marker='^' if s == 'rise' else 'v', markersize=9, markerfacecolor=COLOR[c],
                     markeredgecolor=COLOR[c], label=f'{c} {s}') for c in COMPS for s in ('rise', 'decline')]
    hd += [plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='#777777', markeredgecolor='#777777',
                      label='Competition (filled)'),
           plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='white', markeredgecolor='#777777',
                      label='Compensation (hollow)')]
    ax.legend(handles=hd, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=10)
    im = bx.imshow(M.to_numpy(float), cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                bx.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, color='white' if v > 60 else '#333333')
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels([f'{c.split()[0]}\n{c.split()[1]} {c.split()[2]}\nn={int(n[c]) if np.isfinite(n[c]) else 0}'
                        for c in cols], fontsize=9)
    bx.set_yticks(range(len(FEATS)))
    bx.set_yticklabels(FEATS)
    for y_ in (2.5, 5.5):
        bx.axhline(y_, color='white', lw=3)
    for x_ in (3.5, 7.5):
        bx.axvline(x_, color='white', lw=3)
    for x_ in (1.5, 5.5, 9.5):
        bx.axvline(x_, color='white', lw=1)
    bx.set_xlabel('Edge channel x component change of the origin (n = origins)')
    bx.set_title('(b) Importance Heatmap (%)', fontweight='bold')
    fig.colorbar(im, ax=bx, fraction=0.03, pad=0.03, label='Importance (%)')
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
            explain_seed(cfg, g, x, node, run, s).to_csv(out / f'gnnexplainer_od_change_seed{s}.csv', index=False)
        return
    tab = pd.concat([pd.read_csv(f) for f in sorted(out.glob('gnnexplainer_od_change_seed*.csv'))])
    seeds = sorted(tab.seed.unique())
    agg = tab.groupby(['type', 'class', 'feature'], as_index=False).agg(importance=('importance', 'mean'),
                                                                        importance_sd=('importance', 'std'),
                                                                        n_origins=('n_origins', 'first'),
                                                                        n_edges=('n_edges', 'mean'))
    agg.to_csv(out / 'gnnexplainer_od_change.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    for tname in order:
        plot(agg[agg.type == tname], f'{tname} (GNNExplainer, {len(seeds)} seeds)', out / f'fig_gnnexplainer_od_change_{tname}')
    pd.set_option('display.width', 250)
    print('seeds', seeds)
    print(agg.pivot_table(index=['type', 'feature'], columns='class', values='importance').round(1).to_string())
    print('saved to', out)


if __name__ == '__main__':
    main()
