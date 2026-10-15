"""Which own supply component (TH, SH, CHC) of the origin and which topology of the edge the channel of an outflow edge
(compensation or competition, consensus of a model_v13 run) depends on, by basic configuration type, with the random
forest surrogate of Resili-Net (Yin et al.; explainable classifier trained on the model's grouping).

Unit: every non-self outflow edge A -> B of the consensus (consensus_edges.csv), label its consensus channel, named by
the pressure association of compare_v13 (comparison/mechanism_origin.csv), sample weight its trips. Features: the
rotated component scores TH, SH, CHC of the origin (x B with the rotated ruler, --ruler) and the topology of the edge
(1 adjacent, 0 crossing; internal trips are not assigned to a channel). Groups: basic type of the origin (ruler of the
run, ruler_from, English names of plot_type_change_v13) and the direction of change of each component of the origin
(rise when (x + dx) B > x B, dx from consensus_community.csv, as in the type change figure); groups with fewer than
MIN_ORIGINS origins are left empty. Random forest (sklearn, 300 trees, leaves of at least 5 edges), five folds grouped
by origin; fidelity = trip-weighted ROC AUC on the held-out folds; importance = drop of that AUC when the feature is
permuted in the held-out fold (10 repeats), mean over the folds, negative drops set to 0 and expressed as % of the sum
over the features.
Outputs in results_lineb/<name>/explain: channel_importance.csv and fig_channel_importance_<type>.pdf/.svg/.png
(a: Cleveland dot plot of the importances, b: heatmap, columns the component changes with the number of origins).

python 5lineb/v13_geoflow/explain_channels_v13.py \
    --config 5lineb/v13_geoflow/configs/walk_sup1013_net2_v13_k2_s100_hold_iters7_disencode_routescore_topo_tau05_ruleorigin_road.yaml \
    --ruler results_lineb/walk_sup1013_rot
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
from graph import build_graph  # noqa: E402
from ruler import standardize, measure  # noqa: E402
from run_v2 import frozen  # noqa: E402
from plot_type_change_v13 import TYPE_EN  # noqa: E402

COMPS = ('TH', 'SH', 'CHC')
FEATURES = ['Own TH', 'Own SH', 'Own CHC', 'ADJ / CRS']
COLOR = {'TH': '#0072B2', 'SH': '#E69F00', 'CHC': '#009E73'}
MIN_ORIGINS = 10


def wauc(y, s, w):
    return roc_auc_score(y, s, sample_weight=w)


def importance(X, y, w, groups, seed=0):
    """Fidelity AUC and permutation importance (AUC drop, %) over five folds grouped by origin."""
    rng = np.random.default_rng(seed)
    aucs, drops = [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        rf = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed, n_jobs=-1).fit(X[tr], y[tr], sample_weight=w[tr])
        base = wauc(y[te], rf.predict_proba(X[te])[:, 1], w[te])
        aucs.append(base)
        d = []
        for j in range(X.shape[1]):
            r = []
            for _ in range(10):
                Xp = X[te].copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                r.append(base - wauc(y[te], rf.predict_proba(Xp)[:, 1], w[te]))
            d.append(np.mean(r))
        drops.append(d)
    if not aucs:
        return np.nan, np.full(X.shape[1], np.nan)
    d = np.clip(np.mean(drops, 0), 0, None)
    return float(np.mean(aucs)), 100 * d / d.sum() if d.sum() > 0 else np.zeros_like(d)


def plot(tab, tname, names, path):
    cols = [f'{c} {s}' for c in COMPS for s in ('rise', 'decline')]
    M = tab.pivot(index='feature', columns='group', values='importance').reindex(index=FEATURES, columns=cols)
    n = tab.drop_duplicates('group').set_index('group').n_origins.reindex(cols)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={'width_ratios': [1.15, 1]})
    for i, f in enumerate(FEATURES):
        for c in cols:
            v = M.loc[f, c]
            if np.isfinite(v):
                comp, s = c.split()
                ax.scatter(v, len(FEATURES) - 1 - i, marker='^' if s == 'rise' else 'v', s=90, color=COLOR[comp], zorder=3)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES[::-1])
    ax.axhline(0.5, ls='--', lw=0.8, color='#999999')
    ax.set_xlim(-2, 102)
    ax.grid(axis='y', color='#E5E5E5')
    ax.set_xlabel(f'Importance for the channel ({names}) (%)')
    ax.set_ylabel('Origin supply components and edge topology', fontweight='bold')
    ax.set_title('(a) Per-Component Cleveland Dot Plot', fontweight='bold')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    hd = [plt.Line2D([], [], ls='', marker='^' if s == 'rise' else 'v', markersize=9, color=COLOR[c], label=f'{c} {s}')
          for c in COMPS for s in ('rise', 'decline')]
    ax.legend(handles=hd, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    im = bx.imshow(M.to_numpy(float), cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                bx.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=10, color='white' if v > 60 else '#333333')
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels([f'{c.split()[0]}\n{c.split()[1]}\nn={int(n[c]) if np.isfinite(n[c]) else 0}' for c in cols])
    bx.set_yticks(range(len(FEATURES)))
    bx.set_yticklabels(FEATURES)
    bx.axhline(2.5, color='white', lw=3)
    for x_ in (1.5, 3.5):
        bx.axvline(x_, color='white', lw=3)
    bx.set_xlabel('Component change of the origin')
    bx.set_title('(b) Importance Heatmap (%)', fontweight='bold')
    fig.colorbar(im, ax=bx, fraction=0.04, pad=0.04, label='Importance (%)')
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
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    run = ROOT / 'results_lineb' / cfg['name']
    out = run / 'explain'
    out.mkdir(exist_ok=True)
    g = build_graph(cfg)
    ids = np.asarray(g['ids'])
    T = frozen(ROOT / cfg['ruler_from'])['main']
    U = frozen(ROOT / a.ruler)['main']['U']
    x = standardize(g['raw'], T)
    com = pd.read_csv(run / 'consensus_community.csv', encoding='utf-8-sig').set_index('community_id').loc[ids]
    dx = com[[f'dx_{c}' for c in g['columns']]].to_numpy(float)
    Sb, Se = x @ U, (x + dx) @ U
    types = np.array([TYPE_EN[T['type_names'][c]] for c in measure(x, T)['class']])
    node = pd.DataFrame({'origin_id': ids, 'type': types,
                         **{f'Own {c}': Sb[:, j] for j, c in enumerate(COMPS)},
                         **{f'dir_{c}': np.where(Se[:, j] > Sb[:, j], 'rise', 'decline') for j, c in enumerate(COMPS)}})
    names = pd.read_csv(run / 'comparison' / 'mechanism_origin.csv').set_index('channel').name.to_dict()
    compet = next(c for c, v in names.items() if v == 'competition')
    e = pd.read_csv(run / 'consensus_edges.csv').merge(node, on='origin_id')
    e['ADJ / CRS'] = (e.topology == 'ADJ').astype(float)
    e['y'] = (e.channel == compet).astype(int)
    rows = []
    for tname in [TYPE_EN[t] for t in T['type_names']]:
        et = e[e.type == tname]
        for c in COMPS:
            for s in ('rise', 'decline'):
                sub = et[et[f'dir_{c}'] == s]
                no = sub.origin_id.nunique()
                if no >= MIN_ORIGINS and sub.y.nunique() == 2:
                    auc, imp = importance(sub[FEATURES].to_numpy(float), sub.y.to_numpy(), sub.flow.to_numpy(float),
                                          sub.origin_id.to_numpy())
                else:
                    auc, imp = np.nan, np.full(len(FEATURES), np.nan)
                for f, v in zip(FEATURES, imp):
                    rows.append({'type': tname, 'group': f'{c} {s}', 'feature': f, 'importance': v, 'fidelity_auc': auc,
                                 'n_origins': no, 'n_edges': len(sub),
                                 'competition_trip_share': float((sub.flow * sub.y).sum() / sub.flow.sum()) if len(sub) else np.nan})
        print(tname, 'done', flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'channel_importance.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    for tname, t_ in tab.groupby('type', sort=False):
        plot(t_, tname, 'compensation / competition', out / f'fig_channel_importance_{tname}')
    pd.set_option('display.width', 250)
    print(tab.pivot_table(index=['type', 'feature'], columns='group', values='importance').round(1).to_string())
    print(tab.drop_duplicates(['type', 'group'])[['type', 'group', 'n_origins', 'n_edges', 'fidelity_auc',
                                                     'competition_trip_share']].round(3).to_string(index=False))
    print('saved to', out)


if __name__ == '__main__':
    main()
