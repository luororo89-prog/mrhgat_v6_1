"""Which own supply component (TH, SH, CHC) of the origin and which topology of the edge separate the combinations of
edge channel and component change of the origin, by basic configuration type, with explainable classifiers trained on
the model's grouping (Resili-Net, Yin et al.; random forest and two further classifiers).

Unit: every non-self outflow edge A -> B of the consensus of a model_v13 run (consensus_edges.csv), sample weight its
trips. Label, for each component j of TH, SH, CHC: consensus channel of the edge (compensation or competition, named
by the pressure association of compare_v13) x direction of change of component j of the origin (rise when
(x + dx) B > x B, dx from consensus_community.csv, B the rotated ruler --ruler, as in the type change figure), four
classes; a direction with fewer than MIN_ORIGINS origins in the type is left out. Features: rotated component scores
TH, SH, CHC of the origin and the topology of the edge (1 adjacent, 0 crossing; internal trips have no channel).
Classifiers: random forest (300 trees, leaves of at least 5 edges), histogram gradient boosting, multinomial logistic
regression on standardised features; five folds stratified by label and grouped by origin. Fidelity = trip-weighted
macro one-vs-rest ROC AUC on the held-out folds. Importance of a feature for a class = drop of the one-vs-rest AUC of
that class when the feature is permuted in the held-out fold (5 repeats), mean over the folds, negative drops set to 0,
in % of the sum over the features. The figures use the classifier with the highest mean fidelity over all cells.
Outputs in results_lineb/<name>/explain: channel_change_importance.csv (all classifiers), channel_change_fidelity.csv
and fig_channel_change_importance_<type>.pdf/.svg/.png.

python 5lineb/v13_geoflow/explain_channel_change_v13.py \
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
CHANNELS = ('Comp.', 'Compet.')
FEATURES = ['Own TH', 'Own SH', 'Own CHC', 'ADJ / CRS']
COLOR = {'TH': '#0072B2', 'SH': '#E69F00', 'CHC': '#009E73'}
MIN_ORIGINS = 10
MODELS = {
    'random forest': lambda: RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1),
    'gradient boosting': lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0),
    'logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}


def fit(model, X, y, w):
    if isinstance(model, RandomForestClassifier) or isinstance(model, HistGradientBoostingClassifier):
        return model.fit(X, y, sample_weight=w)
    return model.fit(X, y, logisticregression__sample_weight=w)


def ovr_auc(y, P, w, classes):
    return np.array([roc_auc_score(y == c, P[:, i], sample_weight=w) if 0 < (y == c).sum() < len(y) else np.nan
                     for i, c in enumerate(classes)])


def explain(X, y, w, groups, make, seed=0):
    """Per-class fidelity (one-vs-rest AUC) and permutation importance (%) over five folds grouped by origin."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    aucs, drops = [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[tr])) < len(classes):
            continue
        m = fit(make(), X[tr], y[tr], w[tr])
        base = ovr_auc(y[te], m.predict_proba(X[te]), w[te], classes)
        d = np.zeros((X.shape[1], len(classes)))
        for j in range(X.shape[1]):
            for _ in range(5):
                Xp = X[te].copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                d[j] += (base - ovr_auc(y[te], m.predict_proba(Xp), w[te], classes)) / 5
        aucs.append(base)
        drops.append(d)
    auc = np.nanmean(aucs, 0)
    d = np.clip(np.nanmean(drops, 0), 0, None)
    imp = np.where(d.sum(0) > 0, 100 * d / np.where(d.sum(0) > 0, d.sum(0), 1), 0.0)
    return classes, auc, imp


def plot(tab, tname, path):
    cols = [f'{ch} {c} {s}' for c in COMPS for ch in CHANNELS for s in ('rise', 'decline')]
    M = tab.pivot(index='feature', columns='class', values='importance').reindex(index=FEATURES, columns=cols)
    n = tab.drop_duplicates('class').set_index('class').n_origins.reindex(cols)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 6.8), gridspec_kw={'width_ratios': [0.9, 1.4]})
    for i, f in enumerate(FEATURES):
        for c in cols:
            v = M.loc[f, c]
            if np.isfinite(v):
                ch, comp, s = c.split()
                ax.scatter(v, len(FEATURES) - 1 - i, marker='^' if s == 'rise' else 'v', s=90, zorder=3,
                           facecolor=COLOR[comp] if ch == 'Compet.' else 'white', edgecolor=COLOR[comp], linewidth=1.6)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES[::-1])
    ax.axhline(0.5, ls='--', lw=0.8, color='#999999')
    ax.set_xlim(-2, 102)
    ax.grid(axis='y', color='#E5E5E5')
    ax.set_xlabel('Importance (%)')
    ax.set_ylabel('Origin supply components and edge topology', fontweight='bold')
    ax.set_title('(a) Per-Component Cleveland Dot Plot', fontweight='bold')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    hd = [plt.Line2D([], [], ls='', marker='^' if s == 'rise' else 'v', markersize=9, markerfacecolor=COLOR[c],
                     markeredgecolor=COLOR[c], label=f'{c} {s}') for c in COMPS for s in ('rise', 'decline')]
    hd += [plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='#777777', markeredgecolor='#777777',
                      label='Competition (filled)'),
           plt.Line2D([], [], ls='', marker='o', markersize=9, markerfacecolor='white', markeredgecolor='#777777',
                      label='Compensation (hollow)')]
    ax.legend(handles=hd, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=10)
    im = bx.imshow(M.to_numpy(float), cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                bx.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, color='white' if v > 60 else '#333333')
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels([f'{c.split()[0]}\n{c.split()[1]} {c.split()[2]}\nn={int(n[c]) if np.isfinite(n[c]) else 0}'
                        for c in cols], fontsize=9)
    bx.set_yticks(range(len(FEATURES)))
    bx.set_yticklabels(FEATURES)
    bx.axhline(2.5, color='white', lw=3)
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
    node = pd.DataFrame({'origin_id': ids, 'type': [TYPE_EN[T['type_names'][c]] for c in measure(x, T)['class']],
                         **{f'Own {c}': Sb[:, j] for j, c in enumerate(COMPS)},
                         **{f'dir_{c}': np.where(Se[:, j] > Sb[:, j], 'rise', 'decline') for j, c in enumerate(COMPS)}})
    names = pd.read_csv(run / 'comparison' / 'mechanism_origin.csv').set_index('channel').name.to_dict()
    e = pd.read_csv(run / 'consensus_edges.csv').merge(node, on='origin_id')
    e['ADJ / CRS'] = (e.topology == 'ADJ').astype(float)
    e['ch'] = e.channel.map(lambda c: 'Compet.' if names[c] == 'competition' else 'Comp.')
    rows, fid = [], []
    for tname in [TYPE_EN[t] for t in T['type_names']]:
        et = e[e.type == tname]
        for c in COMPS:
            n_dir = et.groupby(f'dir_{c}').origin_id.nunique()
            keep = [s for s in ('rise', 'decline') if n_dir.get(s, 0) >= MIN_ORIGINS]
            sub = et[et[f'dir_{c}'].isin(keep)]
            y = (sub.ch + f' {c} ' + sub[f'dir_{c}']).to_numpy()
            for mname, make in MODELS.items():
                if len(np.unique(y)) >= 2:
                    classes, auc, imp = explain(sub[FEATURES].to_numpy(float), y, sub.flow.to_numpy(float),
                                                sub.origin_id.to_numpy(), make)
                else:
                    classes, auc, imp = [], [], np.zeros((len(FEATURES), 0))
                fid.append({'type': tname, 'component': c, 'model': mname, 'macro_auc': float(np.nanmean(auc)) if len(auc) else np.nan})
                for k, cl in enumerate(classes):
                    for f, v in zip(FEATURES, imp[:, k]):
                        rows.append({'type': tname, 'model': mname, 'class': cl, 'feature': f, 'importance': v,
                                     'class_auc': auc[k], 'n_origins': int(n_dir.get(cl.split()[2], 0)),
                                     'n_edges': int((y == cl).sum())})
            print(tname, c, 'done', flush=True)
    tab, F = pd.DataFrame(rows), pd.DataFrame(fid)
    tab.to_csv(out / 'channel_change_importance.csv', index=False)
    F.to_csv(out / 'channel_change_fidelity.csv', index=False)
    best = F.groupby('model').macro_auc.mean().idxmax()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    for tname, t_ in tab[tab.model == best].groupby('type', sort=False):
        plot(t_, f'{tname} ({best})', out / f'fig_channel_change_importance_{tname}')
    pd.set_option('display.width', 250)
    print(F.pivot_table(index=['type', 'component'], columns='model', values='macro_auc').round(3).to_string())
    print('model used for the figures:', best)
    print(tab[tab.model == best].pivot_table(index=['type', 'feature'], columns='class', values='importance').round(1).to_string())
    print('saved to', out)


if __name__ == '__main__':
    main()
