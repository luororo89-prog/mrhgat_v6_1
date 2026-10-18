"""Fixed ruler with components that correspond to the three facility tiers (varimax rotation of the frozen PCA).

1  Start from the frozen ruler of results_lineb/walk_v4_pca5 (Yeo-Johnson transform, standardisation, three principal
   components by parallel analysis); the transform, mean and std are kept, so the standardized indicators and every
   supply change dx of the later runs stay valid.
2  Varimax rotation (Kaiser 1958) of the loadings L = U sqrt(lambda); every rotated component is matched to the tier
   whose three indicators load on it most and signed so that its loadings sum positive. Rotated component scores
   S = Z U diag(lambda)^(-1/2) T (unit variance, uncorrelated), so the three axes are TH, SH and CHC.
3  KMeans with k = 4 (n_init 50, random_state 0) on S, the k chosen on S by ruler.select_k (all types with bootstrap
   Jaccard >= 0.80; Calinski-Harabasz, Davies-Bouldin and gap vote for 4); the k selection table is reported. Types are named by the tertile level of each component at the type centre (naming rule of
   ruler.type_names) and ordered by the level profile (TH first, then SH, then CHC).
Output results_lineb/walk_v4_pca5_rot/ruler.json in the layout of the frozen ruler: frozen U holds the score
coefficients, so ruler.measure returns the rotated scores as 'pc' and the type as 'class' ('composite' is not used).

python 5lineb/v11_nodeedge/rotate_ruler_v11.py --config 5lineb/v11_nodeedge/configs/walk_v4_pca5_v11_k3_s10.yaml
    [--src results_lineb/walk_v4_pca5 --dst results_lineb/walk_v4_pca5_rot --k 4]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from graph import build_graph  # noqa: E402
from ruler import standardize, measure, select_k, jaccard_stability  # noqa: E402
from run_v2 import frozen  # noqa: E402

SRC = 'results_lineb/walk_v4_pca5'
DST = 'results_lineb/walk_v4_pca5_rot'
TIER_CN = ('三甲', '二级', '社康')
TIER_EN = ('TH', 'SH', 'CHC')


def varimax(L, iters=500, tol=1e-12):
    p, k = L.shape
    T = np.eye(k)
    d = 0.0
    for _ in range(iters):
        Lr = L @ T
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        T = u @ vt
        if s.sum() < d * (1 + tol):
            break
        d = s.sum()
    return T


def name(lv):
    word = ('低', '中', '高')
    profile = '·'.join(f'{n}{word[v]}' for n, v in zip(TIER_CN, lv))
    high = [n for n, v in zip(TIER_CN, lv) if v == 2]
    if (lv == 2).all():
        short = '充足型'
    elif (lv == 0).all():
        short = '稀缺型'
    elif len(high) == 1:
        short = f'{high[0]}主导型'
    elif len(high) == 2:
        short = f'{high[0]}{high[1]}型'
    else:
        short = '中等型'
    return f'{short}（{profile}）'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--dst', default=DST)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    root = HERE.parents[1]
    R = frozen(root / a.src)['main']
    src = json.loads((root / a.src / 'ruler.json').read_text(encoding='utf-8'))['main']
    g = build_graph(cfg)
    cols = list(g['columns'])
    Z = standardize(g['raw'], R)
    ev = np.asarray(src['eigenvalues'][:R['U'].shape[1]])
    L = R['U'] * np.sqrt(ev)
    T = varimax(L)
    Lr = L @ T
    T = T[:, [int(np.argmax(np.abs(Lr[3 * t:3 * t + 3]).sum(0))) for t in range(3)]]
    T = T * np.where((L @ T).sum(0) < 0, -1.0, 1.0)
    Lr = L @ T
    B = (R['U'] / np.sqrt(ev)) @ T
    S = Z @ B
    var = (Lr ** 2).sum(0)

    table, votes = select_k(S)[1:]
    km = KMeans(a.k, n_init=50, random_state=0).fit(S)
    cuts = np.quantile(S, [1 / 3, 2 / 3], axis=0)
    lv = (km.cluster_centers_[:, None, :] > cuts[None]).sum(1)
    order = np.lexsort(lv.T[::-1])
    centers, lv = km.cluster_centers_[order], lv[order]
    names = [name(v) for v in lv]
    fz = {'columns': cols, 'lambdas': R['lambdas'], 'mean': R['mean'].tolist(), 'std': R['std'].tolist(),
          'mode': 'composite', 'U': B.tolist(), 'weights': (var / var.sum()).tolist(), 'centers': centers.tolist(),
          'type_names': names}
    new = measure(Z, {**fz, 'U': B, 'weights': var / var.sum(), 'centers': centers})['class']
    old = measure(Z, R)['class']
    jac = jaccard_stability(S, a.k)
    report = {'source_ruler': a.src, 'rotation': 'varimax', 'rotation_matrix': T.tolist(),
              'loadings_rotated': pd.DataFrame(Lr, index=cols, columns=TIER_EN).round(3).to_dict(),
              'variance_rotated': var.tolist(), 'variance_rotated_share': (var / var.sum()).tolist(),
              'variance_explained_rotated': (var / len(cols)).tolist(),
              'k_selection': table.to_dict('records'), 'k_votes': votes, 'k': a.k,
              'jaccard': jac.tolist(), 'component_tertile_cuts': cuts.tolist(),
              'type_names': names, 'type_centres': np.round(centers, 3).tolist(),
              'class_sizes': pd.Series(new).value_counts().sort_index().to_dict(),
              'ari_with_source_types': float(adjusted_rand_score(old, new)),
              'crosstab_source_types': pd.crosstab(pd.Series([src['type_names'][c] for c in old], name='source'),
                                                   pd.Series([names[c] for c in new], name='rotated')).to_dict(),
              'frozen': fz}
    out = root / a.dst
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ruler.json').write_text(json.dumps({'main': report}, ensure_ascii=False, indent=1, default=int), encoding='utf-8')
    pd.set_option('display.width', 250)
    print(pd.DataFrame(Lr, index=cols, columns=TIER_EN).round(3).to_string())
    print('rotated variance share', (var / var.sum()).round(3), 'explained', (var / len(cols)).round(3))
    print(pd.DataFrame(table).round(3).to_string(index=False))
    print(pd.DataFrame({'type': names, 'n': np.bincount(new, minlength=a.k), 'jaccard': jac.round(3)[order],
                        **{t: centers[:, j].round(2) for j, t in enumerate(TIER_EN)}}).to_string(index=False))
    print('ARI with source types', round(report['ari_with_source_types'], 3))
    print(pd.crosstab(pd.Series([src['type_names'][c] for c in old], name='source'),
                      pd.Series([names[c] for c in new], name='rotated')).to_string())
    print('saved to', out)


if __name__ == '__main__':
    main()
