"""Nominal supply ruler (线B §2.2): canonical PCA + KMeans, fitted once on nominal indicators and frozen.

measure() only transforms and predicts with the frozen parameters, so a
category change can only come from a change in the indicators.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

TIERS = ('TH', 'SH', 'CHC')


def kmo_bartlett(Z):
    n, p = Z.shape
    R = np.corrcoef(Z, rowvar=False)
    inv = np.linalg.pinv(R)
    partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    off = ~np.eye(p, dtype=bool)
    r2, q2 = (R[off] ** 2).sum(), (partial[off] ** 2).sum()
    kmo = r2 / (r2 + q2)
    sign, logdet = np.linalg.slogdet(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    df = p * (p - 1) / 2
    return float(kmo), float(chi2), float(df), float(stats.chi2.sf(chi2, df))


def parallel_analysis(n, p, reps=1000, q=95, seed=0):
    rng = np.random.default_rng(seed)
    eig = np.empty((reps, p))
    for r in range(reps):
        eig[r] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
    return np.percentile(eig, q, axis=0)


def jaccard_stability(F, k, boots=200, seed=0):
    """Cluster-wise bootstrap Jaccard (Hennig 2007): mean over bootstraps of the best match per original cluster."""
    base = KMeans(k, n_init=50, random_state=0).fit_predict(F)
    rng = np.random.default_rng(seed)
    n = len(F)
    scores = np.zeros((boots, k))
    for b in range(boots):
        idx = rng.integers(0, n, n)
        uniq = np.unique(idx)
        lab = KMeans(k, n_init=10, random_state=b).fit(F[idx]).predict(F[uniq])
        for c in range(k):
            a = set(uniq[base[uniq] == c])
            if not a:
                continue
            scores[b, c] = max(len(a & set(uniq[lab == m])) / len(a | set(uniq[lab == m])) for m in range(k))
    return scores.mean(0)


def gap_statistic(F, ks, refs=100, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = F.min(0), F.max(0)
    out = {}
    for k in ks:
        w = np.log(KMeans(k, n_init=10, random_state=0).fit(F).inertia_)
        ref = np.array([np.log(KMeans(k, n_init=3, random_state=r).fit(rng.uniform(lo, hi, F.shape)).inertia_)
                        for r in range(refs)])
        out[k] = (float(ref.mean() - w), float(ref.std(ddof=1) * np.sqrt(1 + 1 / refs)))
    return out


def select_k(F, candidates=range(3, 9), boots=200, threshold=0.75, seed=0):
    table = []
    for k in candidates:
        jac = jaccard_stability(F, k, boots, seed)
        lab = KMeans(k, n_init=50, random_state=0).fit_predict(F)
        table.append({'k': k, 'jaccard_mean': float(jac.mean()), 'jaccard_min': float(jac.min()),
                      'silhouette': float(silhouette_score(F, lab)), 'calinski_harabasz': float(calinski_harabasz_score(F, lab)),
                      'davies_bouldin': float(davies_bouldin_score(F, lab))})
    table = pd.DataFrame(table)
    gap = gap_statistic(F, list(candidates) + [max(candidates) + 1], seed=seed)
    table['gap'] = [gap[k][0] for k in table.k]
    table['gap_se'] = [gap[k][1] for k in table.k]
    stable = table[table.jaccard_mean >= threshold]
    if stable.empty:
        return None, table, {}
    ks = stable.k.tolist()
    votes = {'silhouette': int(stable.loc[stable.silhouette.idxmax(), 'k']),
             'calinski_harabasz': int(stable.loc[stable.calinski_harabasz.idxmax(), 'k']),
             'davies_bouldin': int(stable.loc[stable.davies_bouldin.idxmin(), 'k'])}
    gap_pick = next((k for k in ks if gap[k][0] >= gap[k + 1][0] - gap[k + 1][1]), max(ks))
    votes['gap'] = int(gap_pick)
    counts = pd.Series(list(votes.values())).value_counts()
    best = int(min(counts[counts == counts.max()].index))
    return best, table, votes


def fit_ruler(raw, columns, original_classes=None, pa_reps=1000, boots=200, seed=0, transform_mode='yeo-johnson',
              log_form=(), force_mode=None, k_override=None):
    report = {'columns': list(columns), 'transform_mode': transform_mode, 'log_form_columns': list(log_form),
              'force_mode': force_mode, 'k_override': k_override}
    skew = stats.skew(raw, axis=0)
    report['describe'] = pd.DataFrame(raw, columns=columns).describe().T.round(6).to_dict()
    report['skewness'] = dict(zip(columns, map(float, skew)))
    lambdas = {}
    if transform_mode == 'yeo-johnson':
        for j, (c, sk) in enumerate(zip(columns, skew)):
            if abs(sk) > 1 and c not in log_form:
                lambdas[c] = float(stats.yeojohnson(raw[:, j])[1])
    report['yeo_johnson_lambda'] = lambdas
    frozen = {'columns': list(columns), 'lambdas': lambdas}
    T = transform(raw, frozen)
    mean, std = T.mean(0), T.std(0)
    frozen.update(mean=mean, std=std)
    Z = (T - mean) / std
    report['skewness_after'] = dict(zip(columns, map(float, stats.skew(Z, axis=0))))
    report['max_abs_z'] = dict(zip(columns, map(float, np.abs(Z).max(0))))
    kmo, chi2, df, p = kmo_bartlett(Z)
    report.update(kmo=kmo, bartlett_chi2=chi2, bartlett_df=df, bartlett_p=p, suitable=bool(kmo >= 0.6 and p < 0.05))
    R = np.corrcoef(Z, rowvar=False)
    eigval, eigvec = np.linalg.eigh(R)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    pa = parallel_analysis(*Z.shape, reps=pa_reps, seed=seed)
    m_pa = max(1, int((eigval > pa).sum()))
    m_kaiser = int((eigval > 1).sum())
    report.update(eigenvalues=eigval.tolist(), parallel_95=pa.tolist(), components_parallel=m_pa, components_kaiser=m_kaiser)
    m = m_pa
    U = eigvec[:, :m].copy()
    loadings = U * np.sqrt(eigval[:m])
    sign = np.where(loadings.sum(0) < 0, -1.0, 1.0)
    U, loadings = U * sign, loadings * sign
    weights = eigval[:m] / eigval[:m].sum()
    report.update(loadings=pd.DataFrame(loadings, index=columns, columns=[f'PC{j + 1}' for j in range(m)]).round(3).to_dict(),
                  variance_weights=weights.tolist(), variance_explained=(eigval[:m] / eigval.sum()).tolist())
    mode = force_mode or ('composite' if report['suitable'] else 'tiers')
    if mode == 'composite':
        frozen.update(mode='composite', U=U, weights=weights)
        space = Z @ U
    else:
        frozen.update(mode='tiers', **tier_ruler(Z, columns, report))
        space = (tier_scores(Z, frozen) - frozen['tier_mean']) / frozen['tier_std']
    k, table, votes = select_k(space, boots=boots, seed=seed)
    report['k_selection'] = table.to_dict('records')
    report['k_votes'] = votes
    report['k_selected_by_rule'] = k
    if k_override is not None:
        k = int(k_override)  # set by the user after reviewing stability
    report['k'] = k
    frozen['centers'] = None
    if k is None:
        report['N1'] = False
        return frozen, report
    km = KMeans(k, n_init=50, random_state=0).fit(space)
    report['ari_across_inits'] = float(np.mean([adjusted_rand_score(KMeans(k, n_init=50, random_state=a).fit_predict(space),
                                                                    KMeans(k, n_init=50, random_state=b).fit_predict(space))
                                                for a, b in [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]]))
    if frozen['mode'] == 'composite':
        centre_score = km.cluster_centers_ @ weights
        order_ = np.argsort(centre_score)
    else:
        order_ = np.lexsort(km.cluster_centers_.T[::-1])
    frozen['centers'] = km.cluster_centers_[order_]
    labels = measure(Z, frozen)['class']
    selected = next(r for r in report['k_selection'] if r['k'] == k)
    report['class_sizes'] = pd.Series(labels).value_counts().sort_index().to_dict()
    if frozen['mode'] == 'tiers':
        names = type_names(frozen)
        frozen['type_names'] = names
        report['type_names'] = names
        report['type_centres_standardized'] = frozen['centers'].round(3).tolist()
        report['N1'] = bool(selected['jaccard_mean'] >= 0.75)
    else:
        report['class_profiles'] = pd.DataFrame(tier_components(Z), columns=TIERS).groupby(labels).mean().round(3).to_dict('index')
        report['N1'] = bool((report['suitable'] or force_mode == 'composite') and selected['jaccard_mean'] >= 0.75)
    if original_classes is not None:
        report['crosstab_original_k5'] = pd.crosstab(labels, original_classes).to_dict()
    return frozen, report


def tier_ruler(Z, columns, report):
    """Within-tier PCA: first component if it explains at least 50% of the tier's variance, else equal weights."""
    W = np.zeros((Z.shape[1], 3))
    info = {}
    for t, name in enumerate(TIERS):
        block = Z[:, 3 * t:3 * t + 3]
        val, vec = np.linalg.eigh(np.corrcoef(block, rowvar=False))
        u = vec[:, np.argmax(val)]
        u = u if u.sum() >= 0 else -u
        explained = float(val.max() / val.sum())
        w = u if explained >= 0.5 else np.full(3, 1 / 3)
        W[3 * t:3 * t + 3, t] = w
        info[name] = {'pc1_explained': explained, 'used': 'pc1' if explained >= 0.5 else 'equal_mean',
                      'weights': dict(zip(columns[3 * t:3 * t + 3], map(float, w)))}
    report['tier_scores'] = info
    T = Z @ W
    cuts = np.percentile(T, [100 / 3, 200 / 3], axis=0)
    report['tier_tertile_cuts'] = dict(zip(TIERS, cuts.T.tolist()))
    return {'tier_W': W, 'tier_mean': T.mean(0), 'tier_std': T.std(0), 'tier_cuts': cuts}


def tier_scores(Z, frozen):
    return Z @ frozen['tier_W']


def tier_levels(T, frozen):
    return (T[:, None, :] > frozen['tier_cuts'][None, :, :]).sum(1)  # 0 low, 1 mid, 2 high


def type_names(frozen):
    """Name configuration types by the level of each tier at the type centre (no ranking between types)."""
    centres = frozen['centers'] * frozen['tier_std'] + frozen['tier_mean']
    levels = tier_levels(centres, frozen)
    word = ('低', '中', '高')
    out = []
    for lv in levels:
        profile = '·'.join(f'{n}{word[v]}' for n, v in zip(('三甲', '二级', '社康'), lv))
        high = [n for n, v in zip(('三甲', '二级', '社康'), lv) if v == 2]
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
        out.append(f'{short}（{profile}）')
    return out


def transform(raw, frozen):
    T = np.array(raw, dtype=float).copy()
    for j, c in enumerate(frozen['columns']):
        if c in frozen['lambdas']:
            T[:, j] = stats.yeojohnson(T[:, j], lmbda=frozen['lambdas'][c])
    return T


def standardize(raw, frozen):
    return (transform(raw, frozen) - frozen['mean']) / frozen['std']


def tier_components(Z):
    return Z.reshape(len(Z), 3, 3).mean(2)


def measure(Z, frozen):
    """Measure standardized indicators with the frozen ruler (composite mode or tier mode)."""
    if frozen['mode'] == 'composite':
        F = Z @ frozen['U']
        out = {'composite': F @ frozen['weights'], 'pc': F, 'tiers': tier_components(Z)}
        space = F
    else:
        T = tier_scores(Z, frozen)
        out = {'composite': None, 'pc': None, 'tiers': T, 'levels': tier_levels(T, frozen)}
        space = (T - frozen['tier_mean']) / frozen['tier_std']
    if frozen.get('centers') is not None:
        d = np.sqrt(((space[:, None, :] - frozen['centers'][None]) ** 2).sum(-1))
        part = np.sort(d, 1)
        out['class'] = d.argmin(1)
        out['boundary'] = part[:, 1] - part[:, 0]
    return out
