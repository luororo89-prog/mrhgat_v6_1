"""Per-channel decoders back to the nine standardized indicators (线B §2.7, §3.1)."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from ruler import measure

ALPHAS = np.logspace(-4, 3, 15)


def fit_decoders(z, x):
    """Ridge decoder D_k: z_{i,k} -> x_i, fitted after routing is frozen."""
    return [RidgeCV(alphas=ALPHAS).fit(z[:, k], x) for k in range(z.shape[1])]


def decode(decoders, k, v):
    return decoders[k].predict(v)


def decoder_check(z, x, frozen, fold, columns):
    """Spatial-block 5-fold held-out R2 per indicator and class agreement under the frozen ruler."""
    nominal = measure(x, frozen)
    result = {}
    for k in range(z.shape[1]):
        pred = np.zeros_like(x)
        for f in np.unique(fold):
            test = fold == f
            pred[test] = RidgeCV(alphas=ALPHAS).fit(z[~test, k], x[~test]).predict(z[test, k])
        decoded = measure(pred, frozen)
        result[k] = {'heldout_r2': {c: float(r2_score(x[:, j], pred[:, j])) for j, c in enumerate(columns)},
                     'heldout_tier_r2': {t: float(r2_score(nominal['tiers'][:, j], decoded['tiers'][:, j]))
                                         for j, t in enumerate(('TH', 'SH', 'CHC'))},
                     'class_agreement': (float((decoded['class'] == nominal['class']).mean())
                                         if 'class' in nominal else None),
                     'insample_r2': float(RidgeCV(alphas=ALPHAS).fit(z[:, k], x).score(z[:, k], x))}
    return result
