"""KMO robustness table for the nine supply indicators (decides PCA composite vs tier ruler).

Transform: by indicator definition (log1p of RA_SH, SA_SH, RA_CHC, SA_CHC, as in the earlier table),
Yeo-Johnson for |skew| > 1, or none; road-disconnected communities kept or excluded. 1000 bootstrap
resamples of communities give the 95% interval. Rule set by the user: if every combination has KMO >= 0.6
and the interval lower bound >= 0.6, the PCA composite is the main ruler; otherwise the tier ruler.

python lineb/kmo_robustness.py --supply <xlsx> --disconnected-from <xlsx of the 70/40 table> --out <csv>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ruler import kmo_bartlett  # noqa: E402

COLS = ['RA_TH', 'SA_TH_per_10000', 'TC_TH', 'RA_SH', 'SA_SH_per_10000', 'TC_SH', 'RA_CHC', 'SA_CHC_per_10000', 'TC_CHC']
DEFINITION_LOG = ['RA_SH', 'SA_SH_per_10000', 'RA_CHC', 'SA_CHC_per_10000']


def transform(X, mode):
    X = X.copy()
    done = []
    for j, c in enumerate(COLS):
        if mode == 'definition' and c in DEFINITION_LOG:
            X[:, j] = np.log1p(X[:, j])
            done.append(f'{c}:log1p')
        elif mode == 'yeo-johnson' and abs(stats.skew(X[:, j])) > 1:
            X[:, j] = stats.yeojohnson(X[:, j])[0]
            done.append(f'{c}:yj')
    return X, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--supply', required=True)
    ap.add_argument('--disconnected-from', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--boots', type=int, default=1000)
    a = ap.parse_args()
    s = pd.read_excel(a.supply)
    ref = pd.read_excel(a.disconnected_from).set_index('community_id').loc[s.community_id]
    disconnected = (ref['RA_TH'].to_numpy() == 0)
    rng = np.random.default_rng(0)
    rows = []
    for mode, label in (('definition', '按指标定义'), ('yeo-johnson', 'Yeo–Johnson'), ('none', '不变换')):
        for drop, dlabel in ((False, '保留'), (True, '排除')):
            keep = ~disconnected if drop else np.ones(len(s), bool)
            X, done = transform(s.loc[keep, COLS].to_numpy(float), mode)
            Z = (X - X.mean(0)) / X.std(0)
            kmo, chi2, df, p = kmo_bartlett(Z)
            boot = []
            for _ in range(a.boots):
                idx = rng.integers(0, len(Z), len(Z))
                Xb = X[idx]
                sd = Xb.std(0)
                if (sd == 0).any():
                    continue
                boot.append(kmo_bartlett((Xb - Xb.mean(0)) / sd)[0])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({'变换方式': label, '道路不连通社区': dlabel, '社区数': int(keep.sum()), 'KMO': round(kmo, 3),
                         '95%CI下限': round(lo, 3), '95%CI上限': round(hi, 3), 'Bartlett p': p,
                         '最大|z|': round(float(np.abs(Z).max()), 2), '变换的指标': ','.join(done)})
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('道路不连通社区:', s.loc[disconnected, 'community_name'].tolist())
    print(R.drop(columns='变换的指标').to_string(index=False))
    ok = (R.KMO >= 0.6).all() and (R['95%CI下限'] >= 0.6).all()
    print('判定:', 'PCA综合标尺为主线' if ok else '分层标尺为主线（PCA综合标尺作对照）')


if __name__ == '__main__':
    main()
