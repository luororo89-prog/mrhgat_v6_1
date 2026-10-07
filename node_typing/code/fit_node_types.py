"""Node types from the nine supply indicators: PCA composite ruler (5 types) as the main fixed scale.

The ruler-fitting part of lineb/measure_pca.py. The nine nominal indicators are transformed (Yeo-Johnson for
|skew| > 1), standardized, projected on the parallel-analysis components and clustered with KMeans (k = 5).
Types are named by the tertile level of each tier component (mean of the tier's three standardized
indicators). The tier ruler (5 types) is fitted as the comparison. Output: ruler.json (frozen parameters and
report, same layout as results_lineb/walk_v4_pca5/ruler.json) and node_types.csv (type of every community).

python node_typing/code/fit_node_types.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ruler import fit_ruler, standardize, measure  # noqa: E402

SUPPLY = '/Users/luoxuan/Documents/博后三胎/input/联通人口_新的九项医疗供给指标_2018_等时圈.xlsx'
OUT = HERE.parent / 'results'
TIERS = ('TH', 'SH', 'CHC')
TIER_CN = ('三甲', '二级', '社康')
COLUMNS = [c for t in TIERS for c in (f'RA_{t}', f'SA_{t}_per_10000', f'TC_{t}')]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def composite_names(frozen, Z):
    """Name each type by the tertile level of its members' mean tier component (mean of the tier's three
    standardized indicators), the same wording as the tier ruler."""
    tiers = Z.reshape(len(Z), 3, 3).mean(2)
    cuts = np.percentile(tiers, [100 / 3, 200 / 3], axis=0)
    cls = measure(Z, frozen)['class']
    centres = np.stack([tiers[cls == c].mean(0) for c in range(len(frozen['centers']))])
    levels = (centres[:, None, :] > cuts[None, :, :]).sum(1)
    word = ('低', '中', '高')
    out = []
    for lv in levels:
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
        out.append(f'{short}（{profile}）')
    return out, cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--supply', default=SUPPLY)
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_excel(a.supply)
    raw = s[COLUMNS].to_numpy(float)
    common = dict(pa_reps=1000, boots=200, transform_mode='yeo-johnson', log_form=[])
    main_r, main_rep = fit_ruler(raw, COLUMNS, None, force_mode='composite', k_override=5, **common)
    cmp_r, cmp_rep = fit_ruler(raw, COLUMNS, None, force_mode='tiers', k_override=5, **common)
    x = standardize(raw, main_r)
    assert np.allclose(x, standardize(raw, cmp_r))
    names, cuts = composite_names(main_r, x)
    main_r['type_names'] = names
    main_rep['type_names'] = names
    main_rep['component_tertile_cuts'] = cuts.tolist()
    (out / 'ruler.json').write_text(json.dumps(jsonable({'main': {**main_rep, 'frozen': main_r}, 'comparison': {**cmp_rep, 'frozen': cmp_r}}),
                                               ensure_ascii=False, indent=1), encoding='utf-8')
    k5 = next(r for r in main_rep['k_selection'] if r['k'] == 5)
    print(f"PCA ruler: KMO {main_rep['kmo']:.3f}, components {main_rep['components_parallel']}, k=5 Jaccard {k5['jaccard_mean']:.3f} "
          f"(min {k5['jaccard_min']:.3f}), N1 {main_rep['N1']}")
    nominal = measure(x, main_r)
    print('types', list(zip(names, np.bincount(nominal['class'], minlength=5).tolist())))
    pd.DataFrame({'community_id': s['community_id'], 'community_name': s['community_name'],
                  'type_id': nominal['class'], 'type_name': [names[c] for c in nominal['class']],
                  **{f'PC{j + 1}': nominal['pc'][:, j] for j in range(nominal['pc'].shape[1])}}
                 ).to_csv(out / 'node_types.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
