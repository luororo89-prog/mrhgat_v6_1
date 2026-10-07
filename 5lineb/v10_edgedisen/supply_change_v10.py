"""Supply change of the trained model_v10 runs: the joint decoder is fitted on the self loop vectors and reads the
concatenated aggregated vectors.

Per seed (channels aligned as in run_v9, channel_order in summary.csv), with all observed outflow edges as messages:
    D: ridge decoder [s_u,1, ..., s_u,K] -> x_u, fitted on the self loop vectors of all communities
    dx_u = D([c_u,1, ..., c_u,K]) - D([s_u,1, ..., s_u,K]),   dx_u,k = change when only block k is replaced by c_u,k
Changes are in standardised indicator units (the fixed ruler's transformation and scaling); tier changes are the mean
of the three indicators of the tier (TH, SH, CHC).
Outputs
  supply_change_overall.csv     mean change and mean absolute change per indicator and tier, seed range
  supply_change_by_type.csv     mean tier change per nominal type, seed range, sign agreed by all seeds
  supply_change_channel_by_type.csv   the same split into the channels
  supply_change_agreement.csv   Spearman correlation of the community changes between every pair of seeds, per tier
  supply_change_community.csv   per community: mean change and seed standard deviation per indicator and tier

python 5lineb/v10_edgedisen/supply_change_v10.py --config 5lineb/v10_edgedisen/configs/walk_v4_pca5_v10_k3_joint_full.yaml
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from graph import build_graph  # noqa: E402
from ruler import standardize, measure, tier_components, TIERS  # noqa: E402
from decode import ALPHAS  # noqa: E402
from run_v2 import frozen  # noqa: E402
from model_v10 import EdgeDisen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    root = HERE.parents[1]
    out = root / 'results_lineb' / cfg['name']
    K = cfg.get('n_channels', 3)
    CH = [f'C{k + 1}' for k in range(K)]
    R = frozen(root / cfg['ruler_from'])['main']
    type_names = json.loads((root / cfg['ruler_from'] / 'ruler.json').read_text(encoding='utf-8'))['main']['type_names']
    g = build_graph(cfg)
    x = standardize(g['raw'], R)
    W = g['W']
    n = len(W)
    obs = (W > 0) & ~np.eye(n, dtype=bool)
    T = [torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
         torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    nominal = measure(x, R)['class']
    order = {int(r.seed): json.loads(str(r.channel_order)) for r in pd.read_csv(out / 'summary.csv').itertuples()}

    per = {}
    for seed, perm in order.items():
        model = EdgeDisen(iters=cfg.get('iters', 5), n_channels=K, self_weight=cfg.get('self_weight', 'one'),
                          out_weight=cfg.get('out_weight'))
        model.load_state_dict(torch.load(out / f'seed{seed}' / 'model.pt'))
        model.eval()
        with torch.no_grad():
            o = model(*T, share, torch.as_tensor(obs), cfg['tau_final'])
        s, c = o['zr'].numpy().astype(float)[:, perm], o['c'].numpy().astype(float)[:, perm]
        S, Cc = s.reshape(n, -1), c.reshape(n, -1)
        dec = RidgeCV(alphas=ALPHAS).fit(S, x)
        base = dec.predict(S)
        d = s.shape[2]
        parts = []
        for k in range(K):
            Sk = S.copy()
            Sk[:, k * d:(k + 1) * d] = Cc[:, k * d:(k + 1) * d]
            parts.append(dec.predict(Sk) - base)
        per[seed] = {'dx': dec.predict(Cc) - base, 'parts': parts}
        print(f'seed {seed} done', flush=True)
    seeds = list(per)
    cols = list(g['columns'])
    DX = np.stack([per[s]['dx'] for s in seeds])
    DT = np.stack([tier_components(per[s]['dx']) for s in seeds])
    PT = np.stack([[tier_components(p) for p in per[s]['parts']] for s in seeds])

    rows = []
    for j, c in enumerate(cols):
        m = DX[:, :, j].mean(1)
        rows.append({'item': c, 'mean_change': m.mean(), 'seed_min': m.min(), 'seed_max': m.max(),
                     'mean_abs_change': np.abs(DX[:, :, j]).mean()})
    for j, t in enumerate(TIERS):
        m = DT[:, :, j].mean(1)
        rows.append({'item': f'tier_{t}', 'mean_change': m.mean(), 'seed_min': m.min(), 'seed_max': m.max(),
                     'mean_abs_change': np.abs(DT[:, :, j]).mean()})
    pd.DataFrame(rows).to_csv(out / 'supply_change_overall.csv', index=False)

    def by_type(arr, extra):
        res = []
        for ci, nm in enumerate(type_names):
            mk = nominal == ci
            for j, t in enumerate(TIERS):
                m = arr[:, mk, j].mean(1)
                res.append({**extra, 'type': nm, 'n': int(mk.sum()), 'tier': t, 'mean_change': m.mean(), 'seed_min': m.min(),
                            'seed_max': m.max(), 'sign_agreed': bool((m > 0).all() or (m < 0).all())})
        return res
    pd.DataFrame(by_type(DT, {})).to_csv(out / 'supply_change_by_type.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([r for k, ch in enumerate(CH) for r in by_type(PT[:, k], {'channel': ch})]).to_csv(
        out / 'supply_change_channel_by_type.csv', index=False, encoding='utf-8-sig')
    agree = [{'seed_a': a_, 'seed_b': b_, **{f'spearman_{t}': spearmanr(DT[i, :, j], DT[k, :, j]).statistic for j, t in enumerate(TIERS)}}
             for (i, a_), (k, b_) in itertools.combinations(enumerate(seeds), 2)]
    pd.DataFrame(agree).to_csv(out / 'supply_change_agreement.csv', index=False)
    com = pd.DataFrame({'community_id': g['ids'], 'name': g['names'], 'type_nominal': [type_names[c] for c in nominal]})
    for j, c in enumerate(cols):
        com[f'dx_{c}'], com[f'dx_{c}_sd'] = DX[:, :, j].mean(0), DX[:, :, j].std(0)
    for j, t in enumerate(TIERS):
        com[f'd_tier_{t}'], com[f'd_tier_{t}_sd'] = DT[:, :, j].mean(0), DT[:, :, j].std(0)
        for k, ch in enumerate(CH):
            com[f'd_tier_{t}_{ch}'] = PT[:, k, :, j].mean(0)
    com.to_csv(out / 'supply_change_community.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(by_type(DT, {})).round(4).to_string(index=False))
    print(pd.DataFrame(agree).round(3).describe().loc[['mean', 'min', 'max']].to_string())
    print(pd.DataFrame([r for k, ch in enumerate(CH) for r in by_type(PT[:, k], {'channel': ch})])
          .pivot_table(index=['type'], columns=['tier', 'channel'], values='mean_change').round(4).to_string())


if __name__ == '__main__':
    main()
