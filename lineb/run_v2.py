"""Line B with the figure-version DisenGAT (model_v2): train, effective supply, channel attribution,
comparison with the reported model under the same frozen rulers.

python lineb/run_v2.py --config lineb/configs/walk_v4_k5_v2.yaml
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from graph import build_graph, load_labels, ADJ, CRS, MECHANISMS  # noqa: E402
from ruler import standardize, measure, TIERS  # noqa: E402
from model import as_tensors, typed  # noqa: E402
from model_v2 import train_v2, effective  # noqa: E402

LV = {0: '低', 1: '中', 2: '高'}


def frozen(run_dir):
    out = {}
    for role, v in json.loads((run_dir / 'ruler.json').read_text(encoding='utf-8')).items():
        f = dict(v['frozen'])
        for key in ('mean', 'std', 'U', 'weights', 'tier_W', 'tier_mean', 'tier_std', 'tier_cuts', 'centers'):
            if f.get(key) is not None:
                f[key] = np.asarray(f[key], dtype=float)
        out[role] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    root = HERE.parent
    out = root / 'results_lineb' / cfg['name']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    logf = (out / 'progress.log').open('a', encoding='utf-8')

    def log(msg):
        line = f'{datetime.now():%H:%M:%S} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    ref_dir = root / cfg['ruler_from']
    F = frozen(ref_dir)
    (out / 'ruler.json').write_text((ref_dir / 'ruler.json').read_text(encoding='utf-8'), encoding='utf-8')
    g = build_graph(cfg)
    x = standardize(g['raw'], F['main'])
    g['x'] = x
    lab_np = load_labels(cfg['labels_file'], g)
    lab = typed(as_tensors(lab_np))
    log(f"labels {len(lab_np['origin'])} edges {np.bincount(lab_np['mech'], minlength=3).tolist()}")
    W, K = g['W'], g['K']
    n = len(W)
    offm = ~np.eye(n, dtype=bool)
    share = np.divide(W, W.sum(1, keepdims=True), out=np.zeros_like(W), where=W.sum(1, keepdims=True) > 0)
    rows = np.asarray(g['origins'])
    nominal = measure(x, F['main'])
    names = F['main']['type_names']
    S_w = F['comparison']['U'] @ F['comparison']['weights']
    tier_W = F['main']['tier_W']
    per = []
    for seed in cfg['seeds']:
        t0 = time.time()
        model, hist = train_v2(g, lab_np, seed, epochs=cfg['epochs'], lr=cfg['lr'], decay=cfg['decay'], margin=cfg['margin'],
                               weight=cfg['anchor_weight'], rec_weight=cfg['rec_weight'], log=log, model_kwargs=cfg['model_kwargs'])
        rdir = out / 'runs' / f'seed{seed}'
        rdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), rdir / 'model.pt')
        pd.DataFrame(hist).to_csv(rdir / 'history.csv', index=False)
        e = effective(model, g, lab, share)
        e_adj = effective(model, g, lab, share, mask_topology=(K == ADJ).astype(float))
        e_crs = effective(model, g, lab, share, mask_topology=(K == CRS).astype(float))
        x_eff = x + e['delta']
        m_eff = measure(x_eff, F['main'])
        m_hat = measure(e['x_hat'], F['main'])
        post = e['posterior']
        ce = float(-(share[rows] * e['logpi'][rows]).sum(1).mean())
        agree = (post[lab_np['origin'], :, lab_np['dest']].argmax(1) == lab_np['mech'])
        fw = W * offm
        r2 = 1 - ((e['x_hat'] - x) ** 2).sum(0) / ((x - x.mean(0)) ** 2).sum(0)
        per.append({'seed': seed, 'ce': ce, 'anchor_agree': float(agree.mean()),
                    **{f'anchor_agree_{m}': float(agree[lab_np['mech'] == k].mean()) for k, m in enumerate(MECHANISMS)},
                    **{f'share_{m}': float((fw * post[:, k, :]).sum() / fw.sum()) for k, m in enumerate(MECHANISMS)},
                    'rec_r2_mean': float(r2.mean()), 'rec_class_agreement': float((m_hat['class'] == nominal['class']).mean()),
                    'class': m_eff['class'], 'levels': m_eff['levels'], 'dT': m_eff['tiers'] - nominal['tiers'],
                    'dS': e['delta'] @ S_w, 'shap_S': e['shapley'] @ S_w,               # [n, 3]
                    'shap_T': np.einsum('ikj,jt->ikt', e['shapley'], tier_W),            # [n, 3, 3]
                    'adj_S': e_adj['delta'] @ S_w, 'crs_S': e_crs['delta'] @ S_w,
                    'seconds': time.time() - t0})
        log(f"seed {seed} ce {ce:.4f} anchor {agree.mean():.3f} rec R2 {r2.mean():.3f} rec class {per[-1]['rec_class_agreement']:.3f} "
            f"shares {[round(per[-1][f'share_{m}'], 3) for m in MECHANISMS]}")
    seeds = pd.DataFrame([{k: v for k, v in p.items() if not isinstance(v, np.ndarray)} for p in per])
    seeds.to_csv(out / 'seed_summary.csv', index=False)
    cls = np.stack([p['class'] for p in per])
    lv = np.stack([p['levels'] for p in per])
    modal = np.array([np.bincount(c, minlength=len(names)).argmax() for c in cls.T])
    robust = ((cls == modal[None]).sum(0) >= 4) & (modal != nominal['class'])
    lv_modal = np.stack([[np.bincount(l, minlength=3).argmax() for l in lv[:, :, j].T] for j in range(3)], 1)
    lv_robust = ((lv == lv_modal[None]).sum(0) >= 4) & (lv_modal != nominal['levels'])
    dS = np.mean([p['dS'] for p in per], 0)
    dT = np.mean([p['dT'] for p in per], 0)
    shapS = np.mean([p['shap_S'] for p in per], 0)
    shapT = np.mean([p['shap_T'] for p in per], 0)
    adjS = np.mean([p['adj_S'] for p in per], 0)
    crsS = np.mean([p['crs_S'] for p in per], 0)
    frame = pd.DataFrame({'community_id': g['ids'], 'name': g['names'], 'type_nominal': [names[c] for c in nominal['class']],
                          'type_adjusted_modal': [names[c] for c in modal], 'robust_type_change': robust, 'dS': dS,
                          **{f'dS_{m}': shapS[:, k] for k, m in enumerate(MECHANISMS)},
                          'dS_adjacent_only': adjS, 'dS_cross_only': crsS})
    for j, t in enumerate(TIERS):
        frame[f'd_{t}'] = dT[:, j]
        for k, m in enumerate(MECHANISMS):
            frame[f'd_{t}_{m}'] = shapT[:, k, j]
        frame[f'{t}_level_nominal'] = [LV[v] for v in nominal['levels'][:, j]]
        frame[f'{t}_level_adjusted_modal'] = [LV[v] for v in lv_modal[:, j]]
        frame[f'{t}_robust_level_change'] = lv_robust[:, j]
    frame.to_csv(out / 'nominal_vs_adjusted.csv', index=False, encoding='utf-8-sig')
    tr = pd.crosstab(frame.type_nominal, frame.type_adjusted_modal)
    tr.to_csv(out / 'transition.csv', encoding='utf-8-sig')
    by = []
    for c, nm in enumerate(names):
        mk = nominal['class'] == c
        f = frame[mk]
        row = {'type': nm, 'n': int(mk.sum()), 'dS': f.dS.mean(), 'rise_share': (f.dS > 0).mean(),
               'robust_type_changes': int(f.robust_type_change.sum()),
               **{f'dS_{m}': f[f'dS_{m}'].mean() for m in MECHANISMS},
               'dS_adjacent_only': f.dS_adjacent_only.mean(), 'dS_cross_only': f.dS_cross_only.mean()}
        for t in TIERS:
            row[f'd_{t}'] = f[f'd_{t}'].mean()
            for m in MECHANISMS:
                row[f'd_{t}_{m}'] = f[f'd_{t}_{m}'].mean()
        by.append(row)
    B = pd.DataFrame(by)
    B.to_csv(out / 'by_type.csv', index=False, encoding='utf-8-sig')
    ref = pd.read_csv(ref_dir / 'nominal_vs_adjusted.csv')
    refc = pd.read_csv(ref_dir / 'comparison_composite.csv')
    comp = {'same_adjusted_type_as_reported': float((frame.type_adjusted_modal.values == ref.class_adjusted_modal.values).mean()),
            'corr_dS_vs_reported': float(np.corrcoef(dS, refc.dS_mean.values)[0, 1]),
            'robust_type_changes': int(robust.sum()), 'robust_type_changes_reported': int(ref.robust_class_change.sum()),
            'jaccard_robust_vs_reported': float((robust & ref.robust_class_change.values).sum() / max((robust | ref.robust_class_change.values).sum(), 1)),
            **{f'corr_d{t}_vs_reported': float(spearmanr(dT[:, j], ref[f'tier_{t}_delta']).statistic) for j, t in enumerate(TIERS)},
            **{f'{t}_level_up/down': f"{int((lv_robust[:, j] & (lv_modal[:, j] > nominal['levels'][:, j])).sum())}/"
                                     f"{int((lv_robust[:, j] & (lv_modal[:, j] < nominal['levels'][:, j])).sum())}" for j, t in enumerate(TIERS)},
            **{f'{t}_seed_agreement': float(np.mean([spearmanr(per[a_]['dT'][:, j], per[b_]['dT'][:, j]).statistic
                                                    for a_ in range(len(per)) for b_ in range(a_ + 1, len(per))])) for j, t in enumerate(TIERS)}}
    (out / 'status.json').write_text(json.dumps({**comp, **seeds.drop(columns='seed').mean().to_dict()}, ensure_ascii=False, indent=1), encoding='utf-8')
    log('done ' + json.dumps(comp, ensure_ascii=False))
    pd.set_option('display.width', 250)
    print(seeds.round(4).to_string(index=False))
    print(B.round(3).T.to_string())
    print(tr.to_string())


if __name__ == '__main__':
    main()
