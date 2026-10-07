"""Line B v9: DisenGCN routing (model_v9), the aggregated channel vectors go straight to the decoders, fixed ruler.

1  Train model_v9 on every seed (training itself predicts a random half of the outflow edges from the other half in
   every epoch). Training cross-entropy of a seed = mean over its last 50 epochs.
2  Final pass with all observed outflow edges as messages at the final temperature: routing shares p_uv,k, unit
   channel vectors zr and aggregated vectors c.
3  Align the anonymous channels of every seed to the seed with the lowest training cross-entropy (Hungarian method on
   the flow-weighted correlation of the routing shares). Stability: matched correlations and flow shares across seeds.
4  Per seed and channel: ridge decoder D_k (zr_k -> x), class agreement of its reconstruction of all communities under
   the fixed ruler (A1), and the
   change of the channel alone, dx_k = D_k(c_k) - D_k(zr_k). With decoder: joint in the config one ridge decoder D reads
   the concatenated channel vectors [zr_1, ..., zr_K] (all channels aggregated and decoded together); it is linear, so
   D([c_1, ..., c_K]) - D([zr_1, ..., zr_K]) = sum_k dx_k with dx_k the change when only block k is replaced by c_k,
   and A1 is the class agreement of the reconstruction by D.
5  Channel effects (tier components of dx_k), channel profiles and routing shares per edge are written for every
   channel. Names: channel_names in the config when given; with three channels and no channel_names, the rule below
   from the effects averaged over the aligned seeds. With more channels and no channel_names the run stops here.
   Rule: base changes the own supply least (mean absolute change); of the other two, compensation mainly raises the
   tier the community lacks and competition mainly lowers the tier the community is high in (share of raise-lacking in
   raise-lacking plus lower-highest).
6  x~ = x + sum of dx_k over the channels named comp or compet (base stays), or over every channel when move_all is set
   in the config (each channel's contribution is reported, under its name when channel_names is given), measured with the fixed PCA ruler (component
   levels by nominal tertiles, five types). A change is robust when at least 80% of the seeds agree.

python 5lineb/v9_disengcn/run_v9.py --config 5lineb/v9_disengcn/configs/walk_v4_pca5_v9.yaml
"""
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE.parent / 'v1_disengat'))
sys.path.insert(0, str(HERE.parent / 'v2_figure'))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from graph import build_graph, CRS  # noqa: E402
from ruler import standardize, measure, tier_components  # noqa: E402
from decode import ALPHAS  # noqa: E402
from run_v2 import frozen  # noqa: E402
from unsupervised import profile  # noqa: E402
from model_v9 import train_v9  # noqa: E402

FIT_KEYS = ('out_weight', 'load_from', 'self_weight', 'stop_window', 'stop_tol', 'n_channels', 'iters', 'epochs', 'lr', 'decay', 'rec_weight', 'tau0', 'tau_final', 'n_stages', 'stage_epochs', 'perturb_scale')
NAMES = ('base', 'comp', 'compet')
PCS = ('PC1', 'PC2', 'PC3')
LV = ('低', '中', '高')


def wcorr(a, b, w):
    a = a - (w * a).sum() / w.sum()
    b = b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def main(train=train_v9):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    root = HERE.parents[1]
    out = root / 'results_lineb' / cfg['name']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    logf = (out / 'progress.log').open('a', encoding='utf-8')

    def log(msg):
        line = f'{datetime.now():%H:%M:%S} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    torch.set_num_threads(cfg.get('threads', 8))
    R = frozen(root / cfg['ruler_from'])['main']
    type_names = json.loads((root / cfg['ruler_from'] / 'ruler.json').read_text(encoding='utf-8'))['main']['type_names']
    g = build_graph(cfg)
    x = standardize(g['raw'], R)
    g['x'] = x
    W = g['W']
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    obs = (W > 0) & off
    oi, di = np.nonzero(obs)
    w = W[obs]
    rows = np.asarray(g['origins'])
    share = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-12), dtype=torch.float32)
    T = [torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(g['t'], dtype=torch.float32),
         torch.as_tensor(g['K'], dtype=torch.long), torch.as_tensor(g['candidates'])]
    K = cfg.get('n_channels', 3)
    CH = [f'C{k + 1}' for k in range(K)]
    log(f"flow threshold {g['theta']} retains {g['retained_share']:.3f}; seeds {cfg['seeds']}")

    runs = {}
    for seed in cfg['seeds']:
        model, hist, tau = train(g, seed, log=log, **{k: cfg[k] for k in FIT_KEYS if k in cfg})
        rdir = out / f'seed{seed}'
        rdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), rdir / 'model.pt')
        pd.DataFrame(hist).to_csv(rdir / 'history.csv', index=False)
        with torch.no_grad():
            fu = model(*T, share, torch.as_tensor(obs), tau)
        ce = float(np.mean([h_['ce'] for h_ in hist[-50:]]))
        p = fu['p'].numpy().astype(float) * off[:, None, :]
        runs[seed] = {'ce': ce, 'p': p, 'E': p.transpose(0, 2, 1)[obs], 'zr': fu['zr'].numpy().astype(float),
                      'c': fu['c'].numpy().astype(float)}
        log(f'seed {seed} train CE {ce:.4f} shares {np.round((w[:, None] * runs[seed]["E"]).sum(0) / w.sum(), 3).tolist()}')

    ref_seed = min(runs, key=lambda s: runs[s]['ce'])
    ref = runs[ref_seed]['E']
    for s, r in runs.items():
        C = np.array([[wcorr(ref[:, i], r['E'][:, j], w) for j in range(K)] for i in range(K)])
        _, perm = linear_sum_assignment(-C)
        r.update(perm=perm.tolist(), E=r['E'][:, perm], p=r['p'][:, perm], zr=r['zr'][:, perm], c=r['c'][:, perm])
    seeds = list(runs)
    stab = pd.DataFrame([{'seed_a': sa, 'seed_b': sb, **{f'corr_{CH[k]}': wcorr(runs[sa]['E'][:, k], runs[sb]['E'][:, k], w) for k in range(K)}}
                         for sa, sb in itertools.combinations(seeds, 2)])
    stab.to_csv(out / 'stability.csv', index=False)

    nominal_class = measure(x, R)['class']
    T0 = tier_components(x)
    low, high = T0.argmin(1), T0.argmax(1)
    idx = np.arange(n)
    naming = []
    for s, r in runs.items():
        r['dx'] = []
        if cfg.get('decoder') == 'joint':
            Z = r['zr'].reshape(n, -1)
            dec = RidgeCV(alphas=ALPHAS).fit(Z, x)
            a1 = float((measure(dec.predict(Z), R)['class'] == nominal_class).mean())
            d = r['zr'].shape[2]
        for k in range(K):
            if cfg.get('decoder') == 'joint':
                Zk = Z.copy()
                Zk[:, k * d:(k + 1) * d] = r['c'][:, k]
                dx = dec.predict(Zk) - dec.predict(Z)
            else:
                dec = RidgeCV(alphas=ALPHAS).fit(r['zr'][:, k], x)
                dx = dec.predict(r['c'][:, k]) - dec.predict(r['zr'][:, k])
                a1 = float((measure(dec.predict(r['zr'][:, k]), R)['class'] == nominal_class).mean())
            r['dx'].append(dx)
            dT = tier_components(dx)
            raise_low, lower_high = float(dT[idx, low][rows].mean()), float(-dT[idx, high][rows].mean())
            naming.append({'seed': s, 'channel': CH[k], 'mean_abs_change': float(np.abs(dT[rows]).mean()),
                           'raise_lacking': raise_low, 'lower_high': lower_high,
                           'raise_share': raise_low / (raise_low + lower_high) if raise_low + lower_high > 0 else np.nan,
                           'flow_share': float((w * r['E'][:, k]).sum() / w.sum()),
                           'A1': a1})
    N = pd.DataFrame(naming)
    tiers = (T0 - T0.mean(0)) / T0.std(0)
    O = W.sum(1)
    pressure = np.log1p(np.divide((W * (g['K'] == CRS) * off).sum(0), O, out=np.zeros(n), where=O > 0))
    prof = pd.concat([profile(r['p'], g, tiers, pressure).assign(seed=s) for s, r in runs.items()])
    prof['channel'] = prof['channel'].map(dict(enumerate(CH)))
    Pm = np.stack([r['E'] for r in runs.values()])
    edge = pd.DataFrame({'origin_id': g['ids'][oi], 'destination_id': g['ids'][di],
                         'topology': [('INT', 'ADJ', 'CRS')[int(v)] for v in g['K'][oi, di]],
                         'flow': w, 'distance_km': g['distance_km'][oi, di]})
    for k, ch in enumerate(CH):
        edge[f'p_{ch}'] = Pm[:, :, k].mean(0)
        edge[f'p_{ch}_range'] = Pm[:, :, k].max(0) - Pm[:, :, k].min(0)
    shares = {s: (w[:, None] * r['E']).sum(0) / w.sum() for s, r in runs.items()}
    pd.DataFrame([{'seed': s, 'train_ce': runs[s]['ce'], 'channel_order': runs[s]['perm'],
                   **{f'share_{ch}': float(shares[s][k]) for k, ch in enumerate(CH)}} for s in seeds]).to_csv(out / 'summary.csv', index=False)
    status = {'n_channels': K, 'reference_seed': ref_seed, 'train_ce': [runs[s]['ce'] for s in seeds],
              **{f'stability_min_corr_{ch}': float(stab[f'corr_{ch}'].min()) for ch in CH},
              **{f'stability_mean_corr_{ch}': float(stab[f'corr_{ch}'].mean()) for ch in CH},
              **{f'share_{ch}': [round(float(shares[s][k]), 4) for s in seeds] for k, ch in enumerate(CH)},
              **{f'A1_min_{ch}': float(N[N.channel == ch].A1.min()) for ch in CH}}

    def name(frame):
        base = int(frame.mean_abs_change.to_numpy().argmin())
        rest = [k for k in range(3) if k != base]
        comp = max(rest, key=lambda k: frame.raise_share.iloc[k])
        return {base: 'base', comp: 'comp', [k for k in rest if k != comp][0]: 'compet'}

    if cfg.get('move_all') and not cfg.get('channel_names'):
        names = None
    elif cfg.get('channel_names'):
        names = {k: cfg['channel_names'][ch] for k, ch in enumerate(CH)}
    elif K == 3:
        names = name(N.groupby('channel', sort=True)[['mean_abs_change', 'raise_share']].mean())
        for s in seeds:
            own = name(N[N.seed == s].sort_values('channel').reset_index(drop=True))
            N.loc[N.seed == s, 'name_this_seed'] = N.loc[N.seed == s, 'channel'].map(lambda c: own[CH.index(c)])
        status['naming_agreement_across_seeds'] = float((N.channel.map(lambda c: names[CH.index(c)]) == N.name_this_seed).mean())
    else:
        names = None
    if names:
        N['name'] = N.channel.map(lambda c: names[CH.index(c)])
        prof['mechanism'] = prof['channel'].map(lambda c: names[CH.index(c)])
        for m in NAMES:
            edge[f'p_{m}'] = edge[[f'p_{CH[k]}' for k in range(K) if names[k] == m]].sum(1)
        status['names'] = {CH[k]: names[k] for k in range(K)}
    N.to_csv(out / 'channel_effects.csv', index=False)
    prof.to_csv(out / 'channel_profiles.csv', index=False)
    edge.to_csv(out / 'edge_routing.csv', index=False)
    if cfg.get('move_all'):
        groups = {(names[k] if names else ch): [k] for k, ch in enumerate(CH)}
    elif names:
        groups = {m: [k for k in range(K) if names[k] == m] for m in ('comp', 'compet')}
        log('names ' + json.dumps(status['names']))
    else:
        (out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding='utf-8')
        log('channel outputs written; names are needed (channel_names in the config) before measuring: ' + json.dumps(status, ensure_ascii=False))
        return

    U = R['U']
    nominal = measure(x, R)
    pc0 = nominal['pc']
    cuts = np.percentile(pc0, [100 / 3, 200 / 3], axis=0)
    lvl = lambda Pc: (Pc[:, None, :] > cuts[None, :, :]).sum(1)
    lv0 = lvl(pc0)
    per = []
    for s, r in runs.items():
        part = {m: sum((r['dx'][k] for k in ks), np.zeros_like(x)) for m, ks in groups.items()}
        dx = sum(part.values())
        me = measure(x + dx, R)
        per.append({'seed': s, 'cls': me['class'], 'lv': lvl(me['pc']), 'dPC': me['pc'] - pc0, 'dx': dx,
                    **{f'dPC_{m}': part[m] @ U for m in groups}})
    need = int(np.ceil(0.8 * len(per)))
    cls = np.stack([q['cls'] for q in per])
    lvs = np.stack([q['lv'] for q in per])
    modal = np.array([np.bincount(c, minlength=len(type_names)).argmax() for c in cls.T])
    robust = ((cls == modal[None]).sum(0) >= need) & (modal != nominal['class'])
    lv_modal = np.stack([[np.bincount(l_, minlength=3).argmax() for l_ in lvs[:, :, j].T] for j in range(3)], 1)
    lv_robust = ((lvs == lv_modal[None]).sum(0) >= need) & (lv_modal != lv0)
    mean = lambda key: np.mean([q[key] for q in per], 0)
    dPC, dx = mean('dPC'), mean('dx')
    frame = pd.DataFrame({'community_id': g['ids'], 'name': g['names'], 'type_nominal': [type_names[c] for c in nominal['class']],
                          'type_effective_modal': [type_names[c] for c in modal], 'robust_type_change': robust})
    for j, pc in enumerate(PCS):
        frame[f'{pc}_nominal'] = pc0[:, j]
        frame[f'd{pc}'] = dPC[:, j]
        for m in groups:
            frame[f'd{pc}_{m}'] = mean(f'dPC_{m}')[:, j]
        frame[f'{pc}_level_nominal'] = [LV[q] for q in lv0[:, j]]
        frame[f'{pc}_level_effective_modal'] = [LV[q] for q in lv_modal[:, j]]
        frame[f'{pc}_robust_level_change'] = lv_robust[:, j]
    for i, c in enumerate(g['columns']):
        frame[f'dx_{c}'] = dx[:, i]
    frame.to_csv(out / 'nominal_vs_effective.csv', index=False, encoding='utf-8-sig')
    pd.crosstab(frame.type_nominal, frame.type_effective_modal).to_csv(out / 'transition.csv', encoding='utf-8-sig')
    by = []
    for c, nm in enumerate(type_names):
        f_ = frame[nominal['class'] == c]
        row = {'type': nm, 'n': len(f_), 'robust_type_changes': int(f_.robust_type_change.sum())}
        for pc in PCS:
            rc = f_[f'{pc}_robust_level_change']
            up = rc & (f_[f'{pc}_level_effective_modal'].map(LV.index) > f_[f'{pc}_level_nominal'].map(LV.index))
            row.update({f'd{pc}': f_[f'd{pc}'].mean(), **{f'd{pc}_{m}': f_[f'd{pc}_{m}'].mean() for m in groups},
                        f'{pc}_level_up/down': f'{int(up.sum())}/{int((rc & ~up).sum())}'})
        by.append(row)
    pd.DataFrame(by).to_csv(out / 'by_type.csv', index=False, encoding='utf-8-sig')

    pairs = list(itertools.combinations(range(len(per)), 2))
    status.update({'robust_type_changes': int(robust.sum()),
              **{f'{pc}_level_up/down': f"{int((lv_robust[:, j] & (lv_modal[:, j] > lv0[:, j])).sum())}/"
                                        f"{int((lv_robust[:, j] & (lv_modal[:, j] < lv0[:, j])).sum())}" for j, pc in enumerate(PCS)},
              **{f'seed_agreement_d{pc}': float(np.mean([spearmanr(per[i]['dPC'][:, j], per[k]['dPC'][:, j]).statistic for i, k in pairs]))
                 for j, pc in enumerate(PCS)}})
    (out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding='utf-8')
    log('done ' + json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
