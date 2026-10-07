"""Unsupervised check of the three routing channels (no anchor loss, centres learned freely).

Variants
  U1  structure kept: the base channel alone routes to the self loop and only the third gate reads inflow
  U2  symmetric: three exchangeable channels, every gate reads inflow, every channel may use the self loop
Anchors are used only after training, to read the channels: confusion with the labels under the best
channel-to-mechanism matching. Channel profiles describe where each channel sends flow (self loop,
adjacent, cross), how far, and the destination-minus-origin tier scores under the frozen main ruler.

python lineb/v1_disengat/unsupervised.py --config lineb/v1_disengat/configs/walk_v4_k5.yaml --supervised results_lineb/walk_v4_k5
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
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
from graph import build_graph, load_labels, INT, ADJ, CRS  # noqa: E402
from ruler import standardize, measure, TIERS  # noqa: E402
from model import DisenGAT, train, as_tensors, typed  # noqa: E402

VARIANTS = {'U1_structure_kept': {'free_centres': True}, 'U2_symmetric': {'free_centres': True, 'symmetric': True}}


def frozen_main(run_dir):
    f = dict(json.loads((run_dir / 'ruler.json').read_text(encoding='utf-8'))['main']['frozen'])
    for key in ('mean', 'std', 'tier_W', 'tier_mean', 'tier_std', 'tier_cuts', 'centers'):
        if f.get(key) is not None:
            f[key] = np.asarray(f[key], dtype=float)
    return f


def forward(model, g, labels):
    T = dict(x=torch.as_tensor(g['x'], dtype=torch.float32), t=torch.as_tensor(g['t'], dtype=torch.float32),
             R=torch.as_tensor(g['K'], dtype=torch.long), u=torch.as_tensor(g['u'], dtype=torch.float32),
             cand=torch.as_tensor(g['candidates']))
    model.eval()
    with torch.no_grad():
        out = model(T['x'], T['t'], T['R'], T['u'], T['cand'], labels)
    return out['posterior'].numpy(), out['lambda'].numpy(), out['logpi'].numpy()


def profile(post, g, tiers, pressure):
    """Flow-weighted description of each channel."""
    W, K, D = g['W'], g['K'], g['distance_km']
    n = len(W)
    rows = []
    total = W.sum()
    dT = tiers[None, :, :] - tiers[:, None, :]
    for k in range(post.shape[1]):
        f = W * post[:, k, :]
        off = f * (1 - np.eye(n))
        r = {'channel': k, 'flow_share': f.sum() / total,
             'self_share_of_channel': np.trace(f) / f.sum(),
             'adjacent_share_of_nonself': off[K == ADJ].sum() / off.sum(),
             'cross_share_of_nonself': off[K == CRS].sum() / off.sum(),
             'mean_km_nonself': (off * D).sum() / off.sum()}
        for j, t in enumerate(TIERS):
            r[f'd_{t}'] = (off * dT[:, :, j]).sum() / off.sum()
            r[f'better_{t}'] = (off * (dT[:, :, j] > 0.5)).sum() / off.sum()
        r['better_any'] = (off * (dT.max(-1) > 0.5)).sum() / off.sum()
        r['parallel_or_worse_all'] = (off * (dT.max(-1) <= 0.5)).sum() / off.sum()
        share_o = off.sum(1) / np.maximum(W.sum(1), 1e-12)
        r['corr_origin_pressure'] = float(np.corrcoef(share_o, pressure)[0, 1])
        rows.append(r)
    return pd.DataFrame(rows)


def edge_vectors(post, W):
    m = (W > 0) & ~np.eye(len(W), dtype=bool)
    return post.transpose(0, 2, 1)[m], W[m]          # [edges, 3], weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--supervised', required=True)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    sup_dir = Path(a.supervised)
    out = Path(a.out) if a.out else sup_dir.parent / (sup_dir.name + '_unsupervised')
    out.mkdir(parents=True, exist_ok=True)
    fr = frozen_main(sup_dir)
    g = build_graph(cfg)
    g['x'] = standardize(g['raw'], fr)
    tiers = measure(g['x'], fr)['tiers']
    tiers = (tiers - fr['tier_mean']) / fr['tier_std']
    labels_np = load_labels(cfg['labels_file'], g)
    labels = typed(as_tensors(labels_np))
    W = g['W']
    O = W.sum(1)
    cross_in = (W * (g['K'] == CRS) * (1 - np.eye(len(W)))).sum(0)
    pressure = np.log1p(np.divide(cross_in, O, out=np.zeros(len(W)), where=O > 0))

    # supervised reference (default behaviour of the model must be unchanged)
    sup = []
    for seed in cfg['seeds']:
        m = DisenGAT(reconstruct=cfg.get('joint_reconstruction', 0) > 0)
        m.load_state_dict(torch.load(sup_dir / 'runs' / f'seed{seed}' / 'model.pt'))
        sup.append(forward(m, g, labels))
    share = np.divide(W, W.sum(1, keepdims=True), out=np.zeros_like(W), where=W.sum(1, keepdims=True) > 0)
    rows_idx = np.asarray(g['origins'])
    ce = lambda logp: float(-(share[rows_idx] * logp[rows_idx]).sum(1).mean())
    print('supervised CE by seed', [round(ce(s[2]), 4) for s in sup], flush=True)
    sup_vec = [edge_vectors(s[0], W)[0] for s in sup]
    ew = edge_vectors(sup[0][0], W)[1]

    summary, profiles = [], []
    for name, kwargs in VARIANTS.items():
        runs = []
        for seed in cfg['seeds']:
            model, res, history, _ = train(g, labels_np, seed, epochs=cfg['epochs'], lr=cfg['lr'], decay=cfg['decay'],
                                           margin=cfg['margin'], weight=0.0, reconstruct=float(cfg.get('joint_reconstruction', 0)),
                                           model_kwargs=kwargs)
            post, lam, logpi = forward(model, g, labels)
            torch.save(model.state_dict(), out / f'{name}_seed{seed}.pt')
            runs.append({'seed': seed, 'post': post, 'lam': lam, 'ce': ce(logpi)})
            print(f'{name} seed {seed} CE {ce(logpi):.4f}', flush=True)
        # match channels of every seed to seed 0 by flow-weighted posterior correlation
        ref = edge_vectors(runs[0]['post'], W)[0]
        for r in runs:
            v = edge_vectors(r['post'], W)[0]
            C = np.array([[np.corrcoef(ref[:, i] * np.sqrt(ew), v[:, j] * np.sqrt(ew))[0, 1] for j in range(3)] for i in range(3)])
            _, perm = linear_sum_assignment(-C)
            r['perm'] = perm
            r['match_corr'] = C[np.arange(3), perm]
            r['post'] = r['post'][:, perm, :]
            r['lam'] = r['lam'][:, perm]
        # read channels with the anchors (evaluation only): best channel -> mechanism assignment
        for r in runs:
            dom = r['post'][labels_np['origin'], :, labels_np['dest']].argmax(1)
            conf = np.zeros((3, 3))
            for c, m_ in zip(dom, labels_np['mech']):
                conf[c, m_] += 1
            ch, mech = linear_sum_assignment(-conf)
            r['channel_to_mech'] = dict(zip(ch.tolist(), mech.tolist()))
            r['anchor_acc'] = conf[ch, mech].sum() / conf.sum()
            r['conf'] = conf
            v = edge_vectors(r['post'], W)[0]
            # agreement with the supervised dominant assignment, after the anchor-based mapping
            mapped = np.zeros_like(v)
            for c_, m_ in r['channel_to_mech'].items():
                mapped[:, m_] = v[:, c_]
            sup_dom = np.stack([s.argmax(1) for s in sup_vec])
            r['edge_agree_sup'] = float(np.mean([np.sum(ew * (mapped.argmax(1) == sd)) / ew.sum() for sd in sup_dom]))
            r['sharpness'] = float(np.sum(ew * v.max(1)) / ew.sum())
            prof = profile(r['post'], g, tiers, pressure)
            prof.insert(0, 'seed', r['seed'])
            prof.insert(0, 'variant', name)
            prof['mapped_mechanism'] = prof['channel'].map(lambda c_: ('base', 'comp', 'compet')[r['channel_to_mech'][c_]])
            profiles.append(prof)
            summary.append({'variant': name, 'seed': r['seed'], 'CE': r['ce'], 'seed_match_corr_min': float(r['match_corr'].min()),
                            'anchor_acc_best_mapping': r['anchor_acc'], 'edge_agree_supervised': r['edge_agree_sup'],
                            'mean_max_posterior': r['sharpness'],
                            'mapping': {f'ch{c_}': ('base', 'comp', 'compet')[m_] for c_, m_ in r['channel_to_mech'].items()},
                            'confusion_rows_channel_cols_base_comp_compet': r['conf'].astype(int).tolist()})
    sup_prof = []
    for s, seed in zip(sup, cfg['seeds']):
        pr = profile(s[0], g, tiers, pressure)
        pr.insert(0, 'seed', seed)
        pr.insert(0, 'variant', 'supervised')
        pr['mapped_mechanism'] = pr['channel'].map(lambda c_: ('base', 'comp', 'compet')[c_])
        sup_prof.append(pr)
        dom = s[0][labels_np['origin'], :, labels_np['dest']].argmax(1)
        summary.append({'variant': 'supervised', 'seed': seed, 'CE': ce(s[2]),
                        'anchor_acc_best_mapping': float((dom == labels_np['mech']).mean()),
                        'mean_max_posterior': float(np.sum(ew * edge_vectors(s[0], W)[0].max(1)) / ew.sum())})
    S = pd.DataFrame(summary)
    P = pd.concat(sup_prof + profiles, ignore_index=True)
    S.to_csv(out / 'summary.csv', index=False)
    P.to_csv(out / 'channel_profiles.csv', index=False)
    pd.set_option('display.width', 250)
    print(S.drop(columns=['confusion_rows_channel_cols_base_comp_compet']).round(4).to_string(index=False))
    cols = ['flow_share', 'self_share_of_channel', 'adjacent_share_of_nonself', 'mean_km_nonself', 'd_TH', 'd_SH', 'd_CHC',
            'better_any', 'parallel_or_worse_all', 'corr_origin_pressure']
    print(P.groupby(['variant', 'channel'])[cols].agg(['mean']).round(3).to_string())
    print(P.groupby(['variant', 'channel'])['mapped_mechanism'].agg(lambda s: s.value_counts().to_dict()).to_string())


if __name__ == '__main__':
    main()
