"""Train and compare OD relation routing using one destination CE and Adam L2.

python 5lineb/v13_geoflow/run_od_relation_v13.py --config \
  5lineb/v13_geoflow/configs/redo_d0.25_od_relation_v13.yaml

Both architectures share MechNet.forward, five routing layers and seven
iterations per layer. Only the representation used inside routing differs.
Both architectures use the same fixed edge splits and message-only OD
context. Held-out counts, including within-community targets, never enter
message weights or pressure. Both compare candidate OD supply pairs with
the origin's aggregated channels using the same prediction head. Pressure
enters channel scores; topology and distance correct the OD score. Model selection uses only
validation destination cross entropy. Test metrics are computed once after
restoring that checkpoint. They are conditional masked-destination metrics,
matching the task of predicting hidden edges among non-message candidates.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from model_od_relation_v13 import ODRelationNet
from model_v13 import MechNet, holdout_mask, road_matrix
from graph import build_graph
from ruler import standardize


def pressure_context(context, raw, columns):
    """Only available incoming messages contribute to the origin pressure."""
    inflow = (context * ~np.eye(len(context), dtype=bool)).sum(0)
    features, service = [], []
    for tier in ('CHC', 'SH', 'TH'):
        resource = raw[:, columns.index(f'RA_{tier}')]
        has = resource > 0
        p = np.zeros(len(raw))
        if has.any():
            values = np.log1p(inflow[has] / resource[has])
            p[has] = (values - values.mean()) / max(values.std(), 1e-12)
        features.append(p)
        service.append(has.astype(float))
    return np.column_stack(features + service).astype(np.float32)


def destination_ce(score, target, candidates):
    """Macro average of origin-normalized multinomial destination losses."""
    total = target.sum(1, keepdim=True)
    active = total[:, 0] > 0
    probability = target / total.clamp_min(1e-12)
    return F.cross_entropy(score[active].masked_fill(~candidates[active], -1e9), probability[active])


def make_model(variant, cfg, n, topology=None):
    model_class = ODRelationNet if variant == 'od_relation' else MechNet
    return model_class(d=cfg['hid_dim'] // 2, iters=cfg['iters'], n_channels=2,
                   out_weight='share', attention=cfg['attention'], base=False,
                   pressure=np.zeros((n, 6)), etype=topology, n=n,
                   proj='linear', routing_layers=cfg['routing_layers'], dropout=cfg['dropout'],
                   route_score=True, rule_origin=True, od_pair_head=True)


def retention_context(context, internal):
    """Internal trips over internal plus available outgoing messages, standardised over communities with trips."""
    total = internal + (context * ~np.eye(len(context), dtype=bool)).sum(1)
    has = total > 0
    r = np.zeros(len(context))
    r[has] = internal[has] / total[has]
    r[has] = (r[has] - r[has].mean()) / max(r[has].std(), 1e-12)
    return r.astype(np.float32)


def forward_context(model, variant, g, tensors, context_np, tau):
    x, distance, topology, candidates = tensors
    context = torch.as_tensor(context_np, dtype=torch.float32)
    pressure = torch.as_tensor(pressure_context(context_np, g['raw'], g['columns']))
    model.inflow.copy_(pressure)
    if getattr(model, 'retention', None) is not None:
        model.retention.copy_(torch.as_tensor(retention_context(context_np, np.diag(g['W']))))
    if hasattr(model, 'set_context'):
        model.set_context(context_np, np.diag(g['W']))
    share = context / context.sum(1, keepdim=True).clamp_min(1e-12)
    return model(x, distance, topology, candidates, share, context > 0, tau)


def evaluate_scores(score, W, held, context_mask, excluded=None):
    """The shared conditional test protocol, plus a full-choice retention diagnostic."""
    n = len(W)
    candidates = ~torch.as_tensor(context_mask)
    if excluded is not None:
        candidates &= ~torch.as_tensor(excluded)
    joint = torch.softmax(score.masked_fill(~candidates, -1e9), 1).numpy()
    nonself = candidates & ~torch.eye(n, dtype=torch.bool)
    logpi = torch.log_softmax(score.masked_fill(~nonself, -1e9), 1).numpy()
    observed = W * held
    total = observed.sum(1)
    rows = total > 0
    target = observed[rows] / total[rows, None]
    ce = -(target * np.where(held[rows], logpi[rows], 0)).sum(1)
    predicted = total[rows, None] * np.exp(logpi[rows])
    mask = nonself.numpy()[rows]
    internal = np.diag(W)[rows] / (np.diag(W)[rows] + total[rows])
    full = torch.softmax(score, 1).numpy()
    true_share = np.divide(np.diag(W), W.sum(1), out=np.zeros(n), where=W.sum(1) > 0)
    valid = W.sum(1) > 0
    return {
        'ce': float(ce.mean()), 'ce_trips': float(np.average(ce, weights=total[rows])),
        'cpc': float(np.minimum(predicted, observed[rows])[mask].sum() / observed[rows][mask].sum()),
        'nrmse': float(np.sqrt(np.mean((predicted[mask] - observed[rows][mask]) ** 2)) / observed[rows][mask].mean()),
        'internal_error_conditional': float(np.abs(np.diag(joint)[rows] - internal).mean()),
        'internal_error_full_diagnostic': float(np.abs(np.diag(full)[valid] - true_share[valid]).mean())}


def train_one(variant, seed, cfg, g, tensors, H, Hv, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = g['W']
    off = ~np.eye(len(W), dtype=bool)
    visible = (W > 0) & off & ~H & ~Hv
    tau = cfg['tau_final']
    model = make_model(variant, cfg, len(W), tensors[2])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    history, best, best_ce, best_epoch, since = [], None, np.inf, 0, 0
    log_file = (output / 'progress.log').open('a', encoding='utf-8')

    def log(message):
        line = f'{time.strftime("%H:%M:%S")} {variant} seed={seed} {message}'
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    log('START destination CE + Adam L2, fixed holdout seed 0')
    start = time.time()
    for epoch in range(cfg['epochs']):
        model.train()
        msg = visible & (rng.random(W.shape) < 0.5)
        context = W * msg
        result = forward_context(model, variant, g, tensors, context, tau)
        target = torch.as_tensor(W * ~msg * ~H * ~Hv, dtype=torch.float32)
        candidates = torch.as_tensor(~msg & ~H & ~Hv)
        if cfg.get('exclude_internal_target'):
            target = target * torch.as_tensor(off, dtype=torch.float32)
            candidates = candidates & torch.as_tensor(off)
        ce = destination_ce(result['score'], target, candidates)
        loss = ce
        for key in ('indep', 'dec', 'stable'):
            if cfg.get(f'{key}_weight') and key in result:
                loss = loss + cfg[f'{key}_weight'] * result[key]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            validation = forward_context(model, variant, g, tensors, W * visible, tau)
            val = evaluate_scores(validation['score'], W, Hv, visible, excluded=H)
        history.append({'epoch': epoch + 1, 'ce': float(ce.detach()), 'val_ce': val['ce'], 'val_cpc': val['cpc']})
        for key in ('indep', 'dec', 'stable'):
            if key in result:
                history[-1][key] = float(result[key].detach())
        if val['ce'] < best_ce:
            best_ce, best_epoch, since = val['ce'], epoch + 1, 0
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            since += 1
        if (epoch + 1) % 25 == 0:
            log(f'epoch={epoch + 1} ce={ce.item():.4f} val_ce={val["ce"]:.4f} val_cpc={val["cpc"]:.4f}')
            pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        if since >= cfg['early']:
            break
    model.load_state_dict(best)
    model.eval()
    test_context = (W > 0) & off & ~H
    with torch.no_grad():
        result = forward_context(model, variant, g, tensors, W * test_context, tau)
        metrics = evaluate_scores(result['score'], W, H, test_context)
    metrics.update(variant=variant, seed=seed, epochs=len(history), best_epoch=best_epoch,
                   best_val_ce=best_ce, seconds=time.time()-start)
    p = result['p'].numpy()
    edge = ((W > 0) & off)
    probability = p.transpose(0, 2, 1)[edge]
    weights = W[edge]
    entropy = -(probability * np.log(probability.clip(1e-30))).sum(1) / np.log(probability.shape[1])
    metrics['routing_entropy_normalized'] = float(np.average(entropy, weights=weights))
    metrics['routing_probability_std_C1'] = float(np.sqrt(np.average((probability[:, 0]-np.average(probability[:, 0], weights=weights))**2, weights=weights)))
    metrics['confident_flow_share_p08'] = float(np.average(probability.max(1) >= 0.8, weights=weights))
    torch.save(model.state_dict(), output / 'model.pt')
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output / 'arrays.npz', p=probability, scores=result['score'].numpy(),
                        h=result['c'].numpy())
    log('DONE ' + json.dumps(metrics))
    log_file.close()


def weighted_corr(a, b, weights):
    a, b = a - np.average(a, weights=weights), b - np.average(b, weights=weights)
    denominator = np.sqrt(np.dot(weights, a*a) * np.dot(weights, b*b))
    return float(np.dot(weights, a*b)/denominator) if denominator > 1e-12 else np.nan


def summarize(output, g, seeds):
    all_metrics, stability = [], []
    W = g['W']
    edge = (W > 0) & ~np.eye(len(W), dtype=bool)
    weights = W[edge]
    oi, di = np.nonzero(edge)
    for variant in ('node_reference', 'od_relation'):
        runs, metrics = {}, {}
        for seed in seeds:
            directory = output / variant / f'seed{seed}'
            if not (directory / 'metrics.json').exists():
                return
            metrics[seed] = json.loads((directory / 'metrics.json').read_text())
            all_metrics.append(metrics[seed])
            runs[seed] = np.load(directory / 'arrays.npz')['p']
        for sa, sb in combinations(seeds, 2):
            a, b = runs[sa], runs[sb]
            C = np.array([[weighted_corr(a[:, i], b[:, j], weights) for j in range(2)] for i in range(2)])
            _, perm = linear_sum_assignment(-np.nan_to_num(C, nan=0))
            bb = b[:, perm]
            stability.append({'variant': variant, 'seed_a': sa, 'seed_b': sb,
                              'routing_corr': weighted_corr(a[:, 0], bb[:, 0], weights),
                              'hard_agreement': float((a.argmax(1) == bb.argmax(1)).mean()),
                              'hard_agreement_flow_weighted': float(np.average(a.argmax(1) == bb.argmax(1), weights=weights)),
                              'probability_abs_difference': float(np.average(np.abs(a[:, 0]-bb[:, 0]), weights=weights))})
        ref_seed = min(seeds, key=lambda s: metrics[s]['best_val_ce'])
        reference = runs[ref_seed]
        aligned = []
        for seed in seeds:
            C = np.array([[weighted_corr(reference[:, i], runs[seed][:, j], weights) for j in range(2)] for i in range(2)])
            _, perm = linear_sum_assignment(-np.nan_to_num(C, nan=0))
            aligned.append(runs[seed][:, perm])
        stack = np.stack(aligned)
        consensus = pd.DataFrame({'origin_id': g['ids'][oi], 'destination_id': g['ids'][di], 'flow': weights,
                                 'p_C1': stack[:, :, 0].mean(0), 'p_C2': stack[:, :, 1].mean(0),
                                 'p_C1_std': stack[:, :, 0].std(0), 'channel_agreement': np.maximum((stack.argmax(2)==0).mean(0), (stack.argmax(2)==1).mean(0))})
        consensus.to_csv(output / f'{variant}_edge_consensus.csv', index=False)
    M, S = pd.DataFrame(all_metrics), pd.DataFrame(stability)
    M.to_csv(output / 'metrics_summary.csv', index=False)
    S.to_csv(output / 'stability.csv', index=False)
    print(M.groupby('variant')[['ce', 'cpc', 'nrmse']].agg(['mean','std']).to_string(), flush=True)
    print(S.groupby('variant').mean(numeric_only=True).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', choices=('node_reference','od_relation','both'), default='both')
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ('supply_file','flow_file','adjacency_file'):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    g = build_graph(cfg)
    R = json.loads((ROOT / cfg['ruler_from'] / 'ruler.json').read_text())['main']['frozen']
    g['x'] = standardize(g['raw'], R)
    W = g['W']
    H = holdout_mask(W, cfg['holdout_share'], cfg['holdout_seed'])
    Hv = holdout_mask(W * ~H, cfg['val_share'], cfg['holdout_seed']+1)
    tensors = [torch.as_tensor(g['x'], dtype=torch.float32), torch.as_tensor(road_matrix(g,cfg['road_impedance']), dtype=torch.float32),
               torch.as_tensor(g['K'], dtype=torch.long), torch.ones(W.shape, dtype=torch.bool)]
    output = ROOT / 'results_od_relation' / cfg['name']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    np.savez_compressed(output / 'fixed_split.npz', ids=g['ids'], W=W, H=H, Hv=Hv)
    variants = ('od_relation','node_reference') if args.variant == 'both' else (args.variant,)
    seeds = args.seeds or cfg['seeds']
    for seed in seeds:
        for variant in variants:
            directory = output / variant / f'seed{seed}'
            if not (directory / 'metrics.json').exists():
                train_one(variant, seed, cfg, g, tensors, H, Hv, directory)
    summarize(output, g, cfg['seeds'])


if __name__ == '__main__':
    main()
