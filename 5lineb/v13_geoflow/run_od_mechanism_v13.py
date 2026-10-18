"""Train and summarise the non-exclusive mechanism model (model_od_mechanism_v13) with the reference procedure.

python 5lineb/v13_geoflow/run_od_mechanism_v13.py --config 5lineb/v13_geoflow/configs/outpatient_d0.25_od_mechanism_v13.yaml \
    [--variants mech_full ...] [--seeds 0 1 2] [--summarize-only]

Training is run_od_relation_v13.train_one (origin-normalised destination CE + Adam coupled L2, fixed test and
validation splits of run_od_ablation_v13.setup, minimum validation CE with early stopping). Results go to
results_od_mechanism/<name>/<variant>/seed<s>. The summary reports per seed and per variant
  prediction    test CE, CPC, NRMSE; for variants with external channels also the test metrics of the same checkpoint
                with both external channel vectors set to zero (aggregation contribution);
  routing       gates q^comp, q^compete and node propensities s; channel correspondence between two seeds is chosen
                on the gates of the available training edges (visible context) and the same permutation is applied to
                the test context; on the observed edges outside the test set in the test context: per-gate cross-seed
                correlation before and after alignment, mean absolute and squared difference of the two-gate vector,
                flow-weighted agreement of the dominant gate and of each gate above 0.5, cross-seed correlation of the
                node propensities, flow-weighted share of edges with both gates above 0.5, mean binary gate entropy;
  channels      mean norm of the base, comp and compete vectors, mean cosine between comp and compete vectors of the
                nodes with outgoing messages, mean normalised entropy of the two attentions.
Checkpoint replay: the test scores of every saved model are recomputed and compared exactly with arrays.npz.
View consistency (M2, stable_weight > 0 in the config): every training epoch draws a second random message mask of the
available training edges (own generator, seed 10000 + seed, so the first view follows the reference sequence); the
destination CE stays that of the first view, and the loss adds stable_weight times the mean over the edges carrying
messages in both views of ||q_ij^(a) - q_ij^(b)||^2 (the two gates of the same OD in the same mechanism order).
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_od_relation_v13 as reference
from run_od_ablation_v13 import setup, digest
from model_od_mechanism_v13 import MECH_VARIANTS, MECHANISMS, make_mechanism_model


def view_consistency(g, H, Hv, seed):
    """forward_context that adds the gate consistency of a second random view in training mode."""
    W = g['W']
    n = len(W)
    visible = (W > 0) & ~np.eye(n, dtype=bool) & ~H & ~Hv
    rng = np.random.default_rng(10_000 + seed)
    original = reference.forward_context

    def dense(out):
        q = torch.zeros(n, n, 2)
        q[out['edge_index'][0], out['edge_index'][1]] = out['q_edge']
        return q

    def forward(model, variant, g_, tensors, context, tau):
        out = original(model, variant, g_, tensors, context, tau)
        if model.training:
            second = visible & (rng.random(W.shape) < 0.5)
            out_b = original(model, variant, g_, tensors, W * second, tau)
            shared = torch.as_tensor((context > 0) & second)
            out['stable'] = (dense(out) - dense(out_b))[shared].square().sum(1).mean()
        return out
    return forward


def predict(model, g, tensors, context, tau, zero_external=False):
    model.eval()
    model.zero_external = zero_external
    with torch.no_grad():
        out = reference.forward_context(model, 'mechanism', g, tensors, context, tau)
    model.zero_external = False
    return out


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    d = np.sqrt(np.dot(w, a * a) * np.dot(w, b * b))
    return float(np.dot(w, a * b) / d) if d > 1e-12 else float('nan')


def binary_entropy(q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def seed_record(variant, seed, cfg, g, tensors, H, Hv, directory):
    W = g['W']
    n = len(W)
    edge = (W > 0) & ~np.eye(n, dtype=bool)
    visible, test_ctx = edge & ~H & ~Hv, edge & ~H
    tau = cfg['tau_final']
    model = make_mechanism_model(variant, cfg, n, tensors[2])
    model.load_state_dict(torch.load(directory / 'model.pt', weights_only=True))
    test = predict(model, g, tensors, W * test_ctx, tau)
    saved = np.load(directory / 'arrays.npz')
    np.testing.assert_array_equal(test['score'].numpy(), saved['scores'])
    metrics = json.loads((directory / 'metrics.json').read_text())
    record = {k: metrics[k] for k in ('ce', 'ce_trips', 'cpc', 'nrmse', 'best_epoch', 'best_val_ce', 'epochs')}
    record.update(variant=variant, seed=seed, replay_exact=True, parameter_count=sum(p.numel() for p in model.parameters()))
    external = [c for c in MECHANISMS if c in MECH_VARIANTS[variant]['channels']]
    if external:
        zero = predict(model, g, tensors, W * test_ctx, tau, zero_external=True)
        z = reference.evaluate_scores(zero['score'], W, H, test_ctx)
        record.update(zero_external_ce=z['ce'], zero_external_cpc=z['cpc'], zero_external_nrmse=z['nrmse'])
    train = predict(model, g, tensors, W * visible, tau)
    q_test = test['p'].numpy().transpose(0, 2, 1)
    q_train = train['p'].numpy().transpose(0, 2, 1)
    keep = test_ctx[edge]
    w = W[edge][keep]
    q = q_test[edge][keep]
    record.update(gate_mean_comp=float(np.average(q[:, 0], weights=w)), gate_mean_compete=float(np.average(q[:, 1], weights=w)),
                  both_gates_above_05=float(np.average((q > 0.5).all(1), weights=w)),
                  gate_entropy_comp=float(np.average(binary_entropy(q[:, 0]), weights=w)),
                  gate_entropy_compete=float(np.average(binary_entropy(q[:, 1]), weights=w)))
    ch = {k: v.numpy() for k, v in test['channels'].items()}
    for k, v in ch.items():
        record[f'norm_{k}'] = float(np.linalg.norm(v, axis=1).mean())
    has = test_ctx.any(1)
    a, b = ch['comp'][has], ch['compete'][has]
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    record['cos_comp_compete'] = float(cos.mean())
    oi = test['edge_index'][0].numpy()
    for k, c in enumerate(MECHANISMS):
        att = test['a_edge'][:, k].numpy()
        ent = np.zeros(n)
        np.add.at(ent, oi, -att * np.log(np.clip(att, 1e-30, None)))
        deg = np.bincount(oi, minlength=n)
        m = deg > 1
        record[f'attention_entropy_{c}'] = float((ent[m] / np.log(deg[m])).mean())
    arrays = dict(q_test=q, q_train=q_train[edge][visible[edge]], s_test=test['s'].numpy(), s_train=train['s'].numpy())
    return record, arrays


def stability(variant, seeds, arrays, g, H, Hv):
    W = g['W']
    n = len(W)
    edge = (W > 0) & ~np.eye(n, dtype=bool)
    w_test = W[edge][(edge & ~H)[edge]]
    w_train = W[edge][(edge & ~H & ~Hv)[edge]]
    rows = []
    for sa, sb in combinations(seeds, 2):
        A, B = arrays[sa], arrays[sb]
        C = np.array([[wcorr(A['q_train'][:, i], B['q_train'][:, j], w_train) for j in range(2)] for i in range(2)])
        _, perm = linear_sum_assignment(-np.nan_to_num(C))
        qa, qb = A['q_test'], B['q_test'][:, perm]
        sa_, sb_ = A['s_test'], B['s_test'][:, perm]
        rows.append(dict(
            variant=variant, seed_a=sa, seed_b=sb, permutation=','.join(map(str, perm)),
            corr_before_comp=wcorr(A['q_test'][:, 0], B['q_test'][:, 0], w_test),
            corr_before_compete=wcorr(A['q_test'][:, 1], B['q_test'][:, 1], w_test),
            corr_after_comp=wcorr(qa[:, 0], qb[:, 0], w_test), corr_after_compete=wcorr(qa[:, 1], qb[:, 1], w_test),
            mean_abs_diff=float(np.average(np.abs(qa - qb).mean(1), weights=w_test)),
            mean_sq_diff=float(np.average(((qa - qb) ** 2).sum(1), weights=w_test)),
            dominant_gate_agreement=float(np.average(qa.argmax(1) == qb.argmax(1), weights=w_test)),
            gate05_agreement_comp=float(np.average((qa[:, 0] > .5) == (qb[:, 0] > .5), weights=w_test)),
            gate05_agreement_compete=float(np.average((qa[:, 1] > .5) == (qb[:, 1] > .5), weights=w_test)),
            propensity_corr_comp=float(np.corrcoef(sa_[:, 0], sb_[:, 0])[0, 1]),
            propensity_corr_compete=float(np.corrcoef(sa_[:, 1], sb_[:, 1])[0, 1])))
    return rows


def summarize(output, cfg, g, tensors, H, Hv, variants, seeds):
    records, stab = [], []
    for variant in variants:
        arrays = {}
        for seed in seeds:
            directory = output / variant / f'seed{seed}'
            if not (directory / 'metrics.json').exists():
                continue
            record, arrays[seed] = seed_record(variant, seed, cfg, g, tensors, H, Hv, directory)
            records.append(record)
        if len(arrays) > 1 and set(MECHANISMS) <= set(MECH_VARIANTS[variant]['channels']):
            stab += stability(variant, sorted(arrays), arrays, g, H, Hv)
    R, S = pd.DataFrame(records), pd.DataFrame(stab)
    R.to_csv(output / 'metrics_per_seed.csv', index=False)
    if len(S):
        S.to_csv(output / 'stability.csv', index=False)
    numeric = [c for c in R.columns if c not in ('variant', 'seed', 'replay_exact')]
    summary = R.groupby('variant')[numeric].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    if len(S):
        summary = summary.join(S.groupby('variant').mean(numeric_only=True).drop(columns=['seed_a', 'seed_b']))
    summary.to_csv(output / 'summary.csv')
    pd.set_option('display.width', 250)
    cols = [c for c in ('ce_mean', 'ce_std', 'cpc_mean', 'cpc_std', 'nrmse_mean', 'zero_external_ce_mean',
                        'zero_external_cpc_mean', 'corr_after_comp', 'corr_after_compete', 'mean_abs_diff',
                        'propensity_corr_comp', 'propensity_corr_compete', 'both_gates_above_05_mean') if c in summary]
    print(summary[cols].round(4).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    cfg, g, tensors, H, Hv = setup(args.config)
    variants = args.variants or cfg['variants']
    seeds = args.seeds or cfg['seeds']
    if any(v not in MECH_VARIANTS for v in variants):
        raise ValueError(f'variants must select from {list(MECH_VARIANTS)}')
    output = ROOT / 'results_od_mechanism' / cfg['name']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    np.savez_compressed(output / 'fixed_split.npz', ids=g['ids'], W=g['W'], H=H, Hv=Hv)
    if not args.summarize_only:
        files = ['model_od_mechanism_v13.py', 'run_od_mechanism_v13.py', 'run_od_relation_v13.py', 'model_v13.py']
        data = [Path(cfg[k]) for k in ('supply_file', 'flow_file', 'adjacency_file')]
        data += [ROOT / cfg['road_impedance'], ROOT / cfg['ruler_from'] / 'ruler.json']
        (output / 'experiment_manifest.json').write_text(json.dumps(dict(
            seeds=cfg['seeds'], torch_version=torch.__version__, variants={v: MECH_VARIANTS[v] for v in cfg['variants']},
            loss='origin-normalized flow CE + Adam coupled L2' + (
                f' + {cfg["stable_weight"]} x two-view gate consistency' if cfg.get('stable_weight') else ' (no auxiliary loss)'),
            selection='minimum validation CE with early stopping',
            source_sha256={f: digest(HERE / f) for f in files},
            data_sha256={str(p.relative_to(ROOT)): digest(p) for p in data},
            notes=['Within-community trips F_ii enter the base channel and the propensities; the training target keeps the '
                   'within-community option (reference train_one); test and validation metrics are non-self only.',
                   'Gates of pairs without a message use a zero flow input; routing diagnostics use the observed edges '
                   'outside the test set in the test context.',
                   'Channel correspondence between seeds is chosen on the training edges only.']),
            indent=2, ensure_ascii=False))
        torch.set_num_threads(cfg['threads'])
        with patch.object(reference, 'make_model', make_mechanism_model):
            for seed in seeds:
                for variant in variants:
                    directory = output / variant / f'seed{seed}'
                    if not all((directory / f).exists() for f in ('model.pt', 'arrays.npz', 'metrics.json')):
                        if cfg.get('stable_weight'):
                            with patch.object(reference, 'forward_context', view_consistency(g, H, Hv, seed)):
                                reference.train_one(variant, seed, cfg, g, tensors, H, Hv, directory)
                        else:
                            reference.train_one(variant, seed, cfg, g, tensors, H, Hv, directory)
    summarize(output, cfg, g, tensors, H, Hv, cfg['variants'], cfg['seeds'])


if __name__ == '__main__':
    main()
