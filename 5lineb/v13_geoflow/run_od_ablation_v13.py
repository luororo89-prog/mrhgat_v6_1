"""Train the selected OD routing variant with the reference training procedure.

Fresh reproduction:
  python 5lineb/v13_geoflow/run_od_ablation_v13.py --config \
    5lineb/v13_geoflow/configs/redo_d0.25_od_ablation_v13.yaml

--reuse-original PATH may reuse the already trained original seeds only after
checking the source, configuration, fixed splits, and checkpoint predictions.
All modified models are trained from the prescribed initialization.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
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
from model_od_ablation_v13 import make_ablation_model, VARIANTS
from model_od_disen_v13 import DISEN_VARIANTS
from model_od_pairgat_v13 import PAIRGAT_VARIANTS
from model_v13 import holdout_mask, road_matrix
from model_v9 import l2n
from graph import build_graph
from ruler import standardize


ORIGINAL_HASHES = {
    'model_v13.py': '24ea68d7d64f59f1c8ad771a9f6cc4d811750347a49cd1123e64799c7ee56bc1',
    'model_od_relation_v13.py': '28144f4ee6f0e735a818d0a54bb78b8d98eb47ed1d0a0dfdccb54b233c76b940',
    'run_od_relation_v13.py': '5135875184449aa3a688e9dede9e4dad546aa202b65791fb266b67c8888f9c50',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(config):
    cfg = yaml.safe_load(Path(config).read_text())
    for key in ('supply_file', 'flow_file', 'adjacency_file'):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    g = build_graph(cfg)
    ruler = json.loads((ROOT / cfg['ruler_from'] / 'ruler.json').read_text())['main']['frozen']
    g['x'] = standardize(g['raw'], ruler)
    W = g['W']
    H = holdout_mask(W, cfg['holdout_share'], cfg['holdout_seed'])
    Hv = holdout_mask(W * ~H, cfg['val_share'], cfg['holdout_seed'] + 1)
    tensors = [torch.tensor(g['x'], dtype=torch.float32),
               torch.tensor(road_matrix(g, cfg['road_impedance']), dtype=torch.float32),
               torch.tensor(g['K'], dtype=torch.long), torch.ones(W.shape, dtype=torch.bool)]
    return cfg, g, tensors, H, Hv


def reuse_original(source, output, cfg, g, tensors, H, Hv):
    source = Path(source).resolve()
    old_cfg = yaml.safe_load((source / 'config.yaml').read_text())
    for key, value in cfg.items():
        if key not in ('name', 'variants', 'supply_file', 'flow_file', 'adjacency_file'):
            assert old_cfg[key] == value, (key, old_cfg[key], value)
    saved_split = np.load(source / 'fixed_split.npz')
    for key, value in dict(ids=g['ids'], W=g['W'], H=H, Hv=Hv).items():
        assert np.array_equal(saved_split[key], value), key
    for name, sha in ORIGINAL_HASHES.items():
        assert digest(HERE / name) == sha, name
    W = g['W']
    edge = (W > 0) & ~np.eye(len(W), dtype=bool)
    checks = []
    for seed in cfg['seeds']:
        src = source / 'od_relation' / f'seed{seed}'
        dst = output / 'original' / f'seed{seed}'
        model = make_ablation_model('original', cfg, len(W), tensors[2])
        model.load_state_dict(torch.load(src / 'model.pt', weights_only=True))
        model.eval()
        with torch.no_grad():
            replay = reference.forward_context(model, 'original', g, tensors, W * (edge & ~H), cfg['tau_final'])
        saved = np.load(src / 'arrays.npz')
        actual_p = replay['p'].numpy().transpose(0, 2, 1)[edge]
        np.testing.assert_array_equal(actual_p, saved['p'])
        np.testing.assert_array_equal(replay['score'].numpy(), saved['scores'])
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.iterdir():
            if path.is_file():
                shutil.copy2(path, dst / path.name)
        metrics = json.loads((dst / 'metrics.json').read_text())
        metrics.update(variant='original', training_origin='reused_verified', source_directory=str(src))
        (dst / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        checks.append(dict(seed=seed, scores_exact=True, routing_exact=True,
                           checkpoint_sha256=digest(src / 'model.pt')))
        print(f'REUSED verified original seed={seed}', flush=True)
    (output / 'original_reuse_checks.json').write_text(json.dumps(checks, indent=2))


def align(a, b, weights):
    C = np.array([[reference.weighted_corr(a[:, i], b[:, j], weights)
                   for j in range(b.shape[1])] for i in range(a.shape[1])])
    _, perm = linear_sum_assignment(-np.nan_to_num(C, nan=0))
    return b[:, perm], perm


def layer_diagnostics(model, variant, g, tensors, context, tau):
    rows = []
    original = model._centres

    def capture(*args, **kwargs):
        centre = original(*args, **kwargs)
        v = l2n(centre).double()
        n = len(v)
        cos = ((v.sum(0).square().sum(-1) - v.square().sum((0, 2))) / (n * (n-1))).numpy()
        rows.append(dict(layer=len(rows)+1, **{f'node_cos_C{k+1}': float(cos[k]) for k in range(len(cos))},
                         node_cos_max=float(cos.max())))
        return centre

    with torch.no_grad(), patch.object(model, '_centres', side_effect=capture):
        result = reference.forward_context(model, variant, g, tensors, context, tau)
    assert len(rows) == 5
    return rows, result


def summarize(output, cfg, g, tensors, H, Hv):
    metrics_rows, stability_rows, layers_rows, replay_checks = [], [], [], []
    W = g['W']
    edge = (W > 0) & ~np.eye(len(W), dtype=bool)
    weights = W[edge]
    oi, di = np.nonzero(edge)
    visible = edge & ~H & ~Hv
    for variant in cfg.get('variants', ['fix1']):
        runs, metrics_by_seed = {}, {}
        for seed in cfg['seeds']:
            directory = output / variant / f'seed{seed}'
            metrics = json.loads((directory / 'metrics.json').read_text())
            saved = np.load(directory / 'arrays.npz')
            probabilities = saved['p']
            runs[seed], metrics_by_seed[seed] = probabilities, metrics
            hard = [float(np.average(probabilities.argmax(1) == k, weights=weights)) for k in range(probabilities.shape[1])]
            hard_C1 = hard[0]
            metrics.update(
                hard_C1_flow_share=hard_C1,
                hard_flow_share=hard,
                soft_flow_share=[float(np.average(probabilities[:, k], weights=weights)) for k in range(probabilities.shape[1])],
                dominant_hard_channel_flow_share=max(hard),
                soft_C1_flow_share=float(np.average(probabilities[:,0], weights=weights)),
                training_origin=metrics.get('training_origin', 'trained_this_comparison'))
            model = make_ablation_model(variant, cfg, len(W), tensors[2])
            model.load_state_dict(torch.load(directory / 'model.pt', weights_only=True))
            model.eval()
            metrics['parameter_count'] = sum(v.numel() for v in model.parameters())
            layers, val_result = layer_diagnostics(model, variant, g, tensors, W*visible, cfg['tau_final'])
            layers_rows += [dict(variant=variant, seed=seed, **row) for row in layers]
            channel_cos = {key: value for key, value in layers[-1].items() if key.startswith('node_cos_C')}
            metrics.update({f'layer5_{key}': value for key, value in channel_cos.items()})
            metrics['layer5_node_cos_max'] = layers[-1]['node_cos_max']
            metrics['layer5_channels_cos_ge_099'] = int(sum(value >= .99 for value in channel_cos.values()))
            val_metrics = reference.evaluate_scores(val_result['score'], W, Hv, visible, excluded=H)
            np.testing.assert_allclose(val_metrics['ce'], metrics['best_val_ce'], atol=2e-6, rtol=0)
            with torch.no_grad():
                replay = reference.forward_context(model, variant, g, tensors, W*(edge & ~H), cfg['tau_final'])
            np.testing.assert_array_equal(replay['score'].numpy(), saved['scores'])
            np.testing.assert_array_equal(replay['p'].numpy().transpose(0,2,1)[edge], probabilities)
            replay_checks.append(dict(variant=variant, seed=seed, test_scores_exact=True,
                                      test_routing_exact=True, validation_checkpoint_verified=True))
            metrics_rows.append(metrics)
            (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            del model, replay, val_result
        for sa, sb in combinations(cfg['seeds'], 2):
            a, b = runs[sa], runs[sb]
            bb, perm = align(a, b, weights)
            agreement = (a.argmax(1) == bb.argmax(1))
            stability_rows.append(dict(
                variant=variant, seed_a=sa, seed_b=sb, channel_permutation=','.join(map(str,perm)),
                routing_corr=float(np.mean([reference.weighted_corr(a[:,k], bb[:,k], weights) for k in range(a.shape[1])])),
                hard_agreement=float(agreement.mean()),
                hard_agreement_flow_weighted=float(np.average(agreement, weights=weights)),
                probability_abs_difference=float(np.average(abs(a-bb).mean(1),weights=weights))))
        ref_seed = min(cfg['seeds'], key=lambda s:metrics_by_seed[s]['best_val_ce'])
        stack = np.stack([align(runs[ref_seed], runs[s], weights)[0] for s in cfg['seeds']])
        K = stack.shape[2]
        pd.DataFrame(dict(origin_id=g['ids'][oi], destination_id=g['ids'][di], flow=weights,
                          **{f'p_C{k+1}': stack[:,:,k].mean(0) for k in range(K)},
                          p_C1_std=stack[:,:,0].std(0),
                          channel_agreement=np.max([(stack.argmax(2)==k).mean(0) for k in range(K)], 0)))\
            .to_csv(output/f'{variant}_edge_consensus.csv',index=False)
    M, S, L = pd.DataFrame(metrics_rows), pd.DataFrame(stability_rows), pd.DataFrame(layers_rows)
    M.to_csv(output/'metrics_summary.csv',index=False)
    S.to_csv(output/'stability.csv',index=False)
    L.to_csv(output/'layers.csv',index=False)
    (output/'replay_checks.json').write_text(json.dumps(replay_checks,indent=2))
    summary=[]
    for variant in cfg.get('variants', ['fix1']):
        m=M[M.variant==variant]; s=S[S.variant==variant]
        row=dict(variant=variant,n_seeds=len(m),parameter_count=int(m.parameter_count.iloc[0]))
        for col in ('ce','cpc','nrmse','best_val_ce','dominant_hard_channel_flow_share',
                    'routing_entropy_normalized','layer5_node_cos_max','routing_probability_std_C1'):
            row[col+'_mean']=float(m[col].mean());row[col+'_sd']=float(m[col].std())
        for col in ('routing_corr','hard_agreement_flow_weighted','probability_abs_difference'):
            row[col+'_mean']=float(s[col].mean());row[col+'_min']=float(s[col].min());row[col+'_max']=float(s[col].max())
        row['collapsed_channels_of_6']=int(m.layer5_channels_cos_ge_099.sum())
        summary.append(row)
    summary=pd.DataFrame(summary);summary.to_csv(output/'comparison_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--reuse-original',type=Path)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    cfg,g,tensors,H,Hv=setup(args.config)
    variants=cfg.get('variants', ['fix1'])
    known={**VARIANTS,**DISEN_VARIANTS,**PAIRGAT_VARIANTS}
    if not variants or any(v not in known for v in variants):
        raise ValueError(f'variants must select from {list(known)}')
    if args.reuse_original and 'original' not in variants:
        raise ValueError('--reuse-original requires original in variants')
    output=ROOT/'results_od_ablation'/cfg['name'];output.mkdir(parents=True,exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    np.savez_compressed(output/'fixed_split.npz',ids=g['ids'],W=g['W'],H=H,Hv=Hv)
    source_files=list(ORIGINAL_HASHES)+['model_od_ablation_v13.py','run_od_ablation_v13.py']
    data_files=[Path(cfg[k]) for k in ('supply_file','flow_file','adjacency_file')]
    data_files += [ROOT/cfg['road_impedance'], ROOT/cfg['ruler_from']/'ruler.json']
    manifest=dict(seeds=cfg['seeds'],torch_version=torch.__version__,threads=cfg['threads'],
                  variants={v:known[v] for v in variants},loss='origin-normalized flow CE + Adam coupled L2',
                  selection='minimum validation CE with unchanged early stopping',
                  baseline_reused=bool(args.reuse_original),
                  source_sha256={name:digest(HERE/name) for name in source_files},
                  data_sha256={str(p.relative_to(ROOT)):digest(p) for p in data_files},
                  notes=['Layer 1 and head share the original pair projection in every variant.',
                         'Fix 1 adds independently optimized half-identity projections in layers 2--5.',
                         'The OD-pair prediction head and its pressure scoring term are unchanged.',
                         'Attention, topology, distance, within-community target treatment are unchanged.',
                         'No residual or auxiliary objective is added.',
                         'Test CE/CPC/NRMSE are conditional nonself hidden-destination metrics.',
                         'CPC/NRMSE use the supplied hidden-origin totals, not predicted totals.',
                         'Within-community error is a diagnostic, not independent held-out evaluation.',
                         'Legacy inactive parameters are retained; parameter counts include them.'])
    if cfg.get('gravity_mlp', 0):
        manifest['notes'][0] = 'Layer 1 uses the original pair projection in every variant.'
        manifest['notes'][2] = (f'The OD-pair head is replaced by the gravity head MLP([H_A, H_B, t_AB]) with '
                                f'{cfg["gravity_mlp"]} hidden units; CE reaches the routing through H.')
        manifest['notes'][3] = 'Attention, routing topology bias, distance, within-community target treatment are unchanged.'
    if any(v in DISEN_VARIANTS for v in variants):
        manifest['notes'][:4] = [
            'disen variants: parameter-free routing layers, the message of A -> B is v_B (DisenGCN official).',
            'disen variants: GATv2 neighbour weights per layer, a starting at zero; _flow adds log flow share.',
            'disen variants: the last routing iteration keeps its length; the head reads dropout(relu(u)) of layer 5.',
            'disen variants: no attention re-aggregation; pressure, routing topology bias, tau, distance unchanged.']
    if any(v in PAIRGAT_VARIANTS for v in variants):
        manifest['notes'][:4] = [
            'pair_gatv2: fix1 OD-pair routing per layer, the origin pair e_ii initialises the centre.',
            'pair_gatv2: GATv2 aggregation inside every channel, routing mass pi_ij p_ij,k as attention prior.',
            'pair_gatv2: layer output keeps its length; the gravity head reads dropout(relu(h)) of layer 5.',
            f'pair_gatv2: loss = CE + {cfg.get("indep_weight", 0)} x mean nHSIC between the channels of layer 5.']
        manifest['loss'] = 'origin-normalized flow CE' + ''.join(
            f' + {cfg[k]} x {name}' for k, name in (('indep_weight', 'channel nHSIC'), ('dec_weight', 'DEC KL')) if cfg.get(k)) \
            + ' + Adam coupled L2'
        manifest['notes'] += [f'{v}: options {PAIRGAT_VARIANTS[v]}, dec_weight {cfg.get("dec_weight", 0)}'
                              for v in variants if v in PAIRGAT_VARIANTS]
    if not args.summarize_only:
        (output/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
        if args.reuse_original:
            reuse_original(args.reuse_original,output,cfg,g,tensors,H,Hv)
        with patch.object(reference,'make_model',make_ablation_model):
            for seed in cfg['seeds']:
                for variant in variants:
                    directory=output/variant/f'seed{seed}'
                    if not all((directory/name).exists() for name in ('model.pt', 'arrays.npz', 'metrics.json')):
                        reference.train_one(variant,seed,cfg,g,tensors,H,Hv,directory)
    summarize(output,cfg,g,tensors,H,Hv)


if __name__=='__main__':
    main()
