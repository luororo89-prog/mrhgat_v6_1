"""E0--E3: unchanged M1 plus independent weak mechanism ranking.

Run with --config and optionally --experiments E0 E1 E2 E3 --weights .01 .1 1.
Every setting uses all configured seeds. E4 is conditional on validation gates.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

import run_od_relation_v13 as training
from run_od_ablation_v13 import setup, digest
from model_od_mechanism_v13 import make_mechanism_model
from mechanism_rank_v13 import MechanismReference, ranked_forward

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def diagnostics(model, g, tensors, H, Hv, ref, directory):
    W=g['W'];off=~np.eye(len(W),dtype=bool)
    records=[];groups=[]
    for split, context_mask, population in (
        ('validation', (W>0)&off&~H&~Hv, Hv),
        ('test', (W>0)&off&~H, H),
        ('visible', (W>0)&off&~H, (W>0)&off&~H)):
        context=W*context_mask
        model.eval()
        with torch.no_grad(): out=training.forward_context(model,'mech_full',g,tensors,context,.5)
        oi,di=np.nonzero(population)
        q=out['p'].numpy().transpose(0,2,1)[oi,di]
        components=ref.components(context,oi,di)
        references,pairs=ref.pairs(context,oi,di,seed=999,max_pairs=32768,visible_only=False)
        np.savez_compressed(directory/f'gates_{split}.npz',origin=oi,destination=di,q=q,
                            references=references,**components)
        for k,name in enumerate(('comp','compete')):
            a,b=pairs[k]
            row=dict(split=split,mechanism=name,n=len(q),pair_count=len(a),mean=q[:,k].mean(),
                     variance=q[:,k].var(),low=(q[:,k]<.05).mean(),high=(q[:,k]>.95).mean(),
                     both_high=(q>.5).all(1).mean())
            for j,label in enumerate(('comp','compete')):
                aa,bb=pairs[j]
                delta=q[aa,k]-q[bb,k]
                row['rank_'+label]=float(np.mean((delta>0)+.5*(delta==0))) if len(delta) else np.nan
                valid=np.isfinite(references[:,j])
                row['spearman_'+label]=float(spearmanr(q[valid,k],references[valid,j]).statistic) if valid.sum()>2 else np.nan
            records.append(row)
            for feature,values in components.items():
                # Cutpoints learned from training references and stored separately;
                # no validation or test refitting of quantile bins.
                cuts=ref.component_cuts[feature]
                bins=np.digitize(values,cuts)
                for b_ in range(len(cuts)+1):
                    use=(bins==b_)&np.isfinite(values)
                    groups.append(dict(split=split,mechanism=name,feature=feature,bin=b_,n=int(use.sum()),
                                       gate_mean=float(q[use,k].mean()) if use.any() else np.nan))
        if split=='test':
            saved=np.load(directory/'arrays.npz')
            np.testing.assert_array_equal(out['score'].numpy(),saved['scores'])
    pd.DataFrame(records).to_csv(directory/'mechanism_diagnostics.csv',index=False)
    pd.DataFrame(groups).to_csv(directory/'quantile_response.csv',index=False)
    (directory/'replay.json').write_text(json.dumps({'scores_exact':True}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--experiments',nargs='+',default=['E0','E1','E2','E3'])
    ap.add_argument('--weights',nargs='+',type=float,default=[.01,.1,1.])
    ap.add_argument('--seeds',nargs='+',type=int)
    args=ap.parse_args()
    cfg,g,tensors,H,Hv=setup(args.config)
    if args.seeds is not None:cfg['seeds']=args.seeds
    W=g['W'];off=~np.eye(len(W),dtype=bool);visible=(W>0)&off&~H&~Hv
    ref=MechanismReference(g,tensors[1].numpy(),tensors[2].numpy(),W*visible)
    oi,di=np.nonzero(visible)
    comp=ref.components(W*visible,oi,di)
    ref.component_cuts={k:np.unique(np.nanquantile(v,[.2,.4,.6,.8])) for k,v in comp.items()}
    output=ROOT/'results_od_mechanism'/cfg['name'];output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'fixed_split.npz',ids=g['ids'],W=W,H=H,Hv=Hv)
    state={**ref.state(),'quantile_cutpoints':{k:v.tolist() for k,v in ref.component_cuts.items()}}
    (output/'reference_rules.json').write_text(json.dumps(state,indent=2))
    files=['model_od_mechanism_v13.py','run_od_relation_v13.py','mechanism_rank_v13.py','run_od_identifiability_v13.py']
    inputs=[Path(cfg[k]) for k in ('supply_file','flow_file','adjacency_file')]
    inputs += [ROOT/cfg['road_impedance'],ROOT/cfg['ruler_from']/'ruler.json',ROOT/'data/processed/communities_candidate.parquet']
    (output/'manifest.json').write_text(json.dumps(dict(torch_version=torch.__version__,
        source_sha256={p:digest(HERE/p) for p in files},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        selection='minimum validation CE; no test tuning; E4 only after validation specificity',
        caveat='Within-community counts remain visible in phi under the unchanged M1 protocol.',
        proxy_interpretation='relative weak references, not causal mechanism labels'),indent=2))
    original=training.forward_context
    for experiment in args.experiments:
        if experiment not in ('E0','E1','E2','E3'):raise ValueError(experiment)
        for weight in ([0.] if experiment=='E0' else args.weights):
            setting=f'{experiment}_w{weight:g}'
            run_cfg={**cfg,'comp_rank_weight':weight if experiment in ('E1','E3') else 0.,
                     'compete_rank_weight':weight if experiment in ('E2','E3') else 0.}
            for seed in cfg['seeds']:
                directory=output/setting/f'seed{seed}'
                directory.mkdir(parents=True,exist_ok=True)
                cp=directory/'config.yaml'
                if cp.exists() and yaml.safe_load(cp.read_text())!=run_cfg:raise ValueError('Refusing config overwrite')
                cp.write_text(yaml.safe_dump(run_cfg,allow_unicode=True))
                if not (directory/'metrics.json').exists():
                    with patch.object(training,'make_model',make_mechanism_model), \
                         patch.object(training,'forward_context',ranked_forward(original,ref)):
                        training.train_one('mech_full',seed,run_cfg,g,tensors,H,Hv,directory)
                model=make_mechanism_model('mech_full',run_cfg,len(W))
                model.load_state_dict(torch.load(directory/'model.pt',weights_only=True))
                diagnostics(model,g,tensors,H,Hv,ref,directory)
                print('DIAGNOSTICS',setting,seed,flush=True)


if __name__=='__main__':main()
