"""Frozen development/audit protocol for factor stabilization."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import shutil
import time
import numpy as np
import torch
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.data import load_graph, masks
from D_mrhgat.synthetic import make_flows
from D_mrhgat.benchmarks import evaluate
from D_mrhgat.stability import fit_baselines, fit_group, factor_consensus

def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def run(stage, directory):
    torch.set_num_threads(3)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    frozen=ROOT/'results/runs/20260928_212256_D6_necessary_gate'
    d=load_graph(frozen/'input_graph.npz',device='cuda')
    a={k:v.cpu().numpy() for k,v in d.items()}
    p0=np.load(frozen/'B_reference.npz')['probability']
    if stage=='development':
        protocol={'data_seed':3,'scenario':'S2','strength':.5,'initialization_seeds':[0,1,2,3,4],
            'profiles':['plain','hsic','factor_hsic'],'folds':5,'epoch_selection':'per-seed median of five fold best epochs',
            'max_epochs':400,'patience':20,'lr':.02,'matrix_l2':1e-4,'routing_iterations':0,'hsic_weight':.1,
            'profile_selection':'minimum mean held-out CE; no truth metrics',
            'factor_alignment':'diagnostic only; forbidden for primary recovery metrics',
            'semantic_ordering':'anonymous during training; posthoc tests determine mechanism; no truth naming',
            'factor_discrimination_weight':.5,
            'audit_data_seeds':[10,11,12,13,14],'audit_initialization_seeds':[20,21,22,23,24],
            'audit_strengths':[.3,.5], 'additional_ensemble_seeds':[30,31,32,33,34],
            'thresholds':{'comp_edge':.85,'comp_node':.85,'compet_edge':.75,'compet_node':.70},
            'original_graph_sha256':hashlib.sha256((frozen/'input_graph.npz').read_bytes()).hexdigest(),
            'causal_claims_allowed':False}
        write(directory/'protocol.json',protocol)
        for name in ('model.py','stability.py','run_stability.py','synthetic.py','benchmarks.py'):
            shutil.copy2(ROOT/'src/D_mrhgat'/name,directory/name)
        F, truth=make_flows(a['x'],a['K'],p0,a['F'].sum(1),'S2',.5,3)
        flows=torch.as_tensor(np.repeat(F[None],5,0),device='cuda')
        sim={**d,'F':flows[0]}
        cv=json.loads((directory/'development_cv.json').read_text(encoding='utf-8')) if (directory/'development_cv.json').exists() else {p:[] for p in protocol['profiles']}
        for profile in protocol['profiles']:
            cv.setdefault(profile,[])
        for fold in range(5):
            tr,va=masks(sim,fold)
            train_rows=tr[None].expand(5,-1)
            eval_rows=va[None].expand(5,-1)
            if all(len(cv[p])>fold for p in protocol['profiles']):
                continue
            base=fit_baselines(d['x'],d['K'],flows,train_rows,eval_rows)
            for profile in protocol['profiles']:
                if len(cv[profile])>fold:
                    continue
                print('CV',fold,profile,flush=True)
                out,states,info=fit_group(d['x'],d['K'],flows,protocol['initialization_seeds'],base,train_rows,eval_rows,profile)
                cv[profile].append(info)
                write(directory/'development_cv.json',cv)
                print('CV_DONE',fold,profile,info['best_epoch'],info['ce'],flush=True)
                del out,states
                torch.cuda.empty_cache()
        scores={p:float(np.mean([v['ce'] for v in cv[p]])) for p in cv}
        selected=min(scores,key=scores.get)
        medians={p:np.median([v['best_epoch'] for v in cv[p]],axis=0).astype(int).tolist() for p in cv}
        selection={'selected_profile':selected,'mean_fivefold_ce':scores,'per_seed_median_epochs':medians,
                   'audit_fixed_epochs':int(np.median(medians[selected])), 'truth_used_for_selection':False}
        write(directory/'selection.json',selection)
        print('SELECTED',json.dumps(selection),flush=True)
        jobs=[('S2',.5,3,protocol['initialization_seeds'],p,medians[p],'development') for p in protocol['profiles']]
    else:
        protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
        selection=json.loads((directory/'selection.json').read_text(encoding='utf-8'))
        jobs=[('S2',phi,seed,protocol['audit_initialization_seeds'],selection['selected_profile'],
               [selection['audit_fixed_epochs']]*5,'audit') for phi in protocol['audit_strengths'] for seed in protocol['audit_data_seeds']]
        jobs += [('S2',.5,10,protocol['additional_ensemble_seeds'],selection['selected_profile'],
                  [selection['audit_fixed_epochs']]*5,'audit_repeat')]
    records=[]
    target=directory/(stage+'_metrics.csv')
    for scenario,phi,data_seed,seeds,profile,epochs,phase in jobs:
        case=directory/f'{phase}_{profile}_{scenario}_{round(phi*100):03d}_data{data_seed}'
        if (case/'metrics.json').exists():
            records.extend(json.loads((case/'metrics.json').read_text(encoding='utf-8')))
            pd.DataFrame(records).to_csv(target,index=False)
            continue
        print('FIT',phase,profile,scenario,phi,data_seed,seeds,epochs,flush=True)
        F,truth=make_flows(a['x'],a['K'],p0,a['F'].sum(1),scenario,phi,data_seed)
        flows=torch.as_tensor(np.repeat(F[None],5,0),device='cuda')
        valid=flows.sum(2)>0
        base=fit_baselines(d['x'],d['K'],flows,valid,valid)
        out,states,info=fit_group(d['x'],d['K'],flows,seeds,base,valid,profile=profile,fixed_epochs=epochs)
        r=out['r'].cpu().numpy()
        diagnostic_consensus,aligned,agreement=factor_consensus(r,F)
        consensus=r.mean(0)
        agreement['matching_used_for_primary_metrics']=False
        bp=torch.softmax((d['x']@base['beta'][0])[None,:]+base['topo'][0][d['K']],1).cpu().numpy()
        case=directory/f'{phase}_{profile}_{scenario}_{round(phi*100):03d}_data{data_seed}'
        case.mkdir(exist_ok=True)
        torch.save(states,case/'models.pt')
        np.savez_compressed(case/'posteriors.npz',raw=r,consensus=consensus,F=F)
        write(case/'fit.json',info)
        write(case/'agreement.json',agreement)
        for i,rr in enumerate([*r,consensus]):
            kind='raw' if i<5 else 'consensus'
            metrics=evaluate(F,a['K'],{'r':torch.as_tensor(rr)},truth,bp)
            metrics.update(profile=profile,phase=phase,scenario=scenario,phi=phi,data_seed=data_seed,
                seed=seeds[i%5] if kind!='consensus' else -1,kind=kind)
            records.append(metrics)
        write(case/'metrics.json',records[-6:])
        pd.DataFrame(records).to_csv(target,index=False)
        print('DONE',json.dumps(records[-1]),json.dumps(agreement),flush=True)
        del out,states,flows
        torch.cuda.empty_cache()
    assert hashlib.sha256((frozen/'input_graph.npz').read_bytes()).hexdigest()==protocol['original_graph_sha256']
    print('COMPLETE',stage,str(directory),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['development','audit'],required=True)
    p.add_argument('--directory',required=True)
    args=p.parse_args()
    run(args.stage,args.directory)
