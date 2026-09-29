"""Independent test of a five-start optimizer; no mechanism evidence in selection.

This is a project adaptation of repeated initialization, not an exact method
from the supplied papers. Keep all starts; select by fitted CE before recovery
evaluation. This supplementary test cannot substitute for the full MD gates.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.data import load_graph
from D_mrhgat.synthetic import make_flows
from D_mrhgat.stability import fit_baselines,fit_group,factor_consensus
from D_mrhgat.benchmarks import evaluate
from D_mrhgat.mechanism_judgment import screen_anonymous_factors

def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def run():
    torch.set_num_threads(3)
    parent=ROOT/'results/runs/20260928_md_stability'
    target=parent/'restart_validation'
    target.mkdir(exist_ok=True)
    selection=json.loads((parent/'selection.json').read_text(encoding='utf-8'))
    protocol={'data_seed':15,'scenario':'S2','strength':.5,'seed_groups':[list(range(i,i+5)) for i in range(40,65,5)],
              'profile':selection['selected_profile'],'epochs':selection['audit_fixed_epochs'],
              'selection':'minimum fitted CE within each group; all starts retained',
              'truth_used_for_selection':False,'mechanism_names_used_for_selection':False,
              'scope':'supplementary optimizer validation, not complete original MD gates'}
    write(target/'protocol.json',protocol)
    write(target/'code_version.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [Path(__file__),ROOT/'src/D_mrhgat/stability.py',ROOT/'src/D_mrhgat/model.py']})
    ref=ROOT/'results/runs/20260928_212256_D6_necessary_gate'
    d=load_graph(ref/'input_graph.npz',device='cuda')
    a={k:v.cpu().numpy() for k,v in d.items()}
    p0=np.load(ref/'B_reference.npz')['probability']
    F,truth=make_flows(a['x'],a['K'],p0,a['F'].sum(1),'S2',.5,15)
    flows=torch.as_tensor(np.repeat(F[None],5,0),device='cuda')
    valid=flows.sum(2)>0
    base=fit_baselines(d['x'],d['K'],flows,valid,valid)
    records=[]
    selected_posteriors=[]
    for group,seeds in enumerate(protocol['seed_groups']):
        case=target/f'group{group}'
        case.mkdir(exist_ok=True)
        if (case/'selection.json').exists():
            raw=np.load(case/'posteriors.npz')['raw']
            info=json.loads((case/'fit.json').read_text(encoding='utf-8'))
        else:
            print('RESTART_GROUP',group,seeds,flush=True)
            out,states,info=fit_group(d['x'],d['K'],flows,seeds,base,valid,
                profile=protocol['profile'],fixed_epochs=[protocol['epochs']]*5)
            raw=out['r'].cpu().numpy()
            torch.save(states,case/'models.pt')
            np.savez_compressed(case/'posteriors.npz',raw=raw,F=F)
            write(case/'fit.json',info)
            del out,states
        best=int(np.argmin(info['ce']))
        # Freeze selection before truth-based evaluation or posthoc hypotheses.
        write(case/'selection.json',dict(index=best,seed=seeds[best],criterion='minimum fitted CE',
                                        ce=info['ce'],truth_used=False))
        selected_posteriors.append(raw[best])
        expected=truth['expected']
        fractions=expected/np.maximum(expected.sum(1)[:,None,:],1e-30)
        for index,r in enumerate(raw):
            orders=([0,1,2],[0,2,1])
            errors=[float((np.abs(r[:,o,:]-fractions)*F[:,None,:]).sum()/F.sum()) for o in orders]
            order=orders[int(np.argmin(errors))]
            values=evaluate(F,a['K'],{'r':torch.as_tensor(r[:,order,:])},truth,p0)
            row=dict(group=group,seed=seeds[index],selected=index==best,ce=info['ce'][index],
                     oracle_order=str(order),oracle_matching_for_evaluation_only=True)
            for k in ('comp_edge_auc','comp_node_auc','compet_edge_auc','compet_node_auc'):
                row[k]=values[k]
            records.append(row)
        write(case/'anonymous_hypothesis_screen.json',screen_anonymous_factors(a['x'],F,a['K'],raw[best]))
        pd.DataFrame(records).to_csv(target/'all_starts.csv',index=False)
        print('RESTART_DONE',group,'selected',seeds[best],flush=True)
        torch.cuda.empty_cache()
    selected=pd.DataFrame(records).query('selected')
    selected.to_csv(target/'selected_starts.csv',index=False)
    _,_,agreement=factor_consensus(np.stack(selected_posteriors),F)
    write(target/'selected_agreement.json',agreement)
    write(target/'status.json',dict(completed_groups=5,completed_fits=25,mechanism_names_confirmed=False,
        null_tests_completed=False,input_graph_sha256=hashlib.sha256((ref/'input_graph.npz').read_bytes()).hexdigest()))
    print('RESTART_COMPLETE',selected.to_json(orient='records'),flush=True)

if __name__=='__main__':
    run()
