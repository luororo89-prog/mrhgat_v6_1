"""Synthetic-only oracle diagnostic inspired by FactorGCN evaluation.
Ground truth is used only to assess recoverability up to permutation, never to
train, select a fitted model, rename real-data channels or replace official AUCs.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.synthetic import make_flows
from D_mrhgat.benchmarks import evaluate

run=ROOT/'results/runs/20260928_md_stability'
reference=ROOT/'results/runs/20260928_212256_D6_necessary_gate'
graph=np.load(reference/'input_graph.npz')
p0=np.load(reference/'B_reference.npz')['probability']
F,truth=make_flows(graph['x'],graph['K'],p0,graph['F'].sum(1),'S2',.5,3)
expected=truth['expected']
fractions=expected/np.maximum(expected.sum(1)[:,None,:],1e-30)
records=[]
for profile in ('plain','hsic','factor_hsic'):
    if not (run/f'development_{profile}_S2_050_data3/posteriors.npz').exists():
        continue
    data=np.load(run/f'development_{profile}_S2_050_data3/posteriors.npz')
    for seed,r in enumerate(data['raw']):
        orders=([0,1,2],[0,2,1])
        costs=[float((np.abs(r[:,order,:]-fractions)*F[:,None,:]).sum()/F.sum()) for order in orders]
        selected=int(np.argmin(costs))
        fixed=evaluate(F,graph['K'],{'r':torch.as_tensor(r)},truth,p0)
        oracle=evaluate(F,graph['K'],{'r':torch.as_tensor(r[:,orders[selected],:])},truth,p0)
        row={'profile':profile,'seed':seed,'oracle_order':str(orders[selected]),
             'fixed_error':costs[0],'oracle_error':costs[selected],
             'truth_used_only_for_oracle_diagnostic':True}
        for key in ('comp_edge_auc','comp_node_auc','compet_edge_auc','compet_node_auc'):
            row['fixed_'+key]=fixed[key]
            row['oracle_'+key]=oracle[key]
        records.append(row)
pd.DataFrame(records).to_csv(run/'permutation_diagnostic.csv',index=False)
print(pd.DataFrame(records).to_string(index=False))
