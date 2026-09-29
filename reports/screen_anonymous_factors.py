"""Export post-fit hypothesis screens without inferring names from directions."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.mechanism_judgment import screen_anonymous_factors

run=ROOT/'results/runs/20260928_md_stability'
graph=np.load(ROOT/'results/runs/20260928_212256_D6_necessary_gate/input_graph.npz')
protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
records=[]
for case in sorted(run.glob('*_S2_*_data*')):
    if not (case/'posteriors.npz').exists():
        continue
    data=np.load(case/'posteriors.npz')
    seeds=(protocol['initialization_seeds'] if case.name.startswith('development_') else
           protocol['additional_ensemble_seeds'] if case.name.startswith('audit_repeat_') else
           protocol['audit_initialization_seeds'])
    case_records=[]
    for seed,r in zip(seeds,data['raw']):
        for values in screen_anonymous_factors(graph['x'],data['F'],graph['K'],r):
            row={'case':case.name,'seed':seed,'factor_index':values['factor_index'],
                 'mechanism_label':values['mechanism_label'],'null_tests_completed':False}
            for hypothesis in ('compensation','competition'):
                for topology,beta in values[hypothesis+'_direction'].items():
                    row[hypothesis+'_'+topology+'_beta']=beta
            case_records.append(row)
    (case/'anonymous_hypothesis_screen.json').write_text(
        json.dumps(case_records,ensure_ascii=False,indent=2),encoding='utf-8')
    records.extend(case_records)
pd.DataFrame(records).to_csv(run/'anonymous_hypothesis_screen.csv',index=False)
files=[ROOT/'src/D_mrhgat/mechanism_judgment.py',ROOT/'src/D_mrhgat/tests_posthoc.py',Path(__file__)]
(run/'posthoc_code_version.json').write_text(json.dumps(
    {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
print('Exported',len(records),'anonymous factor screens; no significance or mechanism name inferred.')
