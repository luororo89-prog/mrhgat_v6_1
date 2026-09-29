"""Twenty-refit NC/ND validation for the surviving aggregate none model."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.data import load_graph,masks
from D_mrhgat.model import MRHGAT
from D_mrhgat.train import fit,fit_baseline
from D_mrhgat.synthetic import make_flows
from D_mrhgat.benchmarks import evaluate
from D_mrhgat.batched_train import fit_batch
from D_mrhgat.tests_posthoc import directional_prerequisites
from D_mrhgat.nulls import null_probability,decide

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def run_null_validation(run,device='cuda',batch_size=8):
    run=Path(run)
    torch.set_num_threads(3)
    current=json.loads((ROOT/'reports/D6_progress.json').read_text(encoding='utf-8'))
    if current.get('status')!='identification_passed_null_validation_pending' or current.get('selected')!='none':
        raise ValueError('All identification gates must pass for the none model first')
    protocol={'repeats':20,'batch_size':batch_size,'seeds':[0,1,2,3,4],
        'order':[['S0',.3],['S1',.3],['S1',.5],['S2',.3],['S2',.5],['S3',.3],['S3',.5]],
        'S0_strengths_equivalent':[.1,.3,.5],
        'early_stop':'false positives >1 or S2 maximum possible detections <4; no discarded completed seeds',
        'NC':'rho_comp goes to base',
        'ND':'rho_compet permuted within supply terciles; base absorbs difference; reject negative base mass',
        'thresholds':{'z':1.96,'holm_p':.05,'S2_min_detection':4,'max_false_positive':1},
        'null_training':'same observed-case seed and selected full-data epoch count; baseline refit on every null',
        'regression':'supply and raw inflow controls only; no capacity, time, or imputed workplace share'}
    write_json(run/'null_validation_protocol.json',protocol)
    files=[Path(__file__),ROOT/'src/D_mrhgat/nulls.py',ROOT/'src/D_mrhgat/batched_train.py',ROOT/'src/D_mrhgat/tests_posthoc.py']
    write_json(run/'null_code_version.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    d=load_graph(run/'input_graph.npz',device=device)
    a={k:v.cpu().numpy() for k,v in d.items()}
    p0=np.load(run/'B_reference.npz')['probability']
    s=a['x'].mean(1)
    result=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    cases,tests,group_results=[],[],[]
    null_count=0
    all_pass=True
    for scenario,phi in protocol['order']:
        group=[]
        for seed in range(5):
            case=run/f'none_{scenario}_{round(phi*100):03d}_seed{seed}'
            generated,truth=make_flows(a['x'],a['K'],p0,a['F'].sum(1),scenario,phi,seed)
            if scenario=='S0':
                for other_phi in (.1,.5):
                    equivalent,_=make_flows(a['x'],a['K'],p0,a['F'].sum(1),scenario,other_phi,seed)
                    np.testing.assert_array_equal(generated,equivalent)
            F=generated
            sim={**d,'F':torch.as_tensor(F,device=device)}
            if (case/'model.pt').exists():
                stored=np.load(case/'synthetic.npz')['F']
                np.testing.assert_array_equal(F,stored)
                metrics=json.loads((case/'metrics.json').read_text(encoding='utf-8'))
                model=MRHGAT(int_channel='none').to(device)
                model.load_state_dict(torch.load(case/'model.pt',map_location=device,weights_only=True))
                with torch.no_grad():
                    output=model(d['x'],sim['F'],d['K'])
            else:
                started=time.time()
                print(f'START observed {scenario} {phi} seed={seed}',flush=True)
                train,evaluation=masks(sim,0)
                bcv,_=fit_baseline(sim,train,evaluation)
                _,_,cvinfo=fit(sim,'none',seed,bcv.state_dict(),train,evaluation)
                bfull,_=fit_baseline(sim)
                model,output,fullinfo=fit(sim,'none',seed,bfull.state_dict(),epochs=cvinfo['best_epoch'])
                with torch.no_grad():
                    bp=bfull(sim['x'],sim['K']).exp().cpu().numpy()
                metrics=evaluate(F,a['K'],output,truth,bp)
                metrics.update(variant='none',scenario=scenario,phi=phi,seed=seed,cv_epoch=cvinfo['best_epoch'],
                    cv_ce=cvinfo['ce'],full_ce=fullinfo['ce'],seconds=time.time()-started)
                result['records'].append(metrics)
                pd.DataFrame(result['records']).to_csv(ROOT/'results/D6_synthetic.csv',index=False)
                case.mkdir(exist_ok=True)
                torch.save(model.state_dict(),case/'model.pt')
                np.savez_compressed(case/'synthetic.npz',F=F,expected=truth['expected'],displacers=truth['displacers'])
                write_json(case/'metrics.json',metrics)
                write_json(case/'training.json',{'cv':cvinfo,'full':fullinfo})
                del bcv,bfull
            observed=directional_prerequisites(a['x'],F,a['K'],output['r'].cpu().numpy(),'none')
            rho=output['rho'].cpu().numpy()
            logp=output['logp'].cpu().numpy()
            del output,model,sim
            detected={}
            valid_case=True
            for kind in (['NC','ND'] if scenario=='S0' else ['ND']):
                flows=[]
                for rep in range(20):
                    scenario_id={'S0':0,'S1':1,'S2':2,'S3':3}[scenario]
                    rng=np.random.default_rng(np.random.SeedSequence([20260928,scenario_id,round(phi*100),seed,kind=='ND',rep]))
                    probability=null_probability(rho,logp,s,kind,rng)
                    flows.append(np.stack([rng.multinomial(int(total),p/p.sum()) for total,p in zip(F.sum(1),probability)]).astype(np.float32))
                np.savez_compressed(case/f'null_flows_{kind}.npz',F=np.stack(flows).astype(np.int32))
                null_stats=[]
                for start in range(0,20,batch_size):
                    stop=min(start+batch_size,20)
                    write_json(ROOT/'reports/D6_progress.json',{'status':'null_validation_running','run_directory':str(run),
                        'scenario':scenario,'phi':phi,'seed':seed,'null_type':kind,'null_range':[start,stop],
                        'completed_null_refits':null_count,'completed_simulations':len(result['records'])})
                    print(f'NULL {scenario} {phi} seed={seed} {kind} {start}:{stop}',flush=True)
                    batch=torch.as_tensor(np.stack(flows[start:stop]),device=device)
                    fitted,params=fit_batch(d['x'],d['K'],batch,'none',[seed]*(stop-start),epochs=metrics['cv_epoch'])
                    for index in range(stop-start):
                        values=directional_prerequisites(a['x'],flows[start+index],a['K'],fitted['r'][index].cpu().numpy(),'none')
                        values.update(rep=start+index,training_seed=seed)
                        null_stats.append(values)
                    torch.save({k:v.cpu() for k,v in params.items()},case/f'null_models_{kind}_{start}_{stop}.pt')
                    null_count+=stop-start
                    del fitted,params,batch
                    if device=='cuda':
                        torch.cuda.empty_cache()
                write_json(case/f'null_statistics_{kind}.json',null_stats)
                keys=['comp_ADJ_beta','comp_CRS_beta'] if kind=='NC' else ['compet_ADJ_beta','compet_CRS_beta']
                decisions=decide(observed,null_stats,keys,[1,1])
                detected[kind]=any(row['detected'] for row in decisions)
                valid_case=valid_case and all(row['valid'] for row in decisions)
                for row in decisions:
                    row.update(scenario=scenario,phi=phi,seed=seed,null_type=kind,replicates=20)
                    tests.append(row)
                pd.DataFrame(tests).to_csv(ROOT/'results/D6_null_tests.csv',index=False)
            positive=any(detected.values()) if scenario=='S0' else detected['ND']
            case_result={'scenario':scenario,'phi':phi,'seed':seed,'valid':valid_case,
                'detected':positive,'null_refits':40 if scenario=='S0' else 20}
            group.append(case_result)
            cases.append(case_result)
            pd.DataFrame(cases).to_csv(ROOT/'results/D6_null_cases.csv',index=False)
            print(json.dumps(case_result,ensure_ascii=False),flush=True)
            hits=sum(r['detected'] for r in group)
            unavoidable_failure=(hits+(5-len(group))<4) if scenario=='S2' else hits>1
            if not valid_case or unavoidable_failure:
                break
        hits=sum(r['detected'] for r in group)
        passed=all(r['valid'] for r in group) and len(group)==5 and (hits>=4 if scenario=='S2' else hits<=1)
        group_results.append({'scenario':scenario,'phi':phi,'completed':len(group),'detections':hits,
            'required':'at least 4/5' if scenario=='S2' else 'at most 1/5','passed':passed})
        result.update(null_groups=group_results,null_cases=cases,null_tests=tests,null_refits=null_count)
        write_json(run/'metrics.json',result)
        if not passed:
            all_pass=False
            break
    final_status='main_synthetic_gates_passed_remaining_checks_pending' if all_pass else 'synthetic_null_gate_not_met'
    result['status']=final_status
    write_json(run/'metrics.json',result)
    report=ROOT/'reports/D6_model_selection.md'
    text=report.read_text(encoding='utf-8')
    text=text.split('## 当前训练状态')[0]
    text+='## 误报与检出对照\n\n每份数据每种对照重训练20次；系数方向符合、z>1.96且机制内Holm校正p<0.05才计检出。\n\n'
    text+='| 情景 | 强度 | 已完成份数 | 检出数 | 要求 | 门槛 |\n|---|---:|---:|---:|---|---|\n'
    for g in group_results:
        text+=f"| {g['scenario']} | {g['phi']:.0%} | {g['completed']}/5 | {g['detections']} | {g['required']} | {'满足' if g['passed'] else '未满足'} |\n"
    text+='\n无机制情景最多允许1/5误报；已出现2份误报时，剩余种子不能改变验收结论，按预定规则停止。S2最多可能检出数不足4时同样停止。未运行的种子不计作阴性或阳性。\n\n'
    text+='S0在三个强度下的生成流量逐元素相同，重复情景只计算一次并记录等价性。\n\n'
    text+='## 当前训练状态\n\n'
    text+=('主要模拟门槛已满足，尚需完成其余预定检查。' if all_pass else
        '当前候选配置未满足完整模拟验收的必要条件，按方案停在模拟阶段；不进入真实MR-HGAT机制训练。')
    text+=f'\n\n已完成{len(result["records"])}次模拟拟合和{null_count}次对照重训练。未生成真实社区机制分解、机制再分类或依赖性韧性解释。\n\n运行记录：{run}\n'
    report.write_text(text,encoding='utf-8')
    write_json(ROOT/'reports/D6_progress.json',{'status':final_status,'run_directory':str(run),'completed_simulations':len(result['records']),
        'completed_null_refits':null_count,'selected':'none' if all_pass else None})
    ready=json.loads((ROOT/'reports/A_readiness.json').read_text(encoding='utf-8'))
    ready.update(status=final_status,ready_for_training=False,blockers=[
        '主要模拟门槛通过，尚有预定检查未完成' if all_pass else '完整模拟验收必要条件未满足，无可进入真实训练的配置'])
    write_json(ROOT/'reports/A_readiness.json',ready)
    print(json.dumps({'status':final_status,'groups':group_results,'null_refits':null_count},ensure_ascii=False),flush=True)
    return all_pass

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    run_null_validation(args.run,args.device,args.batch_size)
