"""Check the plan's mandatory direction before expensive significance nulls."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.data import load_graph,masks
from D_mrhgat.model import MRHGAT
from D_mrhgat.train import fit,fit_baseline
from D_mrhgat.synthetic import make_flows
from D_mrhgat.benchmarks import evaluate
from D_mrhgat.tests_posthoc import directional_prerequisites

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def complete(run,device='cuda'):
    run=Path(run)
    torch.set_num_threads(3)
    files=[Path(__file__),ROOT/'src/D_mrhgat/tests_posthoc.py']
    write_json(run/'direction_check_code.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    result=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    records,decisions=result['records'],result['decisions']
    d=load_graph(run/'input_graph.npz',device=device)
    a={k:v.cpu().numpy() for k,v in d.items()}
    reference=np.load(run/'B_reference.npz')
    p0=reference['probability']
    directions=[]
    accepted=None
    for variant in ('nested','independent','none'):
        decision=next((item for item in decisions if item['variant']==variant),None)
        if decision is None:
            group=[]
            for seed in range(5):
                started=time.time()
                print(f'START variant={variant} seed={seed}',flush=True)
                write_json(ROOT/'reports/D6_progress.json',{'run_directory':str(run),'variant':variant,'seed':seed,
                    'stage':'S2_30_percent_necessary_gate','completed':len(records)})
                F,truth=make_flows(a['x'],a['K'],p0,a['F'].sum(1),'S2',.3,seed)
                sim={**d,'F':torch.as_tensor(F,device=device)}
                train,evaluation=masks(sim,0)
                bcv,_=fit_baseline(sim,train,evaluation)
                _,_,cvinfo=fit(sim,variant,seed,bcv.state_dict(),train,evaluation)
                bfull,_=fit_baseline(sim)
                model,out,fullinfo=fit(sim,variant,seed,bfull.state_dict(),epochs=cvinfo['best_epoch'])
                with torch.no_grad():
                    bp=bfull(sim['x'],sim['K']).exp().cpu().numpy()
                metrics=evaluate(F,a['K'],out,truth,bp)
                metrics.update(variant=variant,scenario='S2',phi=.3,seed=seed,cv_epoch=cvinfo['best_epoch'],
                    cv_ce=cvinfo['ce'],full_ce=fullinfo['ce'],seconds=time.time()-started)
                records.append(metrics)
                group.append(metrics)
                pd.DataFrame(records).to_csv(ROOT/'results/D6_synthetic.csv',index=False)
                case=run/f'{variant}_S2_030_seed{seed}'
                case.mkdir(exist_ok=True)
                torch.save(model.state_dict(),case/'model.pt')
                np.savez_compressed(case/'synthetic.npz',F=F,expected=truth['expected'],displacers=truth['displacers'])
                write_json(case/'metrics.json',metrics)
                write_json(case/'training.json',{'cv':cvinfo,'full':fullinfo})
                print(json.dumps(metrics,ensure_ascii=False),flush=True)
                del model,out,sim,bcv,bfull
                if device=='cuda':
                    torch.cuda.empty_cache()
            means={key:float(np.mean([r[key] for r in group])) for key in (
                'comp_edge_auc','comp_residual_auc','comp_node_auc','compet_edge_auc','compet_node_auc')}
            means['comp_topology_correlation']=float(pearsonr(
                [r['topo_'+k+'_pred'] for r in group for k in ('adj','crs')],
                [r['topo_'+k+'_truth'] for r in group for k in ('adj','crs')]).statistic)
            checks={'comp_edge':means['comp_edge_auc']>=.85,'above_residual':means['comp_edge_auc']>means['comp_residual_auc'],
                'comp_node':means['comp_node_auc']>=.85,'compet_edge':means['compet_edge_auc']>=.75,
                'compet_node':means['compet_node_auc']>=.70,'topology':means['comp_topology_correlation']>=.7}
            decision={'variant':variant,'means':means,'checks':checks,'necessary_gate_passed':all(checks.values())}
            decisions.append(decision)
        # Sign is a necessary condition for the plan's null-based detection.
        if decision['necessary_gate_passed']:
            count=0
            for seed in range(5):
                case=run/f'{variant}_S2_030_seed{seed}'
                F=np.load(case/'synthetic.npz')['F']
                model=MRHGAT(int_channel=variant).to(device)
                model.load_state_dict(torch.load(case/'model.pt',map_location=device,weights_only=True))
                with torch.no_grad():
                    out=model(d['x'],torch.as_tensor(F,device=device),d['K'])
                values=directional_prerequisites(a['x'],F,a['K'],out['r'].cpu().numpy(),variant)
                count+=int(values['required_compet_direction'])
                values.update(variant=variant,seed=seed,scenario='S2',phi=.3)
                directions.append(values)
                del model,out
            decision['direction_eligible_runs']=count
            decision['direction_necessary_passed']=count>=4
            pd.DataFrame(directions).to_csv(ROOT/'results/D6_direction_prerequisites.csv',index=False)
            if count>=4:
                accepted=variant
                break
        else:
            decision['direction_necessary_passed']=None
        write_json(run/'metrics.json',{'decisions':decisions,'records':records,'directions':directions})
    status='necessary_gate_passed_full_validation_pending' if accepted else 'necessary_recovery_gate_not_met'
    write_json(run/'metrics.json',{'decisions':decisions,'records':records,'directions':directions,'status':status})
    rows=['# 模拟识别验收','',
        '当前口径：总人流、原九项供给属性、无出行时间、无容量校正。S2、30%机制强度，每配置5份模拟；空间第0折和邻接缓冲选择轮数。', '',
        '| 内部通道 | 补偿边AUC ≥0.85 | 残差基准 | 补偿节点AUC ≥0.85 | 竞争边AUC ≥0.75 | 竞争节点AUC ≥0.70 | 拓扑相关 ≥0.7 | 关系方向 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for item in decisions:
        m=item['means']
        direction=(str(item['direction_eligible_runs'])+'/5满足必要方向') if 'direction_eligible_runs' in item else '识别门槛未满足，未进入方向检验'
        rows.append(f"| {item['variant']} | {m['comp_edge_auc']:.3f} | {m['comp_residual_auc']:.3f} | {m['comp_node_auc']:.3f} | {m['compet_edge_auc']:.3f} | {m['compet_node_auc']:.3f} | {m['comp_topology_correlation']:.3f} | {direction} |")
    rows+=['','补偿边AUC还须高于残差基准。均值包含全部5个固定种子，未筛选种子或放宽阈值。拓扑相关为5份×ADJ/CRS的10格，不是分级人流检验。', '',
        '## 关系方向的必要条件','',
        '原方案步骤6规定：独立内部通道必须同时满足Self对流入的系数为负、Disp对流入的系数为正。这里以低供给来源的流入人数取代容量校正压力，并控制综合供给评价；不使用时间和未知工作地比例。', '',
        '方向不符合时，无论对照分布如何均不能满足原方案的检出条件。要求5份中至少4份检出，因此至少4份必须先符合方向。方向检查没有p值，不等于显著性检验；尚未运行的对照不得记为通过。', '']
    if directions:
        rows+=['| 配置 | 种子 | Self系数（要求<0） | Disp系数（要求>0） |','|---|---:|---:|---:|']
        for row in directions:
            rows.append(f"| {row['variant']} | {row['seed']} | {row['self_low_inflow_beta']:.4f} | {row['disp_low_inflow_beta']:.4f} |")
    rows+=['','## 当前结论','',
        '必要条件已满足，完整情景与对照验收仍待完成。' if accepted else
        '当前三种配置均未满足进入真实训练所需的全部必要条件。按方案验收规则停在模拟阶段。', '',
        '未执行真实MR-HGAT机制训练、真实机制再分类及依赖性韧性解释；不输出真实社区补偿、竞争或挤出结论。', '',
        '未运行其余情景、强度及20次误报对照，不声称完成完整模拟验证。独立扰动模拟所需的v5完整定义未定位，不自行补造。', '',
        f'当前完整运行记录：{run}','']
    (ROOT/'reports/D6_model_selection.md').write_text('\n'.join(rows),encoding='utf-8')
    write_json(ROOT/'reports/D6_progress.json',{'status':status,'run_directory':str(run),'completed':len(records),'selected':accepted})
    readiness=json.loads((ROOT/'reports/A_readiness.json').read_text(encoding='utf-8'))
    readiness.update(status=status,ready_for_training=False,blockers=[
        '完整模拟检验待完成' if accepted else '模拟必要条件未满足，无可进入真实机制训练的配置'])
    write_json(ROOT/'reports/A_readiness.json',readiness)
    print(json.dumps({'status':status,'decisions':decisions},ensure_ascii=False),flush=True)
    return accepted

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    complete(args.run,args.device)
