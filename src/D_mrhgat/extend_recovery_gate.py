"""Remaining necessary identification tests before significance/null testing."""
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
from D_mrhgat.train import fit,fit_baseline
from D_mrhgat.synthetic import make_flows
from D_mrhgat.benchmarks import evaluate
from D_mrhgat.tests_posthoc import directional_prerequisites

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def render(run,result,status):
    rows=['# 聚合MR-HGAT模拟验收','',
        '当前口径：保留原九项设施评价和总人流；无出行时间、无容量校正、无层级人流拆分。', '',
        '## S2、30%强度的必要识别条件','',
        '| 内部通道 | 补偿边AUC | 残差基准 | 补偿节点AUC | 竞争边AUC | 竞争节点AUC | 拓扑相关 | 方向条件 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for d in result['decisions']:
        m=d['means']
        direction=str(d['direction_eligible_runs'])+'/5满足必要方向' if 'direction_eligible_runs' in d else '识别条件未满足'
        rows.append(f"| {d['variant']} | {m['comp_edge_auc']:.3f} | {m['comp_residual_auc']:.3f} | {m['comp_node_auc']:.3f} | {m['compet_edge_auc']:.3f} | {m['compet_node_auc']:.3f} | {m['comp_topology_correlation']:.3f} | {direction} |")
    rows+=['','## 其余必要识别条件','',
        '| 配置 | 情景 | 强度 | 补偿边AUC | 残差基准 | 补偿节点AUC | 竞争边AUC | 竞争节点AUC | 拓扑相关 | 必要条件 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for d in result.get('extension_decisions',[]):
        m=d['means']
        rows.append(f"| {d['variant']} | {d['scenario']} | {d['phi']:.0%} | {m['comp_edge_auc']:.3f} | {m['comp_residual_auc']:.3f} | {m['comp_node_auc']:.3f} | {m['compet_edge_auc']:.3f} | {m['compet_node_auc']:.3f} | {m['comp_topology_correlation']:.3f} | {'满足' if d['passed'] else '未满足'} |")
    rows+=['','各行均含5个固定种子。补偿边/节点AUC要求≥0.85，补偿边须高于残差基准；S2竞争边AUC要求≥0.75、竞争节点≥0.70；拓扑相关≥0.7。S1/S1s没有竞争真值，竞争AUC的NaN不作为该情景的识别判据。', '',
        '独立内部通道另按原方案步骤6要求Self对低供给来源流入量的系数<0且Disp系数>0。5份中至少4份须先符合方向，才可能满足至少4份检出的条件。方向不符时无需运行对照即可判定不满足必要条件；方向符合不等于统计显著。', '',
        '聚合模型的回归不包含时间、容量或未知工作地比例；流入量为人数。拓扑相关为5份模拟×ADJ/CRS共10格，不是分级人流检验。', '',
        '## 当前训练状态','']
    if status=='necessary_recovery_gate_not_met':
        rows+=['当前没有满足全部必要识别条件的配置。按方案验收规则，停止依赖该门槛的真实MR-HGAT机制训练。']
    else:
        rows+=['尚未完成完整模拟与误报对照验收，暂不能进入真实MR-HGAT机制训练。']
    rows+=['','未产生真实社区的补偿、竞争分解、机制再分类或依赖性的韧性结论。必要条件检查不等于已完成全部情景、全部强度和误报对照。', '',
        f"当前完成{len(result['records'])}次模拟拟合。运行快照、逐次指标及模型参数：{run}",'']
    (ROOT/'reports/D6_model_selection.md').write_text('\n'.join(rows),encoding='utf-8')

def extend(run,device='cuda'):
    run=Path(run)
    torch.set_num_threads(3)
    status=json.loads((ROOT/'reports/D6_progress.json').read_text(encoding='utf-8'))
    variant=status.get('selected')
    if variant is None:
        return False
    result=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    result.setdefault('extension_decisions',[])
    protocol={'order':[['S1',.3],['S1',.5],['S1s',.3],['S1s',.5],['S2',.5]],
        'seeds':[0,1,2,3,4], 'stop_on_any_required_condition_not_met':True,
        'thresholds_unchanged':True,'source':str(Path(__file__)),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(run/'remaining_identification_protocol.json',protocol)
    d=load_graph(run/'input_graph.npz',device=device)
    a={k:v.cpu().numpy() for k,v in d.items()}
    p0=np.load(run/'B_reference.npz')['probability']
    passed=True
    for scenario,phi in protocol['order']:
        group=[]
        for seed in range(5):
            started=time.time()
            write_json(ROOT/'reports/D6_progress.json',{'run_directory':str(run),'variant':variant,
                'stage':'remaining_identification','scenario':scenario,'phi':phi,'seed':seed,'completed':len(result['records']), 'selected':variant})
            print(f'START {variant} {scenario} phi={phi} seed={seed}',flush=True)
            F,truth=make_flows(a['x'],a['K'],p0,a['F'].sum(1),scenario,phi,seed)
            sim={**d,'F':torch.as_tensor(F,device=device)}
            train,evaluation=masks(sim,0)
            bcv,_=fit_baseline(sim,train,evaluation)
            _,_,cvinfo=fit(sim,variant,seed,bcv.state_dict(),train,evaluation)
            bfull,_=fit_baseline(sim)
            model,out,fullinfo=fit(sim,variant,seed,bfull.state_dict(),epochs=cvinfo['best_epoch'])
            with torch.no_grad():
                bp=bfull(sim['x'],sim['K']).exp().cpu().numpy()
            metrics=evaluate(F,a['K'],out,truth,bp)
            if scenario=='S2':
                metrics.update(directional_prerequisites(a['x'],F,a['K'],out['r'].cpu().numpy(),variant))
            metrics.update(variant=variant,scenario=scenario,phi=phi,seed=seed,cv_epoch=cvinfo['best_epoch'],
                cv_ce=cvinfo['ce'],full_ce=fullinfo['ce'],seconds=time.time()-started)
            group.append(metrics)
            result['records'].append(metrics)
            pd.DataFrame(result['records']).to_csv(ROOT/'results/D6_synthetic.csv',index=False)
            case=run/f'{variant}_{scenario}_{round(phi*100):03d}_seed{seed}'
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
            'comp_node':means['comp_node_auc']>=.85,'topology':means['comp_topology_correlation']>=.7}
        if scenario=='S2':
            checks.update(compet_edge=means['compet_edge_auc']>=.75,compet_node=means['compet_node_auc']>=.70,
                direction=sum(r['required_compet_direction'] for r in group)>=4)
        passed=all(checks.values())
        result['extension_decisions'].append({'variant':variant,'scenario':scenario,'phi':phi,'means':means,'checks':checks,'passed':passed})
        result['status']='remaining_identification_in_progress' if passed else 'necessary_recovery_gate_not_met'
        write_json(run/'metrics.json',result)
        render(run,result,result['status'])
        if not passed:
            break
    final_status='identification_passed_null_validation_pending' if passed else 'necessary_recovery_gate_not_met'
    result['status']=final_status
    write_json(run/'metrics.json',result)
    render(run,result,final_status)
    write_json(ROOT/'reports/D6_progress.json',{'status':final_status,'run_directory':str(run),
        'completed':len(result['records']),'selected':variant if passed else None})
    readiness=json.loads((ROOT/'reports/A_readiness.json').read_text(encoding='utf-8'))
    readiness.update(status=final_status,ready_for_training=False,blockers=[
        '完整误报对照验收尚未完成' if passed else '模拟必要识别条件未满足，无可进入真实机制训练的配置'])
    write_json(ROOT/'reports/A_readiness.json',readiness)
    print(json.dumps({'status':final_status,'groups':result['extension_decisions']},ensure_ascii=False),flush=True)
    return passed

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    extend(args.run,args.device)
