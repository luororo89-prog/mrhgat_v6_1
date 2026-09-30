"""Prespecified necessary recovery gate. No real mechanism model is fitted here."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.data import load_graph, masks
from D_mrhgat.train import fit, fit_baseline, seed_all
from D_mrhgat.synthetic import make_flows
from D_mrhgat.benchmarks import evaluate

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def version():
    files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'configs').glob('*')) + [ROOT/'reports/执行口径.md']
    return {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--dependence-weight',type=float,default=.1)
    parser.add_argument('--discrimination-weight',type=float,default=0)
    args = parser.parse_args()
    torch.set_num_threads(3)
    seed_all(0)
    run = ROOT/'results/runs'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_D6_necessary_gate')
    run.mkdir(parents=True)
    (run/'config.yaml').write_bytes((ROOT/'configs/mrhgat.yaml').read_bytes())
    write_json(run/'code_version.json',version())
    graph_file = ROOT/'data/graph/graph_v6.npz'
    (run/'input_graph.npz').write_bytes(graph_file.read_bytes())
    write_json(run/'input_manifest.json',{'graph_sha256':hashlib.sha256(graph_file.read_bytes()).hexdigest(),
        'source_audit':json.loads((ROOT/'reports/A_readiness.json').read_text(encoding='utf-8'))['run_directory']})
    write_json(run/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'device':args.device, 'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        'seeds':[0,1,2,3,4]})
    d = load_graph(device=args.device)
    arrays = {k:v.cpu().numpy() for k,v in d.items()}
    reference, info = fit_baseline(d)
    with torch.no_grad():
        p0 = reference(d['x'],d['K']).exp().cpu().numpy().astype(float)
    np.savez_compressed(run/'B_reference.npz', beta=reference.beta.detach().cpu().numpy(),
        topo=reference.topo.detach().cpu().numpy(), probability=p0)
    params = [{'parameter':f'supply_{i}', 'value':float(v)} for i,v in enumerate(reference.beta.detach().cpu())]
    params += [{'parameter':f'topo_{k}','value':float(v)} for k,v in enumerate(reference.topo.detach().cpu())]
    pd.DataFrame(params).to_csv(ROOT/'results/B_choice_params.csv',index=False)
    baseline_cv = []
    for k in range(5):
        train, evaluation = masks(d,k)
        b, binfo = fit_baseline(d,train,evaluation)
        with torch.no_grad():
            pred = b(d['x'],d['K']).exp()*d['F'].sum(1,keepdim=True)
            actual = d['F'][evaluation]
            cpc = float(2*torch.minimum(pred[evaluation],actual).sum()/(pred[evaluation].sum()+actual.sum()))
        baseline_cv.append({'fold':k,'ce':binfo['ce'],'cpc':cpc,'best_epoch':binfo['best_epoch']})
    pd.DataFrame(baseline_cv).to_csv(ROOT/'results/B_cv.csv',index=False)
    (ROOT/'reports/B_choice.md').write_text('# 无时间、无容量的供给—拓扑基准\n\n'+
        '目的地概率由原九项评价的线性得分和拓扑偏置生成；参数均不表示时间衰减或接诊容量。\n\n'+
        pd.DataFrame(baseline_cv).to_string(index=False)+'\n',encoding='utf-8')
    records, decisions = [], []
    selected = None
    for variant in ('nested','independent','none'):
        group=[]
        for seed in range(5):
            started=time.time()
            write_json(ROOT/'reports/D6_progress.json',{'run_directory':str(run),'variant':variant,'seed':seed,
                'stage':'S2_30_percent_necessary_gate','completed':len(records)})
            print(f'START variant={variant} seed={seed}',flush=True)
            F,truth=make_flows(arrays['x'],arrays['K'],p0,arrays['F'].sum(1),'S2',.3,seed)
            sim={**d,'F':torch.as_tensor(F,device=args.device)}
            train,evaluation=masks(sim,0)
            bcv,_=fit_baseline(sim,train,evaluation)
            _,_,cvinfo=fit(sim,variant,seed,bcv.state_dict(),train,evaluation,
                dependence_weight=args.dependence_weight,discrimination_weight=args.discrimination_weight)
            bfull,_=fit_baseline(sim)
            model,out,fullinfo=fit(sim,variant,seed,bfull.state_dict(),epochs=cvinfo['best_epoch'],
                dependence_weight=args.dependence_weight,discrimination_weight=args.discrimination_weight)
            with torch.no_grad():
                bp=bfull(sim['x'],sim['K']).exp().cpu().numpy()
            metrics=evaluate(F,arrays['K'],out,truth,bp)
            metrics.update(variant=variant,scenario='S2',phi=.3,seed=seed,cv_epoch=cvinfo['best_epoch'],
                cv_ce=cvinfo['ce'],full_ce=fullinfo['ce'],seconds=time.time()-started)
            records.append(metrics)
            group.append(metrics)
            pd.DataFrame(records).to_csv(ROOT/'results/D6_synthetic.csv',index=False)
            case=run/f'{variant}_S2_030_seed{seed}'
            case.mkdir()
            torch.save(model.state_dict(),case/'model.pt')
            np.savez_compressed(case/'synthetic.npz',F=F,expected=truth['expected'],displacers=truth['displacers'])
            write_json(case/'metrics.json',metrics)
            write_json(case/'training.json',{'cv':cvinfo,'full':fullinfo})
            print(json.dumps(metrics,ensure_ascii=False),flush=True)
            del model,out,sim,bcv,bfull
            if args.device=='cuda':
                torch.cuda.empty_cache()
        means={key:float(np.mean([r[key] for r in group])) for key in (
            'comp_edge_auc','comp_residual_auc','comp_node_auc','compet_edge_auc','compet_node_auc')}
        pred=[r['topo_'+k+'_pred'] for r in group for k in ('adj','crs')]
        actual=[r['topo_'+k+'_truth'] for r in group for k in ('adj','crs')]
        means['comp_topology_correlation']=float(pearsonr(pred,actual).statistic)
        checks={'comp_edge':means['comp_edge_auc']>=.85,
            'above_residual':means['comp_edge_auc']>means['comp_residual_auc'],
            'comp_node':means['comp_node_auc']>=.85,
            'compet_edge':means['compet_edge_auc']>=.75,
            'compet_node':means['compet_node_auc']>=.70,
            'topology':means['comp_topology_correlation']>=.7}
        decision={'variant':variant,'means':means,'checks':checks,'necessary_gate_passed':all(checks.values())}
        decisions.append(decision)
        write_json(run/'metrics.json',{'decisions':decisions,'records':records})
        if decision['necessary_gate_passed']:
            selected=variant
            break
    status='necessary_gate_passed_full_validation_pending' if selected else 'necessary_recovery_gate_not_met'
    rows=['# 模拟识别验收','',
        '当前口径：总人流、原九项供给属性、无出行时间、无容量校正。', '',
        '本表为S2、30%机制强度、每配置5份模拟的必要识别门槛；各份均由空间第0折及邻接缓冲选择轮数，再全量拟合模拟人流。', '',
        '| 内部通道 | 补偿边AUC ≥0.85 | 残差基准 | 补偿节点AUC ≥0.85 | 竞争边AUC ≥0.75 | 竞争节点AUC ≥0.70 | 拓扑相关 ≥0.7 | 必要门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for item in decisions:
        m=item['means']
        rows.append(f"| {item['variant']} | {m['comp_edge_auc']:.3f} | {m['comp_residual_auc']:.3f} | {m['comp_node_auc']:.3f} | {m['compet_edge_auc']:.3f} | {m['compet_node_auc']:.3f} | {m['comp_topology_correlation']:.3f} | {'通过' if item['necessary_gate_passed'] else '未达标'} |")
    rows += ['', '补偿边AUC还须高于同份数据拟合的残差基准；所有指标采用5份的均值，未删除不利种子。', '',
        '拓扑相关为5份模拟×ADJ/CRS的10格，不是分层人流检验。', '',
        '## 当前结论','',
        ('必要识别门槛已通过，尚须完成其他强度、情景和误报对照后才能进入真实机制训练。' if selected else
         '三种内部通道均未满足全部必要识别门槛，当前没有可用于真实机制分解的合格配置。按预定验收规则停在模拟阶段。'),'',
        '尚未运行完整情景误报检验、真实MR-HGAT机制训练、机制再分类与依赖性韧性分析；不据本模型输出真实社区的补偿、竞争或挤出结论。', '',
        f'运行记录：{run}', '']
    (ROOT/'reports/D6_model_selection.md').write_text('\n'.join(rows),encoding='utf-8')
    write_json(ROOT/'reports/D6_progress.json',{'status':status,'run_directory':str(run),'completed':len(records),'selected':selected})
    readiness=json.loads((ROOT/'reports/A_readiness.json').read_text(encoding='utf-8'))
    readiness.update(status=status,ready_for_training=False,blockers=[
        '需要完成全部模拟验收' if selected else '模拟必要识别门槛未达标，无合格真实训练配置'])
    write_json(ROOT/'reports/A_readiness.json',readiness)
    print(json.dumps({'status':status,'decisions':decisions,'run':str(run)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
