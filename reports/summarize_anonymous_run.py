"""Summarize actual completed fits, retaining every initialization."""
from pathlib import Path
import hashlib
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
ref=ROOT/'results/runs/20260928_212256_D6_necessary_gate'
graph=np.load(ref/'input_graph.npz')
p0=np.load(ref/'B_reference.npz')['probability']
protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
rows=[]
agreements=[]
for case in sorted(run.glob('*_S2_*_data*')):
    if not (case/'metrics.json').exists():
        continue
    records=json.loads((case/'metrics.json').read_text(encoding='utf-8'))
    meta=records[0]
    saved=np.load(case/'posteriors.npz')
    F,truth=make_flows(graph['x'],graph['K'],p0,graph['F'].sum(1),'S2',meta['phi'],meta['data_seed'])
    np.testing.assert_array_equal(F,saved['F'])
    expected=truth['expected']
    fractions=expected/np.maximum(expected.sum(1)[:,None,:],1e-30)
    for r,record in zip(saved['raw'],[r for r in records if r['kind']=='raw']):
        orders=([0,1,2],[0,2,1])
        errors=[float((np.abs(r[:,order,:]-fractions)*F[:,None,:]).sum()/F.sum()) for order in orders]
        idx=int(np.argmin(errors))
        metrics=evaluate(F,graph['K'],{'r':torch.as_tensor(r[:,orders[idx],:])},truth,p0)
        row={k:record[k] for k in ('phase','profile','phi','data_seed','seed')}
        row.update(case=case.name,synthetic_oracle_order=str(orders[idx]),oracle_error=errors[idx],
                   truth_used_for_evaluation_only=True)
        for name in ('comp_edge_auc','comp_node_auc','compet_edge_auc','compet_node_auc'):
            row['oracle_'+name]=metrics[name]
        rows.append(row)
    agreement=json.loads((case/'agreement.json').read_text(encoding='utf-8'))
    agreements.append(dict(case=case.name,phase=meta['phase'],profile=meta['profile'],
                           phi=meta['phi'],data_seed=meta['data_seed'],**agreement))
df=pd.DataFrame(rows)
df.to_csv(run/'all_synthetic_permutation_diagnostics.csv',index=False)
pd.DataFrame(agreements).to_json(run/'all_restart_agreements.json',orient='records',indent=2)
current_hash=hashlib.sha256((ref/'input_graph.npz').read_bytes()).hexdigest()
assert current_hash==protocol['original_graph_sha256']
selected=df[df.profile==selection['selected_profile']]
cv=json.loads((run/'development_cv.json').read_text(encoding='utf-8'))
status={'training_semantics':'anonymous','mechanism_names_confirmed':False,
        'null_tests_completed':False,'ready_for_real_mechanism_interpretation':False,
        'selected_profile':selection['selected_profile'],
        'completed_cv_fits':sum(len(v)*5 for v in cv.values()),
        'completed_full_fits':len(df),'completed_audit_fits':int((df.phase=='audit').sum()),
        'completed_repeat_fits':int((df.phase=='audit_repeat').sum()),
        'original_graph_sha256':current_hash,'input_unchanged':True,
        'scope':'none configuration, S2 synthetic diagnostic; not complete original D6 gates',
        'permutation_matching':'synthetic evaluation only; not empirical mechanism naming'}
restart=run/'restart_validation'
restart_complete=(restart/'status.json').exists()
status['completed_restart_fits']=25 if restart_complete else 0
status['completed_total_fits']=status['completed_cv_fits']+status['completed_full_fits']+status['completed_restart_fits']
(run/'current_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
text=['# 匿名通道实验结果','',
      '训练采用匿名通道，机制名称由训练后检验确定。输入九项设施评价、总人流与拓扑保持不变。',
      '', '## 当前证据', '',
      f"已完成{status['completed_cv_fits']}次空间交叉验证拟合及{status['completed_full_fits']}次全量拟合。",
      f"配置依据五折留出误差选择，当前为{selection['selected_profile']}。选择未读取机制真值。",'',
      '| 配置 | 五折平均交叉熵 |','|---|---:|']
for profile,value in selection['mean_fivefold_ce'].items():
    text.append(f'| {profile} | {value:.6f} |')
text+=['','## 当前配置的模拟分解诊断','',
       '下表使用模拟真值匹配因素编号，仅用于评价分解的可恢复性。每次初始化均保留。该指标不构成真实数据的机制命名，也不替代原方案全部验收。','',
       '| 数据组 | 拟合次数 | 补偿边AUC最小值 | 补偿节点AUC最小值 | 竞争边AUC最小值 | 竞争节点AUC最小值 |',
       '|---|---:|---:|---:|---:|---:|']
for phase,g in selected.groupby('phase',sort=False):
    cols=['oracle_'+k for k in ('comp_edge_auc','comp_node_auc','compet_edge_auc','compet_node_auc')]
    text.append('| '+phase+' | '+str(len(g))+' | '+' | '.join(f'{g[k].min():.4f}' for k in cols)+' |')
text+=['','## 机制判定状态','',
       '逐通道方向筛查已经输出。NC和ND对照显著性检验尚未完成，目前机制名称均为待判定。',
       '命名程序逐通道检查补偿和竞争两类假设。通过一类时命名为该机制，两类均通过时记为混合待判定。八项候选检验统一做Holm校正。',
       '当前结果不能支持机制识别已稳定的结论，也不能支持真实竞争挤出的因果解释。匿名通道处理可消除编号本身的影响，但成分混合仍需满足分解恢复和对照检验要求。',
       '', '## 文件', '',
       '* `anonymous_hypothesis_screen.csv` 保存每个匿名通道的方向诊断和待判定状态。',
       '* `all_synthetic_permutation_diagnostics.csv` 保存全部初始化的模拟分解评价。',
       '* `all_restart_agreements.json` 保存不使用真值的跨初始化匹配一致性。',
       '* `selection.json` 和 `development_cv.json` 保存选模证据及逐折记录。',
       '* `current_status.json` 保存完成数量、范围与输入校验。',
       '', '模型和判定相关17项代码检查通过。原始输入图SHA256与预登记值一致。',
       '此验证不包含数据准备模块，当前训练环境未安装该模块所需的geopandas。']
if restart_complete:
    chosen=pd.read_csv(restart/'selected_starts.csv')
    agreement=json.loads((restart/'selected_agreement.json').read_text(encoding='utf-8'))
    for group in chosen.group:
        case=restart/f'group{group}'
        choice=json.loads((case/'selection.json').read_text(encoding='utf-8'))
        checkpoint=torch.load(case/'models.pt',map_location='cpu',weights_only=True)
        torch.save({key:value[choice['index']].clone() for key,value in checkpoint.items()},case/'selected_model.pt')
    text+=['','## 五组多起点优化的独立验证','',
           '独立数据种子15，每组五个初始化，共25次拟合。每组按交叉熵最小值选择一个模型，选取完成后再使用模拟真值评价。输入及模型机制指标不参与选取。',
           '', '| 组别 | 所选初始化 | 补偿边AUC | 补偿节点AUC | 竞争边AUC | 竞争节点AUC |',
           '|---|---:|---:|---:|---:|---:|']
    for row in chosen.to_dict('records'):
        text.append('| '+str(row['group'])+' | '+str(row['seed'])+' | '+
                    ' | '.join(f"{row[k]:.4f}" for k in ('comp_edge_auc','comp_node_auc','compet_edge_auc','compet_node_auc'))+' |')
    text+=['',f"所选模型之间匿名匹配后的平均余弦相似度为{agreement['mean_pairwise_matched_cosine']:.4f}，逐边最大后验标签平均不一致率为{agreement['mean_pairwise_edge_label_disagreement']:.2%}。",
           '该结果仅覆盖一份独立S2模拟数据及五组优化重复。完整情景、拓扑恢复和NC与ND对照检验仍需通过，当前不赋予真实数据通道机制名称。',
           '每组模型保存为 `restart_validation/group编号/selected_model.pt`，所有起点模型及其选取依据同时保留。',
           '',f"本次累计完成{status['completed_total_fits']}次MR-HGAT拟合，不含基础模型拟合。"]
(ROOT/'reports/anonymous_channel_results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(status,ensure_ascii=False))
