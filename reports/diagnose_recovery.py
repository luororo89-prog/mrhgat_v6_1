"""Read-only model audit and isolated seed controls; no gate/result replacement."""
from pathlib import Path
import json
import sys
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from D_mrhgat.data import load_graph, shares
from D_mrhgat.model import MRHGAT, cross_entropy
from D_mrhgat.train import fit_baseline, seed_all
from D_mrhgat.synthetic import make_flows
from D_mrhgat.benchmarks import evaluate

RUN = ROOT / 'results/runs/20260928_212256_D6_necessary_gate'
OUT = ROOT / 'reports/D6_cause_diagnosis.json'
torch.set_num_threads(3)
d = load_graph(RUN / 'input_graph.npz', device='cuda')
a = {k: v.cpu().numpy() for k, v in d.items()}
p0 = np.load(RUN / 'B_reference.npz')['probability']
report = {'scope': 'diagnostic only; original inputs, models, gate thresholds and official results unchanged',
          'saved_models': [], 'controls': []}

def save():
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

def summarize(model, sim, truth, bp):
    with torch.no_grad():
        out = model(sim['x'], sim['F'], sim['K'])
        ce = float(cross_entropy(out['logpi'], shares(sim['F']), sim['F'].sum(1)>0))
    F = sim['F'].cpu().numpy()
    r = out['r'].cpu().numpy()
    node = (r * F[:, None, :]).sum(2) / np.maximum(F.sum(1)[:, None], 1)
    valid = F.sum(1)>0
    edge = (F>0) & (a['K']!=0) & (truth['delta']>0)
    frac = truth['expected'] / np.maximum(truth['expected'].sum(1)[:,None,:],1e-30)
    metrics = evaluate(F, a['K'], out, truth, bp)
    metrics.update(ce=ce,
        channel_auc_for_true_comp_edges=[float(roc_auc_score(frac[:,1,:][edge]>.5, r[:,c,:][edge])) for c in range(3)],
        channel_auc_for_low_supply_nodes=[float(roc_auc_score(truth['low'][valid], node[valid,c])) for c in range(3)],
        mean_channel_share_low=node[valid & truth['low']].mean(0).tolist(),
        mean_channel_share_high=node[valid & truth['high']].mean(0).tolist(),
        conservation_max_error=float(np.abs(r.sum(1)-1).max()),
        finite=bool(np.isfinite(r).all()))
    return metrics

for seed in range(5):
    case = RUN / f'none_S2_050_seed{seed}'
    F, truth = make_flows(a['x'], a['K'], p0, a['F'].sum(1), 'S2', .5, seed)
    archived = np.load(case / 'synthetic.npz')
    assert np.array_equal(F, archived['F'])
    assert np.array_equal(truth['expected'], archived['expected'])
    sim = {**d, 'F': torch.as_tensor(F, device='cuda')}
    model = MRHGAT(int_channel='none').cuda()
    model.load_state_dict(torch.load(case/'model.pt', map_location='cuda', weights_only=True))
    metrics = summarize(model, sim, truth, p0)
    old = json.loads((case/'metrics.json').read_text(encoding='utf-8'))
    assert abs(metrics['comp_edge_auc']-old['comp_edge_auc']) < 1e-6
    assert abs(metrics['comp_node_auc']-old['comp_node_auc']) < 1e-6
    metrics.update(data_seed=seed, model_seed=seed, epochs=old['cv_epoch'],
                   original_full_ce=old['full_ce'], data_reproduced_exactly=True)
    report['saved_models'].append(metrics)
    print('SAVED', seed, json.dumps(metrics), flush=True)
    del model
save()

# Cross data seed with model seed, holding training duration fixed.
# Truth is used only for diagnostic scoring and never for selecting a model.
for data_seed in (0, 3):
    F, truth = make_flows(a['x'], a['K'], p0, a['F'].sum(1), 'S2', .5, data_seed)
    sim = {**d, 'F': torch.as_tensor(F, device='cuda')}
    baseline, _ = fit_baseline(sim)
    with torch.no_grad():
        bp = baseline(sim['x'],sim['K']).exp().cpu().numpy()
    for model_seed in (0, 3):
        seed_all(model_seed)
        model = MRHGAT(int_channel='none',base_beta=baseline.beta.detach(),base_topo=baseline.topo.detach()).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=.02)
        target = shares(sim['F'])
        rows = sim['F'].sum(1)>0
        limit = 400 if (data_seed,model_seed)==(3,3) else 108
        for epoch in range(1, limit+1):
            opt.zero_grad(set_to_none=True)
            out = model(sim['x'],sim['F'],sim['K'])
            loss = cross_entropy(out['logpi'],target,rows) + 1e-4*model.regularization()
            loss.backward()
            opt.step()
            if epoch in (40, 80, 108, 200, 400):
                metrics = summarize(model,sim,truth,bp)
                metrics.update(data_seed=data_seed,model_seed=model_seed,epochs=epoch)
                report['controls'].append(metrics)
                save()
                print('CONTROL',data_seed,model_seed,epoch,json.dumps(metrics),flush=True)
        del model, opt, out
        torch.cuda.empty_cache()
save()
