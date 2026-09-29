"""Build the user-approved aggregate graph without travel time or capacity."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]

def spatial_folds(xy, F, x):
    block = KMeans(n_clusters=25, n_init=20, random_state=0).fit_predict(xy)
    S = x.reshape(len(x), 3, 3).mean(2)
    strata = np.column_stack([np.searchsorted(np.quantile(S[:, m], [1/3, 2/3]), S[:, m], side='right') for m in range(3)])
    composition = np.column_stack([np.ones(len(x)), F.sum(1)] + [(strata[:, m] == q).astype(float) for m in range(3) for q in range(3)])
    counts = np.stack([composition[block == b].sum(0) for b in range(25)])
    target = counts.sum(0) / 5
    rng = np.random.default_rng(0)
    best, best_score = None, float('inf')
    for _ in range(2000):
        assignment = rng.permutation(25).reshape(5, 5)
        actual = counts[assignment].sum(1)
        node_ok = np.all(np.abs(actual[:, 0] / target[0] - 1) <= .25)
        present = target[2:] > 0
        proportions = actual[:, 2:][:, present] / actual[:, :1]
        ratios = proportions / (target[2:][present] / target[0])
        strata_ok = np.all((ratios >= .5) & (ratios <= 2))
        nonempty = target > 0
        score = np.max(np.abs(actual[:, nonempty] / target[nonempty] - 1))
        if node_ok and strata_ok and score < best_score:
            best, best_score = assignment.copy(), score
    if best is None:
        raise ValueError('No spatial partition meets the prespecified node balance')
    lookup = np.empty(25, dtype=int)
    for k, blocks in enumerate(best):
        lookup[blocks] = k
    fold = lookup[block]
    for k in range(5):
        for m in range(3):
            for q in range(3):
                if not (strata[:, m] == q).any():
                    continue  # Identical supply values can leave a quantile empty.
                ratio = (strata[fold == k, m] == q).mean() / (strata[:, m] == q).mean()
                if not .5 <= ratio <= 2:
                    raise ValueError('Supply strata spatial balance outside [0.5, 2]')
    return block, fold, best_score

def main():
    p = ROOT / 'data/processed'
    g = ROOT / 'data/graph'
    g.mkdir(exist_ok=True, parents=True)
    supply = pd.read_parquet(p / 'legacy_supply_zscore.parquet')
    ids = supply.OBJECTID.to_numpy()
    x = supply.drop(columns='OBJECTID').to_numpy(float)
    mapping = {v: i for i, v in enumerate(ids)}
    F = np.zeros((len(ids), len(ids)), dtype=np.float64)
    od = pd.read_parquet(p / 'flows_community_aggregate.parquet')
    np.add.at(F, (od.source.map(mapping).to_numpy(), od.target.map(mapping).to_numpy()), od.flow_count.to_numpy())
    assert np.isfinite(x).all() and np.isfinite(F).all() and (F >= 0).all()
    adj = pd.read_parquet(p / 'adjacency.parquet')
    K = np.full(F.shape, 2, dtype=np.int64)
    K[adj.community_id.map(mapping), adj.neighbor_id.map(mapping)] = 1
    np.fill_diagonal(K, 0)
    coords = pd.read_parquet(p / 'communities_candidate.parquet').set_index('community_id').loc[ids, ['cx', 'cy']].to_numpy()
    block, fold, balance = spatial_folds(coords, F, x)
    np.savez_compressed(g / 'graph_v6.npz', x=x.astype(np.float32), F=F.astype(np.float32), K=K, ids=ids, fold=fold)
    pd.DataFrame({'community_id': ids, 'block_id': block, 'fold': fold}).to_parquet(p / 'spatial_folds.parquet', index=False)
    S = x.mean(1)
    low_in = (F * (S[:, None] < S[None, :]) * (K != 0)).sum(0)
    i, j = np.nonzero(F)
    pd.DataFrame({'src': ids[i], 'dst': ids[j], 'topo': K[i,j], 'F': F[i,j],
        'delta': S[j]-S[i], 'low_supply_inflow_src': low_in[i]}).to_parquet(g / 'edge_posthoc.parquet', index=False)
    totals = np.stack([(F * (K == k)).sum(1) for k in range(3)], axis=1)
    valid = F.sum(1) > 0
    dominant = np.argmax(totals[valid], axis=1)
    cv = []
    for k in range(5):
        test = fold == k
        buffer = (K[test] == 1).any(0) & ~test
        train = ~test & ~buffer & valid
        assert train.any() and (test & valid).any()
        cv.append({'fold': k, 'nodes': int(test.sum()), 'origins': int((test & valid).sum()),
            'buffer': int(buffer.sum()), 'training_origins': int(train.sum()), 'outflow': float(F[test].sum())})
    report = {'scope': 'aggregate_no_capacity_no_time', 'nodes': len(ids), 'flow_total': float(F.sum()),
        'nonzero_edges': len(i), 'adjacency_directed_edges': int((K == 1).sum()), 'valid_origins': int(valid.sum()),
        'dominant_topology_counts': {name: int((dominant == k).sum()) for k, name in enumerate(['INT','ADJ','CRS'])},
        'cross_circle_dominant_fraction_valid_origins': float((dominant == 2).mean()),
        'spatial_balance_max_relative_deviation': balance, 'folds': cv,
        'capacity_correction': False, 'travel_time': False, 'tier_flow_allocation': False}
    (ROOT / 'reports/C_graph.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'reports/C_graph.md').write_text('# 聚合图核验\n\n'+
        f'节点{len(ids)}；总流量{F.sum():,.0f}；有效起点{valid.sum()}；非零边{len(i)}。\n\n'+
        f'跨圈主导占有效起点{(dominant==2).mean():.2%}；原稿88.9%属于另一流量口径，本次不要求强行匹配。\n\n'+
        '全部社区为候选目的地；TH/SH/CHC仅为供给属性。图中无时间、容量及事后检验量。\n\n'+
        '空间分折：\n\n'+pd.DataFrame(cv).to_string(index=False)+'\n', encoding='utf-8')
    status_path = ROOT / 'reports/A_readiness.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status.update(status='aggregate_inputs_ready', ready_for_training=False,
        ready_for_synthetic_validation=True, blockers=['真实训练需先通过模拟识别门槛'],
        user_scope={'capacity_correction': False, 'travel_time': False, 'tier_flow_allocation': False},
        current_specification='reports/执行口径.md')
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    run = ROOT / 'results/runs' / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_C_graph')
    run.mkdir(parents=True)
    (run/'config.yaml').write_bytes((ROOT/'configs/mrhgat.yaml').read_bytes())
    (run/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    files = [Path(__file__), ROOT/'configs/mrhgat.yaml', ROOT/'reports/执行口径.md']
    (run/'code_version.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
