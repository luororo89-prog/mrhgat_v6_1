"""OD graph for line B (线B §2.3): flow threshold, candidates, topology, impedance, inflow context, labels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from D_mrhgat.routing_data import load_city, SUPPLY_COLUMNS  # noqa: E402

INT, ADJ, CRS = 0, 1, 2
MECHANISMS = ('base', 'comp', 'compet')


def flow_threshold(W, keep=0.95):
    """Largest integer threshold whose retained edges (count >= threshold) still carry keep of all flow."""
    total = W.sum()
    values = np.sort(np.unique(W[W > 0]))
    best = 1
    for v in values:
        if W[W >= v].sum() >= keep * total:
            best = int(v)
        else:
            break
    return best


def build_graph(cfg, threshold=None):
    city = load_city(supply_file=cfg['supply_file'], flow_file=cfg['flow_file'],
                     adjacency_file=cfg['adjacency_file'], adjacency_column=cfg['adjacency_column'])
    W = city['W'].numpy().astype(float)
    theta = threshold if threshold is not None else flow_threshold(W, cfg.get('flow_keep', 0.95))
    Wt = W * (W >= theta)
    K = city['K'].numpy()
    n = len(W)
    received = Wt.sum(0) > 0
    candidates = np.broadcast_to(received[None, :], (n, n)).copy()
    D = city['distance_km'].numpy()
    off = ~np.eye(n, dtype=bool)
    t = (D - D[off].mean()) / D[off].std()
    O = Wt.sum(1)
    inflow = []
    for r in (ADJ, CRS):
        vol = (Wt * (K == r) * off).sum(0)
        inflow.append(np.log1p(np.divide(vol, O, out=np.zeros(n), where=O > 0)))
    u = np.column_stack(inflow)
    u = (u - u.mean(0)) / u.std(0)
    return {'ids': city['ids'], 'names': city['names'], 'raw': city['x_raw'], 'W_raw': W, 'W': Wt, 'theta': theta,
            'retained_share': float(Wt.sum() / W.sum()), 'K': K, 'distance_km': D, 't': t, 'u': u,
            'candidates': candidates, 'origins': O > 0, 'lon': city['lon'], 'lat': city['lat'],
            'columns': list(SUPPLY_COLUMNS), 'meta': city['meta']}


def load_labels(path, g):
    """Labelled edges (dominant anchors from the neighbour-centre review); edges below the threshold carry no flow."""
    a = pd.read_csv(path, encoding='utf-8-sig')
    a = a[a['anchor_kind'] == 'dominant']
    index = {v: k for k, v in enumerate(g['ids'].tolist())}
    o = a['origin_id'].map(index).to_numpy()
    d = a['destination_id'].map(index).to_numpy()
    k = a['mechanism'].map({m: c for c, m in enumerate(MECHANISMS)}).to_numpy()
    f = g['W'][o, d]
    keep = (f > 0) & (o != d)
    return {'origin': o[keep], 'dest': d[keep], 'mech': k[keep], 'flow': f[keep],
            'confidence': a['confidence'].to_numpy()[keep], 'dropped_below_threshold': int((~keep).sum())}
