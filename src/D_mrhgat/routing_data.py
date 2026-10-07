"""Inputs, spatial nesting and flow-context isolation for the soft-routing GAT.

Static inputs (nine supply features in TH/SH/CHC groups, topology, impedance,
population, coordinates) are known for the whole city. Every quantity derived
from OD rows is rebuilt from W_context, which only contains the rows of
visible origins (修订方案 Section 8). Hidden entries are stored as 0 and
carried with an explicit observed mask.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from .routing_core import ADJACENT, CROSS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path('/Users/luoxuan/Documents/博后三胎/input')
SUPPLY_FILE = '联通人口_九项医疗供给指标_2018.xlsx'
FLOW_FILE = 'flow_grid_d30_home_tid.csv'
TOPOLOGY_FILE = ROOT / 'data/graph/K_walk15_circle_or_boundary_adjacency.npz'
CENTROID_FILE = ROOT / 'data/processed/communities_candidate.parquet'
TIERS = ('TH', 'SH', 'CHC')
SUPPLY_COLUMNS = tuple(c for t in TIERS for c in (f'RA_{t}', f'SA_{t}_per_10000', f'TC_{t}'))
SUMMARY_COLUMNS = ('log_inflow_adjacent', 'log_inflow_cross', 'coverage_count_adjacent',
                   'coverage_count_cross', 'coverage_population_adjacent',
                   'coverage_population_cross', 'has_inflow_adjacent', 'has_inflow_cross')


def great_circle_km(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    dlon = lon[None, :] - lon[:, None]
    dlat = lat[None, :] - lat[:, None]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def read_od(path, ids):
    """OD matrix: headerless csv in community_id order, or csv with community_id index and header."""
    with open(path, encoding='utf-8-sig') as f:
        first = f.readline().split(',')[0].strip()
    if first.replace('.', '', 1).isdigit():
        return np.loadtxt(path, delimiter=',')
    od = pd.read_csv(path, index_col=0, encoding='utf-8-sig')
    if not ((od.index.to_numpy() == ids).all() and (od.columns.astype(int).to_numpy() == ids).all()):
        raise ValueError('OD rows and columns must follow community_id')
    return od.to_numpy(float)


def read_adjacency(path, column, ids):
    """Directed adjacency from a pair table (source_community_id, target_community_id, column in {0, 1})."""
    pairs = pd.read_csv(path, encoding='utf-8-sig')
    index = {v: k for k, v in enumerate(ids.tolist())}
    K = np.full((len(ids), len(ids)), CROSS, dtype=np.int64)
    hit = pairs[pairs[column] == 1]
    K[hit['source_community_id'].map(index), hit['target_community_id'].map(index)] = ADJACENT
    np.fill_diagonal(K, 0)
    return K


def load_city(input_dir=DEFAULT_INPUT, topology_file=TOPOLOGY_FILE, centroid_file=CENTROID_FILE,
              impedance=None, device='cpu', supply_file=None, flow_file=None, adjacency_file=None,
              adjacency_column=None):
    """Load the city graph.

    impedance: optional [n, n] road impedance matrix aligned to community_id
    order. Without it the great-circle distance between population centroids
    is used for every pair, so no pair-specific coverage can leak OD support.
    """
    input_dir = Path(input_dir)
    supply_path = Path(supply_file) if supply_file else input_dir / SUPPLY_FILE
    flow_path = Path(flow_file) if flow_file else input_dir / FLOW_FILE
    supply = pd.read_excel(supply_path)
    ids = supply['community_id'].to_numpy()
    raw = supply[list(SUPPLY_COLUMNS)].to_numpy(float)
    W = read_od(flow_path, ids)
    n = len(ids)
    if W.shape != (n, n) or not np.isfinite(W).all() or (W < 0).any():
        raise ValueError('Flow matrix must be a finite nonnegative square matrix aligned to community_id')
    K = read_adjacency(adjacency_file, adjacency_column, ids) if adjacency_file else np.load(topology_file)['K']
    if K.shape != (n, n) or (np.diag(K) != 0).any() or not np.isin(K[~np.eye(n, dtype=bool)], (ADJACENT, CROSS)).all():
        raise ValueError('Topology must be 0 on the diagonal and adjacent/cross elsewhere')
    centroids = pd.read_parquet(centroid_file, columns=['community_id', 'cx', 'cy', '联通人']).set_index('community_id').loc[ids]
    lon, lat = centroids['cx'].to_numpy(float), centroids['cy'].to_numpy(float)
    distance = great_circle_km(lon, lat)
    impedance_source = 'centroid_great_circle_km'
    if impedance is not None:
        distance = np.asarray(impedance, dtype=float)
        impedance_source = 'user_supplied'
        if distance.shape != (n, n) or not np.isfinite(distance).all():
            raise ValueError('Impedance must be a finite square matrix aligned to community_id')
    std = raw.std(0)
    if (std == 0).any():
        raise ValueError('Constant supply column')
    x = (raw - raw.mean(0)) / std
    log_d = np.log1p(distance)
    off = ~np.eye(n, dtype=bool)
    imp = (log_d - log_d[off].mean()) / log_d[off].std()
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
    return {
        'ids': ids, 'x': tensor(x), 'x_raw': raw, 'W': tensor(W), 'K': tensor(K, torch.long),
        'impedance': tensor(imp), 'distance_km': tensor(distance), 'population': tensor(centroids['联通人'].to_numpy(float)),
        'lon': lon, 'lat': lat,
        # A community without any recorded outflow (internal flow included) is
        # treated as an unobserved origin, never as an origin with true zeros.
        'origin_observed': tensor(W.sum(1) > 0, torch.bool),
        'names': supply['community_name'].astype(str).to_numpy(),
        'meta': {'impedance_source': impedance_source, 'supply_columns': list(SUPPLY_COLUMNS),
                 'supply_file': supply_path.name, 'flow_file': flow_path.name,
                 'adjacency': f'{Path(adjacency_file).name}:{adjacency_column}' if adjacency_file else Path(topology_file).name,
                 'supply_scaling': 'citywide z-score of static supply (no OD input)',
                 'disclosure_threshold': None,
                 'zero_flow_status': 'observed zero inside a visible origin row; rows with zero total are unobserved origins'},
    }


def spatial_blocks(lon, lat, n_blocks=25, seed=0):
    """Contiguous spatial blocks from coordinates only (no OD labels or model outputs)."""
    xy = np.column_stack([(lon - lon.mean()) * 111.32 * np.cos(np.radians(lat.mean())), (lat - lat.mean()) * 110.57])
    return KMeans(n_clusters=n_blocks, n_init=20, random_state=seed).fit_predict(xy)


def balanced_groups(units, sizes, n_groups):
    """Longest-processing-time assignment of units to groups by node count."""
    order = sorted(units, key=lambda u: (-sizes[u], u))
    groups, load = [[] for _ in range(n_groups)], np.zeros(n_groups)
    for u in order:
        g = int(np.argmin(load))
        groups[g].append(u)
        load[g] += sizes[u]
    return [sorted(g) for g in groups]


def outer_folds(block, n_folds=5):
    sizes = np.bincount(block)
    lookup = np.empty(len(sizes), dtype=int)
    for k, blocks in enumerate(balanced_groups(range(len(sizes)), sizes, n_folds)):
        lookup[blocks] = k
    return lookup[block]


def nested_split(block, fold, outer, n_inner=5, query_groups=4, inner_validation=0):
    """Outer test fold, inner validation blocks and support/query groups.

    All roles are unions of spatial blocks of the outer training area.
    """
    sizes = np.bincount(block)
    train_blocks = sorted(set(block[fold != outer].tolist()))
    inner = balanced_groups(train_blocks, sizes, n_inner)
    valid_blocks = inner[inner_validation]
    fit_blocks = [b for b in train_blocks if b not in valid_blocks]
    groups = balanced_groups(fit_blocks, sizes, query_groups)
    member = lambda blocks: torch.as_tensor(np.isin(block, blocks))
    return {'outer': outer, 'test': torch.as_tensor(fold == outer), 'train': torch.as_tensor(fold != outer),
            'valid': member(valid_blocks), 'fit': member(fit_blocks), 'groups': [member(g) for g in groups]}


def build_context(city, visible):
    """Flow context from the rows of visible origins only.

    Entry (v, i) of W_context is known iff origin v is visible. Inflow to i is
    aggregated separately for adjacent and cross sources; coverage uses only
    which origins are visible plus their static population.
    """
    W, K, pop = city['W'], city['K'], city['population']
    n = W.shape[0]
    visible = visible & city['origin_observed']
    W_ctx = W * visible[:, None]
    observed = visible[:, None].expand(n, n)
    summary, positive, source_supply = [], [], []
    volumes, coverage_count, coverage_pop, has = [], [], [], []
    for r in (ADJACENT, CROSS):
        relation = K == r
        known = relation & observed
        volume = (W_ctx * known).sum(0)
        volumes.append(torch.log1p(volume))
        coverage_count.append(known.sum(0) / relation.sum(0).clamp_min(1))
        coverage_pop.append((pop[:, None] * known).sum(0) / (pop[:, None] * relation).sum(0).clamp_min(1e-9))
        pos = known & (W_ctx > 0)
        has.append(pos.any(0).float())
        positive.append(pos)
        weight = W_ctx * known
        source_supply.append((weight.T @ city['x']) / volume.clamp_min(1e-9)[:, None])
    summary = torch.stack(volumes + coverage_count + coverage_pop + has, 1)
    return {'visible': visible, 'W': W_ctx, 'observed': observed, 'log_flow': torch.log1p(W_ctx),
            'positive': torch.stack(positive), 'summary': summary,
            'source_supply': torch.cat(source_supply, 1)}


def grouped_context_schedule(split):
    """(support, query) origin masks; each fit group is the query once per epoch."""
    for g, query in enumerate(split['groups']):
        support = split['fit'] & ~query
        yield g, support, query


def shares(W):
    return W / W.sum(1, keepdim=True).clamp_min(1e-12)


def loss_rows(city, origins):
    return origins & city['origin_observed']


def isolation_record(city, split, contexts, prototype_sources):
    ids = city['ids']
    pick = lambda m: ids[m.cpu().numpy()].tolist()
    observed = city['origin_observed']
    pop = city['population']
    record = {'outer_fold': int(split['outer']), 'test_origins': pick(split['test']),
              'validation_origins': pick(split['valid']), 'fit_origins': pick(split['fit']),
              'query_groups': [pick(g) for g in split['groups']],
              'origins_without_outflow_unobserved': pick(~observed),
              'preprocessing_scope': city['meta']['supply_scaling'] + '; impedance standardized over all pairs (static)',
              'impedance_source': city['meta']['impedance_source'], 'contexts': {}, 'prototype_sources': prototype_sources}
    for name, ctx in contexts.items():
        vis = ctx['visible']
        record['contexts'][name] = {'visible_origins': int(vis.sum()), 'observed_origins': int(observed.sum()),
                                    'visible_origin_fraction': float(vis.sum() / observed.sum()),
                                    'visible_population_fraction': float(pop[vis].sum() / pop[observed].sum()),
                                    'test_rows_visible': int((vis & split['test']).sum())}
    return record
