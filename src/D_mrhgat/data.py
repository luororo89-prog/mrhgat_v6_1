from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {'x', 'F', 'K', 'ids', 'fold'}

def validate_keys(keys):
    if set(keys) != ALLOWED:
        raise ValueError(f'Graph must contain only {sorted(ALLOWED)}; got {sorted(keys)}')

def load_graph(path=None, device='cpu'):
    with np.load(path or ROOT/'data/graph/graph_v6.npz') as archive:
        validate_keys(archive.files)
        data = {k: torch.as_tensor(archive[k], device=device) for k in archive.files}
    data['x'] = data['x'].float()
    data['F'] = data['F'].float()
    data['K'] = data['K'].long()
    return data

def masks(data, k):
    evaluation = data['fold'] == k
    buffer = (data['K'][evaluation] == 1).any(0) & ~evaluation
    valid = data['F'].sum(1) > 0
    return ~evaluation & ~buffer & valid, evaluation & valid

def shares(F):
    return F / F.sum(1, keepdim=True).clamp_min(1)
