import copy
import random
import numpy as np
import torch
from .data import shares
from .model import MRHGAT, SeparatedMRHGAT, ChoiceBaseline, cross_entropy

def hsic(embeddings, rows):
    """Biased linear HSIC over train nodes, normalized by covariance norms.

Returns centered kernel alignment for scale-invariant regularization.
This penalizes second-order dependence, not all statistical dependence.
"""
    weights = rows.to(embeddings.dtype)
    n = weights.sum().clamp_min(1)
    centered = embeddings - (embeddings * weights[:, None, None]).sum(0, keepdim=True) / n
    left, right = centered[:, 0] * weights[:, None], centered[:, 1] * weights[:, None]
    cross = left.T @ right / n
    xx, yy = left.T @ left / n, right.T @ right / n
    return cross.square().sum() / (xx.square().sum() * yy.square().sum()).sqrt().clamp_min(1e-8)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def fit_baseline(data, train_rows=None, eval_rows=None, epochs=400, patience=20):
    model = ChoiceBaseline().to(data['x'].device)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    share = shares(data['F'])
    rows = data['F'].sum(1) > 0 if train_rows is None else train_rows
    best, best_epoch, stale, state = float('inf'), 0, 0, None
    for epoch in range(1, epochs+1):
        opt.zero_grad()
        loss = cross_entropy(model(data['x'], data['K']), share, rows)
        loss.backward()
        opt.step()
        with torch.no_grad():
            score = float(cross_entropy(model(data['x'], data['K']), share, rows if eval_rows is None else eval_rows))
        if score < best - 1e-7:
            best, best_epoch, stale, state = score, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if eval_rows is not None and stale >= patience:
            break
    model.load_state_dict(state)
    return model, {'best_epoch': best_epoch, 'ce': best}

def fit(data, variant, seed, base_state, train_rows=None, eval_rows=None, epochs=400, patience=20,
        dependence_weight=0, discrimination_weight=0, model_kwargs=None):
    seed_all(seed)
    cls = SeparatedMRHGAT if discrimination_weight else MRHGAT
    model = cls(int_channel=variant, base_beta=base_state['beta'], base_topo=base_state['topo'],
                **(model_kwargs or {})).to(data['x'].device)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    share = shares(data['F'])
    rows = data['F'].sum(1) > 0 if train_rows is None else train_rows
    def predict():
        return model(data['x'], data['F'], data['K'], rows) if discrimination_weight else model(data['x'], data['F'], data['K'])
    best, best_epoch, stale, state = float('inf'), 0, 0, None
    trace = []
    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = predict()
        loss = cross_entropy(out['logpi'], share, rows) + 1e-4 * model.regularization()
        if dependence_weight:
            loss = loss + dependence_weight * hsic(out['factor_embeddings'], rows)
        if discrimination_weight:
            loss = loss + discrimination_weight * out['factor_discrimination']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        loss.backward()
        opt.step()
        if eval_rows is not None:
            model.eval()
            with torch.no_grad():
                score = float(cross_entropy(predict()['logpi'], share, eval_rows))
            if score < best - 1e-7:
                best, best_epoch, stale, state = score, epoch, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
        trace.append({'epoch': epoch, 'loss': float(loss.detach())})
    if eval_rows is not None:
        model.load_state_dict(state)
    else:
        best_epoch = epochs
    model.eval()
    with torch.no_grad():
        out = predict()
        ce = float(cross_entropy(out['logpi'], share, rows if eval_rows is None else eval_rows))
    return model, out, {'best_epoch': best_epoch, 'ce': ce, 'epochs_run': epoch, 'trace': trace}
