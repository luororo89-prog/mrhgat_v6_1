import copy
import random
import numpy as np
import torch
from .data import shares
from .model import MRHGAT, ChoiceBaseline, cross_entropy

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

def fit(data, variant, seed, base_state, train_rows=None, eval_rows=None, epochs=400, patience=20):
    seed_all(seed)
    model = MRHGAT(int_channel=variant, base_beta=base_state['beta'], base_topo=base_state['topo']).to(data['x'].device)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    share = shares(data['F'])
    rows = data['F'].sum(1) > 0 if train_rows is None else train_rows
    best, best_epoch, stale, state = float('inf'), 0, 0, None
    trace = []
    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(data['x'], data['F'], data['K'])
        loss = cross_entropy(out['logpi'], share, rows) + 1e-4 * model.regularization()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        loss.backward()
        opt.step()
        if eval_rows is not None:
            model.eval()
            with torch.no_grad():
                score = float(cross_entropy(model(data['x'], data['F'], data['K'])['logpi'], share, eval_rows))
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
        out = model(data['x'], data['F'], data['K'])
        ce = float(cross_entropy(out['logpi'], share, rows if eval_rows is None else eval_rows))
    return model, out, {'best_epoch': best_epoch, 'ce': ce, 'epochs_run': epoch, 'trace': trace}
