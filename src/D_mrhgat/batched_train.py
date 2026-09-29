"""Independent per-seed refits (previously vmap-vectorized across seeds).

Vectorized (vmap) batching was found to diverge substantially from true
independent per-seed training at realistic epoch counts on the real graph
(see reports/D7_batching_divergence.md) and has been removed. Each seed now
runs its own ordinary, non-batched training loop; n_workers>1 runs seeds in
separate OS processes (fully independent, no shared computation graph) for
wall-clock speed, never by re-introducing vectorized batching.
"""
import multiprocessing as mp
import torch
from .model import MRHGAT, ChoiceBaseline
from .train import seed_all

def row_losses(logpi, F):
    totals = F.sum(-1)
    valid = totals > 0
    share = F / totals[..., None].clamp_min(1)
    ce = -(share * logpi).sum(-1)
    return (ce * valid).sum(-1) / valid.sum(-1).clamp_min(1)

def _run_jobs(fn, jobs, n_workers):
    if n_workers <= 1 or len(jobs) <= 1:
        return [fn(*job) for job in jobs]
    with mp.get_context('spawn').Pool(processes=min(n_workers, len(jobs))) as pool:
        return pool.starmap(fn, jobs)

def _fit_one_baseline(x, K, F, epochs):
    model = ChoiceBaseline().to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    best = float('inf')
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        row_losses(model(x, K), F).backward()
        opt.step()
        with torch.no_grad():
            score = float(row_losses(model(x, K), F))
        if score < best - 1e-7:
            best = score
            state = {n: p.detach().clone() for n, p in model.named_parameters()}
    return state

def baseline_batch(x, K, flows, epochs=400, n_workers=1):
    jobs = [(x, K, flows[i], epochs) for i in range(len(flows))]
    results = _run_jobs(_fit_one_baseline, jobs, n_workers)
    return {name: torch.stack([r[name] for r in results]) for name in results[0]}

def _fit_one(x, K, F, variant, seed, base_beta, base_topo, epochs):
    seed_all(seed)
    model = MRHGAT(int_channel=variant, base_beta=base_beta, base_topo=base_topo).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = model(x, F, K)
        loss = row_losses(out['logpi'], F) + 1e-4 * model.regularization()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model(x, F, K)
        if not torch.isfinite(out['logpi']).all():
            raise FloatingPointError('Nonfinite batched null fit')
        detached = {k: v.detach() for k, v in out.items()}
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    return detached, params

def fit_batch(x, K, flows, variant, seeds, epochs, base_states=None, n_workers=1):
    if base_states is None:
        base_states = baseline_batch(x, K, flows, n_workers=n_workers)
    jobs = [(x, K, flows[i], variant, seed, base_states['beta'][i], base_states['topo'][i], epochs)
            for i, seed in enumerate(seeds)]
    results = _run_jobs(_fit_one, jobs, n_workers)
    outs = [r[0] for r in results]
    params = [r[1] for r in results]
    detached = {k: torch.stack([o[k] for o in outs]) for k in outs[0]}
    stacked = {n: torch.stack([p[n] for p in params]) for n in params[0]}
    return detached, stacked
