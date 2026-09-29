"""Literature-informed factor stabilization. No mechanism truth in fitting/matching.

Linear HSIC: Liu et al. Eq. 8, with nodes as sampling units (explicit adaptation).
Factor matching: Yang et al. Hungarian matching, adapted to model-to-model
diagnostics only, never model-to-ground-truth matching or mechanism naming.

Group fits (fit_group/fit_baselines) previously ran all seeds in one seed
group as a single vmap-vectorized computation for speed. That was found to
diverge substantially from true independent per-seed training at realistic
epoch counts on the real graph (see reports/D7_batching_divergence.md) and
has been removed: each seed now runs its own ordinary training loop, with an
optional n_workers>1 to run seeds in separate OS processes for wall-clock
speed — never by re-introducing vectorized cross-seed batching.
"""
import multiprocessing as mp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn
from scipy.optimize import linear_sum_assignment
from .model import MRHGAT, ChoiceBaseline
from .train import seed_all


class SeparatedMRHGAT(MRHGAT):
    """MD forward model plus FactorGCN-style auxiliary factor-graph classifier.

    The shared classifier predicts anonymous factor IDs, never mechanism labels.
    Original gates and destination probabilities are inherited unchanged.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor_layers=nn.ModuleList([nn.Linear(9,16),nn.Linear(16,8),nn.Linear(8,16)])
        self.factor_classifier=nn.Linear(32,2)

    def forward(self, x, F, K, loss_rows=None):
        out=super().forward(x,F,K)
        rows=(F.sum(1)>0) if loss_rows is None else loss_rows
        edges=((F>0)&(K!=0)&rows[:,None]).to(x.dtype)
        # Graph degree normalization, not facility capacity correction.
        norm=(edges.sum(1)[:,None]*edges.sum(0)[None,:]).sqrt().clamp_min(1)
        reps=[]
        for channel in (1,2):
            adjacency=edges*out['r'][:,channel,:]/norm
            outgoing=incoming=x
            for layer in self.factor_layers:
                outgoing=torch.tanh(adjacency@layer(outgoing))
                incoming=torch.tanh(adjacency.T@layer(incoming))
            reps.append(torch.cat([outgoing.mean(0),incoming.mean(0)]))
        logits=self.factor_classifier(torch.stack(reps))
        labels=torch.arange(2,device=x.device)
        out['factor_discrimination']=fn.cross_entropy(logits,labels)
        return out


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


def masked_ce(logpi, flows, rows):
    share = flows / flows.sum(-1, keepdim=True).clamp_min(1)
    ce = -(share * logpi).sum(-1)
    return (ce * rows).sum(-1) / rows.sum(-1).clamp_min(1)


def _run_jobs(fn_, jobs, n_workers):
    if n_workers <= 1 or len(jobs) <= 1:
        return [fn_(*job) for job in jobs]
    with mp.get_context('spawn').Pool(processes=min(n_workers, len(jobs))) as pool:
        return pool.starmap(fn_, jobs)


def _fit_one_baseline(x, K, F, train_rows, eval_rows, epochs):
    model = ChoiceBaseline().to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    best = float('inf')
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        masked_ce(model(x, K), F, train_rows).backward()
        opt.step()
        with torch.no_grad():
            score = float(masked_ce(model(x, K), F, eval_rows))
        if score < best - 1e-7:
            best = score
            state = {n: p.detach().clone() for n, p in model.named_parameters()}
    return state


def fit_baselines(x, K, flows, train_rows, eval_rows, epochs=400, n_workers=1):
    jobs = [(x, K, flows[i], train_rows[i], eval_rows[i], epochs) for i in range(len(flows))]
    results = _run_jobs(_fit_one_baseline, jobs, n_workers)
    return {n: torch.stack([r[n] for r in results]) for n in results[0]}


def _fit_one_group(x, K, F, seed, base_beta, base_topo, train_rows, eval_rows,
                    profile, epochs, fixed_epochs, variant):
    cls = SeparatedMRHGAT if profile == 'factor_hsic' else MRHGAT
    seed_all(seed)
    model = cls(int_channel=variant, base_beta=base_beta, base_topo=base_topo).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=.02)

    def predict():
        return model(x, F, K, train_rows) if profile == 'factor_hsic' else model(x, F, K)

    max_epochs = fixed_epochs if fixed_epochs is not None else epochs
    best = float('inf')
    best_epoch = max_epochs
    stale = 0
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    history = []
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad(set_to_none=True)
        out = predict()
        ce = masked_ce(out['logpi'], F, train_rows)
        reg = model.regularization()
        dep = hsic(out['factor_embeddings'], train_rows) if profile in ('hsic', 'factor_hsic') else ce * 0
        dis = out.get('factor_discrimination', ce * 0)
        loss = ce + 1e-4 * reg + .1 * dep + .5 * dis
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite stabilization objective')
        loss.backward()
        opt.step()
        with torch.no_grad():
            if eval_rows is not None:
                score = masked_ce(predict()['logpi'], F, eval_rows)
                if score < best - 1e-7:
                    best, best_epoch, stale = float(score), epoch, 0
                    state = {n: p.detach().clone() for n, p in model.named_parameters()}
                else:
                    stale += 1
            if epoch == 1 or epoch % 20 == 0:
                history.append({'epoch': epoch, 'ce': float(ce), 'dependence': float(dep),
                                 'factor_discrimination': float(dis)})
        if eval_rows is not None and stale >= 20:
            break
    if eval_rows is None:
        state = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(state[n])
        out = predict()
        final_ce = float(masked_ce(out['logpi'], F, train_rows if eval_rows is None else eval_rows))
        detached = {k: v.detach() for k, v in out.items()}
    return detached, state, {'best_epoch': best_epoch, 'ce': final_ce, 'epochs_run': epoch, 'trace': history}


def fit_group(x, K, flows, seeds, base_states, train_rows, eval_rows=None,
              profile='hsic', epochs=400, fixed_epochs=None, variant='none', n_workers=1):
    jobs = []
    for i, seed in enumerate(seeds):
        ev = eval_rows[i] if eval_rows is not None else None
        fe = fixed_epochs[i] if fixed_epochs is not None else None
        jobs.append((x, K, flows[i], seed, base_states['beta'][i], base_states['topo'][i],
                     train_rows[i], ev, profile, epochs, fe, variant))
    results = _run_jobs(_fit_one_group, jobs, n_workers)
    outs = [r[0] for r in results]
    states = [r[1] for r in results]
    infos = [r[2] for r in results]
    out = {k: torch.stack([o[k] for o in outs]) for k in outs[0]}
    stacked_states = {n: torch.stack([s[n] for s in states]) for n in states[0]}
    info = {'best_epoch': [i['best_epoch'] for i in infos], 'ce': [i['ce'] for i in infos],
            'epochs_run': max(i['epochs_run'] for i in infos), 'trace': [i['trace'] for i in infos]}
    return out, stacked_states, info


def factor_consensus(posteriors, F):
    """All members retained. Base fixed. Match two latent factors to a medoid.

    Factor identities remain anonymous. No supply indicators or mechanism names
    are used in matching; this diagnostic never substitutes for primary scores.
    """
    posteriors = np.asarray(posteriors, dtype=float)
    if posteriors.ndim != 4 or posteriors.shape[2] != 3:
        raise ValueError('Expected members x nodes x three channels x nodes')
    totals = F.sum(1)
    weight = np.sqrt(F / np.maximum(totals[:,None], 1))
    representations = (posteriors[:,:,1:,:] * weight[None,:,None,:]).transpose(0,2,1,3).reshape(len(posteriors),2,-1)
    representations /= np.maximum(np.linalg.norm(representations,axis=-1,keepdims=True),1e-12)
    scores = np.zeros((len(posteriors),len(posteriors)))
    mappings = {}
    for i in range(len(posteriors)):
        for j in range(len(posteriors)):
            cosine = representations[i] @ representations[j].T
            row, col = linear_sum_assignment(-cosine)
            scores[i,j] = cosine[row,col].mean()
            mappings[i,j] = col.tolist()
    reference = int(np.argmax(scores.mean(1)))
    aligned=[]
    permutations=[]
    for i in range(len(posteriors)):
        order=[0]+[c+1 for c in mappings[reference,i]]
        permutations.append(order)
        aligned.append(posteriors[i][:,order,:])
    aligned=np.stack(aligned)
    average=aligned.mean(0)
    disagreements=[]
    for i in range(len(aligned)):
        for j in range(i):
            valid=F>0
            disagreements.append(float((aligned[i].argmax(1)[valid]!=aligned[j].argmax(1)[valid]).mean()))
    info={'reference_member':reference, 'matching_permutations':permutations,
          'mean_pairwise_matched_cosine':float(scores[np.triu_indices(len(scores),1)].mean()),
          'mean_pairwise_edge_label_disagreement':float(np.mean(disagreements)),
          'max_posterior_sum_error':float(np.abs(average.sum(1)-1).max()),
          'truth_used_for_matching':False, 'semantic_naming_performed':False}
    return average, aligned, info
