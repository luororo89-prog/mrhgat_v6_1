"""Node2vec embedding of a weighted undirected graph (Grover & Leskovec 2016): biased second-order random walks and a
skip-gram with negative sampling; d = 16, walks of length 40, 10 per node, window 5, p = 1, q = 0.5, 5 negative
samples, 5 epochs. Used by benchmark_v13.py and model_v13.py.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn


def node2vec(Wsym, seed, d=16, walks=10, length=40, window=5, p=1.0, q=0.5, neg=5, epochs=5):
    """Node2vec embedding [n, d] of the weighted undirected graph Wsym."""
    rng = np.random.default_rng(seed)
    n = len(Wsym)
    adj = Wsym > 0
    nbr = [np.flatnonzero(adj[i]) for i in range(n)]
    wts = [Wsym[i, nbr[i]] for i in range(n)]
    corpus = []
    for _ in range(walks):
        for s in rng.permutation(n):
            if len(nbr[s]) == 0:
                continue
            walk = [s, rng.choice(nbr[s], p=wts[s] / wts[s].sum())]
            while len(walk) < length:
                cur, prev = walk[-1], walk[-2]
                if len(nbr[cur]) == 0:
                    break
                bias = np.where(nbr[cur] == prev, 1 / p, np.where(adj[prev, nbr[cur]], 1.0, 1 / q))
                pr = wts[cur] * bias
                walk.append(rng.choice(nbr[cur], p=pr / pr.sum()))
            corpus.append(walk)
    pairs = [(w_[i], w_[j]) for w_ in corpus for i in range(len(w_))
             for j in range(max(0, i - window), min(len(w_), i + window + 1)) if i != j]
    pairs = torch.as_tensor(np.array(pairs), dtype=torch.long)
    deg = np.bincount(pairs[:, 1].numpy(), minlength=n) ** 0.75
    noise = torch.as_tensor(deg / deg.sum(), dtype=torch.float32)
    torch.manual_seed(seed)
    emb_in, emb_out = nn.Embedding(n, d), nn.Embedding(n, d)
    nn.init.uniform_(emb_in.weight, -0.5 / d, 0.5 / d)
    nn.init.zeros_(emb_out.weight)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=0.01)
    for _ in range(epochs):
        for b in torch.randperm(len(pairs)).split(4096):
            c, o = pairs[b, 0], pairs[b, 1]
            ng = torch.multinomial(noise, len(b) * neg, replacement=True).view(len(b), neg)
            vi = emb_in(c)
            loss = -(fn.logsigmoid((vi * emb_out(o)).sum(1)) + fn.logsigmoid(-(emb_out(ng) * vi[:, None]).sum(2)).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    e = emb_in.weight.detach().numpy()
    return (e - e.mean(0)) / e.std(0)
