"""Neighbour aggregation and adjusted indicators (线B §2.8)."""
import numpy as np

from decode import decode


def adjust(x, z, posterior, W, decoders, moving=(1, 2), fixed_point=False):
    """w_iv,k = pi_iv p_iv,k (v != i); c~_i,k = (1 - a_i,k) z_i,k + sum_v w_iv,k z_v,k;
    x~ = x + sum_{k in moving} [D_k(c~_k) - D_k(z_k)].
    fixed_point: destinations enter with their own aggregated vectors, c~_k = (1 - a_k) z_k + W_k c~_k,
    solved as c~_k = (I - W_k)^-1 (1 - a_k) z_k."""
    n = len(x)
    share = np.divide(W, W.sum(1, keepdims=True), out=np.zeros_like(W), where=W.sum(1, keepdims=True) > 0)
    w = share[:, None, :] * posterior
    idx = np.arange(n)
    w[idx, :, idx] = 0.0
    a = w.sum(2)
    ctil = (1 - a)[:, :, None] * z + np.einsum('nkv,vkd->nkd', w, z)
    if fixed_point:
        for k in range(w.shape[1]):
            ctil[:, k] = np.linalg.solve(np.eye(n) - w[:, k, :], (1 - a[:, k])[:, None] * z[:, k])
    delta = {k: decode(decoders, k, ctil[:, k]) - decode(decoders, k, z[:, k]) for k in range(w.shape[1])}
    xtil = x + sum(delta[k] for k in moving)
    return xtil, delta, a
