"""Soft-routing arithmetic for the heterogeneous GAT (修订方案 Sections 2-3).

Channels are fixed to (base, comp, compet). Topology (internal, adjacent,
cross) is a separate axis: it only shapes candidate sets, it never becomes a
mechanism ID. Channels removed by an ablation keep their slot with zero prior
and zero posterior so every output has the same layout.
"""
import torch

CHANNELS = ('base', 'comp', 'compet')
INTERNAL, ADJACENT, CROSS = 0, 1, 2


def candidate_masks(K, channels=CHANNELS, adjacent_hard_mask=False):
    """C_base = V, C_comp = C_compet = V minus i.

    adjacent_hard_mask restores the earlier restriction (compet to cross
    destinations only) and exists for the ablation in Section 7.2.
    """
    n = K.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=K.device)
    masks = []
    for c in CHANNELS:
        if c not in channels:
            masks.append(torch.zeros_like(eye))
        elif c == 'base':
            masks.append(torch.ones_like(eye))
        elif c == 'compet' and adjacent_hard_mask:
            masks.append(K == CROSS)
        else:
            masks.append(~eye)
    return torch.stack(masks, 1)


def center_increment(delta, mask):
    """Mean-center an increment over the allowed destinations of each origin."""
    m = mask.to(delta.dtype)
    mean = (delta * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1)
    return (delta - mean) * m


def channel_log_probs(scores, masks):
    """Per-channel destination log-probabilities, normalized over each origin's candidates.

    scores and masks are [origin, channel, destination]. Rows without any
    candidate are filled with a finite placeholder before the softmax so that
    no NaN reaches the backward pass, then set to -inf.
    """
    has_any = masks.any(-1, keepdim=True)
    safe = torch.where(masks, scores, torch.full_like(scores, -torch.inf))
    safe = torch.where(has_any, safe, torch.zeros_like(scores))
    logp = torch.log_softmax(safe, -1)
    return torch.where(masks, logp, torch.full_like(logp, -torch.inf))


def mixture(log_omega, logp, masks):
    """Mix channel destination probabilities with node priors.

    log_omega is [origin, channel] with log_omega_base = 0. A channel without
    candidates for an origin gets zero prior. Returns log prior, log pi_hat and
    the posterior p_ij,k = lambda_ik P_k(j|i) / pi_hat_ij.
    """
    available = masks.any(-1)
    log_lambda = torch.log_softmax(log_omega.masked_fill(~available, -torch.inf), 1)
    logjoint = log_lambda[:, :, None] + logp
    logpi = torch.logsumexp(logjoint, 1)
    posterior = torch.exp(logjoint - logpi[:, None, :])
    return log_lambda, logpi, posterior


def two_level(posterior, a_min=0.05, eps=1e-12):
    """a = non-base share of an edge, b = comp share inside the non-base part.

    b is only interpretable where a >= a_min; b_valid carries that flag.
    """
    a = posterior[:, 1] + posterior[:, 2]
    b = posterior[:, 1] / a.clamp_min(eps)
    return a, b, a >= a_min


def recompose(a, b):
    return torch.stack([1 - a, a * b, a * (1 - b)], 1)


def origin_gates(A_C, A_D):
    """Two-level gate parameters equivalent to log omega = (0, A_C, A_D).

    sigmoid(g) is the node-level non-base prior and sigmoid(t) the comp prior
    inside the non-base part.
    """
    return torch.logsumexp(torch.stack([A_C, A_D], -1), -1), A_C - A_D


def priors_from_origin_gates(g, t):
    non_base = torch.sigmoid(g)
    comp = torch.sigmoid(t)
    return torch.stack([1 - non_base, non_base * comp, non_base * (1 - comp)], -1)


def soft_counts(W, posterior):
    """Model accounting of observed counts by channel; adds back to W exactly."""
    return W[:, None, :] * posterior
