"""Loss functions for the BW-Weighted main method.

Combines the standard MBRL losses (dyn / reward / state-delta) with:
- D1 weighting on the per-sample MSE (low-score transitions contribute less)
- D3 BW-MixUp alignment toward synthetic geodesic Gaussians
- E1 EMA trust region from `bw_weighted.EmaPrototype`
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .spd_distances import coral_align_spd
from .utils import covariance, standardize


def coral_bw_align(u, domain_id, n_domains, eps=1e-6):
    """BW alignment of per-domain feature covariances (drop-in for the existing
    `tcla_loss` baseline at distance='bures')."""
    if u.shape[0] == 0 or n_domains <= 1:
        return torch.zeros((), device=u.device, dtype=u.dtype)
    u = standardize(u)
    per_domain = []
    for d in range(n_domains):
        mask = domain_id == d
        if mask.sum() < 2:
            continue
        per_domain.append(covariance(u[mask], eps=1e-6))
    if len(per_domain) <= 1:
        return torch.zeros((), device=u.device, dtype=u.dtype)
    return coral_align_spd(per_domain, distance="bures", eps=eps)


def weighted_dyn_loss(out, weights):
    """Per-sample-weighted dynamics MSE.

    weights: (N,) tensor in [0, 1].
    """
    target = out["dz_true"].detach()
    pred = out["dz_pred"]
    se = (pred - target) ** 2
    if weights is None:
        return se.mean()
    w = weights.unsqueeze(-1).expand_as(se)
    return (w * se).sum() / (w.sum() + 1e-8)


def weighted_reward_loss(r_pred, r_target, weights):
    se = (r_pred - r_target) ** 2
    if weights is None:
        return se.mean()
    return (weights * se).sum() / (weights.sum() + 1e-8)


def weighted_state_loss(ds_pred, ds_target, weights):
    se = (ds_pred - ds_target) ** 2
    if weights is None:
        return se.mean()
    w = weights.unsqueeze(-1).expand_as(se)
    return (w * se).sum() / (w.sum() + 1e-8)


def variance_floor_loss(u, gamma=1.0):
    if u.shape[0] < 2:
        return torch.zeros((), device=u.device, dtype=u.dtype)
    std = u.std(dim=0, unbiased=False)
    return F.relu(gamma - std).mean()
