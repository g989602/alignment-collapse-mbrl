"""Numerics utilities — covariance, seeding, standardization."""
from __future__ import annotations
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def covariance(x, eps=1e-6):
    """Sample covariance with Bessel correction. (N,D) -> (D,D)."""
    n = x.shape[0]
    d = x.shape[1]
    if n < 2:
        return torch.eye(d, device=x.device, dtype=x.dtype) * eps
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    cov = (xc.t() @ xc) / (n - 1)
    return cov + eps * torch.eye(d, device=x.device, dtype=x.dtype)


def standardize(u, eps=1e-5):
    if u.shape[0] < 2:
        return u
    mu = u.mean(dim=0, keepdim=True)
    std = u.std(dim=0, keepdim=True) + eps
    return (u - mu) / std
