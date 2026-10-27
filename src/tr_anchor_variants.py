"""Anchor-family interventions: variants of the EMA prototype anchor.

Each variant keeps the SAME penalty form (mean over domains of
BW^2(minibatch cov, anchor cov)), the same tau/warmup, and the same update
cadence as EmaPrototype; only the anchor construction differs. This isolates
WHAT the trust region needs:

  GlobalEmaPrototype   — one shared EMA anchor for all domains.
                         Tests: is domain-SPECIFIC history necessary?
  ShuffledEmaPrototype — per-domain EMA anchors, but domain d is pulled toward
                         the anchor of domain (d+1) mod K (fixed derangement).
                         Same penalty scale and covariance statistics; only the
                         domain correspondence is broken.
                         Tests: is the CORRECT domain correspondence necessary?
  FrozenEmaPrototype   — per-domain anchors frozen after `freeze_after`
                         updates (early-training covariance as fixed target).
                         Tests: is slow adaptation necessary, or does a fixed
                         early anchor suffice?
"""
from __future__ import annotations
import torch

from .bw_weighted import EmaPrototype
from .spd_distances import bures_wasserstein_distance_sq
from .utils import covariance, standardize


class GlobalEmaPrototype(EmaPrototype):
    """Single EMA anchor shared by every domain (n_domains=1 internally)."""

    def __init__(self, n_domains, feat_dim, tau=0.99, device="cpu", warmup_steps=20):
        super().__init__(1, feat_dim, tau=tau, device=device,
                         warmup_steps=warmup_steps)
        self._true_n_domains = n_domains

    def update(self, u, domain_id, eps=1e-5):
        # all samples update the single global anchor
        super().update(u, torch.zeros_like(domain_id), eps=eps)

    def trust_region_loss(self, u, domain_id, eps=1e-5):
        if self._cov[0] is None or self._n_updates[0] < self.warmup_steps:
            return torch.zeros((), device=u.device, dtype=u.dtype)
        u_std = standardize(u)
        losses = []
        for d in range(self._true_n_domains):
            mask = domain_id == d
            if mask.sum() < 2:
                continue
            cov_d = covariance(u_std[mask], eps=eps)
            losses.append(bures_wasserstein_distance_sq(
                cov_d, self._cov[0].to(cov_d.device), eps=1e-6))
        if not losses:
            return torch.zeros((), device=u.device, dtype=u.dtype)
        return torch.stack(losses).mean()


class ShuffledEmaPrototype(EmaPrototype):
    """Per-domain EMA anchors with a fixed derangement of the correspondence."""

    def trust_region_loss(self, u, domain_id, eps=1e-5):
        u_std = standardize(u)
        losses = []
        for d in range(self.n_domains):
            dp = (d + 1) % self.n_domains  # fixed derangement
            if self._cov[dp] is None or self._n_updates[dp] < self.warmup_steps:
                continue
            mask = domain_id == d
            if mask.sum() < 2:
                continue
            cov_d = covariance(u_std[mask], eps=eps)
            losses.append(bures_wasserstein_distance_sq(
                cov_d, self._cov[dp].to(cov_d.device), eps=1e-6))
        if not losses:
            return torch.zeros((), device=u.device, dtype=u.dtype)
        return torch.stack(losses).mean()


class FrozenEmaPrototype(EmaPrototype):
    """Per-domain anchors frozen after `freeze_after` updates per domain."""

    def __init__(self, n_domains, feat_dim, tau=0.99, device="cpu",
                 warmup_steps=20, freeze_after=40):
        super().__init__(n_domains, feat_dim, tau=tau, device=device,
                         warmup_steps=warmup_steps)
        self.freeze_after = int(freeze_after)

    def update(self, u, domain_id, eps=1e-5):
        if all(n >= self.freeze_after for n in self._n_updates):
            return
        # only update domains still below the freeze point
        u_std = standardize(u).detach()
        for d in range(self.n_domains):
            if self._n_updates[d] >= self.freeze_after:
                continue
            mask = domain_id == d
            if mask.sum() < 2:
                continue
            mu_d = u_std[mask].mean(dim=0).detach()
            cov_d = covariance(u_std[mask], eps=eps).detach()
            if self._mu[d] is None:
                self._mu[d] = mu_d
                self._cov[d] = cov_d
            else:
                self._mu[d] = self.tau * self._mu[d] + (1 - self.tau) * mu_d
                self._cov[d] = self.tau * self._cov[d] + (1 - self.tau) * cov_d
            self._n_updates[d] += 1


class IdentityAnchor(EmaPrototype):
    """Fixed identity-covariance target — BW^2(Sigma_hat_d, I).

    A generic (VICReg-like) geometry target with zero parameters and no
    history. Tests whether a domain-agnostic fixed geometry suffices, vs the
    domain-specific historical anchor of TR."""

    def update(self, u, domain_id, eps=1e-5):
        # no history to maintain; count updates so warmup semantics match
        for d in range(self.n_domains):
            self._n_updates[d] += 1

    def trust_region_loss(self, u, domain_id, eps=1e-5):
        if min(self._n_updates) < self.warmup_steps:
            return torch.zeros((), device=u.device, dtype=u.dtype)
        u_std = standardize(u)
        eye = torch.eye(self.feat_dim, device=u.device, dtype=u.dtype)
        losses = []
        for d in range(self.n_domains):
            mask = domain_id == d
            if mask.sum() < 2:
                continue
            cov_d = covariance(u_std[mask], eps=eps)
            losses.append(bures_wasserstein_distance_sq(cov_d, eye, eps=1e-6))
        if not losses:
            return torch.zeros((), device=u.device, dtype=u.dtype)
        return torch.stack(losses).mean()


ANCHOR_VARIANTS = {
    "global": GlobalEmaPrototype,
    "shuffled": ShuffledEmaPrototype,
    "frozen": FrozenEmaPrototype,
    "identity": IdentityAnchor,
}
