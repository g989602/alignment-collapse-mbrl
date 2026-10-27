"""Three new primitives that compose the main method:

  D1. BW score filter         — `bw_score_filter(...)`
  D3. BW-geodesic MixUp       — `bw_geodesic_mixup(...)`
  E1. EMA env-prototype loss  — `EmaPrototype` class

Each is train-time, mini-batch, closed-form. None requires test-time
projection (the failure mode of TBR-P / BW-projection-support score).
"""
from __future__ import annotations
import torch

from .spd_distances import bures_wasserstein_distance_sq, matrix_sqrt, _spd_eigh
from .utils import covariance, standardize


# --------------------------------------------------------------------------
# D1 — BW Score Filter (VGDF / IGDF / PAR family)
# --------------------------------------------------------------------------

def per_domain_covs(u, domain_id, n_domains, eps=1e-5):
    """Compute the per-domain covariance of features `u` in this minibatch.

    Returns:
        covs: list of (D, D) tensors, one per domain (None if a domain has <2 samples)
        means: list of (D,) tensors
    """
    u = standardize(u)  # scale-invariant; matches our existing alignment loss
    covs, means = [], []
    for d in range(n_domains):
        mask = domain_id == d
        if mask.sum() < 2:
            covs.append(None); means.append(None)
            continue
        ud = u[mask]
        covs.append(covariance(ud, eps=eps))
        means.append(ud.mean(dim=0))
    return covs, means


def bw_score_filter(u, domain_id, n_domains, beta=0.5, eps=1e-5):
    """D1 — per-transition Bures-Wasserstein acceptance score.

    For each sample n in domain d, the proxy "target" is the leave-one-out mean
    of the OTHER domain covariances (BW arithmetic mean — not Frechet-mean — is
    fine here since we only need a smooth target Σ̄_proxy and arithmetic mean is
    PSD on the SPD cone).

    score_n = exp(-β · BW²(Σ_d, Σ̄_proxy_d))     # per-sample weight ∈ (0,1]

    Returns:
        weights:     (N,) tensor in (0, 1]    — multiplier for dyn / reward losses
        log_score:   per-domain dict          — diagnostics
    """
    covs, _ = per_domain_covs(u, domain_id, n_domains, eps=eps)
    valid = [(d, c) for d, c in enumerate(covs) if c is not None]
    if len(valid) <= 1:
        return torch.ones(u.shape[0], device=u.device, dtype=u.dtype), {}

    # compute per-domain BW² to its proxy
    bw2_per_domain = {}
    for d, c_d in valid:
        others = [c for dd, c in valid if dd != d]
        c_proxy = torch.stack(others, dim=0).mean(dim=0)
        bw2_per_domain[d] = bures_wasserstein_distance_sq(c_d, c_proxy, eps=1e-6)

    weights = torch.ones(u.shape[0], device=u.device, dtype=u.dtype)
    log = {}
    for d, bw2 in bw2_per_domain.items():
        score = torch.exp(-beta * bw2)
        mask = domain_id == d
        weights[mask] = score
        log["d" + str(d)] = float(bw2.item())
    return weights, log


def weighted_mse(pred, target, weights, eps=1e-8):
    """Per-sample-weighted MSE that broadcasts `weights` over feature dims."""
    if pred.shape != target.shape:
        target = target.expand_as(pred)
    se = (pred - target) ** 2
    while weights.dim() < se.dim():
        weights = weights.unsqueeze(-1)
    num = (weights * se).sum()
    den = weights.sum() * se.shape[-1] + eps if se.dim() > 1 else weights.sum() + eps
    return num / den


# --------------------------------------------------------------------------
# D3 — BW-Geodesic MixUp on covariances
# --------------------------------------------------------------------------

def bw_geodesic_interpolate(A, B, t, eps=1e-6):
    """Interpolate SPD A,B along the McCann (BW) geodesic at t in [0,1].

    γ(0)=A, γ(1)=B,  T_{A→B} = A^{-1/2}(A^{1/2} B A^{1/2})^{1/2} A^{-1/2},
    γ(t) = ((1-t)I + tT) A ((1-t)I + tT)ᵀ.

    Stays on the SPD manifold for any t ∈ [0,1] (linear interp in matrix space
    can become indefinite if eigen-pairs disagree; the McCann interp does not).
    """
    eA, VA = _spd_eigh(A, eps)
    A_sqrt = VA @ torch.diag(torch.sqrt(eA)) @ VA.t()
    A_inv_sqrt = VA @ torch.diag(1.0 / torch.sqrt(eA)) @ VA.t()
    M = A_sqrt @ B @ A_sqrt
    M_sqrt = matrix_sqrt(M, eps)
    T = A_inv_sqrt @ M_sqrt @ A_inv_sqrt
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    coef = (1.0 - t) * I + t * T
    return coef @ A @ coef.t()


def bw_geodesic_mixup(u, domain_id, n_domains, alpha=0.2, n_pairs=2,
                      eps=1e-5, generator=None):
    """D3 — sample BW-geodesic MixUp Gaussians from minibatch domain covariances.

    Returns a list of synthetic covariance targets the alignment loss can pull
    toward. The covariances stay on the SPD manifold for any λ ∈ [0,1].

    Args:
        alpha: Beta(α, α) parameter; α=0.2 (matches DADS) gives most mass near
               endpoints, with some genuinely interpolated samples.
        n_pairs: number of (i,j) source-pair samples to draw per minibatch.

    Returns:
        synth_covs: list of (D, D) tensors          — synthetic Σ_λ
        synth_means: list of (D,) tensors           — synthetic μ_λ
        pair_log: dict (i, j, λ) per generated pair — diagnostics
    """
    covs, means = per_domain_covs(u, domain_id, n_domains, eps=eps)
    valid = [(d, c, m) for d, (c, m) in enumerate(zip(covs, means)) if c is not None]
    if len(valid) < 2:
        return [], [], []

    rng = torch.Generator(device="cpu") if generator is None else generator
    synth_covs, synth_means, pair_log = [], [], []
    for _ in range(n_pairs):
        idxs = torch.randperm(len(valid), generator=rng if generator is not None else None)[:2]
        (d_i, c_i, m_i), (d_j, c_j, m_j) = valid[int(idxs[0])], valid[int(idxs[1])]
        # λ ~ Beta(α, α)  (PyTorch torch.distributions)
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
        c_lam = bw_geodesic_interpolate(c_i, c_j, lam, eps=1e-6)
        m_lam = (1.0 - lam) * m_i + lam * m_j
        synth_covs.append(c_lam)
        synth_means.append(m_lam)
        pair_log.append({"i": d_i, "j": d_j, "lambda": lam})
    return synth_covs, synth_means, pair_log


def bw_mixup_alignment_loss(u, domain_id, n_domains, synth_covs, eps=1e-5):
    """Alignment loss between each domain's empirical Σ_d and the closest
    synthetic Σ_λ from the BW-MixUp set. This pulls the aligned manifold toward
    the geodesic-interpolated targets.

    L = mean_d  min_λ  BW²(Σ_d, Σ_λ).
    """
    if not synth_covs:
        return torch.zeros((), device=u.device, dtype=u.dtype)
    covs, _ = per_domain_covs(u, domain_id, n_domains, eps=eps)
    valid = [c for c in covs if c is not None]
    if not valid:
        return torch.zeros((), device=u.device, dtype=u.dtype)
    losses = []
    for c_d in valid:
        # min over the synthetic set (closest BW geodesic point)
        bw2s = torch.stack(
            [bures_wasserstein_distance_sq(c_d, c_s, eps=1e-6) for c_s in synth_covs]
        )
        losses.append(bw2s.min())
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------
# E1 — EMA env-prototypes + BW trust-region penalty (Wu 2022 disentangled)
# --------------------------------------------------------------------------

class EmaPrototype:
    """Maintain per-domain EMA prototypes (μ̄_E, Σ̄_E) and emit a BW
    trust-region loss that pulls the current minibatch covariance toward the
    historical prototype.

    Wu et al. 2022 ablation showed that *removing* this EMA target makes the
    method unstable across seeds — exactly the failure mode we're trying to fix.
    """
    def __init__(self, n_domains, feat_dim, tau=0.99, device="cpu", warmup_steps=20):
        self.n_domains = n_domains
        self.feat_dim = feat_dim
        self.tau = float(tau)
        self.device = device
        self.warmup_steps = int(warmup_steps)
        self._mu = [None] * n_domains
        self._cov = [None] * n_domains
        self._n_updates = [0] * n_domains

    def update(self, u, domain_id, eps=1e-5):
        """Update prototypes with this minibatch's per-domain stats. Detached."""
        u = standardize(u).detach()
        for d in range(self.n_domains):
            mask = domain_id == d
            if mask.sum() < 2:
                continue
            mu_d = u[mask].mean(dim=0).detach()
            cov_d = covariance(u[mask], eps=eps).detach()
            if self._mu[d] is None:
                self._mu[d] = mu_d
                self._cov[d] = cov_d
            else:
                self._mu[d] = self.tau * self._mu[d] + (1 - self.tau) * mu_d
                self._cov[d] = self.tau * self._cov[d] + (1 - self.tau) * cov_d
            self._n_updates[d] += 1

    def trust_region_loss(self, u, domain_id, eps=1e-5):
        """Σ_d BW²(Σ̂_d, Σ̄_d_EMA) — train-time trust region.

        Skipped during warmup (first `warmup_steps` updates per domain) so the
        EMA has converged enough to be a meaningful target.
        """
        u_std = standardize(u)
        losses = []
        for d in range(self.n_domains):
            if self._cov[d] is None or self._n_updates[d] < self.warmup_steps:
                continue
            mask = domain_id == d
            if mask.sum() < 2:
                continue
            cov_d = covariance(u_std[mask], eps=eps)
            losses.append(bures_wasserstein_distance_sq(cov_d, self._cov[d].to(cov_d.device), eps=1e-6))
        if not losses:
            return torch.zeros((), device=u.device, dtype=u.dtype)
        return torch.stack(losses).mean()

    def state(self):
        """For checkpointing."""
        return {
            "mu":  [None if m is None else m.cpu() for m in self._mu],
            "cov": [None if c is None else c.cpu() for c in self._cov],
            "n":   list(self._n_updates),
            "tau": self.tau,
        }
