"""SPD-manifold-aware distances between covariance matrices.

All distances assume A, B are symmetric positive definite. We add a small
ridge before eigendecomposition for numerical stability.

References:
- Pennec, Fillard, Ayache (2006) "A Riemannian Framework for Tensor Computing"
- Bhatia, Jain, Lim (2019) "On the Bures-Wasserstein distance between PSD matrices"
- Arsigny et al. (2007) "Log-Euclidean metrics for fast and simple calculus on diffusion tensors"
"""
from __future__ import annotations
import torch


def _spd_eigh(A, eps=1e-6):
    """Symmetric eigendecomposition with numerical floor.

    Tries eigh with progressively larger ridge on convergence failure. This
    matters when source covariances become severely ill-conditioned at high
    domain perturbation -- without retries, eigh raises LinAlgError #15
    ("repeated eigenvalues / ill-conditioned input") and crashes the run.
    """
    n = A.shape[0]
    A = 0.5 * (A + A.t())
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    last_err = None
    for ridge in (eps, 1e-4, 1e-3, 1e-2, 1e-1):
        try:
            eigvals, eigvecs = torch.linalg.eigh(A + ridge * I)
            return eigvals.clamp(min=eps), eigvecs
        except Exception as exc:  # torch._C._LinAlgError or RuntimeError
            last_err = exc
    # final fallback: SVD-based path (always converges for symmetric input)
    U, S, _Vh = torch.linalg.svd(A + 1e-1 * I)
    return S.clamp(min=eps), U


def matrix_sqrt(A, eps=1e-6):
    """Principal square root of an SPD matrix."""
    e, V = _spd_eigh(A, eps)
    return V @ torch.diag(torch.sqrt(e)) @ V.t()


def matrix_log(A, eps=1e-6):
    """Matrix log of an SPD matrix."""
    e, V = _spd_eigh(A, eps)
    return V @ torch.diag(torch.log(e)) @ V.t()


def log_euclidean_distance_sq(A, B, eps=1e-6):
    """Log-Euclidean squared distance: ||log(A) - log(B)||_F^2.

    Cheap, smooth, but not affine-invariant.
    """
    return ((matrix_log(A, eps) - matrix_log(B, eps)) ** 2).sum()


def affine_invariant_distance_sq(A, B, eps=1e-6):
    """AIRM squared distance: ||log(A^{-1/2} B A^{-1/2})||_F^2.

    True SPD geodesic. Affine-invariant. More expensive.
    """
    e, V = _spd_eigh(A, eps)
    A_inv_sqrt = V @ torch.diag(1.0 / torch.sqrt(e)) @ V.t()
    M = A_inv_sqrt @ B @ A_inv_sqrt
    return (matrix_log(M, eps) ** 2).sum()


def bures_wasserstein_distance_sq(A, B, eps=1e-6):
    """Bures-Wasserstein squared distance between SPD matrices.

    d^2(A, B) = tr(A) + tr(B) - 2 tr((A^{1/2} B A^{1/2})^{1/2})

    This is the 2-Wasserstein distance between two centered Gaussians with
    covariances A and B. Geometric, statistical, and SPD-aware.
    """
    A_sqrt = matrix_sqrt(A, eps)
    M = A_sqrt @ B @ A_sqrt
    e_M, _ = _spd_eigh(M, eps)
    M_sqrt_trace = torch.sqrt(e_M).sum()
    return (torch.trace(A) + torch.trace(B) - 2.0 * M_sqrt_trace).clamp(min=0.0)


def coral_align_spd(per_domain_cov, distance="bures", eps=1e-6):
    """Mean SPD-distance between each domain's covariance and the
    arithmetic-mean balanced covariance.

    distance in {"frobenius", "log_eucl", "airm", "bures"}.
    Frobenius is the original CORAL.
    """
    if len(per_domain_cov) <= 1:
        return torch.zeros(())
    stacked = torch.stack(per_domain_cov, dim=0)
    bal = stacked.mean(dim=0)
    losses = []
    for cov in per_domain_cov:
        if distance == "frobenius":
            losses.append(((cov - bal) ** 2).sum())
        elif distance == "log_eucl":
            losses.append(log_euclidean_distance_sq(cov, bal, eps))
        elif distance == "airm":
            losses.append(affine_invariant_distance_sq(cov, bal, eps))
        elif distance == "bures":
            losses.append(bures_wasserstein_distance_sq(cov, bal, eps))
        else:
            raise ValueError("unknown distance: " + distance)
    return torch.stack(losses).mean()
