"""VGDF (Variational Generalized Domain Filter) baseline.

Xu et al. 2023 — uses an ensemble of K dynamics heads. Per-transition
uncertainty (variance of K predictions) acts as a "domain mismatch" proxy:
high-variance transitions are likely outliers, get down-weighted in the
training loss.

This is the most-cited closed-target SOTA in cross-dynamics RL and the
direct competitor to our D1 BW-Score-Filter (both reweight source
transitions, but VGDF uses ensemble disagreement and D1 uses BW-distance
between domain covariances).

Adaptation for our setup:
- K dynamics heads share the same encoder; each predicts a different `dz`
- Per-transition variance = mean over feature dims of var of K predictions
- Weight  w(s,a,s') = exp(-β · normalized_variance) ∈ (0, 1]
- Used to weight L_dyn / L_rew / L_state (same loss surface as D1)
- alignment loss (CORAL-BW) kept identical for fair comparison
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .models import mlp


class EnsembleDynamics(nn.Module):
    """K independent dynamics heads sharing the input space (z, a)."""
    def __init__(self, latent_dim, action_dim, hidden=128, n_heads=5):
        super().__init__()
        self.n_heads = int(n_heads)
        self.heads = nn.ModuleList([
            mlp(latent_dim + action_dim, latent_dim, hidden) for _ in range(self.n_heads)
        ])

    def forward_all(self, z, a):
        """Return stacked predictions of shape (K, B, latent_dim)."""
        x = torch.cat([z, a], dim=-1)
        return torch.stack([h(x) for h in self.heads], dim=0)

    def mean_var(self, z, a):
        preds = self.forward_all(z, a)               # (K, B, D)
        return preds.mean(dim=0), preds.var(dim=0, unbiased=False)


def vgdf_weights(variance, beta=2.0):
    """Convert per-transition variance to a (0, 1] weight.

    variance: (B, D) tensor; we average over D to get a scalar uncertainty per sample.
    """
    var_scalar = variance.mean(dim=-1).detach()          # (B,)
    # robust normalization to keep beta scale-invariant
    med = var_scalar.median().clamp(min=1e-6)
    norm = var_scalar / med                               # ~1.0 around the median
    w = torch.exp(-beta * (norm - 1.0).clamp(min=0.0))    # samples below median ~1.0
    return w


class VgdfWorldModel(nn.Module):
    """World model with VGDF ensemble in place of the single dyn head.

    Drop-in compatible with `train.py` / `eval.py` patterns: exposes the
    standard `predict_delta_latent`, `predict_reward`, etc. The training
    loop is in `vgdf_train.train_vgdf` (separate to keep core train.py clean).
    """
    def __init__(self, state_dim, action_dim, latent_dim=32, feat_dim=16,
                 hidden=128, n_heads=5, n_domains=1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.n_heads = n_heads
        self.n_domains = n_domains

        self.encoder = mlp(state_dim, latent_dim, hidden)
        self.dyn_ensemble = EnsembleDynamics(latent_dim, action_dim, hidden, n_heads)
        self.reward_head = mlp(latent_dim + action_dim, 1, hidden)
        self.state_delta_head = mlp(latent_dim + action_dim, state_dim, hidden)
        self.phi = nn.Sequential(
            mlp(latent_dim + action_dim + latent_dim, feat_dim, hidden),
            nn.LayerNorm(feat_dim),
        )

    def encode(self, state):
        return self.encoder(state)

    def predict_delta_latent(self, z, a):
        # use head 0 at eval time (CEM planner) for deterministic behavior
        return self.dyn_ensemble.heads[0](torch.cat([z, a], dim=-1))

    def predict_reward(self, z, a):
        return self.reward_head(torch.cat([z, a], dim=-1)).squeeze(-1)

    def predict_delta_state(self, z, a):
        return self.state_delta_head(torch.cat([z, a], dim=-1))

    def transition_feature(self, z, a, dz):
        return self.phi(torch.cat([z, a, dz], dim=-1))

    def forward(self, state, action, next_state):
        z = self.encode(state)
        z_next = self.encode(next_state)
        dz_true = z_next - z
        dz_all = self.dyn_ensemble.forward_all(z, action)         # (K, B, D)
        dz_mean = dz_all.mean(dim=0)
        dz_var = dz_all.var(dim=0, unbiased=False)
        r_pred = self.predict_reward(z, action)
        ds_pred = self.predict_delta_state(z, action)
        u = self.transition_feature(z, action, dz_true)
        return {
            "z": z, "z_next": z_next,
            "dz_true": dz_true,
            "dz_pred": dz_mean,
            "dz_all": dz_all,
            "dz_var": dz_var,
            "r_pred": r_pred,
            "ds_pred": ds_pred,
            "u": u,
        }
