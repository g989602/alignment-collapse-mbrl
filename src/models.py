"""World model with transition feature phi(z, a, dz).

LayerNorm at phi output: necessary so an untrained phi does not emit ~1e-8
features, which would collapse the BW alignment loss to ~0 and starve phi of
gradient. Standardize-inside-loss is also necessary, see losses.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def mlp(in_dim, out_dim, hidden, depth=2):
    layers = []
    last = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(last, hidden))
        layers.append(nn.SiLU())
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim=32,
        feat_dim=16,
        hidden=128,
        n_domains=1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.n_domains = n_domains

        self.encoder = mlp(state_dim, latent_dim, hidden)
        self.dyn = mlp(latent_dim + action_dim, latent_dim, hidden)
        self.reward_head = mlp(latent_dim + action_dim, 1, hidden)
        self.state_delta_head = mlp(latent_dim + action_dim, state_dim, hidden)
        self.phi = nn.Sequential(
            mlp(latent_dim + action_dim + latent_dim, feat_dim, hidden),
            nn.LayerNorm(feat_dim),
        )

    def encode(self, state):
        return self.encoder(state)

    def predict_delta_latent(self, z, a):
        return self.dyn(torch.cat([z, a], dim=-1))

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
        dz_pred = self.predict_delta_latent(z, action)
        r_pred = self.predict_reward(z, action)
        ds_pred = self.predict_delta_state(z, action)
        u = self.transition_feature(z, action, dz_true)
        return {
            "z": z,
            "z_next": z_next,
            "dz_true": dz_true,
            "dz_pred": dz_pred,
            "r_pred": r_pred,
            "ds_pred": ds_pred,
            "u": u,
        }

    @torch.no_grad()
    def latent_step(self, z, a):
        dz = self.predict_delta_latent(z, a)
        z2 = z + dz
        r = self.predict_reward(z, a)
        return z2, dz, r
