"""Evaluation: per-domain return + worst-domain summary.

We use a simple shooting CEM with the trained dynamics model so the evaluation
mirrors the policy class used in the source data (random actions during data
collection, planned actions at test time).
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def cem_action(model, state_np, action_dim, action_low, action_high,
               horizon=10, pop=64, iters=2, elite_frac=0.1, device="cpu"):
    """Lightweight CEM planner using the world model's latent rollout."""
    s = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    z = model.encode(s)
    mu = torch.zeros(horizon, action_dim, device=device)
    sigma = torch.ones(horizon, action_dim, device=device) * 0.5
    n_elite = max(1, int(pop * elite_frac))
    for _ in range(iters):
        eps = torch.randn(pop, horizon, action_dim, device=device)
        a = (mu.unsqueeze(0) + sigma.unsqueeze(0) * eps).clamp(action_low, action_high)
        z_p = z.expand(pop, -1).contiguous()
        ret = torch.zeros(pop, device=device)
        for t in range(horizon):
            r = model.predict_reward(z_p, a[:, t])
            dz = model.predict_delta_latent(z_p, a[:, t])
            z_p = z_p + dz
            ret = ret + r
        topk = ret.topk(n_elite).indices
        elite = a[topk]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp(min=1e-3)
    return mu[0].detach().cpu().numpy()


def evaluate(model, env, n_episodes=2, horizon=10, pop=64, iters=2, device="cpu"):
    """Return (per_domain_mean, per_domain_worst, mean_return, worst_return)."""
    per_domain = []
    for d in range(env.num_domains):
        rets = []
        for _ in range(n_episodes):
            s = env.reset(domain_idx=d)
            done = False
            ret = 0.0
            while not done:
                a = cem_action(model, s, env.action_dim,
                               env.action_low, env.action_high,
                               horizon=horizon, pop=pop, iters=iters, device=device)
                s, r, done, _ = env.step(a)
                ret += float(r)
            rets.append(ret)
        per_domain.append(float(np.mean(rets)))
    return {
        "per_domain":   per_domain,
        "mean":         float(np.mean(per_domain)),
        "worst":        float(np.min(per_domain)),
    }
