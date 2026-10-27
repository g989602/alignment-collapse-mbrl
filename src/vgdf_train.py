"""Training loop for the VGDF baseline."""
from __future__ import annotations
import numpy as np
import torch

from .data import TransitionBuffer
from .losses import (
    coral_bw_align,
    weighted_dyn_loss,
    weighted_reward_loss,
    weighted_state_loss,
)
from .vgdf import VgdfWorldModel, vgdf_weights


def train_vgdf(model: VgdfWorldModel, buffer: TransitionBuffer, cfg) -> dict:
    """Drop-in training loop.

    cfg keys (in addition to the standard MBRL ones):
        n_heads:    ensemble size (default 5; configured by model)
        beta_vgdf:  uncertainty-to-weight sharpness (default 2.0)
    """
    device = cfg["device"]
    rng = np.random.default_rng(cfg["seed"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    history = []
    per_domain_size = max(8, cfg["batch_size"] // max(1, buffer.num_domains))
    beta_vgdf = float(cfg.get("beta_vgdf", 2.0))

    n_steps = max(1, len(buffer) // cfg["batch_size"])
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        meters = {k: [] for k in ("total", "dyn_ens", "rew", "state", "align",
                                  "var_min", "var_mean", "w_min", "w_mean")}
        for _ in range(n_steps):
            batch = buffer.sample_balanced(per_domain_size, rng)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["state"], batch["action"], batch["next_state"])
            u = out["u"]
            dom = batch["domain_id"]

            # ---- VGDF: per-transition uncertainty -> weights ----
            dz_var = out["dz_var"]
            weights = vgdf_weights(dz_var, beta=beta_vgdf)

            # ---- ensemble dynamics loss: each head trained on its own MSE,
            #      weighted by the SAME per-transition weight (so high-var
            #      transitions contribute less to all heads).
            target = out["dz_true"].detach()
            dz_all = out["dz_all"]                          # (K, B, D)
            se_per_head = (dz_all - target.unsqueeze(0)) ** 2
            w = weights.view(1, -1, 1)
            l_dyn_ens = (w * se_per_head).sum() / (
                w.expand_as(se_per_head).sum() + 1e-8
            )

            # ---- standard reward / state losses, same weighting
            l_rew  = weighted_reward_loss(out["r_pred"], batch["reward"], weights)
            l_state = weighted_state_loss(
                out["ds_pred"], batch["next_state"] - batch["state"], weights,
            )
            # alignment kept (so VGDF gets the same alignment scaffold as MAIN)
            l_align = coral_bw_align(u, dom, buffer.num_domains)

            total = (
                cfg["w_dyn"]    * l_dyn_ens
                + cfg["w_reward"] * l_rew
                + cfg["w_state"]  * l_state
                + cfg["w_align"]  * l_align
            )
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            meters["total"].append(float(total.item()))
            meters["dyn_ens"].append(float(l_dyn_ens.item()))
            meters["rew"].append(float(l_rew.item()))
            meters["state"].append(float(l_state.item()))
            meters["align"].append(float(l_align.item()))
            meters["var_min"].append(float(dz_var.mean(dim=-1).min().item()))
            meters["var_mean"].append(float(dz_var.mean().item()))
            meters["w_min"].append(float(weights.min().item()))
            meters["w_mean"].append(float(weights.mean().item()))

        epoch_log = {"epoch": epoch}
        for k, v in meters.items():
            epoch_log[k] = float(np.mean(v)) if v else 0.0
        history.append(epoch_log)

    return {"history": history, "n_domains": buffer.num_domains}
