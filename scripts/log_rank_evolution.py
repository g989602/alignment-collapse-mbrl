"""Log effective-rank evolution of per-domain feature covariances during training.

Tracks BW collapse via a complementary lens: effective rank
(Shannon entropy of normalized eigenvalues, exponentiated). Full-rank Σ has
erank = p (= feat_dim); rank-1 collapse has erank = 1.

For BASELINE (BW-CORAL only) and MAIN (TR added), logs per-epoch erank
of the per-domain Σ_d.

Output: checkpoints/RANK_<MAIN|BASELINE>_<env>_seed<seed>/rank_history.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.bw_weighted import (
    EmaPrototype, bw_geodesic_mixup, bw_mixup_alignment_loss, bw_score_filter,
    bures_wasserstein_distance_sq, covariance, standardize,
)
from src.data import TransitionBuffer
from src.losses import (coral_bw_align, weighted_dyn_loss, weighted_reward_loss,
                        weighted_state_loss)
from src.models import WorldModel
from src.mujoco_env import (DEFAULT_SOURCE_SHIFTS, DEFAULT_TARGET_SHIFTS,
                            ENV_DEFAULTS, make_buffers)
from src.utils import seed_everything

ENV_ID = {"walker2d": "Walker2d-v4", "hopper": "Hopper-v4",
          "ant": "Ant-v4", "halfcheetah": "HalfCheetah-v4",
          "humanoid": "Humanoid-v4"}


def effective_rank(sigma: torch.Tensor, eps: float = 1e-12) -> float:
    """Roy-Vetterli effective rank: exp(-sum p_i log p_i) with p_i = lambda_i / sum."""
    eigvals = torch.linalg.eigvalsh(sigma).clamp_min(0.0)
    s = eigvals.sum()
    if s < eps:
        return 0.0
    p = eigvals / s
    p = p[p > eps]
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H).item())


def measure_per_domain(model, buffer, device):
    """Returns list of (cov_matrix, erank) per domain."""
    model.eval()
    out = []
    with torch.no_grad():
        for d in buffer.domain_ids:
            arrs = buffer.all_for_domain(d)
            s = arrs["state"].to(device); a = arrs["action"].to(device); sn = arrs["next_state"].to(device)
            n = s.shape[0]
            idx = torch.randperm(n)[:1000].to(device)
            s, a, sn = s[idx], a[idx], sn[idx]
            z = model.encode(s); zn = model.encode(sn); dz = zn - z
            u = model.transition_feature(z, a, dz)
            u = standardize(u)
            cov = covariance(u, eps=1e-5)
            out.append((cov, effective_rank(cov)))
    return out


def train_with_rank_logging(model, buffer, cfg, use_tr=True):
    device = cfg["device"]
    rng = np.random.default_rng(cfg["seed"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    per_dom = max(8, cfg["batch_size"] // max(1, buffer.num_domains))
    n_steps = max(1, len(buffer) // cfg["batch_size"])

    ema = EmaPrototype(
        n_domains=buffer.num_domains, feat_dim=model.feat_dim,
        tau=cfg.get("ema_tau", 0.99), device=device,
        warmup_steps=cfg.get("ema_warmup", 20),
    )

    history = []
    init = measure_per_domain(model, buffer, device)
    history.append({"epoch": 0, "erank_per_domain": [e for _, e in init],
                    "erank_mean": float(np.mean([e for _, e in init]))})

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for _ in range(n_steps):
            batch = buffer.sample_balanced(per_dom, rng)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["state"], batch["action"], batch["next_state"])
            u = out["u"]; dom = batch["domain_id"]

            l_align = coral_bw_align(u, dom, buffer.num_domains)
            l_dyn = weighted_dyn_loss(out, torch.ones(u.shape[0], device=device, dtype=u.dtype))
            l_rew = weighted_reward_loss(out["r_pred"], batch["reward"], torch.ones(u.shape[0], device=device, dtype=u.dtype))
            l_state = weighted_state_loss(out["ds_pred"], batch["next_state"] - batch["state"], torch.ones(u.shape[0], device=device, dtype=u.dtype))

            if use_tr:
                l_ema = ema.trust_region_loss(u, dom); ema.update(u, dom)
            else:
                l_ema = torch.zeros((), device=device)

            total = (cfg["w_dyn"]*l_dyn + cfg["w_reward"]*l_rew
                     + cfg["w_state"]*l_state + cfg["w_align"]*l_align
                     + (1.0 if use_tr else 0.0) * l_ema)
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        snap = measure_per_domain(model, buffer, device)
        eranks = [e for _, e in snap]
        history.append({"epoch": epoch, "erank_per_domain": eranks,
                        "erank_mean": float(np.mean(eranks))})
        print(f"  epoch {epoch}: erank/domain = {[f'{e:.2f}' for e in eranks]}, mean = {np.mean(eranks):.2f}", flush=True)

    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="walker2d")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--method", required=True, choices=["MAIN", "BASELINE", "VGDF", "VGDFTR"])
    p.add_argument("--feat_dim", type=int, default=None, help="override feat_dim from config")
    p.add_argument("--transitions", type=int, default=None, help="override transitions_per_domain")
    p.add_argument("--label", default="", help="extra suffix on output dir, e.g. fd32")
    args = p.parse_args()

    cfg_file = Path("configs") / (f"baseline_{args.env}.json" if args.method == "BASELINE"
                                   else f"{args.env}.json")
    cfg = json.loads(cfg_file.read_text())
    cfg["seed"] = args.seed; cfg["device"] = args.device
    cfg["epochs"] = args.epochs
    if args.feat_dim is not None: cfg["feat_dim"] = int(args.feat_dim)
    if args.transitions is not None: cfg["transitions_per_domain"] = int(args.transitions)
    seed_everything(cfg["seed"])

    env_id = ENV_ID[args.env]
    defaults = ENV_DEFAULTS.get(env_id, {"max_steps": 200, "transitions_per_domain": 4000})
    src_env, tgt_env, buffers = make_buffers(
        env_id, DEFAULT_SOURCE_SHIFTS, DEFAULT_TARGET_SHIFTS,
        transitions_per_domain=cfg.get("transitions_per_domain", defaults["transitions_per_domain"]),
        seed=cfg["seed"], domain_seed=cfg.get("domain_seed", 42),
        max_steps=cfg.get("max_steps", defaults["max_steps"]),
    )
    buffer = TransitionBuffer(buffers)

    model = WorldModel(state_dim=src_env.state_dim, action_dim=src_env.action_dim,
                       latent_dim=cfg.get("latent_dim", 32), feat_dim=cfg.get("feat_dim", 16),
                       hidden=cfg.get("hidden", 128), n_domains=buffer.num_domains).to(cfg["device"])

    use_tr = (args.method == "MAIN")
    print(f"[{args.env} seed{args.seed} method={args.method} feat_dim={cfg.get('feat_dim',16)}] use_tr={use_tr}")
    history = train_with_rank_logging(model, buffer, cfg, use_tr=use_tr)

    suffix = f"_{args.label}" if args.label else ""
    out_dir = Path("checkpoints") / f"RANK_{args.method}_{args.env}{suffix}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rank_history.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed, "method": args.method,
        "feat_dim": cfg.get("feat_dim", 16), "history": history,
    }, indent=2))
    print(f"wrote {out_dir / 'rank_history.json'}")
    src_env.close(); tgt_env.close()


if __name__ == "__main__":
    main()
