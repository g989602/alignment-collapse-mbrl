"""Driver for the VGDF baseline. Mirrors run_main.py so all logging,
output dirs, and aggregate.py downstream work the same way.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data import TransitionBuffer
from src.eval import evaluate
from src.vgdf import VgdfWorldModel
from src.vgdf_train import train_vgdf
from src.mujoco_env import (
    DEFAULT_SOURCE_SHIFTS, DEFAULT_TARGET_SHIFTS, HARD_TARGET_SHIFTS, ODRL_SOURCE_SHIFTS, ODRL_TARGET_SHIFTS,
    ENV_DEFAULTS, make_buffers,
)
from src.utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_root", default="checkpoints")
    p.add_argument("--prefix", default="VGDF")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=5)
    p.add_argument("--beta_vgdf", type=float, default=2.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--shift_set", default="default", choices=["default", "hard", "odrl"])
    p.add_argument("--transitions", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg["seed"] = int(args.seed)
    if args.device is not None: cfg["device"] = args.device
    if args.epochs is not None: cfg["epochs"] = int(args.epochs)
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["beta_vgdf"] = float(args.beta_vgdf)
    if args.smoke:
        cfg["epochs"] = 1
        cfg["transitions_per_domain"] = 200
        cfg["batch_size"] = 32

    env_id = cfg["env_id"]; env_short = env_id.split("-")[0].lower()
    eff_prefix = args.prefix if args.shift_set == "default" else (f"{args.prefix}_ODRL" if args.shift_set == "odrl" else f"{args.prefix}_HARD")
    out_dir = Path(args.out_root) / f"{eff_prefix}_{env_short}_seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.shift_set == "hard": target_shifts = HARD_TARGET_SHIFTS; src_shifts = DEFAULT_SOURCE_SHIFTS
    elif args.shift_set == "odrl": target_shifts = ODRL_TARGET_SHIFTS; src_shifts = ODRL_SOURCE_SHIFTS
    else: target_shifts = DEFAULT_TARGET_SHIFTS; src_shifts = DEFAULT_SOURCE_SHIFTS
    seed_everything(cfg["seed"])
    log_lines = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); log_lines.append(line)
    log(f"VGDF env={env_id} seed={cfg['seed']} device={cfg['device']} n_heads={args.n_heads} beta={args.beta_vgdf}")

    defaults = ENV_DEFAULTS.get(env_id, {"max_steps": 200, "transitions_per_domain": 4000})
    if args.transitions is not None: cfg["transitions_per_domain"] = int(args.transitions)
    transitions = cfg.get("transitions_per_domain", defaults["transitions_per_domain"])
    max_steps = cfg.get("max_steps", defaults["max_steps"])

    src_env, tgt_env, buffers = make_buffers(
        env_id, src_shifts, target_shifts,
        transitions_per_domain=transitions,
        seed=cfg["seed"], domain_seed=cfg.get("domain_seed", 42),
        max_steps=max_steps,
    )
    buffer = TransitionBuffer(buffers)
    log(f"buffers: {buffer.num_domains} domains, {len(buffer)} total transitions")

    model = VgdfWorldModel(
        state_dim=src_env.state_dim, action_dim=src_env.action_dim,
        latent_dim=cfg.get("latent_dim", 32), feat_dim=cfg.get("feat_dim", 16),
        hidden=cfg.get("hidden", 128), n_heads=args.n_heads,
        n_domains=buffer.num_domains,
    ).to(cfg["device"])

    log("starting train_vgdf ...")
    t0 = time.time()
    info = train_vgdf(model, buffer, cfg)
    train_time = time.time() - t0
    log(f"trained in {train_time:.1f}s; epochs={cfg['epochs']}")

    log("evaluating on source domains ...")
    src_metrics = evaluate(model, src_env, n_episodes=cfg.get("eval_episodes", 2),
                           horizon=cfg.get("plan_horizon", 10),
                           pop=cfg.get("plan_pop", 32),
                           iters=cfg.get("plan_iters", 2),
                           device=cfg["device"])
    log(f"source: mean={src_metrics['mean']:.2f} worst={src_metrics['worst']:.2f}")
    log("evaluating on target domains ...")
    tgt_metrics = evaluate(model, tgt_env, n_episodes=cfg.get("eval_episodes", 2),
                           horizon=cfg.get("plan_horizon", 10),
                           pop=cfg.get("plan_pop", 32),
                           iters=cfg.get("plan_iters", 2),
                           device=cfg["device"])
    log(f"target: mean={tgt_metrics['mean']:.2f} worst={tgt_metrics['worst']:.2f}")

    metrics = {"config": cfg, "history": info["history"],
               "source": src_metrics, "target": tgt_metrics,
               "train_time_s": train_time}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out_dir / "model.pt")
    (out_dir / "log.txt").write_text("\n".join(log_lines))
    log(f"DONE wrote {out_dir}")
    src_env.close(); tgt_env.close()


if __name__ == "__main__":
    main()
