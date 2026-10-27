"""Entry point — one full BW-Weighted main-method run on one (env, seed).

Usage:
    python scripts/run_main.py --config configs/walker2d.json --seed 7

Writes:
    <out_root>/MAIN_<env>_seed<S>/
        metrics.json     full per-epoch losses + final eval
        model.pt         trained world-model state_dict (+ ema_state)
        log.txt          terminal log
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

# Allow `python scripts/run_main.py` from project root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data import TransitionBuffer
from src.eval import evaluate
from src.models import WorldModel
from src.mujoco_env import (
    DEFAULT_SOURCE_SHIFTS,
    DEFAULT_TARGET_SHIFTS,
    HARD_TARGET_SHIFTS,
    ODRL_SOURCE_SHIFTS,
    ODRL_TARGET_SHIFTS,
    ENV_DEFAULTS,
    make_buffers,
)
from src.train import train_bw_weighted
from src.utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_root", default="checkpoints")
    p.add_argument("--prefix", default="MAIN",
                   help="output dir prefix, e.g. MAIN, BASELINE, ORACLE_TARGET")
    p.add_argument("--device", default=None,
                   help="cuda / cpu / cuda:0 ... (overrides config)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--oracle", default="none",
                   choices=["none", "target", "joint"],
                   help="oracle data-collection mode. "
                        "target: random-policy collection from target shifts. "
                        "joint: random collection from source ∪ target. "
                        "none: standard source-only random collection.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config for the unit-test smoke path")
    p.add_argument("--shift_set", default="default",
                   choices=["default", "hard", "odrl"],
                   help="target shift set: default, hard, or odrl (Lyu et al. 2024 Walker2d shifts)")
    p.add_argument("--transitions", type=int, default=None,
                   help="override transitions_per_domain (for size sweep)")
    p.add_argument("--ema_tau", type=float, default=None,
                   help="override EMA decay (default 0.99)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg["seed"] = int(args.seed)

    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        cfg["epochs"] = 1
        cfg["transitions_per_domain"] = 200
        cfg["batch_size"] = 32

    env_id = cfg["env_id"]
    env_short = env_id.split("-")[0].lower()
    # When --shift_set hard is requested, fold "HARD_" into the prefix so the
    # output dir doesn't clash with default-shift runs.
    if args.shift_set == "default":
        eff_prefix = args.prefix
    elif args.shift_set == "odrl":
        eff_prefix = f"{args.prefix}_ODRL"
    else:
        eff_prefix = f"{args.prefix}_HARD"
    out_dir = Path(args.out_root) / f"{eff_prefix}_{env_short}_seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg["seed"])
    log_path = out_dir / "log.txt"
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"env={env_id} seed={cfg['seed']} device={cfg['device']} out={out_dir}")
    defaults = ENV_DEFAULTS.get(env_id, {"max_steps": 200, "transitions_per_domain": 4000})
    if args.transitions is not None:
        cfg["transitions_per_domain"] = int(args.transitions)
    if args.ema_tau is not None: cfg["ema_tau"] = float(args.ema_tau)
    transitions = cfg.get("transitions_per_domain", defaults["transitions_per_domain"])
    max_steps = cfg.get("max_steps", defaults["max_steps"])
    if args.shift_set == "hard":
        target_shifts = HARD_TARGET_SHIFTS
        default_source = DEFAULT_SOURCE_SHIFTS
    elif args.shift_set == "odrl":
        target_shifts = ODRL_TARGET_SHIFTS
        default_source = ODRL_SOURCE_SHIFTS
    else:
        target_shifts = DEFAULT_TARGET_SHIFTS
        default_source = DEFAULT_SOURCE_SHIFTS

    if args.oracle == "target":
        collect_shifts = target_shifts
        log(f"ORACLE-TARGET (random): collecting from {len(collect_shifts)} target shifts")
    elif args.oracle == "joint":
        collect_shifts = default_source + target_shifts
        log(f"ORACLE-JOINT (random): collecting from {len(collect_shifts)} (source+target) shifts")
    else:
        collect_shifts = default_source

    log(f"collecting buffers: {transitions} transitions per domain")
    src_env, tgt_env, buffers = make_buffers(
        env_id, collect_shifts, target_shifts,
        transitions_per_domain=transitions,
        seed=cfg["seed"], domain_seed=cfg.get("domain_seed", 42),
        max_steps=max_steps,
    )
    buffer = TransitionBuffer(buffers)
    log(f"buffers: {buffer.num_domains} domains, {len(buffer)} total transitions")

    state_dim = src_env.state_dim
    action_dim = src_env.action_dim
    latent_dim = cfg.get("latent_dim", 32)
    feat_dim = cfg.get("feat_dim", 16)
    if cfg.get("adaptive_dim", False):
        new_latent = max(latent_dim, 2 * state_dim)
        new_feat   = max(feat_dim, state_dim)
        if (new_latent, new_feat) != (latent_dim, feat_dim):
            log(f"adaptive_dim: latent_dim {latent_dim}→{new_latent}, "
                f"feat_dim {feat_dim}→{new_feat}")
            latent_dim, feat_dim = new_latent, new_feat
    model = WorldModel(
        state_dim=state_dim, action_dim=action_dim,
        latent_dim=latent_dim,
        feat_dim=feat_dim,
        hidden=cfg.get("hidden", 128),
        n_domains=buffer.num_domains,
    ).to(cfg["device"])
    log(f"model: state={state_dim} action={action_dim} latent={cfg.get('latent_dim',32)} feat={cfg.get('feat_dim',16)}")

    log("starting train_bw_weighted ...")
    t0 = time.time()
    train_info = train_bw_weighted(model, buffer, cfg)
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

    metrics = {
        "config":  cfg,
        "history": train_info["history"],
        "source":  src_metrics,
        "target":  tgt_metrics,
        "train_time_s": train_time,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({
        "state_dict": model.state_dict(),
        "ema_state":  getattr(model, "ema_state", None),
        "config":     cfg,
    }, out_dir / "model.pt")
    log_path.write_text("\n".join(log_lines))
    log(f"DONE. wrote {out_dir}/metrics.json + model.pt")
    src_env.close(); tgt_env.close()


if __name__ == "__main__":
    main()
