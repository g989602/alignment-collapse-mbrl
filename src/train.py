"""Training loop for the BW-Weighted main method.

Pipeline per minibatch:
  1. Forward → (z, dz_pred, r_pred, ds_pred, u = phi(z, a, dz_true)).
  2. D1: compute per-sample weights from per-domain BW score.
  3. D3: sample BW-geodesic MixUp synthetic Σ_λ.
  4. E1: update EMA prototypes; compute trust-region BW loss.
  5. Compose total loss; backprop.

The standard alignment baseline (CORAL on bures) is kept as a separate term
so we can ablate D1/D3/E1 cleanly via config flags.
Paper names: D1 = score filter (SF), D3 = geodesic MixUp (GM), E1 = Prototype Trust Region (TR).
"""
from __future__ import annotations
import numpy as np
import torch

from .data import TransitionBuffer
from .models import WorldModel
from .losses import (
    coral_bw_align,
    weighted_dyn_loss,
    weighted_reward_loss,
    weighted_state_loss,
    variance_floor_loss,
)
from .bw_weighted import (
    bw_score_filter,
    bw_geodesic_mixup,
    bw_mixup_alignment_loss,
    EmaPrototype,
)


def train_bw_weighted(model: WorldModel, buffer: TransitionBuffer, cfg) -> dict:
    """Train the BW-Weighted world model.

    cfg: dict-like with keys
        device, lr, epochs, batch_size, seed,
        w_dyn, w_reward, w_state, w_align, w_var,
        w_mixup,    # D3 weight
        w_ema,      # E1 weight
        beta,       # D1 score sharpness
        ema_tau,    # E1 EMA factor
        mixup_alpha, mixup_pairs,
        use_d1, use_d3, use_e1,    # ablation flags
    """
    device = cfg["device"]
    rng = np.random.default_rng(cfg["seed"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    history = []
    per_domain_size = max(8, cfg["batch_size"] // max(1, buffer.num_domains))

    # tr_anchor (anchor-family intervention): "perdomain" (default, unchanged path)
    #             | "global" | "shuffled" | "frozen" — see src/tr_anchor_variants.py
    tr_anchor = cfg.get("tr_anchor", "perdomain")
    if tr_anchor != "perdomain":
        from .tr_anchor_variants import ANCHOR_VARIANTS
        _ema_cls = ANCHOR_VARIANTS[tr_anchor]
    else:
        _ema_cls = EmaPrototype
    ema = _ema_cls(
        n_domains=buffer.num_domains,
        feat_dim=model.feat_dim,
        tau=cfg.get("ema_tau", 0.99),
        device=device,
        warmup_steps=cfg.get("ema_warmup", 20),
    )

    # ---- MAIN+: Polyak averaging on the last `polyak_frac` fraction of training ----
    use_polyak = bool(cfg.get("polyak", False))
    polyak_frac = float(cfg.get("polyak_frac", 0.25))
    polyak_snapshots = []
    polyak_start_epoch = int(cfg["epochs"] * (1 - polyak_frac)) + 1
    if use_polyak:
        print(f"[MAIN+] Polyak averaging from epoch {polyak_start_epoch} "
              f"(last {int(polyak_frac*100)}% of training)", flush=True)

    n_steps = max(1, len(buffer) // cfg["batch_size"])
    # TR penalty cadence — compute l_ema only every k-th step (EMA prototype
    # still updates every step); k=1 (default) is the unchanged path
    _tr_every = int(cfg.get("ema_penalty_every", 1))
    _gstep = 0
    # adaptive tau — single global controller targeting prototype drift
    # (default off). Constants fixed by pre-registration.
    _tau_ad = bool(cfg.get("tau_adaptive", False))
    _ad_win, _ad_frac = 50, 0.01
    _ad_prev = None      # prototype covariance snapshot
    _ad_target = None    # drift target = frac * initial inter-domain separation
    # auto-TR gate — start with TR off, enable permanently when minibatch
    # feature erank falls below 50% of its initial value (default off).
    _auto_tr = bool(cfg.get("auto_tr", False))
    if _auto_tr:
        cfg["use_e1"] = False
    _gate_init_erank = None
    _gate_fired_at = None

    def _mb_erank(u_):
        import torch as _t
        x = u_.detach()
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-5)
        c = (x.t() @ x) / max(1, x.shape[0] - 1)
        c = c + 1e-5 * _t.eye(c.shape[0], device=c.device)
        try:
            ev = _t.linalg.eigvalsh(c).clamp_min(0)
        except Exception:
            ev = _t.linalg.eigvalsh(c.cpu().double() + 1e-6 * _t.eye(c.shape[0], dtype=_t.float64)).clamp_min(0)
        p_ = ev / ev.sum(); p_ = p_[p_ > 1e-12]
        return float((-(p_ * p_.log()).sum()).exp().item())
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        meters = {k: [] for k in ("total", "dyn", "rew", "state", "align",
                                  "mixup", "ema", "var", "score_min", "score_mean")}
        for _ in range(n_steps):
            batch = buffer.sample_balanced(per_domain_size, rng)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["state"], batch["action"], batch["next_state"])
            u = out["u"]
            dom = batch["domain_id"]

            # ---- D1 — BW score filter ----
            if cfg.get("use_d1", True):
                weights, score_log = bw_score_filter(
                    u.detach(), dom, buffer.num_domains,
                    beta=cfg.get("beta", 0.5),
                )
            else:
                weights = torch.ones(u.shape[0], device=device, dtype=u.dtype)
                score_log = {}

            # ---- D3 — BW-Geodesic MixUp synthetic targets ----
            if cfg.get("use_d3", True):
                synth_covs, _, mixup_log = bw_geodesic_mixup(
                    u.detach(), dom, buffer.num_domains,
                    alpha=cfg.get("mixup_alpha", 0.2),
                    n_pairs=cfg.get("mixup_pairs", 2),
                )
                l_mixup = bw_mixup_alignment_loss(u, dom, buffer.num_domains, synth_covs)
            else:
                l_mixup = torch.zeros((), device=device)
                mixup_log = []

            # ---- auto-TR gate check (before E1 block) ----
            if _auto_tr and _gate_fired_at is None and _gstep % 50 == 0:
                er_now = _mb_erank(u)
                if _gate_init_erank is None:
                    _gate_init_erank = er_now
                elif er_now < 0.5 * _gate_init_erank:
                    cfg["use_e1"] = True
                    cfg["w_ema"] = 1.0
                    _gate_fired_at = _gstep
                    print(f"[auto_tr] gate fired at step {_gstep} "
                          f"(erank {er_now:.2f} < 0.5x init {_gate_init_erank:.2f})", flush=True)

            # ---- adaptive tau controller ----
            if _tau_ad and _gstep % _ad_win == 0 and _gstep > 0:
                import torch as _t
                with _t.no_grad():
                    protos = [c.clone() for c in ema._cov]
                    if _ad_target is None:
                        seps = []
                        for i in range(len(protos)):
                            for j in range(i + 1, len(protos)):
                                seps.append(float((protos[i] - protos[j]).norm().item()))
                        _ad_target = _ad_frac * (sum(seps) / max(1, len(seps)))
                        _ad_prev = protos
                    else:
                        drift = float(sum((a - b).norm().item()
                                      for a, b in zip(protos, _ad_prev)) / len(protos))
                        _ad_prev = protos
                        one_m = 1.0 - ema.tau
                        if drift > _ad_target:
                            one_m *= 0.5   # slow the prototype
                        elif drift < 0.25 * _ad_target:
                            one_m *= 2.0   # speed it up
                        ema.tau = float(min(0.99995, max(0.99, 1.0 - one_m)))

            # ---- E1 — EMA env-prototype trust region ----
            if cfg.get("use_e1", True):
                if _tr_every <= 1 or _gstep % _tr_every == 0:
                    l_ema = ema.trust_region_loss(u, dom)
                else:
                    l_ema = torch.zeros((), device=device)
                ema.update(u, dom)
            else:
                l_ema = torch.zeros((), device=device)
            _gstep += 1

            # ---- standard MBRL losses, weighted by D1 score ----
            l_dyn  = weighted_dyn_loss(out, weights)
            l_rew  = weighted_reward_loss(out["r_pred"], batch["reward"], weights)
            l_state = weighted_state_loss(out["ds_pred"], batch["next_state"] - batch["state"], weights)
            l_align = coral_bw_align(u, dom, buffer.num_domains)
            l_var  = variance_floor_loss(u) if cfg.get("w_var", 0) > 0 else torch.zeros((), device=device)

            total = (
                cfg["w_dyn"]    * l_dyn
                + cfg["w_reward"] * l_rew
                + cfg["w_state"]  * l_state
                + cfg["w_align"]  * l_align
                + cfg.get("w_mixup", 1.0) * l_mixup
                + cfg.get("w_ema",   1.0) * l_ema
                + cfg.get("w_var",   0.0) * l_var
            )
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            meters["total"].append(float(total.item()))
            meters["dyn"].append(float(l_dyn.item()))
            meters["rew"].append(float(l_rew.item()))
            meters["state"].append(float(l_state.item()))
            meters["align"].append(float(l_align.item()))
            meters["mixup"].append(float(l_mixup.item()))
            meters["ema"].append(float(l_ema.item()))
            meters["var"].append(float(l_var.item()))
            meters["score_min"].append(float(weights.min().item()))
            meters["score_mean"].append(float(weights.mean().item()))

        epoch_log = {"epoch": epoch}
        for k, v in meters.items():
            epoch_log[k] = float(np.mean(v)) if v else 0.0
        history.append(epoch_log)

        # ---- MAIN+: snapshot params for Polyak averaging ----
        if use_polyak and epoch >= polyak_start_epoch:
            polyak_snapshots.append(
                {k: v.detach().clone() for k, v in model.state_dict().items()
                 if v.dtype.is_floating_point}
            )

    # save the EMA prototype state on the model so eval can attach it
    model.ema_state = ema.state()

    # ---- MAIN+: replace model params with Polyak average ----
    if use_polyak and polyak_snapshots:
        avg_sd = {}
        keys = list(polyak_snapshots[0].keys())
        for k in keys:
            avg_sd[k] = torch.stack([s[k] for s in polyak_snapshots], dim=0).mean(dim=0)
        # Load only floating-point params (skip int buffers like ema._n_updates)
        cur_sd = model.state_dict()
        for k, v in avg_sd.items():
            cur_sd[k] = v
        model.load_state_dict(cur_sd, strict=False)
        print(f"[MAIN+] Polyak-averaged {len(polyak_snapshots)} param snapshots", flush=True)

    return {"history": history, "n_domains": buffer.num_domains,
            "auto_tr_fired_at": _gate_fired_at, "final_tau": getattr(ema, "tau", None),
            "polyak_n_snapshots": len(polyak_snapshots) if use_polyak else 0}
