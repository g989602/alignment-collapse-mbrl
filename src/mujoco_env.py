"""Parametric MuJoCo wrapper. Trimmed from full/src/tcla/mujoco_env.py.

Each domain is a gym MuJoCo env whose underlying MjModel has been modified to
apply a (mass / friction / damping) shift. Multiple domains live side-by-side.
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np


warnings.filterwarnings("ignore", message=".*v4.*out of date.*")


@dataclass
class ShiftSpec:
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    damping_scale: float = 1.0
    gravity_scale: float = 1.0


DEFAULT_SOURCE_SHIFTS = [
    ShiftSpec("src_nominal"),
    ShiftSpec("src_mass_light", mass_scale=0.85),
    ShiftSpec("src_friction_high", friction_scale=1.30),
    ShiftSpec("src_damping_high", damping_scale=1.30),
]

DEFAULT_TARGET_SHIFTS = [
    ShiftSpec("tgt_heavy_lowfric", mass_scale=1.50, friction_scale=0.70),
    ShiftSpec("tgt_light_highdamp", mass_scale=0.50, damping_scale=1.50),
    ShiftSpec("tgt_lowfric_lowdamp", friction_scale=0.50, damping_scale=0.50),
]

# HARD target shifts: more extreme than DEFAULT, to create harder transfer
# tasks especially on robust envs like Ant where the default shifts are easy.
HARD_TARGET_SHIFTS = [
    ShiftSpec("tgt_hvy_lowf",   mass_scale=2.00, friction_scale=0.30),
    ShiftSpec("tgt_lt_hidmp",   mass_scale=0.30, damping_scale=2.00),
    ShiftSpec("tgt_lowf_lowdmp", friction_scale=0.20, damping_scale=0.30),
]

# ODRL-compatible shift sets for Walker2d (matching ODRL's standard friction +
# gravity perturbation magnitudes from Lyu et al. 2024). Source: 4 friction
# perturbations across {0.5, 0.8, 1.2, 2.0}; targets: {friction-0.5, friction-2.0,
# gravity-0.5, gravity-2.0}.
ODRL_SOURCE_SHIFTS = [
    ShiftSpec("odrl_src_fr0.5", friction_scale=0.5),
    ShiftSpec("odrl_src_fr0.8", friction_scale=0.8),
    ShiftSpec("odrl_src_fr1.2", friction_scale=1.2),
    ShiftSpec("odrl_src_fr2.0", friction_scale=2.0),
]
ODRL_TARGET_SHIFTS = [
    ShiftSpec("odrl_tgt_fr0.5", friction_scale=0.5),
    ShiftSpec("odrl_tgt_fr2.0", friction_scale=2.0),
    ShiftSpec("odrl_tgt_gv0.5", gravity_scale=0.5),
    ShiftSpec("odrl_tgt_gv2.0", gravity_scale=2.0),
]


def _apply_shift(env, shift):
    model = env.unwrapped.model
    if shift.mass_scale != 1.0:
        model.body_mass[:] = model.body_mass[:] * float(shift.mass_scale)
    if shift.friction_scale != 1.0:
        model.geom_friction[:, 0] = model.geom_friction[:, 0] * float(shift.friction_scale)
    if shift.damping_scale != 1.0:
        model.dof_damping[:] = model.dof_damping[:] * float(shift.damping_scale)
    if shift.gravity_scale != 1.0:
        # MuJoCo gravity is opt.gravity (xyz vector); scale all components.
        model.opt.gravity[:] = model.opt.gravity[:] * float(shift.gravity_scale)


class MultiDomainMuJoCoEnv:
    def __init__(self, base_env_id, shifts, max_steps=200, seed=0):
        import gymnasium as gym
        self.base_env_id = base_env_id
        self.shifts = list(shifts)
        self.max_steps = int(max_steps)
        self._envs = []
        for i, shift in enumerate(self.shifts):
            env = gym.make(base_env_id)
            env.reset(seed=seed + i)
            _apply_shift(env, shift)
            self._envs.append(env)
        first = self._envs[0]
        self.state_dim = int(first.observation_space.shape[0])
        self.action_dim = int(first.action_space.shape[0])
        self.action_low = float(np.min(first.action_space.low))
        self.action_high = float(np.max(first.action_space.high))
        self._domain_idx = 0
        self._t = 0

    @property
    def num_domains(self):
        return len(self._envs)

    def set_domain(self, idx):
        assert 0 <= idx < self.num_domains
        self._domain_idx = idx

    def reset(self, domain_idx=None):
        if domain_idx is not None:
            self.set_domain(domain_idx)
        s, _ = self._envs[self._domain_idx].reset()
        self._t = 0
        return np.asarray(s, dtype=np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32),
                    self.action_low, self.action_high)
        s2, r, term, trunc, info = self._envs[self._domain_idx].step(a)
        self._t += 1
        done = bool(term) or bool(trunc) or self._t >= self.max_steps
        return np.asarray(s2, dtype=np.float32), float(r), done, {"domain": self._domain_idx}

    def close(self):
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass


def make_buffers(base_env_id, source_shifts, target_shifts,
                 transitions_per_domain, seed, domain_seed=42, max_steps=200):
    src_env = MultiDomainMuJoCoEnv(base_env_id, source_shifts,
                                   max_steps=max_steps, seed=domain_seed)
    tgt_env = MultiDomainMuJoCoEnv(base_env_id, target_shifts,
                                   max_steps=max_steps, seed=domain_seed + 991)

    rng = np.random.default_rng(seed)
    buffers = {}
    for d_idx in range(src_env.num_domains):
        states, actions, next_states, rewards, dones = [], [], [], [], []
        s = src_env.reset(domain_idx=d_idx)
        n_collected = 0
        while n_collected < transitions_per_domain:
            a = rng.uniform(src_env.action_low, src_env.action_high,
                            size=src_env.action_dim).astype(np.float32)
            s2, r, done, _ = src_env.step(a)
            states.append(s); actions.append(a); next_states.append(s2)
            rewards.append(r); dones.append(float(done))
            n_collected += 1
            s = src_env.reset(domain_idx=d_idx) if done else s2
        buffers[d_idx] = {
            "state": np.stack(states).astype(np.float32),
            "action": np.stack(actions).astype(np.float32),
            "next_state": np.stack(next_states).astype(np.float32),
            "reward": np.asarray(rewards, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.float32),
            "domain_id": np.full(len(states), d_idx, dtype=np.int64),
        }
    return src_env, tgt_env, buffers


ENV_DEFAULTS = {
    "HalfCheetah-v4": {"max_steps": 200, "transitions_per_domain": 4000},
    "Hopper-v4":      {"max_steps": 200, "transitions_per_domain": 4000},
    "Walker2d-v4":    {"max_steps": 200, "transitions_per_domain": 4000},
    "Ant-v4":         {"max_steps": 200, "transitions_per_domain": 4000},
}
