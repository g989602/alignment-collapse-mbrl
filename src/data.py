"""Domain-keyed transition buffer with balanced sampling."""
from __future__ import annotations
import numpy as np
import torch


class TransitionBuffer:
    def __init__(self, per_domain):
        self.per_domain = per_domain
        self.domain_ids = sorted(per_domain.keys())

    @property
    def num_domains(self):
        return len(self.domain_ids)

    def __len__(self):
        return sum(self.per_domain[d]["state"].shape[0] for d in self.domain_ids)

    def sample_balanced(self, per_domain_size, rng):
        out = {k: [] for k in ("state", "action", "next_state", "reward", "done", "domain_id")}
        for d in self.domain_ids:
            arrs = self.per_domain[d]
            n = arrs["state"].shape[0]
            idx = rng.integers(0, n, size=per_domain_size)
            for k in out.keys():
                out[k].append(arrs[k][idx])
        merged = {}
        for k, parts in out.items():
            arr = np.concatenate(parts, axis=0)
            t = torch.as_tensor(arr)
            t = t.long() if k == "domain_id" else t.float()
            merged[k] = t
        return merged

    def all_for_domain(self, d):
        arrs = self.per_domain[d]
        return {
            "state": torch.as_tensor(arrs["state"]).float(),
            "action": torch.as_tensor(arrs["action"]).float(),
            "next_state": torch.as_tensor(arrs["next_state"]).float(),
            "reward": torch.as_tensor(arrs["reward"]).float(),
            "done": torch.as_tensor(arrs["done"]).float(),
            "domain_id": torch.as_tensor(arrs["domain_id"]).long(),
        }
