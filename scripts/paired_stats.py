"""Paired comparison of two run families on matched seeds (the statistic used for the paper's paired contrasts).

For each seed present in both families, d = J_A - J_B with J = target worst-domain return (metrics.json:
target.worst). Reports IQM of the per-seed paired differences, a two-sided 95% percentile bootstrap interval
(10,000 seed resamples, numpy default_rng(0)), the one-sided bootstrap p_b = Pr[IQM* <= 0], and wins = #{d > 0}.
With several --pair arguments it also reports Holm-adjusted p-values over those pairs.

Usage:
    python scripts/paired_stats.py --root checkpoints --pair VGDFTR_HARD_walker2d VGDF_HARD_walker2d
    python scripts/paired_stats.py --pair TRONLY_walker2d BASELINE_walker2d --pair MAIN_walker2d BASELINE_walker2d
A family is the run-directory prefix before "_seed<S>".
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean


def iqm(x):
    return float(trim_mean(np.asarray(x, dtype=float), 0.25))


def load(root: Path, family: str) -> dict:
    out = {}
    for d in sorted(root.glob(f"{family}_seed*")):
        f = d / "metrics.json"
        if f.exists():
            out[int(d.name.rsplit("seed", 1)[1])] = float(json.loads(f.read_text())["target"]["worst"])
    return out


def paired_boot(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    boots = np.array([iqm(d[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return dict(n=n, iqm_diff=iqm(d), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                p_b=float((boots <= 0).mean()), wins=int((d > 0).sum()))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("FAMILY_A", "FAMILY_B"), required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for fa, fb in args.pair:
        a, b = load(root, fa), load(root, fb)
        seeds = sorted(set(a) & set(b))
        if len(seeds) < 2:
            print(f"{fa} vs {fb}: fewer than 2 matched seeds, skipped"); continue
        r = paired_boot([a[s] for s in seeds], [b[s] for s in seeds], n_boot=args.n_boot)
        r.update(a=fa, b=fb, iqm_a=iqm([a[s] for s in seeds]), iqm_b=iqm([b[s] for s in seeds]))
        rows.append(r)
    adj = holm([max(r["p_b"], 1.0 / args.n_boot) for r in rows]) if len(rows) > 1 else [None] * len(rows)
    print(f"{'A > B':48s} {'n':>3s} {'IQM A':>8s} {'IQM B':>8s} {'IQM(d)':>8s} {'95% CI':>18s} {'p_b':>7s} {'Holm':>6s} wins")
    for r, h in zip(rows, adj):
        hs = f"{h:6.3f}" if h is not None else "     -"
        print(f"{r['a'] + ' > ' + r['b']:48s} {r['n']:3d} {r['iqm_a']:8.1f} {r['iqm_b']:8.1f} {r['iqm_diff']:+8.1f} "
              f"[{r['lo']:+7.1f},{r['hi']:+7.1f}] {r['p_b']:7.4f} {hs} {r['wins']}/{r['n']}")


if __name__ == "__main__":
    main()
