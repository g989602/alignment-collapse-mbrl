"""Per-family aggregator. Reads checkpoints/<PREFIX>_<env>_seed<S>/metrics.json for every seed and
writes a mean / std / IQM summary of source and target returns.

Usage:
    python scripts/aggregate.py --env walker2d --prefix TRONLY --root checkpoints
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--root", default="checkpoints")
    p.add_argument("--prefix", default="MAIN",
                   help="run-dir prefix, e.g. MAIN or BASELINE")
    args = p.parse_args()

    root = Path(args.root)
    runs = sorted(root.glob(f"{args.prefix}_{args.env}_seed*"))
    if not runs:
        raise SystemExit(f"no runs found under {root}/{args.prefix}_{args.env}_seed*")

    rows = []
    for r in runs:
        mfile = r / "metrics.json"
        if not mfile.exists():
            print(f"  skip {r} (no metrics.json)")
            continue
        m = json.loads(mfile.read_text())
        seed = int(r.name.rsplit("seed", 1)[1])
        rows.append({
            "seed":       seed,
            "src_mean":   m["source"]["mean"],
            "src_worst":  m["source"]["worst"],
            "tgt_mean":   m["target"]["mean"],
            "tgt_worst":  m["target"]["worst"],
            "train_time": m.get("train_time_s", -1),
        })

    if not rows:
        raise SystemExit("no usable runs")

    out_dir = root / f"{args.prefix}_{args.env}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def stats(key):
        vs = [r[key] for r in rows]
        return float(np.mean(vs)), float(np.std(vs)), float(np.min(vs)), float(np.max(vs))

    md = []
    md.append(f"# {args.prefix} — {args.env} ({len(rows)} seeds)\n")
    md.append("| seed | src_mean | src_worst | tgt_mean | tgt_worst | train_s |")
    md.append("|------|----------|-----------|----------|-----------|---------|")
    for r in rows:
        md.append(f"| {r['seed']} | {r['src_mean']:.2f} | {r['src_worst']:.2f} | "
                  f"{r['tgt_mean']:.2f} | {r['tgt_worst']:.2f} | {r['train_time']:.0f} |")
    md.append("")
    md.append("| stat | src_mean | src_worst | tgt_mean | tgt_worst |")
    md.append("|------|----------|-----------|----------|-----------|")
    for stat_name, fn in [("mean", np.mean), ("std", np.std),
                          ("min",  np.min),  ("max",  np.max)]:
        sm = fn([r["src_mean"]  for r in rows])
        sw = fn([r["src_worst"] for r in rows])
        tm = fn([r["tgt_mean"]  for r in rows])
        tw = fn([r["tgt_worst"] for r in rows])
        md.append(f"| {stat_name} | {sm:.2f} | {sw:.2f} | {tm:.2f} | {tw:.2f} |")

    md_text = "\n".join(md)
    out_md = out_dir / "summary.md"
    out_md.write_text(md_text)
    out_dir.joinpath("rows.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_md}")
    print(md_text)


if __name__ == "__main__":
    main()
