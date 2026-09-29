"""Summarize ranking_scaling.py --dump output (merges runs split across processes).

    python scripts/scaling_summary.py results/c_scaling_raw.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    tau = defaultdict(dict)                 # (n, k, method) -> {rep: tau}
    gap = defaultdict(dict)
    for r in rows:
        tau[(r["n"], r["calls_per_model"], r["method"])][r["rep"]] = r["tau"]
        gap[r["n"]][r["rep"]] = r["gap"]
    for n in sorted(gap):
        reps = sorted(gap[n])
        print(f"{n} models, {len(reps)} leaderboards, median neighbour gap "
              f"{np.mean(list(gap[n].values())):.4f}")
        print(f"  {'calls/model':>11s} {'random':>8s} {'indep':>8s} {'coupled':>8s}"
              f"   coupled-indep  [95% CI]   won/tied/lost")
        for k in sorted({k for (nn, k, _) in tau if nn == n}):
            t = {m: np.array([tau[(n, k, m)][r] for r in reps])
                 for m in ("random", "independent", "coupled")}
            d = t["coupled"] - t["independent"]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {k:>11d} {t['random'].mean():8.3f} {t['independent'].mean():8.3f} "
                  f"{t['coupled'].mean():8.3f}   {d.mean():+.3f} [{d.mean() - 1.96 * se:+.3f}, "
                  f"{d.mean() + 1.96 * se:+.3f}]   {(d > 1e-9).sum()}/{(abs(d) <= 1e-9).sum()}"
                  f"/{(d < -1e-9).sum()}")
        print()


if __name__ == "__main__":
    main()
