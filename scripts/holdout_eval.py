"""Adaptive vs. random (multi-seed) ranking agreement on held-out checkpoints.

Kendall tau between theta estimated from a fixed item budget and theta from all items.
Adaptive selection comes from report.offline_curve (deterministic); random selection is
repeated over --seeds seeds and reported as mean +/- sd.

Usage:
  python scripts/holdout_eval.py --data data/fluid_arc.json \
      --params data/fluid_arc_params.json --prefix olmo2-7b/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_eval import data as D  # noqa: E402
from adaptive_eval.irt import ItemBank, estimate_ability  # noqa: E402
from adaptive_eval.report import offline_curve  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--prefix", required=True, help="model-name prefix of the test family, e.g. olmo2-7b/")
    p.add_argument("--budgets", default="20,40,80,120,160")
    p.add_argument("--seeds", type=int, default=20)
    a = p.parse_args()

    d = D.load(a.data)
    bank, _ = ItemBank.load(a.params)
    test = [m for m in d["models"] if m.startswith(a.prefix)]
    if not test:
        sys.exit(f"no models start with '{a.prefix}'")
    idx = {m: i for i, m in enumerate(d["models"])}
    R = d["R"][[idx[m] for m in test]]
    full = np.array([estimate_ability(bank.a, bank.b, r)[0] for r in R])
    ks = [int(x) for x in a.budgets.split(",")]
    adaptive = {r["items"]: r["tau_max_info"] for r in offline_curve(d, bank, test, ks)}

    print(f"{len(test)} test checkpoints")
    for k in ks:
        taus = []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            est = []
            for r in R:
                j = rng.choice(R.shape[1], k, replace=False)
                est.append(estimate_ability(bank.a[j], bank.b[j], r[j])[0])
            taus.append(kendalltau(est, full)[0])
        print(f"k={k:3d}  adaptive {adaptive[k]:.3f}   random {np.mean(taus):.3f} ± {np.std(taus):.3f}"
              f"  (min {min(taus):.2f}, max {max(taus):.2f})")


if __name__ == "__main__":
    main()
