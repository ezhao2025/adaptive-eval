"""Choose the discrimination prior (sigma_log_a) on a validation LM.

Fits 2PL on the non-held-out models minus the validation family, then scores adaptive vs.
random selection on the validation family only. The test family (holdout_models in the data
file) is never touched.

Usage:
  python scripts/sigma_sweep.py --data data/fluid_arc.json --val-prefix amber-7b/
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_eval import data as D  # noqa: E402
from adaptive_eval.irt import fit_2pl  # noqa: E402
from adaptive_eval.report import offline_curve  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val-prefix", required=True, help="model-name prefix of the validation family")
    p.add_argument("--sigmas", default="0.5,0.25,0.1,0.05,0.02")
    p.add_argument("--budgets", default="20,40,80")
    a = p.parse_args()

    d = D.load(a.data)
    idx = {m: i for i, m in enumerate(d["models"])}
    hold = set(d.get("holdout_models", []))
    train = [m for m in d["models"] if m not in hold]
    val = [m for m in train if m.startswith(a.val_prefix)]
    if not val:
        sys.exit(f"no training models start with '{a.val_prefix}'")
    fit_rows = [idx[m] for m in train if not m.startswith(a.val_prefix)]
    budgets = [int(x) for x in a.budgets.split(",")]

    print(f"fit on {len(fit_rows)} models, validate on {len(val)}  (cells: adaptive/random tau)")
    for s in (float(x) for x in a.sigmas.split(",")):
        bank, _ = fit_2pl(d["R"][fit_rows], d["items"], sigma_log_a=s)
        curve = offline_curve(d, bank, val, budgets)
        cells = "  ".join(f"{r['items']}:{r['tau_max_info']:+.2f}/{r['tau_random']:+.2f}" for r in curve)
        print(f"sigma={s:<5} a {bank.a.min():.2f}-{bank.a.max():.2f}  {cells}")


if __name__ == "__main__":
    main()
