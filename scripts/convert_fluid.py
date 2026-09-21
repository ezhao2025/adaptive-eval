"""Convert Fluid Benchmarking per-checkpoint results into adaptive_eval's data JSON.

Each (LM, checkpoint) pair becomes one "model" row. Cells with no result become null.
Checkpoints of the LMs named in --holdout are listed in "holdout_models" so IRT is fit
on the other LMs only (checkpoints of the same LM are highly correlated, so a random
split would leak).

Usage:
  python scripts/convert_fluid.py --benchmark arc_challenge --out data/fluid_arc.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_eval.data import PROVIDER_NAMES, save  # noqa: E402


def read_lm(path: str, prefix: str) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Return (checkpoint names, {checkpoint: {item_id: 0/1}}) for one LM's CSV."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        ckpts = next(reader)[1:]
        results: dict[str, dict[str, int]] = {c: {} for c in ckpts}
        for row in reader:
            if not row or not row[0].startswith(prefix):
                continue
            item = row[0]
            for c, v in zip(ckpts, row[1:]):
                v = v.strip()
                if v and v.lower() != "nan":
                    results[c][item] = int(float(v))
    return ckpts, results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, help="e.g. arc_challenge, mmlu, gsm8k")
    p.add_argument("--src", default="data/fluid/data")
    p.add_argument("--out", required=True)
    p.add_argument("--holdout", default="olmo2-7b",
                   help="comma-separated LMs whose checkpoints are held out of IRT fitting")
    a = p.parse_args()

    prefix = a.benchmark + "_"
    holdout_lms = {x.strip() for x in a.holdout.split(",") if x.strip()}
    models: list[str] = []
    rows: dict[str, dict[str, int]] = {}
    holdout: list[str] = []

    for path in sorted(glob.glob(os.path.join(a.src, "lm_eval_results", "*.csv"))):
        lm = os.path.splitext(os.path.basename(path))[0]
        ckpts, results = read_lm(path, prefix)
        for c in ckpts:
            if not results[c]:
                continue  # checkpoint has no results on this benchmark
            m = f"{lm}/{c}"
            models.append(m)
            rows[m] = results[c]
            if lm in holdout_lms:
                holdout.append(m)

    if not models:
        sys.exit(f"No results for prefix '{prefix}' under {a.src}/lm_eval_results")
    missing = holdout_lms - {m.split("/")[0] for m in models}
    if missing:
        sys.exit(f"--holdout LMs not found: {sorted(missing)}")

    items = sorted({i for r in rows.values() for i in r})
    R = [[rows[m].get(i) for i in items] for m in models]
    n_null = sum(v is None for row in R for v in row)

    save({
        "models": models,
        "items": items,
        "R": R,
        "model_provider": {m: PROVIDER_NAMES[k % len(PROVIDER_NAMES)] for k, m in enumerate(models)},
        "holdout_models": holdout,
        "source": f"allenai/fluid-benchmarking lm_eval_results ({a.benchmark})",
    }, a.out)
    print(f"{len(models)} models ({len(holdout)} held out) x {len(items)} items -> {a.out}")
    print(f"missing cells: {n_null} ({n_null / (len(models) * len(items)):.2%})")


if __name__ == "__main__":
    main()
