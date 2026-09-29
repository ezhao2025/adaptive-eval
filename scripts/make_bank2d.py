"""Export the explanatory 2D item bank for adaptive_eval.c.scheduler.

    python scripts/make_bank2d.py --matrix data/sp6_matrix.json --meta data/sp6_item_meta.json \
        --out data/sp6_bank2d.json

Fits the explanatory 2D + item-residual model on every observed answer and saves each item's
discrimination, difficulty and dimension. --features-only drops the per-item residuals:
use it for items no model has answered yet (their residual would be 0 anyway).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from explanatory_cv import fit, load                                  # noqa: E402

from adaptive_eval.c.ranking import Bank2D                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/sp6_matrix.json")
    p.add_argument("--meta", default="data/sp6_item_meta.json")
    p.add_argument("--out", default="data/sp6_bank2d.json")
    p.add_argument("--features-only", action="store_true")
    a = p.parse_args()
    D = load(a.matrix, a.meta)
    r = fit(D, ~np.isnan(D["R"]), "explanatory 2D + item residual")
    s = D["s_idx"]
    b = D["X"] @ r["beta"] if a.features_only else r["b"]
    bank = Bank2D(D["items"], r["a"][s], b, D["dim_of_subtask"][s])
    bank.save(a.out, {"source": a.matrix, "features_only": a.features_only,
                      "subtasks": D["subtasks"], "dim_of_subtask": D["dim_of_subtask"].tolist()})
    print(f"{len(bank)} items -> {a.out} ({(bank.dim == 0).sum()} counting, "
          f"{(bank.dim == 1).sum()} relations)")


if __name__ == "__main__":
    main()
