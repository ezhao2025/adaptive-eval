"""Step C4: does coupled allocation's edge hold or grow with more models?

    python scripts/ranking_scaling.py --matrix data/sp6_matrix.json --meta data/sp6_item_meta.json

Simulated leaderboards built from the real sp6 fit:
  * true items: the explanatory 2D + item-residual fit on all 746 items and all 8 models
  * the allocator's bank: the same fit's feature-only difficulties (no residual), so it is
    misspecified the way a bank for freshly generated items would be
  * models: theta ~ N(mean, cov) of the 8 fitted 2D abilities; answers ~ Bernoulli
Truth = each simulated model's actual 50/50 family-balanced accuracy on all 746 items.
With more models, neighbours are closer, so ranking gets harder at a fixed calls/model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from explanatory_cv import fit, load                                  # noqa: E402
from ranking_experiment import run                                    # noqa: E402

from adaptive_eval.c.ranking import Bank2D                            # noqa: E402

METHODS = ("random", "independent", "coupled")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/sp6_matrix.json")
    p.add_argument("--meta", default="data/sp6_item_meta.json")
    p.add_argument("--sizes", default="8,16,32,64")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--rep-start", type=int, default=0, help="to split one size across processes")
    p.add_argument("--w-count", type=float, default=0.5)
    p.add_argument("--dump", default="", help="append raw per-leaderboard tau as JSON lines")
    a = p.parse_args()

    D = load(a.matrix, a.meta)
    r = fit(D, ~np.isnan(D["R"]), "explanatory 2D + item residual")
    a_i = r["a"][D["s_idx"]]
    dim = D["dim_of_subtask"][D["s_idx"]]
    true_bank = Bank2D(D["items"], a_i, r["b"], dim)
    model_bank = Bank2D(D["items"], a_i, D["X"] @ r["beta"], dim)
    w = model_bank.weights(a.w_count)
    mu, cov = r["theta"].mean(0), np.cov(r["theta"].T)
    print(f"theta population: mean {np.round(mu, 2)}, sd {np.round(np.sqrt(np.diag(cov)), 2)}, "
          f"corr {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.2f}")
    print(f"allocator bank vs truth: difficulty corr "
          f"{np.corrcoef(model_bank.b, true_bank.b)[0, 1]:.3f}\n")

    per_model = [5, 10, 20, 40, 80]
    L = true_bank.loadings()
    for n in map(int, a.sizes.split(",")):
        budgets = [k * n for k in per_model]
        res = {m: {b: [] for b in budgets} for m in METHODS}
        gaps = []
        for rep in range(a.rep_start, a.rep_start + a.reps):
            rng = np.random.default_rng(10_000 * n + rep)
            theta = rng.multivariate_normal(mu, cov, n)
            Y = (rng.random((n, len(true_bank))) < expit(theta @ L.T - true_bank.b)).astype(int)
            truth = Y @ w
            gaps.append(np.median(np.diff(np.sort(truth))))
            for m in METHODS:
                out = run(m, model_bank, Y, w, budgets, np.random.default_rng(rep))
                for b, (est, _) in out.items():
                    res[m][b].append(kendalltau(est, truth).statistic)
                    if a.dump:
                        with open(a.dump, "a") as f:
                            f.write(json.dumps({"n": n, "rep": rep, "method": m,
                                                "calls_per_model": b // n,
                                                "tau": res[m][b][-1],
                                                "gap": float(gaps[-1])}) + "\n")
        print(f"{n} models ({n * (n - 1) // 2} pairs), median gap between neighbours "
              f"{np.mean(gaps):.4f}; Kendall tau, mean (sd) over {a.reps} leaderboards")
        print(f"  {'calls/model':>11s} " + " ".join(f"{m:>14s}" for m in METHODS)
              + "   coupled-indep (won/tied/lost)")
        for k, b in zip(per_model, budgets):
            t = {m: np.array(res[m][b]) for m in METHODS}
            d = t["coupled"] - t["independent"]
            print(f"  {k:>11d} " + " ".join(f"{t[m].mean():8.3f} ({t[m].std():.3f})"
                                            for m in METHODS)
                  + f"   {d.mean():+.3f} ({(d > 1e-9).sum()}/{(abs(d) <= 1e-9).sum()}"
                    f"/{(d < -1e-9).sum()})")
        print()


if __name__ == "__main__":
    main()
