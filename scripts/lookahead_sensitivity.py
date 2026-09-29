"""Was the lookahead set in choose() tuned? Rerun the scaling simulation with other sets.

    python scripts/lookahead_sensitivity.py --sizes 16,32 --reps 10

Same simulated leaderboards as ranking_scaling.py (same seeds): models drawn from the
fitted sp6 population, answers from the full sp6 fit, allocator bank = feature-only
difficulties. Only the lookahead set passed to choose() changes:
  1-step   (1,)                        the original rule (gains die once no single answer
                                        can flip a pair)
  short    (1, 2, 4, 8)
  default  (1, 2, 4, 8, 16, 32, 64)    what ranking.py uses, fixed before any of these runs
  long     (1, 2, 4, ..., 256)
If default is not clearly better than short and long, it was not tuned into a sweet spot.
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

from adaptive_eval.c.ranking import Bank2D, choose, score, variance_after  # noqa: E402

SETS = {"1-step": (1,), "short": (1, 2, 4, 8), "default": (1, 2, 4, 8, 16, 32, 64),
        "long": tuple(2 ** k for k in range(9))}


def run(bank, Y, w, budgets, lookahead):
    n_m = len(Y)
    ans = [dict() for _ in range(n_m)]
    st = [score(bank, w, a) for a in ans]
    vn = [variance_after(bank, w, a, s) for a, s in zip(ans, st)]
    out, calls = {}, 0
    while calls < max(budgets):
        m, i, _ = choose(st, vn, lookahead=lookahead)
        ans[m][i] = int(Y[m, i])
        st[m] = score(bank, w, ans[m])
        vn[m] = variance_after(bank, w, ans[m], st[m])
        calls += 1
        if calls in budgets:
            out[calls] = np.array([s.s for s in st])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/sp6_matrix.json")
    p.add_argument("--meta", default="data/sp6_item_meta.json")
    p.add_argument("--sizes", default="16,32")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--dump", default="", help="append raw rows as JSON lines")
    a = p.parse_args()

    D = load(a.matrix, a.meta)
    r = fit(D, ~np.isnan(D["R"]), "explanatory 2D + item residual")
    a_i = r["a"][D["s_idx"]]
    dim = D["dim_of_subtask"][D["s_idx"]]
    true_bank = Bank2D(D["items"], a_i, r["b"], dim)
    model_bank = Bank2D(D["items"], a_i, D["X"] @ r["beta"], dim)
    w = model_bank.weights(0.5)
    mu, cov = r["theta"].mean(0), np.cov(r["theta"].T)
    L = true_bank.loadings()
    per_model = [10, 20, 40, 80]
    names = a.sets.split(",")
    for n in map(int, a.sizes.split(",")):
        budgets = [k * n for k in per_model]
        res = {s: {b: [] for b in budgets} for s in names}
        for rep in range(a.reps):
            rng = np.random.default_rng(10_000 * n + rep)       # same as ranking_scaling.py
            theta = rng.multivariate_normal(mu, cov, n)
            Y = (rng.random((n, len(true_bank))) < expit(theta @ L.T - true_bank.b)).astype(int)
            truth = Y @ w
            for sname in names:
                for b, est in run(model_bank, Y, w, budgets, SETS[sname]).items():
                    t = kendalltau(est, truth).statistic
                    res[sname][b].append(t)
                    if a.dump:
                        with open(a.dump, "a") as f:
                            f.write(json.dumps({"n": n, "rep": rep, "set": sname,
                                                "calls_per_model": b // n, "tau": t}) + "\n")
        print(f"\n{n} models, {a.reps} leaderboards: Kendall tau, mean (sd)")
        print(f"  {'calls/model':>11s} " + " ".join(f"{s:>14s}" for s in names))
        for k, b in zip(per_model, budgets):
            print(f"  {k:>11d} " + " ".join(
                f"{np.mean(res[s][b]):8.3f} ({np.std(res[s][b]):.3f})" for s in names))
        if "default" in names:
            for s in names:
                if s == "default":
                    continue
                d = [np.array(res["default"][b]) - np.array(res[s][b]) for b in budgets]
                print(f"  default - {s:8s}: " + "  ".join(
                    f"{k}:{x.mean():+.3f}({(x > 1e-9).sum()}/{(abs(x) <= 1e-9).sum()}/"
                    f"{(x < -1e-9).sum()})" for k, x in zip(per_model, d)))


if __name__ == "__main__":
    main()
