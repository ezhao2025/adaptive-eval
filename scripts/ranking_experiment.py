"""Step C3: does coupled allocation rank models with fewer calls than uncoupled methods?

    python scripts/ranking_experiment.py --matrix data/sp6_matrix.json --meta data/sp6_item_meta.json

Per split (scenes split 50/50 at random):
  * calibration half: fit the explanatory 2D + item-residual bank on every model's answers
  * pool half: rank the models whose pool is fully answered (SmolVLM is not), by replaying
    their logged answers
Truth = each model's actual family-balanced (50/50) accuracy on the pool: an empirical
number, not an IRT estimate. All methods use the same bank and the same score estimator;
they differ only in which (model, item) they ask next:
  random       round-robin over models, random unasked item
  independent  round-robin over models, the item that most shrinks that model's own
               score variance (Design A/B behaviour: each session minds itself)
  coupled      the (model, item) with the largest expected drop in discordant pairs
  coupled-1step   coupled with a one-step lookahead (the first version). Its gains hit 0
                  once no single answer can flip a pair, so later picks were tie-breaks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from explanatory_cv import fit, load                                  # noqa: E402

from adaptive_eval.c.ranking import Bank2D, choose, score, variance_after  # noqa: E402


def make_bank(D, calib_cols, pool_cols):
    obs = ~np.isnan(D["R"])
    mask = obs & calib_cols[None, :]
    r = fit(D, mask, "explanatory 2D + item residual")
    s = D["s_idx"][pool_cols]
    return Bank2D([D["items"][i] for i in np.where(pool_cols)[0]], r["a"][s],
                  r["b"][pool_cols], D["dim_of_subtask"][s])


def run(method, bank, Y, w, budgets, rng=None):
    n_m, n_i = Y.shape
    ans = [dict() for _ in range(n_m)]
    st = [score(bank, w, a) for a in ans]
    vn = [variance_after(bank, w, a, s) for a, s in zip(ans, st)]
    out, calls = {}, 0
    while calls < max(budgets):
        if method == "coupled":
            m, i, _ = choose(st, vn)
        elif method == "coupled-1step":
            m, i, _ = choose(st, vn, lookahead=(1,))
        else:
            m = calls % n_m
            free = np.where(np.isfinite(vn[m]))[0]
            if free.size == 0:
                calls += 1
                continue
            i = int(rng.choice(free)) if method == "random" else int(free[np.argmin(vn[m][free])])
        ans[m][i] = int(Y[m, i])
        st[m] = score(bank, w, ans[m])
        vn[m] = variance_after(bank, w, ans[m], st[m])
        calls += 1
        if calls in budgets:
            out[calls] = (np.array([s.s for s in st]), [len(a) for a in ans])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/sp6_matrix.json")
    p.add_argument("--meta", default="data/sp6_item_meta.json")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--random-draws", type=int, default=5)
    p.add_argument("--w-count", type=float, default=0.5)
    a = p.parse_args()

    D = load(a.matrix, a.meta)
    full = ~np.isnan(D["R"]).any(1)
    names = [m.split("/")[-1] for m, f in zip(D["models"], full) if f]
    n_m = int(full.sum())
    pairs = n_m * (n_m - 1) // 2
    scenes = np.unique(D["scene"])
    per_model = [5, 10, 20, 40, 80]
    budgets = [k * n_m for k in per_model]
    res = {m: {b: [] for b in budgets} for m in ("random", "independent", "coupled-1step", "coupled")}
    alloc = {b: [] for b in budgets}

    for sp in range(a.splits):
        rng = np.random.default_rng(sp)
        pool_scenes = set(rng.permutation(scenes)[: len(scenes) // 2])
        pool = np.array([s in pool_scenes for s in D["scene"]])
        bank = make_bank(D, ~pool, pool)
        w = bank.weights(a.w_count)
        Y = D["R"][full][:, pool]
        truth = Y @ w
        order = " > ".join(f"{names[i]}({truth[i]:.3f})" for i in np.argsort(-truth))
        print(f"split {sp}: {pool.sum()} pool items; truth {order}")
        for method in res:
            draws = a.random_draws if method == "random" else 1
            for dr in range(draws):
                out = run(method, bank, Y, w, budgets, np.random.default_rng(1000 * sp + dr))
                for b, (est, n_ans) in out.items():
                    tau = kendalltau(est, truth).statistic
                    res[method][b].append((tau, round((1 - tau) * pairs / 2)))
                    if method == "coupled":
                        alloc[b].append(n_ans)

    print(f"\nKendall tau vs true pool accuracy (w_count={a.w_count}), "
          f"{n_m} models, {pairs} pairs; mean over {a.splits} splits "
          f"(random: x{a.random_draws} draws)")
    print(f"{'calls/model':>11s} | " + " | ".join(f"{m:>22s}" for m in res))
    print(f"{'':>11s} | " + " | ".join(f"{'tau':>8s} {'sd':>5s} {'wrong':>6s}" for _ in res))
    for k, b in zip(per_model, budgets):
        row = []
        for m in res:
            t = np.array([x[0] for x in res[m][b]])
            wr = np.array([x[1] for x in res[m][b]])
            row.append(f"{t.mean():8.3f} {t.std():5.3f} {wr.mean():6.1f}")
        print(f"{k:>11d} | " + " | ".join(row))

    print("\npaired by split, coupled minus independent: mean tau diff, splits won/tied/lost")
    for k, b in zip(per_model, budgets):
        d = np.array([x[0] for x in res["coupled"][b]]) - \
            np.array([x[0] for x in res["independent"][b]])
        print(f"  {k:3d} calls/model: {d.mean():+.3f}  {(d > 1e-9).sum()}/{(abs(d) <= 1e-9).sum()}"
              f"/{(d < -1e-9).sum()}")

    print("\ncoupled: calls per model at the largest budget (mean over splits)")
    A = np.array(alloc[budgets[-1]]).mean(0)
    for n, c in zip(names, A):
        print(f"  {n:34s} {c:6.1f}")


if __name__ == "__main__":
    main()
