"""Step C6: place new models on an existing leaderboard with a few calls each.

    python scripts/placement_experiment.py --matrix data/sp6_matrix.json \
        --new-matrix data/sp6_new_matrix.json --meta data/sp6_item_meta.json

Leaderboard: the models in --matrix that answered every item (the anchors). Their answers
are known, so their scores are exact and they cost nothing. The item bank is fit on
--matrix only: the new models never touch calibration.

New models (--new-matrix, run on RunPod afterwards, all 746 items each) start with no
answers. Each method spends k calls per new model on average, replaying logged answers:
  random       round-robin over new models, random unasked item
  independent  round-robin over new models, the item that most shrinks that model's own
               score variance
  coupled      the (model, item) that most reduces expected misordered pairs (choose())
Truth = every model's actual 50/50 family-balanced accuracy on all 746 items.
Metrics cover only pairs involving a new model (anchor-anchor pairs are exact by design).

One deterministic run on one item pool is a single draw, so --replicates repeats everything
on random 80% subsets of scenes (truth recomputed on each subset). Every method sees the
same subsets, and the summary compares them paired.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from explanatory_cv import fit, load                                  # noqa: E402

from adaptive_eval.c.ranking import Bank2D, choose, score, variance_after  # noqa: E402


def place(method, bank, w, Y_anchor, Y_new, budgets, rng=None):
    n_a, n_n = len(Y_anchor), len(Y_new)
    ans = [{i: int(y) for i, y in enumerate(row)} for row in Y_anchor] + [{} for _ in Y_new]
    st = [score(bank, w, a) for a in ans]
    vn = [variance_after(bank, w, a, s) for a, s in zip(ans, st)]
    out, calls = {}, 0
    while calls < max(budgets):
        if method == "coupled":
            m, i, _ = choose(st, vn)
        else:
            m = n_a + calls % n_n
            free = np.where(np.isfinite(vn[m]))[0]
            i = int(rng.choice(free)) if method == "random" else int(free[np.argmin(vn[m][free])])
        ans[m][i] = int(Y_new[m - n_a, i])
        st[m] = score(bank, w, ans[m])
        vn[m] = variance_after(bank, w, ans[m], st[m])
        calls += 1
        if calls in budgets:
            out[calls] = (np.array([s.s for s in st]), [len(ans[k]) for k in range(n_a, n_a + n_n)])
    return out


def metrics(est, truth, n_a):
    n = len(truth)
    pairs = [(j, k) for j in range(n) for k in range(j + 1, n) if k >= n_a]   # involve a new model
    right = sum(np.sign(est[j] - est[k]) == np.sign(truth[j] - truth[k]) for j, k in pairs)
    rank_est = (-est).argsort().argsort()
    rank_true = (-truth).argsort().argsort()
    rank_err = np.abs(rank_est[n_a:] - rank_true[n_a:]).mean()
    return right / len(pairs), rank_err, kendalltau(est, truth).statistic, len(pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/sp6_matrix.json")
    p.add_argument("--new-matrix", default="data/sp6_new_matrix.json")
    p.add_argument("--meta", default="data/sp6_item_meta.json")
    p.add_argument("--random-draws", type=int, default=5)
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument("--subset", type=float, default=0.8, help="fraction of scenes per replicate")
    p.add_argument("--w-count", type=float, default=0.5)
    p.add_argument("--out", default="", help="also write the table here")
    a = p.parse_args()

    D = load(a.matrix, a.meta)
    new = json.load(open(a.new_matrix))
    assert new["items"] == D["items"], "new models must be run on the same items"
    r = fit(D, ~np.isnan(D["R"]), "explanatory 2D + item residual")   # old models only
    s = D["s_idx"]
    bank = Bank2D(D["items"], r["a"][s], r["b"], D["dim_of_subtask"][s])
    w = bank.weights(a.w_count)

    full = ~np.isnan(D["R"]).any(1)
    anchors = [m for m, f in zip(D["models"], full) if f]
    Y_anchor = D["R"][full].astype(int)
    Y_new = np.array(new["R"], dtype=float)
    assert not np.isnan(Y_new).any(), "new models need every item answered (truth)"
    Y_new = Y_new.astype(int)
    names = anchors + new["models"]
    truth = np.concatenate([Y_anchor @ w, Y_new @ w])
    n_a, n_n = len(anchors), len(new["models"])

    lines = [f"{n_a} anchors, {n_n} new models, {len(D['items'])} items; truth = 50/50 accuracy"
             f" on all items", "", "true leaderboard (* = new):"]
    for rank, k in enumerate(np.argsort(-truth), 1):
        lines.append(f"  {rank:2d}. {'*' if k >= n_a else ' '} {names[k].split('/')[-1]:36s}"
                     f" {truth[k]:.3f}")

    per_model = [5, 10, 20, 40, 80]
    budgets = [k * n_n for k in per_model]
    methods = ("random", "independent", "coupled")
    res = {m: {b: [] for b in budgets} for m in methods}       # per replicate: mean over draws
    alloc = []
    scenes = np.unique(D["scene"])
    for rep in range(a.replicates):
        rng = np.random.default_rng(rep)
        keep_sc = set(rng.choice(scenes, int(round(a.subset * len(scenes))), replace=False))
        keep = np.array([sc in keep_sc for sc in D["scene"]])
        sub = Bank2D([it for it, k in zip(bank.item_ids, keep) if k], bank.a[keep],
                     bank.b[keep], bank.dim[keep])
        ws = sub.weights(a.w_count)
        truth_r = np.concatenate([Y_anchor[:, keep] @ ws, Y_new[:, keep] @ ws])
        for m in methods:
            draws = []
            for dr in range(a.random_draws if m == "random" else 1):
                out = place(m, sub, ws, Y_anchor[:, keep], Y_new[:, keep], budgets,
                            np.random.default_rng(1000 * rep + dr))
                draws.append(out)
            for b in budgets:
                res[m][b].append(np.mean([metrics(o[b][0], truth_r, n_a)[:2] for o in draws], 0))
                if m == "coupled" and b == budgets[-1]:
                    alloc.append(draws[0][b][1])
    n_pairs = n_a * n_n + n_n * (n_n - 1) // 2
    lines += ["", f"{a.replicates} replicates on {a.subset:.0%} scene subsets; pairs involving a new"
              f" model ordered correctly (of {n_pairs}), mean (sd) over replicates;"
              f" random averages {a.random_draws} draws per replicate",
              f"{'calls/new model':>15s} | " + " | ".join(f"{m:>13s}" for m in methods)
              + " | coupled-indep  coupled-random  (won/tied/lost vs indep)"]
    for k, b in zip(per_model, budgets):
        c = {m: np.array([x[0] for x in res[m][b]]) * n_pairs for m in methods}
        d1, d2 = c["coupled"] - c["independent"], c["coupled"] - c["random"]
        lines.append(f"{k:>15d} | " + " | ".join(f"{c[m].mean():5.1f} ({c[m].std():3.1f})"
                                                   for m in methods)
                     + f" | {d1.mean():+6.2f}        {d2.mean():+6.2f}"
                       f"          {(d1 > 0).sum()}/{(d1 == 0).sum()}/{(d1 < 0).sum()}")
    lines += ["", "mean |rank error| of new models (lower is better)",
              f"{'calls/new model':>15s} | " + " | ".join(f"{m:>11s}" for m in methods)]
    for k, b in zip(per_model, budgets):
        lines.append(f"{k:>15d} | " + " | ".join(
            f"{np.mean([x[1] for x in res[m][b]]):11.2f}" for m in methods))
    lines += ["", f"coupled: mean calls per new model at {per_model[-1]} calls/model on average"]
    for nm, c in zip(new["models"], np.mean(alloc, 0)):
        lines.append(f"  {nm.split('/')[-1]:36s} {c:6.1f}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
