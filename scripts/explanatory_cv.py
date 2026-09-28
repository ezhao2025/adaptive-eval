"""Step C1: can generator features stand in for per-item parameters when there are only 8 models?

    python scripts/explanatory_cv.py --matrix data/sp6_matrix.json --meta data/sp6_item_meta.json

Model (MAP, all variants share one code path):
    logit P(m answers i) = sum_d a_{g(i),d} * theta_{m,d} - b_i
    b_i = x_i . beta + eps_i,   eps_i ~ N(0, tau^2)

    g(i) is the item's discrimination group (the item itself, or its subtask), and
    confirmatory loadings put each subtask on exactly one dimension.

Two held-out tests, 5 folds each, same splits for every variant:
  cells:  hold out 20% of observed (model, item) cells  -> fills in a sparse matrix
  scenes: hold out 20% of scenes, every model's answers -> new generated items, cold start
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

COUNT_DIM = {"count_visible", "count_total", "count_hidden", "tallest_column", "count_ground",
             "count_above_red", "support_on_ground"}          # dim 0: counting / occlusion
# everything else (relation_*, triple_*) -> dim 1: relations


def load(matrix_path, meta_path):
    d = json.load(open(matrix_path))
    meta = json.load(open(meta_path))
    R = np.array([[np.nan if v is None else v for v in row] for row in d["R"]], dtype=float)
    items = d["items"]
    subtasks = sorted({meta[i]["subtask"] for i in items})
    s_idx = np.array([subtasks.index(meta[i]["subtask"]) for i in items])
    tiers = sorted({(meta[i]["scene"]["nx"], meta[i]["scene"]["max_h"]) for i in items})
    t_idx = np.array([tiers.index((meta[i]["scene"]["nx"], meta[i]["scene"]["max_h"]))
                      for i in items])
    hf = np.array([meta[i]["scene"]["hidden_frac"] for i in items])
    hf = (hf - hf.mean()) / hf.std()
    ncubes = np.log(np.array([np.sum(meta[i]["scene"]["heights"]) for i in items]))
    ncubes = (ncubes - ncubes.mean()) / ncubes.std()
    is_count = np.array([meta[i]["subtask"] in COUNT_DIM for i in items], dtype=float)
    scene = np.array([i.rsplit("-", 1)[0] for i in items])
    # design matrix: subtask one-hot (acts as intercepts), tier offsets, occlusion and size,
    # and their interaction with counting (occlusion should only hurt counting)
    X = np.column_stack([np.eye(len(subtasks))[s_idx], np.eye(len(tiers))[t_idx][:, 1:],
                         hf, ncubes, hf * is_count, ncubes * is_count])
    dim_of_subtask = np.array([0 if s in COUNT_DIM else 1 for s in subtasks])
    return dict(R=R, models=d["models"], items=items, subtasks=subtasks, s_idx=s_idx,
                X=X, scene=scene, dim_of_subtask=dim_of_subtask)


VARIANTS = {
    # name: (difficulty source, discrimination group, n_dims)
    "2PL per item (repo, sigma_log_a=0.05)": ("item", "item", 1),
    "Rasch-ish, b per subtask only":       ("subtask", "subtask", 1),
    "explanatory 1D":                       ("features", "subtask", 1),
    "explanatory 1D + item residual":       ("features+resid", "subtask", 1),
    "explanatory 2D (confirmatory)":        ("features", "subtask", 2),
    "explanatory 2D + item residual":       ("features+resid", "subtask", 2),
}


def fit(D, mask, variant, tau=0.5, iters=300):
    diff_src, group, n_dims = VARIANTS[variant]
    R = D["R"]
    n_m, n_i = R.shape
    Y = torch.tensor(np.nan_to_num(R))
    M = torch.tensor(mask.astype(float))
    X = torch.tensor(D["X"])
    s_idx = torch.tensor(D["s_idx"])
    n_sub = len(D["subtasks"])

    theta = torch.zeros(n_m, n_dims, requires_grad=True)
    n_groups = n_i if group == "item" else n_sub
    la = torch.zeros(n_groups, requires_grad=True)
    beta = torch.zeros(X.shape[1], requires_grad=True)
    b_item = torch.zeros(n_i, requires_grad=True)
    params = [theta, la, beta, b_item]
    sig_la = 0.05 if group == "item" else 0.5

    if n_dims == 1:
        load_mask = torch.ones(n_i, 1)
    else:
        dim = torch.tensor(D["dim_of_subtask"])[s_idx]
        load_mask = torch.nn.functional.one_hot(dim, 2).double()

    def parts():
        a = torch.exp(la)[torch.arange(n_i) if group == "item" else s_idx]
        if diff_src == "item":
            b = b_item
        elif diff_src == "subtask":
            b = X[:, :n_sub] @ beta[:n_sub]
        elif diff_src == "features":
            b = X @ beta
        else:
            b = X @ beta + b_item
        z = (theta @ (a[:, None] * load_mask).T) - b[None, :]
        return z, b

    def loss():
        z, _ = parts()
        nll = -(M * (Y * torch.nn.functional.logsigmoid(z)
                     + (1 - Y) * torch.nn.functional.logsigmoid(-z))).sum()
        prior = 0.5 * (theta ** 2).sum() + 0.5 * (la ** 2).sum() / sig_la ** 2 \
            + 0.5 * (beta ** 2).sum() / 2.0 ** 2
        if diff_src == "item":
            prior = prior + 0.5 * (b_item ** 2).sum() / 2.0 ** 2
        elif diff_src == "features+resid":
            prior = prior + 0.5 * (b_item ** 2).sum() / tau ** 2
        return nll + prior

    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        l = loss()
        l.backward()
        return l
    opt.step(closure)
    with torch.no_grad():
        z, b = parts()
    return dict(z=z.detach().numpy(), theta=theta.detach().numpy(), a=torch.exp(la).detach().numpy(),
                b=b.detach().numpy(), beta=beta.detach().numpy(), loss_fn=loss, params=params)


def logloss(R, z, cells):
    y = R[cells]
    p = 1 / (1 + np.exp(-z[cells]))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), \
        float(((p > 0.5) == (y == 1)).mean())


def cv(D, split, k=5, seed=0):
    R = D["R"]
    obs = ~np.isnan(R)
    rng = np.random.default_rng(seed)
    if split == "cells":
        fold = np.where(obs, rng.integers(0, k, R.shape), -1)
    else:
        scenes = np.unique(D["scene"])
        sf = dict(zip(scenes, rng.permutation(len(scenes)) % k))
        fold = np.where(obs, np.array([sf[s] for s in D["scene"]])[None, :], -1)
    out = {}
    for v in VARIANTS:
        ll, acc, n = 0.0, 0.0, 0
        for f in range(k):
            test = fold == f
            r = fit(D, obs & ~test, v)
            l, a = logloss(R, r["z"], test)
            ll += l * test.sum()
            acc += a * test.sum()
            n += test.sum()
        out[v] = (ll / n, acc / n)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/sp6_matrix.json")
    p.add_argument("--meta", default="data/sp6_item_meta.json")
    a = p.parse_args()
    D = load(a.matrix, a.meta)
    obs = ~np.isnan(D["R"])
    base = np.nanmean(D["R"])
    print(f"{len(D['models'])} models x {len(D['items'])} items, {obs.sum()} observed cells, "
          f"{len(D['subtasks'])} subtasks, {D['X'].shape[1]} difficulty features")
    print(f"constant-p baseline log-loss: "
          f"{-(base * np.log(base) + (1 - base) * np.log(1 - base)):.4f}\n")
    for split in ("cells", "scenes"):
        res = cv(D, split)
        print(f"held-out {split} (5-fold): log-loss / accuracy")
        for v, (ll, acc) in res.items():
            print(f"  {v:40s} {ll:.4f}  {acc:.3f}")
        print()


if __name__ == "__main__":
    main()
