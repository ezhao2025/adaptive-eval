"""Response-matrix data: synthetic generator, loader, held-out split.

Data file format (JSON):
  {"models": [...], "items": [...], "R": [[0/1/null, ...], ...],
   "model_provider": {model: provider_name}, "truth": {...optional...}}
Real data (e.g. per-item leaderboard results or your own benchmark runs) just
needs converting into this shape.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.special import expit

PROVIDER_NAMES = ["sim-openai", "sim-anthropic", "sim-google"]


def generate_synthetic(n_models: int, n_items: int, seed: int = 0,
                       noisy_frac: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_models)
    a = rng.lognormal(0, 0.35, n_items)
    b = rng.normal(0, 1.2, n_items)
    noisy = rng.random(n_items) < noisy_frac
    a[noisy] = 0.05                      # near-uninformative items (e.g. mislabeled)
    P = expit(a[None, :] * (theta[:, None] - b[None, :]))
    R = (rng.random(P.shape) < P).astype(int)
    models = [f"model-{i:03d}" for i in range(n_models)]
    return {
        "models": models,
        "items": [f"item-{j:04d}" for j in range(n_items)],
        "R": R.tolist(),
        "model_provider": {m: PROVIDER_NAMES[i % len(PROVIDER_NAMES)] for i, m in enumerate(models)},
        "truth": {"theta": theta.tolist(), "a": a.tolist(), "b": b.tolist()},
    }


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    d["R"] = np.array([[np.nan if v is None else v for v in row] for row in d["R"]], dtype=float)
    return d


def save(d: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(d, f)


def split_models(models: list[str], train_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """IRT is fit on train models only; adaptive eval runs on held-out test models.
    Evaluating on the models used to fit the item parameters would inflate results."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(models))
    k = int(round(train_frac * len(models)))
    return [models[i] for i in sorted(order[:k])], [models[i] for i in sorted(order[k:])]
