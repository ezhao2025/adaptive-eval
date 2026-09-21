"""2PL item response theory: fit item parameters, estimate ability, pick items.

Model: P(correct | theta, a, b) = sigmoid(a * (theta - b))
  theta = model ability, a = item discrimination (>0), b = item difficulty.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class ItemBank:
    item_ids: list[str]
    a: np.ndarray
    b: np.ndarray

    def __len__(self) -> int:
        return len(self.item_ids)

    def index(self) -> dict[str, int]:
        return {iid: i for i, iid in enumerate(self.item_ids)}

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {"item_ids": self.item_ids, "a": self.a.tolist(), "b": self.b.tolist()}
        payload.update(extra or {})
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> tuple["ItemBank", dict]:
        with open(path) as f:
            d = json.load(f)
        bank = cls(d["item_ids"], np.array(d["a"]), np.array(d["b"]))
        return bank, d


def fit_2pl(R: np.ndarray, item_ids: list[str], sigma_log_a: float = 0.5,
            sigma_b: float = 2.0, maxiter: int = 1000) -> tuple[ItemBank, np.ndarray]:
    """Joint MAP fit. R is (models x items) with 1/0, np.nan = not observed.

    Priors: theta ~ N(0,1) (fixes the scale), log a ~ N(0, sigma_log_a^2), b ~ N(0, sigma_b^2).
    """
    mask = ~np.isnan(R)
    Y = np.where(mask, R, 0.0)
    n_m, n_i = R.shape

    # Initialise from logits of observed accuracy.
    p_item = np.clip(np.nanmean(R, axis=0), 0.02, 0.98)
    p_model = np.clip(np.nanmean(R, axis=1), 0.02, 0.98)
    t0 = np.log(p_model / (1 - p_model))
    t0 = (t0 - t0.mean()) / (t0.std() + 1e-9)
    x0 = np.concatenate([t0, np.zeros(n_i), -np.log(p_item / (1 - p_item))])

    def objective(x):
        th, la, b = x[:n_m], x[n_m:n_m + n_i], x[n_m + n_i:]
        a = np.exp(la)
        z = a[None, :] * (th[:, None] - b[None, :])
        ll = -Y * np.logaddexp(0, -z) - (1 - Y) * np.logaddexp(0, z)
        nll = -(ll * mask).sum()
        prior = 0.5 * (th ** 2).sum() + 0.5 * (la ** 2).sum() / sigma_log_a ** 2 \
            + 0.5 * (b ** 2).sum() / sigma_b ** 2
        G = -(Y - expit(z)) * mask                      # d nll / d z
        g_th = (G * a[None, :]).sum(1) + th
        g_la = (G * (th[:, None] - b[None, :])).sum(0) * a + la / sigma_log_a ** 2
        g_b = -(G * a[None, :]).sum(0) + b / sigma_b ** 2
        return nll + prior, np.concatenate([g_th, g_la, g_b])

    res = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    th, la, b = res.x[:n_m], res.x[n_m:n_m + n_i], res.x[n_m + n_i:]
    return ItemBank(list(item_ids), np.exp(la), b), th


def estimate_ability(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                     prior_sd: float = 1.0, iters: int = 50) -> tuple[float, float]:
    """MAP ability from answered items via Newton's method. Returns (theta, standard error)."""
    th = 0.0
    for _ in range(iters):
        P = expit(a * (th - b))
        grad = (a * (y - P)).sum() - th / prior_sd ** 2
        hess = -(a * a * P * (1 - P)).sum() - 1 / prior_sd ** 2
        step = grad / hess
        th = float(np.clip(th - step, -6, 6))
        if abs(step) < 1e-7:
            break
    P = expit(a * (th - b))
    info = (a * a * P * (1 - P)).sum() + 1 / prior_sd ** 2
    return th, float(1 / np.sqrt(info))


def fisher_information(theta: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    P = expit(a * (theta - b))
    return a * a * P * (1 - P)


def select_next(selector: str, theta: float, bank: ItemBank, used: set[int],
                session_id: str, step: int) -> int:
    """Deterministic given (session, step, answers so far) -- required for crash replay."""
    available = np.array([i for i in range(len(bank)) if i not in used])
    if selector == "max_info":
        info = fisher_information(theta, bank.a[available], bank.b[available])
        return int(available[np.argmax(info)])  # argmax breaks ties by lowest index
    if selector == "random":
        rng = np.random.default_rng(zlib.crc32(f"{session_id}:{step}".encode()))
        return int(rng.choice(available))
    raise ValueError(f"unknown selector {selector}")
