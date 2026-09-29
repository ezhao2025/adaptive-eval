"""Design C: adaptive ranking across models with a 2D confirmatory item bank.

Ranking target (per model m), a family-balanced expected accuracy over a fixed item pool:
    s_m = sum_i w_i * y_mi        w_i = w_count / n_count  or  (1 - w_count) / n_rel

s_m is estimated with observed answers where a model has answered and IRT predictions where
it has not, so the estimate equals the true pool score once every item is answered:
    s_hat_m = sum_{answered} w_i y_mi + sum_{unanswered} w_i P_mi
    Var     = g' Sigma_m g + sum_{unanswered} w_i^2 P_mi (1 - P_mi)
where Sigma_m is the Laplace posterior covariance of theta_m and g = d s_hat / d theta
(delta method). The second term is the Bernoulli noise in answers not yet seen.

Objective: the expected number of discordant pairs (= (1 - Kendall tau) * C(n,2) / 2),
    E = sum_{j<k} Phi(-|s_j - s_k| / sqrt(v_j + v_k)).
Gain of asking model m item i = the expected drop in E, holding other models fixed. The
lookahead is preposterior: the answer shrinks v_m and also moves s_m, by a random amount
whose variance equals the variance it removes. Averaging over that move is what lets a
tied pair (|d| ~ 0) still show a gain; a variance-only lookahead scores ties as zero.

Everything here is deterministic given the answers, which Design B's crash replay and
speculation rely on (argmax ties break by lowest index).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm

# Gauss-Hermite nodes for E[f(d + tau * Z)], Z ~ N(0, 1)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(24)
_GH_W = _GH_W / _GH_W.sum()


@dataclass
class Bank2D:
    """Calibrated items. Item i loads on one dimension: logit = a_i * theta[dim_i] - b_i."""
    item_ids: list[str]
    a: np.ndarray            # (n,) discrimination
    b: np.ndarray            # (n,) difficulty (already includes the a scaling; logit offset)
    dim: np.ndarray          # (n,) 0 = counting/occlusion, 1 = relations
    n_dims: int = 2

    def __len__(self) -> int:
        return len(self.item_ids)

    def loadings(self) -> np.ndarray:
        L = np.zeros((len(self), self.n_dims))
        L[np.arange(len(self)), self.dim] = self.a
        return L                                        # logit = L @ theta - b

    def weights(self, w_count: float = 0.5) -> np.ndarray:
        is_c = self.dim == 0
        w = np.where(is_c, w_count / max(is_c.sum(), 1), (1 - w_count) / max((~is_c).sum(), 1))
        return w


def posterior(bank: Bank2D, answered: dict[int, int], iters: int = 50):
    """MAP theta and Laplace covariance under theta ~ N(0, I). Returns (theta, cov)."""
    D = bank.n_dims
    th = np.zeros(D)
    if answered:
        idx = np.fromiter(answered.keys(), int)
        y = np.fromiter(answered.values(), float)
        L = bank.loadings()[idx]
        b = bank.b[idx]
    for _ in range(iters):
        grad, H = -th, -np.eye(D)
        if answered:
            P = expit(L @ th - b)
            grad = grad + L.T @ (y - P)
            H = H - (L * (P * (1 - P))[:, None]).T @ L
        step = np.linalg.solve(H, grad)
        th = np.clip(th - step, -6, 6)
        if np.abs(step).max() < 1e-8:
            break
    info = np.eye(D)
    if answered:
        P = expit(L @ th - b)
        info = info + (L * (P * (1 - P))[:, None]).T @ L
    return th, np.linalg.inv(info)


@dataclass
class ModelState:
    s: float                 # estimated pool score
    v: float                 # its variance
    theta: np.ndarray
    cov: np.ndarray


def score(bank: Bank2D, w: np.ndarray, answered: dict[int, int]) -> ModelState:
    th, cov = posterior(bank, answered)
    L = bank.loadings()
    P = expit(L @ th - bank.b)
    un = np.ones(len(bank), bool)
    obs = 0.0
    for i, y in answered.items():
        un[i] = False
        obs += w[i] * y
    s = obs + (w[un] * P[un]).sum()
    g = ((w * P * (1 - P))[un, None] * L[un]).sum(0)
    v = float(g @ cov @ g + (w[un] ** 2 * P[un] * (1 - P[un])).sum())
    return ModelState(float(s), max(v, 1e-12), th, cov)


def variance_after(bank: Bank2D, w: np.ndarray, answered: dict[int, int],
                   st: ModelState) -> np.ndarray:
    """Var(s_hat) after answering each item (inf for items already answered).

    Rank-one (Sherman-Morrison) update of the covariance at the current theta, and the item
    leaves the unanswered set.
    """
    L = bank.loadings()
    P = expit(L @ st.theta - bank.b)
    q = P * (1 - P)
    un = np.ones(len(bank), bool)
    un[list(answered)] = False
    g = ((w * q)[un, None] * L[un]).sum(0)                  # current gradient
    bern = (w[un] ** 2 * q[un]).sum()
    # per candidate i: g_i = g - w_i q_i L_i ; cov_i = cov - c cov L_i L_i' cov / (1 + c L_i' cov L_i)
    c = q                                                   # item info = q_i L_i L_i'
    Cl = L @ st.cov                                         # (n, D) rows = (cov L_i)'
    lCl = (Cl * L).sum(1)                                   # L_i' cov L_i
    gi = g[None, :] - (w * q)[:, None] * L                  # (n, D)
    gCg = np.einsum("nd,de,ne->n", gi, st.cov, gi)
    gCl = (gi * Cl).sum(1)
    shrink = c * gCl ** 2 / (1 + c * lCl)
    v_new = gCg - shrink + (bern - w ** 2 * q)
    v_new = np.maximum(v_new, 1e-12)
    v_new[~un] = np.inf
    return v_new


def discord(d: np.ndarray, var: np.ndarray) -> np.ndarray:
    """P(the pair is ordered wrongly) given estimated gap d and its variance."""
    return norm.cdf(-np.abs(d) / np.sqrt(var))


def expected_discordant(states: list[ModelState]) -> float:
    s = np.array([x.s for x in states])
    v = np.array([x.v for x in states])
    j, k = np.triu_indices(len(states), 1)
    return float(discord(s[j] - s[k], v[j] + v[k]).sum())


def gains(m: int, states: list[ModelState], v_new: np.ndarray) -> np.ndarray:
    """Expected drop in discordant pairs from asking model m each item (preposterior)."""
    me = states[m]
    others = [k for k in range(len(states)) if k != m]
    d = np.array([me.s - states[k].s for k in others])            # (K,)
    vk = np.array([states[k].v for k in others])                  # (K,)
    now = discord(d, me.v + vk).sum()
    fin = np.isfinite(v_new)
    vn = np.minimum(v_new[fin], me.v)                             # (n',)
    tau = np.sqrt(np.maximum(me.v - vn, 0.0))                     # sd of the mean's move
    # d' = d + tau Z ;  E Phi(-|d'| / sqrt(vn + vk))
    dd = d[None, :, None] + tau[:, None, None] * _GH_X[None, None, :]
    sd = np.sqrt(vn[:, None, None] + vk[None, :, None])
    after = (norm.cdf(-np.abs(dd) / sd) * _GH_W).sum(-1).sum(-1)  # (n',)
    out = np.full(len(v_new), -np.inf)
    out[fin] = now - after
    return out


LOOKAHEAD = (1, 2, 4, 8, 16, 32, 64)


def choose(states: list[ModelState], banks_v_new: list[np.ndarray],
           cost: np.ndarray | None = None,
           lookahead: tuple[int, ...] = LOOKAHEAD) -> tuple[int, int, float]:
    """Coupled allocation: the (model, item) with the largest gain per unit cost.

    Which item: a model's gain depends on the item only through v_new and falls as v_new
    falls, so each model's best item is its argmin v_new. Coupling decides *which model*.
    That makes this O(models^2) instead of O(models^2 x items).

    Which model (KG*): one answer moves a score by ~1/sqrt(n) of its range, so once the
    gap in a pair is several times that, a one-step lookahead sees no chance of a flip and
    scores the pair ~0, even when it is still 20% likely to be misordered. Every model
    then ties at 0 and allocation degenerates to the tie-break. So score each model by
    max_r gain(r answers) / r, approximating r answers as r times this item's information:
        1 / v_r = 1 / v + r * (1 / v_new - 1 / v).
    lookahead=(1,) is the plain one-step rule; choose_full() is its brute-force oracle.

    Returns (model, item, gain per call). Deterministic: ties go to the lowest model/item.
    """
    best = (-1, -1, -np.inf)
    r = np.asarray(lookahead, float)
    for m, v_new in enumerate(banks_v_new):
        i = int(np.argmin(v_new))
        if not np.isfinite(v_new[i]):
            continue                                      # nothing left to ask this model
        v = states[m].v
        n_left = int(np.isfinite(v_new).sum())
        rr = r[r <= n_left]
        dinfo = max(1 / v_new[i] - 1 / v, 0.0)
        v_r = 1 / (1 / v + rr * dinfo)
        g = float((gains(m, states, v_r) / rr).max())
        if cost is not None:
            g = g / cost[m]
        if g > best[2]:
            best = (m, i, g)
    return best


def choose_full(states: list[ModelState], banks_v_new: list[np.ndarray],
                cost: np.ndarray | None = None) -> tuple[int, int, float]:
    """Brute force over every (model, item); same answer as choose(), much slower."""
    best = (-1, -1, -np.inf)
    for m, v_new in enumerate(banks_v_new):
        g = gains(m, states, v_new)
        if cost is not None:
            g = g / cost[m]
        i = int(np.argmax(g))
        if g[i] > best[2]:
            best = (m, i, float(g[i]))
    return best

