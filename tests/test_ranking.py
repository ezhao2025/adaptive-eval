import numpy as np
import pytest
from scipy.special import expit

from adaptive_eval.c.ranking import (Bank2D, choose, choose_full, discord, expected_discordant, gains,
                                     posterior, score, variance_after)


@pytest.fixture
def bank():
    rng = np.random.default_rng(0)
    n = 60
    return Bank2D([f"i{k}" for k in range(n)], np.exp(rng.normal(0, 0.3, n)),
                  rng.normal(0, 1, n), (np.arange(n) % 3 == 0).astype(int))


def answers(bank, theta, idx, seed=0):
    rng = np.random.default_rng(seed)
    P = expit(bank.loadings()[idx] @ theta - bank.b[idx])
    return {int(i): int(r < p) for i, r, p in zip(idx, rng.random(len(idx)), P)}


def test_weights_are_family_balanced(bank):
    w = bank.weights(0.5)
    assert np.isclose(w.sum(), 1)
    assert np.isclose(w[bank.dim == 0].sum(), 0.5)


def test_posterior_moves_toward_answers(bank):
    all_right = {i: 1 for i in range(0, 30)}
    all_wrong = {i: 0 for i in range(0, 30)}
    assert (posterior(bank, all_right)[0] > 0).all()
    assert (posterior(bank, all_wrong)[0] < 0).all()


def test_score_is_exact_when_everything_answered(bank):
    w = bank.weights()
    ans = answers(bank, np.array([0.3, -0.2]), np.arange(len(bank)))
    st = score(bank, w, ans)
    assert np.isclose(st.s, sum(w[i] * y for i, y in ans.items()))
    assert st.v < 1e-9


def test_variance_after_matches_brute_force(bank):
    """Rank-one update == recomputing the variance with the item removed and its info added."""
    w = bank.weights()
    ans = answers(bank, np.array([0.5, 0.1]), np.arange(0, 60, 4))
    st = score(bank, w, ans)
    fast = variance_after(bank, w, ans, st)
    L = bank.loadings()
    P = expit(L @ st.theta - bank.b)
    q = P * (1 - P)
    for i in [1, 2, 3, 5, 59]:
        cov = np.linalg.inv(np.linalg.inv(st.cov) + q[i] * np.outer(L[i], L[i]))
        un = np.ones(len(bank), bool)
        un[list(ans) + [i]] = False
        g = ((w * q)[un, None] * L[un]).sum(0)
        slow = g @ cov @ g + (w[un] ** 2 * q[un]).sum()
        assert np.isclose(fast[i], slow, rtol=1e-9)
    assert np.isinf(fast[list(ans)]).all()
    assert (fast[np.isfinite(fast)] <= st.v + 1e-15).all()


def test_gains_nonnegative_and_positive_for_tied_pair(bank):
    w = bank.weights()
    st = [score(bank, w, {}), score(bank, w, {})]          # identical: gap exactly 0
    g = gains(0, st, variance_after(bank, w, {}, st[0]))
    assert (g >= -1e-12).all() and g.max() > 0


def test_choose_prefers_the_uncertain_close_pair(bank):
    w = bank.weights()
    ans = [{i: 0 for i in range(0, 60, 2)},                          # clearly last, well measured
           answers(bank, np.array([1.0, 1.0]), np.arange(0, 10)),     # two close, few answers
           answers(bank, np.array([1.0, 1.0]), np.arange(10, 20), seed=1)]
    states = [score(bank, w, a) for a in ans]
    vn = [variance_after(bank, w, a, s) for a, s in zip(ans, states)]
    m, i, g = choose(states, vn)
    assert m in (1, 2) and i not in ans[m] and g > 0
    assert choose(states, vn) == (m, i, g)                  # deterministic


def test_expected_discordant_bounds():
    assert np.isclose(discord(np.array([0.0]), np.array([1.0]))[0], 0.5)


def test_fast_choose_matches_brute_force(bank):
    w = bank.weights()
    rng = np.random.default_rng(3)
    for trial in range(15):
        ans = [answers(bank, rng.normal(0, 1, 2), rng.choice(60, rng.integers(0, 40), replace=False),
                       seed=trial * 10 + k) for k in range(5)]
        states = [score(bank, w, a) for a in ans]
        vn = [variance_after(bank, w, a, s) for a, s in zip(ans, states)]
        fm, fi, fg = choose(states, vn, lookahead=(1,))
        bm, bi, bg = choose_full(states, vn)
        assert np.isclose(fg, bg, rtol=1e-9, atol=1e-15)
        assert (fm, fi) == (bm, bi) or np.isclose(fg, bg, rtol=1e-12, atol=1e-18)


def test_lookahead_keeps_gain_alive_when_one_answer_cannot_flip_a_pair(bank):
    """Two nearly-resolved models ~2 sd apart: one answer can't flip them, 32 can."""
    from adaptive_eval.c.ranking import ModelState
    th, cov = np.zeros(2), np.eye(2)
    states = [ModelState(0.50, 0.01 ** 2, th, cov), ModelState(0.52, 0.01 ** 2, th, cov)]
    v_new = [np.full(len(bank), 0.0099 ** 2), np.full(len(bank), 0.0099 ** 2)]
    one = choose(states, v_new, lookahead=(1,))[2]
    kg = choose(states, v_new)[2]
    assert kg > 1e6 * max(one, 1e-300) and kg > 1e-7
