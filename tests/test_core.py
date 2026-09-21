import asyncio
import time

import numpy as np
import pytest

from adaptive_eval.cli import build_context
from adaptive_eval.data import generate_synthetic, split_models
from adaptive_eval.engine import SessionConfig, SimulatedCrash, run_many
from adaptive_eval.irt import estimate_ability, fit_2pl
from adaptive_eval.ratelimit import TokenBucket
from adaptive_eval.storage import EventStore, connect


@pytest.fixture(scope="module")
def setup():
    d = generate_synthetic(40, 150, seed=1)
    d["R"] = np.array(d["R"], dtype=float)
    train, test = split_models(d["models"], 0.7, seed=1)
    bank, _ = fit_2pl(d["R"][[d["models"].index(m) for m in train]], d["items"])
    return d, bank, test


def test_irt_recovers_difficulty(setup):
    d, bank, _ = setup
    assert np.corrcoef(bank.b, d["truth"]["b"])[0, 1] > 0.85


def test_ability_estimate_moves_with_answers():
    a, b = np.ones(10), np.zeros(10)
    hi, _ = estimate_ability(a, b, np.ones(10))
    lo, _ = estimate_ability(a, b, np.zeros(10))
    assert hi > 0 > lo


def test_token_bucket_enforces_rate():
    async def go():
        bucket = TokenBucket(rate_per_sec=20, capacity=5)
        t0 = time.monotonic()
        await asyncio.gather(*(bucket.acquire() for _ in range(25)))
        return time.monotonic() - t0
    assert asyncio.run(go()) >= 0.95        # 5 burst + 20 more at 20/s ~= 1.0s


def test_event_append_is_idempotent(tmp_path):
    store = EventStore(connect(str(tmp_path / "t.db")))
    assert store.append("s", 0, "item_selected", "item-1")
    assert not store.append("s", 0, "item_selected", "item-1")


def test_crash_and_resume_matches_clean_run(setup, tmp_path):
    d, bank, test = setup
    cfg = SessionConfig(se_target=0.3)

    def final(db, crash):
        if crash:
            ctx, _ = build_context(d, bank, str(tmp_path / db))
            res = asyncio.run(run_many(ctx, "r", test, cfg, crash_after_step=4))
            assert all(isinstance(r, SimulatedCrash) for r in res)
        # fresh context = a restarted process; all state must come from the DB
        ctx, conn = build_context(d, bank, str(tmp_path / db))
        asyncio.run(run_many(ctx, "r", test, cfg))
        return conn.execute("SELECT model, theta, n_items FROM sessions ORDER BY model").fetchall()

    assert final("crash.db", True) == final("clean.db", False)
