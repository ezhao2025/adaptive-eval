"""Design B Step 2: A's crash-and-resume guarantee, on Postgres.

Skipped unless PG_DSN is set and reachable. Each test gets throwaway schemas.
"""
import asyncio
import os
import uuid

import asyncpg
import numpy as np
import pytest

from adaptive_eval import data as D
from adaptive_eval.b import pgstore
from adaptive_eval.engine import Context, SessionConfig, SimulatedCrash, run_many
from adaptive_eval.irt import fit_2pl
from adaptive_eval.providers import DEFAULT_PROVIDERS, ReplayProvider
from adaptive_eval.ratelimit import ProviderLimiter

DSN = os.environ.get("PG_DSN")


def _reachable():
    if not DSN:
        return False
    async def ping():
        conn = await asyncpg.connect(DSN, timeout=3)
        await conn.close()
    try:
        asyncio.run(ping())
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="PG_DSN not set or Postgres not reachable")


def make_data(tmp_path):
    d = D.generate_synthetic(12, 60, seed=1)
    D.save(d, str(tmp_path / "d.json"))
    d = D.load(str(tmp_path / "d.json"))
    bank, _ = fit_2pl(d["R"], d["items"])
    return d, bank


def pg_context(d, bank, pool, seed=0):
    midx = {m: i for i, m in enumerate(d["models"])}
    responses = {(m, it): bool(d["R"][midx[m], j]) for m in d["models"]
                 for j, it in enumerate(d["items"])}
    providers = {n: ReplayProvider(n, c, responses, seed) for n, c in DEFAULT_PROVIDERS.items()}
    limiters = {n: ProviderLimiter(c.rpm, c.tpm) for n, c in DEFAULT_PROVIDERS.items()}
    return Context(bank, providers, DEFAULT_PROVIDERS, limiters, pgstore.PgResponseCache(pool),
                   pgstore.PgEventStore(pool), d["model_provider"])


async def sessions(pool, run_name):
    rows = await pool.fetch("SELECT model, theta, n_items FROM sessions"
                            " WHERE run_name=$1 AND status='done' ORDER BY model", run_name)
    return [(r["model"], round(r["theta"], 9), r["n_items"]) for r in rows]


async def drop(schemas):
    conn = await asyncpg.connect(DSN)
    for s in schemas:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{s}" CASCADE')
    await conn.close()


def test_crash_resume_matches_clean_run_on_postgres(tmp_path):
    d, bank = make_data(tmp_path)
    cfg = SessionConfig()
    crash_s, clean_s = f"t_{uuid.uuid4().hex[:8]}", f"t_{uuid.uuid4().hex[:8]}"

    async def main():
        try:
            pool = await pgstore.connect(DSN, crash_s)
            res = await run_many(pg_context(d, bank, pool), "run", d["models"], cfg,
                                 crash_after_step=6)
            assert all(isinstance(r, SimulatedCrash) for r in res)
            await pool.close()

            pool = await pgstore.connect(DSN, crash_s)          # fresh "process": new pool + context
            res = await run_many(pg_context(d, bank, pool, seed=1), "run", d["models"], cfg)
            assert not [r for r in res if isinstance(r, Exception)]
            resumed = await sessions(pool, "run")
            pending_left = await pool.fetchval(
                "SELECT COUNT(*) FROM (SELECT session_id, step FROM events WHERE type='item_selected'"
                " EXCEPT SELECT session_id, step FROM events WHERE type='answer_recorded') x")
            await pool.close()

            pool = await pgstore.connect(DSN, clean_s)
            res = await run_many(pg_context(d, bank, pool), "run", d["models"], cfg)
            assert not [r for r in res if isinstance(r, Exception)]
            clean = await sessions(pool, "run")
            ok, paid = await pool.fetchval("SELECT COUNT(*) FROM call_attempts WHERE status='ok'"), \
                await pool.fetchval("SELECT COUNT(*) FROM events"
                                    " WHERE type='answer_recorded' AND cached=0")
            await pool.close()
        finally:
            await drop([crash_s, clean_s])
        return resumed, clean, pending_left, ok, paid

    resumed, clean, pending_left, ok, paid = asyncio.run(main())
    assert len(clean) == len(d["models"])
    assert resumed == clean              # identical thetas and item counts
    assert pending_left == 0             # every write-ahead item got answered
    assert ok == paid                    # one successful attempt per paid answer


def test_append_is_idempotent_and_batched_on_postgres():
    schema = f"t_{uuid.uuid4().hex[:8]}"

    async def main():
        try:
            pool = await pgstore.connect(DSN, schema)
            store = pgstore.PgEventStore(pool)
            assert await store.append("s", 0, "item_selected", "i1") is True
            assert await store.append("s", 0, "item_selected", "i1") is False
            await store.append_many([("s", 0, "answer_recorded", "i1", 1, 0, 0.1),
                                     ("s", 0, "answer_recorded", "i1", 1, 0, 0.1)])
            n = await pool.fetchval("SELECT COUNT(*) FROM events")
            st = await store.load_state("s")
            await pool.close()
        finally:
            await drop([schema])
        return n, st

    n, st = asyncio.run(main())
    assert n == 2 and st.answered == [("i1", 1)] and st.pending is None


def test_many_processes_can_connect_at_once():
    """Regression: concurrent CREATE ... IF NOT EXISTS used to crash all but one worker."""
    schema = f"t_{uuid.uuid4().hex[:8]}"

    async def main():
        try:
            pools = await asyncio.gather(*(pgstore.connect(DSN, schema) for _ in range(6)))
            for p in pools:
                await p.close()
        finally:
            await drop([schema])

    asyncio.run(main())
