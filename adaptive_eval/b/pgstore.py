"""Design B storage: Postgres (asyncpg) port of A's ResponseCache and EventStore.

Same method names and arguments as adaptive_eval.storage, but every method is async.
The engine awaits storage calls when they return awaitables, so it runs unchanged on
either backend. Postgres is the source of truth; nothing here depends on Redis.
"""
from __future__ import annotations

import json
import time

import asyncpg

from ..storage import ResponseCache as _SqliteCache
from ..storage import SessionState

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, run_name TEXT, model TEXT, provider TEXT,
    config TEXT, status TEXT, theta DOUBLE PRECISION, se DOUBLE PRECISION, n_items INTEGER,
    started_at DOUBLE PRECISION, finished_at DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL, step INTEGER NOT NULL, type TEXT NOT NULL,
    item_id TEXT NOT NULL, correct INTEGER, cached INTEGER, cost_usd DOUBLE PRECISION,
    ts DOUBLE PRECISION,
    UNIQUE (session_id, step, type));
CREATE TABLE IF NOT EXISTS call_attempts (
    id BIGSERIAL PRIMARY KEY, job_id TEXT NOT NULL, attempt INT NOT NULL,
    provider TEXT, model TEXT, item_id TEXT, speculative BOOLEAN,
    status TEXT,            -- ok | transient | fatal
    latency_s REAL, tokens INT, cost_usd REAL, ts TIMESTAMPTZ DEFAULT now(),
    UNIQUE (job_id, attempt));
CREATE INDEX IF NOT EXISTS call_attempts_provider_ts ON call_attempts (provider, ts);
"""

_STATUS = {"ok": "ok", "transient_error": "transient", "error": "fatal"}


async def connect(dsn: str, schema: str | None = None, max_size: int = 10) -> asyncpg.Pool:
    """Open a pool and create the tables. `schema` isolates a run (tests use a fresh one)."""
    settings = {}
    if schema:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        finally:
            await conn.close()
        settings["search_path"] = schema
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=max_size,
                                     server_settings=settings)
    await pool.execute(SCHEMA)
    return pool


class PgResponseCache:
    make_key = staticmethod(_SqliteCache.make_key)   # identical keys on both backends

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.hits = self.misses = 0

    async def get(self, key: str) -> dict | None:
        value = await self.pool.fetchval("SELECT value FROM cache WHERE key=$1", key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(value)

    async def put(self, key: str, value: dict) -> None:
        await self.pool.execute(
            "INSERT INTO cache (key, value, created_at) VALUES ($1,$2,$3)"
            " ON CONFLICT DO NOTHING", key, json.dumps(value), time.time())


class PgEventStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ensure_session(self, session_id, run_name, model, provider, config: dict) -> None:
        await self.pool.execute(
            "INSERT INTO sessions (session_id, run_name, model, provider, config, status,"
            " started_at) VALUES ($1,$2,$3,$4,$5,'running',$6) ON CONFLICT DO NOTHING",
            session_id, run_name, model, provider, json.dumps(config), time.time())

    async def append(self, session_id, step, type_, item_id, correct=None, cached=None,
                     cost=None) -> bool:
        """Returns False if this (session, step, type) already exists -- safe to retry."""
        status = await self.pool.execute(
            "INSERT INTO events (session_id, step, type, item_id, correct, cached, cost_usd, ts)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT DO NOTHING",
            session_id, step, type_, item_id, correct, cached, cost, time.time())
        return status == "INSERT 0 1"

    async def append_many(self, rows: list[tuple]) -> None:
        """Batched write. rows: (session_id, step, type, item_id, correct, cached, cost)."""
        now = time.time()
        await self.pool.executemany(
            "INSERT INTO events (session_id, step, type, item_id, correct, cached, cost_usd, ts)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT DO NOTHING",
            [(*r, now) for r in rows])

    async def load_state(self, session_id) -> SessionState:
        st = SessionState()
        status = await self.pool.fetchval(
            "SELECT status FROM sessions WHERE session_id=$1", session_id)
        st.done = status == "done"
        selected, answers = {}, {}
        for r in await self.pool.fetch(
                "SELECT step, type, item_id, correct FROM events WHERE session_id=$1"
                " ORDER BY step", session_id):
            if r["type"] == "item_selected":
                selected[r["step"]] = r["item_id"]
            else:
                answers[r["step"]] = r["correct"]
        for step in sorted(selected):
            if step in answers:
                st.answered.append((selected[step], answers[step]))
            else:
                st.pending = (step, selected[step])
        return st

    async def log_attempt(self, session_id, step, attempt, provider, model, item_id,
                          outcome, started_at, latency_s, *, job_id=None, speculative=False,
                          tokens=None, cost_usd=None) -> None:
        """One row per provider call, failures included. (job_id, attempt) is unique, so a
        redelivered job's duplicate log is dropped while a re-issued job (new id) is kept."""
        await self.pool.execute(
            "INSERT INTO call_attempts (job_id, attempt, provider, model, item_id, speculative,"
            " status, latency_s, tokens, cost_usd, ts)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,to_timestamp($11)) ON CONFLICT DO NOTHING",
            job_id or f"{session_id}:{step}", attempt, provider, model, item_id, speculative,
            _STATUS.get(outcome, outcome), latency_s, tokens, cost_usd, started_at)

    async def finish(self, session_id, theta, se, n_items) -> None:
        await self.pool.execute(
            "UPDATE sessions SET status='done', theta=$1, se=$2, n_items=$3, finished_at=$4"
            " WHERE session_id=$5", theta, se, n_items, time.time(), session_id)
