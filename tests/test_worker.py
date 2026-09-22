"""Design B Step 5: the worker. Needs both PG_DSN and REDIS_URL; skipped otherwise.
Workers run in-process on one event loop; each test gets its own schema and key prefix."""
import asyncio
import os
import time
import uuid

import asyncpg
import pytest

from adaptive_eval import data as D
from adaptive_eval.b import pgstore
from adaptive_eval.b.queue import Job, JobQueue, connect, real_job_id, spec_job_id
from adaptive_eval.b.worker import Worker
from adaptive_eval.providers import DEFAULT_PROVIDERS, ProviderConfig, ReplayProvider
from adaptive_eval.storage import ResponseCache

DSN, URL = os.environ.get("PG_DSN"), os.environ.get("REDIS_URL")


def _reachable():
    if not (DSN and URL):
        return False
    async def ping():
        conn = await asyncpg.connect(DSN, timeout=3)
        await conn.close()
        r = connect(URL)
        try:
            return await r.ping()
        finally:
            await r.aclose()
    try:
        return asyncio.run(ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="PG_DSN/REDIS_URL not set or not reachable")

DATA = D.generate_synthetic(6, 10, seed=3)
RESPONSES = {(m, it): bool(DATA["R"][i][j]) for i, m in enumerate(DATA["models"])
             for j, it in enumerate(DATA["items"])}


def cfgs(latency=(0.001, 0.005), failure_rate=0.0):
    return {n: ProviderConfig(60_000, 10**9, latency, failure_rate, 0.0025, 0.01)
            for n in DEFAULT_PROVIDERS}


def job(m, it, step, spec=False):
    key = ResponseCache.make_key(model=m, item=it)
    jid = spec_job_id(key) if spec else real_job_id(f"run:{m}", step)
    return Job(jid, DATA["model_provider"][m], m, it, key, spec)


def all_jobs():
    return [job(m, it, j) for m in DATA["models"] for j, it in enumerate(DATA["items"])]


async def harness(body, *, n_workers=3, provider_cfgs=None, setup=None, **worker_kw):
    """Fresh schema + prefix; run setup(q) BEFORE any worker starts; start n in-process
    workers; run body; clean up."""
    schema, prefix = f"t_{uuid.uuid4().hex[:8]}", f"t{uuid.uuid4().hex[:8]}:"
    provider_cfgs = provider_cfgs or cfgs()
    r = connect(URL)
    pool = await pgstore.connect(DSN, schema)
    q = JobQueue(r, list(provider_cfgs), prefix)
    await q.ensure_groups()
    before = await setup(q) if setup else None
    workers = [Worker(f"w{i}", r, pool,
                      {n: ReplayProvider(n, c, RESPONSES, seed=i) for n, c in provider_cfgs.items()},
                      provider_cfgs, prefix=prefix, block_ms=100, **worker_kw)
               for i in range(n_workers)]
    runs = [asyncio.create_task(w.run()) for w in workers]
    try:
        return await body(q, pool, workers, before) if setup else await body(q, pool, workers)
    finally:
        for w in workers:
            w.stopping.set()
        await asyncio.wait_for(asyncio.gather(*runs, return_exceptions=True), 10)
        await pool.close()
        conn = await asyncpg.connect(DSN)
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
        keys = [k async for k in r.scan_iter(f"{prefix}*")]
        if keys:
            await r.delete(*keys)
        await r.aclose()


async def collect(q, n, timeout=20.0):
    got, dupes, deadline = {}, 0, time.monotonic() + timeout
    while len(got) < n and time.monotonic() < deadline:
        for msg_id, f in await q.read_results("test", count=100, block_ms=100):
            dupes += f["job_id"] in got
            got[f["job_id"]] = f
            await q.ack_result(msg_id)
    return got, dupes


def test_workers_answer_every_job_once_despite_transient_errors():
    jobs = all_jobs()

    async def body(q, pool, workers):
        for j in jobs:
            await q.enqueue(j)
        got, _ = await collect(q, len(jobs))
        ok = await pool.fetchval("SELECT COUNT(*) FROM call_attempts WHERE status='ok'")
        transient = await pool.fetchval(
            "SELECT COUNT(*) FROM call_attempts WHERE status='transient'")
        cached = await pool.fetchval("SELECT COUNT(*) FROM cache")
        return got, ok, transient, cached, sum(w.stats["transient"] for w in workers)

    got, ok, transient, cached, stat_transient = asyncio.run(
        harness(body, provider_cfgs=cfgs(failure_rate=0.2)))
    assert len(got) == len(jobs)
    for j in jobs:
        assert got[j.job_id]["error"] == ""
        assert int(got[j.job_id]["correct"]) == int(RESPONSES[(j.model, j.item_id)])
    assert ok == cached == len(jobs)          # exactly one paid call per answer
    assert transient == stat_transient > 0    # every retried failure is on disk


def test_inflight_claim_stops_real_and_spec_jobs_paying_twice():
    m, it = DATA["models"][0], DATA["items"][0]

    async def body(q, pool, workers):
        await q.enqueue(job(m, it, 0, spec=True))
        await q.enqueue(job(m, it, 0))
        got, _ = await collect(q, 1)
        await asyncio.sleep(0.5)              # let the other job finish too
        ok = await pool.fetchval("SELECT COUNT(*) FROM call_attempts WHERE status='ok'")
        return got, ok, [w.stats for w in workers]

    got, ok, stats = asyncio.run(harness(body, n_workers=2,
                                         provider_cfgs=cfgs(latency=(0.3, 0.3))))
    assert list(got) == [job(m, it, 0).job_id]     # only the real job posts a result
    assert ok == 1                                 # the provider was paid once
    assert sum(s["cache_hits"] for s in stats) == 1


def test_job_of_dead_worker_is_reclaimed_and_finished():
    m, it = DATA["models"][1], DATA["items"][1]
    j = job(m, it, 1)

    async def setup(q):
        # a "dead" worker takes the job and never acks, before any live worker exists
        await q.enqueue(j)
        [dead] = await q._read("dead-worker", q.job_streams(False), 1, None)
        return dead

    async def body(q, pool, workers, dead):
        got, _ = await collect(q, 1, timeout=10)
        keys = [r["job_id"] for r in await pool.fetch("SELECT job_id FROM call_attempts")]
        pending = await q.r.xpending(dead.stream, "workers")
        return got, keys, pending, sum(w.stats["reclaimed"] for w in workers)

    got, keys, pending, reclaimed = asyncio.run(
        harness(body, setup=setup, n_workers=1, min_idle_ms=100, reclaim_every_s=0.2))
    assert j.job_id in got and reclaimed >= 1
    assert any(k.endswith("#2") for k in keys)     # logged as the second delivery
    assert pending["pending"] == 0


def test_fatal_error_posts_error_result_and_acks():
    j = Job(real_job_id("run:ghost", 0), "sim-openai", "ghost-model", "no-such-item",
            ResponseCache.make_key(model="ghost", item="none"), False)

    async def body(q, pool, workers):
        await q.enqueue(j)
        got, _ = await collect(q, 1)
        status = await pool.fetchval("SELECT status FROM call_attempts")
        pending = await q.r.xpending(q.stream("sim-openai", False), "workers")
        return got, status, pending

    got, status, pending = asyncio.run(harness(body, n_workers=1))
    assert got[j.job_id]["error"].startswith("LookupError")
    assert status == "fatal" and pending["pending"] == 0
