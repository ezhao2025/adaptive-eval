"""Design B Step 3: Redis Streams job queue. Skipped unless REDIS_URL is reachable.
Each test uses its own key prefix and deletes its keys afterwards."""
import asyncio
import os
import uuid

import pytest

from adaptive_eval.b.queue import Job, JobQueue, connect, real_job_id, spec_job_id

URL = os.environ.get("REDIS_URL")
PROVIDERS = ["sim-openai", "sim-google"]


def _reachable():
    if not URL:
        return False
    async def ping():
        r = connect(URL)
        try:
            return await r.ping()
        finally:
            await r.aclose()
    try:
        return asyncio.run(ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="REDIS_URL not set or Redis not reachable")


def job(n, spec=False, provider="sim-openai"):
    key = f"key{n}"
    return Job(spec_job_id(key) if spec else real_job_id("run:m", n), provider, "m",
               f"item{n}", key, spec)


def run(test):
    """Run test(q) against a fresh, isolated queue; clean up its keys afterwards."""
    async def main():
        r = connect(URL)
        prefix = f"t{uuid.uuid4().hex[:8]}:"
        q = JobQueue(r, PROVIDERS, prefix)
        try:
            await q.ensure_groups()
            await q.ensure_groups()                 # idempotent: BUSYGROUP is ignored
            return await test(q)
        finally:
            keys = [k async for k in r.scan_iter(f"{prefix}*")]
            if keys:
                await r.delete(*keys)
            await r.aclose()
    return asyncio.run(main())


def test_real_lane_is_read_before_spec_lane():
    async def t(q):
        await q.enqueue(job(1, spec=True))
        await q.enqueue(job(2, provider="sim-google"))
        first = await q.read("w1", block_ms=100)
        second = await q.read("w1", block_ms=100)
        empty = await q.read("w1", block_ms=100)
        return first, second, empty
    first, second, empty = run(t)
    assert [d.job.job_id for d in first] == ["run:m:2"] and not first[0].job.speculative
    assert second[0].job.speculative and second[0].job.job_id == "spec:key1"
    assert empty == []


def test_ack_clears_pending():
    async def t(q):
        await q.enqueue(job(1))
        [d] = await q.read("w1", block_ms=100)
        await q.ack(d)
        return await q.r.xpending(d.stream, "workers")
    assert run(t)["pending"] == 0


def test_dead_worker_job_is_reclaimed_with_delivery_count():
    async def t(q):
        await q.enqueue(job(1))
        [dead] = await q.read("dead-worker", block_ms=100)   # read, never acked
        await asyncio.sleep(0.05)
        reclaimed = await q.reclaim("w2", min_idle_ms=20)
        again = await q.reclaim("w3", min_idle_ms=10_000)    # not idle long enough now
        for d in reclaimed:
            await q.ack(d)
        pending = await q.r.xpending(dead.stream, "workers")
        return dead, reclaimed, again, pending
    dead, reclaimed, again, pending = run(t)
    assert [d.job.job_id for d in reclaimed] == [dead.job.job_id]
    assert reclaimed[0].delivery == 2
    assert again == [] and pending["pending"] == 0


def test_results_round_trip():
    async def t(q):
        j = job(1)
        await q.publish_result(j, correct=1, cached=0, cost_usd=0.01)
        [(msg_id, fields)] = await q.read_results("sched", block_ms=100)
        await q.ack_result(msg_id)
        return fields, await q.r.xlen(q.results)
    fields, left = run(t)
    assert fields["job_id"] == "run:m:1" and fields["correct"] == "1" and left == 0
