"""Design B worker: stateless. Pulls jobs from Redis, answers from the cache or the provider,
posts results, and acks last.

Per job:
  1. cache hit -> step 6
  2. claim inflight:{cache_key} (SET NX EX); if another worker holds it, poll the cache
  3. acquire the shared Redis rate limit
  4. call the provider, logging one call_attempts row per attempt
  5. transient error -> retry with backoff + jitter; fatal or exhausted -> error result
  6. write the cache, release the inflight key
  7. XADD results (real jobs only), then XACK

The ack is last, so a crash anywhere before it leaves the job reclaimable: at-least-once.
Duplicate results from a reclaimed job are harmless because the scheduler's event writes
are idempotent (UNIQUE(session_id, step, type)).

    python -m adaptive_eval.b.worker --id w1 --data data/synthetic.json
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import signal
import sys
import time
import zlib
from dataclasses import asdict, replace

import numpy as np
from redis.exceptions import ResponseError

from .. import data as D
from ..real_provider import ANTHROPIC_CONFIG, AnthropicProvider, load_items
from ..providers import DEFAULT_PROVIDERS, ReplayProvider, TransientError
from . import pgstore
from .queue import Delivery, JobQueue, connect
from .ratelimit_redis import limiters_for
from .sim import add_args, provider_configs

def redis_state_lost(e: Exception) -> bool:
    """Redis restarted or was wiped: groups are gone (NOGROUP), or a stream vanished while
    we were blocked reading it (UNBLOCKED ... no longer exists)."""
    msg = str(e)
    return "NOGROUP" in msg or "no longer exists" in msg


# delete the inflight key only if we still own it (it may have expired and been re-claimed)
RELEASE = ("if redis.call('GET', KEYS[1]) == ARGV[1] then "
           "return redis.call('DEL', KEYS[1]) end return 0")


class Worker:
    def __init__(self, worker_id: str, r, pool, providers: dict, provider_cfgs: dict, *,
                 prefix: str = "", concurrency: int = 16, max_retries: int = 6,
                 inflight_ttl_s: int = 60, min_idle_ms: int = 30_000,
                 reclaim_every_s: float = 5.0, block_ms: int = 500, poll_s: float = 0.05,
                 burst_s: float = 1.0, spec_min_free: float = 0.3):
        self.id, self.r, self.prefix = worker_id, r, prefix
        self.providers, self.cfgs = providers, provider_cfgs
        self.q = JobQueue(r, list(provider_cfgs), prefix)
        self.limiters = limiters_for(r, provider_cfgs, burst_s, prefix)
        self.cache = pgstore.PgResponseCache(pool)
        self.store = pgstore.PgEventStore(pool)
        self.concurrency, self.max_retries = concurrency, max_retries
        self.inflight_ttl_s, self.min_idle_ms = inflight_ttl_s, min_idle_ms
        self.reclaim_every_s, self.block_ms, self.poll_s = reclaim_every_s, block_ms, poll_s
        self.spec_min_free = spec_min_free
        self._release = r.register_script(RELEASE)
        self.tasks: set[asyncio.Task] = set()
        self.stopping = asyncio.Event()
        self.stats = dict(jobs=0, cache_hits=0, paid=0, waited_on_inflight=0,
                          transient=0, fatal=0, reclaimed=0, spec_dropped=0)

    # ---- one job -------------------------------------------------------------------
    async def handle(self, d: Delivery) -> None:
        job = d.job
        value, cached, error = await self._resolve(d)
        if not job.speculative:
            if error is None:
                await self.q.publish_result(
                    job, correct=int(value["correct"]), cached=int(cached),
                    cost_usd=0.0 if cached else value["cost_usd"],
                    tokens=value["input_tokens"] + value["output_tokens"], error="")
            else:
                await self.q.publish_result(job, error=error)
        await self.q.ack(d)                        # last: a crash before this = reclaimable
        self.stats["jobs"] += 1

    async def _resolve(self, d: Delivery):
        """Returns (value, cached, error). Pays for a call only while holding inflight."""
        job = d.job
        inflight = f"{self.prefix}inflight:{job.cache_key}"
        token = f"{self.id}:{d.msg_id}"
        waited = False
        while True:
            value = await self.cache.get(job.cache_key)
            if value is not None:
                self.stats["cache_hits"] += 1
                return value, True, None
            if job.speculative and \
                    await self.limiters[job.provider].headroom() < self.spec_min_free:
                self.stats["spec_dropped"] += 1      # bucket is busy: real work comes first
                return None, False, None
            if await self.r.set(inflight, token, nx=True, ex=self.inflight_ttl_s):
                try:
                    value = await self.cache.get(job.cache_key)   # landed just before our claim?
                    if value is not None:
                        self.stats["cache_hits"] += 1
                        return value, True, None
                    value, error = await self._call(d)
                    if error is None:
                        await self.cache.put(job.cache_key, value)
                        self.stats["paid"] += 1
                    return value, False, error
                finally:
                    await self._release(keys=[inflight], args=[token])
            if job.speculative:                    # already being fetched: nothing to warm
                self.stats["spec_dropped"] += 1
                return None, False, None
            if not waited:                         # someone else is fetching it: don't pay twice
                self.stats["waited_on_inflight"] += 1
                waited = True
            await asyncio.sleep(self.poll_s)

    async def _call(self, d: Delivery):
        job = d.job
        provider, cfg = self.providers[job.provider], self.cfgs[job.provider]
        # unique per enqueue (msg_id) and per delivery, so a call re-paid after a reclaim or
        # a re-enqueue gets its own rows instead of colliding on UNIQUE(job_id, attempt)
        attempt_key = f"{job.job_id}@{d.msg_id}#{d.delivery}"
        for attempt in range(self.max_retries + 1):
            await self.limiters[job.provider].acquire(cfg.est_tokens_per_call)
            t0 = time.time()
            try:
                resp = await provider.answer(job.model, job.item_id)
            except TransientError as e:
                await self._log(job, attempt_key, attempt, "transient_error", t0)
                self.stats["transient"] += 1
                if attempt == self.max_retries:
                    return None, f"transient errors exhausted: {e}"
                await asyncio.sleep(min(8.0, 0.1 * 2 ** attempt) * random.uniform(0.5, 1.5))
                continue
            except Exception as e:                 # 4xx, missing item, grader bug: not retryable
                await self._log(job, attempt_key, attempt, "error", t0)
                self.stats["fatal"] += 1
                return None, f"{type(e).__name__}: {e}"
            await self._log(job, attempt_key, attempt, "ok", t0, resp)
            return asdict(resp), None
        raise AssertionError("unreachable")

    async def _log(self, job, attempt_key, attempt, outcome, t0, resp=None):
        await self.store.log_attempt(
            "", -1, attempt, job.provider, job.model, job.item_id, outcome, t0,
            time.time() - t0, job_id=attempt_key, speculative=job.speculative,
            tokens=None if resp is None else resp.input_tokens + resp.output_tokens,
            cost_usd=None if resp is None else resp.cost_usd)

    # ---- main loop -----------------------------------------------------------------
    def _spawn(self, d: Delivery) -> None:
        t = asyncio.create_task(self._safe(d))
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    async def _safe(self, d: Delivery) -> None:
        try:
            await self.handle(d)
        except Exception as e:                     # e.g. Postgres down: leave it unacked
            print(f"[{self.id}] job {d.job.job_id} not acked, left for reclaim: {e!r}",
                  file=sys.stderr)

    async def run(self) -> None:
        await self.q.ensure_groups()
        self._last_reclaim = float("-inf")         # reclaim immediately on (re)start
        while not self.stopping.is_set():
            try:
                await self._step()
            except ResponseError as e:
                if not redis_state_lost(e):
                    raise
                # Redis lost its state (restart or FLUSHALL): recreate the groups and carry
                # on. Postgres still has everything; the scheduler re-enqueues lost jobs.
                print(f"[{self.id}] Redis groups missing, recreating", file=sys.stderr)
                await self.q.ensure_groups()
                await asyncio.sleep(0.2)
        if self.tasks:
            await asyncio.wait(set(self.tasks))

    async def _step(self) -> None:
        free = self.concurrency - len(self.tasks)
        if free <= 0:
            await asyncio.wait(set(self.tasks), return_when=asyncio.FIRST_COMPLETED)
            return
        now = time.monotonic()
        if now - self._last_reclaim >= self.reclaim_every_s:
            self._last_reclaim = now
            for d in await self.q.reclaim(self.id, self.min_idle_ms, count=free):
                self.stats["reclaimed"] += 1
                self._spawn(d)
            return
        # read only as many jobs as there are free slots, so jobs don't sit idle here
        for d in await self.q.read(self.id, count=free, block_ms=self.block_ms):
            self._spawn(d)


def replay_providers(data_path: str, seed: int, cfgs: dict | None = None) -> dict:
    d = D.load(data_path)
    responses = {(m, it): bool(d["R"][i, j]) for i, m in enumerate(d["models"])
                 for j, it in enumerate(d["items"]) if not np.isnan(d["R"][i, j])}
    return {n: ReplayProvider(n, c, responses, seed)
            for n, c in (cfgs or DEFAULT_PROVIDERS).items()}


async def amain(a) -> None:
    r = connect(a.redis_url)
    pool = await pgstore.connect(a.pg_dsn, a.schema)
    if a.provider == "anthropic":                 # real API: one provider, real limits/prices
        cfgs = {"anthropic": replace(ANTHROPIC_CONFIG, rpm=a.rpm, tpm=a.tpm,
                                     usd_per_1k_input=a.usd_per_1k_input,
                                     usd_per_1k_output=a.usd_per_1k_output)}
        providers = {"anthropic": AnthropicProvider(cfgs["anthropic"], load_items(a.items),
                                                    max_tokens=a.max_tokens)}
        print(f"[{a.id}] real provider: {cfgs['anthropic'].rpm} rpm, "
              f"${cfgs['anthropic'].usd_per_1k_input}/1k in, "
              f"${cfgs['anthropic'].usd_per_1k_output}/1k out", file=sys.stderr)
    else:
        cfgs = provider_configs(a.rate_scale, a.latency_scale)
        providers = replay_providers(a.data, zlib.crc32(a.id.encode()), cfgs)
    w = Worker(a.id, r, pool, providers,
               cfgs, prefix=a.prefix, concurrency=a.concurrency,
               min_idle_ms=a.min_idle_ms, inflight_ttl_s=a.inflight_ttl_s,
               spec_min_free=a.spec_min_free)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, w.stopping.set)   # graceful: finish in-flight jobs
    print(f"[{w.id}] started", file=sys.stderr)
    try:
        await w.run()
    finally:
        print(f"[{w.id}] stopped {w.stats}", file=sys.stderr)
        await pool.close()
        await r.aclose()


def main() -> None:
    p = argparse.ArgumentParser(prog="adaptive_eval.b.worker")
    p.add_argument("--id", required=True)
    p.add_argument("--data", default="data/synthetic.json", help="response matrix for replay")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--min-idle-ms", type=int, default=30_000,
                   help="reclaim jobs idle this long; keep it above ~3x p99 job time")
    p.add_argument("--inflight-ttl-s", type=int, default=60,
                   help="how long a dead worker's claim blocks others; keep above max call time")
    p.add_argument("--spec-min-free", type=float, default=0.3,
                   help="drop a speculative job unless this fraction of the bucket is free")
    add_args(p)
    p.add_argument("--provider", choices=["replay", "anthropic"], default="replay")
    p.add_argument("--items", default="data/items_smoke.json",
                   help="question/answer file for --provider anthropic")
    p.add_argument("--rpm", type=float, default=ANTHROPIC_CONFIG.rpm)
    p.add_argument("--tpm", type=float, default=ANTHROPIC_CONFIG.tpm)
    p.add_argument("--usd-per-1k-input", type=float, default=ANTHROPIC_CONFIG.usd_per_1k_input)
    p.add_argument("--usd-per-1k-output", type=float, default=ANTHROPIC_CONFIG.usd_per_1k_output)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--prefix", default="")
    p.add_argument("--schema", default=None)
    p.add_argument("--pg-dsn", default=os.environ.get("PG_DSN"))
    p.add_argument("--redis-url", default=os.environ.get("REDIS_URL"))
    a = p.parse_args()
    if not a.pg_dsn or not a.redis_url:
        sys.exit("set PG_DSN and REDIS_URL (source .env.sh)")
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
