"""Design B Step 4: the Redis limiter holds its rate across 8 separate OS processes.
Skipped unless REDIS_URL is reachable. Each test uses its own key prefix."""
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid

import pytest

from adaptive_eval.b.queue import connect
from adaptive_eval.b.ratelimit_redis import RedisLimiter

URL = os.environ.get("REDIS_URL")


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

RATE, BURST_S, PROCS, PER_PROC = 20.0, 0.25, 8, 10        # capacity = 5 requests
CHILD = """
import asyncio, json, sys, time
from adaptive_eval.b.queue import connect
from adaptive_eval.b.ratelimit_redis import RedisLimiter
async def main(url, prefix, n, rate, burst):
    r = connect(url)
    lim = RedisLimiter(r, "p", rpm=rate * 60, tpm=10**9, burst_s=burst, prefix=prefix)
    out = []
    for _ in range(n):
        await lim.acquire(1)
        out.append(time.time())
    await r.aclose()
    print(json.dumps(out))
asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])))
"""


async def _cleanup(prefix):
    r = connect(URL)
    keys = [k async for k in r.scan_iter(f"{prefix}*")]
    if keys:
        await r.delete(*keys)
    await r.aclose()


def test_eight_processes_share_one_bucket():
    prefix = f"t{uuid.uuid4().hex[:8]}:"
    try:
        procs = [subprocess.Popen([sys.executable, "-c", CHILD, URL, prefix, str(PER_PROC),
                                   str(RATE), str(BURST_S)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(PROCS)]
        grants = []
        for p in procs:
            out, err = p.communicate(timeout=60)
            assert p.returncode == 0, err
            grants += json.loads(out)
    finally:
        asyncio.run(_cleanup(prefix))

    grants.sort()
    cap = RATE * BURST_S
    total = PROCS * PER_PROC
    assert len(grants) == total
    # no 1-second window ever exceeds rate + burst (small tolerance for timestamp jitter)
    worst = max(sum(1 for t in grants if s <= t < s + 1.0) for s in grants)
    assert worst <= RATE + cap + 1, worst
    # and the whole batch can't finish faster than the rate allows
    assert grants[-1] - grants[0] >= (total - cap) / RATE - 0.25


def test_token_bucket_binds_and_denied_call_charges_nothing():
    prefix = f"t{uuid.uuid4().hex[:8]}:"

    async def main():
        r = connect(URL)
        try:
            # plenty of requests/s, but only 1000 tokens/s with a 100-token burst
            lim = RedisLimiter(r, "p", rpm=60_000, tpm=60_000, burst_s=0.1, prefix=prefix)
            await lim.acquire(100)                          # drains the token bucket
            req_before = float(await r.hget(lim.keys[0], "tokens"))
            wait = await lim.try_acquire(100)               # must be denied...
            req_after = float(await r.hget(lim.keys[0], "tokens"))
            t0 = time.monotonic()
            for _ in range(5):
                await lim.acquire(100)
            elapsed = time.monotonic() - t0
            with pytest.raises(ValueError):
                await lim.acquire(101)                      # can never fit
            return wait, req_before, req_after, elapsed
        finally:
            await r.aclose()
            await _cleanup(prefix)

    wait, req_before, req_after, elapsed = asyncio.run(main())
    assert wait > 0
    assert req_after >= req_before          # ...and must not spend a request token
    assert elapsed >= 0.45                  # 5 x 100 tokens at 1000/s after an empty bucket
