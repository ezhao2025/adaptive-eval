"""Design B distributed rate limiter: token buckets in Redis, updated by one Lua script.

The script runs atomically inside Redis and reads Redis's own clock, so every worker
process sees one consistent bucket and worker clock skew doesn't matter.

Each provider has a request bucket and a token bucket, as in A. Both are checked and
charged in the SAME script call: a call is granted only if both have room, and a denied
call charges neither. (Two separate scripts would let the request bucket be spent while
the call then waits on the token bucket, double-charging it on the retry.)
"""
from __future__ import annotations

import asyncio
import random

import redis.asyncio as redis

LUA = """
-- KEYS[1]=request bucket, KEYS[2]=token bucket
-- ARGV: req_rate/s, req_cap, tok_rate/s, tok_cap, n_tokens  ->  0 (granted) or wait_ms
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local function level(key, rate, cap)
  local b = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(b[1]) or cap
  local ts = tonumber(b[2]) or now
  return math.min(cap, tokens + math.max(0, now - ts) * rate)
end
local rr, rc = tonumber(ARGV[1]), tonumber(ARGV[2])
local tr, tc, n = tonumber(ARGV[3]), tonumber(ARGV[4]), tonumber(ARGV[5])
local r = level(KEYS[1], rr, rc)
local k = level(KEYS[2], tr, tc)
local wait = 0
if r < 1 then wait = math.max(wait, (1 - r) / rr) end
if k < n then wait = math.max(wait, (n - k) / tr) end
if wait == 0 then r = r - 1; k = k - n end
redis.call('HSET', KEYS[1], 'tokens', r, 'ts', now)
redis.call('HSET', KEYS[2], 'tokens', k, 'ts', now)
redis.call('PEXPIRE', KEYS[1], 120000)
redis.call('PEXPIRE', KEYS[2], 120000)
return math.ceil(wait * 1000)
"""


PEEK = """
-- read-only: fraction of capacity currently free, min over both buckets (as a string)
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local function level(key, rate, cap)
  local b = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(b[1]) or cap
  local ts = tonumber(b[2]) or now
  return math.min(cap, tokens + math.max(0, now - ts) * rate)
end
local r = level(KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2])) / tonumber(ARGV[2])
local k = level(KEYS[2], tonumber(ARGV[3]), tonumber(ARGV[4])) / tonumber(ARGV[4])
return tostring(math.min(r, k))
"""


class RedisLimiter:
    """Drop-in for A's ProviderLimiter: `await acquire(n_tokens)` blocks until granted.

    Not FIFO, unlike A's in-process limiter: a waiter can be overtaken by a newer caller.
    Jittered sleeps keep this bounded in practice; strict FIFO would need a ticket queue.
    """

    def __init__(self, r: redis.Redis, provider: str, rpm: float, tpm: float,
                 burst_s: float = 1.0, prefix: str = ""):
        tag = f"{prefix}rl:{{{provider}}}"          # hash tag: both keys in one cluster slot
        self.keys = [f"{tag}:req", f"{tag}:tok"]
        self.req_rate, self.tok_rate = rpm / 60.0, tpm / 60.0
        self.req_cap = max(1.0, self.req_rate * burst_s)
        self.tok_cap = self.tok_rate * burst_s
        self.script = r.register_script(LUA)
        self._peek = r.register_script(PEEK)
        self.waits = 0

    async def try_acquire(self, n_tokens: int) -> int:
        """One atomic attempt. Returns 0 if granted, else milliseconds to wait."""
        return int(await self.script(keys=self.keys, args=[
            self.req_rate, self.req_cap, self.tok_rate, self.tok_cap, n_tokens]))

    async def headroom(self) -> float:
        """Fraction of capacity free right now (0..1), without taking anything."""
        return float(await self._peek(keys=self.keys, args=[
            self.req_rate, self.req_cap, self.tok_rate, self.tok_cap]))

    async def acquire(self, n_tokens: int) -> None:
        if n_tokens > self.tok_cap:
            raise ValueError(f"{n_tokens} tokens can never fit a bucket of {self.tok_cap:.0f};"
                             " raise tpm or burst_s")
        while (w := await self.try_acquire(n_tokens)) > 0:
            self.waits += 1
            await asyncio.sleep(w / 1000 * random.uniform(1.0, 1.2))   # jitter: no herd


def limiters_for(r: redis.Redis, provider_cfgs: dict, burst_s: float = 1.0,
                 prefix: str = "") -> dict[str, RedisLimiter]:
    """One RedisLimiter per provider, from A's ProviderConfig (rpm, tpm)."""
    return {name: RedisLimiter(r, name, c.rpm, c.tpm, burst_s, prefix)
            for name, c in provider_cfgs.items()}
