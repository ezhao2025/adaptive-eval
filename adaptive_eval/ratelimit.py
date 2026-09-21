"""Async token buckets. One ProviderLimiter per provider enforces both
requests/minute and tokens/minute across every session sharing that provider."""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate, self.capacity = rate_per_sec, capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()   # waiters are served FIFO -> no starvation

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    async def acquire(self, n: float = 1.0) -> None:
        if n > self.capacity:
            raise ValueError("request larger than bucket capacity")
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return
                await asyncio.sleep((n - self.tokens) / self.rate)


class ProviderLimiter:
    def __init__(self, rpm: int, tpm: int, burst_seconds: float = 1.0):
        self.requests = TokenBucket(rpm / 60, max(1.0, rpm / 60 * burst_seconds))
        self.tokens = TokenBucket(tpm / 60, max(1.0, tpm / 60 * burst_seconds))

    async def acquire(self, est_tokens: int) -> None:
        await self.requests.acquire(1)
        await self.tokens.acquire(est_tokens)
