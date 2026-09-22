"""Design B priority admission.

Per provider, cap real jobs in flight at a window of about rpm/60 x p50 latency x 1.5
(enough to keep the provider busy, no more). Ready jobs beyond the window wait in a heap:

  priority mode:  highest  expected SE reduction / expected cost  first  (breadth)
  nearest mode:   sessions with the fewest calls left to reach the SE target first  (depth)
  fifo mode:      first come, first served (the baseline)

The scheduler computes the priority value for the chosen mode; this class only orders by it.

Expected SE reduction of an item = SE_now - 1/sqrt(info_now + I(theta, item)).
An optional USD budget stops admitting paid work once spent + committed would exceed it.
"""
from __future__ import annotations

import heapq
import itertools
import math

from collections.abc import Awaitable, Callable

from .queue import Job, JobQueue


def expected_se_reduction(se_now: float, item_info: float) -> float:
    info_now = 1.0 / se_now ** 2                 # SE from estimate_ability includes the prior
    return se_now - 1.0 / math.sqrt(info_now + item_info)


class Admission:
    def __init__(self, q: JobQueue, provider_cfgs: dict, *, mode: str = "priority",
                 window_scale: float = 1.5, windows: dict[str, int] | None = None,
                 budget_usd: float | None = None):
        if mode not in ("priority", "nearest", "fifo"):
            raise ValueError(f"unknown admission mode {mode!r}")
        self.q, self.mode, self.budget = q, mode, budget_usd
        windows = windows or {}
        self.window = {p: windows.get(p) or max(1, math.ceil(
            c.rpm / 60 * (sum(c.latency) / 2) * window_scale)) for p, c in provider_cfgs.items()}
        # prior guess of cost per paid call; replaced by a running mean of real costs
        self.exp_cost = {p: c.est_tokens_per_call / 1000 * c.usd_per_1k_input
                         for p, c in provider_cfgs.items()}
        self._n_cost = dict.fromkeys(provider_cfgs, 0)
        self.heap: dict[str, list] = {p: [] for p in provider_cfgs}
        self.inflight: dict[str, set[str]] = {p: set() for p in provider_cfgs}
        self.max_inflight = dict.fromkeys(provider_cfgs, 0)
        self.spent = 0.0
        # called for every admission decision (Design C logs these as 'allocation' events)
        self.on_admit: Callable[[Job, float], Awaitable[None]] | None = None
        self.extra_committed = 0.0               # speculative spend (set by the Speculator)
        self._seq = itertools.count()

    def expected_cost(self, provider: str) -> float:
        return max(self.exp_cost[provider], 1e-9)

    def _committed(self) -> float:
        return self.spent + self.extra_committed + \
            sum(len(s) * self.expected_cost(p) for p, s in self.inflight.items())

    def _affordable(self, provider: str) -> bool:
        return self.budget is None or \
            self._committed() + self.expected_cost(provider) <= self.budget

    async def admit(self, job: Job, priority: float) -> None:
        key = 0.0 if self.mode == "fifo" else -priority      # max-heap on priority
        heapq.heappush(self.heap[job.provider], (key, next(self._seq), priority, job))
        await self._drain(job.provider)

    async def _drain(self, provider: str) -> None:
        h, live = self.heap[provider], self.inflight[provider]
        while h and len(live) < self.window[provider] and self._affordable(provider):
            _, _, priority, job = heapq.heappop(h)
            live.add(job.job_id)
            if self.on_admit is not None:
                await self.on_admit(job, priority)
            self.max_inflight[provider] = max(self.max_inflight[provider], len(live))
            await self.q.enqueue(job)

    async def release(self, provider: str, job_id: str, cost_usd: float, paid: bool) -> None:
        """A result arrived. Duplicates (job not in flight) change nothing."""
        live = self.inflight.get(provider)
        if live is None or job_id not in live:
            return
        live.discard(job_id)
        if paid:
            self.spent += cost_usd
            n = self._n_cost[provider] = self._n_cost[provider] + 1
            self.exp_cost[provider] += (cost_usd - self.exp_cost[provider]) / n
        for p in self.heap:                      # the budget is shared, so any lane may unblock
            await self._drain(p)

    def idle(self) -> bool:
        return not any(self.inflight.values())

    def budget_exhausted(self) -> bool:
        """Work is waiting, nothing is in flight, and nothing more is affordable."""
        return self.budget is not None and self.idle() and any(self.heap.values()) and \
            not any(self._affordable(p) for p, h in self.heap.items() if h)
