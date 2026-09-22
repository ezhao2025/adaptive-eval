"""Design B speculative prefetch.

While session s waits on item i, both possible outcomes are known in advance:
  theta+ = ability if i comes back correct,  theta- = if it comes back wrong
and item selection is a pure function of the answers so far (A kept it deterministic), so
  next+ = the scheduler's choice after "correct",  next- = after "wrong"
are exactly the items the scheduler WILL pick. Fetching them early turns the real next call
into a cache hit.

Speculative jobs only warm the cache. They never write session events, so turning
speculation on or off changes speed and cost, never results.

Guards, all cheap and conservative:
  * skip a candidate that is already cached, in flight, or already speculated
  * enqueue only if the provider's shared bucket has >= min_free of capacity left
    (a read-only peek; workers re-check before calling, and real lanes are read first)
  * hard cap: speculative spend <= max_fraction of total spend (real + speculative)
"""
from __future__ import annotations

from .queue import Job, spec_job_id


class Speculator:
    def __init__(self, sched, limiters: dict, cache, *, min_free: float = 0.3,
                 max_fraction: float = 0.2):
        if not 0 < max_fraction < 1:
            raise ValueError("max_fraction must be in (0, 1)")
        self.sched, self.limiters, self.cache = sched, limiters, cache
        self.min_free, self.max_fraction = min_free, max_fraction
        self.enqueued: set[str] = set()          # cache keys we speculated
        self.committed_usd = 0.0                 # expected cost of everything we enqueued
        self.stats = dict(spec_enqueued=0, spec_skip_cached=0, spec_skip_inflight=0,
                          spec_skip_headroom=0, spec_skip_cap=0, spec_skip_stop=0)

    def _within_cap(self, cost: float) -> bool:
        spec = self.committed_usd + cost
        ok = spec <= self.max_fraction * (self.sched.real_spent + spec)
        adm = self.sched.admission
        if ok and adm is not None and adm.budget is not None:      # budget covers spec too
            ok = adm._committed() + cost <= adm.budget
        return ok

    async def on_real_job(self, s, step: int, item_id: str) -> None:
        """Called right after the real job for (s, step, item_id) is handed to admission."""
        provider = self.sched.model_provider[s.model]
        for correct in (1, 0):
            answered = s.st.answered + [(item_id, correct)]
            nxt = self.sched.plan_next(s.session_id, s.model, answered)
            if nxt is None:                      # the session would stop on this branch
                self.stats["spec_skip_stop"] += 1
                continue
            key = self.sched.cache_key(s.model, nxt)
            if key in self.enqueued:
                continue
            if await self.sched.q.r.exists(f"{self.sched.q.prefix}inflight:{key}"):
                self.stats["spec_skip_inflight"] += 1
                continue
            if await self.cache.get(key) is not None:
                self.stats["spec_skip_cached"] += 1
                continue
            if await self.limiters[provider].headroom() < self.min_free:
                self.stats["spec_skip_headroom"] += 1
                continue
            cost = self.sched.expected_cost(provider)
            if not self._within_cap(cost):
                self.stats["spec_skip_cap"] += 1
                continue
            self.enqueued.add(key)
            self.committed_usd += cost
            if self.sched.admission is not None:
                self.sched.admission.extra_committed += cost
            self.stats["spec_enqueued"] += 1
            await self.sched.q.enqueue(Job(spec_job_id(key), provider, s.model, nxt, key, True))
