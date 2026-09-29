"""Design C scheduler: rank models by allocating calls across them jointly.

Reuses Design B's workers, Redis job queues, Postgres event log and cache keys unchanged.
What changes is who decides the next call. In B every session picks its own next item;
here one allocator (adaptive_eval.c.ranking.choose) picks the (model, item) that most
reduces the expected number of misordered pairs, per expected dollar.

Concurrency rules:
  * at most one call in flight per model. A second in-flight call would be chosen as if
    the first had not happened, and both would be priced off the same stale variance.
  * per-provider windows (as in B's Admission) cap calls in flight per provider, and
    --max-inflight caps the total. Free slots go to the best models by gain.
  * a free slot is left idle when its best call is worth less than --min-relative-gain
    (default 0.5) of the best call anywhere; that call will be available when its
    provider frees up. Filling every slot regardless cost ~0.1 tau on sp6 (see
    scripts/c_concurrency_experiment.py). 0.5 was set once, not tuned.
  * one session per model ("<run>:<model>"); a model's step n is its n-th answer, so B's
    (session, step, type) idempotency keys carry over unchanged.

Recovery. State is a pure function of the event log: each model's answers are rebuilt
from answer_recorded events, pending selections are re-enqueued under the same job ids,
and the allocator resumes from the rebuilt state. What is NOT reproducible under
concurrency is the order results arrive in, so two clean runs can already differ; every
decision is therefore logged as an 'allocation' event (global seq, gain, expected
discordant pairs at the time) and can be audited or replayed. With --max-inflight 1 the
run is a pure function of the answers, and a crashed-and-resumed run equals a clean one.

Stopping: when the expected number of discordant pairs falls below --stop-discordant,
when --max-calls or --budget-usd would be exceeded, or when no call has positive gain.
Scores are written to sessions.theta (pool score) and sessions.se (its sd).

    python -m adaptive_eval.c.scheduler --run-name c-v1 --bank data/sp6_bank2d.json \\
        --data data/sp6_matrix.json --max-calls 800
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .. import data as D
from ..providers import DEFAULT_PROVIDERS
from ..real_provider import item_hashes, load_items
from ..storage import ResponseCache
from ..b import pgstore
from ..b.queue import SCHEDULER, Job, JobQueue, connect, real_job_id
from ..b.scheduler import SchedulerCrash, redis_state_lost
from ..b.sim import add_args, provider_configs
from .ranking import (Bank2D, ModelState, choose, expected_discordant, score,
                      variance_after)

DESIGN = "C-ranking"


@dataclass
class Lane:
    """One model being ranked."""
    session_id: str
    model: str
    provider: str
    answered: dict[int, int] = field(default_factory=dict)   # bank index -> correct
    pending: tuple[int, int] | None = None                    # (step, bank index)
    failed: bool = False
    state: ModelState | None = None
    v_new: np.ndarray | None = None


class RankingScheduler:
    def __init__(self, run_name: str, models: list[str], model_provider: dict[str, str],
                 bank: Bank2D, pool, q: JobQueue, provider_cfgs: dict, *,
                 w_count: float = 0.5, available: dict[str, set[int]] | None = None,
                 windows: dict[str, int] | None = None, max_inflight: int | None = None,
                 max_calls: int | None = None, budget_usd: float | None = None,
                 stop_discordant: float = 0.5, cost_aware: bool = True,
                 min_relative_gain: float = 0.5,
                 prompt_version: str = "v1", decoding: str = "temp0", sample_idx: int = 0,
                 item_hash: dict[str, str] | None = None, consumer: str = "sched-c",
                 status_every_s: float = 5.0, log=print):
        self.run_name, self.models, self.model_provider = run_name, list(models), model_provider
        self.bank, self.idx, self.pool, self.q = bank, bank.index(), pool, q
        self.w = bank.weights(w_count)
        self.w_count = w_count
        self.store = pgstore.PgEventStore(pool)
        self.available, self.cfgs = available, provider_cfgs
        self.windows = {p: (windows or {}).get(p) or max(1, int(np.ceil(
            c.rpm / 60 * (sum(c.latency) / 2) * 1.5))) for p, c in provider_cfgs.items()}
        self.max_inflight, self.max_calls, self.budget = max_inflight, max_calls, budget_usd
        self.stop_discordant, self.cost_aware = stop_discordant, cost_aware
        self.min_relative_gain = min_relative_gain
        self.prompt_version, self.decoding, self.sample_idx = prompt_version, decoding, sample_idx
        self.item_hash, self.consumer = item_hash or {}, consumer
        self.status_every_s, self.log = status_every_s, log
        self.lanes: dict[str, Lane] = {}
        self.order: list[str] = []            # lane order = allocator's model index
        self.seq = 0                          # global allocation counter (from the log)
        self.spent = 0.0
        self._cost_sum: dict[str, float] = {}
        self._cost_n: dict[str, int] = {}
        self.stop_reason: str | None = None
        self.stats = dict(results=0, stale_results=0, reenqueued=0, failed=0, allocations=0,
                          max_inflight_total=0, slots_left_idle=0)

    # ---- helpers -----------------------------------------------------------------
    def cache_key(self, model: str, item_id: str) -> str:
        return ResponseCache.make_key(model=model, item=item_id, prompt=self.prompt_version,
                                      decoding=self.decoding, sample=self.sample_idx,
                                      item_hash=self.item_hash.get(item_id))

    def expected_cost(self, provider: str) -> float:
        if self._cost_n.get(provider):
            return self._cost_sum[provider] / self._cost_n[provider]
        c = self.cfgs.get(provider) or DEFAULT_PROVIDERS.get(provider)
        return 0.002 if c is None else c.est_tokens_per_call / 1000 * c.usd_per_1k_input

    def _inflight(self) -> list[Lane]:
        return [ln for ln in self.lanes.values() if ln.pending is not None]

    def _committed(self) -> float:
        return self.spent + sum(self.expected_cost(ln.provider) for ln in self._inflight())

    def _refresh(self, ln: Lane) -> None:
        ln.state = score(self.bank, self.w, ln.answered)
        v = variance_after(self.bank, self.w, ln.answered, ln.state)
        if self.available is not None:
            allowed = self.available.get(ln.model, set())
            mask = np.ones(len(self.bank), bool)
            mask[list(allowed)] = False
            v[mask] = np.inf
        ln.v_new = v

    def exp_discordant(self) -> float:
        live = [self.lanes[s] for s in self.order if not self.lanes[s].failed]
        return expected_discordant([ln.state for ln in live]) if len(live) > 1 else 0.0

    def calls_made(self) -> int:
        return sum(len(ln.answered) for ln in self.lanes.values()) + len(self._inflight())

    # ---- allocation --------------------------------------------------------------
    async def _fill(self) -> None:
        """Admit calls while there are free slots and a call is worth making."""
        while True:
            live = [self.lanes[s] for s in self.order if not self.lanes[s].failed]
            if len(live) < 2:
                self.stop_reason = self.stop_reason or "fewer than 2 models left"
                return
            if self.exp_discordant() < self.stop_discordant:
                self.stop_reason = self.stop_reason or "ranking settled"
                return
            if self.max_calls is not None and self.calls_made() >= self.max_calls:
                self.stop_reason = self.stop_reason or "max calls"
                return
            inflight = self._inflight()
            if self.max_inflight is not None and len(inflight) >= self.max_inflight:
                return
            busy = {p: 0 for p in self.windows}
            for ln in inflight:
                busy[ln.provider] = busy.get(ln.provider, 0) + 1
            cost = np.array([self.expected_cost(ln.provider) if self.cost_aware else 1.0
                             for ln in live])
            vn = []
            for ln in live:
                ok = ln.pending is None and busy.get(ln.provider, 0) < self.windows.get(
                    ln.provider, 1) and (self.budget is None or self._committed()
                                         + self.expected_cost(ln.provider) <= self.budget)
                vn.append(ln.v_new if ok else np.full(len(self.bank), np.inf))
            states = [ln.state for ln in live]
            m, i, gain = choose(states, vn, cost)
            if inflight and self.min_relative_gain > 0 and m >= 0:
                # the best call anywhere, ignoring windows and budget. If the free slot's
                # best is much worse, wait: that call will be available when a slot frees.
                free = [ln.v_new if ln.pending is None else np.full(len(self.bank), np.inf)
                        for ln in live]
                best_any = choose(states, free, cost)[2]
                if gain < self.min_relative_gain * best_any:
                    self.stats["slots_left_idle"] += 1
                    return
            if m < 0 or not gain > 0:
                if not inflight:
                    if self.budget is not None and any(
                            ln.pending is None and self._committed() + self.expected_cost(
                                ln.provider) > self.budget for ln in live):
                        self.stop_reason = self.stop_reason or "budget"
                    else:
                        self.stop_reason = self.stop_reason or "no call has positive gain"
                return
            await self._dispatch(live[m], i, gain)

    async def _dispatch(self, ln: Lane, i: int, gain: float) -> None:
        assert ln.pending is None and i not in ln.answered, "one call per model, no repeats"
        step, item_id = len(ln.answered), self.bank.item_ids[i]
        await self.store.append(ln.session_id, step, "item_selected", item_id)  # write-ahead
        await self.store.append(ln.session_id, step, "allocation", item_id, detail=json.dumps(
            {"design": DESIGN, "seq": self.seq, "gain_per_usd": gain,
             "expected_discordant": self.exp_discordant(), "provider": ln.provider}))
        self.seq += 1
        self.stats["allocations"] += 1
        ln.pending = (step, i)
        await self._enqueue(ln)

    async def _enqueue(self, ln: Lane) -> None:
        step, i = ln.pending
        item_id = self.bank.item_ids[i]
        await self.q.enqueue(Job(real_job_id(ln.session_id, step), ln.provider, ln.model,
                                 item_id, self.cache_key(ln.model, item_id), False))
        n = len(self._inflight())
        self.stats["max_inflight_total"] = max(self.stats["max_inflight_total"], n)

    # ---- results -----------------------------------------------------------------
    async def on_result(self, f: dict) -> None:
        sid, _, step = f["job_id"].rpartition(":")
        ln = self.lanes.get(sid)
        if ln is None or ln.failed or ln.pending is None or ln.pending[0] != int(step):
            self.stats["stale_results"] += 1
            return
        if f.get("error"):
            await self.pool.execute("UPDATE sessions SET status='failed', finished_at=$1"
                                    " WHERE session_id=$2", time.time(), sid)
            ln.failed, ln.pending = True, None
            self.stats["failed"] += 1
            self.log(f"[sched-c] {sid} failed at step {step}: {f['error']}")
            await self._fill()
            return
        _, i = ln.pending
        cached = int(f["cached"])
        cost = 0.0 if cached else float(f["cost_usd"])
        wrote = await self.store.append(sid, int(step), "answer_recorded", self.bank.item_ids[i],
                                        correct=int(f["correct"]), cached=cached, cost=cost)
        if not wrote:
            self.stats["stale_results"] += 1
            return
        ln.answered[i] = int(f["correct"])
        ln.pending = None
        self.stats["results"] += 1
        if not cached:
            self.spent += cost
            self._cost_sum[ln.provider] = self._cost_sum.get(ln.provider, 0.0) + cost
            self._cost_n[ln.provider] = self._cost_n.get(ln.provider, 0) + 1
        self._refresh(ln)
        await self._fill()

    # ---- startup, recovery, main loop --------------------------------------------
    async def recover(self) -> None:
        await self.q.ensure_groups()
        config = {"design": DESIGN, "w_count": self.w_count, "stop_discordant":
                  self.stop_discordant, "prompt_version": self.prompt_version}
        for m in self.models:
            sid = f"{self.run_name}:{m}"
            p = self.model_provider[m]
            await self.store.ensure_session(sid, self.run_name, m, p, config)
            status = await self.pool.fetchval("SELECT status FROM sessions WHERE session_id=$1",
                                              sid)
            st = await self.store.load_state(sid)
            ln = Lane(sid, m, p, {self.idx[it]: int(c) for it, c in st.answered},
                      failed=status == "failed")
            if st.pending is not None and not ln.failed:
                ln.pending = (st.pending[0], self.idx[st.pending[1]])
            self._refresh(ln)
            self.lanes[sid] = ln
            self.order.append(sid)
        self.seq = int(await self.pool.fetchval(
            "SELECT COUNT(*) FROM events e JOIN sessions s USING (session_id)"
            " WHERE s.run_name=$1 AND e.type='allocation'", self.run_name))
        for ln in self._inflight():             # selected, never answered: re-issue
            await self._enqueue(ln)
            self.stats["reenqueued"] += 1
        start = "0-0"                           # results received but never acked
        while True:
            nxt, _, *_ = await self.q.r.xautoclaim(self.q.results, SCHEDULER, self.consumer, 0,
                                                   start_id=start, count=500)
            if nxt in ("0-0", b"0-0"):
                break
            start = nxt

    async def _handle_batch(self, batch, exit_after: int | None) -> None:
        for msg_id, f in batch:
            if f:
                await self.on_result(f)
            await self.q.ack_result(msg_id)
            if exit_after is not None and self.stats["results"] >= exit_after:
                raise SchedulerCrash(f"simulated crash after {exit_after} results")

    async def run(self, exit_after: int | None = None) -> dict:
        t0 = time.time()
        await self.recover()
        self.log(f"[sched-c] {self.run_name}: {len(self.lanes)} models, "
                 f"{self.calls_made()} calls already logged, {self.stats['reenqueued']} re-enqueued")
        while True:                              # the backlog this consumer already owns
            r = await self.q.r.xreadgroup(SCHEDULER, self.consumer, {self.q.results: "0"},
                                          count=200)
            batch = [(mid, f) for _, entries in (r or []) for mid, f in entries]
            if not batch:
                break
            await self._handle_batch(batch, exit_after)
        await self._fill()
        last = time.monotonic()
        while self._inflight():
            batch = await self.q.read_results(self.consumer, count=200, block_ms=1000)
            await self._handle_batch(batch, exit_after)
            if time.monotonic() - last >= self.status_every_s:
                last = time.monotonic()
                self.log(f"[sched-c] {self.calls_made()} calls, expected discordant pairs "
                         f"{self.exp_discordant():.2f}")
        await self._finish()
        return self.summary(time.time() - t0)

    async def _finish(self) -> None:
        for ln in self.lanes.values():
            if not ln.failed:
                await self.store.finish(ln.session_id, ln.state.s, float(np.sqrt(ln.state.v)),
                                        len(ln.answered))

    def ranking(self) -> list[dict]:
        rows = [{"model": ln.model, "score": round(ln.state.s, 4),
                 "sd": round(float(np.sqrt(ln.state.v)), 4), "calls": len(ln.answered)}
                for ln in self.lanes.values() if not ln.failed]
        return sorted(rows, key=lambda r: -r["score"])

    def summary(self, wall_s: float) -> dict:
        return {"run": self.run_name, "design": DESIGN, "stop_reason": self.stop_reason,
                "calls": self.calls_made(), "spent_usd": round(self.spent, 4),
                "expected_discordant_pairs": round(self.exp_discordant(), 3),
                "wall_clock_s": round(wall_s, 2), "ranking": self.ranking(), **self.stats}


def available_items(d: dict, bank: Bank2D) -> dict[str, set[int]]:
    """Replay mode: per model, the bank items that have a logged answer."""
    didx = {it: j for j, it in enumerate(d["items"])}
    return {m: {k for k, it in enumerate(bank.item_ids)
                if it in didx and not np.isnan(d["R"][i, didx[it]])}
            for i, m in enumerate(d["models"])}


async def amain(a) -> None:
    d = D.load(a.data)
    bank, _ = Bank2D.load(a.bank)
    models = d["models"] if a.models == "all" else [m.strip() for m in a.models.split(",")]
    unknown = [m for m in models if m not in d["models"]]
    if unknown:
        sys.exit(f"not in the data file: {unknown}")
    r = connect(a.redis_url)
    pool = await pgstore.connect(a.pg_dsn, a.schema)
    cfgs = provider_configs(a.rate_scale, a.latency_scale)
    q = JobQueue(r, sorted(set(d["model_provider"].values())), a.prefix)
    sched = RankingScheduler(
        a.run_name, models, d["model_provider"], bank, pool, q, cfgs, w_count=a.w_count,
        available=available_items(d, bank) if a.replay else None,
        max_inflight=a.max_inflight, max_calls=a.max_calls, budget_usd=a.budget_usd,
        stop_discordant=a.stop_discordant, min_relative_gain=a.min_relative_gain,
        item_hash=item_hashes(load_items(a.items)) if a.items else None)
    try:
        if a.exit_after is not None:
            try:
                await sched.run(exit_after=a.exit_after)
            except SchedulerCrash as e:
                print(f"[sched-c] {e}; exiting without cleanup", file=sys.stderr)
                os._exit(1)
        print(json.dumps(await sched.run(), indent=2))
    except Exception as e:
        if redis_state_lost(e):
            print("[sched-c] Redis state was lost. Run the same command again: it re-enqueues"
                  " everything pending from Postgres.", file=sys.stderr)
            sys.exit(3)
        raise
    finally:
        await pool.close()
        await r.aclose()


def main() -> None:
    p = argparse.ArgumentParser(prog="adaptive_eval.c.scheduler")
    p.add_argument("--run-name", required=True)
    p.add_argument("--bank", required=True, help="Bank2D JSON (scripts/make_bank2d.py)")
    p.add_argument("--data", required=True, help="models and model_provider (and, with"
                   " --replay, the logged answers workers replay)")
    p.add_argument("--models", default="all", help="'all' or a comma-separated list")
    p.add_argument("--replay", action="store_true",
                   help="only ask items that have a logged answer for that model")
    p.add_argument("--w-count", type=float, default=0.5)
    p.add_argument("--stop-discordant", type=float, default=0.5,
                   help="stop when expected misordered pairs fall below this")
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--budget-usd", type=float, default=None)
    p.add_argument("--min-relative-gain", type=float, default=0.5,
                   help="leave a free slot idle if its best call is worth less than this"
                        " fraction of the best call anywhere (0 = always fill)")
    p.add_argument("--max-inflight", type=int, default=None,
                   help="total calls in flight; 1 makes the run a pure function of answers")
    add_args(p)
    p.add_argument("--items", default=None)
    p.add_argument("--exit-after", type=int, default=None)
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
