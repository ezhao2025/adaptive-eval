"""Design B scheduler: one asyncio process that owns every session's logic.

It reuses A's IRT code unchanged. Workers only answer (model, item) jobs; everything
adaptive -- theta, SE, stopping, item selection -- happens here, driven by the results
stream. Session state lives only in the Postgres event log, exactly as in A.

Single point of failure BY DESIGN: the scheduler is recoverable, not highly available.
Kill it at any moment and restart it: it rebuilds every session from the log,
re-enqueues pending steps (same job ids, so duplicates are harmless) and picks up
results it had received but not yet acknowledged.

    python -m adaptive_eval.b.scheduler --run-name b-v1 --data data/synthetic.json \\
        --params data/irt_params.json --se-target 0.3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import kendalltau

from .. import data as D
from ..engine import SessionConfig
from ..irt import ItemBank, estimate_ability, fisher_information, select_next
from ..providers import DEFAULT_PROVIDERS
from ..report import full_reference
from ..storage import ResponseCache, SessionState
from . import pgstore
from .admission import Admission, expected_se_reduction
from .queue import SCHEDULER, Job, JobQueue, connect, real_job_id


class SchedulerCrash(Exception):
    """Raised by --exit-after in tests to simulate the process dying mid-run."""


@dataclass
class Session:
    session_id: str
    model: str
    st: SessionState
    theta: float = 0.0
    se: float = float("inf")
    done: bool = False


def available_items(d: dict, bank: ItemBank) -> dict[str, set[int]]:
    """Per model, the bank items that have a logged answer (replay mode)."""
    bidx, didx = bank.index(), {it: j for j, it in enumerate(d["items"])}
    return {m: {bidx[it] for it in bank.item_ids if not np.isnan(d["R"][i, didx[it]])}
            for i, m in enumerate(d["models"])}


class Scheduler:
    def __init__(self, run_name: str, models: list[str], model_provider: dict[str, str],
                 bank: ItemBank, pool, q: JobQueue, cfg: SessionConfig, *,
                 available: dict[str, set[int]] | None = None, consumer: str = "sched",
                 admission: Admission | None = None, status_every_s: float = 5.0, log=print):
        self.run_name, self.models, self.model_provider = run_name, list(models), model_provider
        self.bank, self.idx, self.pool, self.q, self.cfg = bank, bank.index(), pool, q, cfg
        self.store = pgstore.PgEventStore(pool)
        self.available, self.consumer, self.admission = available, consumer, admission
        self.status_every_s, self.log = status_every_s, log
        self.sessions: dict[str, Session] = {}
        self.stats = dict(results=0, stale_results=0, reenqueued=0, failed=0, budget_stopped=0)

    # ---- per-session logic (A's engine, split at the network boundary) -------------
    def _ability(self, s: Session) -> None:
        ii = np.array([self.idx[i] for i, _ in s.st.answered], dtype=int)
        y = np.array([c for _, c in s.st.answered], dtype=float)
        s.theta, s.se = estimate_ability(self.bank.a[ii], self.bank.b[ii], y)

    def _allowed(self, model: str) -> set[int] | None:
        return None if self.available is None else self.available.get(model, set())

    async def admit(self, job: Job, priority: float) -> None:
        """Hand a job to the workers, directly or through windowed priority admission."""
        if self.admission is None:
            await self.q.enqueue(job)
        else:
            await self.admission.admit(job, priority)

    async def _enqueue(self, s: Session, step: int, item_id: str) -> None:
        key = ResponseCache.make_key(model=s.model, item=item_id, prompt=self.cfg.prompt_version,
                                     decoding=self.cfg.decoding, sample=self.cfg.sample_idx)
        provider = self.model_provider[s.model]
        i = self.idx[item_id]
        gain = expected_se_reduction(
            s.se, float(fisher_information(s.theta, self.bank.a[i:i + 1], self.bank.b[i:i + 1])[0]))
        await self.admit(Job(real_job_id(s.session_id, step), provider, s.model, item_id, key,
                             False), priority=self._priority(s, i, provider, gain))

    def _priority(self, s: Session, item: int, provider: str, gain: float) -> float:
        if self.admission is None:
            return 0.0
        if self.admission.mode == "nearest":
            # fewest calls left to finish = highest priority: finish sessions, don't spread
            info_item = float(fisher_information(s.theta, self.bank.a[item:item + 1],
                                                 self.bank.b[item:item + 1])[0])
            info_needed = max(0.0, 1 / self.cfg.se_target ** 2 - 1 / s.se ** 2)
            calls_left = max(self.cfg.min_items - len(s.st.answered),
                             info_needed / max(info_item, 1e-9))
            return -calls_left
        return gain / self.admission.expected_cost(provider)     # SE reduction per dollar

    async def _advance(self, s: Session) -> None:
        """Stop, or select the next item, log it (write-ahead) and enqueue it."""
        n, allowed = len(s.st.answered), self._allowed(s.model)
        pool_size = len(self.bank) if allowed is None else len(allowed)
        if n >= min(self.cfg.max_items, pool_size) or (
                n >= self.cfg.min_items and s.se < self.cfg.se_target):
            await self.store.finish(s.session_id, s.theta, s.se, n)
            s.done = True
            return
        used = {self.idx[i] for i, _ in s.st.answered}
        item_id = self.bank.item_ids[select_next(self.cfg.selector, s.theta, self.bank, used,
                                                 s.session_id, n, allowed)]
        await self.store.append(s.session_id, n, "item_selected", item_id)
        s.st.pending = (n, item_id)
        await self._enqueue(s, n, item_id)

    async def on_result(self, f: dict) -> None:
        sid, _, step = f["job_id"].rpartition(":")
        s = self.sessions.get(sid)
        if s is not None and self.admission is not None:    # free the window slot first
            paid = not f.get("error") and f.get("cached") == "0"
            await self.admission.release(self.model_provider[s.model], f["job_id"],
                                         float(f.get("cost_usd") or 0.0), paid)
        # duplicates (reclaimed or re-enqueued jobs) and other runs' results are dropped
        if s is None or s.done or s.st.pending is None or s.st.pending[0] != int(step):
            self.stats["stale_results"] += 1
            return
        if f.get("error"):
            await self.pool.execute("UPDATE sessions SET status='failed', finished_at=$1"
                                    " WHERE session_id=$2", time.time(), sid)
            s.done = True
            self.stats["failed"] += 1
            self.log(f"[sched] {sid} failed at step {step}: {f['error']}")
            return
        _, item_id = s.st.pending
        cached = int(f["cached"])
        wrote = await self.store.append(sid, int(step), "answer_recorded", item_id,
                                        correct=int(f["correct"]), cached=cached,
                                        cost=0.0 if cached else float(f["cost_usd"]))
        if not wrote:                               # already in the log: not new information
            self.stats["stale_results"] += 1
            return
        s.st.answered.append((item_id, int(f["correct"])))
        s.st.pending = None
        self.stats["results"] += 1
        self._ability(s)
        await self._advance(s)

    # ---- startup, recovery, main loop --------------------------------------------
    async def recover(self) -> None:
        await self.q.ensure_groups()
        for m in self.models:
            sid = f"{self.run_name}:{m}"
            await self.store.ensure_session(sid, self.run_name, m, self.model_provider[m],
                                            asdict(self.cfg))
            status = await self.pool.fetchval("SELECT status FROM sessions WHERE session_id=$1",
                                              sid)
            if status in ("done", "failed", "budget"):
                continue
            s = Session(sid, m, await self.store.load_state(sid))
            self._ability(s)
            self.sessions[sid] = s
            if s.st.pending is not None:            # selected, never answered: re-issue it
                await self._enqueue(s, *s.st.pending)
                self.stats["reenqueued"] += 1
            else:
                await self._advance(s)
        # results a previous scheduler received but never acked: take them over
        start = "0-0"
        while True:
            nxt, _, *_ = await self.q.r.xautoclaim(self.q.results, SCHEDULER, self.consumer, 0,
                                                   start_id=start, count=500)
            if nxt in ("0-0", b"0-0"):
                break
            start = nxt

    async def _handle_batch(self, batch, exit_after: int | None) -> None:
        for msg_id, f in batch:
            if f:                                   # empty = deleted entry
                await self.on_result(f)
            await self.q.ack_result(msg_id)         # ack only after the event is written
            if exit_after is not None and self.stats["results"] >= exit_after:
                raise SchedulerCrash(f"simulated crash after {exit_after} results")

    def pending_sessions(self) -> int:
        return sum(not s.done for s in self.sessions.values())

    async def run(self, exit_after: int | None = None) -> dict:
        t0 = time.time()
        await self.recover()
        self.log(f"[sched] {self.run_name}: {len(self.sessions)} sessions to run"
                 f" ({self.stats['reenqueued']} re-enqueued)")
        while True:                                 # first the backlog we own, then new ones
            r = await self.q.r.xreadgroup(SCHEDULER, self.consumer, {self.q.results: "0"},
                                          count=200)
            batch = [(mid, f) for _, entries in (r or []) for mid, f in entries]
            if not batch:
                break
            await self._handle_batch(batch, exit_after)
        last_status = time.monotonic()
        while self.pending_sessions():
            if self.admission is not None and self.admission.budget_exhausted():
                await self._stop_for_budget()
                break
            batch = await self.q.read_results(self.consumer, count=200, block_ms=1000)
            await self._handle_batch(batch, exit_after)
            if time.monotonic() - last_status >= self.status_every_s:
                last_status = time.monotonic()
                self.log(f"[sched] {self.run_name}: {len(self.sessions) - self.pending_sessions()}"
                         f"/{len(self.sessions)} done, {self.stats['results']} answers")
        return await self.summary(time.time() - t0)

    async def _stop_for_budget(self) -> None:
        """Budget spent: freeze every unfinished session at its current estimate."""
        for s in self.sessions.values():
            if not s.done:
                await self.pool.execute(
                    "UPDATE sessions SET status='budget', theta=$1, se=$2, n_items=$3,"
                    " finished_at=$4 WHERE session_id=$5",
                    s.theta, s.se, len(s.st.answered), time.time(), s.session_id)
                s.done = True
                self.stats["budget_stopped"] += 1
        self.log(f"[sched] {self.run_name}: budget exhausted, "
                 f"{self.stats['budget_stopped']} sessions stopped early")

    async def summary(self, wall_s: float) -> dict:
        row = await self.pool.fetchrow(
            "SELECT COUNT(*) FILTER (WHERE s.status='done') AS done,"
            " COUNT(*) FILTER (WHERE s.status='failed') AS failed,"
            " COUNT(*) FILTER (WHERE s.se < $2) AS reached_se_target FROM sessions s"
            " WHERE s.run_name=$1", self.run_name, self.cfg.se_target)
        ev = await self.pool.fetchrow(
            "SELECT COUNT(*) AS answers, COUNT(*) FILTER (WHERE e.cached=0) AS paid,"
            " COALESCE(SUM(e.cost_usd), 0) AS cost FROM events e JOIN sessions s"
            " USING (session_id) WHERE s.run_name=$1 AND e.type='answer_recorded'", self.run_name)
        extra = {} if self.admission is None else {
            "admission": self.admission.mode, "budget_usd": self.admission.budget,
            "spent_usd": round(self.admission.spent, 4), "windows": self.admission.window,
            "max_inflight": self.admission.max_inflight}
        return {"run": self.run_name, "done": row["done"], "failed": row["failed"],
                "reached_se_target": row["reached_se_target"], **extra,
                "answers": ev["answers"], "paid_calls": ev["paid"],
                "cost_usd": round(float(ev["cost"]), 4), "wall_clock_s": round(wall_s, 2),
                **self.stats}


async def amain(a) -> None:
    d = D.load(a.data)
    bank, meta = ItemBank.load(a.params)
    models = meta["test_models"] if a.models == "test" else d["models"]
    cfg = SessionConfig(selector=a.selector, se_target=a.se_target, max_items=a.max_items)
    r = connect(a.redis_url)
    pool = await pgstore.connect(a.pg_dsn, a.schema)
    q = JobQueue(r, sorted(set(d["model_provider"].values())), a.prefix)
    admission = None if a.admission == "none" else Admission(
        q, DEFAULT_PROVIDERS, mode=a.admission, window_scale=a.window_scale,
        budget_usd=a.budget_usd)
    sched = Scheduler(a.run_name, models, d["model_provider"], bank, pool, q, cfg,
                      available=available_items(d, bank), admission=admission)
    try:
        if a.exit_after is not None:
            try:
                await sched.run(exit_after=a.exit_after)
            except SchedulerCrash as e:
                print(f"[sched] {e}; exiting without cleanup", file=sys.stderr)
                os._exit(1)                         # like kill -9: nothing gets flushed
        summary = await sched.run()
        rows = await pool.fetch("SELECT model, theta FROM sessions WHERE run_name=$1"
                                " AND status IN ('done', 'budget')", a.run_name)
        if len(rows) > 1:                   # ranking agreement with the full benchmark
            ref = full_reference(d, bank, [r["model"] for r in rows])
            summary["kendall_tau_vs_full_theta"] = round(float(kendalltau(
                [r["theta"] for r in rows], [ref[r["model"]]["theta"] for r in rows])[0]), 4)
        print(json.dumps(summary, indent=2))
    finally:
        await pool.close()
        await r.aclose()


def main() -> None:
    p = argparse.ArgumentParser(prog="adaptive_eval.b.scheduler")
    p.add_argument("--run-name", required=True)
    p.add_argument("--data", default="data/synthetic.json")
    p.add_argument("--params", default="data/irt_params.json")
    p.add_argument("--selector", default="max_info")
    p.add_argument("--se-target", type=float, default=0.30)
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--models", default="test", help="'test' (held-out models) or 'all'")
    p.add_argument("--admission", choices=["priority", "nearest", "fifo", "none"],
                   default="priority")
    p.add_argument("--budget-usd", type=float, default=None,
                   help="stop admitting paid calls once this much is spent or committed")
    p.add_argument("--window-scale", type=float, default=1.5)
    p.add_argument("--exit-after", type=int, default=None,
                   help="simulate a crash after this many answers (fault testing)")
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
