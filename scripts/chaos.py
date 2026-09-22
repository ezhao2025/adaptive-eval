"""Design B fault injection: every scenario must end with results IDENTICAL to a clean run.

For each scenario: start workers + scheduler on an isolated Postgres schema and Redis prefix,
wait a random time, inject the fault, restart what died, let the run finish, then compare
every session's (model, theta, n_items) with a clean run of the same configuration.

    python scripts/chaos.py                          # all scenarios once
    python scripts/chaos.py --scenarios random --repeat 5

Scenarios:
  worker       kill -9 one random worker mid-run, restart it
  all-workers  kill -9 every worker, restart them all
  scheduler    kill -9 the scheduler, restart it with the same --run-name
  redis-wipe   FLUSHALL mid-run, restart the scheduler (it re-enqueues from Postgres)
  speculate    no fault; speculation ON vs the clean run with it OFF
  random       several random faults of the kinds above at random times

--speculate runs every fault scenario with speculation on (the clean run stays off).

WARNING: redis-wipe runs FLUSHALL on the whole Redis server. Don't run it while anything
else you care about uses the same Redis.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import signal
import subprocess
import sys
import time
import uuid

import asyncpg

SCENARIOS = ["worker", "all-workers", "scheduler", "redis-wipe"]
RUN_NAME = "chaos"          # same run name everywhere, so session ids match the clean run


class Run:
    def __init__(self, a, label: str, speculate: bool = False):
        self.a, self.label, self.speculate = a, label, speculate
        self.tag = f"chaos_{label.replace('-', '_')}_{uuid.uuid4().hex[:6]}"
        self.workers: dict[int, subprocess.Popen] = {}
        self.sched: subprocess.Popen | None = None
        self.sched_restarts = 0
        self.log = open(f"/tmp/{self.tag}.log", "w")

    def _common(self):
        return ["--schema", self.tag, "--prefix", f"{self.tag}:"]

    def start_worker(self, i: int) -> None:
        self.workers[i] = subprocess.Popen(
            [sys.executable, "-m", "adaptive_eval.b.worker", "--id", f"w{i}",
             "--data", self.a.data, "--min-idle-ms", str(self.a.min_idle_ms),
             "--inflight-ttl-s", str(self.a.inflight_ttl_s), *self._common()],
            stdout=self.log, stderr=self.log)

    def start_scheduler(self) -> None:
        self.sched = subprocess.Popen(
            [sys.executable, "-m", "adaptive_eval.b.scheduler", "--run-name", RUN_NAME,
             "--data", self.a.data, "--params", self.a.params, "--models", self.a.models,
             "--se-target", str(self.a.se_target), *self._common(),
             *(["--speculate"] if self.speculate else [])],
            stdout=self.log, stderr=self.log)

    def start(self) -> None:
        for i in range(self.a.workers):
            self.start_worker(i)
        time.sleep(1.0)
        self.start_scheduler()

    def running(self) -> bool:
        return self.sched is not None and self.sched.poll() is None

    # ---- faults ------------------------------------------------------------------
    def kill_worker(self) -> str:
        i = random.choice(list(self.workers))
        self.workers[i].send_signal(signal.SIGKILL)
        self.workers[i].wait()
        time.sleep(0.5)
        self.start_worker(i)
        return f"kill -9 w{i}"

    def kill_all_workers(self) -> str:
        for p in self.workers.values():
            p.send_signal(signal.SIGKILL)
        for p in self.workers.values():
            p.wait()
        time.sleep(0.5)
        for i in list(self.workers):
            self.start_worker(i)
        return "kill -9 all workers"

    def kill_scheduler(self) -> str:
        self.sched.send_signal(signal.SIGKILL)
        self.sched.wait()
        time.sleep(0.3)
        self.start_scheduler()
        self.sched_restarts += 1
        return "kill -9 scheduler"

    def wipe_redis(self) -> str:
        subprocess.run(["redis-cli", "-u", os.environ["REDIS_URL"], "FLUSHALL"],
                       check=True, capture_output=True)
        # the scheduler exits by itself once it notices; don't wait on that
        time.sleep(0.5)
        if self.running():
            self.sched.send_signal(signal.SIGKILL)
            self.sched.wait()
        self.start_scheduler()
        self.sched_restarts += 1
        return "FLUSHALL + restart scheduler"

    def inject(self, kind: str) -> str:
        return {"worker": self.kill_worker, "all-workers": self.kill_all_workers,
                "scheduler": self.kill_scheduler, "redis-wipe": self.wipe_redis}[kind]()

    # ---- finish ------------------------------------------------------------------
    def finish(self) -> None:
        deadline = time.monotonic() + self.a.timeout
        while True:
            rc = self.sched.wait(timeout=max(1, deadline - time.monotonic()))
            if rc == 0:
                return
            # exit 3 = scheduler saw Redis state was lost; anything else = it died.
            # Either way the documented recovery is: run the same command again.
            self.start_scheduler()
            self.sched_restarts += 1

    def stop(self) -> None:
        for p in list(self.workers.values()) + ([self.sched] if self.sched else []):
            if p.poll() is None:
                p.send_signal(signal.SIGKILL)
                p.wait()
        self.log.close()


async def fetch_results(schema: str):
    conn = await asyncpg.connect(os.environ["PG_DSN"])
    try:
        rows = await conn.fetch(f'SELECT model, theta, n_items, status FROM "{schema}".sessions'
                                " WHERE run_name=$1 ORDER BY model", RUN_NAME)
        paid = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema}".call_attempts WHERE status=\'ok\'')
        orphans = await conn.fetchval(
            f'SELECT COUNT(*) FROM (SELECT session_id, step FROM "{schema}".events'
            f" WHERE type='item_selected' EXCEPT SELECT session_id, step FROM \"{schema}\".events"
            " WHERE type='answer_recorded') x")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()
    return ([(r["model"], round(r["theta"], 9), r["n_items"], r["status"]) for r in rows],
            orphans, paid)


def execute(a, label: str, faults: list[str], speculate: bool = False):
    run = Run(a, label, speculate)
    done = []
    try:
        run.start()
        for kind in faults:
            time.sleep(random.uniform(a.min_delay, a.max_delay))
            if not run.running():
                done.append(f"(run already finished; skipped {kind} - lower --max-delay)")
                break
            done.append(run.inject(kind))
        run.finish()
    finally:
        run.stop()
    rows, orphans, paid = asyncio.run(fetch_results(run.tag))
    run.paid = paid
    return rows, orphans, done, run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", default=",".join(SCENARIOS),
                   help=f"comma list from {SCENARIOS + ['speculate', 'random']}")
    p.add_argument("--speculate", action="store_true",
                   help="run the fault scenarios with speculation on")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--data", default="data/synthetic.json")
    p.add_argument("--params", default="data/irt_params.json")
    p.add_argument("--models", default="all")
    p.add_argument("--se-target", type=float, default=0.2, help="lower = longer runs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min-delay", type=float, default=1.0)
    p.add_argument("--max-delay", type=float, default=4.0)
    p.add_argument("--min-idle-ms", type=int, default=3000,
                   help="worker reclaim timeout; short so killed workers' jobs return fast")
    p.add_argument("--inflight-ttl-s", type=int, default=5,
                   help="short so a killed worker's claim on an item expires quickly")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if not (os.environ.get("PG_DSN") and os.environ.get("REDIS_URL")):
        sys.exit("set PG_DSN and REDIS_URL (source .env.sh)")
    random.seed(a.seed)

    t0 = time.monotonic()
    clean, clean_orphans, _, clean_run = execute(a, "clean", [])
    base = clean_run.paid
    print(f"clean run: {len(clean)} sessions, all done={all(r[3] == 'done' for r in clean)},"
          f" orphans={clean_orphans}, {base} paid calls  ({time.monotonic() - t0:.0f}s)")
    extra: dict[str, list[int]] = {}

    failures = 0
    for rep in range(a.repeat):
        for sc in a.scenarios.split(","):
            if sc == "speculate":
                faults, spec = [], True
            else:
                faults = [random.choice(SCENARIOS) for _ in range(3)] if sc == "random" else [sc]
                spec = a.speculate
            rows, orphans, done, run = execute(a, sc, faults, spec)
            done = done or ["none (speculation on)"]
            ok = rows == clean and orphans == 0
            failures += not ok
            if not spec and sc != "speculate":    # speculation adds its own calls
                extra.setdefault(sc, []).append(run.paid - base)
            print(f"{'PASS' if ok else 'FAIL'}  {sc:12s} faults: {'; '.join(done)}"
                  f"  (scheduler restarts: {run.sched_restarts}, extra paid calls:"
                  f" {run.paid - base:+d})")
            if not ok:
                diff = [(c, g) for c, g in zip(clean, rows) if c != g][:5]
                print(f"      orphans={orphans}, first differences (clean, got): {diff}"
                      f"\n      log: /tmp/{run.tag}.log")
    if extra:
        print(f"\nrecovery cost: extra paid calls vs the clean run ({base} calls)")
        for sc, xs in extra.items():
            print(f"  {sc:12s} mean {sum(xs) / len(xs):+.1f}  (runs: {xs})")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
