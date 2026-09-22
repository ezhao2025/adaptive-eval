"""Step 7 experiment: admission rules under a tight USD budget.

Arms: fifo (baseline), priority (SE reduction per dollar: breadth),
nearest (fewest calls left to reach the SE target: depth).

For each arm and repeat, starts fresh workers + a scheduler on an isolated Postgres schema
and Redis prefix (so no arm can reuse another arm's cached answers), then reports
(a) sessions that reached the SE target and (b) Kendall tau vs the full benchmark,
both measured at budget exhaustion.

    python scripts/admission_experiment.py --budget-usd 0.8 --reps 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import uuid

import asyncpg


def run_arm(a, mode: str, rep: int) -> dict:
    tag = f"exp_{mode}_{rep}_{uuid.uuid4().hex[:6]}"
    env = dict(os.environ)
    common = ["--schema", tag, "--prefix", f"{tag}:"]
    workers = [subprocess.Popen([sys.executable, "-m", "adaptive_eval.b.worker", "--id", f"w{i}",
                                 "--data", a.data, *common],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
               for i in range(a.workers)]
    try:
        time.sleep(1.5)                                   # let workers connect
        out = subprocess.run(
            [sys.executable, "-m", "adaptive_eval.b.scheduler", "--run-name", tag,
             "--data", a.data, "--params", a.params, "--models", a.models,
             "--se-target", str(a.se_target), "--admission", mode,
             "--budget-usd", str(a.budget_usd), *common],
            capture_output=True, text=True, env=env, timeout=a.timeout)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-2000:])
        return json.loads(out.stdout[out.stdout.index("{"):])
    finally:
        for w in workers:
            w.terminate()
        for w in workers:
            w.wait(timeout=15)
        asyncio.run(_drop(tag))


async def _drop(schema: str) -> None:
    conn = await asyncpg.connect(os.environ["PG_DSN"])
    await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    await conn.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--data", default="data/synthetic.json")
    p.add_argument("--params", default="data/irt_params.json")
    p.add_argument("--models", default="all", help="'all' gives more sessions competing")
    p.add_argument("--se-target", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--arms", default="fifo,priority,nearest")
    a = p.parse_args()
    if not (os.environ.get("PG_DSN") and os.environ.get("REDIS_URL")):
        sys.exit("set PG_DSN and REDIS_URL (source .env.sh)")

    results = {m: [] for m in a.arms.split(",")}
    for rep in range(a.reps):
        for mode in results:                              # alternate arms within each rep
            r = run_arm(a, mode, rep)
            results[mode].append(r)
            print(f"rep {rep} {mode:8s} reached_target={r['reached_se_target']:3d}/"
                  f"{r['done'] + r['budget_stopped']}  tau={r.get('kendall_tau_vs_full_theta')}"
                  f"  spent=${r['spent_usd']:.3f}", flush=True)

    def ms(xs):
        return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 \
            else f"{xs[0]:.3f}"
    print(f"\nbudget ${a.budget_usd}, {a.reps} reps (mean ± sd)")
    print(f"{'arm':10s} {'reached SE target':>20s} {'Kendall tau':>18s} {'spent USD':>16s}")
    for mode, rs in results.items():
        print(f"{mode:10s} {ms([r['reached_se_target'] for r in rs]):>20s} "
              f"{ms([r['kendall_tau_vs_full_theta'] for r in rs]):>18s} "
              f"{ms([r['spent_usd'] for r in rs]):>16s}")


if __name__ == "__main__":
    main()
