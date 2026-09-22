"""Step 10.1: throughput vs worker count, at a loose and a tight rate limit.

Each run: N fresh workers (each with a fixed number of concurrent call slots) + a scheduler,
on its own Postgres schema and Redis prefix. Throughput = paid calls / span of their
timestamps in call_attempts. Expect near-linear scaling while workers are the bottleneck,
then a plateau once the shared rate limit binds.

    python scripts/scaling_experiment.py

Defaults: 2 call slots per worker and 4x replay latency (~0.2 s calls), so workers are the
bottleneck. With the raw ~0.05 s replay calls, the single scheduler saturates first and
the linear region is too short to see.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid

import asyncpg


async def _throughput(schema: str) -> tuple[int, float]:
    conn = await asyncpg.connect(os.environ["PG_DSN"])
    try:
        r = await conn.fetchrow(
            f'SELECT COUNT(*) AS n, EXTRACT(EPOCH FROM MAX(ts) - MIN(ts)) AS span'
            f' FROM "{schema}".call_attempts WHERE status=\'ok\'')
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()
    return r["n"], float(r["span"] or 0.0)


def run(a, n_workers: int, rate_scale: float) -> dict:
    tag = f"scale_{n_workers}_{str(rate_scale).replace('.', '_')}_{uuid.uuid4().hex[:6]}"
    knobs = ["--rate-scale", str(rate_scale), "--latency-scale", str(a.latency_scale)]
    common = ["--schema", tag, "--prefix", f"{tag}:"]
    workers = [subprocess.Popen([sys.executable, "-m", "adaptive_eval.b.worker", "--id", f"w{i}",
                                 "--data", a.data, "--concurrency", str(a.slots),
                                 *knobs, *common],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
               for i in range(n_workers)]
    try:
        time.sleep(1.5)
        out = subprocess.run(
            [sys.executable, "-m", "adaptive_eval.b.scheduler", "--run-name", "scale",
             "--data", a.data, "--params", a.params, "--models", a.models,
             "--se-target", str(a.se_target), "--admission", "none", *knobs, *common],
            capture_output=True, text=True, timeout=a.timeout)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-2000:])
        summary = json.loads(out.stdout[out.stdout.index("{"):])
    finally:
        for w in workers:
            w.terminate()
        for w in workers:
            w.wait(timeout=15)
    calls, span = asyncio.run(_throughput(tag))
    return {"workers": n_workers, "calls": calls, "span_s": span,
            "calls_per_s": calls / span if span else 0.0, "wall_s": summary["wall_clock_s"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", default="1,2,4,8")
    p.add_argument("--limits", default="loose=20,tight=1",
                   help="name=rate_scale pairs; 1 = the default simulated limits")
    p.add_argument("--slots", type=int, default=2,
                   help="concurrent calls per worker, so workers (not Python) are the bottleneck")
    p.add_argument("--latency-scale", type=float, default=4.0)
    p.add_argument("--data", default="data/synthetic.json")
    p.add_argument("--params", default="data/irt_params.json")
    p.add_argument("--models", default="all")
    p.add_argument("--se-target", type=float, default=0.3)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    if not (os.environ.get("PG_DSN") and os.environ.get("REDIS_URL")):
        sys.exit("set PG_DSN and REDIS_URL (source .env.sh)")

    counts = [int(x) for x in a.workers.split(",")]
    table = {}
    for pair in a.limits.split(","):
        name, scale = pair.split("=")
        for n in counts:
            r = run(a, n, float(scale))
            table[(name, n)] = r
            print(f"{name:6s} workers={n}  {r['calls_per_s']:7.1f} calls/s"
                  f"  wall={r['wall_s']:6.1f}s  ({r['calls']} calls)", flush=True)

    names = [pair.split("=")[0] for pair in a.limits.split(",")]
    print(f"\n{a.slots} call slots per worker, latency x{a.latency_scale}")
    print(f"{'workers':>8s}" + "".join(f" {n + ' calls/s':>16s} {n + ' wall s':>13s}" for n in names))
    for n in counts:
        print(f"{n:>8d}" + "".join(f" {table[(nm, n)]['calls_per_s']:>16.1f}"
                                   f" {table[(nm, n)]['wall_s']:>13.1f}" for nm in names))


if __name__ == "__main__":
    main()
