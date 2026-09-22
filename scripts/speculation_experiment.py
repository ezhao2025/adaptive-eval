"""Step 8 experiment: speculation off vs on, same data and seed.

Each arm gets fresh workers, a fresh Postgres schema and Redis prefix (no shared cache).
Results must be identical; what changes is cost and speed.

    python scripts/speculation_experiment.py --reps 3
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admission_experiment import _drop  # noqa: E402  (reuse the schema cleanup helper)

import asyncio  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import asyncpg  # noqa: E402


def run_arm(a, speculate: bool, rep: int) -> tuple[dict, list]:
    tag = f"spec_{'on' if speculate else 'off'}_{rep}_{uuid.uuid4().hex[:6]}"
    common = ["--schema", tag, "--prefix", f"{tag}:"]
    workers = [subprocess.Popen([sys.executable, "-m", "adaptive_eval.b.worker", "--id", f"w{i}",
                                 "--data", a.data, "--spec-min-free", str(a.spec_min_free),
                                 *common],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
               for i in range(a.workers)]
    try:
        time.sleep(1.5)
        out = subprocess.run(
            [sys.executable, "-m", "adaptive_eval.b.scheduler", "--run-name", "spec-exp",
             "--data", a.data, "--params", a.params, "--models", a.models,
             "--se-target", str(a.se_target), "--admission", "none", *common,
             *(["--speculate", "--spec-max-fraction", str(a.max_fraction),
                "--spec-min-free", str(a.spec_min_free)] if speculate else [])],
            capture_output=True, text=True, timeout=a.timeout)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-2000:])
        summary = json.loads(out.stdout[out.stdout.index("{"):])
        rows = asyncio.run(_results(tag))
        return summary, rows
    finally:
        for w in workers:
            w.terminate()
        for w in workers:
            w.wait(timeout=15)
        asyncio.run(_drop(tag))


async def _results(schema: str) -> list:
    conn = await asyncpg.connect(os.environ["PG_DSN"])
    try:
        rows = await conn.fetch(f'SELECT model, theta, n_items FROM "{schema}".sessions'
                                " ORDER BY model")
    finally:
        await conn.close()
    return [(r["model"], round(r["theta"], 9), r["n_items"]) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--data", default="data/synthetic.json")
    p.add_argument("--params", default="data/irt_params.json")
    p.add_argument("--models", default="all")
    p.add_argument("--se-target", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-fraction", type=float, default=0.2)
    p.add_argument("--spec-min-free", type=float, default=0.3,
                   help="0 lets speculation use any free capacity (tests the headroom rule)")
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    if not (os.environ.get("PG_DSN") and os.environ.get("REDIS_URL")):
        sys.exit("set PG_DSN and REDIS_URL (source .env.sh)")

    arms = {"off": [], "on": []}
    reference = None
    for rep in range(a.reps):
        for name in arms:
            s, rows = run_arm(a, name == "on", rep)
            reference = reference or rows
            s["identical"] = rows == reference
            arms[name].append(s)
            print(f"rep {rep} spec={name:3s} identical={s['identical']}"
                  f"  wall={s['wall_clock_s']:.1f}s  median_session={s['median_session_latency_s']}s"
                  f"  cost=${s['cost_usd'] + s.get('spec_cost_usd', 0):.4f}"
                  + (f"  hit_rate={s['spec_hit_rate']:.2f}"
                     f"  wasted=${s['wasted_spec_cost_usd']:.4f}" if name == "on" else ""),
                  flush=True)

    def ms(xs):
        return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 \
            else f"{xs[0]:.3f}"
    print(f"\n{a.reps} reps (mean ± sd); results identical in every run: "
          f"{all(s['identical'] for rs in arms.values() for s in rs)}")
    print(f"{'spec':5s} {'wall clock s':>16s} {'median session s':>18s} {'total cost $':>16s}"
          f" {'spec hit rate':>16s} {'wasted spec $':>16s}")
    for name, rs in arms.items():
        total = [s["cost_usd"] + s.get("spec_cost_usd", 0.0) for s in rs]
        hit = ms([s["spec_hit_rate"] for s in rs]) if name == "on" else "-"
        waste = ms([s["wasted_spec_cost_usd"] for s in rs]) if name == "on" else "-"
        print(f"{name:5s} {ms([s['wall_clock_s'] for s in rs]):>16s}"
              f" {ms([s['median_session_latency_s'] for s in rs]):>18s} {ms(total):>16s}"
              f" {hit:>16s} {waste:>16s}")


if __name__ == "__main__":
    main()
