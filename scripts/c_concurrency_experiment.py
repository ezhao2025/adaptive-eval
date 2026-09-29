"""Step C5: what does concurrency cost the allocator? sp6 data through the full B stack.

    python scripts/c_concurrency_experiment.py --matrix data/sp6_matrix.json \
        --meta data/sp6_item_meta.json

Same split as ranking_experiment.py (bank calibrated on half the scenes, the 7 fully
answered models ranked on the other half). Models are spread over the three simulated
providers; workers replay logged answers. With k calls in flight per provider, each
decision is made while up to 3k - 1 answers are still out, so it sees staler state than
the one-at-a-time allocator. Needs PG_DSN and REDIS_URL.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

import asyncpg
import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from explanatory_cv import load                                       # noqa: E402
from ranking_experiment import make_bank                              # noqa: E402

from adaptive_eval.b import pgstore                                   # noqa: E402
from adaptive_eval.b.queue import JobQueue, connect                   # noqa: E402
from adaptive_eval.b.worker import Worker                             # noqa: E402
from adaptive_eval.c.scheduler import RankingScheduler                # noqa: E402
from adaptive_eval.providers import ProviderConfig, ReplayProvider    # noqa: E402

# three providers, loose limits, ~20-60 ms calls: latency-bound, so windows matter
PROV = {f"sim-{k}": ProviderConfig(60_000, 10**9, (0.02, 0.06), 0.0, 0.0025, 0.01)
        for k in ("a", "b", "c")}


async def one_run(dsn, url, bank, names, Y, max_calls, window, seed, rho=0.0):
    schema, prefix = f"cx_{uuid.uuid4().hex[:8]}", f"cx{uuid.uuid4().hex[:8]}:"
    r = connect(url)
    pool = await pgstore.connect(dsn, schema)
    resp = {(m, it): bool(Y[i, j]) for i, m in enumerate(names)
            for j, it in enumerate(bank.item_ids)}
    mp = {m: list(PROV)[k % len(PROV)] for k, m in enumerate(names)}
    workers = [Worker(f"w{i}", r, pool, {n: ReplayProvider(n, c, resp, seed=seed * 10 + i)
                                         for n, c in PROV.items()}, PROV, prefix=prefix,
                      block_ms=50) for i in range(3)]
    tasks = [asyncio.create_task(w.run()) for w in workers]
    try:
        sched = RankingScheduler("cx", names, mp, bank, pool, JobQueue(r, list(PROV), prefix),
                                 PROV, windows={p: max(window, 1) for p in PROV},
                                 max_inflight=1 if window == 0 else None, max_calls=max_calls,
                                 stop_discordant=0.0, cost_aware=False, min_relative_gain=rho,
                                 log=lambda *_: None)
        t0 = time.time()
        s = await sched.run()
        s["wall"] = time.time() - t0
        return s
    finally:
        for w in workers:
            w.stopping.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await pool.close()
        c = await asyncpg.connect(dsn)
        await c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await c.close()
        keys = [k async for k in r.scan_iter(f"{prefix}*")]
        if keys:
            await r.delete(*keys)
        await r.aclose()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/sp6_matrix.json")
    p.add_argument("--meta", default="data/sp6_item_meta.json")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--windows", default="0,1,2,4",
                   help="calls in flight per provider; 0 = one call in flight in total")
    p.add_argument("--calls-per-model", default="20,80")
    p.add_argument("--rhos", default="0,0.5",
                   help="min_relative_gain values (0 = always fill a free slot)")
    a = p.parse_args()
    dsn, url = os.environ.get("PG_DSN"), os.environ.get("REDIS_URL")
    if not dsn or not url:
        sys.exit("set PG_DSN and REDIS_URL (source .env.sh)")

    D = load(a.matrix, a.meta)
    full = ~np.isnan(D["R"]).any(1)
    names = [m for m, f in zip(D["models"], full) if f]
    scenes = np.unique(D["scene"])
    windows = [int(x) for x in a.windows.split(",")]
    per_model = [int(x) for x in a.calls_per_model.split(",")]
    rhos = [float(x) for x in a.rhos.split(",")]
    runs = [(wdw, rho) for wdw in windows for rho in (rhos if wdw > 0 else [0.0])]
    res = {(k, wr): [] for k in per_model for wr in runs}
    for sp in range(a.splits):
        rng = np.random.default_rng(sp)                    # same splits as ranking_experiment
        pool_scenes = set(rng.permutation(scenes)[: len(scenes) // 2])
        pool = np.array([s in pool_scenes for s in D["scene"]])
        bank = make_bank(D, ~pool, pool)
        Y = D["R"][full][:, pool].astype(int)
        truth = dict(zip(names, Y @ bank.weights(0.5)))
        for k in per_model:
            for wdw, rho in runs:
                s = asyncio.run(one_run(dsn, url, bank, names, Y, k * len(names), wdw, sp, rho))
                est = {r["model"]: r["score"] for r in s["ranking"]}
                tau = kendalltau([est[m] for m in names], [truth[m] for m in names]).statistic
                res[(k, (wdw, rho))].append((tau, s["wall"], s["max_inflight_total"]))
        print(f"split {sp} done", flush=True)

    print(f"\n{len(names)} models, 3 providers, windows = calls in flight per provider; "
          f"mean over {a.splits} splits")
    print(f"{'calls/model':>11s} {'window':>6s} {'rho':>4s} {'max in flight':>13s} {'tau':>6s}"
          f" {'sd':>6s} {'wall s':>7s}")
    for k in per_model:
        for wdw, rho in runs:
            x = res[(k, (wdw, rho))]
            t = np.array([v[0] for v in x])
            print(f"{k:>11d} {wdw if wdw else 'seq':>6} {rho:4.1f} {max(v[2] for v in x):>13d}"
                  f" {t.mean():6.3f} {t.std():6.3f} {np.mean([v[1] for v in x]):7.1f}")


if __name__ == "__main__":
    main()
