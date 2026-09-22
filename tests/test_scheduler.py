"""Design B Step 6: scheduler + workers give exactly A's results, even when the scheduler
is killed mid-run and restarted. Needs PG_DSN and REDIS_URL; skipped otherwise."""
import asyncio
import os
import uuid

import asyncpg
import numpy as np
import pytest

from adaptive_eval import data as D
from adaptive_eval.b import pgstore
from adaptive_eval.b.admission import Admission
from adaptive_eval.b.queue import JobQueue, connect
from adaptive_eval.b.ratelimit_redis import limiters_for
from adaptive_eval.b.speculate import Speculator
from adaptive_eval.b.scheduler import Scheduler, SchedulerCrash, available_items
from adaptive_eval.b.worker import Worker
from adaptive_eval.cli import build_context
from adaptive_eval.engine import SessionConfig, run_many
from adaptive_eval.irt import fit_2pl
from adaptive_eval.providers import DEFAULT_PROVIDERS, ProviderConfig, ReplayProvider

DSN, URL = os.environ.get("PG_DSN"), os.environ.get("REDIS_URL")


def _reachable():
    if not (DSN and URL):
        return False
    async def ping():
        conn = await asyncpg.connect(DSN, timeout=3)
        await conn.close()
        r = connect(URL)
        try:
            return await r.ping()
        finally:
            await r.aclose()
    try:
        return asyncio.run(ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="PG_DSN/REDIS_URL not set or not reachable")

FAST = {n: ProviderConfig(60_000, 10**9, (0.001, 0.005), 0.05, 0.0025, 0.01)
        for n in DEFAULT_PROVIDERS}
CFG = SessionConfig()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "d.json"
    D.save(D.generate_synthetic(12, 60, seed=5), str(path))
    d = D.load(str(path))
    bank, _ = fit_2pl(d["R"], d["items"])
    # reference: Design A's single-process engine on SQLite
    ctx, _ = build_context(d, bank, str(tmp_path_factory.mktemp("a") / "a.db"))
    res = asyncio.run(run_many(ctx, "eq", d["models"], CFG))
    ref = sorted((r["model"], round(r["theta"], 9), r["n_items"]) for r in res)
    return d, bank, ref


def responses(d, drop_model=None):
    return {(m, it): bool(d["R"][i, j]) for i, m in enumerate(d["models"])
            for j, it in enumerate(d["items"]) if m != drop_model}


async def run_b(d, bank, *, crash_after=None, drop_model=None, n_workers=3, admission=None,
                speculate=None):
    """admission: None, or dict of Admission kwargs (a fresh controller per scheduler)."""
    schema, prefix = f"t_{uuid.uuid4().hex[:8]}", f"t{uuid.uuid4().hex[:8]}:"
    r = connect(URL)
    pool = await pgstore.connect(DSN, schema)
    resp = responses(d, drop_model)
    workers = [Worker(f"w{i}", r, pool, {n: ReplayProvider(n, c, resp, seed=i)
                                         for n, c in FAST.items()},
                      FAST, prefix=prefix, block_ms=100) for i in range(n_workers)]
    runs = [asyncio.create_task(w.run()) for w in workers]

    controllers = []

    def scheduler():
        q = JobQueue(r, list(FAST), prefix)
        adm = None if admission is None else Admission(q, FAST, **admission)
        controllers.append(adm)
        sched = Scheduler("eq", d["models"], d["model_provider"], bank, pool, q, CFG,
                          available=available_items(d, bank), admission=adm,
                          log=lambda *_: None)
        if speculate is not None:     # dict of Speculator kwargs
            sched.speculator = Speculator(sched, limiters_for(r, FAST, prefix=prefix),
                                          pgstore.PgResponseCache(pool), **speculate)
        return sched
    crashed = False
    try:
        if crash_after is not None:
            try:
                await asyncio.wait_for(scheduler().run(exit_after=crash_after), 60)
            except SchedulerCrash:
                crashed = True
        summary = await asyncio.wait_for(scheduler().run(), 60)   # a fresh scheduler "process"
        rows = await pool.fetch("SELECT model, theta, n_items, status FROM sessions"
                                " WHERE run_name='eq' ORDER BY model")
        summary["_paid_cost"] = float(await pool.fetchval(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM events WHERE type='answer_recorded'"))
        summary["_controllers"] = controllers
        spec = {r["speculative"]: float(r["usd"]) for r in await pool.fetch(
            "SELECT speculative, COALESCE(SUM(cost_usd), 0) AS usd FROM call_attempts"
            " WHERE status='ok' GROUP BY 1")}
        summary["_spec_cost"], summary["_real_cost"] = spec.get(True, 0.0), spec.get(False, 0.0)
        orphans = await pool.fetchval(
            "SELECT COUNT(*) FROM (SELECT session_id, step FROM events WHERE type='item_selected'"
            " EXCEPT SELECT session_id, step FROM events WHERE type='answer_recorded') x")
        return summary, rows, orphans, crashed
    finally:
        for w in workers:
            w.stopping.set()
        await asyncio.wait_for(asyncio.gather(*runs, return_exceptions=True), 10)
        await pool.close()
        conn = await asyncpg.connect(DSN)
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
        keys = [k async for k in r.scan_iter(f"{prefix}*")]
        if keys:
            await r.delete(*keys)
        await r.aclose()


def test_distributed_run_matches_design_a(setup):
    d, bank, ref = setup
    summary, rows, orphans, _ = asyncio.run(run_b(d, bank))
    got = sorted((r["model"], round(r["theta"], 9), r["n_items"]) for r in rows)
    assert got == ref
    assert summary["done"] == len(d["models"]) and summary["failed"] == 0 and orphans == 0


def test_scheduler_killed_mid_run_recovers_to_same_result(setup):
    d, bank, ref = setup
    summary, rows, orphans, crashed = asyncio.run(run_b(d, bank, crash_after=25))
    got = sorted((r["model"], round(r["theta"], 9), r["n_items"]) for r in rows)
    assert crashed
    assert got == ref and orphans == 0
    assert summary["reenqueued"] > 0            # pending steps were re-issued on restart


def test_fatal_error_fails_only_that_session(setup):
    d, bank, ref = setup
    bad = d["models"][0]
    summary, rows, orphans, _ = asyncio.run(run_b(d, bank, drop_model=bad))
    status = {r["model"]: r["status"] for r in rows}
    assert status[bad] == "failed"
    assert all(s == "done" for m, s in status.items() if m != bad)
    good = sorted((r["model"], round(r["theta"], 9), r["n_items"]) for r in rows if r["model"] != bad)
    assert good == [x for x in ref if x[0] != bad]


@pytest.mark.parametrize("mode", ["priority", "nearest"])
def test_priority_admission_changes_order_not_results(setup, mode):
    """Admission reorders calls across sessions; each session's own item sequence is
    deterministic, so final results must still equal Design A exactly."""
    d, bank, ref = setup
    summary, rows, orphans, _ = asyncio.run(
        run_b(d, bank, admission=dict(mode=mode, windows={p: 2 for p in FAST})))
    got = sorted((r["model"], round(r["theta"], 9), r["n_items"]) for r in rows)
    assert got == ref and orphans == 0
    adm = summary["_controllers"][-1]
    assert all(adm.max_inflight[p] <= 2 for p in FAST)          # the window held


@pytest.mark.parametrize("mode", ["priority", "nearest", "fifo"])
def test_budget_stops_the_run_without_overspending(setup, mode):
    d, bank, _ = setup
    full = asyncio.run(run_b(d, bank))[0]["_paid_cost"]
    budget = full / 3
    summary, rows, _, _ = asyncio.run(
        run_b(d, bank, admission=dict(mode=mode, budget_usd=budget, windows={p: 2 for p in FAST})))
    statuses = {r["status"] for r in rows}
    assert "budget" in statuses and statuses <= {"done", "budget"}
    assert summary["_paid_cost"] <= budget * 1.05            # committed-cost accounting
    assert summary["budget_stopped"] > 0


def test_speculation_changes_cost_and_speed_never_results(setup):
    d, bank, ref = setup
    summary, rows, orphans, _ = asyncio.run(
        run_b(d, bank, speculate=dict(min_free=0.0, max_fraction=0.5)))
    got = sorted((r["model"], round(r["theta"], 9), r["n_items"]) for r in rows)
    assert got == ref and orphans == 0          # identical to Design A
    assert summary["spec_enqueued"] > 0 and summary["spec_hits"] > 0
    assert summary["spec_calls"] > 0


def test_speculative_spend_respects_hard_cap(setup):
    d, bank, ref = setup
    f = 0.1
    summary, rows, _, _ = asyncio.run(run_b(d, bank, speculate=dict(min_free=0.0, max_fraction=f)))
    got = sorted((r["model"], round(r["theta"], 9), r["n_items"]) for r in rows)
    assert got == ref
    one_call = 0.0025                            # generous upper bound on one call's cost
    assert summary["_spec_cost"] <= f * (summary["_spec_cost"] + summary["_real_cost"]) + one_call
