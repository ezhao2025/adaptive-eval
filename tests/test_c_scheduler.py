"""Design C scheduler on B's distributed stack. Needs PG_DSN and REDIS_URL; skipped otherwise.

With one call in flight the distributed run must make exactly the offline allocator's
decisions, crash or no crash. With many in flight the order results arrive in is not
reproducible, so those tests check invariants instead: one call per model at a time, no
item asked twice, every answer preceded by a logged allocation, nothing orphaned, and
final scores that are a pure function of the logged answers.
"""
import asyncio
import json
import uuid

import asyncpg
import numpy as np
import pytest
from scipy.special import expit

from adaptive_eval.b import pgstore
from adaptive_eval.b.queue import JobQueue, connect
from adaptive_eval.b.scheduler import SchedulerCrash
from adaptive_eval.b.worker import Worker
from adaptive_eval.c.ranking import Bank2D, choose, expected_discordant, score, variance_after
from adaptive_eval.c.scheduler import RankingScheduler
from adaptive_eval.providers import ReplayProvider

from test_scheduler import DSN, FAST, URL, _reachable

pytestmark = pytest.mark.skipif(not _reachable(), reason="PG_DSN/REDIS_URL not set or not reachable")


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)
    n_items, n_models = 80, 8
    bank = Bank2D([f"it{k:03d}" for k in range(n_items)], np.exp(rng.normal(0, 0.3, n_items)),
                  rng.normal(0, 1, n_items), (np.arange(n_items) % 3 == 0).astype(int))
    theta = rng.multivariate_normal([0, 0], [[0.4, 0.25], [0.25, 0.6]], n_models)
    Y = (rng.random((n_models, n_items)) <
         expit(theta @ bank.loadings().T - bank.b)).astype(int)
    models = [f"m{k}" for k in range(n_models)]
    providers = list(FAST)
    return dict(bank=bank, Y=Y, models=models,
                model_provider={m: providers[k % len(providers)] for k, m in enumerate(models)})


def offline(world, max_calls, stop):
    """The allocator alone, one call at a time: the reference sequence."""
    bank, Y = world["bank"], world["Y"]
    w = bank.weights(0.5)
    ans = [dict() for _ in world["models"]]
    st = [score(bank, w, a) for a in ans]
    vn = [variance_after(bank, w, a, s) for a, s in zip(ans, st)]
    seq = []
    while expected_discordant(st) >= stop and len(seq) < max_calls:
        m, i, g = choose(st, vn, np.ones(len(ans)))
        if m < 0 or not g > 0:
            break
        ans[m][i] = int(Y[m, i])
        seq.append((world["models"][m], bank.item_ids[i]))
        st[m] = score(bank, w, ans[m])
        vn[m] = variance_after(bank, w, ans[m], st[m])
    return seq, [s.s for s in st]


async def run_c(world, *, crash_after=None, drop_model=None, n_workers=3, **kw):
    schema, prefix = f"c_{uuid.uuid4().hex[:8]}", f"c{uuid.uuid4().hex[:8]}:"
    r = connect(URL)
    pool = await pgstore.connect(DSN, schema)
    resp = {(m, it): bool(world["Y"][i, j]) for i, m in enumerate(world["models"])
            for j, it in enumerate(world["bank"].item_ids) if m != drop_model}
    workers = [Worker(f"w{i}", r, pool, {n: ReplayProvider(n, c, resp, seed=i)
                                         for n, c in FAST.items()},
                      FAST, prefix=prefix, block_ms=100) for i in range(n_workers)]
    runs = [asyncio.create_task(w.run()) for w in workers]

    def scheduler():
        return RankingScheduler("rk", world["models"], world["model_provider"], world["bank"],
                                pool, JobQueue(r, list(FAST), prefix), FAST,
                                log=lambda *_: None, **kw)
    crashed = False
    try:
        if crash_after is not None:
            try:
                await asyncio.wait_for(scheduler().run(exit_after=crash_after), 60)
            except SchedulerCrash:
                crashed = True
        summary = await asyncio.wait_for(scheduler().run(), 120)
        ev = await pool.fetch("SELECT session_id, step, type, item_id, correct, detail"
                              " FROM events ORDER BY id")
        orphans = await pool.fetchval(
            "SELECT COUNT(*) FROM (SELECT session_id, step FROM events WHERE type='item_selected'"
            " EXCEPT SELECT session_id, step FROM events WHERE type='answer_recorded') x")
        return summary, ev, orphans, crashed
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


def allocations(ev):
    """(model, item) in allocation order, from the log."""
    rows = [(json.loads(e["detail"])["seq"], e["session_id"].split(":", 1)[1], e["item_id"])
            for e in ev if e["type"] == "allocation"]
    return [(m, it) for _, m, it in sorted(rows)]


def check_invariants(world, summary, ev, orphans):
    assert orphans == 0
    ans = [e for e in ev if e["type"] == "answer_recorded"]
    alloc = {(e["session_id"], e["step"]) for e in ev if e["type"] == "allocation"}
    assert {(e["session_id"], e["step"]) for e in ans} <= alloc     # every answer was allocated
    per = {}
    for e in ans:
        per.setdefault(e["session_id"], []).append(e["item_id"])
    assert all(len(v) == len(set(v)) for v in per.values())         # no item asked twice
    seqs = sorted(json.loads(e["detail"])["seq"] for e in ev if e["type"] == "allocation")
    assert seqs == list(range(len(seqs)))                           # one global order, no gaps
    # final scores are a pure function of the logged answers
    bank, w = world["bank"], world["bank"].weights(0.5)
    idx = bank.index()
    for row in summary["ranking"]:
        a = {idx[e["item_id"]]: e["correct"] for e in ans if e["session_id"] == f"rk:{row['model']}"}
        assert row["calls"] == len(a)
        assert abs(score(bank, w, a).s - row["score"]) < 1e-4


def test_sequential_run_makes_the_offline_allocators_decisions(world):
    ref, ref_scores = offline(world, max_calls=60, stop=0.3)
    summary, ev, orphans, _ = asyncio.run(run_c(world, max_inflight=1, max_calls=60,
                                                stop_discordant=0.3, cost_aware=False))
    assert allocations(ev) == ref
    got = {r["model"]: r["score"] for r in summary["ranking"]}
    assert all(abs(got[m] - s) < 1e-4 for m, s in zip(world["models"], ref_scores))
    check_invariants(world, summary, ev, orphans)


def test_sequential_crash_and_resume_equals_clean_run(world):
    ref, _ = offline(world, max_calls=60, stop=0.3)
    summary, ev, orphans, crashed = asyncio.run(run_c(
        world, crash_after=25, max_inflight=1, max_calls=60, stop_discordant=0.3,
        cost_aware=False))
    assert crashed and allocations(ev) == ref
    check_invariants(world, summary, ev, orphans)


def test_concurrent_run_keeps_invariants(world):
    summary, ev, orphans, _ = asyncio.run(run_c(world, windows={p: 2 for p in FAST},
                                                max_calls=120, stop_discordant=0.0))
    assert summary["max_inflight_total"] > 1                  # concurrency actually happened
    assert summary["max_inflight_total"] <= 2 * len(FAST)     # windows held
    assert summary["slots_left_idle"] > 0                     # the gain floor was exercised
    assert summary["calls"] == 120 and summary["stop_reason"] == "max calls"
    check_invariants(world, summary, ev, orphans)


def test_concurrent_crash_recovers_without_orphans_or_repeats(world):
    summary, ev, orphans, crashed = asyncio.run(run_c(
        world, crash_after=40, windows={p: 2 for p in FAST}, max_calls=120,
        stop_discordant=0.0))
    assert crashed and summary["calls"] == 120
    check_invariants(world, summary, ev, orphans)


def test_budget_is_not_overspent(world):
    full, *_ = asyncio.run(run_c(world, windows={p: 2 for p in FAST}, max_calls=60,
                                 stop_discordant=0.0))
    budget = full["spent_usd"] / 2
    summary, ev, orphans, _ = asyncio.run(run_c(world, windows={p: 2 for p in FAST},
                                                budget_usd=budget, stop_discordant=0.0))
    assert summary["stop_reason"] == "budget"
    assert summary["spent_usd"] <= budget * 1.05
    check_invariants(world, summary, ev, orphans)


def test_failed_model_drops_out_and_the_rest_are_ranked(world):
    bad = world["models"][0]
    summary, ev, orphans, _ = asyncio.run(run_c(world, drop_model=bad, max_calls=60,
                                                windows={p: 2 for p in FAST}))
    ranked = {r["model"] for r in summary["ranking"]}
    assert bad not in ranked and ranked == set(world["models"][1:])
    assert summary["failed"] == 1
