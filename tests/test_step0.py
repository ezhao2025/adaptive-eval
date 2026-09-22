"""Design B Step 0: missing (null) responses and per-attempt call logging."""
import asyncio

import numpy as np

from adaptive_eval import data as D
from adaptive_eval.cli import build_context
from adaptive_eval.engine import SessionConfig, run_many
from adaptive_eval.irt import ItemBank, fit_2pl, select_next
from adaptive_eval.report import offline_curve


def sparse_data(tmp_path, n_models=9, n_items=30, frac=0.3, seed=0):
    d = D.generate_synthetic(n_models, n_items, seed=seed)
    rng = np.random.default_rng(seed)
    for row in d["R"]:
        for j in range(len(row)):
            if rng.random() < frac:
                row[j] = None
    path = tmp_path / "sparse.json"
    D.save(d, str(path))
    return D.load(str(path))


def test_select_next_respects_allowed():
    bank = ItemBank([f"i{k}" for k in range(10)], np.ones(10), np.linspace(-2, 2, 10))
    allowed = {1, 3, 7}
    for sel in ("max_info", "random"):
        assert select_next(sel, 0.0, bank, set(), "s", 0, allowed) in allowed
    assert select_next("max_info", 0.0, bank, {1, 3}, "s", 0, allowed) == 7


def test_sparse_run_uses_only_answerable_items_and_logs_attempts(tmp_path):
    d = sparse_data(tmp_path)
    bank, _ = fit_2pl(d["R"], d["items"])
    ctx, conn = build_context(d, bank, str(tmp_path / "e.db"))
    cfg = SessionConfig(se_target=0.0, max_items=1000)   # never stop early: exhaust every model
    results = asyncio.run(run_many(ctx, "sparse", d["models"], cfg, concurrency=8))
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, errors[:3]

    midx = {m: i for i, m in enumerate(d["models"])}
    for r in results:
        assert r["n_items"] == int((~np.isnan(d["R"][midx[r["model"]]])).sum())

    paid = conn.execute("SELECT COUNT(*) FROM events"
                        " WHERE type='answer_recorded' AND cached=0").fetchone()[0]
    ok, transient = conn.execute("SELECT SUM(outcome='ok'), SUM(outcome='transient_error')"
                                 " FROM call_attempts").fetchone()
    assert ok == paid                      # one successful attempt per paid answer
    assert (transient or 0) == ctx.retries  # every retried failure is on disk


def test_offline_curve_handles_missing(tmp_path):
    d = sparse_data(tmp_path)
    bank, _ = fit_2pl(d["R"], d["items"])
    curve = offline_curve(d, bank, d["models"], [5, 15])
    assert all(np.isfinite(r["tau_max_info"]) and np.isfinite(r["tau_random"]) for r in curve)



def test_sqlite_load_state_ignores_other_event_types(tmp_path):
    from adaptive_eval.storage import EventStore, connect
    store = EventStore(connect(str(tmp_path / "e.db")))
    store.append("s", 0, "item_selected", "i1")
    store.append("s", 0, "allocation", "i1")
    st = store.load_state("s")
    assert st.answered == [] and st.pending == (0, "i1")
