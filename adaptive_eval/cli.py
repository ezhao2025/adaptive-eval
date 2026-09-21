"""CLI: python -m adaptive_eval.cli <command> ..."""
from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from . import data as D
from .engine import Context, SessionConfig, SimulatedCrash, run_many
from .irt import ItemBank, fit_2pl
from .providers import DEFAULT_PROVIDERS, ReplayProvider
from .ratelimit import ProviderLimiter
from .report import dump, items_needed, offline_curve, run_report
from .storage import EventStore, ResponseCache, connect


def cmd_gen(a):
    D.save(D.generate_synthetic(a.models, a.items, a.seed), a.out)
    print(f"wrote {a.out}: {a.models} models x {a.items} items")


def cmd_fit(a):
    d = D.load(a.data)
    if d.get("holdout_models"):
        test = d["holdout_models"]
        train = [m for m in d["models"] if m not in set(test)]
    else:
        train, test = D.split_models(d["models"], a.train_frac, a.seed)
    rows = [d["models"].index(m) for m in train]
    bank, _ = fit_2pl(d["R"][rows], d["items"])
    bank.save(a.out, {"train_models": train, "test_models": test})
    msg = f"fit on {len(train)} models, {len(test)} held out -> {a.out}"
    if "truth" in d:
        r = np.corrcoef(bank.b, d["truth"]["b"])[0, 1]
        msg += f" | corr(fitted b, true b) = {r:.3f}"
    print(msg)


def build_context(d, bank, db_path, seed=0):
    conn = connect(db_path)
    midx = {m: i for i, m in enumerate(d["models"])}
    responses = {(m, it): bool(d["R"][midx[m], j])
                 for m in d["models"] for j, it in enumerate(d["items"])
                 if not np.isnan(d["R"][midx[m], j])}
    providers = {n: ReplayProvider(n, c, responses, seed) for n, c in DEFAULT_PROVIDERS.items()}
    limiters = {n: ProviderLimiter(c.rpm, c.tpm) for n, c in DEFAULT_PROVIDERS.items()}
    ctx = Context(bank, providers, DEFAULT_PROVIDERS, limiters, ResponseCache(conn),
                  EventStore(conn), d["model_provider"])
    return ctx, conn


def cmd_run(a):
    d = D.load(a.data)
    bank, meta = ItemBank.load(a.params)
    models = meta["test_models"] if a.models == "test" else d["models"]
    ctx, conn = build_context(d, bank, a.db)
    cfg = SessionConfig(selector=a.selector, se_target=a.se_target, max_items=a.max_items)
    t0 = time.time()
    results = asyncio.run(run_many(ctx, a.run_name, models, cfg, a.concurrency, a.crash_after))
    crashed = [r for r in results if isinstance(r, SimulatedCrash)]
    errors = [r for r in results if isinstance(r, Exception) and not isinstance(r, SimulatedCrash)]
    print(f"{len(results) - len(crashed) - len(errors)} done, {len(crashed)} crashed, "
          f"{len(errors)} failed in {time.time() - t0:.1f}s | retries={ctx.retries} "
          f"cache hits={ctx.cache.hits} misses={ctx.cache.misses}")
    for e in errors[:3]:
        print("  error:", repr(e))


def cmd_report(a):
    d = D.load(a.data)
    bank, _ = ItemBank.load(a.params)
    print(dump(run_report(connect(a.db), d, bank, a.run_name)))


def cmd_curve(a):
    d = D.load(a.data)
    bank, meta = ItemBank.load(a.params)
    budgets = [int(x) for x in a.budgets.split(",")]
    curve = offline_curve(d, bank, meta["test_models"], budgets)
    print(dump(curve))
    for target in (0.8, 0.85, 0.9):
        ad, rn = items_needed(curve, "max_info", target), items_needed(curve, "random", target)
        print(f"tau>={target}: adaptive needs {ad} items, random needs {rn}")


def main():
    p = argparse.ArgumentParser(prog="adaptive_eval")
    sub = p.add_subparsers(required=True)

    g = sub.add_parser("gen-synthetic"); g.set_defaults(fn=cmd_gen)
    g.add_argument("--models", type=int, default=60); g.add_argument("--items", type=int, default=400)
    g.add_argument("--seed", type=int, default=0); g.add_argument("--out", default="data/synthetic.json")

    f = sub.add_parser("fit"); f.set_defaults(fn=cmd_fit)
    f.add_argument("--data", default="data/synthetic.json"); f.add_argument("--out", default="data/irt_params.json")
    f.add_argument("--train-frac", type=float, default=0.7); f.add_argument("--seed", type=int, default=0)

    common = dict(data="data/synthetic.json", params="data/irt_params.json", db="data/eval.db")
    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run, **common)
    r.add_argument("--run-name", required=True); r.add_argument("--selector", default="max_info")
    r.add_argument("--se-target", type=float, default=0.30); r.add_argument("--max-items", type=int, default=100)
    r.add_argument("--concurrency", type=int, default=64); r.add_argument("--models", default="test")
    r.add_argument("--crash-after", type=int, default=None)
    r.add_argument("--db", default=common["db"])
    r.add_argument("--data", default=common["data"]); r.add_argument("--params", default=common["params"])

    rep = sub.add_parser("report"); rep.set_defaults(fn=cmd_report, **common)
    rep.add_argument("--run-name", required=True); rep.add_argument("--db", default=common["db"])
    rep.add_argument("--data", default=common["data"]); rep.add_argument("--params", default=common["params"])

    c = sub.add_parser("curve"); c.set_defaults(fn=cmd_curve, **common)
    c.add_argument("--budgets", default="5,10,15,20,30,40,60,80,120,160")
    c.add_argument("--data", default=common["data"]); c.add_argument("--params", default=common["params"])

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
