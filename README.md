# adaptive-eval — Design A

An adaptive evaluation service. It uses item response theory (2PL) to pick the most informative
benchmark items for each model, runs many models' sessions concurrently under per-provider rate
limits, caches every response, and survives crashes mid-session.

## Scope

**In scope (Design A):** single process, one asyncio event loop, SQLite storage, and a replay
provider (free, deterministic) with one real-API skeleton. Ability is unidimensional 2PL.
Metrics come from held-out models.

**Out of scope, deferred:**
- **Design B:** distributed workers, Redis rate limits, Postgres, speculative prefetch.
- **Design C:** adaptive *ranking* across models, and multidimensional IRT for your
  spatial-benchmark levels.

**Size:** ~800 lines of Python plus tests in the starter. Expect ~1.5–2.5k lines once you
add real data loaders, a real provider, and a small dashboard.

**Time:** roughly 4–6 weeks at 8–10 hrs/week:

| Weeks | Work |
|-------|------|
| 1 | IRT + synthetic data |
| 2 | Engine, limiter, cache, store |
| 3 | Crash recovery + tests |
| 4 | Experiments + README numbers |
| 5–6 | Real data or a real provider, polish |

**Done means:**
1. Tests pass.
2. The crash demo produces results identical to a clean run.
3. The report shows call reduction, ranking agreement vs. full evaluation, and cache hit rate
   on held-out models.
4. The adaptive-vs-random curve backs the savings claim.

---

## Step 0 — Environment

```bash
mkdir adaptive-eval && cd adaptive-eval
git init
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p adaptive_eval tests data && touch adaptive_eval/__init__.py
```

Layout:

```
adaptive_eval/
  irt.py        # 2PL fit, ability estimate, item selection
  data.py       # synthetic generator, loader, held-out split
  providers.py  # Provider interface, ReplayProvider, real-API skeleton
  ratelimit.py  # async token buckets (requests/min + tokens/min)
  storage.py    # SQLite cache + append-only session event log
  engine.py     # adaptive session coroutine + concurrent scheduler
  report.py     # metrics + adaptive-vs-random curve
  cli.py        # commands
tests/test_core.py
```

Commit after each step: `git add -A && git commit -m "step N: ..."`.

## Step 1 — Data + replay mode (`data.py`)

Build and test the system on synthetic data drawn from a known 2PL model. You know the true
abilities, so you can check that the system recovers them, and it costs nothing.

```bash
python -m adaptive_eval.cli gen-synthetic --models 60 --items 400 --seed 0
```

About 5% of items are generated as near-uninformative, simulating mislabeled items. Each model
is assigned to one of three simulated providers with different limits.

## Step 2 — IRT (`irt.py`)

- **`fit_2pl`:** joint MAP fit with L-BFGS. The prior θ ~ N(0,1) fixes the scale;
  log a and b get weak priors.
- **`estimate_ability`:** 1-D Newton MAP. It returns θ and its standard error
  (1/√information), and the SE drives the stopping rule.
- **`select_next`:** picks the item with maximum Fisher information at the current θ. It is
  deterministic, with ties broken by lowest index. **Determinism is what makes crash replay
  exact.** `random` is the control baseline.

Fit on train models only. The rest are held out:

```bash
python -m adaptive_eval.cli fit --train-frac 0.7
# expect corr(fitted b, true b) around 0.9+
```

## Step 3 — Providers (`providers.py`)

Everything talks to one interface, `async answer(model, item_id) -> Response`.
`ReplayProvider` answers from the response matrix, adds random latency, and throws
`TransientError` at a set rate so retries get exercised. `AnthropicProvider` is an **untested
skeleton** for later. Only map 429, 5xx, and connection errors to `TransientError`; other 4xx
errors are bugs and must not be retried.

## Step 4 — Rate limiting (`ratelimit.py`)

Each provider gets two token buckets, one for requests/min and one for tokens/min. The lock
serves waiters in FIFO order, so no session starves. Every session using a provider shares its
limiter, which is the "interleave many sessions while respecting limits" requirement.

## Step 5 — Cache + event log (`storage.py`)

- **Cache key:** `sha256(model, item, prompt_version, decoding params, sample_idx)`. Anything
  that can change the answer belongs in the key. At temperature > 0 a hit reuses one sample,
  so bump `sample_idx` when you want repeated draws.
- **Event log:** each step writes `item_selected` *before* the API call (write-ahead) and
  `answer_recorded` after it. `UNIQUE(session_id, step, type)` is the idempotency key, so a
  duplicate write after a retry is a no-op.
- **`load_state`:** rebuilds a session from its events. If the last item was selected but
  never answered, it becomes `pending` and is re-issued on resume. θ is always recomputed from
  the log and never stored as the source of truth.

## Step 6 — Engine + scheduler (`engine.py`)

Each session runs this loop:

1. Resume state from the log.
2. Select an item and log it.
3. Check the cache; on a miss, acquire the rate limit, call the provider, and retry with
   exponential backoff plus jitter.
4. Log the answer.
5. Re-estimate θ.
6. Stop when `SE < se_target` (after `min_items`) or at `max_items`.

`run_many` runs all sessions under a semaphore with `return_exceptions=True`, so one bad
session can't take down the others.

```bash
python -m adaptive_eval.cli run --run-name adaptive-v1
python -m adaptive_eval.cli report --run-name adaptive-v1
```

## Step 7 — Tests

```bash
python -m pytest -q
```

The tests cover:
- IRT recovery.
- Ability direction.
- The token bucket enforcing its rate.
- Idempotent event appends.
- **Crash + resume from a fresh process context producing identical final results to a clean
  run.**

## Step 8 — Experiments for the README

**(a) Savings + ranking quality**

```bash
python -m adaptive_eval.cli run --run-name adaptive-v1 --se-target 0.30
python -m adaptive_eval.cli report --run-name adaptive-v1
```

Report `call_reduction_vs_full`, `kendall_tau_vs_full_theta`, and `cost_usd`.

**(b) Fair comparison: adaptive vs. random at equal item budgets.** This is the honest version
of "X% fewer calls at equal ranking accuracy":

```bash
python -m adaptive_eval.cli curve --budgets 5,10,15,20,30,40,60,80,120,160
```

This prints how many items each selector needs to reach τ ≥ 0.8 / 0.85 / 0.9. **With only
~18 held-out models, τ is noisy.** For claims, regenerate with `--models 300`, repeat over
several seeds, and report mean ± spread. Also plot the curve.

**(c) Cache as a cost lever.** Tighten the stopping rule on the same DB. The second run reuses
the first run's items:

```bash
python -m adaptive_eval.cli run --run-name adaptive-v2 --se-target 0.20
python -m adaptive_eval.cli report --run-name adaptive-v2    # see cache_hit_rate, paid_calls
```

**(d) Crash recovery demo**

```bash
python -m adaptive_eval.cli run --run-name demo --db data/crash.db --crash-after 6   # all sessions "crash"
python -m adaptive_eval.cli run --run-name demo --db data/crash.db                   # resume
python -m adaptive_eval.cli run --run-name demo --db data/clean.db                   # uninterrupted
python -m adaptive_eval.cli report --run-name demo --db data/crash.db
python -m adaptive_eval.cli report --run-name demo --db data/clean.db                # same thetas
```

Also try a real kill: start a run with `--se-target 0.1 --max-items 300`, press Ctrl-C (or
`kill -9` the PID) partway through, then rerun the same command.

**Resume bullet template** (fill with *your measured* numbers):

> Built an async adaptive-evaluation service (IRT-based item selection, per-provider rate
> limiting, response caching, event-sourced crash recovery); matched full-benchmark model
> rankings (Kendall τ = __) using __% fewer API calls, vs __% for random sampling at equal
> accuracy.

## Step 9 — Real data, then real models

1. **Real response matrix.** Convert per-item results (public leaderboard details, or the
   released data from the Fluid Benchmarking repo, which is worth checking) into the
   `data.py` JSON format. Use `null` for missing entries. **Handle missing entries in
   selection:** skip items with no logged answer for that model in replay mode and in
   `offline_curve`. The synthetic data has none, so the starter doesn't handle this yet.
2. **Real provider.** Finish `AnthropicProvider` (or others), set a real `ProviderConfig`
   (limits and prices from the provider's docs), and put the image content hash into the
   cache key. Start with a tiny budget and `--max-items 10`.
3. **Your spatial benchmark.** Fit IRT once you have results from enough models. Treat
   15–30+ as a rough floor and check what the literature used.

## Step 10 — Keep the door open for B and C

- **Keep `Provider`, `ResponseCache`, and `EventStore` as the only I/O boundaries.** B swaps
  their implementations (Redis limiter, Postgres store, worker queue) without touching
  `engine.py` logic.
- **Keep selection a pure function of logged state.** B's speculative prefetch needs to ask
  "what would the next item be if this answer is right/wrong?", which is just
  `select_next` on a hypothetical θ.
- **Keep sessions independent here.** C couples them (budget moves between models), so C
  adds a global scheduler that logs its own decisions. Don't bake cross-session state into A.
- **Keep `irt.py` behind a small interface** (`fit`, `estimate`, `information`). C's
  multidimensional IRT replaces θ with a vector without changing the engine loop.

## Known limitations (be upfront about these)

- SQLite writes happen on the event loop thread. That's fine at this scale, and it's a
  bottleneck B removes.
- Unidimensional 2PL. Spatial reasoning probably needs multidimensional IRT (Design C).
- Replay latency and failures are simulated, so real providers will behave differently.
- Items whose parameters were fit on older/weaker models can't separate models stronger than
  all training models, which is a known limitation of fixed IRT banks.
