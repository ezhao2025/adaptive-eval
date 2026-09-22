"""Adaptive sessions + scheduler.

Each session is a coroutine: select item -> log it -> (cache | rate limit -> call) -> log
answer -> update ability -> repeat until precise enough. Many sessions interleave on one
event loop; per-provider limiters keep every provider under its limits.
"""
from __future__ import annotations

import asyncio
import inspect
import random
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from .irt import ItemBank, estimate_ability, select_next
from .providers import Provider, ProviderConfig, TransientError
from .ratelimit import ProviderLimiter
from .storage import EventStore, ResponseCache


class SimulatedCrash(Exception):
    pass


async def _maybe(x):
    """Await x if the storage backend is async (Postgres); pass through if sync (SQLite)."""
    return await x if inspect.isawaitable(x) else x


@dataclass
class SessionConfig:
    selector: str = "max_info"      # or "random" (the control baseline)
    se_target: float = 0.30         # stop when ability standard error drops below this
    min_items: int = 5
    max_items: int = 100
    prompt_version: str = "v1"
    decoding: dict = field(default_factory=lambda: {"temperature": 0, "max_tokens": 50})
    sample_idx: int = 0
    max_retries: int = 6


@dataclass
class Context:
    bank: ItemBank
    providers: dict[str, Provider]
    provider_cfgs: dict[str, ProviderConfig]
    limiters: dict[str, ProviderLimiter]
    cache: ResponseCache
    store: EventStore
    model_provider: dict[str, str]
    retries: int = 0
    available: dict[str, set[int]] | None = None   # per-model answerable items; None = all


async def call_item(ctx: Context, model: str, item_id: str, cfg: SessionConfig,
                    *, session_id: str = "", step: int = -1) -> tuple[dict, bool]:
    """Cache first; on miss, rate-limit, call, retry transient errors with backoff + jitter."""
    pname = ctx.model_provider[model]
    key = ResponseCache.make_key(model=model, item=item_id, prompt=cfg.prompt_version,
                                 decoding=cfg.decoding, sample=cfg.sample_idx)
    hit = await _maybe(ctx.cache.get(key))
    if hit is not None:
        return hit, True
    provider, pcfg = ctx.providers[pname], ctx.provider_cfgs[pname]
    job_id = f"{session_id}:{step}:{uuid.uuid4().hex[:8]}"   # new id per issue of this call
    for attempt in range(cfg.max_retries + 1):
        await ctx.limiters[pname].acquire(pcfg.est_tokens_per_call)
        t0 = time.time()

        async def log(outcome, resp=None):
            await _maybe(ctx.store.log_attempt(
                session_id, step, attempt, pname, model, item_id, outcome, t0,
                time.time() - t0, job_id=job_id,
                tokens=None if resp is None else resp.input_tokens + resp.output_tokens,
                cost_usd=None if resp is None else resp.cost_usd))

        try:
            resp = await provider.answer(model, item_id)
        except TransientError:
            await log("transient_error")
            if attempt == cfg.max_retries:
                raise
            ctx.retries += 1
            await asyncio.sleep(min(8.0, 0.1 * 2 ** attempt) * random.uniform(0.5, 1.5))
            continue
        except Exception:
            await log("error")
            raise
        await log("ok", resp)
        break
    value = asdict(resp)
    await _maybe(ctx.cache.put(key, value))
    return value, False


async def run_session(ctx: Context, run_name: str, model: str, cfg: SessionConfig,
                      crash_after_step: int | None = None) -> dict:
    session_id = f"{run_name}:{model}"
    await _maybe(ctx.store.ensure_session(session_id, run_name, model,
                                          ctx.model_provider[model], asdict(cfg)))
    st = await _maybe(ctx.store.load_state(session_id))          # resume from log if this session crashed
    idx = ctx.bank.index()

    def ability():
        ii = np.array([idx[i] for i, _ in st.answered], dtype=int)
        y = np.array([c for _, c in st.answered], dtype=float)
        return estimate_ability(ctx.bank.a[ii], ctx.bank.b[ii], y)

    theta, se = ability()
    while not st.done:
        if st.pending is None:
            n = len(st.answered)
            allowed = None if ctx.available is None else ctx.available.get(model, set())
            pool = len(ctx.bank) if allowed is None else len(allowed)
            if n >= min(cfg.max_items, pool) or (n >= cfg.min_items and se < cfg.se_target):
                break
            used = {idx[i] for i, _ in st.answered}
            item_id = ctx.bank.item_ids[select_next(cfg.selector, theta, ctx.bank, used,
                                                    session_id, n, allowed)]
            await _maybe(ctx.store.append(session_id, n, "item_selected", item_id))  # write-ahead
            st.pending = (n, item_id)

        step, item_id = st.pending
        if crash_after_step is not None and step >= crash_after_step:
            raise SimulatedCrash(f"{session_id} crashed at step {step}")
        value, cached = await call_item(ctx, model, item_id, cfg, session_id=session_id, step=step)
        await _maybe(ctx.store.append(session_id, step, "answer_recorded", item_id,
                                      correct=int(value["correct"]), cached=int(cached),
                                      cost=0.0 if cached else value["cost_usd"]))
        st.answered.append((item_id, int(value["correct"])))
        st.pending = None
        theta, se = ability()

    await _maybe(ctx.store.finish(session_id, theta, se, len(st.answered)))
    return {"model": model, "theta": theta, "se": se, "n_items": len(st.answered)}


async def run_many(ctx: Context, run_name: str, models: list[str], cfg: SessionConfig,
                   concurrency: int = 64, crash_after_step: int | None = None) -> list:
    sem = asyncio.Semaphore(concurrency)

    async def one(m):
        async with sem:
            return await run_session(ctx, run_name, m, cfg, crash_after_step)

    # return_exceptions: one crashed session must not kill the others
    return await asyncio.gather(*(one(m) for m in models), return_exceptions=True)
