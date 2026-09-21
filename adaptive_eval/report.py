"""Metrics: savings, ranking agreement, cache hit rate, and the adaptive-vs-random curve."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
from scipy.stats import kendalltau

from .irt import ItemBank, estimate_ability, select_next


def full_reference(d: dict, bank: ItemBank, models: list[str]) -> dict[str, dict]:
    """Ground truth = every item answered. Ability uses the same fixed item params."""
    midx = {m: i for i, m in enumerate(d["models"])}
    didx = {it: j for j, it in enumerate(d["items"])}
    cols = np.array([didx[i] for i in bank.item_ids])
    out = {}
    for m in models:
        y = d["R"][midx[m], cols]
        ok = ~np.isnan(y)
        th, _ = estimate_ability(bank.a[ok], bank.b[ok], y[ok])
        out[m] = {"theta": th, "accuracy": float(np.nanmean(y))}
    return out


def run_report(conn: sqlite3.Connection, d: dict, bank: ItemBank, run_name: str) -> dict:
    rows = conn.execute(
        "SELECT model, theta, n_items, started_at, finished_at FROM sessions"
        " WHERE run_name=? AND status='done'", (run_name,)).fetchall()
    if not rows:
        raise SystemExit(f"no finished sessions for run {run_name!r}")
    models = [r[0] for r in rows]
    ref = full_reference(d, bank, models)
    ev = conn.execute(
        "SELECT COUNT(*), SUM(cached), SUM(cost_usd) FROM events e JOIN sessions s"
        " USING(session_id) WHERE s.run_name=? AND e.type='answer_recorded'", (run_name,)).fetchone()
    answered, cached, cost = ev[0], ev[1] or 0, ev[2] or 0.0
    full_calls = len(models) * len(bank)
    theta_ad = [r[1] for r in rows]
    report = {
        "run": run_name, "models": len(models),
        "items_per_model_mean": float(np.mean([r[2] for r in rows])),
        "answers_used": answered, "paid_calls": answered - cached,
        "full_eval_calls": full_calls,
        "call_reduction_vs_full": 1 - answered / full_calls,
        "cache_hit_rate": cached / answered if answered else 0.0,
        "cost_usd": round(cost, 4),
        "kendall_tau_vs_full_theta": kendalltau(theta_ad, [ref[m]["theta"] for m in models])[0],
        "kendall_tau_vs_full_accuracy": kendalltau(theta_ad, [ref[m]["accuracy"] for m in models])[0],
        "wall_clock_s": max(r[4] for r in rows) - min(r[3] for r in rows),
    }
    if "truth" in d:   # synthetic only: compare against the true abilities
        midx = {m: i for i, m in enumerate(d["models"])}
        true = [d["truth"]["theta"][midx[m]] for m in models]
        report["kendall_tau_vs_true_theta"] = kendalltau(theta_ad, true)[0]
    return report


def offline_curve(d: dict, bank: ItemBank, models: list[str], budgets: list[int]) -> list[dict]:
    """Fast simulation (no API, no async): ranking quality at a fixed item budget,
    adaptive vs random. This is where the 'X% fewer calls at equal accuracy' claim comes from."""
    midx = {m: i for i, m in enumerate(d["models"])}
    didx = {it: j for j, it in enumerate(d["items"])}
    ref = full_reference(d, bank, models)
    ref_theta = [ref[m]["theta"] for m in models]
    out = []
    for k in budgets:
        row = {"items": k}
        for sel in ("max_info", "random"):
            thetas = []
            for m in models:
                allowed = {i for i, it in enumerate(bank.item_ids)
                           if not np.isnan(d["R"][midx[m], didx[it]])}
                order, ys, th = [], [], 0.0
                for step in range(min(k, len(allowed))):
                    i = select_next(sel, th, bank, set(order), f"curve:{m}", step, allowed)
                    order.append(i)
                    ys.append(d["R"][midx[m], didx[bank.item_ids[i]]])
                    ii = np.array(order)
                    th, _ = estimate_ability(bank.a[ii], bank.b[ii], np.array(ys))
                thetas.append(th)
            row[f"tau_{sel}"] = float(kendalltau(thetas, ref_theta)[0])
        out.append(row)
    return out


def items_needed(curve: list[dict], selector: str, target_tau: float) -> int | None:
    for row in curve:
        if row[f"tau_{selector}"] >= target_tau:
            return row["items"]
    return None


def dump(obj) -> str:
    return json.dumps(obj, indent=2, default=float)
