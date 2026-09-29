"""Report on a live Design C placement run (adaptive_eval.c.scheduler --live-models).

    python scripts/live_report.py --schema live --run-name live-v1 \
        --bank data/sp6_bank2d.json --anchors data/sp6_matrix.json \
        --batch data/sp6_new_matrix.json

For each live model: where the run placed it vs where its full batch answers (all 746 items,
from scripts/vllm_batch.py) put it; how often a live answer matched the same model's batch
answer on the same item (same prompt, temperature 0); and the real per-call latency.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import asyncpg
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_eval.c.ranking import Bank2D, score  # noqa: E402
from adaptive_eval.c.scheduler import load_anchors  # noqa: E402


async def fetch(dsn, schema, run):
    c = await asyncpg.connect(dsn, server_settings={"search_path": schema})
    try:
        ans = await c.fetch("SELECT s.model, e.item_id, e.correct FROM events e JOIN sessions s"
                            " USING (session_id) WHERE s.run_name=$1 AND"
                            " e.type='answer_recorded' ORDER BY e.id", run)
        lat = await c.fetch("SELECT model, status, latency_s FROM call_attempts"
                            " WHERE NOT speculative")
        return ans, lat
    finally:
        await c.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schema", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--bank", default="data/sp6_bank2d.json")
    p.add_argument("--anchors", default="data/sp6_matrix.json")
    p.add_argument("--batch", default="data/sp6_new_matrix.json")
    p.add_argument("--w-count", type=float, default=0.5)
    a = p.parse_args()
    dsn = os.environ.get("PG_DSN") or sys.exit("set PG_DSN (source .env.sh)")

    bank, _ = Bank2D.load(a.bank)
    w, idx = bank.weights(a.w_count), bank.index()
    anchors = load_anchors(a.anchors)
    batch = json.load(open(a.batch))
    batch_ans = {m: dict(zip(batch["items"], row)) for m, row in zip(batch["models"], batch["R"])}
    rows, lat = asyncio.run(fetch(dsn, a.schema, a.run_name))
    live = {}
    for r in rows:
        live.setdefault(r["model"], {})[r["item_id"]] = int(r["correct"])

    def full_score(answers):
        return float(sum(w[idx[it]] * y for it, y in answers.items() if it in idx))

    anchor_scores = {m: full_score(v) for m, v in anchors.items()}
    for m, got in live.items():
        est = score(bank, w, {idx[it]: y for it, y in got.items()})
        print(f"{m}: {len(got)} live answers")
        est_rank = 1 + sum(s > est.s for s in anchor_scores.values())
        line = f"  placed #{est_rank} of {len(anchors) + 1} (score {est.s:.3f} +/- {np.sqrt(est.v):.3f})"
        if m in batch_ans and all(v is not None for v in batch_ans[m].values()):
            true = full_score(batch_ans[m])
            true_rank = 1 + sum(s > true for s in anchor_scores.values())
            same = [got[it] == batch_ans[m][it] for it in got]
            line += (f"; true #{true_rank} (all 746 batch answers: {true:.3f})\n"
                     f"  live answer graded the same as the batch answer on "
                     f"{sum(same)}/{len(same)} items ({np.mean(same):.1%})")
        print(line)
        mine = [x for x in lat if x["model"] == m]
        ok = [x["latency_s"] for x in mine if x["status"] == "ok"]
        if ok:
            print(f"  latency per call: median {np.median(ok):.2f}s, p95 "
                  f"{np.percentile(ok, 95):.2f}s; {len(mine) - len(ok)} failed attempts retried")
    print("\nleaderboard (anchors exact; * = live, placed from its live answers):")
    board = [(s, m, False) for m, s in anchor_scores.items()]
    board += [(score(bank, w, {idx[it]: y for it, y in got.items()}).s, m, True)
              for m, got in live.items()]
    for k, (s, m, is_live) in enumerate(sorted(board, reverse=True), 1):
        print(f"  {k:2d}. {'*' if is_live else ' '} {m.split('/')[-1]:36s} {s:.3f}")


if __name__ == "__main__":
    main()
