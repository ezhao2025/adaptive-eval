"""Import raw VLM replies (from scripts/vllm_batch.py on a rented GPU) into a run.

    python scripts/import_responses.py --schema sp5 --run-name cal-v1 \
        --items data/cal_items.json --responses responses.jsonl

Grades every reply with the repo's grader -- the same code the API and MLX providers use, so
models are comparable -- and writes one finished session per model, with an answer_recorded
event per item. After this, spatial_matrix.py and the IRT fit see them like any other model.
Re-running is safe: events are idempotent, so a re-import corrects nothing but duplicates
nothing either.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_eval.b import pgstore  # noqa: E402
from adaptive_eval.real_provider import extract, grade, load_items  # noqa: E402


async def main_async(a) -> None:
    items = load_items(a.items)
    by_model: dict[str, list] = defaultdict(list)
    with open(a.responses) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_model[r["model"]].append(r)

    pool = await pgstore.connect(os.environ["PG_DSN"], a.schema)
    store = pgstore.PgEventStore(pool)
    try:
        for model, rows in sorted(by_model.items()):
            sid = f"{a.run_name}:{model}"
            await store.ensure_session(sid, a.run_name, model, "vllm", {"source": a.responses})
            n_ok = n_blank = 0
            for step, r in enumerate(sorted(rows, key=lambda x: x["item_id"])):
                item = items.get(r["item_id"])
                if item is None:
                    continue
                correct = grade(r["text"], item)
                n_ok += correct
                n_blank += extract(r["text"], item) is None
                await store.append(sid, step, "item_selected", r["item_id"])
                await store.append(sid, step, "answer_recorded", r["item_id"],
                                   correct=int(correct), cached=0, cost=0.0)
            await pool.execute(
                "UPDATE sessions SET status='done', n_items=$1, finished_at=$2"
                " WHERE session_id=$3", len(rows), time.time(), sid)
            print(f"{model:45s} {n_ok:4d}/{len(rows)} correct ({n_ok / len(rows):.2f})"
                  f"  {n_blank} unparseable")
    finally:
        await pool.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--schema", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--responses", required=True)
    a = p.parse_args()
    if not os.environ.get("PG_DSN"):
        sys.exit("set PG_DSN (source .env.sh)")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
