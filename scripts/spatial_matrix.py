"""Turn a finished spatial run into a response matrix for IRT.

    python scripts/spatial_matrix.py --schema sp3 --run-name spatial-v1 \
        --items data/spatial_items.json --out data/spatial_matrix.json

Reads answer_recorded events from Postgres and writes adaptive_eval.data's JSON format
(null where a model was never asked an item), then reports coverage per model. Fit it with:
    python -m adaptive_eval.cli fit --data data/spatial_matrix.json --out data/spatial_irt.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import asyncpg


async def fetch(dsn: str, schema: str, run_name: str) -> list:
    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetch(
            f'SELECT s.model, e.item_id, e.correct FROM "{schema}".events e'
            f' JOIN "{schema}".sessions s USING (session_id)'
            f" WHERE e.type='answer_recorded' AND s.run_name=$1", run_name)
    finally:
        await conn.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--schema", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--items", default="data/spatial_items.json")
    p.add_argument("--out", default="data/spatial_matrix.json")
    p.add_argument("--provider", default="anthropic")
    a = p.parse_args()
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        sys.exit("set PG_DSN (source .env.sh)")

    rows = asyncio.run(fetch(dsn, a.schema, a.run_name))
    if not rows:
        sys.exit(f"no answers for run {a.run_name!r} in schema {a.schema!r}")
    items = list(json.load(open(a.items)))
    models = sorted({r["model"] for r in rows})
    answers = {(r["model"], r["item_id"]): int(r["correct"]) for r in rows}
    R = [[answers.get((m, it)) for it in items] for m in models]

    json.dump({"models": models, "items": items, "R": R,
               "model_provider": {m: a.provider for m in models},
               "source": f"{a.schema}.{a.run_name}"}, open(a.out, "w"))
    print(f"{len(models)} models x {len(items)} items -> {a.out}")
    for m, row in zip(models, R):
        seen = [v for v in row if v is not None]
        acc = sum(seen) / len(seen) if seen else 0.0
        print(f"  {m:34s} answered {len(seen):4d}/{len(items)}  accuracy {acc:.2f}")
    flat = [v for row in R for v in row if v is not None]
    print(f"  overall accuracy {sum(flat) / len(flat):.3f}; "
          f"{sum(v is None for row in R for v in row)} missing cells")
    if len(models) < 15:
        print(f"\nNOTE: {len(models)} models is below the usual floor (~15-30) for stable item"
              "\nparameters. Treat the fit as a pipeline check, not a measurement.")


if __name__ == "__main__":
    main()
