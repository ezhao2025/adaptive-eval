"""Generate spatial-benchmark items (and the data/params files to run them).

    python scripts/make_spatial_items.py --scenes 8 --models claude-haiku-4-5-20251001

Difficulty knobs: --nx/--ny (base grid), --max-h (stack height), --fill (density).
Writes data/spatial_items.json, data/spatial_data.json, data/spatial_params.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_eval.spatial.items import build  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=8, help="scenes per difficulty tier")
    p.add_argument("--tiers", default="",
                   help="difficulty tiers as size:fill pairs, e.g. 2:0.7,3:0.85,4:0.9,5:0.95;"
                        " size sets nx, ny and max height. Empty = one tier from --nx/--ny/--max-h")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nx", type=int, default=3)
    p.add_argument("--ny", type=int, default=3)
    p.add_argument("--max-h", type=int, default=3)
    p.add_argument("--fill", type=float, default=0.8)
    p.add_argument("--kinds", default="count,relation")
    p.add_argument("--size", type=int, default=40, help="pixels per cube edge")
    p.add_argument("--models", default="claude-haiku-4-5-20251001",
                   help="comma-separated model ids to evaluate")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--tag", default="spatial")
    a = p.parse_args()

    kinds = tuple(a.kinds.split(","))
    if a.tiers:
        items = {}
        for i, spec in enumerate(a.tiers.split(",")):
            n, fill = spec.split(":")
            n = int(n)
            items.update(build(a.scenes, a.seed + 1000 * i, nx=n, ny=n, max_h=n,
                               fill=float(fill), size=a.size, kinds=kinds,
                               prefix=f"{a.tag}t{n}"))
    else:
        items = build(a.scenes, a.seed, nx=a.nx, ny=a.ny, max_h=a.max_h, fill=a.fill,
                      size=a.size, kinds=kinds, prefix=a.tag)
    ids, models = list(items), a.models.split(",")
    out = pathlib.Path(a.out_dir)
    out.mkdir(exist_ok=True)
    (out / f"{a.tag}_items.json").write_text(json.dumps(items))
    (out / f"{a.tag}_data.json").write_text(json.dumps({
        "models": models, "items": ids, "R": [[1] * len(ids) for _ in models],
        "model_provider": {m: "anthropic" for m in models}}))
    (out / f"{a.tag}_params.json").write_text(json.dumps({
        "item_ids": ids, "a": [1.0] * len(ids),
        "b": [0.0] * len(ids),                 # flat: real parameters need many models first
        "train_models": [], "test_models": models}))
    by_kind: dict[str, int] = {}
    for it in items.values():
        by_kind[it["subtask"]] = by_kind.get(it["subtask"], 0) + 1
    mb = len(json.dumps(items)) / 1e6
    n_tiers = len(a.tiers.split(",")) if a.tiers else 1
    print(f"{len(items)} items from {a.scenes * n_tiers} scenes in {n_tiers} tier(s) "
          f"({mb:.1f} MB) -> {out}/{a.tag}_items.json")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
