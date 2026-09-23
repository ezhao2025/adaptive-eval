"""Build the files for a tiny real-API smoke run (9.2): 10 text items, one model.

    python scripts/make_smoke_run.py --model claude-haiku-4-5-20251001

Writes data/items_smoke.json (questions + answers), data/smoke_data.json (which models and
items exist) and data/smoke_params.json (flat item parameters: this run tests plumbing, not
measurement -- real item parameters need answers from many models first).
"""
from __future__ import annotations

import argparse
import json
import pathlib

ITEMS = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4", ["four"]),
    ("What colour is a ripe banana?", "yellow"),
    ("How many sides does a triangle have?", "3", ["three"]),
    ("What planet do humans live on?", "Earth"),
    ("What is the chemical symbol for water?", "H2O"),
    ("Which ocean is the largest?", "Pacific"),
    ("How many days are in a week?", "7", ["seven"]),
    ("What gas do plants absorb from the air?", "CO2", ["carbon dioxide"]),
    ("What is the freezing point of water in Celsius?", "0", ["zero"]),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--out-dir", default="data")
    a = p.parse_args()
    out = pathlib.Path(a.out_dir)
    out.mkdir(exist_ok=True)

    items = {f"item-{i:04d}": {"question": q, "answer": ans,
                               **({"aliases": rest[0]} if rest else {})}
             for i, (q, ans, *rest) in enumerate(ITEMS)}
    ids = list(items)
    (out / "items_smoke.json").write_text(json.dumps(items, indent=2))
    (out / "smoke_data.json").write_text(json.dumps({
        "models": [a.model], "items": ids,
        "R": [[1] * len(ids)],                      # marks every item answerable; not real data
        "model_provider": {a.model: "anthropic"}}))
    (out / "smoke_params.json").write_text(json.dumps({
        "item_ids": ids, "a": [1.0] * len(ids),
        "b": [round(-1.5 + 3 * i / (len(ids) - 1), 3) for i in range(len(ids))],
        "train_models": [], "test_models": [a.model]}))
    print(f"wrote {out}/items_smoke.json, smoke_data.json, smoke_params.json "
          f"({len(ids)} items, model {a.model})")


if __name__ == "__main__":
    main()
