"""Turn generated scenes into items for the evaluation harness.

Item format matches adaptive_eval.real_provider: question, answer, optional aliases/options,
plus image_b64 and media_type. Extra keys (subtask, scene) are metadata, and they are part of
the item's content hash, so changing a generator setting invalidates old cached answers.

Two knobs matter, and the pilot showed why:
  * hidden_frac rejection-samples structures by how much is occluded. Grid size saturates
    (tiers 3-5 all landed at the same difficulty) because once counting collapses, more cubes
    add nothing; occlusion keeps biting.
  * hard=True picks marked cubes that sit close together on screen. Far-apart pairs made the
    relation items trivial: every model scored 1.00 on left/right.
"""
from __future__ import annotations

import numpy as np

from .render import png_b64, render, visible_pixels
from .scene import (counting_questions, generate, mark_n, relation_questions,
                    support_questions, triple_questions)

KINDS = ("count", "relation", "triple", "support")


def _scene(rng, nx, ny, max_h, fill, size, hidden_frac, tries=40):
    """Sample a structure; if hidden_frac is set, keep re-rolling until it lands in range."""
    best = None
    for _ in range(tries):
        scene = generate(rng, nx, ny, max_h, fill)
        pixels = visible_pixels(scene, size)
        frac = 1 - len(pixels) / scene.total()
        if hidden_frac is None:
            return scene, pixels
        lo, hi = hidden_frac
        if lo <= frac <= hi:
            return scene, pixels
        gap = lo - frac if frac < lo else frac - hi
        if best is None or gap < best[0]:
            best = (gap, scene, pixels)
    return best[1], best[2]


def build(n: int, seed: int = 0, *, nx: int = 3, ny: int = 3, max_h: int = 3,
          fill: float = 0.8, size: int = 40, kinds: tuple[str, ...] = KINDS,
          hidden_frac: tuple[float, float] | None = None, hard: bool = True,
          prefix: str = "sp") -> dict[str, dict]:
    """n scenes -> one item per applicable question. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    n_marks = 3 if "triple" in kinds else 2
    items: dict[str, dict] = {}
    for i in range(n):
        scene, pixels = _scene(rng, nx, ny, max_h, fill, size, hidden_frac)
        scene = mark_n(scene, pixels, rng, n_marks, hard=hard)
        image = png_b64(render(scene, size))
        qs = []
        if "count" in kinds:
            qs += counting_questions(scene, set(pixels))
        if "relation" in kinds:
            qs += relation_questions(scene)
        if "triple" in kinds:
            qs += triple_questions(scene)
        if "support" in kinds:
            qs += support_questions(scene, set(pixels))
        meta = {"nx": nx, "ny": ny, "max_h": max_h, "fill": fill,
                "hidden_frac": round(1 - len(pixels) / scene.total(), 3),
                "heights": scene.heights.tolist(),
                "marks": {k: list(v) for k, v in scene.marks.items()}}
        for q in qs:
            iid = f"{prefix}-{i:04d}-{q['subtask']}"
            items[iid] = {"question": q["question"], "answer": q["answer"],
                          **({"aliases": q["aliases"]} if q.get("aliases") else {}),
                          **({"options": q["options"]} if q.get("options") else {}),
                          "image_b64": image, "media_type": "image/png",
                          "subtask": q["subtask"], "scene": meta}
    return items
