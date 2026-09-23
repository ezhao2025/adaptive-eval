"""Turn generated scenes into items for the evaluation harness.

Item format matches adaptive_eval.real_provider: question, answer, optional aliases, plus
image_b64 and media_type. Extra keys (subtask, scene) are metadata, and they are part of the
item's content hash, so changing a generator setting invalidates old cached answers.
"""
from __future__ import annotations

import numpy as np

from .render import png_b64, render, visible_pixels
from .scene import counting_questions, generate, mark_two, relation_questions


def build(n: int, seed: int = 0, *, nx: int = 3, ny: int = 3, max_h: int = 3,
          fill: float = 0.8, size: int = 28, kinds: tuple[str, ...] = ("count", "relation"),
          prefix: str = "sp") -> dict[str, dict]:
    """n scenes -> one item per applicable question. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    items: dict[str, dict] = {}
    for i in range(n):
        scene = generate(rng, nx, ny, max_h, fill)
        pixels = visible_pixels(scene, size)
        scene = mark_two(scene, pixels, rng)
        image = png_b64(render(scene, size))
        qs = []
        if "count" in kinds:
            qs += counting_questions(scene, set(pixels))
        if "relation" in kinds:
            qs += relation_questions(scene)
        meta = {"nx": nx, "ny": ny, "max_h": max_h, "fill": fill,
                "heights": scene.heights.tolist(), "marks": {k: list(v)
                                                             for k, v in scene.marks.items()}}
        for q in qs:
            iid = f"{prefix}-{i:04d}-{q['subtask']}"
            items[iid] = {"question": q["question"], "answer": q["answer"],
                          **({"aliases": q["aliases"]} if q.get("aliases") else {}),
                          **({"options": q["options"]} if q.get("options") else {}),
                          "image_b64": image, "media_type": "image/png",
                          "subtask": q["subtask"], "scene": meta}
    return items
