"""Stacked-cube structures with per-task ground truth.

A structure is a height map on a grid: column (x, y) holds h[x][y] cubes stacked from the
ground, so every cube is supported (no floating cubes to reason about).

Axes, as stated to the model in every prompt:
  +x goes right, +y goes back (away from the viewer), +z goes up.

Occlusion is NOT computed analytically here. A cube can be fully covered by a combination
of neighbours, not just by the single cube at (x+1, y+1, z+1), so visibility comes from
render.visible_cubes(), which rasterises the scene and reports which cubes actually survive.
Ground truth then matches the image the model is shown, by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

# In this isometric view BOTH +x and +y come toward the viewer (down-right and down-left).
# So screen-left/right is x - y, and nearness to the viewer is x + y. Relation questions are
# asked in the viewer's frame, which is what someone looking at the picture actually sees.
VIEW = ("the cubes are drawn in an isometric view")


@dataclass
class Scene:
    heights: np.ndarray                       # (nx, ny) ints: cubes stacked per column
    marks: dict[str, tuple[int, int, int]] = field(default_factory=dict)   # colour -> cube

    @property
    def cubes(self) -> list[tuple[int, int, int]]:
        nx, ny = self.heights.shape
        return [(x, y, z) for x in range(nx) for y in range(ny)
                for z in range(int(self.heights[x, y]))]

    def occupied(self, x: int, y: int, z: int) -> bool:
        nx, ny = self.heights.shape
        return 0 <= x < nx and 0 <= y < ny and 0 <= z < int(self.heights[x, y])

    # ---- ground truth -------------------------------------------------------------
    def total(self) -> int:
        return int(self.heights.sum())

    def tallest_column_height(self) -> int:
        return int(self.heights.max())

    def footprint(self) -> int:
        return int((self.heights > 0).sum())


def generate(rng: np.random.Generator, nx: int = 3, ny: int = 3, max_h: int = 3,
             fill: float = 0.8) -> Scene:
    """Random structure. nx, ny, max_h and fill are the difficulty knobs."""
    while True:
        h = rng.integers(0, max_h + 1, size=(nx, ny))
        h = h * (rng.random((nx, ny)) < fill)
        if h.sum() >= 2:                      # degenerate structures make trivial items
            return Scene(h.astype(int))


MARK_COLOURS = ("red", "blue", "green")


def mark_n(scene: Scene, pixels: dict, rng: np.random.Generator, n: int = 2,
           min_frac: float = 0.5, hard: bool = False) -> Scene:
    """Colour n clearly visible cubes so relation questions are answerable.

    A cube showing only a sliver is excluded: the question should test spatial reasoning,
    not whether the model can spot 20 pixels of colour. With hard=True, prefer cubes that sit
    close together on screen -- far-apart pairs made left/right trivial (every model scored
    1.00), while near-neighbours force an actual comparison."""
    if not pixels:
        return scene
    cutoff = min_frac * max(pixels.values())
    vis = sorted(c for c, v in pixels.items() if v >= cutoff)
    if len(vis) < n:
        return scene
    best = None
    for _ in range(60):
        pick = [vis[i] for i in rng.choice(len(vis), n, replace=False)]
        if n > 2 and not all(len({f(c) for c in pick}) == n for f in
                             (lambda c: c[0] - c[1], lambda c: c[2], lambda c: c[0] + c[1])):
            continue     # three-way questions need a unique winner on every dimension
        if len({(c[0] - c[1], c[2]) for c in pick}) < n:      # need distinguishable positions
            continue
        spread = max(abs(a[0] - a[1] - (b[0] - b[1])) + abs(a[2] - b[2])
                     for a in pick for b in pick)
        if best is None or (spread < best[0]) == hard:
            best = (spread, pick)
    if best:
        scene.marks = {MARK_COLOURS[i]: tuple(map(int, c)) for i, c in enumerate(best[1])}
    return scene


def mark_two(scene: Scene, pixels: dict, rng: np.random.Generator,
             min_frac: float = 0.5) -> Scene:
    return mark_n(scene, pixels, rng, 2, min_frac)


# ---- questions ---------------------------------------------------------------------
def _number(n: int) -> dict:
    aliases = [NUMBER_WORDS[n]] if n < len(NUMBER_WORDS) else []
    return {"answer": str(n), "aliases": aliases}


def counting_questions(scene: Scene, visible: set) -> list[dict]:
    n_visible, total = len(visible), scene.total()
    """The Spatial-IQ chain: what you can see, what must be there, what is hidden."""
    return [
        {"subtask": "count_visible",
         "question": ("The image shows a structure built from identical cubes stacked on a "
                      "ground plane. How many cubes are at least partly visible from this "
                      "viewpoint?"),
         **_number(n_visible)},
        {"subtask": "count_total",
         "question": ("The image shows a structure built from identical cubes stacked on a "
                      "ground plane. Every cube rests on the ground or directly on another "
                      "cube. How many cubes are in the structure in total, including any you "
                      "cannot see?"),
         **_number(total)},
        {"subtask": "count_hidden",
         "question": ("The image shows a structure built from identical cubes stacked on a "
                      "ground plane. Every cube rests on the ground or directly on another "
                      "cube. How many cubes are completely hidden from this viewpoint?"),
         **_number(total - n_visible)},
        {"subtask": "tallest_column",
         "question": ("The image shows a structure built from identical cubes. How many cubes "
                      "tall is its tallest stack?"),
         **_number(scene.tallest_column_height())},
    ]


def relation_questions(scene: Scene) -> list[dict]:
    """Easy end of the scale: IRT needs items that models can actually get right.
    Everything is phrased in the viewer's frame -- see the note on VIEW above."""
    if "red" not in scene.marks:
        return []
    (rx, ry, rz), (bx, by, bz) = scene.marks["red"], scene.marks["blue"]
    r_side, b_side = rx - ry, bx - by            # screen horizontal
    r_near, b_near = rx + ry, bx + by            # distance toward the viewer
    out = []
    if rz != bz:
        out.append({"subtask": "relation_height",
                    "question": ("In the image, two cubes are coloured. Which one is higher "
                                 "off the ground, the red cube or the blue cube? Answer 'red' "
                                 "or 'blue'."),
                    "answer": "red" if rz > bz else "blue", "options": ["red", "blue"]})
    if r_side != b_side:
        out.append({"subtask": "relation_left_right",
                    "question": ("In the image, two cubes are coloured. Is the red cube to the "
                                 "left or to the right of the blue cube, as you see them? "
                                 "Answer 'left' or 'right'."),
                    "answer": "left" if r_side < b_side else "right",
                    "options": ["left", "right"]})
    if r_near != b_near:
        out.append({"subtask": "relation_near_far",
                    "question": ("In the image, two cubes are coloured. Which one is nearer to "
                                 "you, the viewer? Answer 'red' or 'blue'."),
                    "answer": "red" if r_near > b_near else "blue", "options": ["red", "blue"]})
    return out


def support_questions(scene: Scene, visible: set) -> list[dict]:
    """Mid-difficulty items: they need the structure, not just the surface. The pilot had a
    gap between relations (b about -1.5) and counting (b about +1) with nothing in between."""
    out = [{"subtask": "count_ground",
            "question": ("The image shows a structure built from identical cubes stacked on a "
                         "ground plane. How many cubes are touching the ground?"),
            **_number(scene.footprint())}]
    if "red" in scene.marks:
        rx, ry, rz = scene.marks["red"]
        out.append({"subtask": "support_on_ground",
                    "question": ("In the image, one cube is coloured red. Is the red cube "
                                 "resting directly on the ground? Answer 'yes' or 'no'."),
                    "answer": "yes" if rz == 0 else "no", "options": ["yes", "no"]})
        above = int(scene.heights[rx, ry]) - rz - 1
        out.append({"subtask": "count_above_red",
                    "question": ("In the image, one cube is coloured red. How many cubes are "
                                 "stacked directly on top of it, in the same column?"),
                    **_number(above)})
    return out


def triple_questions(scene: Scene) -> list[dict]:
    """Three-way comparisons: chance is 1/3 instead of 1/2, and the model has to order three
    things rather than compare two. The two-cube versions were at ceiling."""
    if "green" not in scene.marks:
        return []
    m = scene.marks
    names = list(m)
    side = {k: m[k][0] - m[k][1] for k in names}         # screen horizontal
    near = {k: m[k][0] + m[k][1] for k in names}         # toward the viewer
    high = {k: m[k][2] for k in names}
    out = []
    for subtask, values, text in [
            ("triple_leftmost", {k: -v for k, v in side.items()},
         "Which of the three coloured cubes is furthest to the left, as you see them?"),
            ("triple_highest", high, "Which of the three coloured cubes is highest off the "
                                     "ground?"),
            ("triple_nearest", near, "Which of the three coloured cubes is nearest to you, "
                                     "the viewer?")]:
        best = max(values.values())
        winners = [k for k in names if values[k] == best]
        if len(winners) == 1:                            # skip ties: no single right answer
            out.append({"subtask": subtask,
                        "question": f"In the image, three cubes are coloured red, blue and "
                                    f"green. {text} Answer 'red', 'blue' or 'green'.",
                        "answer": winners[0], "options": list(MARK_COLOURS)})
    return out
