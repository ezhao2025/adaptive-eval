"""Spatial benchmark: ground truth must match the rendered image, exactly and repeatably."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from adaptive_eval.spatial.items import build
from adaptive_eval.spatial.render import render, visible_cubes, visible_pixels
from adaptive_eval.spatial.scene import (Scene, counting_questions, generate, mark_n,
                                         mark_two, relation_questions, support_questions,
                                         triple_questions)


def test_solid_block_hides_exactly_its_core():
    """2x2x2: only the cube at (0,0,0) is behind the others."""
    s = Scene(np.full((2, 2), 2))
    assert s.total() == 8
    assert visible_cubes(s) == {c for c in s.cubes if c != (0, 0, 0)}


def test_single_column_is_fully_visible():
    s = Scene(np.array([[3]]))
    assert s.total() == 3 and len(visible_cubes(s)) == 3


def test_neighbours_can_hide_a_cube_together():
    """A cube with nothing at (+1,+1,+1) can still be fully covered by the combination of
    its neighbours. That is why visibility is rasterised rather than derived from a rule."""
    rng = np.random.default_rng(0)
    found = False
    for _ in range(20):
        s = generate(rng, nx=4, ny=4, max_h=4, fill=0.95)
        vis = visible_cubes(s)
        found = any(c not in vis and not s.occupied(c[0] + 1, c[1] + 1, c[2] + 1)
                    for c in s.cubes)
        if found:
            break
    assert found


def test_visible_count_never_exceeds_total_and_matches_pixels():
    rng = np.random.default_rng(3)
    for _ in range(10):
        s = generate(rng, nx=4, ny=4, max_h=4, fill=0.9)
        pixels = visible_pixels(s)
        assert set(pixels) == visible_cubes(s)
        assert 0 < len(pixels) <= s.total()
        assert all(n > 0 for n in pixels.values())


def test_marked_cubes_are_clearly_visible():
    rng = np.random.default_rng(11)
    s = generate(rng, nx=4, ny=4, max_h=4, fill=0.95)
    pixels = visible_pixels(s)
    s = mark_two(s, pixels, rng)
    cutoff = 0.5 * max(pixels.values())
    for cube in s.marks.values():
        assert pixels[cube] >= cutoff


def test_counting_answers_are_consistent():
    rng = np.random.default_rng(5)
    s = generate(rng, nx=3, ny=3, max_h=3)
    vis = visible_cubes(s)
    qs = {q["subtask"]: int(q["answer"]) for q in counting_questions(s, vis)}
    assert qs["count_visible"] + qs["count_hidden"] == qs["count_total"] == s.total()
    assert qs["tallest_column"] == int(s.heights.max())


def test_items_are_deterministic_and_well_formed():
    a = build(3, seed=1)
    b = build(3, seed=1)
    assert a == b and build(3, seed=2) != a
    for iid, it in a.items():
        assert it["question"] and it["answer"]
        img = Image.open(io.BytesIO(base64.b64decode(it["image_b64"])))
        assert img.format == "PNG" and min(img.size) > 50
        assert it["subtask"] in iid


@pytest.mark.parametrize("kinds,expect", [(("count",), "count"), (("relation",), "relation")])
def test_kinds_filter(kinds, expect):
    items = build(2, seed=0, kinds=kinds)
    assert items and all(it["subtask"].startswith(expect) or expect == "count"
                         for it in items.values())


def test_difficulty_knobs_raise_the_hidden_count():
    def mean_hidden(**kw):
        items = build(6, seed=0, kinds=("count",), **kw)
        return np.mean([int(it["answer"]) for iid, it in items.items()
                        if it["subtask"] == "count_hidden"])
    assert mean_hidden(nx=2, ny=2, max_h=2, fill=0.6) < mean_hidden(nx=5, ny=5, max_h=5, fill=0.95)


def test_left_right_matches_what_is_on_screen():
    """Regression: 'left' used to be derived from the x axis, but in this isometric view
    screen position is x - y, so items were mislabelled whenever the y values differed."""
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(25):
        s = generate(rng, 3, 3, 3, 0.85)
        s = mark_two(s, visible_pixels(s, 40), rng)
        answers = {q["subtask"]: q["answer"] for q in relation_questions(s)}
        if "relation_left_right" not in answers:
            continue
        a = np.asarray(render(s, 40)).astype(int)
        red_x = np.argwhere(a[:, :, 0] - a[:, :, 2] > 40)[:, 1].mean()
        blue_x = np.argwhere(a[:, :, 2] - a[:, :, 0] > 40)[:, 1].mean()
        assert answers["relation_left_right"] == ("left" if red_x < blue_x else "right")
        checked += 1
    assert checked >= 5


def _marked(seed=1, n=3, hard=True, nx=4):
    rng = np.random.default_rng(seed)
    s = generate(rng, nx, nx, nx, 0.9)
    px = visible_pixels(s, 40)
    return mark_n(s, px, rng, n, hard=hard), px


def test_triple_questions_have_one_unambiguous_winner():
    s, _ = _marked()
    qs = triple_questions(s)
    assert qs, "three marked cubes should yield three-way questions"
    for q in qs:
        assert q["answer"] in q["options"] == list(("red", "blue", "green"))
    m = s.marks
    answers = {q["subtask"]: q["answer"] for q in qs}
    if "triple_highest" in answers:
        assert answers["triple_highest"] == max(m, key=lambda k: m[k][2])
    if "triple_leftmost" in answers:
        assert answers["triple_leftmost"] == min(m, key=lambda k: m[k][0] - m[k][1])
    if "triple_nearest" in answers:
        assert answers["triple_nearest"] == max(m, key=lambda k: m[k][0] + m[k][1])


def test_support_questions_match_the_structure():
    s, px = _marked()
    qs = {q["subtask"]: q["answer"] for q in support_questions(s, set(px))}
    x, y, z = s.marks["red"]
    assert int(qs["count_ground"]) == s.footprint()
    assert qs["support_on_ground"] == ("yes" if z == 0 else "no")
    assert int(qs["count_above_red"]) == int(s.heights[x, y]) - z - 1


def test_hard_marks_sit_closer_together_than_easy_ones():
    def spread(hard):
        out = []
        for seed in range(8):
            s, _ = _marked(seed=seed, n=2, hard=hard)
            if len(s.marks) < 2:
                continue
            a, b = s.marks.values()
            out.append(abs((a[0] - a[1]) - (b[0] - b[1])) + abs(a[2] - b[2]))
        return np.mean(out)
    assert spread(hard=True) < spread(hard=False)


def test_hidden_fraction_knob_moves_occlusion():
    low = build(6, seed=0, nx=4, ny=4, max_h=4, fill=0.9, kinds=("count",),
                hidden_frac=(0.0, 0.15))
    high = build(6, seed=0, nx=4, ny=4, max_h=4, fill=0.9, kinds=("count",),
                 hidden_frac=(0.35, 0.7))
    def frac(items):
        return np.mean([it["scene"]["hidden_frac"] for it in items.values()])
    assert frac(low) < 0.2 < frac(high)
