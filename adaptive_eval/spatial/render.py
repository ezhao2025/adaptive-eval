"""Isometric renderer for stacked-cube scenes (Pillow only -- no 3D engine needed).

Unit cube at (x, y, z) projects to a hexagon: top rhombus plus left and right faces.
Cubes are drawn back to front by (x + y + z), which is exactly the viewing direction, so
nearer cubes paint over farther ones and occlusion in the image matches Scene.hidden().
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .scene import Scene

GREY = {"top": (222, 222, 226), "left": (168, 170, 178), "right": (196, 198, 205)}
COLOURS = {
    "red": {"top": (242, 140, 130), "left": (186, 66, 56), "right": (214, 96, 86)},
    "blue": {"top": (140, 172, 242), "left": (54, 92, 186), "right": (84, 126, 214)},
    "green": {"top": (150, 214, 150), "left": (44, 132, 58), "right": (74, 170, 88)},
}
EDGE = (60, 62, 70)


def _corners(x: int, y: int, z: int, s: int, ox: int, oy: int):
    """Screen position of the cube's 7 visible corners."""
    cx = ox + (x - y) * s
    cy = oy + (x + y) * s // 2 - z * s
    return {"t0": (cx, cy - s), "t1": (cx + s, cy - s // 2), "t2": (cx, cy),
            "t3": (cx - s, cy - s // 2), "b1": (cx + s, cy + s // 2), "b2": (cx, cy + s),
            "b3": (cx - s, cy + s // 2)}


def render(scene: Scene, size: int = 28, pad: int = 40, bg=(250, 250, 250)) -> Image.Image:
    nx, ny = scene.heights.shape
    nz = max(1, scene.tallest_column_height())
    w = (nx + ny) * size + 2 * pad
    h = (nx + ny) * size // 2 + nz * size + 2 * pad
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    ox, oy = pad + ny * size, pad + nz * size
    mark_of = {c: name for name, c in scene.marks.items()}
    for cube in sorted(scene.cubes, key=lambda c: c[0] + c[1] + c[2]):   # painter's order
        p = _corners(*cube, size, ox, oy)
        palette = COLOURS.get(mark_of.get(cube), GREY)
        d.polygon([p["t0"], p["t1"], p["t2"], p["t3"]], fill=palette["top"], outline=EDGE)
        d.polygon([p["t3"], p["t2"], p["b2"], p["b3"]], fill=palette["left"], outline=EDGE)
        d.polygon([p["t2"], p["t1"], p["b1"], p["b2"]], fill=palette["right"], outline=EDGE)
    return _crop(img, pad, bg)


def _crop(img: Image.Image, pad: int, bg) -> Image.Image:
    """Trim the empty canvas around the structure: a sparse scene should not float in a sea
    of background, and every item should be framed the same way."""
    box = Image.new("RGB", img.size, bg)
    bbox = ImageChops.difference(img, box).getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - pad // 2), max(0, y0 - pad // 2),
                     min(img.width, x1 + pad // 2), min(img.height, y1 + pad // 2)))


def visible_pixels(scene: Scene, size: int = 28, pad: int = 40) -> dict:
    """How many pixels of each cube survive in the rendered image: draw each cube in its own
    ID colour, in the same painter order, then read back what is left. Exact by construction,
    so ground truth always matches the picture the model is shown."""
    cubes = sorted(scene.cubes, key=lambda c: c[0] + c[1] + c[2])
    nx, ny = scene.heights.shape
    nz = max(1, scene.tallest_column_height())
    w = (nx + ny) * size + 2 * pad
    h = (nx + ny) * size // 2 + nz * size + 2 * pad
    ids = Image.new("I", (w, h), 0)
    d = ImageDraw.Draw(ids)
    ox, oy = pad + ny * size, pad + nz * size
    for i, cube in enumerate(cubes, start=1):
        p = _corners(*cube, size, ox, oy)
        for face in (("t0", "t1", "t2", "t3"), ("t3", "t2", "b2", "b3"), ("t2", "t1", "b1", "b2")):
            d.polygon([p[k] for k in face], fill=i, outline=i)
    flat = np.asarray(ids).ravel()
    counts = np.bincount(flat, minlength=len(cubes) + 1)
    return {cubes[i - 1]: int(counts[i]) for i in range(1, len(cubes) + 1) if counts[i]}


def visible_cubes(scene: Scene, size: int = 28, pad: int = 40) -> set:
    return set(visible_pixels(scene, size, pad))


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
