"""Synthetic cage generators for the Part 7 validation suite: tetrahedron, cube (both
convex, no reflex edges), and the non-convex suite (L-shape, U-shape, star, dumbbell
with a thin neck), each built by extruding a hand-specified 2D cross-section so its
reflex edges are known analytically by construction.

Every builder self-checks global winding via cage.winding_number at a known interior
point before returning, and flips all faces if the mesh came out inside-out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cage import winding_number  # noqa: E402


def _verify_and_fix_orientation(V: np.ndarray, F: np.ndarray, interior_point: np.ndarray) -> np.ndarray:
    wn = winding_number(interior_point[None, :], V, F)[0]
    if wn < 0.5:
        if wn > -0.5:
            raise ValueError(f"winding number {wn:.3f} at claimed interior point is neither ~1 nor ~-1; "
                              f"the point is probably not actually interior")
        F = F[:, [0, 2, 1]]
    return F


# ---------------------------------------------------------------------------
# Convex primitives
# ---------------------------------------------------------------------------

def tetrahedron(scale: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    V = scale * np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64)
    F = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ], dtype=np.int64)
    F = _verify_and_fix_orientation(V, F, V.mean(axis=0))
    return V, F


def cube(side: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    h = side / 2.0
    V = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float64)
    quads = [
        (0, 3, 2, 1),  # bottom, -Z
        (4, 5, 6, 7),  # top, +Z
        (0, 1, 5, 4),  # front, -Y
        (3, 7, 6, 2),  # back, +Y
        (0, 4, 7, 3),  # left, -X
        (1, 2, 6, 5),  # right, +X
    ]
    F = np.array([[q[0], q[1], q[2]] for q in quads] + [[q[0], q[2], q[3]] for q in quads], dtype=np.int64)
    F = _verify_and_fix_orientation(V, F, np.zeros(3))
    return V, F


# ---------------------------------------------------------------------------
# 2D cross-section extrusion for the non-convex suite
# ---------------------------------------------------------------------------

def _polygon_signed_area_and_centroid(poly: np.ndarray) -> tuple[float, np.ndarray]:
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    return area, np.array([cx, cy])


def _ear_clip(poly_ccw: np.ndarray) -> list[tuple[int, int, int]]:
    """Standard O(n^2) ear-clipping triangulation of a simple CCW polygon. Handles
    non-convex, non-star-shaped polygons (unlike a single-apex fan), which the U and
    dumbbell cross-sections below need. Returns triangles as (prev, cur, next)
    index triples into ``poly_ccw``, each with the same (CCW) orientation as input.
    """
    def cross2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def point_in_tri(p, a, b, c):
        d1 = cross2(p, a, b)
        d2 = cross2(p, b, c)
        d3 = cross2(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    idx = list(range(len(poly_ccw)))
    tris: list[tuple[int, int, int]] = []

    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 10 * len(poly_ccw) ** 2:
            raise RuntimeError("ear clipping did not converge; polygon may be malformed")

        n = len(idx)
        clipped = False
        for i in range(n):
            prev_i, cur_i, next_i = idx[(i - 1) % n], idx[i], idx[(i + 1) % n]
            a, b, c = poly_ccw[prev_i], poly_ccw[cur_i], poly_ccw[next_i]
            if cross2(a, b, c) <= 0:  # reflex or degenerate vertex: not a valid ear tip
                continue

            is_ear = True
            for j in idx:
                if j in (prev_i, cur_i, next_i):
                    continue
                if point_in_tri(poly_ccw[j], a, b, c):
                    is_ear = False
                    break
            if is_ear:
                tris.append((prev_i, cur_i, next_i))
                idx.pop(i)
                clipped = True
                break

        if not clipped:
            raise RuntimeError("no ear found; polygon may be self-intersecting or wrongly wound")

    tris.append((idx[0], idx[1], idx[2]))
    return tris


def extrude_polygon(poly2d: np.ndarray, z_lo: float, z_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Extrudes a simple 2D polygon (any winding) into a triangulated solid, capped
    top/bottom via ear clipping (robust for non-star-shaped polygons like the U)."""
    area, _ = _polygon_signed_area_and_centroid(poly2d)
    if area < 0:
        poly2d = poly2d[::-1]

    cap_tris = _ear_clip(poly2d)

    n = len(poly2d)
    bottom = np.column_stack([poly2d, np.full(n, z_lo)])
    top = np.column_stack([poly2d, np.full(n, z_hi)])
    V = np.vstack([bottom, top])

    bottom_idx = np.arange(n)
    top_idx = np.arange(n) + n

    F = []
    for (p, c, nx) in cap_tris:
        F.append([top_idx[p], top_idx[c], top_idx[nx]])       # top cap, CCW -> +Z
        F.append([bottom_idx[p], bottom_idx[nx], bottom_idx[c]])  # bottom cap, reversed -> -Z
    for i in range(n):
        j = (i + 1) % n
        F.append([bottom_idx[i], bottom_idx[j], top_idx[j]])    # side wall, standard extrusion winding
        F.append([bottom_idx[i], top_idx[j], top_idx[i]])

    F = np.array(F, dtype=np.int64)

    # Any point strictly inside a cap triangle, offset to mid-height, is a valid
    # interior point for the global orientation self-check.
    p, c, nx = cap_tris[0]
    interior_xy = (poly2d[p] + poly2d[c] + poly2d[nx]) / 3.0
    interior_point = np.array([interior_xy[0], interior_xy[1], (z_lo + z_hi) / 2.0])
    F = _verify_and_fix_orientation(V, F, interior_point)
    return V, F


def l_shape(z_half: float = 0.3, xy_scale: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    poly = np.array([
        [0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2],
    ], dtype=np.float64)
    poly = (poly - poly.mean(axis=0)) * xy_scale
    return extrude_polygon(poly, -z_half, z_half)


def u_shape(z_half: float = 0.3, xy_scale: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    poly = np.array([
        [0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3],
    ], dtype=np.float64)
    poly = (poly - poly.mean(axis=0)) * xy_scale
    return extrude_polygon(poly, -z_half, z_half)


def star_shape(n_points: int = 5, outer_r: float = 0.35, inner_r: float = 0.15,
               z_half: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    k = 2 * n_points
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, k, endpoint=False)
    radii = np.where(np.arange(k) % 2 == 0, outer_r, inner_r)
    poly = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    return extrude_polygon(poly, -z_half, z_half)


def dumbbell(z_half: float = 0.25, xy_scale: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Two boxes connected by a thin neck: the neck's two pairs of reflex corners are
    the primary event surfaces this shape exists to exercise."""
    poly = np.array([
        [0, 0], [1, 0], [1, 0.4], [1.6, 0.4], [1.6, 0], [2.6, 0],
        [2.6, 1], [1.6, 1], [1.6, 0.6], [1, 0.6], [1, 1], [0, 1],
    ], dtype=np.float64)
    poly = (poly - poly.mean(axis=0)) * xy_scale
    return extrude_polygon(poly, -z_half, z_half)


NON_CONVEX_SUITE = {
    "l_shape": l_shape,
    "u_shape": u_shape,
    "star": star_shape,
    "dumbbell": dumbbell,
}

CONVEX_SUITE = {
    "tetrahedron": tetrahedron,
    "cube": cube,
}
