"""Exact port of ``SphereWeightCalculator::TexelCoordSolidAngle``
(viewer/app/Rendering/PMVC/SphereWeightCalculator.h) — the analytic closed-form solid
angle subtended by one cubemap texel, in steradians.

The formula is face-independent (by symmetry of the cube map): it only depends on the
texel's (x, y) location within its face and the face resolution, so one (size, size)
table covers all 6 faces. Total over a full cubemap is 4*pi steradians.
"""
from __future__ import annotations

import numpy as np


def _area_element(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.arctan2(x * y, np.sqrt(x * x + y * y + 1.0))


def texel_solid_angle_table(size: int) -> np.ndarray:
    """Returns a (size, size) float64 array: solid angle of texel (x, y) in steradians.

    Direct port of TexelCoordSolidAngle's four-corner AreaElement combination.
    """
    x = np.arange(size, dtype=np.float64)
    y = np.arange(size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")  # xx[i,j]=i (texelX), yy[i,j]=j (texelY)

    inv_res = 1.0 / size
    U = 2.0 * (xx + 0.5) / size - 1.0
    V = 2.0 * (yy + 0.5) / size - 1.0

    x0 = U - inv_res
    y0 = V - inv_res
    x1 = U + inv_res
    y1 = V + inv_res

    solid_angle = (
        _area_element(x0, y0) - _area_element(x0, y1)
        - _area_element(x1, y0) + _area_element(x1, y1)
    )
    return solid_angle


def face_to_sphere_dirs(size: int) -> np.ndarray:
    """Unit ray direction of every texel center of every face, matching
    ``SphereWeightCalculator::FaceToSphere``. Returns (6, size, size, 3).

    Face convention (world-axis-aligned, fixed per query point — matches the
    real cubemap renderer's per-face view basis):
      0:+X  1:-X  2:+Y  3:-Y  4:+Z  5:-Z
    """
    x = np.arange(size, dtype=np.float64)
    y = np.arange(size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    U = 2.0 * (xx + 0.5) / size - 1.0
    V = 2.0 * (yy + 0.5) / size - 1.0

    dirs = np.empty((6, size, size, 3), dtype=np.float64)
    # face 0: +X -> (1, -v, -u)
    dirs[0, ..., 0] = 1.0
    dirs[0, ..., 1] = -V
    dirs[0, ..., 2] = -U
    # face 1: -X -> (-1, -v, u)
    dirs[1, ..., 0] = -1.0
    dirs[1, ..., 1] = -V
    dirs[1, ..., 2] = U
    # face 2: +Y -> (u, 1, v)
    dirs[2, ..., 0] = U
    dirs[2, ..., 1] = 1.0
    dirs[2, ..., 2] = V
    # face 3: -Y -> (u, -1, -v)
    dirs[3, ..., 0] = U
    dirs[3, ..., 1] = -1.0
    dirs[3, ..., 2] = -V
    # face 4: +Z -> (u, -v, 1)
    dirs[4, ..., 0] = U
    dirs[4, ..., 1] = -V
    dirs[4, ..., 2] = 1.0
    # face 5: -Z -> (-u, -v, -1)
    dirs[5, ..., 0] = -U
    dirs[5, ..., 1] = -V
    dirs[5, ..., 2] = -1.0

    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / norm
