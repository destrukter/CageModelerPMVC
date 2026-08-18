"""NumPy port of the Ring pipeline's accumulation shader (PMVCCompute.comp) driving
the software rasterizer (rasterizer.py) at arbitrary query points. This one module IS
both "reference PMVC" and "the variant" - they differ only in ``PMVCParams``, exactly
mirroring how the real app runs both through one GPU pipeline with different push
constants (RingComputeStrategy / CubemapManager::ComputeCoordinates).

Hit-weight rule ported from ``CubemapRenderInstance::HitWeight``:
  hit_count == 3: hit 0 -> alpha, hit 1 -> beta, hit 2 -> theta.
  otherwise:      even hit index -> 1.0, odd -> 0.0 (matches depth-peeling layers
                   that exist only so the next layer can be peeled against them).

Emission ported from ``PMVCCompute.comp``'s sampleHit/emitHit:
  - non-combined hits are deposited independently on their own triangle
    (lambda[i] += bary[i] * weight; wsum += weight);
  - the "energy preserving" three-hit variant (subtract_second_from_first=True)
    instead deposits max(alpha*w0 - beta*w1, 0) once, on hit 0's triangle only.
  hit 2 (theta), when present, is always independent/additive either way, matching
  CubemapRenderInstance dispatching it on its own (no "second hit" partner).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from rasterizer import CubemapRaster, RayHits


@dataclasses.dataclass(frozen=True)
class PMVCParams:
    """near/far of None means "compute from the cage/query points", matching
    ``CubemapRenderInstance::UpdateProjectionPlanes``: far = the cage's own extent
    (that function uses the true max pairwise cage-vertex distance; we use the
    bounding-box diagonal as a cheap, slightly-conservative proxy), near = the
    closest any query point gets to the cage surface. Fixed defaults of
    near=1e-4/far=1.0 (the shader's push-constant struct defaults, only actually
    reached if UpdateProjectionPlanes bails out early) are systematically wrong for
    cages that aren't ~unit scale with hit distances << 1: (1-depth) = near*(far-t)
    / (t*(far-near)) only approximates the true 1/t MVC weighting when t << far, and
    silently biases the reconstruction otherwise (verified: linear-precision error on
    a unit-ish cube stayed ~1.2% across face_size 16-128 with far=1.0 fixed, i.e. it
    was a systematic weighting bias, not shrinking discretization noise).
    """
    face_size: int = 32
    hit_count: int = 1
    alpha: float = 1.0
    beta: float = 0.0
    theta: float = 0.0
    subtract_second_from_first: bool = False
    solid_angle_only: bool = False  # PMVCO: weight by solid angle alone
    near: float | None = None
    far: float | None = None
    # Multiplies the auto-computed far plane (no effect if `far` is set explicitly).
    # far_scale=1.0 is production-faithful and carries a real, understood ~1-2%
    # linear-precision bias on convex cages even at infinite cubemap resolution
    # (verified: the bias shrinks as far/cage_scale grows and vanishes as far ->
    # infinity, since (1-depth) = near*(far-t)/(t*(far-near)) only -> the exact
    # near/t MVC weighting when far >> t; UpdateProjectionPlanes sizes far to the
    # cage's own extent, which is not >> t). The validation suite's roundoff-level
    # correctness gates use a large far_scale to reach that limit and isolate
    # "is the FD/continuity machinery correct" from "what does the shipped near/far
    # heuristic itself cost in linear precision" (a separate, reportable finding).
    far_scale: float = 1.0


# The two schemes under test in this study, both driven by this one evaluator.
REFERENCE_PMVC = PMVCParams(hit_count=1, alpha=1.0, beta=0.0, theta=0.0)
VARIANT_ENERGY_PRESERVING = PMVCParams(
    hit_count=3, alpha=1.0, beta=1.0, theta=1.0, subtract_second_from_first=True
)
# Secondary variant configs from evaluation/gen_eval_configs.py, kept for context /
# the negativity comparison the energy-preserving fix was written to address.
VARIANT_3HIT_A1_B0_T1 = PMVCParams(hit_count=3, alpha=1.0, beta=0.0, theta=1.0)
VARIANT_3HIT_A1_BM1_T1 = PMVCParams(hit_count=3, alpha=1.0, beta=-1.0, theta=1.0)


def hit_weight(params: PMVCParams, hit_index: int) -> float:
    if params.hit_count == 3:
        return {0: params.alpha, 1: params.beta, 2: params.theta}.get(hit_index, 0.0)
    return 1.0 if hit_index % 2 == 0 else 0.0


def _encode_depth(t: np.ndarray, near: float, far: float) -> np.ndarray:
    """Ports the shader's zEye -> [0,1] perspective depth mapping (inverse of the
    reconstruction used by the interior-distance variant); depth=0 at near, 1 at far.
    """
    denom = t * (far - near)
    denom_safe = np.where(np.abs(denom) < 1e-300, 1.0, denom)
    depth = far * (t - near) / denom_safe
    return np.clip(depth, 0.0, 1.0)


def _sample_hit_weight(
    t: np.ndarray, valid: np.ndarray, solid_angle: np.ndarray, near: float, far: float,
    solid_angle_only: bool,
) -> np.ndarray:
    """Per-ray unweighted hit magnitude (sampleHit, before the pass's alpha/beta/theta
    is applied). Always >= 0, zero where invalid."""
    if solid_angle_only:
        w = solid_angle.copy()
    else:
        depth = _encode_depth(np.where(valid, t, far), near, far)
        no_hit = depth >= 0.999999
        w = np.where(no_hit, 0.0, solid_angle * (1.0 - depth))
    return np.where(valid, np.maximum(w, 0.0), 0.0)


def _point_to_triangle_distance(p: np.ndarray, V0: np.ndarray, V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Vectorized closest-point-on-triangle distance from one point to M triangles:
    min of the 3 edges' point-to-segment distance and, where the perpendicular foot
    on the plane falls inside the triangle, the plane distance. Used only to
    calibrate the near plane (see below), not in the hot per-ray path."""
    best = None
    for a, b in ((V0, V1), (V1, V2), (V2, V0)):
        d = b - a
        t = np.clip(
            np.einsum("mi,mi->m", p[None, :] - a, d) / np.maximum(np.einsum("mi,mi->m", d, d), 1e-300),
            0.0, 1.0,
        )
        closest = a + t[:, None] * d
        dist = np.linalg.norm(p[None, :] - closest, axis=1)
        best = dist if best is None else np.minimum(best, dist)

    n = np.cross(V1 - V0, V2 - V0)
    n_norm = np.maximum(np.linalg.norm(n, axis=1), 1e-300)
    n_unit = n / n_norm[:, None]
    to_p = p[None, :] - V0
    signed = np.einsum("mi,mi->m", to_p, n_unit)
    plane_dist = np.abs(signed)
    proj = p[None, :] - signed[:, None] * n_unit

    v0v1, v0v2, v0p = V1 - V0, V2 - V0, proj - V0
    d00 = np.einsum("mi,mi->m", v0v1, v0v1)
    d01 = np.einsum("mi,mi->m", v0v1, v0v2)
    d11 = np.einsum("mi,mi->m", v0v2, v0v2)
    d20 = np.einsum("mi,mi->m", v0p, v0v1)
    d21 = np.einsum("mi,mi->m", v0p, v0v2)
    denom = d00 * d11 - d01 * d01
    denom_safe = np.where(np.abs(denom) < 1e-300, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom_safe
    w = (d00 * d21 - d01 * d20) / denom_safe
    u = 1.0 - v - w
    inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)

    return np.where(inside, np.minimum(best, plane_dist), best)


def _emit(lam: np.ndarray, wsum: float, bary: np.ndarray, tri_idx: np.ndarray,
          contrib: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    """lambda[F[tri,k]] += bary[:,k] * contrib ; wsum += sum(contrib) -- vectorized
    scatter-add over rays, mirroring emitHit's atomicAdds."""
    nonzero = contrib != 0.0
    if not np.any(nonzero):
        return lam, wsum
    verts = F[tri_idx[nonzero]]  # (n,3)
    weighted = contrib[nonzero, None] * bary[nonzero]  # (n,3)
    np.add.at(lam, verts.reshape(-1), weighted.reshape(-1))
    wsum += float(contrib[nonzero].sum())
    return lam, wsum


def evaluate_pmvc(
    points: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    params: PMVCParams,
    raster: CubemapRaster | None = None,
) -> np.ndarray:
    """Evaluates normalized PMVC/variant coordinates at arbitrary query points.

    Returns (K, n_cage_vertices). Rows sum to 1 wherever wsum != 0 (partition of
    unity), matching RingComputeStrategy::Readback's normalization.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if raster is None or raster.face_size != params.face_size:
        raster = CubemapRaster(params.face_size)

    near, far = params.near, params.far
    if far is None:
        # Matches UpdateProjectionPlanes: true max pairwise cage-vertex distance
        # (not a bounding-box proxy -- verified that far being even a few x too
        # small measurably biases linear precision away from true MVC, since
        # (1-depth) = near*(far-t)/(t*(far-near)) only -> near/t as far >> t).
        diff = V[:, None, :] - V[None, :, :]
        far = max(float(np.sqrt((diff ** 2).sum(axis=2)).max()), 1e-3) * params.far_scale
    if near is None:
        # Matches UpdateProjectionPlanes: min point-to-triangle distance, but over
        # the query points of *this* evaluation (its analogue of "every deformable
        # mesh vertex"). A vertex-distance proxy was tried and rejected: it sets
        # near too large for points close to a triangle's interior but far from
        # its vertices, clipping real hits in other ray directions.
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        base_far = far / params.far_scale  # unscaled, cage-scale reference for the floor
        near_val = min(
            (float(_point_to_triangle_distance(p, v0, v1, v2).min()) for p in points),
            default=base_far * 1e-4,
        )
        # Floored relative to the *unscaled* cage extent, not the (possibly
        # far_scale-inflated) far plane: near only needs to stay << the typical hit
        # distance t (~cage scale) to keep (1-depth) ~= near/t well above the
        # 0.999999 no-hit sentinel; it does not need to track far_scale.
        near = max(near_val, 1e-4 * base_far)
        if near >= far:
            near = max(far * 0.001, 1e-4 * base_far)

    n_cage = V.shape[0]
    hits_per_point = raster.cast(points, V, F, n_hits=params.hit_count, near=near, far=far)
    solid_angle = raster.solid_angle

    out = np.zeros((len(points), n_cage), dtype=np.float64)
    for pi, hits in enumerate(hits_per_point):
        lam = np.zeros(n_cage, dtype=np.float64)
        wsum = 0.0

        combine = params.hit_count == 3 and params.subtract_second_from_first
        if combine:
            w0 = _sample_hit_weight(hits.t[0], hits.valid[0], solid_angle, near, far,
                                     params.solid_angle_only)
            w1 = _sample_hit_weight(hits.t[1], hits.valid[1], solid_angle, near, far,
                                     params.solid_angle_only)
            contrib = np.where(hits.valid[0], np.maximum(params.alpha * w0 - params.beta * w1, 0.0), 0.0)
            lam, wsum = _emit(lam, wsum, hits.bary[0], hits.tri_idx[0], contrib, F)
            start_hi = 2
        else:
            start_hi = 0

        for hi in range(start_hi, params.hit_count):
            hw = hit_weight(params, hi)
            if hw == 0.0:
                continue
            wv = _sample_hit_weight(hits.t[hi], hits.valid[hi], solid_angle, near, far,
                                     params.solid_angle_only)
            contrib = np.where(hits.valid[hi], hw * wv, 0.0)
            lam, wsum = _emit(lam, wsum, hits.bary[hi], hits.tri_idx[hi], contrib, F)

        if wsum != 0.0:
            lam = lam / wsum
        out[pi] = lam

    return out
