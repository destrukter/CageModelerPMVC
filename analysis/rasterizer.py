"""Vectorized software cubemap rasterizer, evaluable at arbitrary query points.

Cube-map rasterization from a fixed camera position is, per texel, exactly "which
triangle does the ray through this texel's center first hit" (rasterization with a
z-buffer and analytic per-texel ray casting agree exactly for a pinhole/perspective
camera — a rasterizer just finds the same nearest-hit-along-the-texel-ray answer via
scan conversion instead of an explicit intersection test). Depth peeling for the
multi-hit PMVC variants is the same statement for the 2nd, 3rd, ... nearest hit. This
module implements it as batched ray-triangle intersection (Moeller-Trumbore) against
the fixed, world-axis-aligned texel ray directions from ``solid_angle.py``, which
means it can be evaluated at any query point (not just mesh vertices, unlike the
shipped GPU pipeline) while remaining numerically equivalent to what the real
CubemapRenderInstance / PMVCCompute.comp pipeline computes.

One query point is processed per iteration (rays x triangles vectorized fully inside);
that is enough for the probe-based Part 1-4/7 workloads (hundreds-thousands of query
points). A batched-over-query-points version would be needed for full Part 5 voxel-grid
production runs and is not implemented here.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from solid_angle import face_to_sphere_dirs, texel_solid_angle_table


@dataclasses.dataclass
class RayHits:
    """Per query point, up to ``n_hits`` depth-peeled layers per ray.

    All arrays shaped (n_hits, n_rays) where n_rays = 6 * face_size * face_size
    (raveled face-major, then row-major within a face, matching
    ``face_to_sphere_dirs``'s (6, size, size) layout).
    """
    valid: np.ndarray       # bool
    tri_idx: np.ndarray     # int64, meaningless where not valid
    bary: np.ndarray        # float64 (n_hits, n_rays, 3): (w0, w1, w2) for F[tri_idx]
    t: np.ndarray           # float64 Euclidean hit distance along the ray


class CubemapRaster:
    """Precomputes the fixed texel ray directions + solid-angle table once, then
    casts against a given cage's triangles for any number of query points."""

    def __init__(self, face_size: int = 32):
        self.face_size = face_size
        dirs = face_to_sphere_dirs(face_size)  # (6,size,size,3)
        self.dirs = dirs.reshape(-1, 3)  # (n_rays, 3)
        # Solid angle is face-independent; tile the (size,size) table across the 6
        # faces so it matches the (6,size,size) -> (n_rays,) raveling of self.dirs.
        sa = texel_solid_angle_table(face_size)  # (size,size)
        self.solid_angle = np.broadcast_to(sa[None, :, :], (6, face_size, face_size)).reshape(-1)
        self.n_rays = self.dirs.shape[0]

    def cast(
        self,
        points: np.ndarray,
        V: np.ndarray,
        F: np.ndarray,
        n_hits: int,
        near: float,
        far: float,
        eps: float = 1e-9,
    ) -> list[RayHits]:
        """points: (K,3). Returns a list of K RayHits."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        edge1 = v1 - v0  # (M,3)
        edge2 = v2 - v0  # (M,3)
        M = F.shape[0]
        D = self.dirs  # (R,3)
        R = self.n_rays

        # h, a, f depend only on the (fixed) ray directions and the triangles, so
        # they're computed once per cage, reused for every query point.
        h = np.cross(D[:, None, :], edge2[None, :, :])  # (R,M,3)
        a = np.einsum("rmi,mi->rm", h, edge1)  # (R,M)
        degenerate = np.abs(a) < eps
        a_safe = np.where(degenerate, 1.0, a)
        f = 1.0 / a_safe

        results: list[RayHits] = []
        for o in points:
            s = o - v0  # (M,3)
            q = np.cross(s, edge1)  # (M,3)

            u = f * np.einsum("rmi,mi->rm", h, s)  # (R,M)
            v = f * np.einsum("ri,mi->rm", D, q)  # (R,M)

            results.append(self._finish_point(u, v, f, q, edge2, degenerate, near, far, n_hits, eps))

        return results

    def _finish_point(self, u, v, f, q, edge2, degenerate, near, far, n_hits, eps) -> RayHits:
        # t = f * dot(q, edge2), but q is (M,3) (ray-independent) and f is (R,M):
        t_per_tri = np.einsum("mi,mi->m", q, edge2)  # (M,) per-triangle dot, ray-independent
        t = f * t_per_tri[None, :]  # (R,M)

        valid = (
            (~degenerate)
            & (u >= -eps) & (u <= 1.0 + eps)
            & (v >= -eps) & (v <= 1.0 + eps)
            & ((u + v) <= 1.0 + eps)
            & (t > near) & (t < far)
        )

        t_masked = np.where(valid, t, np.inf)  # (R,M)
        R, M = t_masked.shape
        if M == 0:
            n_rays = R
            return RayHits(
                valid=np.zeros((n_hits, n_rays), dtype=bool),
                tri_idx=np.zeros((n_hits, n_rays), dtype=np.int64),
                bary=np.zeros((n_hits, n_rays, 3), dtype=np.float64),
                t=np.full((n_hits, n_rays), np.inf),
            )

        # A pool larger than n_hits, so that de-duplicating exact-t coincidences
        # below (see dup_tol) still leaves n_hits genuine hits when possible: a ray
        # exactly in the plane of a quad's diagonal split intersects both of that
        # quad's coplanar triangles at *identical* t (verified: t1-t0 == 0.0 to the
        # ULP for such rays on the validation cube/tetrahedron), which is the same
        # physical crossing counted twice, not a second depth-peeled layer.
        pool = min(M, n_hits + 8)
        order = np.argpartition(t_masked, kth=pool - 1, axis=1)[:, :pool]
        row_idx = np.arange(R)[:, None]
        t_pool = t_masked[row_idx, order]
        sort_within = np.argsort(t_pool, axis=1)
        order = np.take_along_axis(order, sort_within, axis=1)
        t_pool = np.take_along_axis(t_pool, sort_within, axis=1)
        u_pool = np.take_along_axis(u, order, axis=1)
        v_pool = np.take_along_axis(v, order, axis=1)
        valid_pool = np.isfinite(t_pool)

        # Sequential de-dup: a candidate is a duplicate of the *previously accepted*
        # one if their t's coincide to a tight relative tolerance. Genuine distinct
        # surface layers (real depth peeling) are never this close in practice.
        dup_tol = 1e-9
        accepted_t = np.zeros(R)
        has_accepted = np.zeros(R, dtype=bool)
        keep = np.zeros((R, pool), dtype=bool)
        for j in range(pool):
            tj = t_pool[:, j]
            is_dup = has_accepted & (np.abs(tj - accepted_t) <= dup_tol * np.maximum(1.0, np.abs(accepted_t)))
            kj = valid_pool[:, j] & ~is_dup
            keep[:, j] = kj
            accepted_t = np.where(kj, tj, accepted_t)
            has_accepted = has_accepted | kj

        rank = np.cumsum(keep, axis=1) - 1  # per-row rank among kept candidates

        valid_k = np.zeros((R, n_hits), dtype=bool)
        t_k = np.full((R, n_hits), np.inf)
        tri_k = np.zeros((R, n_hits), dtype=np.int64)
        u_k = np.zeros((R, n_hits))
        v_k = np.zeros((R, n_hits))
        for slot in range(n_hits):
            slot_mask = keep & (rank == slot)  # (R,pool); at most one True per row
            has_slot = slot_mask.any(axis=1)
            col = np.argmax(slot_mask, axis=1)  # arbitrary (0) where has_slot is False
            valid_k[:, slot] = has_slot
            t_k[has_slot, slot] = t_pool[np.arange(R)[has_slot], col[has_slot]]
            tri_k[has_slot, slot] = order[np.arange(R)[has_slot], col[has_slot]]
            u_k[has_slot, slot] = u_pool[np.arange(R)[has_slot], col[has_slot]]
            v_k[has_slot, slot] = v_pool[np.arange(R)[has_slot], col[has_slot]]

        w0 = 1.0 - u_k - v_k
        bary = np.stack([w0, u_k, v_k], axis=-1)  # (R,n_hits,3)

        return RayHits(
            valid=valid_k.T,               # (n_hits, R)
            tri_idx=tri_k.T,               # (n_hits, R)
            bary=np.transpose(bary, (1, 0, 2)),  # (n_hits, R, 3)
            t=t_k.T,                       # (n_hits, R)
        )
