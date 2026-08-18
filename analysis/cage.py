"""Cage/mesh geometry: OBJ loading, reflex-edge detection, analytic PMVC event-surface
enumeration (Part 2a), and a generalized-winding-number interior test used both to clip
candidate surfaces to the cage interior and to gate FD stencils (Part 6).

Independent of igl: the OBJ parser and quad-triangulation exactly mirror
``cagedeformations/src/LoadMesh.cpp``'s ``load_cage`` (diagonal split poly[0,1,2] +
poly[0,2,3]), so cages loaded here match what the viewer would load.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parses vertex positions and polygonal faces out of a Wavefront OBJ.

    Faces with more than 3 vertices are fan-triangulated is NOT what the cage loader
    does for quads (it uses the diagonal split, see ``triangulate_polygon``); this
    function returns the raw polygons (as a ragged list turned into a triangulated
    array via ``triangulate_polygon``) to mirror ``load_cage`` exactly.
    """
    verts: list[list[float]] = []
    polys: list[list[int]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v ") or line.startswith("v\t"):
                parts = line.split()[1:4]
                verts.append([float(p) for p in parts])
            elif line.startswith("f ") or line.startswith("f\t"):
                parts = line.split()[1:]
                # OBJ face refs may be "v", "v/vt", "v/vt/vn" or "v//vn"; 1-indexed.
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                polys.append(idx)

    V = np.asarray(verts, dtype=np.float64)
    F = triangulate_polygons(polys)
    return V, F


def triangulate_polygons(polys: list[list[int]]) -> np.ndarray:
    """Diagonal-split triangulation matching ``load_cage(..., triangulate_quads=true)``.

    Triangles pass through unchanged; quads split into (0,1,2) + (0,2,3). Any other
    polygon size is rejected, matching the C++ assert.
    """
    tris: list[list[int]] = []
    for poly in polys:
        if len(poly) == 3:
            tris.append(list(poly))
        elif len(poly) == 4:
            tris.append([poly[0], poly[1], poly[2]])
            tris.append([poly[0], poly[2], poly[3]])
        else:
            raise ValueError(f"unsupported polygon with {len(poly)} vertices")
    return np.asarray(tris, dtype=np.int64)


def write_obj(path: str | Path, V: np.ndarray, F: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for v in V:
            handle.write(f"v {v[0]!r} {v[1]!r} {v[2]!r}\n")
        for f in F:
            handle.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


# ---------------------------------------------------------------------------
# Basic per-triangle geometry
# ---------------------------------------------------------------------------

def face_normals(V: np.ndarray, F: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Per-triangle normals via the right-hand rule on (v1-v0, v2-v0).

    Cages are assumed consistently wound with outward-facing normals (standard OBJ /
    viewer convention); reflex-edge classification below depends on this.
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    if normalize:
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        norm = np.where(norm < 1e-300, 1.0, norm)
        n = n / norm
    return n


def fix_outward_orientation(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Flips any triangle whose normal points toward the mesh centroid, so every
    face ends up outward-facing. Only reliable for star-shaped-from-centroid
    meshes (fine for the synthetic validation shapes built in this project); a
    general (possibly non-convex, non-star-shaped) cage must already be correctly
    wound on load.
    """
    F = F.copy()
    centroid = V.mean(axis=0)
    v0 = V[F[:, 0]]
    n = face_normals(V, F, normalize=False)
    flip = np.einsum("ij,ij->i", n, centroid - v0) > 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def cage_diameter(V: np.ndarray) -> float:
    lo, hi = V.min(axis=0), V.max(axis=0)
    return float(np.linalg.norm(hi - lo))


# ---------------------------------------------------------------------------
# Generalized winding number (Jacobson et al.) — the interior test used both to clip
# event surfaces to the cage interior (Part 2a) and to gate FD stencils (Part 6).
# ---------------------------------------------------------------------------

def winding_number(points: np.ndarray, V: np.ndarray, F: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Generalized winding number of a closed (possibly non-convex) triangle mesh.

    ``points``: (K, 3). Returns (K,) floats, ~1 inside a consistently-oriented closed
    surface, ~0 outside, non-integer near the boundary or if the mesh has holes.
    Uses the robust Van Oosterom-Strackee signed solid-angle-per-triangle formula,
    vectorized over points and triangles with chunking to bound memory.
    """
    points = np.asarray(points, dtype=np.float64)
    K = points.shape[0]
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.empty(K, dtype=np.float64)

    tri_elems = F.shape[0]
    pts_per_chunk = max(1, chunk // max(1, tri_elems))

    for start in range(0, K, pts_per_chunk):
        end = min(K, start + pts_per_chunk)
        p = points[start:end]  # (k,3)

        ra = v0[None, :, :] - p[:, None, :]  # (k,M,3)
        rb = v1[None, :, :] - p[:, None, :]
        rc = v2[None, :, :] - p[:, None, :]

        na = np.linalg.norm(ra, axis=2)
        nb = np.linalg.norm(rb, axis=2)
        nc = np.linalg.norm(rc, axis=2)

        numerator = np.einsum("kmi,kmi->km", ra, np.cross(rb, rc))
        denominator = (
            na * nb * nc
            + np.einsum("kmi,kmi->km", ra, rb) * nc
            + np.einsum("kmi,kmi->km", rb, rc) * na
            + np.einsum("kmi,kmi->km", rc, ra) * nb
        )

        solid_angle = 2.0 * np.arctan2(numerator, denominator)
        out[start:end] = solid_angle.sum(axis=1) / (4.0 * np.pi)

    return out


def inside_cage(points: np.ndarray, V: np.ndarray, F: np.ndarray, tol: float = 0.5) -> np.ndarray:
    """Boolean interior test: winding number closer to 1 than to 0."""
    return winding_number(points, V, F) > tol


# ---------------------------------------------------------------------------
# Reflex-edge detection
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ReflexEdge:
    a: int
    b: int
    face1: int
    face2: int
    apex1: int  # vertex of face1 not on the edge
    apex2: int  # vertex of face2 not on the edge
    dihedral_angle: float  # interior dihedral angle, radians; > pi => reflex


def _edge_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def build_edge_adjacency(F: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """edge(sorted vtx pair) -> list of (face_idx, apex_vertex_idx)."""
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for fi, tri in enumerate(F):
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            apex = tri[(k + 2) % 3]
            key = _edge_key(int(a), int(b))
            adjacency.setdefault(key, []).append((fi, int(apex)))
    return adjacency


def classify_reflex(V: np.ndarray, F: np.ndarray, tol: float = 1e-9) -> list[ReflexEdge]:
    """Reflex (concave) edges: interior dihedral angle > pi.

    Convexity test: for a manifold edge shared by faces with outward normals n1
    (face1) and apex2 (the vertex of face2 not on the edge), the edge is convex iff
    apex2 lies on the interior side of face1's plane (dot(n1, apex2 - a) <= 0), which
    is exactly the defining half-space property of a convex polyhedron. Reflex edges
    are the ones that violate it. Cross-checked against the signed dihedral angle
    computed via the standard atan2 formula.
    """
    normals = face_normals(V, F)
    adjacency = build_edge_adjacency(F)

    result: list[ReflexEdge] = []
    for (a, b), faces in adjacency.items():
        if len(faces) != 2:
            continue

        (f1, apex1), (f2, apex2) = faces
        n1, n2 = normals[f1], normals[f2]
        pa = V[a]

        edge_dir = V[b] - V[a]
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-300:
            continue
        edge_dir = edge_dir / edge_len

        # Authoritative reflex test: a convex polyhedron is the intersection of the
        # outward-normal half-spaces of its faces, so every other vertex (in
        # particular the opposite face's apex) satisfies dot(n, v - a) <= 0 on a
        # convex edge; d > 0 means the edge is reflex. Symmetric under swapping
        # (f1,apex1) <-> (f2,apex2), unlike a signed atan2-based angle (which
        # depends on adjacency dict iteration order picking f1 vs f2 first, and
        # silently gives the wrong sign for whichever order it didn't happen to be
        # derived against - caught by exactly this disagreement check on the cube).
        d1 = float(np.dot(n1, V[apex2] - pa))
        d2 = float(np.dot(n2, V[apex1] - V[b]))
        is_reflex = (d1 + d2) * 0.5 > tol

        # Interior dihedral angle, also order-independent: the unsigned angle
        # between the normals is symmetric in n1/n2 by itself (arccos(dot(n1,n2)));
        # only whether it's pi-minus or pi-plus depends on convexity, which the
        # half-space test above already resolved.
        unsigned_normal_angle = float(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
        angle = np.pi + unsigned_normal_angle if is_reflex else np.pi - unsigned_normal_angle

        result.append(ReflexEdge(
            a=int(a), b=int(b), face1=f1, face2=f2, apex1=apex1, apex2=apex2,
            dihedral_angle=angle,
        ))

    return [e for e in result if e.dihedral_angle > np.pi + tol]


# ---------------------------------------------------------------------------
# Candidate event surfaces (Part 2a)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PlanarSurface:
    """A bounded planar patch: point ``origin`` + orthonormal in-plane basis (u_dir,
    v_dir) + a convex polygon in local (u, v) coordinates. ``normal`` is u_dir x v_dir.
    ``kind`` / ``meta`` describe which analytic rule produced it, for reporting.
    """
    origin: np.ndarray
    u_dir: np.ndarray
    v_dir: np.ndarray
    normal: np.ndarray
    poly2d: np.ndarray  # (P, 2)
    kind: str
    meta: dict

    def to_3d(self, uv: np.ndarray) -> np.ndarray:
        """uv: (..., 2) -> (..., 3)."""
        return self.origin + uv[..., 0:1] * self.u_dir + uv[..., 1:2] * self.v_dir

    def poly3d(self) -> np.ndarray:
        return self.to_3d(self.poly2d)


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / max(np.linalg.norm(normal), 1e-300)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, normal)
    u = u / max(np.linalg.norm(u), 1e-300)
    v = np.cross(normal, u)
    return u, v


def _clip_halfplane(poly: np.ndarray, normal2d: np.ndarray, offset: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a 2D polygon by the half-plane {p : normal2d.p <= offset}."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        cur_in = np.dot(normal2d, cur) <= offset
        nxt_in = np.dot(normal2d, nxt) <= offset
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            d = nxt - cur
            denom = np.dot(normal2d, d)
            t = (offset - np.dot(normal2d, cur)) / denom if abs(denom) > 1e-300 else 0.0
            out.append(cur + t * d)
    return np.asarray(out) if out else np.zeros((0, 2))


def _clip_to_convex_polygon(poly: np.ndarray, clip_poly: np.ndarray) -> np.ndarray:
    """Clips ``poly`` against the convex polygon ``clip_poly`` (CCW), both 2D."""
    result = poly
    n = len(clip_poly)
    for i in range(n):
        a, b = clip_poly[i], clip_poly[(i + 1) % n]
        edge = b - a
        # Inward normal for a CCW polygon: rotate edge by -90deg.
        normal2d = np.array([edge[1], -edge[0]])
        offset = np.dot(normal2d, a)
        result = _clip_halfplane(result, normal2d, offset)
        if len(result) == 0:
            break
    return result


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Simple monotone-chain convex hull, CCW, no scipy dependency needed here."""
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def _aabb_hull_in_plane(V: np.ndarray, origin: np.ndarray, u_dir: np.ndarray, v_dir: np.ndarray) -> np.ndarray:
    lo, hi = V.min(axis=0), V.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rel = corners - origin
    uv = np.stack([rel @ u_dir, rel @ v_dir], axis=1)
    return _convex_hull_2d(uv)


def _make_surface(origin, normal, u_dir, v_dir, u_range, v_range, V, kind, meta) -> PlanarSurface | None:
    rect = np.array([
        [u_range[0], v_range[0]],
        [u_range[1], v_range[0]],
        [u_range[1], v_range[1]],
        [u_range[0], v_range[1]],
    ])
    hull = _aabb_hull_in_plane(V, origin, u_dir, v_dir)
    if len(hull) < 3:
        return None
    clipped = _clip_to_convex_polygon(rect, hull)
    if len(clipped) < 3:
        return None
    return PlanarSurface(origin=origin, u_dir=u_dir, v_dir=v_dir, normal=normal,
                          poly2d=clipped, kind=kind, meta=meta)


def candidate_ev_surfaces(V: np.ndarray, F: np.ndarray, edges: list[ReflexEdge]) -> list[PlanarSurface]:
    """EV candidates: for each reflex edge, the two adjacent face planes extended
    beyond the edge (away from their own triangle) into the cage interior."""
    normals = face_normals(V, F)
    diam = cage_diameter(V)
    margin = diam * 1.5
    surfaces: list[PlanarSurface] = []

    for e in edges:
        a, b = V[e.a], V[e.b]
        edge_dir = b - a
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-300:
            continue
        edge_dir = edge_dir / edge_len

        for face_idx, apex_idx in ((e.face1, e.apex1), (e.face2, e.apex2)):
            n = normals[face_idx]
            perp = np.cross(n, edge_dir)
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-300:
                continue
            perp = perp / perp_norm

            # Which side of line(e) the triangle's own apex sits on, so the surface
            # extends to the OPPOSITE side (beyond the edge, away from the triangle).
            apex_side = np.dot(V[apex_idx] - a, perp)
            sign = -1.0 if apex_side > 0 else 1.0
            v_dir = perp * sign

            surf = _make_surface(
                origin=a, normal=n, u_dir=edge_dir, v_dir=v_dir,
                u_range=(-margin, edge_len + margin), v_range=(0.0, margin),
                V=V, kind="EV",
                meta={"edge": (e.a, e.b), "face": face_idx, "dihedral": e.dihedral_angle},
            )
            if surf is not None:
                surfaces.append(surf)

    return surfaces


def candidate_ve_surfaces(V: np.ndarray, F: np.ndarray, edges: list[ReflexEdge]) -> list[PlanarSurface]:
    """VE candidates: for every cage vertex v and every reflex edge e, the plane
    through v and e."""
    diam = cage_diameter(V)
    margin = diam * 1.5
    surfaces: list[PlanarSurface] = []
    n_verts = V.shape[0]

    for e in edges:
        a, b = V[e.a], V[e.b]
        edge_dir = b - a
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-300:
            continue
        edge_dir = edge_dir / edge_len

        for vi in range(n_verts):
            if vi in (e.a, e.b):
                continue
            v = V[vi]
            to_v = v - a
            # Component of (v - a) perpendicular to the edge direction.
            perp = to_v - np.dot(to_v, edge_dir) * edge_dir
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-6 * max(diam, 1e-9):
                continue  # v (nearly) colinear with e: plane undefined
            perp = perp / perp_norm
            normal = np.cross(edge_dir, perp)
            normal = normal / max(np.linalg.norm(normal), 1e-300)

            surf = _make_surface(
                origin=a, normal=normal, u_dir=edge_dir, v_dir=perp,
                u_range=(-margin, edge_len + margin), v_range=(-margin, margin),
                V=V, kind="VE",
                meta={"edge": (e.a, e.b), "vertex": vi, "dihedral": e.dihedral_angle},
            )
            if surf is not None:
                surfaces.append(surf)

    return surfaces


def all_candidate_surfaces(V: np.ndarray, F: np.ndarray) -> tuple[list[ReflexEdge], list[PlanarSurface]]:
    edges = classify_reflex(V, F)
    surfaces = candidate_ev_surfaces(V, F, edges) + candidate_ve_surfaces(V, F, edges)
    return edges, surfaces


# ---------------------------------------------------------------------------
# Probe-point sampling on a candidate surface
# ---------------------------------------------------------------------------

def sample_probe_points(
    surface: PlanarSurface,
    V: np.ndarray,
    F: np.ndarray,
    n_samples: int,
    max_probe_offset: float,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> np.ndarray:
    """Rejection-samples up to ``n_samples`` points on ``surface`` that lie strictly
    inside the cage, with both x +/- max_probe_offset*normal also inside (so the
    widest FD stencil used downstream stays valid). Returns (<=n_samples, 3)."""
    poly = surface.poly2d
    lo, hi = poly.min(axis=0), poly.max(axis=0)

    found: list[np.ndarray] = []
    tries = 0
    batch = max(64, n_samples * 4)
    while len(found) < n_samples and tries < max_tries:
        tries += 1
        uv = rng.uniform(lo, hi, size=(batch, 2))
        inside_poly = _points_in_convex_polygon(uv, poly)
        uv = uv[inside_poly]
        if len(uv) == 0:
            continue
        pts = surface.to_3d(uv)

        n = surface.normal
        p_plus = pts + max_probe_offset * n
        p_minus = pts - max_probe_offset * n
        ok = inside_cage(pts, V, F) & inside_cage(p_plus, V, F) & inside_cage(p_minus, V, F)
        for p in pts[ok]:
            found.append(p)
            if len(found) >= n_samples:
                break

    return np.asarray(found[:n_samples]) if found else np.zeros((0, 3))


def _points_in_convex_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """points: (K,2), poly: (P,2) CCW or CW convex polygon. Returns (K,) bool.

    A point is inside iff every edge's cross product has the same sign (all
    non-negative for CCW, all non-positive for CW) - tested independently so either
    winding order works.
    """
    n = len(poly)
    all_nonneg = np.ones(len(points), dtype=bool)
    all_nonpos = np.ones(len(points), dtype=bool)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        edge = b - a
        rel = points - a
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        all_nonneg &= cross >= -1e-9
        all_nonpos &= cross <= 1e-9
    return all_nonneg | all_nonpos
