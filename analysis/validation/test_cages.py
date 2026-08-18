"""Part 7 validation, cage correctness gates:
  - tetrahedron: PMVC reduces to barycentric coordinates (roundoff-close, in the
    far_scale >> 1 limit -- see pmvc.PMVCParams.far_scale), zero kinks anywhere.
  - cube: both reference PMVC and the variant reduce to the same MVC-equivalent
    weights (roundoff-close), zero reflex edges / zero candidate event surfaces.
  - non-convex suite: nonzero analytic candidate surfaces, and reference PMVC shows
    a nonzero gradient jump across them (the established C0-not-C1 baseline). If
    this gate fails the whole pipeline is untrustworthy for the real comparison.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cage  # noqa: E402
import continuity  # noqa: E402
import pmvc  # noqa: E402
from rasterizer import CubemapRaster  # noqa: E402
from shapes import tetrahedron, cube, NON_CONVEX_SUITE  # noqa: E402

CALIBRATED = dataclasses.replace(pmvc.REFERENCE_PMVC, far_scale=1e4, face_size=48)
CALIBRATED_VARIANT = dataclasses.replace(pmvc.VARIANT_ENERGY_PRESERVING, far_scale=1e4, face_size=48)


def barycentric_tetrahedron(points: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Exact barycentric coordinates of points inside tetrahedron V (4,3)."""
    T = (V[1:] - V[0]).T  # (3,3)
    Tinv = np.linalg.inv(T)
    lam123 = (Tinv @ (points - V[0]).T).T  # (K,3)
    lam0 = 1.0 - lam123.sum(axis=1)
    return np.column_stack([lam0, lam123])


# ---------------------------------------------------------------------------
# Tetrahedron
# ---------------------------------------------------------------------------

def test_tetrahedron_matches_barycentric_coordinates():
    V, F = tetrahedron()
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 12:
        p = rng.uniform(V.min(axis=0), V.max(axis=0))
        if cage.inside_cage(p[None, :], V, F)[0]:
            pts.append(p)
    pts = np.array(pts)

    w = pmvc.evaluate_pmvc(pts, V, F, CALIBRATED)
    exact = barycentric_tetrahedron(pts, V)
    err = np.abs(w - exact).max()
    assert err < 1e-4, f"tetrahedron PMVC vs exact barycentric max error {err:.3e}"


def test_tetrahedron_linear_precision():
    V, F = tetrahedron()
    pts = np.array([[0.0, 0.0, 0.0], [0.02, -0.01, 0.03], [-0.03, 0.02, 0.01]])
    w = pmvc.evaluate_pmvc(pts, V, F, CALIBRATED)
    recon = w @ V
    err = np.abs(recon - pts).max()
    assert err < 1e-4, f"tetrahedron linear precision error {err:.3e}"


def test_tetrahedron_zero_kinks():
    """grad w_i is a known constant on a tetrahedron (barycentric coords are exactly
    linear): probing along random interior lines must show ~zero value/gradient
    jump, once h is kept above the discretization noise floor (Part 1, hard-asserted
    below rather than just documented).

    Uses jump_diagnostics' Richardson-extrapolated jump magnitude rather than the
    log-log slope classifier: on this cage the true noise floor is ~1e-10 (float64
    rounding on an exactly-linear function), so D2/D3 are indistinguishable from
    pure noise over the whole usable h range, and a least-squares slope fit through
    8 points of pure noise easily reports a spurious "growing" slope by chance --
    confirmed by checking those probes' extrapolated gradient_jump directly, which
    lands at ~1e-7, i.e. genuinely zero at the precision this pipeline can resolve.
    """
    V, F = tetrahedron()
    raster = CubemapRaster(CALIBRATED.face_size)
    rng = np.random.default_rng(1)

    max_grad_jump = 0.0
    for _ in range(5):
        x0 = rng.uniform(V.min(axis=0) * 0.5, V.max(axis=0) * 0.5)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        if not cage.inside_cage(x0[None, :], V, F)[0]:
            continue

        def f(t, x0=x0, n=n):
            pts = x0[None, :] + t[:, None] * n[None, :]
            return pmvc.evaluate_pmvc(pts, V, F, CALIBRATED, raster)

        delta = continuity.estimate_noise_floor(f, 0.0, span=3e-2)
        h_floor = continuity.h_star(delta) * 2.0  # safety margin above h*
        h = np.geomspace(3e-2, max(h_floor, 1e-8), 8)
        continuity.assert_above_noise_floor(h, delta, label="tetrahedron zero-kink probe")

        jd = continuity.jump_diagnostics(f, 0.0, h)
        max_grad_jump = max(max_grad_jump, float(np.nanmax(np.abs(jd.gradient_jump))))

    assert max_grad_jump < 1e-4, f"tetrahedron should have zero kinks, saw jump {max_grad_jump:.3e}"


# ---------------------------------------------------------------------------
# Cube
# ---------------------------------------------------------------------------

def test_cube_zero_reflex_edges_and_candidates():
    V, F = cube()
    edges = cage.classify_reflex(V, F)
    assert len(edges) == 0
    _, surfaces = cage.all_candidate_surfaces(V, F)
    assert len(surfaces) == 0


def test_cube_both_schemes_agree_and_reduce_to_mvc_like_precision():
    V, F = cube()
    pts = np.array([[0.0, 0.0, 0.0], [0.05, -0.03, 0.07], [0.1, 0.1, -0.12]])

    w_ref = pmvc.evaluate_pmvc(pts, V, F, CALIBRATED)
    w_var = pmvc.evaluate_pmvc(pts, V, F, CALIBRATED_VARIANT)

    # On a fully convex cage no ray ever gets a second/third depth-peeled hit, so
    # the three-hit variant must degenerate to exactly the same result as the
    # single-hit reference (this is a correctness gate on the variant's handling of
    # "no further hit", not just a numerical closeness check).
    diff = np.abs(w_ref - w_var).max()
    assert diff < 1e-9, f"variant should exactly match reference PMVC on a convex cage, diff={diff:.3e}"

    recon = w_ref @ V
    err = np.abs(recon - pts).max()
    assert err < 1e-3, f"cube linear precision error {err:.3e}"
    assert w_ref.min() > -1e-9, "no negative weights expected on a convex cage"


# ---------------------------------------------------------------------------
# Non-convex suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(NON_CONVEX_SUITE.keys()))
def test_non_convex_suite_has_analytic_candidates(name):
    V, F = NON_CONVEX_SUITE[name]()
    edges = cage.classify_reflex(V, F)
    assert len(edges) > 0, f"{name}: expected reflex edges"
    _, surfaces = cage.all_candidate_surfaces(V, F)
    assert len(surfaces) > 0, f"{name}: expected candidate event surfaces"


@pytest.mark.parametrize("name", list(NON_CONVEX_SUITE.keys()))
def test_reference_pmvc_shows_nonzero_gradient_jump(name):
    """The critical gate: if reference PMVC shows *no* gradient jump on an EV
    surface of a non-convex cage, the pipeline (rasterizer/pmvc/continuity) is
    broken and nothing downstream can be trusted -- do not relax this test."""
    V, F = NON_CONVEX_SUITE[name]()
    edges = cage.classify_reflex(V, F)
    surfaces = cage.candidate_ev_surfaces(V, F, edges)
    assert surfaces, f"{name}: no EV surfaces to probe"

    params = dataclasses.replace(pmvc.REFERENCE_PMVC, face_size=48)
    raster = CubemapRaster(params.face_size)
    # A fixed per-shape seed: Python's hash() of a str is randomized per process
    # (PYTHONHASHSEED), so it silently resampled different probe points on every
    # run -- not reproducible, and the reason this test flaked on l_shape earlier.
    seed = abs(int.from_bytes(name.encode(), "little")) % (2 ** 31)
    rng = np.random.default_rng(seed)
    # Floor informed by the noise-floor measurement in test_tetrahedron_zero_kinks
    # at the same face_size (D3 there was still noise-only at h=6.7e-4 and clearly
    # blown up by h=5e-6); stay well clear of that.
    h = np.geomspace(1e-2, 1e-3, 7)

    best_jump = 0.0
    found_c0_not_c1 = False
    for surf in surfaces:
        pts = cage.sample_probe_points(surf, V, F, n_samples=5, max_probe_offset=h.max() * 3, rng=rng)
        for x0 in pts:
            n = surf.normal

            def f(t, x0=x0, n=n):
                p = x0[None, :] + t[:, None] * n[None, :]
                return pmvc.evaluate_pmvc(p, V, F, params, raster)

            dd = continuity.divided_differences(f, 0.0, h)
            classes = continuity.classify_continuity_vector(dd)
            jd = continuity.jump_diagnostics(f, 0.0, h)
            jump = float(np.nanmax(jd.gradient_jump))
            best_jump = max(best_jump, jump)
            if any(c.classification == "C0_NOT_C1" for c in classes):
                found_c0_not_c1 = True

    assert found_c0_not_c1, (
        f"{name}: reference PMVC showed no C0-not-C1 kink on any sampled EV surface "
        f"point (best gradient jump seen: {best_jump:.3e}) -- pipeline is broken"
    )
    assert best_jump > 1e-3, f"{name}: gradient jump too small to trust ({best_jump:.3e})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
