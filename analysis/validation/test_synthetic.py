"""Part 7 validation: the kink detector itself, on functions whose continuity class is
known exactly in closed form. Must pass before any PMVC/cage result is trusted."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import continuity  # noqa: E402

H = np.geomspace(1e-1, 1e-6, 12)


def test_kink_detector_abs_matches_analytic_D2():
    """w = |x|: D2_h must equal 2/h exactly (task's specified self-check)."""
    dd = continuity.divided_differences(lambda t: np.abs(t), 0.0, H)
    assert np.allclose(dd.D2, 2.0 / H, rtol=0, atol=1e-9)
    # D1 is exactly 0 for every h by symmetry of |x| about 0.
    assert np.allclose(dd.D1, 0.0, atol=1e-12)


def test_kink_detector_abs_classification_and_jump():
    dd = continuity.divided_differences(lambda t: np.abs(t), 0.0, H)
    cls = continuity.classify_continuity(dd)
    assert cls.classification == "C0_NOT_C1"

    jd = continuity.jump_diagnostics(lambda t: np.abs(t), 0.0, H)
    assert jd.value_jump < 1e-8
    assert abs(jd.gradient_jump - 2.0) < 1e-6
    assert jd.cross_check_diff < 1e-6


def test_kink_detector_sign_matches_analytic_D1():
    """w = sign(x): D1_h must equal 1/h exactly (task's C0 self-check)."""
    dd = continuity.divided_differences(lambda t: np.sign(t), 0.0, H)
    assert np.allclose(dd.D1, 1.0 / H, rtol=0, atol=1e-6)


def test_kink_detector_sign_classification_and_jump():
    dd = continuity.divided_differences(lambda t: np.sign(t), 0.0, H)
    cls = continuity.classify_continuity(dd)
    assert cls.classification == "C0_BREAK"

    jd = continuity.jump_diagnostics(lambda t: np.sign(t), 0.0, H)
    assert abs(jd.value_jump - 2.0) < 1e-6


def test_kink_detector_smooth_no_false_positive():
    """w = sin(x): every diagnostic must stay bounded (no false kink).

    D1/D2 have O(h^2) centered-difference truncation error, so the tolerance is
    scaled with h^2 rather than fixed; only the smallest (but still well above the
    float64 noise floor) h is checked tightly.
    """
    h = np.geomspace(1e-1, 1e-4, 10)  # stays above the float64 noise floor (~1e-5/1e-6)
    dd = continuity.divided_differences(lambda t: np.sin(t), 0.3, h)
    cls = continuity.classify_continuity(dd)
    assert cls.classification == "C2_OR_BETTER"
    assert np.allclose(dd.D1, np.cos(0.3), atol=0.3 * h ** 2)
    assert np.allclose(dd.D2, -np.sin(0.3), atol=2.0 * h ** 2)
    assert abs(dd.D1[-1] - np.cos(0.3)) < 1e-6
    assert abs(dd.D2[-1] - (-np.sin(0.3))) < 1e-4

    jd = continuity.jump_diagnostics(lambda t: np.sin(t), 0.3, h)
    # Linear-in-h Richardson extrapolation of a smooth function leaves an O(h_max^2)
    # residual bias (it only removes the linear term), not roundoff -- loose
    # tolerances here on purpose; the real test is that nothing looks like a jump.
    assert jd.value_jump < 1e-3
    assert jd.gradient_jump < 1e-3


def test_bisection_locates_known_kink():
    # |x - 0.37| has its kink at t=0.37; a fixed-h centered D1 sweeps smoothly from
    # -1 (left of the kink) to +1 (right of it), giving a clean sign change to
    # bracket -- unlike a one-sided-vs-one-sided gap, which is exactly 0 (not
    # sign-definite) at any t whose +/-h window doesn't straddle the kink at all.
    f = lambda t: np.abs(t - 0.37)  # noqa: E731
    h_probe = 1e-3

    def gap(t):
        return continuity.divided_differences(f, t, np.array([h_probe])).D1[0]

    t_star = continuity.bisect_surface_location(gap, 0.0, 1.0, tol=1e-9)
    assert abs(t_star - 0.37) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
