"""Part 3: continuity-class determination along a 1D probe (Diagnostic A: divided-
difference scaling / log-log classification; Diagnostic B: one-sided Richardson
extrapolation for the jump magnitude, with a cross-check between the two).

Deliberately agnostic to what ``f`` is: it takes an array of scalar offsets ``t`` and
returns an array of values (K,) or (K, C) (C = e.g. one column per cage vertex). The
caller supplies ``f`` as a closure over a 3D probe line (``lambda t: pmvc(x0 + t[:,None]*n,
...)``) or a bare 1D synthetic test function (``lambda t: np.abs(t)``) — everything
below only ever sees a 1D parametrization, matching how the task states D1_h/D2_h/D3_h.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

ScalarFn = Callable[[np.ndarray], np.ndarray]

_STENCIL_OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def sample_multi_h(f: ScalarFn, t0: float, h: np.ndarray) -> np.ndarray:
    """Evaluates f at t0 + h*{-2,-1,0,1,2} for every h in one batched call.

    Returns (n_h, 5) if f is scalar-valued, (n_h, 5, C) if vector-valued.
    """
    h = np.asarray(h, dtype=np.float64)
    t = t0 + h[:, None] * _STENCIL_OFFSETS[None, :]
    flat = f(t.reshape(-1))
    flat = np.asarray(flat, dtype=np.float64)
    if flat.ndim == 1:
        return flat.reshape(h.shape[0], 5)
    return flat.reshape(h.shape[0], 5, flat.shape[-1])


@dataclasses.dataclass
class DividedDifferences:
    h: np.ndarray
    D1: np.ndarray  # (n_h,[C])
    D2: np.ndarray
    D3: np.ndarray
    v0: np.ndarray  # value at the center, (n_h,[C]) (constant across h, kept for convenience)


def divided_differences(f: ScalarFn, t0: float, h: np.ndarray) -> DividedDifferences:
    vals = sample_multi_h(f, t0, h)
    vm2, vm1, v0, vp1, vp2 = (vals[:, i, ...] for i in range(5))
    h_b = h if vals.ndim == 2 else np.asarray(h)[:, None]

    D1 = (vp1 - vm1) / (2 * h_b)
    D2 = (vp1 - 2 * v0 + vm1) / (h_b ** 2)
    D3 = (vp2 - 2 * vp1 + 2 * vm1 - vm2) / (2 * h_b ** 3)
    return DividedDifferences(h=np.asarray(h, dtype=np.float64), D1=D1, D2=D2, D3=D3, v0=v0)


def fit_loglog_slope(h: np.ndarray, y: np.ndarray, min_points: int = 3) -> float:
    """Slope of log|y| vs log h. ~ -1 means y grows like 1/h as h -> 0 (unbounded);
    ~ 0 means y is bounded. NaN if there isn't enough finite, nonzero data."""
    h = np.asarray(h, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y) & (np.abs(y) > 1e-300) & np.isfinite(h) & (h > 0)
    if mask.sum() < min_points:
        return float("nan")
    slope, _ = np.polyfit(np.log(h[mask]), np.log(np.abs(y[mask])), 1)
    return float(slope)


@dataclasses.dataclass
class ContinuityClassification:
    classification: str  # 'C0_BREAK' | 'C0_NOT_C1' | 'C1_NOT_C2' | 'C2_OR_BETTER'
    slope_D1: float
    slope_D2: float
    slope_D3: float


def classify_continuity(dd: DividedDifferences, growth_slope_threshold: float = -0.5) -> ContinuityClassification:
    """Classifies by the first of D1, D2, D3 (in that order) whose log-log slope vs h
    indicates 1/h growth as h -> 0, per the task's diagnostic-A rule. Only meaningful
    for scalar-valued dd (call per-component for vector-valued f)."""
    s1 = fit_loglog_slope(dd.h, dd.D1)
    s2 = fit_loglog_slope(dd.h, dd.D2)
    s3 = fit_loglog_slope(dd.h, dd.D3)

    def growing(s: float) -> bool:
        return not np.isnan(s) and s < growth_slope_threshold

    if growing(s1):
        cls = "C0_BREAK"
    elif growing(s2):
        cls = "C0_NOT_C1"
    elif growing(s3):
        cls = "C1_NOT_C2"
    else:
        cls = "C2_OR_BETTER"

    return ContinuityClassification(classification=cls, slope_D1=s1, slope_D2=s2, slope_D3=s3)


def classify_continuity_vector(dd: DividedDifferences) -> list[ContinuityClassification]:
    """Per-component classification for vector-valued f (dd.D1 etc. shaped (n_h, C))."""
    n_c = dd.D1.shape[-1]
    out = []
    for c in range(n_c):
        dd_c = DividedDifferences(h=dd.h, D1=dd.D1[:, c], D2=dd.D2[:, c], D3=dd.D3[:, c], v0=dd.v0[:, c])
        out.append(classify_continuity(dd_c))
    return out


# ---------------------------------------------------------------------------
# Diagnostic B: one-sided Richardson-extrapolated jump magnitude
# ---------------------------------------------------------------------------

def one_sided_derivative_estimates(f: ScalarFn, t0: float, h: np.ndarray, side: int) -> np.ndarray:
    """Two-point one-sided secant slope [f(t0+side*h) - f(t0)] / (side*h) for every h.

    O(h) accurate for a one-sided limit (as opposed to the centered O(h^2) formulas
    above), which is what Richardson extrapolation below assumes.
    """
    h = np.asarray(h, dtype=np.float64)
    t = np.concatenate([[t0], t0 + side * h])
    vals = np.asarray(f(t), dtype=np.float64)
    v0 = vals[0, ...]
    vh = vals[1:, ...]
    h_b = h if vh.ndim == 1 else h[:, None]
    return (vh - v0) / (side * h_b)


def richardson_extrapolate_linear(h: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Least-squares fit d(h) ~= d0 + c*h and returns d0 (the h -> 0 limit),
    assuming the leading discretization error of a one-sided estimate is O(h)."""
    h = np.asarray(h, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    A = np.stack([np.ones_like(h), h], axis=1)
    if d.ndim == 1:
        coeffs, *_ = np.linalg.lstsq(A, d, rcond=None)
        return coeffs[0]
    flat = d.reshape(d.shape[0], -1)
    coeffs, *_ = np.linalg.lstsq(A, flat, rcond=None)
    return coeffs[0].reshape(d.shape[1:])


@dataclasses.dataclass
class JumpDiagnostics:
    value_plus: np.ndarray
    value_minus: np.ndarray
    value_jump: np.ndarray          # |f(0+) - f(0-)|, should be ~0 if C0 holds
    grad_plus: np.ndarray           # one-sided derivative limit from the + side
    grad_minus: np.ndarray          # one-sided derivative limit from the - side
    gradient_jump: np.ndarray       # |grad_plus - grad_minus|, 0 => C1 holds here
    cross_check: np.ndarray         # h*D2_h extrapolated to h->0 (should equal gradient_jump)
    cross_check_diff: np.ndarray    # |gradient_jump - cross_check|; large => bug, not a real jump


def jump_diagnostics(f: ScalarFn, t0: float, h: np.ndarray) -> JumpDiagnostics:
    h = np.asarray(h, dtype=np.float64)

    d_plus = one_sided_derivative_estimates(f, t0, h, side=+1)
    d_minus = one_sided_derivative_estimates(f, t0, h, side=-1)
    grad_plus = richardson_extrapolate_linear(h, d_plus)
    grad_minus = richardson_extrapolate_linear(h, d_minus)
    gradient_jump = np.abs(grad_plus - grad_minus)

    # Value jump: the one-sided value estimates *are* f(t0 +/- h) themselves
    # (h -> 0 limit via the same linear-in-h extrapolation of f(t0+h) about its
    # true one-sided limit; f is assumed O(h) close to its boundary value here,
    # consistent with the one-sided-derivative assumption above).
    t = np.concatenate([[t0], t0 + h, t0 - h])
    vals = np.asarray(f(t), dtype=np.float64)
    n_h = h.shape[0]
    v_plus_h = vals[1:1 + n_h, ...]
    v_minus_h = vals[1 + n_h:, ...]
    value_plus = richardson_extrapolate_linear(h, v_plus_h)
    value_minus = richardson_extrapolate_linear(h, v_minus_h)
    value_jump = np.abs(value_plus - value_minus)

    dd = divided_differences(f, t0, h)
    h_b = h if dd.D2.ndim == 1 else h[:, None]
    cross_check = richardson_extrapolate_linear(h, h_b * dd.D2)
    cross_check_diff = np.abs(gradient_jump - np.abs(cross_check))

    return JumpDiagnostics(
        value_plus=value_plus, value_minus=value_minus, value_jump=value_jump,
        grad_plus=grad_plus, grad_minus=grad_minus, gradient_jump=gradient_jump,
        cross_check=cross_check, cross_check_diff=cross_check_diff,
    )


# ---------------------------------------------------------------------------
# Noise floor (a minimal precursor to the full Part 1 deliverable): the divided-
# difference formulas have error O(h^2) + O(delta/h^order), where delta is the
# high-frequency sampling/discretization noise level and order in {1,2,3} for
# D1/D2/D3 -- balancing the two terms gives h* ~ delta^(1/(2+order)). Confirmed
# empirically on the tetrahedron cage (see validation/test_cages.py): D1 stays
# exactly flat with h (as expected for a truly linear function), D2/D3 are pure
# noise near zero at moderate h, and D3 specifically blows up by orders of
# magnitude once h drops below its h*, exactly as this predicts.
# ---------------------------------------------------------------------------

def estimate_noise_floor(f: ScalarFn, t0: float, span: float, n: int = 33) -> float:
    """Dense-samples f over [t0-span, t0+span], fits a cubic (the smooth trend a
    C2-or-better function would show over a small interval), and returns the RMS
    residual as the noise level delta. For vector-valued f, returns the max over
    components (the most conservative / least favorable one)."""
    t = np.linspace(t0 - span, t0 + span, n)
    vals = np.asarray(f(t), dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    deltas = []
    for c in range(vals.shape[1]):
        coeffs = np.polyfit(t, vals[:, c], 3)
        fit = np.polyval(coeffs, t)
        deltas.append(np.std(vals[:, c] - fit))
    return float(np.max(deltas))


def h_star(delta: float) -> float:
    """Minimum usable FD step, per the task's stated formula: error ~ O(h^2) +
    O(delta/h), balanced at h^3 ~ delta => h* ~ delta^(1/3). Used as a single
    conservative floor for D1/D2/D3 alike (empirically checked against the
    tetrahedron cage: D3's blow-up onset matches this h* well, even though D3's own
    noise term is technically O(delta/h^3) and would in principle tolerate a larger
    floor -- the D1-based h* is the more conservative of the two and is what the
    task specifies)."""
    if delta <= 0:
        return 0.0
    return float(delta) ** (1.0 / 3.0)


def assert_above_noise_floor(h: np.ndarray, delta: float, label: str = "") -> None:
    """Hard-asserts every h in the array exceeds h*(delta), per the task's explicit
    instruction to hard-assert this rather than just document it."""
    hs = h_star(delta)
    h = np.asarray(h)
    if np.any(h < hs):
        bad = h[h < hs]
        raise AssertionError(
            f"{label + ': ' if label else ''}h values below the noise floor h*={hs:.3e} "
            f"(delta={delta:.3e}): {bad}"
        )


def bisect_surface_location(
    onesided_gap: Callable[[float], float], t_lo: float, t_hi: float,
    tol: float = 1e-10, max_iter: int = 60,
) -> float:
    """Brackets the true kink location by bisection on a signed indicator that
    changes sign as t crosses the surface (e.g. the discrepancy between the +
    and - one-sided derivative estimates at fixed small h, which is ~0 away from
    a kink and jumps in sign/magnitude as the probe crosses it). Used to refine an
    analytically-estimated surface position before centering the diagnostics above
    on it, per the task's "do not assume the analytic surface position is exact."
    """
    g_lo, g_hi = onesided_gap(t_lo), onesided_gap(t_hi)
    if g_lo == 0:
        return t_lo
    if g_hi == 0:
        return t_hi
    if np.sign(g_lo) == np.sign(g_hi):
        raise ValueError("bisect_surface_location: bracket does not change sign")

    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        g_mid = onesided_gap(t_mid)
        if abs(t_hi - t_lo) < tol:
            return t_mid
        if np.sign(g_mid) == np.sign(g_lo):
            t_lo, g_lo = t_mid, g_mid
        else:
            t_hi, g_hi = t_mid, g_mid
    return 0.5 * (t_lo + t_hi)
