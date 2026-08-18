"""Part 3 (scoped): paired continuity-class diagnostics for reference PMVC vs the
energy-preserving variant, on every analytic EV candidate surface of the two
non-convex cages in scope (l_shape, dumbbell). Same probe points, same h range,
both schemes -- a paired comparison, not two independent runs.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

import cage
import continuity
import pmvc
from rasterizer import CubemapRaster
from validation.shapes import l_shape, dumbbell

CAGES = {"l_shape": l_shape, "dumbbell": dumbbell}
SCHEMES = {
    "reference_pmvc": pmvc.REFERENCE_PMVC,
    "variant_energy_preserving": pmvc.VARIANT_ENERGY_PRESERVING,
}
FACE_SIZE = 32
H = np.geomspace(2e-2, 2e-3, 8)  # comfortably above the largest measured h* (1.4e-3) across all 4 (cage,scheme) pairs
N_PROBES_PER_SURFACE = 4


@dataclasses.dataclass
class ProbeResult:
    cage_name: str
    surface_kind: str
    surface_idx: int
    surface_meta: dict
    probe_idx: int
    x0: list
    scheme: str
    classification: str
    slope_D1: float
    slope_D2: float
    slope_D3: float
    value_jump: float
    gradient_jump: float
    cross_check_diff: float
    worst_component: int
    delta: float
    h_star: float


def deep_interior_noise_floor(V, F, surfaces, params: pmvc.PMVCParams, raster: CubemapRaster,
                               rng: np.random.Generator, n_candidates: int = 200) -> float:
    """Part 1's method, reused per (cage, scheme): the noise floor is a property of
    the cubemap discretization away from any real discontinuity, measured ONCE and
    then applied as the floor for every probe near a surface -- NOT recomputed at
    each probe location, which would be circular (a cubic fit centered on a real
    kink is dominated by the kink itself, not noise, and reports a spuriously huge
    delta; confirmed this rejected every single probe on the first attempt here).
    """
    best = None
    for _ in range(n_candidates):
        p = rng.uniform(V.min(axis=0), V.max(axis=0))
        if not cage.inside_cage(p[None, :], V, F)[0]:
            continue
        min_margin = min((abs(np.dot(p - s.origin, s.normal)) for s in surfaces), default=np.inf)
        if best is None or min_margin > best[1]:
            best = (p, min_margin)
        if best[1] > 0.05:  # comfortably far from every candidate surface
            break

    x0, _ = best
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)

    def f(t):
        pts = x0[None, :] + t[:, None] * n[None, :]
        return pmvc.evaluate_pmvc(pts, V, F, params, raster)

    # span must stay small enough that a cubic fit actually captures the smooth
    # trend -- checked directly: at span=H.max()=1e-2 the "residual" here is
    # dominated by genuine quartic-and-up curvature (it shrinks ~span^4 as span
    # shrinks, a clean power law, not a plateau), not noise, and gave a wildly
    # inflated delta that rejected every probe. A much narrower span isolates the
    # actual sample-to-sample jitter instead.
    return continuity.estimate_noise_floor(f, 0.0, span=1e-4, n=51)


def run_cage(cage_name: str, builder) -> tuple[list[ProbeResult], np.ndarray, np.ndarray, list]:
    V, F = builder()
    edges = cage.classify_reflex(V, F)
    ev_surfaces = cage.candidate_ev_surfaces(V, F, edges)

    rng = np.random.default_rng(abs(int.from_bytes(cage_name.encode(), "little")) % (2 ** 31))
    results: list[ProbeResult] = []

    rasters = {name: CubemapRaster(FACE_SIZE) for name in SCHEMES}

    noise_floor = {}
    for scheme_name, base_params in SCHEMES.items():
        params = dataclasses.replace(base_params, face_size=FACE_SIZE)
        delta = deep_interior_noise_floor(V, F, ev_surfaces, params, rasters[scheme_name], rng)
        hs = continuity.h_star(delta)
        noise_floor[scheme_name] = (delta, hs)
        print(f"  {cage_name}/{scheme_name}: deep-interior delta={delta:.3e}  h*={hs:.3e}")
        if H.min() < hs:
            print(f"  WARNING: H.min()={H.min():.3e} is below h*={hs:.3e} for {scheme_name}; "
                  f"probes on this scheme will be skipped")

    for si, surf in enumerate(ev_surfaces):
        pts = cage.sample_probe_points(surf, V, F, n_samples=N_PROBES_PER_SURFACE,
                                        max_probe_offset=H.max() * 3, rng=rng)
        for pi, x0 in enumerate(pts):
            n = surf.normal
            for scheme_name, base_params in SCHEMES.items():
                params = dataclasses.replace(base_params, face_size=FACE_SIZE)
                raster = rasters[scheme_name]

                def f(t, x0=x0, n=n, params=params, raster=raster):
                    p = x0[None, :] + t[:, None] * n[None, :]
                    return pmvc.evaluate_pmvc(p, V, F, params, raster)

                delta, hs = noise_floor[scheme_name]
                # Hard gate (Part 1): if the chosen H dips below h*, this probe's
                # result is not trustworthy -- skip it rather than silently keep it.
                if H.min() < hs:
                    continue

                dd = continuity.divided_differences(f, 0.0, H)
                classes = continuity.classify_continuity_vector(dd)
                jd = continuity.jump_diagnostics(f, 0.0, H)

                worst = int(np.nanargmax(np.abs(jd.gradient_jump)))
                c = classes[worst]

                results.append(ProbeResult(
                    cage_name=cage_name, surface_kind=surf.kind, surface_idx=si,
                    surface_meta={k: (list(v) if isinstance(v, tuple) else v) for k, v in surf.meta.items()},
                    probe_idx=pi, x0=x0.tolist(), scheme=scheme_name,
                    classification=c.classification, slope_D1=c.slope_D1, slope_D2=c.slope_D2,
                    slope_D3=c.slope_D3, value_jump=float(jd.value_jump[worst]),
                    gradient_jump=float(jd.gradient_jump[worst]),
                    cross_check_diff=float(jd.cross_check_diff[worst]),
                    worst_component=worst, delta=delta, h_star=hs,
                ))

    return results, V, F, ev_surfaces


def main():
    out_dir = Path(__file__).parent / "part3_out"
    out_dir.mkdir(exist_ok=True)

    all_results: list[ProbeResult] = []
    for cage_name, builder in CAGES.items():
        t0 = time.time()
        results, V, F, surfaces = run_cage(cage_name, builder)
        all_results.extend(results)
        print(f"{cage_name}: {len(surfaces)} EV surfaces, {len(results)} probe results "
              f"in {time.time() - t0:.1f}s")

    with open(out_dir / "probe_results.json", "w") as fh:
        json.dump([dataclasses.asdict(r) for r in all_results], fh, indent=2)

    # Headline summary
    print("\n=== Summary ===")
    for cage_name in CAGES:
        for scheme_name in SCHEMES:
            rows = [r for r in all_results if r.cage_name == cage_name and r.scheme == scheme_name]
            if not rows:
                print(f"{cage_name:10s} {scheme_name:26s}  (no valid probes)")
                continue
            n_c0not1 = sum(1 for r in rows if r.classification == "C0_NOT_C1")
            n_break = sum(1 for r in rows if r.classification == "C0_BREAK")
            jumps = [r.gradient_jump for r in rows]
            print(f"{cage_name:10s} {scheme_name:26s} n={len(rows):3d}  "
                  f"C0_NOT_C1={n_c0not1:3d}  C0_BREAK={n_break:3d}  "
                  f"jump median={np.median(jumps):.3e}  max={np.max(jumps):.3e}")

    return all_results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
