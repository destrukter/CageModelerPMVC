"""Part 4 (discrete, since the variant has no continuous eps): sweep the three-hit
configs already identified in evaluation/gen_eval_configs.py --
a1_b0_t1 (drop the 2nd hit, add a 3rd), a1_bm1_t1 (naive negative-beta subtraction,
the one the energy-preserving fix replaced), a1_b1_t1_energy (the variant) -- against
the hit_count=1 reference. Reuses part3_continuity.py's saved probe points for the
jump-magnitude axis (paired at the same locations/h) and a separate small set of
interior points for the linear-precision axis.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

import cage
import continuity
import pmvc
from rasterizer import CubemapRaster
from validation.shapes import l_shape, dumbbell

CAGES = {"l_shape": l_shape, "dumbbell": dumbbell}
CONFIGS = {
    "reference (hit_count=1)": pmvc.REFERENCE_PMVC,
    "3hit a1_b0_t1": pmvc.VARIANT_3HIT_A1_B0_T1,
    "3hit a1_bm1_t1 (naive)": pmvc.VARIANT_3HIT_A1_BM1_T1,
    "3hit a1_b1_t1 (energy preserving)": pmvc.VARIANT_ENERGY_PRESERVING,
}
FACE_SIZE = 32
H = np.geomspace(2e-2, 2e-3, 8)


def jump_magnitude_by_config(probe_records: list[dict]) -> dict[str, list[float]]:
    """Re-evaluates gradient_jump at the exact probe points part3 already sampled,
    for the two configs not already covered there."""
    out: dict[str, list[float]] = {name: [] for name in CONFIGS}

    by_cage: dict[str, tuple] = {name: builder() for name, builder in CAGES.items()}
    surfaces_by_cage = {}
    for cage_name, (V, F) in by_cage.items():
        edges = cage.classify_reflex(V, F)
        surfaces_by_cage[cage_name] = cage.candidate_ev_surfaces(V, F, edges)

    # Unique (cage, surface, probe) points, taking x0 from the reference rows.
    unique = {}
    for r in probe_records:
        if r["scheme"] != "reference_pmvc":
            continue
        key = (r["cage_name"], r["surface_idx"], r["probe_idx"])
        unique[key] = r

    # Reference and energy-preserving variant results are already computed in part3.
    for r in probe_records:
        if r["scheme"] == "reference_pmvc":
            out["reference (hit_count=1)"].append(r["gradient_jump"])
        elif r["scheme"] == "variant_energy_preserving":
            out["3hit a1_b1_t1 (energy preserving)"].append(r["gradient_jump"])

    for config_name in ("3hit a1_b0_t1", "3hit a1_bm1_t1 (naive)"):
        params = dataclasses.replace(CONFIGS[config_name], face_size=FACE_SIZE)
        raster = CubemapRaster(FACE_SIZE)
        for (cage_name, surface_idx, probe_idx), r in unique.items():
            V, F = by_cage[cage_name]
            surf = surfaces_by_cage[cage_name][surface_idx]
            x0 = np.array(r["x0"])
            n = surf.normal
            comp = r["worst_component"]

            def f(t, x0=x0, n=n, params=params, raster=raster):
                p = x0[None, :] + t[:, None] * n[None, :]
                return pmvc.evaluate_pmvc(p, V, F, params, raster)

            jd = continuity.jump_diagnostics(f, 0.0, H)
            out[config_name].append(float(jd.gradient_jump[comp]))

    return out


def linear_precision_by_config(n_points_per_cage: int = 8) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {name: [] for name in CONFIGS}
    for cage_name, builder in CAGES.items():
        V, F = builder()
        rng = np.random.default_rng(abs(int.from_bytes((cage_name + "_lp").encode(), "little")) % (2 ** 31))
        pts = []
        while len(pts) < n_points_per_cage:
            p = rng.uniform(V.min(axis=0), V.max(axis=0))
            if cage.inside_cage(p[None, :], V, F)[0]:
                pts.append(p)
        pts = np.array(pts)

        for config_name, base_params in CONFIGS.items():
            params = dataclasses.replace(base_params, face_size=FACE_SIZE)
            w = pmvc.evaluate_pmvc(pts, V, F, params)
            recon = w @ V
            err = np.linalg.norm(recon - pts, axis=1)
            out[config_name].extend(err.tolist())
    return out


def main():
    out_dir = Path(__file__).parent / "part3_out"
    with open(out_dir / "probe_results.json") as fh:
        probe_records = json.load(fh)

    jumps = jump_magnitude_by_config(probe_records)
    precision = linear_precision_by_config()

    result = {}
    print(f"{'config':38s} {'median jump':>14s} {'median lin.prec.err':>22s}")
    for name in CONFIGS:
        mj = float(np.median(jumps[name])) if jumps[name] else float("nan")
        mp = float(np.median(precision[name])) if precision[name] else float("nan")
        print(f"{name:38s} {mj:14.3e} {mp:22.3e}")
        result[name] = {"jumps": jumps[name], "linear_precision_errors": precision[name],
                         "median_jump": mj, "median_linear_precision_error": mp}

    with open(out_dir / "part4_sweep.json", "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
