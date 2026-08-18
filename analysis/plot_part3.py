"""Part 8 (scoped): visualizations built from part3_continuity.py's saved probe
results -- jump-magnitude histogram (both schemes overlaid) and one representative
line-probe figure (w, |grad w|, components; both schemes; 3 values of h)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import continuity
import pmvc
from rasterizer import CubemapRaster
from validation.shapes import l_shape, dumbbell

COLOR_REF = "#2a78d6"
COLOR_VAR = "#eb6834"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_SURFACE = "#fcfcfb"

CAGES = {"l_shape": l_shape, "dumbbell": dumbbell}
SCHEMES = {"reference_pmvc": pmvc.REFERENCE_PMVC, "variant_energy_preserving": pmvc.VARIANT_ENERGY_PRESERVING}


def plot_histogram(data: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    for scheme_name, color, label in (
        ("reference_pmvc", COLOR_REF, "reference PMVC"),
        ("variant_energy_preserving", COLOR_VAR, "variant (energy preserving)"),
    ):
        jumps = np.array([r["gradient_jump"] for r in data if r["scheme"] == scheme_name])
        jumps = jumps[jumps > 1e-6]
        log_jumps = np.log10(jumps)
        ax.hist(log_jumps, bins=18, color=color, alpha=0.55, label=f"{label} (n={len(jumps)})",
                 edgecolor=COLOR_SURFACE, linewidth=0.5)
        ax.axvline(np.median(log_jumps), color=color, lw=1.5, ls="--")

    ax.set_xlabel("log10(gradient jump magnitude)", color=COLOR_TEXT)
    ax.set_ylabel("probe count", color=COLOR_TEXT)
    ax.set_title("Gradient jump magnitude across all EV-surface probes (l_shape + dumbbell)",
                 fontsize=10.5, color=COLOR_TEXT)
    ax.tick_params(colors=COLOR_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_MUTED)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def plot_line_probe(record: dict, out_path: Path) -> None:
    V, F = CAGES[record["cage_name"]]()
    x0 = np.array(record["x0"])
    surf_meta = record["surface_meta"]
    # Recompute the surface normal from the same analytic construction used to
    # generate it (cheaper and exactly reproducible vs. serializing the surface).
    import cage as cage_mod
    edges = cage_mod.classify_reflex(V, F)
    surfaces = cage_mod.candidate_ev_surfaces(V, F, edges)
    surf = surfaces[record["surface_idx"]]
    n = surf.normal
    comp = record["worst_component"]

    t_dense = np.linspace(-3e-2, 3e-2, 401)
    h_marks = [1e-2, 3e-3, 1e-3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=COLOR_SURFACE)
    for ax in axes:
        ax.set_facecolor(COLOR_SURFACE)

    for scheme_name, color, label in (
        ("reference_pmvc", COLOR_REF, "reference PMVC"),
        ("variant_energy_preserving", COLOR_VAR, "variant"),
    ):
        params = dataclasses.replace(SCHEMES[scheme_name], face_size=32)
        raster = CubemapRaster(32)

        def f(t, params=params, raster=raster):
            p = x0[None, :] + t[:, None] * n[None, :]
            return pmvc.evaluate_pmvc(p, V, F, params, raster)

        w = f(t_dense)[:, comp]
        axes[0].plot(t_dense, w, color=color, lw=1.8, label=label)

        for h, alpha in zip(h_marks, (1.0, 0.6, 0.35)):
            dd = continuity.divided_differences(f, 0.0, np.array([h]))
            grad = dd.D1[0, comp]
            # local tangent line at +-h around 0, showing the FD slope estimate
            tt = np.array([-h, h])
            axes[1].plot(tt, grad * tt, color=color, lw=1.5, alpha=alpha,
                         label=(f"{label}, h={h:.0e}" if h == h_marks[0] else None))

    axes[0].axvline(0, color=COLOR_MUTED, lw=1, ls=":")
    axes[0].set_xlabel("t along surface normal", color=COLOR_TEXT)
    axes[0].set_ylabel(f"w (cage vertex {comp})", color=COLOR_TEXT)
    axes[0].set_title("w along the probe line", fontsize=10, color=COLOR_TEXT)
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].axvline(0, color=COLOR_MUTED, lw=1, ls=":")
    axes[1].set_xlabel("t", color=COLOR_TEXT)
    axes[1].set_ylabel("FD slope estimate * t (local tangent)", color=COLOR_TEXT)
    axes[1].set_title("D1_h tangent at 3 values of h (kink visible as slope mismatch)",
                       fontsize=10, color=COLOR_TEXT)
    axes[1].legend(frameon=False, fontsize=7.5)

    for ax in axes:
        ax.tick_params(colors=COLOR_MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_MUTED)

    fig.suptitle(f"{record['cage_name']} / {surf_meta} / probe {record['probe_idx']}",
                 fontsize=9.5, color=COLOR_MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "part3_out"
    with open(out_dir / "probe_results.json") as fh:
        data = json.load(fh)

    plot_histogram(data, out_dir / "gradient_jump_histogram.png")

    ref_rows = [r for r in data if r["scheme"] == "reference_pmvc"]
    biggest = max(ref_rows, key=lambda r: r["gradient_jump"])
    plot_line_probe(biggest, out_dir / "line_probe_example.png")
