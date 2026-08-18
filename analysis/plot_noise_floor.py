"""Renders the Part 1 FD-gradient-error-vs-h plot from noise_floor.py's saved
curves: log-log, both branches (O(h^2) truncation, O(delta/h) noise), h* marked.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_REF = "#2a78d6"
COLOR_VAR = "#eb6834"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_SURFACE = "#fcfcfb"


def plot(out_dir: Path, png_path: Path) -> None:
    ref = np.load(out_dir / "fd_error_curve_tetrahedron_reference_pmvc.npz")
    var = np.load(out_dir / "fd_error_curve_tetrahedron_variant_energy_preserving.npz")

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    h_ref, e_ref, hstar, delta = ref["h"], ref["error"], float(ref["h_star"]), float(ref["delta"])
    h_var, e_var = var["h"], var["error"]

    ax.plot(h_ref, e_ref, color=COLOR_REF, lw=2, zorder=3, label="reference PMVC")
    ax.plot(h_var, e_var, color=COLOR_VAR, lw=1.5, ls=(0, (4, 2)), zorder=4,
            marker="o", ms=3.5, markevery=2, label="variant (energy preserving)")

    # Asymptotic guide lines: O(h^2) truncation branch and O(delta/h) noise branch,
    # anchored through h* so they visibly bracket the measured V-shape.
    h_trunc = h_ref[h_ref >= hstar]
    h_noise = h_ref[h_ref <= hstar]
    e_at_hstar = np.interp(hstar, h_ref[::-1], e_ref[::-1])
    guide_trunc = e_at_hstar * (h_trunc / hstar) ** 2
    guide_noise = e_at_hstar * (hstar / np.maximum(h_noise, 1e-300))

    ax.plot(h_trunc, guide_trunc, color=COLOR_MUTED, lw=1, ls=":", zorder=1)
    ax.plot(h_noise, guide_noise, color=COLOR_MUTED, lw=1, ls=":", zorder=1)

    ax.axvline(hstar, color=COLOR_MUTED, lw=1, ls="-", alpha=0.6, zorder=1)
    ax.annotate(f"h* = {hstar:.2e}\n(delta = {delta:.2e})",
                xy=(hstar, e_at_hstar), xytext=(hstar * 2.2, e_at_hstar * 6),
                fontsize=9, color=COLOR_TEXT,
                arrowprops=dict(arrowstyle="-", color=COLOR_MUTED, lw=0.8))

    ax.text(h_trunc[len(h_trunc) // 3], guide_trunc[len(h_trunc) // 3] * 2.2,
            "O(h^2) truncation", fontsize=8.5, color=COLOR_MUTED, style="italic")
    ax.text(h_noise[len(h_noise) // 2] * 0.5, guide_noise[len(h_noise) // 2] * 1.3,
            "O(delta/h) noise", fontsize=8.5, color=COLOR_MUTED, style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h", color=COLOR_TEXT)
    ax.set_ylabel("FD gradient error  |D1_h - true gradient|", color=COLOR_TEXT)
    ax.set_title("Noise floor: FD gradient error vs h (tetrahedron, deep interior)",
                 color=COLOR_TEXT, fontsize=11)
    ax.invert_xaxis()
    ax.tick_params(colors=COLOR_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_MUTED)
    ax.grid(True, which="both", ls="-", lw=0.4, color="#e5e4df", zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "noise_floor_out"
    plot(out_dir, out_dir / "fd_gradient_error_vs_h.png")
