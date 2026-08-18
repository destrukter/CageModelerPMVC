"""Quick manual smoke test (not the full pytest suite) exercising the rasterizer +
pmvc pipeline end to end before building the FD/continuity machinery on top of it."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cage  # noqa: E402
import pmvc  # noqa: E402
from rasterizer import CubemapRaster  # noqa: E402
from shapes import tetrahedron, cube, l_shape, u_shape, star_shape, dumbbell  # noqa: E402


def report_reflex(name, V, F):
    edges = cage.classify_reflex(V, F)
    print(f"{name}: V={len(V)} F={len(F)} reflex_edges={len(edges)}")
    return edges


print("=== reflex edge counts ===")
Vt, Ft = tetrahedron()
report_reflex("tetrahedron", Vt, Ft)
Vc, Fc = cube()
report_reflex("cube", Vc, Fc)
for name, builder in [("l_shape", l_shape), ("u_shape", u_shape), ("star", star_shape), ("dumbbell", dumbbell)]:
    V, F = builder()
    edges = report_reflex(name, V, F)
    assert len(edges) > 0, f"{name} should have reflex edges"

print("\n=== PMVC sanity on cube (center should be ~1/8 each, reference PMVC hit_count=1) ===")
raster = CubemapRaster(face_size=32)
t0 = time.time()
w = pmvc.evaluate_pmvc(np.zeros((1, 3)), Vc, Fc, pmvc.REFERENCE_PMVC, raster)
print("elapsed", time.time() - t0, "s")
print("weights:", w[0])
print("sum:", w[0].sum())
print("max abs deviation from 1/8:", np.max(np.abs(w[0] - 0.125)))

print("\n=== PMVC sanity on cube, variant (energy preserving, hit_count=3) ===")
w2 = pmvc.evaluate_pmvc(np.zeros((1, 3)), Vc, Fc, pmvc.VARIANT_ENERGY_PRESERVING, raster)
print("weights:", w2[0])
print("sum:", w2[0].sum())

print("\n=== Negativity check: reference PMVC on non-convex dumbbell at a few interior points ===")
Vd, Fd = dumbbell()
rng = np.random.default_rng(0)
pts = []
while len(pts) < 20:
    p = rng.uniform(Vd.min(axis=0), Vd.max(axis=0))
    if cage.inside_cage(p[None, :], Vd, Fd)[0]:
        pts.append(p)
pts = np.array(pts)
w3 = pmvc.evaluate_pmvc(pts, Vd, Fd, pmvc.REFERENCE_PMVC, raster)
print("min weight:", w3.min(), "max |sum-1|:", np.max(np.abs(w3.sum(axis=1) - 1)))

print("\n=== naive (non-energy-preserving) 3-hit variant a1_bm1_t1: expect possible negatives ===")
w4 = pmvc.evaluate_pmvc(pts, Vd, Fd, pmvc.VARIANT_3HIT_A1_BM1_T1, raster)
print("min weight:", w4.min(), "fraction negative:", (w4 < -1e-9).mean())

print("\n=== energy-preserving 3-hit variant on same points: expect no negatives ===")
w5 = pmvc.evaluate_pmvc(pts, Vd, Fd, pmvc.VARIANT_ENERGY_PRESERVING, raster)
print("min weight:", w5.min(), "fraction negative:", (w5 < -1e-9).mean())
