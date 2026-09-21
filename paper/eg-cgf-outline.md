# Positive Mean Value Coordinates with Multiple Hits and Interior Distances

Structural outline for a short paper (Eurographics / Computer Graphics Forum).
Target: 8 pages + references in the `egpubl` CGF style, or 4 pages if submitted as an
EG short paper — the section budget below gives both numbers as `short / full`.

The outline records *what* each section argues and which artifact in this repository
supplies the evidence, so writing is a matter of expanding the bullets, not of
rediscovering the results.

---

## Title / author block

- Working title above. Alternatives:
  - *Multi-Hit Positive Mean Value Coordinates with Cage-Aware Interior Distances*
  - *Smoother Positive Mean Value Coordinates by Depth Peeling and Heat-Method Distances*
- CCS: Computing methodologies → Computer graphics → Shape modeling → Mesh geometry models.
- Keywords: cage-based deformation, generalized barycentric coordinates, mean value
  coordinates, visibility, geodesic distance, GPU.

## Abstract (150–200 words)

One sentence each:
1. Cage-based deformation needs coordinates that are positive and visibility-aware;
   PMVC is the standard answer.
2. PMVC's single-hit rasterized visibility test makes its weights discontinuous in the
   gradient across visibility-event surfaces, and its Euclidean hit distance ignores the
   cage geometry, so thin non-convex walls leak influence.
3. We contribute (a) a multi-hit (depth-peeled) PMVC weighting with an *energy-preserving*
   hit combination that is provably non-negative, and (b) an interior-distance PMVC
   variant whose hit distance is a heat-method geodesic inside the cage volume.
4. Both live in one unified GPU pipeline; flags select the variant.
5. Result: median gradient jump across visibility-event surfaces drops 2–3.7x at a
   5–10% linear-precision cost, with no loss of positivity or partition of unity.

## 1. Introduction — 0.75 / 1.25 pages

- Cage-based deformation in production; why *positive* coordinates matter (no inverted
  or over-shooting influence, intuitive handles).
- The two defects we attack, stated as the paper's two research questions:
  - **Q1 (multiple hits).** PMVC discards everything a ray hits after the first cage
    triangle. The first hit changes identity abruptly as the mesh vertex crosses an
    edge–vertex (EV) event surface, and the weight is only C⁰ there.
  - **Q2 (interior distance).** The PMVC weight `solidAngle * (1 - d)` uses the Euclidean
    ray length `d`, which crosses cage walls. In a non-convex cage (U/L shapes,
    dumbbells, limbs of a character) two points separated by a thin wall are treated as
    close.
- Contributions, as a bulleted list of four:
  1. A multi-hit PMVC family parameterized by per-layer weights (α, β, θ) that subsumes
     reference PMVC and the offset variant PMVCO as special cases.
  2. An energy-preserving hit combination that redirects the second hit's subtraction
     onto the triangle that was over-counted, with a proof of non-negativity and of
     partition of unity.
  3. An interior-distance weighting built on a heat-method solve over a voxelization of
     the cage interior, injected as a *detour* (interior − Euclidean) so the scheme
     degenerates exactly to PMVC wherever the cage is locally convex.
  4. A continuity study with an explicit noise-floor gate, quantifying what the variants
     buy and what they cost.
- Honest framing sentence (carry it from `analysis/REPORT.md`): the variants do **not**
  make PMVC C¹; they shrink the gradient jump. Say so here, not only in Section 7.

## 2. Related work — 0.75 / 1 page

Four short paragraphs, no survey prose:
- **Generalized barycentric coordinates.** MVC, harmonic/BBW, LBC, Green, MEC, MLC,
  QMVC, SOMIG — all implemented in the host system, so comparison is cheap; cite the
  EG STAR on cage-based deformation as the umbrella reference.
- **Positivity and visibility.** Lipman et al.'s PMVC; spherical-rasterization
  implementations; the offset variant.
- **Depth peeling / order-independent visibility** as the mechanism we borrow for
  multiple hits.
- **Geodesic and interior distances.** Heat method (Crane et al. 2013); volumetric /
  diffusion distances used as shape-aware kernels in skinning weights.

## 3. Background and notation — 0.5 / 0.75 pages

- Cage `C = (V_C, F_C)`, mesh vertex `p`, coordinates `λ_i(p)`, deformation
  `p' = Σ λ_i v'_i`; requirements: partition of unity, linear precision, non-negativity.
- The rasterized PMVC estimator: cubemap around `p`, per-texel solid angle `Ω_t`,
  first-hit triangle `T(t)` with barycentrics `b(t)`, unnormalized weight
  `λ_i += b_i(t) · Ω_t · (1 − d_t)`, `wsum += Ω_t · (1 − d_t)`.
- Where the gradient discontinuity comes from: `T(t)` is piecewise constant in `p`;
  its jump set is the EV/VE/EEE event surfaces of the cage.
- One figure slot: **Figure 1**, 2D diagram of a ray hitting a non-convex cage twice,
  annotating hit 1 / hit 2 / the over-counted triangle and the Euclidean-vs-interior
  path. This figure carries both contributions at once.

## 4. Method — 2.5 / 3.5 pages (the core)

### 4.1 A unified multi-hit weighting

- Depth peeling gives hits `h_0, h_1, h_2, …` per texel, `d_{h_0} < d_{h_1} < …`.
- Generalized estimator: each layer `k` carries a weight `w_k`; reference PMVC is
  `w = (1, 0, 1, 0, …)` (odd layers are the negative-orientation layers, dropped);
  the three-hit variant is `w = (α, β, θ)`; PMVCO drops the distance term entirely.
- Table 1 in this subsection: variant → (hit count, α, β, θ, flags) → what it reduces to.

### 4.2 Two ways to spend the second hit

- **Naive (signed) combination.** β < 0 deposits negative mass on the *second* hit's own
  triangle. Effect: the gradient jump collapses (≈50x on the dumbbell probe set), but
  coordinates go negative — up to −0.016, on 22% of interior sample points — which
  breaks the one property PMVC exists to provide.
- **Energy-preserving combination.** The second hit gets no contribution of its own; it
  removes `β·w_1` from the *first* hit of the same ray, deposited on the first hit's
  triangle: `emit(T_0, max(α·w_0 − β·w_1, 0))`.
- **Proposition 1 (non-negativity).** Depth peeling gives `d_{h_1} > d_{h_0}`, hence
  `w_1 < w_0`, hence `α·w_0 − β·w_1 ≥ 0` for `β ≤ α`; the clamp covers the interior-detour
  case where a detour may reorder the two hit distances. Two-sentence proof.
- **Proposition 2 (partition of unity).** Both combinations deposit a single scalar over
  one triangle's barycentrics, which sum to one, so `Σ_i λ_i = wsum` holds regardless of
  how hits are combined. One sentence — it is the same argument for both rules.
- State the trade-off explicitly: the energy-preserving rule is *not* the smoothest
  member of the family, it is the smoothest provably positive one.

### 4.3 Interior distances as a detour

- Voxelize the cage interior (boundary shell + flood-filled exterior + interior labels,
  resolution `N` on the longest axis, 2-voxel margin).
- Per cage vertex, one heat-method solve on the solid voxels: `(I − tΔ)u = δ`,
  `X = −∇u/|∇u|`, `Δφ = ∇·X`, with exterior voxels excluded from every stencil, which is
  exactly the zero-Neumann wall condition. Trilinear sampling at mesh vertices;
  Euclidean fallback for unreachable pairs.
- **The key design decision:** store and interpolate the *detour* `δ_ic = d_interior −
  d_euclid ≥ 0`, not the absolute interior distance. Two consequences to argue:
  1. Where the cage is locally convex the detour is 0 and the scheme is *bit-identical*
     to PMVC — no regression on convex cages, which the noise-floor measurements confirm.
  2. Interpolating absolute distances barycentrically overestimates mid-triangle hit
     distance on coarse cages; interpolating the detour does not.
- Ray-space injection: reconstruct eye-space depth `z`, lengthen by `detour · cosθ_t` for
  the texel direction, re-encode with the same projection mapping. Give the three
  formulas; this is where the method is reproducible or not.
- **Figure 2:** interior-distance color map vs Euclidean distance map on the same cage
  vertex, same fixed `distance_max`, viridis + isolines (exactly what
  `MeshExportDistanceFieldOperation` writes). The isoline bending around a non-convex
  neck is the whole argument in one image.

### 4.4 Orthogonality

- One short paragraph: the flags are independent, so `{1-hit, 3-hit} × {Euclidean,
  interior}` are all valid configurations, and the evaluation treats them as a grid.

## 5. GPU implementation — 0.75 / 1 page

- Ring pipeline: a ring of cubemap render slots, filled / dispatched / read back by a
  worker pool, one cubemap per mesh vertex, six faces, per-face command buffers, two
  ping-pong slots so two peeled layers of the same cubemap are in flight together.
- One compute shader for every variant; the differences are push constants
  (`uHitWeightA/B`, flags `interiorDistance | solidAngleOnly | subtractSecondFromFirst`).
  Emphasize this: it makes the comparison in Section 6 apples-to-apples — the variants
  differ only in the constants, never in the code path.
- Accumulation by `atomicAdd` into λ and `wsum`; per-vertex near plane from the minimum
  point-to-cage-triangle distance (degenerate-triangle handling).
- Cost model: heat solves are `|V_C|` Poisson solves, done once per (cage, mesh) pair and
  reused across all deformed cages; the three-hit variant costs one extra peel pass and
  one extra color image. **Table 2:** timings per stage (the harness already writes
  `results/timings_<stem>.json`).

## 6. Evaluation — 2 / 2.5 pages

State the harness once: `evaluation/gen_eval_configs.py` emits the cartesian product of
model entries and coordinate setups; configs are deterministic and byte-reproducible.

### 6.1 Protocol and noise floor
- Why a noise floor is needed at all: finite-difference probes of a rasterized estimator
  measure discretization jitter unless `h > h* = δ^{1/3}`.
- Report `δ` and `h*` per (cage, scheme), face_size 32; note that reference and variant
  are numerically identical on convex cages — a prediction of the method, not a
  coincidence.
- Note the two distinct noise floors on the tetrahedron (local dense-line scatter vs the
  ~1e-9 floor against the analytic barycentric gradient) so the reader is not confused by
  the two numbers.

### 6.2 Continuity class
- Two diagnostics: log-log FD slope, and Richardson-extrapolated one-sided jumps.
- **Result:** both schemes are C⁰ and not C¹; *zero* probes show a real value
  discontinuity (value jumps are 2–4 orders below gradient jumps). Report that
  Diagnostic A's raw threshold mislabels ~30–40% of probes as C⁰-break, and that
  Diagnostic B is the reliable one — a methodological point worth keeping, not hiding.
- **Table 3:** median / max gradient jump per (cage, scheme). **Figure 3:** jump
  histogram with medians marked; **Figure 4:** the largest-jump line probe, showing the
  reference bending more sharply than the variant.

### 6.3 The smoothing / positivity trade-off
- **Table 4** (the paper's central table): reference vs `α1β0θ1` vs naive `α1β−1θ1` vs
  energy-preserving `α1β1θ1`, columns = median gradient jump, median linear-precision
  error, negativity. This is the table that makes the design decision in 4.2 legible.
- Note that the linear-precision penalty (+5–12%) comes from adding hits at all, not from
  the combination rule.

### 6.4 Interior distance
- Qualitative: influence maps and distance maps on non-convex cages (U-shape, dumbbell,
  and a character limb — armadillo/beast from the model set); show that influence no
  longer bleeds across a thin wall.
- Quantitative: convex cages reproduce PMVC exactly (zero detour); resolution sweep of
  the voxel grid `N` against distance error and solve time.

### 6.5 Comparison to other coordinates
- A single figure of the same deformation under MVC / QMVC / Green / MEC / MLC / PMVC /
  ours, since the host system computes all of them from one config. Keep it to one row
  of images plus one sentence — the paper's claim is about PMVC, not about winning a
  bake-off.

## 7. Limitations and future work — 0.4 / 0.5 pages

Write this from `analysis/REPORT.md`'s "what was NOT tested" list; it is already honest
and specific:
- Only EV event surfaces were probed. VE surfaces are enumerated but unprobed; the
  stochastic segment sweep intended to catch EEE events was not run, so EEE behavior is
  unconfirmed in either direction.
- Sample sizes (4 probes/surface, 8–16 interior points/cage) establish direction and
  magnitude, not confidence intervals — the dumbbell max-jump reversal (variant worse at
  the extreme while better at the median) could flip with more samples. Report it anyway.
- The continuity numbers come from a CPU reimplementation validated against synthetic and
  analytic gates; a cross-check against the shader's own `weights.dmat` is outstanding.
- Voxel resolution bounds how thin a cage wall the interior distance can resolve (a wall
  thinner than one voxel leaks through the flood fill).
- Grid-sweep maps of negativity / partition of unity / deformation quality were replaced
  by point samples.
- Future: a continuous blend parameter between the naive and energy-preserving rules;
  interior distances on a tetrahedral embedding instead of a voxel grid; per-vertex
  adaptive hit counts driven by local cage convexity.

## 8. Conclusion — 0.2 / 0.25 pages

Three sentences: the family and its two useful members; what the measurements support
(smaller jumps, preserved positivity, cage-aware falloff) and what they do not (a
continuity-class improvement); the pipeline is open in the host project.

## Acknowledgements / reproducibility statement

One paragraph naming the repository, the config generator, and the exact coordinate
setups used for every table, so a reader can regenerate the numbers.

---

## Figure and table budget

| # | Item | Section | Source in repo |
|---|------|---------|----------------|
| Fig. 1 | 2D diagram: two hits + Euclidean vs interior path | 3 | new vector art |
| Fig. 2 | Euclidean vs interior distance map, fixed `distance_max` | 4.3 | `*_distance_map.obj` exports |
| Fig. 3 | Gradient-jump histogram, both schemes | 6.2 | `part3_out/gradient_jump_histogram.png` |
| Fig. 4 | Largest-jump line probe | 6.2 | `part3_out/line_probe_example.png` |
| Fig. 5 | Trade-off scatter (jump vs linear precision vs negativity) | 6.3 | `part3_out/tradeoff.png` |
| Fig. 6 | Influence maps across non-convex cages | 6.4 | `influence_map.obj` exports |
| Fig. 7 | Coordinate comparison row | 6.5 | evaluation batch |
| Tab. 1 | Variant → parameter mapping | 4.1 | shader flags |
| Tab. 2 | Per-stage timings | 5 | `results/timings_*.json` |
| Tab. 3 | Gradient jump per cage and scheme | 6.2 | `part3_out/probe_results.json` |
| Tab. 4 | Four-config trade-off | 6.3 | `part3_out/part4_sweep.json` |

For an EG short paper, cut Figures 4, 6 and 7 and Table 2, fold Section 5 into 4, and
reduce Section 6 to the noise floor, Table 4 and one interior-distance figure.

## Writing order

1. Section 4 (method) — everything else is scaffolding around it.
2. Section 6 with the tables filled from the existing JSON, so the claims are fixed
   before the prose commits to them.
3. Sections 1, 7 and the abstract last, phrased to match the measured numbers exactly.
