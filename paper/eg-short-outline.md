# Positive Mean Value Coordinates with Multiple Hits and Interior Distances

Structure for a **4-page Eurographics short paper** (`egpubl`, two columns, references
included in the 4 pages). Budgets below are in words of body text and in column-inches;
the two together are what keep the draft from overrunning.

## The page arithmetic, first

Four pages = 8 columns. Fixed overhead comes off the top:

| Block | Cost | Note |
|---|---|---|
| Title, authors, abstract, CCS | 0.6 col | |
| Teaser (Fig. 1) | 1.0 col | full-width on p.1, spans both columns |
| Figure 2 + caption | 0.9 col | |
| Figure 3 + caption | 0.7 col | |
| Table 1 + caption | 0.6 col | |
| References (~20 entries) | 0.9 col | hard cap: 20, not 40 |
| **Overhead total** | **4.7 col** | |
| **Body text remaining** | **3.3 col ≈ 2000 words** | this is the real budget |

2000 words is roughly one conference talk. Every section below carries its share, and
the shares add up to 2000 — if a section grows, name the one that shrinks.

## What gets cut from the full-length version

Decided, not deferred. The 8-page outline is in git history (`paper/eg-cgf-outline.md`,
commit 1d3f5a9) if the work is ever expanded again.

- **Cut entirely:** the coordinate bake-off against MVC/QMVC/Green/MEC/MLC; the per-stage
  timing table; the separate related-work section; the noise-floor section; the two-
  diagnostic continuity methodology; the voxel-resolution sweep; Figures 4, 6, 7.
- **Demoted to one sentence each:** the noise-floor gate (a clause in the protocol), the
  C⁰-not-C¹ finding (a clause in the results), the partition-of-unity argument (a clause
  in §3.2), the GPU pipeline (a paragraph in §3.4).
- **Kept at full strength:** the energy-preserving combination and its positivity
  proposition, the detour formulation of interior distance, and Table 1. These three are
  the paper.

## Framing decision

A 4-page paper supports one headline, not two. The lead is the **multi-hit
energy-preserving weighting**, because it is the part with a proof and a number
(2–3.7× smaller gradient jumps, positivity preserved). **Interior distance is presented
as the second half of one idea** — both contributions modify the same per-texel hit term,
one in the *weight* and one in the *distance* — rather than as an independent second
contribution. That framing is what makes both fit; presenting them as two separate
papers-in-one is what would not fit.

---

## Title block and abstract — 120 words

Four sentences: PMVC's first-hit rule is blind to what a ray hits next and its Euclidean
hit distance crosses cage walls; we generalize the per-texel hit term along both axes; the
energy-preserving combination is provably non-negative and cuts median gradient jumps
2–3.7×; interior distances make influence follow the cage volume and reduce exactly to
PMVC where the cage is convex.

**Fig. 1 (teaser, full width, p.1):** three panels on one non-convex cage — reference
PMVC influence map | ours | the 2D ray diagram annotating hit 1, hit 2, the over-counted
triangle, and the Euclidean-vs-interior path. The diagram panel does the work that
Section 3's prose would otherwise have to do, which is why it is worth a full column.

## 1. Introduction — 350 words

- Para 1 (100 w): cage-based deformation; positivity is the property PMVC exists to
  provide; the rasterized estimator `λ_i += b_i(t)·Ω_t·(1−d_t)` stated inline here, since
  there is no room for a background section.
- Para 2 (150 w): the two defects, one sentence of cause each. The first-hit triangle
  `T(t)` is piecewise constant in `p`, so the weight is only C⁰ across visibility-event
  surfaces. The distance `d_t` is Euclidean, so a thin non-convex wall does not separate
  points that it geometrically should.
- Para 3 (100 w): contributions as three bullets — the (α, β, θ) multi-hit family
  subsuming PMVC and PMVCO; the energy-preserving combination with a positivity proof;
  the detour formulation of interior distance. Last sentence states the honest limit up
  front: this reduces the size of the gradient jump, it does not remove it.

Related work is 3–4 sentences inside Para 1, cited inline (MVC, PMVC, the EG STAR on
cage-based deformation, the heat method, depth peeling). No `\section{Related Work}`.

## 2. The generalized hit term — 200 words

One short section that sets up both contributions at once, so §3 can be pure method.

- Depth peeling yields hits `h_0, h_1, …` per texel with `d_{h_0} < d_{h_1} < …`.
- Generalized estimator: layer `k` carries weight `w_k` and distance `d_k`. Reference PMVC
  is `w = (1,0,1,0,…)`; PMVCO drops the distance term; our variant is `w = (α, β, θ)`.
- One sentence naming the two axes — *which hits contribute* (§3.1–3.2) and *what distance
  each hit reports* (§3.3) — and that they are independent flags, so the four combinations
  are all valid configurations.

## 3. Method — 900 words (the paper)

### 3.1 Spending the second hit — 200 w
The naive signed rule (β < 0, negative mass on the second hit's own triangle) collapses the
gradient jump ~50× but drives coordinates negative — up to −0.016 on 22% of dumbbell
interior points. State it as the motivating failure, in three sentences; it justifies
everything in §3.2.

### 3.2 The energy-preserving combination — 300 w
- The rule: the second hit contributes nothing of its own; it removes `β·w_1` from the
  first hit of the same ray, deposited on the first hit's triangle —
  `emit(T_0, max(α·w_0 − β·w_1, 0))`. Displayed equation.
- **Proposition 1.** Depth peeling gives `d_{h_1} > d_{h_0}`, hence `w_1 < w_0`, hence the
  contribution is non-negative for `β ≤ α`; the clamp covers the case where an interior
  detour reorders the two hit distances. Proof in two sentences, inline, not a numbered
  environment.
- Partition of unity in one sentence: a single scalar deposited over one triangle's
  barycentrics, which sum to one, so `Σλ_i = wsum` under any combination rule.
- Closing sentence, the framing that earns the design: this is not the smoothest member
  of the family, it is the smoothest provably positive one.

### 3.3 Interior distance as a detour — 300 w
- Voxelize the cage interior (boundary shell, flood-filled exterior, interior); one
  heat-method solve per cage vertex on the solid voxels, `(I − tΔ)u = δ`, `X = −∇u/|∇u|`,
  `Δφ = ∇·X`; excluding exterior voxels from every stencil *is* the zero-Neumann wall
  condition — say it in that one clause, do not derive it.
- **The design decision, given the most space in this subsection:** store and interpolate
  the detour `δ = d_interior − d_euclid ≥ 0`, not the absolute distance. Two consequences:
  convex regions have zero detour and reduce bit-identically to PMVC; barycentric
  interpolation of absolute distances would overestimate mid-triangle hit distance on
  coarse cages.
- Ray-space injection in three displayed formulas (eye-space depth, lengthen by
  `detour·cosθ_t`, re-encode). These are the reproducibility-critical lines — keep all
  three even under pressure.

### 3.4 Implementation — 100 w
One paragraph, no subsections: a ring of cubemap slots filled, dispatched and read back by
a worker pool; one compute shader for every variant, differing only in push constants.
Last sentence earns its keep by noting that this is *why* §4's comparison is
apples-to-apples — the variants share a code path.

## 4. Results — 400 words

No subsections; four paragraphs.

- **Protocol (80 w).** Deterministic config generator; finite-difference probes on two
  non-convex cages (l_shape, dumbbell) plus a convex control, gated above a measured
  noise floor `h* = δ^{1/3}` — the gate gets one clause, its derivation gets none.
- **Continuity (100 w).** Both schemes are C⁰ and not C¹; no probe shows a real value
  discontinuity (value jumps 2–4 orders below gradient jumps). Median gradient jump drops
  3.7× on l_shape and ~2× on dumbbell. One clause admitting the max-jump reversal on
  dumbbell — cheap to include and it is what a reviewer will otherwise catch.
- **Trade-off (120 w) — carries Table 1**, the four configurations × {median gradient
  jump, median linear-precision error, negativity}. The sentence that matters: the
  linear-precision cost (+5–12%) comes from adding hits at all, not from the combination
  rule.
- **Interior distance (100 w) — carries Fig. 3.** Convex cages reproduce PMVC exactly
  (zero detour, identical noise floor); on non-convex cages influence stops crossing thin
  walls. Qualitative only at this length, and say so rather than implying a measurement.

**Fig. 2:** trade-off scatter, jump vs linear precision, negativity marked
(`part3_out/tradeoff.png`).
**Fig. 3:** Euclidean vs interior distance map, same cage vertex, same fixed
`distance_max`, isolines visible (`*_distance_map.obj` exports).
**Table 1:** the four-config comparison (`part3_out/part4_sweep.json`).

## 5. Conclusion and limitations — 150 words

Merged, limitations first — at this length a separate limitations section reads as padding,
and burying them reads as evasion.

- Two sentences of limitation, chosen as the two a reviewer would raise anyway: only
  edge–vertex event surfaces were probed (EEE behavior unconfirmed), and the continuity
  numbers come from a CPU reimplementation validated against analytic gates rather than a
  cross-check against the shader's own output. Voxel resolution bounding resolvable wall
  thickness gets a half-sentence.
- Two sentences of conclusion: what the family gives (smaller jumps at preserved
  positivity, cage-aware falloff, one pipeline) and what it does not (a better continuity
  class).
- One sentence pointing at the open implementation and the config generator, which is the
  whole reproducibility statement at this length.

---

## Figure and table budget (final — 3 figures, 1 table)

| # | Item | Placement | Source |
|---|------|-----------|--------|
| Fig. 1 | Teaser: PMVC vs ours vs ray diagram | p.1, full width | exports + new vector art |
| Fig. 2 | Trade-off scatter | §4 | `part3_out/tradeoff.png` |
| Fig. 3 | Euclidean vs interior distance map | §4 | `*_distance_map.obj` exports |
| Tab. 1 | Four-config trade-off | §4 | `part3_out/part4_sweep.json` |

## If it still overruns

Cut in this order, first item first:
1. Figure 2 — Table 1 already carries those numbers; the scatter is redundant with it.
2. §3.1 down to a single sentence inside §3.2.
3. The convex control from the protocol paragraph (keep the claim in §3.3, drop the
   measurement).
4. References from 20 to 15.

Do **not** cut, in any order: Proposition 1, the three ray-space formulas, Table 1, or the
sentence admitting C⁰-not-C¹.

## Writing order

1. Table 1 and the three figures — at 4 pages the artwork determines the text budget, so
   fix it before writing a sentence.
2. §3.2 and §3.3, the two subsections that must survive intact.
3. §4, phrased against the finished table.
4. Abstract and §1 last, mirroring §4's numbers exactly.
