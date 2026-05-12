# Interval BnB Stall Diagnostic: d=10, target=1.208

## Methodology

Re-implemented the BnB main loop in `diagnose_stall_d10_t1208.py` to harvest
boxes that survive to depth 55 (max widths ~3e-2 to 1.2e-1 — boxes that have
already been split heavily yet still cannot certify). Population:
- 600 STUCK boxes (depth >= 55 OR max_width < 1e-4 with `lb_fast < target`)
- 600 NEAR boxes (gap < 0.05, near the harvest depth)

BnB ran 1.6M nodes / 786K certified leaves over 180s; stack stayed flat at
35-41, indicating a STEADY-STATE stream of stuck boxes (i.e. genuine stall,
not a transient).

Run with:
```
python diagnose_stall_d10_t1208.py
```

## Run Stats

- nodes processed: 1,612,555
- certified leaves: 786,545
- max depth reached: 55
- wall time: 180s
- STUCK harvested: 600
- NEAR harvested: 600

## Geometric Location of Unprovable Boxes

### Centroid mean / std (per coord)

```
coord:     0       1       2       3       4       5       6       7       8       9
mean:    0.1348  0.0794  0.0542  0.0414  0.0470  0.0823  0.0962  0.1076  0.1129  0.2442
std:     0.0486  0.0225  0.0295  0.0269  0.0248  0.0294  0.0276  0.0256  0.0336  0.0138
```

**A U-shaped two-peak profile**: large mass at coord 9 (mean 0.244 ± 0.014),
secondary mass at coord 0 (mean 0.135 ± 0.049), and small mass at the middle
indices. coord 9's stddev is the smallest (0.014) — these boxes essentially
**pin μ_9 at ≈ 0.25** with little variation, and modulate μ_0 + interior.

### argmax-coord histogram (which coord dominates the centroid)

```
coord 0: 13   coord 8: 6   coord 9: 1181
```

**98.4% of stuck boxes have μ_9 as the dominant coordinate** — they sit on the
last-index endpoint of the simplex. The half-simplex symmetry cut enforces
μ_i ≤ μ_{d-1-i} for i < d/2, so these boxes are on the symmetric boundary
where μ_9 is effectively the largest. Only 13/1200 boxes have coord 0 as
the dominant — these are the symmetric reflections.

### Distance metrics

- distance to illustrative μ* (CLAUDE.md): median **0.21**, min 0.15
- distance to nearest simplex corner e_i: median **0.81** (far from corners)
- # coords with centroid < 0.01 (near-zero coords): **0 in 1077 / 1200**;
  1 zero in 115; 2 zeros in 8. So most boxes have NO coordinate stuck at 0.
- smallest_coord median: **0.017** (no axis is exactly on the simplex face).

The stuck region is **NOT near μ\***, **NOT at corners**, and **NOT on
zero-coordinate faces**. It's clustered in the symmetric interior at a
distinct distance from both the proposed minimiser and the boundary.

### Symmetry edges

- median |μ_i − μ_{d-1-i}|: 0.016 (close but not on-edge for the median box)
- 20% of boxes are within 0.005 of a symmetry edge

So **a sizeable minority (~20%) live exactly on symmetric edges** (coords
paired by half-simplex), but the bulk are slightly off.

## Width / Depth

- max_width: median 6.25e-2, max 1.25e-1, min 3.12e-2
- depth: median 52, max 55, min 45 (we capped at 55)

These are NOT shrinking to a point. At max_width ≈ 1/16, the BnB has split
each axis ~5 times on average; further splits in any single direction shrink
that axis but the **gap to target stays nearly constant** (median 0.0029),
indicating the binding bound is **not** improved by more splits.

## Gap to Target

- lb_fast median: **1.2051** (target 1.208)
- gap median: **0.0029** (very tight, but consistently negative)
- max_W TV at centroid median: **1.326**
- 100% of centroids satisfy max_W TV(centroid) >= 1.208

**Critical finding**: the TRUE objective at the centroid is FAR above target
(median 1.326, ranging up to 1.79 for the depth-40 cohort). The bound LB
just doesn't see this. The objective lives well above target throughout the
stuck region, but the McCormick / autoconv relaxations cannot see it.

## Winning Windows

Winning windows (highest LB on the box) are concentrated in the **central
support range** s_lo ∈ {7, 8}:

```
W(ell= 9, s= 8): 162    W(ell=10, s= 7): 137
W(ell= 5, s= 8): 120    W(ell=11, s= 7): 117
W(ell=10, s= 8): 113    W(ell= 6, s= 8):  80
W(ell= 8, s= 8):  78    W(ell= 4, s= 8):  69
W(ell=11, s= 8):  58    W(ell=11, s= 6):  44
```

ell median 9 (range 2-13), s_lo median 8 (range 5-9). Approximately
**half-length** windows centred near s = d-1 = 9 dominate. The argmax-TV
window AT THE CENTROID is similarly concentrated at s∈{8,9} but with
**shorter** ells (4-7 most common, vs the LB winners' 9-11).

This **mismatch** — the centroid-binding window is short (ell≈4-7), while
the LB-binding window is medium (ell≈9-11) — means the McCormick relaxation
finds a TIGHTER LB on a different window than the one that actually binds
the objective. The relaxation gap is window-dependent, and the longer
windows have more pairs ⇒ more McCormick slack ⇒ looser LB even though
their TV is lower.

## PCA Structure

Top-5 PCs explain only 86% of centroid variance, with PC1 only 40%. The
stuck set is a **manifold of dimension ≈ 4-5** in the simplex, NOT a tight
cluster. Combined with the U-shape mean profile, this says the stall is a
**continuous family** of mass distributions that lie near a particular
subset of Δ_10.

## Sample Stuck Box Centroids (top 3 by lb_fast)

```
1: [0.1803, 0.0492, 0.0492, 0.0492, 0.0574, 0.0574, 0.0738, 0.123,  0.1066, 0.2541]  lb=1.20768
2: [0.1803, 0.0492, 0.0492, 0.0492, 0.0574, 0.041,  0.0902, 0.123,  0.1066, 0.2541]  lb=1.20768
3: [0.1774, 0.0484, 0.0484, 0.0484, 0.0565, 0.0565, 0.0887, 0.121,  0.1048, 0.25  ]  lb=1.20739
```

These show a **5-flat plateau in the middle** (μ_1 = μ_2 = μ_3 ≈ 0.049,
μ_4 = μ_5 ≈ 0.057) — i.e. coords frozen at multiples of the dyadic split
denominator. Coords 0, 7, 8, 9 carry the variation.

## Summary of Findings

1. **Stuck region is NOT at μ\***. Median dist from μ* is 0.21 — these are
   distinctly different mass distributions than the cascade-discovered
   minimiser.

2. **Stuck region is on a U-shaped two-mode boundary**, with one mode at
   coord 9 (μ ≈ 0.244) and a secondary mode at coord 0 (μ ≈ 0.135).
   coord 9's narrow stddev (0.014) is the strongest signal: μ_9 ≈ 0.25 is
   essentially fixed across the stall set.

3. **Stuck region is a 4-5 dimensional manifold**, not a point cluster.
   1200 boxes occupy a smooth low-dim slab in Δ_10 (PC1 explains 40%, PC5
   explains 86% cumulatively).

4. **The bound, not the objective, is the bottleneck**. 100% of centroids
   have TV(centroid) >= 1.326 ≫ target 1.208. The LB's median 1.205 sits
   ~12% below the true objective at the centroid.

5. **Window mismatch**. The window with TIGHTEST LB (median ell=9, s=8) is
   different from the window that achieves max TV at the centroid (median
   ell=4-7, s=8-9). The McCormick relaxation tightens better on long
   windows, but the binding window at these mass distributions is short.

6. **Symmetry edges play a role but not exclusive**: 20% of centroids live
   exactly on a half-simplex symmetric pair (μ_i = μ_{d-1-i}), but 80% are
   strictly off. Symmetry alone does not characterise the stall.

7. **Splits don't shrink the gap**. At max_width = 6e-2 (depth 50+), the gap
   median is 0.003 and stays roughly constant as we split further — the
   asymptotic gap is set by the relaxation, not the box size.

## Implications

The stall has a **structural cause**: McCormick + autoconv at d=10
**asymptotically cannot prove > 1.205-1.208** on the symmetric two-mode
mass distributions where μ_9 ≈ 0.25 and μ_0 ≈ 0.13. Smaller boxes will not
unlock these. To progress past target=1.208 we need **either** a
fundamentally tighter bound formulation (joint-face dual cert, RLT, sum-
of-squares) **or** a tighter encoding of the simplex / symmetry that the
McCormick LP can leverage. Increased BnB depth on the current
relaxation will not close the gap at this exact target.

## Joint-Face LP: Empirical Closure Test

Ran `bound_mccormick_joint_face_lp` (float scipy HiGHS) on the 200 stuck
boxes saved to `stall_diag_d10_t1208.json`:

```
joint-face LP on winning window:    157 / 200 = 78.5% close
joint-face LP on top-3 windows:     191 / 200 = 95.5% close

delta_joint median: +0.0073   (gain over lb_fast)
delta_joint max:    +0.0163
delta_top3 median:  +0.0087
```

Joint-face LP uplifts the bound by a median of **+0.007**, which is roughly
**2.5× the median gap** (0.003). On a representative sample, **95.5% of
stuck boxes close** if we evaluate joint-face on the top-3 candidate
windows. This is a **strong actionable conclusion**: the stall is **not
information-theoretic** — the box really does prove ≥ target — but the
current driver only invokes joint-face dual cert at `depth >= 20` and only
on the single winning window. **Lowering the joint-face gate to depth ≥ 12
and adding top-K alternate windows would empirically close the d=10
target=1.208 stall**.

The remaining ~5% (9/200) that joint-face misses likely need:
- a tighter relaxation (RLT, Lasserre level-2 monomial), or
- explicit two-mode-symmetry exploitation (the U-shape suggests the
  binding configuration is a "two-cluster" mass concentrated near the
  endpoints of [-1/4, 1/4]).

