# L^{3/2} Moment-Lasserre SDP — Pilot Report

**Date:** 2026-04-29
**Code:** [lasserre/threepoint_l32_moment.py](threepoint_l32_moment.py)
**Tests:** [tests/test_threepoint_l32_moment.py](../tests/test_threepoint_l32_moment.py)

## TL;DR — encoding is sound but VACUOUS in practice

The L^{3/2} bound encoded via auxiliary density `g` and Putinar localizer
`g^2 >= f^3` is a SOUND relaxation but does **NOT bind** at any tested
B in [0.5, 5.0] or k in {6, 7, 8}, with or without simultaneous L^infty
constraint. The relaxation collapses similarly to V2's pseudo-measure
problem: it allows singular `g` with small `int g` and large `int g^2`,
which trivially satisfies the moment-level Putinar constraint without
forcing actual L^{3/2} norm control.

**Lambda > 1.2802 NOT achieved through this encoding alone.**

## 1. Encoding Implemented

### Variables

* `m_a` (a = 0..2k) — 1D moments of f, V3 baseline
* `g_{ab}` (a+b <= 2k) or `y_{abc}` (a+b+c <= 2k) — V3 baseline 2D / 3D
* `z_a` (a = 0..2k) — 1D moments of auxiliary density g
* `mu_{a, j, k}` — joint moments `int x^a f^j g^k dx`, with
  a in [0, 2 K_joint], j in [0, 4], k in [0, 2]
  (allocated lazily; only even a are CVXPY variables, odd a forced 0
  by reflection symmetry)

### Linear linkage equations

For all even a in [0, max_a]:

* `mu_{a,0,0} = J_a` (Lebesgue moment, constant)
* `mu_{a,1,0} = m_a` (or `y_{a,0,0}` in 3pt)
* `mu_{a,0,1} = z_a`

### PSD constraints (in addition to V3 baseline blocks)

1. **Hausdorff PSD on z** (g >= 0 measure on [-1, 1] rescaled):
   * `M_k(z) >> 0`, size k+1
   * Box localizer `(1 - x^2) M_{k-1}(z) >> 0`, size k
2. **Joint moment matrix PSD** at order K_joint = max(1, k-2):
   * Basis `B_J = {x^a f^j g^k : a <= K_joint, j in {0,1,2}, k in {0,1}}`
   * Size 6*(K_joint+1)
3. **Putinar localizer for `g^2 - f^3 >= 0`** at order K_L = max(1, k-2):
   * `H[i,j] = mu_{i+j, 0, 2} - mu_{i+j, 3, 0}`, Hankel size K_L+1, PSD
   * Box-localizer version size K_L

### Linear bound

* `z_0 <= B_resc`, where `B_resc = B^{3/2} * mu_scale^{1/2}`, encoding
  `int g_orig du <= B^{3/2}` via the rescaling formulas (see code).

### Soundness sketch

Putinar PSD on `(mu_{a, 0, 2} - mu_{a, 3, 0})` implies that the
sequence `(mu_{a, 0, 2} - mu_{a, 3, 0})` is a Hausdorff moment sequence
of some positive measure on [-1/4, 1/4]. Combined with joint-PSD
consistency, this is **necessary** for `g^2 >= f^3` pointwise in the
exact problem; for actual integrable functions f, g it gives
`int g >= int f^{3/2}`, hence `||f||_{3/2}^{3/2} <= z_0 <= B_resc`.
The linkage is sound; the relaxation is sub-tight.

## 2. Numerical Results

All numbers are 2pt baseline (`build_2pt_l32`); 3pt yields the same
qualitative finding.

### Sweep: pure L^{3/2}, no L^infty

| k | B in {1.5, 1.55, 1.6, 1.7, 1.8, 2.0} |       lambda      | z_0 binding? |
|--:|--------------------------------------|-----------------:|-------------:|
| 6 | all values                           | 0.22676 (constant) | NO          |
| 7 | all values                           | 0.23909 (constant) | NO          |

* Lambda is independent of B to ~5 digits — encoding has no effect.
* `z_0` realized = ~0.16 to ~0.88 (rescaled), always strictly below B_resc.

### V3 baseline comparison (no L^{3/2}, no L^infty)

* k=6: V3 lambda = 0.22676  =  matches our encoding exactly
* k=7: V3 lambda = 0.23909  =  matches our encoding exactly

So the L^{3/2} encoding adds *zero* information.

### L^{3/2} combined with L^infty

| k | B    | f_infty | lambda             |
|--:|-----:|--------:|-------------------:|
| 7 | 1.5  | 2.10    | 1.342348           |
| 7 | 5.0  | 2.10    | 1.342348           |
| 7 | 1.5  | 2.15    | 1.280304           |
| 7 | 5.0  | 2.15    | 1.280304           |

The L^{3/2} adds nothing on top of L^infty either.

### Diagnostic: realized joint moments at B=0.5 (extreme)

```
int g    = 0.16          (bound 0.18, NOT binding)
int g^2  = 33.5          (HUGE)
int f^3  = 19.1          (HUGE)
int f^4  = 110.3         (very HUGE)
```

`g^2 >= f^3` as integrals (33.5 > 19.1) and as Hausdorff measures (PSD
satisfied). But these singular pseudo-measures don't correspond to
actual L^infty-bounded functions.

## 3. Does lambda > 1.2802 emerge?

**No.** The only way to get lambda above 1.2802 in the tests above is via
L^infty bound at f_infty <= 2.15, which yields ~1.28 (already in V3).
The L^{3/2} encoding contributes zero lift.

## 4. Honest Assessment of Soundness

The encoding is **sound** (the relaxation only adds necessary conditions
for `||f||_{3/2} <= B`, no spurious cuts). But it is **vacuous** at the
relaxation orders (k <= 8) and basis sizes I used. The relaxation
admits singular pseudo-densities for `g` that satisfy all constraints
without controlling `int g` in any meaningful way.

**Why the collapse:**

* The PSD constraint `H[i,j] = mu_{a, 0, 2} - mu_{a, 3, 0}` with
  Hausdorff support is sound but trivially satisfied — for any
  pseudo-measure `f`, we can choose `g`-moments to match.
* The joint-PSD constraint `H[(a,j,k),(a',j',k')] = mu_{a+a', j+j', k+k'}`
  imposes a Cauchy-Schwarz-like relation on cross-moments but
  doesn't force `int g^2` to be related to `(int g)^2` and pointwise
  `g`-values.
* Without a separate `||g||_infty` (or `||f||_infty`) bound, the
  relaxation always collapses to near-Dirac pseudo-measures, exactly
  as V2/V3 documented for the basic CD formulation.

This is the same fundamental issue that forced V3 to introduce the
L^infty bound. **L^{3/2} is structurally insufficient to exclude the
singular pseudo-densities** that drive the trivial relaxation value.

**To make L^{3/2} effective**, one would need either:

1. A higher-order joint-moment hierarchy that captures the non-linear
   relation `g >= f^{3/2}` more tightly (likely intractable at k > 4).
2. An auxiliary L^infty-on-g constraint (which moves the problem back
   to the L^infty regime).
3. A different formulation, e.g., conditioning on `g = f^{3/2}` via
   exact-equality moment matching — but this requires either the
   moments of `f^{3/2}` (not polynomial in m) or a discretization,
   which the discretized variant in `threepoint_l32.py` already does
   (and that one *does* bind, at the cost of being non-rigorous over
   continuous f).

## 5. Status

* Code: [lasserre/threepoint_l32_moment.py](threepoint_l32_moment.py) (~370 lines)
* Tests: [tests/test_threepoint_l32_moment.py](../tests/test_threepoint_l32_moment.py) — 5 tests, all pass
* Sweep results: [results/l32_moment_sweep.json](../results/l32_moment_sweep.json)

**Posterior P(this exact path beats 1.2802 with extra effort): < 5%.**
The structural barrier is the same as V2's CD pseudo-measure collapse.
Recommend halting work on this exact moment-Lasserre L^{3/2} encoding
and pivoting to either (a) discretized L^{3/2} (already in
`threepoint_l32.py`) with rigorous discretization-error bounds, or
(b) the L^infty + analytical-bridge path that V3 already identified
as the most promising line.
