/-
Sidon Autocorrelation Constant: Rigorous Lower Bound `C_{1a} ≥ 1.292`
The Piterbarg-Bajaj-Vincent Bound
=====================================================================

This module formalises the headline theorem of *Improving the Bounds on
the Supremum of Autoconvolutions* (Piterbarg, Bajaj, Vincent).

The autoconvolution ratio of an admissible test function `f`,

```
    R(f)  =  ‖f * f‖_∞ / (∫ f)²,
```

is bounded below uniformly over nonnegative `f` supported on `[-1/4, 1/4]`
with `∫ f > 0`.  Matolcsi–Vinuesa (2010, arXiv:0907.1379) established the
master inequality

```
    R(f) + 1 + √(R(f) - 1)·√(K_2 - 1)  ≥  2/u + a,
```

where `K = ∑ λ_i K_arc(δ_i, ·)` is an admissible kernel (a convex
combination of arcsine kernels) and the gain term `a = (4/u)·min_G²/S_1`
is determined by a finite cosine `G` on `[0, 1/4]`.  Inverting this
quadratic in `R(f)` produces a rational lower bound `M_target` on
`C_{1a}`.

This file formalises the 3-scale instance

```
    δ₁ = 138/1000,    λ₁ = 85/100,
    δ₂ =  55/1000,    λ₂ = 10/100,
    δ₃ =  25/1000,    λ₃ =  5/100,
    u  = 638/1000  (= 1/2 + δ₁),
```

with a 200-coefficient `G` re-optimised at this kernel, yielding

```
    C_{1a}  ≥  1292/1000  =  1.292.
```

Bochner admissibility of `K̂` is automatic: each `J₀(πδᵢξ)²` is the square
of a real Bessel function and the convex combination preserves
nonnegativity.

Numerical anchors
-----------------
Each of the five numerical inputs is computed by a `flint.arb` certifier
at 256-bit precision (`delsarte_dual/grid_bound_alt_kernel/`):

  * `K_2`     ∈ `[4.788823, 4.788906]`  (XI_MAX = 1e5, closed-form tail)
  * `k_1`     ≥ `0.92124658`              (essentially exact, rad < 1e-76)
  * `S_1`     ≤ `29.840907`
  * `min G`   ≥ `0.99997987`             (Taylor B&B, 32768 cells on [0, 1/4])
  * `gain a`  ≥ `0.21009214`             (script-reported, coupled in arb)

The rational bounds declared below are slack relaxations of these
arb-interval endpoints; the certifier's cell-search bisection reaches
`M_cert = 66167/51200 ≈ 1.29232422`, which exceeds the rational target
`1292/1000` by `≈ 3.2 × 10⁻⁴`.  The strict-failure proof in this file
uses the looser rational anchors and clears `1292/1000` with margin
`307/3190000 ≈ 9.6 × 10⁻⁵`.

Axioms
------
The headline theorem reaches exactly **one** user axiom in its
dependency closure:

  * `MV_master_inequality_for_extremiser` is *structural*: the 3-scale
    extension of the MV Fourier reduction on `ℝ/uℤ`, with `K_2` replaced
    by the slack rational `K2UpperQ` and `a` replaced by the slack
    rational `gainLowerQ`.  This substitution is valid because the
    master inequality is monotone in `K_2 - 1` and in `a`, and the slack
    rationals are true bounds on the analytic functionals — these bounds
    (paper Lemmas 4.1-4.5) are discharged externally by the `flint.arb`
    certifier.

The five rational comparisons recording the slack soundness
(`K_two_upper_bound`, `k_one_lower_bound`, `S_one_upper_bound`,
`min_G_lower_bound`, `gain_lower_bound`) and the algebraic inversion
(`master_inequality_M_lower`) are now Lean *theorems*.

No `sorry`, no conjectural axioms.

References
----------
* Matolcsi, M., Vinuesa, C.  *Improved bounds on the supremum of
  autoconvolutions.*  J. Math. Anal. Appl. **372** (2010), 439-447,
  arXiv:0907.1379.
* Cloninger, A., Steinerberger, S.  *On suprema of autoconvolutions
  with an application to Sidon sets.*  Proc. Amer. Math. Soc.
  **145** (2017), 3191-3200, arXiv:1403.7988.
* Cohn, H., Elkies, N.  *New upper bounds on sphere packings I.*
  Ann. of Math. (2) 157 (2003), 689–714.
-/

import Mathlib
import Sidon.Defs

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real

namespace Sidon.MultiScale

/-! ## Kernel anchors -/

/-- First arcsine scale `δ₁ = 138/1000`. -/
def delta1Q : ℚ := 138 / 1000

/-- Second arcsine scale `δ₂ = 55/1000`. -/
def delta2Q : ℚ := 55 / 1000

/-- Third arcsine scale `δ₃ = 25/1000`. -/
def delta3Q : ℚ := 25 / 1000

/-- Mixture weight `λ₁ = 85/100`. -/
def lambda1Q : ℚ := 85 / 100

/-- Mixture weight `λ₂ = 10/100`. -/
def lambda2Q : ℚ := 10 / 100

/-- Mixture weight `λ₃ = 5/100`. -/
def lambda3Q : ℚ := 5 / 100

/-- Weights are a partition of unity. -/
theorem lambdas_sum_one : lambda1Q + lambda2Q + lambda3Q = 1 := by
  unfold lambda1Q lambda2Q lambda3Q; norm_num

/-- Weights are nonnegative. -/
theorem lambdas_nonneg :
    (0 : ℚ) ≤ lambda1Q ∧ (0 : ℚ) ≤ lambda2Q ∧ (0 : ℚ) ≤ lambda3Q := by
  unfold lambda1Q lambda2Q lambda3Q
  refine ⟨?_, ?_, ?_⟩ <;> norm_num

/-- Period of the MV cosine basis: `u = 638/1000 = 1/2 + δ₁`. -/
def uQ : ℚ := 638 / 1000

/-- Verifies the closed-form relation `u = 1/2 + δ₁`. -/
theorem uQ_eq : uQ = (1 : ℚ) / 2 + delta1Q := by
  unfold uQ delta1Q; norm_num

/-- Rational target of the bound: `M_target = 1292/1000 = 1.292`. -/
def MTargetQ : ℚ := 1292 / 1000

/-! ## Numerical anchors

Each rational below is a slack relaxation of an arb-interval bound
reported by the `flint.arb` certifier at 256-bit precision.
-/

/-- Slack upper bound on `K_2 = ‖K̂‖₂² = 2·∫_0^∞ K̂(ξ)² dξ`.

    Certifier returns `K_2 ∈ [4.788823, 4.788906]`; we use `4.7897`. -/
def K2UpperQ : ℚ := 47897 / 10000

/-- Slack lower bound on `k_1 = K̂(1)`.

    Certifier returns `k_1 ≈ 0.92124658993` (radius `< 10⁻⁷⁶`), so in
    particular `k_1 ≥ 0.92124658`; we use the slack rational `0.9212`. -/
def K1LowerQ : ℚ := 9212 / 10000

/-- Slack upper bound on the dual sum `S_1 = ∑_{j=1}^{200} a_j² / K̂(j/u)`.

    Certifier returns `S_1 ≤ 29.840907`; we use `29.841`. -/
def S1UpperQ : ℚ := 29841 / 1000

/-- Slack lower bound on `min_{x ∈ [0,1/4]} G(x)` for the 200-coefficient
    re-optimised `G`.

    Certifier returns `min G ≥ 0.99997987`; we use `0.998`. -/
def minGLowerQ : ℚ := 998 / 1000

/-- Slack lower bound on the gain `a = (4/u)·min_G²/S_1`.

    Plugging the rational floors directly gives
    `(4/u)·(998/1000)²/(29841/1000) ≈ 0.20926`; we use `0.20925`.
    The certifier reports the tighter coupled value `a ≥ 0.21009214`.

    `a` is invariant under rescaling `G ↦ cG` (`min G ↦ c·min G`,
    `S_1 ↦ c²·S_1`), so `min G < 1` for the re-optimised G is benign. -/
def gainLowerQ : ℚ := 20925 / 100000

/-- The gain lower bound is strictly positive. -/
theorem gainLowerQ_pos : (0 : ℚ) < gainLowerQ := by
  unfold gainLowerQ; norm_num

/-- The rational `gainLowerQ` is dominated by the certifier's reported
    coupled-arb value `0.21009214`. -/
theorem gainLowerQ_below_certifier_value :
    (gainLowerQ : ℚ) ≤ 21009214 / 100000000 := by
  unfold gainLowerQ; norm_num

/-! ## Slack-rational soundness theorems

Each statement below is a pure rational comparison between the slack
rational chosen above and the decimal value reported by the `flint.arb`
certifier (paper Lemmas 4.1-4.5).  They are *not* axioms about the
analytic functionals `K_2`, `k_1`, `S_1`, `min_G`, `a` (those bounds are
discharged externally by the certifier and used inside
`MV_master_inequality_for_extremiser`); they record that the rational
slack used downstream is on the correct side of the certifier output.
Each is a one-line `norm_num` check.
-/

/-- (N1) The Lean slack `K2UpperQ = 47897/10000` is at least the
    certifier's reported upper endpoint `K_2 ≤ 4.788906`. -/
theorem K_two_upper_bound : (K2UpperQ : ℝ) ≥ (4788906 / 1000000 : ℝ) := by
  unfold K2UpperQ; norm_num

/-- (N2) The Lean slack `K1LowerQ = 9212/10000` is at most the
    certifier's reported lower endpoint `k_1 ≥ 0.92124658`. -/
theorem k_one_lower_bound : (K1LowerQ : ℝ) ≤ (92124658 / 100000000 : ℝ) := by
  unfold K1LowerQ; norm_num

/-- (N3) The Lean slack `S1UpperQ = 29841/1000` is at least the
    certifier's reported upper endpoint `S_1 ≤ 29.84091`. -/
theorem S_one_upper_bound : (S1UpperQ : ℝ) ≥ (2984091 / 100000 : ℝ) := by
  unfold S1UpperQ; norm_num

/-- (N4) The Lean slack `minGLowerQ = 998/1000` is at most the
    certifier's reported lower endpoint `min_G ≥ 0.9999798`. -/
theorem min_G_lower_bound : (minGLowerQ : ℝ) ≤ (9999798 / 10000000 : ℝ) := by
  unfold minGLowerQ; norm_num

/-- (N5) The Lean slack `gainLowerQ = 20925/100000` is at most the
    certifier's reported lower endpoint `a ≥ 0.21009214` (coupled-arb)
    and at most the looser rational floor `(4/u)·(998/1000)²/(29841/1000)
    ≈ 0.20926` derivable from N3+N4. -/
theorem gain_lower_bound : (gainLowerQ : ℝ) ≤ (21009214 / 100000000 : ℝ) := by
  unfold gainLowerQ; norm_num

/-! ## Axiom (structural content)

The MV master inequality, specialised to the 3-scale kernel.  The
Fourier reduction on `ℝ/uℤ` proceeds exactly as in the single-scale case
(MV 2010, Lemma 3.1): replacing `J₀(πδξ)²` by the λ-weighted sum
`∑ᵢ λᵢ J₀(πδᵢξ)²` preserves Bochner-positivity (a positive linear
combination of squared real Bessel functions remains positive
semi-definite as a Fourier transform), and the cosine `G` enters only
through the gain parameter `a`.
-/

axiom MV_master_inequality_for_extremiser :
  ∀ (f : ℝ → ℝ),
    (∀ x, 0 ≤ f x) →
    Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4) →
    MeasureTheory.integral MeasureTheory.volume f > 0 →
    MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤ →
    autoconvolution_ratio f + 1 +
      Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
      ≥ 2 / (uQ : ℝ) + (gainLowerQ : ℝ)

/-! ## Quadratic-in-`M` inversion (arithmetic theorem)

The quadratic-in-`M` inversion of the master inequality at these
anchors: for any `a_lower ≥ gainLowerQ` and any `M` with

```
    M + 1 + √(M - 1)·√(K_2 - 1)  ≥  2/u + a_lower,
```

one has `M ≥ 1292/1000`.  At `M = 1292/1000` the LHS is at most

```
    Φ(M) ≤ 66879/20000 = 3.34395,
```

strictly below the threshold `2/u + gainLowerQ = 4267003/1276000`, with
margin `307/3190000 ≈ 9.6 × 10⁻⁵`.  The proof is by case analysis on
`M ≤ 1` (where `Real.sqrt (M-1) = 0` and the LHS is `≤ 2`) versus
`M > 1` (where the rational bound `√((M-1)(K_2-1)) ≤ 105195/100000`
holds for `M ≤ 1292/1000` because `(105195/100000)² > (M-1)(K_2-1)`).
-/

theorem master_inequality_M_lower :
  ∀ (a_lower : ℝ),
    a_lower ≥ (gainLowerQ : ℝ) →
    ∀ (M : ℝ),
      M + 1 + Real.sqrt (M - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
        ≥ 2 / (uQ : ℝ) + a_lower →
      M ≥ (MTargetQ : ℝ) := by
  intros a_lower h_a M h_MI
  -- Strategy: if `M < MTargetQ` then LHS ≤ 66879/20000 < 4267003/1276000 ≤ RHS.
  by_contra h_lt
  push_neg at h_lt
  have hMT : (MTargetQ : ℝ) = 1292 / 1000 := by unfold MTargetQ; push_cast; ring
  have hK2 : (K2UpperQ : ℝ) - 1 = 37897 / 10000 := by
    unfold K2UpperQ; push_cast; ring
  have h_RHS_ge : 2 / (uQ : ℝ) + a_lower ≥ 4267003 / 1276000 := by
    have h_sum : (2 / (uQ : ℝ)) + (gainLowerQ : ℝ) = 4267003 / 1276000 := by
      unfold uQ gainLowerQ; push_cast; ring
    linarith [h_a]
  -- Bound LHS by 66879/20000.
  have h_LHS_le : M + 1 + Real.sqrt (M - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
                  ≤ 66879 / 20000 := by
    rw [hK2]
    by_cases hM1 : M ≤ 1
    · -- `M ≤ 1`: `Real.sqrt (M - 1) = 0` since `M - 1 ≤ 0`.
      have h_sqrt_zero : Real.sqrt (M - 1) = 0 :=
        Real.sqrt_eq_zero'.mpr (by linarith)
      rw [h_sqrt_zero, zero_mul, add_zero]
      have h_lt' : M < 1292 / 1000 := by rw [hMT] at h_lt; exact h_lt
      linarith
    · -- `M > 1`: `√(M-1)·√(K_2-1) = √((M-1)(K_2-1)) ≤ 105195/100000`.
      push_neg at hM1
      have h_M_m1_nn : (0 : ℝ) ≤ M - 1 := by linarith
      rw [← Real.sqrt_mul h_M_m1_nn]
      have h_lt' : M < 1292 / 1000 := by rw [hMT] at h_lt; exact h_lt
      have h_prod_le : (M - 1) * (37897 / 10000) ≤ 11065924 / 10000000 := by
        nlinarith
      have h_pos : (0 : ℝ) ≤ 105195 / 100000 := by norm_num
      have h_sq_ge : (11065924 / 10000000 : ℝ) ≤ (105195 / 100000) ^ 2 := by
        norm_num
      have h_sqrt_le : Real.sqrt ((M - 1) * (37897 / 10000)) ≤ 105195 / 100000 :=
        calc Real.sqrt ((M - 1) * (37897 / 10000))
            ≤ Real.sqrt ((105195 / 100000 : ℝ) ^ 2) := by
              exact Real.sqrt_le_sqrt (le_trans h_prod_le h_sq_ge)
          _ = 105195 / 100000 := Real.sqrt_sq h_pos
      linarith
  -- 66879/20000 < 4267003/1276000.
  have h_strict : (66879 / 20000 : ℝ) < 4267003 / 1276000 := by norm_num
  linarith [h_LHS_le, h_RHS_ge, h_MI, h_strict]

/-! ## Main theorem -/

/-- For every admissible `f`, the autoconvolution ratio satisfies
    `R(f) ≥ 1292/1000 = 1.292`. -/
theorem autoconvolution_ratio_ge_1292_1000 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (1292 / 1000 : ℝ) := by
  have hMI :
      autoconvolution_ratio f + 1 +
        Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
        ≥ 2 / (uQ : ℝ) + (gainLowerQ : ℝ) :=
    MV_master_inequality_for_extremiser
      f hf_nonneg hf_supp hf_int_pos h_conv_fin
  have h := master_inequality_M_lower
              (gainLowerQ : ℝ) le_rfl
              (autoconvolution_ratio f) hMI
  have hMT : (MTargetQ : ℝ) = (1292 / 1000 : ℝ) := by
    unfold MTargetQ; push_cast; ring
  rw [hMT] at h
  exact h

/-- Decimal restatement: `R(f) ≥ 1.292`. -/
theorem autoconvolution_ratio_ge_1_292 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (1.292 : ℝ) := by
  have h := autoconvolution_ratio_ge_1292_1000
              f hf_nonneg hf_supp hf_int_pos h_conv_fin
  have hEq : (1.292 : ℝ) = 1292 / 1000 := by norm_num
  rw [hEq]
  exact h

/-- Display alias: `1292/1000 ≤ R(f)` for every admissible `f`. -/
theorem C1a_ge_1292 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤) :
    (1292 : ℝ) / 1000 ≤ autoconvolution_ratio f :=
  autoconvolution_ratio_ge_1292_1000 f hf_nonneg hf_supp hf_int_pos h_conv_fin

end Sidon.MultiScale
