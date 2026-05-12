/-
Sidon Autocorrelation Project — J_0 Bessel Infrastructure

The arcsine kernel's Fourier coefficients are J_0(π j δ / u)².  The multi-scale
Bochner-positive kernels K_hat_multiscale used in the K26 lower bound work
require:

  J_0(z) := ∑_{k=0}^∞ (-1)^k (z/2)^{2k} / (k!)²       (Taylor series)
  J_0(0) = 1
  J_0(z) ∈ ℝ for z ∈ ℝ
  |J_0(z)| ≤ 1 for all z ∈ ℝ            (max attained at 0)
  J_0(z)² ≥ 0                            (real-square)
  Rigorous rational interval bounds at specific z (z = π·δ for δ∈{0.138,0.055})

As of Lean 4 / mathlib4 commit `f897ebc…` there is NO `Real.BesselJ` or
`besselJ` in the library (only Bessel's *inequality*, an unrelated functional-
analysis statement).  We therefore build the J_0 infrastructure from scratch
using the Taylor series, paired with a small number of named axioms that
package well-known classical facts (uniform |J_0| ≤ 1, partial-sum remainder
estimates).  These axioms can later be replaced with full mathlib proofs if and
when `Mathlib.Analysis.SpecialFunctions.Bessel.Basic` materializes.

Strategy: Option A/C hybrid.
  • Define J_0 by its Taylor series (computable in principle, noncomputable as
    used here).
  • Prove the algebraic facts (J_0(0) = 1, J_0 z ∈ ℝ trivially, J_0(z)² ≥ 0)
    *from the definition*, no axioms.
  • Axiomatize the analytic uniform bound |J_0(z)| ≤ 1 and the
    Taylor-truncation-with-remainder rational interval bound at specific
    rationals.  These are classical and could be discharged by interval
    arithmetic.

Axioms introduced:
  • besselJ0_summable      — Taylor series is summable for every real z
  • besselJ0_abs_le_one    — classical uniform bound |J_0 z| ≤ 1 (Watson §2.5)
  • J0sq_interval_138      — rigorous rational interval at z = π · 0.138
  • J0sq_interval_055      — rigorous rational interval at z = π · 0.055
  • J0sq_interval_ratio    — rigorous rational interval at z = π · 0.138 / 0.638
-/

import Mathlib
import Sidon.Defs

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace Sidon.Bessel

-- ═══════════════════════════════════════════════════════════════════════════════
-- 1. Taylor series definition of J_0
-- ═══════════════════════════════════════════════════════════════════════════════

/-- k-th term of the J_0 Taylor series:
      a_k(z) = (-1)^k · (z/2)^(2k) / (k!)²
    The series converges for all complex z; here we restrict to ℝ. -/
noncomputable def besselJ0_term (k : ℕ) (z : ℝ) : ℝ :=
  ((-1)^k / ((k.factorial : ℝ)^2)) * (z / 2)^(2 * k)

/-- J_0 Bessel function defined by its Taylor series. -/
noncomputable def besselJ0 (z : ℝ) : ℝ :=
  ∑' k : ℕ, besselJ0_term k z

/-- Square of J_0. -/
noncomputable def J0sq (z : ℝ) : ℝ := (besselJ0 z) ^ 2

/-- Truncated Taylor series: ∑_{k < N} a_k(z).  Used for rigorous evaluation
    by combining with an explicit remainder bound. -/
noncomputable def besselJ0_partial (N : ℕ) (z : ℝ) : ℝ :=
  ∑ k ∈ Finset.range N, besselJ0_term k z

-- ═══════════════════════════════════════════════════════════════════════════════
-- 2. Algebraic facts proven directly from the definition
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The 0-th term at z = 0 is 1; all higher terms vanish. -/
theorem besselJ0_term_zero_of_pos {k : ℕ} (hk : 0 < k) : besselJ0_term k 0 = 0 := by
  unfold besselJ0_term
  have h2k : 2 * k ≠ 0 := by omega
  have : (0 / (2 : ℝ))^(2 * k) = 0 := by
    rw [zero_div]; exact zero_pow h2k
  rw [this]; ring

theorem besselJ0_term_zero_zero : besselJ0_term 0 0 = 1 := by
  unfold besselJ0_term
  simp

/-- J_0(0) = 1. -/
theorem besselJ0_zero : besselJ0 0 = 1 := by
  unfold besselJ0
  have hsupp : Function.support (fun k => besselJ0_term k 0) ⊆ {0} := by
    intro k hk
    by_contra hne
    have hk0 : k ≠ 0 := by simpa using hne
    have hpos : 0 < k := Nat.pos_of_ne_zero hk0
    exact hk (besselJ0_term_zero_of_pos hpos)
  rw [tsum_eq_sum (s := ({0} : Finset ℕ)) ?_]
  · simp [besselJ0_term_zero_zero]
  · intro k hk
    simp at hk
    exact besselJ0_term_zero_of_pos (Nat.pos_of_ne_zero hk)

/-- J_0(z)² ≥ 0 (real square). -/
theorem J0sq_nonneg (z : ℝ) : 0 ≤ J0sq z := by
  unfold J0sq; exact sq_nonneg _

/-- J_0(0)² = 1. -/
theorem J0sq_zero : J0sq 0 = 1 := by
  unfold J0sq; rw [besselJ0_zero]; ring

/-- J_0 is real-valued by construction (vacuous in ℝ → ℝ). -/
theorem besselJ0_real (z : ℝ) : ∃ y : ℝ, besselJ0 z = y := ⟨besselJ0 z, rfl⟩

-- ═══════════════════════════════════════════════════════════════════════════════
-- 3. Classical analytic axioms
--    These could be discharged via mathlib's analytic-function or interval-
--    arithmetic infrastructure; we package them as named axioms for now.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The Taylor series for J_0 is summable at every real z.
    Classical: ratio test gives ratio (z/(2(k+1)))² → 0. -/
axiom besselJ0_summable (z : ℝ) : Summable (fun k => besselJ0_term k z)

/-- Classical uniform bound: |J_0(z)| ≤ 1 for all real z.
    Reference: Watson, *Theory of Bessel Functions*, §2.5, eq. (2);
    or Abramowitz–Stegun §9.1.60 (with J_0(0) = 1 attaining the max). -/
axiom besselJ0_abs_le_one (z : ℝ) : |besselJ0 z| ≤ 1

/-- Consequence: J_0(z)² ≤ 1. -/
theorem J0sq_le_one (z : ℝ) : J0sq z ≤ 1 := by
  unfold J0sq
  have h := besselJ0_abs_le_one z
  have : (besselJ0 z)^2 = |besselJ0 z|^2 := by rw [sq_abs]
  rw [this]
  have h0 : 0 ≤ |besselJ0 z| := abs_nonneg _
  nlinarith [h, h0]

/-- Combined: J_0(z)² ∈ [0,1] for every real z. -/
theorem J0sq_bounds (z : ℝ) : 0 ≤ J0sq z ∧ J0sq z ≤ 1 :=
  ⟨J0sq_nonneg z, J0sq_le_one z⟩

-- ═══════════════════════════════════════════════════════════════════════════════
-- 4. Rigorous rational interval bounds at K26 evaluation points
--
-- We need J_0(π·δ)² at specific δ values used in the multi-scale arcsine
-- kernel construction.  Numerically:
--   J_0(π·0.138)² ≈ J_0(0.4335)² ≈ (0.95374)² ≈ 0.90963
--   J_0(π·0.055)² ≈ J_0(0.1728)² ≈ (0.99255)² ≈ 0.98515
--   J_0(π·0.138/0.638)² ≈ J_0(0.6794)² ≈ (0.88823)² ≈ 0.78896
--
-- The intervals below are loose enough to be defended by a 6-term truncation
-- plus standard Taylor-remainder bound; tightening them is an interval
-- arithmetic exercise.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Rigorous rational interval for J_0(π·0.138)².
    Witness value 0.90963; loose envelope [0.90, 0.92]. -/
axiom J0sq_interval_138 :
  (90 : ℝ)/100 ≤ J0sq (Real.pi * (138 : ℝ)/1000) ∧
  J0sq (Real.pi * (138 : ℝ)/1000) ≤ (92 : ℝ)/100

/-- Rigorous rational interval for J_0(π·0.055)².
    Witness value 0.98515; loose envelope [0.98, 0.99]. -/
axiom J0sq_interval_055 :
  (98 : ℝ)/100 ≤ J0sq (Real.pi * (55 : ℝ)/1000) ∧
  J0sq (Real.pi * (55 : ℝ)/1000) ≤ (99 : ℝ)/100

/-- Rigorous rational interval for J_0(π·0.138/0.638)² (single-coordinate
    evaluation at u = 0.638 used in the ratio-driven version of K26).
    Witness value 0.78896; loose envelope [0.78, 0.80]. -/
axiom J0sq_interval_ratio :
  (78 : ℝ)/100 ≤ J0sq (Real.pi * (138 : ℝ)/1000 / ((638 : ℝ)/1000)) ∧
  J0sq (Real.pi * (138 : ℝ)/1000 / ((638 : ℝ)/1000)) ≤ (80 : ℝ)/100

-- ═══════════════════════════════════════════════════════════════════════════════
-- 5. Multi-scale Bochner positivity
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Two-scale arcsine-kernel Fourier coefficient:
      K̂(ξ) = λ · J_0(π δ_1 ξ)² + (1-λ) · J_0(π δ_2 ξ)²,  λ ∈ [0,1].
    By Bochner this is the Fourier coefficient of a positive measure
    (convex combination of two arcsine-kernels). -/
noncomputable def K_hat_multiscale (δ₁ δ₂ lam ξ : ℝ) : ℝ :=
  lam * J0sq (Real.pi * δ₁ * ξ) + (1 - lam) * J0sq (Real.pi * δ₂ * ξ)

/-- K̂ ≥ 0 trivially because both J_0² terms are nonneg and λ, 1-λ ∈ [0,1]. -/
theorem K_hat_multiscale_nonneg (δ₁ δ₂ lam ξ : ℝ)
    (h0 : 0 ≤ lam) (h1 : lam ≤ 1) :
    0 ≤ K_hat_multiscale δ₁ δ₂ lam ξ := by
  unfold K_hat_multiscale
  have hJ1 : 0 ≤ J0sq (Real.pi * δ₁ * ξ) := J0sq_nonneg _
  have hJ2 : 0 ≤ J0sq (Real.pi * δ₂ * ξ) := J0sq_nonneg _
  have hlamc : 0 ≤ 1 - lam := by linarith
  have t1 : 0 ≤ lam * J0sq (Real.pi * δ₁ * ξ) := mul_nonneg h0 hJ1
  have t2 : 0 ≤ (1 - lam) * J0sq (Real.pi * δ₂ * ξ) := mul_nonneg hlamc hJ2
  linarith

/-- K̂ ≤ 1 (since both J_0² ≤ 1 and λ, 1-λ form a convex combination). -/
theorem K_hat_multiscale_le_one (δ₁ δ₂ lam ξ : ℝ)
    (h0 : 0 ≤ lam) (h1 : lam ≤ 1) :
    K_hat_multiscale δ₁ δ₂ lam ξ ≤ 1 := by
  unfold K_hat_multiscale
  have hJ1 : J0sq (Real.pi * δ₁ * ξ) ≤ 1 := J0sq_le_one _
  have hJ2 : J0sq (Real.pi * δ₂ * ξ) ≤ 1 := J0sq_le_one _
  have hJ1n : 0 ≤ J0sq (Real.pi * δ₁ * ξ) := J0sq_nonneg _
  have hJ2n : 0 ≤ J0sq (Real.pi * δ₂ * ξ) := J0sq_nonneg _
  have hlamc : 0 ≤ 1 - lam := by linarith
  have t1 : lam * J0sq (Real.pi * δ₁ * ξ) ≤ lam * 1 := mul_le_mul_of_nonneg_left hJ1 h0
  have t2 : (1 - lam) * J0sq (Real.pi * δ₂ * ξ) ≤ (1 - lam) * 1 :=
    mul_le_mul_of_nonneg_left hJ2 hlamc
  nlinarith [t1, t2, h0, h1, hJ1n, hJ2n]

/-- Three-scale variant occasionally needed. -/
noncomputable def K_hat_threescale (δ₁ δ₂ δ₃ μ₁ μ₂ ξ : ℝ) : ℝ :=
  μ₁ * J0sq (Real.pi * δ₁ * ξ) +
  μ₂ * J0sq (Real.pi * δ₂ * ξ) +
  (1 - μ₁ - μ₂) * J0sq (Real.pi * δ₃ * ξ)

theorem K_hat_threescale_nonneg (δ₁ δ₂ δ₃ μ₁ μ₂ ξ : ℝ)
    (h1 : 0 ≤ μ₁) (h2 : 0 ≤ μ₂) (h3 : μ₁ + μ₂ ≤ 1) :
    0 ≤ K_hat_threescale δ₁ δ₂ δ₃ μ₁ μ₂ ξ := by
  unfold K_hat_threescale
  have hJ1 : 0 ≤ J0sq (Real.pi * δ₁ * ξ) := J0sq_nonneg _
  have hJ2 : 0 ≤ J0sq (Real.pi * δ₂ * ξ) := J0sq_nonneg _
  have hJ3 : 0 ≤ J0sq (Real.pi * δ₃ * ξ) := J0sq_nonneg _
  have h4 : 0 ≤ 1 - μ₁ - μ₂ := by linarith
  have t1 : 0 ≤ μ₁ * J0sq (Real.pi * δ₁ * ξ) := mul_nonneg h1 hJ1
  have t2 : 0 ≤ μ₂ * J0sq (Real.pi * δ₂ * ξ) := mul_nonneg h2 hJ2
  have t3 : 0 ≤ (1 - μ₁ - μ₂) * J0sq (Real.pi * δ₃ * ξ) := mul_nonneg h4 hJ3
  linarith

end Sidon.Bessel

end -- noncomputable section
