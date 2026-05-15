/-
Bessel function `J_0` defined from scratch via its power series.

This file builds the Bessel function `J_0` directly from its absolutely
convergent power series

    J_0(x)  =  ∑_{k=0}^∞ (-1)^k (x/2)^{2k} / (k!)²,

independently of any prior development.  The basic analytic properties
(summability, boundary value, evenness, continuity) are proven here, and
the Fourier-cosine integral representation tying `J_0` to the arcsine
kernel is also established:

    J_0(2π δ ξ)  =  (1/π) ∫_{-δ}^{δ} cos(2π ξ x) / sqrt(δ² - x²) dx  (δ > 0).

The integral representation is decomposed into:

  * `cos_taylor_series`        — Taylor expansion of `Real.cos`.
  * `wallis_sin_pow_even`      — `∫₀^π sin²ᵏθ dθ = π · C(2k,k)/4^k`.
  * `wallis_sin_pow_even_sym`  — same identity on `[-π/2, π/2]`.
  * `arcsine_moment_integral`  — closed form for `∫ x²ᵏ/√(δ²-x²) dx`.
  * `besselJ0_arcsine_fourier` — the final integral representation.

This file contains no `sorry` and introduces no new axioms beyond the
three Lean core axioms (`Classical.choice`, `propext`, `Quot.sound`).
-/

import Mathlib
import Sidon.Defs

set_option linter.mathlibStandardSet false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real

namespace Sidon.Bessel

/-! ## Power-series definition -/

/-- The `k`-th term of the `J_0` power series:
    `T(x, k) = (-1)^k · (x/2)^{2k} / (k!)²`. -/
noncomputable def besselJ0Term (x : ℝ) (k : ℕ) : ℝ :=
  ((-1) ^ k : ℝ) * (x / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2

/-- Bessel `J_0` power series:
    `J_0(x) = ∑_{k=0}^∞ (-1)^k (x/2)^{2k} / (k!)²`. -/
noncomputable def besselJ0 (x : ℝ) : ℝ :=
  ∑' k : ℕ, besselJ0Term x k

/-! ## Summability -/

/-- Auxiliary: factorial-squared bound `1 / (k!)² ≤ 1 / k!`. -/
lemma inv_factorial_sq_le_inv_factorial (k : ℕ) :
    (1 : ℝ) / ((k.factorial : ℝ) ^ 2) ≤ 1 / (k.factorial : ℝ) := by
  have hk : (1 : ℝ) ≤ (k.factorial : ℝ) := by
    exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.factorial_pos k).ne'
  have hk_pos : (0 : ℝ) < (k.factorial : ℝ) := by
    exact_mod_cast Nat.factorial_pos k
  rw [pow_two, one_div, one_div, mul_inv]
  have h_le : (k.factorial : ℝ)⁻¹ ≤ 1 := by
    rw [inv_le_one_iff₀]; right; exact hk
  have hinv_pos : (0 : ℝ) < (k.factorial : ℝ)⁻¹ := inv_pos.mpr hk_pos
  calc (k.factorial : ℝ)⁻¹ * (k.factorial : ℝ)⁻¹
      ≤ 1 * (k.factorial : ℝ)⁻¹ := by
        apply mul_le_mul_of_nonneg_right h_le (le_of_lt hinv_pos)
    _ = (k.factorial : ℝ)⁻¹ := by ring

/-- Absolute value of the `k`-th Bessel term:
    `|T(x, k)| = (x/2)^{2k} / (k!)²`. -/
lemma abs_besselJ0Term (x : ℝ) (k : ℕ) :
    |besselJ0Term x k| = (x / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2 := by
  unfold besselJ0Term
  rw [abs_div, abs_mul]
  have h1 : |((-1 : ℝ) ^ k)| = 1 := by
    rw [abs_pow]; simp
  have h2 : |((k.factorial : ℝ) ^ 2)| = (k.factorial : ℝ) ^ 2 := by
    rw [abs_pow, abs_of_nonneg]
    exact Nat.cast_nonneg _
  have h3 : |((x / 2) ^ (2 * k))| = (x / 2) ^ (2 * k) := by
    rw [show 2 * k = 2 * k from rfl, pow_mul]
    rw [abs_pow, abs_pow, sq_abs]
  rw [h1, h2, h3, one_mul]

/-- The Bessel `J_0` series is summable for every real `x`. -/
theorem besselJ0_summable (x : ℝ) : Summable (besselJ0Term x) := by
  -- Dominate by the exponential series: |T(x,k)| ≤ ((x/2)²)^k / k!.
  apply Summable.of_norm_bounded
    (g := fun k : ℕ => ((x / 2) ^ 2) ^ k / (k.factorial : ℝ))
    ?_ ?_
  · -- The dominating series ∑ y^k/k! is summable.
    exact Real.summable_pow_div_factorial _
  · intro k
    rw [Real.norm_eq_abs, abs_besselJ0Term]
    -- (x/2)^{2k} = ((x/2)²)^k
    have hk : (x / 2) ^ (2 * k) = ((x / 2) ^ 2) ^ k := by
      rw [pow_mul]
    rw [hk]
    -- ((x/2)²)^k / (k!)² ≤ ((x/2)²)^k / k!
    have h_kfact_pos : (0 : ℝ) < (k.factorial : ℝ) := by
      exact_mod_cast Nat.factorial_pos k
    have h_kfact_sq_pos : (0 : ℝ) < (k.factorial : ℝ) ^ 2 := by positivity
    have h_num_nn : (0 : ℝ) ≤ ((x / 2) ^ 2) ^ k := by positivity
    rw [div_le_div_iff₀ h_kfact_sq_pos h_kfact_pos]
    have h_kfact_ge_one : (1 : ℝ) ≤ (k.factorial : ℝ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.factorial_pos k).ne'
    calc ((x / 2) ^ 2) ^ k * (k.factorial : ℝ)
        ≤ ((x / 2) ^ 2) ^ k * (k.factorial : ℝ) ^ 2 := by
          apply mul_le_mul_of_nonneg_left _ h_num_nn
          rw [pow_two]
          calc (k.factorial : ℝ)
              = (k.factorial : ℝ) * 1 := by ring
            _ ≤ (k.factorial : ℝ) * (k.factorial : ℝ) := by
                apply mul_le_mul_of_nonneg_left h_kfact_ge_one (le_of_lt h_kfact_pos)
      _ = ((x / 2) ^ 2) ^ k * (k.factorial : ℝ) ^ 2 := rfl

/-! ## Boundary value `J_0(0) = 1` -/

/-- For `k ≥ 1`, the `k`-th Bessel term at `x = 0` vanishes. -/
lemma besselJ0Term_zero_succ (k : ℕ) : besselJ0Term 0 (k + 1) = 0 := by
  unfold besselJ0Term
  have h2k : 2 * (k + 1) = 2 * k + 2 := by ring
  rw [h2k]
  have : (0 / 2 : ℝ) ^ (2 * k + 2) = 0 := by
    rw [zero_div]
    rw [show 2 * k + 2 = (2 * k + 1) + 1 from rfl, pow_succ]
    ring
  rw [this]; ring

/-- The 0-th Bessel term at `x = 0` is `1`. -/
lemma besselJ0Term_zero_zero : besselJ0Term 0 0 = 1 := by
  unfold besselJ0Term
  simp

/-- `J_0(0) = 1`. -/
theorem besselJ0_zero : besselJ0 0 = 1 := by
  unfold besselJ0
  rw [tsum_eq_single 0]
  · exact besselJ0Term_zero_zero
  · intro k hk
    obtain ⟨j, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hk
    exact besselJ0Term_zero_succ j

/-! ## Evenness `J_0(-x) = J_0(x)` -/

/-- Each Bessel term is invariant under `x ↦ -x`. -/
lemma besselJ0Term_neg (x : ℝ) (k : ℕ) : besselJ0Term (-x) k = besselJ0Term x k := by
  unfold besselJ0Term
  have h : (-x / 2) ^ (2 * k) = (x / 2) ^ (2 * k) := by
    have : -x / 2 = -(x / 2) := by ring
    rw [this]
    rw [show 2 * k = 2 * k from rfl, pow_mul]
    rw [show 2 * k = 2 * k from rfl, pow_mul (x/2) 2 k]
    congr 1
    ring
  rw [h]

/-- `J_0` is even: `J_0(-x) = J_0(x)`. -/
theorem besselJ0_even (x : ℝ) : besselJ0 (-x) = besselJ0 x := by
  unfold besselJ0
  apply tsum_congr
  intro k
  exact besselJ0Term_neg x k

/-! ## Continuity -/

/-- Term-by-term continuity: each Bessel term `T(·, k)` is continuous. -/
lemma besselJ0Term_continuous (k : ℕ) : Continuous (fun x => besselJ0Term x k) := by
  unfold besselJ0Term
  exact (continuous_const.mul ((continuous_id.div_const 2).pow (2*k))).div_const _

/-- Pointwise summability bound: on `‖x‖ ≤ R`, the `k`-th term is bounded
    by `(R/2)^{2k}/(k!)²`, which is summable in `k`. -/
lemma besselJ0Term_norm_bound {R : ℝ} (hR : 0 ≤ R) (x : ℝ) (hx : |x| ≤ R) (k : ℕ) :
    ‖besselJ0Term x k‖ ≤ (R / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2 := by
  rw [Real.norm_eq_abs, abs_besselJ0Term]
  have h_kfact_sq_pos : (0 : ℝ) < (k.factorial : ℝ) ^ 2 := by
    have : (0 : ℝ) < (k.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos k
    positivity
  rw [div_le_div_iff₀ h_kfact_sq_pos h_kfact_sq_pos]
  apply mul_le_mul_of_nonneg_right _ (le_of_lt h_kfact_sq_pos)
  -- Need: (x/2)^{2k} ≤ (R/2)^{2k}
  rw [show 2 * k = 2 * k from rfl, pow_mul, pow_mul]
  apply pow_le_pow_left₀ _ _ k
  · positivity
  · -- (x/2)² ≤ (R/2)²
    rw [show ((x/2)^2 : ℝ) = (x/2)*(x/2) from by ring]
    rw [show ((R/2)^2 : ℝ) = (R/2)*(R/2) from by ring]
    have habs_x : |x/2| ≤ R/2 := by
      rw [abs_div]
      simp
      linarith
    have habs_x_nn : 0 ≤ |x/2| := abs_nonneg _
    have : (x/2)*(x/2) ≤ |x/2| * |x/2| := by
      have := abs_mul_self (x/2)
      nlinarith [abs_nonneg (x/2), sq_abs (x/2)]
    calc (x/2)*(x/2) ≤ |x/2| * |x/2| := this
      _ ≤ (R/2) * (R/2) := by
        apply mul_le_mul habs_x habs_x habs_x_nn (by linarith)

/-- The summable dominating sequence used for the Weierstrass `M`-test. -/
lemma summable_besselJ0_majorant (R : ℝ) :
    Summable (fun k : ℕ => (R / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2) := by
  -- Same dominate-by-exp trick as in `besselJ0_summable`.
  refine Summable.of_nonneg_of_le
    (g := fun k : ℕ => (R / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2)
    (f := fun k : ℕ => ((R / 2) ^ 2) ^ k / (k.factorial : ℝ))
    ?_ ?_ (Real.summable_pow_div_factorial _)
  · -- 0 ≤ (R/2)^(2k)/(k!)^2
    intro k
    have h_kfact_sq_pos : (0 : ℝ) < (k.factorial : ℝ) ^ 2 := by
      have : (0 : ℝ) < (k.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos k
      positivity
    refine div_nonneg ?_ (le_of_lt h_kfact_sq_pos)
    rw [show 2 * k = 2 * k from rfl, pow_mul]
    positivity
  · -- (R/2)^(2k)/(k!)^2 ≤ ((R/2)^2)^k/k!
    intro k
    show (R / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2 ≤
         ((R / 2) ^ 2) ^ k / (k.factorial : ℝ)
    have hk : (R / 2) ^ (2 * k) = ((R / 2) ^ 2) ^ k := by rw [pow_mul]
    rw [hk]
    have h_kfact_pos : (0 : ℝ) < (k.factorial : ℝ) := by
      exact_mod_cast Nat.factorial_pos k
    have h_kfact_sq_pos : (0 : ℝ) < (k.factorial : ℝ) ^ 2 := by positivity
    have h_num_nn : (0 : ℝ) ≤ ((R / 2) ^ 2) ^ k := by positivity
    rw [div_le_div_iff₀ h_kfact_sq_pos h_kfact_pos]
    have h_kfact_ge_one : (1 : ℝ) ≤ (k.factorial : ℝ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.factorial_pos k).ne'
    calc ((R / 2) ^ 2) ^ k * (k.factorial : ℝ)
        ≤ ((R / 2) ^ 2) ^ k * (k.factorial : ℝ) ^ 2 := by
          apply mul_le_mul_of_nonneg_left _ h_num_nn
          rw [pow_two]
          calc (k.factorial : ℝ)
              = (k.factorial : ℝ) * 1 := by ring
            _ ≤ (k.factorial : ℝ) * (k.factorial : ℝ) := by
                apply mul_le_mul_of_nonneg_left h_kfact_ge_one (le_of_lt h_kfact_pos)

/-- `besselJ0` is continuous on every closed ball.  This is the
    Weierstrass `M`-test: the partial sums converge uniformly on `[-R, R]`
    against the summable majorant `(R/2)^{2k}/(k!)²`. -/
theorem besselJ0_continuousOn_closed_ball (R : ℝ) :
    ContinuousOn besselJ0 (Set.Icc (-R) R) := by
  by_cases hR : 0 ≤ R
  · -- Uniform-on-compacts convergence ⇒ continuity of limit.
    have h_unif : TendstoUniformlyOn
        (fun N x => ∑ k ∈ Finset.range N, besselJ0Term x k)
        besselJ0 Filter.atTop (Set.Icc (-R) R) := by
      have h := tendstoUniformlyOn_tsum_nat
        (u := fun k : ℕ => (R / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2)
        (summable_besselJ0_majorant R)
        (s := Set.Icc (-R) R)
        (f := fun k x => besselJ0Term x k)
        (by
          intro k x hx
          have hxR : |x| ≤ R := by
            rcases hx with ⟨h1, h2⟩
            rw [abs_le]; exact ⟨h1, h2⟩
          exact besselJ0Term_norm_bound hR x hxR k)
      -- `besselJ0` is defined as `∑' k, besselJ0Term x k`.
      have h_eq : (fun x => ∑' k : ℕ, besselJ0Term x k) = besselJ0 := rfl
      rw [← h_eq]
      exact h
    apply h_unif.continuousOn
    apply Filter.Frequently.of_forall
    intro n
    apply continuousOn_finset_sum
    intro k _
    exact (besselJ0Term_continuous k).continuousOn
  · -- If R < 0 the interval is empty, so continuity-on-it is vacuous.
    push_neg at hR
    have h_empty : Set.Icc (-R) R = ∅ := by
      apply Set.Icc_eq_empty
      linarith
    rw [h_empty]
    exact continuousOn_empty _

/-- `besselJ0` is continuous on all of `ℝ`. -/
theorem besselJ0_continuous : Continuous besselJ0 := by
  rw [continuous_iff_continuousAt]
  intro x
  -- It suffices to show continuity on `[-(|x|+1), |x|+1]`, a closed
  -- ball containing `x` in its interior.
  have h_in : x ∈ Set.Icc (-(|x| + 1)) (|x| + 1) := by
    refine ⟨?_, ?_⟩
    · have := neg_abs_le x; linarith
    · have := le_abs_self x; linarith
  have h_cts := besselJ0_continuousOn_closed_ball (|x| + 1)
  -- ContinuousOn on an interval that contains `x` in its interior gives `ContinuousAt`.
  have h_interior : x ∈ interior (Set.Icc (-(|x| + 1)) (|x| + 1)) := by
    rw [interior_Icc]
    refine ⟨?_, ?_⟩
    · have := neg_abs_le x
      have h_abs_pos : -(|x| + 1) < -|x| := by linarith
      linarith [this, abs_nonneg x]
    · have := le_abs_self x; linarith
  refine h_cts.continuousAt ?_
  exact Filter.mem_of_superset (IsOpen.mem_nhds isOpen_interior h_interior) interior_subset

/-! ## Fourier-cosine integral representation

The Fourier-cosine identity

    J_0(2π δ ξ)  =  (1/π) · ∫_{-δ}^{δ} cos(2π ξ x) / sqrt(δ² - x²) dx

proceeds by:

1.  Expand `cos(2π ξ x)` as its Taylor series in `x` (`cos_taylor_series`).
2.  Substitute `x = δ sin θ` (`arcsine_moment_integral`).
3.  Evaluate `∫ sin²ᵏθ dθ` (`wallis_sin_pow_even_sym`).
4.  Match against the `J_0` series at `2π δ ξ`.
-/

/-- Closed-form factorial identity used inside `besselJ0_arcsine_fourier`:
    `C(2k, k) / 4^k = (2k)! / ((k!)² · 4^k)`. -/
lemma central_binomial_factorial_form (k : ℕ) :
    ((Nat.choose (2 * k) k : ℝ) / (4 : ℝ) ^ k) =
      (Nat.factorial (2 * k) : ℝ) /
        ((k.factorial : ℝ) ^ 2 * (4 : ℝ) ^ k) := by
  have h_fact_pos : (0 : ℝ) < (k.factorial : ℝ) := by
    exact_mod_cast Nat.factorial_pos k
  have h_choose_eq :
      (Nat.choose (2 * k) k : ℝ) * ((k.factorial : ℝ) ^ 2) =
        (Nat.factorial (2 * k) : ℝ) := by
    have h_choose_nat :
        (2 * k).choose k * k.factorial * (2 * k - k).factorial =
          (2 * k).factorial := Nat.choose_mul_factorial_mul_factorial (by omega : k ≤ 2 * k)
    have h_sub : 2 * k - k = k := by omega
    rw [h_sub] at h_choose_nat
    have := congrArg (fun n : ℕ => (n : ℝ)) h_choose_nat
    simp only at this
    push_cast at this
    nlinarith [this]
  have h_pow_4_pos : (0 : ℝ) < (4 : ℝ) ^ k := by positivity
  have h_sq_pos : (0 : ℝ) < (k.factorial : ℝ) ^ 2 := by positivity
  field_simp
  linarith [h_choose_eq]

/-- Taylor series for `Real.cos`: `cos y = ∑_k (-1)^k y^(2k)/(2k)!`. -/
theorem cos_taylor_series (y : ℝ) :
    Real.cos y = ∑' k : ℕ, ((-1) ^ k : ℝ) * y ^ (2 * k) / ((2 * k).factorial : ℝ) :=
  Real.cos_eq_tsum y

/-! ### Wallis-type integral identities -/

/-- The product `∏_{i<k} (2i+1)/(2i+2)` equals `C(2k,k)/4^k`. -/
lemma prod_wallis_eq_choose (k : ℕ) :
    (∏ i ∈ Finset.range k, (2 * (i : ℝ) + 1) / (2 * i + 2)) =
      (Nat.choose (2 * k) k : ℝ) / (4 : ℝ) ^ k := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Finset.prod_range_succ, ih]
    have hk_fact_pos : (0 : ℝ) < (k.factorial : ℝ) := by
      exact_mod_cast Nat.factorial_pos k
    -- Use central_binomial_factorial_form to switch both sides to factorial form.
    rw [central_binomial_factorial_form k, central_binomial_factorial_form (k + 1)]
    have h_2k1_fact : ((2 * (k + 1)).factorial : ℝ) =
        ((2 * k + 1) * (2 * k + 2)) * ((2 * k).factorial : ℝ) := by
      have h1 : 2 * (k + 1) = 2 * k + 2 := by ring
      rw [h1]
      have h2 : (2 * k + 2).factorial = (2 * k + 2) * (2 * k + 1) * (2 * k).factorial := by
        rw [Nat.factorial_succ, Nat.factorial_succ]; ring
      have h2' : ((2 * k + 2).factorial : ℝ) =
          ((2 * k + 2) * (2 * k + 1) * (2 * k).factorial : ℕ) := by exact_mod_cast h2
      rw [h2']
      push_cast; ring
    have h_kp_fact : ((k + 1).factorial : ℝ) = (k + 1) * (k.factorial : ℝ) := by
      have : (k + 1).factorial = (k + 1) * k.factorial := Nat.factorial_succ k
      exact_mod_cast this
    have h_4_kp : (4 : ℝ) ^ (k + 1) = 4 * 4 ^ k := by rw [pow_succ]; ring
    rw [h_2k1_fact, h_kp_fact, h_4_kp]
    field_simp
    ring

/-- Wallis: `∫_0^π sin²ᵏθ dθ = π · C(2k,k)/4^k`. -/
theorem wallis_sin_pow_even (k : ℕ) :
    (∫ θ in (0 : ℝ)..Real.pi, (Real.sin θ) ^ (2 * k)) =
      Real.pi * (Nat.choose (2 * k) k : ℝ) / (4 : ℝ) ^ k := by
  rw [integral_sin_pow_even, prod_wallis_eq_choose]
  ring

/-- Auxiliary: `∫_{-π/2}^{0} sin²ᵏθ dθ = ∫_0^{π/2} sin²ᵏθ dθ`
    (by substitution `θ ↦ -θ` and evenness of `sin²ᵏ`). -/
lemma integral_sin_pow_even_neg_half (k : ℕ) :
    (∫ θ in (-(Real.pi/2))..(0 : ℝ), (Real.sin θ) ^ (2 * k)) =
      ∫ θ in (0 : ℝ)..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
  have h := intervalIntegral.integral_comp_neg
    (a := 0) (b := Real.pi / 2) (f := fun θ => (Real.sin θ) ^ (2 * k))
  -- h : ∫ x in 0..π/2, sin(-x)^(2k) = ∫ x in -(π/2)..-0, sin x ^(2k)
  have h_neg_zero : -(0 : ℝ) = 0 := neg_zero
  rw [h_neg_zero] at h
  -- now h : ∫ x in 0..π/2, sin(-x)^(2k) = ∫ x in -(π/2)..0, sin x ^(2k)
  rw [← h]
  -- show: ∫ θ in 0..π/2, sin(-θ)^(2k) = ∫ θ in 0..π/2, sin θ^(2k)
  apply intervalIntegral.integral_congr
  intro u _
  show (Real.sin (-u)) ^ (2 * k) = (Real.sin u) ^ (2 * k)
  rw [Real.sin_neg]
  rw [show 2 * k = 2 * k from rfl, neg_pow, pow_mul, neg_one_sq, one_pow, one_mul]

/-- `∫_{-π/2}^{π/2} sin²ᵏθ dθ = 2 ∫_0^{π/2} sin²ᵏθ dθ`. -/
lemma integral_sin_pow_even_sym_half (k : ℕ) :
    (∫ θ in (-(Real.pi/2))..(Real.pi/2), (Real.sin θ) ^ (2 * k)) =
      2 * ∫ θ in (0 : ℝ)..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
  have h_split : (∫ θ in (-(Real.pi/2))..(Real.pi/2), (Real.sin θ) ^ (2 * k)) =
      (∫ θ in (-(Real.pi/2))..(0 : ℝ), (Real.sin θ) ^ (2 * k)) +
        ∫ θ in (0 : ℝ)..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
    rw [← intervalIntegral.integral_add_adjacent_intervals
      (a := -(Real.pi/2)) (b := 0) (c := Real.pi/2)]
    all_goals exact (Real.continuous_sin.pow _).intervalIntegrable _ _
  rw [h_split, integral_sin_pow_even_neg_half]
  ring

/-- Auxiliary: `∫_{π/2}^{π} sin²ᵏθ dθ = ∫_0^{π/2} sin²ᵏθ dθ`
    (by substitution `θ ↦ π - θ` and `sin(π - θ) = sin θ`). -/
lemma integral_sin_pow_even_upper_half (k : ℕ) :
    (∫ θ in (Real.pi/2)..Real.pi, (Real.sin θ) ^ (2 * k)) =
      ∫ θ in (0 : ℝ)..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
  have h := intervalIntegral.integral_comp_sub_left
    (a := 0) (b := Real.pi/2) (d := Real.pi)
    (f := fun θ => (Real.sin θ) ^ (2 * k))
  -- h : ∫ x in 0..π/2, sin(π - x)^(2k) = ∫ x in π - π/2..π - 0, sin x ^(2k)
  have hsimp1 : Real.pi - 0 = Real.pi := sub_zero _
  have hsimp2 : Real.pi - Real.pi / 2 = Real.pi / 2 := by ring
  rw [hsimp1, hsimp2] at h
  -- h : ∫ x in 0..π/2, sin(π - x)^(2k) = ∫ x in π/2..π, sin x ^(2k)
  rw [← h]
  apply intervalIntegral.integral_congr
  intro u _
  show (Real.sin (Real.pi - u)) ^ (2 * k) = (Real.sin u) ^ (2 * k)
  rw [Real.sin_pi_sub]

/-- `∫_0^π sin²ᵏθ dθ = 2 ∫_0^{π/2} sin²ᵏθ dθ`. -/
lemma integral_sin_pow_even_full_half (k : ℕ) :
    (∫ θ in (0 : ℝ)..Real.pi, (Real.sin θ) ^ (2 * k)) =
      2 * ∫ θ in (0 : ℝ)..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
  have h_split : (∫ θ in (0 : ℝ)..Real.pi, (Real.sin θ) ^ (2 * k)) =
      (∫ θ in (0 : ℝ)..(Real.pi/2), (Real.sin θ) ^ (2 * k)) +
        ∫ θ in (Real.pi/2)..Real.pi, (Real.sin θ) ^ (2 * k) := by
    rw [← intervalIntegral.integral_add_adjacent_intervals
      (a := 0) (b := Real.pi/2) (c := Real.pi)]
    all_goals exact (Real.continuous_sin.pow _).intervalIntegrable _ _
  rw [h_split, integral_sin_pow_even_upper_half]
  ring

/-- Symmetric form: `∫_{-π/2}^{π/2} sin²ᵏθ dθ = π · C(2k,k)/4^k`. -/
theorem wallis_sin_pow_even_sym (k : ℕ) :
    (∫ θ in (-(Real.pi/2))..(Real.pi/2), (Real.sin θ) ^ (2 * k)) =
      Real.pi * (Nat.choose (2 * k) k : ℝ) / (4 : ℝ) ^ k := by
  rw [integral_sin_pow_even_sym_half]
  have h1 := wallis_sin_pow_even k
  rw [integral_sin_pow_even_full_half] at h1
  linarith

/-! ### Arcsine moment integral via substitution `x = δ sin θ`. -/

/-- `δ · sin '' Icc (-π/2) (π/2) = Icc (-δ) δ` for `δ > 0`. -/
lemma image_delta_sin_Icc {δ : ℝ} (hδ : 0 < δ) :
    (fun θ => δ * Real.sin θ) '' Set.Icc (-(Real.pi/2)) (Real.pi/2) =
      Set.Icc (-δ) δ := by
  have h_sin_image : Real.sin '' Set.Icc (-(Real.pi / 2)) (Real.pi / 2) =
      Set.Icc (-1) 1 := Real.bijOn_sin.image_eq
  ext y
  simp only [Set.mem_image, Set.mem_Icc]
  constructor
  · rintro ⟨θ, hθ, rfl⟩
    have h_sinθ_mem : Real.sin θ ∈ Set.Icc (-1 : ℝ) 1 := Real.sin_mem_Icc θ
    obtain ⟨h1, h2⟩ := h_sinθ_mem
    constructor
    · -- -δ ≤ δ * sin θ. Since sin θ ≥ -1 and δ > 0, δ * sin θ ≥ -δ.
      nlinarith
    · -- δ * sin θ ≤ δ.
      nlinarith
  · rintro ⟨h1, h2⟩
    -- y ∈ [-δ, δ], find θ such that δ sin θ = y, i.e., sin θ = y/δ.
    have h_y_div : -1 ≤ y/δ ∧ y/δ ≤ 1 := by
      refine ⟨?_, ?_⟩
      · rw [le_div_iff₀ hδ]; linarith
      · rw [div_le_iff₀ hδ]; linarith
    have h_y_div_mem : y/δ ∈ Set.Icc (-1 : ℝ) 1 := h_y_div
    rw [← h_sin_image] at h_y_div_mem
    obtain ⟨θ, hθ, hsinθ⟩ := h_y_div_mem
    refine ⟨θ, hθ, ?_⟩
    field_simp at hsinθ
    rw [← hsinθ]
    ring

/-- `δ · sin` is strictly monotone on `Ioo (-π/2) (π/2)`, for `δ > 0`. -/
lemma strictMonoOn_delta_sin {δ : ℝ} (hδ : 0 < δ) :
    StrictMonoOn (fun θ => δ * Real.sin θ) (Set.Ioo (-(Real.pi/2)) (Real.pi/2)) := by
  intro a ha b hb hab
  have h_sin_lt : Real.sin a < Real.sin b := by
    apply Real.strictMonoOn_sin
    · exact ⟨le_of_lt ha.1, le_of_lt ha.2⟩
    · exact ⟨le_of_lt hb.1, le_of_lt hb.2⟩
    · exact hab
  exact mul_lt_mul_of_pos_left h_sin_lt hδ

/-- `δ · sin` is monotone on `Ioo (-π/2) (π/2)`, for `δ > 0`. -/
lemma monotoneOn_delta_sin {δ : ℝ} (hδ : 0 < δ) :
    MonotoneOn (fun θ => δ * Real.sin θ) (Set.Ioo (-(Real.pi/2)) (Real.pi/2)) :=
  (strictMonoOn_delta_sin hδ).monotoneOn

/-- `δ · sin '' Ioo (-π/2) (π/2) = Ioo (-δ) δ` for `δ > 0`. -/
lemma image_delta_sin_Ioo {δ : ℝ} (hδ : 0 < δ) :
    (fun θ => δ * Real.sin θ) '' Set.Ioo (-(Real.pi/2)) (Real.pi/2) =
      Set.Ioo (-δ) δ := by
  ext y
  simp only [Set.mem_image, Set.mem_Ioo]
  constructor
  · rintro ⟨θ, hθ, rfl⟩
    -- θ ∈ (-π/2, π/2). Then sin θ is in (-1, 1) by strict monotonicity.
    have h_sin_lt : Real.sin θ < 1 := by
      have h := Real.strictMonoOn_sin
        ⟨le_of_lt hθ.1, le_of_lt hθ.2⟩
        (by refine ⟨?_, le_refl _⟩; linarith [Real.pi_pos] :
          (Real.pi/2 : ℝ) ∈ Set.Icc (-(Real.pi/2)) (Real.pi/2)) hθ.2
      rw [Real.sin_pi_div_two] at h
      exact h
    have h_sin_gt : -1 < Real.sin θ := by
      have h := Real.strictMonoOn_sin
        (by refine ⟨le_refl _, ?_⟩; linarith [Real.pi_pos] :
          (-(Real.pi/2) : ℝ) ∈ Set.Icc (-(Real.pi/2)) (Real.pi/2))
        ⟨le_of_lt hθ.1, le_of_lt hθ.2⟩ hθ.1
      rw [Real.sin_neg, Real.sin_pi_div_two] at h
      exact h
    refine ⟨?_, ?_⟩
    · -- -δ < δ * sin θ.  Equivalent to -1 < sin θ since δ > 0.
      nlinarith
    · -- δ * sin θ < δ.
      nlinarith
  · rintro ⟨h1, h2⟩
    -- y ∈ (-δ, δ), find θ with δ sin θ = y, i.e., sin θ = y/δ.
    have h_y_div_lt1 : y/δ < 1 := by
      rw [div_lt_iff₀ hδ]; linarith
    have h_y_div_gt_neg : -1 < y/δ := by
      rw [lt_div_iff₀ hδ]; linarith
    -- Use arcsine to recover θ.
    refine ⟨Real.arcsin (y/δ), ?_, ?_⟩
    · refine ⟨?_, ?_⟩
      · -- arcsin (y/δ) > -π/2
        have h_eq : Real.arcsin (-1) = -(Real.pi/2) := Real.arcsin_neg_one
        rw [← h_eq]
        exact Real.arcsin_lt_arcsin (by norm_num : (-1 : ℝ) ≤ -1) h_y_div_gt_neg
          (le_of_lt h_y_div_lt1)
      · -- arcsin (y/δ) < π/2
        have h_eq : Real.arcsin 1 = Real.pi/2 := Real.arcsin_one
        rw [← h_eq]
        exact Real.arcsin_lt_arcsin (le_of_lt h_y_div_gt_neg) h_y_div_lt1
          (by norm_num : (1 : ℝ) ≤ 1)
    · have h_sin : Real.sin (Real.arcsin (y/δ)) = y/δ :=
        Real.sin_arcsin (le_of_lt h_y_div_gt_neg) (le_of_lt h_y_div_lt1)
      rw [h_sin]; field_simp

/-- HasDerivWithinAt for `θ ↦ δ sin θ` on `Ioo (-π/2) (π/2)`. -/
lemma hasDerivWithinAt_delta_sin {δ : ℝ} (θ : ℝ)
    (_ : θ ∈ Set.Ioo (-(Real.pi/2)) (Real.pi/2)) :
    HasDerivWithinAt (fun u => δ * Real.sin u) (δ * Real.cos θ)
      (Set.Ioo (-(Real.pi/2)) (Real.pi/2)) θ := by
  have h : HasDerivAt (fun u => δ * Real.sin u) (δ * Real.cos θ) θ := by
    have h_sin : HasDerivAt Real.sin (Real.cos θ) θ := Real.hasDerivAt_sin θ
    exact h_sin.const_mul δ
  exact h.hasDerivWithinAt

/-- The integrand simplification: on the open interval `(-π/2, π/2)`,
    where `cos θ > 0`, we have

    `δ cos θ · ((δ sin θ)^{2k} / sqrt(δ² - (δ sin θ)²)) = δ^{2k} · sin^{2k}θ`. -/
lemma arcsine_composite_simplify {δ : ℝ} (hδ : 0 < δ) (k : ℕ) {θ : ℝ}
    (hθ : θ ∈ Set.Ioo (-(Real.pi/2)) (Real.pi/2)) :
    δ * Real.cos θ * ((δ * Real.sin θ) ^ (2 * k) /
      Real.sqrt (δ^2 - (δ * Real.sin θ)^2)) =
    δ ^ (2 * k) * (Real.sin θ) ^ (2 * k) := by
  have h_cos_pos : 0 < Real.cos θ := Real.cos_pos_of_mem_Ioo hθ
  have h_sin_sq : (Real.sin θ) ^ 2 + (Real.cos θ) ^ 2 = 1 := Real.sin_sq_add_cos_sq θ
  have h_cos_sq : (Real.cos θ) ^ 2 = 1 - (Real.sin θ) ^ 2 := by linarith
  have h_factor : δ ^ 2 - (δ * Real.sin θ) ^ 2 = δ ^ 2 * (Real.cos θ) ^ 2 := by
    rw [h_cos_sq]; ring
  rw [h_factor]
  have h_sqrt : Real.sqrt (δ ^ 2 * (Real.cos θ) ^ 2) = δ * Real.cos θ := by
    rw [show δ ^ 2 * (Real.cos θ) ^ 2 = (δ * Real.cos θ) ^ 2 from by ring]
    exact Real.sqrt_sq (by positivity : 0 ≤ δ * Real.cos θ)
  rw [h_sqrt]
  have h_mul_sin : (δ * Real.sin θ) ^ (2 * k) = δ ^ (2 * k) * (Real.sin θ) ^ (2 * k) := by
    rw [mul_pow]
  rw [h_mul_sin]
  have h_prod_pos : 0 < δ * Real.cos θ := mul_pos hδ h_cos_pos
  field_simp

/-- The arcsine moment integral, stated over `Set.Ioo`.  This is the
    main step in the Bessel integral representation. -/
theorem arcsine_moment_integral {δ : ℝ} (hδ : 0 < δ) (k : ℕ) :
    (∫ x in Set.Ioo (-δ) δ, x ^ (2 * k) / Real.sqrt (δ ^ 2 - x ^ 2)) =
      Real.pi * δ ^ (2 * k) * (Nat.choose (2 * k) k : ℝ) / (4 : ℝ) ^ k := by
  -- Step 1: Apply change-of-variable `x = δ sin θ` via
  -- `integral_image_eq_integral_deriv_smul_of_monotoneOn`.
  set s : Set ℝ := Set.Ioo (-(Real.pi/2)) (Real.pi/2)
  set f : ℝ → ℝ := fun θ => δ * Real.sin θ with hf_def
  set f' : ℝ → ℝ := fun θ => δ * Real.cos θ with hf'_def
  set g : ℝ → ℝ := fun x => x ^ (2 * k) / Real.sqrt (δ ^ 2 - x ^ 2) with hg_def
  have hs : MeasurableSet s := measurableSet_Ioo
  have hf' : ∀ θ ∈ s, HasDerivWithinAt f (f' θ) s θ :=
    fun θ hθ => hasDerivWithinAt_delta_sin θ hθ
  have hf_mono : MonotoneOn f s := monotoneOn_delta_sin hδ
  have h_change :=
    MeasureTheory.integral_image_eq_integral_deriv_smul_of_monotoneOn hs hf' hf_mono g
  -- h_change : ∫ x in f '' s, g x = ∫ x in s, f' x • g (f x)
  have h_img : f '' s = Set.Ioo (-δ) δ := image_delta_sin_Ioo hδ
  rw [h_img] at h_change
  -- Now: ∫ x in Set.Ioo (-δ) δ, g x = ∫ θ in s, f' θ • g (f θ)
  rw [h_change]
  -- Step 2: Simplify the integrand on `s` to `δ^{2k} · sin^{2k}θ`.
  have h_simplify :
      (∫ θ in s, f' θ • g (f θ)) =
      ∫ θ in s, δ ^ (2 * k) * (Real.sin θ) ^ (2 * k) := by
    apply MeasureTheory.setIntegral_congr_fun hs
    intro θ hθ
    show f' θ * g (f θ) = δ ^ (2 * k) * (Real.sin θ) ^ (2 * k)
    exact arcsine_composite_simplify hδ k hθ
  rw [h_simplify]
  -- Step 3: Pull out the constant and rewrite as intervalIntegral.
  rw [MeasureTheory.integral_const_mul]
  -- Now: δ^(2k) * ∫ θ in s, sin^(2k) θ
  -- s = Ioo (-π/2) (π/2). Convert to intervalIntegral on [-π/2, π/2].
  have h_int_eq :
      (∫ θ in s, (Real.sin θ) ^ (2 * k)) =
        ∫ θ in (-(Real.pi/2))..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
    rw [intervalIntegral.integral_of_le
      (by linarith [Real.pi_pos] : -(Real.pi/2) ≤ Real.pi/2)]
    show (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), (Real.sin θ) ^ (2 * k)) = _
    rw [← MeasureTheory.integral_Ioc_eq_integral_Ioo]
  rw [h_int_eq, wallis_sin_pow_even_sym]
  ring

/-! ### Bessel J_0 — Fourier-cosine integral representation.

This is the final assembly: the identity

    J_0(2π δ ξ) = (1/π) ∫_{-δ}^{δ} cos(2π ξ x) / sqrt(δ² - x²) dx

for `δ > 0` and arbitrary real `ξ`.  The proof proceeds via:

1.  Substitute `x = δ sin θ` in the integral on the RHS to obtain

        ∫_{-π/2}^{π/2} cos(2π ξ δ sin θ) dθ.

2.  Expand `cos` as its Taylor series and exchange sum and integral
    (dominated convergence; uses `intervalIntegral.tsum_intervalIntegral_eq_of_summable_norm`).

3.  Identify each integral as a Wallis moment.

4.  Match against the `J_0` series.
-/

/-- The "trigonometric" form of the Fourier-cosine identity: substituting
    `x = δ sin θ` converts the arcsine-kernel integral to a `θ`-integral. -/
theorem arcsine_fourier_substitution {δ : ℝ} (hδ : 0 < δ) (ξ : ℝ) :
    (∫ x in Set.Ioo (-δ) δ, Real.cos (2 * Real.pi * ξ * x) /
        Real.sqrt (δ ^ 2 - x ^ 2)) =
      ∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2),
          Real.cos (2 * Real.pi * ξ * δ * Real.sin θ) := by
  set s : Set ℝ := Set.Ioo (-(Real.pi/2)) (Real.pi/2)
  set f : ℝ → ℝ := fun θ => δ * Real.sin θ with hf_def
  set f' : ℝ → ℝ := fun θ => δ * Real.cos θ with hf'_def
  set g : ℝ → ℝ := fun x => Real.cos (2 * Real.pi * ξ * x) /
    Real.sqrt (δ ^ 2 - x ^ 2) with hg_def
  have hs : MeasurableSet s := measurableSet_Ioo
  have hf' : ∀ θ ∈ s, HasDerivWithinAt f (f' θ) s θ :=
    fun θ hθ => hasDerivWithinAt_delta_sin θ hθ
  have hf_mono : MonotoneOn f s := monotoneOn_delta_sin hδ
  have h_change :=
    MeasureTheory.integral_image_eq_integral_deriv_smul_of_monotoneOn hs hf' hf_mono g
  have h_img : f '' s = Set.Ioo (-δ) δ := image_delta_sin_Ioo hδ
  rw [h_img] at h_change
  rw [h_change]
  -- Now simplify: f'(θ) • g(f(θ)) = cos(2π ξ δ sin θ) on the open interval.
  apply MeasureTheory.setIntegral_congr_fun hs
  intro θ hθ
  show f' θ * g (f θ) = Real.cos (2 * Real.pi * ξ * δ * Real.sin θ)
  show δ * Real.cos θ * (Real.cos (2 * Real.pi * ξ * (δ * Real.sin θ)) /
    Real.sqrt (δ ^ 2 - (δ * Real.sin θ) ^ 2)) =
    Real.cos (2 * Real.pi * ξ * δ * Real.sin θ)
  have h_cos_pos : 0 < Real.cos θ := Real.cos_pos_of_mem_Ioo hθ
  have h_sin_sq : (Real.sin θ) ^ 2 + (Real.cos θ) ^ 2 = 1 := Real.sin_sq_add_cos_sq θ
  have h_cos_sq : (Real.cos θ) ^ 2 = 1 - (Real.sin θ) ^ 2 := by linarith
  have h_factor : δ ^ 2 - (δ * Real.sin θ) ^ 2 = δ ^ 2 * (Real.cos θ) ^ 2 := by
    rw [h_cos_sq]; ring
  rw [h_factor]
  have h_sqrt : Real.sqrt (δ ^ 2 * (Real.cos θ) ^ 2) = δ * Real.cos θ := by
    rw [show δ ^ 2 * (Real.cos θ) ^ 2 = (δ * Real.cos θ) ^ 2 from by ring]
    exact Real.sqrt_sq (by positivity : 0 ≤ δ * Real.cos θ)
  rw [h_sqrt]
  have h_prod_pos : 0 < δ * Real.cos θ := mul_pos hδ h_cos_pos
  have h_arg : 2 * Real.pi * ξ * (δ * Real.sin θ) = 2 * Real.pi * ξ * δ * Real.sin θ := by ring
  rw [h_arg]
  field_simp

/-! ### Sum-integral exchange for `cos(c sin θ)`.

The Bessel integral identity

    ∫_{-π/2}^{π/2} cos(c sin θ) dθ = π J_0(c)

is established by Taylor-expanding `cos` and exchanging sum and integral.
-/

/-- The `k`-th Taylor term of `cos(c sin θ)`. -/
noncomputable def cosCSinTerm (c : ℝ) (k : ℕ) (θ : ℝ) : ℝ :=
  ((-1) ^ k : ℝ) * (c * Real.sin θ) ^ (2 * k) / ((2 * k).factorial : ℝ)

/-- Each Taylor term is continuous. -/
lemma continuous_cosCSinTerm (c : ℝ) (k : ℕ) : Continuous (cosCSinTerm c k) := by
  unfold cosCSinTerm
  refine Continuous.div_const ?_ _
  refine Continuous.mul continuous_const ?_
  exact (continuous_const.mul Real.continuous_sin).pow _

/-- Termwise sum of the Taylor terms equals `cos(c sin θ)`. -/
lemma tsum_cosCSinTerm_eq (c θ : ℝ) :
    ∑' k : ℕ, cosCSinTerm c k θ = Real.cos (c * Real.sin θ) := by
  rw [cos_taylor_series (c * Real.sin θ)]
  rfl

/-- Each Taylor term `cosCSinTerm c k` is bounded in absolute value by
    `|c|^{2k} · (sin θ)^{2k} / (2k)!`. -/
lemma abs_cosCSinTerm_le (c : ℝ) (k : ℕ) (θ : ℝ) :
    |cosCSinTerm c k θ| =
      |c| ^ (2 * k) * (Real.sin θ) ^ (2 * k) / ((2 * k).factorial : ℝ) := by
  unfold cosCSinTerm
  rw [abs_div]
  have h_fact_pos : (0 : ℝ) < ((2 * k).factorial : ℝ) := by
    exact_mod_cast Nat.factorial_pos _
  rw [abs_of_pos h_fact_pos]
  congr 1
  rw [abs_mul]
  have h1 : |((-1 : ℝ) ^ k)| = 1 := by rw [abs_pow]; simp
  rw [h1, one_mul, mul_pow, abs_mul]
  -- |c * sin θ|^(2k) = |c|^(2k) * |sin θ|^(2k) (already by mul_pow + abs_mul before pow)
  -- but we want |c|^(2k) · sin^(2k) θ (no abs on sin since it gets squared evenly)
  rw [abs_pow, abs_pow]
  -- |c|^(2k) * |sin θ|^(2k) = |c|^(2k) * sin^(2k) θ since (2k) is even.
  have h_sin_even : |Real.sin θ| ^ (2 * k) = (Real.sin θ) ^ (2 * k) := by
    rw [show 2 * k = 2 * k from rfl, pow_mul, sq_abs, ← pow_mul]
  rw [h_sin_even]

/-- Each Taylor term is integrable on `Set.Ioo (-π/2, π/2)`. -/
lemma integrableOn_cosCSinTerm (c : ℝ) (k : ℕ) :
    MeasureTheory.IntegrableOn (cosCSinTerm c k)
      (Set.Ioo (-(Real.pi/2)) (Real.pi/2)) := by
  -- The term is continuous on ℝ, hence integrable on any bounded measurable set.
  have h_cont : Continuous (cosCSinTerm c k) := continuous_cosCSinTerm c k
  have h_subset : Set.Ioo (-(Real.pi/2)) (Real.pi/2) ⊆
      Set.Icc (-(Real.pi/2)) (Real.pi/2) := Set.Ioo_subset_Icc_self
  have h_compact : IsCompact (Set.Icc (-(Real.pi/2)) (Real.pi/2)) := isCompact_Icc
  have h_int_Icc : MeasureTheory.IntegrableOn (cosCSinTerm c k)
      (Set.Icc (-(Real.pi/2)) (Real.pi/2)) :=
    h_cont.continuousOn.integrableOn_compact h_compact
  exact h_int_Icc.mono_set h_subset

/-- The integral of `|F_k|` over `(-π/2, π/2)` equals `π · |c|^{2k} · C(2k,k) / (4^k · (2k)!)`. -/
lemma integral_abs_cosCSinTerm (c : ℝ) (k : ℕ) :
    (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), ‖cosCSinTerm c k θ‖) =
      Real.pi * |c| ^ (2 * k) * (Nat.choose (2 * k) k : ℝ) /
        ((4 : ℝ) ^ k * ((2 * k).factorial : ℝ)) := by
  have h_pi_pos := Real.pi_pos
  -- Use `MeasureTheory.setIntegral_congr_fun`.
  have h_eq : ∀ θ ∈ Set.Ioo (-(Real.pi/2)) (Real.pi/2),
      ‖cosCSinTerm c k θ‖ =
      |c| ^ (2 * k) * (Real.sin θ) ^ (2 * k) / ((2 * k).factorial : ℝ) := by
    intro θ _
    rw [Real.norm_eq_abs, abs_cosCSinTerm_le]
  rw [MeasureTheory.setIntegral_congr_fun measurableSet_Ioo h_eq]
  -- Now: ∫ θ in Ioo, |c|^(2k) * sin^(2k) θ / (2k)! = ?
  have h_fact_pos : (0 : ℝ) < ((2 * k).factorial : ℝ) := by
    exact_mod_cast Nat.factorial_pos _
  -- Pull out constants.
  have h_form : ∀ θ : ℝ,
      |c| ^ (2 * k) * (Real.sin θ) ^ (2 * k) / ((2 * k).factorial : ℝ) =
      (|c| ^ (2 * k) / ((2 * k).factorial : ℝ)) * (Real.sin θ) ^ (2 * k) := by
    intro θ; ring
  simp_rw [h_form]
  rw [MeasureTheory.integral_const_mul]
  -- Now: (|c|^(2k)/(2k)!) * ∫ θ in Ioo, sin^(2k) θ
  -- Convert Ioo integral to intervalIntegral and use wallis_sin_pow_even_sym.
  have h_int_eq :
      (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), (Real.sin θ) ^ (2 * k)) =
        ∫ θ in (-(Real.pi/2))..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
    rw [intervalIntegral.integral_of_le (by linarith : -(Real.pi/2) ≤ Real.pi/2)]
    rw [← MeasureTheory.integral_Ioc_eq_integral_Ioo]
  rw [h_int_eq, wallis_sin_pow_even_sym]
  field_simp

/-- Summability of the integrals of `|F_k|`. -/
lemma summable_integral_abs_cosCSinTerm (c : ℝ) :
    Summable (fun k : ℕ =>
      ∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), ‖cosCSinTerm c k θ‖) := by
  -- Apply the formula `∫|F_k| = π · |c|^(2k) · C(2k,k) / (4^k · (2k)!)`.
  -- We use `central_binomial_factorial_form` to rewrite this as
  -- `π · |c|^(2k) / ((k!)² · 4^k · 1)` (just `π / (k!)^2 · (|c|/2)^(2k)`).
  -- This is `π * summable_besselJ0_majorant`.
  have h_each : ∀ k : ℕ,
      (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), ‖cosCSinTerm c k θ‖) =
      Real.pi * ((|c| / 2) ^ (2 * k) / ((k.factorial : ℝ) ^ 2)) := by
    intro k
    rw [integral_abs_cosCSinTerm c k]
    -- π * |c|^(2k) * C(2k,k) / (4^k * (2k)!) = π * (|c|/2)^(2k) / (k!)^2
    have h_cbf := central_binomial_factorial_form k
    have h_fact_pos : (0 : ℝ) < ((2 * k).factorial : ℝ) := by
      exact_mod_cast Nat.factorial_pos _
    have h_kfact_pos : (0 : ℝ) < (k.factorial : ℝ) := by
      exact_mod_cast Nat.factorial_pos _
    have h_4k_pos : (0 : ℝ) < (4 : ℝ) ^ k := by positivity
    have h_kfact_sq_pos : (0 : ℝ) < (k.factorial : ℝ) ^ 2 := by positivity
    have h_cbf_simple : (Nat.choose (2 * k) k : ℝ) * ((k.factorial : ℝ) ^ 2) =
        ((2 * k).factorial : ℝ) := by
      have h := h_cbf
      field_simp at h
      linarith [h]
    -- Express (|c|/2)^(2k) = |c|^(2k) / 4^k.
    have h_half_pow : ((|c| / 2) ^ (2 * k) : ℝ) = |c| ^ (2 * k) / (4 : ℝ) ^ k := by
      rw [div_pow]
      congr 1
      rw [show (2 * k : ℕ) = 2 * k from rfl, pow_mul]
      norm_num
    rw [h_half_pow]
    -- Now: π * |c|^(2k) * C(2k,k) / (4^k * (2k)!) = π * (|c|^(2k)/4^k) / (k!)^2
    field_simp
    -- After field_simp, both sides should be cross-multiplied. Use h_cbf_simple.
    have : |c| ^ (2 * k) * (Nat.choose (2 * k) k : ℝ) * ((k.factorial : ℝ) ^ 2) =
           |c| ^ (2 * k) * ((2 * k).factorial : ℝ) := by
      rw [mul_assoc, h_cbf_simple]
    nlinarith [this, Real.pi_pos, h_fact_pos, h_kfact_pos, h_4k_pos, h_kfact_sq_pos,
               pow_nonneg (abs_nonneg c) (2 * k)]
  -- Now use `Summable.congr` to switch to the simpler form.
  refine Summable.congr ?_ (fun k => (h_each k).symm)
  -- We need `Summable (fun k => π * ((|c|/2)^(2k) / (k!)^2))`.
  have h_maj := summable_besselJ0_majorant |c|
  exact h_maj.mul_left Real.pi

/-- **Bessel's integral formula**: for all real `c`,

    π · J_0(c) = ∫_{-π/2}^{π/2} cos(c sin θ) dθ.

This is the cleanest form of the Bessel integral representation. -/
theorem besselJ0_eq_cos_csin_integral (c : ℝ) :
    Real.pi * besselJ0 c =
      ∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), Real.cos (c * Real.sin θ) := by
  -- Step 1: Express RHS as ∫ ∑' F_k.
  have h_RHS_sum :
      (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), Real.cos (c * Real.sin θ)) =
      ∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), ∑' k : ℕ, cosCSinTerm c k θ := by
    apply MeasureTheory.setIntegral_congr_fun measurableSet_Ioo
    intro θ _
    exact (tsum_cosCSinTerm_eq c θ).symm
  rw [h_RHS_sum]
  -- Step 2: Swap sum and integral using `integral_tsum_of_summable_integral_norm`.
  have h_swap :
      (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), ∑' k : ℕ, cosCSinTerm c k θ) =
      ∑' k : ℕ, ∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), cosCSinTerm c k θ := by
    rw [← MeasureTheory.integral_tsum_of_summable_integral_norm
      (F := fun k θ => cosCSinTerm c k θ)
      (μ := MeasureTheory.volume.restrict (Set.Ioo (-(Real.pi/2)) (Real.pi/2)))
      (fun k => integrableOn_cosCSinTerm c k)
      (summable_integral_abs_cosCSinTerm c)]
  rw [h_swap]
  -- Step 3: Each integral becomes the J_0 series term times π.
  -- ∫ F_k = (-1)^k · c^(2k) / (2k)! · ∫ sin^(2k) θ dθ = (-1)^k · c^(2k) / (2k)! · π C(2k,k)/4^k
  have h_each : ∀ k : ℕ,
      (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), cosCSinTerm c k θ) =
      Real.pi * besselJ0Term c k := by
    intro k
    unfold cosCSinTerm besselJ0Term
    -- F_k(θ) = (-1)^k (c sin θ)^(2k) / (2k)!
    -- = (-1)^k c^(2k) sin^(2k) θ / (2k)!
    have h_form : ∀ θ : ℝ,
        ((-1) ^ k : ℝ) * (c * Real.sin θ) ^ (2 * k) / ((2 * k).factorial : ℝ) =
        ((-1) ^ k * c ^ (2 * k) / ((2 * k).factorial : ℝ)) * (Real.sin θ) ^ (2 * k) := by
      intro θ
      rw [mul_pow]; ring
    simp_rw [h_form]
    rw [MeasureTheory.integral_const_mul]
    -- Now: (-1)^k c^(2k) / (2k)! * ∫ sin^(2k)
    have h_int_eq :
        (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2), (Real.sin θ) ^ (2 * k)) =
          ∫ θ in (-(Real.pi/2))..(Real.pi/2), (Real.sin θ) ^ (2 * k) := by
      rw [intervalIntegral.integral_of_le
        (by linarith [Real.pi_pos] : -(Real.pi/2) ≤ Real.pi/2)]
      rw [← MeasureTheory.integral_Ioc_eq_integral_Ioo]
    rw [h_int_eq, wallis_sin_pow_even_sym]
    -- ((-1)^k * c^(2k) / (2k)!) * (π · C(2k,k) / 4^k) = π * besselJ0Term c k
    have h_cbf := central_binomial_factorial_form k
    have h_fact_pos : (0 : ℝ) < ((2 * k).factorial : ℝ) := by
      exact_mod_cast Nat.factorial_pos _
    have h_kfact_pos : (0 : ℝ) < (k.factorial : ℝ) := by
      exact_mod_cast Nat.factorial_pos _
    have h_4k_pos : (0 : ℝ) < (4 : ℝ) ^ k := by positivity
    have h_kfact_sq_pos : (0 : ℝ) < (k.factorial : ℝ) ^ 2 := by positivity
    have h_cbf_simple : (Nat.choose (2 * k) k : ℝ) * ((k.factorial : ℝ) ^ 2) =
        ((2 * k).factorial : ℝ) := by
      have h := h_cbf
      field_simp at h
      linarith [h]
    -- Rewrite (c/2)^(2k) = c^(2k) / 4^k.
    have h_half_pow : ((c / 2) ^ (2 * k) : ℝ) = c ^ (2 * k) / (4 : ℝ) ^ k := by
      rw [div_pow]
      congr 1
      rw [show (2 * k : ℕ) = 2 * k from rfl, pow_mul]
      norm_num
    rw [h_half_pow]
    -- LHS: ((-1)^k * c^(2k) / (2k)!) * (π * C(2k,k) / 4^k)
    -- RHS: π * ((-1)^k * c^(2k) / 4^k / (k!)^2)
    field_simp
    have h_aux : c ^ (2 * k) * (Nat.choose (2 * k) k : ℝ) * ((k.factorial : ℝ) ^ 2) =
                 c ^ (2 * k) * ((2 * k).factorial : ℝ) := by
      rw [mul_assoc, h_cbf_simple]
    nlinarith [h_aux]
  -- Step 4: Sum: ∑ π * besselJ0Term c k = π * ∑ besselJ0Term c k = π * besselJ0 c.
  simp_rw [h_each]
  rw [tsum_mul_left]
  rfl

/-- **Bessel `J_0` Fourier-cosine integral representation.**  For all
    `δ > 0` and `ξ : ℝ`,

    `J_0(2π δ ξ) = (1/π) · ∫_{-δ}^{δ} cos(2π ξ x) / sqrt(δ² - x²) dx`. -/
theorem besselJ0_arcsine_fourier {δ : ℝ} (hδ : 0 < δ) (ξ : ℝ) :
    besselJ0 (2 * Real.pi * δ * ξ) =
      (1 / Real.pi) * ∫ x in Set.Ioo (-δ) δ,
        Real.cos (2 * Real.pi * ξ * x) / Real.sqrt (δ ^ 2 - x ^ 2) := by
  -- Step 1: Use `arcsine_fourier_substitution` to switch to θ integral.
  rw [arcsine_fourier_substitution hδ ξ]
  -- Step 2: Use `besselJ0_eq_cos_csin_integral` with c = 2π δ ξ.
  have h_pi_pos : 0 < Real.pi := Real.pi_pos
  have h_pi_ne : Real.pi ≠ 0 := ne_of_gt h_pi_pos
  have h_bessel := besselJ0_eq_cos_csin_integral (2 * Real.pi * δ * ξ)
  -- h_bessel : π * J_0(2πδξ) = ∫ θ in Ioo (-π/2)(π/2), cos((2πδξ) sin θ)
  -- We want: J_0(2πδξ) = (1/π) * ∫ ...
  -- Note: cos((2πδξ) sin θ) = cos(2πξ · δ sin θ), same as RHS.
  have h_arg_eq : ∀ θ : ℝ, (2 * Real.pi * δ * ξ) * Real.sin θ =
      2 * Real.pi * ξ * δ * Real.sin θ := by intro θ; ring
  -- Rewrite the integrand.
  have h_int_eq :
      (∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2),
        Real.cos ((2 * Real.pi * δ * ξ) * Real.sin θ)) =
      ∫ θ in Set.Ioo (-(Real.pi/2)) (Real.pi/2),
        Real.cos (2 * Real.pi * ξ * δ * Real.sin θ) := by
    apply MeasureTheory.setIntegral_congr_fun measurableSet_Ioo
    intro θ _
    show Real.cos (2 * Real.pi * δ * ξ * Real.sin θ) =
         Real.cos (2 * Real.pi * ξ * δ * Real.sin θ)
    rw [h_arg_eq]
  rw [← h_int_eq, ← h_bessel]
  field_simp

end Sidon.Bessel
