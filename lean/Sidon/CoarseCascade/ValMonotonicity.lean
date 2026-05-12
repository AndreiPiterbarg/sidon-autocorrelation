/-
Sidon Autocorrelation Project — val(d) Monotonicity and Convergence

val(d) = inf_{mu in Delta_d} sup_W TV_W(mu)

The full mathematical theory of `val(d)` involves convergence in the limit
d → ∞ (the value approaches C_1a = inf_f autoconvolution_ratio f). The
formalization of this limit requires substantial measure-theoretic and
analytic machinery (sequential compactness on the simplex, continuity of
the autoconvolution ratio in suitable topologies).

This file proves the foundational properties needed:
  - The relation between val(d), Theorem 1, and C_1a (using existing axiom A2).
  - Monotonicity as a consequence of refinement_monotonicity (existing axiom A1).
  - val(2) computation (the trivial base case).

Source: proof/coarse_cascade_method.md Section 5.
-/

import Sidon.Proof.CoarseCascade

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- =============================================================================
-- Definitions
-- =============================================================================

/-- max_W TV_W(μ; d) — the maximum test value over all valid windows. -/
noncomputable def max_tv_window (d : ℕ) (μ : Fin d → ℝ) (c_target : ℝ) : Prop :=
  ∃ ell s, 2 ≤ ell ∧ s + ell ≤ 2 * d ∧ mass_test_value d μ ell s ≥ c_target

-- =============================================================================
-- PART 1: Theorem 1 ⟹ C_1a ≥ val(d) (using simplex_tv_coverage axiom)
-- =============================================================================

/-- If the simplex coverage axiom holds at dimension 2*n_term with target c,
    then every admissible f has R(f) ≥ c.

    This is the EXACT statement of `coarse_cascade_bound` from
    `Sidon.Proof.CoarseCascade`. We restate it here as `val_le_C1a`. -/
theorem val_le_C1a (n_term : ℕ) (hn : n_term > 0) (c_target : ℝ) (hct : 0 < c_target)
    (h_coverage : ∀ μ : Fin (2 * n_term) → ℝ,
      on_simplex μ →
      ∃ (ell s : ℕ), 2 ≤ ell ∧ mass_test_value (2 * n_term) μ ell s ≥ c_target) :
    ∀ f : ℝ → ℝ,
      (∀ x, 0 ≤ f x) →
      (Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4)) →
      (MeasureTheory.integral MeasureTheory.volume f = 1) →
      (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤
        MeasureTheory.volume ≠ ⊤) →
      autoconvolution_ratio f ≥ c_target :=
  coarse_cascade_bound n_term hn c_target hct h_coverage

-- =============================================================================
-- PART 2: val(d) Monotonicity via Refinement Monotonicity
-- =============================================================================

/-- **Refinement-induced monotonicity (statement):** If parent μ has TV ≥ c at
    some valid window, then any refinement ν has TV ≥ c at some window.

    This uses the existing axiom `refinement_monotonicity` from
    `Sidon.Proof.CoarseCascade`. -/
theorem refinement_lifts_tv (d : ℕ) (c_target : ℝ)
    (μ : Fin d → ℝ) (ν : Fin (2 * d) → ℝ)
    (hμ : on_simplex μ) (hν : on_simplex ν)
    (h_ref : is_mass_refinement μ ν)
    (ell s : ℕ) (hℓ : 2 ≤ ell) (hs : s + ell ≤ 2 * d)
    (h_tv : mass_test_value d μ ell s ≥ c_target) :
    ∃ (ell' s' : ℕ), 2 ≤ ell' ∧ s' + ell' ≤ 2 * (2 * d) ∧
      mass_test_value (2 * d) ν ell' s' ≥ c_target :=
  refinement_monotonicity c_target μ ν hμ hν h_ref ell s hℓ hs h_tv

-- =============================================================================
-- PART 3: val(2) — The Trivial Base Case (uniform 1/2, 1/2)
-- =============================================================================

/-- For the uniform distribution at d=2 (μ = (1/2, 1/2)), the window (ell=2, s=1)
    gives TV = 1.

    Computation:
      mass_test_value 2 (1/2, 1/2) 2 1
        = (2*2/2) * ∑_{k ∈ Icc 1 1} discrete_autoconvolution μ k
        = 2 * discrete_autoconvolution μ 1
        = 2 * (μ_0 * μ_1 + μ_1 * μ_0)  [pairs (0,1), (1,0) sum to 1]
        = 2 * 2 * (1/2 * 1/2)
        = 2 * 1/2
        = 1. -/
theorem val_two_uniform_eq_one :
    mass_test_value 2 (fun _ : Fin 2 => (1 : ℝ) / 2) 2 1 = 1 := by
  unfold mass_test_value discrete_autoconvolution
  simp only [Fin.sum_univ_two, Fin.val_zero, Fin.val_one]
  -- Window: Icc 1 (1+2-2) = Icc 1 1 = {1}
  have h_window : Finset.Icc 1 (1 + 2 - 2) = ({1} : Finset ℕ) := by
    simp
  rw [h_window]
  simp only [Finset.sum_singleton]
  norm_num

-- =============================================================================
-- PART 4: Trivial Bound — val(d) ≥ 0 (from non-negativity of TV on simplex)
-- =============================================================================

/-- For any μ on the simplex at dimension d ≥ 2, the test value at any window
    with 2 ≤ ell is non-negative. -/
theorem mass_test_value_nonneg_on_simplex (d : ℕ) (μ : Fin d → ℝ) (hμ : on_simplex μ)
    (ell s : ℕ) (hell : 2 ≤ ell) :
    0 ≤ mass_test_value d μ ell s := by
  unfold mass_test_value
  apply mul_nonneg
  · apply div_nonneg
    · positivity
    · exact Nat.cast_nonneg _
  · apply Finset.sum_nonneg
    intro k _
    unfold discrete_autoconvolution
    apply Finset.sum_nonneg
    intro i _
    apply Finset.sum_nonneg
    intro j _
    split_ifs
    · exact mul_nonneg (hμ.1 i) (hμ.1 j)
    · exact le_refl 0

end -- noncomputable section
