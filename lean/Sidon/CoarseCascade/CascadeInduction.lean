/-
Sidon Autocorrelation Project — Coarse Cascade Induction

The cascade induction principle for the coarse grid:

  If at every level, either a composition is directly pruned (TV >= c_target)
  or all of its children are pruned, then all compositions at the starting
  dimension are "cascade-pruned" and the bound C_1a >= c_target holds.

This parallels CascadeInduction.lean in Algorithm/ but for the coarse grid
(no correction, mass-based TV, constant S).

Source: proof/coarse_cascade_method.md Section 8.2.

NOTE: This file proves the CASCADE STRUCTURE (predicate definitions and
induction step). The full bridge from "cascade pruned" to "C_1a ≥ c_target"
is established in `Sidon.Proof.CoarseCascade` (`coarse_cascade_bound`)
using the existing axiom `simplex_tv_coverage`.
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
-- Coarse Cascade Pruning Predicate
-- =============================================================================

/-- A composition is coarse-cascade-pruned if either:
    1. Directly pruned: exists window with TV >= c_target, OR
    2. All children at dimension 2d are coarse-cascade-pruned.

    Note: uses mass_test_value (no correction), not test_value (with correction).
    Note: child constraint is exact split (not ±1 rounding as in fine grid). -/
inductive CoarseCascadePruned (S : ℕ) (c_target : ℝ) :
    (d : ℕ) → (Fin d → ℕ) → Prop where
  | direct {d : ℕ} {c : Fin d → ℕ}
      (h : ∃ ell s, 2 ≤ ell ∧
        mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell s ≥ c_target) :
      CoarseCascadePruned S c_target d c
  | refine {d : ℕ} {c : Fin d → ℕ}
      (h : ∀ child : Fin (2 * d) → ℕ,
        (∀ i : Fin d, child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = c i) →
        (∑ j, child j = S) →
        CoarseCascadePruned S c_target (2 * d) child) :
      CoarseCascadePruned S c_target d c

-- =============================================================================
-- PART 1: Cascade Induction Step
-- =============================================================================

/-- If the cascade at level k has 0 survivors (all children of all parents
    are pruned), then all parents are coarse-cascade-pruned via the refine case. -/
theorem cascade_induction_step (S : ℕ) (c_target : ℝ)
    (d : ℕ) (parent : Fin d → ℕ) (_h_sum : ∑ i, parent i = S)
    (h_all_children_pruned : ∀ child : Fin (2 * d) → ℕ,
      (∀ i : Fin d,
        child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = parent i) →
      (∑ j, child j = S) →
      CoarseCascadePruned S c_target (2 * d) child) :
    CoarseCascadePruned S c_target d parent :=
  CoarseCascadePruned.refine h_all_children_pruned

-- =============================================================================
-- PART 2: Cascade Pruned Implies Bound (high-level statement)
-- =============================================================================

/-- **Direct-case soundness:** If a composition c at dimension d is "directly pruned"
    (TV at the grid point ≥ c_target), then this is exactly the witness used by
    `CoarseCascadePruned.direct`.

    The full implication "CoarseCascadePruned → C_1a ≥ c_target" requires the
    existing axiom `refinement_monotonicity` to handle the inductive `refine` case;
    that bridge is established in `Sidon.Proof.CoarseCascade.coarse_cascade_bound`. -/
theorem coarse_cascade_directly_pruned_bound (S : ℕ) (_hS : S > 0)
    (c_target : ℝ) (_hct : 0 < c_target)
    (d : ℕ) (_hd : d > 0) (c : Fin d → ℕ) (_hc_sum : ∑ i, c i = S)
    (h_direct : ∃ ell s, 2 ≤ ell ∧
        mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell s ≥ c_target) :
    CoarseCascadePruned S c_target d c :=
  CoarseCascadePruned.direct h_direct

-- =============================================================================
-- PART 3: Subtree Pruning Justification (with Refinement Monotonicity hypothesis)
-- =============================================================================

/-- If a parent is directly pruned (TV ≥ c_target with a window in the cascade's
    normal regime s+ell ≤ 2*d) AND refinement monotonicity holds (as a hypothesis),
    then every valid child is also CoarseCascadePruned.

    This justifies skipping the subtree of a directly pruned parent in the cascade.

    HYPOTHESES:
    - `h_direct`: parent has TV ≥ c_target at SOME window with s+ell ≤ 2*d
      (the cascade always operates in this regime — outside it, the convolution
      indices > 2d-2 contribute zero).
    - `h_mono`: refinement monotonicity for the SPECIFIC parent + child pair.
      Captured by the existing AXIOM `refinement_monotonicity` in `Sidon.Proof.CoarseCascade`. -/
theorem directly_pruned_implies_descendants_pruned (S : ℕ) (hS : S > 0) (c_target : ℝ)
    (d : ℕ) (parent : Fin d → ℕ) (_h_sum : ∑ i, parent i = S)
    (h_direct : ∃ ell s, 2 ≤ ell ∧ s + ell ≤ 2 * d ∧
      mass_test_value d (fun i => (parent i : ℝ) / (S : ℝ)) ell s ≥ c_target)
    (h_mono : ∀ (μ : Fin d → ℝ) (ν : Fin (2 * d) → ℝ),
      on_simplex μ → on_simplex ν → is_mass_refinement μ ν →
      ∀ ell s, 2 ≤ ell → s + ell ≤ 2 * d →
        mass_test_value d μ ell s ≥ c_target →
        ∃ ell' s', 2 ≤ ell' ∧ mass_test_value (2 * d) ν ell' s' ≥ c_target) :
    ∀ child : Fin (2 * d) → ℕ,
      (∀ i : Fin d,
        child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = parent i) →
      (∑ j, child j = S) →
      CoarseCascadePruned S c_target (2 * d) child := by
  intro child h_split h_child_sum
  obtain ⟨ell, s, hℓ, h_window_valid, h_tv⟩ := h_direct
  -- Construct the simplex/refinement hypotheses needed by h_mono.
  have hS_ne : (S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h_parent_simplex : on_simplex (fun i => (parent i : ℝ) / (S : ℝ)) := by
    refine ⟨?_, ?_⟩
    · intro i; apply div_nonneg
      · exact Nat.cast_nonneg _
      · exact Nat.cast_nonneg _
    · rw [← Finset.sum_div]
      have h_cast : (∑ i : Fin d, (parent i : ℝ)) = ((∑ i, parent i : ℕ) : ℝ) := by
        push_cast
        rfl
      rw [h_cast, _h_sum]
      field_simp
  have h_child_simplex : on_simplex (fun j => (child j : ℝ) / (S : ℝ)) := by
    refine ⟨?_, ?_⟩
    · intro j; apply div_nonneg
      · exact Nat.cast_nonneg _
      · exact Nat.cast_nonneg _
    · rw [← Finset.sum_div]
      have h_cast : (∑ i : Fin (2 * d), (child i : ℝ)) = ((∑ i, child i : ℕ) : ℝ) := by
        push_cast
        rfl
      rw [h_cast, h_child_sum]
      field_simp
  have h_refine : is_mass_refinement
      (fun i => (parent i : ℝ) / (S : ℝ))
      (fun j => (child j : ℝ) / (S : ℝ)) := by
    refine ⟨?_, ?_⟩
    · intro i
      have h_eq : (child ⟨2 * i.val, by omega⟩ : ℝ) + (child ⟨2 * i.val + 1, by omega⟩ : ℝ) =
          (parent i : ℝ) := by
        have h_nat : child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = parent i :=
          h_split i
        exact_mod_cast h_nat
      have : (child ⟨2 * i.val, by omega⟩ : ℝ) / (S : ℝ) +
             (child ⟨2 * i.val + 1, by omega⟩ : ℝ) / (S : ℝ) =
             ((child ⟨2 * i.val, by omega⟩ : ℝ) + (child ⟨2 * i.val + 1, by omega⟩ : ℝ)) / (S : ℝ) := by
        field_simp
      rw [this, h_eq]
    · intro j; apply div_nonneg
      · exact Nat.cast_nonneg _
      · exact Nat.cast_nonneg _
  obtain ⟨ell', s', hℓ', h_tv'⟩ := h_mono _ _ h_parent_simplex h_child_simplex h_refine
    ell s hℓ h_window_valid h_tv
  exact CoarseCascadePruned.direct ⟨ell', s', hℓ', h_tv'⟩

end -- noncomputable section
