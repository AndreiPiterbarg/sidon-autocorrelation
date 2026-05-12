/-
Sidon Autocorrelation Project — Subtree Pruning Soundness

When cursors a_0, ..., a_pos are assigned (bins 0..2*pos+1 fixed),
the partial autoconvolution of assigned bins is a LOWER BOUND on the
full autoconvolution (since all masses are non-negative).

If ws_partial > thr[ell] for any window fully within the assigned range,
then the full window sum will also exceed the threshold, and the entire
subtree below can be pruned.

Source: proof/coarse_cascade_method.md Section 6.5.
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
-- PART 1: Partial Autoconvolution Lower Bound
-- =============================================================================

/-- Partial autoconvolution: only sum over assigned bins [0, p). -/
def partial_autoconvolution {d : ℕ} (a : Fin d → ℝ) (p : ℕ) (k : ℕ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d,
    if i.val < p ∧ j.val < p ∧ i.val + j.val = k then a i * a j else 0

/-- **Partial conv is a lower bound on full conv** when all entries are non-negative. -/
theorem partial_conv_le_full {d : ℕ} (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i)
    (p : ℕ) (k : ℕ) :
    partial_autoconvolution a p k ≤ discrete_autoconvolution a k := by
  unfold partial_autoconvolution discrete_autoconvolution
  apply Finset.sum_le_sum
  intro i _
  apply Finset.sum_le_sum
  intro j _
  by_cases h_kij : i.val + j.val = k
  · by_cases h_pij : i.val < p ∧ j.val < p
    · simp [h_pij.1, h_pij.2, h_kij]
    · -- partial term is 0, full term is a_i * a_j ≥ 0
      have h_full : (if i.val + j.val = k then a i * a j else 0) = a i * a j := by simp [h_kij]
      have h_partial : (if i.val < p ∧ j.val < p ∧ i.val + j.val = k then a i * a j else 0) = 0 := by
        rcases Classical.em (i.val < p) with hi | hi
        · rcases Classical.em (j.val < p) with hj | hj
          · exact absurd ⟨hi, hj⟩ h_pij
          · simp [hj]
        · simp [hi]
      rw [h_partial, h_full]
      exact mul_nonneg (ha i) (ha j)
  · -- both terms are 0
    simp [h_kij]

/-- Windowed partial sum is a lower bound on windowed full sum. -/
theorem partial_window_sum_le_full {d : ℕ} (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i)
    (p : ℕ) (ell s : ℕ) :
    ∑ k ∈ Finset.Icc s (s + ell - 2), partial_autoconvolution a p k ≤
    ∑ k ∈ Finset.Icc s (s + ell - 2), discrete_autoconvolution a k := by
  apply Finset.sum_le_sum
  intro k _
  exact partial_conv_le_full a ha p k

-- =============================================================================
-- PART 2: Subtree Pruning Soundness
-- =============================================================================

/-- **Subtree Pruning Theorem:**
    If the partial mass_test_value of assigned bins [0, p) already
    exceeds the threshold for some window, then the FULL mass_test_value
    also exceeds the threshold, regardless of how bins [p, d) are assigned. -/
theorem subtree_pruning_sound {d : ℕ}
    (c_target : ℝ)
    (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i)
    (p : ℕ) (_hp : p ≤ d)
    (ell s : ℕ) (_hell : 2 ≤ ell)
    (h_partial_exceeds :
      mass_test_value d (fun i => if i.val < p then a i else 0) ell s ≥ c_target) :
    -- For ANY assignment of bins [p, d) extending a:
    ∀ b : Fin d → ℝ,
      (∀ i, 0 ≤ b i) →
      (∀ i : Fin d, i.val < p → b i = a i) →
      mass_test_value d b ell s ≥ c_target := by
  intro b hb h_agree
  -- Show: mass_test_value d (a-with-restriction) ≤ mass_test_value d b
  -- because b agrees with a on [0, p) and is non-negative on [p, d).
  -- Specifically, the restriction (i ↦ if i.val < p then a i else 0) is ≤ b pointwise.
  have h_restrict_le : ∀ i : Fin d, (if i.val < p then a i else 0) ≤ b i := by
    intro i
    by_cases hip : i.val < p
    · simp only [hip, if_true]; rw [h_agree i hip]
    · simp [hip, hb i]
  -- mass_test_value is monotone in non-negative inputs (sum of products of non-negative entries)
  have h_restrict_nn : ∀ i : Fin d, 0 ≤ (if i.val < p then a i else 0) := by
    intro i
    split_ifs with h
    · exact ha i
    · exact le_refl 0
  have h_mass_le : mass_test_value d (fun i => if i.val < p then a i else 0) ell s ≤
      mass_test_value d b ell s := by
    unfold mass_test_value discrete_autoconvolution
    have hd_nn : 0 ≤ (2 * (d : ℝ) / (ell : ℝ)) := by
      apply div_nonneg
      · positivity
      · exact Nat.cast_nonneg _
    apply mul_le_mul_of_nonneg_left _ hd_nn
    apply Finset.sum_le_sum
    intro k _
    apply Finset.sum_le_sum
    intro i _
    apply Finset.sum_le_sum
    intro j _
    split_ifs with h_eq
    · -- i + j = k case: compare products
      -- (if i<p then a i else 0) * (if j<p then a j else 0) ≤ b i * b j
      apply mul_le_mul (h_restrict_le i) (h_restrict_le j) (h_restrict_nn j) (hb i)
    · exact le_refl 0
  exact le_trans h_partial_exceeds h_mass_le

-- =============================================================================
-- PART 3: Incremental Convolution Update (Trivial Identity)
-- =============================================================================

/-- Incremental update identity: the difference between two convolutions. -/
theorem incremental_conv_decompose {d : ℕ} (a_old a_new : Fin d → ℝ) (k : ℕ) :
    discrete_autoconvolution a_new k - discrete_autoconvolution a_old k =
    discrete_autoconvolution a_new k - discrete_autoconvolution a_old k := rfl

-- =============================================================================
-- PART 4: Incremental Update Correctness (Undo on Backtrack)
-- =============================================================================

/-- Undo property: subtracting the same delta restores the old convolution.
    This justifies the backtracking in the DFS cascade. -/
theorem incremental_conv_undo {d : ℕ} (a_old a_new : Fin d → ℝ) (k : ℕ) :
    let delta := discrete_autoconvolution a_new k - discrete_autoconvolution a_old k
    discrete_autoconvolution a_new k - delta = discrete_autoconvolution a_old k := by
  simp [sub_sub_cancel]

end -- noncomputable section
