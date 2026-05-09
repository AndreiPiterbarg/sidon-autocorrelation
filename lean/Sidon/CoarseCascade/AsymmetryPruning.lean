/-
Sidon Autocorrelation Project — Asymmetry Pruning for Coarse Grid

If the left-half mass fraction >= sqrt(c/2), the autoconvolution peak
is already >= c from the left half alone. This prunes compositions
with extreme left-right imbalance.

On the coarse grid this is EXACT (no correction needed), unlike the
fine grid where the asymmetry threshold must account for discretization error.

Source: run_cascade_coarse.py asymmetry_prune_mask_coarse, pruning.py asymmetry_threshold.
-/

import Sidon.Proof.CoarseCascade
import Sidon.CoarseCascade.IntegerThreshold

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
-- Asymmetry Bound (Coarse Grid — No Correction)
-- =============================================================================

/-- Left-half mass: sum of masses in bins [0, d/2). -/
def left_mass {d : ℕ} (μ : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, if i.val < d / 2 then μ i else 0

/-- Right-half mass: sum of masses in bins [d/2, d). -/
def right_mass {d : ℕ} (μ : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, if i.val ≥ d / 2 then μ i else 0

/-- Helper: For the window (ell = d, s = 0) on dimension d, the test value equals
    2 * ∑_{k=0..d-2} ∑_{i+j=k} μ_i μ_j. -/
private lemma full_left_window_eq {d : ℕ} (μ : Fin d → ℝ) (hd2 : d ≥ 2) :
    mass_test_value d μ d 0 =
    2 * ∑ k ∈ Finset.Icc 0 (d - 2), discrete_autoconvolution μ k := by
  unfold mass_test_value
  have hd_ne : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h_factor : (2 * (d : ℝ) / (d : ℝ)) = 2 := by
    field_simp
  rw [h_factor]
  have h_window : Finset.Icc 0 (0 + d - 2) = Finset.Icc 0 (d - 2) := by
    have : 0 + d - 2 = d - 2 := by omega
    rw [this]
  rw [h_window]

/-- Helper: For the window (ell = d, s = d), the test value equals
    2 * ∑_{k=d..2d-2} ∑_{i+j=k} μ_i μ_j. -/
private lemma right_window_eq {d : ℕ} (μ : Fin d → ℝ) (hd2 : d ≥ 2) :
    mass_test_value d μ d d =
    2 * ∑ k ∈ Finset.Icc d (2 * d - 2), discrete_autoconvolution μ k := by
  unfold mass_test_value
  have hd_ne : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h_factor : (2 * (d : ℝ) / (d : ℝ)) = 2 := by
    field_simp
  rw [h_factor]
  have h_window : Finset.Icc d (d + d - 2) = Finset.Icc d (2 * d - 2) := by
    have : d + d - 2 = 2 * d - 2 := by omega
    rw [this]
  rw [h_window]

/-- Helper: A subset of pairs (i, j) with i+j ≤ d - 2 lower-bounds the window sum. -/
private lemma subset_pairs_lower_bound {d : ℕ} (μ : Fin d → ℝ) (hμ : ∀ i, 0 ≤ μ i) (hd2 : d ≥ 2)
    (M : ℕ) (_hM : M ≤ d / 2) :
    ∑ k ∈ Finset.Icc 0 (d - 2), discrete_autoconvolution μ k ≥
    ∑ i : Fin d, ∑ j : Fin d,
      if i.val < M ∧ j.val < M then μ i * μ j else 0 := by
  have h_pairs_in_window : ∀ (i j : Fin d), i.val < M → j.val < M → i.val + j.val ≤ d - 2 := by
    intro i j hi hj
    have hM_le : M ≤ d / 2 := _hM
    have h_div : 2 * (d / 2) ≤ d := Nat.mul_div_le d 2
    omega
  have h_eq : ∑ i : Fin d, ∑ j : Fin d,
      (if i.val < M ∧ j.val < M then μ i * μ j else 0) =
      ∑ k ∈ Finset.Icc 0 (d - 2), ∑ i : Fin d, ∑ j : Fin d,
        if i.val < M ∧ j.val < M ∧ i.val + j.val = k then μ i * μ j else 0 := by
    rw [show ∑ k ∈ Finset.Icc 0 (d - 2), ∑ i : Fin d, ∑ j : Fin d,
        (if i.val < M ∧ j.val < M ∧ i.val + j.val = k then μ i * μ j else 0) =
        ∑ i : Fin d, ∑ j : Fin d, ∑ k ∈ Finset.Icc 0 (d - 2),
        (if i.val < M ∧ j.val < M ∧ i.val + j.val = k then μ i * μ j else 0) from by
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl ?_
      intro i _
      rw [Finset.sum_comm]]
    refine Finset.sum_congr rfl ?_
    intro i _
    refine Finset.sum_congr rfl ?_
    intro j _
    by_cases hij : i.val < M ∧ j.val < M
    · have h_in_range : i.val + j.val ∈ Finset.Icc 0 (d - 2) := by
        simp only [Finset.mem_Icc]
        exact ⟨Nat.zero_le _, h_pairs_in_window i j hij.1 hij.2⟩
      have h_unique : ∑ k ∈ Finset.Icc 0 (d - 2),
          (if i.val < M ∧ j.val < M ∧ i.val + j.val = k then μ i * μ j else 0) = μ i * μ j := by
        rw [Finset.sum_eq_single (i.val + j.val)]
        · simp [hij.1, hij.2]
        · intro b _ hb_ne
          have h_not_eq : ¬ i.val + j.val = b := fun h => hb_ne h.symm
          simp [hij.1, hij.2, h_not_eq]
        · intro h_not_in
          exact absurd h_in_range h_not_in
      rw [h_unique]
      simp [hij.1, hij.2]
    · push_neg at hij
      have h_lhs : (if i.val < M ∧ j.val < M then μ i * μ j else 0) = 0 := by
        rcases Classical.em (i.val < M) with hi | hi
        · have hj : ¬ j.val < M := by have := hij hi; omega
          simp [hi, hj]
        · simp [hi]
      have h_rhs : ∑ k ∈ Finset.Icc 0 (d - 2),
          (if i.val < M ∧ j.val < M ∧ i.val + j.val = k then μ i * μ j else 0) = 0 := by
        apply Finset.sum_eq_zero
        intro k _
        rcases Classical.em (i.val < M) with hi | hi
        · have hj : ¬ j.val < M := by have := hij hi; omega
          simp [hi, hj]
        · simp [hi]
      rw [h_lhs, h_rhs]
  rw [h_eq]
  apply Finset.sum_le_sum
  intro k _
  unfold discrete_autoconvolution
  apply Finset.sum_le_sum
  intro i _
  apply Finset.sum_le_sum
  intro j _
  by_cases h_eq : i.val + j.val = k
  · by_cases h_ij : i.val < M ∧ j.val < M
    · simp [h_eq, h_ij.1, h_ij.2]
    · push_neg at h_ij
      simp only [h_eq, and_true]
      rcases Classical.em (i.val < M) with hi | hi
      · simp only [hi, true_and]
        have hj : ¬ j.val < M := by have := h_ij hi; omega
        simp [hj, mul_nonneg (hμ i) (hμ j)]
      · simp [hi, mul_nonneg (hμ i) (hμ j)]
  · simp [h_eq]

/-- Helper: A subset of pairs (i, j) with i, j ≥ d/2 lies in window [d, 2d-2]
    when d is EVEN. (For odd d the pair (d/2, d/2) lies at k = d-1, just below window.) -/
private lemma right_subset_pairs_lower_bound {d : ℕ} (μ : Fin d → ℝ) (hμ : ∀ i, 0 ≤ μ i)
    (hd2 : d ≥ 2) (h_even : Even d) :
    ∑ k ∈ Finset.Icc d (2 * d - 2), discrete_autoconvolution μ k ≥
    ∑ i : Fin d, ∑ j : Fin d,
      if i.val ≥ d / 2 ∧ j.val ≥ d / 2 then μ i * μ j else 0 := by
  have h_d_eq : 2 * (d / 2) = d := by
    rcases h_even with ⟨k, hk⟩
    omega
  have h_pairs_in_window : ∀ (i j : Fin d), i.val ≥ d / 2 → j.val ≥ d / 2 →
      d ≤ i.val + j.val ∧ i.val + j.val ≤ 2 * d - 2 := by
    intro i j hi hj
    have hi_lt := i.isLt
    have hj_lt := j.isLt
    refine ⟨?_, ?_⟩
    · have : 2 * (d / 2) ≤ i.val + j.val := by omega
      omega
    · omega
  have h_eq : ∑ i : Fin d, ∑ j : Fin d,
      (if i.val ≥ d / 2 ∧ j.val ≥ d / 2 then μ i * μ j else 0) =
      ∑ k ∈ Finset.Icc d (2 * d - 2), ∑ i : Fin d, ∑ j : Fin d,
        if i.val ≥ d / 2 ∧ j.val ≥ d / 2 ∧ i.val + j.val = k then μ i * μ j else 0 := by
    rw [show ∑ k ∈ Finset.Icc d (2 * d - 2), ∑ i : Fin d, ∑ j : Fin d,
        (if i.val ≥ d / 2 ∧ j.val ≥ d / 2 ∧ i.val + j.val = k then μ i * μ j else 0) =
        ∑ i : Fin d, ∑ j : Fin d, ∑ k ∈ Finset.Icc d (2 * d - 2),
        (if i.val ≥ d / 2 ∧ j.val ≥ d / 2 ∧ i.val + j.val = k then μ i * μ j else 0) from by
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl ?_
      intro i _
      rw [Finset.sum_comm]]
    refine Finset.sum_congr rfl ?_
    intro i _
    refine Finset.sum_congr rfl ?_
    intro j _
    by_cases hij : i.val ≥ d / 2 ∧ j.val ≥ d / 2
    · have h_in_range : i.val + j.val ∈ Finset.Icc d (2 * d - 2) := by
        simp only [Finset.mem_Icc]
        exact h_pairs_in_window i j hij.1 hij.2
      have h_unique : ∑ k ∈ Finset.Icc d (2 * d - 2),
          (if i.val ≥ d / 2 ∧ j.val ≥ d / 2 ∧ i.val + j.val = k then μ i * μ j else 0) = μ i * μ j := by
        rw [Finset.sum_eq_single (i.val + j.val)]
        · simp [hij.1, hij.2]
        · intro b _ hb_ne
          have h_not_eq : ¬ i.val + j.val = b := fun h => hb_ne h.symm
          simp [hij.1, hij.2, h_not_eq]
        · intro h_not_in
          exact absurd h_in_range h_not_in
      rw [h_unique]
      simp [hij.1, hij.2]
    · push_neg at hij
      have h_lhs : (if i.val ≥ d / 2 ∧ j.val ≥ d / 2 then μ i * μ j else 0) = 0 := by
        rcases Classical.em (i.val ≥ d / 2) with hi | hi
        · simp [hi, hij hi]
        · simp [hi]
      have h_rhs : ∑ k ∈ Finset.Icc d (2 * d - 2),
          (if i.val ≥ d / 2 ∧ j.val ≥ d / 2 ∧ i.val + j.val = k then μ i * μ j else 0) = 0 := by
        apply Finset.sum_eq_zero
        intro k _
        rcases Classical.em (i.val ≥ d / 2) with hi | hi
        · simp [hi, hij hi]
        · simp [hi]
      rw [h_lhs, h_rhs]
  rw [h_eq]
  apply Finset.sum_le_sum
  intro k _
  unfold discrete_autoconvolution
  apply Finset.sum_le_sum
  intro i _
  apply Finset.sum_le_sum
  intro j _
  by_cases h_eq : i.val + j.val = k
  · by_cases h_ij : i.val ≥ d / 2 ∧ j.val ≥ d / 2
    · simp [h_eq, h_ij.1, h_ij.2]
    · push_neg at h_ij
      simp only [h_eq, and_true]
      rcases Classical.em (i.val ≥ d / 2) with hi | hi
      · simp only [hi, true_and]
        have hj : ¬ j.val ≥ d / 2 := by
          have h := h_ij hi
          omega
        simp [hj, mul_nonneg (hμ i) (hμ j)]
      · simp [hi, mul_nonneg (hμ i) (hμ j)]
  · simp [h_eq]

/-- Helper: ∑_{i,j < M} μ_i μ_j = (∑_{i < M} μ_i)^2. -/
private lemma sum_square_left {d : ℕ} (μ : Fin d → ℝ) (M : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, (if i.val < M ∧ j.val < M then μ i * μ j else 0) =
    (∑ i : Fin d, if i.val < M then μ i else 0) ^ 2 := by
  rw [sq]
  rw [Finset.sum_mul_sum]
  refine Finset.sum_congr rfl ?_
  intro i _
  refine Finset.sum_congr rfl ?_
  intro j _
  by_cases hi : i.val < M
  · by_cases hj : j.val < M
    · simp [hi, hj]
    · simp [hi, hj]
  · simp [hi]

/-- Helper: ∑_{i,j ≥ M} μ_i μ_j = (∑_{i ≥ M} μ_i)^2. -/
private lemma sum_square_right {d : ℕ} (μ : Fin d → ℝ) (M : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, (if i.val ≥ M ∧ j.val ≥ M then μ i * μ j else 0) =
    (∑ i : Fin d, if i.val ≥ M then μ i else 0) ^ 2 := by
  rw [sq]
  rw [Finset.sum_mul_sum]
  refine Finset.sum_congr rfl ?_
  intro i _
  refine Finset.sum_congr rfl ?_
  intro j _
  by_cases hi : i.val ≥ M
  · by_cases hj : j.val ≥ M
    · simp [hi, hj]
    · simp [hi, hj]
  · simp [hi]

/-- Helper: when left_mass μ ≥ sqrt(c/2), then (left_mass μ)^2 ≥ c/2. -/
private lemma left_mass_sq_bound {d : ℕ} (μ : Fin d → ℝ) (c_target : ℝ)
    (_hct : 0 < c_target)
    (h_left : left_mass μ ≥ Real.sqrt (c_target / 2)) :
    (left_mass μ) ^ 2 ≥ c_target / 2 := by
  have h_quot_nn : 0 ≤ c_target / 2 := by linarith
  have h_sqrt_nn : 0 ≤ Real.sqrt (c_target / 2) := Real.sqrt_nonneg _
  have h_left_nn : 0 ≤ left_mass μ := le_trans h_sqrt_nn h_left
  have h_sq_ge : (left_mass μ) ^ 2 ≥ Real.sqrt (c_target / 2) ^ 2 := by
    apply sq_le_sq'
    · linarith
    · exact h_left
  have h_sqrt_sq : Real.sqrt (c_target / 2) ^ 2 = c_target / 2 := Real.sq_sqrt h_quot_nn
  rw [h_sqrt_sq] at h_sq_ge
  exact h_sq_ge

/-- Helper: when right_mass μ ≥ sqrt(c/2), then (right_mass μ)^2 ≥ c/2. -/
private lemma right_mass_sq_bound {d : ℕ} (μ : Fin d → ℝ) (c_target : ℝ)
    (_hct : 0 < c_target)
    (h_right : right_mass μ ≥ Real.sqrt (c_target / 2)) :
    (right_mass μ) ^ 2 ≥ c_target / 2 := by
  have h_quot_nn : 0 ≤ c_target / 2 := by linarith
  have h_sqrt_nn : 0 ≤ Real.sqrt (c_target / 2) := Real.sqrt_nonneg _
  have h_right_nn : 0 ≤ right_mass μ := le_trans h_sqrt_nn h_right
  have h_sq_ge : (right_mass μ) ^ 2 ≥ Real.sqrt (c_target / 2) ^ 2 := by
    apply sq_le_sq'
    · linarith
    · exact h_right
  have h_sqrt_sq : Real.sqrt (c_target / 2) ^ 2 = c_target / 2 := Real.sq_sqrt h_quot_nn
  rw [h_sqrt_sq] at h_sq_ge
  exact h_sq_ge

/-- Helper: on the simplex with `Even d`, right_mass = 1 - left_mass. -/
private lemma left_plus_right_eq_sum {d : ℕ} (μ : Fin d → ℝ) :
    left_mass μ + right_mass μ = ∑ i, μ i := by
  unfold left_mass right_mass
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl ?_
  intro i _
  by_cases hi : i.val < d / 2
  · have hi_neg : ¬ i.val ≥ d / 2 := by omega
    simp [hi, hi_neg]
  · push_neg at hi
    simp [show ¬ (i.val < d / 2) from by omega, hi]

/-- **Asymmetry pruning theorem (left half):**
    If left_mass(μ) ≥ sqrt(c_target / 2), then the window (ell = d, s = 0)
    over the left half of the convolution support gives TV ≥ c_target. -/
theorem asymmetry_prune_sound {d : ℕ} (hd2 : d ≥ 2)
    (μ : Fin d → ℝ) (hμ : on_simplex μ) (c_target : ℝ) (hct : 0 < c_target)
    (h_left : left_mass μ ≥ Real.sqrt (c_target / 2)) :
    ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target := by
  refine ⟨d, 0, hd2, ?_⟩
  rw [full_left_window_eq μ hd2]
  have h_pairs_lb : ∑ k ∈ Finset.Icc 0 (d - 2), discrete_autoconvolution μ k ≥
      (left_mass μ) ^ 2 := by
    have hMle : (d / 2) ≤ d / 2 := le_refl _
    have h_pairs := subset_pairs_lower_bound μ hμ.1 hd2 (d / 2) hMle
    rw [sum_square_left μ (d / 2)] at h_pairs
    exact h_pairs
  have h_sq := left_mass_sq_bound μ c_target hct h_left
  linarith

/-- **Asymmetry pruning theorem (right half, requires Even d):**
    If right_mass(μ) ≥ sqrt(c_target / 2) and d is even, then the window
    (ell = d, s = d) over the right half of the convolution support gives TV ≥ c_target. -/
theorem asymmetry_prune_right {d : ℕ} (hd2 : d ≥ 2) (h_even : Even d)
    (μ : Fin d → ℝ) (hμ : on_simplex μ) (c_target : ℝ) (hct : 0 < c_target)
    (h_right : 1 - left_mass μ ≥ Real.sqrt (c_target / 2)) :
    ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target := by
  refine ⟨d, d, hd2, ?_⟩
  rw [right_window_eq μ hd2]
  -- Convert h_right into right_mass μ ≥ sqrt(c_target / 2)
  have h_lr : left_mass μ + right_mass μ = 1 := by
    rw [left_plus_right_eq_sum μ]; exact hμ.2
  have h_right' : right_mass μ ≥ Real.sqrt (c_target / 2) := by linarith
  have h_pairs_lb : ∑ k ∈ Finset.Icc d (2 * d - 2), discrete_autoconvolution μ k ≥
      (right_mass μ) ^ 2 := by
    have h_pairs := right_subset_pairs_lower_bound μ hμ.1 hd2 h_even
    rw [sum_square_right μ (d / 2)] at h_pairs
    exact h_pairs
  have h_sq := right_mass_sq_bound μ c_target hct h_right'
  linarith

/-- The asymmetry pruning rule: only compositions with left_frac in
    (1 - sqrt(c/2), sqrt(c/2)) need full window scanning. (Requires Even d for the right case.) -/
theorem asymmetry_needs_check {d : ℕ} (hd2 : d ≥ 2) (h_even : Even d)
    (μ : Fin d → ℝ) (hμ : on_simplex μ) (c_target : ℝ) (hct : 0 < c_target) :
    (left_mass μ ≥ Real.sqrt (c_target / 2) ∨
     left_mass μ ≤ 1 - Real.sqrt (c_target / 2)) →
    ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target := by
  rintro (h_left | h_right)
  · exact asymmetry_prune_sound hd2 μ hμ c_target hct h_left
  · apply asymmetry_prune_right hd2 h_even μ hμ c_target hct
    linarith

end -- noncomputable section
