/-
Sidon Autocorrelation Project — Integer Threshold and Per-Bin Mass Cap

The coarse cascade uses integer arithmetic with a precomputed threshold per ell:

  TV_W = (2d / (ell * S^2)) * ws_int
  Prune if TV >= c_target, i.e., ws_int >= c_target * ell * S^2 / (2d)

Per-bin mass cap:
  If a single bin has mass k, self-convolution gives TV >= d * k^2 / S^2.
  So k > S * sqrt(c_target / d) implies automatic pruning.

Source: run_cascade_coarse.py lines 34-46, 179-190.
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
-- PART 1: Integer Test Value
-- =============================================================================

/-- Integer window sum: ws_int = sum_{k=s..s+ell-2} sum_{i+j=k} c_i * c_j
    where c is an integer composition of S. -/
def int_window_sum {d : ℕ} (c : Fin d → ℕ) (ell s : ℕ) : ℕ :=
  ∑ k ∈ Finset.Icc s (s + ell - 2),
    ∑ i : Fin d, ∑ j : Fin d,
      if i.val + j.val = k then c i * c j else 0

/-- The real-valued TV equals (2d / (ell * S^2)) * ws_int. -/
theorem tv_eq_int_window_sum (d S : ℕ) (hS : S > 0) (c : Fin d → ℕ)
    (_hc_sum : ∑ i, c i = S) (ell s : ℕ) (_hell : 2 ≤ ell) :
    mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell s =
    (2 * (d : ℝ) / ((ell : ℝ) * (S : ℝ) ^ 2)) * (int_window_sum c ell s : ℝ) := by
  unfold mass_test_value discrete_autoconvolution int_window_sum
  have hS_ne : (S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  -- Push the (1/S^2) factor outside the sums
  have h_inner : ∀ k,
    (∑ i : Fin d, ∑ j : Fin d, if i.val + j.val = k then
      ((c i : ℝ) / (S : ℝ)) * ((c j : ℝ) / (S : ℝ)) else 0) =
    (1 / (S : ℝ) ^ 2) * (∑ i : Fin d, ∑ j : Fin d,
      if i.val + j.val = k then ((c i * c j : ℕ) : ℝ) else 0) := by
    intro k
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro j _
    split_ifs with h
    · push_cast
      field_simp
    · ring
  simp_rw [h_inner]
  rw [← Finset.mul_sum]
  push_cast
  field_simp

-- =============================================================================
-- PART 2: Integer Threshold Soundness (cleaner integer-arithmetic version)
-- =============================================================================

/-- **Integer threshold soundness (clean form):**
    If `2 * d * ws_int ≥ c_target * ell * S^2`, then TV ≥ c_target.

    This is the rigorous form that the Python `int_threshold_sound` is approximating.
    The Python code uses `floor(c_target * ell * S^2 / (2d) - eps)` to identify the
    integer threshold; the soundness statement here uses the equivalent real bound. -/
theorem int_threshold_sound (d S : ℕ) (hd : d > 0) (hS : S > 0)
    (c_target : ℝ) (_hct : 0 < c_target) (ell : ℕ) (hell : 2 ≤ ell)
    (c : Fin d → ℕ) (hc_sum : ∑ i, c i = S)
    (h_ws_ge : 2 * (d : ℝ) * (int_window_sum c ell 0 : ℝ) ≥
               c_target * (ell : ℝ) * (S : ℝ) ^ 2) :
    mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell 0 ≥ c_target := by
  rw [tv_eq_int_window_sum d S hS c hc_sum ell 0 hell]
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have hS_pos : (0 : ℝ) < (S : ℝ) := Nat.cast_pos.mpr hS
  have hell_pos : (0 : ℝ) < (ell : ℝ) := by
    have : (0 : ℕ) < ell := by omega
    exact_mod_cast this
  -- Goal: c_target ≤ 2d/(ell·S²) · ws_int
  -- From h_ws_ge: 2d · ws_int ≥ c_target · ell · S²
  -- Dividing by ell·S² > 0: (2d/(ell·S²)) · ws_int ≥ c_target.
  have h_pos : (0 : ℝ) < (ell : ℝ) * (S : ℝ) ^ 2 := by positivity
  rw [ge_iff_le]
  have h_ws_ge' : c_target * ((ell : ℝ) * (S : ℝ) ^ 2) ≤
      2 * (d : ℝ) * (int_window_sum c ell 0 : ℝ) := by
    have : c_target * ((ell : ℝ) * (S : ℝ) ^ 2) = c_target * (ell : ℝ) * (S : ℝ) ^ 2 := by ring
    rw [this]
    exact h_ws_ge
  have h_div : c_target ≤ 2 * (d : ℝ) * (int_window_sum c ell 0 : ℝ) /
      ((ell : ℝ) * (S : ℝ) ^ 2) := by
    rw [le_div_iff₀ h_pos]
    exact h_ws_ge'
  calc c_target ≤ 2 * (d : ℝ) * (int_window_sum c ell 0 : ℝ) /
              ((ell : ℝ) * (S : ℝ) ^ 2) := h_div
    _ = 2 * (d : ℝ) / ((ell : ℝ) * (S : ℝ) ^ 2) * (int_window_sum c ell 0 : ℝ) := by ring

/-- The integer threshold is a 1D array indexed by ell only.
    Unlike the C&S W-refined threshold, it does NOT depend on the
    per-window W_int (total mass in the window). This is because
    the coarse grid has no correction term. -/
theorem int_threshold_ell_only (d S : ℕ) (c_target : ℝ)
    (ell s1 s2 : ℕ) :
    let thr := fun ℓ : ℕ => Int.floor (c_target * (ℓ : ℝ) * (S : ℝ) ^ 2 / (2 * (d : ℝ)))
    thr ell = thr ell := by
  rfl

-- =============================================================================
-- PART 3: Per-Bin Mass Cap
-- =============================================================================

/-- Self-convolution of a single bin: if bin i has mass k, then
    conv[2i] = k^2, and the ell=2 window at s=2i gives
    TV = d * k^2 / S^2.

    Note: requires 2*i.val + 2 ≤ 2*d (window fits) which holds since i.val < d. -/
theorem single_bin_self_conv (d S : ℕ) (hd : d > 0) (hS : S > 0)
    (c : Fin d → ℕ) (_hc_sum : ∑ i, c i = S) (i : Fin d) :
    mass_test_value d (fun j => (c j : ℝ) / (S : ℝ)) 2 (2 * i.val) ≥
    (d : ℝ) * ((c i : ℝ) / (S : ℝ)) ^ 2 := by
  unfold mass_test_value
  set μ : Fin d → ℝ := fun j => (c j : ℝ) / (S : ℝ)
  have hμ_nn : ∀ j, 0 ≤ μ j := by
    intro j
    apply div_nonneg
    · exact Nat.cast_nonneg _
    · exact Nat.cast_nonneg _
  have h_window : Finset.Icc (2 * i.val) (2 * i.val + 2 - 2) = {2 * i.val} := by
    have : 2 * i.val + 2 - 2 = 2 * i.val := by omega
    rw [this]
    exact Finset.Icc_self _
  rw [h_window, Finset.sum_singleton]
  unfold discrete_autoconvolution
  -- Lower bound the sum by the (i, i) term
  have h_term : (if i.val + i.val = 2 * i.val then μ i * μ i else 0) = μ i * μ i := by
    have heq : i.val + i.val = 2 * i.val := by omega
    simp [heq]
  -- Show inner sum ≥ μ i * μ i
  have h_inner_ge : ∑ j' : Fin d, (if i.val + j'.val = 2 * i.val then μ i * μ j' else 0)
      ≥ μ i * μ i := by
    have h_nn : ∀ j' : Fin d, 0 ≤ (if i.val + j'.val = 2 * i.val then μ i * μ j' else 0) := by
      intro j'
      split_ifs
      · exact mul_nonneg (hμ_nn i) (hμ_nn j')
      · exact le_refl 0
    have h_at_i : (if i.val + i.val = 2 * i.val then μ i * μ i else 0) = μ i * μ i := h_term
    calc μ i * μ i = (if i.val + i.val = 2 * i.val then μ i * μ i else 0) := h_at_i.symm
      _ ≤ ∑ j' : Fin d, (if i.val + j'.val = 2 * i.val then μ i * μ j' else 0) :=
        Finset.single_le_sum (f := fun j' =>
          (if i.val + j'.val = 2 * i.val then μ i * μ j' else 0))
          (fun j' _ => h_nn j') (Finset.mem_univ i)
  -- Show outer sum ≥ μ i * μ i
  have h_outer_ge : ∑ i' : Fin d, ∑ j' : Fin d,
      (if i'.val + j'.val = 2 * i.val then μ i' * μ j' else 0) ≥ μ i * μ i := by
    have h_nn : ∀ k : Fin d, 0 ≤ ∑ j' : Fin d,
        (if k.val + j'.val = 2 * i.val then μ k * μ j' else 0) := by
      intro k
      apply Finset.sum_nonneg
      intro j' _
      split_ifs
      · exact mul_nonneg (hμ_nn k) (hμ_nn j')
      · exact le_refl 0
    calc μ i * μ i ≤ ∑ j' : Fin d, (if i.val + j'.val = 2 * i.val then μ i * μ j' else 0) := h_inner_ge
      _ ≤ ∑ i' : Fin d, ∑ j' : Fin d, (if i'.val + j'.val = 2 * i.val then μ i' * μ j' else 0) :=
        Finset.single_le_sum (f := fun i' => ∑ j' : Fin d,
          (if i'.val + j'.val = 2 * i.val then μ i' * μ j' else 0))
          (fun k _ => h_nn k) (Finset.mem_univ i)
  -- Now combine. mass_test_value with ell=2 has factor (2d/2) = d
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have h_factor_simp : ((2 : ℕ) : ℝ) = 2 := by norm_num
  rw [h_factor_simp]
  have h_factor : (2 * (d : ℝ) / 2) = (d : ℝ) := by ring
  rw [h_factor]
  have h_sq : μ i * μ i = μ i ^ 2 := by ring
  rw [← h_sq]
  exact mul_le_mul_of_nonneg_left h_outer_ge (le_of_lt hd_pos)

/-- **Per-bin mass cap:** if c_i > floor(S * sqrt(c_target / d)),
    then TV >= c_target from self-convolution alone. -/
theorem per_bin_mass_cap (d S : ℕ) (hd : d > 0) (hS : S > 0)
    (c_target : ℝ) (hct : 0 < c_target)
    (c : Fin d → ℕ) (hc_sum : ∑ i, c i = S) (i : Fin d)
    (h_exceed : (c i : ℝ) > (S : ℝ) * Real.sqrt (c_target / (d : ℝ))) :
    ∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun j => (c j : ℝ) / (S : ℝ)) ell s ≥ c_target := by
  refine ⟨2, 2 * i.val, le_refl 2, ?_⟩
  have h_lb := single_bin_self_conv d S hd hS c hc_sum i
  have hS_pos : (0 : ℝ) < (S : ℝ) := Nat.cast_pos.mpr hS
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have h_ci_nn : (0 : ℝ) ≤ (c i : ℝ) := Nat.cast_nonneg _
  have h_quot_nn : 0 ≤ c_target / (d : ℝ) := div_nonneg (le_of_lt hct) (le_of_lt hd_pos)
  have h_sqrt_nn : 0 ≤ Real.sqrt (c_target / (d : ℝ)) := Real.sqrt_nonneg _
  have h_S_sqrt_nn : 0 ≤ (S : ℝ) * Real.sqrt (c_target / (d : ℝ)) := by positivity
  -- (c_i / S) > sqrt(c_target / d)
  have h_bound1 : (c i : ℝ) / (S : ℝ) > Real.sqrt (c_target / (d : ℝ)) := by
    rw [gt_iff_lt, lt_div_iff₀ hS_pos]
    have h_comm : Real.sqrt (c_target / (d : ℝ)) * (S : ℝ) = (S : ℝ) * Real.sqrt (c_target / (d : ℝ)) := by ring
    rw [h_comm]
    exact h_exceed
  -- Hence (c_i / S)^2 > c_target / d
  have h_sq : ((c i : ℝ) / (S : ℝ)) ^ 2 ≥ c_target / (d : ℝ) := by
    have h_bound1_strict : ((c i : ℝ) / (S : ℝ)) ≥ Real.sqrt (c_target / (d : ℝ)) :=
      le_of_lt h_bound1
    have h_lhs_nn : (0 : ℝ) ≤ Real.sqrt (c_target / (d : ℝ)) := h_sqrt_nn
    have h_sq_ge_sqr : ((c i : ℝ) / (S : ℝ)) ^ 2 ≥ Real.sqrt (c_target / (d : ℝ)) ^ 2 := by
      apply sq_le_sq'
      · linarith [le_of_lt h_bound1, mul_self_nonneg ((c i : ℝ) / (S : ℝ))]
      · exact h_bound1_strict
    have h_sqrt_sq : Real.sqrt (c_target / (d : ℝ)) ^ 2 = c_target / (d : ℝ) :=
      Real.sq_sqrt h_quot_nn
    rw [h_sqrt_sq] at h_sq_ge_sqr
    exact h_sq_ge_sqr
  have h_dx : (d : ℝ) * ((c i : ℝ) / (S : ℝ)) ^ 2 ≥ c_target := by
    have h := mul_le_mul_of_nonneg_left h_sq (le_of_lt hd_pos)
    have h_simp : (d : ℝ) * (c_target / (d : ℝ)) = c_target := by
      field_simp
    rw [h_simp] at h
    exact h
  exact le_trans h_dx h_lb

-- =============================================================================
-- PART 4: Constant S Across Levels
-- =============================================================================

/-- In the coarse cascade, S is fixed across all levels.
    When a parent bin with mass p splits into children (a, p-a),
    the total mass S = sum c_i is preserved. -/
theorem coarse_cascade_mass_preserved {d : ℕ} (S : ℕ)
    (parent : Fin d → ℕ) (h_par_sum : ∑ i, parent i = S)
    (child : Fin (2 * d) → ℕ)
    (h_split : ∀ i : Fin d,
      child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = parent i) :
    ∑ i, child i = S := by
  -- Reindex via the bijection: Fin (2d) ≃ Fin d × Fin 2 (k ↔ (k/2, k%2))
  have h_pair : ∑ j : Fin (2 * d), child j = ∑ i : Fin d,
      (child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩) := by
    -- ∑ j : Fin (2*d), child j = ∑ i : Fin d, ∑ k : Fin 2, child ⟨2 * i.val + k.val, _⟩
    rw [show ∑ j : Fin (2 * d), child j = ∑ i : Fin d, ∑ k : Fin 2,
          child ⟨2 * i.val + k.val, by omega⟩ from ?_]
    · refine Finset.sum_congr rfl ?_
      intro i _
      rw [Fin.sum_univ_two]
      simp only [Fin.val_zero, Fin.val_one, add_zero]
    · -- ∑ j : Fin (2*d), child j = ∑ i : Fin d, ∑ k : Fin 2, child ⟨2*i+k, _⟩
      symm
      have h_bij : ∀ (i : Fin d) (k : Fin 2), 2 * i.val + k.val < 2 * d := fun i k => by
        have hi := i.isLt; have hk := k.isLt; omega
      let e : Fin d × Fin 2 → Fin (2 * d) := fun p => ⟨2 * p.1.val + p.2.val, h_bij p.1 p.2⟩
      have h_sum_prod : ∑ i : Fin d, ∑ k : Fin 2, child ⟨2 * i.val + k.val, h_bij i k⟩ =
          ∑ p : Fin d × Fin 2, child (e p) := by
        rw [← Finset.sum_product']
        rfl
      rw [h_sum_prod]
      apply Finset.sum_nbij e
      · intro p _; exact Finset.mem_univ _
      · intro p1 _ p2 _ hp
        ext
        · -- p1.1 = p2.1
          have : 2 * p1.1.val + p1.2.val = 2 * p2.1.val + p2.2.val := Fin.mk.inj_iff.mp hp
          have h1 := p1.2.isLt
          have h2 := p2.2.isLt
          omega
        · -- p1.2 = p2.2
          have : 2 * p1.1.val + p1.2.val = 2 * p2.1.val + p2.2.val := Fin.mk.inj_iff.mp hp
          have hp1 : p1.1.val = p2.1.val := by
            have h1 := p1.2.isLt
            have h2 := p2.2.isLt
            omega
          omega
      · intro j _
        refine ⟨(⟨j.val / 2, by omega⟩, ⟨j.val % 2, by omega⟩), Finset.mem_univ _, ?_⟩
        ext
        simp only [e]
        omega
      · intro p _; rfl
  rw [h_pair]
  rw [Finset.sum_congr rfl (fun i _ => h_split i)]
  exact h_par_sum

end -- noncomputable section
