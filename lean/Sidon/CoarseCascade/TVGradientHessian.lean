/-
Sidon Autocorrelation Project — TV Gradient and Hessian (Proof Stubs)

TV_W(mu) is a DEGREE-2 POLYNOMIAL in mu, so its Taylor expansion is exact:

  TV_W(mu + delta) = TV_W(mu) + grad(TV_W) . delta + (2d/ell) * Q(delta)

where Q(delta) = delta^T A delta, A_{ij} = 1_{s <= i+j <= s+ell-2}.

Key facts:
  - grad_i = (4d/ell) * sum_{j: s <= i+j <= s+ell-2} mu_j
  - H_{ij} = (4d/ell) * 1_{s <= i+j <= s+ell-2}  (constant Hessian)
  - Taylor expansion is EXACT (no remainder) since TV is quadratic.

Source: run_cascade_coarse_v2.py lines 56-72, proof/coarse_cascade_method.md Section 7.
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
-- PART 1: TV is a Quadratic Form
-- =============================================================================

/-- The window indicator matrix A_{ij} = 1 if s <= i+j <= s+ell-2, else 0. -/
def window_indicator (d ell s : ℕ) (i j : Fin d) : ℝ :=
  if s ≤ i.val + j.val ∧ i.val + j.val ≤ s + ell - 2 then 1 else 0

/-- TV_W(mu, ell, s) = (2d/ell) * mu^T A mu where A is the window indicator matrix.
    This establishes TV as a quadratic form. -/
theorem tv_as_quadratic_form (d : ℕ) (μ : Fin d → ℝ) (ell s : ℕ) (hell : 2 ≤ ell) :
    mass_test_value d μ ell s =
    (2 * (d : ℝ) / (ell : ℝ)) *
      ∑ i : Fin d, ∑ j : Fin d,
        window_indicator d ell s i j * μ i * μ j := by
  unfold mass_test_value discrete_autoconvolution window_indicator
  congr 1
  -- Goal: ∑ k ∈ Icc s (s+ell-2), ∑ i, ∑ j, (if i.1+j.1=k then μ i * μ j else 0)
  --     = ∑ i, ∑ j, (if s ≤ i.1+j.1 ∧ i.1+j.1 ≤ s+ell-2 then 1 else 0) * μ i * μ j
  rw [Finset.sum_comm]
  -- ∑ i, ∑ k ∈ Icc s (s+ell-2), ∑ j, (if i.1+j.1=k then μ i * μ j else 0)
  refine Finset.sum_congr rfl ?_
  intro i _
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl ?_
  intro j _
  -- Goal: ∑ k ∈ Icc s (s+ell-2), (if i.1+j.1 = k then μ i * μ j else 0)
  --     = (if s ≤ i.1+j.1 ∧ i.1+j.1 ≤ s+ell-2 then 1 else 0) * μ i * μ j
  rw [Finset.sum_ite_eq (Finset.Icc s (s + ell - 2)) (i.val + j.val) (fun _ => μ i * μ j)]
  -- Goal: (if i.1+j.1 ∈ Icc s (s+ell-2) then μ i * μ j else 0)
  --     = (if s ≤ i.1+j.1 ∧ i.1+j.1 ≤ s+ell-2 then 1 else 0) * μ i * μ j
  by_cases h : s ≤ i.val + j.val ∧ i.val + j.val ≤ s + ell - 2
  · have hmem : i.val + j.val ∈ Finset.Icc s (s + ell - 2) := by
      rw [Finset.mem_Icc]; exact h
    simp [hmem, h]
  · have hmem : ¬ (i.val + j.val ∈ Finset.Icc s (s + ell - 2)) := by
      rw [Finset.mem_Icc]; exact h
    simp [hmem, h]

-- =============================================================================
-- PART 2: Gradient of TV
-- =============================================================================

/-- The gradient of TV_W with respect to mu_i:
    grad_i(mu) = (4d/ell) * sum_{j: s <= i+j <= s+ell-2} mu_j

    This comes from differentiating the quadratic form: d/d(mu_i) [mu^T A mu] = 2*(A*mu)_i,
    and A is symmetric. -/
def tv_gradient (d : ℕ) (μ : Fin d → ℝ) (ell s : ℕ) (i : Fin d) : ℝ :=
  (4 * (d : ℝ) / (ell : ℝ)) *
    ∑ j : Fin d, if s ≤ i.val + j.val ∧ i.val + j.val ≤ s + ell - 2
                 then μ j else 0

/-- **Exact Taylor expansion** for TV_W (quadratic, so no remainder term):

    TV_W(mu + delta) = TV_W(mu) + grad . delta + (2d/ell) * Q(delta)

    where Q(delta) = sum_{k in window} sum_{i+j=k} delta_i * delta_j.

    This is EXACT because TV_W is a degree-2 polynomial.

    Source: run_cascade_coarse_v2.py lines 57-59. -/
theorem tv_taylor_exact (d : ℕ) (μ δ : Fin d → ℝ)
    (ell s : ℕ) (hell : 2 ≤ ell) :
    mass_test_value d (fun i => μ i + δ i) ell s =
    mass_test_value d μ ell s +
    (∑ i : Fin d, tv_gradient d μ ell s i * δ i) +
    (2 * (d : ℝ) / (ell : ℝ)) *
      (∑ i : Fin d, ∑ j : Fin d,
        window_indicator d ell s i j * δ i * δ j) := by
  -- Abbreviate the constant factor and the indicator.
  set C : ℝ := (2 * (d : ℝ) / (ell : ℝ)) with hC
  set W : Fin d → Fin d → ℝ := fun i j => window_indicator d ell s i j with hW
  -- Rewrite both TV-instances as quadratic forms in W.
  rw [tv_as_quadratic_form d (fun i => μ i + δ i) ell s hell,
      tv_as_quadratic_form d μ ell s hell]
  -- After the rewrites, the goal (modulo defeq W = window_indicator) is:
  --   C * Σ Σ W i j * (μ i + δ i) * (μ j + δ j)
  --   =  C * Σ Σ W i j * μ i * μ j
  --     + Σ tv_gradient d μ ell s i * δ i
  --     + C * Σ Σ W i j * δ i * δ j
  -- We rewrite the bilinear product (μ + δ)(μ + δ) into 4 terms and split the sum.
  -- Step 1: split the LHS double sum into Σ Σ μμ + Σ Σ μδ + Σ Σ δμ + Σ Σ δδ.
  have h_split_lhs :
      (∑ i : Fin d, ∑ j : Fin d, W i j * (μ i + δ i) * (μ j + δ j)) =
      (∑ i : Fin d, ∑ j : Fin d, W i j * μ i * μ j) +
      (∑ i : Fin d, ∑ j : Fin d, W i j * μ i * δ j) +
      (∑ i : Fin d, ∑ j : Fin d, W i j * δ i * μ j) +
      (∑ i : Fin d, ∑ j : Fin d, W i j * δ i * δ j) := by
    -- Each inner sum equals the corresponding 4-term split.
    have h_inner : ∀ i : Fin d,
        (∑ j : Fin d, W i j * (μ i + δ i) * (μ j + δ j)) =
        (∑ j : Fin d, W i j * μ i * μ j) +
        (∑ j : Fin d, W i j * μ i * δ j) +
        (∑ j : Fin d, W i j * δ i * μ j) +
        (∑ j : Fin d, W i j * δ i * δ j) := by
      intro i
      rw [show (∑ j : Fin d, W i j * (μ i + δ i) * (μ j + δ j)) =
            (∑ j : Fin d, (W i j * μ i * μ j + W i j * μ i * δ j +
              W i j * δ i * μ j + W i j * δ i * δ j)) from by
        refine Finset.sum_congr rfl ?_
        intro j _; ring]
      simp [Finset.sum_add_distrib]
    -- Sum h_inner over i, then split the outer sum.
    rw [show (∑ i : Fin d, ∑ j : Fin d, W i j * (μ i + δ i) * (μ j + δ j)) =
          (∑ i : Fin d, ((∑ j : Fin d, W i j * μ i * μ j) +
            (∑ j : Fin d, W i j * μ i * δ j) +
            (∑ j : Fin d, W i j * δ i * μ j) +
            (∑ j : Fin d, W i j * δ i * δ j))) from by
      refine Finset.sum_congr rfl ?_
      intro i _; exact h_inner i]
    simp [Finset.sum_add_distrib]
  rw [h_split_lhs]
  -- Step 2: relate the two cross-sums via i↔j symmetry of W (W_ij = W_ji).
  have hW_symm : ∀ i j : Fin d, W i j = W j i := by
    intro i j
    simp only [hW, window_indicator]
    rw [Nat.add_comm]
  -- Σ Σ W μ_i δ_j = Σ Σ W_ji μ_j δ_i (swap names)
  --              = Σ Σ W δ_j μ_i  (no, careful)
  -- We just need: Σ Σ W_ij μ_i δ_j = Σ Σ W_ij δ_i μ_j.
  have h_swap :
      (∑ i : Fin d, ∑ j : Fin d, W i j * μ i * δ j) =
      (∑ i : Fin d, ∑ j : Fin d, W i j * δ i * μ j) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl ?_
    intro j _
    refine Finset.sum_congr rfl ?_
    intro i _
    rw [hW_symm i j]
    ring
  rw [h_swap]
  -- Now the cross part becomes 2 * Σ Σ W δ_i μ_j.
  -- Step 3: Show Σ tv_gradient_i * δ_i = C * (2 * Σ Σ W δ_i μ_j).
  have h_grad_eq :
      (∑ i : Fin d, tv_gradient d μ ell s i * δ i) =
      (2 * C) * (∑ i : Fin d, ∑ j : Fin d, W i j * δ i * μ j) := by
    -- Unfold tv_gradient and show the indicator equals W * μ.
    unfold tv_gradient
    have h_grad_unfold : ∀ i : Fin d,
        (∑ j : Fin d,
          (if s ≤ i.val + j.val ∧ i.val + j.val ≤ s + ell - 2 then μ j else 0)) =
        (∑ j : Fin d, W i j * μ j) := by
      intro i
      refine Finset.sum_congr rfl ?_
      intro j _
      simp only [hW, window_indicator]
      by_cases h : s ≤ i.val + j.val ∧ i.val + j.val ≤ s + ell - 2
      · simp [h]
      · simp [h]
    simp_rw [h_grad_unfold]
    -- Goal:
    --   ∑ i, (4d/ell) * (∑ j, W i j * μ j) * δ i = (2 * C) * (∑ i, ∑ j, W i j * δ i * μ j)
    -- Reduce both sides to ∑ i, ∑ j, (4d/ell) * (W i j * δ i * μ j) and compare.
    have hC4 : (4 * (d : ℝ) / (ell : ℝ)) = 2 * C := by
      simp only [hC]; ring
    -- Push the (4d/ell) factor through both sums on each side and compare termwise.
    rw [show (∑ i : Fin d, (4 * (d : ℝ) / (ell : ℝ)) *
                  (∑ j : Fin d, W i j * μ j) * δ i) =
        (∑ i : Fin d, ∑ j : Fin d,
          (4 * (d : ℝ) / (ell : ℝ)) * (W i j * δ i * μ j)) from by
      refine Finset.sum_congr rfl ?_
      intro i _
      rw [Finset.mul_sum, Finset.sum_mul]
      refine Finset.sum_congr rfl ?_
      intro j _
      ring]
    rw [show (2 * C) * (∑ i : Fin d, ∑ j : Fin d, W i j * δ i * μ j) =
        (∑ i : Fin d, ∑ j : Fin d,
          (4 * (d : ℝ) / (ell : ℝ)) * (W i j * δ i * μ j)) from by
      rw [hC4, Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro i _
      rw [Finset.mul_sum]]
  rw [h_grad_eq]
  ring

/-- Gradient correctness (Taylor first-order form).

    The original statement `(TV(μ+tδ) - TV(μ))/t = grad·δ + t * Q(δ)` is malformed
    at t = 0 (LHS would be 0/0).  We instead state the equivalent **multiplicative**
    form (from `tv_taylor_exact` applied to `t·δ`):

      TV(μ + tδ) = TV(μ) + t * (grad · δ) + t² * (2d/ell) * Q(δ).

    This is well-defined for all t ∈ ℝ and immediately implies the original
    quotient identity whenever t ≠ 0. -/
theorem tv_gradient_correct (d : ℕ) (μ δ : Fin d → ℝ)
    (ell s : ℕ) (hell : 2 ≤ ell) (t : ℝ) :
    mass_test_value d (fun i => μ i + t * δ i) ell s =
    mass_test_value d μ ell s +
    t * (∑ i : Fin d, tv_gradient d μ ell s i * δ i) +
    t ^ 2 * (2 * (d : ℝ) / (ell : ℝ)) * ∑ i : Fin d, ∑ j : Fin d,
      window_indicator d ell s i j * δ i * δ j := by
  -- Apply tv_taylor_exact with δ replaced by t·δ.
  have h := tv_taylor_exact d μ (fun i => t * δ i) ell s hell
  -- h : TV(μ + t·δ) = TV(μ) + Σ grad_i * (t·δ_i)
  --                    + (2d/ell) * Σ ∑ W_ij * (t δ_i) * (t δ_j)
  rw [h]
  -- Simplify each piece.
  have h_grad :
      (∑ i : Fin d, tv_gradient d μ ell s i * (t * δ i)) =
      t * (∑ i : Fin d, tv_gradient d μ ell s i * δ i) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _; ring
  have h_q :
      (2 * (d : ℝ) / (ell : ℝ)) *
        (∑ i : Fin d, ∑ j : Fin d,
          window_indicator d ell s i j * (t * δ i) * (t * δ j)) =
      t ^ 2 * (2 * (d : ℝ) / (ell : ℝ)) *
        ∑ i : Fin d, ∑ j : Fin d,
          window_indicator d ell s i j * δ i * δ j := by
    rw [show (∑ i : Fin d, ∑ j : Fin d,
              window_indicator d ell s i j * (t * δ i) * (t * δ j)) =
        t ^ 2 * ∑ i : Fin d, ∑ j : Fin d,
              window_indicator d ell s i j * δ i * δ j from by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro i _
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro j _
      ring]
    ring
  rw [h_grad, h_q]

-- =============================================================================
-- PART 3: Hessian of TV (Constant)
-- =============================================================================

/-- The Hessian of TV_W is the CONSTANT matrix:
    H_{ij} = (4d/ell) * 1_{s <= i+j <= s+ell-2}

    This is independent of mu (TV is quadratic, so the Hessian is constant). -/
def tv_hessian (d ell s : ℕ) (i j : Fin d) : ℝ :=
  (4 * (d : ℝ) / (ell : ℝ)) * window_indicator d ell s i j

/-- The Hessian is symmetric: H_{ij} = H_{ji}. -/
theorem tv_hessian_symmetric (d ell s : ℕ) (i j : Fin d) :
    tv_hessian d ell s i j = tv_hessian d ell s j i := by
  unfold tv_hessian window_indicator
  rw [Nat.add_comm i.val j.val]

/-- The Hessian has non-negative entries (but is NOT necessarily positive semidefinite).
    Counterexample: d=2, ell=2, s=1 gives A = [[0,1],[1,0]], eigenvalues +1 and -1. -/
theorem tv_hessian_entries_nonneg (d ell s : ℕ) (i j : Fin d) :
    0 ≤ tv_hessian d ell s i j := by
  unfold tv_hessian window_indicator
  apply mul_nonneg
  · apply div_nonneg
    · positivity
    · exact Nat.cast_nonneg _
  · split_ifs
    · exact zero_le_one
    · exact le_refl 0

-- =============================================================================
-- PART 4: Gradient Non-negativity on the Simplex
-- =============================================================================

/-- Gradient entries are non-negative when mu is on the simplex.
    grad_i = (4d/ell) * sum of non-negative masses. -/
theorem tv_gradient_nonneg (d : ℕ) (μ : Fin d → ℝ) (hμ : on_simplex μ)
    (ell s : ℕ) (i : Fin d) :
    0 ≤ tv_gradient d μ ell s i := by
  unfold tv_gradient
  apply mul_nonneg
  · apply div_nonneg
    · positivity
    · exact Nat.cast_nonneg _
  · apply Finset.sum_nonneg
    intro j _
    split_ifs
    · exact hμ.1 j
    · exact le_refl 0

end -- noncomputable section
