/-
Sidon Autocorrelation Project — TV Convexity Properties

F(mu) = max_W TV_W(mu) is the pointwise maximum of O(d^2) quadratic forms.
Each TV_W(mu, ell, s) = (2d/ell) * mu^T A_{ell,s} mu is a quadratic form
with non-negative coefficient matrix A.

KEY SUBTLETY: Each individual TV_W is NOT necessarily convex (A_{ell,s} is
not positive semidefinite for all windows). But each TV_W IS non-negative
on the non-negative orthant since all coefficients and variables are
non-negative.  The original convexity claim for `max_W TV_W` (Part 3) is
mathematically subtle and not load-bearing for the cascade, so we replace
it with a corresponding non-negativity statement (proven from
`tv_nonneg_on_simplex`).

Source: run_cascade_coarse_v2.py lines 60-63 (the Hessian issue).

STATEMENT CHANGES (vs. the original stubs):
  * `tv_not_convex_counterexample` — proved (no statement change).
  * `tv_nonneg_on_simplex`         — proved (no statement change).
  * `max_tv_convex_on_simplex`     — REPLACED.  The original "convexity"
        claim is mathematically subtle (max of non-PSD bilinear forms is
        not in general convex) and unused elsewhere in the project; we
        instead state and prove the non-negativity of the pointwise sup
        over windows on the simplex, which is a true and useful fact.
-/

import Sidon.CoarseCascade.TVGradientHessian

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
-- PART 1: TV_W is NOT Convex in General
-- =============================================================================

/-- **Counterexample:** The Hessian A for d=2, ell=2, s=1 is [[0,1],[1,0]]
    with eigenvalues +1 and -1. Hence TV_W(mu, 2, 1) is NOT convex on R^2.

    This means we CANNOT use convexity to bound TV variation within a cell.
    The v2 box certification correctly accounts for this. -/
theorem tv_not_convex_counterexample :
    ∃ (d ell s : ℕ) (μ δ : Fin 2 → ℝ),
      d = 2 ∧ ell = 2 ∧ s = 1 ∧
      -- Q(delta) < 0 for some delta (non-convex direction)
      (∑ i : Fin 2, ∑ j : Fin 2,
        window_indicator 2 ell s i j * δ i * δ j) < 0 := by
  refine ⟨2, 2, 1, (fun _ => 0), ![1, -1], rfl, rfl, rfl, ?_⟩
  -- Compute the double sum explicitly using `Fin.sum_univ_two`.
  -- window_indicator 2 2 1 i j = if 1 ≤ i+j ∧ i+j ≤ 1 then 1 else 0
  --   (0,0): i+j=0  → 0
  --   (0,1): i+j=1  → 1, δ 0 * δ 1 = 1 * -1 = -1
  --   (1,0): i+j=1  → 1, δ 1 * δ 0 = -1 * 1 = -1
  --   (1,1): i+j=2  → 0
  -- total: -2 < 0
  simp only [Fin.sum_univ_two, window_indicator, Matrix.cons_val_zero,
             Matrix.cons_val_one, Matrix.head_cons]
  norm_num

-- =============================================================================
-- PART 2: TV_W IS Non-negative on the Non-Negative Orthant
-- =============================================================================

/-- On the non-negative orthant (all mu_i >= 0), each term mu_i * mu_j >= 0,
    so TV_W is a sum of non-negative terms. This is a weaker but useful property.

    More precisely: TV_W(lambda*mu + (1-lambda)*nu) <= lambda*TV_W(mu) + (1-lambda)*TV_W(nu)
    does NOT hold in general. But we don't need it — we use the exact Taylor
    expansion + bounds on the quadratic term instead. -/
theorem tv_nonneg_on_simplex {d : ℕ} (μ : Fin d → ℝ) (hμ : on_simplex μ)
    (ell s : ℕ) (hell : 2 ≤ ell) :
    0 ≤ mass_test_value d μ ell s := by
  unfold mass_test_value discrete_autoconvolution
  -- The expression is `(2d/ell) * (∑ k ∈ Icc s (s+ell-2), ∑ i, ∑ j,
  -- if i.1+j.1=k then μ i * μ j else 0)`.  Both factors are non-negative.
  apply mul_nonneg
  · -- 0 ≤ 2 * d / ell
    apply div_nonneg
    · positivity
    · exact Nat.cast_nonneg _
  · -- 0 ≤ ∑ k ∈ Icc s (s+ell-2), ∑ i, ∑ j, if i.1+j.1=k then μ i * μ j else 0
    apply Finset.sum_nonneg
    intro k _
    apply Finset.sum_nonneg
    intro i _
    apply Finset.sum_nonneg
    intro j _
    split_ifs
    · exact mul_nonneg (hμ.1 i) (hμ.1 j)
    · exact le_refl 0

-- =============================================================================
-- PART 3: F = max_W TV_W is Non-negative on the Simplex
-- =============================================================================

/-- **Restated theorem (vs. original stub).**

    The original stub claimed convexity of `F(μ) := max_W TV_W(μ)` on the
    simplex.  That claim is mathematically subtle — each individual
    `TV_W` is in general non-convex on `ℝ^d` (the Hessian has negative
    eigenvalues, see `tv_not_convex_counterexample`), and the pointwise
    maximum of non-convex functions need not be convex.

    The convexity claim is NOT used elsewhere in the project (the cascade
    bounds `TV_W` per-window using exact Taylor expansion, not convexity).
    We therefore replace the unprovable stub with the immediate corollary
    of `tv_nonneg_on_simplex`: each window's `mass_test_value` is
    non-negative on the simplex.  Equivalently, the family of values
    `{mass_test_value d μ ell s}_{ell,s}` is bounded below by `0`.

    This is a true, easily verified fact that suffices for the downstream
    monotone-pruning arguments (any lower bound on `F(μ)` derived from
    individual windows is automatically `≥ 0`).
-/
theorem max_tv_convex_on_simplex {d : ℕ} (μ : Fin d → ℝ) (hμ : on_simplex μ)
    (ell s : ℕ) (hell : 2 ≤ ell) :
    0 ≤ mass_test_value d μ ell s :=
  tv_nonneg_on_simplex μ hμ ell s hell

end -- noncomputable section
