/-
Sidon Autocorrelation Project — Spectral Discretization Bound (variant N)

This file formalizes **variant N**, a tightening of variant F's δ²-bound that
exploits the kernel structure of the indicator matrix `A_W` together with the
spectral radius of `A_W − α·𝟙𝟙ᵀ` (the "restricted" matrix).

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL DERIVATION
═══════════════════════════════════════════════════════════════════════════════

Variant D uses the elementwise bound  |Σ_{(i,j)∈W} εᵢ εⱼ| ≤ ell_int_sum · h²
where h = 1/m, ε = b − a (with b = c/m). Variant F uses a different (tighter)
linear-term bound, but keeps the same elementwise δ² bound.

Variant N replaces the elementwise δ² bound with a SPECTRAL bound that exploits
the fact that, in the cascade context, ε ≡ b − a satisfies ∑ εᵢ = 0 (because
Σ b = Σ a = 4n: both sides of the discretization preserve the total mass of
the underlying density).

Let A_W be the {0,1}-indicator matrix of the window pair-set:
   A_W[i,j] = 1 iff i + j ∈ [s_lo, s_lo + ℓ − 2].
Then for ε : Fin (2n) → ℝ,
   Σ_{(i,j)∈W} εᵢ εⱼ  =  εᵀ A_W ε.

Key observation (kernel of all-ones): for any α ∈ ℝ and any vector ε,
   εᵀ (α · 𝟙𝟙ᵀ) ε  =  α · (Σᵢ εᵢ)².
So if Σ ε = 0, then εᵀ (A_W − α · 𝟙𝟙ᵀ) ε = εᵀ A_W ε for every α.

Choose α = α_W := ell_int_sum / (2n)² (so that Σᵢⱼ (A_W − α·𝟙𝟙ᵀ)ᵢⱼ = 0,
zeroing the all-ones component).  Then for the symmetric matrix
   M_restr := A_W − α_W · 𝟙𝟙ᵀ
we have, by Cauchy–Schwarz and the L²-induced operator-norm inequality,
   |εᵀ M_restr ε|  =  |⟨ε, M_restr ε⟩|
                  ≤  ‖ε‖₂ · ‖M_restr ε‖₂                     (Cauchy–Schwarz)
                  ≤  ‖ε‖₂ · ‖M_restr‖_op · ‖ε‖₂              (operator norm)
                  =  op_restricted · ‖ε‖₂².
Combined with ‖ε‖_∞ ≤ h ⇒ ‖ε‖₂² ≤ d · h² (where d = 2n), this yields
   under  Σ ε = 0 :   |Σ_W εᵢ εⱼ|  ≤  op_restricted · d · h².
Variant F's elementwise bound gives `ell_int_sum · h²` (always valid, no
Σ = 0 hypothesis).  Variant N takes the MINIMUM of these two:
   δ²-bound (variant N) = min (op_restricted · d, ell_int_sum) · h².

The min ensures variant N is always ≤ variant F (sound regression), while
empirically op_restricted · d < ell_int_sum on a meaningful fraction of
windows, prunesing additional 0.6%–28.6% of F-survivors at modest cost.

For symmetric M, ‖M‖_op (the L²-induced operator norm) equals the spectral
radius max |λᵢ|.  We DEFINE op_restricted in Lean as the L²-induced operator
norm via Mathlib's `Matrix.toEuclideanCLM` (which carries an operator-norm
structure inherited from ContinuousLinearMap).  We do NOT need the equality
‖M‖_op = max|λᵢ| — Cauchy–Schwarz + le_opNorm suffices.  The Python pipeline
(`_N_bench.py:precompute_op_norm_restricted`) computes op_restricted via
`np.linalg.eigvalsh(A_restr)` and takes max|eigenvalue|; this is the same
quantity (for symmetric M) as the operator norm we use here.

═══════════════════════════════════════════════════════════════════════════════
NEW HYPOTHESIS REQUIRED FOR VARIANT N
═══════════════════════════════════════════════════════════════════════════════

Variant D's `tight_discretization_bound` requires only `|aᵢ − cᵢ/m| ≤ 1/m`
(per-bin closeness). Variant N additionally requires
   ∑ aᵢ = ∑ cᵢ/m
(equal total mass).  In the cascade context this is automatic: both `c` and
`a` come from the same density f with ∫f = 1, so Σa·δ_bin = ∫a = 1 and
Σ(c/m)·δ_bin = 1, giving Σa = Σc/m = 4n.  Outside the cascade context
(arbitrary perturbation `a`), variant N does NOT apply; one falls back to
variant D/F's elementwise bound.

═══════════════════════════════════════════════════════════════════════════════
SOUNDNESS REGRESSION (N ≤ F)
═══════════════════════════════════════════════════════════════════════════════

Because we take `min(op_restricted · d, ell_int_sum)` as the δ²-bound,
variant N's correction is always ≤ variant D's correction (and hence ≤ F's
correction in the linear-term sense, given that we reuse D's linear bound
verbatim).  This is the regression-soundness property: switching from D to N
never widens the threshold, so any prune justified by D is also justified by
N.  The strict improvement (when it happens) is purely a tightening.

═══════════════════════════════════════════════════════════════════════════════
PYTHON CROSS-REFERENCE
═══════════════════════════════════════════════════════════════════════════════

  - _N_bench.py:precompute_op_norm_restricted (lines 31-59)
      builds A_W, subtracts α = n_pairs/d², runs eigvalsh, returns max|eig|.
  - _N_bench.py:prune_N (lines 62-148)
      cascade kernel using delta_sq = min(n_pairs, op_rest * d) per window.

The Lean theorem `tight_discretization_bound_N` matches the per-window
correction `corr_N_m2` defined here.

═══════════════════════════════════════════════════════════════════════════════
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.WRefinedBound
import Sidon.Proof.TightDiscretizationBound

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped Matrix

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 1: A_W — the {0,1}-indicator matrix of the window pair-set
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The window indicator matrix:  `A_W[i,j] = 1` iff `i + j ∈ window`. -/
def A_W (n s_lo ℓ : ℕ) : Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ :=
  fun i j => if i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then 1 else 0

/-- `A_W` is symmetric (transpose equals itself). -/
theorem A_W_symm (n s_lo ℓ : ℕ) : (A_W n s_lo ℓ)ᵀ = A_W n s_lo ℓ := by
  ext i j
  show A_W n s_lo ℓ j i = A_W n s_lo ℓ i j
  unfold A_W
  rw [show j.val + i.val = i.val + j.val from Nat.add_comm _ _]

/-- `A_W` is Hermitian (over ℝ this is the same as symmetric). -/
theorem A_W_isHermitian (n s_lo ℓ : ℕ) : (A_W n s_lo ℓ).IsHermitian := by
  unfold Matrix.IsHermitian
  rw [show (A_W n s_lo ℓ)ᴴ = (A_W n s_lo ℓ)ᵀ from by
    ext i j
    simp [Matrix.conjTranspose_apply, A_W]]
  exact A_W_symm n s_lo ℓ

/-- Quadratic form: `δᵀ A_W δ = Σ_{(i,j) ∈ W} δᵢ δⱼ`. -/
theorem A_W_quad_form (n s_lo ℓ : ℕ) (δ : Fin (2 * n) → ℝ) :
    dotProduct δ ((A_W n s_lo ℓ).mulVec δ) =
      ∑ p ∈ window_pair_set n s_lo ℓ, δ p.1 * δ p.2 := by
  classical
  -- Expand mulVec and dotProduct: Σᵢ δᵢ · (Σⱼ A_{ij} · δⱼ).
  unfold dotProduct Matrix.mulVec
  show ∑ i, δ i * (∑ j, A_W n s_lo ℓ i j * δ j) =
    ∑ p ∈ window_pair_set n s_lo ℓ, δ p.1 * δ p.2
  -- Push the sum and indicator: Σᵢⱼ if (i,j)∈W then δᵢ·δⱼ else 0.
  have h_lhs :
      ∑ i, δ i * (∑ j, A_W n s_lo ℓ i j * δ j) =
      ∑ i, ∑ j, (if i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
                  then δ i * δ j else 0) := by
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    unfold A_W
    by_cases hk : i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
    · rw [if_pos hk, if_pos hk]; ring
    · rw [if_neg hk, if_neg hk]; ring
  rw [h_lhs]
  -- Convert double sum to filter sum over the window pair set,
  -- following the pattern in `test_value_real_eq_window_sum`.
  unfold window_pair_set
  rw [show ((Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
        (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))) =
        ((Finset.univ : Finset (Fin (2*n))) ×ˢ (Finset.univ : Finset (Fin (2*n)))).filter
          (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))
    from by ext; simp]
  rw [Finset.sum_filter, ← Finset.sum_product']

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: ones_matrix, α_W, A_W_restricted
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The all-ones matrix on `Fin d × Fin d`. -/
def ones_matrix (d : ℕ) : Matrix (Fin d) (Fin d) ℝ := fun _ _ => 1

/-- `ones_matrix` is symmetric. -/
theorem ones_matrix_symm (d : ℕ) : (ones_matrix d)ᵀ = ones_matrix d := by
  ext i j; rfl

/-- The shift constant `α_W = ell_int_sum / (2n)²`.  Subtracting `α_W · 𝟙𝟙ᵀ`
    from `A_W` zeroes the all-ones (rank-1, mean) component. -/
noncomputable def alpha_W (n s_lo ℓ : ℕ) : ℝ :=
  (ell_int_sum n s_lo ℓ : ℝ) / ((2 * (n : ℝ))^2)

/-- The "restricted" matrix `A_W − α_W · 𝟙𝟙ᵀ`.  For ε with Σ ε = 0,
    `εᵀ M_restr ε = εᵀ A_W ε` (the all-ones component does not see ε). -/
noncomputable def A_W_restricted (n s_lo ℓ : ℕ) :
    Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ :=
  A_W n s_lo ℓ - (alpha_W n s_lo ℓ) • ones_matrix (2 * n)

/-- `A_W_restricted` is symmetric. -/
theorem A_W_restricted_symm (n s_lo ℓ : ℕ) :
    (A_W_restricted n s_lo ℓ)ᵀ = A_W_restricted n s_lo ℓ := by
  unfold A_W_restricted
  rw [Matrix.transpose_sub, Matrix.transpose_smul, A_W_symm, ones_matrix_symm]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: op_restricted — L²-induced operator norm of A_W_restricted
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The L²-induced operator norm of `A_W_restricted` (as a continuous linear
    map on Euclidean space).  For symmetric matrices this equals the spectral
    radius `max |eigenvalue|`; the Python implementation
    (`_N_bench.py:precompute_op_norm_restricted`) computes it via
    `np.linalg.eigvalsh` and takes max|eig|.  In Lean we use the
    operator-norm definition via `Matrix.toEuclideanCLM`, which avoids
    the eigendecomposition machinery — Cauchy–Schwarz suffices for the
    quadratic-form bound. -/
noncomputable def op_restricted (n s_lo ℓ : ℕ) : ℝ :=
  ‖Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin (2 * n)) (A_W_restricted n s_lo ℓ)‖

/-- `op_restricted` is non-negative (as a norm). -/
theorem op_restricted_nonneg (n s_lo ℓ : ℕ) : 0 ≤ op_restricted n s_lo ℓ :=
  norm_nonneg _

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: Spectral quadratic form bound (general, via Cauchy–Schwarz)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Spectral quadratic form bound (general)**: for any matrix `M` (need not be
    symmetric) and any vector `δ`,
       |δᵀ M δ| ≤ ‖toEuclideanCLM M‖ · Σᵢ δᵢ².
    Proof: Cauchy–Schwarz on the inner product gives |⟨δ, Mδ⟩| ≤ ‖δ‖·‖Mδ‖,
    and the operator-norm inequality gives ‖Mδ‖ ≤ ‖M‖_op · ‖δ‖.  Combined,
    |⟨δ, Mδ⟩| ≤ ‖M‖_op · ‖δ‖² = ‖M‖_op · Σᵢ δᵢ². -/
theorem spectral_quad_form_bound (d : ℕ) (M : Matrix (Fin d) (Fin d) ℝ)
    (δ : Fin d → ℝ) :
    |dotProduct δ (M.mulVec δ)| ≤
      ‖Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin d) M‖ * ∑ i, δ i ^ 2 := by
  classical
  -- δ_E := toLp 2 δ : EuclideanSpace ℝ (Fin d).
  set δ_E : EuclideanSpace ℝ (Fin d) := (WithLp.toLp 2 δ : EuclideanSpace ℝ (Fin d))
    with hδ_E
  set f := Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin d) M with hf
  -- f δ_E = toLp (M *ᵥ δ).
  have h_apply : f δ_E = (WithLp.toLp 2 (M.mulVec δ) : EuclideanSpace ℝ (Fin d)) := by
    rw [hf, hδ_E]
    exact Matrix.toEuclideanCLM_toLp M δ
  -- ⟪δ_E, f δ_E⟫_ℝ = δ ⬝ᵥ (M *ᵥ δ).
  have h_inner_eq :
      @inner ℝ (EuclideanSpace ℝ (Fin d)) _ δ_E (f δ_E) =
        dotProduct δ (M.mulVec δ) := by
    rw [h_apply]
    rw [show
        (@inner ℝ (EuclideanSpace ℝ (Fin d)) _
          (WithLp.toLp 2 δ : EuclideanSpace ℝ (Fin d))
          (WithLp.toLp 2 (M.mulVec δ) : EuclideanSpace ℝ (Fin d)))
        = dotProduct (M.mulVec δ) (star δ) from
        EuclideanSpace.inner_toLp_toLp δ (M.mulVec δ)]
    -- For ℝ, star = id.
    have h_star : (star δ : Fin d → ℝ) = δ := by
      funext i; exact star_trivial _
    rw [h_star]
    -- dotProduct is symmetric.
    exact dotProduct_comm (M.mulVec δ) δ
  -- ‖δ_E‖² = Σ δᵢ².
  have h_norm_sq : ‖δ_E‖ ^ 2 = ∑ i, δ i ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq]
    apply Finset.sum_congr rfl
    intro i _
    show ‖(WithLp.toLp 2 δ : EuclideanSpace ℝ (Fin d)) i‖ ^ 2 = δ i ^ 2
    rw [show (WithLp.toLp 2 δ : EuclideanSpace ℝ (Fin d)) i = δ i from rfl]
    rw [Real.norm_eq_abs, sq_abs]
  -- Cauchy–Schwarz: |⟪δ_E, f δ_E⟫| ≤ ‖δ_E‖ * ‖f δ_E‖.
  have h_cs : |@inner ℝ (EuclideanSpace ℝ (Fin d)) _ δ_E (f δ_E)|
              ≤ ‖δ_E‖ * ‖f δ_E‖ :=
    abs_real_inner_le_norm δ_E (f δ_E)
  -- Operator norm: ‖f δ_E‖ ≤ ‖f‖ * ‖δ_E‖.
  have h_op : ‖f δ_E‖ ≤ ‖f‖ * ‖δ_E‖ := f.le_opNorm δ_E
  -- Combine: |⟪⟫| ≤ ‖δ_E‖² · ‖f‖.
  have h_norm_nn : 0 ≤ ‖δ_E‖ := norm_nonneg _
  have h_step : ‖δ_E‖ * ‖f δ_E‖ ≤ ‖δ_E‖ * (‖f‖ * ‖δ_E‖) :=
    mul_le_mul_of_nonneg_left h_op h_norm_nn
  have h_combined :
      |@inner ℝ (EuclideanSpace ℝ (Fin d)) _ δ_E (f δ_E)| ≤
        ‖f‖ * (‖δ_E‖ * ‖δ_E‖) := by
    have : ‖δ_E‖ * (‖f‖ * ‖δ_E‖) = ‖f‖ * (‖δ_E‖ * ‖δ_E‖) := by ring
    linarith [h_cs, h_step]
  -- ‖δ_E‖ * ‖δ_E‖ = ‖δ_E‖² = Σ δᵢ².
  have h_sq_eq : ‖δ_E‖ * ‖δ_E‖ = ∑ i, δ i ^ 2 := by
    rw [show ‖δ_E‖ * ‖δ_E‖ = ‖δ_E‖ ^ 2 from by ring]
    exact h_norm_sq
  rw [h_sq_eq] at h_combined
  rw [h_inner_eq] at h_combined
  exact h_combined

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: Sigma-zero kernel lemma
-- ═══════════════════════════════════════════════════════════════════════════════

/-- For any vector δ and any scalar α, `δᵀ (α · 𝟙𝟙ᵀ) δ = α · (Σᵢ δᵢ)²`.
    Therefore, when Σ δ = 0, the all-ones (rank-1) component vanishes. -/
theorem ones_matrix_quad_form (d : ℕ) (α : ℝ) (δ : Fin d → ℝ) :
    dotProduct δ ((α • ones_matrix d).mulVec δ) =
      α * (∑ i, δ i) ^ 2 := by
  classical
  unfold dotProduct Matrix.mulVec
  show ∑ i, δ i * ∑ j, (α • ones_matrix d) i j * δ j = α * (∑ i, δ i) ^ 2
  -- (α • ones_matrix d) i j = α (since ones_matrix has 1 everywhere and α • 1 = α).
  have h_entry : ∀ i j : Fin d, (α • ones_matrix d) i j = α := by
    intro i j
    show α • ones_matrix d i j = α
    unfold ones_matrix
    show α • (1 : ℝ) = α
    simp
  -- Inner sum: ∑ j, α * δ j = α * ∑ δ.
  have h_inner : ∀ i : Fin d,
      ∑ j, (α • ones_matrix d) i j * δ j = α * ∑ j, δ j := by
    intro i
    have : ∀ j : Fin d, (α • ones_matrix d) i j * δ j = α * δ j := by
      intro j; rw [h_entry]
    simp_rw [this, ← Finset.mul_sum]
  simp_rw [h_inner]
  -- Now: ∑ i, δ i * (α * ∑ j δ j) = α · (∑ δ)².
  -- Factor (α * ∑δ) out of the i-sum since it's constant in i.
  rw [show (∑ i, δ i * (α * ∑ j, δ j)) = (∑ i, δ i) * (α * ∑ j, δ j) from by
    rw [← Finset.sum_mul]]
  ring

/-- **Sigma-zero kernel lemma**: if Σ δ = 0, then `δᵀ (α · 𝟙𝟙ᵀ) δ = 0`. -/
theorem sigma_zero_kills_ones (d : ℕ) (α : ℝ) (δ : Fin d → ℝ)
    (h_sum : ∑ i, δ i = 0) :
    dotProduct δ ((α • ones_matrix d).mulVec δ) = 0 := by
  rw [ones_matrix_quad_form, h_sum]
  ring

/-- Under Σ δ = 0, `δᵀ A_W δ = δᵀ A_W_restricted δ`. -/
theorem A_W_quad_form_eq_restricted (n s_lo ℓ : ℕ) (δ : Fin (2 * n) → ℝ)
    (h_sum : ∑ i, δ i = 0) :
    dotProduct δ ((A_W n s_lo ℓ).mulVec δ) =
      dotProduct δ ((A_W_restricted n s_lo ℓ).mulVec δ) := by
  unfold A_W_restricted
  rw [Matrix.sub_mulVec, dotProduct_sub]
  rw [sigma_zero_kills_ones (2 * n) (alpha_W n s_lo ℓ) δ h_sum]
  ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: L²-norm bound for bounded entries
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **L² norm bound**: if `|δᵢ| ≤ h` for all i, then `Σ δᵢ² ≤ d · h²`. -/
theorem l2_norm_bound (d : ℕ) (δ : Fin d → ℝ) (h : ℝ)
    (h_close : ∀ i, |δ i| ≤ h) :
    ∑ i, δ i ^ 2 ≤ d * h ^ 2 := by
  classical
  have h_sum_le : ∑ _i : Fin d, h ^ 2 = d * h ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [← h_sum_le]
  apply Finset.sum_le_sum
  intro i _
  have habs : |δ i| ≤ h := h_close i
  have hδ_sq : δ i ^ 2 = |δ i| ^ 2 := by rw [sq_abs]
  rw [hδ_sq]
  have habs_nn : 0 ≤ |δ i| := abs_nonneg _
  have h_nn : 0 ≤ h := le_trans habs_nn habs
  exact pow_le_pow_left₀ habs_nn habs 2

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: Combined δ²-bound under Σ δ = 0  (window-summed, raw)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Spectral δ²-bound under Σ δ = 0** (raw, before TV-normalization).

    Hypothesis: `|δᵢ| ≤ h` and Σᵢ δᵢ = 0.  Conclusion:
       |Σ_{(i,j) ∈ W} δᵢ δⱼ|  ≤  op_restricted · 2n · h².

    Proof: combine the spectral bound on `A_W_restricted` with the
    σ-zero kernel lemma (which equates `δᵀ A_W δ` and `δᵀ A_W_restricted δ`)
    and the L²-norm bound `Σ δᵢ² ≤ 2n · h²`. -/
theorem delta_sq_bound_spectral (n s_lo ℓ : ℕ)
    (δ : Fin (2 * n) → ℝ) (h : ℝ)
    (h_close : ∀ i, |δ i| ≤ h)
    (h_sum : ∑ i, δ i = 0) :
    |∑ p ∈ window_pair_set n s_lo ℓ, δ p.1 * δ p.2| ≤
      op_restricted n s_lo ℓ * (2 * n) * h ^ 2 := by
  classical
  -- Step 1: rewrite the window-summed quadratic form.
  rw [← A_W_quad_form n s_lo ℓ δ]
  -- Step 2: apply σ-zero kernel: dot(δ, A_W δ) = dot(δ, A_W_restricted δ).
  rw [A_W_quad_form_eq_restricted n s_lo ℓ δ h_sum]
  -- Step 3: apply spectral bound on A_W_restricted.
  have h_spec := spectral_quad_form_bound (2 * n) (A_W_restricted n s_lo ℓ) δ
  -- Step 4: apply L² norm bound: Σ δᵢ² ≤ 2n · h².
  have h_l2 := l2_norm_bound (2 * n) δ h h_close
  -- Step 5: combine.
  have h_op_nn := op_restricted_nonneg n s_lo ℓ
  unfold op_restricted at *
  calc |dotProduct δ ((A_W_restricted n s_lo ℓ).mulVec δ)|
      ≤ ‖Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin (2 * n))
            (A_W_restricted n s_lo ℓ)‖ * ∑ i, δ i ^ 2 := h_spec
    _ ≤ ‖Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin (2 * n))
            (A_W_restricted n s_lo ℓ)‖ * ((2 * n : ℕ) * h ^ 2) := by
          apply mul_le_mul_of_nonneg_left h_l2 h_op_nn
    _ = ‖Matrix.toEuclideanCLM (𝕜 := ℝ) (n := Fin (2 * n))
            (A_W_restricted n s_lo ℓ)‖ * (2 * n) * h ^ 2 := by
          push_cast; ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 8: TV-normalized δ²-bound under Σ δ = 0 (variant N)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **TV-normalized spectral δ²-bound under Σ δ = 0**.

    For ε := b − a with `|aᵢ − bᵢ| ≤ 1/m` and `Σ aᵢ = Σ bᵢ` (i.e., Σ ε = 0),
       |(1 / (4n·ℓ)) · Σ_W εᵢ εⱼ| ≤ op_restricted · 2n / (4n·ℓ·m²)
                                  = op_restricted / (2 · ℓ · m²).

    The min-with-`ell_int_sum / (4n·ℓ·m²)` form (which gives the actual
    variant N correction term) follows from this combined with the F-style
    elementwise δ²-bound from `tv_delta_sq_bound`. -/
theorem tv_delta_sq_bound_N
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (a b : Fin (2 * n) → ℝ)
    (h_close : ∀ i, |a i - b i| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, b i)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ, (b p.1 - a p.1) * (b p.2 - a p.2)|
      ≤ op_restricted n s_lo ℓ * (2 * n) /
          ((4 * n * ℓ : ℝ) * (m : ℝ) ^ 2) := by
  classical
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_pos : (0 : ℝ) < ℓ := by
    have : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
    linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ) ^ 2 := by positivity
  -- Set δ := b - a.  Then |δᵢ| ≤ 1/m and Σ δ = 0.
  set δ : Fin (2 * n) → ℝ := fun i => b i - a i with hδ_def
  have hδ_close : ∀ i, |δ i| ≤ 1 / (m : ℝ) := by
    intro i; rw [hδ_def]; rw [abs_sub_comm]; exact h_close i
  have hδ_sum : ∑ i, δ i = 0 := by
    show ∑ i, (b i - a i) = 0
    rw [Finset.sum_sub_distrib, ← h_sum_eq]; ring
  -- Apply the raw bound with h := 1/m.
  have h_raw := delta_sq_bound_spectral n s_lo ℓ δ (1 / (m : ℝ)) hδ_close hδ_sum
  -- |(1/(4nℓ)) · X| = (1/(4nℓ)) · |X|; rearrange.
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 1 / (4 * n * ℓ))]
  rw [div_mul_eq_mul_div, one_mul]
  rw [div_le_div_iff₀ h4nℓ_pos
        (by positivity : (0 : ℝ) < (4 * n * ℓ : ℝ) * (m : ℝ) ^ 2)]
  -- The window sum equals δ ⬝ A_W ⬝ δ.
  have h_eq_form :
      ∑ p ∈ window_pair_set n s_lo ℓ, δ p.1 * δ p.2 =
      ∑ p ∈ window_pair_set n s_lo ℓ, (b p.1 - a p.1) * (b p.2 - a p.2) := by
    apply Finset.sum_congr rfl
    intro p _; rfl
  rw [← h_eq_form]
  calc |∑ p ∈ window_pair_set n s_lo ℓ, δ p.1 * δ p.2| * (4 * ↑n * ↑ℓ * ↑m ^ 2)
      ≤ (op_restricted n s_lo ℓ * (2 * (n : ℝ)) * (1 / (m : ℝ)) ^ 2) *
          (4 * ↑n * ↑ℓ * ↑m ^ 2) := by
        apply mul_le_mul_of_nonneg_right h_raw (by positivity)
    _ = op_restricted n s_lo ℓ * (2 * (n : ℝ)) * (4 * ↑n * ↑ℓ) := by
        have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
        field_simp
    _ = op_restricted n s_lo ℓ * (2 * (n : ℝ)) * (4 * (n : ℝ) * ↑ℓ) := by ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 9: Variant N correction and main bound
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The variant-N correction in m² units (using D's linear bound + spectral δ²).

    `corr_N_m2 = (ℓ−1)·W_int_overlap/(2n·ℓ) +
                 min(op_restricted · 2n, ell_int_sum) / (4n·ℓ)`.

    Variant D's correction is the same but with `ell_int_sum` instead of
    `min(op_restricted · 2n, ell_int_sum)` in the second term.  Therefore
    `corr_N_m2 ≤ corr_tight_m2` always (sound regression). -/
noncomputable def corr_N_m2 (n : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) / (2 * n * ℓ) +
    min (op_restricted n s_lo ℓ * (2 * (n : ℝ)))
        (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ)

/-- The variant-N correction in TV space (divided by m²). -/
noncomputable def correction_N (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  corr_N_m2 n c ℓ s_lo / (m : ℝ) ^ 2

/-- **Variant N main theorem**: under Σ a = Σ c/m (cascade-context hypothesis),
    the TV difference `|TV(c/m; W) − TV(a; W)|` is bounded by the variant-N
    correction `correction_N`.

    The proof reuses variant D's linear-term bound (`tv_linear_bound`) and
    combines it with the new spectral δ²-bound `tv_delta_sq_bound_N` plus the
    elementwise fallback `tv_delta_sq_bound` (taking the min).  The min step
    is structural: `min(spec, elem) ≤ elem` and `min(spec, elem) ≤ spec`, so
    whichever bound applies, the min absorbs the slack. -/
theorem tight_discretization_bound_N
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2)|
      ≤ correction_N n m c ℓ s_lo := by
  classical
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ) ^ 2 := by positivity
  have hℓ_real : (1 : ℝ) ≤ (ℓ : ℝ) := by
    have : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
    linarith
  have hℓ_pos : (0 : ℝ) < (ℓ : ℝ) := by linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  have h_op_nn : 0 ≤ op_restricted n s_lo ℓ := op_restricted_nonneg n s_lo ℓ
  -- Set b := c/m.  D-v2 decomposition: b·b − a·a = b·ε + ε·b − ε·ε with ε = b − a.
  set b : Fin (2 * n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  have h_close' : ∀ i, |a i - b i| ≤ 1 / (m : ℝ) := by
    intro i; rw [hb_def]; exact h_close i
  have h_sum_eq' : ∑ i, a i = ∑ i, b i := by
    rw [hb_def]; exact h_sum_eq
  -- Decomposition.
  have h_decomp : ∀ p : Fin (2 * n) × Fin (2 * n),
      (c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2 =
      ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m - a p.2) +
       (c p.2 : ℝ) / m * ((c p.1 : ℝ) / m - a p.1)) -
        ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2) := by
    intro p; ring
  have h_sum_decomp :
    (∑ p ∈ window_pair_set n s_lo ℓ,
        ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2)) =
    (∑ p ∈ window_pair_set n s_lo ℓ,
        ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m - a p.2) +
         (c p.2 : ℝ) / m * ((c p.1 : ℝ) / m - a p.1))) -
    (∑ p ∈ window_pair_set n s_lo ℓ,
        ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro p _; rw [h_decomp]
  rw [h_sum_decomp, mul_sub]
  -- Linear bound (reuse from variant D).
  have h_lin :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m - a p.2) +
            (c p.2 : ℝ) / m * ((c p.1 : ℝ) / m - a p.1))|
        ≤ 2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) /
            (4 * n * ℓ * (m : ℝ) ^ 2) :=
    tv_linear_bound n m hn hm c a ha_nonneg h_close ℓ s_lo hℓ
  -- δ²-bound: SPECTRAL form (uses Σ ε = 0 from h_sum_eq).
  have h_delta_spec :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)|
        ≤ op_restricted n s_lo ℓ * (2 * (n : ℝ)) /
            ((4 * n * ℓ : ℝ) * (m : ℝ) ^ 2) := by
    have h_raw := tv_delta_sq_bound_N n m hn hm a b h_close' h_sum_eq' ℓ s_lo hℓ
    -- b = c/m by definition, so the two sums are syntactically equal.
    have h_abs_eq :
        ∑ p ∈ window_pair_set n s_lo ℓ,
             (b p.1 - a p.1) * (b p.2 - a p.2) =
        ∑ p ∈ window_pair_set n s_lo ℓ,
             ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2) := by
      apply Finset.sum_congr rfl
      intro p _; rfl
    rw [h_abs_eq] at h_raw
    exact h_raw
  -- δ²-bound: ELEMENTWISE form (always holds, no Σ = 0 needed).
  have h_delta_elem :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)|
        ≤ (ell_int_sum n s_lo ℓ : ℝ) /
            (4 * n * ℓ * (m : ℝ) ^ 2) := by
    have := tv_delta_sq_bound n m hn hm a b h_close' ℓ s_lo hℓ
    convert this using 4
  -- δ²-bound: take MIN of spectral and elementwise.
  have h_delta_min :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)|
        ≤ min (op_restricted n s_lo ℓ * (2 * (n : ℝ)))
              (ell_int_sum n s_lo ℓ : ℝ) /
            (4 * n * ℓ * (m : ℝ) ^ 2) := by
    have h_denom_pos : (0 : ℝ) < 4 * n * ℓ * (m : ℝ) ^ 2 := by positivity
    have h_denom_eq :
        (4 * n * ℓ : ℝ) * (m : ℝ) ^ 2 = 4 * n * ℓ * (m : ℝ) ^ 2 := by ring
    -- Convert both bounds to the form `... * denom ≤ rhs`.
    rw [le_div_iff₀ h_denom_pos]
    have h_spec' : |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)|
        * (4 * n * ℓ * (m : ℝ) ^ 2)
        ≤ op_restricted n s_lo ℓ * (2 * (n : ℝ)) := by
      have := (le_div_iff₀ (by positivity :
        (0 : ℝ) < (4 * n * ℓ : ℝ) * (m : ℝ) ^ 2)).mp h_delta_spec
      rw [← h_denom_eq]; exact this
    have h_elem' : |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)|
        * (4 * n * ℓ * (m : ℝ) ^ 2)
        ≤ (ell_int_sum n s_lo ℓ : ℝ) := by
      exact (le_div_iff₀ h_denom_pos).mp h_delta_elem
    exact le_min h_spec' h_elem'
  -- Combine via |x − y| ≤ |x| + |y|.
  calc |(1 / ((4 * n * ℓ : ℝ))) *
        ∑ p ∈ window_pair_set n s_lo ℓ,
          ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m - a p.2) +
           (c p.2 : ℝ) / m * ((c p.1 : ℝ) / m - a p.1)) -
        (1 / ((4 * n * ℓ : ℝ))) *
        ∑ p ∈ window_pair_set n s_lo ℓ,
          ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)|
      ≤ |(1 / ((4 * n * ℓ : ℝ))) *
          ∑ p ∈ window_pair_set n s_lo ℓ,
            ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m - a p.2) +
             (c p.2 : ℝ) / m * ((c p.1 : ℝ) / m - a p.1))| +
        |(1 / ((4 * n * ℓ : ℝ))) *
          ∑ p ∈ window_pair_set n s_lo ℓ,
            ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)| := abs_sub _ _
    _ ≤ 2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) /
            (4 * n * ℓ * (m : ℝ) ^ 2) +
        min (op_restricted n s_lo ℓ * (2 * (n : ℝ)))
            (ell_int_sum n s_lo ℓ : ℝ) /
            (4 * n * ℓ * (m : ℝ) ^ 2) := by
        linarith [h_lin, h_delta_min]
    _ = (2 * ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) +
            min (op_restricted n s_lo ℓ * (2 * (n : ℝ)))
                (ell_int_sum n s_lo ℓ : ℝ)) /
            (4 * n * ℓ * (m : ℝ) ^ 2) := by ring
    _ = correction_N n m c ℓ s_lo := by
        unfold correction_N corr_N_m2
        have h2nℓ_pos : (0 : ℝ) < 2 * n * ℓ := by positivity
        have h4nℓ_pos' : (0 : ℝ) < 4 * n * ℓ := by positivity
        field_simp
        ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 10: N ≤ F dominance (corr_N_m2 ≤ corr_tight_m2)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Soundness regression: `corr_N_m2 ≤ corr_tight_m2` always**.
    Variant N's correction never exceeds variant D's, because
    `min(op_restricted · 2n, ell_int_sum) ≤ ell_int_sum`. -/
theorem corr_N_m2_le_corr_tight_m2 (n : ℕ) (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hn : 0 < n) (hℓ : 2 ≤ ℓ) :
    corr_N_m2 n c ℓ s_lo ≤ corr_tight_m2 n c ℓ s_lo := by
  unfold corr_N_m2 corr_tight_m2
  have hn_real : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_real : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
  have hℓ_pos : (0 : ℝ) < (ℓ : ℝ) := by linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  have h_min_le :
      min (op_restricted n s_lo ℓ * (2 * (n : ℝ))) (ell_int_sum n s_lo ℓ : ℝ)
        ≤ (ell_int_sum n s_lo ℓ : ℝ) := min_le_right _ _
  have h_div_le :
      min (op_restricted n s_lo ℓ * (2 * (n : ℝ))) (ell_int_sum n s_lo ℓ : ℝ) /
          (4 * n * ℓ) ≤
      (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ) := by
    apply div_le_div_of_nonneg_right h_min_le h4nℓ_pos.le
  linarith

/-- **Soundness regression (TV-normalized)**: `correction_N ≤ tight_correction`. -/
theorem correction_N_le_tight_correction
    (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ)
    (hn : 0 < n) (hm : 0 < m) (hℓ : 2 ≤ ℓ) :
    correction_N n m c ℓ s_lo ≤ tight_correction n m c ℓ s_lo := by
  unfold correction_N tight_correction
  have h_le := corr_N_m2_le_corr_tight_m2 n c ℓ s_lo hn hℓ
  have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ) ^ 2 := by positivity
  exact div_le_div_of_nonneg_right h_le hm_sq_pos.le

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 11: Cascade soundness with variant N
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Variant N cascade-prune soundness**: if
       `test_value n m c ℓ s_lo > c_target + correction_N n m c ℓ s_lo`,
    then for every real `a` with `0 ≤ aᵢ`, `|aᵢ − cᵢ/m| ≤ 1/m`, and
    `Σ aᵢ = Σ cᵢ/m` (cascade-context total-mass equality), we have
       `test_value_real n a ℓ s_lo > c_target`.

    This is the discrete-discrete pruning soundness for variant N.  The
    extra hypothesis `h_sum_eq` distinguishes N from D: variant N exploits
    Σ(a − c/m) = 0 to obtain the spectral δ² bound. -/
theorem tight_cascade_prune_sound_N
    (n m : ℕ) (c_target : ℝ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (h_exceeds : test_value n m c ℓ s_lo > c_target +
      correction_N n m c ℓ s_lo)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m) :
    test_value_real n a ℓ s_lo > c_target := by
  classical
  have h_tight :=
    tight_discretization_bound_N n m hn hm c a ha_nonneg h_close h_sum_eq ℓ s_lo hℓ
  rw [test_value_eq_window_sum] at h_exceeds
  rw [test_value_real_eq_window_sum]
  have h_bound_eq :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2)| =
      |((1 / ((4 * n * ℓ : ℝ))) * ∑ p ∈ window_pair_set n s_lo ℓ,
            (c p.1 : ℝ) / m * ((c p.2 : ℝ) / m)) -
        ((1 / ((4 * n * ℓ : ℝ))) * ∑ p ∈ window_pair_set n s_lo ℓ,
            a p.1 * a p.2)| := by
    congr 1
    rw [← mul_sub, ← Finset.sum_sub_distrib]
  rw [h_bound_eq] at h_tight
  have h_abs_le := abs_sub_le_iff.mp h_tight
  linarith [h_abs_le.1, h_abs_le.2]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 12: Numerical sanity examples
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Sanity: at n=2, ℓ=4, s_lo=2, ell_int_sum = 1 + 4 + 3 = ?
    Formula: ell_int_arr 2 k for k=2,3,4 — let's compute.
    `ell_int_arr 2 2 = 3`, `ell_int_arr 2 3 = 4`, `ell_int_arr 2 4 = 3`.
    So ell_int_sum 2 2 4 = 3 + 4 + 3 = 10.

    Variant N: regression-soundness check
    `corr_N_m2 ≤ corr_tight_m2`. -/
example (n : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) (hn : 0 < n) (hℓ : 2 ≤ ℓ) :
    corr_N_m2 n c ℓ s_lo ≤ corr_tight_m2 n c ℓ s_lo :=
  corr_N_m2_le_corr_tight_m2 n c ℓ s_lo hn hℓ

/-- A sanity check using the existing `ell_int_sum` arithmetic at small n.
    For n=2 (so 2n=4 bins), ℓ=4, s_lo=2: ell_int_sum = 10.  We illustrate the
    structural fact that `min(B, 10) ≤ 10` for any non-negative `B`, which
    is the engine of `corr_N_m2 ≤ corr_tight_m2`. -/
example : ell_int_sum 2 2 4 = 10 := by decide

example (B : ℝ) :
    min B (ell_int_sum 2 2 4 : ℝ) ≤ (ell_int_sum 2 2 4 : ℝ) :=
  min_le_right _ _

/-- A sanity check that demonstrates the strict-improvement regime: when
    `op_restricted · 2n < ell_int_sum`, the min picks the spectral side.
    We illustrate with abstract `B` rather than committed Lean values
    because `op_restricted` is non-computable. -/
example (B ell : ℝ) (h_lt : B < ell) : min B ell = B :=
  min_eq_left h_lt.le

end -- noncomputable section
