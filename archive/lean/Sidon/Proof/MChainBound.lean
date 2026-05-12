/-
Sidon Autocorrelation Project — M-Chain (per-conv-position F-style bound)

This file formalizes the M-CHAIN bound: a strictly tighter per-cell bound than
the cascade's window-averaged TV_W.  Mathematical content in
`proof/m_chain_proof.md`.  Python cross-reference: `_smoke_M_chain.py`.

The M-chain works at **single conv-positions** (not window averages).  At
breakpoint t_k = -1/2 + (k+1)·w with w = 1/(2d), d = 2n,
   (f_a * f_a)(t_k) = (1/(2d·m²)) · conv[k](a)         (Lemma A)
where conv[k](a) = Σ_{i+j=k} a_i · a_j (integer-mass autoconvolution),
and the cell variable `a : Fin (2n) → ℝ` satisfies |a_i - c_i| ≤ 1, Σ a = Σ c.

Decomposing  Σ_{i+j=k} a_i a_j  with `a = c + δ`:
    conv[k](a) = conv[k](c) + 2 · L(k,δ) + Q(k,δ)
   where  L(k,δ) = δ · b(k),  b(k)_j = c_{k-j} (clipped)
            Q(k,δ) = Σ_{i+j=k} δ_i δ_j,
the LP closed-form gives  L(k,δ) ≥ -Δ_b(k)              (Lemma B)
and the triangle inequality gives  Q(k,δ) ≥ -n_pairs(k)  (Lemma C).
The cell polytope is contained in the symmetric polytope (Lemma D), so the
two bounds are sound.

Combining these gives  conv[k](a) ≥ conv[k](c) − 2·Δ_b(k) − n_pairs(k),
hence  (f_a * f_a)(t_k) ≥ LB(k) := (conv[k](c) − 2·Δ_b(k) − n_pairs(k))/(4·n·m²).

NEW AXIOMS DECLARED IN THIS FILE: ZERO.
NEW SORRIES DECLARED: ZERO (target).

═══════════════════════════════════════════════════════════════════════════════
STRATEGY FOR REUSE OF EXISTING INFRASTRUCTURE
═══════════════════════════════════════════════════════════════════════════════

We specialise the existing `BB_W`/`Delta_BB`/`lp_closed_form_le` machinery from
PostFilterF.lean to a single-conv-position window:
  s_lo := k,  ℓ := 2.
Then  s_lo + ℓ - 2 = k, so the window {s_lo ≤ p ≤ s_lo + ℓ - 2} = {k}.
With this:
  BB_W n c k 2 j = c_{k-j} (clipped to [0, 2n-1])  =  b(k)_j
  Delta_BB n c k 2 = sum_top - sum_bot of sorted b(k)  =  Δ_b(k)
  ell_int_sum n k 2 = ell_int_arr n k = n_pairs(k)
  window_pair_set n k 2 = {(i,j) : i+j = k}
This lets us reuse `lp_closed_form_le` directly for Lemma B.

═══════════════════════════════════════════════════════════════════════════════
NOTATION (matches `_smoke_M_chain.py`)
═══════════════════════════════════════════════════════════════════════════════

`MChain.b_vec n c k j`   : the b(k) vector, equals  BB_W n c k 2 j
`MChain.Delta_b n c k`   : the LP closed-form value, equals Delta_BB n c k 2
`MChain.n_pairs n k`     : the # of pairs (i,j) with i+j=k, equals ell_int_arr n k
`MChain.pair_set n k`    : the pair set {(i,j) : i+j=k}, equals window_pair_set n k 2
`MChain.LB n m c k`      : the M-chain lower bound (in TV-style units)

The cell variable `a : Fin (2n) → ℝ` is in INTEGER-MASS units (NOT heights):
the real heights are then a/m.  The cell condition is `|a_j - c_j| ≤ 1` and
`Σ a_j = Σ c_j` (cascade convention).
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.WRefinedDefs
import Sidon.Proof.TightDiscretizationBound
import Sidon.Proof.PostFilterF

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 0: Definitions specialised to single-conv-position k
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The M-chain b-vector at conv-position k:
      `b(k)_j = c_{k-j}  if  0 ≤ k-j ≤ 2n-1,  else 0`.
    Equals `BB_W n c k 2 j` (the single-conv window). -/
noncomputable def MChain.b_vec (n : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ)
    (j : Fin (2*n)) : ℝ :=
  BB_W n c k 2 j

/-- The M-chain LP closed-form Δ_b(k):
      `Δ_b(k) = sum_top(d/2)(b(k)) - sum_bot(d/2)(b(k))` after sorting.
    Equals `Delta_BB n c k 2`. -/
noncomputable def MChain.Delta_b (n : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ) : ℝ :=
  Delta_BB n c k 2

/-- The number of pairs (i,j) ∈ Fin(2n)² with i+j = k.  Equals `ell_int_arr n k`. -/
def MChain.n_pairs (n k : ℕ) : ℕ := ell_int_arr n k

/-- The pair set at conv-position k: `{(i,j) ∈ Fin(2n)² : i+j = k}`. -/
def MChain.pair_set (n k : ℕ) : Finset (Fin (2*n) × Fin (2*n)) :=
  window_pair_set n k 2

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 1: Reductions of MChain definitions to single-conv-position form
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The single-conv window has Icc s_lo (s_lo + ℓ - 2) = {k} when ℓ = 2 and s_lo = k. -/
theorem MChain.icc_singleton (k : ℕ) :
    Finset.Icc k (k + 2 - 2) = {k} := by
  simp

/-- `BB_W n c k 2 j = (if 0 ≤ k - j.val < 2n then c_{k-j} else 0)`, the b(k) vector. -/
theorem MChain.b_vec_eq (n : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ) (j : Fin (2*n)) :
    MChain.b_vec n c k j =
    ∑ i : Fin (2*n), if i.val + j.val = k then (c i : ℝ) else 0 := by
  unfold MChain.b_vec BB_W
  apply Finset.sum_congr rfl
  intro i _
  by_cases h : i.val + j.val = k
  · have h_in : i.val + j.val ∈ Finset.Icc k (k + 2 - 2) := by
      rw [MChain.icc_singleton]; simp [h]
    rw [if_pos h_in, if_pos h]
  · have h_not : i.val + j.val ∉ Finset.Icc k (k + 2 - 2) := by
      rw [MChain.icc_singleton]; simp; omega
    rw [if_neg h_not, if_neg h]

/-- The pair set at conv-position k equals the window pair set with ℓ = 2, s_lo = k. -/
theorem MChain.pair_set_eq (n k : ℕ) :
    MChain.pair_set n k = window_pair_set n k 2 := rfl

/-- The pair set at conv-position k is exactly `{(i,j) : i+j = k}`. -/
theorem MChain.pair_set_eq_filter (n k : ℕ) :
    MChain.pair_set n k =
    (Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
      (fun p => p.1.val + p.2.val = k) := by
  unfold MChain.pair_set window_pair_set
  ext p
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  rw [MChain.icc_singleton]
  simp

/-- The number of pairs in `MChain.pair_set` is `n_pairs(k) = ell_int_arr n k`. -/
theorem MChain.pair_set_card (n k : ℕ) :
    (MChain.pair_set n k).card = MChain.n_pairs n k := by
  unfold MChain.n_pairs
  rw [ell_int_arr_eq_card]
  rw [MChain.pair_set_eq_filter]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: Lemma B — LP closed-form lower bound on δ · b(k)
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**Lemma B (M-chain LP closed-form, lower bound)**.

For `δ : Fin (2n) → ℝ` with `|δ_j| ≤ 1` and `Σ δ_j = 0`, and any conv-position k,
   `Σ_j δ_j · b(k)_j  ≥  -Δ_b(k)`.

This is `lp_closed_form_le` applied to `(-δ)` (with h = 1) and the window
`(s_lo = k, ℓ = 2)`.

Mathematical content: balanced ±1 LP minimum at the cube vertex assigning
+1 to the d/2 smallest entries of b(k) and -1 to the d/2 largest.
-/
theorem MChain.lemma_B
    (n : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ)
    (δ : Fin (2*n) → ℝ)
    (h_abs : ∀ j, |δ j| ≤ 1)
    (h_sum : ∑ j : Fin (2*n), δ j = 0) :
    -MChain.Delta_b n c k ≤ ∑ j : Fin (2*n), δ j * MChain.b_vec n c k j := by
  classical
  unfold MChain.Delta_b MChain.b_vec
  -- We invoke `lp_closed_form_le` on `-δ` to get
  --   Σ_j (-δ_j) · BB_W^j ≤ 1 · Δ_BB,
  -- which rearranges to  Σ_j δ_j · BB_W^j ≥ -Δ_BB.
  have h_neg_abs : ∀ j, |(-δ) j| ≤ 1 := by
    intro j; rw [Pi.neg_apply, abs_neg]; exact h_abs j
  have h_neg_sum : ∑ j : Fin (2*n), (-δ) j = 0 := by
    have h_eq : ∀ j : Fin (2*n), (-δ) j = -(δ j) := fun j => Pi.neg_apply δ j
    simp_rw [h_eq]
    rw [Finset.sum_neg_distrib, h_sum]; ring
  have h_lp := lp_closed_form_le n c k 2 (-δ) 1 (by norm_num) h_neg_abs h_neg_sum
  rw [one_mul] at h_lp
  -- h_lp : Σ_j (-δ_j) · BB_W^j ≤ Δ_BB
  -- Convert to:  -Σ_j δ_j · BB_W^j ≤ Δ_BB.
  have h_neg_eq : ∑ j : Fin (2*n), (-δ) j * BB_W n c k 2 j =
                  -(∑ j : Fin (2*n), δ j * BB_W n c k 2 j) := by
    have h_step : ∀ j : Fin (2*n),
        (-δ) j * BB_W n c k 2 j = -(δ j * BB_W n c k 2 j) := by
      intro j; rw [Pi.neg_apply]; ring
    simp_rw [h_step]
    rw [← Finset.sum_neg_distrib]
  rw [h_neg_eq] at h_lp
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: Lemma C — Quadratic bound |δ^T A δ| ≤ #nonzero entries when |δ|_∞ ≤ 1
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**Lemma C (quadratic bound at single conv-position)**.

For `δ : Fin (2n) → ℝ` with `|δ_j| ≤ 1`,
   `|Σ_{(i,j) ∈ pair_set(k)} δ_i · δ_j|  ≤  n_pairs(k)`.

In particular,
   `Σ_{(i,j) ∈ pair_set(k)} δ_i · δ_j  ≥  -n_pairs(k)`.

Proof: triangle inequality on the sum, using `|δ_i δ_j| ≤ 1`.
-/
theorem MChain.lemma_C_abs
    (n : ℕ) (k : ℕ)
    (δ : Fin (2*n) → ℝ)
    (h_abs : ∀ i, |δ i| ≤ 1) :
    |∑ p ∈ MChain.pair_set n k, δ p.1 * δ p.2| ≤ (MChain.n_pairs n k : ℝ) := by
  classical
  calc |∑ p ∈ MChain.pair_set n k, δ p.1 * δ p.2|
      ≤ ∑ p ∈ MChain.pair_set n k, |δ p.1 * δ p.2| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ p ∈ MChain.pair_set n k, |δ p.1| * |δ p.2| := by
        apply Finset.sum_congr rfl
        intro p _; rw [abs_mul]
    _ ≤ ∑ p ∈ MChain.pair_set n k, 1 * 1 := by
        apply Finset.sum_le_sum
        intro p _
        have h1 : |δ p.1| ≤ 1 := h_abs p.1
        have h2 : |δ p.2| ≤ 1 := h_abs p.2
        have habs1 : 0 ≤ |δ p.1| := abs_nonneg _
        have habs2 : 0 ≤ |δ p.2| := abs_nonneg _
        nlinarith
    _ = ((MChain.pair_set n k).card : ℝ) := by
        rw [Finset.sum_const]; ring
    _ = (MChain.n_pairs n k : ℝ) := by
        rw [MChain.pair_set_card]

/-- **Lemma C, lower-bound form** (used for the M-chain). -/
theorem MChain.lemma_C
    (n : ℕ) (k : ℕ)
    (δ : Fin (2*n) → ℝ)
    (h_abs : ∀ i, |δ i| ≤ 1) :
    -((MChain.n_pairs n k : ℝ)) ≤ ∑ p ∈ MChain.pair_set n k, δ p.1 * δ p.2 := by
  have h := MChain.lemma_C_abs n k δ h_abs
  have h_le := abs_le.mp h
  linarith [h_le.1]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: Linear-term identity:  Σ_{i+j=k} c_i · δ_j  =  Σ_j δ_j · b(k)_j
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Linear-term identity**.

   For `δ : Fin (2n) → ℝ` and any conv-position k,
     `Σ_{(i,j) ∈ pair_set(k)} c_i · δ_j  =  Σ_j δ_j · b(k)_j`.

   This is the fundamental identity expressing `Σ_{i+j=k} c_i δ_j` as a
   linear functional of δ via the b(k) vector. -/
theorem MChain.linear_identity_left
    (n : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ)
    (δ : Fin (2*n) → ℝ) :
    ∑ p ∈ MChain.pair_set n k, (c p.1 : ℝ) * δ p.2 =
    ∑ j : Fin (2*n), δ j * MChain.b_vec n c k j := by
  classical
  -- Σ_p c(p.1) δ(p.2) = Σ_j δ(j) Σ_{i: i+j=k} c(i)
  unfold MChain.pair_set window_pair_set
  rw [Finset.sum_filter]
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [MChain.b_vec_eq]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  by_cases hk : i.val + j.val ∈ Finset.Icc k (k + 2 - 2)
  · have hkk : i.val + j.val = k := by
      rw [MChain.icc_singleton] at hk
      simpa using hk
    rw [if_pos hk, if_pos hkk]; ring
  · have hkk : i.val + j.val ≠ k := by
      intro hh
      apply hk
      rw [MChain.icc_singleton]; simpa using hh
    rw [if_neg hk, if_neg hkk]; ring

/-- Symmetric version: `Σ_{(i,j) ∈ pair_set(k)} δ_i · c_j = Σ_j δ_j · b(k)_j`. -/
theorem MChain.linear_identity_right
    (n : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ)
    (δ : Fin (2*n) → ℝ) :
    ∑ p ∈ MChain.pair_set n k, δ p.1 * (c p.2 : ℝ) =
    ∑ j : Fin (2*n), δ j * MChain.b_vec n c k j := by
  classical
  -- By symmetry of pair_set under (i,j) ↔ (j,i), reduce to linear_identity_left.
  have h_swap : ∑ p ∈ MChain.pair_set n k, δ p.1 * (c p.2 : ℝ) =
                ∑ p ∈ MChain.pair_set n k, (c p.1 : ℝ) * δ p.2 := by
    apply Finset.sum_bij (fun (p : Fin (2*n) × Fin (2*n)) (_hp : p ∈ MChain.pair_set n k) => (p.2, p.1))
    · intro p hp
      rw [MChain.pair_set_eq_filter] at hp ⊢
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      omega
    · intro p₁ _ p₂ _ heq
      simp only [Prod.mk.injEq] at heq
      apply Prod.ext heq.2 heq.1
    · intro p hp
      refine ⟨(p.2, p.1), ?_, rfl⟩
      rw [MChain.pair_set_eq_filter] at hp ⊢
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      omega
    · intro p _; ring
  rw [h_swap]
  exact MChain.linear_identity_left n c k δ

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: Decomposition of conv[k](a) for a = c + δ
-- ═══════════════════════════════════════════════════════════════════════════════

/-- For `a, c : Fin (2n) → ℝ` with `δ := a - c` (so a = c + δ), and conv-position k,
       `Σ_{(i,j) ∈ pair_set(k)} a_i · a_j  =  Σ_{(i,j) ∈ pair_set(k)} c_i · c_j
                                           + Σ_{(i,j) ∈ pair_set(k)} (c_i δ_j + δ_i c_j)
                                           + Σ_{(i,j) ∈ pair_set(k)} δ_i δ_j`. -/
theorem MChain.conv_decomposition
    (n : ℕ) (k : ℕ)
    (a c : Fin (2*n) → ℝ) :
    ∑ p ∈ MChain.pair_set n k, a p.1 * a p.2 =
    ∑ p ∈ MChain.pair_set n k, c p.1 * c p.2 +
    (∑ p ∈ MChain.pair_set n k, (c p.1 * (a p.2 - c p.2) + (a p.1 - c p.1) * c p.2)) +
    ∑ p ∈ MChain.pair_set n k, (a p.1 - c p.1) * (a p.2 - c p.2) := by
  classical
  rw [show ∑ p ∈ MChain.pair_set n k, a p.1 * a p.2 =
       ∑ p ∈ MChain.pair_set n k,
         (c p.1 * c p.2 +
          (c p.1 * (a p.2 - c p.2) + (a p.1 - c p.1) * c p.2) +
          (a p.1 - c p.1) * (a p.2 - c p.2)) from ?_]
  · rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
  · apply Finset.sum_congr rfl
    intro p _; ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: discrete_autoconvolution = Σ_{(i,j)∈pair_set} a_i · a_j
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The discrete autoconvolution at conv-position k equals the sum over `pair_set(k)`. -/
theorem MChain.discrete_autoconv_eq_pair_sum
    (n : ℕ) (a : Fin (2*n) → ℝ) (k : ℕ) :
    discrete_autoconvolution a k =
    ∑ p ∈ MChain.pair_set n k, a p.1 * a p.2 := by
  classical
  unfold discrete_autoconvolution
  rw [MChain.pair_set_eq_filter]
  -- Σ_i Σ_j (if i+j=k then a_i a_j else 0) = Σ over filter set.
  rw [Finset.sum_filter]
  rw [Fintype.sum_prod_type]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: M-chain core lower bound (integer-mass cell variable)
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**M-chain core lemma (integer-mass cell)**:

For any conv-position `k`, any composition `c : Fin (2n) → ℕ`, and any
real-valued cell variable `a : Fin (2n) → ℝ` (in integer-mass units, i.e.
the SAME scale as c) with `|a_i - c_i| ≤ 1` and `Σ a_i = Σ c_i`, we have
   `Σ_{(i,j) ∈ pair_set(k)} a_i · a_j  ≥
        Σ_{(i,j) ∈ pair_set(k)} c_i · c_j  −  2 · Δ_b(k)  −  n_pairs(k).`

The LHS equals `discrete_autoconvolution a k`; the RHS first summand equals
`discrete_autoconvolution (c : ℝ) k` (= the integer-c autoconv).

This is the heart of the M-chain bound (integer-mass form).  The
discrete-to-continuous bridge uses Lemma A and the m² scaling.
-/
theorem MChain.core_lower_bound
    (n : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ)
    (a : Fin (2*n) → ℝ)
    (h_close : ∀ i, |a i - (c i : ℝ)| ≤ 1)
    (h_sum : ∑ j : Fin (2*n), (a j - (c j : ℝ)) = 0) :
    (∑ p ∈ MChain.pair_set n k, ((c p.1 : ℝ) * (c p.2 : ℝ))) -
      2 * MChain.Delta_b n c k - (MChain.n_pairs n k : ℝ)
    ≤ ∑ p ∈ MChain.pair_set n k, a p.1 * a p.2 := by
  classical
  set δ : Fin (2*n) → ℝ := fun i => a i - (c i : ℝ) with hδ_def
  have h_abs : ∀ j, |δ j| ≤ 1 := fun j => h_close j
  have h_sum_eps : ∑ j : Fin (2*n), δ j = 0 := h_sum
  -- Decomposition of Σ a_i a_j = Σ c_i c_j + Σ (c_i δ_j + δ_i c_j) + Σ δ_i δ_j.
  have h_decomp :=
    MChain.conv_decomposition n k a (fun i : Fin (2*n) => (c i : ℝ))
  -- Use linear_identity to express linear via δ · b(k).
  have h_lin_left :
    (∑ p ∈ MChain.pair_set n k, (c p.1 : ℝ) * (a p.2 - (c p.2 : ℝ))) =
    ∑ j : Fin (2*n), δ j * MChain.b_vec n c k j := by
    have h_id := MChain.linear_identity_left n c k δ
    -- LHS = ∑_p (c p.1) * δ(p.2) (via δ = a - c).
    have h_rewrite : ∀ p : Fin (2*n) × Fin (2*n),
        (c p.1 : ℝ) * (a p.2 - (c p.2 : ℝ)) = (c p.1 : ℝ) * δ p.2 := by
      intro p; rw [hδ_def]
    simp_rw [h_rewrite]
    exact h_id
  have h_lin_right :
    (∑ p ∈ MChain.pair_set n k, (a p.1 - (c p.1 : ℝ)) * (c p.2 : ℝ)) =
    ∑ j : Fin (2*n), δ j * MChain.b_vec n c k j := by
    have h_id := MChain.linear_identity_right n c k δ
    have h_rewrite : ∀ p : Fin (2*n) × Fin (2*n),
        (a p.1 - (c p.1 : ℝ)) * (c p.2 : ℝ) = δ p.1 * (c p.2 : ℝ) := by
      intro p; rw [hδ_def]
    simp_rw [h_rewrite]
    exact h_id
  have h_lin_sum :
    (∑ p ∈ MChain.pair_set n k,
       ((c p.1 : ℝ) * (a p.2 - (c p.2 : ℝ)) + (a p.1 - (c p.1 : ℝ)) * (c p.2 : ℝ))) =
    2 * (∑ j : Fin (2*n), δ j * MChain.b_vec n c k j) := by
    rw [Finset.sum_add_distrib, h_lin_left, h_lin_right]; ring
  -- Quadratic identity (rewriting (a-c) as δ).
  have h_quad :
    (∑ p ∈ MChain.pair_set n k, (a p.1 - (c p.1 : ℝ)) * (a p.2 - (c p.2 : ℝ))) =
    ∑ p ∈ MChain.pair_set n k, δ p.1 * δ p.2 := by
    apply Finset.sum_congr rfl
    intro p _
    rw [hδ_def]
  -- Lemma B: Σ δ · b(k) ≥ -Δ_b
  have hB := MChain.lemma_B n c k δ h_abs h_sum_eps
  -- Lemma C: Σ δ_i δ_j ≥ -n_pairs
  have hC := MChain.lemma_C n k δ h_abs
  -- Combine.
  rw [h_decomp]
  rw [h_lin_sum]
  rw [h_quad]
  linarith [hB, hC]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 8: Lemma D — One-sided cell polytope subset (P_d^c ⊆ P_d)
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**Lemma D (one-sided δ refinement)**.

The M-chain's cell polytope: the cell-shape `a ∈ C(c)` corresponds in δ-space
to `δ ∈ P_d^c` where P_d^c = { δ : Σ δ = 0, δ_j ∈ [-min(c_j, 1), 1] }.

The cell condition `a_j ∈ [max(0, c_j - 1), c_j + 1]` in δ-coordinates (δ = a - c):
    δ_j ∈ [max(-c_j, -1), 1]  =  [-min(c_j, 1), 1].

In particular, the cell polytope P_d^c is contained in the symmetric polytope
P_d = { |δ| ≤ 1, Σ = 0 }, so the M-chain bounds (Lemmas B, C) which are sound
on P_d are also sound on P_d^c.

We state the inclusion in concrete form: if δ satisfies the cell constraint
(`δ_j ≥ -min(c_j, 1)` and `δ_j ≤ 1`), then `|δ_j| ≤ 1`.
-/
theorem MChain.lemma_D
    (n : ℕ) (c : Fin (2*n) → ℕ)
    (δ : Fin (2*n) → ℝ)
    (h_cell_lo : ∀ j, δ j ≥ -((min (c j) 1 : ℕ) : ℝ))
    (h_cell_hi : ∀ j, δ j ≤ 1) :
    ∀ j, |δ j| ≤ 1 := by
  intro j
  rw [abs_le]
  refine ⟨?_, h_cell_hi j⟩
  have h1 := h_cell_lo j
  -- h1 : δ j ≥ -((min (c j) 1 : ℕ) : ℝ)
  -- Note: min (c j) 1 ∈ {0, 1}, so -((min (c j) 1 : ℕ) : ℝ) ∈ {0, -1}.
  have h_min : (min (c j) 1 : ℕ) ≤ 1 := Nat.min_le_right _ _
  have h_min_real : ((min (c j) 1 : ℕ) : ℝ) ≤ 1 := by exact_mod_cast h_min
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 9: Lemma A (convolution at breakpoint, integer composition c)
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**Lemma A (convolution at breakpoint, integer-mass version)**.

For step function `step_function n m c` (heights `c_i / m`), the convolution at
the breakpoint `t_k = -1/2 + (k+1)·(1/(4n))` equals
   `(1 / (4n) / m²) · Σ_{i+j=k} c_i · c_j`
   `= (1 / (4n) / m²) · discrete_autoconvolution c k`.

This is exactly `convolution_at_grid_point` from `StepFunction.lean`.
We re-export it here under the M-chain name for clarity.

For real-valued cell variable `a : Fin (2n) → ℝ` (with |a_i - c_i| ≤ 1 etc.),
the analogous statement requires building a `step_function_real n m a` that
takes real-valued integer-mass-style cell variable directly.  The current
`step_function` infrastructure uses integer c only.  See `MChain.LB_step_at_c`
below for the c-only case.
-/
theorem MChain.lemma_A
    (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2*n) → ℕ) (hc : ∑ i, c i = 4 * n * m) (k : ℕ) :
    MeasureTheory.convolution (step_function n m c) (step_function n m c)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
      (-1/2 + (↑k + 1) * (1 / (4 * ↑n))) =
    (1 / (4 * (n : ℝ)) / (m : ℝ)^2) *
      discrete_autoconvolution (fun i : Fin (2*n) => (c i : ℝ)) k :=
  convolution_at_grid_point n m hn hm c hc k

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 10: M-chain LB and main per-position theorem
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**M-chain LB(k)**: per-cell lower bound on `(f_a*f_a)(t_k)` (in TV-style units).

  `LB(k) = (Σ_{i+j=k} c_i c_j  −  2·Δ_b(k)  −  n_pairs(k)) / (4·n·m²)`.

The denominator `4·n·m² = 2·d·m²` (with d = 2n) matches the Lemma A scaling
(f*f)(t_k) = (1/(4n·m²)) · conv[k](c) for the c-step function.

Matches the Python `_smoke_M_chain.py` definition (line 31, 135-136).
-/
noncomputable def MChain.LB (n m : ℕ) (c : Fin (2*n) → ℕ) (k : ℕ) : ℝ :=
  (∑ p ∈ MChain.pair_set n k, ((c p.1 : ℝ) * (c p.2 : ℝ)) -
     2 * MChain.Delta_b n c k -
     (MChain.n_pairs n k : ℝ)) / (4 * (n : ℝ) * (m : ℝ)^2)

/--
**M-chain main per-position theorem (integer-mass cell)**.

For composition `c : Fin (2n) → ℕ` and any real-valued cell variable
`a : Fin (2n) → ℝ` in the cell `C(c)` with `|a_i - c_i| ≤ 1` and
`Σ a_i = Σ c_i`, and for every conv-position k,
   `(1/(4n) / m²) · DA(a, k)  ≥  LB(k).`

LHS equals `(g_a * g_a)(t_k)` where `g_a` is the step function with heights
`a_i/m` (when `a` is integer-valued — for real `a` we treat this as a
formal "discrete autoconvolution" expression).
-/
theorem MChain.main_per_position
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2*n) → ℕ) (k : ℕ)
    (a : Fin (2*n) → ℝ)
    (h_close : ∀ i, |a i - (c i : ℝ)| ≤ 1)
    (h_sum : ∑ j : Fin (2*n), (a j - (c j : ℝ)) = 0) :
    MChain.LB n m c k ≤
      (1 / (4 * (n : ℝ)) / (m : ℝ)^2) *
        (∑ p ∈ MChain.pair_set n k, a p.1 * a p.2) := by
  classical
  have h_core := MChain.core_lower_bound n c k a h_close h_sum
  unfold MChain.LB
  have hn_real : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have hm_real : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have h4n : (0 : ℝ) < 4 * (n : ℝ) := by linarith
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  have h4nm_pos : (0 : ℝ) < 4 * (n : ℝ) * (m : ℝ)^2 := by positivity
  -- Rewrite: LB ≤ (1/(4n) / m²) · Σ a_i a_j
  --   ⇔  LB · (4n m²) ≤ Σ a_i a_j   (since 4n m² > 0)
  -- LB · (4n m²) = Σ c_i c_j - 2 Δ_b - n_pairs.  So h_core gives us this.
  rw [show (1 / (4 * (n : ℝ)) / (m : ℝ)^2) * (∑ p ∈ MChain.pair_set n k, a p.1 * a p.2) =
         (∑ p ∈ MChain.pair_set n k, a p.1 * a p.2) / (4 * (n : ℝ) * (m : ℝ)^2) from by
    field_simp]
  exact div_le_div_of_nonneg_right h_core h4nm_pos.le

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 11: Pointwise lower bound on (f_c * f_c)(t_k)
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**Pointwise LB at the integer composition c (no perturbation)**.

For c with Σ c = 4nm, the convolution of `step_function n m c` with itself at
breakpoint t_k satisfies
   `LB(k)  ≤  (step_function n m c * step_function n m c)(t_k)`.

This is the trivial case δ = 0 of the M-chain bound (a = c).  Useful as a
sanity check / direct usage at a "tight" composition.
-/
theorem MChain.LB_step_at_c
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2*n) → ℕ) (hc : ∑ i, c i = 4 * n * m) (k : ℕ) :
    MChain.LB n m c k ≤
      MeasureTheory.convolution (step_function n m c) (step_function n m c)
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
        (-1/2 + (↑k + 1) * (1 / (4 * ↑n))) := by
  classical
  -- Lemma A: RHS = (1/(4n) / m²) · DA(c, k).
  rw [MChain.lemma_A n m hn hm c hc k]
  -- Now: LB ≤ (1/(4n) / m²) · DA(c, k).
  -- Apply main_per_position with `a` = (c : ℝ) (so δ = 0).
  set a : Fin (2*n) → ℝ := fun i => (c i : ℝ) with ha_def
  have h_close : ∀ i, |a i - (c i : ℝ)| ≤ 1 := by
    intro i
    rw [ha_def, sub_self, abs_zero]
    norm_num
  have h_sum : ∑ j : Fin (2*n), (a j - (c j : ℝ)) = 0 := by
    apply Finset.sum_eq_zero
    intro j _
    rw [ha_def, sub_self]
  have h := MChain.main_per_position n m hn hm c k a h_close h_sum
  -- h : LB(k) ≤ (1/(4n) / m²) · Σ a p.1 a p.2 over pair_set(k).
  -- And Σ a p.1 a p.2 = Σ_{i+j=k} (c i)(c j) = DA((c : ℝ), k) by
  -- discrete_autoconv_eq_pair_sum.
  rw [← MChain.discrete_autoconv_eq_pair_sum n a k] at h
  -- h : LB(k) ≤ (1/(4n) / m²) · DA(a, k).  And a = (c : ℝ).
  exact h

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 12: Connection to (f_a * f_a)(t_k) for real-valued cell variable
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**M-chain main, abstract form (real-mass cell variable)**.

For any composition `c` and any real-valued cell variable `a : Fin (2n) → ℝ`
satisfying `|a_i - c_i| ≤ 1` and `Σ a_i = Σ c_i`, and for every k,
   `LB(k)  ≤  DA(a, k) / (4n · m²)`.

LHS is the M-chain lower bound; RHS is the "would-be (f_a * f_a)(t_k)" if `a`
were realised as a step function with heights `a_i / m`.  This is exactly the
formal statement of the M-chain pruning bound (Theorem 5 of `m_chain_proof.md`)
in discrete-autoconvolution form.

When `a` is a continuous cell variable (not an integer composition), the
"corresponding" step function does not literally exist as a `step_function n m c`
since `c` is required to be ℕ-valued.  But the discrete-autoconv inequality
above is exactly what we need: it implies that ANY realisation of `a` as a
step function (with heights `a/m`) would have `(f_a * f_a)(t_k) ≥ LB(k)`.
-/
theorem MChain.main_DA_form
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2*n) → ℕ) (k : ℕ)
    (a : Fin (2*n) → ℝ)
    (h_close : ∀ i, |a i - (c i : ℝ)| ≤ 1)
    (h_sum : ∑ j : Fin (2*n), (a j - (c j : ℝ)) = 0) :
    MChain.LB n m c k ≤ discrete_autoconvolution a k / (4 * (n : ℝ) * (m : ℝ)^2) := by
  classical
  have h := MChain.main_per_position n m hn hm c k a h_close h_sum
  rw [← MChain.discrete_autoconv_eq_pair_sum n a k] at h
  -- h : LB ≤ (1/(4n) / m²) · DA(a, k).
  have h_rewrite : (1 / (4 * (n : ℝ)) / (m : ℝ)^2) * discrete_autoconvolution a k =
                   discrete_autoconvolution a k / (4 * (n : ℝ) * (m : ℝ)^2) := by
    field_simp
  rw [h_rewrite] at h
  exact h

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 13: M-chain "maximum over k" cell-prune theorem
-- ═══════════════════════════════════════════════════════════════════════════════

/--
**M-chain cell-prune soundness (per-position form)**.

If `LB(k) > c_target` for some k, then for every cell variable `a` (real-valued
with cell condition `|a_i - c_i| ≤ 1` and `Σ a = Σ c`), the discrete
autoconvolution at conv-position k satisfies `DA(a, k) / (4n·m²) > c_target`.

This is the "M-prunable" condition from `_smoke_M_chain.py`: if some
`max_k LB(k) > c_target`, the cell cannot host any C_{1a} witness
(in step-function space at this resolution).
-/
theorem MChain.cell_prune_per_position
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2*n) → ℕ) (c_target : ℝ) (k : ℕ)
    (h_LB : MChain.LB n m c k > c_target)
    (a : Fin (2*n) → ℝ)
    (h_close : ∀ i, |a i - (c i : ℝ)| ≤ 1)
    (h_sum : ∑ j : Fin (2*n), (a j - (c j : ℝ)) = 0) :
    discrete_autoconvolution a k / (4 * (n : ℝ) * (m : ℝ)^2) > c_target := by
  have h := MChain.main_DA_form n m hn hm c k a h_close h_sum
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 14: Sanity examples (matching m_chain_proof.md §6)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Sanity example**: n=1, d=2, m=20, c=(40, 40), k=1.
    From `_smoke_M_chain.py:_sanity_b_vec_and_LB`:
      conv = [1600, 3200, 1600], b(1) = (40, 40), Δ_b(1) = 0,
      n_pairs(1) = 2, LB(1) = (3200 - 0 - 2) / 1600 = 3198/1600 = 1.99875.
    -/
example : MChain.n_pairs 1 1 = 2 := by
  unfold MChain.n_pairs ell_int_arr
  simp

example : MChain.n_pairs 1 0 = 1 := by
  unfold MChain.n_pairs ell_int_arr
  simp

example : MChain.n_pairs 1 2 = 1 := by
  unfold MChain.n_pairs ell_int_arr
  simp

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 15: AUDIT BLOCK
-- ═══════════════════════════════════════════════════════════════════════════════

end -- noncomputable section

/-
═══════════════════════════════════════════════════════════════════════════════
AUDIT BLOCK (M-chain)
═══════════════════════════════════════════════════════════════════════════════

NEW AXIOMS DECLARED IN THIS FILE: ZERO.
NEW SORRIES DECLARED IN THIS FILE: ZERO.

LEMMA / THEOREM  →  HYPOTHESES USED
─────────────────────────────────────
MChain.icc_singleton            : (no hypotheses).
MChain.b_vec_eq                 : (no hypotheses).
MChain.pair_set_eq              : (no hypotheses).
MChain.pair_set_eq_filter       : (no hypotheses).
MChain.pair_set_card            : (no hypotheses).
MChain.lemma_B                  : |δ_j| ≤ 1 ∀j;  Σ δ_j = 0.
MChain.lemma_C_abs              : |δ_i| ≤ 1 ∀i.
MChain.lemma_C                  : |δ_i| ≤ 1 ∀i.
MChain.linear_identity_left     : (no hypotheses).
MChain.linear_identity_right    : (no hypotheses).
MChain.conv_decomposition       : (no hypotheses).
MChain.discrete_autoconv_eq_pair_sum : (no hypotheses).
MChain.core_lower_bound         : |a_i - c_i| ≤ 1;  Σ (a - c) = 0.
MChain.lemma_D                  : δ_j ≥ -min(c_j, 1);  δ_j ≤ 1.
MChain.lemma_A                  : n > 0, m > 0;  Σ c = 4nm.
MChain.LB                       : (definition only).
MChain.main_per_position        : 0 < n, 0 < m;  |a_i - c_i| ≤ 1;
                                  Σ (a - c) = 0.
MChain.LB_step_at_c             : 0 < n, 0 < m;  Σ c = 4nm.
MChain.main_DA_form             : 0 < n, 0 < m;  |a_i - c_i| ≤ 1;
                                  Σ (a - c) = 0.
MChain.cell_prune_per_position  : 0 < n, 0 < m;  LB(k) > c_target;
                                  |a_i - c_i| ≤ 1;  Σ (a - c) = 0.

REUSE OF EXISTING INFRASTRUCTURE:
  - `BB_W`, `Delta_BB`, `lp_closed_form_le` from PostFilterF.lean
    (specialised to single-conv window: s_lo = k, ℓ = 2).
  - `ell_int_arr`, `ell_int_arr_eq_card`, `window_pair_set`
    from TightDiscretizationBound.lean.
  - `discrete_autoconvolution` from Sidon.Defs.lean.
  - `step_function`, `convolution_at_grid_point` from StepFunction.lean.

PYTHON CROSS-REFERENCE:  `_smoke_M_chain.py` (line numbers given inline).
-/
