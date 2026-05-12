/-
Sidon Autocorrelation Project — Post-Filter F (LP-tight linear bound)

This file formalizes **variant F**, the LP-tight linear-correction tightening
of variant D.  The key new ingredient is the **LP closed-form**:

  For B : Fin (2k) → ℝ and δ with Σ δ = 0 and |δⱼ| ≤ h,
     Σⱼ δⱼ Bⱼ ≤ h · Δ_B
  where  Δ_B = (Σ of top k of sorted B) − (Σ of bot k of sorted B).

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL CONTENT
═══════════════════════════════════════════════════════════════════════════════

Variant F replaces variant D's loose linear bound `(ℓ−1) · W_int_overlap`
with the LP-tight `Δ_BB`, where for window W:
  BB^W_j := ∑_{i : (i,j) ∈ W} c_i  (the "row-i mass" indexed by j),
  Δ_BB   := (sum of top n of sorted BB^W) − (sum of bot n of sorted BB^W).

The variant-F correction in m² units is:
  corr_F_m2 = Δ_BB / (2n·ℓ) + ell_int_sum / (4n·ℓ).

Compare to variant D:
  corr_tight_m2 = (ℓ−1) · W_int / (2n·ℓ) + ell_int_sum / (4n·ℓ).

Since Δ_BB ≤ Σⱼ BB^W_j = Σᵢ cᵢ · Nᵢ ≤ (ℓ−1) · W_int_overlap, we have
  corr_F_m2 ≤ corr_tight_m2 (F is tighter than D).

PYTHON CROSS-REFERENCE: `_M1_bench.py:prune_F` (lines 229-335).

NEW AXIOMS DECLARED IN THIS FILE: ZERO.
═══════════════════════════════════════════════════════════════════════════════
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.WRefinedDefs
import Sidon.Proof.TightDiscretizationBound

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
-- Part 1: BB_W — the row-mass indexed by j, BB_W^j = ∑_{i : (i,j) ∈ W} c_i
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The "column count" function: BB_W^j = Σ_{i : (i,j) ∈ W} c_i.

    For a window W = {(i,j) : i+j ∈ [s_lo, s_lo+ℓ−2]}, this counts the integer
    mass `c_i` of row i for each fixed column j such that (i,j) ∈ W.

    Matches Python `_M1_bench.py:prune_F` lines 297-309 (`BB[j]` array). -/
noncomputable def BB_W (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ)
    (j : Fin (2 * n)) : ℝ :=
  ∑ i : Fin (2 * n), if i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
                     then (c i : ℝ) else 0

/-- BB_W is non-negative. -/
theorem BB_W_nonneg (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) (j : Fin (2 * n)) :
    0 ≤ BB_W n c s_lo ℓ j := by
  unfold BB_W
  apply Finset.sum_nonneg
  intro i _
  by_cases hk : i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
  · rw [if_pos hk]; exact Nat.cast_nonneg _
  · rw [if_neg hk]

/-- The total of BB_W over all j equals Σᵢ cᵢ · Nᵢ. -/
theorem sum_BB_W_eq_sum_c_N (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    ∑ j : Fin (2 * n), BB_W n c s_lo ℓ j =
      ∑ i : Fin (2 * n), (c i : ℝ) * (N_row n s_lo ℓ i : ℝ) := by
  classical
  unfold BB_W
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  -- ∑_j (if (i,j) ∈ W then c_i else 0) = c_i · #{j : (i,j) ∈ W} = c_i · N_row i.
  unfold N_row
  rw [Finset.card_filter]
  push_cast
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hk : i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
  · rw [if_pos hk, if_pos hk]; ring
  · rw [if_neg hk, if_neg hk]; ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: Delta_BB — sorted-extremes value (top-half sum − bottom-half sum)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The sorting permutation `sigma_BB` for `BB_W n c s_lo ℓ`.  This permutation
    rearranges `Fin (2n)` so that `BB_W ∘ sigma_BB` is monotone (ascending). -/
noncomputable def sigma_BB (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    Equiv.Perm (Fin (2 * n)) :=
  Tuple.sort (BB_W n c s_lo ℓ)

/-- `BB_W ∘ sigma_BB` is monotone (ascending). -/
theorem BB_sigma_monotone (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    Monotone ((BB_W n c s_lo ℓ) ∘ (sigma_BB n c s_lo ℓ)) := by
  unfold sigma_BB
  exact Tuple.monotone_sort (BB_W n c s_lo ℓ)

/-- The top-half index set: {j : Fin (2n) | n ≤ j.val}. -/
def top_half_set (n : ℕ) : Finset (Fin (2 * n)) :=
  (Finset.univ : Finset (Fin (2 * n))).filter (fun j => n ≤ j.val)

/-- The bottom-half index set: {j : Fin (2n) | j.val < n}. -/
def bot_half_set (n : ℕ) : Finset (Fin (2 * n)) :=
  (Finset.univ : Finset (Fin (2 * n))).filter (fun j => j.val < n)

/-- The two halves are disjoint. -/
theorem top_bot_disjoint (n : ℕ) :
    Disjoint (top_half_set n) (bot_half_set n) := by
  unfold top_half_set bot_half_set
  rw [Finset.disjoint_filter]
  intro j _ hj
  omega

/-- Their union is the universe. -/
theorem top_bot_union_eq_univ (n : ℕ) :
    (top_half_set n) ∪ (bot_half_set n) = (Finset.univ : Finset (Fin (2 * n))) := by
  unfold top_half_set bot_half_set
  ext j
  simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
  -- Goal: n ≤ ↑j ∨ ↑j < n.
  exact le_or_gt n j.val

/-- The bottom-half set has card `n`. -/
theorem bot_half_card (n : ℕ) : (bot_half_set n).card = n := by
  classical
  -- Compute LHS = #{j : Fin (2n) | j.val < n} via biject to Fin n.
  have h_card_n : (Finset.univ : Finset (Fin n)).card = n :=
    Finset.card_univ.trans (Fintype.card_fin n)
  have h_eq : (bot_half_set n).card = (Finset.univ : Finset (Fin n)).card := by
    apply Finset.card_bij (s := bot_half_set n) (t := (Finset.univ : Finset (Fin n)))
      (i := fun (j : Fin (2 * n)) (hj : j ∈ bot_half_set n) =>
        (⟨j.val, by
          have : j.val < n := by
            unfold bot_half_set at hj
            simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
            exact hj
          exact this⟩ : Fin n))
    · intro j _; exact Finset.mem_univ _
    · intro j₁ hj₁ j₂ hj₂ heq
      apply Fin.ext
      exact Fin.mk_eq_mk.mp heq
    · intro k _
      refine ⟨(⟨k.val, by have := k.2; omega⟩ : Fin (2 * n)), ?_, ?_⟩
      · unfold bot_half_set
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact k.2
      · rfl
  rw [h_eq, h_card_n]

/-- The top-half set has card `n`. -/
theorem top_half_card (n : ℕ) : (top_half_set n).card = n := by
  classical
  -- Total card = 2n; bot card = n; top + bot = univ; disjoint ⇒ top = 2n − n = n.
  have h_disj := top_bot_disjoint n
  have h_union := top_bot_union_eq_univ n
  have h_total : (top_half_set n).card + (bot_half_set n).card =
                 (Finset.univ : Finset (Fin (2 * n))).card := by
    rw [← Finset.card_union_of_disjoint h_disj, h_union]
  rw [bot_half_card n, Finset.card_univ, Fintype.card_fin] at h_total
  omega

/-- The variant-F LP closed-form value for BB_W: top-half sum minus bottom-half sum
    over the sort order.

    `Delta_BB n c s_lo ℓ = (Σ over top-half-sorted) − (Σ over bot-half-sorted)`.

    Matches Python `_M1_bench.py:prune_F` lines 311-320. -/
noncomputable def Delta_BB (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) : ℝ :=
  ∑ j ∈ top_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) -
  ∑ j ∈ bot_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j)

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: Δ_BB ≤ Σ_j BB^W_j  (used for F ≤ D dominance)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The sum over both halves equals the total. -/
theorem sum_top_add_bot (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    ∑ j ∈ top_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) +
    ∑ j ∈ bot_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) =
    ∑ j : Fin (2 * n), BB_W n c s_lo ℓ j := by
  classical
  rw [← Finset.sum_union (top_bot_disjoint n), top_bot_union_eq_univ n]
  -- Now LHS = Σ_j (BB_W ∘ σ) j, RHS = Σ_j BB_W j; equal by Equiv.sum_comp.
  exact Equiv.sum_comp (sigma_BB n c s_lo ℓ) (BB_W n c s_lo ℓ)

/-- Sum over bot half is non-negative. -/
theorem sum_bot_nonneg (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    0 ≤ ∑ j ∈ bot_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) := by
  apply Finset.sum_nonneg
  intro j _
  exact BB_W_nonneg n c s_lo ℓ _

/-- Sum over top half is non-negative. -/
theorem sum_top_nonneg (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    0 ≤ ∑ j ∈ top_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) := by
  apply Finset.sum_nonneg
  intro j _
  exact BB_W_nonneg n c s_lo ℓ _

/-- **Δ_BB ≤ total mass**: `Δ_BB ≤ Σⱼ BB^W_j`.

    This is the key step for F ≤ D dominance: top-half - bot-half ≤ top-half + bot-half
    when bot-half ≥ 0. -/
theorem Delta_BB_le_total (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    Delta_BB n c s_lo ℓ ≤ ∑ j : Fin (2 * n), BB_W n c s_lo ℓ j := by
  unfold Delta_BB
  have h_sum := sum_top_add_bot n c s_lo ℓ
  have h_bot_nn := sum_bot_nonneg n c s_lo ℓ
  linarith

/-- The pairing bijection between bot_half and top_half: `j ↦ ⟨j.val + n, _⟩`. -/
noncomputable def shift_n (n : ℕ) (j : Fin (2 * n)) (hj : j.val < n) : Fin (2 * n) :=
  ⟨j.val + n, by omega⟩

/-- For `BB_W ∘ σ` monotone, `j < n ⇒ (BB_W ∘ σ)(j) ≤ (BB_W ∘ σ)(j + n)`. -/
theorem BB_sigma_pair_le (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ)
    (j : Fin (2 * n)) (hj : j.val < n) :
    BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) ≤
    BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ (shift_n n j hj)) := by
  have h_mono := BB_sigma_monotone n c s_lo ℓ
  apply h_mono
  show j ≤ shift_n n j hj
  show j.val ≤ j.val + n
  omega

/-- Δ_BB is non-negative.  After ascending-sort, top half ≥ bot half. -/
theorem Delta_BB_nonneg (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) :
    0 ≤ Delta_BB n c s_lo ℓ := by
  classical
  unfold Delta_BB
  -- Strategy: rewrite both sums using the obvious bijection
  -- `bot_half_set ≃ Fin n` via j ↦ ⟨j.val, _⟩, and similarly top via j ↦ ⟨j.val - n, _⟩.
  -- Then both sums become ∑_{k : Fin n} of the "k-th" sorted element, with
  -- bot at position k (sorted ascending) ≤ top at position k+n (sorted ascending).
  have h_bot_eq :
      ∑ j ∈ bot_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) =
      ∑ k : Fin n, BB_W n c s_lo ℓ
        (sigma_BB n c s_lo ℓ (⟨k.val, by have := k.2; omega⟩ : Fin (2 * n))) := by
    apply Finset.sum_bij (fun (j : Fin (2 * n)) (hj : j ∈ bot_half_set n) =>
        (⟨j.val, by
          have h := (Finset.mem_filter.mp hj).2
          unfold bot_half_set at hj
          simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
          exact hj⟩ : Fin n))
    · intro j _; exact Finset.mem_univ _
    · intro j₁ _ j₂ _ heq
      apply Fin.ext
      exact Fin.mk_eq_mk.mp heq
    · intro k _
      refine ⟨⟨k.val, by have := k.2; omega⟩, ?_, ?_⟩
      · unfold bot_half_set
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact k.2
      · rfl
    · intro j _; rfl
  have h_top_eq :
      ∑ j ∈ top_half_set n, BB_W n c s_lo ℓ (sigma_BB n c s_lo ℓ j) =
      ∑ k : Fin n, BB_W n c s_lo ℓ
        (sigma_BB n c s_lo ℓ (⟨k.val + n, by have := k.2; omega⟩ : Fin (2 * n))) := by
    apply Finset.sum_bij (fun (j : Fin (2 * n)) (hj : j ∈ top_half_set n) =>
        (⟨j.val - n, by
          unfold top_half_set at hj
          simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
          have hjlt : j.val < 2 * n := j.2
          omega⟩ : Fin n))
    · intro j _; exact Finset.mem_univ _
    · intro j₁ hj₁ j₂ hj₂ heq
      unfold top_half_set at hj₁ hj₂
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj₁ hj₂
      apply Fin.ext
      have h := Fin.mk_eq_mk.mp heq
      omega
    · intro k _
      refine ⟨⟨k.val + n, by have := k.2; omega⟩, ?_, ?_⟩
      · unfold top_half_set
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]; omega
      · apply Fin.ext; show k.val + n - n = k.val; omega
    · intro j hj
      unfold top_half_set at hj
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
      apply congrArg
      apply congrArg
      apply Fin.ext
      show j.val = (j.val - n) + n
      omega
  rw [h_bot_eq, h_top_eq]
  have h_diff :
      ∑ k : Fin n, BB_W n c s_lo ℓ
        (sigma_BB n c s_lo ℓ (⟨k.val + n, by have := k.2; omega⟩ : Fin (2 * n))) -
      ∑ k : Fin n, BB_W n c s_lo ℓ
        (sigma_BB n c s_lo ℓ (⟨k.val, by have := k.2; omega⟩ : Fin (2 * n))) =
      ∑ k : Fin n, (BB_W n c s_lo ℓ
        (sigma_BB n c s_lo ℓ (⟨k.val + n, by have := k.2; omega⟩ : Fin (2 * n))) -
       BB_W n c s_lo ℓ
        (sigma_BB n c s_lo ℓ (⟨k.val, by have := k.2; omega⟩ : Fin (2 * n)))) := by
    rw [← Finset.sum_sub_distrib]
  rw [h_diff]
  apply Finset.sum_nonneg
  intro k _
  have h_mono := BB_sigma_monotone n c s_lo ℓ
  have hle : (⟨k.val, by have := k.2; omega⟩ : Fin (2 * n)) ≤
             (⟨k.val + n, by have := k.2; omega⟩ : Fin (2 * n)) := by
    show k.val ≤ k.val + n
    omega
  have h_le := h_mono hle
  -- h_le has form `(BB_W n c s_lo ℓ ∘ sigma_BB n c s_lo ℓ) ⟨k.val,_⟩ ≤
  --                (BB_W n c s_lo ℓ ∘ sigma_BB n c s_lo ℓ) ⟨k.val + n,_⟩`.
  -- Function.comp_apply unfolds to direct application.
  simp only [Function.comp_apply] at h_le
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: LP closed-form lemma (the critical new content)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **LP closed-form lemma (for BB_W)**:

    For any δ : Fin (2n) → ℝ with Σ δ = 0 and |δⱼ| ≤ h (h ≥ 0),
       Σⱼ δⱼ · BB_W^j ≤ h · Δ_BB.

    PROOF (by reduction to sorted form via `sigma_BB`):
    1. Set σ = `sigma_BB n c s_lo ℓ`.  Then `BB_W ∘ σ` is monotone (ascending).
    2. By `Equiv.sum_comp` applied to `σ`, we have
         Σⱼ δⱼ BB_W^j = Σⱼ (δ ∘ σ)(j) · (BB_W ∘ σ)(j).
       Also Σ (δ ∘ σ) = Σ δ = 0 and |(δ ∘ σ) j| ≤ h.
    3. Pick pivot μ = (BB_W ∘ σ)(⟨n, _⟩) when n ≥ 1; for n = 0 the sums are empty.
    4. Use Σ (δ ∘ σ) = 0 to substitute Σⱼ (δ∘σ)(j) (BB_W∘σ)(j) =
       Σⱼ (δ∘σ)(j) ((BB_W∘σ)(j) - μ).
    5. For j with j.val < n: (BB_W∘σ)(j) ≤ μ, so the term ≤ h(μ - (BB_W∘σ)(j)).
    6. For j with j.val ≥ n: (BB_W∘σ)(j) ≥ μ, so the term ≤ h((BB_W∘σ)(j) - μ).
    7. Sum: h · ((Σ_top - Σ_bot) - n·μ + n·μ) = h · Δ_BB.

    Mathematical reference: `_M1_bench.py:1-49` (LP duality argument). -/
theorem lp_closed_form_le (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ)
    (δ : Fin (2 * n) → ℝ) (h : ℝ) (h_nn : 0 ≤ h)
    (h_close : ∀ j, |δ j| ≤ h)
    (h_sum : ∑ j : Fin (2 * n), δ j = 0) :
    ∑ j : Fin (2 * n), δ j * BB_W n c s_lo ℓ j ≤ h * Delta_BB n c s_lo ℓ := by
  classical
  set σ := sigma_BB n c s_lo ℓ with hσ_def
  set B := BB_W n c s_lo ℓ with hB_def
  -- Step 1: reindex Σ_j δ_j B_j = Σ_k (δ ∘ σ)_k (B ∘ σ)_k via Equiv.sum_comp.
  have h_reindex : ∑ j : Fin (2 * n), δ j * B j =
                   ∑ k : Fin (2 * n), δ (σ k) * B (σ k) := by
    rw [← Equiv.sum_comp σ (fun j => δ j * B j)]
  rw [h_reindex]
  -- Step 2: also reindex Σ δ = 0 to Σ (δ ∘ σ) = 0.
  have h_sum_perm : ∑ k : Fin (2 * n), δ (σ k) = 0 := by
    rw [Equiv.sum_comp σ δ]; exact h_sum
  -- Step 3: also reindex |·| ≤ h to (δ ∘ σ).
  have h_close_perm : ∀ k, |δ (σ k)| ≤ h := fun k => h_close (σ k)
  -- Step 4: |·| ≤ h ⇒ -h ≤ δ_k ≤ h.
  have h_abs_iff : ∀ k, -h ≤ δ (σ k) ∧ δ (σ k) ≤ h := by
    intro k
    have h_abs := h_close_perm k
    exact abs_le.mp h_abs
  -- Step 5: for n = 0 the bound is trivial; for n > 0, use the median pivot.
  by_cases hn : n = 0
  · -- Empty case: Fin (2*0) is empty, both sums are 0.
    subst hn
    have hdelta : Delta_BB 0 c s_lo ℓ = 0 := by
      unfold Delta_BB
      have h_top : top_half_set 0 = ∅ := by
        unfold top_half_set
        ext j; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty]
        have := j.2; omega
      have h_bot : bot_half_set 0 = ∅ := by
        unfold bot_half_set
        ext j; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty]
        have := j.2; omega
      rw [h_top, h_bot]
      simp
    rw [hdelta]
    -- LHS sum is over Fin 0 which is empty.
    have h_emp : ∀ k : Fin (2 * 0), False := fun k => Nat.not_lt_zero k.val (by simpa using k.2)
    have h_lhs : ∑ k : Fin (2 * 0), δ (σ k) * B (σ k) = 0 := by
      apply Finset.sum_eq_zero
      intro k _
      exact (h_emp k).elim
    rw [h_lhs]
    linarith
  -- Now n ≥ 1.  Pick pivot μ = (B ∘ σ)(⟨n, _⟩).
  push_neg at hn
  have hn_pos : 0 < n := Nat.pos_of_ne_zero hn
  have hn_lt : n < 2 * n := by omega
  set μ := B (σ ⟨n, hn_lt⟩) with hμ_def
  -- Σ (δ∘σ)(k) (B∘σ)(k) = Σ (δ∘σ)(k) ((B∘σ)(k) - μ) using Σ (δ∘σ) = 0.
  have h_pivot : ∑ k : Fin (2 * n), δ (σ k) * B (σ k) =
                 ∑ k : Fin (2 * n), δ (σ k) * (B (σ k) - μ) := by
    -- Σ δ_k B_k = Σ δ_k (B_k - μ) + Σ δ_k μ = Σ δ_k (B_k - μ) + μ · Σ δ_k = ... + 0.
    have h_split : ∀ k : Fin (2 * n),
        δ (σ k) * B (σ k) = δ (σ k) * (B (σ k) - μ) + δ (σ k) * μ := by
      intro k; ring
    simp_rw [h_split]
    rw [Finset.sum_add_distrib]
    rw [show ∑ k : Fin (2 * n), δ (σ k) * μ =
         (∑ k : Fin (2 * n), δ (σ k)) * μ from (Finset.sum_mul _ _ _).symm]
    rw [h_sum_perm]; ring
  rw [h_pivot]
  -- Split Fin (2n) into top_half_set ∪ bot_half_set.
  have h_split_sum :
      ∑ k : Fin (2 * n), δ (σ k) * (B (σ k) - μ) =
      ∑ k ∈ top_half_set n, δ (σ k) * (B (σ k) - μ) +
      ∑ k ∈ bot_half_set n, δ (σ k) * (B (σ k) - μ) := by
    rw [← Finset.sum_union (top_bot_disjoint n), top_bot_union_eq_univ n]
  rw [h_split_sum]
  -- Bot half: B(σ k) ≤ μ (k < n, σ ascending). δ(σ k)(B(σ k) - μ) ≤ -h(B(σ k) - μ) = h(μ - B(σ k)).
  have h_mono_BB : Monotone (B ∘ σ) := by
    rw [hσ_def, hB_def]; exact Tuple.monotone_sort _
  have h_bot_le :
      ∑ k ∈ bot_half_set n, δ (σ k) * (B (σ k) - μ) ≤
      ∑ k ∈ bot_half_set n, h * (μ - B (σ k)) := by
    apply Finset.sum_le_sum
    intro k hk
    unfold bot_half_set at hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    -- B(σ k) ≤ μ since k.val < n and σ is sorted ascending.
    have h_le_pivot : B (σ k) ≤ μ := by
      have h_le_n : k ≤ (⟨n, hn_lt⟩ : Fin (2 * n)) := by
        show k.val ≤ n
        omega
      have := h_mono_BB h_le_n
      simp only [Function.comp_apply] at this
      exact this
    -- δ(σ k) (B(σ k) - μ) ≤ h(μ - B(σ k))   when |δ(σ k)| ≤ h.
    have h_neg : B (σ k) - μ ≤ 0 := by linarith
    have h_abs := h_abs_iff k
    -- δ(σ k) (B(σ k) - μ) ≤ -h · (B(σ k) - μ) = h(μ - B(σ k))
    have hδ : -h ≤ δ (σ k) := h_abs.1
    -- Multiply both sides of -h ≤ δ(σ k) by (μ - B(σ k)) ≥ 0:
    have h_pos : 0 ≤ μ - B (σ k) := by linarith
    -- δ(σ k) · (B(σ k) - μ) = -δ(σ k) · (μ - B(σ k))
    -- and -δ(σ k) ≤ h, so δ(σ k) · (B(σ k) - μ) ≤ h · (μ - B(σ k)).
    nlinarith [h_pos, hδ]
  -- Top half: B(σ k) ≥ μ (k ≥ n, σ ascending). δ(σ k)(B(σ k) - μ) ≤ h(B(σ k) - μ).
  have h_top_le :
      ∑ k ∈ top_half_set n, δ (σ k) * (B (σ k) - μ) ≤
      ∑ k ∈ top_half_set n, h * (B (σ k) - μ) := by
    apply Finset.sum_le_sum
    intro k hk
    unfold top_half_set at hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    -- B(σ k) ≥ μ since k.val ≥ n.
    have h_ge_pivot : μ ≤ B (σ k) := by
      have h_ge_n : (⟨n, hn_lt⟩ : Fin (2 * n)) ≤ k := by
        show n ≤ k.val
        exact hk
      have := h_mono_BB h_ge_n
      simp only [Function.comp_apply] at this
      exact this
    have h_pos : 0 ≤ B (σ k) - μ := by linarith
    have h_abs := h_abs_iff k
    have hδ : δ (σ k) ≤ h := h_abs.2
    nlinarith [h_pos, hδ]
  -- Combine: Σ_top + Σ_bot ≤ h · (Σ_top (B-μ) + Σ_bot (μ-B)) = h · Δ_BB.
  -- Note Σ_top h(B-μ) + Σ_bot h(μ-B) = h·(Σ_top B - n·μ + n·μ - Σ_bot B) = h·Δ_BB.
  -- Use: |top half| = n, |bot half| = n.
  have h_top_eq :
      ∑ k ∈ top_half_set n, h * (B (σ k) - μ) =
      h * (∑ k ∈ top_half_set n, B (σ k)) - h * n * μ := by
    rw [show (h * (∑ k ∈ top_half_set n, B (σ k)) - h * n * μ) =
            h * (∑ k ∈ top_half_set n, B (σ k)) -
            h * (∑ _ ∈ top_half_set n, μ) from by
        rw [show (∑ _ ∈ top_half_set n, μ) = (top_half_set n).card • μ from
          Finset.sum_const _]
        rw [top_half_card]
        ring]
    rw [show h * (∑ k ∈ top_half_set n, B (σ k)) =
          ∑ k ∈ top_half_set n, h * B (σ k) from Finset.mul_sum _ _ _]
    rw [show h * (∑ _ ∈ top_half_set n, μ) =
          ∑ _ ∈ top_half_set n, h * μ from Finset.mul_sum _ _ _]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro k _; ring
  have h_bot_eq :
      ∑ k ∈ bot_half_set n, h * (μ - B (σ k)) =
      h * n * μ - h * (∑ k ∈ bot_half_set n, B (σ k)) := by
    rw [show (h * n * μ - h * (∑ k ∈ bot_half_set n, B (σ k))) =
            h * (∑ _ ∈ bot_half_set n, μ) -
            h * (∑ k ∈ bot_half_set n, B (σ k)) from by
        rw [show (∑ _ ∈ bot_half_set n, μ) = (bot_half_set n).card • μ from
          Finset.sum_const _]
        rw [bot_half_card]
        ring]
    rw [show h * (∑ _ ∈ bot_half_set n, μ) =
          ∑ _ ∈ bot_half_set n, h * μ from Finset.mul_sum _ _ _]
    rw [show h * (∑ k ∈ bot_half_set n, B (σ k)) =
          ∑ k ∈ bot_half_set n, h * B (σ k) from Finset.mul_sum _ _ _]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro k _; ring
  have h_sum_le_diff :
      (∑ k ∈ top_half_set n, h * (B (σ k) - μ)) +
      (∑ k ∈ bot_half_set n, h * (μ - B (σ k))) =
      h * Delta_BB n c s_lo ℓ := by
    rw [h_top_eq, h_bot_eq]
    unfold Delta_BB
    rw [hσ_def, hB_def]
    ring
  linarith [h_top_le, h_bot_le, h_sum_le_diff]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: Linear-term bound via LP closed-form (variant F)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Linear bound (variant F, raw)**:
       |Σ_W (b_i ε_j + b_j ε_i)| ≤ 2 · Δ_BB / m²
    where b_i = c_i/m and ε_i = c_i/m - a_i, given Σ a = Σ c/m (so Σ ε = 0).

    The bound is achieved via:
       Σ_W b_i ε_j = (1/m) Σ_j ε_j BB^j  ≤  (1/m) · (1/m) · Δ_BB  (LP closed-form, h = 1/m)
    and similarly for the swap term.

    Hypotheses: m > 0, ℓ ≥ 2, |a_i - c_i/m| ≤ 1/m, Σ a = Σ c/m. -/
theorem linear_window_bound_F
    (n m : ℕ) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |∑ p ∈ window_pair_set n s_lo ℓ,
       ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
        (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))|
      ≤ 2 * Delta_BB n c s_lo ℓ / (m : ℝ)^2 := by
  classical
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  set W := window_pair_set n s_lo ℓ with hW_def
  set b : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  set ε : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m - a i with hε_def
  -- Σ ε = 0 from h_sum_eq.
  have h_eps_sum : ∑ i, ε i = 0 := by
    show ∑ i, ((c i : ℝ) / m - a i) = 0
    rw [Finset.sum_sub_distrib]
    rw [← h_sum_eq]
    ring
  have h_eps_close : ∀ i, |ε i| ≤ 1 / (m : ℝ) := by
    intro i
    rw [hε_def]
    show |(c i : ℝ)/m - a i| ≤ 1 / (m : ℝ)
    rw [abs_sub_comm]
    exact h_close i
  -- Σ_W b_i ε_j = (1/m) · Σ_j ε_j BB^j by interchange.
  have h_first :
      ∑ p ∈ W, b p.1 * ε p.2 =
      (1 / (m : ℝ)) * ∑ j : Fin (2*n), ε j * BB_W n c s_lo ℓ j := by
    -- Σ_p b(p.1) ε(p.2) = Σ_j ε(j) Σ_{i:(i,j)∈W} b(i)
    --                  = Σ_j ε(j) · (1/m) · BB^j
    --                  = (1/m) · Σ_j ε(j) · BB^j.
    rw [hW_def]
    -- First, rewrite the filter sum as a double sum with indicator.
    have h_double :
        ∑ p ∈ window_pair_set n s_lo ℓ, b p.1 * ε p.2 =
        ∑ i : Fin (2*n), ∑ j : Fin (2*n),
          (if i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
           then b i * ε j else 0) := by
      unfold window_pair_set
      rw [Finset.sum_filter]
      rw [Fintype.sum_prod_type]
    rw [h_double]
    -- Swap summation order then collapse inner.
    rw [Finset.sum_comm]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    -- Inner: Σ_i (if (i,j)∈W then b_i ε_j else 0) = ε_j · (1/m) · BB^j.
    rw [show (1 : ℝ) / m * (ε j * BB_W n c s_lo ℓ j) =
            ε j * ((1 / m) * BB_W n c s_lo ℓ j) from by ring]
    unfold BB_W
    rw [Finset.mul_sum, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    by_cases hk : i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
    · rw [if_pos hk, if_pos hk]
      rw [hb_def]
      field_simp
    · rw [if_neg hk, if_neg hk]; ring
  -- Symmetric: Σ_W b_j ε_i.
  have h_swap :
      ∑ p ∈ W, b p.2 * ε p.1 = ∑ p ∈ W, b p.1 * ε p.2 := by
    apply Finset.sum_bij (fun (p : Fin (2*n) × Fin (2*n)) (_hp : p ∈ W) => (p.2, p.1))
    · intro p hp
      rw [hW_def] at hp ⊢
      unfold window_pair_set at hp ⊢
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      rw [show p.2.val + p.1.val = p.1.val + p.2.val from Nat.add_comm _ _]
      exact hp
    · intro p₁ _ p₂ _ heq
      simp only [Prod.mk.injEq] at heq
      apply Prod.ext heq.2 heq.1
    · intro p hp
      refine ⟨(p.2, p.1), ?_, rfl⟩
      rw [hW_def] at hp ⊢
      unfold window_pair_set at hp ⊢
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      rw [show p.2.val + p.1.val = p.1.val + p.2.val from Nat.add_comm _ _]
      exact hp
    · intros; rfl
  -- Combined LHS = 2 · (1/m) · Σ ε · BB^j.
  have h_combined :
      ∑ p ∈ W, (b p.1 * ε p.2 + b p.2 * ε p.1) =
      (2 / (m : ℝ)) * ∑ j : Fin (2*n), ε j * BB_W n c s_lo ℓ j := by
    rw [Finset.sum_add_distrib, h_first, h_swap, h_first]
    ring
  -- Bound the absolute value.
  have h_lhs_eq :
      (∑ p ∈ window_pair_set n s_lo ℓ,
       ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
        (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))) =
      ∑ p ∈ W, (b p.1 * ε p.2 + b p.2 * ε p.1) := by
    rw [hW_def, hb_def, hε_def]
  rw [h_lhs_eq]
  rw [h_combined]
  -- |2/m · X| = 2/m · |X|.
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 2 / m)]
  -- |Σ ε_j BB^j| ≤ Δ_BB / m by LP closed form (with h := 1/m).
  have h_lp_pos : ∑ j : Fin (2 * n), ε j * BB_W n c s_lo ℓ j ≤
      (1 / (m : ℝ)) * Delta_BB n c s_lo ℓ :=
    lp_closed_form_le n c s_lo ℓ ε (1 / (m : ℝ)) (by positivity) h_eps_close h_eps_sum
  -- For the lower bound: apply to -ε.
  have h_lp_neg : -(∑ j : Fin (2 * n), ε j * BB_W n c s_lo ℓ j) ≤
      (1 / (m : ℝ)) * Delta_BB n c s_lo ℓ := by
    have h_neg_close : ∀ j, |-ε j| ≤ 1 / (m : ℝ) := by
      intro j; rw [abs_neg]; exact h_eps_close j
    have h_neg_sum : ∑ j : Fin (2 * n), -ε j = 0 := by
      have : ∑ j : Fin (2 * n), -ε j = -(∑ j : Fin (2 * n), ε j) := by
        rw [← Finset.sum_neg_distrib]
      rw [this, h_eps_sum]
      ring
    have h_lp := lp_closed_form_le n c s_lo ℓ (fun j => -ε j) (1 / (m : ℝ))
              (by positivity) h_neg_close h_neg_sum
    -- Σ_j (-ε)_j BB^j = -Σ_j ε_j BB^j.
    have h_neg_eq : ∑ j : Fin (2 * n), (-ε j) * BB_W n c s_lo ℓ j =
                    -(∑ j : Fin (2 * n), ε j * BB_W n c s_lo ℓ j) := by
      have h_step : ∀ j : Fin (2 * n),
          (-ε j) * BB_W n c s_lo ℓ j = -(ε j * BB_W n c s_lo ℓ j) := by
        intro j; ring
      simp_rw [h_step]
      rw [← Finset.sum_neg_distrib]
    rw [h_neg_eq] at h_lp
    linarith
  have h_abs_bound :
      |∑ j : Fin (2 * n), ε j * BB_W n c s_lo ℓ j| ≤
      (1 / (m : ℝ)) * Delta_BB n c s_lo ℓ := by
    rw [abs_le]
    constructor <;> linarith [h_lp_pos, h_lp_neg]
  calc 2 / (m : ℝ) * |∑ j : Fin (2 * n), ε j * BB_W n c s_lo ℓ j|
      ≤ 2 / (m : ℝ) * ((1 / (m : ℝ)) * Delta_BB n c s_lo ℓ) := by
        apply mul_le_mul_of_nonneg_left h_abs_bound (by positivity)
    _ = 2 * Delta_BB n c s_lo ℓ / (m : ℝ)^2 := by
        have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
        field_simp

/-- **Linear bound (variant F, TV-normalized)**. -/
theorem tv_linear_bound_F
    (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
          (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))|
      ≤ 2 * Delta_BB n c s_lo ℓ / (4 * n * ℓ * (m : ℝ)^2) := by
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_pos : (0 : ℝ) < ℓ := by
    have : (2 : ℝ) ≤ ℓ := by exact_mod_cast hℓ
    linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 1 / (4 * n * ℓ))]
  have h_raw := linear_window_bound_F n m hm c a h_close h_sum_eq ℓ s_lo hℓ
  rw [div_mul_eq_mul_div, one_mul]
  rw [div_le_div_iff₀ h4nℓ_pos (by positivity : (0 : ℝ) < 4 * n * ℓ * (m : ℝ)^2)]
  calc |∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
          (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))|
       * (4 * ↑n * ↑ℓ * ↑m ^ 2)
      ≤ (2 * Delta_BB n c s_lo ℓ / (m : ℝ)^2) * (4 * ↑n * ↑ℓ * ↑m ^ 2) := by
        exact mul_le_mul_of_nonneg_right h_raw (by positivity)
    _ = (2 * Delta_BB n c s_lo ℓ) * (4 * ↑n * ↑ℓ) := by
        field_simp

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: corr_F_m2 and the F ≤ D dominance
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The variant-F correction in m² units.

    `corr_F_m2 = Δ_BB / (2n·ℓ) + ell_int_sum / (4n·ℓ)`.

    Compare to variant D (`corr_tight_m2`): D uses `(ℓ−1)·W_int` in the linear
    coefficient, while F uses `Δ_BB`.  Since `Δ_BB ≤ Σⱼ BB^j = Σᵢ cᵢ Nᵢ ≤
    (ℓ−1)·W_int`, we have `corr_F_m2 ≤ corr_tight_m2` (F is uniformly tighter). -/
noncomputable def corr_F_m2 (n : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  Delta_BB n c s_lo ℓ / (2 * n * ℓ) +
    (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ)

/-- The variant-F correction in TV space (divided by m²). -/
noncomputable def correction_F (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  corr_F_m2 n c ℓ s_lo / (m : ℝ)^2

/-- **F ≤ D dominance**:  `corr_F_m2 ≤ corr_tight_m2`.

    Proof: Δ_BB ≤ Σⱼ BB^j = Σᵢ cᵢ Nᵢ ≤ (ℓ-1) · W_int_overlap.  Combined with
    the matching ell_int_sum term, F's correction never exceeds D's. -/
theorem corr_F_m2_le_corr_tight_m2 (n : ℕ) (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hn : 0 < n) (hℓ : 2 ≤ ℓ) :
    corr_F_m2 n c ℓ s_lo ≤ corr_tight_m2 n c ℓ s_lo := by
  unfold corr_F_m2 corr_tight_m2
  have hn_real : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_real : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
  have hℓ_pos : (0 : ℝ) < (ℓ : ℝ) := by linarith
  have h2nℓ_pos : (0 : ℝ) < 2 * n * ℓ := by positivity
  -- Step 1: Δ_BB ≤ Σ BB^j = Σ c_i N_i.
  have h1 : Delta_BB n c s_lo ℓ ≤ ∑ j : Fin (2 * n), BB_W n c s_lo ℓ j :=
    Delta_BB_le_total n c s_lo ℓ
  rw [sum_BB_W_eq_sum_c_N] at h1
  -- Step 2: Σ c_i N_i ≤ (ℓ-1) · W_int_overlap (from variant D's helper).
  have h2 : (∑ i : Fin (2 * n), (c i : ℝ) * (N_row n s_lo ℓ i : ℝ)) ≤
        ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) :=
    sum_c_n_le_ell_W_overlap n c s_lo ℓ hℓ
  -- Combine: Δ_BB ≤ (ℓ-1) · W_int_overlap.
  have h_delta_le : Delta_BB n c s_lo ℓ ≤
      ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) := le_trans h1 h2
  -- Divide by 2n·ℓ.
  have h_div : Delta_BB n c s_lo ℓ / (2 * n * ℓ) ≤
      ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) / (2 * n * ℓ) :=
    div_le_div_of_nonneg_right h_delta_le h2nℓ_pos.le
  linarith

/-- **F ≤ D dominance (TV-normalized)**: `correction_F ≤ tight_correction`. -/
theorem correction_F_le_tight_correction
    (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ)
    (hn : 0 < n) (hm : 0 < m) (hℓ : 2 ≤ ℓ) :
    correction_F n m c ℓ s_lo ≤ tight_correction n m c ℓ s_lo := by
  unfold correction_F tight_correction
  have h_le := corr_F_m2_le_corr_tight_m2 n c ℓ s_lo hn hℓ
  have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ) ^ 2 := by positivity
  exact div_le_div_of_nonneg_right h_le hm_sq_pos.le

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: tight_discretization_bound_F (main F bound)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Main variant-F discretization bound** (discrete-discrete).

    For real `a` with `0 ≤ aᵢ`, `|aᵢ − cᵢ/m| ≤ 1/m`, AND `Σ aᵢ = Σ cᵢ/m`
    (cascade-context total-mass equality), the TV difference
    `|TV(c/m; W) − TV(a; W)|` is bounded by the variant-F correction
    `correction_F`.

    Like variant N, F requires the additional hypothesis `Σ a = Σ c/m`
    (because the LP closed-form uses `Σ ε = 0`).  Outside the cascade
    context (arbitrary `a` without Σ = 0), one falls back to variant D's
    elementwise linear bound.  In the cascade context (this theorem)
    F is uniformly tighter. -/
theorem tight_discretization_bound_F
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
      ≤ correction_F n m c ℓ s_lo := by
  classical
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ) ^ 2 := by positivity
  have hℓ_real : (1 : ℝ) ≤ (ℓ : ℝ) := by
    have : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
    linarith
  have hℓ_pos : (0 : ℝ) < (ℓ : ℝ) := by linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  -- Set up b = c/m. Decomposition (D-v2 b-route):
  --   b_ib_j − a_ia_j = b_iε_j + b_jε_i − ε_iε_j   (ε = b − a).
  set b : Fin (2 * n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  have h_close' : ∀ i, |a i - b i| ≤ 1 / (m : ℝ) := by
    intro i; rw [hb_def]; exact h_close i
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
  -- Linear bound: variant F via LP closed-form.
  have h_lin :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m - a p.2) +
            (c p.2 : ℝ) / m * ((c p.1 : ℝ) / m - a p.1))|
        ≤ 2 * Delta_BB n c s_lo ℓ / (4 * n * ℓ * (m : ℝ) ^ 2) :=
    tv_linear_bound_F n m hn hm c a h_close h_sum_eq ℓ s_lo hℓ
  -- δ²-bound: same as variant D.
  have h_delta :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2)|
        ≤ (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ * (m : ℝ) ^ 2) := by
    have := tv_delta_sq_bound n m hn hm a b h_close' ℓ s_lo hℓ
    convert this using 4
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
    _ ≤ 2 * Delta_BB n c s_lo ℓ / (4 * n * ℓ * (m : ℝ) ^ 2) +
        (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ * (m : ℝ) ^ 2) := by
        linarith [h_lin, h_delta]
    _ = (2 * Delta_BB n c s_lo ℓ +
            (ell_int_sum n s_lo ℓ : ℝ)) / (4 * n * ℓ * (m : ℝ) ^ 2) := by ring
    _ = correction_F n m c ℓ s_lo := by
        unfold correction_F corr_F_m2
        have h2nℓ_pos : (0 : ℝ) < 2 * n * ℓ := by positivity
        have h4nℓ_pos' : (0 : ℝ) < 4 * n * ℓ := by positivity
        field_simp
        ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 8: Cascade soundness (variant F)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Variant F cascade-prune soundness**.

    If `test_value n m c ℓ s_lo > c_target + correction_F n m c ℓ s_lo`,
    then for every real `a` with `0 ≤ aᵢ`, `|aᵢ − cᵢ/m| ≤ 1/m`, and
    `Σ aᵢ = Σ cᵢ/m` (cascade-context total-mass equality), we have
       `test_value_real n a ℓ s_lo > c_target`. -/
theorem tight_cascade_prune_sound_F
    (n m : ℕ) (c_target : ℝ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (h_exceeds : test_value n m c ℓ s_lo > c_target +
      correction_F n m c ℓ s_lo)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m) :
    test_value_real n a ℓ s_lo > c_target := by
  classical
  have h_tight :=
    tight_discretization_bound_F n m hn hm c a ha_nonneg h_close h_sum_eq ℓ s_lo hℓ
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

end -- noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- AUDIT BLOCK (variant F)
-- ═══════════════════════════════════════════════════════════════════════════════
/-
COEFFICIENT VARIANT:  F (LP-tight linear, b-route).
  D:  corr_tight = (ℓ−1)·W_int_overlap / (2n·ℓ) + ell_int_sum / (4n·ℓ)
  F:  corr_F     = Δ_BB / (2n·ℓ) + ell_int_sum / (4n·ℓ)
  N:  corr_N     = (ℓ−1)·W_int / (2n·ℓ) + min(op·2n, ell_int_sum) / (4n·ℓ)

  Each replaces a different term:
    F: tightens the linear coefficient (Δ_BB ≤ (ℓ-1)·W_int_overlap).
    N: tightens the δ² coefficient (spectral bound when Σ ε = 0).

NEW HYPOTHESIS REQUIRED FOR VARIANT F:
  Like variant N, F requires Σ aᵢ = Σ cᵢ/m (total-mass equality).  This is
  because the LP closed-form is over {δ : |δ_j| ≤ h, Σ δ = 0}; without Σ = 0,
  the LP relaxation is the same as variant D.

HYPOTHESES USED PER LEMMA:
  BB_W_nonneg                  No hypotheses.
  sum_BB_W_eq_sum_c_N          No hypotheses.
  BB_sigma_monotone            No hypotheses.
  bot_half_card                No hypotheses.
  top_half_card                No hypotheses.
  Delta_BB_le_total            No hypotheses.
  Delta_BB_nonneg              No hypotheses.
  lp_closed_form_le            0 ≤ h, |δⱼ| ≤ h, Σ δ = 0.
  linear_window_bound_F        m > 0, 2 ≤ ℓ, |aᵢ - cᵢ/m| ≤ 1/m, Σ a = Σ c/m.
  tv_linear_bound_F            n > 0, m > 0, 2 ≤ ℓ.
  corr_F_m2_le_corr_tight_m2   0 < n, 2 ≤ ℓ.
  correction_F_le_tight_correction  0 < n, 0 < m, 2 ≤ ℓ.
  tight_discretization_bound_F  0 < n, 0 < m, 2 ≤ ℓ;  0 ≤ aᵢ;
                                |aᵢ - cᵢ/m| ≤ 1/m;  Σ a = Σ c/m.
  tight_cascade_prune_sound_F   0 < n, 0 < m, 2 ≤ ℓ;
                                test_value(c/m) > c_target + correction_F;
                                0 ≤ aᵢ; |aᵢ - cᵢ/m| ≤ 1/m; Σ a = Σ c/m.

NEW AXIOMS DECLARED IN THIS FILE:  ZERO.
-/
