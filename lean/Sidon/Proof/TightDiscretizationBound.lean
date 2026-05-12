/-
Sidon Autocorrelation Project — Tight Discretization Bound (Layer 1, D-v2)

This file proves the **discrete-discrete** tight bound on the difference of
two test values when one composition is the integer-quanta discretization
and the other is any real-valued composition close to it.

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL BACKGROUND (refinement of Cloninger & Steinerberger, arXiv:1403.7988)
═══════════════════════════════════════════════════════════════════════════════

For real composition vectors a, b : Fin (2n) → ℝ with
  - a_i ≥ 0
  - b_i = c_i / m for some c : Fin (2n) → ℕ
  - ‖a − b‖_∞ ≤ 1/m

and a window W = {(i,j) ∈ Fin(2n)² : i+j ∈ [s_lo, s_lo+ℓ−2]}, we prove:

  |TV(b; W) − TV(a; W)| ≤ corr_tight / m²

where TV(x; W) = (1 / (4n·ℓ)) · Σ_{(i,j) ∈ W} x_i x_j and (D-v2)
  corr_tight = (ℓ−1)·W_int / (2n·ℓ) + ell_int_sum / (4n·ℓ),
  W_int      = Σ_{i ∈ overlap} c_i  (overlap = bins i with at least one j s.t. (i,j) ∈ W),
  ell_int_sum = #{(i,j) ∈ Fin(2n)² : i+j ∈ window},
  ell_int_arr[k] = #{(i,j) ∈ Fin(2n)² : i+j = k} = max(0, 2n − |k+1 − 2n|).

D-v2 DERIVATION (b-route, strictly tighter than the legacy D-v1 a-route):
  Set ε_i := b_i − a_i = c_i/m − a_i, so |ε_i| ≤ 1/m and b_i = c_i/m ≥ 0.
  Decomposition (key identity):
    b_i b_j − a_i a_j  =  b_i ε_j + b_j ε_i − ε_i ε_j.
  Triangle bounds, term-by-term:
    |Σ_W b_i ε_j|  ≤  Σ_W b_i · (1/m)  =  (1/m²) Σ_i c_i N_i
                                        ≤  (ℓ−1)·W_int / m².
    Same for |Σ_W b_j ε_i|.
    |Σ_W ε_i ε_j|  ≤  ell_int_sum / m².
  Total raw bound (linear part has NO `+ ell_int_sum/m²` slack):
    |Σ_W (b_i b_j − a_i a_j)|  ≤  2·(ℓ−1)·W_int / m²  +  ell_int_sum / m².
  Dividing by 4n·ℓ gives `corr_tight_m2 / m²` with the D-v2 coefficients
    (ℓ−1)·W_int / (2n·ℓ)  +  ell_int_sum / (4n·ℓ).
  The legacy D-v1 (with coefficient 3 on ell_int_sum / (4n·ℓ)) used the
  alternative decomposition  b_i b_j − a_i a_j = a_i ε_j + a_j ε_i + ε_i ε_j
  and bounded `a_i ≤ (c_i+1)/m`, picking up an extra `2·ell_int_sum/m²`
  of slack.  D-v2 is uniformly tighter (≤ D-v1 pointwise).

THIS FILE PROVES ONLY THE DISCRETE-DISCRETE BOUND (no continuous f, no measure
theory, no new axioms).

KNOT-ALIGNMENT IDENTITY: ell_int_arr matches the Python prune_D ell_int_arr:
  ell_int_arr[k] = max(0, 2n − |k+1 − 2n|).
For n=2 the array is [1,2,3,4,3,2,1] (verified by `decide` below).

PYTHON CROSS-REFERENCE:
  - run_cascade.py:51-181  (_prune_dynamic_int32; ell_prefix at 105-115)
  - run_cascade.py:188-295 (_prune_dynamic_int64; ell_prefix at 224-234)
  - _stage1_bench.py:298-381 (prune_D — D-v1 still wired in production, lines 366-373)

NOTE: Python `_stage1_bench.py:prune_D` and `run_cascade.py` currently use
D-v1 (factor 3 on ell_int_sum / (4n·ℓ)); the Lean theorem here proves the
strictly tighter D-v2 (factor 1).  Python will be upgraded to D-v2 in a
parallel patch.

INDEXING CONVENTIONS (verified against Defs.lean and run_cascade.py):
  - n  ↔ n_half          (Python parameter)
  - 2n = d              (number of bins)
  - S = 4·n·m           (total mass quanta)
  - bin width δ = 1/(4n)
  - height in bin i: a_i = c_i / m
  - conv index k ∈ [0, 4n−2]; pairs (i,j) ∈ Fin(2n)² with i+j = k
  - window in conv space: [s_lo, s_lo+ℓ−2], spans ℓ−1 integer positions
  - TV normalization: 1/(4n·ℓ)  (NOT 1/(4n·(ℓ−1)))

POINTWISE DOMINANCE (D-v2): UNIVERSAL.
  Under D-v2 the inequality `corr_tight_m2 ≤ 1 + W_int/(2n)` (i.e.
  `corr_tight_m2 ≤ corr_W`) holds for ALL (n,m,c,ℓ,s_lo) with ℓ ≥ 2.
  Sketch: the inequality reduces to `ell_int_sum − 2·W_int ≤ 4n·ℓ`,
  and ell_int_sum ≤ (ℓ−1)·2n ≤ 2n·ℓ ≤ 4n·ℓ (with W_int ≥ 0).
  See `corr_tight_m2_le_corr_w_refined` at the bottom of this file.

  The legacy D-v1 (factor 3) FAILED this dominance: at n=10, ℓ=4, s_lo=20,
  c=(40,0,…,0) we had corr_tight_v1 = 162/160 > 1 = corr_W.  This is now a
  historical artifact of the looser a-route bound; D-v2 closes that gap.

═══════════════════════════════════════════════════════════════════════════════
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.WRefinedDefs

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
-- Part 1: ell_int_arr — the count of (i,j) pairs in [0, 2n−1]² with i+j = k
-- ═══════════════════════════════════════════════════════════════════════════════

/-- ell_int_arr n k = #{(i,j) ∈ [0, 2n−1]² : i+j = k}.
    Closed form: max(0, 2n − |k+1 − 2n|).
    Matches Python prune_D `ell_int_arr` at _stage1_bench.py:316-325 and
    run_cascade.py:105-115. -/
def ell_int_arr (n k : ℕ) : ℕ :=
  if k + 1 ≤ 2 * n then k + 1
  else if k + 1 < 4 * n then 4 * n - 1 - k
  else 0

-- Sanity: n=2 gives [1, 2, 3, 4, 3, 2, 1] for k=0..6.
example : ell_int_arr 2 0 = 1 := by decide
example : ell_int_arr 2 1 = 2 := by decide
example : ell_int_arr 2 2 = 3 := by decide
example : ell_int_arr 2 3 = 4 := by decide
example : ell_int_arr 2 4 = 3 := by decide
example : ell_int_arr 2 5 = 2 := by decide
example : ell_int_arr 2 6 = 1 := by decide
example : ell_int_arr 2 7 = 0 := by decide
-- Out-of-range: ell_int_arr 2 100 = 0 (no pairs i+j = 100 with i,j ∈ [0,3]).
example : ell_int_arr 2 100 = 0 := by decide

/-- ell_int_arr n k matches the cardinality of pairs (i,j) in Fin (2n) × Fin (2n)
    with i + j = k. -/
theorem ell_int_arr_eq_card (n k : ℕ) :
    ell_int_arr n k =
    ((Finset.univ : Finset (Fin (2 * n) × Fin (2 * n))).filter
      (fun p => p.1.val + p.2.val = k)).card := by
  classical
  -- Strategy: build a bijection from the filter set to a `Finset.Ico` of the right size
  -- using Finset.card_image_of_injOn.
  rcases le_or_gt (k + 1) (2 * n) with h1 | h1
  · -- Case 1: k + 1 ≤ 2n, i.e., k ≤ 2n − 1. Valid i ∈ [0, k]. count = k + 1.
    rw [show ell_int_arr n k = k + 1 from by simp [ell_int_arr, h1]]
    have h2n : k < 2 * n := by omega
    -- The filter set is in bijection with Finset.range (k+1) via i ↦ (i, k-i).
    -- Show: Σ |filter| = (range (k+1)).card via card_eq_of_bij.
    have h_bij : ((Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
        (fun p => p.1.val + p.2.val = k)).card = (Finset.range (k+1)).card := by
      apply Finset.card_bij (fun (p : Fin (2*n) × Fin (2*n)) _ => p.1.val)
      · intro p hp
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
        have hi : p.1.val < 2 * n := p.1.2
        simp only [Finset.mem_range]
        omega
      · intro p₁ hp₁ p₂ hp₂ heq
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp₁ hp₂
        ext
        · exact heq
        · -- p_1.2.val determined by p_1.1.val + p_1.2.val = k.
          have h1' := hp₁
          have h2' := hp₂
          have : p₁.2.val = p₂.2.val := by
            have e1 : p₁.2.val = k - p₁.1.val := by omega
            have e2 : p₂.2.val = k - p₂.1.val := by omega
            rw [e1, e2, heq]
          exact this
      · intro i hi
        simp only [Finset.mem_range] at hi
        refine ⟨(⟨i, by omega⟩, ⟨k - i, by omega⟩), ?_, rfl⟩
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        omega
    rw [h_bij, Finset.card_range]
  · -- Case 2: k + 1 > 2n.  Sub-cases on k+1 < 4n or k+1 ≥ 4n.
    rcases lt_or_ge (k + 1) (4 * n) with h2 | h2
    · -- Case 2a: 2n < k + 1 < 4n.  k ∈ [2n, 4n−2].
      -- Valid i ∈ [k - (2n - 1), 2n - 1].
      rw [show ell_int_arr n k = 4 * n - 1 - k from by
        simp [ell_int_arr, not_le.mpr h1, h2]]
      have h_bij : ((Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
          (fun p => p.1.val + p.2.val = k)).card =
          (Finset.Ico (k - (2 * n - 1)) (2 * n)).card := by
        apply Finset.card_bij (fun (p : Fin (2*n) × Fin (2*n)) _ => p.1.val)
        · intro p hp
          simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
          have hi : p.1.val < 2 * n := p.1.2
          have hj : p.2.val < 2 * n := p.2.2
          simp only [Finset.mem_Ico]
          omega
        · intro p₁ hp₁ p₂ hp₂ heq
          simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp₁ hp₂
          ext
          · exact heq
          · have e1 : p₁.2.val = k - p₁.1.val := by omega
            have e2 : p₂.2.val = k - p₂.1.val := by omega
            rw [e1, e2, heq]
        · intro i hi
          simp only [Finset.mem_Ico] at hi
          refine ⟨(⟨i, hi.2⟩, ⟨k - i, by omega⟩), ?_, rfl⟩
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          omega
      rw [h_bij, Nat.card_Ico]
      omega
    · -- Case 2b: k + 1 ≥ 4n.  Empty filter set.
      rw [show ell_int_arr n k = 0 from by
        simp [ell_int_arr, not_le.mpr h1, not_lt.mpr h2]]
      symm
      rw [Finset.card_eq_zero]
      apply Finset.eq_empty_of_forall_notMem
      rintro ⟨⟨i, hi⟩, ⟨j, hj⟩⟩
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      omega

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: ell_int_sum — total pair count over a window
-- ═══════════════════════════════════════════════════════════════════════════════

/-- ell_int_sum n s_lo ℓ = Σ_{k ∈ [s_lo, s_lo+ℓ−2]} ell_int_arr n k.
    Equivalently, #{(i,j) ∈ Fin(2n)² : i+j ∈ window}. -/
def ell_int_sum (n s_lo ℓ : ℕ) : ℕ :=
  ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ell_int_arr n k

/-- ell_int_sum equals the cardinality of pairs in the window (as a Finset filter). -/
theorem ell_int_sum_eq_card (n s_lo ℓ : ℕ) :
    ell_int_sum n s_lo ℓ =
    ((Finset.univ : Finset (Fin (2 * n) × Fin (2 * n))).filter
      (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))).card := by
  classical
  unfold ell_int_sum
  -- Use biUnion: filter set = ⋃_k (filter by i+j = k).
  have h_eq : ((Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
      (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))) =
      (Finset.Icc s_lo (s_lo + ℓ - 2)).biUnion (fun k =>
        (Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
          (fun p => p.1.val + p.2.val = k)) := by
    ext ⟨⟨i, hi⟩, ⟨j, hj⟩⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_biUnion]
    constructor
    · intro h; exact ⟨i + j, h, rfl⟩
    · rintro ⟨k, hk, hijk⟩; rw [hijk]; exact hk
  rw [h_eq]
  -- Pairwise disjointness
  have h_disj : ((Finset.Icc s_lo (s_lo + ℓ - 2) : Finset ℕ) : Set ℕ).PairwiseDisjoint
      (fun k => (Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
        (fun p => p.1.val + p.2.val = k)) := by
    intro k _ k' _ hkk'
    rw [Function.onFun]
    rw [Finset.disjoint_left]
    intro p hp hp'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp hp'
    exact hkk' (hp.symm.trans hp')
  rw [Finset.card_biUnion h_disj]
  simp_rw [← ell_int_arr_eq_card]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: window pair set, row counts, and overlap mass
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The window pair set: pairs (i,j) ∈ Fin(2n)² with i+j ∈ window. -/
def window_pair_set (n s_lo ℓ : ℕ) : Finset (Fin (2*n) × Fin (2*n)) :=
  (Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
    (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))

/-- The window pair set is symmetric under (i,j) ↔ (j,i). -/
theorem pair_set_symmetric (n s_lo ℓ : ℕ) :
    ∀ p, p ∈ window_pair_set n s_lo ℓ ↔
           (p.2, p.1) ∈ window_pair_set n s_lo ℓ := by
  intro p
  unfold window_pair_set
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  rw [show p.2.val + p.1.val = p.1.val + p.2.val from Nat.add_comm _ _]

/-- The "row count" N_i for fixed i: number of j with (i,j) ∈ W. -/
def N_row (n s_lo ℓ : ℕ) (i : Fin (2*n)) : ℕ :=
  ((Finset.univ : Finset (Fin (2*n))).filter
    (fun j => i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))).card

/-- For fixed i, the column-set is a subset of an interval of length ℓ-1. -/
theorem n_i_le_ell_minus_1 (n s_lo ℓ : ℕ) (hℓ : 2 ≤ ℓ) (i : Fin (2*n)) :
    N_row n s_lo ℓ i ≤ ℓ - 1 := by
  classical
  unfold N_row
  -- For ℓ ≥ 2: the window has ℓ-1 distinct integer values, so N_row ≤ ℓ-1.
  have h_card_range : (Finset.range (ℓ - 1)).card = ℓ - 1 := Finset.card_range _
  rw [← h_card_range]
  apply Finset.card_le_card_of_injOn
    (fun j : Fin (2 * n) => i.val + j.val - s_lo)
  · intro j hj
    -- hj : j ∈ ↑(filter ... univ), unfold to extract membership condition
    have hj_mem : i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) := by
      have : j ∈ ((Finset.univ : Finset (Fin (2*n))).filter
          (fun j => i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))) := hj
      rw [Finset.mem_filter] at this
      exact this.2
    rw [Finset.mem_Icc] at hj_mem
    have h1 : s_lo ≤ i.val + j.val := hj_mem.1
    have h2 : i.val + j.val ≤ s_lo + ℓ - 2 := hj_mem.2
    have h2' : i.val + j.val ≤ s_lo + (ℓ - 2) := by omega
    have : (fun j : Fin (2*n) => i.val + j.val - s_lo) j ∈ Finset.range (ℓ - 1) := by
      rw [Finset.mem_range]
      simp only
      omega
    exact this
  · intro j hj j' hj' hjj'
    have hj_mem : i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) := by
      have : j ∈ ((Finset.univ : Finset (Fin (2*n))).filter
          (fun j => i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))) := hj
      rw [Finset.mem_filter] at this; exact this.2
    have hj'_mem : i.val + j'.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) := by
      have : j' ∈ ((Finset.univ : Finset (Fin (2*n))).filter
          (fun j => i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))) := hj'
      rw [Finset.mem_filter] at this; exact this.2
    rw [Finset.mem_Icc] at hj_mem hj'_mem
    have h1 : s_lo ≤ i.val + j.val := hj_mem.1
    have h2 : i.val + j.val ≤ s_lo + ℓ - 2 := hj_mem.2
    have h1' : s_lo ≤ i.val + j'.val := hj'_mem.1
    have h2' : i.val + j'.val ≤ s_lo + ℓ - 2 := hj'_mem.2
    simp only at hjj'
    apply Fin.ext
    omega

/-- The sum of N_i over all i equals ell_int_sum. -/
theorem sum_n_i_eq_ell_int_sum (n s_lo ℓ : ℕ) :
    ∑ i : Fin (2*n), N_row n s_lo ℓ i = ell_int_sum n s_lo ℓ := by
  classical
  rw [ell_int_sum_eq_card]
  unfold N_row
  -- Strategy: rewrite both sides as sums of indicator functions, then use Fubini.
  have h_lhs :
      ∑ i : Fin (2 * n),
        ((Finset.univ : Finset (Fin (2 * n))).filter
          (fun j => i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))).card =
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        (if i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then 1 else 0) := by
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.card_filter]
  have h_rhs :
      ((Finset.univ : Finset (Fin (2 * n) × Fin (2 * n))).filter
        (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))).card =
      ∑ p : Fin (2 * n) × Fin (2 * n),
        (if p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then 1 else 0) := by
    rw [Finset.card_filter]
  rw [h_lhs, h_rhs]
  -- Convert ∑_p f to ∑_i ∑_j by switching to product sum. Use Fintype.sum_prod_type.
  rw [Fintype.sum_prod_type]

/-- The overlap mass: Σ_i c_i over bins i with N_i ≥ 1.
    This is the natural quantity for the linear bound: N_i = 0 outside overlap,
    so c_i N_i = 0 outside overlap, and inside, N_i ≤ ℓ-1. -/
def W_int_overlap (n : ℕ) (c : Fin (2 * n) → ℕ) (s_lo ℓ : ℕ) : ℕ :=
  ∑ i : Fin (2 * n), if N_row n s_lo ℓ i ≥ 1 then c i else 0

/-- **Characterization of overlap range.**
    For `i : Fin (2n)` and ℓ ≥ 2, `N_row i ≥ 1` iff `i.val` lies in the same
    `[lo, hi]` overlap range used by `W_int_for_window`:
      lo = if s_lo + 1 ≤ 2n then 0 else s_lo - 2n + 1,
      hi = min (s_lo + ℓ - 2) (2n - 1).

    This is the key step in proving `W_int_overlap = W_int_for_window`. -/
theorem n_row_pos_iff (n : ℕ) (hn : 0 < n) (s_lo ℓ : ℕ) (hℓ : 2 ≤ ℓ)
    (i : Fin (2 * n)) :
    1 ≤ N_row n s_lo ℓ i ↔
      ((if s_lo + 1 ≤ 2 * n then 0 else s_lo - 2 * n + 1) ≤ i.val ∧
        i.val ≤ min (s_lo + ℓ - 2) (2 * n - 1)) := by
  classical
  have hd_pos : 0 < 2 * n := by omega
  have hi_lt : i.val < 2 * n := i.2
  unfold N_row
  rw [Nat.one_le_iff_ne_zero]
  rw [show (((Finset.univ : Finset (Fin (2 * n))).filter
              (fun j => i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))).card ≠ 0) ↔
          (((Finset.univ : Finset (Fin (2 * n))).filter
              (fun j => i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))).Nonempty) from by
    rw [Finset.card_ne_zero]]
  constructor
  · -- forward: ∃ j with i+j ∈ Icc s_lo (s_lo+ℓ-2)  →  lo ≤ i.val ≤ hi.
    rintro ⟨j, hj⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Icc] at hj
    have hj_lt : j.val < 2 * n := j.2
    refine ⟨?_, ?_⟩
    · -- lo ≤ i.val
      split_ifs with h
      · exact Nat.zero_le _
      · -- s_lo + 1 > 2n, so s_lo ≥ 2n. lo = s_lo - 2n + 1.
        -- From s_lo ≤ i+j and j < 2n: s_lo - 2n + 1 ≤ i.val.
        omega
    · -- i.val ≤ min (s_lo+ℓ-2) (2n-1)
      have h_le : i.val ≤ s_lo + ℓ - 2 := by
        have := hj.2
        have h_jnn : 0 ≤ j.val := Nat.zero_le _
        omega
      have h_le2 : i.val ≤ 2 * n - 1 := by omega
      exact Nat.le_min.mpr ⟨h_le, h_le2⟩
  · -- backward: lo ≤ i.val ≤ hi → ∃ j with i+j ∈ Icc s_lo (s_lo+ℓ-2).
    rintro ⟨h_lo, h_hi⟩
    have h_hi1 : i.val ≤ s_lo + ℓ - 2 := le_trans h_hi (Nat.min_le_left _ _)
    have h_hi2 : i.val ≤ 2 * n - 1 := le_trans h_hi (Nat.min_le_right _ _)
    -- choose j_val = if i.val ≤ s_lo then s_lo - i.val else 0
    by_cases h_cmp : i.val ≤ s_lo
    · -- Pick j = s_lo - i.val. Then i.val + j = s_lo ∈ Icc s_lo (s_lo+ℓ-2).
      -- Need j < 2n, i.e., s_lo - i.val < 2n.
      have hj_lt : s_lo - i.val < 2 * n := by
        split_ifs at h_lo with h
        · -- s_lo + 1 ≤ 2n, so s_lo ≤ 2n - 1, hence s_lo - i.val ≤ s_lo ≤ 2n - 1 < 2n.
          omega
        · -- s_lo ≥ 2n, lo = s_lo - 2n + 1, h_lo says i.val ≥ s_lo - 2n + 1.
          -- So s_lo - i.val ≤ s_lo - (s_lo - 2n + 1) = 2n - 1 < 2n.
          omega
      refine ⟨⟨s_lo - i.val, hj_lt⟩, ?_⟩
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Icc]
      refine ⟨?_, ?_⟩
      · -- s_lo ≤ i.val + (s_lo - i.val) = s_lo
        omega
      · -- i.val + (s_lo - i.val) = s_lo ≤ s_lo + ℓ - 2 since ℓ ≥ 2.
        omega
    · -- i.val > s_lo. Pick j = 0. Then i.val + 0 = i.val.
      push_neg at h_cmp
      have hj_lt : (0 : ℕ) < 2 * n := hd_pos
      refine ⟨⟨0, hj_lt⟩, ?_⟩
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Icc]
      refine ⟨?_, ?_⟩
      · -- s_lo ≤ i.val + 0 = i.val (since i.val > s_lo).
        omega
      · -- i.val + 0 = i.val ≤ s_lo + ℓ - 2 from h_hi1.
        omega

/-- **Bijection lemma:** `W_int_overlap` and `W_int_for_window` agree.

    Both sums equal `∑_{i : Fin(2n), lo ≤ i.val ≤ hi} c i`, where
      lo = max(0, s_lo - 2n + 1),  hi = min(s_lo + ℓ - 2, 2n - 1).

    `W_int_overlap` (this file) sums over `Fin (2n)` with the indicator
    `N_row ≥ 1`; `W_int_for_window` (WRefinedBound) sums over `Finset.Icc lo hi`
    with the dependent guard `i < d`.  Both index the same set of contributing
    bins.

    This is the missing piece needed to derive the W-refined axiom from the
    tight axiom (`cs_eq1_tight ⇒ cs_eq1_w_refined`). -/
theorem W_int_overlap_eq_for_window (n : ℕ) (hn : 0 < n) (c : Fin (2 * n) → ℕ)
    (s_lo ℓ : ℕ) (hℓ : 2 ≤ ℓ) :
    W_int_overlap n c s_lo ℓ = W_int_for_window n c ℓ s_lo := by
  classical
  have hd_pos : 0 < 2 * n := by omega
  -- Local abbreviations matching W_int_for_window.
  let lo : ℕ := if s_lo + 1 ≤ 2 * n then 0 else s_lo - 2 * n + 1
  let hi : ℕ := min (s_lo + ℓ - 2) (2 * n - 1)
  -- Step 1: Reduce W_int_overlap to a sum over Fin (2n) with the lo ≤ i.val ≤ hi guard.
  have h_overlap :
      W_int_overlap n c s_lo ℓ =
      ∑ i : Fin (2 * n), if (lo ≤ i.val ∧ i.val ≤ hi) then c i else 0 := by
    unfold W_int_overlap
    apply Finset.sum_congr rfl
    intro i _
    have h_iff : (1 ≤ N_row n s_lo ℓ i) ↔ (lo ≤ i.val ∧ i.val ≤ hi) :=
      n_row_pos_iff n hn s_lo ℓ hℓ i
    by_cases hN : 1 ≤ N_row n s_lo ℓ i
    · rw [if_pos hN, if_pos (h_iff.mp hN)]
    · rw [if_neg hN, if_neg (fun h => hN (h_iff.mpr h))]
  -- Step 2: branch on lo ≤ hi.
  rw [h_overlap]
  show (∑ i : Fin (2 * n), if (lo ≤ i.val ∧ i.val ≤ hi) then c i else 0) =
       W_int_for_window n c ℓ s_lo
  unfold W_int_for_window
  show (∑ i : Fin (2 * n), if (lo ≤ i.val ∧ i.val ≤ hi) then c i else 0) =
       (if lo ≤ hi then ∑ k ∈ Finset.Icc lo hi,
          (if h : k < 2 * n then c ⟨k, h⟩ else 0) else 0)
  by_cases h_lo_hi : lo ≤ hi
  · rw [if_pos h_lo_hi]
    have h_hi_lt : hi < 2 * n := by
      have : hi ≤ 2 * n - 1 := Nat.min_le_right _ _
      omega
    -- LHS: ∑ i : Fin (2n), if guard then c i else 0
    --    = ∑ i ∈ Finset.univ.filter guard, c i.
    rw [show (∑ i : Fin (2 * n), if (lo ≤ i.val ∧ i.val ≤ hi) then c i else 0) =
            ∑ i ∈ (Finset.univ : Finset (Fin (2 * n))).filter
              (fun i => lo ≤ i.val ∧ i.val ≤ hi), c i from by
      rw [Finset.sum_filter]]
    -- Use Finset.sum_bij from Icc lo hi to the filtered Fin-d set, via k ↦ ⟨k, _⟩.
    -- Direction goal: filtered_sum = Icc_sum, so `(Icc_sum = filtered_sum).symm`.
    symm
    refine Finset.sum_bij (fun (k : ℕ) (hk : k ∈ Finset.Icc lo hi) =>
        (⟨k, lt_of_le_of_lt (Finset.mem_Icc.mp hk).2 h_hi_lt⟩ : Fin (2 * n))) ?_ ?_ ?_ ?_
    · -- maps into the filtered set
      intro k hk
      rw [Finset.mem_Icc] at hk
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      exact hk
    · -- injectivity
      intro k₁ hk₁ k₂ hk₂ heq
      exact Fin.mk_eq_mk.mp heq
    · -- surjectivity
      intro i hi_mem
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi_mem
      refine ⟨i.val, ?_, ?_⟩
      · rw [Finset.mem_Icc]; exact hi_mem
      · rfl
    · -- value match
      intro k hk
      rw [Finset.mem_Icc] at hk
      have hk_lt : k < 2 * n := lt_of_le_of_lt hk.2 h_hi_lt
      rw [dif_pos hk_lt]
  · rw [if_neg h_lo_hi]
    -- If lo > hi, no i ∈ Fin (2n) satisfies (lo ≤ i.val ∧ i.val ≤ hi).
    apply Finset.sum_eq_zero
    intro i _
    rw [if_neg]
    rintro ⟨h1, h2⟩
    exact h_lo_hi (le_trans h1 h2)

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: corr_tight — the tight correction in m² units
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The TIGHT correction in m² units (D-v2), replacing the W-refined `1 + W_int/(2n)`.

    corr_tight = (ℓ−1)·W_int / (2n·ℓ) + ell_int_sum / (4n·ℓ)        (D-v2)

    Here W_int is `W_int_overlap`: the sum of c_i over bins i that overlap
    the window (i.e., have N_i ≥ 1).  This is the natural quantity for the
    linear bound (Σ c_i N_i ≤ (ℓ-1) · W_int_overlap).

    The actual correction (in TV space) is corr_tight / m².  See
    `tight_correction` below.

    Note: Python `_stage1_bench.py:prune_D` (lines 366-373) currently uses the
    legacy D-v1 form
      corr_w_v1 = (ell-1) * W_int / (2 * n_half * ell)
                + 3.0 * ell_int_sum / (4 * n_half * ell)
    which is uniformly looser than D-v2.  Python will be upgraded in a
    parallel patch. -/
noncomputable def corr_tight_m2 (n : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) / (2 * n * ℓ) +
    (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ)

/-- The tight correction in TV space (divided by m²). -/
noncomputable def tight_correction (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  corr_tight_m2 n c ℓ s_lo / (m : ℝ)^2

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: TV delta-squared bound
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Tight δ² bound** (raw, before dividing by 4n·ℓ):
      |Σ_{(i,j) ∈ W} δ_i δ_j| ≤ ell_int_sum / m²
    where δ_i = b_i − a_i with ‖δ‖_∞ ≤ 1/m.
    Uses only triangle inequality and the |δ_i δ_j| ≤ ‖δ‖_∞² bound. -/
theorem delta_sq_window_bound
    (n m : ℕ) (hm : m > 0)
    (a b : Fin (2 * n) → ℝ)
    (h_close : ∀ i, |a i - b i| ≤ 1 / (m : ℝ))
    (ℓ s_lo : ℕ) :
    |∑ p ∈ window_pair_set n s_lo ℓ, (b p.1 - a p.1) * (b p.2 - a p.2)|
      ≤ (ell_int_sum n s_lo ℓ : ℝ) / (m : ℝ)^2 := by
  classical
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  set W := window_pair_set n s_lo ℓ
  calc |∑ p ∈ W, (b p.1 - a p.1) * (b p.2 - a p.2)|
      ≤ ∑ p ∈ W, |(b p.1 - a p.1) * (b p.2 - a p.2)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ p ∈ W, |b p.1 - a p.1| * |b p.2 - a p.2| := by
        apply Finset.sum_congr rfl
        intro p _; rw [abs_mul]
    _ ≤ ∑ p ∈ W, (1 / (m : ℝ)) * (1 / (m : ℝ)) := by
        apply Finset.sum_le_sum; intro p _
        have h1 : |b p.1 - a p.1| ≤ 1 / (m : ℝ) := by
          rw [abs_sub_comm]; exact h_close p.1
        have h2 : |b p.2 - a p.2| ≤ 1 / (m : ℝ) := by
          rw [abs_sub_comm]; exact h_close p.2
        have habs1 : 0 ≤ |b p.1 - a p.1| := abs_nonneg _
        have habs2 : 0 ≤ |b p.2 - a p.2| := abs_nonneg _
        have h1m : 0 ≤ 1 / (m : ℝ) := by positivity
        nlinarith [habs1, habs2, h1, h2]
    _ = (W.card : ℝ) * ((1 / (m : ℝ)) * (1 / (m : ℝ))) := by
        rw [Finset.sum_const]; ring
    _ = (W.card : ℝ) / (m : ℝ)^2 := by field_simp
    _ = (ell_int_sum n s_lo ℓ : ℝ) / (m : ℝ)^2 := by
        show (W.card : ℝ) / (m : ℝ)^2 = (ell_int_sum n s_lo ℓ : ℝ) / (m : ℝ)^2
        congr 1
        show (W.card : ℝ) = (ell_int_sum n s_lo ℓ : ℝ)
        have := ell_int_sum_eq_card n s_lo ℓ
        show (W.card : ℝ) = ((ell_int_sum n s_lo ℓ : ℕ) : ℝ)
        rw [this]
        rfl

/-- **Tight δ² bound** (TV-normalized form). -/
theorem tv_delta_sq_bound
    (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (a b : Fin (2 * n) → ℝ)
    (h_close : ∀ i, |a i - b i| ≤ 1 / (m : ℝ))
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ, (b p.1 - a p.1) * (b p.2 - a p.2)|
      ≤ (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ * (m : ℝ)^2) := by
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_pos : (0 : ℝ) < ℓ := by
    have : (2 : ℝ) ≤ ℓ := by exact_mod_cast hℓ
    linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 1 / (4 * n * ℓ))]
  have h_raw := delta_sq_window_bound n m hm a b h_close ℓ s_lo
  rw [div_mul_eq_mul_div, one_mul]
  rw [div_le_div_iff₀ h4nℓ_pos (by positivity : (0 : ℝ) < 4 * n * ℓ * (m : ℝ)^2)]
  calc |∑ p ∈ window_pair_set n s_lo ℓ, (b p.1 - a p.1) * (b p.2 - a p.2)| * (4 * ↑n * ↑ℓ * ↑m ^ 2)
      ≤ ((ell_int_sum n s_lo ℓ : ℝ) / (m : ℝ)^2) * (4 * ↑n * ↑ℓ * ↑m ^ 2) := by
        exact mul_le_mul_of_nonneg_right h_raw (by positivity)
    _ = (ell_int_sum n s_lo ℓ : ℝ) * (4 * ↑n * ↑ℓ) := by
        field_simp

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: a_i ≤ (c_i+1)/m
-- ═══════════════════════════════════════════════════════════════════════════════

/-- From `0 ≤ a_i` and `|a_i - c_i/m| ≤ 1/m`, deduce `a_i ≤ (c_i + 1)/m`. -/
theorem a_le_c_plus_one_div_m (m : ℕ) (hm : m > 0)
    (a_i : ℝ) (c_i : ℕ)
    (h_close : |a_i - (c_i : ℝ) / m| ≤ 1 / (m : ℝ)) :
    a_i ≤ ((c_i : ℝ) + 1) / m := by
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have h := abs_le.mp h_close
  have h1 : a_i - (c_i : ℝ) / m ≤ 1 / m := h.2
  have : a_i ≤ (c_i : ℝ) / m + 1 / m := by linarith
  rw [show ((c_i : ℝ) + 1) / m = (c_i : ℝ) / m + 1 / m from by field_simp]
  exact this

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: Linear-term bound (a_i δ_j) — uses N_i ≤ ℓ-1 and Σ N_i = ell_int_sum
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Helper: Σ_i c_i N_i ≤ (ℓ-1) · W_int_overlap.
    The key is that N_i ≤ ℓ-1 for all i, and N_i = 0 (so c_i N_i = 0) outside overlap.
    Inside overlap (N_i ≥ 1), c_i N_i ≤ c_i (ℓ-1).  -/
theorem sum_c_n_le_ell_W_overlap (n : ℕ) (c : Fin (2 * n) → ℕ)
    (s_lo ℓ : ℕ) (hℓ : 2 ≤ ℓ) :
    (∑ i : Fin (2 * n), (c i : ℝ) * (N_row n s_lo ℓ i : ℝ)) ≤
      ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) := by
  classical
  have hℓ_real : (1 : ℝ) ≤ (ℓ : ℝ) := by
    have h2 : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
    linarith
  -- First: rewrite RHS to a sum form.
  have h_rhs : ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) =
      ∑ i : Fin (2 * n),
        ((ℓ : ℝ) - 1) * (if N_row n s_lo ℓ i ≥ 1 then (c i : ℝ) else 0) := by
    unfold W_int_overlap
    push_cast
    rw [Finset.mul_sum]
  rw [h_rhs]
  -- Now: ∑ c_i N_i ≤ ∑ (ℓ-1) · [N_i ≥ 1 ? c_i : 0]
  apply Finset.sum_le_sum
  intro i _
  by_cases hN : N_row n s_lo ℓ i ≥ 1
  · -- Overlap case.
    rw [if_pos hN]
    have hN_le : N_row n s_lo ℓ i ≤ ℓ - 1 := n_i_le_ell_minus_1 n s_lo ℓ hℓ i
    have hN_real : (N_row n s_lo ℓ i : ℝ) ≤ ((ℓ : ℝ) - 1) := by
      have hcast : ((N_row n s_lo ℓ i : ℕ) : ℝ) ≤ ((ℓ - 1 : ℕ) : ℝ) := by
        exact_mod_cast hN_le
      rw [show ((ℓ - 1 : ℕ) : ℝ) = (ℓ : ℝ) - 1 from by
        rw [Nat.cast_sub (by omega : 1 ≤ ℓ)]; simp] at hcast
      exact hcast
    have hc_nn : (0 : ℝ) ≤ (c i : ℝ) := Nat.cast_nonneg _
    calc (c i : ℝ) * (N_row n s_lo ℓ i : ℝ)
        ≤ (c i : ℝ) * ((ℓ : ℝ) - 1) := by
          apply mul_le_mul_of_nonneg_left hN_real hc_nn
      _ = ((ℓ : ℝ) - 1) * (c i : ℝ) := by ring
  · -- Outside overlap: N_i = 0, so c_i N_i = 0.
    have hN0 : N_row n s_lo ℓ i = 0 := by
      push_neg at hN; omega
    rw [if_neg hN, hN0]
    simp

/-- **Linear bound** (raw, before dividing by 4n·ℓ; D-v2 b-route):
      |Σ_W (b_i ε_j + b_j ε_i)| ≤ 2·(ℓ-1)·W_int_overlap / m²
    where b_i = c_i/m and ε_i = c_i/m - a_i = b_i - a_i.
    Hypotheses: |a_i - c_i/m| ≤ 1/m, m > 0.  (a_i ≥ 0 is NOT required.)
    The `ha_nonneg` parameter is preserved for API compatibility but unused. -/
theorem linear_window_bound
    (n m : ℕ) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (_ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |∑ p ∈ window_pair_set n s_lo ℓ,
       ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
        (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))|
      ≤ 2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) / (m : ℝ)^2 := by
  classical
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  set W := window_pair_set n s_lo ℓ with hW_def
  set b : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  set ε : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m - a i with hε_def
  have hb_nonneg : ∀ i, 0 ≤ b i := by
    intro i; rw [hb_def]; positivity
  have h_abs_eps : ∀ i, |ε i| ≤ 1 / (m : ℝ) := by
    intro i
    rw [hε_def]
    show |(c i : ℝ)/m - a i| ≤ 1 / (m : ℝ)
    rw [abs_sub_comm]
    exact h_close i
  -- Symmetry helper: Σ_p b(p.2) = Σ_p b(p.1)
  have h_swap_b : ∑ p ∈ W, b p.2 = ∑ p ∈ W, b p.1 := by
    apply Finset.sum_bij (fun (p : Fin (2*n) × Fin (2*n)) _ => (p.2, p.1))
    · intro p hp
      simp only [hW_def] at hp ⊢
      unfold window_pair_set at hp ⊢
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      rw [show p.2.val + p.1.val = p.1.val + p.2.val from Nat.add_comm _ _]
      exact hp
    · intro p₁ _ p₂ _ heq
      simp only [Prod.mk.injEq] at heq
      apply Prod.ext heq.2 heq.1
    · intro p hp
      refine ⟨(p.2, p.1), ?_, rfl⟩
      simp only [hW_def] at hp ⊢
      unfold window_pair_set at hp ⊢
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      rw [show p.2.val + p.1.val = p.1.val + p.2.val from Nat.add_comm _ _]
      exact hp
    · intros; rfl
  -- Σ_p b p.1 = Σ_i b_i · N_i
  have h_proj_b_to_row : ∑ p ∈ W, b p.1 = ∑ i : Fin (2*n), b i * (N_row n s_lo ℓ i : ℝ) := by
    show ∑ p ∈ window_pair_set n s_lo ℓ, b p.1 =
      ∑ i : Fin (2*n), b i * (N_row n s_lo ℓ i : ℝ)
    unfold window_pair_set
    rw [Finset.sum_filter]
    rw [Fintype.sum_prod_type]
    apply Finset.sum_congr rfl
    intro i _
    have h_inner : ∀ j : Fin (2*n),
        (if i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then b i else (0 : ℝ)) =
        b i * (if i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then (1:ℝ) else 0) := by
      intro j
      by_cases hk : i.val + j.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
      · rw [if_pos hk, if_pos hk]; ring
      · rw [if_neg hk, if_neg hk]; ring
    simp_rw [h_inner]
    rw [← Finset.mul_sum]
    congr 1
    unfold N_row
    rw [Finset.card_filter]
    push_cast
    rfl
  -- Now the main calc.  Goal is to bound
  --   |∑ (b p.1 * ε p.2 + b p.2 * ε p.1)| ≤ 2·(ℓ-1)·W_int_overlap / m².
  calc |∑ p ∈ W, (b p.1 * ε p.2 + b p.2 * ε p.1)|
      ≤ ∑ p ∈ W, |b p.1 * ε p.2 + b p.2 * ε p.1| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ p ∈ W, (|b p.1 * ε p.2| + |b p.2 * ε p.1|) := by
        apply Finset.sum_le_sum
        intro p _; exact abs_add_le _ _
    _ = ∑ p ∈ W, (b p.1 * |ε p.2| + b p.2 * |ε p.1|) := by
        apply Finset.sum_congr rfl
        intro p _
        rw [abs_mul, abs_mul, abs_of_nonneg (hb_nonneg _), abs_of_nonneg (hb_nonneg _)]
    _ ≤ ∑ p ∈ W, (b p.1 * (1 / (m : ℝ)) + b p.2 * (1 / (m : ℝ))) := by
        apply Finset.sum_le_sum
        intro p _
        have hb1 := hb_nonneg p.1
        have hb2 := hb_nonneg p.2
        have hd1 := h_abs_eps p.1
        have hd2 := h_abs_eps p.2
        have habs1 : 0 ≤ |ε p.1| := abs_nonneg _
        have habs2 : 0 ≤ |ε p.2| := abs_nonneg _
        nlinarith [habs1, habs2, hd1, hd2, hb1, hb2]
    _ = (∑ p ∈ W, b p.1 + ∑ p ∈ W, b p.2) * (1 / (m : ℝ)) := by
        rw [add_mul]
        rw [Finset.sum_mul, Finset.sum_mul, ← Finset.sum_add_distrib]
    _ = 2 * (∑ p ∈ W, b p.1) * (1 / (m : ℝ)) := by
        rw [h_swap_b]; ring
    _ = 2 / (m : ℝ) * (∑ i : Fin (2*n), b i * (N_row n s_lo ℓ i : ℝ)) := by
        rw [h_proj_b_to_row]; ring
    _ = 2 / (m : ℝ)^2 * (∑ i : Fin (2*n), (c i : ℝ) * (N_row n s_lo ℓ i : ℝ)) := by
        -- b_i = c_i/m, so 2/m · ∑ b_i N_i = 2/m² · ∑ c_i N_i.
        have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
        rw [Finset.mul_sum, Finset.mul_sum]
        refine Finset.sum_congr rfl ?_
        intro i _
        rw [hb_def]
        field_simp
    _ ≤ 2 / (m : ℝ)^2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) := by
        have h_sum_cN := sum_c_n_le_ell_W_overlap n c s_lo ℓ hℓ
        have h2m_pos : (0 : ℝ) ≤ 2 / (m : ℝ)^2 := by positivity
        nlinarith [h_sum_cN]
    _ = 2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) / (m : ℝ)^2 := by
        field_simp

/-- **Linear bound** (TV-normalized form, D-v2 b-route). -/
theorem tv_linear_bound
    (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
          (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))|
      ≤ 2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) /
          (4 * n * ℓ * (m : ℝ)^2) := by
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_pos : (0 : ℝ) < ℓ := by
    have : (2 : ℝ) ≤ ℓ := by exact_mod_cast hℓ
    linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 1 / (4 * n * ℓ))]
  have h_raw := linear_window_bound n m hm c a ha_nonneg h_close ℓ s_lo hℓ
  rw [div_mul_eq_mul_div, one_mul]
  rw [div_le_div_iff₀ h4nℓ_pos (by positivity : (0 : ℝ) < 4 * n * ℓ * (m : ℝ)^2)]
  calc |∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
          (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))|
       * (4 * ↑n * ↑ℓ * ↑m ^ 2)
      ≤ (2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) / (m : ℝ)^2) *
          (4 * ↑n * ↑ℓ * ↑m ^ 2) := by
        exact mul_le_mul_of_nonneg_right h_raw (by positivity)
    _ = (2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ))) * (4 * ↑n * ↑ℓ) := by
        field_simp

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 8: Combined bound — tight_discretization_bound
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **The main tight discretization bound** (discrete-discrete, D-v2).
    For real a with a_i ≥ 0 and |a_i - c_i/m| ≤ 1/m, the TV difference
    |TV(c/m; W) - TV(a; W)| ≤ corr_tight_m2 / m² = tight_correction.

    D-v2 derivation: with b_i = c_i/m and ε_i = b_i - a_i, the identity
      b_i b_j - a_i a_j = b_i ε_j + b_j ε_i - ε_i ε_j
    plus the linear bound |Σ_W (b_iε_j + b_jε_i)| ≤ 2(ℓ-1)·W_int/m² (NO
    `+ ell_int_sum/m²` slack) and the δ²-bound |Σ_W ε_iε_j| ≤ ell_int_sum/m²
    combine to (2(ℓ-1)·W_int + ell_int_sum)/m², which after dividing by 4n·ℓ
    is exactly corr_tight_m2 / m². -/
theorem tight_discretization_bound
    (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2)|
      ≤ tight_correction n m c ℓ s_lo := by
  classical
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  have hℓ_real : (1 : ℝ) ≤ (ℓ : ℝ) := by
    have : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
    linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by
    have hℓ_pos : (0 : ℝ) < ℓ := by linarith
    positivity
  -- Set up b = c/m.  D-v2 decomposition:
  --   b_ib_j − a_ia_j = b_iε_j + b_jε_i − ε_iε_j   (ε := b − a).
  set b : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  have h_close' : ∀ i, |a i - b i| ≤ 1 / (m : ℝ) := by
    intro i; rw [hb_def]; exact h_close i
  -- The new (b-route) decomposition.
  have h_decomp : ∀ p : Fin (2*n) × Fin (2*n),
      (c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2 =
      ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
       (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1)) -
        ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2) := by
    intro p; ring
  -- Rewrite the LHS using the decomposition.
  have h_sum_decomp :
    (∑ p ∈ window_pair_set n s_lo ℓ,
        ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2)) =
    (∑ p ∈ window_pair_set n s_lo ℓ,
        ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
         (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))) -
    (∑ p ∈ window_pair_set n s_lo ℓ,
        ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro p _; rw [h_decomp]
  rw [h_sum_decomp, mul_sub]
  -- Now apply triangle inequality and the two component bounds.
  have h_lin : |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
          (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))|
      ≤ 2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) /
          (4 * n * ℓ * (m : ℝ)^2) :=
    tv_linear_bound n m hn hm c a ha_nonneg h_close ℓ s_lo hℓ
  -- For ε² (= δ² with δ_i = c_i/m - a_i), reuse `tv_delta_sq_bound`.
  have h_delta : |(1 / ((4 * n * ℓ : ℝ))) *
       ∑ p ∈ window_pair_set n s_lo ℓ,
         ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)|
      ≤ (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ * (m : ℝ)^2) := by
    have := tv_delta_sq_bound n m hn hm a b h_close' ℓ s_lo hℓ
    convert this using 4
  -- Combine via triangle inequality `|x - y| ≤ |x| + |y|`.
  calc |(1 / ((4 * n * ℓ : ℝ))) *
        ∑ p ∈ window_pair_set n s_lo ℓ,
          ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
           (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1)) -
        (1 / ((4 * n * ℓ : ℝ))) *
        ∑ p ∈ window_pair_set n s_lo ℓ,
          ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)|
      ≤ |(1 / ((4 * n * ℓ : ℝ))) *
          ∑ p ∈ window_pair_set n s_lo ℓ,
            ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
             (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))| +
        |(1 / ((4 * n * ℓ : ℝ))) *
          ∑ p ∈ window_pair_set n s_lo ℓ,
            ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)| := abs_sub _ _
    _ ≤ 2 * (((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ)) /
            (4 * n * ℓ * (m : ℝ)^2) +
        (ell_int_sum n s_lo ℓ : ℝ) / (4 * n * ℓ * (m : ℝ)^2) := by
        linarith [h_lin, h_delta]
    _ = (2 * ((ℓ : ℝ) - 1) * (W_int_overlap n c s_lo ℓ : ℝ) +
            (ell_int_sum n s_lo ℓ : ℝ)) / (4 * n * ℓ * (m : ℝ)^2) := by
        ring
    _ = tight_correction n m c ℓ s_lo := by
        unfold tight_correction corr_tight_m2
        have hℓ_pos : (0 : ℝ) < ℓ := by linarith
        have h2nℓ_pos : (0 : ℝ) < 2 * n * ℓ := by positivity
        have h4nℓ_pos' : (0 : ℝ) < 4 * n * ℓ := by positivity
        field_simp
        ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 9: Cascade soundness — discrete-discrete only
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The "real test value": same as test_value but with real-valued composition `a`. -/
noncomputable def test_value_real (n : ℕ) (a : Fin (2 * n) → ℝ) (ℓ s_lo : ℕ) : ℝ :=
  (1 / (4 * n * ℓ : ℝ)) *
    ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then a i * a j else 0

/-- Equivalent form of test_value_real over the window pair set. -/
theorem test_value_real_eq_window_sum
    (n : ℕ) (a : Fin (2 * n) → ℝ) (ℓ s_lo : ℕ) :
    test_value_real n a ℓ s_lo =
    (1 / (4 * n * ℓ : ℝ)) * ∑ p ∈ window_pair_set n s_lo ℓ, a p.1 * a p.2 := by
  classical
  unfold test_value_real
  congr 1
  -- Need: triple sum = sum over filtered pairs.
  rw [show ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
       ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
         (if i.1 + j.1 = k then a i * a j else 0) =
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
       ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
         (if i.1 + j.1 = k then a i * a j else 0) from ?_]
  · -- Now collapse the k-sum.
    rw [show ∑ i : Fin (2*n), ∑ j : Fin (2*n),
          ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
            (if i.1 + j.1 = k then a i * a j else 0) =
          ∑ i : Fin (2*n), ∑ j : Fin (2*n),
            (if i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then a i * a j else 0) from ?_]
    · -- Now switch to filter sum.
      unfold window_pair_set
      rw [show ((Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
            (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))) =
            ((Finset.univ : Finset (Fin (2*n))) ×ˢ (Finset.univ : Finset (Fin (2*n)))).filter
              (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)) from by
              ext; simp]
      rw [Finset.sum_filter, ← Finset.sum_product']
    · apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      by_cases hk : i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
      · rw [if_pos hk]
        rw [Finset.sum_eq_single (i.1 + j.1)]
        · simp
        · intro k _ hk'; rw [if_neg]; exact fun h => hk' h.symm
        · intro hk'; exact absurd hk hk'
      · rw [if_neg hk]
        apply Finset.sum_eq_zero
        intro k hk_in
        rw [if_neg]
        intro hijk; rw [hijk] at hk; exact hk hk_in
  · rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.sum_comm]

/-- Equivalent form of test_value over the window pair set (using real heights c/m). -/
theorem test_value_eq_window_sum
    (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) :
    test_value n m c ℓ s_lo =
    (1 / (4 * n * ℓ : ℝ)) * ∑ p ∈ window_pair_set n s_lo ℓ,
      (c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) := by
  classical
  unfold test_value
  simp only []
  unfold discrete_autoconvolution
  congr 1
  rw [show ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
       ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
         (if i.1 + j.1 = k then (c i : ℝ)/m * ((c j : ℝ)/m) else 0) =
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
       ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
         (if i.1 + j.1 = k then (c i : ℝ)/m * ((c j : ℝ)/m) else 0) from ?_]
  · rw [show ∑ i : Fin (2*n), ∑ j : Fin (2*n),
          ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
            (if i.1 + j.1 = k then (c i : ℝ)/m * ((c j : ℝ)/m) else 0) =
          ∑ i : Fin (2*n), ∑ j : Fin (2*n),
            (if i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then
              (c i : ℝ)/m * ((c j : ℝ)/m) else 0) from ?_]
    · unfold window_pair_set
      rw [show ((Finset.univ : Finset (Fin (2*n) × Fin (2*n))).filter
            (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2))) =
            ((Finset.univ : Finset (Fin (2*n))) ×ˢ (Finset.univ : Finset (Fin (2*n)))).filter
              (fun p => p.1.val + p.2.val ∈ Finset.Icc s_lo (s_lo + ℓ - 2)) from by
              ext; simp]
      rw [Finset.sum_filter, ← Finset.sum_product']
    · apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      by_cases hk : i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
      · rw [if_pos hk]
        rw [Finset.sum_eq_single (i.1 + j.1)]
        · simp
        · intro k _ hk'; rw [if_neg]; exact fun h => hk' h.symm
        · intro hk'; exact absurd hk hk'
      · rw [if_neg hk]
        apply Finset.sum_eq_zero
        intro k hk_in
        rw [if_neg]
        intro hijk; rw [hijk] at hk; exact hk hk_in
  · rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.sum_comm]

/-- **Tight cascade-prune soundness (discrete-discrete)**.
    If `test_value n m c ℓ s_lo > c_target + tight_correction n m c ℓ s_lo`,
    then for every real-valued `a` with `0 ≤ a_i` and `|a_i - c_i/m| ≤ 1/m`,
    we have `test_value_real n a ℓ s_lo > c_target`.

    This is the discrete-discrete pruning soundness (no continuous f). -/
theorem tight_cascade_prune_sound
    (n m : ℕ) (c_target : ℝ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (h_exceeds : test_value n m c ℓ s_lo > c_target +
      tight_correction n m c ℓ s_lo)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ)) :
    test_value_real n a ℓ s_lo > c_target := by
  classical
  -- We have: test_value(c/m) > c_target + tight_correction.
  -- And: |test_value(c/m) - test_value_real(a)| ≤ tight_correction.
  -- So test_value_real(a) ≥ test_value(c/m) - tight_correction > c_target.
  have h_tight := tight_discretization_bound n m hn hm c a ha_nonneg h_close ℓ s_lo hℓ
  -- Rewrite the bound's LHS into test_value(c/m) - test_value_real(a) form.
  rw [test_value_eq_window_sum] at h_exceeds
  rw [test_value_real_eq_window_sum]
  have h_bound_eq :
      |(1 / ((4 * n * ℓ : ℝ))) *
         ∑ p ∈ window_pair_set n s_lo ℓ,
           ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2)| =
      |((1 / ((4 * n * ℓ : ℝ))) * ∑ p ∈ window_pair_set n s_lo ℓ,
            (c p.1 : ℝ) / m * ((c p.2 : ℝ) / m)) -
        ((1 / ((4 * n * ℓ : ℝ))) * ∑ p ∈ window_pair_set n s_lo ℓ, a p.1 * a p.2)| := by
    congr 1
    rw [← mul_sub, ← Finset.sum_sub_distrib]
  rw [h_bound_eq] at h_tight
  have h_abs_le := abs_sub_le_iff.mp h_tight
  linarith [h_abs_le.1, h_abs_le.2]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 10: Pointwise dominance over corr_W (D-v2: UNIVERSAL)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Sanity: for n=10, ℓ=4, s_lo=20, ell_int_sum = 54.
    (At n=10, 2n=20.  Formula gives ell_int_arr 10 20 = 19, ell_int_arr 10 21 = 18,
    ell_int_arr 10 22 = 17.  Sum = 54.) -/
example : ell_int_sum 10 20 4 = 54 := by decide

/-- HISTORICAL D-v1 dominance counterexample (no longer applies under D-v2):
    With the legacy factor-3 correction `corr_tight_v1`, at the parameters
    n=10, ℓ=4, s_lo=20, c=(40,0,…,0) we had W_int_overlap=0, ell_int_sum=54,
    so corr_tight_v1 = 0 + 3·54/(4·10·4) = 162/160 = 81/80 > 1 = corr_W.
    Under D-v2 the same parameters give corr_tight_m2 = 0 + 54/160 = 27/80 < 1,
    matching the universal D-v2 dominance proven below. -/
example :
    corr_tight_m2 10 (fun i => if i.val = 0 then 40 else 0) 4 20 = 27 / 80 := by
  unfold corr_tight_m2
  have h_ell : ell_int_sum 10 20 4 = 54 := by decide
  have h_W : W_int_overlap 10 (fun i => if i.val = 0 then 40 else 0) 20 4 = 0 := by
    unfold W_int_overlap
    apply Finset.sum_eq_zero
    intro i _
    by_cases hi : i.val = 0
    · -- For i = 0, N_row = 0 (no j has 0 + j ∈ [20, 22]).
      have hN : N_row 10 20 4 i = 0 := by
        unfold N_row
        rw [Finset.card_eq_zero]
        apply Finset.eq_empty_of_forall_notMem
        intro j hj
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Icc] at hj
        rw [hi] at hj
        have hj_bound : j.val < 20 := j.2
        omega
      simp [hN]
    · -- For i ≠ 0, c i = 0.
      simp [hi]
  rw [h_W, h_ell]
  push_cast
  norm_num

/-- ell_int_sum ≤ (2n)·(ℓ-1).  Each of the 2n rows contributes N_i ≤ ℓ-1. -/
theorem ell_int_sum_le (n s_lo ℓ : ℕ) (hℓ : 2 ≤ ℓ) :
    ell_int_sum n s_lo ℓ ≤ 2 * n * (ℓ - 1) := by
  classical
  rw [← sum_n_i_eq_ell_int_sum n s_lo ℓ]
  calc ∑ i : Fin (2 * n), N_row n s_lo ℓ i
      ≤ ∑ _i : Fin (2 * n), (ℓ - 1) :=
        Finset.sum_le_sum (fun i _ => n_i_le_ell_minus_1 n s_lo ℓ hℓ i)
    _ = (Finset.univ : Finset (Fin (2*n))).card * (ℓ - 1) := by
        rw [Finset.sum_const, smul_eq_mul]
    _ = 2 * n * (ℓ - 1) := by
        rw [Finset.card_univ, Fintype.card_fin]

/-- **D-v2 universal pointwise dominance**: for any (n,ℓ,s_lo,c) with n>0 and ℓ≥2,
      corr_tight_m2 ≤ 1 + W_int_overlap/(2n).
    The RHS is the D-v0 W-refined correction in m² units.  Under D-v1 this
    inequality FAILED (see the historical counterexample above); D-v2 closes it.

    Proof sketch:  the inequality reduces (after multiplying by 4n·ℓ) to
      2(ℓ-1)·W + ell_int_sum ≤ 4n·ℓ + 2ℓ·W,
    i.e. `ell_int_sum ≤ 4n·ℓ + 2W`.  This holds because ell_int_sum ≤ 2n(ℓ-1)
    and 2n(ℓ-1) ≤ 4n·ℓ + 2W (with W ≥ 0). -/
theorem corr_tight_m2_le_corr_w_refined
    (n : ℕ) (hn : 0 < n) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    corr_tight_m2 n c ℓ s_lo ≤ 1 + (W_int_overlap n c s_lo ℓ : ℝ) / (2 * n) := by
  unfold corr_tight_m2
  have hn_real : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_real : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
  have hℓ_pos : (0 : ℝ) < (ℓ : ℝ) := by linarith
  have h2nℓ_pos : (0 : ℝ) < 2 * (n : ℝ) * ℓ := by positivity
  have h4nℓ_pos : (0 : ℝ) < 4 * (n : ℝ) * ℓ := by positivity
  have h_ell_le : (ell_int_sum n s_lo ℓ : ℝ) ≤ 2 * (n : ℝ) * ((ℓ : ℝ) - 1) := by
    have hnat := ell_int_sum_le n s_lo ℓ hℓ
    have hcast1 : ((ell_int_sum n s_lo ℓ : ℕ) : ℝ) ≤ ((2 * n * (ℓ - 1) : ℕ) : ℝ) := by
      exact_mod_cast hnat
    have h_sub : ((ℓ - 1 : ℕ) : ℝ) = (ℓ : ℝ) - 1 := by
      rw [Nat.cast_sub (by omega : 1 ≤ ℓ)]; simp
    have h_cast : ((2 * n * (ℓ - 1) : ℕ) : ℝ) = 2 * (n : ℝ) * ((ℓ : ℝ) - 1) := by
      push_cast
      rw [h_sub]
    rw [h_cast] at hcast1
    exact hcast1
  have h_W_nn : (0 : ℝ) ≤ (W_int_overlap n c s_lo ℓ : ℝ) := Nat.cast_nonneg _
  -- Multiply through by 4nℓ to clear denominators.
  rw [div_add_div _ _ (by positivity : (2 * (n : ℝ) * ℓ) ≠ 0)
        (by positivity : (4 * (n : ℝ) * ℓ) ≠ 0)]
  rw [show (1 : ℝ) + (W_int_overlap n c s_lo ℓ : ℝ) / (2 * n) =
          (2 * (n : ℝ) + (W_int_overlap n c s_lo ℓ : ℝ)) / (2 * n) from by
        field_simp]
  rw [div_le_div_iff₀ (by positivity : (0 : ℝ) < 2 * (n : ℝ) * ℓ * (4 * (n : ℝ) * ℓ))
                      (by positivity : (0 : ℝ) < 2 * (n : ℝ))]
  -- Goal: numerator * 2n ≤ (2n + W) * (2nℓ · 4nℓ).
  -- Simplify both sides via nlinarith.
  nlinarith [h_ell_le, h_W_nn, hn_real, hℓ_pos,
             sq_nonneg ((n : ℝ) * ℓ), mul_pos hn_real hℓ_pos,
             mul_nonneg (by linarith : (0:ℝ) ≤ (ℓ:ℝ) - 1) h_W_nn]

end -- noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- AUDIT BLOCK (final, D-v2)
-- ═══════════════════════════════════════════════════════════════════════════════
/-
COEFFICIENT VARIANT:  D-v2 (b-route, strictly tighter than legacy D-v1).
  D-v2:  corr_tight = (ℓ−1)·W_int / (2n·ℓ) +     ell_int_sum / (4n·ℓ)
  D-v1:  corr_tight = (ℓ−1)·W_int / (2n·ℓ) + 3·  ell_int_sum / (4n·ℓ)   (legacy)

HYPOTHESES USED PER LEMMA:

  ell_int_arr_eq_card        (n k : ℕ).  No hypotheses.
  ell_int_sum_eq_card        (n s_lo ℓ : ℕ).  No hypotheses.
  pair_set_symmetric         (n s_lo ℓ : ℕ).  No hypotheses.
  n_i_le_ell_minus_1         (hℓ : 1 ≤ ℓ).  Strict positivity of ℓ would make `ℓ - 1`
                                            well-behaved, but ℓ = 0 also works trivially.
  sum_n_i_eq_ell_int_sum     No hypotheses (relies on Finset.card identities).
  a_le_c_plus_one_div_m      m > 0  (kept for API; unused in D-v2 b-route).
  sum_c_n_le_ell_W_overlap   1 ≤ ℓ  (so ℓ - 1 well-defined as a Nat).
  delta_sq_window_bound      m > 0; |a_i - b_i| ≤ 1/m.
  tv_delta_sq_bound          n > 0, m > 0, 2 ≤ ℓ.
  linear_window_bound (D-v2) m > 0, 2 ≤ ℓ; |a_i - c_i/m| ≤ 1/m.
                             a_i ≥ 0 NOT required by the b-route argument
                             (parameter `_ha_nonneg` preserved for API stability).
  tv_linear_bound  (D-v2)    n > 0, m > 0, 2 ≤ ℓ; a_i ≥ 0 (passed through);
                             |a_i - c_i/m| ≤ 1/m.
  tight_discretization_bound n > 0, m > 0, 2 ≤ ℓ; a_i ≥ 0; |a_i - c_i/m| ≤ 1/m.
  tight_cascade_prune_sound  n > 0, m > 0, 2 ≤ ℓ;
                              test_value(c/m) > c_target + tight_correction;
                              a_i ≥ 0; |a_i - c_i/m| ≤ 1/m.
  ell_int_sum_le             1 ≤ ℓ.  Counts per-row N_i ≤ ℓ-1.
  corr_tight_m2_le_corr_w_refined  0 < n, 2 ≤ ℓ.  D-v2 universal dominance.

SIMPLEX CONSTRAINT (Σ c_i = 4nm):  NOT REQUIRED.  All bounds in this file are
LOCAL (per-window) and do not use the global mass constraint.  This is consistent
with the W-refined proof structure where the simplex constraint enters only at
the final `cascade_all_pruned_w` axiom level.

OFF-BY-ONE FOR ℓ:  Window length is ℓ (number of bins), conv positions span
[s_lo, s_lo + ℓ - 2] which has ℓ - 1 integer values.  The TV normalization is
1/(4n·ℓ) (not 1/(4n·(ℓ-1))) — this matches the Python convention.
N_i ≤ ℓ - 1, since each i is in at most ℓ - 1 distinct positions of the conv index.

SIGN CONVENTION (D-v2):  We define ε_i = b_i - a_i = c_i/m - a_i (so b = c/m).
  Decomposition: b_ib_j − a_ia_j = b_iε_j + b_jε_i − ε_iε_j.
  The δ²-bound (`delta_sq_window_bound`) and the linear bound combine via
  `|x − y| ≤ |x| + |y|` rather than `|x + y|`.

POINTWISE DOMINANCE (D-v2):  UNIVERSAL.  See `corr_tight_m2_le_corr_w_refined`.

PYTHON CROSS-REFERENCE STATUS:
  Lean (this file):                D-v2 (factor 1, strictly tighter).
  Python `_stage1_bench.py:prune_D`: still D-v1 (factor 3); upgrade is a parallel patch.
  Python `run_cascade.py:_prune_dynamic_int*`: still D-v1; upgrade pending.

NEW AXIOMS DECLARED IN THIS FILE:  ZERO.  Only existing axioms (cs_eq1_w_refined,
cascade_all_pruned_w from WRefinedBound) are imported via dependencies.
-/
