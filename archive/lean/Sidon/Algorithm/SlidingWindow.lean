import Mathlib

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
-- Sliding Window and Zero-Bin Skip (Claims 4.12, 4.13)
-- Source: prompt15_sliding_window_and_zero_skip.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.12: Sliding window inductive step — W_{s+1} = W_s + A[s+n_cv] - A[s]. -/
-- Source: output (16).lean (UUID: 873cc3c5) — PROVED (with dite indexing)
theorem sliding_window_step {N : ℕ} (A : Fin N → ℤ) (n_cv s : ℕ)
    (hs : s + n_cv < N)
    (W_s : ℤ) (hW : W_s = ∑ k ∈ Finset.Ico s (s + n_cv), if h : k < N then A ⟨k, h⟩ else 0) :
    W_s + A ⟨s + n_cv, hs⟩ - A ⟨s, by omega⟩ =
    ∑ k ∈ Finset.Ico (s + 1) (s + 1 + n_cv), if h : k < N then A ⟨k, h⟩ else 0 := by
  rw [ Finset.sum_Ico_eq_sub _ ] at * <;> norm_num at *;
  rw [ Finset.sum_range_succ ] ; simp +decide [ add_right_comm, *, Finset.sum_range_succ ] ; ring;
  grind +ring

/-- Claim 4.13: Zero term vanishes in products. -/
theorem zero_term_vanishes (a b : ℤ) (hb : b = 0) : a * b = 0 := by
  subst hb; ring

-- Filtering out c_j = 0 terms doesn't change a sum of products
theorem sum_filter_zero {d : ℕ} (c : Fin d → ℤ) (f : Fin d → ℤ) :
    ∑ j : Fin d, c j * f j =
    ∑ j ∈ (Finset.univ.filter fun j => c j ≠ 0), c j * f j := by
  symm
  apply Finset.sum_subset (Finset.filter_subset _ _)
  intro j _ hj
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at hj
  simp [hj]

-- Autoconvolution with zero-skip = full autoconvolution
-- Source: output (16).lean (UUID: 873cc3c5) — PROVED
theorem autoconv_zero_skip {d : ℕ} (c : Fin d → ℤ) (t : ℕ) :
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c i * c j else 0) =
    (∑ i ∈ (Finset.univ.filter fun i => c i ≠ 0),
      ∑ j ∈ (Finset.univ.filter fun j => c j ≠ 0),
        if i.1 + j.1 = t then c i * c j else 0) := by
  simp +contextual [ Finset.sum_filter ];
  exact Finset.sum_congr rfl fun i hi => by by_cases hi0 : c i = 0 <;> simp +decide [ hi0 ] ; exact Finset.sum_congr rfl fun j hj => by aesop;

-- Cross-term zero-skip: exact for unchanged-bin cross-terms
theorem cross_term_zero_skip {d : ℕ} (c : Fin d → ℤ) (delta : ℤ)
    (S : Finset (Fin d)) :
    (∑ q ∈ S, delta * c q) =
    (∑ q ∈ S.filter (fun q => c q ≠ 0), delta * c q) := by
  symm
  apply Finset.sum_subset (Finset.filter_subset _ _)
  intro q hqS hq
  have hcq : c q = 0 := by
    by_contra h
    exact hq (Finset.mem_filter.mpr ⟨hqS, h⟩)
  simp [hcq]

end -- noncomputable section
