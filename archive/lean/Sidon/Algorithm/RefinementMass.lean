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
-- Refinement Mass Preservation (Claims 3.2c, 4.6)
-- Source: b66ccc2f-25d7-46ad-80f3-eb01a82a1669-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Each parent bin splits into an even-odd child pair summing to the parent. -/
theorem child_bin_pair_sum (d : ℕ) (_hd : d > 0)
    (parent : Fin d → ℕ) (a : Fin d → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i)
    (i : Fin d) :
    child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩ = parent i := by
  rw [hc_even, hc_odd]
  simp [ha i]

/-- Claim 3.2c: Children preserve total mass. -/
theorem child_preserves_total_mass (d : ℕ) (hd : d > 0) (m : ℕ)
    (parent : Fin d → ℕ) (hp : ∑ i, parent i = m)
    (a : Fin d → ℕ) (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j, child j = m := by
  have h_split_sum : ∑ j : Fin (2 * d), child j = ∑ i : Fin d, (child ⟨2 * i, by omega⟩ + child ⟨2 * i + 1, by omega⟩) := by
    have h_split : Finset.range (2 * d) = Finset.image (fun i => 2 * i) (Finset.range d) ∪ Finset.image (fun i => 2 * i + 1) (Finset.range d) := by
      ext i
      simp [Finset.mem_range, Finset.mem_image];
      exact ⟨ fun hi => by rcases Nat.even_or_odd' i with ⟨ k, rfl | rfl ⟩ <;> [ left; right ] <;> exact ⟨ k, by linarith, rfl ⟩, fun hi => by rcases hi with ( ⟨ k, hk, rfl ⟩ | ⟨ k, hk, rfl ⟩ ) <;> linarith ⟩;
    rw [ Finset.sum_fin_eq_sum_range ];
    rw [ h_split, Finset.sum_union ];
    · norm_num [ Finset.sum_add_distrib, Finset.sum_range ];
      exact Finset.sum_congr rfl fun i hi => by split_ifs <;> linarith [ Fin.is_lt i ] ;
    · norm_num [ Finset.disjoint_right ];
      intros; omega;
  grind

/-- Claim 4.6: Left-half sum is invariant under refinement. -/
theorem left_half_sum_invariant (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℕ) (ha : ∀ i, a i ≤ parent i)
    (child : Fin (4 * n) → ℕ)
    (hc_even : ∀ i : Fin (2 * n), child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin (2 * n), child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j : Fin (2 * n), (child ⟨j.1, by omega⟩ : ℕ) =
    ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  have h_split : ∑ j : Fin (2 * n), child ⟨j.val, by linarith [Fin.is_lt j]⟩ = ∑ i : Fin n, (child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩) := by
    have h_split : Finset.range (2 * n) = Finset.image (fun i => 2 * i) (Finset.range n) ∪ Finset.image (fun i => 2 * i + 1) (Finset.range n) := by
      ext i
      simp [Finset.mem_range, Finset.mem_image];
      exact ⟨ fun hi => by rcases Nat.even_or_odd' i with ⟨ k, rfl | rfl ⟩ <;> [ left; right ] <;> exact ⟨ k, by linarith, rfl ⟩, fun hi => by rcases hi with ( ⟨ k, hk, rfl ⟩ | ⟨ k, hk, rfl ⟩ ) <;> linarith ⟩
    generalize_proofs at *;
    rw [ Finset.sum_fin_eq_sum_range ] ; simp_all +decide [ Finset.sum_add_distrib ] ; (
    rw [ Finset.sum_union ] <;> norm_num [ Finset.sum_image, Finset.sum_range ];
    · exact Finset.sum_congr rfl fun i hi => by split_ifs <;> linarith [ Fin.is_lt i ] ;
    · norm_num [ Finset.disjoint_right ] ; omega;;)
  generalize_proofs at *;
  grind

/-- Any two refinements of the same parent have equal left-half sums. -/
theorem left_half_sum_same_for_all_children (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ)
    (a₁ a₂ : Fin (2 * n) → ℕ)
    (ha₁ : ∀ i, a₁ i ≤ parent i) (ha₂ : ∀ i, a₂ i ≤ parent i)
    (child₁ child₂ : Fin (4 * n) → ℕ)
    (hc₁_even : ∀ i : Fin (2 * n), child₁ ⟨2 * i.1, by omega⟩ = a₁ i)
    (hc₁_odd : ∀ i : Fin (2 * n), child₁ ⟨2 * i.1 + 1, by omega⟩ = parent i - a₁ i)
    (hc₂_even : ∀ i : Fin (2 * n), child₂ ⟨2 * i.1, by omega⟩ = a₂ i)
    (hc₂_odd : ∀ i : Fin (2 * n), child₂ ⟨2 * i.1 + 1, by omega⟩ = parent i - a₂ i) :
    ∑ j : Fin (2 * n), (child₁ ⟨j.1, by omega⟩ : ℕ) =
    ∑ j : Fin (2 * n), (child₂ ⟨j.1, by omega⟩ : ℕ) := by
  convert left_half_sum_invariant n hn parent a₁ ha₁ child₁ hc₁_even hc₁_odd using 1;
  apply left_half_sum_invariant n hn parent a₂ ha₂ child₂ hc₂_even hc₂_odd

end -- noncomputable section
