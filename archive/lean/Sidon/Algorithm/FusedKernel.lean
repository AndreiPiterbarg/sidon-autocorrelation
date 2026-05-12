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
-- Fused Kernel and Quick-Check (Claims 4.1, 4.3)
-- Source: e868a126-2d3d-4a3f-8940-ed4c553ac681-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.1: There exists a bijection between Fin(∏(hi-lo+1)) and the Cartesian product
    ∀ i, Fin(hi i - lo i + 1). The computational code uses an odometer traversal to realize
    this bijection; here we prove only the existence (cardinality matching). -/
theorem cartesian_product_bijection {d : ℕ} (lo hi : Fin d → ℕ) :
    ∃ (f : Fin (∏ i, (hi i - lo i + 1)) → (∀ i : Fin d, Fin (hi i - lo i + 1))),
      Function.Bijective f := by
  have h_bij : Nonempty (Fin (∏ i, (hi i - lo i + 1)) ≃ (∀ i, Fin (hi i - lo i + 1))) := by
    refine' ⟨ Fintype.equivOfCardEq _ ⟩ ; aesop;
  exact ⟨ _, Equiv.bijective h_bij.some ⟩

/-- Claim 4.3: Witness extraction — a specific (ℓ, s) exceeding the threshold implies
    the existence of such a pair (existential introduction from a concrete witness). -/
theorem witness_extraction {_d : ℕ} (ws : ℕ → ℕ → ℤ) (dyn : ℕ → ℕ → ℤ)
    (ℓ_star s_star : ℕ) (h : ws ℓ_star s_star > dyn ℓ_star s_star) :
    ∃ ℓ s, ws ℓ s > dyn ℓ s :=
  ⟨ℓ_star, s_star, h⟩

/-- W_int fast-path update correctness. -/
theorem w_int_fast_update (lo_bin hi_bin : ℕ) (c c' : ℕ → ℤ)
    (p : ℕ)
    (h_same : ∀ i, i ≠ 2*p ∧ i ≠ 2*p+1 → c' i = c i)
    (W_old : ℤ) (hW : W_old = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i)
    (_delta : ℤ) (_hd : _delta = (c' (2*p) - c (2*p)) + (c' (2*p+1) - c (2*p+1))) :
    ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i =
      W_old + (if 2*p ∈ Finset.Icc lo_bin hi_bin then c' (2*p) - c (2*p) else 0)
           + (if 2*p+1 ∈ Finset.Icc lo_bin hi_bin then c' (2*p+1) - c (2*p+1) else 0) := by
  have h_split : ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i + ∑ i ∈ Finset.Icc lo_bin hi_bin, (if i = 2 * p then (c' (2 * p) - c (2 * p)) else 0) + ∑ i ∈ Finset.Icc lo_bin hi_bin, (if i = 2 * p + 1 then (c' (2 * p + 1) - c (2 * p + 1)) else 0) := by
    have h_split : ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i = ∑ i ∈ Finset.Icc lo_bin hi_bin, (c i + (if i = 2 * p then c' (2 * p) - c (2 * p) else 0) + (if i = 2 * p + 1 then c' (2 * p + 1) - c (2 * p + 1) else 0)) := by
      grind;
    rw [ h_split, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib ];
  simp [h_split, hW]

end -- noncomputable section
