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
-- Gray Code Kernel (Claims 4.9, 4.10, 4.11)
-- Source: prompt14_gray_code_kernel.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.9: There exists a bijection between Fin(∏ rᵢ) and the dependent product
    ∀ i, Fin(rᵢ). The computational code uses a Gray code traversal; here we prove only
    the existence via cardinality matching (Fintype.equivOfCardEq). -/
-- Source: output (15).lean (UUID: 7753e964) — PROVED
theorem dependent_product_bijection {k : ℕ} (r : Fin k → ℕ) :
    ∃ (f : Fin (∏ i, r i) → (∀ i : Fin k, Fin (r i))),
      Function.Bijective f := by
  have h_equiv : Nonempty (Fin (∏ i, r i) ≃ (∀ i, Fin (r i))) := by
    refine' ⟨ Fintype.equivOfCardEq _ ⟩ ; aesop;
  exact ⟨ _, Equiv.bijective h_equiv.some ⟩

/-- Claim 4.10: Cross-term split for arbitrary position. -/
-- Source: output (15).lean (UUID: 7753e964) — PROVED
theorem cross_term_split {d : ℕ} (p : ℕ) (hp : 2*p+1 < d)
    (f : Fin d → ℤ) :
    (∑ q : Fin d, if q.1 ≠ 2*p ∧ q.1 ≠ 2*p+1 then f q else 0) =
    (∑ q ∈ (Finset.range (2*p)).attach, f ⟨q.1, Nat.lt_trans (Finset.mem_range.mp q.2) (Nat.lt_trans (Nat.lt_succ_self _) hp)⟩) +
    (∑ q ∈ (Finset.Ico (2*p+2) d).attach, f ⟨q.1, (Finset.mem_Ico.mp q.2).2⟩) := by
  simp +decide [ Finset.sum_ite ];
  convert Finset.sum_union ?_ using 2;
  rotate_left;
  rotate_left;
  rotate_left;
  exact Finset.univ.filter fun x => x.val < 2 * p;
  exact Finset.univ.filter fun x => x.val > 2 * p + 1;
  infer_instance;
  · exact Finset.disjoint_filter.mpr fun _ _ _ _ => by linarith;
  · grind;
  · refine' Finset.sum_bij ( fun x hx => ⟨ x, by linarith [ Finset.mem_range.mp x.2 ] ⟩ ) _ _ _ _ <;> aesop;
  · refine' Finset.sum_bij ( fun x hx => ⟨ x, by linarith [ Finset.mem_Ico.mp x.2 ] ⟩ ) _ _ _ _ <;> aesop

/-- Claim 4.11: W_int correctness under Gray code updates. -/
-- Source: output (15).lean (UUID: 7753e964) — PROVED
theorem w_int_gray_update (lo_bin hi_bin : ℕ) (c c' : ℕ → ℤ)
    (p : ℕ)
    (h_same : ∀ i, i ≠ 2*p ∧ i ≠ 2*p+1 → c' i = c i)
    (W_old : ℤ) (hW : W_old = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i) :
    ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i =
      W_old + (if 2*p ∈ Finset.Icc lo_bin hi_bin then c' (2*p) - c (2*p) else 0)
           + (if (2*p+1) ∈ Finset.Icc lo_bin hi_bin then c' (2*p+1) - c (2*p+1) else 0) := by
  have h_split_sum : ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i + ∑ i ∈ Finset.Icc lo_bin hi_bin, (if i = 2 * p then c' (2 * p) - c (2 * p) else 0) + ∑ i ∈ Finset.Icc lo_bin hi_bin, (if i = 2 * p + 1 then c' (2 * p + 1) - c (2 * p + 1) else 0) := by
    simpa only [ ← Finset.sum_add_distrib ] using Finset.sum_congr rfl fun i hi => by
      by_cases hi1 : i = 2 * p <;> by_cases hi2 : i = 2 * p + 1 <;> simp_all
  simp_all

end -- noncomputable section
