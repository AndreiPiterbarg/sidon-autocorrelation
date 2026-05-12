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
-- Incremental Autoconvolution (Claim 4.2)
-- Source: 305874b1-3eed-4942-afb4-5daac0ccf2ac-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Integer autoconvolution (ℤ-valued, for exact computation). -/
def int_autoconvolution {d : ℕ} (c : Fin d → ℤ) (t : ℕ) : ℤ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t then c i * c j else 0

/-- The delta between new and old autoconvolution. -/
def autoconv_delta {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) : ℤ :=
  int_autoconvolution c' t - int_autoconvolution c t

/-- Delta equals the sum of per-entry differences. -/
theorem delta_eq_sum {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) :
    autoconv_delta c c' t =
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c' i * c' j - c i * c j else 0 := by
  simp [autoconv_delta, int_autoconvolution];
  simp [← Finset.sum_sub_distrib];
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  aesop

/-- Terms where neither index changed contribute zero. -/
theorem unchanged_terms_zero {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (i j : Fin d) (hi : i ∉ S) (hj : j ∉ S) :
    c' i * c' j - c i * c j = 0 := by
  simp [hS i hi, hS j hj]

/-- Delta decomposes into three disjoint groups by membership in S. -/
theorem delta_three_way_split {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (t : ℕ) :
    autoconv_delta c c' t =
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∈ S ∧ j ∈ S then c' i * c' j - c i * c j else 0) +
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∈ S ∧ j ∉ S then c' i * c' j - c i * c j else 0) +
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∉ S ∧ j ∈ S then c' i * c' j - c i * c j else 0) := by
  rw [ ← Finset.sum_add_distrib, ← Finset.sum_add_distrib ];
  convert delta_eq_sum c c' t using 2;
  rename_i i hi; rw [ ← Finset.sum_add_distrib, ← Finset.sum_add_distrib ] ; congr ; ext j ; aesop;

/-- Cross-terms factor when one index is unchanged. -/
theorem cross_term_simplify {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (i j : Fin d) (_hi : i ∈ S) (hj : j ∉ S) :
    c' i * c' j - c i * c j = (c' i - c i) * c j := by
  rw [hS j hj];
  ring

/-- Claim 4.2: Incremental update is bit-exact. old_conv + delta = new_conv. -/
theorem incremental_update_correct {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) :
    int_autoconvolution c t + autoconv_delta c c' t = int_autoconvolution c' t := by
  rw [show autoconv_delta c c' t = int_autoconvolution c' t - int_autoconvolution c t from rfl]
  ring

/-- The four membership groups are exhaustive. -/
theorem groups_exhaustive {d : ℕ} (S : Finset (Fin d)) (i j : Fin d) :
    (i ∈ S ∧ j ∈ S) ∨ (i ∈ S ∧ j ∉ S) ∨ (i ∉ S ∧ j ∈ S) ∨ (i ∉ S ∧ j ∉ S) := by
  tauto

/-- The four membership groups are pairwise disjoint. -/
theorem groups_disjoint {d : ℕ} (S : Finset (Fin d)) (i j : Fin d) :
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∈ S ∧ j ∉ S)) ∧
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∈ S)) ∧
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∉ S)) ∧
    ¬((i ∈ S ∧ j ∉ S) ∧ (i ∉ S ∧ j ∈ S)) ∧
    ¬((i ∈ S ∧ j ∉ S) ∧ (i ∉ S ∧ j ∉ S)) ∧
    ¬((i ∉ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∉ S)) := by
  tauto

end -- noncomputable section
