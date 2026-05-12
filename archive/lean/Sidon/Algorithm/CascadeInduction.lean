/-
Sidon Autocorrelation Project — Cascade Induction (Claim 3.4)

Fine grid: at level L with half-dimension 2^L * n, d = 2*(2^L*n) bins,
S = 4*(2^L*n)*m.
-/

import Mathlib
import Sidon.Defs

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
-- Cascade Induction (Claim 3.4)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Merge consecutive child bin pairs back to parent resolution. -/
def merge_pairs {d : ℕ} (child : Fin (2 * d) → ℕ) : Fin d → ℕ :=
  fun i => child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩

-- merge_pairs preserves total mass
theorem merge_pairs_sum {d m : ℕ} (child : Fin (2 * d) → ℕ) (hc : ∑ i, child i = m) :
    ∑ i, merge_pairs child i = m := by
  unfold merge_pairs;
  rw [ ← hc, eq_comm ];
  clear hc;
  induction' d with d ih <;> simp_all +decide [ Nat.mul_succ, Fin.sum_univ_castSucc ] ; ring!;

-- Claim 3.4: If all compositions at resolution d_L are pruned, then c ≥ c_target
-- Fine grid: S = 4 * (2^L * n) * m at level L.
theorem cascade_completeness_step
  (n m : ℕ) (c_target : ℝ)
  (_hn : n > 0) (_hm : m > 0) (_hct : 0 < c_target)
  (L : ℕ)
  (tv : (d : ℕ) → (m : ℕ) → (Fin (2 * d) → ℕ) → ℕ → ℕ → ℝ)
  (discretize : (ℝ → ℝ) → (d : ℕ) → (m : ℕ) → Fin d → ℕ)
  (h_discretize_sum : ∀ f, ∑ i, discretize f (2 * (2^L * n)) m i = 4 * (2^L * n) * m)
  (h_pruning_sound : ∀ (n m : ℕ) (c_target : ℝ) (L : ℕ) (c : Fin (2 * (2^L * n)) → ℕ) (ℓ s_lo : ℕ) (f : ℝ → ℝ),
      tv (2^L * n) m c ℓ s_lo > c_target + 2 / (m : ℝ) + 1 / (m : ℝ)^2 →
      discretize f (2 * (2^L * n)) m = c →
      autoconvolution_ratio f ≥ c_target)
  (h_all_pruned : ∀ c : Fin (2 * (2^L * n)) → ℕ, ∑ i, c i = 4 * (2^L * n) * m →
      ∃ ℓ s_lo, tv (2^L * n) m c ℓ s_lo > c_target + 2 / (m : ℝ) + 1 / (m : ℝ)^2) :
  ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
    Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
    MeasureTheory.integral MeasureTheory.volume f ≠ 0 →
    autoconvolution_ratio f ≥ c_target := by
  intro f hf_nonneg hf_supp hf_int_ne_zero
  let c := discretize f (2 * (2^L * n)) m
  have hc_sum : ∑ i, c i = 4 * (2^L * n) * m := h_discretize_sum f
  obtain ⟨ℓ, s_lo, h_val⟩ := h_all_pruned c hc_sum
  exact h_pruning_sound n m c_target L c ℓ s_lo f h_val rfl

end -- noncomputable section
