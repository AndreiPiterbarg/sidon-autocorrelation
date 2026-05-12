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
-- Reversal Symmetry (Claims 3.3a, 3.3e)
-- Source: 99433443-0e82-4f9f-a69b-82ba51cd0537-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Reversal of a composition (ℕ-valued). -/
def rev_vector {d : ℕ} (c : Fin d → ℕ) : Fin d → ℕ :=
  fun i => c ⟨d - 1 - i.1, by omega⟩

/-- Reversal of a real-valued vector. -/
def rev_vector_real {d : ℕ} (a : Fin d → ℝ) : Fin d → ℝ :=
  fun i => a ⟨d - 1 - i.1, by omega⟩

/-- Claim 3.3a: conv[k](a) = conv[2d-2-k](rev(a)). -/
theorem autoconv_reversal_symmetry {d : ℕ} (hd : d > 0) (a : Fin d → ℝ)
    (k : ℕ) (hk : k ≤ 2 * d - 2) :
    discrete_autoconvolution a k =
    discrete_autoconvolution (rev_vector_real a) (2 * d - 2 - k) := by
  unfold discrete_autoconvolution;
  have h_change : ∑ i : Fin d, ∑ j : Fin d, (if i.val + j.val = k then a i * a j else 0) = ∑ i : Fin d, ∑ j : Fin d, (if (d - 1 - i.val) + (d - 1 - j.val) = k then a (Fin.mk (d - 1 - i.val) (by omega)) * a (Fin.mk (d - 1 - j.val) (by omega)) else 0) := by
    apply Finset.sum_bij (fun i _ => Fin.mk (d - 1 - i.val) (by
    exact lt_of_le_of_lt ( Nat.sub_le _ _ ) ( Nat.pred_lt hd.ne' )))
    all_goals generalize_proofs at *;
    · exact fun _ _ => Finset.mem_univ _;
    · grind;
    · exact fun b _ => ⟨ ⟨ d - 1 - b, by omega ⟩, Finset.mem_univ _, by simp +decide [ tsub_tsub_cancel_of_le ( show b.val ≤ d - 1 from Nat.le_sub_one_of_lt b.2 ) ] ⟩;
    · intro i hi;
      apply Finset.sum_bij (fun j _ => Fin.mk (d - 1 - j.val) (by
      exact lt_of_le_of_lt (Nat.sub_le _ _) (Nat.pred_lt hd.ne')))
      all_goals generalize_proofs at *;
      · exact fun _ _ => Finset.mem_univ _;
      · grind;
      · exact fun j _ => ⟨ ⟨ d - 1 - j, by omega ⟩, Finset.mem_univ _, by simp +decide [ Nat.sub_sub_self ( show j.val ≤ d - 1 from Nat.le_sub_one_of_lt j.2 ) ] ⟩;
      · simp +decide [ Nat.sub_sub_self ( Nat.le_sub_one_of_lt i.2 ), Nat.sub_sub_self ( Nat.le_sub_one_of_lt ( Fin.is_lt _ ) ) ];
  convert h_change using 4;
  omega

/-- Claim 3.3e (helper): left sum + reversed left sum = S (fine grid, S = 4nm). -/
theorem left_sum_reversal (n : ℕ) (hn : n > 0) (m : ℕ)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = 4 * n * m) :
    (∑ i : Fin n, c ⟨i.1, by omega⟩) +
    (∑ i : Fin n, c ⟨2 * n - 1 - i.1, by omega⟩) = 4 * n * m := by
  rw [ ← hc, eq_comm ]
  generalize_proofs at *;
  have h_split : Finset.sum (Finset.univ : Finset (Fin (2 * n))) (fun i => c i) = Finset.sum (Finset.image (fun i : Fin n => ⟨i.val, by omega⟩) Finset.univ) (fun i => c i) + Finset.sum (Finset.image (fun i : Fin n => ⟨2 * n - 1 - i.val, by omega⟩) Finset.univ) (fun i => c i) := by
    rw [ ← Finset.sum_union ] <;> congr! 1
    generalize_proofs at *; (
    apply Finset.ext
    intro j
    simp [Finset.mem_union, Finset.mem_image];
    by_cases h : j.val < n <;> [ exact Or.inl ⟨ ⟨ j.val, by linarith ⟩, rfl ⟩ ; exact Or.inr ⟨ ⟨ 2 * n - 1 - j.val, by omega ⟩, by simp +decide [ Nat.sub_sub_self ( show j.val ≤ 2 * n - 1 from Nat.le_sub_one_of_lt <| Fin.is_lt j ) ] ⟩ ])
    generalize_proofs at *; (
    have h_disjoint : ∀ i j : Fin n, i.val ≠ 2 * n - 1 - j.val := by
      intro i j; omega;
    generalize_proofs at *; (
    simp +decide [ Finset.disjoint_left ];
    exact fun i j => Ne.symm ( h_disjoint i j )));
  rw [ h_split, Finset.sum_image, Finset.sum_image ] <;> simp +decide [ Fin.ext_iff ];
  exact fun i _ j _ hij => Fin.ext <| by injection hij with hij; rw [ tsub_right_inj ] at hij <;> linarith [ Fin.is_lt i, Fin.is_lt j, Nat.sub_add_cancel ( by linarith [ Fin.is_lt i, Fin.is_lt j ] : 1 ≤ 2 * n ) ] ;

/-- Claim 3.3e: asymmetry condition is symmetric under reversal. -/
theorem asymmetry_reversal_symmetric (n : ℕ) (hn : n > 0) (m : ℕ) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = 4 * n * m)
    (threshold : ℝ) (ht : 0 ≤ threshold) (ht1 : threshold ≤ 1)
    (L : ℝ) (hL : L = (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℝ)) / (4 * n * m))
    (h_prune : L ≥ threshold ∨ 1 - L ≥ threshold) :
    let L_rev := (∑ i : Fin n, (c ⟨2 * n - 1 - i.1, by omega⟩ : ℝ)) / (4 * n * m)
    L_rev ≥ threshold ∨ 1 - L_rev ≥ threshold := by
  have h_sum : L + (∑ i : Fin n, (c ⟨2 * n - 1 - i.val, by omega⟩ : ℝ)) / (4 * n * m) = 1 := by
    have h_sum : (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℝ)) + (∑ i : Fin n, (c ⟨2 * n - 1 - i.1, by omega⟩ : ℝ)) = 4 * n * m := by
      norm_cast; exact left_sum_reversal n hn m c hc;
    generalize_proofs at *; (
    rw [ hL, ← add_div, h_sum, div_self ( by positivity ) ]);
  grind +ring

end -- noncomputable section
