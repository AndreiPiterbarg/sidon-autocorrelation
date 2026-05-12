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
-- Subtree Pruning (Claim 4.4)
-- Source: prompt12_subtree_pruning.lean
-- ═══════════════════════════════════════════════════════════════════════════════

-- Inequality 1: partial conv ≤ full conv (restricting to i,j < 2p gives subset of nonneg terms)
-- Source: output (13).lean (UUID: d7ccdaef) — PROVED
theorem partial_conv_le_full_conv {d : ℕ} (c : Fin d → ℤ) (hc : ∀ i, 0 ≤ c i)
    (p : ℕ) (_hp : 2 * p ≤ d) (t : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d,
      (if i.1 + j.1 = t ∧ i.1 < 2*p ∧ j.1 < 2*p then c i * c j else 0) ≤
    ∑ i : Fin d, ∑ j : Fin d,
      (if i.1 + j.1 = t then c i * c j else 0) := by
  apply Finset.sum_le_sum; intro i _; apply Finset.sum_le_sum; intro j _; split_ifs <;> simp_all +decide;
  apply mul_nonneg (hc i) (hc j)

-- Inequality 2: W_int bounded for children in subtree (Even d version, fully proved)
-- Source: output (13).lean (UUID: d7ccdaef) — PROVED
theorem w_int_bounded_unfixed {d : ℕ} (hd : Even d) (child : Fin d → ℕ) (parent : Fin (d/2) → ℕ)
    (p : ℕ) (_hp : 2*p ≤ d)
    (h_split : ∀ q : Fin (d/2), p ≤ q.1 →
      (if h : 2*q.1 < d then child ⟨2*q.1, h⟩ else 0) +
      (if h : 2*q.1+1 < d then child ⟨2*q.1+1, h⟩ else 0) = parent q)
    (lo hi : ℕ) (hlo : lo ≤ hi) (hhi : hi < d) :
    ∑ i ∈ Finset.Icc (max lo (2*p)) hi, (if h : i < d then child ⟨i, h⟩ else 0) ≤
    ∑ q ∈ Finset.filter (fun q => 2*q ≤ hi ∧ lo ≤ 2*q+1)
      (Finset.Icc p (d/2 - 1)), (if h : q < d/2 then parent ⟨q, h⟩ else 0) := by
  set f : ℕ → ℕ := fun i => i / 2
  set S := Finset.Icc (max lo (2 * p)) hi
  set Q := Finset.filter (fun q => 2 * q ≤ hi ∧ lo ≤ 2 * q + 1) (Finset.Icc p (d / 2 - 1)) with hQ_def
  have h_f_S : S ⊆ Finset.biUnion Q (fun q => Finset.Icc (2 * q) (2 * q + 1)) := by
    simp +zetaDelta at *;
    intro i hi; simp_all +decide;
    exact ⟨ i / 2, ⟨ ⟨ by omega, Nat.le_sub_one_of_lt ( Nat.div_lt_of_lt_mul <| by linarith [ Nat.div_mul_cancel ( even_iff_two_dvd.mp hd ) ] ) ⟩, by omega, by omega ⟩, by omega, by omega ⟩;
  have h_sum_bound : ∑ i ∈ S, (if h : i < d then child ⟨i, h⟩ else 0) ≤ ∑ q ∈ Q, (∑ i ∈ Finset.Icc (2 * q) (2 * q + 1), (if h : i < d then child ⟨i, h⟩ else 0)) := by
    refine' le_trans ( Finset.sum_le_sum_of_subset h_f_S ) _;
    rw [ Finset.sum_biUnion ];
    exact fun a ha b hb hab => Finset.disjoint_left.mpr fun x hx₁ hx₂ => hab <| by linarith [ Finset.mem_Icc.mp hx₁, Finset.mem_Icc.mp hx₂ ] ;
  refine le_trans h_sum_bound <| Finset.sum_le_sum fun q hq => ?_;
  split_ifs <;> simp_all +decide;
  · erw [ Finset.sum_Ico_succ_top ] <;> norm_num [ ← h_split ⟨ q, by linarith ⟩ hq.1.1 ];
  · grind

theorem w_int_bounded_corrected {d : ℕ} (hd : Even d) (child : Fin d → ℕ) (parent : Fin (d/2) → ℕ)
    (p : ℕ) (hp : 2*p ≤ d)
    (h_split : ∀ q : Fin (d/2), p ≤ q.1 →
      (if h : 2*q.1 < d then child ⟨2*q.1, h⟩ else 0) +
      (if h : 2*q.1+1 < d then child ⟨2*q.1+1, h⟩ else 0) = parent q)
    (lo hi : ℕ) (hlo : lo ≤ hi) (hhi : hi < d) :
    ∑ i ∈ Finset.Icc lo hi, (if h : i < d then child ⟨i, h⟩ else 0) ≤
    (∑ i ∈ Finset.Icc lo (min hi (2*p-1)), (if h : i < d then child ⟨i, h⟩ else 0)) +
    (∑ q ∈ Finset.filter (fun q => 2*q ≤ hi ∧ lo ≤ 2*q+1)
      (Finset.Icc p (d/2 - 1)), (if h : q < d/2 then parent ⟨q, h⟩ else 0)) := by
  have h_split_sum : ∑ i ∈ Finset.Icc lo hi, (if h : i < d then child ⟨i, h⟩ else 0) ≤ (∑ i ∈ Finset.Icc lo (min hi (2*p-1)), (if h : i < d then child ⟨i, h⟩ else 0)) + (∑ i ∈ Finset.Icc (max lo (2*p)) hi, (if h : i < d then child ⟨i, h⟩ else 0)) := by
    cases max_cases lo ( 2 * p ) <;> simp_all +decide;
    rw [ ← Finset.sum_union ];
    · refine Finset.sum_le_sum_of_subset ?_;
      exact fun x hx => if hx' : x ≤ 2 * p - 1 then Finset.mem_union_left _ <| Finset.mem_Icc.mpr ⟨ Finset.mem_Icc.mp hx |>.1, le_min ( Finset.mem_Icc.mp hx |>.2 ) hx' ⟩ else Finset.mem_union_right _ <| Finset.mem_Icc.mpr ⟨ by omega, Finset.mem_Icc.mp hx |>.2 ⟩;
    · exact Finset.disjoint_left.mpr fun x hx₁ hx₂ => by linarith [ Finset.mem_Icc.mp hx₁, Finset.mem_Icc.mp hx₂, min_le_left hi ( 2 * p - 1 ), min_le_right hi ( 2 * p - 1 ), Nat.sub_add_cancel ( by linarith : 1 ≤ 2 * p ) ] ;
  refine le_trans h_split_sum <| add_le_add_left ?_ _;
  convert w_int_bounded_unfixed hd child parent p hp h_split lo hi hlo hhi using 1

theorem w_int_bounded {d : ℕ} (hd : Even d) (child : Fin d → ℕ) (parent : Fin (d/2) → ℕ)
    (p : ℕ) (hp : 2*p ≤ d)
    (h_split : ∀ q : Fin (d/2), p ≤ q.1 →
      (if h : 2*q.1 < d then child ⟨2*q.1, h⟩ else 0) +
      (if h : 2*q.1+1 < d then child ⟨2*q.1+1, h⟩ else 0) = parent q)
    (lo hi : ℕ) (hlo : lo ≤ hi) (hhi : hi < d) :
    ∑ i ∈ Finset.Icc lo hi, (if h : i < d then child ⟨i, h⟩ else 0) ≤
    (∑ i ∈ Finset.Icc lo (min hi (2*p-1)), (if h : i < d then child ⟨i, h⟩ else 0)) +
    (∑ q ∈ Finset.filter (fun q => 2*q ≤ hi ∧ lo ≤ 2*q+1)
      (Finset.Icc p (d/2 - 1)), (if h : q < d/2 then parent ⟨q, h⟩ else 0)) := by
  exact w_int_bounded_corrected hd child parent p hp h_split lo hi hlo hhi

-- Inequality 3: dyn_it is non-decreasing in W
theorem dyn_it_mono (base s : ℝ) (hs : 0 < s) (W1 W2 : ℝ) (hW : W1 ≤ W2) :
    ⌊(base + 2 * W1) * s⌋ ≤ ⌊(base + 2 * W2) * s⌋ := by
  apply Int.floor_le_floor
  apply mul_le_mul_of_nonneg_right
  · linarith
  · exact le_of_lt hs

-- Chain: subtree pruning is sound
theorem subtree_pruning_chain (ws_partial ws_full dyn_max dyn_actual : ℤ)
    (h1 : ws_full ≥ ws_partial)
    (h2 : ws_partial > dyn_max)
    (h3 : dyn_max ≥ dyn_actual) :
    ws_full > dyn_actual := by
  omega

end -- noncomputable section
