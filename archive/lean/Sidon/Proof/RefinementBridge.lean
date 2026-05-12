/-
Sidon Autocorrelation Project — Refinement Bridge

Proves that canonical_discretization at doubled resolution produces a
valid child of the coarser discretization (with ±1 floor rounding tolerance).

Key result: refinement_preserves_discretization
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.TestValueBounds

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 16000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 40000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- Floor Rounding Lemma
-- ═══════════════════════════════════════════════════════════════════════════════

/-- floor(2x).natAbs is between 2*floor(x).natAbs and 2*floor(x).natAbs+1. -/
private lemma floor_double_natAbs_bounds (x : ℝ) (hx : 0 ≤ x) :
    2 * ⌊x⌋.natAbs ≤ ⌊2 * x⌋.natAbs ∧ ⌊2 * x⌋.natAbs ≤ 2 * ⌊x⌋.natAbs + 1 := by
  have hfx : (0 : ℤ) ≤ ⌊x⌋ := Int.floor_nonneg.mpr hx
  have hf2x : (0 : ℤ) ≤ ⌊2 * x⌋ := Int.floor_nonneg.mpr (by linarith)
  have hlo : 2 * ⌊x⌋ ≤ ⌊2 * x⌋ := by
    rw [Int.le_floor]; push_cast; linarith [Int.floor_le x]
  have hhi : ⌊2 * x⌋ ≤ 2 * ⌊x⌋ + 1 := by
    have : ⌊2 * x⌋ < 2 * ⌊x⌋ + 2 := by
      rw [Int.floor_lt]; push_cast; linarith [Int.lt_floor_add_one x]
    omega
  have h1 : (⌊x⌋.natAbs : ℤ) = ⌊x⌋ := Int.natAbs_of_nonneg hfx
  have h2 : (⌊2 * x⌋.natAbs : ℤ) = ⌊2 * x⌋ := Int.natAbs_of_nonneg hf2x
  constructor <;> omega

-- ═══════════════════════════════════════════════════════════════════════════════
-- Bin Mass Pairing
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Two consecutive fine-grid bins sum to the corresponding coarse-grid bin. -/
private lemma bin_masses_pair_eq (f : ℝ → ℝ) (n : ℕ) (hn : n > 0) (j : Fin (2 * n))
    (hf : MeasureTheory.Integrable f) :
    bin_masses f (2 * n) ⟨2 * j.1, by omega⟩ +
    bin_masses f (2 * n) ⟨2 * j.1 + 1, by omega⟩ =
    bin_masses f n j := by
  simp only [bin_masses, MeasureTheory.integral_indicator measurableSet_Ico]
  have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
  push_cast
  have hδ : (0 : ℝ) < 1 / (4 * (2 * ↑n)) := by positivity
  rw [← MeasureTheory.setIntegral_union _ measurableSet_Ico hf.integrableOn hf.integrableOn]
  · congr 1
    rw [Set.Ico_union_Ico_eq_Ico (by nlinarith) (by nlinarith)]
    ext; simp; field_simp; ring
  · rw [Set.disjoint_left]
    intro y hy1 hy2; exact absurd (lt_of_lt_of_le hy1.2 hy2.1) (lt_irrefl y)

-- ═══════════════════════════════════════════════════════════════════════════════
-- Cumulative Mass Doubling
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Helper: sum with condition j < k+1 splits into sum with j < k plus term at k. -/
private lemma sum_lt_succ_eq {α : Type*} [AddCommMonoid α] {N : ℕ} (g : Fin N → α) (k : ℕ)
    (hk : k < N) :
    (∑ j : Fin N, if j.1 < k + 1 then g j else 0) =
    (∑ j : Fin N, if j.1 < k then g j else 0) + g ⟨k, hk⟩ := by
  have step1 : (∑ j : Fin N, if j.1 < k + 1 then g j else 0) =
      (∑ j : Fin N, if j.1 < k then g j else 0) +
      (∑ j : Fin N, if j.1 = k then g j else 0) := by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl; intro j _
    split_ifs with h1 h2 <;> simp_all <;> omega
  have step2 : (∑ j : Fin N, if j.1 = k then g j else 0) = g ⟨k, hk⟩ := by
    rw [Finset.sum_eq_single_of_mem ⟨k, hk⟩ (Finset.mem_univ _)]
    · simp
    · intro b _ hb; rw [ne_eq, Fin.ext_iff] at hb; exact if_neg hb
  rw [step1, step2]

/-- Partial sum of fine-grid bin masses up to index 2k equals the partial sum
    of coarse-grid bin masses up to index k. -/
lemma cum_mass_fine_eq_coarse (f : ℝ → ℝ) (n : ℕ) (hn : n > 0)
    (hf : MeasureTheory.Integrable f)
    (k : ℕ) (hk : k ≤ 2 * n) :
    (∑ j : Fin (2 * (2 * n)), if j.1 < 2 * k then bin_masses f (2 * n) j else 0) =
    (∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) := by
  induction k with
  | zero => simp
  | succ k' ih =>
    have hk'_lt : k' < 2 * n := by omega
    have hk'_le : k' ≤ 2 * n := by omega
    have h2k'_lt : 2 * k' < 2 * (2 * n) := by omega
    have h2k'1_lt : 2 * k' + 1 < 2 * (2 * n) := by omega
    -- Fine sum: split off the top two bins
    have h_fine_2k1 := sum_lt_succ_eq (fun j => bin_masses f (2 * n) j) (2 * k' + 1) h2k'1_lt
    have h_fine_2k := sum_lt_succ_eq (fun j => bin_masses f (2 * n) j) (2 * k') h2k'_lt
    -- 2*(k'+1) = (2*k'+1) + 1
    show (∑ j : Fin (2 * (2 * n)), if j.1 < (2 * k' + 1) + 1 then
        bin_masses f (2 * n) j else 0) = _
    rw [h_fine_2k1]
    -- Now: (∑ j<2k'+1 ...) + bin(2k'+1) = ...
    -- Split the remaining sum at 2k'
    conv_lhs => arg 1; rw [h_fine_2k]
    -- Now: ((∑ j<2k' ...) + bin(2k')) + bin(2k'+1) = ...
    rw [ih hk'_le, sum_lt_succ_eq (fun j => bin_masses f n j) k' hk'_lt, add_assoc]
    congr 1
    exact bin_masses_pair_eq f n hn ⟨k', hk'_lt⟩ hf

/-- Target cumulative mass at doubled resolution, evaluated at aligned boundary 2k,
    equals twice the target cumulative mass at the coarse resolution at k. -/
lemma target_cum_mass_doubling (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (_hm : m > 0)
    (_hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (k : ℕ) (hk : k ≤ 2 * n) :
    let target_coarse := (∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) /
      (∑ j : Fin (2 * n), bin_masses f n j) * (4 * n * m : ℝ)
    let target_fine := (∑ j : Fin (2 * (2 * n)), if j.1 < 2 * k then bin_masses f (2 * n) j else 0) /
      (∑ j : Fin (2 * (2 * n)), bin_masses f (2 * n) j) * (4 * (2 * n) * m : ℝ)
    target_fine = 2 * target_coarse := by
  simp only []
  have hf : MeasureTheory.Integrable f := by
    by_contra h; exact absurd hf_int (by rw [MeasureTheory.integral_undef h]; norm_num)
  have h_cum := cum_mass_fine_eq_coarse f n hn hf k hk
  have h_total : ∑ j : Fin (2 * (2 * n)), bin_masses f (2 * n) j =
      ∑ j : Fin (2 * n), bin_masses f n j := by
    have h1 := sum_bin_masses_eq_one n hn f hf_supp hf_int
    have h2 := sum_bin_masses_eq_one (2 * n) (by omega) f hf_supp hf_int
    linarith
  rw [h_cum, h_total]
  ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Canonical Cumulative Distribution: Fine vs Coarse
-- ═══════════════════════════════════════════════════════════════════════════════

/-- D_fine(2k) is between 2*D_coarse(k) and 2*D_coarse(k)+1. -/
lemma ccd_fine_vs_coarse (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (k : ℕ) (hk : k ≤ 2 * n) :
    2 * canonical_cumulative_distribution f n m k ≤
      canonical_cumulative_distribution f (2 * n) m (2 * k) ∧
    canonical_cumulative_distribution f (2 * n) m (2 * k) ≤
      2 * canonical_cumulative_distribution f n m k + 1 := by
  unfold canonical_cumulative_distribution; simp only []
  have hf : MeasureTheory.Integrable f := by
    by_contra h; exact absurd hf_int (by rw [MeasureTheory.integral_undef h]; norm_num)
  have h_cum := cum_mass_fine_eq_coarse f n hn hf k hk
  have h_total : ∑ j : Fin (2 * (2 * n)), bin_masses f (2 * n) j =
      ∑ j : Fin (2 * n), bin_masses f n j := by
    linarith [sum_bin_masses_eq_one n hn f hf_supp hf_int,
              sum_bin_masses_eq_one (2 * n) (by omega) f hf_supp hf_int]
  rw [h_cum, h_total]
  set tc := (∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) /
    (∑ j, bin_masses f n j) with htc_def
  have htc_nn : 0 ≤ tc := by
    apply div_nonneg
    · exact Finset.sum_nonneg fun j _ => by
        split_ifs <;> [exact bin_masses_nonneg f hf_nonneg n j; exact le_refl 0]
    · exact Finset.sum_nonneg fun j _ => bin_masses_nonneg f hf_nonneg n j
  have h_fine_eq : tc * (↑(4 * (2 * n) * m) : ℝ) = 2 * (tc * (↑(4 * n * m) : ℝ)) := by
    push_cast; ring
  rw [h_fine_eq]
  exact floor_double_natAbs_bounds (tc * ↑(4 * n * m)) (mul_nonneg htc_nn (by positivity))

-- ═══════════════════════════════════════════════════════════════════════════════
-- Main Bridge Theorem
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Canonical discretization at doubled resolution produces a valid child. -/
theorem refinement_preserves_discretization
    (f : ℝ → ℝ) (n_half m : ℕ) (hn : n_half > 0) (hm : m > 0)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    is_valid_child n_half
      (canonical_discretization f n_half m)
      (canonical_discretization f (2 * n_half) m) := by
  unfold is_valid_child
  constructor
  · -- Part 1: Total mass doubles
    have h_mass_nz_c : ∑ j : Fin (2 * n_half), bin_masses f n_half j ≠ 0 := by
      rw [sum_bin_masses_eq_one n_half hn f hf_supp hf_int]; exact one_ne_zero
    have h_mass_nz_f : ∑ j : Fin (2 * (2 * n_half)), bin_masses f (2 * n_half) j ≠ 0 := by
      rw [sum_bin_masses_eq_one (2 * n_half) (by omega) f hf_supp hf_int]; exact one_ne_zero
    rw [canonical_discretization_sum_eq_m f (2 * n_half) m (by omega) hm h_mass_nz_f hf_nonneg]
    rw [canonical_discretization_sum_eq_m f n_half m hn hm h_mass_nz_c hf_nonneg]
    ring
  · -- Part 2: Each bin pair sum deviates by at most ±1
    intro i
    simp only []
    have h_mass_nz_c : ∑ j : Fin (2 * n_half), bin_masses f n_half j ≠ 0 := by
      rw [sum_bin_masses_eq_one n_half hn f hf_supp hf_int]; exact one_ne_zero
    have h_mass_nz_f : ∑ j : Fin (2 * (2 * n_half)), bin_masses f (2 * n_half) j ≠ 0 := by
      rw [sum_bin_masses_eq_one (2 * n_half) (by omega) f hf_supp hf_int]; exact one_ne_zero
    have hDc_2n := canonical_cumulative_distribution_2n f n_half m hn hm h_mass_nz_c
    have hDf_4n := canonical_cumulative_distribution_2n f (2 * n_half) m (by omega) hm h_mass_nz_f
    -- Rewrite bins as CCD differences
    have h_parent := canonical_discretization_eq_diff f n_half m hDc_2n i
    have h_child0 := canonical_discretization_eq_diff f (2 * n_half) m hDf_4n
      ⟨2 * i.1, by omega⟩
    have h_child1 := canonical_discretization_eq_diff f (2 * n_half) m hDf_4n
      ⟨2 * i.1 + 1, by omega⟩
    -- Normalize index: 2*i.1+1+1 = 2*(i.1+1)
    have h_idx : 2 * i.1 + 1 + 1 = 2 * (i.1 + 1) := by omega
    rw [h_idx] at h_child1
    rw [h_child0, h_child1, h_parent]
    -- Monotonicity
    have hDf_mono := canonical_cumulative_distribution_mono f hf_nonneg (2 * n_half) m
    have hDc_mono := canonical_cumulative_distribution_mono f hf_nonneg n_half m
    -- CCD bounds
    have hbds_i := ccd_fine_vs_coarse f n_half m hn hm hf_nonneg hf_supp hf_int i.1 (by omega)
    have hbds_i1 := ccd_fine_vs_coarse f n_half m hn hm hf_nonneg hf_supp hf_int (i.1 + 1) (by omega)
    -- Monotonicity facts
    have hm1 : canonical_cumulative_distribution f (2 * n_half) m (2 * i.1) ≤
               canonical_cumulative_distribution f (2 * n_half) m (2 * i.1 + 1) :=
      hDf_mono (by omega)
    have hm2 : canonical_cumulative_distribution f (2 * n_half) m (2 * i.1 + 1) ≤
               canonical_cumulative_distribution f (2 * n_half) m (2 * (i.1 + 1)) :=
      hDf_mono (by omega)
    have hm3 : canonical_cumulative_distribution f n_half m i.1 ≤
               canonical_cumulative_distribution f n_half m (i.1 + 1) :=
      hDc_mono (by omega)
    obtain ⟨hfi_lo, hfi_hi⟩ := hbds_i
    obtain ⟨hfi1_lo, hfi1_hi⟩ := hbds_i1
    -- pair_sum = D_f(2*(i+1)) - D_f(2*i)
    -- 2*parent = 2*(D_c(i+1) - D_c(i))
    -- D_f(2*(i+1)) ∈ [2*D_c(i+1), 2*D_c(i+1)+1]
    -- D_f(2*i) ∈ [2*D_c(i), 2*D_c(i)+1]
    -- pair_sum ∈ [2*parent-1, 2*parent+1]
    -- Normalize Fin.val ⟨k, _⟩ → k so omega sees equal CCD arguments
    dsimp only [] at *
    omega

end -- noncomputable section
