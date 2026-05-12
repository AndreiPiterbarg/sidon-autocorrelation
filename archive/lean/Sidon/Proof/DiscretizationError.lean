/-
Sidon Autocorrelation Project — Discretization Error and Correction Terms

Discretization error bound, contributing bins characterization,
correction term bound, and dynamic threshold soundness.
(Claims 1.2, 1.3, 1.4)
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.TestValueBounds

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
-- Discretization Error and Correction Terms (Section 18c)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Target cumulative mass (before flooring). -/
noncomputable def target_cum_mass (f : ℝ → ℝ) (n m : ℕ) (k : ℕ) : ℝ :=
  let S := 4 * n * m
  let masses := bin_masses f n
  let total_mass := ∑ j, masses j
  let cum_mass := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
  (cum_mass) / total_mass * S

-- Helper lemmas for cumulative mass bounds
private lemma target_cum_mass_eq (n m : ℕ) (f : ℝ → ℝ)
    (hμ_sum : ∑ j : Fin (2 * n), bin_masses f n j = 1) (k : ℕ) :
    target_cum_mass f n m k = (∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) * (4 * ↑n * ↑m) := by
  unfold target_cum_mass; simp only []
  rw [hμ_sum, div_one]
  congr 1; push_cast; ring

private lemma target_cum_mass_nonneg (n m : ℕ) (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hμ_sum : ∑ j : Fin (2 * n), bin_masses f n j = 1) (k : ℕ) :
    0 ≤ target_cum_mass f n m k := by
  rw [target_cum_mass_eq n m f hμ_sum k]
  apply mul_nonneg
  · exact Finset.sum_nonneg fun j _ => by split_ifs <;> [exact bin_masses_nonneg f hf_nonneg n j; exact le_refl 0]
  · positivity

private lemma ccd_eq_floor_natAbs (n m : ℕ) (f : ℝ → ℝ) (k : ℕ) :
    canonical_cumulative_distribution f n m k = ⌊target_cum_mass f n m k⌋.natAbs := by
  unfold canonical_cumulative_distribution target_cum_mass; rfl

private lemma ccd_cast_eq (n m : ℕ) (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hμ_sum : ∑ j : Fin (2 * n), bin_masses f n j = 1) (k : ℕ) :
    (canonical_cumulative_distribution f n m k : ℝ) = ⌊target_cum_mass f n m k⌋ := by
  rw [ccd_eq_floor_natAbs]
  have h_nn : (0 : ℤ) ≤ ⌊target_cum_mass f n m k⌋ :=
    Int.floor_nonneg.mpr (target_cum_mass_nonneg n m f hf_nonneg hμ_sum k)
  rw [Nat.cast_natAbs, Int.cast_abs, abs_of_nonneg (Int.cast_nonneg.mpr h_nn)]

private lemma partial_sum_discretization (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (k : ℕ) (hk : k ≤ 2 * n) :
    (∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ,
      (canonical_discretization f n m i : ℝ)) =
    (canonical_cumulative_distribution f n m k : ℝ) := by
  have hμ_sum := sum_bin_masses_eq_one n hn f hf_supp hf_int
  have h_mass_nz : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0 := by rw [hμ_sum]; exact one_ne_zero
  have h_D_2n := canonical_cumulative_distribution_2n f n m hn hm h_mass_nz
  have h_mono := canonical_cumulative_distribution_mono f hf_nonneg n m
  -- Each c_i = D(i+1) - D(i)
  have h_eq_diff : ∀ i : Fin (2 * n), canonical_discretization f n m i =
      canonical_cumulative_distribution f n m (i.1 + 1) - canonical_cumulative_distribution f n m i.1 :=
    canonical_discretization_eq_diff f n m h_D_2n
  -- Filter {i : Fin(2n) | i.val < k} is equivalent to Finset.range k mapped into Fin(2n)
  have h_filter_eq : Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ =
      Finset.image (fun i : Fin k => ⟨i.val, by omega⟩) Finset.univ := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · intro h; exact Finset.mem_image.mpr ⟨⟨i.val, h⟩, Finset.mem_univ _, rfl⟩
    · intro h; obtain ⟨j, -, rfl⟩ := Finset.mem_image.mp h; exact j.isLt
  rw [h_filter_eq]
  rw [Finset.sum_image (by intro a _ b _ hab; ext; simpa using hab)]
  -- Now we sum D(i+1) - D(i) over Fin k, telescoping to D(k) - D(0)
  conv_lhs =>
    arg 2; ext i
    rw [h_eq_diff ⟨i.val, by omega⟩]
  -- Sum of (D(i+1) - D(i)) as ℝ
  have h_sum_telescope : ∑ i : Fin k, ((canonical_cumulative_distribution f n m (i.val + 1) : ℝ) -
      (canonical_cumulative_distribution f n m i.val : ℝ)) =
      (canonical_cumulative_distribution f n m k : ℝ) - (canonical_cumulative_distribution f n m 0 : ℝ) := by
    rw [Fin.sum_univ_eq_sum_range (fun i => (canonical_cumulative_distribution f n m (i + 1) : ℝ) -
        (canonical_cumulative_distribution f n m i : ℝ)) k]
    exact Finset.sum_range_sub (fun i => (canonical_cumulative_distribution f n m i : ℝ)) k
  -- D is monotone, so D(i+1) - D(i) ≥ 0 in ℕ, and (ℕ cast to ℝ) agrees with ℝ subtraction
  have h_cast_sub : ∀ i : Fin k,
      (↑(canonical_cumulative_distribution f n m (i.val + 1) - canonical_cumulative_distribution f n m i.val) : ℝ) =
      (canonical_cumulative_distribution f n m (i.val + 1) : ℝ) -
      (canonical_cumulative_distribution f n m i.val : ℝ) := by
    intro i
    rw [Nat.cast_sub (h_mono (Nat.le_succ _))]
  simp_rw [h_cast_sub]
  rw [h_sum_telescope]
  rw [canonical_cumulative_distribution_zero]
  simp

private lemma partial_sum_mu (n : ℕ) (f : ℝ → ℝ) (k : ℕ) :
    ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ, bin_masses f n i =
    ∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0 := by
  rw [Finset.sum_filter]

-- ═══════════════════════════════════════════════════════════════════════════════
-- C&S Lemma 3: Tighter per-window bound (no 4n/ℓ factor)
--
-- Cloninger & Steinerberger, arXiv:1403.7988, Lemma 3:
--   (g*g)(x) ≤ (f*f)(x) + 2/m + 1/m²  (POINTWISE for all x)
--
-- Since test values are averages of (g*g) over windows, and averaging
-- preserves pointwise inequalities:
--   TV_g(ℓ,s) ≤ TV_f(ℓ,s) + 2/m + 1/m²
--   TV_g(ℓ,s) ≤ ||f*f||_∞ + 2/m + 1/m²
--
-- This is strictly tighter than the (4n/ℓ)·(1/m²+2W/m) bound above.
-- The code now uses this tighter bound for pruning.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **C&S Lemma 3 per-window bound (axiom).**

    Cloninger & Steinerberger (2017), Lemma 3 (arXiv:1403.7988):
    The pointwise discretization error satisfies
      |(g*g)(x) - (f*f)(x)| ≤ 2/m + 1/m²
    where g is the step function with heights c_i/m on the fine grid.

    Fine grid (C&S B_{n,m}): canonical_discretization rounds to S = 4nm quanta, giving
    heights a_i = c_i/m that are multiples of 1/m.  The rounding error
    per bin satisfies |a_i - ideal_i| ≤ 1/m (C&S Lemma 2), hence:
      |(g*g)(x) - (f*f)(x)| ≤ 2·||ε||_∞ + ||ε||_∞² ≤ 2/m + 1/m²

    Since test values are window averages of the autoconvolution, and
    averaging preserves pointwise bounds:
      TV_discrete(c, ℓ, s) - TV_continuous(f, ℓ, s) ≤ 2/m + 1/m²

    This is the ONLY mathematical axiom in the formalization. It encodes
    a published, peer-reviewed result. Full formalization would require
    ~200-300 lines of piecewise integration (MeasureTheory.integral_indicator)
    and the correspondence between continuous and discrete autoconvolution
    at grid points, which exceeds current Mathlib infrastructure.

    CPU code equivalent: pruning.py correction(m) = 2/m + 1/m².
    CPU threshold flag: --use_flat_threshold ensures the cascade uses
    this exact correction (2m+1 in integer units). -/
axiom cs_lemma3_per_window (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    test_value n m (canonical_discretization f n m) ℓ s_lo - test_value_continuous n f ℓ s_lo ≤
      2 / m + 1 / m ^ 2

/-- C&S Lemma 3 correction bound: R(f) ≥ TV(c,ℓ,s) - (2/m + 1/m²).
    Uses the pointwise bound, strictly tighter than correction_term_bound. -/
theorem correction_term_bound_cs (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    autoconvolution_ratio f ≥
      test_value n m (canonical_discretization f n m) ℓ s_lo - (2 / m + 1 / m ^ 2) := by
  have h_cont : autoconvolution_ratio f ≥ test_value_continuous n f ℓ s_lo :=
    continuous_test_value_le_ratio n hn f hf_nonneg hf_supp hf_int h_conv_fin ℓ s_lo hℓ
  have h_disc := cs_lemma3_per_window n m hn hm f hf_nonneg hf_supp hf_int ℓ s_lo hℓ
  linarith

/-- Dynamic threshold soundness using C&S Lemma 3 (tighter bound).
    The correction is 2/m + 1/m² independent of window length — no (4n/ℓ) factor.
    This is what the code now uses for pruning. -/
theorem dynamic_threshold_sound_cs (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (_hct : 0 < c_target)
    (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (h_exceeds : test_value n m c ℓ s_lo > c_target + 2 / m + 1 / m ^ 2) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f = 1 →
      MeasureTheory.eLpNorm (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ →
      canonical_discretization f n m = c →
      autoconvolution_ratio f ≥ c_target := by
  intro f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  have hbound := correction_term_bound_cs n m hn hm f hf_nonneg hf_supp hf_int h_conv_fin ℓ s_lo hℓ
  rw [hdisc] at hbound
  linarith

end -- noncomputable section
