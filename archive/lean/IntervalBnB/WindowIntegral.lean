/-
IntervalBnB — Lemma 1.3 (window integral lower bound).

For any admissible `f` and any window `W`,
    ∫_{I_W} (f*f)(t) dt  ≥  ∑_{i,j : i+j ∈ K_W} μ_i μ_j.

The proof has two measure-theoretic ingredients:

  (i)  Fubini converts the LHS to
       ∫∫ f(x) f(y) · 1[x+y ∈ I_W] dx dy.

  (ii) Split the (x,y) integral over the d×d bin pairs (B_i, B_j).
       For pairs with i+j ∈ K_W, Lemma 1.1(a) gives indicator = 1 on the
       whole rectangle, so the contribution is μ_i μ_j. Drop all other
       (nonneg) pair contributions.

The combinatorial bin-decomposition step (ii) is fully proved here.
The single Fubini identity (i) is captured as an atomic
measure-theoretic step in the helper `window_integral_pointwise_lb`;
see the comment there for why this is genuinely atomic in the current
Mathlib.
-/

import IntervalBnB.Defs
import IntervalBnB.PairSumGeometry
import IntervalBnB.Averaging

set_option linter.mathlibStandardSet false
set_option autoImplicit false
set_option relaxedAutoImplicit false

open scoped BigOperators
open scoped Classical
open MeasureTheory

noncomputable section

namespace IntervalBnB

variable {d : ℕ}

/-!
### Measurability of the bin sets.
-/

lemma bin_measurable (d : ℕ) (i : Fin d) : MeasurableSet (Bin d i) := by
  unfold Bin; exact measurableSet_Ico

/-!
### Bin indices are pairwise disjoint.
-/

lemma bin_disjoint (d : ℕ) {i j : Fin d} (hij : i ≠ j) :
    Disjoint (Bin d i) (Bin d j) := by
  wlog h : i.val < j.val generalizing i j
  · have hji : j.val < i.val := by
      have : i.val ≠ j.val := fun eq => hij (Fin.ext eq)
      omega
    exact (this hij.symm hji).symm
  have hd_pos : 0 < d := by
    have := i.isLt; omega
  have hd_R : (0 : ℝ) < (2 * d : ℝ) := by
    have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd_pos
    linarith
  rw [Set.disjoint_iff]
  rintro x ⟨⟨_, hx_i⟩, ⟨hx_j, _⟩⟩
  have h_i1_le_j : i.val + 1 ≤ j.val := h
  have h_cast : ((i.val : ℝ) + 1) ≤ (j.val : ℝ) := by
    have : ((i.val + 1 : ℕ) : ℝ) ≤ ((j.val : ℕ) : ℝ) := by exact_mod_cast h_i1_le_j
    push_cast at this; linarith
  have h_div : ((i.val : ℝ) + 1) / (2*d) ≤ (j.val : ℝ) / (2*d) :=
    div_le_div_of_nonneg_right h_cast (le_of_lt hd_R)
  linarith

/-- At most one bin contains any given `x`. -/
lemma bin_unique (d : ℕ) {x : ℝ} {i j : Fin d}
    (hxi : x ∈ Bin d i) (hxj : x ∈ Bin d j) : i = j := by
  by_contra hij
  have hd : Disjoint (Bin d i) (Bin d j) := bin_disjoint d hij
  exact (Set.disjoint_iff.mp hd) ⟨hxi, hxj⟩

/-!
### Bin mass in terms of set integrals.
-/

lemma bin_mass_eq_setIntegral (d : ℕ) (f : ℝ → ℝ) (i : Fin d) :
    bin_mass d f i = ∫ x in Bin d i, f x ∂volume := by
  unfold bin_mass
  exact integral_indicator (bin_measurable d i)

/-!
### Bin cover of `[-1/4, 1/4)`.
-/

/-- Every `x ∈ [-1/4, 1/4)` is in exactly one `Bin d i`. -/
lemma bin_cover (d : ℕ) (hd : 0 < d) {x : ℝ}
    (hx : x ∈ Set.Ico (-(1 : ℝ)/4) (1/4)) :
    ∃ i : Fin d, x ∈ Bin d i := by
  rcases hx with ⟨hx1, hx2⟩
  have hd_R : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  have h2d_R : (0 : ℝ) < (2*d : ℝ) := by linarith
  set τ : ℝ := 2 * d * (x + 1/4) with hτ
  have hτ_nn : 0 ≤ τ := by
    have : 0 ≤ x + 1/4 := by linarith
    show 0 ≤ 2 * d * (x + 1/4); positivity
  have hτ_lt : τ < d := by
    show 2 * d * (x + 1/4) < d
    have : x + 1/4 < 1/2 := by linarith
    have hmul := mul_lt_mul_of_pos_left this h2d_R
    have hrw : (2 * d : ℝ) * (1/2) = d := by ring
    linarith
  set n : ℕ := Nat.floor τ with hn
  have hn_lt : n < d := by
    show Nat.floor τ < d
    have h1 : (Nat.floor τ : ℝ) ≤ τ := Nat.floor_le hτ_nn
    have h2 : τ < d := hτ_lt
    have : (Nat.floor τ : ℝ) < (d : ℝ) := lt_of_le_of_lt h1 h2
    exact_mod_cast this
  refine ⟨⟨n, hn_lt⟩, ?_, ?_⟩
  · show -(1 : ℝ)/4 + (n : ℝ)/(2*d) ≤ x
    have h_floor_le : (n : ℝ) ≤ τ := Nat.floor_le hτ_nn
    have hdiv : (n : ℝ)/(2*d) ≤ (x + 1/4) := by
      rw [div_le_iff₀ h2d_R]
      have : (n : ℝ) ≤ 2 * d * (x + 1/4) := h_floor_le
      linarith
    linarith
  · show x < -(1 : ℝ)/4 + ((n : ℝ) + 1)/(2*d)
    have h_lt_floor1 : τ < (n : ℝ) + 1 := by
      have := Nat.lt_floor_add_one τ
      push_cast at this; exact this
    have hdiv : (x + 1/4) < ((n : ℝ) + 1)/(2*d) := by
      rw [lt_div_iff₀ h2d_R]
      have : 2 * d * (x + 1/4) < (n : ℝ) + 1 := h_lt_floor1
      linarith
    linarith

/-!
### Bin sum equals 1.

`∑ i : Fin d, bin_mass d f i = 1` for admissible `f`.  This is an
elementary consequence of:
  (i) `bin_mass d f i = ∫ (indicator B_i · f)`,
  (ii) `∑ i, indicator B_i x = indicator ([-1/4, 1/4)) x` (disjoint cover),
  (iii) `∫ f · 1_{[-1/4,1/4)} = ∫ f = 1` (since `supp f ⊆ [-1/4, 1/4]`
        and `{1/4}` is a null set).
-/

/-- The sum of bin indicators equals the indicator of `[-1/4, 1/4)`. -/
lemma sum_bin_indicator_eq (d : ℕ) (hd : 0 < d) (x : ℝ) :
    (∑ i : Fin d, Set.indicator (Bin d i) (fun _ => (1 : ℝ)) x) =
      Set.indicator (Set.Ico (-(1 : ℝ)/4) (1/4)) (fun _ => (1 : ℝ)) x := by
  by_cases hx : x ∈ Set.Ico (-(1 : ℝ)/4) (1/4)
  · rw [Set.indicator_of_mem hx]
    obtain ⟨i₀, hi₀⟩ := bin_cover d hd hx
    rw [Finset.sum_eq_single i₀]
    · exact Set.indicator_of_mem hi₀ _
    · intro j _ hj
      have hx_not_j : x ∉ Bin d j := fun hj_mem =>
        hj (bin_unique d hj_mem hi₀)
      exact Set.indicator_of_notMem hx_not_j _
    · simp
  · rw [Set.indicator_of_notMem hx]
    apply Finset.sum_eq_zero
    intro i _
    apply Set.indicator_of_notMem
    -- x ∉ Bin d i because Bin d i ⊆ Ico(-1/4, 1/4).
    intro hxi
    apply hx
    -- Bin d i = Ico(-1/4 + i/(2d), -1/4 + (i+1)/(2d)) ⊆ Ico(-1/4, 1/4)
    have hd_R : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
    have h2d_R : (0 : ℝ) < (2*d : ℝ) := by linarith
    have hi_lt : i.val < d := i.isLt
    have hi_cast : (i.val : ℝ) < (d : ℝ) := by exact_mod_cast hi_lt
    rcases hxi with ⟨hxi1, hxi2⟩
    refine ⟨?_, ?_⟩
    · -- -1/4 ≤ x
      have hnn : (0 : ℝ) ≤ (i.val : ℝ) / (2*d) := by
        apply div_nonneg
        · exact_mod_cast Nat.zero_le _
        · linarith
      linarith
    · -- x < 1/4
      -- x < -1/4 + (i+1)/(2d) ≤ -1/4 + d/(2d) = -1/4 + 1/2 = 1/4
      have : (i.val : ℝ) + 1 ≤ d := by
        have : i.val + 1 ≤ d := hi_lt
        have hh : ((i.val + 1 : ℕ) : ℝ) ≤ ((d : ℕ) : ℝ) := by exact_mod_cast this
        push_cast at hh; linarith
      have hd_div : ((i.val : ℝ) + 1) / (2*d) ≤ (d : ℝ) / (2*d) :=
        div_le_div_of_nonneg_right this (le_of_lt h2d_R)
      have hrw : (d : ℝ) / (2*d) = 1/2 := by
        field_simp
      linarith

/-- Disjoint cover: each bin is a subset of `[-1/4, 1/4)`. -/
lemma bin_subset_ref (d : ℕ) (hd : 0 < d) (i : Fin d) :
    Bin d i ⊆ Set.Ico (-(1 : ℝ)/4) (1/4) := by
  intro x hx
  have hs := sum_bin_indicator_eq d hd x
  -- If x ∈ Bin d i, then the sum is ≥ 1 (the i-th term is 1).
  -- This forces the indicator of the full interval at x to be 1,
  -- i.e. x ∈ [-1/4, 1/4).
  by_contra hnot
  rw [Set.indicator_of_notMem hnot] at hs
  have h_i_term : Set.indicator (Bin d i) (fun _ => (1 : ℝ)) x = 1 :=
    Set.indicator_of_mem hx _
  have h_sum_pos :
      (1 : ℝ) ≤ ∑ j : Fin d, Set.indicator (Bin d j) (fun _ => (1 : ℝ)) x := by
    have h_i_in_sum :
        Set.indicator (Bin d i) (fun _ => (1 : ℝ)) x
          ≤ ∑ j : Fin d, Set.indicator (Bin d j) (fun _ => (1 : ℝ)) x := by
      apply Finset.single_le_sum (f := fun j => Set.indicator (Bin d j) (fun _ => (1 : ℝ)) x)
      · intro j _
        by_cases hx_j : x ∈ Bin d j
        · rw [Set.indicator_of_mem hx_j]; exact zero_le_one
        · rw [Set.indicator_of_notMem hx_j]
      · exact Finset.mem_univ i
    linarith [h_i_term]
  rw [hs] at h_sum_pos; linarith

/-- The sum of bin-indicator-weighted `f` values equals `f` except at
    the single point `1/4` (a null set), hence equals `f` a.e. -/
lemma sum_bin_indicator_mul_f_ae (d : ℕ) (hd : 0 < d) (f : ℝ → ℝ)
    (hf : Admissible f) :
    (fun x => ∑ i : Fin d, Set.indicator (Bin d i) f x) =ᵐ[volume] f := by
  have h_set : ∀ x, x ≠ (1/4 : ℝ) →
      (∑ i : Fin d, Set.indicator (Bin d i) f x) = f x := by
    intro x hx_ne
    by_cases hx_in : x ∈ Set.Ico (-(1 : ℝ)/4) (1/4)
    · obtain ⟨i₀, hi₀⟩ := bin_cover d hd hx_in
      rw [Finset.sum_eq_single i₀]
      · exact Set.indicator_of_mem hi₀ _
      · intro j _ hj
        have hx_not_j : x ∉ Bin d j := fun hj_mem =>
          hj (bin_unique d hj_mem hi₀)
        exact Set.indicator_of_notMem hx_not_j f
      · intro h; exact absurd (Finset.mem_univ _) h
    · have h_sum_zero : ∑ i : Fin d, Set.indicator (Bin d i) f x = 0 := by
        apply Finset.sum_eq_zero
        intro i _
        apply Set.indicator_of_notMem
        intro hxi
        exact hx_in (bin_subset_ref d hd i hxi)
      rw [h_sum_zero]
      by_contra hfx
      have hx_mem : x ∈ Set.Icc (-(1 : ℝ)/4) (1/4) := hf.support x (Ne.symm hfx)
      rcases hx_mem with ⟨h1, h2⟩
      exact hx_in ⟨h1, lt_of_le_of_ne h2 hx_ne⟩
  -- Convert to ae equality using ae_iff.
  rw [Filter.EventuallyEq, MeasureTheory.ae_iff]
  -- Goal: volume {x | ¬ (... = f x)} = 0, reduce to volume {1/4} = 0.
  have h_subset : {x : ℝ | ¬ ((∑ i : Fin d, Set.indicator (Bin d i) f x) = f x)}
                    ⊆ {(1/4 : ℝ)} := by
    intro x hx
    by_contra hne_singleton
    have hx_ne : x ≠ (1/4 : ℝ) := hne_singleton
    exact hx (h_set x hx_ne)
  exact le_antisymm
    (le_trans (measure_mono h_subset) (le_of_eq Real.volume_singleton))
    (zero_le _)

-- (The helper `integral_f_Ico_eq` is not needed; the a.e. equality is
-- proved above via `sum_bin_indicator_mul_f_ae`.)

/-- **Lemma C**: the bin masses sum to 1. -/
theorem bin_sum_eq_one (d : ℕ) (hd : 0 < d) (f : ℝ → ℝ) (hf : Admissible f) :
    ∑ i : Fin d, bin_mass d f i = 1 := by
  -- Step 1: ∑ i, ∫ indicator B_i f = ∫ ∑ i, indicator B_i f
  have h_step1 :
      ∑ i : Fin d, bin_mass d f i
        = ∫ x, ∑ i : Fin d, Set.indicator (Bin d i) f x ∂volume := by
    unfold bin_mass
    rw [integral_finset_sum]
    intro i _
    exact hf.integrable.indicator (bin_measurable d i)
  rw [h_step1]
  -- Step 2: the integrand = f a.e. (by sum_bin_indicator_mul_f_ae)
  have h_ae : (fun x => ∑ i : Fin d, Set.indicator (Bin d i) f x) =ᵐ[volume] f :=
    sum_bin_indicator_mul_f_ae d hd f hf
  rw [MeasureTheory.integral_congr_ae h_ae]
  exact hf.integral_one

/-!
### Lemma 1.3 (window integral lower bound).

Full proof via 2D product integration on `volume.prod volume`.

Two ingredients:

  1. **Fubini on the convolution.** `(f * f)(t) = ∫ x, f(x) f(t-x) dx`
     by definition.  The Fubini identity + translation invariance give
         `∫ t, 1[t∈I] · (f*f)(t) dt = ∫ p, f(p.1) f(p.2) 1[p.1+p.2∈I] d(vol × vol)`.

  2. **Bin decomposition + drop boundary.**  Restrict to each bin
     rectangle `B_i × B_j` (they are pairwise disjoint as products).
     For pairs `(i,j)` with `i+j ∈ K_W`, Lemma 1.1(a) gives
     `1[x+y∈I_W] = 1` on `B_i × B_j`, so `setIntegral_prod_mul` yields
     the contribution `μ_i μ_j`.  Since the integrand is nonneg and
     the kept rectangles are disjoint subsets of the plane, their
     contributions sum to something ≤ the total integral.
-/

section WindowIntegralCore

variable (f : ℝ → ℝ)

/-- Measurability of the convolution integrand `(t, x) ↦ f(x) · f(t-x)`. -/
private lemma autoconv_integrand_measurable (hf : Admissible f) :
    Measurable (fun p : ℝ × ℝ => f p.2 * f (p.1 - p.2)) := by
  have h1 : Measurable (fun p : ℝ × ℝ => f p.2) := hf.measurable.comp measurable_snd
  have h2 : Measurable (fun p : ℝ × ℝ => f (p.1 - p.2)) :=
    hf.measurable.comp (measurable_fst.sub measurable_snd)
  exact h1.mul h2

/-- `autoconv f t = ∫ x, f x * f (t - x) dx`. -/
private lemma autoconv_eq_integral (t : ℝ) :
    autoconv f t = ∫ x, f x * f (t - x) ∂MeasureTheory.volume := by
  unfold autoconv
  rw [MeasureTheory.convolution_def]
  simp [ContinuousLinearMap.mul_apply']

/-- Window interval is measurable. -/
private lemma window_interval_measurableSet (W : Window d) :
    MeasurableSet (window_interval W) := measurableSet_Icc

/-- The 2D integrand `f(x) * f(t-x)` is integrable on `ℝ × ℝ`. -/
private lemma convolution_integrand_integrable (hf : Admissible f) :
    MeasureTheory.Integrable
      (fun p : ℝ × ℝ => f p.2 * f (p.1 - p.2))
      (MeasureTheory.volume.prod MeasureTheory.volume) := by
  have h := MeasureTheory.Integrable.convolution_integrand
    (ContinuousLinearMap.mul ℝ ℝ) hf.integrable hf.integrable
  -- h : Integrable (fun p => (mul ℝ ℝ) (f p.2) (f (p.1 - p.2)))
  -- Rewrite (mul ℝ ℝ) (f p.2) (f (p.1 - p.2)) = f p.2 * f (p.1 - p.2).
  simpa [ContinuousLinearMap.mul_apply'] using h

/-- The 3D integrand `(t, x) ↦ f(x) f(t-x) · 1[t ∈ I_W]` is integrable. -/
private lemma windowed_convolution_integrand_integrable
    (hf : Admissible f) (I : Set ℝ) (hI : MeasurableSet I) :
    MeasureTheory.Integrable
      (fun p : ℝ × ℝ => Set.indicator I (fun _ => (1 : ℝ)) p.1 *
        (f p.2 * f (p.1 - p.2)))
      (MeasureTheory.volume.prod MeasureTheory.volume) := by
  -- Pointwise bound by |f(x) f(t-x)|, integrable. Use Integrable.mono.
  have hbase := convolution_integrand_integrable f hf
  refine hbase.mono ?_ ?_
  · -- AEStronglyMeasurable
    have h_ind : Measurable
        (fun p : ℝ × ℝ => Set.indicator I (fun _ => (1 : ℝ)) p.1) := by
      have : Measurable (fun t : ℝ => Set.indicator I (fun _ => (1 : ℝ)) t) :=
        (measurable_const.indicator hI)
      exact this.comp measurable_fst
    have h_core : Measurable (fun p : ℝ × ℝ => f p.2 * f (p.1 - p.2)) :=
      autoconv_integrand_measurable f hf
    exact (h_ind.mul h_core).aestronglyMeasurable
  · -- ‖ind · core‖ ≤ ‖core‖
    refine Filter.Eventually.of_forall (fun p => ?_)
    show ‖Set.indicator I (fun _ => (1 : ℝ)) p.1 * (f p.2 * f (p.1 - p.2))‖ ≤
         ‖f p.2 * f (p.1 - p.2)‖
    by_cases hp : p.1 ∈ I
    · rw [Set.indicator_of_mem hp, one_mul]
    · rw [Set.indicator_of_notMem hp, zero_mul, norm_zero]
      exact norm_nonneg _

/-- **Fubini step.** The window integral equals the 2D product integral.
    `∫ t, 1[t∈I_W] · (f*f)(t) dt = ∫ x, ∫ y, f(x) * f(y) * 1[x+y∈I_W] dy dx`. -/
private lemma integral_indicator_autoconv_eq_prod
    (hf : Admissible f) (W : Window d) :
    MeasureTheory.integral MeasureTheory.volume
        (Set.indicator (window_interval W) (autoconv f))
      = ∫ x, ∫ y, f x * f y *
          Set.indicator (window_interval W) (fun _ => (1 : ℝ)) (x + y) ∂MeasureTheory.volume
          ∂MeasureTheory.volume := by
  set I : Set ℝ := window_interval W
  have hI : MeasurableSet I := window_interval_measurableSet W
  -- Rewrite LHS: ∫ (indicator I autoconv) = ∫ t, 1[t∈I] * autoconv f t
  have h_lhs :
      MeasureTheory.integral MeasureTheory.volume (Set.indicator I (autoconv f))
        = ∫ t, Set.indicator I (fun _ => (1 : ℝ)) t * autoconv f t ∂MeasureTheory.volume := by
    apply MeasureTheory.integral_congr_ae
    refine Filter.Eventually.of_forall (fun t => ?_)
    by_cases ht : t ∈ I
    · simp [Set.indicator_of_mem ht]
    · simp [Set.indicator_of_notMem ht]
  rw [h_lhs]
  -- Next: autoconv f t = ∫ x, f x * f (t - x) dx
  have h_conv : ∀ t, autoconv f t = ∫ x, f x * f (t - x) ∂MeasureTheory.volume :=
    autoconv_eq_integral f
  have h_rewrite :
      (fun t => Set.indicator I (fun _ => (1 : ℝ)) t * autoconv f t)
        = (fun t => Set.indicator I (fun _ => (1 : ℝ)) t *
            ∫ x, f x * f (t - x) ∂MeasureTheory.volume) := by
    funext t; rw [h_conv t]
  rw [h_rewrite]
  -- Pull constant out of inner integral:
  have h_pull : (fun t => Set.indicator I (fun _ => (1 : ℝ)) t *
        ∫ x, f x * f (t - x) ∂MeasureTheory.volume)
      = (fun t => ∫ x, Set.indicator I (fun _ => (1 : ℝ)) t * (f x * f (t - x))
          ∂MeasureTheory.volume) := by
    funext t
    rw [MeasureTheory.integral_const_mul]
  rw [h_pull]
  -- Now swap the integrals using Fubini
  -- View the integrand as uncurry g where g (t,x) = 1[t∈I] * (f x * f (t-x))
  set g : ℝ → ℝ → ℝ := fun t x =>
    Set.indicator I (fun _ => (1 : ℝ)) t * (f x * f (t - x))
  have h_int : MeasureTheory.Integrable
      (Function.uncurry g) (MeasureTheory.volume.prod MeasureTheory.volume) := by
    show MeasureTheory.Integrable (fun p : ℝ × ℝ =>
        Set.indicator I (fun _ => (1 : ℝ)) p.1 * (f p.2 * f (p.1 - p.2)))
      (MeasureTheory.volume.prod MeasureTheory.volume)
    exact windowed_convolution_integrand_integrable f hf I hI
  have h_swap :
      ∫ t, ∫ x, g t x ∂MeasureTheory.volume ∂MeasureTheory.volume
        = ∫ x, ∫ t, g t x ∂MeasureTheory.volume ∂MeasureTheory.volume :=
    MeasureTheory.integral_integral_swap h_int
  show ∫ t, ∫ x, g t x ∂MeasureTheory.volume ∂MeasureTheory.volume
      = ∫ x, ∫ y, f x * f y * Set.indicator I (fun _ => (1 : ℝ)) (x + y)
          ∂MeasureTheory.volume ∂MeasureTheory.volume
  rw [h_swap]
  -- Inner integral: change of variables t ↦ y = t - x, i.e. t = x + y.
  congr 1; funext x
  -- ∫ t, 1[t∈I] * (f x * f (t - x)) dt
  -- = f x * ∫ t, 1[t∈I] * f(t - x) dt
  have h_factor :
      (fun t => g t x) = (fun t => f x * (Set.indicator I (fun _ => (1 : ℝ)) t * f (t - x))) := by
    funext t
    show Set.indicator I (fun _ => (1 : ℝ)) t * (f x * f (t - x))
      = f x * (Set.indicator I (fun _ => (1 : ℝ)) t * f (t - x))
    ring
  rw [h_factor]
  rw [MeasureTheory.integral_const_mul]
  -- ∫ t, 1[t∈I] f(t-x) dt = ∫ y, 1[x+y∈I] f(y) dy via translation (t → y = t - x, so t = x + y)
  have h_trans :
      ∫ t, Set.indicator I (fun _ => (1 : ℝ)) t * f (t - x) ∂MeasureTheory.volume
        = ∫ y, Set.indicator I (fun _ => (1 : ℝ)) (y + x) * f y ∂MeasureTheory.volume := by
    -- The map t ↦ t - x is measure preserving; but easier: use integral_sub_right_eq_self
    -- ∫ t, h(t) dt = ∫ y, h(y + x) dy via integral_add_right_eq_self.
    -- integral_add_right_eq_self h x : ∫ y, h(y + x) dy = ∫ t, h(t) dt
    have := MeasureTheory.integral_add_right_eq_self (μ := MeasureTheory.volume)
      (fun t : ℝ => Set.indicator I (fun _ => (1 : ℝ)) t * f (t - x)) x
    -- this: ∫ y, h(y + x) dy = ∫ t, h(t) dt
    rw [← this]
    congr 1; funext y
    congr 2
    ring
  rw [h_trans]
  -- ∫ y, 1[y+x∈I] * f y dy = ∫ y, f x * f y * 1[x+y∈I] dy / f x  -- but we have f x extracted
  -- Actually we want: f x * ∫ y, 1[y+x∈I]*f(y) dy = ∫ y, f x * f y * 1[x+y∈I] dy
  rw [← MeasureTheory.integral_const_mul]
  congr 1; funext y
  have h_comm : y + x = x + y := by ring
  rw [h_comm]
  ring

/-- Nonnegativity of the 2D integrand on bin rectangles. -/
private lemma bin_rectangle_integrand_nonneg
    (hf : Admissible f) (W : Window d) (i j : Fin d) (x y : ℝ) :
    0 ≤ Set.indicator (Bin d i) f x * Set.indicator (Bin d j) f y *
        Set.indicator (window_interval W) (fun _ => (1 : ℝ)) (x + y) := by
  apply mul_nonneg
  · apply mul_nonneg
    · by_cases hx : x ∈ Bin d i
      · rw [Set.indicator_of_mem hx]; exact hf.nonneg x
      · rw [Set.indicator_of_notMem hx]
    · by_cases hy : y ∈ Bin d j
      · rw [Set.indicator_of_mem hy]; exact hf.nonneg y
      · rw [Set.indicator_of_notMem hy]
  · by_cases hxy : (x + y) ∈ window_interval W
    · rw [Set.indicator_of_mem hxy]
      exact zero_le_one
    · rw [Set.indicator_of_notMem hxy]

/-- On `Bin i × Bin j` with `i+j ∈ K_W`, the window indicator is identically 1. -/
private lemma window_indicator_on_good_bins
    (W : Window d) (i j : Fin d)
    (hij : (i.val + j.val) ∈ pair_sum_support W)
    (x y : ℝ) (hx : x ∈ Bin d i) (hy : y ∈ Bin d j) :
    Set.indicator (window_interval W) (fun _ => (1 : ℝ)) (x + y) = 1 := by
  rw [Set.indicator_of_mem (bin_pair_sum_subset_window W i j hij hx hy)]

/-- The `ℝ × ℝ` version of `integral_indicator_autoconv_eq_prod`:
    `∫ t, 1[t∈I_W] · (f*f)(t) dt = ∫ p in vol×vol, f(p.1) * f(p.2) * 1[p.1+p.2∈I_W]`. -/
private lemma integral_indicator_autoconv_eq_prod_integral
    (hf : Admissible f) (W : Window d) :
    MeasureTheory.integral MeasureTheory.volume
        (Set.indicator (window_interval W) (autoconv f))
      = ∫ p : ℝ × ℝ, f p.1 * f p.2 *
          Set.indicator (window_interval W) (fun _ => (1 : ℝ)) (p.1 + p.2)
          ∂(MeasureTheory.volume.prod MeasureTheory.volume) := by
  rw [integral_indicator_autoconv_eq_prod f hf W]
  -- iterated = product integral via integral_prod (need integrability)
  symm
  -- Integrability of the 2D integrand.
  set I : Set ℝ := window_interval W
  have hI : MeasurableSet I := window_interval_measurableSet W
  have h_int_2d : MeasureTheory.Integrable
      (fun p : ℝ × ℝ => f p.1 * f p.2 *
        Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2))
      (MeasureTheory.volume.prod MeasureTheory.volume) := by
    -- Bound by |f p.2 * f (p.1 + p.2 - p.2)| via a substitution? Simpler route:
    -- ∫⁻ |f(x)| |f(y)| d(vol × vol) = (∫⁻ |f|)² = ∫|f| ·  ∫|f| finite.
    -- Use the same convolution integrand lemma, with substitution.
    -- Easier: show integrability via `Integrable.prod_mul`.
    have h_prod : MeasureTheory.Integrable
        (fun p : ℝ × ℝ => f p.1 * f p.2)
        (MeasureTheory.volume.prod MeasureTheory.volume) := by
      have := hf.integrable.mul_prod hf.integrable
      simpa using this
    refine h_prod.mono ?_ ?_
    · have h1 : Measurable (fun p : ℝ × ℝ => f p.1 * f p.2) := by
        exact (hf.measurable.comp measurable_fst).mul (hf.measurable.comp measurable_snd)
      have h2 : Measurable (fun p : ℝ × ℝ =>
          Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)) := by
        have hm_add : Measurable (fun p : ℝ × ℝ => p.1 + p.2) := measurable_fst.add measurable_snd
        have hm_ind : Measurable (fun t : ℝ => Set.indicator I (fun _ => (1 : ℝ)) t) :=
          (measurable_const.indicator hI)
        exact hm_ind.comp hm_add
      exact (h1.mul h2).aestronglyMeasurable
    · refine Filter.Eventually.of_forall (fun p => ?_)
      show ‖f p.1 * f p.2 * Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)‖ ≤
           ‖f p.1 * f p.2‖
      by_cases hp : p.1 + p.2 ∈ I
      · rw [Set.indicator_of_mem hp, mul_one]
      · rw [Set.indicator_of_notMem hp, mul_zero, norm_zero]
        exact norm_nonneg _
  rw [MeasureTheory.integral_prod _ h_int_2d]

/-- Integral of `f(p.1) * f(p.2) * 1[p.1+p.2∈I]` on a rectangle `B_i × B_j`
    where `i+j ∈ K_W`, equals `μ_i * μ_j`.
    (Admissibility is not strictly needed here; only the measurability/geometry
    of the bins matters.) -/
private lemma setIntegral_good_rectangle
    (W : Window d) (i j : Fin d)
    (hij : (i.val + j.val) ∈ pair_sum_support W) :
    ∫ p in (Bin d i) ×ˢ (Bin d j), f p.1 * f p.2 *
        Set.indicator (window_interval W) (fun _ => (1 : ℝ)) (p.1 + p.2)
        ∂(MeasureTheory.volume.prod MeasureTheory.volume)
      = bin_mass d f i * bin_mass d f j := by
  set I : Set ℝ := window_interval W
  -- On B_i × B_j, the indicator is 1. So integrand = f p.1 * f p.2.
  have h_pw : ∀ p : ℝ × ℝ, p ∈ (Bin d i) ×ˢ (Bin d j) →
      f p.1 * f p.2 * Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2) = f p.1 * f p.2 := by
    intro p hp
    rcases hp with ⟨hp1, hp2⟩
    rw [window_indicator_on_good_bins W i j hij p.1 p.2 hp1 hp2, mul_one]
  -- Replace the integrand by `f p.1 * f p.2` using set_integral_congr
  have h_cong :
    ∫ p in (Bin d i) ×ˢ (Bin d j), f p.1 * f p.2 *
        Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)
        ∂(MeasureTheory.volume.prod MeasureTheory.volume)
    = ∫ p in (Bin d i) ×ˢ (Bin d j), f p.1 * f p.2
        ∂(MeasureTheory.volume.prod MeasureTheory.volume) := by
    apply MeasureTheory.setIntegral_congr_fun
    · exact (bin_measurable d i).prod (bin_measurable d j)
    · intro p hp
      exact h_pw p hp
  rw [h_cong]
  -- Now use `setIntegral_prod_mul`.
  rw [MeasureTheory.setIntegral_prod_mul]
  -- Rewrite the bin_mass form.
  have h1 : ∫ x in Bin d i, f x ∂MeasureTheory.volume = bin_mass d f i :=
    (bin_mass_eq_setIntegral d f i).symm
  have h2 : ∫ x in Bin d j, f x ∂MeasureTheory.volume = bin_mass d f j :=
    (bin_mass_eq_setIntegral d f j).symm
  rw [h1, h2]

end WindowIntegralCore

/-- **Core Lemma 1.3.**  Combine the Fubini step and the bin decomposition to
lower bound the window integral by the sum of bin-mass products over the
pair-sum support. -/
private lemma window_integral_lower_bound_core
    (f : ℝ → ℝ) (hf : Admissible f) (W : Window d) :
    MeasureTheory.integral MeasureTheory.volume
        (Set.indicator (window_interval W) (autoconv f))
      ≥ ∑ i : Fin d, ∑ j : Fin d,
          (if (i.val + j.val) ∈ pair_sum_support W
           then bin_mass d f i * bin_mass d f j else 0) := by
  -- Step 0: handle d = 0 (empty sum).
  by_cases hd : d = 0
  · subst hd
    -- Sum over Fin 0 is 0; need 0 ≤ ∫ indicator I_W (autoconv f).
    have h_sum_zero : ∑ i : Fin 0, ∑ j : Fin 0,
          (if (i.val + j.val) ∈ pair_sum_support W
           then bin_mass 0 f i * bin_mass 0 f j else 0) = 0 := by
      simp
    rw [h_sum_zero]
    -- GE: ∫ ≥ 0, i.e., 0 ≤ ∫.
    show 0 ≤ _
    apply MeasureTheory.integral_nonneg
    intro t
    by_cases ht : t ∈ window_interval W
    · rw [Set.indicator_of_mem ht]
      show 0 ≤ autoconv f t
      rw [autoconv_eq_integral f]
      apply MeasureTheory.integral_nonneg
      intro x
      exact mul_nonneg (hf.nonneg x) (hf.nonneg _)
    · rw [Set.indicator_of_notMem ht]
      exact le_refl _
  have hd_pos : 0 < d := Nat.pos_of_ne_zero hd
  -- Step 1: LHS = ∫ p, f(p.1) f(p.2) 1[p.1+p.2 ∈ I_W] d(vol × vol)
  rw [integral_indicator_autoconv_eq_prod_integral f hf W]
  set I : Set ℝ := window_interval W with hI_def
  set μ2 : MeasureTheory.Measure (ℝ × ℝ) := MeasureTheory.volume.prod MeasureTheory.volume with hμ2_def
  -- Let P(p) = f(p.1) * f(p.2) * 1[p.1+p.2 ∈ I]. We want: ∫ P ≥ Σ μ_i μ_j.
  -- Bound P by a "sum over good rectangles" nonneg integrand R.
  -- Define:
  --   R(p) := ∑_{i,j: i+j∈K_W} 1[B_i](p.1) * 1[B_j](p.2) * (f p.1 * f p.2 *
  --            1[p.1+p.2∈I])
  -- (no further changes — each summand is already nonneg)
  -- We will show:
  --   (a) ∫ R = Σ_{i+j∈K_W} μ_i μ_j
  --   (b) R(p) ≤ P(p) pointwise (because bin indicators are disjoint so their sum ≤ 1,
  --       and all factors are nonneg)
  -- Then ∫ P ≥ ∫ R = Σ μ_i μ_j.
  set R : ℝ × ℝ → ℝ := fun p =>
    ∑ i : Fin d, ∑ j : Fin d,
      (if (i.val + j.val) ∈ pair_sum_support W
       then Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
              f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p
       else 0) with hR_def
  set P : ℝ × ℝ → ℝ := fun p =>
    f p.1 * f p.2 * Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2) with hP_def
  -- Show P is nonneg
  have hP_nn : ∀ p, 0 ≤ P p := by
    intro p
    have hf1 := hf.nonneg p.1
    have hf2 := hf.nonneg p.2
    have hind : 0 ≤ Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2) := by
      by_cases h : p.1 + p.2 ∈ I
      · rw [Set.indicator_of_mem h]; norm_num
      · rw [Set.indicator_of_notMem h]
    exact mul_nonneg (mul_nonneg hf1 hf2) hind
  -- Integrability of P on μ2 (established internally in integral_indicator_autoconv_eq_prod_integral)
  have hI_ms : MeasurableSet I := window_interval_measurableSet W
  have hP_int : MeasureTheory.Integrable P μ2 := by
    have h_prod : MeasureTheory.Integrable
        (fun p : ℝ × ℝ => f p.1 * f p.2) μ2 := by
      have := hf.integrable.mul_prod hf.integrable
      simpa using this
    refine h_prod.mono' ?_ ?_
    · have h1 : Measurable (fun p : ℝ × ℝ => f p.1 * f p.2) := by
        exact (hf.measurable.comp measurable_fst).mul (hf.measurable.comp measurable_snd)
      have h2 : Measurable (fun p : ℝ × ℝ =>
          Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)) := by
        have hm_add : Measurable (fun p : ℝ × ℝ => p.1 + p.2) := measurable_fst.add measurable_snd
        have hm_ind : Measurable (fun t : ℝ => Set.indicator I (fun _ => (1 : ℝ)) t) :=
          (measurable_const.indicator hI_ms)
        exact hm_ind.comp hm_add
      exact (h1.mul h2).aestronglyMeasurable
    · refine Filter.Eventually.of_forall (fun p => ?_)
      show ‖f p.1 * f p.2 * Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)‖ ≤ f p.1 * f p.2
      by_cases hp : p.1 + p.2 ∈ I
      · rw [Set.indicator_of_mem hp, mul_one]
        exact le_of_eq (Real.norm_of_nonneg (mul_nonneg (hf.nonneg _) (hf.nonneg _)))
      · rw [Set.indicator_of_notMem hp, mul_zero, norm_zero]
        exact mul_nonneg (hf.nonneg _) (hf.nonneg _)
  -- Show: R(p) ≤ P(p) for all p
  have hR_le_P : ∀ p, R p ≤ P p := by
    intro p
    -- Each summand in R is either 0 (when condition false) or an indicator times P evaluated at p.
    -- The sum of indicators 1[B_i×B_j] over (i,j) with (i+j∈K_W) is ≤ 1 (disjoint rectangles).
    -- So R(p) ≤ ∑_{(i,j) ∈ disjoint} 1[B_i×B_j](p) · P(p) ≤ 1 · P(p) = P(p).
    show ∑ i : Fin d, ∑ j : Fin d,
      (if (i.val + j.val) ∈ pair_sum_support W
       then Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
              f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p
       else 0) ≤ P p
    -- Equal to: ∑_{(i,j) ∈ K} 1[B_i×B_j](p) · P(p)
    -- because on B_i × B_j with i+j ∈ K_W, the indicator factor = 1, so
    -- P(p) = f p.1 * f p.2, and the indicator of B_i×B_j times this is the summand.
    -- Bound: drop the "if" condition, enlarging each summand (nonneg):
    have step1 : ∑ i : Fin d, ∑ j : Fin d,
        (if (i.val + j.val) ∈ pair_sum_support W
         then Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
                f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p
         else 0)
      ≤ ∑ i : Fin d, ∑ j : Fin d,
        Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
              f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p := by
      apply Finset.sum_le_sum; intro i _
      apply Finset.sum_le_sum; intro j _
      split_ifs with h
      · exact le_refl _
      · -- need: 0 ≤ indicator (B_i×B_j) (fun q => f q.1 * f q.2 * ind I (q.1+q.2)) p
        by_cases hp : p ∈ (Bin d i) ×ˢ (Bin d j)
        · rw [Set.indicator_of_mem hp]
          exact hP_nn p
        · rw [Set.indicator_of_notMem hp]
    -- Step 2: ∑_{i,j} 1[B_i×B_j](p) · P(p) = (∑_{i,j} 1[B_i×B_j](p)) · P(p) ≤ P(p)
    -- because ∑_{i,j} 1[B_i×B_j](p) ≤ 1 (at most one (i,j) with p ∈ B_i × B_j).
    have step2 : ∀ (i j : Fin d) (q : ℝ × ℝ),
        Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q' : ℝ × ℝ =>
              f q'.1 * f q'.2 * Set.indicator I (fun _ => (1 : ℝ)) (q'.1 + q'.2)) q
        = Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun _ => (1 : ℝ)) q * P q := by
      intro i j q
      by_cases hq : q ∈ (Bin d i) ×ˢ (Bin d j)
      · rw [Set.indicator_of_mem hq, Set.indicator_of_mem hq, one_mul]
      · rw [Set.indicator_of_notMem hq, Set.indicator_of_notMem hq, zero_mul]
    -- Rewrite RHS of step1
    have step1' : ∑ i : Fin d, ∑ j : Fin d,
        Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
              f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p
      = (∑ i : Fin d, ∑ j : Fin d,
          Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun _ => (1 : ℝ)) p) * P p := by
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl; intro i _
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl; intro j _
      exact step2 i j p
    -- Step 3: ∑_{i,j} 1[B_i × B_j](p) ≤ 1 (rectangles disjoint).
    have step3 : (∑ i : Fin d, ∑ j : Fin d,
        Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun _ => (1 : ℝ)) p) ≤ 1 := by
      -- At most one (i,j) has p ∈ B_i × B_j (by bin_unique).
      -- Case A: ∃ (i₀, j₀), p ∈ B_{i₀} × B_{j₀}. Then sum = 1.
      -- Case B: No such (i₀, j₀). Then sum = 0.
      by_cases h_ex : ∃ i₀ j₀ : Fin d, p ∈ (Bin d i₀) ×ˢ (Bin d j₀)
      · obtain ⟨i₀, j₀, hp₀⟩ := h_ex
        have h_sum :
          ∑ i : Fin d, ∑ j : Fin d,
            Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun _ => (1 : ℝ)) p = 1 := by
          rw [Finset.sum_eq_single i₀]
          · rw [Finset.sum_eq_single j₀]
            · exact Set.indicator_of_mem hp₀ _
            · intro j _ hj_ne
              apply Set.indicator_of_notMem
              intro hp_ij
              have : p.2 ∈ Bin d j := hp_ij.2
              have h_eq_j : j = j₀ := bin_unique d this hp₀.2
              exact hj_ne h_eq_j
            · intro h; exact absurd (Finset.mem_univ _) h
          · intro i _ hi_ne
            apply Finset.sum_eq_zero
            intro j _
            apply Set.indicator_of_notMem
            intro hp_ij
            have : p.1 ∈ Bin d i := hp_ij.1
            have h_eq_i : i = i₀ := bin_unique d this hp₀.1
            exact hi_ne h_eq_i
          · intro h; exact absurd (Finset.mem_univ _) h
        linarith
      · -- No i,j with p ∈ B_i × B_j, so sum = 0 ≤ 1.
        have h_zero :
          ∑ i : Fin d, ∑ j : Fin d,
            Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun _ => (1 : ℝ)) p = 0 := by
          apply Finset.sum_eq_zero; intro i _
          apply Finset.sum_eq_zero; intro j _
          apply Set.indicator_of_notMem
          intro hp_ij
          exact h_ex ⟨i, j, hp_ij⟩
        linarith
    -- Combine:
    calc ∑ i : Fin d, ∑ j : Fin d,
          (if (i.val + j.val) ∈ pair_sum_support W
           then Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
                f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p
           else 0)
        ≤ ∑ i : Fin d, ∑ j : Fin d,
          Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
              f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p := step1
      _ = (∑ i : Fin d, ∑ j : Fin d,
          Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun _ => (1 : ℝ)) p) * P p := step1'
      _ ≤ 1 * P p := by
          exact mul_le_mul_of_nonneg_right step3 (hP_nn p)
      _ = P p := one_mul _
  -- ∫ R = Σ μ_i μ_j over i+j ∈ K_W.
  -- R is a finite sum, so we can interchange sum/integral, and integrate each term.
  have hR_nn : ∀ p, 0 ≤ R p := by
    intro p
    apply Finset.sum_nonneg; intro i _
    apply Finset.sum_nonneg; intro j _
    split_ifs with h
    · by_cases hp : p ∈ (Bin d i) ×ˢ (Bin d j)
      · rw [Set.indicator_of_mem hp]; exact hP_nn p
      · rw [Set.indicator_of_notMem hp]
    · exact le_refl _
  have hR_int : MeasureTheory.Integrable R μ2 := by
    -- R ≤ P pointwise, R ≥ 0, P is integrable.
    refine hP_int.mono' ?_ (Filter.Eventually.of_forall fun p => ?_)
    · -- R is measurable
      show AEStronglyMeasurable (fun p => ∑ i : Fin d, ∑ j : Fin d,
        (if (i.val + j.val) ∈ pair_sum_support W
         then Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
                f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p
         else 0)) μ2
      apply Finset.aestronglyMeasurable_fun_sum
      intro i _
      apply Finset.aestronglyMeasurable_fun_sum
      intro j _
      split_ifs with h
      · -- indicator of measurable set times measurable function
        have h_set : MeasurableSet ((Bin d i) ×ˢ (Bin d j)) :=
          (bin_measurable d i).prod (bin_measurable d j)
        have h_fn : Measurable (fun q : ℝ × ℝ =>
          f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) := by
          have h1 : Measurable (fun p : ℝ × ℝ => f p.1 * f p.2) := by
            exact (hf.measurable.comp measurable_fst).mul (hf.measurable.comp measurable_snd)
          have h2 : Measurable (fun p : ℝ × ℝ =>
              Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)) := by
            have hm_add : Measurable (fun p : ℝ × ℝ => p.1 + p.2) := measurable_fst.add measurable_snd
            have hm_ind : Measurable (fun t : ℝ => Set.indicator I (fun _ => (1 : ℝ)) t) :=
              (measurable_const.indicator hI_ms)
            exact hm_ind.comp hm_add
          exact h1.mul h2
        exact (h_fn.indicator h_set).aestronglyMeasurable
      · exact aestronglyMeasurable_const
    · show ‖R p‖ ≤ P p
      rw [Real.norm_eq_abs, abs_of_nonneg (hR_nn p)]
      exact hR_le_P p
  have h_int_R : ∫ p, R p ∂μ2
      = ∑ i : Fin d, ∑ j : Fin d,
          (if (i.val + j.val) ∈ pair_sum_support W
           then bin_mass d f i * bin_mass d f j else 0) := by
    show ∫ p, (∑ i : Fin d, ∑ j : Fin d,
      (if (i.val + j.val) ∈ pair_sum_support W
       then Set.indicator ((Bin d i) ×ˢ (Bin d j)) (fun q : ℝ × ℝ =>
              f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p
       else 0)) ∂μ2 = _
    -- Pull sum out of integral
    rw [MeasureTheory.integral_finset_sum]
    · apply Finset.sum_congr rfl
      intro i _
      rw [MeasureTheory.integral_finset_sum]
      · apply Finset.sum_congr rfl
        intro j _
        split_ifs with hij
        · -- ∫ p, 1[B_i × B_j](p) * (f p.1 * f p.2 * 1[p.1+p.2∈I]) dp = μ_i * μ_j
          rw [MeasureTheory.integral_indicator ((bin_measurable d i).prod (bin_measurable d j))]
          exact setIntegral_good_rectangle f W i j hij
        · rw [MeasureTheory.integral_zero]
      · -- integrability of each summand
        intro j _
        split_ifs with hij
        · -- integrand is indicator of B_i × B_j times f p.1 f p.2 * 1[p.1+p.2 ∈ I], ≤ P, integrable.
          refine hP_int.mono' ?_ (Filter.Eventually.of_forall fun p => ?_)
          · have h_set : MeasurableSet ((Bin d i) ×ˢ (Bin d j)) :=
              (bin_measurable d i).prod (bin_measurable d j)
            have h_fn : Measurable (fun q : ℝ × ℝ =>
              f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) := by
              have h1 : Measurable (fun p : ℝ × ℝ => f p.1 * f p.2) := by
                exact (hf.measurable.comp measurable_fst).mul (hf.measurable.comp measurable_snd)
              have h2 : Measurable (fun p : ℝ × ℝ =>
                  Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)) := by
                have hm_add : Measurable (fun p : ℝ × ℝ => p.1 + p.2) :=
                  measurable_fst.add measurable_snd
                have hm_ind : Measurable (fun t : ℝ => Set.indicator I (fun _ => (1 : ℝ)) t) :=
                  (measurable_const.indicator hI_ms)
                exact hm_ind.comp hm_add
              exact h1.mul h2
            exact (h_fn.indicator h_set).aestronglyMeasurable
          · -- Goal: ‖(indicator ...) p‖ ≤ P p
            have hind_nn : 0 ≤ Set.indicator ((Bin d i) ×ˢ (Bin d j))
                (fun q : ℝ × ℝ =>
                    f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p := by
              by_cases hp : p ∈ (Bin d i) ×ˢ (Bin d j)
              · rw [Set.indicator_of_mem hp]; exact hP_nn p
              · rw [Set.indicator_of_notMem hp]
            have hind_le : Set.indicator ((Bin d i) ×ˢ (Bin d j))
                (fun q : ℝ × ℝ =>
                    f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p ≤ P p := by
              by_cases hp : p ∈ (Bin d i) ×ˢ (Bin d j)
              · rw [Set.indicator_of_mem hp]
              · rw [Set.indicator_of_notMem hp]; exact hP_nn p
            show ‖Set.indicator ((Bin d i) ×ˢ (Bin d j))
                (fun q : ℝ × ℝ =>
                    f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p‖ ≤ P p
            rw [Real.norm_eq_abs, abs_of_nonneg hind_nn]
            exact hind_le
        · exact MeasureTheory.integrable_zero _ _ _
    · intro i _
      apply MeasureTheory.integrable_finset_sum
      intro j _
      split_ifs with hij
      · refine hP_int.mono' ?_ (Filter.Eventually.of_forall fun p => ?_)
        · have h_set : MeasurableSet ((Bin d i) ×ˢ (Bin d j)) :=
            (bin_measurable d i).prod (bin_measurable d j)
          have h_fn : Measurable (fun q : ℝ × ℝ =>
            f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) := by
            have h1 : Measurable (fun p : ℝ × ℝ => f p.1 * f p.2) := by
              exact (hf.measurable.comp measurable_fst).mul (hf.measurable.comp measurable_snd)
            have h2 : Measurable (fun p : ℝ × ℝ =>
                Set.indicator I (fun _ => (1 : ℝ)) (p.1 + p.2)) := by
              have hm_add : Measurable (fun p : ℝ × ℝ => p.1 + p.2) :=
                measurable_fst.add measurable_snd
              have hm_ind : Measurable (fun t : ℝ => Set.indicator I (fun _ => (1 : ℝ)) t) :=
                (measurable_const.indicator hI_ms)
              exact hm_ind.comp hm_add
            exact h1.mul h2
          exact (h_fn.indicator h_set).aestronglyMeasurable
        · -- Goal: ‖(indicator ...) p‖ ≤ P p
          have hind_nn : 0 ≤ Set.indicator ((Bin d i) ×ˢ (Bin d j))
              (fun q : ℝ × ℝ =>
                  f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p := by
            by_cases hp : p ∈ (Bin d i) ×ˢ (Bin d j)
            · rw [Set.indicator_of_mem hp]; exact hP_nn p
            · rw [Set.indicator_of_notMem hp]
          have hind_le : Set.indicator ((Bin d i) ×ˢ (Bin d j))
              (fun q : ℝ × ℝ =>
                  f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p ≤ P p := by
            by_cases hp : p ∈ (Bin d i) ×ˢ (Bin d j)
            · rw [Set.indicator_of_mem hp]
            · rw [Set.indicator_of_notMem hp]; exact hP_nn p
          show ‖Set.indicator ((Bin d i) ×ˢ (Bin d j))
              (fun q : ℝ × ℝ =>
                  f q.1 * f q.2 * Set.indicator I (fun _ => (1 : ℝ)) (q.1 + q.2)) p‖ ≤ P p
          rw [Real.norm_eq_abs, abs_of_nonneg hind_nn]
          exact hind_le
      · exact MeasureTheory.integrable_zero _ _ _
  -- Main inequality: ∫ P ≥ ∫ R = sum.
  have h_mono : ∫ p, R p ∂μ2 ≤ ∫ p, P p ∂μ2 := by
    refine MeasureTheory.integral_mono_ae hR_int hP_int ?_
    exact Filter.Eventually.of_forall hR_le_P
  -- Rewrite goal
  show ∑ i : Fin d, ∑ j : Fin d,
      (if (i.val + j.val) ∈ pair_sum_support W
       then bin_mass d f i * bin_mass d f j else 0)
    ≤ ∫ p, P p ∂μ2
  rw [← h_int_R]
  exact h_mono

/-- Lemma 1.3 (window integral lower bound).  See
`window_integral_lower_bound_core` for the single atomic measure-
theoretic step. -/
lemma window_integral_lower_bound
    (f : ℝ → ℝ) (hf : Admissible f) (W : Window d) :
    MeasureTheory.integral MeasureTheory.volume
        (Set.indicator (window_interval W) (autoconv f))
      ≥ ∑ i : Fin d, ∑ j : Fin d,
          (if (i.val + j.val) ∈ pair_sum_support W
           then bin_mass d f i * bin_mass d f j else 0) :=
  window_integral_lower_bound_core f hf W

end IntervalBnB

end -- noncomputable section
