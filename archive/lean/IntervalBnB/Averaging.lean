/-
IntervalBnB — Lemma 1.2 (averaging: max ≥ average).

For any admissible `f` and any window `W`,
    max_{t ∈ [-1/2, 1/2]} (f * f)(t)  ≥  (1/|I_W|) · ∫_{I_W} (f * f)(t) dt.

Mathematically this is "the max is at least the average", a direct
consequence of `∫_I h ≤ |I| · (ess sup h)` for `h ≥ 0` on `I`.
-/

import IntervalBnB.Defs
import IntervalBnB.ScaleFactor

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
### The reference window `[-1/2, 1/2]`.
-/

/-- `window_interval W ⊆ [-1/2, 1/2]`. -/
lemma window_interval_subset_ref (W : Window d) (hd : 0 < d) :
    window_interval W ⊆ Set.Icc (-(1 : ℝ)/2) (1/2) := by
  intro t ht
  rcases ht with ⟨h1, h2⟩
  refine ⟨?_, ?_⟩
  · -- -1/2 ≤ t since sLo ≥ 0
    have h_sLo_nn : (0 : ℝ) ≤ (W.sLo : ℝ) := by exact_mod_cast Nat.zero_le _
    have hd_pos : (0 : ℝ) < (2*d : ℝ) := by
      have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
      linarith
    have : (0 : ℝ) ≤ (W.sLo : ℝ) / (2*d) :=
      div_nonneg h_sLo_nn (le_of_lt hd_pos)
    linarith
  · -- t ≤ 1/2 since sLo + ell ≤ 2d
    have h_le : W.sLo + W.ell ≤ 2*d := W.sLo_add_ell_le
    have h_le_R : ((W.sLo : ℝ) + (W.ell : ℝ)) ≤ (2*d : ℝ) := by
      have : ((W.sLo + W.ell : ℕ) : ℝ) ≤ ((2*d : ℕ) : ℝ) := by exact_mod_cast h_le
      push_cast at this; linarith
    have hd_pos : (0 : ℝ) < (2*d : ℝ) := by
      have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
      linarith
    have h_div : ((W.sLo : ℝ) + W.ell) / (2*d) ≤ 1 := by
      rw [div_le_one hd_pos]; exact h_le_R
    linarith

/-!
### Autoconv measurability & nonnegativity.
-/

/-- The autoconvolution of a nonneg function is nonneg everywhere
    (provided the defining integral exists).  For admissible `f`, the
    integrand `t ↦ f(x) * f(t - x)` is nonneg, so the integral is
    nonneg. -/
lemma autoconv_nonneg {f : ℝ → ℝ} (hf : Admissible f) (t : ℝ) :
    0 ≤ autoconv f t := by
  unfold autoconv MeasureTheory.convolution
  apply MeasureTheory.integral_nonneg
  intro x
  simp only [ContinuousLinearMap.mul_apply']
  exact mul_nonneg (hf.nonneg x) (hf.nonneg (t - x))

/-- Autoconv of a Bochner-integrable `f` is Bochner-integrable. -/
lemma autoconv_integrable {f : ℝ → ℝ} (hf : Admissible f) :
    Integrable (autoconv f) volume := by
  unfold autoconv
  exact hf.integrable.integrable_convolution _ hf.integrable

/-!
### The averaging inequality.
-/

/-- Lemma 1.2 (averaging). For admissible `f`, window `W`, and `d ≥ 1`,
    `essSupConv f · |I_W| ≥ ∫_{I_W} (f * f)`.

    Proof: `f * f ≥ 0` on ℝ and is essentially bounded by `essSupConv f`
    on `[-1/2, 1/2]` (by definition of `essSupConv`). Restricting to
    `I_W ⊆ [-1/2, 1/2]` and integrating yields the claim via
    `norm_setIntegral_le_of_norm_le_const_ae`. -/
lemma averaging_bound (f : ℝ → ℝ) (hf : Admissible f) (W : Window d)
    (hd : 0 < d) :
    essSupConv f * ((W.ell : ℝ)/(2*d))
      ≥ MeasureTheory.integral MeasureTheory.volume
          (Set.indicator (window_interval W) (autoconv f)) := by
  set I : Set ℝ := window_interval W with hI_def
  set J : Set ℝ := Set.Icc (-(1 : ℝ)/2) (1/2) with hJ_def
  set g : ℝ → ℝ := autoconv f with hg_def
  -- Window interval is measurable
  have hI_meas : MeasurableSet I := by
    show MeasurableSet (window_interval W)
    unfold window_interval; exact measurableSet_Icc
  have hJ_meas : MeasurableSet J := measurableSet_Icc
  have hI_sub : I ⊆ J := window_interval_subset_ref W hd
  -- The indicator integral equals the set integral
  have hind :
      MeasureTheory.integral volume (Set.indicator I g)
        = ∫ t in I, g t ∂volume := integral_indicator hI_meas
  rw [hind]
  -- Volume of I is finite (in fact equals ell/(2d))
  have hvol_real : (volume I).toReal = (W.ell : ℝ)/(2*d) :=
    window_interval_volume W hd
  have hvol_lt : volume I < ⊤ := by
    have : (volume I).toReal = (W.ell : ℝ)/(2*d) := hvol_real
    -- toReal finite value implies < ⊤ unless = ⊤ with toReal = 0
    by_contra h
    push_neg at h
    have h_eq : volume I = ⊤ := le_antisymm le_top h
    rw [h_eq, ENNReal.toReal_top] at this
    have hpos : (0 : ℝ) < (W.ell : ℝ)/(2*d) := window_interval_length_pos W hd
    linarith
  -- essSupConv f equals (eLpNorm (indicator J g) ⊤).toReal; its eLpNorm is < ⊤
  have heLp_lt : eLpNorm (Set.indicator J g) ⊤ volume < ⊤ := hf.autoconv_ess_lt_top
  -- Convert eLpNorm to eLpNormEssSup
  have heLp_eq : eLpNorm (Set.indicator J g) ⊤ volume
      = eLpNormEssSup (Set.indicator J g) volume := eLpNorm_exponent_top
  -- Nonneg a.e. bound for g on J: ‖indicator J g t‖ = |g t| = g t on J.
  -- Key lemma: ∀ᵐ t, ‖(indicator J g) t‖ₑ ≤ eLpNormEssSup (indicator J g) volume.
  have h_ae :=
    enorm_ae_le_eLpNormEssSup (Set.indicator J g) volume
  -- Convert enorm-bound to norm-bound using finiteness
  have h_essSup_fin :
      eLpNormEssSup (Set.indicator J g) volume ≠ ⊤ := by
    rw [← heLp_eq]; exact heLp_lt.ne
  -- Define C := essSupConv f (the real toReal value)
  set C : ℝ := essSupConv f with hC
  have hC_def : (C : ℝ) = (eLpNormEssSup (Set.indicator J g) volume).toReal := by
    show essSupConv f = (eLpNormEssSup (Set.indicator J g) volume).toReal
    unfold essSupConv; rw [heLp_eq]
  -- The essSup ENNReal equals ENNReal.ofReal C (since value is finite and ≥ 0)
  have hC_nn : 0 ≤ C := by
    rw [hC_def]; exact ENNReal.toReal_nonneg
  have hC_enn : eLpNormEssSup (Set.indicator J g) volume = ENNReal.ofReal C := by
    rw [hC_def]; exact (ENNReal.ofReal_toReal h_essSup_fin).symm
  -- Transfer enorm-bound to pointwise real bound on J
  -- ‖indicator J g t‖ₑ ≤ ENNReal.ofReal C   ⟹   indicator J g t ≤ C  (a.e.)
  -- We take the bound only on ae t, with ‖·‖ = ‖indicator J g t‖ = |g t| for t ∈ J, and 0 off J
  -- Simpler: prove indicator J g ≤ C a.e. (pointwise real).
  have h_ae_real : ∀ᵐ t ∂volume, Set.indicator J g t ≤ C := by
    filter_upwards [h_ae] with t ht
    rw [hC_enn] at ht
    -- ht : ‖indicator J g t‖ₑ ≤ ENNReal.ofReal C
    -- Use: for a real number y, ‖y‖ₑ = ENNReal.ofReal |y|
    have h1 : ‖Set.indicator J g t‖ₑ = ENNReal.ofReal |Set.indicator J g t| :=
      Real.enorm_eq_ofReal_abs _
    rw [h1] at ht
    have habs : |Set.indicator J g t| ≤ C := (ENNReal.ofReal_le_ofReal_iff hC_nn).mp ht
    exact (le_abs_self _).trans habs
  -- Now bound a.e. on I ⊆ J
  have h_ae_I : ∀ᵐ t ∂(volume.restrict I), g t ≤ C := by
    rw [ae_restrict_iff' hI_meas]
    filter_upwards [h_ae_real] with t ht ht_in_I
    have ht_in_J : t ∈ J := hI_sub ht_in_I
    have : Set.indicator J g t = g t := Set.indicator_of_mem ht_in_J _
    rw [this] at ht
    exact ht
  -- Nonneg a.e. on I (true globally, but we only need ae)
  have hg_nn : ∀ t, 0 ≤ g t := autoconv_nonneg hf
  -- Use norm_setIntegral_le_of_norm_le_const_ae with C
  have h_abs_ae : ∀ᵐ t ∂(volume.restrict I), ‖g t‖ ≤ C := by
    filter_upwards [h_ae_I] with t ht
    rw [Real.norm_eq_abs, abs_of_nonneg (hg_nn t)]
    exact ht
  have h_bound := norm_setIntegral_le_of_norm_le_const_ae hvol_lt h_abs_ae
  -- h_bound : ‖∫ t in I, g t ∂volume‖ ≤ C * (volume.real I)
  -- volume.real I = (volume I).toReal = ell/(2d)
  have hreal_eq : volume.real I = (W.ell : ℝ)/(2*d) := by
    show (volume (window_interval W)).toReal = (W.ell : ℝ)/(2*d)
    exact window_interval_volume W hd
  rw [hreal_eq] at h_bound
  -- ‖∫‖ ≤ C · (ell/(2d));  since ∫ ≥ 0, ‖∫‖ = ∫
  have h_int_nn : 0 ≤ ∫ t in I, g t ∂volume := by
    apply MeasureTheory.integral_nonneg
    intro t
    exact hg_nn t
  rw [Real.norm_eq_abs, abs_of_nonneg h_int_nn] at h_bound
  -- Goal: C * (ell/(2d)) ≥ ∫, i.e. ∫ ≤ C * (ell/(2d))
  exact h_bound

end IntervalBnB

end -- noncomputable section
