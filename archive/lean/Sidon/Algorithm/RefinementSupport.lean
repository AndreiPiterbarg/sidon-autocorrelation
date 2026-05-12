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
-- Refinement & Support Properties (Claims 2.2, 2.3)
-- Source: output (8).lean (UUID: 8b7ac59c)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Support of convolution is contained in Minkowski sum of supports. -/
theorem support_convolution_subset_add {f : ℝ → ℝ} {s : Set ℝ} (hf : Function.support f ⊆ s) :
    Function.support (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊆ s + s := by
  intro x hx
  obtain ⟨y, hy1, hy2⟩ : ∃ y, f y ≠ 0 ∧ f (x - y) ≠ 0 := by
    contrapose! hx; simp_all +decide [ MeasureTheory.convolution ] ;
    exact MeasureTheory.integral_eq_zero_of_ae <| Filter.Eventually.of_forall fun t => by by_cases h : f t = 0 <;> aesop;
  exact ⟨ y, hf hy1, x - y, hf hy2, by ring ⟩

/-- The boundary between the first n bins and the last n bins is exactly at x = 0. -/
theorem left_frac_exact (n m : ℕ) (hn : n > 0) (_hm : m > 0)
    (c : Fin (2 * n) → ℕ) (_hc : ∑ i, c i = 4 * n * m) :
    let δ := (1 : ℝ) / (4 * n)
    (-1/4 : ℝ) + n * δ = 0 := by
  field_simp [hn]
  ring

/-- Asymmetry threshold can be compared directly — no margin needed. -/
theorem asymmetry_no_margin (c_target : ℝ) (_hct : 0 < c_target)
    (L : ℝ) (_hL : L ≥ Real.sqrt (c_target / 2))
    (h_bound : ∀ L', 2 * L' ^ 2 ≤ c_target → L' < L) :
    2 * L ^ 2 ≥ c_target := by
  contrapose! h_bound;
  exact ⟨ L, by linarith, le_rfl ⟩

/-- Pointwise convolution integrand inequality for 0 ≤ f ≤ g. -/
theorem convolution_integrand_le {f g : ℝ → ℝ} (hf : 0 ≤ f) (hg : 0 ≤ g) (h_le : f ≤ g) (x t : ℝ) :
    f t * f (x - t) ≤ g t * g (x - t) := by
  exact mul_le_mul ( h_le _ ) ( h_le _ ) ( hf _ ) ( hg _ )

/-- Integral of pointwise-bounded convolution integrands. -/
theorem integral_convolution_le {f g : ℝ → ℝ} (x : ℝ)
    (h_le : ∀ t, f t * f (x - t) ≤ g t * g (x - t))
    (hf_int : MeasureTheory.Integrable (fun t => f t * f (x - t)) MeasureTheory.volume)
    (hg_int : MeasureTheory.Integrable (fun t => g t * g (x - t)) MeasureTheory.volume) :
    ∫ t, f t * f (x - t) ∂MeasureTheory.volume ≤ ∫ t, g t * g (x - t) ∂MeasureTheory.volume := by
  apply_rules [ MeasureTheory.integral_mono ]

/-- Measure of support of autoconvolution bounded by 2δ. -/
theorem measure_support_convolution_bound {g : ℝ → ℝ} {a δ : ℝ} (_hδ : 0 < δ)
    (hg_supp : Function.support g ⊆ Set.Ioo a (a + δ)) :
    MeasureTheory.volume (Function.support (MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)) ≤ ENNReal.ofReal (2 * δ) := by
  have h_support : Function.support (MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊆ Set.Ioo (a + a) (a + a + 2 * δ) := by
    intro x hx; have := support_convolution_subset_add hg_supp; simp_all +decide [ Set.subset_def ] ; (
    obtain ⟨ y, hy, z, hz, rfl ⟩ := this x hx; constructor <;> linarith [ hy.1, hy.2, hz.1, hz.2 ] ;);
  exact le_trans ( MeasureTheory.measure_mono h_support ) ( by simp +decide [ two_mul ] )

end -- noncomputable section
