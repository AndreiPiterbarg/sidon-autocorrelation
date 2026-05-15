/-
Sidon Autocorrelation Project — MV Lemma 3.1 Eq.(1) + Arcsine Mass
==================================================================

This file discharges two `ExtremiserPrimitives` bundle fields for the
3-scale arcsine kernel `K_ms` of `Sidon.MultiScale`:

  * **Arcsine mass identity.**  `∫ K_arc(δ, ·) = 1` for each `δ > 0`,
     and `∫ K_ms = 1`.
  * **MV Lemma 3.1 Eq.(1) discharge.**  The MV Lemma 3.1 Eq.(1)
     conclusion `∫(f*f)·K_ms ≤ ‖f*f‖_∞` for any admissible nonneg `f`
     with finite `‖f*f‖_∞`, packaged in the `R(f)·(∫f)²` form needed
     by the bundle.

Strategy.

  * The mass identity uses `Sidon.Bessel.arcsine_moment_integral` at
    `k = 0`, which gives `∫_{Ioo(-δ, δ)} (δ²-x²)^{-1/2} dx = π`.
    Multiplying by `1/π` yields `∫ K_arc(δ, ·) = 1`.  Then `K_ms` is
    `∑ λᵢ · K_arc(δᵢ, ·)` with `∑ λᵢ = 1`, so `∫ K_ms = 1` by
    linearity.
  * The Eq.(1) discharge invokes `Sidon.MV.mv_eq1` on `(f, K_ms)` with
    the a.e. bound `(f*f)(x) ≤ ‖f*f‖_∞.toReal` (from
    `ae_le_eLpNormEssSup`).  The main routing work is converting the
    enorm-form a.e. bound into a real a.e. bound, and threading
    integrability of `(f*f)·K_ms`.

No `sorry`, no new axioms.

References
----------
* MV 2010, arXiv:0907.1379, Lemma 3.1 Eq.(1).
* `Sidon.Bessel.arcsine_moment_integral` (Bessel.lean).
* `Sidon.MV.mv_eq1` (MVLemmas.lean).
* `Sidon.MultiScale` (MultiScale.lean):
    - `K_arc`, `K_ms`,
    - `delta_positivities`, `lambdas_sum_one`, `lambdas_nonneg`,
    - `K_arc_nonneg`, `K_ms_nonneg`.
-/

import Mathlib
import Sidon.Defs
import Sidon.MVLemmas
import Sidon.MultiScale
import Sidon.Bessel

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real
open MeasureTheory

namespace Sidon.BundleEq1

/-! ## Arcsine mass: integral of `K_arc(δ, ·)` equals `1`

`K_arc(δ, x) = (η_δ * η_δ)(x)` is defined as the autoconvolution of
the half-arcsine density `η_δ` (mass 1 on `(-δ/2, δ/2)`).  The mass
identity follows from `MeasureTheory.integral_convolution`:
`∫(η * η) = (∫η)·(∫η) = 1`.

These wrappers re-export `Sidon.MultiScale.K_arc_integral_eq_one`
and `Sidon.MultiScale.K_arc_integrable` (proved upstream in
`Sidon.MultiScale`) under the names used in this file.
-/

/-- Each arcsine kernel integrates to `1` (re-exported from
`Sidon.MultiScale`). -/
theorem K_arc_integral_eq_one (δ : ℝ) (hδ : 0 < δ) :
    ∫ x, Sidon.MultiScale.K_arc δ x ∂volume = 1 :=
  Sidon.MultiScale.K_arc_integral_eq_one δ hδ

/-- Each `K_arc(δ, ·)` is integrable (re-exported from `Sidon.MultiScale`). -/
theorem K_arc_integrable (δ : ℝ) (hδ : 0 < δ) :
    Integrable (Sidon.MultiScale.K_arc δ) volume :=
  Sidon.MultiScale.K_arc_integrable δ hδ

/-! ## `K_ms` mass and integrability

`K_ms` is a finite linear combination of arcsine kernels, all of
which are integrable (by `K_arc_integrable`).  Its mass equals
`λ₁ + λ₂ + λ₃ = 1` by `lambdas_sum_one`.
-/

/-- The multi-scale kernel is integrable as a sum of integrable terms. -/
theorem K_ms_integrable :
    Integrable Sidon.MultiScale.K_ms volume := by
  unfold Sidon.MultiScale.K_ms
  have hδ := Sidon.MultiScale.delta_positivities
  have h1 : Integrable
      (fun x => Sidon.MultiScale.lambda1 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x)
      volume :=
    (K_arc_integrable _ hδ.1).const_mul _
  have h2 : Integrable
      (fun x => Sidon.MultiScale.lambda2 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x)
      volume :=
    (K_arc_integrable _ hδ.2.1).const_mul _
  have h3 : Integrable
      (fun x => Sidon.MultiScale.lambda3 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta3 x)
      volume :=
    (K_arc_integrable _ hδ.2.2).const_mul _
  exact (h1.add h2).add h3

/-- The multi-scale arcsine kernel integrates to `1`. -/
theorem K_ms_integral_eq_one :
    ∫ x, Sidon.MultiScale.K_ms x ∂volume = 1 := by
  unfold Sidon.MultiScale.K_ms
  have hδ := Sidon.MultiScale.delta_positivities
  -- Step 1: distribute integral across the three λᵢ·K_arc(δᵢ, ·) terms.
  have h_int_step :
      ∫ x, (Sidon.MultiScale.lambda1 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x +
              Sidon.MultiScale.lambda2 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x +
              Sidon.MultiScale.lambda3 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta3 x) ∂volume
        = Sidon.MultiScale.lambda1 * ∫ x, Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x ∂volume
          + Sidon.MultiScale.lambda2 * ∫ x, Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x ∂volume
          + Sidon.MultiScale.lambda3 * ∫ x, Sidon.MultiScale.K_arc Sidon.MultiScale.delta3 x ∂volume := by
    have h1 : Integrable
        (fun x => Sidon.MultiScale.lambda1 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x)
        volume :=
      (K_arc_integrable _ hδ.1).const_mul _
    have h2 : Integrable
        (fun x => Sidon.MultiScale.lambda2 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x)
        volume :=
      (K_arc_integrable _ hδ.2.1).const_mul _
    have h3 : Integrable
        (fun x => Sidon.MultiScale.lambda3 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta3 x)
        volume :=
      (K_arc_integrable _ hδ.2.2).const_mul _
    -- Use `integral_add` twice (pointwise add forms), then `integral_const_mul`
    -- on each scalar-multiplied term.
    have step_a :
        ∫ x, (Sidon.MultiScale.lambda1 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x +
                Sidon.MultiScale.lambda2 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x) +
              Sidon.MultiScale.lambda3 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta3 x ∂volume
          = (∫ x, Sidon.MultiScale.lambda1 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x +
                Sidon.MultiScale.lambda2 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x ∂volume) +
            ∫ x, Sidon.MultiScale.lambda3 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta3 x ∂volume :=
      integral_add (h1.add h2) h3
    have step_b :
        ∫ x, Sidon.MultiScale.lambda1 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x +
              Sidon.MultiScale.lambda2 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x ∂volume
          = (∫ x, Sidon.MultiScale.lambda1 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta1 x ∂volume) +
            ∫ x, Sidon.MultiScale.lambda2 * Sidon.MultiScale.K_arc Sidon.MultiScale.delta2 x ∂volume :=
      integral_add h1 h2
    rw [step_a, step_b, integral_const_mul, integral_const_mul, integral_const_mul]
  rw [h_int_step]
  rw [K_arc_integral_eq_one _ hδ.1, K_arc_integral_eq_one _ hδ.2.1,
      K_arc_integral_eq_one _ hδ.2.2]
  -- Now we have λ₁·1 + λ₂·1 + λ₃·1 = λ₁ + λ₂ + λ₃ = 1.
  have hsum := Sidon.MultiScale.lambdas_sum_one
  -- hsum : lambda1Q + lambda2Q + lambda3Q = 1 (rational form)
  have hsum_real :
      Sidon.MultiScale.lambda1 + Sidon.MultiScale.lambda2 + Sidon.MultiScale.lambda3 = 1 := by
    unfold Sidon.MultiScale.lambda1 Sidon.MultiScale.lambda2 Sidon.MultiScale.lambda3
    have := congrArg (fun q : ℚ => (q : ℝ)) hsum
    push_cast at this
    linarith
  linarith [hsum_real]

/-! ## Discharge of MV Eq.(1) for `K_ms`

This packages the application of `Sidon.MV.mv_eq1` with `K = K_ms`.

The discharge produces `LHS1 := ∫ (f*f) · K_ms ≤ ‖f*f‖_∞.toReal`,
which is the natural ("unnormalised") form of MV Eq.(1).  The bundle
field `hEq1 : LHS1 ≤ autoconvolution_ratio f` is the **normalised
form** under `∫f = 1`, in which case `R(f) = ‖f*f‖_∞.toReal`.

We provide both:
  * `hEq1_unnormalised`: `LHS1 ≤ ‖f*f‖_∞.toReal`, valid for any
    admissible nonneg `f` with `eLpNorm (f*f) ⊤ volume ≠ ⊤`.
  * `hEq1_discharge`: the bundle form `LHS1 ≤ R(f)`, requiring the
    extra hypothesis `∫f = 1` (which the bundle implicitly assumes).
-/

/-- Abbreviation for the convolution `f*f`. -/
private noncomputable def conv (f : ℝ → ℝ) : ℝ → ℝ :=
  MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume

/-- Definitional unfolding of `conv`. -/
private lemma conv_def (f : ℝ → ℝ) :
    conv f = MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume := rfl

/-- The `LHS1` field of the `ExtremiserPrimitives` bundle: the
integral `∫ (f*f) · K_ms`. -/
noncomputable def LHS1 (f : ℝ → ℝ) : ℝ :=
  ∫ x, (conv f) x * Sidon.MultiScale.K_ms x ∂volume

/-- Nonnegativity of `f*f` for nonneg `f`. -/
lemma conv_nonneg {f : ℝ → ℝ} (hf : ∀ x, 0 ≤ f x) (x : ℝ) :
    0 ≤ conv f x := by
  unfold conv
  exact convolution_nonneg hf hf x

/-- A.e. bound: `(f*f)(x) ≤ (eLpNorm (f*f) ⊤ volume).toReal` whenever
`eLpNorm (f*f) ⊤ volume ≠ ⊤` and `f*f` is a.e. nonneg.

This converts the abstract enorm/essSup bound `ae_le_eLpNormEssSup`
into a real-valued a.e. bound. -/
lemma conv_ae_le_eLpNorm_toReal {f : ℝ → ℝ} (hf : ∀ x, 0 ≤ f x)
    (h_conv_fin : eLpNorm (conv f) ⊤ volume ≠ ⊤) :
    ∀ᵐ x ∂volume, conv f x ≤ (eLpNorm (conv f) ⊤ volume).toReal := by
  -- ae_le_eLpNormEssSup gives ‖(conv f) x‖ₑ ≤ eLpNormEssSup (conv f) volume.
  have h_ae : ∀ᵐ x ∂volume, ‖conv f x‖ₑ ≤ eLpNormEssSup (conv f) volume :=
    ae_le_eLpNormEssSup
  -- eLpNorm ⊤ = eLpNormEssSup.
  have h_top : eLpNorm (conv f) ⊤ volume = eLpNormEssSup (conv f) volume :=
    eLpNorm_exponent_top
  rw [h_top] at h_conv_fin
  filter_upwards [h_ae] with x hx
  have h_nonneg : 0 ≤ conv f x := conv_nonneg hf x
  -- ‖(conv f) x‖ₑ = ENNReal.ofReal ((conv f) x) since (conv f) x ≥ 0.
  have h_enorm : ‖conv f x‖ₑ = ENNReal.ofReal (conv f x) :=
    Real.enorm_of_nonneg h_nonneg
  rw [h_enorm] at hx
  -- hx : ENNReal.ofReal ((conv f) x) ≤ eLpNormEssSup (conv f) volume
  -- Convert to .toReal inequality.
  have h_eq : (conv f) x = (ENNReal.ofReal (conv f x)).toReal :=
    (ENNReal.toReal_ofReal h_nonneg).symm
  rw [h_eq]
  exact ENNReal.toReal_mono h_conv_fin hx

/-- `(eLpNorm (f*f) ⊤ volume).toReal` is nonnegative. -/
lemma eLpNorm_toReal_nonneg (f : ℝ → ℝ) :
    0 ≤ (eLpNorm (conv f) ⊤ volume).toReal :=
  ENNReal.toReal_nonneg

/-- Integrability of `(f*f) · K_ms` from the a.e. bound on `f*f` and
the structural a.e. strong measurability of `f*f`.

The `AEStronglyMeasurable` input is a structural prerequisite: it
holds whenever `f` is integrable (then mathlib's
`MeasureTheory.convolution.aestronglyMeasurable` applies), or
whenever `eLpNorm (conv f) ⊤ volume < ⊤` (then `MemLp.aestronglyMeasurable`
applies after building the `MemLp` certificate).  We expose it as a
hypothesis to keep this lemma free of namespace-specific imports. -/
lemma conv_K_ms_integrable {f : ℝ → ℝ} (hf : ∀ x, 0 ≤ f x)
    (h_conv_meas : AEStronglyMeasurable (conv f) volume)
    (h_conv_fin : eLpNorm (conv f) ⊤ volume ≠ ⊤) :
    Integrable (fun x => conv f x * Sidon.MultiScale.K_ms x) volume := by
  -- f*f is bounded a.e. by C = (eLpNorm (f*f) ⊤ volume).toReal.
  set C : ℝ := (eLpNorm (conv f) ⊤ volume).toReal
  have h_bound : ∀ᵐ x ∂volume, ‖conv f x‖ ≤ C := by
    have h := conv_ae_le_eLpNorm_toReal hf h_conv_fin
    filter_upwards [h] with x hx
    have h_nonneg : 0 ≤ conv f x := conv_nonneg hf x
    rw [Real.norm_of_nonneg h_nonneg]
    exact hx
  -- K_ms is integrable.
  have h_K_int : Integrable Sidon.MultiScale.K_ms volume := K_ms_integrable
  -- Apply Integrable.bdd_mul with `f := conv f` (bounded a.e. by C) and
  -- `g := K_ms` (integrable).
  exact Integrable.bdd_mul h_K_int h_conv_meas h_bound

/-- **Unnormalised form.**  The MV Lemma 3.1 Eq.(1) discharge
for `(f, K_ms)`, in the natural unnormalised form
`∫(f*f)·K_ms ≤ ‖f*f‖_∞.toReal`.

This is the form directly produced by `Sidon.MV.mv_eq1`; the
bundle's `hEq1` (`LHS1 ≤ R(f)`) follows under the additional
hypothesis `∫f = 1`. -/
theorem hEq1_unnormalised (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (_hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (_hf_int : Integrable f volume)
    (h_conv_fin : eLpNorm (conv f) ⊤ volume ≠ ⊤)
    (h_prod_int :
      Integrable (fun x => conv f x * Sidon.MultiScale.K_ms x) volume) :
    LHS1 f ≤ (eLpNorm (conv f) ⊤ volume).toReal := by
  unfold LHS1
  -- Apply Sidon.MV.mv_eq1 with K = K_ms and Minf = (eLpNorm (f*f) ⊤ volume).toReal.
  set Minf : ℝ := (eLpNorm (conv f) ⊤ volume).toReal
  have hM_nn : 0 ≤ Minf := eLpNorm_toReal_nonneg f
  have h_K_nn : ∀ x, 0 ≤ Sidon.MultiScale.K_ms x := Sidon.MultiScale.K_ms_nonneg
  have h_K_int : Integrable Sidon.MultiScale.K_ms volume := K_ms_integrable
  have h_K_one : ∫ x, Sidon.MultiScale.K_ms x ∂volume = 1 := K_ms_integral_eq_one
  have h_ae : ∀ᵐ x ∂volume, conv f x ≤ Minf :=
    conv_ae_le_eLpNorm_toReal hf_nonneg h_conv_fin
  -- Unfold `conv` to apply `mv_eq1` directly.
  show ∫ x, conv f x * Sidon.MultiScale.K_ms x ∂volume ≤ Minf
  exact Sidon.MV.mv_eq1 f Sidon.MultiScale.K_ms
    hf_nonneg h_K_nn h_K_int h_K_one Minf hM_nn h_prod_int h_ae

/-- **Normalised form.**  The MV Lemma 3.1 Eq.(1) discharge in
the bundle form `LHS1 ≤ R(f)`, valid under the normalisation
`∫f = 1` (the bundle's implicit convention).

Proof: by `hEq1_unnormalised`, `LHS1 ≤ (eLpNorm (f*f) ⊤ volume).toReal`.
Under `∫f = 1`, `autoconvolution_ratio f = (eLpNorm (f*f) ⊤ volume).toReal / 1 =
(eLpNorm (f*f) ⊤ volume).toReal`, so the conclusion follows. -/
theorem hEq1_discharge (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int : Integrable f volume)
    (hf_int_one : ∫ x, f x ∂volume = 1)
    (h_conv_fin : eLpNorm (conv f) ⊤ volume ≠ ⊤)
    (h_prod_int :
      Integrable (fun x => conv f x * Sidon.MultiScale.K_ms x) volume) :
    LHS1 f ≤ autoconvolution_ratio f := by
  -- Unfold autoconvolution_ratio:
  -- autoconvolution_ratio f = (eLpNorm (f*f) ⊤ volume).toReal / (∫f)²
  unfold autoconvolution_ratio
  -- With ∫f = 1, (∫f)² = 1, so the ratio equals the norm.
  rw [hf_int_one]
  simp only [one_pow, div_one]
  -- Now goal: LHS1 f ≤ (eLpNorm (conv f) ⊤ volume).toReal
  -- (after unfolding the local `convolution` and `eLpNorm` to match).
  -- Note autoconvolution_ratio unfolds `let conv := ...` so the
  -- expression has the spelled-out convolution; our `conv` is the
  -- same via `conv_def`.
  exact hEq1_unnormalised f hf_nonneg hf_supp hf_int h_conv_fin h_prod_int

/-- **Scale-invariant form.**  The MV Lemma 3.1 Eq.(1) discharge
in the dimensionally consistent form `LHS1 ≤ R(f) · (∫f)²`, valid for
any admissible nonneg `f` with strictly positive integral and finite
convolution.

This form does not require the normalisation `∫f = 1`; it expresses
the inequality in terms of the homogeneous-of-degree-2 product
`R(f) · (∫f)²`, which by definition equals `‖f*f‖_∞.toReal`.

Useful for integration when working with un-normalised admissible
functions. -/
theorem hEq1_general (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int : Integrable f volume)
    (hf_int_pos : 0 < ∫ x, f x ∂volume)
    (h_conv_fin : eLpNorm (conv f) ⊤ volume ≠ ⊤)
    (h_prod_int :
      Integrable (fun x => conv f x * Sidon.MultiScale.K_ms x) volume) :
    LHS1 f ≤ autoconvolution_ratio f * (∫ x, f x ∂volume) ^ 2 := by
  -- Unfold autoconvolution_ratio:
  -- autoconvolution_ratio f * (∫f)² = (‖f*f‖_∞.toReal / (∫f)²) * (∫f)²
  --                                 = ‖f*f‖_∞.toReal  (since ∫f > 0).
  have h_sq_pos : (0 : ℝ) < (∫ x, f x ∂volume) ^ 2 := by positivity
  have h_sq_ne : (∫ x, f x ∂volume) ^ 2 ≠ 0 := ne_of_gt h_sq_pos
  unfold autoconvolution_ratio
  -- Goal: LHS1 f ≤ ((eLpNorm (conv f) ⊤ volume).toReal / (∫f)²) * (∫f)²
  rw [div_mul_cancel₀ _ h_sq_ne]
  -- Goal: LHS1 f ≤ (eLpNorm (conv f) ⊤ volume).toReal
  exact hEq1_unnormalised f hf_nonneg hf_supp hf_int h_conv_fin h_prod_int

end Sidon.BundleEq1
