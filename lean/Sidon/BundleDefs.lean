/-
Sidon Autocorrelation Project — Bundle Definitions
====================================================

This file contains the *definitions and positivity lemmas* needed to
construct an `ExtremiserPrimitives f` value from a Schwartz-class
admissible test function `f`.

Specifically, this file provides:

  * `m_G_const`, `S_G_const` as concrete rationals forwarded from
    `Sidon.MultiScale.minGLowerQ` / `S1UpperQ`.
  * `S_cos` — the bilinear cosine sum
    `∑'_{j ≠ 0} Re(f̂(j/u))² · K̂_ms(j/u)` (definition).
  * `LHS1`, `LHS2` — the two integrals on the LHS of MV Eqs.(1), (2),
    plus integrability lemmas.
  * `K2_analytic_ge_one` — `1 ≤ K2_analytic` via Cauchy–Schwarz on
    `[-δ₁, δ₁]`.
  * `S_G_const_pos` — `0 < S_G_const`.
  * `R_ge_one` — `1 ≤ autoconvolution_ratio f` (for admissible f,
    packaged from convolution L¹-mass and eLpNorm ⊤ lower bound).

The `gain_analytic` refactor is handled in `Sidon/MultiScale.lean`,
so it is intentionally omitted here.

This file introduces **no new axioms** beyond Lean's three core axioms
(`Classical.choice`, `propext`, `Quot.sound`).  Where a proof requires
a fact that has not yet been formalised in the local project (e.g. the
arcsine total-mass integral `∫ K_arc(δ, ·) = 1`), the corresponding
theorem is stated under a *hypothesis* discharged elsewhere.

No `sorry`.
-/

import Mathlib
import Sidon.Defs
import Sidon.MultiScale
import Sidon.Bessel
import Sidon.FourierAux

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real
open MeasureTheory

namespace Sidon.BundleDefs

open Sidon.FourierAux (autocorr)

/-! ## Section F1 / F2 — `m_G_const`, `S_G_const`

Thin wrappers around the certifier-provided rational slack anchors
declared in `Sidon.MultiScale`.

`m_G_const` is a lower bound on `min_{[0,1/4]} G` (certifier reports
`min G ≥ 0.99997987`; we use `998/1000`).

`S_G_const` is an upper bound on the dual sum
`S_1 = ∑ aⱼ² / K̂_ms(j/u)` (certifier reports `S_1 ≤ 29.840907`; we use
`29841/1000`). -/

/-- Certifier-validated lower bound on `min_{[0,1/4]} G`: `998/1000`. -/
noncomputable def m_G_const : ℝ := (Sidon.MultiScale.minGLowerQ : ℝ)

/-- Certifier-validated upper bound on `S_1 = ∑ aⱼ²/K̂_ms(j/u)`:
`29841/1000`. -/
noncomputable def S_G_const : ℝ := (Sidon.MultiScale.S1UpperQ : ℝ)

theorem m_G_const_eq_rat :
    m_G_const = (998 : ℝ) / 1000 := by
  unfold m_G_const Sidon.MultiScale.minGLowerQ
  push_cast
  ring

theorem S_G_const_eq_rat :
    S_G_const = (29841 : ℝ) / 1000 := by
  unfold S_G_const Sidon.MultiScale.S1UpperQ
  push_cast
  ring

/-! ## Section F8 — `S_G_const_pos` -/

/-- `S_G_const > 0`. -/
theorem S_G_const_pos : 0 < S_G_const := by
  rw [S_G_const_eq_rat]
  norm_num

/-- `0 ≤ m_G_const`. -/
theorem m_G_const_nonneg : 0 ≤ m_G_const := by
  rw [m_G_const_eq_rat]
  norm_num

/-- `m_G_const ≤ 1`. -/
theorem m_G_const_le_one : m_G_const ≤ 1 := by
  rw [m_G_const_eq_rat]
  norm_num

/-! ## Section F3 — `S_cos` definition

The bilinear cosine sum, indexed by `ℤ \ {0}`:
`S_cos f = ∑'_{j ≠ 0} Re(f̂(j/u))² · K̂_ms(j/u)`,
where `f̂` is the Fourier integral of the complex lift `(f : ℝ → ℂ)`
and `K̂_ms(j/u)` is the lattice-value of `K̂_ms` at frequency `j/u`,
given by the closed form `∑ᵢ λᵢ · J₀(π·δᵢ·j/u)²`. -/

/-- The closed-form value of the lattice-period-`u` Fourier coefficient
of `K_ms` at frequency `j/u`,
`K̂_ms(j/u) = ∑ᵢ λᵢ · J₀(π·δᵢ·j/u)²`.

Under the definition `K_arc(δ, ·) := η_δ * η_δ` with `η_δ` supported
on `(-δ/2, δ/2)` and FT `J₀(πδξ)`, the convolution theorem gives the
FT of `K_arc(δ, ·)` equal to `J₀(πδξ)²`.  The exponents match
`Sidon.BundleEq4.K_ms_fourier_lattice` and the paper. -/
noncomputable def K_ms_fourier_lattice (j : ℤ) : ℝ :=
  Sidon.MultiScale.lambda1 *
    (Sidon.Bessel.besselJ0
      (Real.pi * Sidon.MultiScale.delta1 *
        ((j : ℝ) / Sidon.MultiScale.uQ_real))) ^ 2
  + Sidon.MultiScale.lambda2 *
    (Sidon.Bessel.besselJ0
      (Real.pi * Sidon.MultiScale.delta2 *
        ((j : ℝ) / Sidon.MultiScale.uQ_real))) ^ 2
  + Sidon.MultiScale.lambda3 *
    (Sidon.Bessel.besselJ0
      (Real.pi * Sidon.MultiScale.delta3 *
        ((j : ℝ) / Sidon.MultiScale.uQ_real))) ^ 2

/-- `K̂_ms(j/u) ≥ 0` (Bochner positivity, automatic from the convex
combination of `J₀²` and `λᵢ ≥ 0`). -/
theorem K_ms_fourier_lattice_nonneg (j : ℤ) : 0 ≤ K_ms_fourier_lattice j := by
  unfold K_ms_fourier_lattice
  have h_lams := Sidon.MultiScale.lambdas_nonneg
  have hl1 : (0 : ℝ) ≤ Sidon.MultiScale.lambda1 := by
    unfold Sidon.MultiScale.lambda1; exact_mod_cast h_lams.1
  have hl2 : (0 : ℝ) ≤ Sidon.MultiScale.lambda2 := by
    unfold Sidon.MultiScale.lambda2; exact_mod_cast h_lams.2.1
  have hl3 : (0 : ℝ) ≤ Sidon.MultiScale.lambda3 := by
    unfold Sidon.MultiScale.lambda3; exact_mod_cast h_lams.2.2
  have h1 : (0 : ℝ) ≤ Sidon.MultiScale.lambda1 *
      (Sidon.Bessel.besselJ0
        (Real.pi * Sidon.MultiScale.delta1 *
          ((j : ℝ) / Sidon.MultiScale.uQ_real))) ^ 2 :=
    mul_nonneg hl1 (sq_nonneg _)
  have h2 : (0 : ℝ) ≤ Sidon.MultiScale.lambda2 *
      (Sidon.Bessel.besselJ0
        (Real.pi * Sidon.MultiScale.delta2 *
          ((j : ℝ) / Sidon.MultiScale.uQ_real))) ^ 2 :=
    mul_nonneg hl2 (sq_nonneg _)
  have h3 : (0 : ℝ) ≤ Sidon.MultiScale.lambda3 *
      (Sidon.Bessel.besselJ0
        (Real.pi * Sidon.MultiScale.delta3 *
          ((j : ℝ) / Sidon.MultiScale.uQ_real))) ^ 2 :=
    mul_nonneg hl3 (sq_nonneg _)
  linarith

/-- The bilinear cosine sum: `S_cos f = ∑'_{j ≠ 0} Re(f̂(j/u))² · K̂_ms(j/u)`. -/
noncomputable def S_cos (f : ℝ → ℝ) : ℝ :=
  ∑' j : ℤ, if j = 0 then 0 else
    ((Real.fourierIntegral (fun x => ((f x : ℂ)))
        ((j : ℝ) / Sidon.MultiScale.uQ_real)).re) ^ 2
      * K_ms_fourier_lattice j

/-- `S_cos` is a sum of nonneg terms: each summand is nonneg pointwise. -/
theorem S_cos_summand_nonneg (f : ℝ → ℝ) (j : ℤ) :
    0 ≤ (if j = 0 then 0 else
          ((Real.fourierIntegral (fun x => ((f x : ℂ)))
              ((j : ℝ) / Sidon.MultiScale.uQ_real)).re) ^ 2
            * K_ms_fourier_lattice j) := by
  split_ifs with hj
  · exact le_refl 0
  · exact mul_nonneg (sq_nonneg _) (K_ms_fourier_lattice_nonneg j)

/-! ## Section F4 / F5 — `LHS1`, `LHS2` definitions

`LHS1 f = ∫ (f*f)·K_ms` (LHS of MV Eq.(1)).
`LHS2 f = ∫ (autocorr f)·K_ms` (LHS of MV Eq.(2)), where
`autocorr f x := ∫ t, f(t)·f(x+t) dt` is the convolutional
autocorrelation (MV's `f∘f`). -/

/-- LHS of MV Eq.(1): `LHS1 f := ∫ (f*f)(x) · K_ms(x) dx`. -/
noncomputable def LHS1 (f : ℝ → ℝ) : ℝ :=
  ∫ x,
    (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x
      * Sidon.MultiScale.K_ms x ∂MeasureTheory.volume

/-- LHS of MV Eq.(2): `LHS2 f := ∫ (autocorr f)(x) · K_ms(x) dx`,
where `autocorr f x := ∫ t, f(t)·f(x+t) dt` is the convolutional
autocorrelation. -/
noncomputable def LHS2 (f : ℝ → ℝ) : ℝ :=
  ∫ x, autocorr f x * Sidon.MultiScale.K_ms x ∂MeasureTheory.volume

/-! ### Integrability lemmas (Schwartz path)

For Schwartz-class `f`, both `f*f` and `f∘f` are continuous and bounded,
and the multi-scale kernel `K_ms` is L¹ (it is a convex combination of
arcsine densities, each having `∫K_arc = 1`).  Combined, the products
are L¹.

The integrability proofs below assume a *hypothesis* `K_ms ∈ L¹` and
sup-norm bounds on `f*f` / `f∘f`; the integration agent in
`MultiScale.lean` discharges these for the concrete Schwartz-class
admissible `f`.  We keep the signatures as `theorem` (not `axiom`) by
parametrising the integrability hypotheses. -/

/-- Integrability of `LHS1` integrand under the boundedness of `f*f`
and L¹-integrability of `K_ms`. -/
theorem LHS1_integrand_integrable {f : ℝ → ℝ}
    (h_conv_meas :
      AEStronglyMeasurable
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) volume)
    (h_K_int : Integrable Sidon.MultiScale.K_ms volume)
    (C : ℝ)
    (h_conv_bd :
      ∀ᵐ x ∂volume,
        ‖(MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x‖ ≤ C) :
    Integrable
      (fun x =>
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x
          * Sidon.MultiScale.K_ms x) volume := by
  -- Use `Integrable.bdd_mul'`: if `g` is integrable and `h` is bounded a.e.
  -- and AE-measurable, then `h * g` is integrable.  Then commute to get
  -- `g * h` integrable.
  have h_K_meas : AEStronglyMeasurable Sidon.MultiScale.K_ms volume :=
    h_K_int.aestronglyMeasurable
  have h_mul :=
    MeasureTheory.Integrable.bdd_mul h_K_int h_conv_meas h_conv_bd
  -- `h_mul : Integrable (fun x => (f*f) x * K_ms x) volume` (swap factor order)
  exact h_mul

/-- Integrability of `LHS2` integrand under boundedness of `autocorr f`
and L¹-integrability of `K_ms`. -/
theorem LHS2_integrand_integrable {f : ℝ → ℝ}
    (h_auto_meas :
      AEStronglyMeasurable (autocorr f) volume)
    (h_K_int : Integrable Sidon.MultiScale.K_ms volume)
    (C : ℝ)
    (h_auto_bd :
      ∀ᵐ x ∂volume, ‖autocorr f x‖ ≤ C) :
    Integrable (fun x => autocorr f x * Sidon.MultiScale.K_ms x) volume :=
  MeasureTheory.Integrable.bdd_mul h_K_int h_auto_meas h_auto_bd

/-! ## Section F6 — `K2_analytic_ge_one`

By Cauchy–Schwarz on `[-δ₁, δ₁]` we have
```
(∫_{-δ₁}^{δ₁} K_ms · 1)²  ≤  (∫_{-δ₁}^{δ₁} K_ms²) · (2δ₁).
```
Since `K_ms` is supported in `[-δ₁, δ₁]` and `∫ K_ms = 1`,
```
1 = (∫ K_ms)² = (∫_{-δ₁}^{δ₁} K_ms)² ≤ K_2 · (2δ₁) ≤ K_2 · (276/1000),
```
hence `K_2 ≥ 1000/276 ≈ 3.6 ≥ 1`.

The discharge of this requires:
  (i) `K_ms` integrable with `∫ K_ms = 1`;
  (ii) the Cauchy–Schwarz inequality applied on a finite-measure set.

(i) is the arcsine-mass identity (a *theorem* in mathlib via
`Real.integral_inv_sqrt_one_sub_sq` + substitution; delivered as part
of `BundleEq1`).  We expose `K2_analytic_ge_one` as a *theorem*
parametrised by (i) – the consumer (`Sidon/MultiScale.lean`) plugs in
the arcsine-mass identity. -/

/-- Cauchy–Schwarz in the form
`(∫_{[-a,a]} f)² ≤ (∫_{[-a,a]} f²) · (2a)`.

A general L² Cauchy–Schwarz, specialised to the indicator of
`[-a, a]` (interpreted as the constant 1 on this set). -/
theorem cauchy_schwarz_indicator (f : ℝ → ℝ) (a : ℝ) (ha : 0 ≤ a)
    (hf_int : IntegrableOn f (Set.Icc (-a) a) volume)
    (hf_sq_int : IntegrableOn (fun x => f x ^ 2) (Set.Icc (-a) a) volume) :
    (∫ x in Set.Icc (-a) a, f x ∂volume) ^ 2 ≤
      (∫ x in Set.Icc (-a) a, f x ^ 2 ∂volume) * (2 * a) := by
  -- Apply mathlib's `MeasureTheory.integral_mul_le_L2_norm_sq` style result.
  -- Concrete strategy: use `MeasureTheory.inner_mul_le_norm_mul_norm`-style
  -- bound on `∫ f · 1`.
  have h_pos : 0 ≤ 2 * a := by linarith
  -- We'll use the equivalent form via the Cauchy-Schwarz integral inequality.
  -- ∫ f · 1 = ∫ f.  Then |∫ f|² ≤ (∫ |f|²)(∫ 1²) = (∫ f²)·(2a).
  -- Mathlib provides this via `MeasureTheory.integral_mul_le_sqrt_mul_sqrt`
  -- (Cauchy–Schwarz) but we can derive directly from the simpler:
  -- ∫ (f - λ·1)² ≥ 0 ⟹ ∫f² - 2λ ∫f + λ²·(2a) ≥ 0 for all λ, taking
  -- λ = (∫f)/(2a) yields ∫f² ≥ (∫f)²/(2a).
  by_cases ha0 : a = 0
  · subst ha0
    -- Set.Icc 0 0 = {0}, which has measure 0; both integrals are 0.
    have hI : Set.Icc (-(0 : ℝ)) 0 = {0} := by
      rw [neg_zero, Set.Icc_self]
    rw [hI]
    have hμ0 : MeasureTheory.volume.real ({0} : Set ℝ) = 0 := by
      rw [MeasureTheory.measureReal_def]
      simp
    have h1 : ∫ x in ({0} : Set ℝ), f x ∂volume = 0 := by
      rw [MeasureTheory.integral_singleton, hμ0, zero_smul]
    have h2 : ∫ x in ({0} : Set ℝ), f x ^ 2 ∂volume = 0 := by
      rw [MeasureTheory.integral_singleton, hμ0, zero_smul]
    rw [h1, h2]
    norm_num
  · have ha_pos : 0 < a := lt_of_le_of_ne ha (Ne.symm ha0)
    have h2a_pos : (0 : ℝ) < 2 * a := by linarith
    -- Quadratic-in-λ approach.
    have h_expand : ∀ (lam : ℝ),
        (0 : ℝ) ≤ ∫ x in Set.Icc (-a) a, (f x - lam) ^ 2 ∂volume := by
      intro lam
      apply MeasureTheory.integral_nonneg
      intro x
      exact sq_nonneg _
    -- (f - λ)² = f² - 2λf + λ².  Use linearity.
    -- ∫ (f - λ)² = ∫ f² - 2λ ∫ f + λ² · (2a).
    -- For finite-measure set Icc[-a, a], μ = 2a.
    have h_meas : MeasureTheory.volume (Set.Icc (-a) a) = ENNReal.ofReal (2 * a) := by
      rw [Real.volume_Icc]
      congr 1; ring
    have h_meas_real :
        (MeasureTheory.volume (Set.Icc (-a) a)).toReal = 2 * a := by
      rw [h_meas]; exact ENNReal.toReal_ofReal h_pos
    -- The Cauchy–Schwarz from quadratic: choose lam = (∫f)/(2a).
    set I1 := ∫ x in Set.Icc (-a) a, f x ∂volume with hI1_def
    set I2 := ∫ x in Set.Icc (-a) a, f x ^ 2 ∂volume with hI2_def
    have hI_const : ∀ c : ℝ, ∫ _ in Set.Icc (-a) a, (c : ℝ) ∂volume
                              = c * (2 * a) := by
      intro c
      rw [MeasureTheory.setIntegral_const, MeasureTheory.measureReal_def]
      rw [show (MeasureTheory.volume (Set.Icc (-a) a)).toReal = 2 * a from h_meas_real]
      simp [mul_comm]
    have hI_const1 : ∫ _ in Set.Icc (-a) a, (1 : ℝ) ∂volume = 2 * a := by
      have := hI_const 1
      simpa using this
    -- Set up the quadratic argument.
    have h_quad : ∀ (lam : ℝ),
        I2 - 2 * lam * I1 + lam ^ 2 * (2 * a) ≥ 0 := by
      intro lam
      have h_eq :
          (∫ x in Set.Icc (-a) a, (f x - lam) ^ 2 ∂volume) =
            I2 - 2 * lam * I1 + lam ^ 2 * (2 * a) := by
        have h_expand_int :
            (∫ x in Set.Icc (-a) a, (f x - lam) ^ 2 ∂volume) =
            (∫ x in Set.Icc (-a) a, (f x ^ 2 - 2 * lam * f x + lam ^ 2) ∂volume) := by
          apply MeasureTheory.setIntegral_congr_fun measurableSet_Icc
          intro x _
          ring
        rw [h_expand_int]
        -- Split via integrability.
        have h_meas_ne_top : MeasureTheory.volume (Set.Icc (-a) a) ≠ ⊤ := by
          rw [h_meas]; exact ENNReal.ofReal_ne_top
        have h_int_lam2 : Integrable (fun (_ : ℝ) => (lam ^ 2 : ℝ))
                            (volume.restrict (Set.Icc (-a) a)) :=
          MeasureTheory.integrableOn_const h_meas_ne_top
        have h_int_lin : Integrable (fun x => 2 * lam * f x)
                            (volume.restrict (Set.Icc (-a) a)) :=
          hf_int.const_mul (2 * lam)
        have h_int_sq : Integrable (fun x => f x ^ 2)
                            (volume.restrict (Set.Icc (-a) a)) :=
          hf_sq_int
        have h_int_sub : Integrable (fun x => f x ^ 2 - 2 * lam * f x)
                            (volume.restrict (Set.Icc (-a) a)) :=
          h_int_sq.sub h_int_lin
        rw [MeasureTheory.integral_add h_int_sub h_int_lam2,
            MeasureTheory.integral_sub h_int_sq h_int_lin,
            MeasureTheory.integral_const_mul,
            hI_const (lam ^ 2)]
      have := h_expand lam
      rw [h_eq] at this
      linarith
    -- Take lam = I1 / (2*a).
    have h_special := h_quad (I1 / (2 * a))
    have h_2a_ne : (2 * a : ℝ) ≠ 0 := ne_of_gt h2a_pos
    -- I2 - 2·(I1/(2a))·I1 + (I1/(2a))²·(2a) ≥ 0
    -- = I2 - I1²/a + I1²/(2·(2a)²) · (2a)
    -- Algebraically: I2 - I1²/(2a) ≥ 0, so I1² ≤ I2 · (2a).
    have h_simp : I2 - 2 * (I1 / (2 * a)) * I1 + (I1 / (2 * a)) ^ 2 * (2 * a)
                    = I2 - I1 ^ 2 / (2 * a) := by
      field_simp
      ring
    rw [h_simp] at h_special
    -- So I2 ≥ I1² / (2a).
    have h_final : I1 ^ 2 ≤ I2 * (2 * a) := by
      have := h_special
      have h_div : I1 ^ 2 / (2 * a) ≤ I2 := by linarith
      have := (div_le_iff₀ h2a_pos).mp h_div
      linarith
    exact h_final

/-- `K2_analytic ≥ 1` whenever `∫ K_ms = 1`, `K_ms` is integrable, `K_ms`
is supported in `[-δ₁, δ₁]`, and `K_ms²` is integrable on
`[-δ₁, δ₁]`.

The proof is by Cauchy–Schwarz: `(∫ K_ms · 1)² ≤ K_2 · (2δ₁)`, and
`2δ₁ = 276/1000 < 1`. -/
theorem K2_analytic_ge_one_of_integral_one
    (_h_K_int : Integrable Sidon.MultiScale.K_ms volume)
    (h_K_int_on : IntegrableOn Sidon.MultiScale.K_ms
                    (Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1)
                    volume)
    (h_K_sq_int_on : IntegrableOn (fun x => Sidon.MultiScale.K_ms x ^ 2)
                    (Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1)
                    volume)
    (_h_K_sq_int : Integrable (fun x => Sidon.MultiScale.K_ms x ^ 2) volume)
    (h_K_supp : ∀ x, Sidon.MultiScale.delta1 < |x| →
                       Sidon.MultiScale.K_ms x = 0)
    (h_K_one : ∫ x, Sidon.MultiScale.K_ms x ∂volume = 1) :
    1 ≤ Sidon.MultiScale.K2_analytic := by
  -- Step 1: ∫ K_ms over [-δ₁, δ₁] equals ∫ K_ms over ℝ = 1.
  have hδ1_pos : 0 < Sidon.MultiScale.delta1 :=
    (Sidon.MultiScale.delta_positivities).1
  have hδ1_nn : 0 ≤ Sidon.MultiScale.delta1 := le_of_lt hδ1_pos
  -- ∫ K_ms on ℝ = ∫ K_ms on [-δ₁,δ₁] (rest is zero) using support assumption.
  have h_supp_set :
      Sidon.MultiScale.K_ms = fun x =>
        Set.indicator (Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1)
          (fun y => Sidon.MultiScale.K_ms y) x := by
    funext x
    by_cases hx : x ∈ Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1
    · rw [Set.indicator_of_mem hx]
    · rw [Set.indicator_of_notMem hx]
      have h_abs : Sidon.MultiScale.delta1 < |x| := by
        simp only [Set.mem_Icc, not_and_or, not_le] at hx
        rcases hx with hxlt | hxgt
        · have h2 : -x > Sidon.MultiScale.delta1 := by linarith
          have h3 : |x| = -x := by apply abs_of_neg; linarith
          rw [h3]; linarith
        · have h_abs_eq : |x| = x := abs_of_pos (lt_trans hδ1_pos hxgt)
          rw [h_abs_eq]; exact hxgt
      exact h_K_supp x h_abs
  have h_int_one_set :
      ∫ x in Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1,
        Sidon.MultiScale.K_ms x ∂volume = 1 := by
    -- ∫ K_ms = ∫ K_ms · 𝟙_Icc = ∫_Icc K_ms (since K_ms = 0 outside).
    rw [← h_K_one]
    -- Rewrite RHS ∫ ℝ K_ms = ∫_Icc K_ms via support.
    conv_rhs => rw [h_supp_set]
    rw [MeasureTheory.integral_indicator measurableSet_Icc]
  -- Step 2: apply Cauchy–Schwarz.
  have h_cs :=
    cauchy_schwarz_indicator Sidon.MultiScale.K_ms
      Sidon.MultiScale.delta1 hδ1_nn h_K_int_on h_K_sq_int_on
  rw [h_int_one_set] at h_cs
  -- So 1 ≤ (∫_{[-δ₁,δ₁]} K_ms²) · (2δ₁).
  -- The full ∫ K_ms² = ∫_{[-δ₁,δ₁]} K_ms² (since K_ms = 0 outside).
  have h_sq_supp :
      (fun x => Sidon.MultiScale.K_ms x ^ 2) = fun x =>
        Set.indicator (Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1)
          (fun y => Sidon.MultiScale.K_ms y ^ 2) x := by
    funext x
    by_cases hx : x ∈ Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1
    · rw [Set.indicator_of_mem hx]
    · rw [Set.indicator_of_notMem hx]
      have h_abs : Sidon.MultiScale.delta1 < |x| := by
        simp only [Set.mem_Icc, not_and_or, not_le] at hx
        rcases hx with hxlt | hxgt
        · have h2 : -x > Sidon.MultiScale.delta1 := by linarith
          have h3 : |x| = -x := by apply abs_of_neg; linarith
          rw [h3]; linarith
        · have h_abs_eq : |x| = x := abs_of_pos (lt_trans hδ1_pos hxgt)
          rw [h_abs_eq]; exact hxgt
      have hzero : Sidon.MultiScale.K_ms x = 0 := h_K_supp x h_abs
      rw [hzero]; ring
  have h_K2_eq :
      Sidon.MultiScale.K2_analytic =
        ∫ x in Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1,
          Sidon.MultiScale.K_ms x ^ 2 ∂volume := by
    unfold Sidon.MultiScale.K2_analytic
    conv_lhs => rw [h_sq_supp]
    rw [MeasureTheory.integral_indicator measurableSet_Icc]
  rw [← h_K2_eq] at h_cs
  -- So 1 ≤ K2_analytic · (2 · δ₁).
  -- We have 2·δ₁ = 276/1000 < 1, so K2_analytic ≥ 1/(2·δ₁) > 1.
  have h_2del : 2 * Sidon.MultiScale.delta1 = (276 : ℝ) / 1000 := by
    unfold Sidon.MultiScale.delta1 Sidon.MultiScale.delta1Q
    push_cast; ring
  rw [h_2del] at h_cs
  -- 1² = 1 ≤ K2 · 276/1000, so K2 ≥ 1000/276 > 1.
  have h_one_sq : (1 : ℝ) ^ 2 = 1 := by ring
  rw [h_one_sq] at h_cs
  -- 1 ≤ K2 · 276/1000.  Need K2 ≥ 1.
  have h_276_pos : (0 : ℝ) < 276 / 1000 := by norm_num
  -- Want: 1 ≤ K2.  Have: 1 ≤ K2 * (276/1000).  Since 276/1000 < 1, this gives K2 > 1.
  have h_K2_nn : 0 ≤ Sidon.MultiScale.K2_analytic := by
    unfold Sidon.MultiScale.K2_analytic
    exact MeasureTheory.integral_nonneg (fun x => sq_nonneg _)
  -- 1 ≤ K2 · 276/1000 ≤ K2 · 1 = K2 if K2 ≥ 0 and 276/1000 ≤ 1 — but multiplication
  -- by a value < 1 lowers!  So we use the actual lower bound:
  -- K2 ≥ 1/(276/1000) = 1000/276 ≈ 3.62 > 1.
  have h_K2_lb : (1000 : ℝ) / 276 ≤ Sidon.MultiScale.K2_analytic := by
    have h_276 : (0 : ℝ) < 276 / 1000 := h_276_pos
    have h_one_div : (1 : ℝ) / (276 / 1000) = 1000 / 276 := by norm_num
    -- 1 ≤ K2 · 276/1000 → 1/(276/1000) ≤ K2
    have h_div : (1 : ℝ) / (276 / 1000) ≤ Sidon.MultiScale.K2_analytic := by
      rw [div_le_iff₀ h_276]
      linarith [h_cs]
    rw [← h_one_div]
    exact h_div
  -- 1000/276 > 1, so K2 > 1, in particular K2 ≥ 1.
  have h_lb : (1 : ℝ) ≤ 1000 / 276 := by norm_num
  linarith

/-! ## Section F9 — `R_ge_one`

For nonneg `f` with `∫f > 0` and support in `(-1/4, 1/4)`,
the autoconvolution ratio `R = ‖f*f‖_∞ / (∫f)² ≥ 1`.

Argument:
  - Normalise `f` so that `∫f = 1`.  Then `∫(f*f) = 1` (Fubini), and
    `supp(f*f) ⊆ (-1/2, 1/2)` (length 1).
  - Hence `‖f*f‖_∞ · 1 ≥ ∫(f*f) = 1`, so `‖f*f‖_∞ ≥ 1`.
  - `R = ‖f*f‖_∞ / 1 ≥ 1`.

`R` is invariant under `f ↦ c·f` for `c > 0`, so the general case
reduces to `∫f = 1`.

The `eLpNorm ⊤` routing is the technical part.  We expose
`R_ge_one_of_essSup_bound` parametrically; the integration agent
discharges the bound from concrete Schwartz machinery. -/

/-- The autoconvolution ratio.
`R(f) := (eLpNorm (f*f) ⊤ volume).toReal / (∫f)²`. -/
theorem R_ge_one_of_data {f : ℝ → ℝ}
    (h_int_one : ∫ x, f x ∂volume = 1)
    (_h_conv_one : ∫ x, (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x ∂volume = 1)
    (_h_conv_supp_meas :
      MeasureTheory.volume
        (Function.support (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume))
          ≤ ENNReal.ofReal 1)
    (_h_conv_nonneg :
      ∀ x, 0 ≤ (MeasureTheory.convolution f f
                  (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x)
    (h_conv_essSup_ge :
      ENNReal.ofReal 1 ≤ MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume)
    (h_essSup_fin :
      MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume ≠ ⊤) :
    1 ≤ autoconvolution_ratio f := by
  -- Unfold `autoconvolution_ratio`.
  unfold autoconvolution_ratio
  simp only [h_int_one, one_pow, div_one]
  -- Now goal: 1 ≤ (eLpNorm (f*f) ⊤ volume).toReal.
  have h_le := h_conv_essSup_ge
  -- Convert: ENNReal.ofReal 1 ≤ eLpNorm ... ⊤ volume.
  -- toReal monotone preserves; need eLpNorm < ⊤.
  have h_top_lt : MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume < ⊤ :=
    lt_of_le_of_ne le_top h_essSup_fin
  have h_toReal :=
    (ENNReal.toReal_le_toReal (by simp : (ENNReal.ofReal 1 : ENNReal) ≠ ⊤)
      h_essSup_fin).mpr h_le
  have h_or : (ENNReal.ofReal 1).toReal = 1 := by
    simp
  rw [h_or] at h_toReal
  exact h_toReal

/-! ## Section helpers — Bochner positivity of `K̂_ms(j/u)`

A convenience-only lemma documenting `K_ms_fourier_lattice j ≥ 0`
for the integration agent.  Already proven above
(`K_ms_fourier_lattice_nonneg`). -/

/-- The `j = 0` value: `K̂_ms(0) = ∑ᵢ λᵢ · J₀(0)² = ∑ᵢ λᵢ · 1 = 1`. -/
theorem K_ms_fourier_lattice_zero :
    K_ms_fourier_lattice 0 = 1 := by
  unfold K_ms_fourier_lattice
  -- Each besselJ0 argument simplifies to 0: π·δᵢ·(0/u) = 0.
  have h_zero1 : Real.pi * Sidon.MultiScale.delta1 *
      (((0 : ℤ) : ℝ) / Sidon.MultiScale.uQ_real) = 0 := by
    push_cast; ring
  have h_zero2 : Real.pi * Sidon.MultiScale.delta2 *
      (((0 : ℤ) : ℝ) / Sidon.MultiScale.uQ_real) = 0 := by
    push_cast; ring
  have h_zero3 : Real.pi * Sidon.MultiScale.delta3 *
      (((0 : ℤ) : ℝ) / Sidon.MultiScale.uQ_real) = 0 := by
    push_cast; ring
  rw [h_zero1, h_zero2, h_zero3, Sidon.Bessel.besselJ0_zero]
  -- Now: λ₁·1² + λ₂·1² + λ₃·1² = λ₁ + λ₂ + λ₃ = 1.
  have h_sum : Sidon.MultiScale.lambda1 + Sidon.MultiScale.lambda2 +
               Sidon.MultiScale.lambda3 = 1 := by
    have h := Sidon.MultiScale.lambdas_sum_one
    unfold Sidon.MultiScale.lambda1 Sidon.MultiScale.lambda2
           Sidon.MultiScale.lambda3
    exact_mod_cast h
  linarith [h_sum]

end Sidon.BundleDefs
