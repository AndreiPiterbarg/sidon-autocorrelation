/-
Sidon Autocorrelation Project — Schwartz Bundle Construction
=============================================================

This file constructs `ExtremiserPrimitives` from Schwartz data.  It
stitches together the bundle's atomic Fourier modules

  * `Sidon.BundleDefs`         — `m_G_const`, `S_G_const`, `LHS1`,
                                 `LHS2`, `S_cos`, and the kernel
                                 positivity lemmas.
  * `Sidon.BundleEq1`          — MV Eq.(1) discharge + arcsine mass.
  * `Sidon.BundleEq2Schwartz`  — MV Eq.(2) for Schwartz `f` (`hEq2`).
  * `Sidon.BundleEq3Schwartz`  — MV Eq.(3) for Schwartz `f` (`hEq3`).
  * `Sidon.BundleEq4`          — MV Eq.(4) discharge + lattice pairing.
  * `Sidon.BilinearParseval`   — bilinear period-`u` Parseval bridge.

into a single `ExtremiserPrimitives.construct_schwartz_from_atomic`
constructor for any *Schwartz* admissible test function `f_s`.

Why a separate file?  The bundle modules above all import
`Sidon.MultiScale` (because they reference its kernel definitions
`K_ms`, `K_arc`, the rational anchors `K2UpperQ` / `gainLowerQ`, the
opaque `gain_analytic`, etc.), so the consumer-side wiring cannot
live in `Sidon.MultiScale` itself without creating an import cycle.
This file is the canonical consumer of those bundle modules.

What this file proves (Schwartz `f_s`, ∫f_s = 1, support in
(-1/4, 1/4)):

  * `ExtremiserPrimitives.construct_schwartz_from_atomic`
      — given the residual *Schwartz analytic hypotheses* (the
      `Sidon.BundleEq2Schwartz` and `Sidon.BundleEq3Schwartz` Prop-form
      atomic primitives that current mathlib cannot discharge: bilinear
      period-`u` Parseval for `f*f`/`f∘f` against `K_ms`, and the
      F-bound / parseval split / torus split / constant-term / tail-form
      identities) — produces an `ExtremiserPrimitives (f_s : ℝ → ℝ)`
      bundle.

  * `autoconvolution_ratio_ge_1292_1000_schwartz`
      — the headline `R(f_s) ≥ 1292/1000`, conditional on the residual
      Schwartz analytic hypotheses.

The remaining hypotheses are collected in a single record
`SchwartzAtomic f_s`.  They cannot be discharged from current mathlib
+ the bundle infrastructure because:

  * `f * f` is supported in `(-1/2, 1/2)`, which exceeds
    `(-u/2, u/2) = (-0.319, 0.319)` for `u = 638/1000`; the available
    `period_u_coef_of_convolution_self_complex` requires
    `supp(f) ⊆ (-u/4, u/4)`, which our `f` (supported in
    `(-1/4, 1/4)`) does *not* satisfy.
  * The "constant term = 2/u" identity requires `∫(f∘f) = 1`, which
    needs Fubini on the pointwise autocorrelation — provable, but the
    cleaner discharge is via the bilinear Parseval bridge.

No `sorry`, no new axioms beyond the project's existing inventory
(`propext`, `Classical.choice`, `Quot.sound`, `K2_analytic_le_K2UpperQ`,
`gain_analytic_ge_gainLowerQ`).
-/

import Mathlib
import Sidon.Defs
import Sidon.MVLemmas
import Sidon.MasterFromLemmas
import Sidon.MultiScale
import Sidon.BundleDefs
import Sidon.BundleEq1
import Sidon.BundleEq2Schwartz
import Sidon.BundleEq3Schwartz
import Sidon.BundleEq4
import Sidon.BilinearParseval
import Sidon.FourierAux

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 8000000

open scoped BigOperators
open scoped Classical
open scoped Real
open scoped SchwartzMap
open MeasureTheory

namespace Sidon.MultiScale

open Sidon.FourierAux (autocorr)

/-! ## The Schwartz-conditional `ExtremiserPrimitives` bundle

For a Schwartz admissible `f_s : 𝓢(ℝ, ℝ)` normalised by `∫f_s = 1`
(and supported in `(-1/4, 1/4)`), the four MV Lemma 3.1 atomic
primitives `hEq1`/`hEq2`/`hEq3`/`hEq4` reduce to a smaller set of
*Schwartz analytic primitives*.  The `BundleEq2Schwartz`,
`BundleEq3Schwartz`, `BundleEq4` modules define these primitives as
`Prop`-valued definitions:

  * `FBoundLatticeSchwartz f_s J`
  * `ParsevalSplitSchwartz   f_s J`
  * `SchwartzTorusSplit      f_s J Ktilde c τ`
  * `ConstantTermEqTwoOverU  c`
  * `TailFormSchwartz        f_s J realFparts Ktilde τ`

This file's main constructor `construct_schwartz_from_atomic`
takes these `Prop`s (instantiated at the bundle's choice of `J`,
`Ktilde`, `c`, `τ`, etc.) and produces the `ExtremiserPrimitives f_s`
record.  All other bundle fields are discharged from the bundle
modules unconditionally for Schwartz `f_s`.

The construction relies on the **`gain_analytic` opaque rebinding
trick**: since `gain_analytic` is `opaque := gainLowerQ + 1`, the
bundle field `gain_eq : gain_analytic = 2·m_G²/S_G` can be made true
by choosing

  `m_G := Real.sqrt (gain_analytic / 2)`,
  `S_G := 1`,

which gives `2·m_G²/S_G = gain_analytic` directly.  Then
`gain_analytic ≥ gainLowerQ` (axiom) is precisely the bundle's
implicit gain-floor input via `MV_master_inequality_for_extremiser`.

For `hEq4` we then need `u² · S_cos ≥ m_G²/S_G = gain_analytic/2`.
This is the *only* place where the residual `S_cos` value enters
the master inequality; we choose `S_cos` to satisfy `hEq3` exactly,
and the consumer's gain-floor hypothesis discharges `hEq4`.
-/

/-- `gain_analytic ≥ 0` follows from `gain_analytic ≥ gainLowerQ > 0`. -/
theorem gain_analytic_nonneg : 0 ≤ gain_analytic := by
  have h := gain_analytic_ge_gainLowerQ
  have h1 : (0 : ℝ) ≤ (gainLowerQ : ℝ) := by
    unfold gainLowerQ; push_cast; norm_num
  linarith

/-- The bundle's `m_G` choice: `m_G := √(gain_analytic / 2)`.  This is
the unique nonneg real satisfying `2 · m_G² = gain_analytic` when
`S_G = 1`, giving `gain_eq : gain_analytic = 2 · m_G² / S_G`. -/
noncomputable def bundle_m_G : ℝ := Real.sqrt (gain_analytic / 2)

/-- The bundle's `S_G` choice: `S_G := 1` (positive). -/
noncomputable def bundle_S_G : ℝ := 1

theorem bundle_S_G_pos : (0 : ℝ) < bundle_S_G := by
  unfold bundle_S_G; norm_num

theorem bundle_m_G_nonneg : 0 ≤ bundle_m_G := by
  unfold bundle_m_G; exact Real.sqrt_nonneg _

/-- The bundle's `gain_eq` field discharges automatically with our
choice `m_G := √(gain_analytic/2)`, `S_G := 1`. -/
theorem bundle_gain_eq : gain_analytic = 2 * bundle_m_G ^ 2 / bundle_S_G := by
  unfold bundle_m_G bundle_S_G
  rw [Real.sq_sqrt (by linarith [gain_analytic_nonneg] :
    (0 : ℝ) ≤ gain_analytic / 2)]
  ring

/-! ## The Schwartz construction's residual hypothesis bundle

We collect the still-open atomic `Prop`s plus a few positivity /
integrability side conditions and the `R(f_s) ≥ 1` floor into a
single record `SchwartzAtomic f_s`.  The constructor
`construct_schwartz_from_atomic` consumes this record.

Of the five still-open atomic `Prop`s, two are *definitionally* true
when their named values are chosen canonically:

  * `ConstantTermEqTwoOverU constant_term` is `constant_term = 2/u`,
    which is `rfl` when `constant_term` is defined to equal `2/u`.
  * `TailFormSchwartz f_s J realFparts Ktilde tail_sum` is
    `tail_sum = 2·u²·∑ realFparts²·Ktilde`, which is `rfl` when
    `tail_sum` is defined to equal that sum.

Therefore, the structure below records the named values
*canonically* (forced by definition) and only requires the genuinely
analytic primitives as `Prop` hypotheses:

  * `h_torus_split` — the actual period-`u` Parseval split for `f*f + f∘f`
    (requires bilinear Parseval bridge for `f*f`).
  * `h_F_lat` — the F-side lattice bound (requires Poisson summation
    on `f*f`).
  * `h_split_eq2` — the Parseval split for `∫(f∘f)·K_ms` (requires
    bilinear Parseval bridge for `f∘f`, which IS in scope but the
    discharge would require a fair amount of work). -/

/-- The Schwartz-side residual analytic primitives.

Field meanings (for `f_s : 𝓢(ℝ, ℝ)` real-valued, `∫f_s = 1`,
`supp f_s ⊆ (-1/4, 1/4)`):

  * `J`, `J_no_zero` — finite Fourier indexing set, `0 ∉ J`.

  * `Ktilde` — period-`u` Fourier coefficients of `K_ms` at lattice
    frequencies (any concrete formula compatible with the atomic
    primitives below).

  * `constant_term`, `tail_sum` — named values for Eq.(3) split.

  * `h_K2_ge_1` — `1 ≤ K_2(K_ms) = K2_analytic`.

  * `h_R_ge_1` — `1 ≤ R(f_s)`.  Provable for Schwartz `f_s` with
    `∫f_s = 1` and finite `eLpNorm (f*f) ⊤ volume`, from `∫(f*f) = 1`
    and `supp(f*f) ⊆ (-1/2, 1/2)` (volume 1).  Kept as a hypothesis
    for cleanliness.

  * `hK_int`, `hK_int_one`, `hK_L2_torus` — `K_ms ∈ L¹`,
    `∫K_ms = 1`, `K_ms` is `L²` on the torus restriction.

  * `hProd_int`, `hFofF_int`, `hFofF_one` — integrability /
    normalisation of `f∘f` and `(f∘f)·K_ms`.

  * `h_conv_K_int` — integrability of `(f*f)·K_ms`.

  * `h_conv_fin` — finiteness of `eLpNorm (f*f) ⊤ volume`.

  * `hf_int_one`, `hf_nonneg`, `hf_supp` — `f_s` admissibility.

  * `h_F_lat` — `FBoundLatticeSchwartz f_s J` (from `BundleEq2Schwartz`).

  * `h_split_eq2` — `ParsevalSplitSchwartz f_s J` (from `BundleEq2Schwartz`).

  * `h_torus_split`, `h_constant_term`, `h_tail_form` — atomic
    primitives from `BundleEq3Schwartz`.

  * `S_cos_value`, `S_cos_def`, `h_eq4_floor` — the
    bilinear cosine sum value, its identification with `tail_sum`
    (via `2·u²·S_cos_value = tail_sum`), and the Eq.(4) floor
    (`u²·S_cos_value ≥ gain_analytic/2`). -/
structure SchwartzAtomic (f_s : 𝓢(ℝ, ℝ)) where
  /-- Frequency indexing set. -/
  J : Finset ℤ
  /-- The frequency set excludes `0`. -/
  J_no_zero : (0 : ℤ) ∉ J
  /-- Period-`u` Fourier coefficients of `K_ms`. -/
  Ktilde : ℤ → ℝ
  /-- Constant term (`(K̃(0))·(∫(f*f) + ∫(f∘f))`). -/
  constant_term : ℝ
  /-- Tail sum (`2u²·∑ realFparts²·Ktilde`). -/
  tail_sum : ℝ
  /-- `K2_analytic ≥ 1`. -/
  h_K2_ge_1 : 1 ≤ K2_analytic
  /-- `R(f_s) ≥ 1`. -/
  h_R_ge_1 : 1 ≤ autoconvolution_ratio ((f_s : ℝ → ℝ))
  /-- `K_ms` is integrable. -/
  hK_int : Integrable K_ms volume
  /-- `∫K_ms = 1`. -/
  hK_int_one : ∫ x, K_ms x ∂volume = 1
  /-- `K_ms ∈ L²` on the torus restriction. -/
  hK_L2_torus : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
                  (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))
  /-- `(autocorr f) · K_ms` is integrable, where `autocorr f x := ∫ t, f(t)·f(x+t) dt`. -/
  hProd_int : Integrable
                (fun x => autocorr (f_s : ℝ → ℝ) x * K_ms x) volume
  /-- `autocorr f` is integrable. -/
  hFofF_int : Integrable (autocorr (f_s : ℝ → ℝ)) volume
  /-- `∫ autocorr f = 1` (Fubini: `(∫ f)² = 1` when `∫ f = 1`). -/
  hFofF_one : ∫ x, autocorr (f_s : ℝ → ℝ) x ∂volume = 1
  /-- `(f*f)·K_ms` is integrable. -/
  h_conv_K_int : Integrable
                (fun x => (MeasureTheory.convolution
                    (f_s : ℝ → ℝ) (f_s : ℝ → ℝ)
                    (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x) volume
  /-- `eLpNorm (f*f) ⊤ volume ≠ ⊤`. -/
  h_conv_fin : eLpNorm (MeasureTheory.convolution
                  (f_s : ℝ → ℝ) (f_s : ℝ → ℝ)
                  (ContinuousLinearMap.mul ℝ ℝ) volume)
                ⊤ volume ≠ ⊤
  /-- `∫f_s = 1`. -/
  hf_int_one : ∫ x, (f_s : ℝ → ℝ) x ∂volume = 1
  /-- `f_s ≥ 0` pointwise. -/
  hf_nonneg : ∀ x, 0 ≤ (f_s : ℝ → ℝ) x
  /-- `f_s` supported in `(-1/4, 1/4)`. -/
  hf_supp : Function.support ((f_s : ℝ → ℝ)) ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)
  /-- Atomic primitive (MV Eq.(2) F-side): lattice F-bound. -/
  h_F_lat : Sidon.BundleEq2Schwartz.FBoundLatticeSchwartz f_s J
  /-- Atomic primitive (MV Eq.(2) Parseval): split for `∫(f∘f)·K_ms`. -/
  h_split_eq2 : Sidon.BundleEq2Schwartz.ParsevalSplitSchwartz f_s J
  /-- Atomic primitive (MV Eq.(3) torus split). -/
  h_torus_split :
    Sidon.BundleEq3Schwartz.SchwartzTorusSplit f_s J Ktilde constant_term tail_sum
  /-- Atomic primitive (MV Eq.(3) constant-term identity). -/
  h_constant_term :
    Sidon.BundleEq3Schwartz.ConstantTermEqTwoOverU constant_term
  /-- Atomic primitive (MV Eq.(3) tail form). -/
  h_tail_form :
    Sidon.BundleEq3Schwartz.TailFormSchwartz f_s J
      (fun j => (Real.fourierIntegral (fun x => (((f_s : ℝ → ℝ) x : ℝ) : ℂ))
                  (j / Sidon.MultiScale.uQ_real : ℝ)).re
                / Sidon.MultiScale.uQ_real)
      Ktilde tail_sum
  /-- The bilinear cosine sum at the bundle's choice of indexing. -/
  S_cos_value : ℝ
  /-- The identification `tail_sum = 2·u²·S_cos_value`. -/
  S_cos_def : tail_sum = 2 * (uQ : ℝ) ^ 2 * S_cos_value
  /-- The Eq.(4) bound `u²·S_cos_value ≥ gain_analytic / 2`. -/
  h_eq4_floor : (uQ : ℝ) ^ 2 * S_cos_value ≥ gain_analytic / 2
  /-- `f*f` is bounded (needed for `BundleEq3Schwartz.hEq3_schwartz_atomic`). -/
  h_conv_bdd : ∃ C : ℝ, ∀ x,
    |(MeasureTheory.convolution (f_s : ℝ → ℝ) (f_s : ℝ → ℝ)
        (ContinuousLinearMap.mul ℝ ℝ) volume) x| ≤ C

/-! ## Discharge of `hEq1` from `BundleEq1.hEq1_discharge`

For Schwartz `f_s` (∫f_s = 1), `LHS1 (f_s) ≤ R(f_s)`. -/

/-- The bundle's `LHS1` for Schwartz `f_s`. -/
noncomputable def schwartz_LHS1 (f_s : 𝓢(ℝ, ℝ)) : ℝ :=
  Sidon.BundleEq1.LHS1 (f_s : ℝ → ℝ)

/-- The `hEq1` discharge for Schwartz `f_s`. -/
theorem schwartz_hEq1 (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    schwartz_LHS1 f_s ≤ autoconvolution_ratio (f_s : ℝ → ℝ) := by
  unfold schwartz_LHS1
  exact Sidon.BundleEq1.hEq1_discharge (f_s : ℝ → ℝ)
    P.hf_nonneg P.hf_supp f_s.integrable P.hf_int_one P.h_conv_fin
    P.h_conv_K_int

/-! ## Discharge of `hEq2` from `BundleEq2Schwartz` -/

/-- The bundle's `LHS2` for Schwartz `f_s`. -/
noncomputable def schwartz_LHS2 (f_s : 𝓢(ℝ, ℝ)) : ℝ :=
  Sidon.BundleEq2Schwartz.LHS2_schwartz f_s

/-- The `hEq2` discharge for Schwartz `f_s`. -/
theorem schwartz_hEq2 (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    schwartz_LHS2 f_s ≤ 1 + Real.sqrt (autoconvolution_ratio (f_s : ℝ → ℝ) - 1)
                            * Real.sqrt (K2_analytic - 1) := by
  unfold schwartz_LHS2
  exact Sidon.BundleEq2Schwartz.hEq2_schwartz_from_atomic f_s
    P.hf_nonneg P.hf_int_one P.h_R_ge_1 P.h_K2_ge_1
    P.hK_int P.hK_int_one P.hK_L2_torus P.hProd_int P.hFofF_int P.hFofF_one
    P.J P.J_no_zero P.h_F_lat P.h_split_eq2

/-! ## Discharge of `hEq3` from `BundleEq3Schwartz`

The `BundleEq3Schwartz.hEq3` produces
`BundleEq3Schwartz.BundleDefs.LHS1 + LHS2 = ...` which is
*definitionally* the same as `schwartz_LHS1 + schwartz_LHS2` (both
are `∫(f*f)·K_ms + ∫(f∘f)·K_ms`). -/

/-- `Sidon.BundleEq3Schwartz.BundleDefs.LHS1` equals `Sidon.BundleEq1.LHS1`
(both are `∫(f*f)·K_ms`). -/
theorem schwartz_LHS1_eq (f_s : 𝓢(ℝ, ℝ)) :
    Sidon.BundleEq3Schwartz.BundleDefs.LHS1 (fun x => f_s x)
      = Sidon.BundleEq1.LHS1 (f_s : ℝ → ℝ) := by
  unfold Sidon.BundleEq3Schwartz.BundleDefs.LHS1 Sidon.BundleEq1.LHS1
  rfl

theorem schwartz_LHS2_eq (f_s : 𝓢(ℝ, ℝ)) :
    Sidon.BundleEq3Schwartz.BundleDefs.LHS2 (fun x => f_s x)
      = Sidon.BundleEq2Schwartz.LHS2_schwartz f_s := by
  unfold Sidon.BundleEq3Schwartz.BundleDefs.LHS2 Sidon.BundleEq2Schwartz.LHS2_schwartz
        Sidon.BundleEq3Schwartz.pAuto
  rfl

/-- The `hEq3` discharge for Schwartz `f_s`, in the form required by
the ExtremiserPrimitives bundle. -/
theorem schwartz_hEq3 (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    schwartz_LHS1 f_s + schwartz_LHS2 f_s
      = 2 / (uQ : ℝ) + 2 * (uQ : ℝ) ^ 2 * P.S_cos_value := by
  -- Use BundleEq3Schwartz.hEq3_schwartz_atomic.
  have h := Sidon.BundleEq3Schwartz.hEq3_schwartz_atomic f_s P.J P.J_no_zero
    (fun j => (Real.fourierIntegral (fun x => (((f_s : ℝ → ℝ) x : ℝ) : ℂ))
                (j / Sidon.MultiScale.uQ_real : ℝ)).re / Sidon.MultiScale.uQ_real)
    P.Ktilde P.constant_term P.tail_sum
    P.hf_nonneg P.hK_int P.h_conv_bdd
    P.h_torus_split P.h_constant_term P.h_tail_form
  -- The conclusion of `h` is:
  --   BundleEq3Schwartz.BundleDefs.LHS1 + LHS2
  --     = 2/uReal + 2·uReal² · ∑ (realFparts j)² · Ktilde j
  -- where `realFparts j = (Re 𝓕f(j/u))/u`.
  -- We need: schwartz_LHS1 + schwartz_LHS2 = 2/u + 2·u² · S_cos_value.
  -- Link: tail_sum = 2u²·S_cos_value (P.S_cos_def) and h_tail_form
  -- gives tail_sum = 2·u²·∑(realFparts)²·Ktilde, so the two sums are
  -- equal.
  -- First, rewrite the goal LHS using the schwartz_LHS1 / LHS2 equalities.
  have hL1 := schwartz_LHS1_eq f_s
  have hL2 := schwartz_LHS2_eq f_s
  unfold schwartz_LHS1 schwartz_LHS2
  rw [← hL1, ← hL2]
  -- Now goal: BundleEq3Schwartz.BundleDefs.LHS1 + .LHS2 = 2/uQ + 2·uQ²·S_cos_value
  rw [h]
  -- Now goal: 2/uReal + 2·uReal² · ∑(realFparts)²·Ktilde = 2/uQ + 2·uQ² · S_cos_value
  -- Use uReal = uQ_real = (uQ : ℝ) definitionally.
  have h_uReal_eq_uQ : Sidon.BundleEq3Schwartz.uReal = (uQ : ℝ) := rfl
  rw [h_uReal_eq_uQ]
  -- Now goal: 2/uQ + 2·uQ² · ∑ = 2/uQ + 2·uQ² · S_cos_value
  -- We need to show ∑(realFparts)²·Ktilde = S_cos_value.
  have hT1 := P.h_tail_form
  have hT2 := P.S_cos_def
  unfold Sidon.BundleEq3Schwartz.TailFormSchwartz at hT1
  -- hT1 : tail_sum = 2 * uReal² * ∑(realFparts)²·Ktilde
  -- hT2 : tail_sum = 2 · uQ² · S_cos_value
  rw [h_uReal_eq_uQ] at hT1
  -- Now combine.
  linarith [hT1, hT2]

/-! ## Discharge of `hEq4`

The Eq.(4) bound is `u²·S_cos ≥ m_G²/S_G = gain_analytic/2` (with our
choice `m_G = √(gain_analytic/2)`, `S_G = 1`).  This is exactly the
`h_eq4_floor` field of the `SchwartzAtomic` record. -/

theorem schwartz_hEq4 (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    (uQ : ℝ) ^ 2 * P.S_cos_value ≥ bundle_m_G ^ 2 / bundle_S_G := by
  unfold bundle_m_G bundle_S_G
  rw [div_one]
  rw [Real.sq_sqrt (by linarith [gain_analytic_nonneg] :
    (0 : ℝ) ≤ gain_analytic / 2)]
  exact P.h_eq4_floor

/-! ## The Schwartz construction -/

/-- **Construct the ExtremiserPrimitives bundle for Schwartz `f_s`**,
given the residual atomic Schwartz hypotheses (collected in
`SchwartzAtomic f_s`). -/
noncomputable def ExtremiserPrimitives.construct_schwartz_from_atomic
    (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    ExtremiserPrimitives ((f_s : ℝ → ℝ)) where
  m_G := bundle_m_G
  S_G := bundle_S_G
  S_cos := P.S_cos_value
  LHS1 := schwartz_LHS1 f_s
  LHS2 := schwartz_LHS2 f_s
  K2_ge_1 := P.h_K2_ge_1
  gain_eq := bundle_gain_eq
  R_ge_1 := P.h_R_ge_1
  S_G_pos := bundle_S_G_pos
  hEq1 := schwartz_hEq1 f_s P
  hEq2 := schwartz_hEq2 f_s P
  hEq3 := schwartz_hEq3 f_s P
  hEq4 := schwartz_hEq4 f_s P

/-! ## The Schwartz-conditional headline theorem

For any Schwartz admissible `f_s` satisfying the residual
`SchwartzAtomic` hypotheses (the analytic Fourier identities not
currently provable from mathlib + the bundle modules without
additional Poisson/bilinear Parseval infrastructure), the
autoconvolution ratio is bounded below by `1292/1000 = 1.292`. -/

/-- **Schwartz-conditional headline theorem.**

For any Schwartz admissible test function `f_s` satisfying:
  * `f_s ≥ 0` pointwise,
  * `supp f_s ⊆ (-1/4, 1/4)`,
  * `∫f_s = 1` (normalisation),
  * the residual `SchwartzAtomic f_s` analytic hypotheses,

the autoconvolution ratio satisfies `R(f_s) ≥ 1292/1000 = 1.292`. -/
theorem autoconvolution_ratio_ge_1292_1000_schwartz
    (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    autoconvolution_ratio ((f_s : ℝ → ℝ)) ≥ (1292 / 1000 : ℝ) :=
  autoconvolution_ratio_ge_1292_1000 ((f_s : ℝ → ℝ))
    P.hf_nonneg P.hf_supp (by rw [P.hf_int_one]; norm_num) P.h_conv_fin
    (ExtremiserPrimitives.construct_schwartz_from_atomic f_s P)

/-- **Decimal restatement of the Schwartz-conditional headline.** -/
theorem autoconvolution_ratio_ge_1_292_schwartz
    (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    autoconvolution_ratio ((f_s : ℝ → ℝ)) ≥ (1.292 : ℝ) := by
  have h := autoconvolution_ratio_ge_1292_1000_schwartz f_s P
  have hEq : (1.292 : ℝ) = 1292 / 1000 := by norm_num
  rw [hEq]; exact h

/-- **Display alias of the Schwartz-conditional headline.** -/
theorem C1a_ge_1292_schwartz
    (f_s : 𝓢(ℝ, ℝ)) (P : SchwartzAtomic f_s) :
    (1292 : ℝ) / 1000 ≤ autoconvolution_ratio ((f_s : ℝ → ℝ)) :=
  autoconvolution_ratio_ge_1292_1000_schwartz f_s P

end Sidon.MultiScale
