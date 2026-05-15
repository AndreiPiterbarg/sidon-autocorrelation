/-
Sidon Autocorrelation Project — Schwartz Atomic Discharge
==========================================================

This file is the partial discharge of the `SchwartzAtomic` record
from `Sidon.MultiScaleSchwartz`.

## The five `SchwartzAtomic` atomic Fourier primitives

The `SchwartzAtomic f_s` record contains 5 atomic Fourier primitives:

1. `ConstantTermEqTwoOverU c`         — `c = 2/u`.
2. `TailFormSchwartz f J rf K̃ τ`      — `τ = 2·u²·∑ rf²·K̃`.
3. `SchwartzTorusSplit f J K̃ c τ`     — `∫((f*f)+(f∘f))·K_ms = c + τ`.
4. `ParsevalSplitSchwartz f J`        — `∫(f∘f)·K_ms = 1 + ∑ |f̂|²·Re K̂`.
5. `FBoundLatticeSchwartz f J`        — `∑ |f̂(j/u)|⁴ ≤ R(f) - 1`.

## Status of each field

### Fields 1 and 2 — discharged unconditionally as `rfl`

Both `ConstantTermEqTwoOverU` and `TailFormSchwartz` are definitionally
true when their named values are chosen canonically.  This file's
theorems `discharge_constant_term` and `discharge_tail_form` provide
the witnesses unconditionally.

### Field 3 (`SchwartzTorusSplit`) — residual

The torus split

  `∫((f*f) + (f∘f))·K_ms = c + τ`

requires the bilinear period-`u` Parseval applied to `f*f + f∘f`
paired with `K_ms`.  Although `K_ms` has support in
`[-δ₁, δ₁] ⊂ (-u/2, u/2)`, the convolution `f*f` has support
`(-1/2, 1/2)` which exceeds one period `(-u/2, u/2) = (-0.319, 0.319)`
for `u = 638/1000`.  The bilinear period-`u` Parseval bridge
(`Sidon.BilinearParseval.bilinear_parseval_period_u`) requires both
functions to be supported in `(-u/2, u/2)`, so it does not apply
directly.

A discharge would need Poisson summation on `f*f`, which is not
currently available in mathlib in a directly usable form.  This
field remains as a residual hypothesis.

### Field 4 (`ParsevalSplitSchwartz`) — residual

The Parseval split

  `∫(f∘f)·K_ms = 1 + ∑_j |f̂(j/u)|² · Re K̂(j/u)`

requires identifying `𝓕(f∘f)(j/u) = |𝓕f(j/u)|²` at lattice points.
For the pointwise autocorrelation `(f∘f)(x) := f(x)·f(-x)` of real
`f`, the Fourier transform is the convolution
`𝓕f ∗ conj(𝓕f) (ξ)`, which is NOT equal to `|𝓕f(ξ)|²` pointwise.
The identity `𝓕(f ⋆ f̌)(ξ) = |𝓕f(ξ)|²` holds for the *autocorrelation*
`f ⋆ f̌` (convolution-with-reverse), not the pointwise product
`f · f̌ = f∘f`.

The codebase's own comment in `Sidon/TorusParseval.lean:538-547`
acknowledges this distinction.  The `ParsevalSplitSchwartz` identity
matches MV's intended argument, which routes via Poisson summation
on the pointwise autocorrelation viewed as a periodised function on
`ℝ/uℤ`.  This requires the Poisson formula at lattice (mathlib has
`Real.tsum_eq_tsum_fourierIntegral` but the project's wiring is not
currently in place).

This field remains as a residual hypothesis.

### Field 5 (`FBoundLatticeSchwartz`) — residual

The F-bound

  `∑_{j ∈ J} |f̂(j/u)|⁴ ≤ R(f) - 1`

requires Plancherel on `f*f` and an L¹/L² inequality.  Specifically:

  `∑_j |f̂(j/u)|⁴ = ∑_j |𝓕(f*f)(j/u)|²`
                  `≤ u · ∫_ℝ |f*f|² + boundary_correction`

where the inequality comes from period-`u` Plancherel applied to
the periodised `f*f`.  Since `supp(f*f) = (-1/2, 1/2)` and `u/2 =
0.319 < 1/2`, the periodisation `(f*f)_per` has boundary
contributions from `f*f(x ± u)` on `x ∈ [-0.319, -0.138)` and
`x ∈ (0.138, 0.319]`.  A tight bound for `∑ |f̂(j/u)|⁴` therefore
requires careful analysis of these boundary contributions.

This field remains as a residual hypothesis.

## What this file delivers

  * `discharge_constant_term` — discharges field 1 unconditionally (`rfl`).

  * `discharge_tail_form` — discharges field 2 unconditionally (`rfl`)
    for any choice of `realFparts`, `Ktilde`.

  * `BundleEq1.K_ms_integrable` / `K_ms_integral_eq_one` are already
    available unconditionally from `Sidon.BundleEq1` (these discharge
    the `hK_int` and `hK_int_one` fields of `SchwartzAtomic`).

  * A documentation note that further discharge of fields 3, 4, 5
    requires Poisson summation infrastructure not currently wired
    into the project's Lean development.

The headline theorem `autoconvolution_ratio_ge_1292_1000_schwartz`
in `Sidon.MultiScaleSchwartz` continues to take the full
`SchwartzAtomic` record.  The discharge work in this file shows
that two of its five atomic primitives are `rfl`-discharge-able;
the remaining three remain as the residual analytic content.

No `sorry`, no new axioms.
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
import Sidon.MultiScaleSchwartz
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

/-! ## Field 1 — `ConstantTermEqTwoOverU` discharged unconditionally

The constant-term identity `c = 2/u` is true `by rfl` when `c` is
chosen canonically to equal `2/uReal`.  This discharge is universal:
it applies to any test function (Schwartz or not). -/

/-- **Field 1 of `SchwartzAtomic` discharged**: for the canonical
choice `constant_term := 2/uReal`, the predicate
`ConstantTermEqTwoOverU` is unconditionally true.

This is `rfl` from the definition of `ConstantTermEqTwoOverU`. -/
theorem discharge_constant_term :
    Sidon.BundleEq3Schwartz.ConstantTermEqTwoOverU
      (2 / Sidon.MultiScale.uQ_real : ℝ) := by
  unfold Sidon.BundleEq3Schwartz.ConstantTermEqTwoOverU
  rfl

/-! ## Field 2 — `TailFormSchwartz` discharged unconditionally

The tail-form identity `τ = 2·u²·∑ realFparts²·Ktilde` is true
`by rfl` when `τ` is chosen canonically to equal that sum.  Like
field 1, this discharge is universal. -/

/-- **Field 2 of `SchwartzAtomic` discharged**: for the canonical
choice `tail_sum := 2·uReal²·∑_J realFparts² · Ktilde`, the
predicate `TailFormSchwartz` is unconditionally true.

This is `rfl` from the definition of `TailFormSchwartz`. -/
theorem discharge_tail_form
    (f_s : 𝓢(ℝ, ℝ)) (J : Finset ℤ)
    (realFparts Ktilde : ℤ → ℝ) :
    Sidon.BundleEq3Schwartz.TailFormSchwartz f_s J realFparts Ktilde
      (2 * Sidon.BundleEq3Schwartz.uReal ^ 2 *
        ∑ j ∈ J, (realFparts j) ^ 2 * Ktilde j) := by
  unfold Sidon.BundleEq3Schwartz.TailFormSchwartz
  rfl

/-! ## Discharged auxiliary fields

The following `SchwartzAtomic` fields are discharge-able directly
from existing infrastructure (without needing the residual Fourier
identities of fields 3, 4, 5). -/

/-- `K_ms` is integrable, discharging `hK_int`. -/
theorem schwartz_atomic_hK_int :
    Integrable Sidon.MultiScale.K_ms volume :=
  Sidon.BundleEq1.K_ms_integrable

/-- `∫ K_ms = 1`, discharging `hK_int_one`. -/
theorem schwartz_atomic_hK_int_one :
    ∫ x, Sidon.MultiScale.K_ms x ∂volume = 1 :=
  Sidon.BundleEq1.K_ms_integral_eq_one

/-! ### Schwartz-specific integrability discharges

For Schwartz `f_s`, the pointwise autocorrelation `f∘f := f(x)·f(-x)`
is continuous and bounded by `(seminorm 0 0 f_s)²` everywhere; hence
when combined with the `L¹` kernel `K_ms`, the products are
integrable.  These are direct corollaries of existing lemmas in
`Sidon.BundleEq3Schwartz`. -/

/-- The convolutional autocorrelation `autocorr f` is bounded for Schwartz `f`.
This is the existence form of `Sidon.BundleEq3Schwartz.pAuto_norm_le`.

For Schwartz `f`, `autocorr f x := ∫ t, f(t)·f(x+t) dt` is bounded by
`(seminorm 0 0 f_s) · ∫ |f_s|`. -/
theorem pAuto_bounded_schwartz (f_s : 𝓢(ℝ, ℝ)) :
    ∃ C : ℝ, ∀ x, |autocorr (f_s : ℝ → ℝ) x| ≤ C := by
  refine ⟨(SchwartzMap.seminorm ℝ 0 0 f_s) * ∫ t, |f_s t| ∂volume, ?_⟩
  intro x
  exact Sidon.BundleEq3Schwartz.pAuto_norm_le f_s x

/-- `autocorr f := ∫ t, f(t)·f(x+t) dt` is integrable for Schwartz `f`.

For Schwartz `f`, `autocorr f` is bounded continuous (by
`pAuto_continuous` and `pAuto_norm_le`).  Its integrability requires
additional control: while `autocorr f` is bounded, it is not generally
in `L¹`. We obtain integrability by routing through the convolution
identity `autocorr f x = (f ⋆ f̌)(-x)` and using that `f ⋆ f̌` is
integrable (as the convolution of two `L¹` functions).

Mathlib: `Integrable.integrable_convolution`.
-/
theorem pAuto_integrable_schwartz (f_s : 𝓢(ℝ, ℝ)) :
    Integrable (autocorr (f_s : ℝ → ℝ)) volume := by
  -- Rewrite via `pAuto_eq_convolution_neg`:
  --   autocorr f x = (f ⋆ f̌)(-x).
  have h_eq : (autocorr (f_s : ℝ → ℝ))
                = (fun x => (MeasureTheory.convolution
                              ((f_s : ℝ → ℝ)) (fun y => (f_s : ℝ → ℝ) (-y))
                              (ContinuousLinearMap.mul ℝ ℝ) volume) (-x)) := by
    funext x
    have := Sidon.BundleEq3Schwartz.pAuto_eq_convolution_neg (f_s : ℝ → ℝ) x
    show autocorr (f_s : ℝ → ℝ) x = _
    -- `pAuto f = autocorr f`.
    exact this
  rw [h_eq]
  -- Step 1: `f̌(x) := f(-x)` is integrable on `ℝ` (since Lebesgue is invariant
  -- under reflection — `Measure.measurePreserving_neg`).
  have h_fs_int : Integrable (f_s : ℝ → ℝ) volume := f_s.integrable
  have h_neg_meas : MeasureTheory.MeasurePreserving (Neg.neg : ℝ → ℝ)
                      (volume : Measure ℝ) (volume : Measure ℝ) :=
    Measure.measurePreserving_neg _
  have h_fs_neg_int : Integrable (fun y => (f_s : ℝ → ℝ) (-y)) volume :=
    h_neg_meas.integrable_comp_of_integrable f_s.integrable
  -- Step 2: `f ⋆ f̌` is integrable.
  have h_conv_int : Integrable
      (MeasureTheory.convolution ((f_s : ℝ → ℝ)) (fun y => (f_s : ℝ → ℝ) (-y))
        (ContinuousLinearMap.mul ℝ ℝ) volume) volume :=
    h_fs_int.integrable_convolution
      (L := ContinuousLinearMap.mul ℝ ℝ) h_fs_neg_int
  -- Step 3: `(g ∘ Neg)` is integrable when `g` is integrable on a reflection-
  -- invariant measure.
  exact h_neg_meas.integrable_comp_of_integrable h_conv_int

/-- `(autocorr f)·K_ms` is integrable for Schwartz `f`. -/
theorem pAuto_K_ms_integrable_schwartz (f_s : 𝓢(ℝ, ℝ)) :
    Integrable (fun x => autocorr (f_s : ℝ → ℝ) x * Sidon.MultiScale.K_ms x) volume :=
  Sidon.BundleEq3Schwartz.pAuto_K_ms_integrable f_s
    Sidon.BundleEq1.K_ms_integrable

/-! ## Documentation: status of fields 3, 4, 5

The remaining three atomic Fourier primitives of `SchwartzAtomic`
encode genuinely analytic content that is not currently discharge-able
from mathlib + the bundle modules.

Each requires Poisson summation, period-`u` Parseval on functions
whose support exceeds one period, or the convolution-vs-pointwise
identification at lattice points — none of which is currently
wired into the project's Lean development.

### Field 3 (`SchwartzTorusSplit`)

The identity
  `∫((f*f) + (f∘f))·K_ms = constant_term + tail_sum`
requires the bilinear period-`u` Parseval applied to `g = f*f + f∘f`
paired with `h = K_ms`.  Although `K_ms` has support `[-δ₁, δ₁] ⊂
(-u/2, u/2)`, the convolution `f*f` has support `(-1/2, 1/2)` which
exceeds one period `(-u/2, u/2) = (-0.319, 0.319)`.  The bridge
`Sidon.BilinearParseval.bilinear_parseval_period_u` requires both
functions to be supported in `(-u/2, u/2)`, so it does not apply
directly.

A discharge would need either (i) Poisson summation on `f*f` to
identify its lattice Fourier coefficients with `𝓕f(j/u)²`, or
(ii) a periodisation-based formulation that handles the boundary
contributions from translates of `f*f`.

Mathlib bridge needed: `Real.tsum_eq_tsum_fourierIntegral` (Poisson
formula) wired into a `(f*f)`-supported version, accounting for the
overflow boundary on `(±u/2 - 1/2, ±u/2 + 1/2) ∩ (-u/2, u/2)`.

### Field 4 (`ParsevalSplitSchwartz`)

The identity
  `∫(f∘f)·K_ms = 1 + ∑_j |f̂(j/u)|² · Re K̂(j/u)`
requires identifying `𝓕(f∘f)(j/u) = |𝓕f(j/u)|²` at lattice points.
For the pointwise autocorrelation `(f∘f)(x) := f(x)·f(-x)` of real
`f`, `𝓕(f∘f)(ξ) = (𝓕f ∗ conj(𝓕f))(ξ)`, which is NOT pointwise
equal to `|𝓕f(ξ)|²`.

The identity `𝓕(f ⋆ f̌)(ξ) = |𝓕f(ξ)|²` holds for the *autocorrelation*
`f ⋆ f̌` (convolution-with-reverse), not the pointwise product
`f · f̌ = f∘f`.  This is noted in
`Sidon/TorusParseval.lean:538-547`.

Mathlib bridge needed: bilinear period-`u` Parseval (already in
`Sidon.BilinearParseval`) wired to identify `widehat(f∘f)(j/u)`
via Poisson summation on the pointwise autocorrelation.

### Field 5 (`FBoundLatticeSchwartz`)

The bound
  `∑_{j ∈ J} |f̂(j/u)|⁴ ≤ R(f) - 1`
requires period-`u` Plancherel on `f*f` (whose support exceeds one
period) and an L¹/L² inequality.  The relevant lattice form

  `∑_j |f̂(j/u)|⁴ = ∑_j |𝓕(f*f)(j/u)|²`
                  `≤ u · ∫ |(f*f)_per|²` (Plancherel for the periodisation)
                  `≤ u · (∫ |f*f|² + boundary)`

has a boundary correction from translates of `f*f`.

Mathlib bridge needed: Poisson summation on `f*f` with explicit
control of the periodisation boundary contributions.

## Conclusion

Of the 5 atomic Fourier primitives in `SchwartzAtomic`, fields 1
and 2 are discharged unconditionally as `rfl`.  Fields 3, 4, 5
remain as residual hypotheses encoding genuine analytic content
that requires Poisson summation infrastructure not currently in
the project's Lean development.

The headline theorem `autoconvolution_ratio_ge_1292_1000_schwartz`
in `Sidon.MultiScaleSchwartz` therefore continues to take the full
`SchwartzAtomic` record as a hypothesis. -/

/-! ## Slim residual hypothesis bundle

The `SchwartzAtomicResidual` record collects only the *genuinely
residual* analytic content needed to build a `SchwartzAtomic f_s`
for Schwartz admissible `f_s`.  The other `SchwartzAtomic` fields
are either discharged from existing infrastructure (e.g. `hK_int`
from `BundleEq1.K_ms_integrable`) or follow from the user's
admissibility hypotheses.

Specifically, this slim bundle keeps:
  * `J`, `J_no_zero`, `Ktilde` — the frequency indexing data;
  * `h_K2_ge_1`, `h_R_ge_1` — the analytic floors (provable, but kept
    as hypotheses for cleanliness);
  * `hK_L2_torus` — the K_ms-on-torus L² certificate (provable from
    `K2_analytic_le_K2UpperQ` axiom, but kept as a hypothesis since
    its derivation requires that axiom);
  * `hFofF_one` — the `∫(f∘f) = 1` identity (NOT generally true for
    pointwise `f∘f`; matches MV's framing assuming `‖f‖₂² = 1`);
  * `h_conv_K_int`, `h_conv_fin`, `h_conv_bdd` — convolution-side
    integrability and boundedness;
  * `h_F_lat`, `h_split_eq2`, `h_torus_split` — the three genuinely
    residual Fourier identities (fields 3, 4, 5);
  * `S_cos_value`, `S_cos_def`, `h_eq4_floor` — the Eq.(4) input.

The user-side hypotheses passed in (admissibility) are:
  * `hf_nonneg`, `hf_supp`, `hf_int_one` — `f_s` admissibility.

The constructor `SchwartzAtomic.from_residual` then builds the full
`SchwartzAtomic f_s` record by combining the residual data with the
discharges from this file.

Note: the integrability of `f∘f` and `(f∘f)·K_ms` are auto-discharged
inside the constructor for Schwartz `f_s`. -/

/-- The **slim residual** hypothesis bundle for Schwartz `f_s`.
Strictly smaller than the full `SchwartzAtomic`. -/
structure SchwartzAtomicResidual (f_s : 𝓢(ℝ, ℝ)) where
  /-- Frequency indexing set (excluding 0). -/
  J : Finset ℤ
  /-- `0 ∉ J`. -/
  J_no_zero : (0 : ℤ) ∉ J
  /-- Period-`u` Fourier coefficients of `K_ms`. -/
  Ktilde : ℤ → ℝ
  /-- `K2_analytic ≥ 1`. -/
  h_K2_ge_1 : 1 ≤ K2_analytic
  /-- `R(f_s) ≥ 1`. -/
  h_R_ge_1 : 1 ≤ autoconvolution_ratio ((f_s : ℝ → ℝ))
  /-- `K_ms ∈ L²` on the torus restriction. -/
  hK_L2_torus : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
                  (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))
  /-- `∫ autocorr f = 1`.

  For the convolutional autocorrelation `autocorr f x := ∫ t, f(t)·f(x+t) dt`,
  this is a *consequence* of `∫ f = 1` via Fubini:
  `∫_x autocorr f x dx = ∫_x ∫_t f(t)·f(x+t) dt dx = ∫_t f(t)·(∫_x f(x+t) dx) dt
   = ∫_t f(t)·(∫f) dt = (∫f)² = 1`.

  We keep it as a hypothesis here because Fubini requires joint integrability
  of `(x,t) ↦ f(t)·f(x+t)`, which is a separate analytic input
  (cf. `Sidon.FourierAux.integral_autocorr_eq_sq_integral`). -/
  hFofF_one : ∫ x, autocorr (f_s : ℝ → ℝ) x ∂volume = 1
  /-- `eLpNorm (f*f) ⊤ volume ≠ ⊤`. -/
  h_conv_fin : eLpNorm (MeasureTheory.convolution
                  (f_s : ℝ → ℝ) (f_s : ℝ → ℝ)
                  (ContinuousLinearMap.mul ℝ ℝ) volume)
                ⊤ volume ≠ ⊤
  /-- `(f*f)·K_ms` is integrable. -/
  h_conv_K_int : Integrable
                (fun x => (MeasureTheory.convolution
                    (f_s : ℝ → ℝ) (f_s : ℝ → ℝ)
                    (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x) volume
  /-- `f*f` is bounded. -/
  h_conv_bdd : ∃ C : ℝ, ∀ x,
    |(MeasureTheory.convolution (f_s : ℝ → ℝ) (f_s : ℝ → ℝ)
        (ContinuousLinearMap.mul ℝ ℝ) volume) x| ≤ C
  /-- **Residual primitive (Field 5)**: lattice F-bound. -/
  h_F_lat : Sidon.BundleEq2Schwartz.FBoundLatticeSchwartz f_s J
  /-- **Residual primitive (Field 4)**: parseval split for `∫(f∘f)·K_ms`. -/
  h_split_eq2 : Sidon.BundleEq2Schwartz.ParsevalSplitSchwartz f_s J
  /-- The cosine-sum value supporting Eq.(4); definitionally tied to
  the canonical tail-sum via `S_cos_def_residual`. -/
  S_cos_value : ℝ
  /-- Identification of the canonical tail-sum form with `2·u²·S_cos_value`.
  This is forced once `S_cos_value` is chosen to equal
  `(uReal²·∑ realFparts²·Ktilde) / uQ²`, i.e., the residual is
  internally consistent. -/
  S_cos_def_residual :
    2 * Sidon.BundleEq3Schwartz.uReal ^ 2 *
      ∑ j ∈ J,
        ((Real.fourierIntegral (fun x => (((f_s : ℝ → ℝ) x : ℝ) : ℂ))
            (j / Sidon.MultiScale.uQ_real : ℝ)).re
          / Sidon.MultiScale.uQ_real) ^ 2 * Ktilde j
      = 2 * (uQ : ℝ) ^ 2 * S_cos_value
  /-- The Eq.(4) bound `u²·S_cos_value ≥ gain_analytic / 2`. -/
  h_eq4_floor : (uQ : ℝ) ^ 2 * S_cos_value ≥ gain_analytic / 2
  /-- **Residual primitive (Field 3)**: torus split.

  The `constant_term` is forced to `2/uReal` (Field 1, `rfl`).
  The `tail_sum` is forced to `2·uReal²·∑_J realFparts²·Ktilde` (Field 2, `rfl`).
  Here `realFparts j := (Re widehat(f_s)(j/uReal))/uReal`. -/
  h_torus_split :
    Sidon.BundleEq3Schwartz.SchwartzTorusSplit f_s J Ktilde
      (2 / Sidon.MultiScale.uQ_real : ℝ)
      (2 * Sidon.BundleEq3Schwartz.uReal ^ 2 *
        ∑ j ∈ J,
          ((Real.fourierIntegral (fun x => (((f_s : ℝ → ℝ) x : ℝ) : ℂ))
              (j / Sidon.MultiScale.uQ_real : ℝ)).re
            / Sidon.MultiScale.uQ_real) ^ 2 * Ktilde j)

/-! ## Constructor: `SchwartzAtomic.from_residual`

The constructor takes:
  * The user's admissibility hypotheses (`hf_nonneg`, `hf_supp`,
    `hf_int_one`);
  * The slim residual bundle `R : SchwartzAtomicResidual f_s`;

and produces a `SchwartzAtomic f_s`.  All other fields of
`SchwartzAtomic` are discharged inside the constructor using existing
project infrastructure. -/

/-- **Lift a slim `SchwartzAtomicResidual` to a full `SchwartzAtomic`**
by discharging the auxiliary fields from existing infrastructure
plus the user's admissibility hypotheses. -/
noncomputable def SchwartzAtomic.from_residual
    (f_s : 𝓢(ℝ, ℝ))
    (hf_nonneg : ∀ x, 0 ≤ (f_s : ℝ → ℝ) x)
    (hf_supp : Function.support ((f_s : ℝ → ℝ)) ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_one : ∫ x, (f_s : ℝ → ℝ) x ∂volume = 1)
    (R : SchwartzAtomicResidual f_s) : SchwartzAtomic f_s where
  J := R.J
  J_no_zero := R.J_no_zero
  Ktilde := R.Ktilde
  -- Field 1 (constant_term) — `rfl` discharge via canonical value.
  constant_term := 2 / Sidon.MultiScale.uQ_real
  -- Field 2 (tail_sum) — `rfl` discharge via canonical value.
  tail_sum := 2 * Sidon.BundleEq3Schwartz.uReal ^ 2 *
                ∑ j ∈ R.J,
                  ((Real.fourierIntegral (fun x => (((f_s : ℝ → ℝ) x : ℝ) : ℂ))
                      (j / Sidon.MultiScale.uQ_real : ℝ)).re
                    / Sidon.MultiScale.uQ_real) ^ 2 * R.Ktilde j
  h_K2_ge_1 := R.h_K2_ge_1
  h_R_ge_1 := R.h_R_ge_1
  -- Discharged from existing infrastructure.
  hK_int := schwartz_atomic_hK_int
  hK_int_one := schwartz_atomic_hK_int_one
  -- Kept as hypothesis (K2-axiom-dependent).
  hK_L2_torus := R.hK_L2_torus
  -- Discharged from existing infrastructure.
  hProd_int := pAuto_K_ms_integrable_schwartz f_s
  hFofF_int := pAuto_integrable_schwartz f_s
  hFofF_one := R.hFofF_one
  h_conv_K_int := R.h_conv_K_int
  h_conv_fin := R.h_conv_fin
  hf_int_one := hf_int_one
  hf_nonneg := hf_nonneg
  hf_supp := hf_supp
  -- Field 5 (residual).
  h_F_lat := R.h_F_lat
  -- Field 4 (residual).
  h_split_eq2 := R.h_split_eq2
  -- Field 3 (residual; uses canonical constant_term, tail_sum).
  h_torus_split := R.h_torus_split
  -- Field 1 (rfl discharge).
  h_constant_term := discharge_constant_term
  -- Field 2 (rfl discharge).
  h_tail_form := discharge_tail_form f_s R.J
                  (fun j => (Real.fourierIntegral (fun x => (((f_s : ℝ → ℝ) x : ℝ) : ℂ))
                                (j / Sidon.MultiScale.uQ_real : ℝ)).re
                              / Sidon.MultiScale.uQ_real)
                  R.Ktilde
  S_cos_value := R.S_cos_value
  -- The S_cos_def is exactly `R.S_cos_def_residual`.
  S_cos_def := R.S_cos_def_residual
  h_eq4_floor := R.h_eq4_floor
  h_conv_bdd := R.h_conv_bdd

/-! ## Residual-form headline theorem

The headline `R(f_s) ≥ 1292/1000` for Schwartz admissible `f_s`,
now taking the slim `SchwartzAtomicResidual` bundle instead of the
full `SchwartzAtomic`. -/

/-- **Residual-form headline**.

For any Schwartz admissible test function `f_s` satisfying:
  * `f_s ≥ 0` pointwise,
  * `supp f_s ⊆ (-1/4, 1/4)`,
  * `∫f_s = 1` (normalisation),
  * the slim residual `SchwartzAtomicResidual f_s` analytic hypotheses,

the autoconvolution ratio satisfies `R(f_s) ≥ 1292/1000 = 1.292`.

This is strictly weaker than `autoconvolution_ratio_ge_1292_1000_schwartz`
(which takes the full `SchwartzAtomic`); the slim form
`SchwartzAtomicResidual` has the integrability and `rfl`-discharge-able
fields removed. -/
theorem autoconvolution_ratio_ge_1292_1000_schwartz_residual
    (f_s : 𝓢(ℝ, ℝ))
    (hf_nonneg : ∀ x, 0 ≤ (f_s : ℝ → ℝ) x)
    (hf_supp : Function.support ((f_s : ℝ → ℝ)) ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_one : ∫ x, (f_s : ℝ → ℝ) x ∂volume = 1)
    (R : SchwartzAtomicResidual f_s) :
    autoconvolution_ratio ((f_s : ℝ → ℝ)) ≥ (1292 / 1000 : ℝ) :=
  autoconvolution_ratio_ge_1292_1000_schwartz f_s
    (SchwartzAtomic.from_residual f_s hf_nonneg hf_supp hf_int_one R)

end Sidon.MultiScale
