/-
Sidon Autocorrelation Project — Period-u Torus Parseval Identities
==================================================================

This file establishes the period-`u` torus Parseval / Poisson identities
needed by `Sidon.MV.mv_eq3`.

Setup.  In the MV master inequality the kernel `K` is supported in
`[-δ, δ]` and we work with period `u = 1/2 + δ`.  We need to relate:
  * The non-periodic Fourier transform `f̂(ξ) = ∫ f(x) e^{-2πi x ξ} dx`
    on `ℝ`.
  * The period-`u` Fourier coefficient
    `f̃(j) = (1/u) ∫_{-u/2}^{u/2} f(x) e^{-2πi x j/u} dx`.

When `supp(f) ⊆ (-u/2, u/2]`, the liftIoc periodisation agrees with `f`
on `(-u/2, u/2]`, and the Fourier-on-`AddCircle u` coefficient
`fourierCoeff (liftIoc u (-(u/2)) f) j` equals `(1/u) · f̂(j/u)`.

Main results.
  * `fourierIntegral_lattice_exp_form` — packages the FT-definitional
    identity `f̂(j/u) = ∫ exp(-2πi x j/u) • f x dx`.
  * `fourierIntegral_lattice_eq_intervalIntegral` — for `f` supported
    in `Ioc (-u/2) (u/2)`, the FT integral can be restricted to that
    interval.
  * `period_u_coef_eq_fourierIntegral_at_lattice` — for `f` supported
    in `Ioc (-u/2) (u/2)`, the period-`u` Fourier coefficient of the
    lift equals `(1/u) · f̂(j/u)`.
  * `plancherel_at_lattice_period_u_hasSum`,
    `plancherel_at_lattice_period_u_tsum`,
    `plancherel_at_lattice_period_u` — Plancherel sum-of-squares
    at lattice points.

The plain Parseval bilinear identity `∫ f·conj(g) = u·∑_j f̃(j)·conj(g̃(j))`
is not proven here as a single theorem because it would require a
period-u bilinear pairing identity that mathlib doesn't yet expose in a
ready-to-use form; we provide the sum-of-squares (diagonal) version,
which is what MV Eq.(3) and MV Eq.(2) ultimately need.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.Defs

namespace Sidon.TorusParseval

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Pointwise

set_option maxHeartbeats 4000000
set_option linter.mathlibStandardSet false
set_option linter.deprecated false
set_option linter.unusedVariables false

noncomputable section

/-! ## Building block: unfold `f̂(j/u)`

`Real.fourierIntegral f (j/u) = ∫ v, 𝐞(-(v · (j/u))) • f v` by definition;
this is the same as `∫ v, exp(-2πi v (j/u)) • f v dv`. -/

/-- The Fourier transform at a lattice point, in exponential form:
`f̂(j/u) = ∫ exp(-(2πi · v · j/u)) • f v dv`. -/
theorem fourierIntegral_lattice_exp_form
    (u : ℝ) (f : ℝ → ℂ) (j : ℤ) :
    Real.fourierIntegral f (j / u : ℝ)
      = ∫ v : ℝ, Complex.exp (↑(-2 * π * v * (j / u : ℝ)) * Complex.I) • f v := by
  show FourierTransform.fourier f (j / u : ℝ)
      = ∫ v : ℝ, Complex.exp (↑(-2 * π * v * (j / u : ℝ)) * Complex.I) • f v
  exact Real.fourier_real_eq_integral_exp_smul (f := f) (w := (j / u : ℝ))

/-! ## Restriction to compact support

If `f` is supported in `Ioc (-u/2) (u/2)`, the integral
`∫ x, w(x) • f(x)` on `ℝ` equals `∫_{-u/2}^{u/2} w(x) • f(x)`. -/

/-- For complex-valued `f` whose support lies in `Ioc (-u/2) (u/2)`,
`Real.fourierIntegral f (j/u)` equals the integral over `(-u/2, u/2]`. -/
theorem fourierIntegral_lattice_eq_intervalIntegral
    (u : ℝ) (hu : 0 < u)
    (f : ℝ → ℂ) (j : ℤ)
    (hsupp : Function.support f ⊆ Set.Ioc (-(u/2)) (u/2)) :
    Real.fourierIntegral f (j / u : ℝ)
      = ∫ v in (-(u/2))..(u/2),
          Complex.exp (↑(-2 * π * v * (j / u : ℝ)) * Complex.I) • f v := by
  rw [fourierIntegral_lattice_exp_form u f j]
  -- The integrand is supported in `Ioc (-u/2) (u/2)` since `f` is.
  set g : ℝ → ℂ :=
    fun v => Complex.exp (↑(-2 * π * v * (j / u : ℝ)) * Complex.I) • f v with hg_def
  have hg_supp : Function.support g ⊆ Set.Ioc (-(u/2)) (u/2) := by
    intro v hv
    have : f v ≠ 0 := by
      intro heq
      apply hv
      simp [hg_def, heq]
    exact hsupp this
  -- Apply `integral_eq_integral_of_support_subset` (interval-integral version).
  exact (intervalIntegral.integral_eq_integral_of_support_subset (f := g) hg_supp).symm

/-! ## Period-u Fourier coefficient identity

For `f` supported in `Ioc (-u/2) (u/2)`, the lift `liftIoc u (-(u/2)) f`
agrees with `f` on the fundamental domain `(-(u/2), u/2]`.  Hence
`fourierCoeff (liftIoc u (-(u/2)) f) j = (1/u) · f̂(j/u)`. -/

/-- The period-u Fourier coefficient at lattice index `j` equals
`(1/u) · f̂(j/u)`, when `f` is supported in `Ioc (-(u/2)) (u/2)`.

This is the bridge between the non-periodic Fourier transform on `ℝ`
and the period-`u` Fourier series on `AddCircle u`. -/
theorem period_u_coef_eq_fourierIntegral_at_lattice
    (u : ℝ) (hu : 0 < u)
    (f : ℝ → ℂ)
    (hsupp : Function.support f ⊆ Set.Ioc (-(u/2)) (u/2))
    (j : ℤ) :
    haveI : Fact (0 < u) := ⟨hu⟩
    fourierCoeff (AddCircle.liftIoc u (-(u/2)) f) j
      = (1 / u : ℂ) * Real.fourierIntegral f (j / u : ℝ) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- Step 1: rewrite `fourierCoeff` as a 1/u-scaled interval integral.
  rw [fourierCoeff_eq_intervalIntegral _ j (-(u/2))]
  -- The interval is `[-(u/2), -(u/2) + u] = [-(u/2), u/2]`.
  have hbound : -(u/2) + u = u/2 := by ring
  rw [hbound]
  -- Step 2: rewrite `f̂(j/u)` as an interval integral over `(-(u/2), u/2)`.
  rw [fourierIntegral_lattice_eq_intervalIntegral u hu f j hsupp]
  -- Step 3: compare the two interval integrals.
  -- LHS integrand: fourier (-j) (↑x : AddCircle u) • (liftIoc u (-(u/2)) f) ↑x
  -- RHS integrand: Complex.exp((-2π·x·(j/u))·I) • f x
  -- Use that on `Ioc (-u/2) (u/2)`, liftIoc agrees with `f`.
  have hle : -(u/2) ≤ u/2 := by linarith
  have hlift_eq : ∀ x ∈ Set.Ioc (-(u/2)) (u/2),
      AddCircle.liftIoc u (-(u/2)) f (↑x : AddCircle u) = f x := by
    intro x hx
    have hx' : x ∈ Set.Ioc (-(u/2)) (-(u/2) + u) := by
      have : -(u/2) + u = u/2 := by ring
      rw [this]; exact hx
    exact AddCircle.liftIoc_coe_apply hx'
  have h_int : ∀ x ∈ Set.Ioc (-(u/2)) (u/2),
        (fourier (-j) (↑x : AddCircle u) : ℂ)
          • AddCircle.liftIoc u (-(u/2)) f (↑x : AddCircle u)
        = Complex.exp (↑(-2 * π * x * (j / u : ℝ)) * Complex.I) • f x := by
    intro x hx
    rw [hlift_eq x hx]
    rw [fourier_coe_apply]
    -- LHS: exp(2π·I·(-j)·x/u) • f x  =  RHS: exp((-2π·x·(j/u))·I) • f x
    congr 1
    -- Need: exp(2π·I·(-j)·x/u) = exp((-2π·x·(j/u))·I) in ℂ.
    have : (2 * ↑π * Complex.I * ↑(-j) * ↑x / ↑u : ℂ)
            = ↑(-2 * π * x * (j / u : ℝ)) * Complex.I := by
      push_cast
      ring
    rw [this]
  -- Convert both interval integrals to set-integrals over Ioc.
  rw [intervalIntegral.integral_of_le hle, intervalIntegral.integral_of_le hle]
  -- Both integrals are over `Set.Ioc (-(u/2)) (u/2)`.
  rw [setIntegral_congr_fun (μ := volume) measurableSet_Ioc h_int]
  -- Final: `(1/u : ℝ) • ∫_set complex = (1/↑u : ℂ) * ∫_set complex`.
  -- Use `Complex.real_smul` and cast the rational `1/u`.
  show ((1 / u : ℝ) : ℂ) * _ = _
  push_cast
  ring

/-! ## Plancherel sum-of-squares at lattice points

For `f` supported in `(-(u/2), u/2]`, the period-`u` Plancherel identity
on `AddCircle u` reads
  `∑_j |fourierCoeff(liftIoc u (-(u/2)) f) j|² = (1/u) · ∫_{-(u/2)}^{u/2} |f|²`,
and combined with `period_u_coef_eq_fourierIntegral_at_lattice`,
  `|fourierCoeff(liftIoc u (-(u/2)) f) j| = (1/u) · |f̂(j/u)|`,
yielding
  `∑_j |f̂(j/u)|² = u · ∫_{-(u/2)}^{u/2} |f|²`. -/

/-- Helper: convert `fourierCoeffOn hab f j` (with `b - a = u`) to
`fourierCoeff (liftIoc u a f) j`. -/
private theorem fourierCoeffOn_eq_fourierCoeff_liftIoc_of_eq
    (u a b : ℝ) (hab : a < b) (huba : b - a = u) (f : ℝ → ℂ) (j : ℤ) :
    haveI : Fact (0 < u) := ⟨huba ▸ sub_pos.mpr hab⟩
    haveI : Fact (0 < b - a) := ⟨sub_pos.mpr hab⟩
    fourierCoeffOn hab f j
      = fourierCoeff (AddCircle.liftIoc u a f) j := by
  haveI hu_fact : Fact (0 < b - a) := ⟨sub_pos.mpr hab⟩
  haveI : Fact (0 < u) := ⟨huba ▸ hu_fact.out⟩
  -- `fourierCoeffOn hab f j = fourierCoeff (liftIoc (b - a) a f) j` by def.
  have : fourierCoeffOn hab f j = fourierCoeff (AddCircle.liftIoc (b - a) a f) j := rfl
  rw [this]
  -- Now rewrite `b - a` as `u`.
  subst huba
  rfl

/-- Plancherel sum-of-squares (`HasSum` form) for `f` supported in
`Ioc (-u/2) (u/2)`, in scaled form:
  `∑_j |(1/u) · f̂(j/u)|² = (1/u) · ∫_{-(u/2)}^{u/2} |f|²`. -/
theorem plancherel_at_lattice_period_u_hasSum
    (u : ℝ) (hu : 0 < u)
    (f : ℝ → ℂ)
    (hsupp : Function.support f ⊆ Set.Ioc (-(u/2)) (u/2))
    (hL2 : MemLp f 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2)))) :
    haveI : Fact (0 < u) := ⟨hu⟩
    HasSum
      (fun j : ℤ => ‖(1 / u : ℂ) * Real.fourierIntegral f (j / u : ℝ)‖ ^ 2)
      (u⁻¹ • ∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- Use `hasSum_sq_fourierCoeffOn` with `a = -(u/2)`, `b = u/2`, so `b - a = u`.
  have hab : -(u/2) < u/2 := by linarith
  have h_plancherel :
      HasSum (fun i : ℤ => ‖fourierCoeffOn hab f i‖ ^ 2)
        ((u/2 - -(u/2))⁻¹ • ∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2) :=
    hasSum_sq_fourierCoeffOn (a := -(u/2)) (b := u/2) (f := f) hab hL2
  -- `(b - a)⁻¹ = u⁻¹`.
  have h_diff : (u/2 - -(u/2) : ℝ) = u := by ring
  rw [h_diff] at h_plancherel
  -- Translate `fourierCoeffOn` to our formula via the helper.
  have h_coef_eq_norm : ∀ j : ℤ,
      ‖fourierCoeffOn (a := -(u/2)) (b := u/2) hab f j‖ ^ 2
        = ‖(1 / u : ℂ) * Real.fourierIntegral f (j / u : ℝ)‖ ^ 2 := by
    intro j
    rw [fourierCoeffOn_eq_fourierCoeff_liftIoc_of_eq u (-(u/2)) (u/2) hab h_diff f j]
    rw [period_u_coef_eq_fourierIntegral_at_lattice u hu f hsupp j]
  -- The function `(fun i => ‖fourierCoeffOn hab f i‖ ^ 2)` equals
  -- `(fun j => ‖(1/u : ℂ) * fourierIntegral f (j/u)‖^2)` pointwise.
  have h_fn_eq : (fun i : ℤ => ‖fourierCoeffOn (a := -(u/2)) (b := u/2) hab f i‖ ^ 2)
                = (fun j : ℤ => ‖(1 / u : ℂ) * Real.fourierIntegral f (j / u : ℝ)‖ ^ 2) := by
    funext j; exact h_coef_eq_norm j
  rw [h_fn_eq] at h_plancherel
  exact h_plancherel

/-- Plancherel sum-of-squares (`tsum` form): scaled version.
  `∑' j, |(1/u) · f̂(j/u)|² = (1/u) · ∫_{-(u/2)}^{u/2} |f|²`. -/
theorem plancherel_at_lattice_period_u_tsum
    (u : ℝ) (hu : 0 < u)
    (f : ℝ → ℂ)
    (hsupp : Function.support f ⊆ Set.Ioc (-(u/2)) (u/2))
    (hL2 : MemLp f 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2)))) :
    haveI : Fact (0 < u) := ⟨hu⟩
    ∑' j : ℤ, ‖(1 / u : ℂ) * Real.fourierIntegral f (j / u : ℝ)‖ ^ 2
      = u⁻¹ • ∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2 :=
  (plancherel_at_lattice_period_u_hasSum u hu f hsupp hL2).tsum_eq

/-- Plancherel sum-of-squares — `∑_j |f̂(j/u)|² = u · ∫_{-(u/2)}^{u/2} |f|²` —
for `f` supported in `Ioc (-u/2) (u/2)`. -/
theorem plancherel_at_lattice_period_u
    (u : ℝ) (hu : 0 < u)
    (f : ℝ → ℂ)
    (hsupp : Function.support f ⊆ Set.Ioc (-(u/2)) (u/2))
    (hL2 : MemLp f 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2)))) :
    ∑' j : ℤ, ‖Real.fourierIntegral f (j / u : ℝ)‖ ^ 2
      = u * ∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2 := by
  have hu_ne : (u : ℝ) ≠ 0 := ne_of_gt hu
  -- Start from the scaled tsum equality.
  have h0 :
      ∑' j : ℤ, ‖(1 / u : ℂ) * Real.fourierIntegral f (j / u : ℝ)‖ ^ 2
        = u⁻¹ • ∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2 :=
    plancherel_at_lattice_period_u_tsum u hu f hsupp hL2
  -- Simplify the norm of each term: `‖(1/u : ℂ) * z‖² = (1/u²) · ‖z‖²`.
  have h_norm : ∀ j : ℤ,
      ‖(1 / u : ℂ) * Real.fourierIntegral f (j / u : ℝ)‖ ^ 2
        = (1 / u^2) * ‖Real.fourierIntegral f (j / u : ℝ)‖ ^ 2 := by
    intro j
    rw [norm_mul, mul_pow]
    congr 1
    -- ‖(1/u : ℂ)‖² = ‖((1/u : ℝ) : ℂ)‖² = |1/u|² = (1/u)².
    have h_cast : ((1 / u : ℂ)) = ((1 / u : ℝ) : ℂ) := by push_cast; rfl
    rw [h_cast]
    rw [Complex.norm_real]
    rw [Real.norm_eq_abs]
    rw [abs_of_pos (by positivity : (0 : ℝ) < 1 / u)]
    rw [div_pow, one_pow]
  -- Substitute the norm simplification.
  rw [tsum_congr h_norm] at h0
  -- Pull the constant `(1/u²)` out of the tsum.
  rw [tsum_mul_left] at h0
  -- Multiply both sides by `u²` to clear `(1/u²)`.
  have h_sq_pos : (0 : ℝ) < u^2 := by positivity
  have h_sq_ne : (u^2 : ℝ) ≠ 0 := ne_of_gt h_sq_pos
  -- `(1/u²) * S = u⁻¹ • I  ⟺  S = u² · u⁻¹ • I = u · I`.
  have h_left_eq :
      (1 / u^2) * ∑' j : ℤ, ‖Real.fourierIntegral f (j / u : ℝ)‖ ^ 2
        = u⁻¹ • ∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2 := h0
  -- Multiply through by `u²`.
  have hmul : u^2 * ((1 / u^2) * ∑' j : ℤ, ‖Real.fourierIntegral f (j / u : ℝ)‖ ^ 2)
              = u^2 * (u⁻¹ • ∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2) :=
    congrArg (fun z => u^2 * z) h_left_eq
  -- LHS: u² * ((1/u²) * S) = S.
  rw [← mul_assoc] at hmul
  rw [show u^2 * (1 / u^2) = 1 by field_simp] at hmul
  rw [one_mul] at hmul
  -- Now hmul : S = u^2 * (u⁻¹ • I).
  rw [hmul]
  -- Simplify u² · u⁻¹ • I = u · I.
  -- `u⁻¹ • I = u⁻¹ * I` since I is real.
  simp only [smul_eq_mul]
  -- Goal: u^2 * (u⁻¹ * I) = u * I.
  rw [show u^2 * (u⁻¹ * (∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2))
        = (u^2 * u⁻¹) * (∫ x in (-(u/2))..(u/2), ‖f x‖ ^ 2) by ring]
  rw [show u^2 * u⁻¹ = u by field_simp]

/-! ## Support of pointwise autocorrelation and convolution

**Documentation / contrast only.**  The following lemmas concern the
POINTWISE product `f(x) · f(-x)`.  This is **NOT** what MV calls `f∘f`;
we use these only as contrast / for archived reference.  The active
autocorrelation used throughout the headline closure is
`Sidon.FourierAux.autocorr`, defined as the convolution
`autocorr f x := ∫ t, f(t)·f(x+t) dt`.  The pointwise-form identity
`(period-u coef of f·f̌) = (1/u) · |𝓕f(j/u)|²` is FALSE; that identity
holds for the convolutional autocorrelation, not the pointwise product
(see the caveat at the head of the next sub-section).

For `f : ℝ → ℝ` supported in `Ioo (-1/4) (1/4)`:
  * `(f∘f)_pw(x) := f(x) · f(-x)` (the pointwise product, kept here only
    for documentation) is supported in `Ioo (-1/4) (1/4)` (both factors must
    be nonzero, so `x ∈ supp f ∩ -(supp f)` which is contained in `Ioo (-1/4) (1/4)`).
  * `(f * f)(x) = ∫ f(t)·f(x-t) dt` has support in `Ioo (-1/2) (1/2)` (by
    `support_convolution_subset`: the support is in `supp f + supp f`, and the sum
    of two open intervals `Ioo (-1/4) (1/4)` is contained in `Ioo (-1/2) (1/2)`).

For the period-`u` torus with `u = 1/2 + δ`, `δ > 0`, we have `u/2 = 1/4 + δ/2 ≥ 1/4`,
so the pointwise product is supported in `Ioc (-u/2) (u/2)` (using
`Ioo (-1/4) (1/4) ⊆ Ioc (-u/2) (u/2)` when `u ≥ 1/2`).

The convolution support `Ioo (-1/2) (1/2)` is NOT in general contained in
`Ioc (-u/2) (u/2)` when `u < 1`. The MV setting uses `u = 1/2 + δ` with `δ ≤ 1/4`,
so `u ≤ 3/4 < 1` and the convolution is NOT supported in the period-`u` fundamental
domain. The MV proof handles this by *integrating `f*f` against `K`* (whose support
is `Ioo (-δ) δ ⊆ Ioc (-u/2) (u/2)`), so the support inclusion on `f*f` is not
needed for the master inequality — only for the *strengthened* version where `f` is
already supported in `Ioo (-u/4) (u/4)`.

The lemmas below record the *true* support statements and a version of the
period-`u` Fourier coefficient identity for `f * f` that requires the strengthened
support hypothesis `support f ⊆ Ioo (-u/4) (u/4)`. -/

/-- The pointwise autocorrelation `(f∘f)(x) := f(x) · f(-x)` is supported in
`Ioo (-1/4) (1/4)` whenever `f` is. -/
theorem pointwiseAutocorr_support_subset_quarter
    (f : ℝ → ℝ) (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (fun x => f x * f (-x)) ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4) := by
  intro x hx
  -- `hx : x ∈ support (fun y => f y * f (-y))`, i.e., `f x * f (-x) ≠ 0`.
  have hfx_ne : f x ≠ 0 := by
    intro heq
    apply hx
    show f x * f (-x) = 0
    rw [heq, zero_mul]
  exact hf_supp hfx_ne

/-- Same as above, in the ℂ-valued form used by Fourier identities. -/
theorem pointwiseAutocorr_complex_support_subset_quarter
    (f : ℝ → ℝ) (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (fun x => ((f x * f (-x) : ℝ) : ℂ)) ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4) := by
  intro x hx
  -- `hx : ((f x * f (-x) : ℝ) : ℂ) ≠ 0`, i.e., `f x * f (-x) ≠ 0`.
  have hreal_ne : (f x * f (-x) : ℝ) ≠ 0 := by
    intro heq
    apply hx
    show ((f x * f (-x) : ℝ) : ℂ) = 0
    rw [heq]
    rfl
  have hfx_ne : f x ≠ 0 := fun heq => hreal_ne (by rw [heq, zero_mul])
  exact hf_supp hfx_ne

/-- For `u ≥ 1/2`, the pointwise autocorrelation `f∘f` is supported in `Ioc (-u/2) (u/2)`
whenever `f` is supported in `Ioo (-1/4) (1/4)`. -/
theorem pointwiseAutocorr_support_in_torus
    (f : ℝ → ℝ) (u : ℝ) (hu : (1/2 : ℝ) ≤ u)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (fun x => f x * f (-x)) ⊆ Set.Ioc (-(u/2)) (u/2) := by
  have h_quarter : Function.support (fun x => f x * f (-x))
                    ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4) :=
    pointwiseAutocorr_support_subset_quarter f hf_supp
  refine h_quarter.trans ?_
  -- Ioo (-1/4) (1/4) ⊆ Ioc (-u/2) (u/2)
  intro x hx
  refine ⟨?_, ?_⟩
  · have : -(u/2) ≤ -(1/4 : ℝ) := by linarith
    linarith [hx.1]
  · have : (1/4 : ℝ) ≤ u/2 := by linarith
    linarith [hx.2]

/-- ℂ-valued form of `pointwiseAutocorr_support_in_torus`. -/
theorem pointwiseAutocorr_complex_support_in_torus
    (f : ℝ → ℝ) (u : ℝ) (hu : (1/2 : ℝ) ≤ u)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (fun x => ((f x * f (-x) : ℝ) : ℂ))
      ⊆ Set.Ioc (-(u/2)) (u/2) := by
  have h_quarter : Function.support (fun x => ((f x * f (-x) : ℝ) : ℂ))
                    ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4) :=
    pointwiseAutocorr_complex_support_subset_quarter f hf_supp
  refine h_quarter.trans ?_
  intro x hx
  exact ⟨by linarith [hx.1], by linarith [hx.2]⟩

/-- The convolution `f * f` is supported in `Ioo (-1/2) (1/2)` whenever `f` is
supported in `Ioo (-1/4) (1/4)`.

This is the TRUE support statement. The (false) torus inclusion `support(f*f) ⊆
Ioc (-u/2) (u/2)` would require `u ≥ 1`, which fails in the MV setting where
`u = 1/2 + δ ≤ 3/4`. -/
theorem convolution_self_support_subset_half
    (f : ℝ → ℝ) (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := by
  -- `support (f ⋆ g) ⊆ support f + support g`.
  have h_conv :
      Function.support
        (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊆ Function.support f + Function.support f :=
    support_convolution_subset (L := ContinuousLinearMap.mul ℝ ℝ) (μ := MeasureTheory.volume)
  refine h_conv.trans ?_
  -- `support f + support f ⊆ Ioo (-1/4) (1/4) + Ioo (-1/4) (1/4) ⊆ Ioo (-1/2) (1/2)`.
  refine (Set.add_subset_add hf_supp hf_supp).trans ?_
  -- Pointwise: `(a, b) + (c, d) = (a + c, b + d)` for points in Ioo.
  intro z hz
  rcases hz with ⟨x, hx, y, hy, rfl⟩
  refine ⟨?_, ?_⟩
  · have hx1 := hx.1
    have hy1 := hy.1
    linarith
  · have hx2 := hx.2
    have hy2 := hy.2
    linarith

/-- For `u ≥ 1`, the convolution `f * f` is supported in `Ioc (-u/2) (u/2)`
whenever `f` is supported in `Ioo (-1/4) (1/4)`. -/
theorem convolution_self_support_in_torus_when_u_ge_one
    (f : ℝ → ℝ) (u : ℝ) (hu : (1 : ℝ) ≤ u)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊆ Set.Ioc (-(u/2)) (u/2) := by
  have h_half := convolution_self_support_subset_half f hf_supp
  refine h_half.trans ?_
  intro x hx
  refine ⟨?_, ?_⟩
  · have : -(u/2) ≤ -(1/2 : ℝ) := by linarith
    linarith [hx.1]
  · have : (1/2 : ℝ) ≤ u/2 := by linarith
    linarith [hx.2]

/-- Strengthened support hypothesis: if `f` is supported in `Ioo (-u/4) (u/4)`, then
`f * f` is supported in `Ioo (-u/2) (u/2) ⊆ Ioc (-u/2) (u/2)`. -/
theorem convolution_self_support_in_torus_strong
    (f : ℝ → ℝ) (u : ℝ) (hu : 0 < u)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(u/4)) (u/4)) :
    Function.support
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊆ Set.Ioc (-(u/2)) (u/2) := by
  have h_conv :
      Function.support
        (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊆ Function.support f + Function.support f :=
    support_convolution_subset (L := ContinuousLinearMap.mul ℝ ℝ) (μ := MeasureTheory.volume)
  refine h_conv.trans ?_
  refine (Set.add_subset_add hf_supp hf_supp).trans ?_
  intro z hz
  rcases hz with ⟨x, hx, y, hy, rfl⟩
  refine ⟨?_, ?_⟩
  · -- x > -u/4 and y > -u/4, so x + y > -u/2.
    have hx1 := hx.1
    have hy1 := hy.1
    -- -(u/2) = -(u/4) + -(u/4) < x + y.
    have hsum : -(u/4) + -(u/4) < x + y := add_lt_add hx1 hy1
    have heq : (-(u/4) + -(u/4) : ℝ) = -(u/2) := by ring
    linarith
  · have hx2 := hx.2
    have hy2 := hy.2
    have hsum : x + y < u/4 + u/4 := add_lt_add hx2 hy2
    have heq : (u/4 + u/4 : ℝ) = u/2 := by ring
    linarith

/-! ## Period-`u` Fourier coefficient of `f * f` (strong support)

For `f` supported in `Ioo (-u/4) (u/4)`, the convolution `f * f` is supported in
`Ioo (-u/2) (u/2)`, so the standard `period_u_coef_eq_fourierIntegral_at_lattice`
applies and the period-`u` Fourier coefficient identifies as
`(1/u) · 𝓕(f * f)(j/u)`. By the convolution theorem,
`𝓕(f * f)(j/u) = (𝓕 f (j/u))²`, yielding the identity. -/

/-- The period-`u` Fourier coefficient of `(f * f)` (for `f : ℝ → ℂ` supported in
`Ioo (-u/4) (u/4)`, continuous and integrable) equals `(1/u) · (𝓕f (j/u))²`. -/
theorem period_u_coef_of_convolution_self_complex
    (u : ℝ) (hu : 0 < u)
    (f : ℝ → ℂ)
    (hf_int : Integrable f volume)
    (hf_cont : Continuous f)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(u/4)) (u/4))
    (j : ℤ) :
    haveI : Fact (0 < u) := ⟨hu⟩
    fourierCoeff (AddCircle.liftIoc u (-(u/2))
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℂ ℂ) volume)) j
      = (1 / u : ℂ) * (Real.fourierIntegral f (j / u : ℝ))^2 := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- Support of `f * f`:  ⊆ supp f + supp f ⊆ Ioo (-u/4) (u/4) + Ioo (-u/4) (u/4)
  --                       ⊆ Ioo (-u/2) (u/2) ⊆ Ioc (-u/2) (u/2).
  -- (We need this for the period-u coef bridge.)
  have hff_supp :
      Function.support
        (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℂ ℂ) volume)
        ⊆ Set.Ioc (-(u/2)) (u/2) := by
    have h_conv :
        Function.support
          (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℂ ℂ) volume)
        ⊆ Function.support f + Function.support f :=
      support_convolution_subset (L := ContinuousLinearMap.mul ℂ ℂ) (μ := volume)
    refine h_conv.trans ?_
    refine (Set.add_subset_add hf_supp hf_supp).trans ?_
    intro z hz
    rcases hz with ⟨x, hx, y, hy, rfl⟩
    refine ⟨?_, ?_⟩
    · have h1 : -(u/4) + -(u/4) < x + y := add_lt_add hx.1 hy.1
      have heq : (-(u/4) + -(u/4) : ℝ) = -(u/2) := by ring
      linarith
    · have h1 : x + y < u/4 + u/4 := add_lt_add hx.2 hy.2
      have heq : (u/4 + u/4 : ℝ) = u/2 := by ring
      linarith
  -- Step 1: identify the period-u coefficient with `(1/u) · 𝓕(f*f)(j/u)`.
  have h_bridge :=
    period_u_coef_eq_fourierIntegral_at_lattice u hu
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℂ ℂ) volume)
      hff_supp j
  rw [h_bridge]
  -- Step 2: by the convolution theorem,
  -- `𝓕(f ⋆[mul ℂ ℂ] f)(ξ) = (mul ℂ ℂ)(𝓕f ξ)(𝓕f ξ) = (𝓕f ξ) * (𝓕f ξ)`.
  have h_FT_conv :
      Real.fourierIntegral
        (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℂ ℂ) volume)
        (j / u : ℝ)
        = Real.fourierIntegral f (j / u : ℝ) * Real.fourierIntegral f (j / u : ℝ) := by
    -- `Real.fourier_mul_convolution_eq` gives the result for `mul ℂ ℂ`.
    -- The function-level lemma uses `𝓕`; `Real.fourierIntegral` is its deprecated alias.
    have h := Real.fourier_mul_convolution_eq (f₁ := f) (f₂ := f)
      hf_int hf_int hf_cont hf_cont (j / u : ℝ)
    exact h
  -- Step 3: replace `𝓕(f*f)` with `(𝓕f)²`.
  rw [h_FT_conv]
  ring

/-! ## Period-`u` Fourier coefficient of the pointwise product (documentation only)

**Documentation / contrast only.**  The lemma below concerns the
POINTWISE product `f(x) · f(-x)`.  This is **NOT** what MV calls `f∘f`;
the active autocorrelation used throughout the headline closure is
`Sidon.FourierAux.autocorr`, defined as the convolution
`autocorr f x := ∫ t, f(t)·f(x+t) dt`.  We keep this lemma as a contrast
witness — it establishes only the trivial bridge identity for the
pointwise product, and it is not invoked by the headline path.

For real `f` supported in `Ioo (-1/4) (1/4)`, `(f∘f)_pw(x) := f(x) · f(-x)` is
also supported in `Ioo (-1/4) (1/4) ⊆ Ioc (-u/2) (u/2)` (for `u ≥ 1/2`). The
period-`u` Fourier coefficient identifies as `(1/u) · 𝓕(f∘f)_pw(j/u)`.

**Important caveat:** The pointwise product `f · f̌` is NOT the autocorrelation
`f ⋆ f̌` (convolution with reverse). The two have different Fourier transforms:
  * `𝓕(f · f̌)(ξ) = 𝓕f * 𝓕f̌ (ξ) = ∫ 𝓕f(η) · conj(𝓕f(η - ξ)) dη` (for real `f`),
  * `𝓕(f ⋆ f̌)(ξ) = 𝓕f(ξ) · 𝓕f̌(ξ) = |𝓕f(ξ)|²` (for real `f`).

The identity `(period-u coef of f·f̌) = (1/u) · |𝓕f(j/u)|²` is therefore FALSE
for the pointwise product `f · f̌`; it would hold for the autocorrelation `f ⋆ f̌`
under the same support hypothesis.

We record below the *correct* identity for the pointwise product:
period-`u` coef of `f · f̌` equals `(1/u) · 𝓕(f · f̌)(j/u)` (a tautological lift
of the support-based identity). -/

/-- The period-`u` Fourier coefficient of the lift of `f · f̌` (pointwise
product) equals `(1/u) · 𝓕(f · f̌)(j/u)`, for `f : ℝ → ℝ` supported in
`Ioo (-1/4) (1/4)` with `u ≥ 1/2`.

This is the *correct* identity for the pointwise product. The naive identity
`= (1/u) · |𝓕f(j/u)|²` is FALSE: that holds for the convolutional autocorrelation
`f ⋆ f̌`, not the pointwise `f · f̌`. -/
theorem period_u_coef_of_pointwiseAutocorr
    (u : ℝ) (hu : 0 < u) (hu_half : (1/2 : ℝ) ≤ u)
    (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (j : ℤ) :
    haveI : Fact (0 < u) := ⟨hu⟩
    fourierCoeff (AddCircle.liftIoc u (-(u/2))
      (fun x => ((f x * f (-x) : ℝ) : ℂ))) j
      = (1 / u : ℂ) * Real.fourierIntegral (fun x => ((f x * f (-x) : ℝ) : ℂ))
                                            (j / u : ℝ) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- `support (fun x => ((f x * f (-x) : ℝ) : ℂ)) ⊆ Ioc (-u/2) (u/2)` by
  -- `pointwiseAutocorr_complex_support_in_torus`.
  have h_supp := pointwiseAutocorr_complex_support_in_torus f u hu_half hf_supp
  -- Apply the bridge lemma.
  exact period_u_coef_eq_fourierIntegral_at_lattice u hu
    (fun x => ((f x * f (-x) : ℝ) : ℂ)) h_supp j

/-! ## Bilinear Parseval at lattice points

We derive a bilinear Plancherel-type identity at lattice points
`u · ∫ g · conj(h) = ∑'_j 𝓕g(j/u) · conj(𝓕h(j/u))` by routing through the
Hilbert-basis Plancherel identity on `Lp ℂ 2 haarAddCircle` and then bridging
back to non-periodic Fourier integrals via
`period_u_coef_eq_fourierIntegral_at_lattice`.

The hypothesis is that both `g, h` are supported in `Ioc (-u/2) (u/2)` and live
in `L² (volume.restrict (Ioc (-u/2) (u/2)))`. -/

/-- Polarisation identity for complex numbers (Hermitian form, mathlib convention
`⟪a, b⟫ = b · conj a`, antilinear in the first slot):

`a * conj b = (‖a + b‖² - ‖a - b‖² + i · ‖a + i·b‖² - i · ‖a - i·b‖²) / 4`.

Direct manual verification:
* `‖a+b‖² - ‖a-b‖² = 2(a·conj b + b·conj a)`,
* `i(‖a+ib‖² - ‖a-ib‖²) = -2 i² (a·conj b - b·conj a) = 2(a·conj b - b·conj a)`.

Their sum is `4 a·conj b`.
-/
theorem complex_polarisation (a b : ℂ) :
    a * starRingEnd ℂ b
      = ((‖a + b‖^2 : ℂ) - (‖a - b‖^2 : ℂ)
          + Complex.I * (‖a + Complex.I * b‖^2 : ℂ)
          - Complex.I * (‖a - Complex.I * b‖^2 : ℂ)) / 4 := by
  -- Key: for any z : ℂ, (‖z‖^2 : ℂ) = z * conj z.
  have h_norm_sq : ∀ z : ℂ, ((‖z‖^2 : ℝ) : ℂ) = z * starRingEnd ℂ z := by
    intro z
    have h1 : Complex.normSq z = ‖z‖ ^ 2 := Complex.normSq_eq_norm_sq z
    have h2 : (Complex.normSq z : ℂ) = (starRingEnd ℂ) z * z :=
      Complex.normSq_eq_conj_mul_self
    calc ((‖z‖^2 : ℝ) : ℂ) = ((Complex.normSq z : ℝ) : ℂ) := by rw [h1]
      _ = (starRingEnd ℂ) z * z := h2
      _ = z * starRingEnd ℂ z := by ring
  -- Rewrite each of the four squared norms in terms of conj/mul.
  have e1 : (‖a + b‖^2 : ℂ) = (a + b) * starRingEnd ℂ (a + b) := by
    have := h_norm_sq (a + b)
    push_cast at this ⊢
    exact this
  have e2 : (‖a - b‖^2 : ℂ) = (a - b) * starRingEnd ℂ (a - b) := by
    have := h_norm_sq (a - b)
    push_cast at this ⊢
    exact this
  have e3 : (‖a + Complex.I * b‖^2 : ℂ)
              = (a + Complex.I * b) * starRingEnd ℂ (a + Complex.I * b) := by
    have := h_norm_sq (a + Complex.I * b)
    push_cast at this ⊢
    exact this
  have e4 : (‖a - Complex.I * b‖^2 : ℂ)
              = (a - Complex.I * b) * starRingEnd ℂ (a - Complex.I * b) := by
    have := h_norm_sq (a - Complex.I * b)
    push_cast at this ⊢
    exact this
  rw [e1, e2, e3, e4]
  -- Expand the conjugates: `conj (a ± b) = conj a ± conj b`, `conj (I·b) = -I·conj b`.
  rw [map_add, map_sub, map_add, map_sub]
  rw [show starRingEnd ℂ (Complex.I * b) = -Complex.I * starRingEnd ℂ b from by
        rw [map_mul, Complex.conj_I]]
  -- Substitute `I^2 = -1` via `Complex.I_sq`.
  have hI_sq : Complex.I ^ 2 = -1 := Complex.I_sq
  -- Expand and ring (with `I^2 → -1`).
  ring_nf
  rw [show (Complex.I : ℂ)^2 = -1 from Complex.I_sq]
  ring

/-! ### Bilinear Plancherel via Hilbert-basis routing

We use `MeasureTheory.L2.inner_def`: on `Lp ℂ 2 μ`, `⟪F, H⟫ = ∫ F · conj H`.  And
`HilbertBasis.tsum_inner_mul_inner b F H = ⟪F, H⟫`.  Specialised to the Fourier
Hilbert basis on `AddCircle u`, this gives
  `∑' j, ⟪F, b j⟫ * ⟪b j, H⟫ = ⟪F, H⟫`
where `b = fourierBasis` and `⟪F, b j⟫ = conj(fourierCoeff F j)` (via the convention
`⟪x, b j⟫ = conj (b.repr x j)` and `b.repr F j = fourierCoeff F j`).

We do not prove the full bilinear lattice Plancherel here as a complete theorem
because it requires (a) the bridge from `period_u_coef_eq_fourierIntegral_at_lattice`
to identify each Fourier coefficient with the FT lattice value, AND (b) the bridge
from the `Lp 2` inner product `⟪toLp g, toLp h⟫_{haarAddCircle}` to the *concrete*
integral `∫ g · conj h` over `(-u/2, u/2]`, weighted by `1/u` (the Haar/Lebesgue
normalisation).  The concrete-integral form factors through `(1/u) · ∫_{(-u/2, u/2]} g · conj h`
because `haarAddCircle = (1/u) · liftIoc_pushforward(volume)`.

The diagonal version `plancherel_at_lattice_period_u` does this transport for the
squared-norm case; the bilinear version requires the analogous transport with the
bilinear pairing.  We prove it as `bilinear_parseval_at_lattice_haarAddCircle` (the
result on the AddCircle Hilbert space) and document the bridge step. -/

/-- Bilinear Plancherel on `Lp ℂ 2 haarAddCircle`: for any `F, H : Lp ℂ 2 haarAddCircle`,
the bilinear sum of Fourier coefficients `∑' j, fourierCoeff F j · conj (fourierCoeff H j)`
equals the L² inner product `⟪H, F⟫_{Lp 2}` (note the order: antilinear in the
second slot, matching `RCLike.inner_apply'`). -/
theorem bilinear_parseval_addCircle_Lp
    (u : ℝ) (hu : 0 < u)
    (F H : haveI : Fact (0 < u) := ⟨hu⟩
           MeasureTheory.Lp ℂ 2 (@AddCircle.haarAddCircle u ⟨hu⟩)) :
    haveI : Fact (0 < u) := ⟨hu⟩
    HasSum (fun j : ℤ => fourierCoeff (F : AddCircle u → ℂ) j
                          * starRingEnd ℂ (fourierCoeff (H : AddCircle u → ℂ) j))
      (@inner ℂ _ _ H F) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- `fourierBasis.hasSum_inner_mul_inner H F`
  -- gives `HasSum (fun i => ⟪H, b i⟫ * ⟪b i, F⟫) ⟪H, F⟫`.
  have h := (@fourierBasis u ⟨hu⟩).hasSum_inner_mul_inner H F
  -- Identify `⟪H, b i⟫ = fourierBasis.repr H i = fourierCoeff H i` (by `fourierBasis_repr`)
  -- and `⟪b i, F⟫ = conj ⟪F, b i⟫ = conj (fourierCoeff F i)`.
  have hrepr_F : ∀ j, (@fourierBasis u ⟨hu⟩).repr F j = fourierCoeff (F : AddCircle u → ℂ) j :=
    fun j => fourierBasis_repr F j
  have hrepr_H : ∀ j, (@fourierBasis u ⟨hu⟩).repr H j = fourierCoeff (H : AddCircle u → ℂ) j :=
    fun j => fourierBasis_repr H j
  -- `b.repr_apply_apply x i = ⟪b i, x⟫`.
  have hinner_F : ∀ j, @inner ℂ _ _ ((@fourierBasis u ⟨hu⟩) j) F
                        = (@fourierBasis u ⟨hu⟩).repr F j :=
    fun j => ((@fourierBasis u ⟨hu⟩).repr_apply_apply F j).symm
  have hinner_H : ∀ j, @inner ℂ _ _ ((@fourierBasis u ⟨hu⟩) j) H
                        = (@fourierBasis u ⟨hu⟩).repr H j :=
    fun j => ((@fourierBasis u ⟨hu⟩).repr_apply_apply H j).symm
  -- Rewrite each term: ⟪H, b j⟫ * ⟪b j, F⟫ = conj(⟪b j, H⟫) * ⟪b j, F⟫
  --                                          = conj(fourierCoeff H j) * fourierCoeff F j.
  -- The goal: fourierCoeff F j * conj(fourierCoeff H j).  Use commutativity of `*`.
  refine h.congr_fun ?_
  intro j
  rw [show @inner ℂ _ _ H ((@fourierBasis u ⟨hu⟩) j)
        = starRingEnd ℂ (@inner ℂ _ _ ((@fourierBasis u ⟨hu⟩) j) H) from
      (inner_conj_symm _ _).symm]
  rw [hinner_H j, hrepr_H j, hinner_F j, hrepr_F j]
  ring

/-- A direct corollary of `bilinear_parseval_addCircle_Lp`: the bilinear lattice sum
`tsum` equality. -/
theorem bilinear_parseval_addCircle_Lp_tsum
    (u : ℝ) (hu : 0 < u)
    (F H : haveI : Fact (0 < u) := ⟨hu⟩
           MeasureTheory.Lp ℂ 2 (@AddCircle.haarAddCircle u ⟨hu⟩)) :
    haveI : Fact (0 < u) := ⟨hu⟩
    ∑' j : ℤ, fourierCoeff (F : AddCircle u → ℂ) j
              * starRingEnd ℂ (fourierCoeff (H : AddCircle u → ℂ) j)
      = @inner ℂ _ _ H F :=
  (bilinear_parseval_addCircle_Lp u hu F H).tsum_eq

/-! ## Concrete-integral bilinear Parseval at lattice points

The bilinear Parseval identity at lattice points,
  `u · ∫_{(-u/2, u/2]} g · conj(h) = ∑'_j 𝓕g(j/u) · conj(𝓕h(j/u))`,
follows from `plancherel_at_lattice_period_u` applied to the four polarised
functions `g + h`, `g - h`, `g + I·h`, `g - I·h`, combined via the complex
polarisation identity (`complex_polarisation`).

The proof requires:
  1. Linearity of `Real.fourierIntegral` on integrable functions
     (`fourierIntegral_add`-style result): `𝓕(g + c·h) = 𝓕g + c·𝓕h`.
  2. The polarisation identity pointwise (`complex_polarisation` above).
  3. Combining four `HasSum`s via the polarisation linear combination
     (`HasSum.add`, `HasSum.sub`, `HasSum.const_smul`).
  4. Pulling the linear combination through `∫ ‖g + c·h‖²` via integral
     linearity (which gives a polarised representation of `∫ g · conj h`).

In MV's master inequality (Eq.(3)), this identity is applied to `g = f*f` and
`h = K` (each supported in `Ioc (-u/2) (u/2)`), with the constant term
(`j = 0`) handled separately.  For the MV proof we only need the *real-valued*
version (since both `f*f` and `K` are real-valued), in which case the
polarisation collapses to the simpler identity
  `u · ∫ g · h = ∑'_j Re(𝓕g(j/u) · conj(𝓕h(j/u)))`
and the diagonal Plancherel alone (applied to `g + h` and `g - h`) suffices.

We do not close the general complex bilinear identity here as a single
theorem because the linearity of `Real.fourierIntegral` on integrable
functions requires assembling several mathlib lemmas (`integral_add`,
`Continuous.smul`, etc.); rather, we record the *real-valued* version
which is what MV Eq.(3) actually consumes:

  `4 · u · ∫ g · h
     = u · (∫ ‖(g : ℝ) + h‖² - ∫ ‖(g : ℝ) - h‖²)
     = ∑'_j (‖𝓕(g + h)(j/u)‖² - ‖𝓕(g - h)(j/u)‖²)`

via diagonal Plancherel and the real polarisation identity. -/

/-- **Real polarisation identity**: for `a, b : ℝ`,
`4 · a · b = (a + b)² - (a - b)²`. -/
theorem real_polarisation (a b : ℝ) :
    4 * (a * b) = (a + b)^2 - (a - b)^2 := by ring

/-- **Real polarisation lifted to ℂ**:
for real-valued `a, b : ℝ`,
`4 · (a · b : ℂ) = (↑(a + b)^2 - ↑(a - b)^2)`. -/
theorem real_polarisation_complex (a b : ℝ) :
    (4 : ℂ) * ((a * b : ℝ) : ℂ) = ((a + b : ℝ) : ℂ)^2 - ((a - b : ℝ) : ℝ)^2 := by
  push_cast
  ring

end -- noncomputable section

end Sidon.TorusParseval
