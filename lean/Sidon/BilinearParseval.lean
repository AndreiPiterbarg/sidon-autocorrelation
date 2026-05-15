/-
Sidon Autocorrelation Project — Bilinear Period-`u` Parseval Bridge
====================================================================

This file provides the bilinear period-`u` Parseval bridge: the
passage from the abstract `Lp 2` inner product
`bilinear_parseval_addCircle_Lp` (in `Sidon.TorusParseval`) to the
concrete-integral form

  ```
  u · ∫_ℝ g(x) · conj(h(x)) dx
    = ∑' j : ℤ, 𝓕g(j/u) · conj(𝓕h(j/u))
  ```

for `g, h : ℝ → ℂ` integrable, square-integrable, and compactly
supported in `Ioo (-(u/2)) (u/2)`.

The diagonal (`g = h`) version is the existing
`plancherel_at_lattice_period_u`.  The bilinear extension is what the
MV proof of `mv_eq2` (the `1 + √(R-1)·√(K₂-1)` bound) consumes.

Main results.
  * `memLp_liftIoc_haarAddCircle` — `MemLp` of the lift on `haarAddCircle`.
  * `fourierCoeff_toLp_eq_period_u` — the Fourier coefficient identification
    `fourierCoeff (toLp (liftIoc g)) j = (1/u) · 𝓕g(j/u)` (via the ae-bridge
    + `period_u_coef_eq_fourierIntegral_at_lattice`).
  * `inner_toLp_eq_intervalIntegral` — the `Lp 2 haarAddCircle` inner-product
    bridge to the concrete `(1/u) · ∫_{(-u/2..u/2)} g · conj h` integral.
  * `bilinear_parseval_period_u` — the consumer-facing form:
    `u · ∫_ℝ g · conj h = ∑' j, 𝓕g(j/u) · conj(𝓕h(j/u))`.
  * `bilinear_parseval_period_u_inv` — the equivalent rescaled form
    `∫_ℝ g · conj h = (1/u) · ∑' j, 𝓕g(j/u) · conj(𝓕h(j/u))`.

**Note on convention.** The natural statement matching the diagonal
`plancherel_at_lattice_period_u` (`∑' j, |𝓕f(j/u)|² = u · ∫ |f|²`) is
the form `u · ∫ g · conj h = ∑' j, 𝓕g(j/u) · conj(𝓕h(j/u))`, NOT
`∫ = u · ∑'`.  The user-facing `bilinear_parseval_period_u_inv` form
expresses the same identity as `∫ = (1/u) · ∑'`.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.Defs
import Sidon.TorusParseval

namespace Sidon.BilinearParseval

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Pointwise

set_option maxHeartbeats 4000000
set_option linter.mathlibStandardSet false
set_option linter.deprecated false
set_option linter.unusedVariables false

noncomputable section

/-! ## L² promotion of L¹ ∩ L² functions to `Lp 2 haarAddCircle`

Given `g : ℝ → ℂ` with `MemLp g 2 volume` and `support g ⊆ Ioo (-(u/2)) (u/2)`,
the lifted function `liftIoc u (-(u/2)) g : AddCircle u → ℂ` is in
`Lp 2 haarAddCircle`. -/

/-- If `g : ℝ → ℂ` is square-integrable on all of `ℝ`, then its restriction
to `Ioc (-(u/2)) (u/2)` is square-integrable.  This is the standard
`MemLp.restrict` lemma in disguise. -/
private theorem memLp_restrict_of_memLp_volume
    (u : ℝ) (hu : 0 < u)
    (g : ℝ → ℂ) (hg_L2 : MemLp g 2 volume) :
    MemLp g 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2))) :=
  hg_L2.restrict (Set.Ioc (-(u/2)) (u/2))

/-- Bridge: `liftIoc u (-(u/2)) g` is in `Lp 2 haarAddCircle`, when
`g` is `MemLp g 2 volume.restrict (Ioc (-(u/2)) (u/2))`. -/
theorem memLp_liftIoc_haarAddCircle
    (u : ℝ) (hu : 0 < u)
    (g : ℝ → ℂ)
    (hg_L2_restrict : MemLp g 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2)))) :
    haveI : Fact (0 < u) := ⟨hu⟩
    MemLp (AddCircle.liftIoc u (-(u/2)) g) 2 (@AddCircle.haarAddCircle u ⟨hu⟩) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- `hg_L2_restrict` has type `MemLp g 2 (volume.restrict (Ioc (-(u/2)) (u/2)))`,
  -- and `memLp_liftIoc` expects `MemLp g 2 (volume.restrict (Ioc t (t + T)))`.
  -- We set `t := -(u/2)` and rely on `t + T = u/2`.
  have h_eq : (-(u/2) + u : ℝ) = u/2 := by ring
  have hg_L2_restrict' :
      MemLp g 2 (volume.restrict (Set.Ioc (-(u/2)) (-(u/2) + u))) := by
    rw [h_eq]; exact hg_L2_restrict
  have h_lift_L2 : MemLp (AddCircle.liftIoc u (-(u/2)) g) 2 (volume : Measure (AddCircle u)) :=
    hg_L2_restrict'.memLp_liftIoc
  exact h_lift_L2.haarAddCircle

/-! ## Identification of Fourier coefficients with `𝓕g(j/u)`

For `g : ℝ → ℂ` supported in `Ioc (-(u/2)) (u/2)`, `MemLp` on the
restriction implies that the `fourierCoeff` of the *Lp-class* representative
of `liftIoc u (-(u/2)) g` agrees with `(1/u) · 𝓕g(j/u)`.  This is the
ae-bridge step. -/

/-- The Fourier coefficient of the `Lp 2 haarAddCircle` representative of
`liftIoc u (-(u/2)) g` equals `(1/u) · 𝓕g(j/u)`, when `g` is supported in
`Ioc (-(u/2)) (u/2)`. -/
theorem fourierCoeff_toLp_eq_period_u
    (u : ℝ) (hu : 0 < u)
    (g : ℝ → ℂ)
    (hg_supp : Function.support g ⊆ Set.Ioc (-(u/2)) (u/2))
    (hg_L2_restrict : MemLp g 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2))))
    (j : ℤ) :
    haveI : Fact (0 < u) := ⟨hu⟩
    fourierCoeff
        ((memLp_liftIoc_haarAddCircle u hu g hg_L2_restrict).toLp
          (AddCircle.liftIoc u (-(u/2)) g) : AddCircle u → ℂ) j
      = (1 / u : ℂ) * Real.fourierIntegral g (j / u : ℝ) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- Step 1: `fourierCoeff` of the `toLp` equals `fourierCoeff` of `liftIoc g`
  -- (by `fourierCoeff_congr_ae` with the `coeFn_toLp` ae-equality).
  have h_ae :
      ((memLp_liftIoc_haarAddCircle u hu g hg_L2_restrict).toLp
          (AddCircle.liftIoc u (-(u/2)) g) : AddCircle u → ℂ)
        =ᵐ[@AddCircle.haarAddCircle u ⟨hu⟩] AddCircle.liftIoc u (-(u/2)) g :=
    MemLp.coeFn_toLp _
  rw [fourierCoeff_congr_ae h_ae]
  -- Step 2: apply `period_u_coef_eq_fourierIntegral_at_lattice`.
  exact Sidon.TorusParseval.period_u_coef_eq_fourierIntegral_at_lattice u hu g hg_supp j

/-! ## Lp inner product = (1/u) · concrete integral

The `Lp 2 haarAddCircle` inner product of two `toLp` representatives
`F = (liftIoc g).toLp`, `H = (liftIoc h).toLp` equals
`(1/u) · ∫_{(-(u/2))..(u/2)} g · conj h`, which (since `g, h` are supported
in `Ioo (-(u/2)) (u/2)` hence in `Ioc (-(u/2)) (u/2)`) equals
`(1/u) · ∫_ℝ g · conj h`. -/

/-- The L² inner product of `(liftIoc g).toLp` and `(liftIoc h).toLp`
on `Lp 2 haarAddCircle` equals `(1/u) · ∫ x in (-(u/2))..(u/2), g x · conj(h x)`,
when both `g, h` are supported in `Ioc (-(u/2)) (u/2)`. -/
theorem inner_toLp_eq_intervalIntegral
    (u : ℝ) (hu : 0 < u)
    (g h : ℝ → ℂ)
    (hg_L2_restrict : MemLp g 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2))))
    (hh_L2_restrict : MemLp h 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2)))) :
    haveI : Fact (0 < u) := ⟨hu⟩
    @inner ℂ _ _
      ((memLp_liftIoc_haarAddCircle u hu h hh_L2_restrict).toLp
        (AddCircle.liftIoc u (-(u/2)) h))
      ((memLp_liftIoc_haarAddCircle u hu g hg_L2_restrict).toLp
        (AddCircle.liftIoc u (-(u/2)) g))
      = (u⁻¹ : ℂ) * ∫ x in (-(u/2))..(u/2), g x * starRingEnd ℂ (h x) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- Build the `toLp` representatives once for clarity.
  set F := (memLp_liftIoc_haarAddCircle u hu g hg_L2_restrict).toLp
    (AddCircle.liftIoc u (-(u/2)) g) with hF_def
  set H := (memLp_liftIoc_haarAddCircle u hu h hh_L2_restrict).toLp
    (AddCircle.liftIoc u (-(u/2)) h) with hH_def
  -- Step 1: unfold the L² inner product via `L2.inner_def`:
  --   `⟪H, F⟫ = ∫ a : AddCircle u, ⟪H a, F a⟫_ℂ ∂haarAddCircle`.
  -- Now `⟪H a, F a⟫_ℂ = F a · conj(H a)` (RCLike.inner_apply).
  -- Step 2: identify the integral pointwise-ae with `∫ liftIoc g · conj (liftIoc h)`.
  have h_F_ae : (F : AddCircle u → ℂ) =ᵐ[@AddCircle.haarAddCircle u ⟨hu⟩]
                  AddCircle.liftIoc u (-(u/2)) g :=
    MemLp.coeFn_toLp _
  have h_H_ae : (H : AddCircle u → ℂ) =ᵐ[@AddCircle.haarAddCircle u ⟨hu⟩]
                  AddCircle.liftIoc u (-(u/2)) h :=
    MemLp.coeFn_toLp _
  -- Express ⟪H, F⟫ as an integral via L2.inner_def, then unfold the inner.
  have h_step1 :
      @inner ℂ _ _ H F
        = ∫ a : AddCircle u, (F : AddCircle u → ℂ) a *
                              starRingEnd ℂ ((H : AddCircle u → ℂ) a)
                ∂(@AddCircle.haarAddCircle u ⟨hu⟩) := by
    rw [L2.inner_def]
    refine integral_congr_ae ?_
    refine Filter.Eventually.of_forall (fun a => ?_)
    exact RCLike.inner_apply _ _
  rw [h_step1]
  -- Step 3: replace `F a` by `liftIoc g a` and `H a` by `liftIoc h a` ae.
  have h_step2 :
      (∫ a : AddCircle u, (F : AddCircle u → ℂ) a *
                          starRingEnd ℂ ((H : AddCircle u → ℂ) a)
                ∂(@AddCircle.haarAddCircle u ⟨hu⟩))
        = ∫ a : AddCircle u,
              AddCircle.liftIoc u (-(u/2)) g a *
              starRingEnd ℂ (AddCircle.liftIoc u (-(u/2)) h a)
              ∂(@AddCircle.haarAddCircle u ⟨hu⟩) := by
    refine integral_congr_ae ?_
    filter_upwards [h_F_ae, h_H_ae] with a ha_F ha_H
    rw [ha_F, ha_H]
  rw [h_step2]
  -- Step 4: `liftIoc g · conj (liftIoc h)` is pointwise equal to `liftIoc (g · conj h)`.
  have h_step3 :
      (fun a : AddCircle u =>
        AddCircle.liftIoc u (-(u/2)) g a * starRingEnd ℂ (AddCircle.liftIoc u (-(u/2)) h a))
      = (fun a : AddCircle u =>
        AddCircle.liftIoc u (-(u/2)) (fun x => g x * starRingEnd ℂ (h x)) a) := by
    funext a
    simp only [AddCircle.liftIoc, Function.comp_apply, Set.restrict_apply]
  -- Rewrite the integrand using h_step3.
  rw [show (fun a : AddCircle u =>
              AddCircle.liftIoc u (-(u/2)) g a *
              starRingEnd ℂ (AddCircle.liftIoc u (-(u/2)) h a))
          = (fun a : AddCircle u =>
              AddCircle.liftIoc u (-(u/2)) (fun x => g x * starRingEnd ℂ (h x)) a)
    from h_step3]
  -- Step 5: haarAddCircle = (1/u) · volume; pass to the volume integral.
  rw [AddCircle.integral_haarAddCircle]
  -- Step 6: `∫ AddCircle u, liftIoc f ∂volume = ∫_{-(u/2)..u/2} f dx`.
  have h_lift_to_interval :
      (∫ t : AddCircle u, AddCircle.liftIoc u (-(u/2))
                  (fun x => g x * starRingEnd ℂ (h x)) t)
        = ∫ a in (-(u/2))..(-(u/2) + u), (fun x => g x * starRingEnd ℂ (h x)) a :=
    @AddCircle.integral_liftIoc_eq_intervalIntegral u ⟨hu⟩ ℂ _ _ (-(u/2))
      (fun x => g x * starRingEnd ℂ (h x))
  rw [h_lift_to_interval]
  have h_eq : (-(u/2) + u : ℝ) = u/2 := by ring
  rw [h_eq]
  -- Final: convert `u⁻¹ • (∫ ...)` (real smul on complex) to `(u⁻¹ : ℂ) * (∫ ...)`.
  show ((u : ℝ)⁻¹ : ℝ) • _ = (u⁻¹ : ℂ) * _
  rw [Complex.real_smul]
  push_cast
  ring

/-! ## Concrete bilinear Parseval at lattice points

The bilinear Parseval identity in concrete-integral form. -/

/-- **Bilinear period-`u` Parseval (concrete integral form)**:
for `g, h : ℝ → ℂ` integrable, square-integrable, and compactly supported
in `Ioo (-(u/2)) (u/2)`:

  `u · ∫_ℝ g(x) · conj(h(x)) dx = ∑' j : ℤ, 𝓕g(j/u) · conj(𝓕h(j/u))`.

(Equivalently `∫_ℝ g · conj h = (1/u) · ∑' j, 𝓕g(j/u) · conj(𝓕h(j/u))`,
see `bilinear_parseval_period_u_inv`.)

Compare to the diagonal version `plancherel_at_lattice_period_u`
(`∑' j, ‖𝓕f(j/u)‖² = u · ∫ ‖f‖²`).  The bilinear version specialised to
`g = h` gives `u · ∫ |g|² = ∑' j, |𝓕g(j/u)|²`, matching the diagonal. -/
theorem bilinear_parseval_period_u
    (u : ℝ) (hu : 0 < u)
    (g h : ℝ → ℂ)
    (hg_int : Integrable g volume) (hh_int : Integrable h volume)
    (hg_supp : Function.support g ⊆ Set.Ioo (-(u/2)) (u/2))
    (hh_supp : Function.support h ⊆ Set.Ioo (-(u/2)) (u/2))
    (hg_L2 : MemLp g 2 volume) (hh_L2 : MemLp h 2 volume) :
    u * ∫ x, g x * starRingEnd ℂ (h x) ∂volume
      = ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                  * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ)) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- Trivial preliminaries.
  have hu_ne : (u : ℝ) ≠ 0 := ne_of_gt hu
  have hu_ne_c : (u : ℂ) ≠ 0 := by exact_mod_cast hu_ne
  -- Support inclusion `Ioo ⊆ Ioc`.
  have hg_supp_ioc : Function.support g ⊆ Set.Ioc (-(u/2)) (u/2) :=
    hg_supp.trans Set.Ioo_subset_Ioc_self
  have hh_supp_ioc : Function.support h ⊆ Set.Ioc (-(u/2)) (u/2) :=
    hh_supp.trans Set.Ioo_subset_Ioc_self
  -- L² on the restriction.
  have hg_L2_restrict : MemLp g 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2))) :=
    memLp_restrict_of_memLp_volume u hu g hg_L2
  have hh_L2_restrict : MemLp h 2 (volume.restrict (Set.Ioc (-(u/2)) (u/2))) :=
    memLp_restrict_of_memLp_volume u hu h hh_L2
  -- Build the `Lp 2 haarAddCircle` representatives.
  set F := (memLp_liftIoc_haarAddCircle u hu g hg_L2_restrict).toLp
    (AddCircle.liftIoc u (-(u/2)) g) with hF_def
  set H := (memLp_liftIoc_haarAddCircle u hu h hh_L2_restrict).toLp
    (AddCircle.liftIoc u (-(u/2)) h) with hH_def
  -- Apply `bilinear_parseval_addCircle_Lp_tsum`.
  have h_lp_tsum :
      ∑' j : ℤ, fourierCoeff (F : AddCircle u → ℂ) j
                * starRingEnd ℂ (fourierCoeff (H : AddCircle u → ℂ) j)
      = @inner ℂ _ _ H F :=
    Sidon.TorusParseval.bilinear_parseval_addCircle_Lp_tsum u hu F H
  -- Replace `fourierCoeff F j` with `(1/u) · 𝓕g(j/u)` and analogously for `H`.
  have h_coef_g : ∀ j : ℤ, fourierCoeff (F : AddCircle u → ℂ) j
                    = (1 / u : ℂ) * Real.fourierIntegral g (j / u : ℝ) := by
    intro j
    rw [hF_def]
    exact fourierCoeff_toLp_eq_period_u u hu g hg_supp_ioc hg_L2_restrict j
  have h_coef_h : ∀ j : ℤ, fourierCoeff (H : AddCircle u → ℂ) j
                    = (1 / u : ℂ) * Real.fourierIntegral h (j / u : ℝ) := by
    intro j
    rw [hH_def]
    exact fourierCoeff_toLp_eq_period_u u hu h hh_supp_ioc hh_L2_restrict j
  -- Substitute into the tsum.
  have h_tsum_subst :
      ∑' j : ℤ, fourierCoeff (F : AddCircle u → ℂ) j
                * starRingEnd ℂ (fourierCoeff (H : AddCircle u → ℂ) j)
      = ∑' j : ℤ, (1 / u : ℂ) * Real.fourierIntegral g (j / u : ℝ)
                  * starRingEnd ℂ ((1 / u : ℂ) * Real.fourierIntegral h (j / u : ℝ)) := by
    refine tsum_congr (fun j => ?_)
    rw [h_coef_g j, h_coef_h j]
  -- Simplify each term to extract `(1/u²)` as a constant multiplier.
  have h_term_simplify : ∀ j : ℤ,
      ((1 / u : ℂ) * Real.fourierIntegral g (j / u : ℝ))
        * starRingEnd ℂ ((1 / u : ℂ) * Real.fourierIntegral h (j / u : ℝ))
      = ((1 / u^2 : ℝ) : ℂ)
          * (Real.fourierIntegral g (j / u : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ))) := by
    intro j
    rw [map_mul]
    rw [show starRingEnd ℂ (1 / u : ℂ) = (1 / u : ℂ) by
        have h_cast : ((1 / u : ℂ)) = ((1 / u : ℝ) : ℂ) := by push_cast; rfl
        rw [h_cast, Complex.conj_ofReal]]
    have h_cast2 : (((1 / u^2 : ℝ)) : ℂ) = (1 / u : ℂ) * (1 / u : ℂ) := by
      push_cast
      field_simp
    rw [h_cast2]
    ring
  have h_tsum_factor :
      ∑' j : ℤ, (1 / u : ℂ) * Real.fourierIntegral g (j / u : ℝ)
                  * starRingEnd ℂ ((1 / u : ℂ) * Real.fourierIntegral h (j / u : ℝ))
      = ((1 / u^2 : ℝ) : ℂ)
          * ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                      * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ)) := by
    rw [← tsum_mul_left]
    refine tsum_congr (fun j => ?_)
    exact h_term_simplify j
  -- Combine: tsum-of-coefs = (1/u²) · target_tsum = ⟪H, F⟫.
  have h_eq1 :
      ((1 / u^2 : ℝ) : ℂ)
          * ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                      * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ))
      = @inner ℂ _ _ H F := by
    rw [← h_tsum_factor, ← h_tsum_subst, h_lp_tsum]
  -- ⟪H, F⟫ = (1/u) · ∫ g · conj h over interval = (1/u) · ∫ g · conj h over ℝ.
  have h_inner :
      @inner ℂ _ _ H F
      = (u⁻¹ : ℂ) * ∫ x in (-(u/2))..(u/2), g x * starRingEnd ℂ (h x) := by
    rw [hF_def, hH_def]
    exact inner_toLp_eq_intervalIntegral u hu g h hg_L2_restrict hh_L2_restrict
  -- Replace the interval integral by the full real integral (support of `g · conj h`
  -- is in `Ioo (-u/2) (u/2)` ⊆ `Ioc (-u/2) (u/2)`).
  have h_prod_supp : Function.support (fun x => g x * starRingEnd ℂ (h x))
                      ⊆ Set.Ioc (-(u/2)) (u/2) := by
    intro x hx
    -- `g x * conj (h x) ≠ 0` requires `g x ≠ 0`, hence `x ∈ supp g ⊆ Ioo (-u/2) (u/2)`.
    have hx_prod_ne : g x * starRingEnd ℂ (h x) ≠ 0 := hx
    have hg_ne : g x ≠ 0 := by
      intro h_g_zero
      apply hx_prod_ne
      rw [h_g_zero, zero_mul]
    have hx_in_g : x ∈ Function.support g := hg_ne
    exact hg_supp_ioc hx_in_g
  have h_interval_eq_full :
      (∫ x in (-(u/2))..(u/2), g x * starRingEnd ℂ (h x))
      = ∫ x, g x * starRingEnd ℂ (h x) ∂volume := by
    rw [intervalIntegral.integral_of_le (by linarith : -(u/2) ≤ u/2)]
    -- Now goal: `∫ x in Ioc (-(u/2)) (u/2), ... ∂volume = ∫ x, ... ∂volume`.
    refine MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ?_
    intro x hx
    -- For x ∉ Ioc (-u/2) (u/2): show `g x * conj (h x) = 0`.
    -- By contraposition: if `g x * conj (h x) ≠ 0`, then `x ∈ supp ⊆ Ioc`.
    by_contra h_ne
    exact hx (h_prod_supp h_ne)
  -- Combine all the rewrites.
  rw [h_interval_eq_full] at h_inner
  -- We have:
  --   h_eq1   : (1/u²) · target = ⟪H, F⟫
  --   h_inner : ⟪H, F⟫ = (1/u) · ∫ g · conj h
  -- Combining: (1/u²) · target = (1/u) · ∫
  -- Equivalently: target = u · ∫.
  have h_combine : ((1 / u^2 : ℝ) : ℂ)
        * ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                    * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ))
      = (u⁻¹ : ℂ) * ∫ x, g x * starRingEnd ℂ (h x) ∂volume := by
    rw [h_eq1, h_inner]
  -- Multiply both sides by `u²` to solve for the target tsum.
  have h_tsum_eq :
      ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ))
      = (u : ℂ) * ∫ x, g x * starRingEnd ℂ (h x) ∂volume := by
    have h_step : ((u^2 : ℝ) : ℂ) *
        (((1 / u^2 : ℝ) : ℂ) *
          ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                    * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ)))
        = ((u^2 : ℝ) : ℂ) *
          ((u⁻¹ : ℂ) * ∫ x, g x * starRingEnd ℂ (h x) ∂volume) := by
      rw [h_combine]
    -- Simplify both sides: LHS = tsum, RHS = u · ∫.
    have h_LHS : ((u^2 : ℝ) : ℂ) *
        (((1 / u^2 : ℝ) : ℂ) *
          ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                    * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ)))
        = ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                    * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ)) := by
      rw [← mul_assoc]
      have h1 : ((u^2 : ℝ) : ℂ) * ((1 / u^2 : ℝ) : ℂ) = 1 := by
        push_cast
        field_simp
      rw [h1, one_mul]
    have h_RHS : ((u^2 : ℝ) : ℂ) *
        ((u⁻¹ : ℂ) * ∫ x, g x * starRingEnd ℂ (h x) ∂volume)
        = (u : ℂ) * ∫ x, g x * starRingEnd ℂ (h x) ∂volume := by
      rw [← mul_assoc]
      have h2 : ((u^2 : ℝ) : ℂ) * (u⁻¹ : ℂ) = (u : ℂ) := by
        push_cast
        field_simp
      rw [h2]
    rw [← h_LHS, h_step, h_RHS]
  -- Final: `u · ∫ = tsum`.
  rw [h_tsum_eq]

/-- **Bilinear period-`u` Parseval (concrete integral form, divided)**:
`∫_ℝ g · conj h = (1/u) · ∑' j : ℤ, 𝓕g(j/u) · conj(𝓕h(j/u))`.

This is the equivalent form of `bilinear_parseval_period_u`. -/
theorem bilinear_parseval_period_u_inv
    (u : ℝ) (hu : 0 < u)
    (g h : ℝ → ℂ)
    (hg_int : Integrable g volume) (hh_int : Integrable h volume)
    (hg_supp : Function.support g ⊆ Set.Ioo (-(u/2)) (u/2))
    (hh_supp : Function.support h ⊆ Set.Ioo (-(u/2)) (u/2))
    (hg_L2 : MemLp g 2 volume) (hh_L2 : MemLp h 2 volume) :
    ∫ x, g x * starRingEnd ℂ (h x) ∂volume
      = (1 / u : ℂ) * ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                                * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ)) := by
  have h_main := bilinear_parseval_period_u u hu g h hg_int hh_int
    hg_supp hh_supp hg_L2 hh_L2
  -- `u · ∫ = tsum` ⟹ `∫ = (1/u) · tsum`.
  have hu_ne : (u : ℂ) ≠ 0 := by exact_mod_cast (ne_of_gt hu)
  have : (1 / u : ℂ) * ((u : ℂ) * ∫ x, g x * starRingEnd ℂ (h x) ∂volume)
        = (1 / u : ℂ) * ∑' j : ℤ, Real.fourierIntegral g (j / u : ℝ)
                                  * starRingEnd ℂ (Real.fourierIntegral h (j / u : ℝ)) := by
    rw [h_main]
  rw [← this]
  -- `(1/u) · (u · ∫) = (1/u · u) · ∫ = ∫`.
  rw [← mul_assoc]
  rw [show (1 / u : ℂ) * (u : ℂ) = 1 by field_simp]
  rw [one_mul]

end -- noncomputable section

end Sidon.BilinearParseval
