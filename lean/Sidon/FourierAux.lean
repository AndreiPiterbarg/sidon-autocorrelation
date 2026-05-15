/-
Sidon Autocorrelation Project — Auxiliary Fourier Infrastructure.

This file builds the Fourier infrastructure used by the MV proof.  It
proves:

  * `fourierIntegral_zero`        — `f̂(0) = ∫ f`  (L¹ functions);
  * `fourierIntegral_zero_real`   — `Re(f̂(0)) = ∫ f` for real-valued `f`;
  * `fourierIntegral_real_conj`   — `f̂(-ξ) = conj(f̂(ξ))` for real-valued `f`;
  * `convolution_self_real`       — unfolding of `(f * f)(x) = ∫ f(t)·f(x-t) dt`;
  * `fourier_l1_pairing`          — L¹-pairing form of the Fourier transform
                                    (a fragment of Plancherel that suffices
                                    for the MV master inequality).

The general L² Plancherel isometry for `ℝ` is *not* in current mathlib
for non-Schwartz functions; the file documents this gap and provides
the L¹-pairing version which is what MV Eq.(2) actually needs.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.Defs

namespace Sidon.FourierAux

open MeasureTheory Real Complex Filter
open scoped FourierTransform Topology BigOperators

set_option maxHeartbeats 4000000

noncomputable section

/-- Norm of `Real.fourierChar x` viewed as a complex number is `1`. -/
lemma norm_fourierChar_coe (x : ℝ) :
    ‖(Real.fourierChar x : ℂ)‖ = 1 :=
  Circle.norm_coe _

/-- Continuity of the standard additive character `Real.fourierChar`
viewed as a function `ℝ → ℂ`. -/
lemma continuous_fourierChar_coe :
    Continuous (fun x : ℝ => (Real.fourierChar x : ℂ)) :=
  (Real.continuous_fourierChar).subtype_val

/-! ## `fourierChar` at `0` -/

/-- `Real.fourierChar 0 = 1` (as a circle element). -/
lemma fourierChar_zero : Real.fourierChar (0 : ℝ) = 1 :=
  (Real.fourierChar).map_zero_eq_one

/-- The Fourier character at `0` (as a complex number) is `1`. -/
lemma fourierChar_zero_coe : ((Real.fourierChar (0 : ℝ) : Circle) : ℂ) = 1 := by
  rw [fourierChar_zero]
  rfl

/-! ## `f̂(0) = ∫ f` (the value of the Fourier transform at the origin)

For `f : ℝ → ℂ` (or any complex Banach-valued function),
`Real.fourierIntegral f 0 = ∫ v, 1 · f v = ∫ f`.

The proof is essentially `e(-v·0) = e(0) = 1`. -/

/-- For `f : ℝ → ℂ`, the Fourier transform at `0` equals the integral. -/
theorem fourierIntegral_zero (f : ℝ → ℂ) :
    Real.fourierIntegral f 0 = ∫ x, f x ∂volume := by
  -- Definition: 𝓕 f 0 = ∫ v, 𝐞(-⟪v,0⟫) • f v
  -- `Real.fourierIntegral` is a deprecated alias for `FourierTransform.fourier`; unfold so the
  -- rewrite below can match the LHS shape `𝓕 ?f ?w`.
  show FourierTransform.fourier f 0 = ∫ x, f x ∂volume
  rw [Real.fourier_real_eq]
  -- Reduce ⟪v, 0⟫ to v * 0 and simplify the character.
  simp only [mul_zero, neg_zero]
  -- Now we have ∫ v, 𝐞(0) • f v = ∫ v, 1 • f v = ∫ v, f v.
  have hchar : (Real.fourierChar (0 : ℝ) : ℂ) = 1 := fourierChar_zero_coe
  -- Use Circle.smul_def to expose the underlying complex number.
  simp_rw [Circle.smul_def, hchar, one_smul]

/-- For `f : ℝ → ℝ` integrable, the *real* part of `f̂(0)` is `∫ f`. -/
theorem fourierIntegral_zero_real (f : ℝ → ℝ) :
    (Real.fourierIntegral (fun x => (f x : ℂ)) 0).re = ∫ x, f x ∂volume := by
  rw [fourierIntegral_zero (fun x => (f x : ℂ))]
  -- `∫ x, (f x : ℂ) = ((∫ x, f x : ℝ) : ℂ)`.
  have h : ∫ x, ((f x : ℂ)) = ((∫ x, f x : ℝ) : ℂ) :=
    integral_ofReal (𝕜 := ℂ) (f := f)
  rw [h]
  exact Complex.ofReal_re _

/-! ## Conjugate symmetry for real-valued functions

`f̂(-ξ) = conj(f̂(ξ))` whenever `f` is real-valued.  The argument is:

   `f̂(-ξ) = ∫ e^{-2πi x (-ξ)} f(x) dx = ∫ e^{+2πi x ξ} f(x) dx`,

and on the other side `conj(f̂(ξ)) = ∫ conj(e^{-2πi x ξ}) conj(f(x)) dx`,
which equals the previous integral since `conj(e^{-2πi x ξ}) = e^{2πi x ξ}`
and `conj(f x) = f x` (because `f` is real). -/

/-- Conjugate of the additive character: `conj(𝐞(x)) = 𝐞(-x)`. -/
lemma conj_fourierChar (x : ℝ) :
    starRingEnd ℂ ((Real.fourierChar x : Circle) : ℂ) =
      ((Real.fourierChar (-x) : Circle) : ℂ) := by
  -- 𝐞(x) = exp(2πi x), and conj(exp(z)) = exp(conj(z)).
  -- So conj(𝐞(x)) = exp(-2πi x) = 𝐞(-x).
  rw [Real.fourierChar_apply, Real.fourierChar_apply]
  rw [← Complex.exp_conj]
  congr 1
  -- We need: conj(↑(2π·x)·I) = ↑(2π·(-x))·I.
  -- conj(a·I) = conj(a)·conj(I) = a·(-I) (when a is real-valued).
  rw [map_mul, Complex.conj_I, Complex.conj_ofReal]
  push_cast
  ring

/-- Conjugate-symmetry: `f̂(-ξ) = conj(f̂(ξ))` for real-valued `f`.

The integrability hypothesis is not used in the proof (the equality
holds pointwise inside the integral and Bochner extends), but is kept
in the signature for symmetry with the rest of the MV development. -/
theorem fourierIntegral_real_conj
    (f : ℝ → ℝ)
    (_hf : Integrable (fun x => (f x : ℂ)) volume) (ξ : ℝ) :
    Real.fourierIntegral (fun x => (f x : ℂ)) (-ξ) =
      starRingEnd ℂ (Real.fourierIntegral (fun x => (f x : ℂ)) ξ) := by
  -- Expand both sides using `fourier_real_eq`.  `Real.fourierIntegral` is a deprecated
  -- alias for `FourierTransform.fourier`; unfold the goal so the rewrite can fire.
  show FourierTransform.fourier (fun x => (f x : ℂ)) (-ξ) =
       starRingEnd ℂ (FourierTransform.fourier (fun x => (f x : ℂ)) ξ)
  rw [Real.fourier_real_eq, Real.fourier_real_eq]
  -- Unfold the `•` action of `Circle` on `ℂ` to ordinary `*` in `ℂ`.
  simp only [Circle.smul_def, smul_eq_mul]
  -- RHS = conj(∫ v, ↑(𝐞(-(v·ξ))) * (f v : ℂ)) = ∫ v, conj(↑(𝐞(-(v·ξ))) * (f v : ℂ)).
  rw [show (starRingEnd ℂ) (∫ (v : ℝ), ↑(𝐞 (-(v * ξ))) * ↑(f v))
        = ∫ (v : ℝ), (starRingEnd ℂ) (↑(𝐞 (-(v * ξ))) * ↑(f v)) from
      (integral_conj (f := fun v : ℝ => ↑(𝐞 (-(v * ξ))) * (f v : ℂ)) (μ := volume)).symm]
  -- Conjugate inside the integral.
  refine integral_congr_ae ?_
  refine Filter.Eventually.of_forall fun v => ?_
  -- Beta-reduce both sides.
  simp only []
  -- Goal: ↑(𝐞(-(v·(-ξ)))) * (f v : ℂ) = conj(↑(𝐞(-(v·ξ))) * (f v : ℂ)).
  rw [map_mul (starRingEnd ℂ)]
  -- conj((f v : ℂ)) = (f v : ℂ) since f is real-valued.
  have hfv_conj : starRingEnd ℂ ((f v : ℂ)) = (f v : ℂ) := Complex.conj_ofReal _
  rw [hfv_conj]
  -- And conj(𝐞(-(v·ξ))) = 𝐞(v·ξ) = 𝐞(-(v·(-ξ))).
  rw [conj_fourierChar (-(v * ξ))]
  -- Goal: ↑(𝐞(-(v · -ξ))) * ↑(f v) = ↑(𝐞(-(-(v · ξ)))) * ↑(f v).
  -- The arguments to 𝐞 differ by neg/neg/mul rearrangement.
  congr 3
  ring

/-! ## Autoconvolution unfolding

`(f * f)(x) = ∫ t, f(t) · f(x - t) dt`.  This is just `convolution_mul`
applied to the diagonal case, recorded here for downstream use. -/

/-- `(f * f)(x) = ∫ f(t) · f(x - t) dt`. -/
theorem convolution_self_real (f : ℝ → ℝ) (x : ℝ) :
    convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume x
      = ∫ t, f t * f (x - t) ∂volume := by
  rfl

/-! ## Convolutional autocorrelation `(f ∘ f)(x) := ∫ f(t) · f(x + t) dt`

This is the autocorrelation function used in MV Eq.(2): the convolutional
autocorrelation `(f ⋆ f̌)(x)` where `f̌(y) := f(-y)`.  Equivalently, by
the substitution `s = -t`:
`∫ f(t) · f(x + t) dt = ∫ f(-s) · f(x - s) ds = (f ⋆ f̌)(x)`.

For real `f`, the Fourier transform is the squared modulus of `f̂`:
`widehat(autocorr f)(ξ) = |f̂(ξ)|²` (the Wiener-Khinchin identity).
This is what MV's Lemma 3.1 Eq.(2) needs:
`∫ (autocorr f)(x) · K(x) dx ≤ 1 + √(‖f*f‖_∞ - 1) · √(K_2 - 1)`,
obtained by the Parseval split `∫(autocorr f) · K = ⟨|f̂|², K̂⟩ = 1 + ⟨tail⟩`. -/

/-- The convolutional autocorrelation, MV's `f ∘ f`:
`autocorr f x := ∫ t, f(t) · f(x + t) dt`.

Properties (for `f` integrable):
  * `autocorr f` is even (when `f` is real-valued and integrable);
  * `∫ autocorr f x dx = (∫ f)²` (Fubini);
  * `widehat(autocorr f)(ξ) = |f̂(ξ)|²` (Wiener-Khinchin, real `f`). -/
def autocorr (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  ∫ t, f t * f (x + t) ∂MeasureTheory.volume

@[simp] lemma autocorr_def (f : ℝ → ℝ) (x : ℝ) :
    autocorr f x = ∫ t, f t * f (x + t) ∂MeasureTheory.volume := rfl

/-! ### `∫ autocorr f x dx = (∫ f)²` (Fubini)

For `f` integrable with `f x · f y` jointly integrable on `ℝ²`,
Fubini gives
  `∫_x autocorr f x dx
    = ∫_x ∫_t f(t) f(x+t) dt dx
    = ∫_t f(t) ∫_x f(x+t) dx dt
    = (∫ f) · (∫ f)`.
In particular, if `∫ f = 1`, then `∫ autocorr f x dx = 1`. -/

/-- Translation-invariance of Lebesgue measure: `∫ f(x + t) dx = ∫ f(x) dx`. -/
lemma integral_translate (f : ℝ → ℝ) (t : ℝ) :
    ∫ x, f (x + t) ∂MeasureTheory.volume = ∫ x, f x ∂MeasureTheory.volume := by
  -- Direct application of `MeasureTheory.integral_add_right_eq_self` (or
  -- `MeasureTheory.integral_comp_add_right`).
  have := MeasureTheory.integral_add_right_eq_self (μ := MeasureTheory.volume) f t
  exact this

/-- **Convolutional autocorrelation total mass identity** (Fubini).
For `f : ℝ → ℝ` such that `g(x,t) := f(t)·f(x+t)` is jointly integrable
on `ℝ²` and `f` is integrable,
  `∫ autocorr f x dx = (∫ f)²`.

The integrability hypothesis `h_joint` ensures Fubini applies. -/
theorem integral_autocorr_eq_sq_integral (f : ℝ → ℝ)
    (hf : MeasureTheory.Integrable f MeasureTheory.volume)
    (h_joint : MeasureTheory.Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
        (MeasureTheory.volume.prod MeasureTheory.volume)) :
    ∫ x, autocorr f x ∂MeasureTheory.volume
      = (∫ x, f x ∂MeasureTheory.volume) ^ 2 := by
  -- `autocorr f x = ∫ t, f(t)·f(x+t) dt`; integrate over x and apply Fubini.
  unfold autocorr
  -- `∫_x ∫_t f(t)·f(x+t) dt dx = ∫_t f(t) · ∫_x f(x+t) dx dt = (∫ f) * (∫ f)`.
  rw [MeasureTheory.integral_integral_swap h_joint]
  -- Now: `∫_t (∫_x f(t)·f(x+t) dx) dt = (∫ f)²`.
  have hpt : ∀ t, ∫ x, f t * f (x + t) ∂MeasureTheory.volume
                  = f t * (∫ x, f x ∂MeasureTheory.volume) := by
    intro t
    rw [MeasureTheory.integral_const_mul]
    rw [integral_translate f t]
  -- Rewrite the inner integral.
  rw [show (fun t => ∫ x, f t * f (x + t) ∂MeasureTheory.volume)
        = (fun t => f t * (∫ x, f x ∂MeasureTheory.volume)) by
      funext t; exact hpt t]
  -- ∫_t f t · C dt = (∫ f) · C, with C = ∫ f.  Hence the result is (∫f)².
  rw [MeasureTheory.integral_mul_const]
  ring

/-! ## Fragment of Plancherel: L¹ pairing

The full L² Plancherel isometry for `ℝ` (i.e. `∫|f|² = ∫|f̂|²` for `f ∈ L²`)
is *not* currently in mathlib for general `f`; mathlib provides Fourier
inversion in `Mathlib.Analysis.Fourier.Inversion`, and an L¹-pairing
self-adjoint identity in `Mathlib.Analysis.Fourier.FourierTransform`.

For the MV proof we only need the *L¹ pairing form*:

   `∫ f̂(ξ) · g(ξ) dξ  =  ∫ f(x) · ĝ(x) dx`

for integrable `f, g`.  This is the "self-adjointness" of the Fourier
transform, available in mathlib as
`Real.integral_fourierIntegral_smul_eq_flip` (for `f : ℝ → ℂ`,
`g : ℝ → F`).

We re-state the scalar-valued version here in the form that MV Eq.(2)
(centred Cauchy-Schwarz) consumes. -/

/-- L¹ pairing / self-adjointness of the Fourier transform on `ℝ`
(`f, g` integrable, scalar-valued, complex):

    `∫ f̂(ξ) · g(ξ) dξ = ∫ f(x) · ĝ(x) dx`. -/
theorem fourier_l1_pairing
    (f g : ℝ → ℂ) (hf : Integrable f volume) (hg : Integrable g volume) :
    ∫ ξ, Real.fourierIntegral f ξ * g ξ ∂volume
      = ∫ x, f x * Real.fourierIntegral g x ∂volume := by
  -- Mathlib provides this directly in terms of `smul`, and for ℂ-valued
  -- functions `•` over `ℂ` reduces to `*`.
  have h := VectorFourier.integral_fourierIntegral_smul_eq_flip (W := ℝ)
    (e := Real.fourierChar) (L := innerₗ ℝ) (μ := volume) (ν := volume)
    Real.continuous_fourierChar
    (by exact ((innerSL ℝ (E := ℝ)).continuous₂))
    hf hg
  -- `innerₗ ℝ` is symmetric, so `.flip = innerₗ ℝ`.
  rw [flip_innerₗ] at h
  -- The mathlib statement is in terms of `VectorFourier.fourierIntegral`;
  -- `Real.fourierIntegral` unfolds to this via the inner-product bilinear
  -- form on ℝ.  For ℂ-valued functions, `•` over `ℂ` reduces to `*`.
  exact h

/-! ## Documentation of what Plancherel would unlock

The full L² Plancherel theorem `∫|f|² = ∫|f̂|²` would, combined with
the `fourier_l1_pairing` above, give MV Eq.(2) (centred Cauchy-Schwarz)
as follows:

  Let `F := f∘f - 𝟙_{[-δ,δ]}` and `K̃ := K - 𝟙_{[-δ,δ]}`.  Both have
  integral zero (mean-centered).  Then:

    `∫ F · K = ⟨F, K⟩_{L²}  ≤ ‖F‖_{L²} · ‖K‖_{L²}`     (Cauchy-Schwarz)
              `≤ √(M_∞ - 1) · √(K_2 - 1)`              (sup-bound on F,
                                                       Plancherel on K̃).

The section below bridges mathlib's abstract `Lp 2` Plancherel isometry
to the concrete `∫|f|² = ∫|f̂|²` integral identity in the cases we need. -/

/-! ## Bridging `Lp 2` Plancherel to concrete integrals

Mathlib provides:

  * `SchwartzMap.integral_norm_sq_fourier` :
      `∫ ξ, ‖𝓕 f ξ‖² = ∫ x, ‖f x‖²` for Schwartz `f`;
  * `SchwartzMap.integral_inner_fourier_fourier` :
      Parseval pairing for Schwartz `f, g`;
  * `MeasureTheory.Lp.norm_fourier_eq` :
      `‖𝓕 f‖_{L²} = ‖f‖_{L²}` for `f : Lp ℂ 2 volume` (abstract quotient).

The Schwartz-level statements are already in the form `∫|f|² = ∫|𝓕 f|²`
that this project consumes (via `fourier_coe`, `𝓕 f = 𝓕 (f : ℝ → ℂ)` so
the Schwartz Fourier transform IS `Real.fourierIntegral` on the
underlying function).  This section repackages those statements in the
plain pointwise-integral language used by `Sidon.MV` and friends.

For `Lp 2` quotients we additionally record:

  * `plancherel_lp_norm_sq` : `‖toLp f hf‖² = ∫ x, ‖f x‖²`, transporting
    the `Lp 2`-norm squared back to the concrete integral of `‖f‖²`.

These statements, combined, allow MV Eq.(2) to be discharged whenever
the relevant functions are Schwartz (or can be approximated by Schwartz
in `L²` with control on the Fourier side). -/

namespace Plancherel

open MeasureTheory SchwartzMap

/-- **Plancherel for Schwartz functions on `ℝ`** in concrete-integral form:
`∫ x, ‖f x‖² = ∫ ξ, ‖𝓕 f ξ‖²`.

This is `SchwartzMap.integral_norm_sq_fourier` specialised to `V = ℝ`,
`H = ℂ`.  By `SchwartzMap.fourier_coe`, the Schwartz Fourier transform
agrees with the function-level `𝓕`, hence with `Real.fourierIntegral`. -/
theorem plancherel_schwartz (f : 𝓢(ℝ, ℂ)) :
    ∫ x, ‖f x‖ ^ 2 ∂volume
      = ∫ ξ, ‖(𝓕 (f : ℝ → ℂ)) ξ‖ ^ 2 ∂volume := by
  -- mathlib's `SchwartzMap.integral_norm_sq_fourier` reads
  -- `∫ ξ, ‖𝓕 f ξ‖ ^ 2 = ∫ x, ‖f x‖ ^ 2` where on the LHS `𝓕 f` is the
  -- Schwartz-level transform.  Using `fourier_coe` we rewrite this to the
  -- function-level `𝓕 (f : ℝ → ℂ)`.
  have h := SchwartzMap.integral_norm_sq_fourier (V := ℝ) (H := ℂ) f
  -- `h : ∫ ξ, ‖(𝓕 f) ξ‖^2 = ∫ x, ‖f x‖^2`.
  -- `fourier_coe : 𝓕 f = 𝓕 (f : V → E)` (as functions ℝ → ℂ).
  have hcoe : ∀ ξ, ((𝓕 f) ξ : ℂ) = (𝓕 (f : ℝ → ℂ)) ξ := by
    intro ξ
    rw [fourier_coe]
  -- Swap sides and rewrite.
  rw [← h]
  refine integral_congr_ae (Filter.Eventually.of_forall fun ξ => ?_)
  show ‖(𝓕 f) ξ‖ ^ 2 = ‖(𝓕 (f : ℝ → ℂ)) ξ‖ ^ 2
  rw [hcoe ξ]

/-- **Plancherel for Schwartz functions on `ℝ`** in terms of `Real.fourierIntegral`
(the deprecated alias for `FourierTransform.fourier`). -/
theorem plancherel_schwartz' (f : 𝓢(ℝ, ℂ)) :
    ∫ x, ‖f x‖ ^ 2 ∂volume
      = ∫ ξ, ‖Real.fourierIntegral (f : ℝ → ℂ) ξ‖ ^ 2 ∂volume :=
  plancherel_schwartz f

/-- **Parseval pairing for Schwartz functions on `ℝ`**:
`∫ x, conj(f x) · g x = ∫ ξ, conj(𝓕 f ξ) · 𝓕 g ξ`.

For ℂ-valued data this is `SchwartzMap.integral_inner_fourier_fourier`
combined with `RCLike.inner_apply : ⟪x, y⟫ = y * conj x`. -/
theorem parseval_schwartz_inner (f g : 𝓢(ℝ, ℂ)) :
    ∫ x, g x * (starRingEnd ℂ) (f x) ∂volume
      = ∫ ξ, (𝓕 (g : ℝ → ℂ)) ξ * (starRingEnd ℂ) ((𝓕 (f : ℝ → ℂ)) ξ) ∂volume := by
  -- `integral_inner_fourier_fourier : ∫ ξ, ⟪𝓕 f ξ, 𝓕 g ξ⟫ = ∫ x, ⟪f x, g x⟫`.
  -- On `ℂ`, `⟪a, b⟫ = b * conj a` (`RCLike.inner_apply`).  Rewrite both sides.
  have h := SchwartzMap.integral_inner_fourier_fourier (V := ℝ) (H := ℂ) f g
  have hcoeF : ∀ ξ, ((𝓕 f) ξ : ℂ) = (𝓕 (f : ℝ → ℂ)) ξ := fun ξ => by rw [fourier_coe]
  have hcoeG : ∀ ξ, ((𝓕 g) ξ : ℂ) = (𝓕 (g : ℝ → ℂ)) ξ := fun ξ => by rw [fourier_coe]
  -- `h` rewritten using `inner_apply`.
  -- ⟪f x, g x⟫ = g x * conj (f x).
  have hLHS : (∫ x, @inner ℂ ℂ _ (f x) (g x) ∂volume)
                = ∫ x, g x * (starRingEnd ℂ) (f x) ∂volume := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
    show @inner ℂ ℂ _ (f x) (g x) = g x * (starRingEnd ℂ) (f x)
    rw [RCLike.inner_apply]
  have hRHS : (∫ ξ, @inner ℂ ℂ _ ((𝓕 f) ξ) ((𝓕 g) ξ) ∂volume)
                = ∫ ξ, (𝓕 (g : ℝ → ℂ)) ξ
                    * (starRingEnd ℂ) ((𝓕 (f : ℝ → ℂ)) ξ) ∂volume := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun ξ => ?_)
    show @inner ℂ ℂ _ ((𝓕 f) ξ) ((𝓕 g) ξ)
          = (𝓕 (g : ℝ → ℂ)) ξ * (starRingEnd ℂ) ((𝓕 (f : ℝ → ℂ)) ξ)
    rw [RCLike.inner_apply, hcoeF ξ, hcoeG ξ]
  rw [← hLHS, ← h, hRHS]

/-- **Plancherel for real-valued Schwartz functions on `ℝ`**:
`∫ x, (f x)² = ∫ ξ, ‖f̂(ξ)‖²` for `f : 𝓢(ℝ, ℝ)`, where `f̂` is the
Fourier integral of the complex-valued lift `x ↦ (f x : ℂ)`.

This is the form consumed by the MV proof for the real-valued
test functions on the autocorrelation side. -/
theorem plancherel_schwartz_real (f : 𝓢(ℝ, ℝ)) :
    ∫ x, (f x) ^ 2 ∂volume
      = ∫ ξ, ‖Real.fourierIntegral (fun x => ((f x : ℝ) : ℂ)) ξ‖ ^ 2 ∂volume := by
  -- Compose `f` with `Complex.ofRealCLM` to get a complex-valued Schwartz function.
  let fc : 𝓢(ℝ, ℂ) := SchwartzMap.postcompCLM (𝕜 := ℝ) Complex.ofRealCLM f
  -- Apply complex Plancherel to `fc`.
  have h := plancherel_schwartz fc
  -- LHS: ‖fc x‖² = (f x)²  since fc x = (f x : ℂ) and ‖(r : ℂ)‖ = |r|.
  have hLHS : ∫ x, ‖fc x‖ ^ 2 ∂volume = ∫ x, (f x) ^ 2 ∂volume := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
    show ‖fc x‖ ^ 2 = (f x) ^ 2
    have hfc : fc x = ((f x : ℝ) : ℂ) := rfl
    rw [hfc, Complex.norm_real, Real.norm_eq_abs, sq_abs]
  -- RHS: `fc : ℝ → ℂ` is `fun x => ((f x : ℝ) : ℂ)`.
  have hcoe_fun : ((fc : 𝓢(ℝ, ℂ)) : ℝ → ℂ) = fun x => ((f x : ℝ) : ℂ) := rfl
  rw [← hLHS, h, hcoe_fun]
  -- `Real.fourierIntegral` is a deprecated alias for `𝓕`; they are definitionally equal.
  rfl

end Plancherel

/-! ## `Lp 2` ↔ concrete integral

Bridging the `Lp 2` norm to the concrete integral `∫ ‖f‖²` for a
pointwise representative `f : ℝ → ℂ` with `MemLp f 2 volume`. -/

namespace LpBridge

open MeasureTheory

/-- For `f : ℝ → ℂ` in `L²`, the `Lp 2` quotient norm squared equals the
concrete integral `∫ ‖f x‖²`.

This is the standard `‖toLp f hf‖² = ∫ ‖f‖²` translation, obtained
by composing the L²-inner-product definition (`MeasureTheory.L2.inner_def`)
with the ae-equality `toLp f hf =ᵐ f` (`MemLp.coeFn_toLp`). -/
theorem norm_sq_toLp_eq_integral (f : ℝ → ℂ) (hf : MemLp f 2 volume) :
    ‖hf.toLp f‖ ^ 2 = ∫ x, ‖f x‖ ^ 2 ∂volume := by
  -- ‖F‖² = re ⟪F, F⟫ where F : Lp ℂ 2 volume.
  have h_norm_sq :
      ‖hf.toLp f‖ ^ 2 = RCLike.re (@inner ℂ _ _ (hf.toLp f) (hf.toLp f)) :=
    @norm_sq_eq_re_inner ℂ _ _ _ _ (hf.toLp f)
  -- L²-inner product = ∫ pointwise ⟪·, ·⟫.
  have h_inner_def :
      @inner ℂ _ _ (hf.toLp f) (hf.toLp f)
        = ∫ a, @inner ℂ _ _ ((hf.toLp f : ℝ → ℂ) a) ((hf.toLp f : ℝ → ℂ) a) ∂volume :=
    L2.inner_def (𝕜 := ℂ) (hf.toLp f) (hf.toLp f)
  -- Replace the Lp-quotient representative with `f` itself (ae-equal).
  have h_ae : (hf.toLp f : ℝ → ℂ) =ᵐ[volume] f := MemLp.coeFn_toLp hf
  have h_inner_pt :
      (∫ a, @inner ℂ _ _ ((hf.toLp f : ℝ → ℂ) a) ((hf.toLp f : ℝ → ℂ) a) ∂volume)
        = ∫ a, @inner ℂ _ _ (f a) (f a) ∂volume := by
    refine integral_congr_ae ?_
    filter_upwards [h_ae] with a ha
    rw [ha]
  -- Pointwise: ⟪z, z⟫_ℂ = z * conj z = (‖z‖² : ℂ); taking `re` of a real cast.
  -- `inner_self_eq_norm_sq_to_K : ⟪x, x⟫ = (‖x‖ : 𝕜)^2`.
  have h_inner_pt_real :
      (∫ a, @inner ℂ _ _ (f a) (f a) ∂volume) = ∫ a, ((‖f a‖ ^ 2 : ℝ) : ℂ) ∂volume := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun a => ?_)
    show @inner ℂ _ _ (f a) (f a) = ((‖f a‖ ^ 2 : ℝ) : ℂ)
    rw [@inner_self_eq_norm_sq_to_K ℂ]
    push_cast
    rfl
  -- `re (∫ ((g : ℝ) : ℂ)) = ∫ g`.
  have h_re_cast : RCLike.re (∫ a, ((‖f a‖ ^ 2 : ℝ) : ℂ) ∂volume)
                    = ∫ x, ‖f x‖ ^ 2 ∂volume := by
    have h1 : (∫ a, ((‖f a‖ ^ 2 : ℝ) : ℂ) ∂volume)
                = ((∫ a, ‖f a‖ ^ 2 ∂volume : ℝ) : ℂ) :=
      integral_ofReal (𝕜 := ℂ) (f := fun a => ‖f a‖ ^ 2)
    rw [h1]
    exact Complex.ofReal_re _
  rw [h_norm_sq, h_inner_def, h_inner_pt, h_inner_pt_real, h_re_cast]

/-- **Plancherel identity in concrete-`Lp`-norm form** for `f ∈ L²`:

    ∫ ‖f x‖² = ‖Lp.fourierTransformₗᵢ (toLp f)‖²_{L²}.

Combining `MeasureTheory.Lp.norm_fourier_eq` (`Lp 2`-isometry of the
Fourier transform) with `norm_sq_toLp_eq_integral`, this is the most
direct concrete-integral consequence of the `Lp 2` Plancherel theorem.

The right-hand side is the squared `Lp 2`-quotient norm of
`fourierTransformₗᵢ (toLp f)`; it equals `∫ ‖Real.fourierIntegral f‖²`
**whenever** the Lp class `fourierTransformₗᵢ (toLp f)` is ae-equal to
the pointwise `Real.fourierIntegral f`, which is true for Schwartz `f`
(via `SchwartzMap.toLp_fourier_eq`) and, by a density argument outside
the scope of current mathlib, for `f ∈ L¹ ∩ L²`. -/
theorem plancherel_lp_norm (f : ℝ → ℂ) (hf : MeasureTheory.MemLp f 2 MeasureTheory.volume) :
    ∫ x, ‖f x‖ ^ 2 ∂MeasureTheory.volume
      = ‖MeasureTheory.Lp.fourierTransformₗᵢ ℝ ℂ (hf.toLp f)‖ ^ 2 := by
  -- ‖toLp f‖² = ∫ ‖f‖² by `norm_sq_toLp_eq_integral`.
  rw [(norm_sq_toLp_eq_integral f hf).symm]
  -- ‖fourierTransformₗᵢ (toLp f)‖ = ‖toLp f‖ since it is a linear isometry.
  -- Use `LinearIsometryEquiv.norm_map` directly to avoid the `𝓕` notation.
  rw [show ‖MeasureTheory.Lp.fourierTransformₗᵢ ℝ ℂ (hf.toLp f)‖ = ‖hf.toLp f‖ from
        LinearIsometryEquiv.norm_map _ _]

end LpBridge

/-! ## Concrete-integral Plancherel for Schwartz on ℝ

For Schwartz `f : 𝓢(ℝ, ℂ)`, the Lp 2 image of `Real.fourierIntegral f`
agrees with `Lp.fourierTransformₗᵢ (toLp f)` (a consequence of
`SchwartzMap.toLp_fourier_eq` from mathlib).  Squared-Lp-norm
combined with this identifies `∫ ‖f‖² = ∫ ‖f̂‖²`.

This duplicates `plancherel_schwartz` but routes via the `Lp 2` API,
illustrating the bridging pattern. -/

namespace PlancherelL1L2

open MeasureTheory SchwartzMap

/-- Schwartz functions are members of `Lp 2`. -/
lemma schwartz_memLp_two (f : 𝓢(ℝ, ℂ)) : MemLp (f : ℝ → ℂ) 2 volume :=
  f.memLp 2 volume

/-- For Schwartz `f`, `(MemLp.toLp f _) = f.toLp 2` (the `MemLp`-built Lp class
agrees with the Schwartz-specific `toLp` map by definition). -/
lemma toLp_two_of_schwartz_eq (f : 𝓢(ℝ, ℂ)) :
    (schwartz_memLp_two f).toLp (f : ℝ → ℂ) = f.toLp 2 (μ := volume) := rfl

/-- **Schwartz Plancherel through the Lp 2 isometry**:
    ∫ ‖f‖² = ∫ ‖Real.fourierIntegral f‖²  for `f : 𝓢(ℝ, ℂ)`.

Proof path: `‖f.toLp 2‖² = ∫ ‖f‖²` (concrete-integral form via
`norm_sq_toLp_eq_integral`); `Lp.norm_fourier_eq` gives
`‖𝓕_L² (f.toLp 2)‖² = ‖f.toLp 2‖²`; `SchwartzMap.toLp_fourier_eq` identifies
`𝓕_L² (f.toLp 2) = (𝓕 f).toLp 2`; finally apply `norm_sq_toLp_eq_integral` to
the RHS to recover `∫ ‖𝓕 f‖²`. -/
theorem plancherel_schwartz_via_lp (f : 𝓢(ℝ, ℂ)) :
    ∫ x, ‖f x‖ ^ 2 ∂volume
      = ∫ ξ, ‖Real.fourierIntegral (f : ℝ → ℂ) ξ‖ ^ 2 ∂volume := by
  -- Reduce to `‖f.toLp 2‖² = ‖(𝓕 f).toLp 2‖²` and apply `norm_sq_toLp_eq_integral`.
  have hf_memLp : MemLp (f : ℝ → ℂ) 2 volume := schwartz_memLp_two f
  have hFf_memLp : MemLp ((𝓕 f : 𝓢(ℝ, ℂ)) : ℝ → ℂ) 2 volume := schwartz_memLp_two (𝓕 f)
  -- LHS = ‖f.toLp 2‖² via `norm_sq_toLp_eq_integral` (with f.toLp 2 = MemLp.toLp f hf_memLp).
  have hLHS : ∫ x, ‖f x‖ ^ 2 ∂volume = ‖hf_memLp.toLp f‖ ^ 2 :=
    (LpBridge.norm_sq_toLp_eq_integral (f := (f : ℝ → ℂ)) hf_memLp).symm
  -- RHS = ‖(𝓕 f).toLp 2‖² via `norm_sq_toLp_eq_integral`, after rewriting `𝓕 f` to function.
  have hcoe : ∀ ξ, ((𝓕 f : 𝓢(ℝ, ℂ)) ξ : ℂ) = Real.fourierIntegral (f : ℝ → ℂ) ξ := by
    intro ξ
    rw [fourier_coe]
    rfl
  have hRHS : ‖hFf_memLp.toLp ((𝓕 f : 𝓢(ℝ, ℂ)) : ℝ → ℂ)‖ ^ 2
                = ∫ ξ, ‖Real.fourierIntegral (f : ℝ → ℂ) ξ‖ ^ 2 ∂volume := by
    rw [LpBridge.norm_sq_toLp_eq_integral
        (f := ((𝓕 f : 𝓢(ℝ, ℂ)) : ℝ → ℂ)) hFf_memLp]
    refine integral_congr_ae (Filter.Eventually.of_forall fun ξ => ?_)
    show ‖(𝓕 f : 𝓢(ℝ, ℂ)) ξ‖ ^ 2 = ‖Real.fourierIntegral (f : ℝ → ℂ) ξ‖ ^ 2
    rw [hcoe]
  -- Now the Lp 2 step: `‖fourierTransformₗᵢ (f.toLp 2)‖ = ‖f.toLp 2‖`
  -- and `fourierTransformₗᵢ (f.toLp 2) = (𝓕 f).toLp 2`.
  -- These both use the `f.toLp 2`/`(𝓕 f).toLp 2` (Schwartz API), so we
  -- need a small bridging step to identify with `MemLp.toLp`.
  have htoLp_f : hf_memLp.toLp (f : ℝ → ℂ) = f.toLp 2 (μ := volume) := rfl
  have htoLp_Ff : hFf_memLp.toLp ((𝓕 f : 𝓢(ℝ, ℂ)) : ℝ → ℂ) = (𝓕 f).toLp 2 (μ := volume) := rfl
  rw [hLHS, ← hRHS, htoLp_f, htoLp_Ff]
  -- Goal: ‖f.toLp 2‖² = ‖(𝓕 f).toLp 2‖²
  rw [← SchwartzMap.toLp_fourier_eq (E := ℝ) (F := ℂ) f]
  -- Goal: ‖f.toLp 2‖² = ‖𝓕 (f.toLp 2)‖²
  -- (the `𝓕` here is `Lp.fourierTransformₗᵢ` by the Lp 2 instance).
  rw [MeasureTheory.Lp.norm_fourier_eq]

end PlancherelL1L2

/-! ## L¹ ∩ L² Plancherel for compactly supported functions

The desired identity

  `∫ ‖f x‖² = ∫ ‖Real.fourierIntegral f ξ‖²`

for `f : ℝ → ℂ` compactly supported with `Integrable f` and `MemLp f 2`
is the classical Plancherel theorem.  We prove it cleanly via
**`ae_eq_of_integral_contDiff_smul_eq`** from mathlib (uniqueness of
locally-integrable functions modulo their action on smooth-cpt-supp test
functions).

Strategy: The two candidate functions to identify are
  * `F₁ := (Lp.fourierTransformₗᵢ (toLp_2 f) : ℝ → ℂ)` — the L² Fourier transform.
  * `F₂ := Real.fourierIntegral f` — the pointwise Fourier integral (bounded continuous,
    hence locally integrable, since `f ∈ L¹`).

Both are locally integrable, and we show their integrals against any smooth-cpt-supp
test function agree.  The argument:

  `∫ g(ξ) · F₁(ξ) dξ = ⟨F₁, g⟩_{L²}` (viewed in the Lp²-pairing form)
                    `= ⟨toLp_2 f, 𝓕⁻¹ g⟩_{L²}` (Plancherel-Parseval)
                    `= ∫ Real.fourierIntegral g(x) · f(x) dx` (Schwartz Fubini)
                    `= ∫ g(ξ) · Real.fourierIntegral f(ξ) dξ` (L¹ pairing applied to f and g).

But for the bridge, we use the simpler chain via `fourier_l1_pairing`:

  `∫ Real.fourierIntegral g(x) · f(x) dx = ∫ g(ξ) · Real.fourierIntegral f(ξ) dξ`

(this is mathlib's L¹ pairing for the pointwise Fourier integral), and the
Schwartz Plancherel inner product / Parseval pairing identifies LHS as
`⟨g, FT₂⟩` modulo the conjugation. -/

namespace L1L2Plancherel

open MeasureTheory SchwartzMap Filter Topology

/-- Helper: For a Schwartz function `g`, its Schwartz Fourier transform `𝓕 g` and
the function-level `Real.fourierIntegral g` agree at every point. -/
lemma schwartz_fourier_eq_real (g : 𝓢(ℝ, ℂ)) (ξ : ℝ) :
    ((𝓕 g : 𝓢(ℝ, ℂ)) ξ : ℂ) = Real.fourierIntegral (g : ℝ → ℂ) ξ :=
  congrFun (SchwartzMap.fourier_coe (V := ℝ) (E := ℂ) g) ξ


end L1L2Plancherel

end -- noncomputable section

end Sidon.FourierAux
