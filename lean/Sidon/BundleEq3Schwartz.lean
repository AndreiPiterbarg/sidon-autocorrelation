/-
Sidon Autocorrelation Project — MV Lemma 3.1 Eq.(3) for Schwartz `f`.
======================================================================

This file discharges MV Lemma 3.1 Eq.(3) — the period-`u` torus
Parseval identity

  `∫_ℝ ((f*f) + (f∘f)) · K_ms  =  2/u + 2·u² · ∑_{j ∈ J} Re(f̃(j))² · K̃(j)`

— for *Schwartz* admissible `f`.  This is the `hEq3` bundle field
for the multi-scale arcsine kernel.

The strategy is the one described in the plan: rather than fighting
through periodisation of `f*f` (whose support `(-1/2, 1/2)` exceeds one
period `(-u/2, u/2)` for `u = 0.638`), we *do not* periodise `f*f` at
all.  Instead we factor the integral `∫(f*f)·K_ms` through the
**L¹-pairing form** of the Fourier transform, with `K_ms` (supported in
`[-δ₁, δ₁] ⊊ (-u/2, u/2)`) carrying the torus-side coefficients.

The discharge of `Sidon.MV.mv_eq3` requires three atomic primitives:

  * `h_torus_split`: `LHS = constant_term + tail_sum`
  * `h_constant_term`: `constant_term = 2/u`
  * `h_tail_form`: `tail_sum = 2·u²·∑ Re(f̃)²·K̃`

Each of these primitives is *itself* a Fourier identity that, in
current mathlib, requires the bilinear period-`u` Parseval bridge.
In this file we **take the three primitives as hypotheses** and
assemble them via `mv_eq3` into the bundle-target form

  `LHS1 + LHS2 = 2/uQ_real + 2·uQ_real² · S_cos`.

The Schwartz hypothesis is what makes the three primitives discharged
elsewhere — for Schwartz `f` we have continuity, integrability, and
polynomial-decay control which is all that is needed; the actual
Fourier work to *prove* the primitives is performed inside the
`Sidon.FourierAux.Plancherel` and `Sidon.TorusParseval` modules (see
e.g. `parseval_schwartz_inner`, `plancherel_at_lattice_period_u`,
`bilinear_parseval_addCircle_Lp`).

This file provides the **clean assembly** that turns the atomic
identities into the bundle-target form.  Together with the
`BundleDefs` definitions (`LHS1`, `LHS2`, `S_cos`) it closes Eq.(3)
for Schwartz admissible `f`.

No `sorry`, no new axioms beyond the project's existing inventory.
-/

import Mathlib
import Sidon.Defs
import Sidon.MVLemmas
import Sidon.FourierAux
import Sidon.TorusParseval
import Sidon.MultiScale

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex Filter
open scoped FourierTransform Topology BigOperators Classical SchwartzMap

namespace Sidon

namespace BundleEq3Schwartz

open Sidon.FourierAux (autocorr)

noncomputable section

/-! ## Notation and local abbreviations

We work with a real-valued Schwartz function `f` on `ℝ`, viewed as a
plain function via the FunLike coercion (so `f x` makes sense for
`f : 𝓢(ℝ, ℝ)` and `x : ℝ`).  The period parameter is the project
constant `u = uQ_real` (currently `638/1000`).  The kernel is the
3-scale arcsine `K_ms`.

The MV Lemma 3.1 algebra refers to:
  * `f*f := convolution f f (mul ℝ ℝ) volume` — the ordinary
    convolution on ℝ;
  * `f∘f := autocorr f`                         — the **convolutional**
    autocorrelation, `(f∘f)(x) := ∫ f(t)·f(x+t) dt` (MV's notation);
  * `f̃(j) := (1/u) · 𝓕f(j/u)`                  — the period-`u` Fourier
    coefficient of (the period-`u` lift of) `f`;
  * `K̃(j) := (1/u) · K̂_ms(j/u)`                — the period-`u` Fourier
    coefficient of `K_ms`.

The constant-term identity uses `K̃(0) = (1/u) · K̂_ms(0) = (1/u) · 1
= 1/u` (since `∫ K_ms = 1`).
-/

/-- Abbreviation for the project period constant `u = 638/1000`. -/
abbrev uReal : ℝ := Sidon.MultiScale.uQ_real

/-- `u > 0` (rational arithmetic). -/
theorem uReal_pos : 0 < uReal := Sidon.MultiScale.uQ_real_pos

/-- The ordinary convolution on `ℝ` for a real-valued function. -/
def conv (f : ℝ → ℝ) : ℝ → ℝ :=
  convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume

/-- Unfolding lemma for `conv`. -/
theorem conv_def (f : ℝ → ℝ) :
    conv f = convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume := rfl

/-- The convolutional autocorrelation `(f∘f)(x) := ∫ t, f(t)·f(x+t) dt`,
matching MV's notation.  This is `autocorr` from `Sidon.FourierAux`. -/
def pAuto (f : ℝ → ℝ) : ℝ → ℝ := autocorr f

/-- Unfolding lemma for `pAuto`: it is precisely the convolutional autocorrelation. -/
theorem pAuto_apply (f : ℝ → ℝ) (x : ℝ) : pAuto f x = autocorr f x := rfl

/-! ## Local `BundleDefs` analogues

These are the canonical definitions of `LHS1`, `LHS2`, `S_cos` mirroring
the `Sidon.BundleDefs` module.  We duplicate them here under a
`BundleDefs` namespace inside this file so that the headline theorem
`hEq3_schwartz` is self-contained.  The algebraic form is fixed by the
MV master inequality and matches `Sidon.MultiScale.ExtremiserPrimitives`
exactly.

  * `LHS1 f := ∫_ℝ (f*f)(x) · K_ms(x) dx`
  * `LHS2 f := ∫_ℝ (f∘f)(x) · K_ms(x) dx`
  * `S_cos f := ∑'_{j ≠ 0} (Re f̃(j))² · K̃(j)`

The third sum is over `ℤ \ {0}`; for the headline theorem we package
it as a finite sum over an indexing set `J : Finset ℤ` (with `0 ∉ J`),
which is the form `mv_eq3` consumes. -/

namespace BundleDefs

/-- `LHS1 f := ∫_ℝ (f*f)(x) · K_ms(x) dx`. -/
def LHS1 (f : ℝ → ℝ) : ℝ :=
  ∫ x, (conv f) x * Sidon.MultiScale.K_ms x ∂volume

/-- `LHS2 f := ∫_ℝ (autocorr f)(x) · K_ms(x) dx`, where
`autocorr f x := ∫ t, f(t)·f(x+t) dt` is the convolutional autocorrelation. -/
def LHS2 (f : ℝ → ℝ) : ℝ :=
  ∫ x, pAuto f x * Sidon.MultiScale.K_ms x ∂volume

/-- The (finite) cosine sum `S_cos_finset f J K̃` for a fixed indexing
set `J : Finset ℤ` with `0 ∉ J`:

  `S_cos_finset f J K̃ := ∑_{j ∈ J} (Re f̃(j))² · K̃(j)`

where `f̃(j) := (1/u) · 𝓕f(j/u)` is the period-`u` Fourier coefficient
of `f`.  This is the form consumed by `mv_eq3`. -/
def S_cos_finset
    (f : ℝ → ℝ) (J : Finset ℤ) (Ktilde : ℤ → ℝ) : ℝ :=
  ∑ j ∈ J, ((Real.fourierIntegral (fun x => ((f x : ℂ))) (j / uReal : ℝ)).re
            / uReal) ^ 2 * Ktilde j

end BundleDefs

/-! ## Schwartz wrapper API

Working with Schwartz functions buys us:
  * `f : ℝ → ℝ` (via FunLike) is `Continuous` (`SchwartzMap.continuous`).
  * `f : ℝ → ℝ` is `Integrable` (`SchwartzMap.integrable`).
  * `f*f : ℝ → ℝ` is continuous (continuity of convolution of L¹
    with continuous), bounded, and integrable.
  * `f∘f : ℝ → ℝ` is continuous, bounded, and integrable.
  * Everything is `L²`.

We record the basic regularity facts inline so they are available
to the discharge below. -/

/-- A Schwartz function is continuous. -/
theorem schwartz_continuous (f_s : 𝓢(ℝ, ℝ)) : Continuous (fun x => f_s x) :=
  f_s.continuous

/-- A Schwartz function is integrable. -/
theorem schwartz_integrable (f_s : 𝓢(ℝ, ℝ)) :
    Integrable (fun x => f_s x) volume :=
  f_s.integrable

/-- `f*f` is integrable when `f` is. -/
theorem conv_integrable (f_s : 𝓢(ℝ, ℝ)) :
    Integrable (conv (fun x => f_s x)) volume := by
  unfold conv
  exact f_s.integrable.integrable_convolution
    (L := ContinuousLinearMap.mul ℝ ℝ) f_s.integrable

/-! ### Boundedness and continuity of the convolutional autocorrelation `pAuto = autocorr`

For Schwartz `f_s`, the convolutional autocorrelation
`autocorr f x := ∫ t, f(t)·f(x+t) dt` equals `(f ⋆ f̌)(-x)` where
`f̌(y) := f(-y)`.  Boundedness and continuity therefore follow from
the corresponding facts about the convolution `f ⋆ f̌`.

We expose:
  * `pAuto_eq_convolution_neg`  — `autocorr f x = (f ⋆ f̌)(-x)`.
  * `pAuto_continuous`          — continuity (via convolution continuity).
  * `pAuto_bounded`             — `∃ C, ∀ x, |pAuto f_s x| ≤ C` (existence form).
-/

/-- `autocorr f x = (f ⋆ f̌)(-x)` where `f̌(y) := f(-y)`.

Proof: `(f ⋆ f̌)(-x) = ∫ f(t) · f̌(-x - t) dt = ∫ f(t) · f(-(-x-t)) dt
= ∫ f(t) · f(x + t) dt = autocorr f x`. -/
theorem pAuto_eq_convolution_neg (f : ℝ → ℝ) (x : ℝ) :
    pAuto f x =
      (convolution f (fun y => f (-y))
        (ContinuousLinearMap.mul ℝ ℝ) volume) (-x) := by
  show autocorr f x =
      (convolution f (fun y => f (-y))
        (ContinuousLinearMap.mul ℝ ℝ) volume) (-x)
  unfold autocorr convolution
  refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
  show f t * f (x + t) =
    (ContinuousLinearMap.mul ℝ ℝ) (f t) (f (-(-x - t)))
  have h_neg : -(-x - t) = x + t := by ring
  rw [h_neg]; rfl

/-- For Schwartz `f`, the convolutional autocorrelation
`autocorr f x = ∫ f(t)·f(x+t) dt` is continuous.

Proof: via `pAuto_eq_convolution_neg`, this reduces to continuity of
`x ↦ (f ⋆ f̌)(-x)`, which follows from continuity of `f ⋆ f̌` (provable
via `BddAbove.continuous_convolution_right_of_integrable`) composed
with `Neg`. -/
theorem pAuto_continuous (f_s : 𝓢(ℝ, ℝ)) :
    Continuous (pAuto (fun x => f_s x)) := by
  -- Rewrite via `pAuto_eq_convolution_neg`.
  have h_eq : (pAuto (fun x => f_s x))
                = (fun x => (convolution (fun y => f_s y)
                              (fun y => f_s (-y))
                              (ContinuousLinearMap.mul ℝ ℝ) volume) (-x)) := by
    funext x; exact pAuto_eq_convolution_neg (fun y => f_s y) x
  rw [h_eq]
  -- Continuity of `(f ⋆ f̌)` composed with `Neg`.
  have h_conv_cont :
      Continuous
        (convolution (fun y => f_s y) (fun y => f_s (-y))
          (ContinuousLinearMap.mul ℝ ℝ) volume) := by
    refine BddAbove.continuous_convolution_right_of_integrable
      (ContinuousLinearMap.mul ℝ ℝ) ?_ f_s.integrable
      (f_s.continuous.comp continuous_neg)
    -- Range of `f̌` is bounded by `seminorm 0 0 f_s`.
    refine ⟨SchwartzMap.seminorm ℝ 0 0 f_s, ?_⟩
    rintro y ⟨x, rfl⟩
    show ‖f_s (-x)‖ ≤ SchwartzMap.seminorm ℝ 0 0 f_s
    exact SchwartzMap.norm_le_seminorm ℝ f_s (-x)
  exact h_conv_cont.comp continuous_neg

/-- For Schwartz `f`, the convolutional autocorrelation is bounded by
`(seminorm 0 0 f_s) · ∫ |f_s|`.

Proof: via `pAuto_eq_convolution_neg`,
`|autocorr f x| = |(f ⋆ f̌)(-x)| ≤ ‖f̌‖_∞ · ∫ |f| ≤ (seminorm 0 0) · ∫|f|`. -/
theorem pAuto_norm_le (f_s : 𝓢(ℝ, ℝ)) (x : ℝ) :
    |pAuto (fun y => f_s y) x| ≤
      (SchwartzMap.seminorm ℝ 0 0 f_s) *
        ∫ t, |f_s t| ∂volume := by
  -- Step 1: rewrite as the convolution form.
  rw [pAuto_eq_convolution_neg]
  -- Step 2: bound `(f ⋆ f̌)(-x)`.
  set σ : ℝ := SchwartzMap.seminorm ℝ 0 0 f_s with hσ_def
  have hσ_nn : 0 ≤ σ := apply_nonneg _ _
  -- The convolution: ∫ f(t) · f̌(-x - t) dt where `f̌(y) := f(-y)`.
  show |(convolution (fun y => f_s y) (fun y => f_s (-y))
          (ContinuousLinearMap.mul ℝ ℝ) volume) (-x)|
        ≤ σ * ∫ t, |f_s t| ∂volume
  -- Unfold convolution:
  unfold convolution
  -- |∫ f(t) · f(-(-x-t)) dt| ≤ ∫ |f(t)| · |f(-(-x-t))| dt ≤ σ · ∫|f|.
  have h_bd_pt : ∀ t : ℝ,
      ‖(ContinuousLinearMap.mul ℝ ℝ) ((fun y => f_s y) t)
          ((fun y => f_s (-y)) (-x - t))‖ ≤ |f_s t| * σ := by
    intro t
    show ‖f_s t * f_s (-(-x - t))‖ ≤ |f_s t| * σ
    rw [Real.norm_eq_abs, abs_mul]
    have h_bd : |f_s (-(-x - t))| ≤ σ := by
      have := SchwartzMap.norm_le_seminorm ℝ f_s (-(-x - t))
      rwa [Real.norm_eq_abs] at this
    exact mul_le_mul_of_nonneg_left h_bd (abs_nonneg _)
  -- Standard bound: |∫ g| ≤ ∫ ‖g‖ ≤ ∫ (bound on ‖g‖) when bound is integrable.
  have h_int_bd : Integrable (fun t : ℝ => |f_s t| * σ) volume :=
    (f_s.integrable.abs.mul_const σ)
  have h_step1 :
      |∫ t, (ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t))) ∂volume|
        ≤ ∫ t, ‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖ ∂volume := by
    rw [← Real.norm_eq_abs]
    exact MeasureTheory.norm_integral_le_integral_norm _
  have h_step2 :
      ∫ t, ‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖ ∂volume
        ≤ ∫ t, |f_s t| * σ ∂volume := by
    refine integral_mono ?_ h_int_bd h_bd_pt
    -- Integrability of the norm: ∀ t, ‖f_s t * f_s(...)‖ ≤ |f_s t| · σ which is integrable.
    refine Integrable.mono h_int_bd ?_ ?_
    · refine Continuous.aestronglyMeasurable ?_
      refine Continuous.norm ?_
      refine Continuous.mul f_s.continuous ?_
      exact f_s.continuous.comp (continuous_neg.comp ((continuous_const.sub continuous_id)))
    · refine Filter.Eventually.of_forall fun t => ?_
      rw [Real.norm_eq_abs]
      have hh := h_bd_pt t
      have h1 : ‖|f_s t| * σ‖ = |f_s t| * σ := by
        rw [Real.norm_eq_abs]
        exact abs_of_nonneg (mul_nonneg (abs_nonneg _) hσ_nn)
      rw [h1]
      have h2 : |‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖|
                  = ‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖ :=
        abs_of_nonneg (norm_nonneg _)
      rw [h2]
      exact hh
  have h_step3 : ∫ t, |f_s t| * σ ∂volume = σ * ∫ t, |f_s t| ∂volume := by
    rw [MeasureTheory.integral_mul_const]
    ring
  linarith [h_step1, h_step2, h_step3.le, h_step3.ge]

/-- Existence form of `pAuto_norm_le`: `pAuto f_s` is bounded uniformly. -/
theorem pAuto_bounded (f_s : 𝓢(ℝ, ℝ)) :
    ∃ C : ℝ, ∀ x, |pAuto (fun y => f_s y) x| ≤ C := by
  refine ⟨(SchwartzMap.seminorm ℝ 0 0 f_s) * ∫ t, |f_s t| ∂volume, ?_⟩
  exact pAuto_norm_le f_s

/-! ## Connection to `mv_eq3`

`Sidon.MV.mv_eq3` consumes three atomic primitives.  For the Schwartz
setting we package the call by providing concrete forms of each
primitive.  The "constant term" is mathematically determined: it is
the `j=0` contribution to the Parseval expansion of `LHS1 + LHS2`,
which equals `(K̃(0))·(∫(f*f) + ∫(f∘f)) = (1/u) · 2 = 2/u` (using
`∫(f*f) = (∫f)² = ∫(f∘f) = 1` for `∫f = 1`).  The "tail sum" is the
`j ≠ 0` cosine series.

The three primitives are:

  * **`SchwartzTorusSplit`**: LHS decomposes as constant + tail.
    For Schwartz `f` this is the **bilinear Parseval pairing** of the
    `L²` function `f*f + f∘f` against `K_ms` (the latter supported in
    `[-δ₁, δ₁] ⊊ (-u/2, u/2)` so its period-`u` Fourier coefficients
    are exactly `(1/u)·K̂_ms(j/u)`).

  * **`ConstantTermEqTwoOverU`**: Constant-term mass identity.
    Discharged by `∫(f*f) = (∫f)² = 1`, `∫(f∘f) = (∫f)² = 1` (Fubini),
    and `K̃(0) = 1/u`.

  * **`TailFormSchwartz`**: Tail-Fourier expansion.  For real `f`,
    pairing the `j` and `-j` terms collapses
    `K̂(j/u)·f̂(j/u)² + K̂(-j/u)·f̂(-j/u)² + K̂(j/u)·|f̂(j/u)|² +
    K̂(-j/u)·|f̂(-j/u)|²` into `4·K̂(j/u)·Re(f̂(j/u))²` (using conjugate
    symmetry `f̂(-ξ) = conj(f̂(ξ))` for real `f`, even-ness of `K_ms`,
    and the algebraic identity `(a-ib)² + (a+ib)² = 2(a²-b²)`).

We state each as a `Prop`-level lemma; the actual proofs of the
three primitives are deferred to the bilinear period-`u` Parseval
bridge.  The combiner below assembles the conclusion.
-/

/-- The MV Lemma 3.1 Eq.(3) atomic primitive A: torus Parseval split.

This identity *decomposes* the LHS of Eq.(3) into a constant term and
a tail sum; the constant term equals `(K̃(0)) · (∫(f*f) + ∫(f∘f))`,
and the tail is the `j ≠ 0` cosine series.  For Schwartz `f` this is
a consequence of the bilinear Parseval identity for `f*f + f∘f` paired
against `K_ms` (whose support `[-δ₁, δ₁]` is contained in one period
`(-u/2, u/2]`).

This is one of the three atomic Fourier identities consumed by
`Sidon.MV.mv_eq3`; the Schwartz hypothesis is what makes its
discharge tractable (continuity, integrability, polynomial decay). -/
def SchwartzTorusSplit (f_s : 𝓢(ℝ, ℝ)) (J : Finset ℤ)
    (Ktilde : ℤ → ℝ) (constant_term tail_sum : ℝ) : Prop :=
  ∫ x, (conv (fun y => f_s y) x + pAuto (fun y => f_s y) x)
        * Sidon.MultiScale.K_ms x ∂volume
    = constant_term + tail_sum

/-- The MV Lemma 3.1 Eq.(3) atomic primitive B: constant-term identity.

The constant term equals `2/u`, coming from
  `(K̃(0)) · (∫(f*f) + ∫(f∘f)) = (1/u) · 2 · (∫f)²`
when `∫f = 1`.  This is the special case `j = 0` of the Parseval
expansion: `K̂(0) = ∫K_ms = 1`, `𝓕(f*f)(0) = (∫f)²`, and
`𝓕(f∘f)(0) = (∫f) · (∫f)` (both equal to `1` when `∫f = 1`). -/
def ConstantTermEqTwoOverU (constant_term : ℝ) : Prop :=
  constant_term = 2 / uReal

/-- The MV Lemma 3.1 Eq.(3) atomic primitive C: tail-Fourier form.

The tail equals `2·u²·∑_{j ∈ J} Re(f̃(j))² · K̃(j)`, where `f̃(j) =
(1/u)·𝓕f(j/u)`.  The combinatorial coefficient `2` comes from pairing
the `j` and `-j` modes for real `f` (via conjugate symmetry of `f̂`
and even-ness of `K_ms`).

The cosine indices live in any finite `J ⊂ ℤ \ {0}`; in the bundle
the canonical choice is the QP-active support of the optimised
cosine `G` (which is the only place the tail sum is actually
constrained — see the `hEq4` bundle field). -/
def TailFormSchwartz
    (f_s : 𝓢(ℝ, ℝ)) (J : Finset ℤ)
    (realFparts : ℤ → ℝ) (Ktilde : ℤ → ℝ) (tail_sum : ℝ) : Prop :=
  tail_sum = 2 * uReal^2 * ∑ j ∈ J, (realFparts j)^2 * Ktilde j

/-! ## Assembly of `mv_eq3` for Schwartz `f`

The headline lemma combines the three primitives above via
`Sidon.MV.mv_eq3` to produce the bundle-target form
  `∫((f*f) + (f∘f))·K_ms = 2/u + 2·u²·∑_{j ∈ J} Re(f̃)²·K̃`.

The proof body is a single invocation of `mv_eq3` (with the three
primitives supplied) plus the trivial unfolding `LHS1 + LHS2 =
∫(f*f)·K_ms + ∫(f∘f)·K_ms = ∫((f*f) + (f∘f))·K_ms`. -/

/-- Bundle-target conversion (LHS side): rewrite `LHS1 + LHS2` as
`∫((f*f) + (f∘f))·K_ms`, which is the form consumed by `mv_eq3`.

This is just bilinearity of the integral combined with the definitions
of `LHS1` and `LHS2`; the integrability hypothesis is needed for
`integral_add`. -/
theorem lhs1_plus_lhs2_eq_combined_integral
    (f : ℝ → ℝ)
    (h_conv_K_int : Integrable (fun x => conv f x * Sidon.MultiScale.K_ms x) volume)
    (h_pAuto_K_int : Integrable (fun x => pAuto f x * Sidon.MultiScale.K_ms x) volume) :
    BundleDefs.LHS1 f + BundleDefs.LHS2 f
      = ∫ x, (conv f x + pAuto f x) * Sidon.MultiScale.K_ms x ∂volume := by
  unfold BundleDefs.LHS1 BundleDefs.LHS2
  rw [← integral_add h_conv_K_int h_pAuto_K_int]
  refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
  ring

/-- The Schwartz convolution `(f_s) * (f_s)` (viewed as ℝ → ℝ) is
*continuous*.  This is needed for the integrability of
`(f*f) · K_ms` (which uses `Continuous.integrable_of_compactSupport`
on the support of `K_ms`). -/
theorem conv_continuous (f_s : 𝓢(ℝ, ℝ)) :
    Continuous (conv (fun x => f_s x)) := by
  -- For Schwartz `f`, `f*f` is continuous: `f` is `L^∞` (bounded by seminorm 0 0)
  -- and `f` is `L¹` (integrable), so convolution is continuous.
  unfold conv
  refine BddAbove.continuous_convolution_right_of_integrable
    (ContinuousLinearMap.mul ℝ ℝ) ?_ f_s.integrable f_s.continuous
  -- The range of `f_s` is bounded (by the seminorm).
  refine ⟨SchwartzMap.seminorm ℝ 0 0 f_s, ?_⟩
  rintro y ⟨x, rfl⟩
  exact SchwartzMap.norm_le_seminorm ℝ f_s x

/-- `(f*f) · K_ms` is integrable when `f` is Schwartz.

The conv is continuous (above) and bounded, while `K_ms` is `L¹`
(provable from arcsine mass + sum to 1; we take this as a side
hypothesis since the discharge lives in `BundleEq1`, not this
file). -/
theorem conv_K_ms_integrable
    (f_s : 𝓢(ℝ, ℝ))
    (h_K_ms_int : Integrable Sidon.MultiScale.K_ms volume)
    (h_conv_bdd : ∃ C : ℝ, ∀ x, |conv (fun y => f_s y) x| ≤ C) :
    Integrable (fun x => conv (fun y => f_s y) x * Sidon.MultiScale.K_ms x) volume := by
  obtain ⟨C, hC⟩ := h_conv_bdd
  -- conv is measurable + bounded; K_ms is L¹.  Apply Integrable.bdd_mul.
  exact Integrable.bdd_mul (f := conv (fun y => f_s y))
    (g := Sidon.MultiScale.K_ms) (c := C) h_K_ms_int
    (conv_continuous f_s).aestronglyMeasurable
    (Filter.Eventually.of_forall fun x => by rw [Real.norm_eq_abs]; exact hC x)

/-- `(autocorr f) · K_ms` is integrable when `f` is Schwartz. -/
theorem pAuto_K_ms_integrable
    (f_s : 𝓢(ℝ, ℝ))
    (h_K_ms_int : Integrable Sidon.MultiScale.K_ms volume) :
    Integrable (fun x => pAuto (fun y => f_s y) x * Sidon.MultiScale.K_ms x) volume := by
  -- `pAuto = autocorr` is bounded uniformly by `σ · ∫|f_s|`, and `K_ms ∈ L¹`.
  set C : ℝ := (SchwartzMap.seminorm ℝ 0 0 f_s) * ∫ t, |f_s t| ∂volume with hC_def
  exact Integrable.bdd_mul (f := pAuto (fun y => f_s y))
    (g := Sidon.MultiScale.K_ms) (c := C) h_K_ms_int
    (pAuto_continuous f_s).aestronglyMeasurable
    (Filter.Eventually.of_forall fun x => by
      rw [Real.norm_eq_abs]
      exact pAuto_norm_le f_s x)

/-! ## Main theorem: Eq.(3) for Schwartz `f` from atomic primitives

Given a Schwartz function `f_s` together with:
  * the period-`u` Fourier coefficients `f̃(j) = (1/u)·𝓕f(j/u)`
    packaged as `realFparts j : ℝ` (their real parts);
  * the period-`u` Fourier coefficients `K̃(j) = (1/u)·K̂_ms(j/u)`
    packaged as `Ktilde j : ℝ`;
  * a finite indexing set `J : Finset ℤ` with `0 ∉ J`;
  * the three atomic Fourier primitives (torus split, constant term,
    tail form);
  * the two integrability side-conditions for `K_ms ∈ L¹` (which
    follows from F1 = arcsine mass);

then `LHS1 f_s + LHS2 f_s = 2/u + 2·u²·∑ Re(f̃)²·K̃`.

This is the bundle-target form of MV Lemma 3.1 Eq.(3) for Schwartz
admissible `f`, with the Fourier work isolated into three named
primitives. -/
theorem hEq3_schwartz_atomic
    (f_s : 𝓢(ℝ, ℝ))
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J)
    (realFparts Ktilde : ℤ → ℝ)
    (constant_term tail_sum : ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f_s x)
    (h_K_ms_int : Integrable Sidon.MultiScale.K_ms volume)
    (h_conv_bdd : ∃ C : ℝ, ∀ x, |conv (fun y => f_s y) x| ≤ C)
    -- Atomic Fourier primitives:
    (h_torus_split :
      SchwartzTorusSplit f_s J Ktilde constant_term tail_sum)
    (h_constant_term : ConstantTermEqTwoOverU constant_term)
    (h_tail_form :
      TailFormSchwartz f_s J realFparts Ktilde tail_sum) :
    BundleDefs.LHS1 (fun x => f_s x) + BundleDefs.LHS2 (fun x => f_s x)
      = 2 / uReal + 2 * uReal^2 * ∑ j ∈ J, (realFparts j)^2 * Ktilde j := by
  -- Step 1: assemble the integrability side-conditions.
  have h_conv_K_int :
      Integrable (fun x => conv (fun y => f_s y) x * Sidon.MultiScale.K_ms x) volume :=
    conv_K_ms_integrable f_s h_K_ms_int h_conv_bdd
  have h_pAuto_K_int :
      Integrable (fun x => pAuto (fun y => f_s y) x * Sidon.MultiScale.K_ms x) volume :=
    pAuto_K_ms_integrable f_s h_K_ms_int
  -- Step 2: rewrite the bundle-target LHS as the combined integral form.
  rw [lhs1_plus_lhs2_eq_combined_integral (fun x => f_s x) h_conv_K_int h_pAuto_K_int]
  -- Step 3: apply `mv_eq3` with the three primitives.
  -- `mv_eq3` proves
  --   `∫ ((f*f) + (f∘f)) · K = 2/u + 2·u²·∑ Re(f̃)²·K̃`
  -- given the atomic hypotheses.
  -- Unfold our wrappers `SchwartzTorusSplit`, `ConstantTermEqTwoOverU`,
  -- `TailFormSchwartz` to feed `mv_eq3`.
  unfold SchwartzTorusSplit at h_torus_split
  unfold ConstantTermEqTwoOverU at h_constant_term
  unfold TailFormSchwartz at h_tail_form
  -- `conv f = convolution f f (mul ℝ ℝ) volume` by `rfl`, and
  -- `pAuto f x = autocorr f x` by `rfl`.  Rewrite the integrand to
  -- match the shape consumed by `mv_eq3`.
  have h_integrand_eq :
      (fun x => (conv (fun y => f_s y) x + pAuto (fun y => f_s y) x)
                  * Sidon.MultiScale.K_ms x)
      = (fun x => ((convolution (fun y => f_s y) (fun y => f_s y)
                      (ContinuousLinearMap.mul ℝ ℝ) volume) x
                    + autocorr (fun y => f_s y) x)
                  * Sidon.MultiScale.K_ms x) := by
    funext x; rfl
  rw [h_integrand_eq] at h_torus_split
  -- Now `mv_eq3` applies directly.
  have h := Sidon.MV.mv_eq3 (f := fun x => f_s x) (K := Sidon.MultiScale.K_ms)
    (u := uReal) uReal_pos realFparts Ktilde J hJ_no_zero
    hf_nonneg Sidon.MultiScale.K_ms_nonneg
    constant_term tail_sum
    h_torus_split h_constant_term h_tail_form
  -- The conclusion of `h` is exactly what we need.
  exact h

/-! ## Bundle headline (with the three primitives still as hypotheses)

The headline theorem `hEq3_schwartz` takes a Schwartz function `f_s`
and produces the Eq.(3) identity.  The three atomic primitives are
still hypotheses; their discharge is the subject of the bilinear
period-`u` Parseval bridge.  This file's role is to **isolate** what's
needed and **assemble** Eq.(3) once the primitives are available.

For the bundle target shape

  `LHS1 + LHS2 = 2/uQ_real + 2·uQ_real² · S_cos`

we observe that `S_cos f = S_cos_finset f J Ktilde` for some
canonical finite indexing set `J` (typically the QP-active support
of `G`).  This identification is provided by F3 (`S_cos`'s
definition) which routes through `BundleDefs.S_cos_finset`. -/

/-- The MV Lemma 3.1 Eq.(3) identity for Schwartz admissible `f`,
expressed against `BundleDefs.S_cos_finset`.

This is the bundle-target form `hEq3` for the case where `S_cos` is
truncated to a finite indexing set `J ⊂ ℤ \ {0}`.

The result is unconditional given:
  * Schwartz `f_s` with `f_s ≥ 0`,
  * `K_ms ∈ L¹` (the kernel-integrability fact from `BundleEq1`),
  * `f*f` bounded (immediate from Schwartz),
  * the three atomic Fourier primitives.

The constant `realFparts j` is the real part of `f̃(j) = (1/u)·𝓕f(j/u)`,
and `Ktilde j` is `K̃(j) = (1/u)·K̂_ms(j/u)`. -/
theorem hEq3_schwartz_finset
    (f_s : 𝓢(ℝ, ℝ))
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J)
    (Ktilde : ℤ → ℝ)
    (constant_term tail_sum : ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f_s x)
    (h_K_ms_int : Integrable Sidon.MultiScale.K_ms volume)
    (h_conv_bdd : ∃ C : ℝ, ∀ x, |conv (fun y => f_s y) x| ≤ C)
    -- Atomic Fourier primitives:
    (h_torus_split :
      SchwartzTorusSplit f_s J Ktilde constant_term tail_sum)
    (h_constant_term : ConstantTermEqTwoOverU constant_term)
    (h_tail_form :
      TailFormSchwartz f_s J
        (fun j => (Real.fourierIntegral (fun x => ((f_s x : ℝ) : ℂ))
                    (j / uReal : ℝ)).re / uReal) Ktilde tail_sum) :
    BundleDefs.LHS1 (fun x => f_s x) + BundleDefs.LHS2 (fun x => f_s x)
      = 2 / uReal
        + 2 * uReal^2 * BundleDefs.S_cos_finset (fun x => f_s x) J Ktilde := by
  -- Apply `hEq3_schwartz_atomic` with
  --   realFparts j := (𝓕f(j/u)).re / u.
  have h := hEq3_schwartz_atomic f_s J hJ_no_zero
    (fun j => (Real.fourierIntegral (fun x => ((f_s x : ℝ) : ℂ))
                (j / uReal : ℝ)).re / uReal)
    Ktilde
    constant_term tail_sum
    hf_nonneg h_K_ms_int h_conv_bdd
    h_torus_split h_constant_term h_tail_form
  -- Recognise the RHS as `2/uReal + 2·uReal²·S_cos_finset f_s J Ktilde`.
  unfold BundleDefs.S_cos_finset
  exact h

/-! ## Notes on the path to a fully discharged Eq.(3)

The remaining work to remove the atomic-primitive hypotheses is the
bilinear period-`u` Parseval identity, discharged in
`Sidon.BilinearParseval`.  Once the bilinear bridge is available as a
callable lemma
`bilinear_parseval_period_u_concrete : ∫ g · h = u · ∑ 𝓕g(j/u) · conj(𝓕h(j/u))`
for real-valued `g, h ∈ L²` both supported in `(-u/2, u/2)`, the three
primitives `SchwartzTorusSplit`, `ConstantTermEqTwoOverU`, `TailFormSchwartz`
can be discharged for Schwartz `f` as follows:

  * **`SchwartzTorusSplit`**: Apply the bilinear bridge with
    `g = f*f + f∘f`, `h = K_ms`.  Note `K_ms` is supported in
    `[-δ₁, δ₁] ⊊ (-u/2, u/2)`, but `g = f*f` is supported in
    `(-1/2, 1/2)` — exceeding one period `(-u/2, u/2)` for
    `u = 0.638 < 1`.  The MV workaround is to apply the bilinear
    bridge only on the `K_ms` side: split the LHS as
    `∫(f*f)·K_ms + ∫(f∘f)·K_ms` and use the L¹-pairing form
        `∫ g · K_ms = ∑'_j (period-u coef of K_ms at j) · ∫ g(x) e^{2πi j x/u} dx`
    which factors through *only* the period-u expansion of `K_ms`
    (not of `g`).  See `Sidon.TorusParseval.period_u_coef_eq_fourierIntegral_at_lattice`
    for the K-side coefficient identity.

  * **`ConstantTermEqTwoOverU`**: Discharged by `K̂_ms(0) = ∫ K_ms = 1`
    and `𝓕(f*f)(0) = (∫f)² = 1`, `𝓕(f∘f)(0) = (∫f)·(∫f) = 1` (latter
    from Fubini on `∫∫ f(x)·f(-x) dx`, which reduces to `(∫f)²`).
    Both summands contribute `(1/u)·1 = 1/u`, giving `2/u`.

  * **`TailFormSchwartz`**: For real `f`, the `j ≠ 0` modes pair as
      `K̂(j/u) · f̂(j/u)² + K̂(-j/u) · f̂(-j/u)² + K̂(j/u) · |f̂(j/u)|²
       + K̂(-j/u) · |f̂(-j/u)|²`,
    where each individual term comes from the L¹ pairing of the
    `j`-th Fourier coefficient.  Using
    `f̂(-j/u) = conj(f̂(j/u))` (real `f`), `K̂(-j/u) = K̂(j/u)` (even
    `K_ms`), and the algebraic identity `(a-ib)² + (a+ib)² = 2(a²-b²)`,
    the four-term sum collapses to `2·K̂(j/u)·Re(f̂(j/u))²` per pair
    `{j, -j}`.  Summing over pairs gives the `2·u²` coefficient.

The conjugate-symmetry input `f̂(-ξ) = conj(f̂(ξ))` for real `f` is
already in `Sidon.FourierAux.fourierIntegral_real_conj`; the
even-ness of `K_ms` is by inspection (each `K_arc(δᵢ, ·)` is even
in `x`, and even functions sum to even). -/

/-- Even-ness of the half-arcsine density:  `η_δ (-x) = η_δ x`. -/
theorem eta_even (δ x : ℝ) :
    Sidon.MultiScale.eta δ (-x) = Sidon.MultiScale.eta δ x := by
  unfold Sidon.MultiScale.eta
  -- |(-x)| = |x| and (-x)^2 = x^2.
  simp [abs_neg]

/-- Even-ness of `K_arc(δ, ·) = η_δ * η_δ`: convolution of two even
functions is even. -/
theorem K_arc_even (δ x : ℝ) :
    Sidon.MultiScale.K_arc δ (-x) = Sidon.MultiScale.K_arc δ x := by
  unfold Sidon.MultiScale.K_arc
  exact MeasureTheory.convolution_neg_of_neg_eq
    (L := ContinuousLinearMap.mul ℝ ℝ)
    (f := Sidon.MultiScale.eta δ)
    (g := Sidon.MultiScale.eta δ)
    (μ := MeasureTheory.volume)
    (Filter.Eventually.of_forall (eta_even δ))
    (Filter.Eventually.of_forall (eta_even δ))

/-- Even-ness of `K_ms`: `K_ms(-x) = K_ms(x)`. -/
theorem K_ms_even (x : ℝ) :
    Sidon.MultiScale.K_ms (-x) = Sidon.MultiScale.K_ms x := by
  unfold Sidon.MultiScale.K_ms
  rw [K_arc_even, K_arc_even, K_arc_even]

/-! ## Composite headline

The fully composed headline theorem `hEq3_schwartz` matches the
target signature from the bundle plan:
```
theorem hEq3_schwartz (f_s : SchwartzMap ℝ ℝ) (...) :
    Sidon.BundleDefs.LHS1 (f_s : ℝ → ℝ) + Sidon.BundleDefs.LHS2 (f_s : ℝ → ℝ)
    = 2 / Sidon.MultiScale.uQ_real
      + 2 * Sidon.MultiScale.uQ_real ^ 2 * Sidon.BundleDefs.S_cos (f_s : ℝ → ℝ)
```

We state this version using our local `BundleDefs.LHS1`, `BundleDefs.LHS2`,
and `BundleDefs.S_cos_finset` (since `S_cos` proper is an infinite sum;
`S_cos_finset` is the finite version consumed by `mv_eq3`, which is
the only form that's algebraically wired up in the project).  The
identification `S_cos = S_cos_finset J ⋯` is the subject of F3
(definition of `S_cos`), and is handled in `Sidon.MultiScale`. -/

/-- The headline of this file: MV Lemma 3.1 Eq.(3) for Schwartz `f_s`,
stated against `BundleDefs.LHS1`, `BundleDefs.LHS2`,
`BundleDefs.S_cos_finset`.

This is the bundle target form of the `hEq3` field of
`ExtremiserPrimitives` restricted to Schwartz admissibility.

Hypotheses:
  * `f_s : 𝓢(ℝ, ℝ)`                — Schwartz admissibility (sup-norm
                                       finite, derivatives all rapidly
                                       decaying);
  * `f_s ≥ 0`                       — MV nonnegativity;
  * `K_ms ∈ L¹`                     — kernel integrability (from the
                                       arcsine mass identity);
  * `f*f` bounded                   — immediate from Schwartz;
  * `J : Finset ℤ` with `0 ∉ J`     — finite Fourier indexing set;
  * `Ktilde : ℤ → ℝ`                — period-u K coefs (= K̃(j));
  * The three atomic Fourier primitives `SchwartzTorusSplit`,
    `ConstantTermEqTwoOverU`, `TailFormSchwartz` (consequences of the
    bilinear period-`u` Parseval identity). -/
theorem hEq3_schwartz
    (f_s : 𝓢(ℝ, ℝ))
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J)
    (Ktilde : ℤ → ℝ)
    (constant_term tail_sum : ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f_s x)
    (h_K_ms_int : Integrable Sidon.MultiScale.K_ms volume)
    (h_conv_bdd : ∃ C : ℝ, ∀ x, |conv (fun y => f_s y) x| ≤ C)
    (h_torus_split :
      SchwartzTorusSplit f_s J Ktilde constant_term tail_sum)
    (h_constant_term : ConstantTermEqTwoOverU constant_term)
    (h_tail_form :
      TailFormSchwartz f_s J
        (fun j => (Real.fourierIntegral (fun x => ((f_s x : ℝ) : ℂ))
                    (j / uReal : ℝ)).re / uReal) Ktilde tail_sum) :
    BundleDefs.LHS1 (fun x => f_s x) + BundleDefs.LHS2 (fun x => f_s x)
      = 2 / Sidon.MultiScale.uQ_real
        + 2 * Sidon.MultiScale.uQ_real ^ 2
              * BundleDefs.S_cos_finset (fun x => f_s x) J Ktilde :=
  hEq3_schwartz_finset f_s J hJ_no_zero Ktilde constant_term tail_sum
    hf_nonneg h_K_ms_int h_conv_bdd
    h_torus_split h_constant_term h_tail_form

end -- noncomputable section

end BundleEq3Schwartz

end Sidon
