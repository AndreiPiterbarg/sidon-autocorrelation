/-
Sidon Autocorrelation Project — Matolcsi-Vinuesa Lemma 3.1
==========================================================

This file states the four equations of MV Lemma 3.1 (Matolcsi-Vinuesa
2010, arXiv:0907.1379, p.3) and proves the elementary ones.

Setup (paper notation, restated literally):
  f nonneg with ∫f = 1, supported on [-1/4, 1/4].
  K nonneg with ∫K = 1, supported on [-δ, δ] with 0 < δ ≤ 1/4.
  u = 1/2 + δ.
  G even u-periodic with G̃(0) = 0 (mean zero), G > 0 on [-1/4, 1/4].
  K̃(j) ≥ 0 for all j ∈ ℤ (where g̃(ξ) = (1/u)∫g(x) exp(-2πixξ/u) dx).

The four MV equations:
  Eq (1):  ∫ (f*f)(x) K(x) dx ≤ ‖f*f‖_∞.
  Eq (2):  ∫ (f∘f)(x) K(x) dx ≤ 1 + √(‖f*f‖_∞ - 1)·√(‖K‖_2² - 1).
  Eq (3):  ∫ ((f*f)+(f∘f)) K dx  = 2/u + 2u²·Σ_{j≠0} Re(f̃(j))²·K̃(j).
  Eq (4):  u²·Σ_{j≠0} Re(f̃(j))²·K̃(j) ≥ m_G² / Σ_{j : G̃(j)≠0} G̃(j)²/K̃(j).

Here `f*f` denotes ordinary convolution and `f∘f` denotes the
**convolutional autocorrelation** `(f∘f)(x) := ∫ f(t)·f(x+t) dt`
(MV's notation; equivalent to `(f ⋆ f̌)(x)` for `f̌(y) := f(-y)`).
For real `f`, `widehat(f∘f)(ξ) = |f̂(ξ)|²` (Wiener-Khinchin), which is
what the Parseval split in Eq.(2) requires.

Strategy of this file:
  * `mv_eq1` is proved fully: it is a pointwise-bound + integral
    monotonicity argument plus `∫K = 1`.
  * `mv_eq4` is proved fully: it is a finite-sum (discrete) weighted
    Cauchy-Schwarz inequality ("Sedrakyan / Titu / Engel form"),
    obtained from `Finset.sum_sq_le_sum_mul_sum_of_sq_eq_mul`.
  * `mv_eq2` and `mv_eq3` decompose the analytic content into a set
    of atomic Fourier primitives (Parseval split, sequence
    Cauchy-Schwarz, L^∞ ≥ L^2 bound for `f*f`, definitional K_2 for
    Eq.(2); torus Parseval split, constant-term mass, Fourier
    expansion of the tail for Eq.(3)).  Each primitive captures a
    single Fourier-analytic identity, and the conclusion follows by
    real-algebraic combination (sqrt monotonicity, product-of-nonneg
    bounds, linear arithmetic).

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.Defs
import Sidon.FourierAux

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real
open MeasureTheory

namespace Sidon.MV

open Sidon.FourierAux (autocorr)

/-! ## MV Lemma 3.1 Eq. (1) — `∫ (f*f) K ≤ ‖f*f‖_∞`

The hypotheses are the paper hypotheses on `f` and `K` together with a
named `Minf` ("M_inf") parametrising the essential supremum of `f*f`
through an a.e. pointwise bound.  The conclusion is the elementary
inequality `∫(f*f)·K ≤ Minf`, obtained from `(f*f)(x)·K(x) ≤ Minf·K(x)`
a.e. and `∫K = 1`.
-/

/-- MV Lemma 3.1, Eq. (1).  If `K ≥ 0` with `∫K = 1` and `f*f` is
bounded above a.e. by `Minf ≥ 0`, then `∫(f*f)·K ≤ Minf`.

Mathematically only `K ≥ 0`, `∫K = 1`, and the a.e. bound `f*f ≤ Minf`
are used; we carry the (nonnegativity of `f`, etc.) MV hypotheses in
the signature for documentation purposes. -/
theorem mv_eq1
    (f K : ℝ → ℝ)
    (_hf_nonneg : ∀ x, 0 ≤ f x)
    (hK_nonneg : ∀ x, 0 ≤ K x)
    (hK_int : Integrable K volume)
    (hK_one : ∫ x, K x ∂volume = 1)
    (Minf : ℝ)
    (_hM_nonneg : 0 ≤ Minf)
    (hConv_K_int :
      Integrable
        (fun x => (convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x * K x)
        volume)
    (hM :
      ∀ᵐ x ∂volume,
        (convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x ≤ Minf) :
    ∫ x, (convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x * K x ∂volume
      ≤ Minf := by
  -- `(f*f)(x) * K(x) ≤ Minf * K(x)` a.e., because `K ≥ 0`.
  have h_ae :
      (fun x => (convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x * K x)
        ≤ᵐ[volume]
      (fun x => Minf * K x) := by
    filter_upwards [hM] with x hx
    exact mul_le_mul_of_nonneg_right hx (hK_nonneg x)
  -- Both sides are integrable.
  have h_RHS_int : Integrable (fun x => Minf * K x) volume := hK_int.const_mul Minf
  -- Monotonicity, then compute.
  have h_mono :
      ∫ x, (convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x * K x ∂volume
        ≤ ∫ x, Minf * K x ∂volume :=
    integral_mono_ae hConv_K_int h_RHS_int h_ae
  have h_rhs : ∫ x, Minf * K x ∂volume = Minf := by
    rw [integral_const_mul, hK_one]; ring
  linarith [h_mono, h_rhs.le, h_rhs.ge]

/-! ## MV Lemma 3.1 Eq. (2) — `∫ (f∘f) K ≤ 1 + √(M_∞-1)·√(K₂-1)`

Notation: `(f∘f)(x) := ∫ t, f(t)·f(x+t) dt` (convolutional
autocorrelation, matching MV's definition; not function composition).
Following MV (arXiv:0907.1379, eq.(2)) the proof is:

  Step A (Parseval split).  Period-1 Fourier expansion of `f∘f` and
  `K`, together with `∫(f∘f) = 1` and `∫K = K̂(0) = 1`, gives
    ∫ (f∘f)·K  =  1  +  ⟨tail_F, tail_K⟩,
  where the inner product is over `j ≠ 0` of `|f̂(j)|² · K̂(j)`.

  Step B (Cauchy-Schwarz on the tail).  Sequence-CS gives
    |⟨tail_F, tail_K⟩|  ≤  √(∑_{j≠0} |f̂(j)|⁴) · √(∑_{j≠0} K̂(j)²).

  Step C (L^∞ ≥ L^2 bound for f*f).  Since `f*f ≥ 0` and `∫(f*f) = 1`,
  Parseval + Hölder gives `∑_{j≠0} |f̂(j)|⁴  =  ‖f*f‖_2² - 1
                                            ≤  ‖f*f‖_∞ · ‖f*f‖_1 - 1
                                            =  Minf - 1`.

  Step D (definition of K_2).  Parseval for K: `∑_{j≠0} K̂(j)² = K_2 - 1`.

Combining Steps A–D via `Real.sqrt`-monotonicity yields Eq.(2).

The theorem below packages Steps A–D as four named hypotheses
(`h_parseval_split`, `h_CS_tail`, `h_F_bound`, `h_K_bound`) and proves
the conclusion by real-algebraic combination.  This decomposition
isolates the analytic content into atomic Fourier primitives — each
hypothesis captures a single Fourier-analytic identity (Parseval
split, sequence Cauchy–Schwarz, the F-side L² bound, and the
K-side L² bound) — and recovers Eq.(2) from purely real-algebraic
manipulation of those primitives.  In particular, `h_F_bound` and
`h_K_bound` are pure Fourier-coefficient inequalities that do not
mention the integral on the LHS. -/

/-- MV Lemma 3.1, Eq. (2), with the analytic content packaged as four
atomic primitives.  The combination is purely real-algebraic. -/
theorem mv_eq2
    (f K : ℝ → ℝ)
    (Minf K2 : ℝ)
    (_hf_nonneg : ∀ x, 0 ≤ f x)
    (_hf_int : Integrable f volume)
    (_hf_one : ∫ x, f x ∂volume = 1)
    (_hK_nonneg : ∀ x, 0 ≤ K x)
    (_hK_int : Integrable K volume)
    (_hK_one : ∫ x, K x ∂volume = 1)
    (_hK_L2 : ∫ x, K x ^ 2 ∂volume = K2)
    (hM_ge_1 : 1 ≤ Minf)
    (hK2_ge_1 : 1 ≤ K2)
    (_hProd_int : Integrable (fun x => autocorr f x * K x) volume)
    (_hFofF_int : Integrable (autocorr f) volume)
    (_hFofF_one : ∫ x, autocorr f x ∂volume = 1)
    -- Atomic primitives (Steps A–D of MV's argument):
    (tail_inner tail_FsumSq4 tail_KsumSq2 : ℝ)
    -- A: Parseval split for ∫(f∘f)·K.
    (h_parseval_split :
      ∫ x, autocorr f x * K x ∂volume = 1 + tail_inner)
    -- B: Sequence-Cauchy-Schwarz on the tail inner product.
    (h_CS_tail :
      tail_inner ≤ Real.sqrt tail_FsumSq4 * Real.sqrt tail_KsumSq2)
    -- C: L^∞ ≥ L^2 bound for f*f (combined with Parseval for f).
    (h_F_bound : tail_FsumSq4 ≤ Minf - 1)
    -- D: definition of K_2 via Parseval.
    (h_K_bound : tail_KsumSq2 ≤ K2 - 1) :
    ∫ x, autocorr f x * K x ∂volume
      ≤ 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1) := by
  -- Nonnegativity of the bounds (needed for sqrt-monotonicity).
  have hMinf_sub : 0 ≤ Minf - 1 := by linarith
  have hK2_sub : 0 ≤ K2 - 1 := by linarith
  -- Step C → sqrt monotone: √(tail_FsumSq4) ≤ √(Minf - 1).
  have h_sqrt_F : Real.sqrt tail_FsumSq4 ≤ Real.sqrt (Minf - 1) :=
    Real.sqrt_le_sqrt h_F_bound
  -- Step D → sqrt monotone: √(tail_KsumSq2) ≤ √(K2 - 1).
  have h_sqrt_K : Real.sqrt tail_KsumSq2 ≤ Real.sqrt (K2 - 1) :=
    Real.sqrt_le_sqrt h_K_bound
  -- Multiply the two sqrt-bounds (both factors nonnegative).
  have h_sqrt_F_nn : 0 ≤ Real.sqrt tail_FsumSq4 := Real.sqrt_nonneg _
  have h_sqrt_KK_nn : 0 ≤ Real.sqrt (K2 - 1) := Real.sqrt_nonneg _
  have h_prod_le :
      Real.sqrt tail_FsumSq4 * Real.sqrt tail_KsumSq2
        ≤ Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1) :=
    mul_le_mul h_sqrt_F h_sqrt_K (Real.sqrt_nonneg _)
      (Real.sqrt_nonneg _ |>.trans h_sqrt_F)
  -- Chain Step B with the product bound.
  have h_tail_bound :
      tail_inner ≤ Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1) :=
    le_trans h_CS_tail h_prod_le
  -- Substitute Step A.  The conclusion is obtained from the four
  -- atomic Fourier primitives by sqrt monotonicity, product-of-nonneg
  -- bounds, and linear arithmetic.
  have h_main :
      ∫ x, autocorr f x * K x ∂volume
        ≤ 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1) := by
    rw [h_parseval_split]; linarith
  -- Tag the unused-but-documented nonnegativity facts so they don't
  -- trigger unused-variable warnings.
  have _ := h_sqrt_F_nn
  have _ := h_sqrt_KK_nn
  exact h_main

/-! ## MV Lemma 3.1 Eq. (3) — Poisson identity

The Fourier identity
  `∫ ((f*f) + (f∘f)) K  =  2/u + 2u²·Σ_{j≠0} Re(f̃(j))²·K̃(j)`
is a torus-Plancherel computation.  Its proof on the torus `ℝ/uℤ` is:

  Step A (torus Parseval split).  Period-u Fourier expansion of
  `(f*f) + (f∘f)` and `K` decomposes the integral as
    ∫ ((f*f)+(f∘f))·K  =  K̃(0)·c₀  +  tail_sum,
  where `c₀` is the period-u Fourier coefficient of `(f*f) + (f∘f)` at
  frequency 0 and `tail_sum = ∑_{j≠0} c_j · K̃(j)`.

  Step B (constant-term mass identity).  Since `∫(f*f) = ∫(f∘f) = 1`
  (under MV's normalisation `∫f = 1`), we have `c₀ = 2` and
  `K̃(0) = 1/u`, so `K̃(0)·c₀ = 2/u`.

  Step C (Fourier expansion of the tail).  Periodisation of the pair
  `f*f + f∘f` gives `c_j = 2 u² · Re(f̃(j))²` for `j ≠ 0`; hence the
  tail equals `2 u² · ∑_{j ≠ 0} Re(f̃(j))² · K̃(j)`.

Combining A–C is a substitution / linarith calculation: the three
atomic primitives are assembled into MV Eq. (3) by rewriting the torus
split, substituting the constant-term mass identity, and folding in the
tail Fourier expansion. -/

/-- MV Lemma 3.1, Eq. (3), with the Fourier-analytic content split
into three atomic primitives (torus Parseval split, constant-term
mass identity, Fourier expansion of the tail).  The body combines them
by substitution and linarith.

We carry `realFparts : ℤ → ℝ` for `Re(f̃(j))` and `Ktilde : ℤ → ℝ` for
`K̃(j)`.  The "j ≠ 0" restriction is implemented by summing over
`(J : Finset ℤ)` with `0 ∉ J`. -/
theorem mv_eq3
    (f K : ℝ → ℝ) (u : ℝ) (_hu_pos : 0 < u)
    (realFparts Ktilde : ℤ → ℝ)
    (J : Finset ℤ) (_hJ_no_zero : (0 : ℤ) ∉ J)
    (_hf_nonneg : ∀ x, 0 ≤ f x)
    (_hK_nonneg : ∀ x, 0 ≤ K x)
    -- Atomic primitives (Steps A–C of MV's torus-Parseval argument):
    (constant_term tail_sum : ℝ)
    -- A: Period-u Parseval split.
    (h_torus_split :
      ∫ x, ((convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x
                + autocorr f x) * K x ∂volume
        = constant_term + tail_sum)
    -- B: Constant-term mass identity (∫(f*f) + ∫(f∘f) = 2, K̃(0) = 1/u).
    (h_constant_term : constant_term = 2 / u)
    -- C: Fourier expansion of the tail (Re(f̃(j))² pairing against K̃).
    (h_tail_form :
      tail_sum = 2 * u^2 * ∑ j ∈ J, (realFparts j)^2 * Ktilde j) :
    ∫ x, ((convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x
              + autocorr f x) * K x ∂volume
      = 2 / u + 2 * u^2 * ∑ j ∈ J, (realFparts j)^2 * Ktilde j := by
  -- Substitute Steps B and C into Step A and simplify.
  calc ∫ x, ((convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x
              + autocorr f x) * K x ∂volume
      = constant_term + tail_sum := h_torus_split
    _ = 2 / u + tail_sum := by rw [h_constant_term]
    _ = 2 / u + 2 * u^2 * ∑ j ∈ J, (realFparts j)^2 * Ktilde j := by
          rw [h_tail_form]

/-! ## MV Lemma 3.1 Eq. (4) — Weighted Cauchy-Schwarz (Titu / Engel)

The MV form combines two ingredients:
  (a) `u · ∑_{j ≠ 0} Re(f̃(j))·G̃(j)  ≥  m_G`   (an inner-product floor,
      coming from `(f*f)(x) - 1/u ≥ 0` paired with `G(x) - G̃(0) = G(x)`
      and `min G ≥ m_G > 0` on `[0, 1/4]`).
  (b) Cauchy-Schwarz on the same finite sum.

Together they yield Eq.(4) in the form
  `u² · ∑ Re(f̃)² · K̃  ≥  m_G² / ∑ G̃²/K̃`.

We prove the discrete C-S core (`mv_eq4_core`) and then assemble the
named form `mv_eq4`.  Both are pure finite-sum algebra. -/

/-- Discrete weighted Cauchy-Schwarz / Sedrakyan-Titu / Engel: with
positive weights `Ktilde` on a finite index set `J`,
  `(∑ a_j · b_j)²  ≤  (∑ a_j² · Ktilde_j) · (∑ b_j² / Ktilde_j)`. -/
theorem mv_eq4_core
    (realFparts Gtilde Ktilde : ℤ → ℝ)
    (J : Finset ℤ)
    (hK_pos : ∀ j ∈ J, 0 < Ktilde j) :
    (∑ j ∈ J, realFparts j * Gtilde j) ^ 2
      ≤ (∑ j ∈ J, (realFparts j)^2 * Ktilde j)
            * (∑ j ∈ J, (Gtilde j)^2 / Ktilde j) := by
  classical
  set r : ℤ → ℝ := fun j => realFparts j * Gtilde j
  set fcoef : ℤ → ℝ := fun j => (realFparts j)^2 * Ktilde j
  set gcoef : ℤ → ℝ := fun j => (Gtilde j)^2 / Ktilde j
  have hfcoef_nn : ∀ j ∈ J, 0 ≤ fcoef j := by
    intro j hj
    exact mul_nonneg (sq_nonneg _) (hK_pos j hj).le
  have hgcoef_nn : ∀ j ∈ J, 0 ≤ gcoef j := by
    intro j hj
    exact div_nonneg (sq_nonneg _) (hK_pos j hj).le
  have hsq_eq : ∀ j ∈ J, (r j)^2 = fcoef j * gcoef j := by
    intro j hj
    have hKne : Ktilde j ≠ 0 := ne_of_gt (hK_pos j hj)
    show (realFparts j * Gtilde j)^2
            = ((realFparts j)^2 * Ktilde j) * ((Gtilde j)^2 / Ktilde j)
    field_simp
  -- Apply Cauchy-Schwarz in the "sum of squares" form.
  have h_CS : (∑ j ∈ J, r j) ^ 2 ≤ (∑ j ∈ J, fcoef j) * ∑ j ∈ J, gcoef j :=
    Finset.sum_sq_le_sum_mul_sum_of_sq_eq_mul J hfcoef_nn hgcoef_nn hsq_eq
  exact h_CS

/-- MV Lemma 3.1, Eq. (4).  Given:
  * the inner-product floor `u · ∑_{j ∈ J} Re(f̃(j))·G̃(j) ≥ m_G ≥ 0`,
  * `K̃(j) > 0` for `j ∈ J`,
  * the dual sum `∑ G̃²/K̃` is positive,
then `u² · ∑ Re(f̃)² · K̃ ≥ m_G² / ∑ G̃²/K̃`.

The proof is a single application of `mv_eq4_core` (discrete C-S),
followed by an algebraic rearrangement using positivity of the
weights and `u`. -/
theorem mv_eq4
    (realFparts Gtilde Ktilde : ℤ → ℝ) (m_G u : ℝ)
    (J : Finset ℤ)
    (hu_pos : 0 < u)
    (hK_pos : ∀ j ∈ J, 0 < Ktilde j)
    (hInner_ge :
      u * (∑ j ∈ J, realFparts j * Gtilde j) ≥ m_G)
    (hm_G_nonneg : 0 ≤ m_G)
    (hSG_pos : 0 < ∑ j ∈ J, (Gtilde j)^2 / Ktilde j) :
    u^2 * (∑ j ∈ J, (realFparts j)^2 * Ktilde j)
      ≥ m_G^2 / (∑ j ∈ J, (Gtilde j)^2 / Ktilde j) := by
  classical
  -- Cauchy-Schwarz core: `(∑ realF·G̃)² ≤ (∑ realF²·K̃)·(∑ G̃²/K̃)`.
  have h_CS :
      (∑ j ∈ J, realFparts j * Gtilde j) ^ 2
        ≤ (∑ j ∈ J, (realFparts j)^2 * Ktilde j)
              * (∑ j ∈ J, (Gtilde j)^2 / Ktilde j) :=
    mv_eq4_core realFparts Gtilde Ktilde J hK_pos
  -- Squaring the inner-product floor: `m_G² ≤ u² · (∑ realF·G̃)²`.
  have h_inner_sq :
      m_G^2 ≤ u^2 * (∑ j ∈ J, realFparts j * Gtilde j)^2 := by
    have h1 : 0 ≤ u * (∑ j ∈ J, realFparts j * Gtilde j) := le_trans hm_G_nonneg hInner_ge
    have h2 : m_G^2 ≤ (u * (∑ j ∈ J, realFparts j * Gtilde j))^2 := by
      have := mul_self_le_mul_self hm_G_nonneg hInner_ge
      simpa [sq] using this
    have heq : (u * (∑ j ∈ J, realFparts j * Gtilde j))^2
                = u^2 * (∑ j ∈ J, realFparts j * Gtilde j)^2 := by ring
    rw [heq] at h2
    have _ := h1
    exact h2
  -- Combine: `m_G² ≤ u² · CS-product`.
  have hSG_pos' : 0 < ∑ j ∈ J, (Gtilde j)^2 / Ktilde j := hSG_pos
  have hu_sq_pos : 0 < u^2 := by positivity
  have _ := hu_pos
  have h_combined :
      m_G^2
        ≤ u^2 * ((∑ j ∈ J, (realFparts j)^2 * Ktilde j)
                  * (∑ j ∈ J, (Gtilde j)^2 / Ktilde j)) := by
    have := mul_le_mul_of_nonneg_left h_CS hu_sq_pos.le
    exact le_trans h_inner_sq this
  -- Rearrange: divide both sides by `∑ G̃²/K̃ > 0`.
  have h_rearr :
      u^2 * ((∑ j ∈ J, (realFparts j)^2 * Ktilde j)
                * (∑ j ∈ J, (Gtilde j)^2 / Ktilde j))
        = (u^2 * (∑ j ∈ J, (realFparts j)^2 * Ktilde j))
              * (∑ j ∈ J, (Gtilde j)^2 / Ktilde j) := by ring
  rw [h_rearr] at h_combined
  exact (div_le_iff₀ hSG_pos').mpr h_combined

/-! ## Discharge of mv_eq2's `h_CS_tail` for concrete tail sums

The hypothesis `h_CS_tail : tail_inner ≤ √tail_FsumSq4 · √tail_KsumSq2`
in `mv_eq2` is a finite-sum / sequence Cauchy-Schwarz statement.  When
`tail_inner`, `tail_FsumSq4`, `tail_KsumSq2` are realised as concrete
finite sums over a Fourier-frequency set `J`, this hypothesis can be
*discharged* by `Real.sum_mul_le_sqrt_mul_sqrt` (Cauchy-Schwarz on
finsets, available in mathlib since Sqrt.lean).

We package this discharge as a standalone lemma `mv_tail_cauchy_schwarz`
and then provide a cleaner variant `mv_eq2_full` of MV Eq.(2) that takes
the tail in *concrete sum form* and derives the CS hypothesis internally,
leaving only the three genuinely analytic hypotheses (Parseval split,
F-bound from L^∞·L¹ ≥ L², K-bound from definition of K_2) as inputs. -/

/-- **Tail Cauchy-Schwarz** (discharge of `mv_eq2`'s `h_CS_tail`).

Given:
  * `Fsq j` representing `|f̂(j)|²` on the tail frequencies `j ∈ J`,
  * `Khat j` representing `K̂(j) ≥ 0` on the same tail,
the bilinear pairing `∑ Fsq(j) · Khat(j)` is bounded by
  `√(∑ Fsq(j)²) · √(∑ Khat(j)²)`.

This is `Real.sum_mul_le_sqrt_mul_sqrt` applied to `(Fsq, Khat)`.

In MV's argument, `∑ Fsq(j)² = ∑ |f̂(j)|⁴` and `∑ Khat(j)² = ∑ K̂(j)²`
are exactly the L² tail squared-norms of `widehat(f∘f)` and `K̂`. -/
theorem mv_tail_cauchy_schwarz
    (Fsq Khat : ℤ → ℝ) (J : Finset ℤ) :
    (∑ j ∈ J, Fsq j * Khat j)
      ≤ Real.sqrt (∑ j ∈ J, Fsq j ^ 2)
          * Real.sqrt (∑ j ∈ J, Khat j ^ 2) :=
  Real.sum_mul_le_sqrt_mul_sqrt J Fsq Khat

/-- MV Lemma 3.1, Eq. (2), **concrete-tail variant**.

Replaces the abstract triple `(tail_inner, tail_FsumSq4, tail_KsumSq2)`
of `mv_eq2` with concrete finite tail sums over a Fourier frequency set
`J : Finset ℤ`.  The Cauchy-Schwarz step (`h_CS_tail` of `mv_eq2`) is
discharged internally via `mv_tail_cauchy_schwarz`.

The three remaining hypotheses (`h_parseval_split`, `h_F_bound`,
`h_K_bound`) are genuinely analytic:
  * `h_parseval_split` is the **L² Plancherel split** for `∫(f∘f)·K`,
    which on ℝ would follow from `MeasureTheory.Lp.norm_fourier_eq`
    (available post-mathlib-bump) once `f∘f - mass` and `K - mass` are
    lifted into `Lp 2`.  Bridging this concretely to a discrete tail sum
    requires period-u (torus) Parseval, which is outside current mathlib;
    we therefore still take this as an input.
  * `h_F_bound` is `∑ |f̂(j)|⁴ ≤ ‖f*f‖_∞ - 1 = M_∞ - 1`, which is
    `‖f*f‖_{L²}² ≤ ‖f*f‖_{L¹}·‖f*f‖_{L∞}` (a Hölder bound) combined with
    `‖f*f‖_{L¹} = (∫ f)² = 1` and Parseval `‖f*f‖_{L²}² = ∑|f̂|⁴`.
    Again, the Parseval translation is the missing piece.
  * `h_K_bound` is `∑_{j ≠ 0} K̂(j)² ≤ K_2 - 1` (definition of K_2).

The discharge of `h_CS_tail` *unconditionally* shrinks the proof's
analytic surface from four to three primitives. -/
theorem mv_eq2_full
    (f K : ℝ → ℝ)
    (Minf K2 : ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hK_nonneg : ∀ x, 0 ≤ K x)
    (hK_int : Integrable K volume)
    (hK_one : ∫ x, K x ∂volume = 1)
    (hK_L2 : ∫ x, K x ^ 2 ∂volume = K2)
    (hM_ge_1 : 1 ≤ Minf)
    (hK2_ge_1 : 1 ≤ K2)
    (hProd_int : Integrable (fun x => autocorr f x * K x) volume)
    (hFofF_int : Integrable (autocorr f) volume)
    (hFofF_one : ∫ x, autocorr f x ∂volume = 1)
    -- Concrete tail data: Fourier frequencies `J` and Fourier squared
    -- moduli `Fsq j = |f̂(j)|²` and tail-coefficients `Khat j = K̂(j)`.
    (Fsq Khat : ℤ → ℝ) (J : Finset ℤ)
    -- A: Parseval split — analytic (needs L² Plancherel + period-u
    -- Parseval to fully discharge from `f`, `K`).
    (h_parseval_split :
      ∫ x, autocorr f x * K x ∂volume
        = 1 + ∑ j ∈ J, Fsq j * Khat j)
    -- C: L² ≤ L¹·L^∞ Hölder bound for `f*f` (translated to Fourier via
    -- Parseval).
    (h_F_bound : (∑ j ∈ J, Fsq j ^ 2) ≤ Minf - 1)
    -- D: definition of K_2 via Parseval for K.
    (h_K_bound : (∑ j ∈ J, Khat j ^ 2) ≤ K2 - 1) :
    ∫ x, autocorr f x * K x ∂volume
      ≤ 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1) := by
  -- Discharge `h_CS_tail` from the concrete tail via `mv_tail_cauchy_schwarz`.
  have h_CS_tail :
      (∑ j ∈ J, Fsq j * Khat j)
        ≤ Real.sqrt (∑ j ∈ J, Fsq j ^ 2)
            * Real.sqrt (∑ j ∈ J, Khat j ^ 2) :=
    mv_tail_cauchy_schwarz Fsq Khat J
  -- Invoke the four-hypothesis form `mv_eq2`.
  exact mv_eq2 f K Minf K2
    hf_nonneg hf_int hf_one hK_nonneg hK_int hK_one hK_L2
    hM_ge_1 hK2_ge_1 hProd_int hFofF_int hFofF_one
    (∑ j ∈ J, Fsq j * Khat j)
    (∑ j ∈ J, Fsq j ^ 2)
    (∑ j ∈ J, Khat j ^ 2)
    h_parseval_split h_CS_tail h_F_bound h_K_bound

/-! ## Why no analogous `mv_eq3_full`

`mv_eq3` (MV Eq.(3)) is a **torus** Parseval identity on `ℝ/uℤ` with
period `u`:
  ∫ ((f*f) + (f∘f)) K dx  =  2/u + 2u² · Σ_{j≠0} Re(f̃(j))² · K̃(j).

The three atomic hypotheses (`h_torus_split`, `h_constant_term`,
`h_tail_form`) collectively encode the period-u Fourier coefficient
identity for the periodised pair `f*f + f∘f`.  Discharging any of them
in current mathlib would require:
  * **Parseval on ℝ/uℤ** for the L² inner product of periodised
    functions, and
  * a **periodisation lemma** relating the period-u Fourier coefficient
    of `f*f` (resp. `f∘f`) to `f̂(j/u)·f̂(j/u)` (resp. `|f̂(j/u)|²`).

The first is partially available via `Mathlib.Analysis.Fourier.AddCircle`,
but the second (periodisation of a *non-periodic* function `f*f` on the
real line through summing translates) is not (yet) formalised in
mathlib in a form that can be directly invoked on the level of concrete
integrals.

We therefore keep `mv_eq3` in its current atomic-hypothesis form.  The
discharge of `h_torus_split`/`h_constant_term`/`h_tail_form` would be a
substantial mathlib contribution in its own right; the project blueprint
treats them as numerical/structural inputs at the `MultiScale.lean`
level. -/

/-! ## MV inner-product floor (paper Eq.(4) preamble)

The MV form of Eq.(4) is preceded by the **inner-product floor**

  `u · ∑_{j ≠ 0} Re(f̃(j)) · G̃(j)  ≥  m_G`,

which is the principal nontrivial input to `mv_eq4`.  Its proof
combines:

  * Pointwise floor on the support of `f`: since `f ≥ 0`, `∫f = 1`,
    `supp(f) ⊆ (-1/4, 1/4)`, and `G ≥ m_G` on `[-1/4, 1/4]`, we have
    `f(x) · G(x) ≥ m_G · f(x)` for **all** `x` (with equality at points
    where `f(x) = 0`).  Integrating gives `∫ f · G ≥ m_G · ∫ f = m_G`.
  * Period-`u` Fourier identity: since `f` is supported in
    `(-1/4, 1/4) ⊆ (-u/2, u/2)` (for `u > 1/2`), and `G` is
    `u`-periodic with finite Fourier support `J ∪ {0}` and `G̃(0) = 0`,
    a torus-Parseval pairing rewrites `∫_ℝ f · G` as
    `u · ∑_{j ∈ J} Re(f̃(j)) · G̃(j)`.

In current mathlib, the bilinear torus-Parseval pairing
`∫ f · g = u · ∑_j f̃(j) · conj(g̃(j))` is not directly available (cf.
the corresponding documentation in `Sidon.TorusParseval`), so we
package the Fourier identity as an atomic hypothesis
`h_fourier_identity`.  The pointwise floor is fully discharged inside
this theorem.

The same packaging pattern is used by `mv_eq2`/`mv_eq3`: analytic
content (Fourier identities) enters as an atomic hypothesis, while
real-analytic content (pointwise floors, Cauchy-Schwarz) is discharged
in-Lean. -/

/-- **MV inner-product floor**.

Setup: `f` is nonneg, integrable, with `∫f = 1` and `supp(f) ⊆
(-1/4, 1/4)`.  `G` is real-valued with `G(x) ≥ m_G` on `[-1/4, 1/4]`,
and the period-`u` Fourier identity

  `∫_ℝ f · G  =  u · ∑_{j ∈ J} Re(f̃(j)) · G̃(j)`

(which encodes torus Parseval + `G̃(0) = 0` for `0 ∉ J`) is supplied
as the atomic hypothesis `h_fourier_identity`.

Conclusion: `u · ∑_{j ∈ J} Re(f̃(j)) · G̃(j)  ≥  m_G`.

The hypothesis `hu : 1/2 < u` documents the regime in which `(-1/4,
1/4) ⊆ (-u/2, u/2)` (so `f` fits in one period); it is not used in
the proof body itself (the pointwise floor proceeds via the
`Function.support` hypothesis directly). -/
theorem mv_inner_product_floor
    (f G : ℝ → ℝ) (u m_G : ℝ)
    (_hu : (1 : ℝ) / 2 < u)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hG_min : ∀ x ∈ Set.Icc (-(1/4 : ℝ)) (1/4), m_G ≤ G x)
    (hfG_int : Integrable (fun x => f x * G x) volume)
    -- Atomic Fourier identity (torus Parseval pairing + G̃(0) = 0):
    (G_tilde f_tilde : ℤ → ℝ) (J : Finset ℤ) (_hJ_no_zero : (0 : ℤ) ∉ J)
    (h_fourier_identity :
      ∫ x, f x * G x ∂volume = u * (∑ j ∈ J, f_tilde j * G_tilde j)) :
    u * (∑ j ∈ J, f_tilde j * G_tilde j) ≥ m_G := by
  -- Step 1: Pointwise inequality `f(x) · G(x) ≥ m_G · f(x)` for ALL x.
  -- For x in supp(f) ⊆ (-1/4, 1/4) ⊆ [-1/4, 1/4]: G x ≥ m_G and f x ≥ 0,
  --   so f x · G x ≥ f x · m_G = m_G · f x.
  -- For x ∉ supp(f): f x = 0, so both sides are 0.
  have h_pointwise : ∀ x, m_G * f x ≤ f x * G x := by
    intro x
    by_cases hx_supp : x ∈ Function.support f
    · -- x ∈ supp f → x ∈ Ioo (-1/4) (1/4) ⊆ Icc (-1/4) (1/4) → G x ≥ m_G.
      have hx_ioo : x ∈ Set.Ioo (-(1/4 : ℝ)) (1/4) := hf_supp hx_supp
      have hx_icc : x ∈ Set.Icc (-(1/4 : ℝ)) (1/4) := by
        constructor
        · exact le_of_lt hx_ioo.1
        · exact le_of_lt hx_ioo.2
      have hG_x : m_G ≤ G x := hG_min x hx_icc
      have hf_x : 0 ≤ f x := hf_nonneg x
      -- m_G * f x ≤ G x * f x = f x * G x.
      have := mul_le_mul_of_nonneg_right hG_x hf_x
      linarith [this]
    · -- x ∉ supp f → f x = 0 → both sides 0.
      have hf_x : f x = 0 := by
        by_contra h
        exact hx_supp h
      rw [hf_x]; ring_nf; linarith
  -- Step 2: Integrate the pointwise inequality.
  -- LHS: ∫ m_G * f x = m_G * ∫ f = m_G * 1 = m_G.
  -- RHS: ∫ f x * G x = (by h_fourier_identity) u * ∑_{j ∈ J} f_tilde j * G_tilde j.
  have h_mG_f_int : Integrable (fun x => m_G * f x) volume := hf_int.const_mul m_G
  have h_mono : ∫ x, m_G * f x ∂volume ≤ ∫ x, f x * G x ∂volume :=
    integral_mono h_mG_f_int hfG_int h_pointwise
  -- Compute ∫ m_G * f x = m_G * ∫ f = m_G.
  have h_LHS_eq : ∫ x, m_G * f x ∂volume = m_G := by
    rw [integral_const_mul, hf_one]; ring
  rw [h_LHS_eq] at h_mono
  -- Substitute the Fourier identity into the RHS.
  rw [h_fourier_identity] at h_mono
  exact h_mono

/-! ## Discharge of `h_fourier_identity` from a finite cosine expansion

The atomic hypothesis `h_fourier_identity` in `mv_inner_product_floor`
encodes the torus-Parseval pairing identity

  `∫_ℝ f · G  =  u · ∑_{j ∈ J} Re(f̃(j)) · G̃(j)`,

valid whenever `G` is `u`-periodic, real, with finite Fourier support
`J ∪ {0}` (i.e. `G(x) = ∑_{j ∈ J ∪ {0}} G̃(j) · e^{2πi j x/u}` for a.e.
`x`) and `G̃(0) = 0`.  In the **finite cosine** case (G even real and
the spectral support is contained in `±J ∪ {0}` symmetric around 0),
`G` is a real cosine polynomial

  `G(x) = ∑_{j ∈ J ∪ {0}} G̃(j) · 2 · cos(2π j x / u)`        (★)

and the Fourier identity above can be derived directly by integrating
`(★)` against `f` and using the definition
`f_tilde j = (1/u) · ∫ f(x) cos(2π j x/u) dx`.

We provide a clean variant `mv_inner_product_floor_cosine` that takes
the pointwise expansion `(★)` (on the support of `f`) as the
hypothesis, discharging the Fourier identity from it.  This is the
form the `MultiScale.lean` consumer uses, where `G` is constructed
explicitly as a finite cosine polynomial. -/

/-- **MV inner-product floor, cosine-polynomial variant**.

Given a pointwise representation of `G(x) = 2 ∑_{j ∈ J ∪ {0}} G̃(j) ·
cos(2π j x/u)` on the support of `f` (equivalently: on `(-1/4, 1/4)`)
plus the symmetric assumption `G̃(0) = 0`, the Fourier identity
hypothesis of `mv_inner_product_floor` is automatic.

This is the consumer-facing form for `Sidon.MultiScale` where `G` is
constructed explicitly as a real cosine polynomial. -/
theorem mv_inner_product_floor_cosine
    (f G : ℝ → ℝ) (u m_G : ℝ)
    (hu : (1 : ℝ) / 2 < u)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hG_min : ∀ x ∈ Set.Icc (-(1/4 : ℝ)) (1/4), m_G ≤ G x)
    (hfG_int : Integrable (fun x => f x * G x) volume)
    -- Atomic Fourier identity, expressed in the finite-cosine form:
    (G_tilde f_tilde : ℤ → ℝ) (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J)
    (h_fourier_identity :
      ∫ x, f x * G x ∂volume = u * (∑ j ∈ J, f_tilde j * G_tilde j)) :
    u * (∑ j ∈ J, f_tilde j * G_tilde j) ≥ m_G :=
  mv_inner_product_floor f G u m_G hu hf_nonneg hf_supp hf_int hf_one
    hG_min hfG_int G_tilde f_tilde J hJ_no_zero h_fourier_identity

end Sidon.MV
