/-
Sidon Autocorrelation Project — MV Lemma 3.1 Eq.(4) (hEq4)
==========================================================

This module discharges MV Lemma 3.1 Eq.(4) — the discrete weighted
Cauchy-Schwarz / Sedrakyan-Titu identity — for **admissible** test
functions `f` paired with a finite real cosine polynomial `G`.

Concretely:

  * `lattice_pairing` — for `f : ℝ → ℝ` integrable and supported in
    `Ioo (-1/4) (1/4)`, and `G(x) = ∑_{j ∈ J} G̃(j) · 2 · cos(2π j x / u)`
    a real cosine polynomial whose Fourier support `J` avoids `0` and is
    symmetric in `j`, the lattice pairing identity holds:

      ∫_ℝ f(x) · G(x) dx  =  u · ∑_{j ∈ J} f_tilde_real(j) · G̃(j),

    where `f_tilde_real(j) := (2/u) · ∫ f(x) · cos(2π j x / u) dx` is the
    real period-u Fourier coefficient of `f`.

  * `K_ms_fourier_lattice_nonneg` — the Bochner positivity of the
    multi-scale kernel at lattice frequencies `K̂_ms(j/u) ≥ 0`, expressed
    as a convex combination of squared Bessel terms
    `λᵢ · J₀(πδᵢ · j/u)² ≥ 0`.

  * `hEq4_discharge` — the MV Eq.(4) bound `u² · S_cos ≥ m_G² / S_G`
    for the admissible `f`, derived from `Sidon.MV.mv_eq4` with the
    inner-product floor discharged via `Sidon.MV.mv_inner_product_floor`
    + `lattice_pairing`.

This file introduces **no axioms** beyond the two numerical kernel-
specific axioms already in `Sidon.MultiScale`
(`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`).

References:
  * `Sidon.MV.mv_eq4` and `Sidon.MV.mv_inner_product_floor` for the MV
    Eq.(4) skeleton.
  * `Sidon.MultiScale.{uQ_real, lambda1, lambda2, lambda3, delta1,
    delta2, delta3}` for the kernel constants.
  * `Sidon.Bessel.besselJ0` for the real Bessel function `J₀`.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.Defs
import Sidon.Bessel
import Sidon.MVLemmas
import Sidon.MultiScale

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real
open MeasureTheory

namespace Sidon.BundleEq4

/-! ## Real period-u Fourier coefficient (cosine part)

For `f : ℝ → ℝ` integrable, the real period-u Fourier coefficient at
lattice index `j` is

  `f_tilde_real(j) := (2/u) · ∫ f(x) · cos(2π j x / u) dx`,

with `u := Sidon.MultiScale.uQ_real`. This is the convention chosen so
that

  `∫ f(x) · 2 · cos(2π j x / u) dx  =  u · f_tilde_real(j)`,

matching the MV `h_fourier_identity` hypothesis exactly. -/

/-- The real period-u Fourier coefficient of `f` at lattice frequency
`j/u`: `f_tilde_real(j) := (2/u) · ∫ f(x) · cos(2π j x / u) dx`. -/
noncomputable def f_tilde_real (f : ℝ → ℝ) (j : ℤ) : ℝ :=
  (2 / Sidon.MultiScale.uQ_real) *
    ∫ x, f x * Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real) ∂volume

/-! ## Lattice pairing identity

Given a real cosine polynomial `G(x) = ∑_{j ∈ J} G̃(j) · 2 · cos(2π j x / u)`
with `G̃(0) = 0` (encoded by `0 ∉ J`) and real symmetry of `G̃`, plus an
integrable `f` supported in the open quarter interval, we have

  `∫_ℝ f · G  =  u · ∑_{j ∈ J} f_tilde_real(j) · G̃(j)`. -/

/-- Integrability of `f(x) · cos(α·x)` for integrable `f`. -/
private lemma integrable_f_mul_cos (f : ℝ → ℝ) (hf : Integrable f volume) (α : ℝ) :
    Integrable (fun x => f x * Real.cos (α * x)) volume := by
  -- `f` is integrable and `cos(α·x)` is bounded by `1`, so the product is integrable.
  refine hf.mul_bdd (c := 1)
    (Real.continuous_cos.comp (continuous_const.mul continuous_id)).aestronglyMeasurable
    ?_
  refine Filter.Eventually.of_forall (fun x => ?_)
  have := Real.abs_cos_le_one (α * x)
  simpa [Real.norm_eq_abs] using this

/-- Integrability of `f(x) · cos(α·x + β)` for integrable `f`, in the
form needed for the lattice pairing (the cos argument has the shape
`2π·j·x/u`). -/
private lemma integrable_f_mul_cos_lattice (f : ℝ → ℝ) (hf : Integrable f volume)
    (j : ℤ) :
    Integrable
      (fun x => f x * Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))
      volume := by
  have h_rewrite : (fun x : ℝ => f x * Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))
      = (fun x : ℝ => f x * Real.cos ((2 * Real.pi * (j : ℝ) / Sidon.MultiScale.uQ_real) * x)) := by
    funext x
    congr 2
    ring
  rw [h_rewrite]
  exact integrable_f_mul_cos f hf (2 * Real.pi * (j : ℝ) / Sidon.MultiScale.uQ_real)

/-- **Lattice pairing identity** for a finite real cosine polynomial.

Given:
  * `f : ℝ → ℝ` integrable, supported in `Ioo (-1/4) (1/4)`,
  * `G̃ : ℤ → ℝ`, `J : Finset ℤ` with `0 ∉ J` and `J` symmetric (`-j ∈ J`
    whenever `j ∈ J`), and `G̃` real-symmetric (`G̃(-j) = G̃(j)`),

the integral pairing
`∫_ℝ f(x) · (∑_{j ∈ J} G̃(j) · 2 · cos(2π j x / u)) dx`
equals `u · ∑_{j ∈ J} f_tilde_real(j) · G̃(j)`.

The proof is a direct expansion: distribute the integral over the finite
sum (using `MeasureTheory.integral_finset_sum`), evaluate each term as
`G̃(j) · 2 · ∫ f(x)·cos(...) dx = G̃(j) · u · f_tilde_real(j)`, and
collect. -/
theorem lattice_pairing
    (f : ℝ → ℝ) (hf_int : Integrable f volume)
    (_hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (G_tilde : ℤ → ℝ) (J : Finset ℤ)
    (_hJ_no_zero : (0 : ℤ) ∉ J) (_hJ_sym : ∀ j ∈ J, -j ∈ J)
    (_hG_real_sym : ∀ j, G_tilde (-j) = G_tilde j) :
    ∫ x, f x * (∑ j ∈ J, G_tilde j * 2 *
        Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real)) ∂volume
      = Sidon.MultiScale.uQ_real *
          ∑ j ∈ J, (f_tilde_real f j) * G_tilde j := by
  classical
  have hu_pos : 0 < Sidon.MultiScale.uQ_real := Sidon.MultiScale.uQ_real_pos
  have hu_ne : Sidon.MultiScale.uQ_real ≠ 0 := ne_of_gt hu_pos
  -- Step 1: rewrite the integrand as a finite sum and use linearity of `∫`.
  have h_dist :
      (fun x : ℝ => f x * (∑ j ∈ J, G_tilde j * 2 *
            Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real)))
        = (fun x : ℝ => ∑ j ∈ J, f x * (G_tilde j * 2 *
            Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))) := by
    funext x
    rw [Finset.mul_sum]
  rw [h_dist]
  -- Step 2: integrability of each summand.
  have h_summand_int : ∀ j ∈ J, Integrable
      (fun x : ℝ => f x * (G_tilde j * 2 *
          Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))) volume := by
    intro j _hj
    -- f(x) · (G̃(j) · 2 · cos(...)) = G̃(j) · 2 · (f(x) · cos(...)).
    have h_eq : (fun x : ℝ => f x * (G_tilde j * 2 *
                  Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real)))
                = (fun x : ℝ => (G_tilde j * 2) *
                  (f x * Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))) := by
      funext x; ring
    rw [h_eq]
    exact (integrable_f_mul_cos_lattice f hf_int j).const_mul (G_tilde j * 2)
  -- Step 3: distribute the integral over the finset sum.
  rw [MeasureTheory.integral_finset_sum (μ := volume) J h_summand_int]
  -- Step 4: evaluate each integral.
  --   ∫ x, f x * (G̃(j) · 2 · cos(...)) dx
  --     = (G̃(j) · 2) · ∫ x, f x · cos(...) dx
  --     = G̃(j) · u · (2/u) · ∫ f · cos
  --     = G̃(j) · u · f_tilde_real(j)
  --     = u · (f_tilde_real(j) · G̃(j))
  have h_each : ∀ j ∈ J,
      ∫ x, f x * (G_tilde j * 2 *
          Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real)) ∂volume
        = Sidon.MultiScale.uQ_real * (f_tilde_real f j * G_tilde j) := by
    intro j _hj
    -- Pull constants out of the integral.
    have h_eq : (fun x : ℝ => f x * (G_tilde j * 2 *
                  Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real)))
                = (fun x : ℝ => (G_tilde j * 2) *
                  (f x * Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))) := by
      funext x; ring
    rw [h_eq]
    rw [MeasureTheory.integral_const_mul]
    -- Now goal: (G̃(j) · 2) · ∫ f · cos = u · (f_tilde_real(j) · G̃(j)).
    -- Unfold f_tilde_real and clear denominator.
    show (G_tilde j * 2) *
          (∫ x, f x *
            Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real) ∂volume)
          = Sidon.MultiScale.uQ_real * (f_tilde_real f j * G_tilde j)
    unfold f_tilde_real
    field_simp
  -- Step 5: substitute each integral and pull `u` out.
  rw [Finset.sum_congr rfl h_each]
  rw [← Finset.mul_sum]

/-! ## Bochner positivity of `K̂_ms(j/u)` at lattice frequencies

The multi-scale arcsine kernel `K_ms = λ₁ · K_arc(δ₁) + λ₂ · K_arc(δ₂) +
λ₃ · K_arc(δ₃)` has Fourier transform

  `K̂_ms(ξ) = λ₁ · J₀(πδ₁ξ)² + λ₂ · J₀(πδ₂ξ)² + λ₃ · J₀(πδ₃ξ)²`,

which is `≥ 0` at every frequency by Bochner positivity (each summand is
a nonnegative scalar times the square of a real Bessel function).

We define the value at lattice points `K_ms_fourier_lattice j :=
K̂_ms(j/u)` symbolically as the sum of `λᵢ · J₀(πδᵢ · j/u)²`, and prove
nonnegativity directly.

For strict positivity we rely on the certifier output: the QP-optimised
cosine `G` is chosen on a frequency set `J` for which the certifier
verifies `K̂_ms(j/u) > 0`; this is consumed downstream as a hypothesis
parameter. -/

/-- The Bessel-squared evaluation `J₀(π δᵢ · j/u)²` for the i-th scale,
expressed via `Sidon.Bessel.besselJ0`. -/
noncomputable def J0_sq (δ : ℝ) (j : ℤ) : ℝ :=
  (Sidon.Bessel.besselJ0 (Real.pi * δ * (j / Sidon.MultiScale.uQ_real))) ^ 2

/-- The kernel's lattice Fourier value, as the convex combination of
squared Bessel terms:
`K_ms_fourier_lattice j := λ₁ · J₀(πδ₁ · j/u)² + λ₂ · J₀(πδ₂ · j/u)² +
                           λ₃ · J₀(πδ₃ · j/u)²`. -/
noncomputable def K_ms_fourier_lattice (j : ℤ) : ℝ :=
  Sidon.MultiScale.lambda1 * J0_sq Sidon.MultiScale.delta1 j
    + Sidon.MultiScale.lambda2 * J0_sq Sidon.MultiScale.delta2 j
    + Sidon.MultiScale.lambda3 * J0_sq Sidon.MultiScale.delta3 j

/-- Each squared Bessel term is nonnegative. -/
lemma J0_sq_nonneg (δ : ℝ) (j : ℤ) : 0 ≤ J0_sq δ j := by
  unfold J0_sq
  exact sq_nonneg _

/-- **Bochner positivity at lattice frequencies**: `K̂_ms(j/u) ≥ 0`.

Each summand `λᵢ · J₀(πδᵢ · j/u)²` is the product of a nonnegative
weight `λᵢ ≥ 0` and a nonnegative squared Bessel value `J₀(πδᵢ · j/u)²
≥ 0`. -/
theorem K_ms_fourier_lattice_nonneg (j : ℤ) :
    0 ≤ K_ms_fourier_lattice j := by
  unfold K_ms_fourier_lattice
  have hlams := Sidon.MultiScale.lambdas_nonneg
  have hl1 : 0 ≤ Sidon.MultiScale.lambda1 := by
    unfold Sidon.MultiScale.lambda1; exact_mod_cast hlams.1
  have hl2 : 0 ≤ Sidon.MultiScale.lambda2 := by
    unfold Sidon.MultiScale.lambda2; exact_mod_cast hlams.2.1
  have hl3 : 0 ≤ Sidon.MultiScale.lambda3 := by
    unfold Sidon.MultiScale.lambda3; exact_mod_cast hlams.2.2
  have h1 : 0 ≤ Sidon.MultiScale.lambda1 * J0_sq Sidon.MultiScale.delta1 j :=
    mul_nonneg hl1 (J0_sq_nonneg _ _)
  have h2 : 0 ≤ Sidon.MultiScale.lambda2 * J0_sq Sidon.MultiScale.delta2 j :=
    mul_nonneg hl2 (J0_sq_nonneg _ _)
  have h3 : 0 ≤ Sidon.MultiScale.lambda3 * J0_sq Sidon.MultiScale.delta3 j :=
    mul_nonneg hl3 (J0_sq_nonneg _ _)
  linarith

/-! ## Bundle constants `m_G_const` and `S_G_const`

For the MV Eq.(4) bound we need a concrete minimum `m_G` of the cosine
`G` on `[0, 1/4]` and a concrete dual sum `S_G = ∑ G̃²/K̂_ms`.

These constants are *defined* here as the slack-rational floors from
`Sidon.MultiScale`; their justification is the certifier's arb-interval
verification (paper Lemmas 4.3-4.5).  Downstream consumers may pin these
to the actual certifier outputs; for the internal purposes of this file
we only need them to be positive reals. -/

/-- The slack-rational floor on `min_{x ∈ [0, 1/4]} G(x)`, real-coerced.
This equals `Sidon.MultiScale.minGLowerQ = 998/1000` and serves as the
`m_G` parameter in the MV Eq.(4) bound. -/
noncomputable def m_G_const : ℝ := (Sidon.MultiScale.minGLowerQ : ℝ)

/-- The bundle's `S_G` denominator: a fixed positive real, chosen here
as a placeholder slack rational; downstream consumers may pin this to
the actual certifier-reported value `S_1 = ∑ G̃(j)²/K̂_ms(j/u)` for the
QP-optimised cosine `G`.

For the internal interface of this file, only `S_G_const > 0` is
required. -/
noncomputable def S_G_const : ℝ := (Sidon.MultiScale.S1UpperQ : ℝ)

/-- `m_G_const > 0` (it equals `998/1000`). -/
theorem m_G_const_pos : 0 < m_G_const := by
  unfold m_G_const
  unfold Sidon.MultiScale.minGLowerQ
  push_cast
  norm_num

/-- `m_G_const ≥ 0`. -/
theorem m_G_const_nonneg : 0 ≤ m_G_const := le_of_lt m_G_const_pos

/-- `S_G_const > 0` (it equals `29841/1000`). -/
theorem S_G_const_pos : 0 < S_G_const := by
  unfold S_G_const
  unfold Sidon.MultiScale.S1UpperQ
  push_cast
  norm_num

/-! ## Bundle-level `S_cos` definition

`S_cos f := ∑_{j ∈ J} (Re f̃(j))² · K̂_ms(j/u)` for the QP-optimised
frequency set `J`; the analytic floor `u² · S_cos ≥ m_G² / S_G` is what
MV Eq.(4) provides.

We define `S_cos f J` as a concrete finite sum over a parametric
finite frequency set `J : Finset ℤ`; the downstream consumer pins `J`
to the certifier's frequency set. -/

/-- The bilinear cosine sum `S_cos := ∑_{j ∈ J} f_tilde_real(j)² ·
K̂_ms(j/u)`, parametrised by a finite frequency set `J`. -/
noncomputable def S_cos (f : ℝ → ℝ) (J : Finset ℤ) : ℝ :=
  ∑ j ∈ J, (f_tilde_real f j) ^ 2 * K_ms_fourier_lattice j

/-! ## `hEq4_discharge`

The MV Eq.(4) bound `u² · S_cos ≥ m_G² / S_G` is the consumer-facing
form of `Sidon.MV.mv_eq4` after the inner-product floor has been
discharged via `Sidon.MV.mv_inner_product_floor` + `lattice_pairing`.

The hypotheses are the standard MV admissibility (`f ≥ 0`, support,
∫f = 1) plus:
  * The cosine polynomial expansion of `G` on `[0, 1/4]` (so the
    pointwise floor `G ≥ m_G` is meaningful).
  * Strict positivity `K̂_ms(j/u) > 0` on `J` (consumed as a hypothesis,
    discharged externally by the certifier output).
  * Positivity of `S_G_const`. -/

/-- **hEq4 discharge** for an admissible `f` paired with a finite
real cosine polynomial `G` on the QP-frequency set `J`.

Setup:
  * `f : ℝ → ℝ` nonneg, integrable, `∫f = 1`, supported in `Ioo (-1/4) (1/4)`.
  * `G(x) = ∑_{j ∈ J} G̃(j) · 2 · cos(2π j x / u)` on the support of `f`
    (and indeed pointwise everywhere).
  * `G̃(0)` excluded from the sum (`0 ∉ J`).
  * `G̃` real-symmetric, `J` symmetric in `j ↦ -j`.
  * `G ≥ m_G_const` on `[-1/4, 1/4]`.
  * `K̂_ms(j/u) > 0` for every `j ∈ J` (certifier output).

Conclusion: `u² · S_cos f J ≥ m_G_const² / S_G_const`, where
`S_G_const > 0` is taken to be the dual sum at the certifier's
arb-verified upper bound on `∑ G̃²/K̂_ms`. -/
theorem hEq4_discharge
    (f : ℝ → ℝ) (G : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (G_tilde : ℤ → ℝ) (J : Finset ℤ)
    (hJ_no_zero : (0 : ℤ) ∉ J) (hJ_sym : ∀ j ∈ J, -j ∈ J)
    (hG_real_sym : ∀ j, G_tilde (-j) = G_tilde j)
    (hG_cos_expansion :
      ∀ x, G x = ∑ j ∈ J, G_tilde j * 2 *
                  Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))
    (hG_min : ∀ x ∈ Set.Icc (-(1/4 : ℝ)) (1/4), m_G_const ≤ G x)
    (hfG_int : Integrable (fun x => f x * G x) volume)
    -- Strict positivity of K̂_ms at the QP frequencies (certifier output):
    (hK_pos : ∀ j ∈ J, 0 < K_ms_fourier_lattice j)
    -- The bundle's S_G_const is identified with the dual sum ∑ G̃²/K̂_ms on J:
    (hSG_eq : S_G_const = ∑ j ∈ J, (G_tilde j) ^ 2 / K_ms_fourier_lattice j) :
    Sidon.MultiScale.uQ_real ^ 2 * S_cos f J ≥ m_G_const ^ 2 / S_G_const := by
  -- Step A: Discharge `h_fourier_identity` via `lattice_pairing`.
  have h_fourier_identity :
      ∫ x, f x * G x ∂volume
        = Sidon.MultiScale.uQ_real *
          (∑ j ∈ J, (f_tilde_real f j) * G_tilde j) := by
    -- Replace `G x` by its cosine expansion and apply `lattice_pairing`.
    have h_integrand_eq :
        (fun x => f x * G x)
          = (fun x => f x * (∑ j ∈ J, G_tilde j * 2 *
              Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))) := by
      funext x
      rw [hG_cos_expansion x]
    rw [show (∫ x, f x * G x ∂volume)
          = ∫ x, f x * (∑ j ∈ J, G_tilde j * 2 *
              Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real)) ∂volume
        from by rw [h_integrand_eq]]
    exact lattice_pairing f hf_int hf_supp G_tilde J hJ_no_zero hJ_sym hG_real_sym
  -- Step B: Apply `mv_inner_product_floor` to obtain the inner-product floor.
  have h_inner_floor :
      Sidon.MultiScale.uQ_real *
        (∑ j ∈ J, (f_tilde_real f j) * G_tilde j) ≥ m_G_const :=
    Sidon.MV.mv_inner_product_floor f G Sidon.MultiScale.uQ_real m_G_const
      Sidon.MultiScale.uQ_real_gt_half
      hf_nonneg hf_supp hf_int hf_one
      hG_min hfG_int
      G_tilde (f_tilde_real f) J hJ_no_zero
      h_fourier_identity
  -- Step C: Apply `mv_eq4` to derive the discrete C-S bound.
  have h_eq4 :
      Sidon.MultiScale.uQ_real ^ 2 *
        (∑ j ∈ J, (f_tilde_real f j) ^ 2 * K_ms_fourier_lattice j)
        ≥ m_G_const ^ 2 /
          (∑ j ∈ J, (G_tilde j) ^ 2 / K_ms_fourier_lattice j) := by
    -- We need positivity of `∑ G̃²/K̂_ms` to apply `mv_eq4`.
    have hSG_pos_sum :
        0 < ∑ j ∈ J, (G_tilde j) ^ 2 / K_ms_fourier_lattice j := by
      rw [← hSG_eq]
      exact S_G_const_pos
    exact Sidon.MV.mv_eq4 (f_tilde_real f) G_tilde K_ms_fourier_lattice
      m_G_const Sidon.MultiScale.uQ_real J
      Sidon.MultiScale.uQ_real_pos
      hK_pos h_inner_floor m_G_const_nonneg hSG_pos_sum
  -- Step D: Repackage via `S_cos` and `S_G_const`.
  unfold S_cos
  rw [hSG_eq]
  exact h_eq4

/-! ## Convenience: parameter-free m_G² / S_G floor

When the cosine polynomial `G` happens to attain the bundle's
`m_G_const` and the dual sum equals `S_G_const`, the hEq4 conclusion
collapses to the standard MV form `u²·S_cos ≥ m_G²/S_G`.  This is the
shape consumed by `Sidon.MultiScale.MV_master_inequality_from_MV_lemmas`
as the `hEq4` field. -/

/-- Restatement of `hEq4_discharge` in the `u²·S_cos ≥ m_G²/S_G` form
that consumers expect, parametrised by the **bundle constants**
`m_G_const`, `S_G_const`. -/
theorem hEq4_consumer_form
    (f : ℝ → ℝ) (G : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (G_tilde : ℤ → ℝ) (J : Finset ℤ)
    (hJ_no_zero : (0 : ℤ) ∉ J) (hJ_sym : ∀ j ∈ J, -j ∈ J)
    (hG_real_sym : ∀ j, G_tilde (-j) = G_tilde j)
    (hG_cos_expansion :
      ∀ x, G x = ∑ j ∈ J, G_tilde j * 2 *
                  Real.cos (2 * Real.pi * j * x / Sidon.MultiScale.uQ_real))
    (hG_min : ∀ x ∈ Set.Icc (-(1/4 : ℝ)) (1/4), m_G_const ≤ G x)
    (hfG_int : Integrable (fun x => f x * G x) volume)
    (hK_pos : ∀ j ∈ J, 0 < K_ms_fourier_lattice j)
    (hSG_eq : S_G_const = ∑ j ∈ J, (G_tilde j) ^ 2 / K_ms_fourier_lattice j) :
    Sidon.MultiScale.uQ_real ^ 2 * S_cos f J ≥ m_G_const ^ 2 / S_G_const :=
  hEq4_discharge f G hf_nonneg hf_supp hf_int hf_one
    G_tilde J hJ_no_zero hJ_sym hG_real_sym hG_cos_expansion hG_min hfG_int
    hK_pos hSG_eq

end Sidon.BundleEq4
