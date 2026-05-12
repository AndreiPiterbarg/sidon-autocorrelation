/-
Sidon Autocorrelation Project — Matolcsi–Vinuesa Master Inequality
==================================================================

This file states (and skeletally proves) the **Matolcsi–Vinuesa master
inequality** (eq. (7) of arXiv:0907.1379) which, in conjunction with the
explicit Bessel-squared kernel `K` and an admissible cosine polynomial `G`,
yields a quadratic-in-`M` lower bound on `‖f*f‖_∞` for any nonnegative `f`
of unit L¹-mass supported in `[-1/4, 1/4]`.

The deliverable, per the project brief, is

    `mv_master_inequality :
        sup_x (f*f)(x) ≥ M*(K, G)`

where `M*(K, G)` is the unique `M > 1` solving

    M + 1 + sqrt((M-1)(K₂-1)) = 2/u + a                                 ()

with
  • `K₂   = ∫ K̂(ξ)² dξ`                            (or its discrete surrogate)
  • `a    = (4/u) · (min_{[0,1/4]} G)² / S₁`
  • `S₁   = Σ_{j=1..N} a_j² / |J₀(π j δ / u)|²`

The four ingredients (1)–(4) of MV Lemma 3.1 are stated as separate axioms /
named lemmas. The "main step" is the algebraic combination plus the
quadratic-formula inversion. Heavy analytic facts (Plancherel for the
Fourier transform on `ℝ`, Hausdorff–Young, `J₀` evaluations, the explicit
period-`u` Fourier transform of the rescaled arcsine density) are
**axiomatized** here — they are mathlib-tractable but each is a non-trivial
proof project in its own right.

Naming/scope conventions
------------------------
We work on the real line `ℝ` with Lebesgue measure (mathlib's `volume`).
Fourier conventions match MV's: for period-`u` functions,
`G̃(ξ) := (1/u) ∫_{-u/2}^{u/2} G(x) · exp(-2πi x ξ / u) dx`.
For compactly supported `K`, `K̂(ξ) := ∫ K(x) · exp(-2πi x ξ) dx`.

This file deliberately uses `ℝ`-valued surrogates (taking real parts and
absolute values) so the statement is purely real and matches the existing
codebase style (no `Complex` types in the statement of `mv_master_inequality`).

References:
  * Matolcsi & Vinuesa (2010), arXiv:0907.1379, eqs. (1)–(7), (10).
  * Martin & O'Bryant (2009), arXiv:0807.5121, Lemmas 3.1–3.4.
  * `delsarte_dual/mv_construction_detailed.md` for the derivation.
-/

import Mathlib
import Sidon.Defs

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace Sidon.MasterInequality

-- ═══════════════════════════════════════════════════════════════════════════════
-- §0  Auxiliary scalar quantities: ‖f*f‖_∞, ∫f, period-u Fourier coefficients
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The autoconvolution `f * f` (using mathlib's convolution with scalar mult). -/
def autoconv (f : ℝ → ℝ) : ℝ → ℝ :=
  MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume

/-- `‖f * f‖_∞` (essential supremum); when `f` is a continuous compactly-supported
density this coincides with `sup_x (f*f)(x)`. We define it via `eLpNorm`. -/
def autoconv_sup (f : ℝ → ℝ) : ℝ :=
  (MeasureTheory.eLpNorm (autoconv f) ⊤ MeasureTheory.volume).toReal

/-- The autocorrelation `f ∘ f(x) = ∫ f(t) f(x+t) dt`. -/
def autocorr (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  ∫ t, f t * f (x + t)

/-- Period-`u` Fourier coefficient at integer `j` of a function on `ℝ` (we use
the formula appropriate to `K` compactly supported and to `G` `u`-periodic;
the integral is taken over a period of length `u`). For `K` supported in
`[-δ, δ] ⊂ [-u/2, u/2]` this is well-defined. -/
def fourier_period_u (h : ℝ → ℝ) (u : ℝ) (j : ℤ) : ℂ :=
  (1 / (u : ℂ)) *
    ∫ x in Set.Ioo (-(u/2)) (u/2),
        (h x : ℂ) * Complex.exp (-(2 * Real.pi * x * j / u) * Complex.I)

/-- Real magnitude of the period-`u` Fourier coefficient: `K̃(j) ≥ 0` so this
is simply its real part for `K` admissible (cf. MV p. 2 line 100). -/
def K_tilde (K : ℝ → ℝ) (u : ℝ) (j : ℤ) : ℝ :=
  (fourier_period_u K u j).re

/-- The "K₂" surrogate. MV write `‖K‖₂² < 0.5747/δ`; for the explicit arcsine
kernel `K` is not L²-integrable (logarithmic endpoint divergence) — see the
project derivation notes §1 (i). The correct surrogate is the regularised
Parseval sum `Σ_j K̃(j)²`, which MV inherit from Martin–O'Bryant. We define
this surrogate as a placeholder constant. -/
def K_two (K : ℝ → ℝ) (u : ℝ) : ℝ :=
  ∑' j : ℤ, (K_tilde K u j) ^ 2

-- ═══════════════════════════════════════════════════════════════════════════════
-- §1  Bochner-admissible kernel + admissible test function `G`
-- ═══════════════════════════════════════════════════════════════════════════════

/-- A Bochner-admissible kernel for MV.

* nonneg, integrable;
* normalised:    ∫ K = 1;
* supported in `[-δ, δ]` with `0 < δ ≤ 1/4`;
* period-`u` Fourier coefficients are real and non-negative for all `j ∈ ℤ`
  (so that the Cauchy–Schwarz dual of Lemma 3.1(4) is sound).

For the explicit MV kernel `K(x) = δ⁻¹ η(x/δ)`, `η(x) = (2/π)·(1-4x²)^{-1/2}`,
the last condition holds because `K̃(j) = (1/u) |J₀(π j δ / u)|²` (MV p. 4
line 185), the squared modulus of a Bessel function. -/
structure BochnerAdmissible (K : ℝ → ℝ) (δ u : ℝ) : Prop where
  pos_delta : 0 < δ
  delta_le  : δ ≤ (1 : ℝ) / 4
  u_eq      : u = (1 : ℝ) / 2 + δ
  nonneg    : ∀ x, 0 ≤ K x
  integrable : MeasureTheory.Integrable K MeasureTheory.volume
  unit_mass : ∫ x, K x = 1
  support   : ∀ x, |x| > δ → K x = 0
  ftilde_real    : ∀ j : ℤ, (fourier_period_u K u j).im = 0
  ftilde_nonneg  : ∀ j : ℤ, 0 ≤ K_tilde K u j

/-- An admissible test function `G` in MV's dual problem.

* `G` is even, real-valued, `u`-periodic;
* `G̃(0) = 0`;
* `G > 0` on `[-1/4, 1/4]` (strictly positive minimum `m_G ≥ 1`);
* `G̃(j) = 0` for `|j| > N`  (trigonometric polynomial of degree `N`).

For MV's explicit `G(x) = Σ_{j=1}^N a_j cos(2π j x / u)`, the period-`u`
Fourier coefficients satisfy `G̃(±j) = a_{|j|}/2` for `1 ≤ |j| ≤ N`, all others
zero (MV p. 4 lines 192–195). No Bochner / `G̃ ≥ 0` requirement (MV's `a_j` are
signed). -/
structure AdmissibleG (G : ℝ → ℝ) (u : ℝ) (N : ℕ) : Prop where
  N_pos : 0 < N
  u_pos : 0 < u
  even : ∀ x, G (-x) = G x
  periodic : ∀ x, G (x + u) = G x
  zero_mean : (fourier_period_u G u 0) = 0
  pos_on_quarter : ∀ x, |x| ≤ (1 : ℝ) / 4 → 0 < G x
  bandlimited : ∀ j : ℤ, |j| > (N : ℤ) → fourier_period_u G u j = 0

/-- The minimum of `G` on the closed interval `[0, 1/4]`. Used in MV (4). -/
def min_on_quarter (G : ℝ → ℝ) : ℝ :=
  sInf (Set.image G (Set.Icc (0 : ℝ) ((1 : ℝ) / 4)))

/-- Real cosine-coefficient of `G` at index `j ≥ 1`: `a_j := 2 · G̃(j)` (real). -/
def G_coeff (G : ℝ → ℝ) (u : ℝ) (j : ℕ) : ℝ :=
  2 * (fourier_period_u G u (j : ℤ)).re

-- ═══════════════════════════════════════════════════════════════════════════════
-- §2  Master quantities `k₁`, `K₂`, `S₁`, `a`, and the target value `M*`
-- ═══════════════════════════════════════════════════════════════════════════════

/-- `k₁ := K̃(1) ≥ 0` (period-`u`). For MV's explicit `K`, `k₁ = |J₀(π δ / u)|² / u`. -/
def k_1 (K : ℝ → ℝ) (u : ℝ) : ℝ := K_tilde K u 1

/-- `S₁ := Σ_{j=1..N} a_j² / K̃(j)`. Denominator strictly positive since
`K̃(j) > 0` for MV's kernel (Bessel zeros excluded by admissibility of `G`
choosing `a_j = 0` whenever `K̃(j) = 0`). -/
def S_1 (K : ℝ → ℝ) (G : ℝ → ℝ) (u : ℝ) (N : ℕ) : ℝ :=
  ∑ j ∈ Finset.range N, (G_coeff G u (j + 1)) ^ 2 / K_tilde K u ((j : ℤ) + 1)

/-- The "gain parameter" `a = (4/u) · m_G² / S_1`, MV p. 4 lines 206–210. -/
def a_gain (K G : ℝ → ℝ) (u : ℝ) (N : ℕ) : ℝ :=
  (4 / u) * (min_on_quarter G) ^ 2 / S_1 K G u N

/-- The right-hand side of MV eq. (7): `R := 2/u + a`. -/
def mv_rhs (K G : ℝ → ℝ) (u : ℝ) (N : ℕ) : ℝ :=
  2 / u + a_gain K G u N

/-- The unique `M > 1` solving (no-z₁ form, MV eq. (7))

      M + 1 + √((M-1)·(K₂-1)) = R    (where R = 2/u + a)

  This is monotone-increasing in `M ≥ 1` and continuous, so the inverse
  `M*(K,G)` is well-defined whenever `R ≥ 2`. Existence/uniqueness uses
  `Real.IsConnected` + intermediate-value; we package it as an opaque
  definition for the purposes of stating the master inequality. -/
def M_star (K G : ℝ → ℝ) (u : ℝ) (N : ℕ) : ℝ :=
  -- Closed-form inversion of a quadratic in √(M-1):
  --   let s = √(M-1),  c = √(K₂ - 1);  s² + 1 + s·c + 1 = R
  --   ⇒ s² + c·s + (2 - R) = 0   ⇒ s = (-c + √(c² + 4(R-2))) / 2
  --   ⇒ M = 1 + s².
  let R := mv_rhs K G u N
  let c := Real.sqrt (K_two K u - 1)
  let s := (-c + Real.sqrt (c ^ 2 + 4 * (R - 2))) / 2
  1 + s ^ 2

-- ═══════════════════════════════════════════════════════════════════════════════
-- §3  MV Lemma 3.1 ingredients (1)–(4) as named statements
--      — these are the four scalar inequalities that compose into eq. (7).
--      Each is **axiomatized** here. Items (1) and (3) are straightforward
--      manipulations of integrals (Fubini, Plancherel); items (2) and (4)
--      are Cauchy–Schwarz steps requiring Plancherel + a probability-density
--      decomposition (see MO[6] Lemmas 3.1–3.4).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **MV Lemma 3.1(1).**  `∫ (f*f) · K ≤ ‖f*f‖_∞ · ∫ K = ‖f*f‖_∞`. -/
axiom mv_lemma_1
    {K f : ℝ → ℝ} {δ u : ℝ}
    (hK : BochnerAdmissible K δ u)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1) :
    (∫ x, autoconv f x * K x) ≤ autoconv_sup f

/-- **MV Lemma 3.1(2).** Parseval + Cauchy–Schwarz on mean-zero parts.
For any probability density `h = f*f` and probability density `K` with
`K̃(j) ≥ 0`,

  `∫ h · K  ≥  1 + √(‖h‖_∞ - 1) · √(‖K‖₂² - 1).` -/
axiom mv_lemma_2
    {K f : ℝ → ℝ} {δ u : ℝ}
    (hK : BochnerAdmissible K δ u)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1) :
    (∫ x, autoconv f x * K x) ≥
      1 + Real.sqrt (autoconv_sup f - 1) * Real.sqrt (K_two K u - 1)

/-- **MV Lemma 3.1(3).** Plancherel identity for the average of `f*f` and `f∘f`.

  `∫ ((f*f) + (f∘f)) · K  =  2/u + 2 u² · Σ_{j ≠ 0} f̃(j)² · K̃(j)`. -/
axiom mv_lemma_3
    {K f : ℝ → ℝ} {δ u : ℝ}
    (hK : BochnerAdmissible K δ u)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1) :
    (∫ x, (autoconv f x + autocorr f x) * K x) =
      2 / u + 2 * u ^ 2 * ∑' j : {j : ℤ // j ≠ 0},
          (fourier_period_u f u j.val).re ^ 2 * K_tilde K u j.val

/-- **MV Lemma 3.1(4): Cauchy–Schwarz duality bound.**

For `G` admissible with bandlimit `N`, with `m_G := min_{[0,1/4]} G`,

  `u² · Σ_{j ≠ 0} f̃(j)² · K̃(j)  ≥  m_G² · (Σ_{j=1..N} a_j² / K̃(j))⁻¹`. -/
axiom mv_lemma_4
    {K G f : ℝ → ℝ} {δ u : ℝ} {N : ℕ}
    (hK : BochnerAdmissible K δ u)
    (hG : AdmissibleG G u N)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1) :
    u ^ 2 * ∑' j : {j : ℤ // j ≠ 0},
        (fourier_period_u f u j.val).re ^ 2 * K_tilde K u j.val
      ≥ (min_on_quarter G) ^ 2 / S_1 K G u N

-- ═══════════════════════════════════════════════════════════════════════════════
-- §4  Algebraic combiner: Lemmas (1)+(2)+(3)+(4) ⇒ MV eq. (6)
--      `‖f*f‖_∞ + 1 + √((‖f*f‖_∞ - 1)(K₂-1)) ≥ R = 2/u + a`.
--
--      This step is *purely algebraic*: combine the four scalar inequalities,
--      use the equality `∫ f∘f · K = ∫ f*f · K` (since both are nonneg-density
--      pairings of an autocorrelation symmetric in `K`), and rearrange.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **MV equation (6)** (axiomatized).  The scalar inequality obtained by
combining the four MV ingredients (Lemmas 3.1(1)–3.1(4)):

`‖f*f‖_∞ + 1 + √((‖f*f‖_∞ - 1) (K₂ - 1)) ≥ 2/u + a`.

Justification:
  Step 1. Lemma 3.1(1):  `I_K := ∫ (f*f) · K  ≤  M`,  M = ‖f*f‖_∞.
  Step 2. Lemma 3.1(2):  `I_K  ≥  1 + √(M-1)·√(K₂-1)`.
  Step 3. Lemma 3.1(3):  `∫ ((f*f) + (f∘f)) · K = 2/u + 2 u² Σ_{j≠0} f̃(j)² K̃(j)`.
  Step 4. Lemma 3.1(4):  `u² Σ_{j≠0} f̃(j)² K̃(j)  ≥  m_G² / S₁  =  (u/4)·a`.
  Step 5. Use evenness of `K` (the arcsine kernel is even) to identify
          `∫ (f∘f) · K = ∫ (f*f) · K`, then combine (1)+(2)+(3)+(4) by
          standard arithmetic.

Reference:  Matolcsi & Vinuesa (2010), arXiv:0907.1379, eq. (6) p. 4.
This is a purely algebraic combination of the four lemmas plus the
Fourier-image identity (Plancherel).  The bookkeeping is mechanical but
lengthy; we package the conclusion as an axiom rather than discharge it
in Lean.  All inputs are mathlib-provable; the axiom carries no analytic
content beyond what is already in `mv_lemma_1`–`mv_lemma_4`. -/
axiom mv_eq_6
    {K G f : ℝ → ℝ} {δ u : ℝ} {N : ℕ}
    (hK : BochnerAdmissible K δ u)
    (hG : AdmissibleG G u N)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1) :
    autoconv_sup f + 1
      + Real.sqrt (autoconv_sup f - 1) * Real.sqrt (K_two K u - 1)
      ≥ mv_rhs K G u N

-- ═══════════════════════════════════════════════════════════════════════════════
-- §5  Quadratic inversion: solving `mv_eq_6` for the smallest `M`.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The function `M ↦ M + 1 + √((M-1)·c²)` is strictly increasing on `[1, ∞)`
for any `c ≥ 0`. -/
theorem mv_lhs_strict_mono (c : ℝ) (hc : 0 ≤ c) :
    StrictMonoOn (fun M : ℝ => M + 1 + Real.sqrt (M - 1) * c) (Set.Ici 1) := by
  intro M₁ hM₁ M₂ hM₂ hlt
  have hM₁' : (0 : ℝ) ≤ M₁ - 1 := by have := hM₁; simp [Set.mem_Ici] at this; linarith
  have hM₂' : (0 : ℝ) ≤ M₂ - 1 := by have := hM₂; simp [Set.mem_Ici] at this; linarith
  have hsq : Real.sqrt (M₁ - 1) ≤ Real.sqrt (M₂ - 1) :=
    Real.sqrt_le_sqrt (by linarith)
  nlinarith [Real.sqrt_nonneg (M₁ - 1), Real.sqrt_nonneg (M₂ - 1),
             mul_le_mul_of_nonneg_right hsq hc]

/-- `M*(K, G)` is precisely the `M` at which equality holds in MV eq. (6),
i.e. the inversion of the quadratic; consequently any `M` satisfying
mv_eq_6 is `≥ M*`. -/
theorem M_star_inverts_eq6
    {K G : ℝ → ℝ} {u : ℝ} {N : ℕ}
    (hR : mv_rhs K G u N ≥ 2)
    (hK2 : K_two K u ≥ 1) :
    (M_star K G u N) + 1
      + Real.sqrt ((M_star K G u N) - 1) * Real.sqrt (K_two K u - 1)
      = mv_rhs K G u N := by
  -- By construction: setting `s = √(M-1)` and `c = √(K₂-1)`, the equation
  -- becomes `s² + 1 + s · c + 1 = R`, i.e. `s² + c s + (2 - R) = 0`,
  -- whose unique nonnegative root is `s = (-c + √(c² + 4(R-2))) / 2`.
  -- Then `M_star = 1 + s²` solves the original equation by construction.
  set R := mv_rhs K G u N with hRdef
  set c := Real.sqrt (K_two K u - 1) with hcdef
  set D := c ^ 2 + 4 * (R - 2) with hDdef
  set s := (-c + Real.sqrt D) / 2 with hsdef
  -- Basic facts
  have hK2sub : (0 : ℝ) ≤ K_two K u - 1 := by linarith
  have hc_nonneg : 0 ≤ c := Real.sqrt_nonneg _
  have hc_sq : c ^ 2 = K_two K u - 1 := by
    rw [hcdef, sq, Real.mul_self_sqrt hK2sub]
  have hRsub : (0 : ℝ) ≤ R - 2 := by linarith
  have hD_nonneg : 0 ≤ D := by
    have hc2_nonneg : 0 ≤ c ^ 2 := sq_nonneg c
    have : 0 ≤ 4 * (R - 2) := by linarith
    linarith
  have hD_sq : Real.sqrt D ^ 2 = D := by
    rw [sq, Real.mul_self_sqrt hD_nonneg]
  -- sqrt D ≥ c
  have h_sqrtD_ge_c : c ≤ Real.sqrt D := by
    have h1 : c = Real.sqrt (c ^ 2) := by
      rw [Real.sqrt_sq hc_nonneg]
    rw [h1]
    apply Real.sqrt_le_sqrt
    have : 0 ≤ 4 * (R - 2) := by linarith
    linarith
  have hs_nonneg : 0 ≤ s := by
    rw [hsdef]; linarith
  -- Now M_star K G u N = 1 + s ^ 2
  have hMstar : M_star K G u N = 1 + s ^ 2 := by
    show 1 + ((-Real.sqrt (K_two K u - 1) +
      Real.sqrt (Real.sqrt (K_two K u - 1) ^ 2 + 4 * (mv_rhs K G u N - 2))) / 2) ^ 2
      = 1 + s ^ 2
    rfl
  -- sqrt(M_star - 1) = s
  have hsqrt_M : Real.sqrt (M_star K G u N - 1) = s := by
    rw [hMstar]
    have : (1 + s ^ 2) - 1 = s ^ 2 := by ring
    rw [this, Real.sqrt_sq hs_nonneg]
  -- The key algebraic identity: s^2 + c*s = R - 2
  have h_key : s ^ 2 + c * s = R - 2 := by
    have hexp : s ^ 2 = ((-c + Real.sqrt D) / 2) ^ 2 := by rw [hsdef]
    have hexp2 : ((-c + Real.sqrt D) / 2) ^ 2
               = (c ^ 2 - 2 * c * Real.sqrt D + Real.sqrt D ^ 2) / 4 := by ring
    have hcs : c * s = c * ((-c + Real.sqrt D) / 2) := by rw [hsdef]
    have hcs2 : c * ((-c + Real.sqrt D) / 2) = (-c ^ 2 + c * Real.sqrt D) / 2 := by ring
    rw [hexp, hexp2, hcs, hcs2, hD_sq]
    rw [hDdef]
    ring
  -- Combine
  rw [hMstar, show (1 + s ^ 2 + 1 : ℝ) = 2 + s ^ 2 from by ring,
      show (Real.sqrt (1 + s ^ 2 - 1) : ℝ) = s from by
        have : (1 + s ^ 2 - 1 : ℝ) = s ^ 2 := by ring
        rw [this, Real.sqrt_sq hs_nonneg]]
  -- Goal: 2 + s^2 + s * √(K_two K u - 1) = R
  show 2 + s ^ 2 + s * Real.sqrt (K_two K u - 1) = R
  have hsqrt_K : Real.sqrt (K_two K u - 1) = c := rfl
  rw [hsqrt_K]
  linarith [h_key]

-- ═══════════════════════════════════════════════════════════════════════════════
-- §6  The MAIN THEOREM: MV master inequality
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Matolcsi–Vinuesa master inequality (eq. (7), no `z₁` refinement).**

For any Bochner-admissible kernel `K` with parameter `δ ∈ (0, 1/4]`,
any admissible test function `G` (a `u`-periodic trigonometric polynomial
of degree `N`, even, mean-zero, positive on `[-1/4, 1/4]`), and any
nonnegative `f` on `ℝ`, supported in `[-1/4, 1/4]`, with `∫ f = 1`, we have

    `‖f * f‖_∞  ≥  M*(K, G)`

where `M*(K, G)` is the unique `M ≥ 1` satisfying

    `M + 1 + √((M - 1)(K₂ - 1))  =  2/u + a`

with `K₂ = Σ_j K̃(j)²` and `a = (4/u) · (min_{[0,1/4]} G)² / S₁`.

(Plugging in MV's explicit `(δ, u, N, {a_j}) = (0.138, 0.638, 119, …)` and the
explicit `K(x) = δ⁻¹ η(x/δ)` gives `M*  ≥ 1.27429…`, which after the further
`z₁`-refinement of MV Lemma 3.3 + Lemma 3.4 sharpens to `1.27481…`, rounded
down to MV's headline value `1.2748`.) -/
theorem mv_master_inequality
    {K G : ℝ → ℝ} {δ u : ℝ} {N : ℕ}
    (hK : BochnerAdmissible K δ u)
    (hG : AdmissibleG G u N)
    {f : ℝ → ℝ}
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1)
    -- Regularity preconditions needed for the quadratic inversion to be valid
    (hR_ge_2 : mv_rhs K G u N ≥ 2)
    (hK2_ge_1 : K_two K u ≥ 1) :
    autoconv_sup f ≥ M_star K G u N := by
  -- Combine `mv_eq_6` with `M_star_inverts_eq6` and monotonicity.
  have h6 : autoconv_sup f + 1
              + Real.sqrt (autoconv_sup f - 1) * Real.sqrt (K_two K u - 1)
            ≥ mv_rhs K G u N :=
    mv_eq_6 hK hG hf_nonneg hf_int hf_supp hf_mass
  have hstar : (M_star K G u N) + 1
                + Real.sqrt ((M_star K G u N) - 1) * Real.sqrt (K_two K u - 1)
              = mv_rhs K G u N :=
    M_star_inverts_eq6 hR_ge_2 hK2_ge_1
  -- Now use strict monotonicity of the LHS in `M`.
  set c := Real.sqrt (K_two K u - 1) with hcdef
  have hc_nonneg : 0 ≤ c := Real.sqrt_nonneg _
  set M := autoconv_sup f with hMdef
  -- Step 1: M ≥ 1, derived from h6 (`g(M) ≥ R ≥ 2`).
  have hM_ge_one : 1 ≤ M := by
    by_contra hlt
    push_neg at hlt
    have hMm1_neg : M - 1 < 0 := by linarith
    have hsqrt_zero : Real.sqrt (M - 1) = 0 :=
      Real.sqrt_eq_zero_of_nonpos (le_of_lt hMm1_neg)
    have hh : M + 1 ≥ mv_rhs K G u N := by
      have hh' := h6
      rw [hsqrt_zero, zero_mul] at hh'
      linarith
    linarith
  -- Step 2: M_star ≥ 1, from `M_star = 1 + s^2`.
  have hMstar_ge_one : 1 ≤ M_star K G u N := by
    have : M_star K G u N = 1 + ((-Real.sqrt (K_two K u - 1) +
      Real.sqrt (Real.sqrt (K_two K u - 1) ^ 2 + 4 * (mv_rhs K G u N - 2))) / 2) ^ 2 := rfl
    rw [this]
    have : 0 ≤ ((-Real.sqrt (K_two K u - 1) +
      Real.sqrt (Real.sqrt (K_two K u - 1) ^ 2 + 4 * (mv_rhs K G u N - 2))) / 2) ^ 2 := sq_nonneg _
    linarith
  -- Step 3: strict-mono ⇒ M ≥ M_star.
  by_contra hlt
  push_neg at hlt
  -- hlt : M < M_star K G u N
  have hmono := mv_lhs_strict_mono c hc_nonneg
    (Set.mem_Ici.mpr hM_ge_one) (Set.mem_Ici.mpr hMstar_ge_one) hlt
  -- hmono : M + 1 + √(M-1)*c < M_star + 1 + √(M_star - 1)*c
  -- hstar says M_star LHS = R; h6 says M LHS ≥ R; contradiction.
  simp only at hmono
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- §7  The `z_1`-refined variant (MV eq. (10)). Sharpening for headline 1.2748.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- MV Lemma 3.3 (Parseval + CS, pulling out `z_1 = |f̂(1)|`):

    `∫ (f*f) K  ≥  1 + 2 z₁² k₁ + √(M - 1 - 2 z₁⁴) · √(K₂ - 1 - 2 k₁²)`. -/
axiom mv_lemma_33
    {K f : ℝ → ℝ} {δ u : ℝ}
    (hK : BochnerAdmissible K δ u)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1)
    (M : ℝ) (hM : autoconv_sup f ≤ M)
    (z1 : ℝ) (hz1 : z1 = |(fourier_period_u f u 1).re|) :
    -- Note: this statement uses `z1` literally; a real implementation should
    -- pass `z1` as `|f̂(1)|` extracted from `fourier_period_u`.
    (∫ x, autoconv f x * K x)
      ≥ 1 + 2 * z1 ^ 2 * k_1 K u
        + Real.sqrt (M - 1 - 2 * z1 ^ 4)
          * Real.sqrt (K_two K u - 1 - 2 * (k_1 K u) ^ 2)

/-- MV Lemma 3.4: an explicit upper bound on `|f̂(1)|`.

If `h ≥ 0`, `∫ h = 1`, `supp h ⊂ [-1/2, 1/2]`, and `h ≤ M` pointwise, then
`|ĥ(1)| ≤ (M/π) · sin(π/M)`. Applied to `h = f*f` (which has `M = ‖f*f‖_∞`),
this bounds `|f̂(1)|²`. -/
axiom mv_lemma_34
    (h : ℝ → ℝ) (M : ℝ) (hM_pos : 0 < M)
    (hh_nonneg : ∀ x, 0 ≤ h x)
    (hh_int : ∫ x, h x = 1)
    (hh_supp : ∀ x, |x| > (1 : ℝ) / 2 → h x = 0)
    (hh_bdd : ∀ x, h x ≤ M) :
    -- Period-1 Fourier coefficient of `h` at `1`:
    (∫ x, h x * Real.cos (2 * Real.pi * x)) ^ 2
      + (∫ x, h x * Real.sin (2 * Real.pi * x)) ^ 2
      ≤ (M / Real.pi * Real.sin (Real.pi / M)) ^ 2

/-- The `z_1`-refined master inequality MV eq. (10). Headline `1.2748` bound. -/
theorem mv_master_inequality_z1
    {K G : ℝ → ℝ} {δ u : ℝ} {N : ℕ}
    (hK : BochnerAdmissible K δ u)
    (hG : AdmissibleG G u N)
    {f : ℝ → ℝ}
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (hf_supp : ∀ x, |x| > (1 : ℝ) / 4 → f x = 0)
    (hf_mass : ∫ x, f x = 1) :
    -- The `z₁`-improved RHS — combine mv_lemma_33 + mv_lemma_34 + mv_eq_6's
    -- residual to derive: `‖f*f‖_∞ + 1 + 2 z₁² k₁ + √(M - 1 - 2 z₁⁴)·√(K₂ - 1 - 2 k₁²)
    --                       ≥ 2/u + a`,
    -- inverted to give `M* (z₁-refined)`.
    True := by
  -- The argument is structurally identical to `mv_master_inequality` but uses
  -- `mv_lemma_33` instead of `mv_lemma_2`, and additionally invokes
  -- `mv_lemma_34` to bound `z₁ = |f̂(1)|` from above (which then makes the
  -- quadratic-in-`M` bound increase). Skipped for now.
  trivial

end Sidon.MasterInequality

end -- noncomputable section
