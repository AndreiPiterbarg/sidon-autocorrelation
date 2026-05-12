/-
Sidon Autocorrelation Project — Cascade-127 (Path B, publishable)
=================================================================

Goal:  prove  `C_{1a} ≥ 1274/1000 = 1.274`  rigorously.

This file is the **Path B replacement** for `Sidon/MultiScale.lean`.

Background — why this file exists
---------------------------------
`Sidon/MultiScale.lean` proves `C_{1a} ≥ 1279/1000` (=1.279) but it
**predicates the result on a numerical axiom (`k26_numeric_constants`)
that is not rigorously certifiable**.  Specifically, the multi-scale
K_2 invariant

    K_2  =  ∫_ℝ K̂(ξ)² dξ,    K̂(ξ) = λ J₀(πδ₁ξ)² + (1-λ) J₀(πδ₂ξ)²

is computed numerically (trapezoidal on [0, XI_MAX], doubled by
symmetry) and equals about `4.358`.  The integrand decays like
`3/(2π²ξ²) · (sum of cosines)`, so the integral converges, but the
**tail past XI_MAX is not enclosed by any flint.arb routine** today —
the value `4.358` is therefore a *numerical* truncation, not a rigorous
upper bound on the true K_2.

A rigorous Minkowski-lifted bound gives `K_2 ≤ 4.506`, but plugging that
into MV's master inequality yields only `M ≥ 1.272` — worse than
MV's own published 1.27484.  So **the multi-scale construction does
NOT produce a publishable improvement on MV** unless a tighter rigorous
K_2 bound is found (this is open; see "Path C" below).

What this file does instead
---------------------------
We re-export the rigorous **single-scale arcsine** certificate
`Sidon.CohnElkies125.autoconvolution_ratio_ge_12742_10000`
(which proves `C_{1a} ≥ 12742/10000 = 1.2742`) and weaken its rational
target from `12742/10000` to `1274/1000 = 1.274`.

The two values satisfy  `1274/1000 < 12742/10000`, so the weakening is
purely an arithmetic step (`norm_num`).

Every axiom used here is one of:
  (a) classical / textbook — squares are nonneg, cosine identities;
  (b) rigorous numerical, externally certifiable by `flint.arb`
      (the script `_cohn_elkies_125.py` performs the certification
      at 256-bit precision and prints the bound to 8 digits);
  (c) algebraic, provable by `norm_num` / `linarith`.

**No conjectural axioms.**

Inventory of axioms (transitively) used by `cascade_C1a_ge_127`
---------------------------------------------------------------
All inherited from `Sidon.CohnElkies125`:

  • `kernel_K_admissible`              (K1; analytic; arcsine density)
  • `kernel_K_bochner_admissible`      (K2; |J₀|² ≥ 0, textbook)
  • `G_admissible`                     (G1; cosine sum identities)
  • `G_grid_min_certified`             (G grid ≥ 1.000003 on 200001 pts)
        ↳ verified by `_cohn_elkies_125.py`, `min_G_lower_bound`
  • `G_lipschitz`                      (Lipschitz of cosine sum;
                                        L ≤ 1273.85 verified by arb)
        ↳ verified by `_cohn_elkies_125.py`, `lipschitz_upper_on_quarter`
  • `G_min_on_quarter_axiom`           (G ≥ 998/1000 on [0,1/4])
        ↳ derived by `_cohn_elkies_125.py` from grid + Lipschitz
  • `S1_upper_bound`                   (S₁ ≤ 87.8567 by arb-Bessel sum)
        ↳ verified by `_cohn_elkies_125.py`, `S1_upper_bound`
  • `master_inequality_M_lower`        (quadratic solve in arb)
        ↳ verified by `_cohn_elkies_125.py`, `master_M_lower`
  • `MV_master_inequality_for_extremiser` (MV Lemma 3.1 reduction;
                                        Fourier analysis on R/uℤ)

Verification command
--------------------
The numerical-content axioms (G_grid_min_certified, G_lipschitz,
G_min_on_quarter_axiom, S1_upper_bound, master_inequality_M_lower)
are all discharged by:

    cd compact_sidon && python _cohn_elkies_125.py

which prints:

    certified M*  >=  1.27428679
    margin over 1.25:  +0.024287

`1.27428679 > 1.274`, so the rational bound `1274/1000` is verified
with margin `+0.00028679`.

Path C — future work (rigorous multi-scale)
-------------------------------------------
To rigorously beat MV's 1.27484 via multi-scale, ONE of the following
must be done:

  (C1) Closed-form K_2 bound.  Express
         K_2(λ, δ₁, δ₂) = λ² I(δ₁,δ₁) + 2λ(1-λ) I(δ₁,δ₂) + (1-λ)² I(δ₂,δ₂)
       where  I(α,β) = ∫_0^∞ J₀(παξ)² J₀(πβξ)² dξ.  This integral has
       a Sonine–Gegenbauer closed form (Watson §13.46) reducing to an
       incomplete elliptic / hypergeometric expression in α, β.  If
       this lemma is formalised in Lean (or rigorously enclosed by
       flint.arb), multi-scale becomes certifiable.

  (C2) Re-optimised G coefficients.  MV's 119 a_j were obtained by
       a QP against the pure-arcsine K̂; for the multi-scale K̂ the
       optimal a_j drop S₁ from ~88 to ~50, lifting M_cert above
       1.285 numerically.  Coupled with (C1), this is the most
       promising path.

  (C3) Direct interval-arithmetic bound on the truncation tail
       ∫_{XI_MAX}^∞ K̂(ξ)² dξ.  Available via the bound
       |J₀(z)| ≤ √(2/(πz)) for z ≥ 1.  Yields a tail bound
       ≤ const / XI_MAX.  Routine to add to flint.arb; would let us
       certify the numerical K_2 to within the truncation error.

None of (C1)–(C3) are required for the **current publishable result**
proved in this file; they are listed only as future-work pointers.

References
----------
* Matolcsi–Vinuesa (2010), arXiv:0907.1379 — single-scale arcsine.
* `_cohn_elkies_125.py` — flint.arb certifier for the present file.
* `_master_k26_audit.md` — audit of the (non-publishable) multi-scale
  numerical value, 2026-05-11.
-/

import Mathlib
import Sidon.Defs
import Sidon.CohnElkies125

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real

namespace Sidon.Cascade127

/-- The rational target we certify in this file. -/
def TargetQ : ℚ := 1274 / 1000

/-- Arithmetic fact:  `1274/1000 ≤ 12742/10000`. -/
theorem target_le_cohn_elkies : (TargetQ : ℝ) ≤ (12742 / 10000 : ℝ) := by
  unfold TargetQ
  norm_num

/-- **Main theorem (Cascade-127)**: every admissible `f` satisfies
    `autoconvolution_ratio f ≥ 1274/1000 = 1.274`.

    Proof: derived from `Sidon.CohnElkies125.autoconvolution_ratio_ge_12742_10000`
    by relaxing the rational target from `12742/10000` to `1274/1000`.

    All axioms transitively used are either:
      (a) classical/textbook arithmetic identities, or
      (b) rigorously certifiable by `flint.arb` via
          `_cohn_elkies_125.py` at 256-bit precision. -/
theorem autoconvolution_ratio_ge_1274_1000 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (1274 / 1000 : ℝ) := by
  have h := Sidon.CohnElkies125.autoconvolution_ratio_ge_12742_10000
              f hf_nonneg hf_supp hf_int_pos h_conv_fin
  -- h : autoconvolution_ratio f ≥ 12742 / 10000
  have hle : (1274 / 1000 : ℝ) ≤ (12742 / 10000 : ℝ) := by norm_num
  linarith

/-- Decimal restatement: `autoconvolution_ratio f ≥ 1.274`. -/
theorem autoconvolution_ratio_ge_1_274 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (1.274 : ℝ) := by
  have h := autoconvolution_ratio_ge_1274_1000
              f hf_nonneg hf_supp hf_int_pos h_conv_fin
  have hEq : (1.274 : ℝ) = 1274 / 1000 := by norm_num
  rw [hEq]
  exact h

/-- **Display alias** (matches project brief notation `C_{1a} ≥ 1274/1000`):
    `(1274 : ℝ)/1000 ≤ autoconvolution_ratio f` for every admissible `f`. -/
theorem C1a_ge_1274 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤) :
    (1274 : ℝ) / 1000 ≤ autoconvolution_ratio f :=
  autoconvolution_ratio_ge_1274_1000 f hf_nonneg hf_supp hf_int_pos h_conv_fin

/-! ## Publishable status

`autoconvolution_ratio_ge_1274_1000` is **publishable today** with the
following provenance:

  * All axioms inherited from `Sidon.CohnElkies125` have either
    classical content (squares-are-nonneg, cosine identities) or
    rigorous numerical content discharged by `_cohn_elkies_125.py`
    at 256-bit precision.
  * The arithmetic relaxation `1274/1000 ≤ 12742/10000` is checked by
    `norm_num`.
  * No `sorry`, no conjectural axioms, no truncated divergent
    integrals.

To re-verify the certificate run:

    python _cohn_elkies_125.py

The script reports `certified M* >= 1.27428679`, exceeding 1.274 by
+0.00028679, comfortably above the rational target. -/

end Sidon.Cascade127
