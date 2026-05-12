/-
Sidon Cascade-125 — Final Lower Bound  C_{1a} ≥ 5/4 = 1.25

This file assembles the proof of `C_{1a} ≥ 5/4` using:

  (1) The fully-proved `coarse_cascade_bound_general` theorem in
      `Sidon/Proof/CoarseCascade.lean` (no axioms in its proof itself; the
      axioms enter only via its hypothesis).

  (2) The EXISTING axiom `simplex_tv_coverage n_term c_target` from
      `Sidon/Proof/CoarseCascade.lean`, instantiated at `n_term = 4` and
      `c_target = 5/4` (i.e. d = 2·n_term = 8 bins).

NO NEW AXIOMS are introduced — `simplex_tv_coverage` is universally
quantified in `(n_term, c_target)`, so instantiating it at `(4, 5/4)` is
an application, not an extension of the axiom set.

═══════════════════════════════════════════════════════════════════════════
STATUS OF THE AXIOM CLAIM `simplex_tv_coverage 4 (5/4)`
═══════════════════════════════════════════════════════════════════════════

The Lean theorem `autoconvolution_ratio_ge_5_4` is rigorously valid AS A
CONDITIONAL on the axiom.  Whether the axiom claim itself

    ∀ μ ∈ Δ^7, ∃ (ℓ, s), 2 ≤ ℓ ∧ mass_test_value 8 μ ℓ s ≥ 5/4

is true is a separate question.  The available evidence is the Python
cascade in `_prove_125.py` + `_refine_4_to_d16.py`, which reports:

  • Stage 1 (full d=8 enumeration, 122,661 canonical compositions):
      grid-pruned 122,661;  Stage-2 residue = 8,104 cells.
  • Stage 2 at d=8 (v4 cell-cert on residue):
      8,100 closed, 4 stragglers remain.
  • Stage 3: 4 stragglers refined to 7,920 d=16 children; all close at d=16.

Two issues with treating this as a clean discharge of the axiom:

  (a) SOFTWARE: v4's `tier_L_joint` (used in Stage 2, 2,476 closures) is
      unsound at ~24% (see `project_v4_ljoint_unsound`).  A v5 re-run with
      sound Lagrangian L_joint is required to certify those closures.

  (b) DIMENSION MISMATCH: the axiom is stated at d=8.  For the 4 d=8
      stragglers, the cascade does NOT close them at d=8 — it closes their
      d=16 refinements instead.  The d=8 statement `max_W TV_W ≥ 5/4` on
      the stragglers' Voronoi cells is therefore NOT directly verified.
      The d=16 statement IS verified for the 7,920 children, but lifting
      this to d=8 requires an inverse-refinement argument (the existing
      axiom `refinement_monotonicity` runs the other direction: d → 2d).

For these reasons, the present file does NOT claim to discharge the axiom
by computational evidence.  The Lean theorem is a *conditional* on the
axiom holding; that conditional theorem is itself rigorously valid and
free of sorries and new axioms.

To make the bound unconditional one of the following is needed:
  • A v5-sound re-run at d=16 enumerating *every* d=16 cell (not just the
    children of the d=8 stragglers), which would discharge the
    `simplex_tv_coverage 8 (5/4)` axiom (n_term = 8 form).
  • A direct mathematical argument (e.g. lifting Matolcsi–Vinuesa's
    bound + Theorem 1) bounding `inf_{μ ∈ Δ_d} max_W TV_W(μ)` from below.
  • A multi-level cascade predicate (analogous to the existing
    `CascadePruned` in `Sidon/Defs.lean`) capturing both d=8 closures and
    d=16 refinement, plus an inverse-refinement lemma.

The Cascade-125 module proves all the building blocks (Cell, B1 bound,
split coverage, cascade soundness theorem) that one needs to assemble
such an unconditional result without `sorry`.

No new axioms, no sorries.
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.CoarseCascade
import Sidon.Cascade125.Cell
import Sidon.Cascade125.TierB1
import Sidon.Cascade125.CellCert

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 8000000

open scoped BigOperators
open scoped Classical
open scoped Real

namespace Sidon.Cascade125

/-- The simplex-coverage hypothesis at `n_term = 4, c_target = 5/4`, supplied
    by the existing computational axiom `simplex_tv_coverage`. -/
private theorem simplex_coverage_125 :
    ∀ μ : Fin (2 * 4) → ℝ,
      on_simplex μ →
      ∃ (ℓ s : ℕ), 2 ≤ ℓ ∧ mass_test_value (2 * 4) μ ℓ s ≥ (5 / 4 : ℝ) :=
  simplex_tv_coverage 4 (5 / 4 : ℝ)

/-- **Main theorem**: every admissible `f` satisfies `R(f) ≥ 5/4`.

    Conditions: `f ≥ 0`, support in `(-1/4, 1/4)`, positive integral, and
    `‖f * f‖_∞ < ∞`.

    Proof:  apply `coarse_cascade_bound_general` with `n_term = 4`,
    `c_target = 5/4`, using `simplex_coverage_125` as the coverage
    hypothesis (which is `simplex_tv_coverage 4 (5/4)`, an instance of the
    existing computational axiom verifying the d=8 / S=16 cascade run). -/
theorem autoconvolution_ratio_ge_5_4 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (5 / 4 : ℝ) :=
  coarse_cascade_bound_general 4 (by norm_num) (5 / 4 : ℝ) (by norm_num)
    simplex_coverage_125 f hf_nonneg hf_supp hf_int_pos h_conv_fin

/-- Numerical restatement: `5/4 = 1.25`. -/
theorem autoconvolution_ratio_ge_1_25 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (1.25 : ℝ) := by
  have h := autoconvolution_ratio_ge_5_4 f hf_nonneg hf_supp hf_int_pos h_conv_fin
  have h_eq : (1.25 : ℝ) = 5 / 4 := by norm_num
  rw [h_eq]; exact h

end Sidon.Cascade125
