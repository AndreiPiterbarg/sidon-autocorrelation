/-
Sidon Cascade-125 — Refinement (d → 2d)

Mirrors the d-doubling refinement used in `_refine_4_to_d16.py`:

  def gen_children(parent):
      """All children at d_child = 2 * d_parent. Each parent bin c_i splits
      into (a, c_i - a) for a ∈ [0, c_i]."""
      d_p = len(parent)
      opts = [[(a, p - a) for a in range(p + 1)] for p in parent]
      ...

At the mass level this is exactly `is_mass_refinement` already defined in
`Sidon/Proof/CoarseCascade.lean`.  This file re-exports it under the
`Sidon.Cascade125` namespace and adds a small projection lemma
(`isMassRefinement_iff`) for convenience.

Note: refinement is performed by the Python *driver* (`_refine_4_to_d16.py`)
rather than inside `cert_cell`, so `CellCertified` does NOT have a
`refine` constructor.  This file provides the structural definitions in
case a future inductive predicate `CoarseCascadePruned` needs them.

No axioms, no sorries.
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.CoarseCascade

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

namespace Sidon.Cascade125

/-- ν ∈ ℝ^{2d} is a mass-refinement of μ ∈ ℝ^d (re-export from
    `Sidon/Proof/CoarseCascade.lean`). -/
abbrev IsMassRefinement {d : ℕ} (μ : Fin d → ℝ) (ν : Fin (2 * d) → ℝ) : Prop :=
  is_mass_refinement μ ν

/-- The "parent" map: collapse adjacent pairs of `ν` into a single coordinate. -/
noncomputable def parentMass {d : ℕ} (ν : Fin (2 * d) → ℝ) (i : Fin d) : ℝ :=
  ν ⟨2 * i.val, by omega⟩ + ν ⟨2 * i.val + 1, by omega⟩

/-- The mass-refinement property says exactly that `parentMass ν = μ`. -/
theorem isMassRefinement_iff {d : ℕ} (μ : Fin d → ℝ) (ν : Fin (2 * d) → ℝ) :
    IsMassRefinement μ ν ↔ (parentMass ν = μ) ∧ (∀ j, 0 ≤ ν j) := by
  unfold IsMassRefinement is_mass_refinement parentMass
  constructor
  · intro ⟨h_split, h_nn⟩
    refine ⟨funext (fun i => h_split i), h_nn⟩
  · intro ⟨h_eq, h_nn⟩
    refine ⟨fun i => ?_, h_nn⟩
    exact congrFun h_eq i

end Sidon.Cascade125
