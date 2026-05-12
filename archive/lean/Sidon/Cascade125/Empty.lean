/-
Sidon Cascade-125 — Empty Cell Tests (B34)

Mirrors the sum-based empty tests in `_coarse_bnb_v4.py::is_cell_empty`:

  def is_cell_empty(cell, eps=1e-12) -> bool:
      if cell.lo.sum() > 1.0 + eps:    # B34a
          return True
      if cell.hi.sum() < 1.0 - eps:    # B34b
          return True
      ...

For a box ∩ simplex with `lo ≥ 0`, the condition `Σlo ≤ 1 ≤ Σhi` is
NECESSARY for nonemptiness (immediate from `lo ≤ μ ≤ hi` and `Σμ = 1`); it
is also sufficient (by IVT on `t ↦ (1-t)·lo + t·hi`), though we only need
necessity for the cascade and prove only that direction here.

(The Python file additionally runs a B35 single-coord rest-sum test that is
*not* sound on its own — counter-example `lo=[0.5, 0], hi=[1.0, 0.3]` has
μ=[0.7, 0.3] feasible yet B35 fires.  We omit B35 from the Lean
formalisation.  B34a + B34b are themselves complete in the sense above.)

No axioms, no sorries.
-/

import Mathlib
import Sidon.Cascade125.Cell

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

noncomputable section

namespace Sidon.Cascade125

namespace Cell

variable {d : ℕ}

/-- B34a: sum of lower bounds > 1 ⟹ no probability vector in cell. -/
theorem empty_of_lo_sum_gt_one (cell : Cell d)
    (h : (∑ i, cell.lo i) > 1) :
    ∀ μ : Fin d → ℝ, cell.Mem μ → ¬ (∑ i, μ i = 1) := by
  intro μ hμ hsum
  have h_lo_le_sum : (∑ i, cell.lo i) ≤ ∑ i, μ i :=
    Finset.sum_le_sum (fun i _ => (hμ i).1)
  rw [hsum] at h_lo_le_sum
  linarith

/-- B34b: sum of upper bounds < 1 ⟹ no probability vector in cell. -/
theorem empty_of_hi_sum_lt_one (cell : Cell d)
    (h : (∑ i, cell.hi i) < 1) :
    ∀ μ : Fin d → ℝ, cell.Mem μ → ¬ (∑ i, μ i = 1) := by
  intro μ hμ hsum
  have h_sum_le_hi : (∑ i, μ i) ≤ ∑ i, cell.hi i :=
    Finset.sum_le_sum (fun i _ => (hμ i).2)
  rw [hsum] at h_sum_le_hi
  linarith

end Cell

end Sidon.Cascade125

end -- noncomputable section
