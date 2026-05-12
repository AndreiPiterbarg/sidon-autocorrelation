/-
Sidon Cascade-125 — Cell Certification Cascade

Mirrors the recursive certification cascade in `_coarse_bnb_v4.py::cert_cell`:

  def cert_cell(cell, windows, c_target, max_depth=6, current_depth=0):
      # Order: empty → B1 → F → L_single → L_joint → split
      if is_cell_empty(cell):
          return CertResult(certified=True, tier_used='empty')
      if not cell.is_simplex_feasible():
          return CertResult(certified=True, tier_used='empty')

      # Tier B1 (μ-corner)
      for W in windows:
          if tier_B1_mu_corner(cell, W, c_target) > 0:
              return CertResult(certified=True, tier_used='B1')

      # Tier F / L / L_joint  (LP/SDP — not formalised in Lean)
      ...

      # Split
      sub1, sub2 = cell.split(axis)
      r1 = cert_cell(sub1, ...);  r2 = cert_cell(sub2, ...)
      if r1.certified and r2.certified:
          return CertResult(certified=True, tier_used='split')

We capture the cascade as an inductive predicate `CellCertified c_target cell`
with three constructors matching the Python algorithm's intra-cell branches
that are PROVABLE in Lean (no LP/SDP machinery needed):

  • `empty`  — cell is empty (no probability vector lies in it)
  • `tierB1` — some window's B1 corner bound dominates `c_target`
  • `split`  — both axis-split subcells are certified

The Python's F / L_single / L_joint tiers are LP/SDP relaxations.  Their
soundness reduces to "for every μ in the cell, the SDP-min lower bounds
μᵀ A_W μ"; this requires LP/SDP duality which we do not formalise here.
Their effect on the overall `C_{1a} ≥ 5/4` claim is absorbed at the
top level via the existing `simplex_tv_coverage` axiom — see `Bound.lean`.

The Python's d → 2d refinement step (in `_refine_4_to_d16.py`) is a
*driver-level* operation in the Python and not part of `cert_cell`; we
likewise expose `IsMassRefinement` in `Refinement.lean` as a structural
definition (re-exporting `is_mass_refinement` from `Sidon.Proof.CoarseCascade`)
but do not include a `refine` constructor in `CellCertified`.

We then prove the **soundness theorem**: a certified cell implies every
probability vector in it has some window with `mass_test_value ≥ c_target`.

No axioms, no sorries.
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.CoarseCascade
import Sidon.Cascade125.Cell
import Sidon.Cascade125.Empty
import Sidon.Cascade125.TierB1
import Sidon.Cascade125.Refinement

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical

namespace Sidon.Cascade125

/-- A *probability point* in a cell: lies in the box and sums to 1. -/
def Cell.ProbMem {d : ℕ} (cell : Cell d) (μ : Fin d → ℝ) : Prop :=
  cell.Mem μ ∧ (∑ i, μ i = 1)

/-- A cell is *certified* at threshold `c_target` if the cascade closes it.
    Constructors mirror the four PROVABLE branches of `cert_cell`. -/
inductive CellCertified {d : ℕ} (c_target : ℝ) : Cell d → Prop
  /-- Empty cell: no probability vector lies in it.  Vacuously certified. -/
  | empty {cell : Cell d}
      (h : ∀ μ : Fin d → ℝ, cell.Mem μ → ¬ (∑ i, μ i = 1)) :
      CellCertified c_target cell
  /-- B1 corner tier: some window's B1 bound dominates `c_target`.
      We carry `2 ≤ ℓ` (matching `mass_test_value_le_ratio`'s requirement). -/
  | tierB1 {cell : Cell d} (ℓ s : ℕ) (_hℓ : 2 ≤ ℓ)
      (h : cellB1Bound cell ℓ s ≥ c_target) :
      CellCertified c_target cell
  /-- Split: split along axis at value `mid ∈ [lo[axis], hi[axis]]`,
      both halves certified. -/
  | split {cell : Cell d} (axis : Fin d) (mid : ℝ)
      (h_in : cell.lo axis ≤ mid ∧ mid ≤ cell.hi axis)
      (h_lo_nn : 0 ≤ mid)
      (cert_lo : CellCertified c_target (cell.splitLo axis mid h_in))
      (cert_hi : CellCertified c_target (cell.splitHi axis mid h_in h_lo_nn)) :
      CellCertified c_target cell

/-- **Soundness theorem**.  If a cell is certified, then every probability
    vector in it has some window for which `mass_test_value ≥ c_target`.
    The returned `ℓ` satisfies `2 ≤ ℓ`, matching the hypothesis of
    `coarse_cascade_bound_general`. -/
theorem CellCertified.sound {d : ℕ} (c_target : ℝ) (cell : Cell d)
    (hcert : CellCertified c_target cell) :
    ∀ μ : Fin d → ℝ, cell.ProbMem μ →
      ∃ ℓ s, 2 ≤ ℓ ∧ mass_test_value d μ ℓ s ≥ c_target := by
  induction hcert with
  | @empty cell h_empty =>
    intro μ ⟨hμ_mem, hμ_sum⟩
    exact absurd hμ_sum (h_empty μ hμ_mem)
  | @tierB1 cell ℓ s hℓ h_bound =>
    intro μ ⟨hμ_mem, _⟩
    have hℓ_pos : 0 < ℓ := lt_of_lt_of_le (by norm_num : (0 : ℕ) < 2) hℓ
    exact ⟨ℓ, s, hℓ, cell_certified_by_B1 cell ℓ s hℓ_pos c_target h_bound μ hμ_mem⟩
  | @split cell axis mid h_in h_lo_nn _cert_lo _cert_hi ih_lo ih_hi =>
    intro μ ⟨hμ_mem, hμ_sum⟩
    rcases Cell.mem_split_cover cell axis mid h_in h_lo_nn hμ_mem with h | h
    · exact ih_lo μ ⟨h, hμ_sum⟩
    · exact ih_hi μ ⟨h, hμ_sum⟩

end Sidon.Cascade125
