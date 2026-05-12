/-
Sidon Cascade-125 — Cell (Box) Definition

Mirrors the Python `Cell` class in `_coarse_bnb_v3.py`:

  @dataclass
  class Cell:
      lo: np.ndarray            -- per-coord lower bound, ≥ 0
      hi: np.ndarray            -- per-coord upper bound, ≥ lo

      def is_simplex_feasible(self) -> bool:
          return (self.lo.sum() ≤ 1 + 1e-12 and self.hi.sum() ≥ 1 - 1e-12)

      @classmethod
      def from_integer_composition(cls, c, S):
          h = 1 / (2 * S)
          lo = max(c/S - h, 0)
          hi = c/S + h

This file defines `Cell d`, membership `μ ∈ cell`, simplex-feasibility, the
axis-aligned `split`, and the `fromIntegerComposition` constructor.

No axioms, no sorries.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

noncomputable section

namespace Sidon.Cascade125

/-- An axis-aligned box [lo, hi] in ℝ^d with all `lo i ≥ 0` and `lo i ≤ hi i`. -/
structure Cell (d : ℕ) where
  lo : Fin d → ℝ
  hi : Fin d → ℝ
  lo_nonneg : ∀ i, 0 ≤ lo i
  lo_le_hi : ∀ i, lo i ≤ hi i

namespace Cell

variable {d : ℕ}

/-- Membership: `μ` lies in the box (per-coordinate). -/
def Mem (cell : Cell d) (μ : Fin d → ℝ) : Prop :=
  ∀ i, cell.lo i ≤ μ i ∧ μ i ≤ cell.hi i

/-- Every point in the cell is coordinate-wise nonneg. -/
theorem nonneg_of_mem (cell : Cell d) {μ : Fin d → ℝ}
    (hμ : cell.Mem μ) (i : Fin d) : 0 ≤ μ i :=
  le_trans (cell.lo_nonneg i) (hμ i).1

/-- Simplex-feasibility predicate matching Python's `is_simplex_feasible`. -/
def IsSimplexFeasible (cell : Cell d) : Prop :=
  (∑ i, cell.lo i) ≤ 1 ∧ 1 ≤ ∑ i, cell.hi i

/-- If a cell contains a probability vector, it is simplex-feasible. -/
theorem isSimplexFeasible_of_mem (cell : Cell d) {μ : Fin d → ℝ}
    (hμ : cell.Mem μ) (hsum : ∑ i, μ i = 1) : cell.IsSimplexFeasible := by
  refine ⟨?_, ?_⟩
  · calc ∑ i, cell.lo i
        ≤ ∑ i, μ i := Finset.sum_le_sum (fun i _ => (hμ i).1)
      _ = 1 := hsum
  · calc (1 : ℝ) = ∑ i, μ i := hsum.symm
      _ ≤ ∑ i, cell.hi i := Finset.sum_le_sum (fun i _ => (hμ i).2)

/-- Split a cell along `axis` at value `mid`.  Returns the "lower" half. -/
def splitLo (cell : Cell d) (axis : Fin d) (mid : ℝ)
    (h_in : cell.lo axis ≤ mid ∧ mid ≤ cell.hi axis) : Cell d where
  lo := cell.lo
  hi := fun i => if i = axis then mid else cell.hi i
  lo_nonneg := cell.lo_nonneg
  lo_le_hi := by
    intro i
    by_cases hi_eq : i = axis
    · -- i = axis, so hi becomes `mid`
      subst hi_eq
      simp
      exact h_in.1
    · simp [hi_eq]
      exact cell.lo_le_hi i

/-- Split a cell along `axis` at value `mid`.  Returns the "upper" half. -/
def splitHi (cell : Cell d) (axis : Fin d) (mid : ℝ)
    (h_in : cell.lo axis ≤ mid ∧ mid ≤ cell.hi axis)
    (h_lo_nonneg : 0 ≤ mid) : Cell d where
  lo := fun i => if i = axis then mid else cell.lo i
  hi := cell.hi
  lo_nonneg := by
    intro i
    by_cases hi_eq : i = axis
    · subst hi_eq; simp; exact h_lo_nonneg
    · simp [hi_eq]; exact cell.lo_nonneg i
  lo_le_hi := by
    intro i
    by_cases hi_eq : i = axis
    · subst hi_eq; simp; exact h_in.2
    · simp [hi_eq]; exact cell.lo_le_hi i

/-- Splitting covers: every point in `cell` lies in `splitLo` or `splitHi`. -/
theorem mem_split_cover (cell : Cell d) (axis : Fin d) (mid : ℝ)
    (h_in : cell.lo axis ≤ mid ∧ mid ≤ cell.hi axis) (h_lo_nn : 0 ≤ mid)
    {μ : Fin d → ℝ} (hμ : cell.Mem μ) :
    (cell.splitLo axis mid h_in).Mem μ ∨ (cell.splitHi axis mid h_in h_lo_nn).Mem μ := by
  by_cases hle : μ axis ≤ mid
  · left
    intro i
    refine ⟨(hμ i).1, ?_⟩
    by_cases h_eq : i = axis
    · subst h_eq; simp [splitLo]; exact hle
    · simp [splitLo, h_eq]; exact (hμ i).2
  · right
    push_neg at hle
    intro i
    refine ⟨?_, (hμ i).2⟩
    by_cases h_eq : i = axis
    · subst h_eq; simp [splitHi]; exact le_of_lt hle
    · simp [splitHi, h_eq]; exact (hμ i).1

/-- Construct a cell from an integer composition `c` of `S` parts, using
    radius `1/S` (matches the existing `voronoi_coverage` theorem; this is
    looser than Python's `1/(2S)` but suffices and is fully sound). -/
def fromIntegerComposition (c : Fin d → ℕ) (S : ℕ) (_hS : S > 0) : Cell d where
  lo := fun i => max ((c i : ℝ) / S - 1 / S) 0
  hi := fun i => (c i : ℝ) / S + 1 / S
  lo_nonneg := fun _ => le_max_right _ _
  lo_le_hi := by
    intro i
    have h₁ : ((c i : ℝ) / S - 1 / S) ≤ (c i : ℝ) / S + 1 / S := by
      have h1S : (0 : ℝ) ≤ 1 / S := by positivity
      linarith
    have h₂ : (0 : ℝ) ≤ (c i : ℝ) / S + 1 / S := by
      have hc_nn : 0 ≤ (c i : ℝ) / S := div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
      have h1S : 0 ≤ (1 : ℝ) / S := by positivity
      linarith
    exact max_le h₁ h₂

/-- An integer composition's Voronoi neighborhood (radius 1/S) contains every
    point that the cumulative-rounding Voronoi witness lands within `1/S` of
    the integer composition. -/
theorem mem_fromIntegerComposition_of_voronoi
    {S : ℕ} (hS : S > 0) (c : Fin d → ℕ) (μ : Fin d → ℝ)
    (hμ_nn : ∀ i, 0 ≤ μ i)
    (h_err : ∀ i, |μ i - (c i : ℝ) / (S : ℝ)| < 1 / (S : ℝ)) :
    (Cell.fromIntegerComposition c S hS).Mem μ := by
  intro i
  have h_err_i := h_err i
  rw [abs_lt] at h_err_i
  obtain ⟨h_lo_err, h_hi_err⟩ := h_err_i
  refine ⟨?_, ?_⟩
  · -- lo i = max ((c i)/S - 1/S) 0 ≤ μ i
    show max ((c i : ℝ) / S - 1 / S) 0 ≤ μ i
    apply max_le
    · linarith
    · exact hμ_nn i
  · -- μ i ≤ hi i = (c i)/S + 1/S
    show μ i ≤ (c i : ℝ) / S + 1 / S
    linarith

end Cell

end Sidon.Cascade125

end -- noncomputable section
