/-
Sidon Autocorrelation Project — W-Refined Definitions (axiom-free)

This file factors out the **definitions** `W_int_for_window` and
`w_refined_correction` from `WRefinedBound.lean` so that they can be used by
both `TightDiscretizationBound.lean` (for the bijection lemma against
`W_int_overlap`) and `TightContinuousBridge.lean` (which now PROVES
`cs_eq1_w_refined` as a theorem from `cs_eq1_tight` + the dominance theorem
+ the bijection lemma).

The reason for factoring: `WRefinedBound.lean` previously declared
`cs_eq1_w_refined` as an axiom and is imported by both downstream files;
moving the axiom-deletion+theorem-proof into `TightContinuousBridge` would
create a circular import (WRefinedBound → TightContinuousBridge → WRefinedBound).
By splitting the **definitions** (this file) from the **axiom + downstream
machinery** (`WRefinedBound.lean`), we break the cycle:

    WRefinedDefs (no axiom)
      ↓
      ├── TightDiscretizationBound  (proves bijection W_int_overlap = W_int_for_window)
      │     ↓
      │     TightContinuousBridge  (axiom: cs_eq1_tight; theorem: cs_eq1_w_refined)
      │       ↓
      │       WRefinedBound  (uses cs_eq1_w_refined as a THEOREM, downstream
      │                       cascade machinery, computational axiom
      │                       cascade_all_pruned_w)
      │
      └── (other files importing only the definitions)

NEW AXIOMS DECLARED IN THIS FILE: ZERO.
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

-- ═══════════════════════════════════════════════════════════════════════════════
-- W_int_for_window: integer mass in bins overlapping a convolution window.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- W_int: the integer mass in bins contributing to a convolution window.

    For a window [s_lo, s_lo + ℓ - 2] in convolution space, the contributing
    bins are those i ∈ [0, d-1] such that ∃ j ∈ [0, d-1], s_lo ≤ i+j ≤ s_lo+ℓ-2.

    This simplifies to i ∈ [max(0, s_lo - d + 1), min(s_lo + ℓ - 2, d - 1)].

    W_int = ∑_{i in contributing range} c_i

    Matches CPU code: solvers.py lines 134-139 (lo_bin, hi_bin, prefix sum).
    Matches GPU code: cascade_kernel.cu lines 313-318 (sliding window on child[]). -/
def W_int_for_window (n : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℕ :=
  let d := 2 * n
  let lo := if s_lo + 1 ≤ d then 0 else s_lo - d + 1
  let hi := min (s_lo + ℓ - 2) (d - 1)
  if lo ≤ hi then
    ∑ i ∈ Finset.Icc lo hi, if h : i < d then c ⟨i, by omega⟩ else 0
  else 0

-- ═══════════════════════════════════════════════════════════════════════════════
-- w_refined_correction: per-window correction in TV space.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The W-refined correction for a specific window (ℓ, s_lo) on composition c.
    correction_w = W_int/(2n·m²) + 1/m² = (1 + W_int/(2n)) / m²

    This is the per-window correction from C&S equation (1), strictly tighter
    than the flat C&S Lemma 3 correction (2/m + 1/m²). -/
noncomputable def w_refined_correction (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  let W := W_int_for_window n c ℓ s_lo
  (W : ℝ) / (2 * n * (m : ℝ) ^ 2) + 1 / (m : ℝ) ^ 2

end -- noncomputable section
