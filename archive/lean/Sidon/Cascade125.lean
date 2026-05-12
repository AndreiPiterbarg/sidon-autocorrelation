/-
Sidon Cascade-125 — Module Index

Lean formalisation of the coarse-cascade proof that `C_{1a} ≥ 5/4 = 1.25`,
matching the Python cascade in `_prove_125.py` and `_refine_4_to_d16.py`.

Sections (one Lean file per cascade section in `_coarse_bnb_v4.py`):

  • Cell        — Cell box, simplex feasibility, axis split, integer-composition
                  cell.  Mirrors the Python `Cell` class.
  • Empty       — B34 (lo-sum / hi-sum) empty tests.  Mirrors the SOUND part
                  of `is_cell_empty`.  We deliberately OMIT Python's B35
                  single-coord test (mathematically unsound — see
                  `_b35_audit.py` for a counter-example).
  • TierB1      — μ-corner lower bound on `mass_test_value`.  Mirrors
                  `tier_B1_mu_corner`.  FULLY PROVED.
  • Refinement  — d → 2d mass-refinement (re-export of `is_mass_refinement`
                  from `Sidon/Proof/CoarseCascade.lean`).  Structural
                  definition; not used by `CellCertified` (refinement is a
                  driver-level operation in `_refine_4_to_d16.py`).
  • CellCert    — `CellCertified` inductive predicate (empty / tierB1 /
                  split) plus the soundness theorem `CellCertified.sound`.
                  Mirrors the *intra-cell* recursion in `cert_cell`.
  • Bound       — Final theorem `autoconvolution_ratio_ge_5_4` and its
                  numerical restatement `autoconvolution_ratio_ge_1_25`,
                  obtained from `coarse_cascade_bound_general` applied to
                  the existing `simplex_tv_coverage 4 (5/4 : ℝ)` axiom.
                  See `Bound.lean` for a detailed note on the
                  conditional-on-axiom status of the bound.

No new axioms are introduced and no Lean proofs use `sorry`.
-/

import Sidon.Cascade125.Cell
import Sidon.Cascade125.Empty
import Sidon.Cascade125.TierB1
import Sidon.Cascade125.Refinement
import Sidon.Cascade125.CellCert
import Sidon.Cascade125.Bound
