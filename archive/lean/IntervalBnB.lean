/-
IntervalBnB — Root Module.

Rigorous derivation of `C_{1a} ≥ val(d)` (Theorem 1 of
`interval_bnb/THEOREM.md`), plus the σ-symmetry reduction to the
half-simplex `H_d` (Section 3).

Modules:
  * Defs.lean             — core definitions (bins, windows, simplex, val_d)
  * PairSumGeometry.lean  — Lemma 1.1 (pair-sum geometry)
  * Averaging.lean        — Lemma 1.2 (max ≥ average)
  * WindowIntegral.lean   — Lemma 1.3 (Tonelli + drop-nonneg-terms)
  * Symmetry.lean         — Section 3 (σ on windows and distributions)
  * ScaleFactor.lean      — Section 2 (|I_W| = ℓ/(2d))
  * MainTheorem.lean      — Theorem 1 (C_{1a} ≥ val(d))
-/

import IntervalBnB.Defs
import IntervalBnB.PairSumGeometry
import IntervalBnB.Averaging
import IntervalBnB.WindowIntegral
import IntervalBnB.Symmetry
import IntervalBnB.ScaleFactor
import IntervalBnB.MainTheorem
