/-
Sidon Autocorrelation Project — Coarse Cascade Proof Stubs

All proofs needed for the coarse-grid cascade method (Theorem 1, no correction).
These correspond to the algorithmic changes in:
  - cloninger-steinerberger/cpu/run_cascade_coarse.py (v1)
  - cloninger-steinerberger/cpu/run_cascade_coarse_v2.py (v2, sound box cert)
  - proof/coarse_cascade_method.md (mathematical writeup)

Proof dependency graph:
  TVGradientHessian ← BoxCertification ← CascadeInduction
  TVGradientHessian ← TVConvexity
  RefinementMonotonicity ← ValMonotonicity
  RefinementMonotonicity ← CascadeInduction (subtree pruning justification)

STATUS: All files contain `sorry` stubs. No proofs are complete.

KEY OPEN PROBLEM: RefinementMonotonicity is an unproven conjecture
(183K+ empirical tests, zero violations). Proving it would make the
entire coarse cascade result unconditional.
-/

-- Core mathematical properties of TV
import Sidon.CoarseCascade.TVGradientHessian
import Sidon.CoarseCascade.TVConvexity

-- The main conjecture
import Sidon.CoarseCascade.RefinementMonotonicity

-- Box certification (continuous coverage)
import Sidon.CoarseCascade.BoxCertification

-- Integer arithmetic correctness
import Sidon.CoarseCascade.IntegerThreshold

-- Search space reduction
import Sidon.CoarseCascade.SubtreePruning
import Sidon.CoarseCascade.ReversalSymmetry
import Sidon.CoarseCascade.AsymmetryPruning
import Sidon.CoarseCascade.QuickCheck

-- Cascade structure
import Sidon.CoarseCascade.CascadeInduction
import Sidon.CoarseCascade.ValMonotonicity

-- Correctness of cascade mechanics (2026-04-13 fixes)
import Sidon.CoarseCascade.FeasibilityFilter
import Sidon.CoarseCascade.CanonicalCompletion
