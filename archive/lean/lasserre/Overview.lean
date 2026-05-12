/-
Lasserre SDP Proof Inventory - Overview

STATUS: all theorems in this tree are stubs.

Goal:
  document every proof obligation needed for the current Lasserre SDP code path
  to count as a valid proof of a new lower bound on the Sidon
  autocorrelation constant.

Dependency graph:
  Core -> ReducedMomentSet -> CliqueSparsity -> Consistency -> WindowConstraints
  WindowConstraints + ConstraintGeneration + Search -> Soundness
  Soundness + NumericalCertification + Bridge -> final publishable theorem

The modules below are meant to mirror the current implementation, especially:
  * tests/lasserre_highd.py
  * lasserre/core.py
  * lasserre/precompute.py
  * lasserre/cliques.py
  * lasserre/solvers.py
-/

import lasserre.Defs
import lasserre.Core
import lasserre.ReducedMomentSet
import lasserre.CliqueSparsity
import lasserre.Consistency
import lasserre.WindowConstraints
import lasserre.ConstraintGeneration
import lasserre.Search
import lasserre.NumericalCertification
import lasserre.Soundness
import lasserre.Bridge
import lasserre.Guardrails
-- Proved mathematical foundations (no sorry)
import lasserre.ProofPartialConsistency
import lasserre.ProofRelaxation
import lasserre.ProofPSDSubmatrix
import lasserre.ProofSimplexMoments
import lasserre.ProofSoundnessChain
