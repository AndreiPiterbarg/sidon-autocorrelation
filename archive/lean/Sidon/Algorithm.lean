/-
Sidon Autocorrelation Project — Algorithm Correctness Proofs

Formal verification that the cascade branch-and-prune algorithm is correct.
These proofs justify the computational axiom `cascade_all_pruned`:
every optimization used in the CPU/GPU cascade is proved sound.

Sections:
  • Enumeration: composition counting, Cartesian product bijection
  • Symmetry: reversal symmetry halves the search space
  • Cascade: refinement preserves mass, cascade induction step
  • Kernel: incremental convolution, Gray code, sliding window
  • Pruning: subtree pruning, Gray code subtree pruning
  • Optimization: sparse cross-terms, arc consistency
  • Background: refinement support, correction support, essential supremum
-/

-- Enumeration
import Sidon.Algorithm.CompositionEnum
import Sidon.Algorithm.FusedKernel

-- Symmetry
import Sidon.Algorithm.ReversalSymmetry

-- Cascade structure
import Sidon.Algorithm.RefinementMass
import Sidon.Algorithm.CascadeInduction

-- Kernel correctness
import Sidon.Algorithm.IncrementalAutoconv
import Sidon.Algorithm.GrayCode
import Sidon.Algorithm.SlidingWindow

-- Pruning
import Sidon.Algorithm.SubtreePruning
import Sidon.Algorithm.GrayCodeSubtreePruning

-- Optimization
import Sidon.Algorithm.SparseCrossTerm
import Sidon.Algorithm.ArcConsistency

-- Mathematical background
import Sidon.Algorithm.RefinementSupport
import Sidon.Algorithm.CorrectionSupport
import Sidon.Algorithm.EssSup

-- Block mass invariant (parent-level pruning)
import Sidon.Algorithm.BlockMassInvariant

-- GPU kernel soundness (stubs — not yet proved)
import Sidon.Algorithm.GpuKernelSoundness
