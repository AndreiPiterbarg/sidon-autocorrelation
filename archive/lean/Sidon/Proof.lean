/-
Sidon Autocorrelation Project — Main Proof

The complete proof chain for the main theorem:
  autoconvolution_ratio_ge_32_25 : C₁ₐ ≥ 32/25 = 1.28

Proof structure:
  Defs → Foundational → StepFunction → TestValueBounds → DiscretizationError → FinalResult
  with CauchySchwarz and AsymmetryBound as dependencies of TestValueBounds.

W-Refined proof chain (tighter thresholds):
  WRefinedBound extends DiscretizationError with per-window W-refined correction.
  Uses C&S equation (1) instead of Lemma 3 for ~3-5× tighter threshold.

Point-Eval proof chain (strictly tightest, no averaging slack):
  PointEvalBound replaces window-averaging with EXACT pointwise evaluation
  at conv-bin lattice points (where (g*g) achieves its piecewise-linear
  breakpoints).  Empirically dominates W-refined at every benchmark config
  (zero P-not-A in the SOTA sweep).

Axioms:
  • cs_lemma3_per_window — C&S Lemma 3 flat bound (2/m + 1/m²)
  • cs_eq1_tight        — C&S eq(1) tight (D-v2) per-window bound
                          (cs_eq1_w_refined now derived as a theorem)
  • cs_eq1_pointeval    — C&S eq(1) pointwise at lattice (point-eval)
  • cascade_all_pruned  — computational result with flat threshold
  • cascade_all_pruned_w — computational result with W-refined threshold
  • cascade_all_pruned_p — computational result with point-eval threshold
-/

import Sidon.Proof.Foundational
import Sidon.Proof.CauchySchwarz
import Sidon.Proof.AsymmetryBound
import Sidon.Proof.StepFunction
import Sidon.Proof.TestValueBounds
import Sidon.Proof.DiscretizationError
import Sidon.Proof.RefinementBridge
import Sidon.Proof.FinalResult
import Sidon.Proof.WRefinedDefs
import Sidon.Proof.TightDiscretizationBound
import Sidon.Proof.WRefinedBound
import Sidon.Proof.SpectralDiscretizationBound
import Sidon.Proof.TightContinuousBridge
import Sidon.Proof.CoarseCascade
import Sidon.Proof.PostFilterF
import Sidon.Proof.PostFilterQ
import Sidon.Proof.PostFilterL
import Sidon.Proof.PostFilterChain
import Sidon.Proof.PointEvalBound
