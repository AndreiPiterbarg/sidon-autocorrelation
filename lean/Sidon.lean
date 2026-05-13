/-
Sidon Autocorrelation Constant: rigorous lower bound `C_{1a} ≥ 1.292`
— the Piterbarg-Bajaj-Vincent Bound.

This is the top-level entry point of the Lean formalisation of the
paper *A New Lower Bound for the Supremum of Autoconvolutions*.  The
proof of the headline theorem lives in `Sidon.MultiScale`.

Construction.  Three-scale arcsine kernel applied to the
Matolcsi–Vinuesa (2010) master inequality, with all numerical anchors
discharged by a `flint.arb` certifier at 256-bit precision (see
`delsarte_dual/grid_bound_alt_kernel/`).

Headline.  `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
           (and equivalently `Sidon.MultiScale.autoconvolution_ratio_ge_1_292`,
           `Sidon.MultiScale.C1a_ge_1292`).

Axioms.    The headline theorem reaches exactly **one** user axiom in
           its dependency closure:
           `Sidon.MultiScale.MV_master_inequality_for_extremiser`
           (the 3-scale master inequality with the slack rationals
           `K2UpperQ` and `gainLowerQ` substituted for the analytic
           `K_2` and `a`).  The quadratic inversion
           `master_inequality_M_lower` and the five slack-soundness
           statements (`K_two_upper_bound`, `k_one_lower_bound`,
           `S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound`)
           are Lean *theorems*.

No `sorry`, no conjectural axioms.  Run `lake env lean AxiomCheck.lean`
to print the axiom inventory of the headline theorem.
-/

import Sidon.Defs
import Sidon.MultiScale
