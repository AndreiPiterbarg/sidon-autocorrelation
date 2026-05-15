/-
Sidon Autocorrelation Constant: rigorous lower bound `C_{1a} ≥ 1.292`
— the Piterbarg-Bajaj-Vincent Bound.

This is the top-level entry point of the Lean formalisation of the
paper *A New Lower Bound for the Supremum of Autoconvolutions*.  The
proof of the headline theorem lives in `Sidon.MultiScale`; a
Schwartz-class variant lives in `Sidon.MultiScaleSchwartz`.

Construction.  Three-scale arcsine kernel applied to the
Matolcsi–Vinuesa (2010) master inequality, with all numerical anchors
discharged by a `flint.arb` certifier at 256-bit precision (see
`delsarte_dual/grid_bound_alt_kernel/`).

Headline theorems.
  * `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
    (and equivalently `Sidon.MultiScale.autoconvolution_ratio_ge_1_292`,
    `Sidon.MultiScale.C1a_ge_1292`) — the headline, parameterised on
    an `ExtremiserPrimitives f` bundle of analytic primitives.
  * `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000_schwartz`
    (and `Sidon.MultiScale.C1a_ge_1292_schwartz`) — the Schwartz-class
    instance, where the bundle is constructed unconditionally from
    `Sidon.MultiScale.ExtremiserPrimitives.construct_schwartz_from_atomic`.

Axioms.    The headline theorems reach exactly **two** numerical-only
           user axioms in their dependency closure:
             * `Sidon.MultiScale.K2_analytic_le_K2UpperQ`
               (`K_2(K_ms) ≤ 47897/10000`, paper Lemma 4.2), and
             * `Sidon.MultiScale.gain_analytic_ge_gainLowerQ`
               (`gain_analytic ≥ 20925/100000`, paper Lemmas 4.3–4.5).
           Both are certifier outputs of the form
           "`flint.arb` evaluated this functional".

           `Sidon.MultiScale.MV_master_inequality_for_extremiser` is
           now a Lean **theorem**, composed from the zero-axiom
           wire-ups `MV_master_via_slack_monotonicity` and
           `MV_master_inequality_from_MV_lemmas`.  The quadratic
           inversion `master_inequality_M_lower` and the five
           slack-soundness statements (`K_two_upper_bound`,
           `k_one_lower_bound`, `S_one_upper_bound`,
           `min_G_lower_bound`, `gain_lower_bound`) are likewise
           Lean *theorems*.

Layout.    The formalisation totals ≈ 8.4 kLoC across this root entry
           plus the 15 modules in `Sidon/` (Defs, Bessel,
           BilinearParseval, BundleDefs, BundleEq1, BundleEq2Schwartz,
           BundleEq3Schwartz, BundleEq4, FourierAux, MasterFromLemmas,
           MultiScale, MultiScaleSchwartz, MVLemmas,
           SchwartzAtomicDischarge, TorusParseval).

No `sorry`, no conjectural axioms.  Run `lake env lean AxiomCheck.lean`
to print the axiom inventory of the headline theorems.
-/

import Sidon.Defs
import Sidon.MultiScale
import Sidon.MultiScaleSchwartz
import Sidon.SchwartzAtomicDischarge
import Sidon.TorusParseval
