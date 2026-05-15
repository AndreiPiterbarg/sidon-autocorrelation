/-
Axiom inventory check for the headline theorem `C_{1a} ≥ 1.292`.

Run from the `lean/` directory:

    lake env lean AxiomCheck.lean

Expected output: Lean's three core axioms

  * `Classical.choice`
  * `propext`
  * `Quot.sound`

together with exactly **two** verifiable-by-computation numerical
axioms declared in `Sidon.MultiScale`:

  * `Sidon.MultiScale.K2_analytic_le_K2UpperQ`
      `K_2(K_ms) ≤ 47897/10000` (paper Lemma 4.2).
  * `Sidon.MultiScale.gain_analytic_ge_gainLowerQ`
      `gain_analytic ≥ 20925/100000` (paper Lemmas 4.3–4.5).

These are the only user axioms reached by the proofs of
`autoconvolution_ratio_ge_1292_1000` and its Schwartz-class variant
`autoconvolution_ratio_ge_1292_1000_schwartz`.  Both are certifier
outputs of the analytic functionals — `flint.arb` at 256-bit precision
on the explicit 3-scale arcsine kernel `K_ms` and the QP-optimised
cosine `G`.

For each of `C1a_ge_1292`, `autoconvolution_ratio_ge_1292_1000`,
`C1a_ge_1292_schwartz`, `autoconvolution_ratio_ge_1292_1000_schwartz`,
and `autoconvolution_ratio_ge_1292_1000_schwartz_residual`
the printout should be the 5-element multiset
`{propext, Classical.choice, Quot.sound,
  Sidon.MultiScale.K2_analytic_le_K2UpperQ,
  Sidon.MultiScale.gain_analytic_ge_gainLowerQ}`.

The MV-master wire-ups `MV_master_via_slack_monotonicity` and
`MV_master_inequality_from_MV_lemmas` (printed separately below) reach
**none** of the user axioms — they are pure algebraic/monotonicity
content and depend only on Lean's three core axioms.

The Schwartz-class bundle constructor
`ExtremiserPrimitives.construct_schwartz_from_atomic` reaches only
`gain_analytic_ge_gainLowerQ` (it uses the gain-floor axiom to anchor
its `m_G`-choice; it does *not* use the `K_2` axiom).

The algebraic inversion `master_inequality_M_lower` and the five
slack-soundness statements (`K_two_upper_bound`, `k_one_lower_bound`,
`S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound`) are Lean
*theorems* and therefore do not appear in any of the listings.

Any other axiom in the printout indicates a regression.
-/

import Sidon.MultiScale
import Sidon.MultiScaleSchwartz
import Sidon.SchwartzAtomicDischarge

#print axioms Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000
#print axioms Sidon.MultiScale.C1a_ge_1292
#print axioms Sidon.MultiScale.MV_master_via_slack_monotonicity
#print axioms Sidon.MultiScale.MV_master_inequality_from_MV_lemmas

#print axioms Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000_schwartz
#print axioms Sidon.MultiScale.C1a_ge_1292_schwartz
#print axioms Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000_schwartz_residual
#print axioms Sidon.MultiScale.ExtremiserPrimitives.construct_schwartz_from_atomic
