/-
Axiom inventory check for the headline theorem `C_{1a} ≥ 1.292`.

Run from the `lean/` directory:

    lake env lean AxiomCheck.lean

Expected output: Lean's three core axioms

  * `Classical.choice`
  * `propext`
  * `Quot.sound`

together with exactly one user axiom declared in `Sidon.MultiScale`:

  * `Sidon.MultiScale.MV_master_inequality_for_extremiser`

This is the only user axiom reached by the proof of
`autoconvolution_ratio_ge_1292_1000`.  It asserts the MV master
inequality with the slack rationals `K2UpperQ = 47897/10000` and
`gainLowerQ = 20925/100000` directly substituted for `K_2` and `a`; the
substitution is justified by the certifier's bounds on the analytic
functionals (paper Lemmas 4.1-4.5).

The algebraic inversion `master_inequality_M_lower` and the five
slack-soundness statements (`K_two_upper_bound`, `k_one_lower_bound`,
`S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound`) are Lean
*theorems* and therefore do not appear in this listing.

Any other axiom in the printout indicates a regression.
-/

import Sidon.MultiScale

#print axioms Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000
#print axioms Sidon.MultiScale.C1a_ge_1292
