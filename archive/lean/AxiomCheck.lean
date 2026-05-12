/-
Axiom inventory check for the Cascade-125 main theorem.

Run:  lake env lean AxiomCheck.lean
to inspect the axiom dependencies of `autoconvolution_ratio_ge_5_4`.

We expect to see ONLY:
  • Classical / propext / Quot.sound / Classical.choice
    (logical foundations — automatically available in Mathlib)
  • simplex_tv_coverage          (PRE-EXISTING — computational, in CoarseCascade.lean)
  • refinement_monotonicity      (PRE-EXISTING — conjecture, in CoarseCascade.lean)

NO new axioms should appear.
-/
import Sidon.Cascade125

#print axioms Sidon.Cascade125.autoconvolution_ratio_ge_5_4
#print axioms Sidon.Cascade125.autoconvolution_ratio_ge_1_25
#print axioms Sidon.Cascade125.CellCertified.sound
#print axioms Sidon.Cascade125.cellB1Bound_le_mass_test_value
