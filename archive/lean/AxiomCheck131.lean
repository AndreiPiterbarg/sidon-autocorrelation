/-
Axiom inventory check for Cascade-131.

Run:  lake env lean AxiomCheck131.lean

We expect to see ONLY:
  * Lean core axioms (Classical.choice / propext / Quot.sound)
  * The seven NEW v6 axioms declared in `Sidon/Cascade131.lean`:
      - K_two_upper_bound_v6
      - k_one_lower_bound_v6
      - S_one_upper_bound_v6
      - min_G_lower_bound_v6
      - gain_lower_bound_v6
      - MV_master_inequality_for_extremiser_v6
      - master_inequality_M_lower_v6

  (The brief mentions "5 numerical axioms + MV master inequality + master
  inequality quadratic solver" = 7 named axioms total.  This matches the
  Cascade-130 pattern.)
-/
import Sidon.Cascade131

#print axioms Sidon.Cascade131.autoconvolution_ratio_ge_1293_1000
#print axioms Sidon.Cascade131.autoconvolution_ratio_ge_1_293
#print axioms Sidon.Cascade131.C1a_ge_1293
