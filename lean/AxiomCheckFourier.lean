/-
Axiom inventory check for the Plancherel bridge theorems in
`Sidon.FourierAux`.  Run from the `lean/` directory:

    lake env lean AxiomCheckFourier.lean

Expected output: Lean's three core axioms only

  * `Classical.choice`
  * `propext`
  * `Quot.sound`
-/

import Sidon.FourierAux

#print axioms Sidon.FourierAux.Plancherel.plancherel_schwartz
#print axioms Sidon.FourierAux.Plancherel.plancherel_schwartz'
#print axioms Sidon.FourierAux.Plancherel.parseval_schwartz_inner
#print axioms Sidon.FourierAux.Plancherel.plancherel_schwartz_real
#print axioms Sidon.FourierAux.LpBridge.norm_sq_toLp_eq_integral
#print axioms Sidon.FourierAux.LpBridge.plancherel_lp_norm
#print axioms Sidon.FourierAux.PlancherelL1L2.plancherel_schwartz_via_lp
