/-
Axiom inventory check for `Sidon.BundleDefs`.

Run from the `lean/` directory:

    lake env lean AxiomCheckBundleDefs.lean

Expected output: only Lean's three core axioms

  * `Classical.choice`
  * `propext`
  * `Quot.sound`

(no new user axioms beyond those already present in `Sidon.MultiScale`).
-/

import Sidon.BundleDefs

#print axioms Sidon.BundleDefs.m_G_const
#print axioms Sidon.BundleDefs.S_G_const
#print axioms Sidon.BundleDefs.m_G_const_eq_rat
#print axioms Sidon.BundleDefs.S_G_const_eq_rat
#print axioms Sidon.BundleDefs.S_G_const_pos
#print axioms Sidon.BundleDefs.m_G_const_nonneg
#print axioms Sidon.BundleDefs.m_G_const_le_one
#print axioms Sidon.BundleDefs.K_ms_fourier_lattice
#print axioms Sidon.BundleDefs.K_ms_fourier_lattice_nonneg
#print axioms Sidon.BundleDefs.K_ms_fourier_lattice_zero
#print axioms Sidon.BundleDefs.S_cos
#print axioms Sidon.BundleDefs.S_cos_summand_nonneg
#print axioms Sidon.BundleDefs.LHS1
#print axioms Sidon.BundleDefs.LHS2
#print axioms Sidon.BundleDefs.LHS1_integrand_integrable
#print axioms Sidon.BundleDefs.LHS2_integrand_integrable
#print axioms Sidon.BundleDefs.cauchy_schwarz_indicator
#print axioms Sidon.BundleDefs.K2_analytic_ge_one_of_integral_one
#print axioms Sidon.BundleDefs.R_ge_one_of_data
