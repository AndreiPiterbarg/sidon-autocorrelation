import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom RawSolverStatusInsufficient : BaseConfig -> Prop
axiom VerifiedInfeasibilityCertificateRequired : BaseConfig -> Prop
axiom VerifiedFeasibilityWitnessRequired : BaseConfig -> Prop
axiom VerifiedSpectralResidualBoundsRequired : BaseConfig -> Prop
axiom CertificateCheckerYieldsRigorousLowerBound :
  BaseConfig -> SolverOutput -> Prop

/-- Formal proof obligation documenting that raw MOSEK, SCS, or Clarabel status
codes are not themselves a formal proof object. -/
theorem raw_solver_status_insufficient (cfg : BaseConfig) :
    RawSolverStatusInsufficient cfg := by
  sorry

/-- Needed to justify each "infeasible" branch used during bisection. -/
theorem verified_infeasibility_certificate_required (cfg : BaseConfig) :
    VerifiedInfeasibilityCertificateRequired cfg := by
  sorry

/-- Needed to justify the final feasible witness extracted at the reported
boundary. -/
theorem verified_feasibility_witness_required (cfg : BaseConfig) :
    VerifiedFeasibilityWitnessRequired cfg := by
  sorry

/-- Needed because the implementation relies on PSD and spectral tests computed
numerically. -/
theorem verified_spectral_residual_bounds_required (cfg : BaseConfig) :
    VerifiedSpectralResidualBoundsRequired cfg := by
  sorry

/-- Solver-agnostic obligation for turning approximate conic output into a
rigorous published lower bound. -/
theorem certificate_checker_yields_rigorous_lower_bound
    (cfg : BaseConfig) (out : SolverOutput) :
    NumericalCertificationMatchesCode cfg ->
    VerifiedLowerBound cfg out ->
    CertificateCheckerYieldsRigorousLowerBound cfg out := by
  sorry

end Lasserre

end
