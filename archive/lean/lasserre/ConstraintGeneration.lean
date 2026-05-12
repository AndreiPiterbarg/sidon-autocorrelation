import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom ActiveWindowSubproblemIsRelaxation : BaseConfig -> Prop
axiom AddingViolatedWindowsPreservesSoundness : BaseConfig -> Prop
axiom NoReportedViolationsImpliesFullFeasibility : BaseConfig -> Prop
axiom ConstraintGenerationFixedPointCorrect : BaseConfig -> SolverOutput -> Prop

/-- Covers the active-window subset model solved during CG. -/
theorem active_window_subproblem_is_relaxation (cfg : BaseConfig) :
    ConstraintGenerationMatchesCode cfg ->
    ActiveWindowSubproblemIsRelaxation cfg := by
  sorry

/-- Covers the "add most violated windows" loop in `solve_highd_sparse` and
`solve_enhanced`. -/
theorem adding_violated_windows_preserves_soundness (cfg : BaseConfig) :
    ConstraintGenerationMatchesCode cfg ->
    ActiveWindowSubproblemIsRelaxation cfg ->
    AddingViolatedWindowsPreservesSoundness cfg := by
  sorry

/-- Exact obligation behind the CG stopping criterion. -/
theorem no_reported_violations_implies_full_feasibility (cfg : BaseConfig) :
    ConstraintGenerationMatchesCode cfg ->
    NoReportedViolationsImpliesFullFeasibility cfg := by
  sorry

/-- If the loop halts with no missed violations and all subproblem bounds are
checked soundly, then the returned fixed point is a valid lower bound for the
full sparse relaxation. -/
theorem constraint_generation_fixed_point_correct
    (cfg : BaseConfig) (out : SolverOutput) :
    ConstraintGenerationMatchesCode cfg ->
    AddingViolatedWindowsPreservesSoundness cfg ->
    NoReportedViolationsImpliesFullFeasibility cfg ->
    ConvergedWithoutMissedViolations cfg out ->
    ConstraintGenerationFixedPointCorrect cfg out := by
  sorry

end Lasserre

end
