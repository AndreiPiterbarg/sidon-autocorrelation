import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom ScalarWindowConstraintSound : BaseConfig -> Prop
axiom WindowCoverageMapCorrect : BaseConfig -> Prop
axiom CoveredWindowPSDIsPrincipalSubmatrix : BaseConfig -> Prop
axiom ViolationCheckerReportsOnlyGenuineViolations : BaseConfig -> Prop
axiom UncoveredWindowsUseScalarOnly : BaseConfig -> Prop
axiom PartialQPSDIsUnsoundShortcut : BaseConfig -> Prop

/-- Covers scalar window inequalities from `lasserre/precompute.py` and
`tests/lasserre_highd.py`. -/
theorem scalar_window_constraint_sound (cfg : BaseConfig) :
    WindowSystemMatchesCode cfg ->
    ScalarWindowConstraintSound cfg := by
  sorry

/-- Covers `window_covering` inside `_precompute_highd`. -/
theorem window_coverage_map_correct (cfg : BaseConfig) :
    WindowSystemMatchesCode cfg ->
    WindowCoverageMapCorrect cfg := by
  sorry

/-- Covers `_add_window_psd_highd` and the covered-window branch of the
violation checker. -/
theorem covered_window_psd_is_principal_submatrix (cfg : BaseConfig) :
    WindowSystemMatchesCode cfg ->
    WindowCoverageMapCorrect cfg ->
    CoveredWindowPSDIsPrincipalSubmatrix cfg := by
  sorry

/-- Covers `_check_violations_highd` and `_batch_check_violations`. -/
theorem violation_checker_reports_only_genuine_violations (cfg : BaseConfig) :
    WindowSystemMatchesCode cfg ->
    CoveredWindowPSDIsPrincipalSubmatrix cfg ->
    ViolationCheckerReportsOnlyGenuineViolations cfg := by
  sorry

/-- Exact obligation behind the "covered windows get PSD, uncovered windows get
only the scalar constraint" design. -/
theorem uncovered_windows_use_scalar_only (cfg : BaseConfig) :
    WindowSystemMatchesCode cfg ->
    WindowCoverageMapCorrect cfg ->
    UncoveredWindowsUseScalarOnly cfg := by
  sorry

/-- Guardrail theorem documenting why partial-Q PSD is not an admissible
shortcut for uncovered windows. -/
theorem partial_q_psd_is_unsound_shortcut (cfg : BaseConfig) :
    PartialQPSDIsUnsoundShortcut cfg := by
  sorry

end Lasserre

end
