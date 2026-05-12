import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom ExactHighDRelaxationLowerBoundSound : BaseConfig -> Prop
axiom EnhancedFullModeSound : EnhancedConfig -> Prop
axiom EnhancedSparseModeSound : EnhancedConfig -> Prop
axiom EnhancedDSOSModeSound : EnhancedConfig -> Prop
axiom EnhancedBMModeSoundAfterVerification : EnhancedConfig -> Prop
axiom CGSolverSound : BaseConfig -> Prop
axiom FusionSolverSound : BaseConfig -> Prop

/-- Main mathematical soundness statement for the clique-restricted high-d
relaxation built in `tests/lasserre_highd.py`. -/
theorem exact_highd_relaxation_lower_bound_sound (cfg : BaseConfig) :
    RestrictedRelaxationSound cfg ->
    ExactHighDRelaxationLowerBoundSound cfg := by
  sorry

/-- Public API theorem for `solve_highd_sparse`. -/
theorem solve_highd_sparse_output_le_valD
    (cfg : HighDConfig) (out : SolverOutput) :
    SolverOutputMatchesHighDCode cfg out ->
    RestrictedRelaxationSound cfg.base ->
    ConvergedWithoutMissedViolations cfg.base out ->
    VerifiedLowerBound cfg.base out ->
    out.lowerBound <= valD cfg.d := by
  sorry

/-- Public API theorem for `solve_enhanced` in full PSD mode. -/
theorem solve_enhanced_full_output_le_valD
    (cfg : EnhancedConfig) (out : SolverOutput) :
    cfg.psdMode = .full ->
    SolverOutputMatchesEnhancedCode cfg out ->
    RestrictedRelaxationSound cfg.base ->
    VerifiedLowerBound cfg.base out ->
    EnhancedFullModeSound cfg ->
    out.lowerBound <= valD cfg.d := by
  sorry

/-- Public API theorem for `solve_enhanced` in sparse PSD mode. -/
theorem solve_enhanced_sparse_output_le_valD
    (cfg : EnhancedConfig) (out : SolverOutput) :
    cfg.psdMode = .sparse ->
    SolverOutputMatchesEnhancedCode cfg out ->
    RestrictedRelaxationSound cfg.base ->
    VerifiedLowerBound cfg.base out ->
    EnhancedSparseModeSound cfg ->
    out.lowerBound <= valD cfg.d := by
  sorry

/-- Public API theorem for `solve_enhanced` in DSOS mode. -/
theorem solve_enhanced_dsos_output_le_valD
    (cfg : EnhancedConfig) (out : SolverOutput) :
    cfg.psdMode = .dsos ->
    SolverOutputMatchesEnhancedCode cfg out ->
    RestrictedRelaxationSound cfg.base ->
    VerifiedLowerBound cfg.base out ->
    EnhancedDSOSModeSound cfg ->
    out.lowerBound <= valD cfg.d := by
  sorry

/-- Public API theorem for `solve_enhanced` in BM mode after verification. -/
theorem solve_enhanced_bm_output_le_valD
    (cfg : EnhancedConfig) (out : SolverOutput) :
    cfg.psdMode = .bm ->
    SolverOutputMatchesEnhancedCode cfg out ->
    RestrictedRelaxationSound cfg.base ->
    VerifiedLowerBound cfg.base out ->
    EnhancedBMModeSoundAfterVerification cfg ->
    out.lowerBound <= valD cfg.d := by
  sorry

/-- Public API theorem for `solve_cg`. -/
theorem solve_cg_output_le_valD
    (cfg : BaseConfig) (out : SolverOutput) :
    SolverOutputMatchesCGCode cfg out ->
    RestrictedRelaxationSound cfg ->
    VerifiedLowerBound cfg out ->
    CGSolverSound cfg ->
    out.lowerBound <= valD cfg.d := by
  sorry

/-- Public API theorem for `solve_lasserre_fusion`. -/
theorem solve_fusion_output_le_valD
    (cfg : BaseConfig) (out : SolverOutput) :
    SolverOutputMatchesFusionCode cfg out ->
    RestrictedRelaxationSound cfg ->
    VerifiedLowerBound cfg out ->
    FusionSolverSound cfg ->
    out.lowerBound <= valD cfg.d := by
  sorry

end Lasserre

end
