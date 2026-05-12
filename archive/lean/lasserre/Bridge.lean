import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

/-- Bridge from the discrete `val(d)` problem to the continuous Sidon
autocorrelation constant. -/
theorem valD_le_C1a (cfg : BaseConfig) :
    DiscreteBridgeEstablished cfg ->
    ContinuousBridgeEstablished cfg ->
    valD cfg.d <= C1a := by
  sorry

/-- If a rigorously certified Lasserre lower bound beats the current record,
then the current record is strictly below `C1a`. -/
theorem rigorous_lb_over_record_implies_new_best
    (cfg : BaseConfig) (out : SolverOutput) :
    VerifiedLowerBound cfg out ->
    out.lowerBound <= valD cfg.d ->
    valD cfg.d <= C1a ->
    provesNewBest out ->
    currentRecord < C1a := by
  sorry

/-- Single theorem packaging the full publication chain for a Lasserre-based
world-record lower bound. -/
theorem full_publication_chain
    (cfg : HighDConfig) (out : SolverOutput) :
    SolverOutputMatchesHighDCode cfg out ->
    RestrictedRelaxationSound cfg.base ->
    ConvergedWithoutMissedViolations cfg.base out ->
    VerifiedLowerBound cfg.base out ->
    DiscreteBridgeEstablished cfg.base ->
    ContinuousBridgeEstablished cfg.base ->
    provesNewBest out ->
    currentRecord < C1a := by
  sorry

end Lasserre

end
