import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom EnumMonomialsComplete : BaseConfig -> Prop
axiom CollectMomentsMatchesDegreeBound : BaseConfig -> Prop
axiom WindowMatricesMatchDiscreteProblem : BaseConfig -> Prop
axiom HashCollisionCheckSufficient : BaseConfig -> Prop
axiom HashLookupCorrectWhenChecksPass : BaseConfig -> Prop

/-- Covers `lasserre/core.py:enum_monomials`. -/
theorem enum_monomials_complete (cfg : BaseConfig) :
    CoreEncodingMatchesCode cfg -> EnumMonomialsComplete cfg := by
  sorry

/-- Covers `lasserre/core.py:collect_moments`. -/
theorem collect_moments_degree_complete (cfg : BaseConfig) :
    CoreEncodingMatchesCode cfg -> CollectMomentsMatchesDegreeBound cfg := by
  sorry

/-- Covers `lasserre/core.py:build_window_matrices`. -/
theorem window_matrices_match_discrete_problem (cfg : BaseConfig) :
    CoreEncodingMatchesCode cfg -> WindowMatricesMatchDiscreteProblem cfg := by
  sorry

/-- Covers the runtime hash-collision assertions used by the high-d code path. -/
theorem hash_collision_check_is_sufficient (cfg : BaseConfig) :
    CoreEncodingMatchesCode cfg -> HashCollisionCheckSufficient cfg := by
  sorry

/-- Covers `_build_hash_table` and `_hash_lookup` under the checked no-collision
assumption. -/
theorem hash_lookup_correct_when_checks_pass (cfg : BaseConfig) :
    CoreEncodingMatchesCode cfg ->
    HashCollisionCheckSufficient cfg ->
    HashLookupCorrectWhenChecksPass cfg := by
  sorry

end Lasserre

end
