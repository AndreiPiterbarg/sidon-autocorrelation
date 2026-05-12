import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom FullConsistencyEqualitySound : BaseConfig -> Prop
axiom PartialConsistencyInequalitySound : BaseConfig -> Prop
axiom BuildConsistencySparseMatchesSemantics : BaseConfig -> Prop
axiom PartialEqualityWouldStrengthenModel : BaseConfig -> Prop

/-- Covers full equality rows from `tests/lasserre_highd.py:_build_consistency_sparse`. -/
theorem full_consistency_equality_sound (cfg : BaseConfig) :
    ConsistencySystemMatchesCode cfg ->
    FullConsistencyEqualitySound cfg := by
  sorry

/-- Covers partial inequality rows from `tests/lasserre_highd.py:_build_consistency_sparse`. -/
theorem partial_consistency_inequality_sound (cfg : BaseConfig) :
    ConsistencySystemMatchesCode cfg ->
    PartialConsistencyInequalitySound cfg := by
  sorry

/-- Exact proof obligation matching the sparse consistency builder used in the
Round 0 model and the main CG model. -/
theorem build_consistency_sparse_matches_semantics (cfg : BaseConfig) :
    ConsistencySystemMatchesCode cfg ->
    FullConsistencyEqualitySound cfg ->
    PartialConsistencyInequalitySound cfg ->
    BuildConsistencySparseMatchesSemantics cfg := by
  sorry

/-- Guardrail theorem documenting why missing-child equality is forbidden. -/
theorem partial_equality_would_strengthen_model (cfg : BaseConfig) :
    PartialEqualityWouldStrengthenModel cfg := by
  sorry

end Lasserre

end
