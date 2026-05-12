import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom ReducedSetContainsAllLowDegreeMoments : BaseConfig -> Prop
axiom ReducedSetContainsCliqueMomentEntries : BaseConfig -> Prop
axiom ReducedSetContainsCliqueLocalizingEntries : BaseConfig -> Prop
axiom FullEqualityChainHoldsUpToDegreeTwoKMinusTwo : BaseConfig -> Prop
axiom PrecomputeHighDIsSemanticallyComplete : BaseConfig -> Prop

/-- Covers `tests/lasserre_highd.py:_build_reduced_moment_set` for the global
degree `<= 2k-1` chain. -/
theorem reduced_set_contains_all_low_degree_moments (cfg : BaseConfig) :
    ReducedMomentSetMatchesCode cfg ->
    ReducedSetContainsAllLowDegreeMoments cfg := by
  sorry

/-- Covers clique moment entries inserted by `_build_reduced_moment_set`. -/
theorem reduced_set_contains_clique_moment_entries (cfg : BaseConfig) :
    ReducedMomentSetMatchesCode cfg ->
    ReducedSetContainsCliqueMomentEntries cfg := by
  sorry

/-- Covers clique localizing entries inserted by `_build_reduced_moment_set`. -/
theorem reduced_set_contains_clique_localizing_entries (cfg : BaseConfig) :
    ReducedMomentSetMatchesCode cfg ->
    ReducedSetContainsCliqueLocalizingEntries cfg := by
  sorry

/-- This is the exact tightness-critical statement from the code comments:
all degree `<= 2k-2` consistency rows have their full equality chain present. -/
theorem reduced_set_gives_full_low_degree_chain (cfg : BaseConfig) :
    ReducedMomentSetMatchesCode cfg ->
    ReducedSetContainsAllLowDegreeMoments cfg ->
    FullEqualityChainHoldsUpToDegreeTwoKMinusTwo cfg := by
  sorry

/-- Covers `tests/lasserre_highd.py:_precompute_highd`. -/
theorem precompute_highd_semantically_complete (cfg : BaseConfig) :
    ReducedMomentSetMatchesCode cfg ->
    ReducedSetContainsAllLowDegreeMoments cfg ->
    ReducedSetContainsCliqueMomentEntries cfg ->
    ReducedSetContainsCliqueLocalizingEntries cfg ->
    PrecomputeHighDIsSemanticallyComplete cfg := by
  sorry

end Lasserre

end
