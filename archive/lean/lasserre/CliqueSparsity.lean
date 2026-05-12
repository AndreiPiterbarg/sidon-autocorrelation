import lasserre.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

axiom BandedCliquesHaveRunningIntersection : BaseConfig -> Prop
axiom CliqueBasisSupportedInsideClique : BaseConfig -> Prop
axiom CliqueMomentPSDIsPrincipalSubmatrix : BaseConfig -> Prop
axiom CliqueLocalizingPSDIsPrincipalSubmatrix : BaseConfig -> Prop
axiom FullMkMinusOnePSDProvidesCrossCliqueCoupling : BaseConfig -> Prop
axiom SparseRelaxationWeakerThanFullRelaxation : BaseConfig -> Prop

/-- Covers `lasserre/cliques.py:_build_banded_cliques`. -/
theorem banded_cliques_have_running_intersection (cfg : BaseConfig) :
    CliqueSystemMatchesCode cfg ->
    BandedCliquesHaveRunningIntersection cfg := by
  sorry

/-- Covers `lasserre/cliques.py:_build_clique_basis`. -/
theorem clique_basis_supported_inside_clique (cfg : BaseConfig) :
    CliqueSystemMatchesCode cfg ->
    CliqueBasisSupportedInsideClique cfg := by
  sorry

/-- Covers the soundness of clique moment PSD constraints in both
`lasserre/cliques.py` and `tests/lasserre_highd.py`. -/
theorem clique_moment_psd_is_principal_submatrix (cfg : BaseConfig) :
    CliqueSystemMatchesCode cfg ->
    CliqueMomentPSDIsPrincipalSubmatrix cfg := by
  sorry

/-- Covers clique localizing PSD soundness. -/
theorem clique_localizing_psd_is_principal_submatrix (cfg : BaseConfig) :
    CliqueSystemMatchesCode cfg ->
    CliqueLocalizingPSDIsPrincipalSubmatrix cfg := by
  sorry

/-- Covers the extra full `M_(k-1)` PSD constraint added by the high-d code. -/
theorem full_mk_minus_one_psd_provides_cross_clique_coupling (cfg : BaseConfig) :
    CliqueSystemMatchesCode cfg ->
    FullMkMinusOnePSDProvidesCrossCliqueCoupling cfg := by
  sorry

/-- Formal obligation behind the statement "narrower bandwidth gives a weaker
but still sound relaxation". -/
theorem sparse_relaxation_weaker_than_full_relaxation (cfg : BaseConfig) :
    CliqueSystemMatchesCode cfg ->
    SparseRelaxationWeakerThanFullRelaxation cfg := by
  sorry

end Lasserre

end
