/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# Relaxation Soundness for Optimization Problems

This file proves the abstract mathematical facts underlying relaxation-based
lower bounds. The key idea: weakening constraints enlarges the feasible set,
which can only decrease (or preserve) the infimum of the objective.

## Code correspondence
- `lasserre_highd.py` Soundness Theorem (lines 29-46): "lb ≤ val(d)"
- `run_scs_direct.py` bisection loop: feasibility at t implies SDP_value ≤ t
- `lasserre/cliques.py`: sparse relaxation is weaker than full → sound

## Chain of relaxations
```
lb_bisection ≤ SDP_clique_value ≤ SDP_full_value ≤ val(d) ≤ C₁ₐ
```
Each ≤ follows from: the right-hand problem is a restriction of the left.
-/
import Mathlib

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Set

namespace Lasserre.Relaxation

/-! ## Abstract relaxation theory -/

/-- Core relaxation lemma: if S ⊆ T (S is the original feasible set,
    T is the relaxation's feasible set), then sInf T ≤ sInf S.

    This captures: weakening constraints enlarges the feasible set,
    which can only decrease the infimum.

    Application to Lasserre: the clique-restricted moment conditions are
    NECESSARY for a true measure, so every true moment vector is in the
    relaxation's feasible set. -/
theorem relaxation_infimum_le {S T : Set ℝ} (h : S ⊆ T)
    (hS : S.Nonempty) (hT_bdd : BddBelow T) :
    sInf T ≤ sInf S :=
  csInf_le_csInf hT_bdd hS h

/-- If a specific point x₀ is feasible with objective ≤ v,
    then the optimum is ≤ v.

    This captures: if true moments y* achieve val(d), and y* is feasible
    for the SDP, then SDP_value ≤ val(d).

    Used in: `run_scs_direct.py` bisection — if feasible at t_val,
    then the SDP value ≤ t_val. -/
theorem opt_le_of_feasible {S : Set ℝ} {v : ℝ}
    (hv : v ∈ S) (hS_bdd : BddBelow S) :
    sInf S ≤ v :=
  csInf_le hS_bdd hv

/-! ## Relaxation chain -/

/-- Transitivity of relaxation bounds: if S₁ ⊆ S₂ ⊆ S₃, then
    sInf S₃ ≤ sInf S₂ ≤ sInf S₁.

    Corresponds to the chain:
    SDP_clique ⊆ SDP_full ⊆ {all achievable objectives}
    ⟹ lb_clique ≤ lb_full ≤ val(d) -/
theorem relaxation_chain {S₁ S₂ S₃ : Set ℝ}
    (h₁₂ : S₁ ⊆ S₂) (h₂₃ : S₂ ⊆ S₃)
    (hS₁ : S₁.Nonempty) (hS₂_bdd : BddBelow S₂) (hS₃_bdd : BddBelow S₃) :
    sInf S₃ ≤ sInf S₁ := by
  have h₁₃ : S₁ ⊆ S₃ := Subset.trans h₁₂ h₂₃
  exact csInf_le_csInf hS₃_bdd hS₁ h₁₃

/-- The bisection lower bound is ≤ the SDP optimum.

    In the bisection, `lo` is set to values where feasibility FAILS.
    Since the SDP optimum is the boundary between feasible and infeasible,
    lo ≤ SDP_optimum.

    More precisely: if for all t < lb, the SDP at t is infeasible,
    then lb ≤ SDP_value. This follows from: SDP_value is feasible. -/
theorem bisection_lb_le_opt {S : Set ℝ} {lb : ℝ}
    (h_infeas : ∀ t ∈ S, lb ≤ t) (hS : S.Nonempty) (hS_bdd : BddBelow S) :
    lb ≤ sInf S := by
  exact le_csInf hS h_infeas

/-! ## Specific application to moment relaxations -/

/-- Adding constraints can only increase the optimum (for minimization).
    Removing a PSD constraint is a relaxation.

    This justifies: clique-restricted PSD ⊆ full PSD constraints ⟹
    the clique relaxation's optimum ≤ full Lasserre optimum. -/
theorem fewer_constraints_smaller_opt {FeasFull FeasRelax : Set ℝ}
    (h : FeasFull ⊆ FeasRelax)
    (hFull : FeasFull.Nonempty) (hRelax_bdd : BddBelow FeasRelax) :
    sInf FeasRelax ≤ sInf FeasFull :=
  relaxation_infimum_le h hFull hRelax_bdd

/-- The full Lasserre relaxation chain: if lb is a lower bound on the SDP
    optimum, and the SDP optimum is ≤ val(d) (because true moments are
    feasible), then lb ≤ val(d).

    This is the mathematical core of the soundness proof in
    `lasserre_highd.py` lines 29-46. -/
theorem lb_le_valD {lb sdp_val valD : ℝ}
    (h_lb : lb ≤ sdp_val) (h_sdp : sdp_val ≤ valD) :
    lb ≤ valD :=
  le_trans h_lb h_sdp

end Lasserre.Relaxation
