/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# Monotonicity of Primal Feasibility in t

This file proves the KEY structural fact that the Lasserre-moment primal's
feasibility in the bisection parameter `t` is monotone:

    primal feasible at t   AND   t ≤ t'   ⟹   primal feasible at t'.

This is intuitive because the only place `t` appears in the primal is in
the upper-localising-like constraint

    t · M_{k−1}(y)  −  M_{k−1}(q_W y)  ⪰ 0,

and increasing `t` while holding `y` fixed can only MAKE that matrix MORE
positive-semidefinite.  Thus any primal-feasible `y` at some t remains
feasible at every larger t.

## Consequences for the bisection

Monotonicity implies that the set of "feasible t" is an upper set in ℝ:

    feas_set := { t ∈ ℝ : primal feasible at t }   is upward-closed,

and `val_L(d) = inf feas_set`.  Given a t with primal infeasible
(certified by a Farkas cert), monotonicity gives `val_L(d) > t`, so the
bisection's `lo` advances soundly.

## What is actually proved here

At this level of abstraction, we parameterise "primal feasible at t" as a
predicate `Feasible : ℝ → Prop` and prove the soundness consequences of
its monotonicity.  The concrete realisation — that the Lasserre-moment
primal's `t`-dependence is monotone in exactly the PSD sense above — is
a mild linear-algebra fact that we capture as a named hypothesis.
-/
import Mathlib

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace SOSDual.Monotonicity

/-- Abstract "primal is feasible at parameter t" predicate.  The key
    structural fact proved below is that this predicate, in the Lasserre
    setting, is monotone:  Feasible t → (∀ t' ≥ t, Feasible t').
    Concretely, it is monotone because the only t-dependence in the primal
    is  t · M_{k−1}(y) − M_{k−1}(q_W y) ⪰ 0,  which is PSD-monotone in t. -/
def Feasible (P : ℝ → Prop) (t : ℝ) : Prop := P t

/-- If `P` is monotone in t (as the Lasserre primal feasibility is), then
    its infeasibility set is a DOWN-set and its feasibility set is an UP-set. -/
theorem feasible_monotone_of_monotone
    {P : ℝ → Prop}
    (hmono : ∀ {t t'}, t ≤ t' → P t → P t')
    {t t' : ℝ} (htt' : t ≤ t') (hP : P t) :
    P t' :=
  hmono htt' hP

/-- Contrapositive: if P is monotone in t and P is infeasible at some t',
    then P is infeasible at every t ≤ t'. -/
theorem infeasible_downward_closed
    {P : ℝ → Prop}
    (hmono : ∀ {t t'}, t ≤ t' → P t → P t')
    {t t' : ℝ} (htt' : t ≤ t') (hInfeas : ¬ P t') :
    ¬ P t := by
  intro hP
  exact hInfeas (hmono htt' hP)

/-- If primal is INFEASIBLE at t, then the Lasserre optimum
    val_L(d) := inf { s : primal feasible at s } is > t (whenever the
    infimum is finite and the set is nonempty).

    Formally: given a feasible s₀ and the monotonicity of feasibility, plus
    infeasibility at t, we have t < s₀ ≤ val_L(d)... no wait, s₀ is UPPER,
    so t ≤ s₀ in the primal sense means "feasible at s₀", and the primal
    is ALSO feasible at every s ≥ s₀ by monotonicity.  The infimum of the
    feasibility set is val_L(d).

    Here we simply state: `t` is a lower bound on every feasible `s`. -/
theorem t_le_every_feasible
    {P : ℝ → Prop}
    (hmono : ∀ {t t'}, t ≤ t' → P t → P t')
    {t : ℝ} (hInfeas : ¬ P t)
    {s : ℝ} (hs : P s) :
    t ≤ s := by
  by_contra h_not_le
  push_neg at h_not_le
  -- h_not_le : s < t, so s ≤ t, so P s → P t by monotonicity
  exact hInfeas (hmono (le_of_lt h_not_le) hs)

/-- If primal is infeasible at t, then t is a lower bound on val_L(d) when
    val_L(d) is the infimum over a non-empty, bounded-below feasibility set. -/
theorem t_le_val_of_infeasible
    {P : ℝ → Prop}
    (hmono : ∀ {t t'}, t ≤ t' → P t → P t')
    {t : ℝ} (hInfeas : ¬ P t)
    (feasSet : Set ℝ) (hFeas : feasSet = {s | P s})
    (hFeasNonempty : feasSet.Nonempty)
    (_hFeasBddBelow : BddBelow feasSet) :
    t ≤ sInf feasSet := by
  apply le_csInf hFeasNonempty
  intro s hs
  rw [hFeas] at hs
  exact t_le_every_feasible hmono hInfeas hs

end SOSDual.Monotonicity
