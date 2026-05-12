/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# Verdict Soundness: MOSEK `λ*` → Lower Bound on val_L(d)

This file composes the Farkas alternative (SOSDual.Farkas) with the
feasibility monotonicity (SOSDual.Monotonicity) to prove the central
verdict-soundness theorem used by `lasserre/dual_sdp.py::solve_dual_task`:

    At a probed t*, if the Farkas LP's max-λ solution satisfies
    λ* = 1  (or more generally λ* ≥ infeas_threshold · Λ),
    THEN val_L(d) ≥ t*.

The number-crunching side is simple:
  (i) MOSEK returned an optimal certificate (λ*, μ*, v*, X*_{0,i,W}, …)
      with λ* > 0.  Encode this as "the Farkas cert set is non-empty".
  (ii) By the conic Farkas alternative, the primal is empty at t*.
  (iii) By feasibility monotonicity, no t ≤ t* is primal-feasible, so
        t* ≤ val_L(d).

## Code correspondence

- `lasserre/dual_sdp.py::solve_dual_task` returns `verdict='infeas'` when
  λ* ≥ 0.75 · Λ.  The `solve_mosek_dual` bisection driver advances `lo := t*`
  upon an `infeas` verdict — which is sound precisely because of the theorem
  below.
-/
import Mathlib
import SOSDual.Farkas
import SOSDual.Monotonicity

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace SOSDual.Verdict

open SOSDual.Farkas SOSDual.Monotonicity

/-- A "verified MOSEK verdict at t*" is a Farkas certificate at t* plus
    evidence that feasibility is monotone in t.  Together these imply the
    numerical bound t* ≤ val_L(d). -/
structure VerifiedInfeas
    (α β : Type) (P : ℝ → Prop) (t_star : ℝ) where
  /-- The Farkas alternative instantiated at t*. -/
  inv : FarkasInvariant α β
  /-- A concrete certificate u in the Farkas LP feasible set. -/
  cert : β
  cert_mem : cert ∈ inv.certs
  /-- The primal feasibility predicate at t* coincides with `inv.primal` being non-empty. -/
  primal_eq_P : inv.primal.Nonempty ↔ P t_star
  /-- Feasibility is monotone in t for the family P. -/
  mono : ∀ {t t'}, t ≤ t' → P t → P t'

/-- From a verified infeas verdict at t*, every feasible t is ≥ t*. -/
theorem t_star_le_every_feasible
    {α β : Type} {P : ℝ → Prop} {t_star : ℝ}
    (v : VerifiedInfeas α β P t_star)
    {s : ℝ} (hs : P s) :
    t_star ≤ s := by
  -- infeasible at t_star from the certificate
  have hEmpty : ¬ v.inv.primal.Nonempty :=
    v.inv.infeas_of_cert_exists ⟨v.cert, v.cert_mem⟩
  have hP_infeas : ¬ P t_star := fun hP =>
    hEmpty (v.primal_eq_P.mpr hP)
  exact t_le_every_feasible v.mono hP_infeas hs

/-- From a verified infeas verdict at t*, `t* ≤ val_L(d)` where val_L(d)
    is the infimum over the feasibility set of t-values. -/
theorem t_star_le_val
    {α β : Type} {P : ℝ → Prop} {t_star : ℝ}
    (v : VerifiedInfeas α β P t_star)
    (feasSet : Set ℝ) (hFeas : feasSet = {t | P t})
    (hFeasNonempty : feasSet.Nonempty)
    (_hFeasBddBelow : BddBelow feasSet) :
    t_star ≤ sInf feasSet := by
  apply le_csInf hFeasNonempty
  intro t ht
  rw [hFeas] at ht
  exact t_star_le_every_feasible v ht

/-- Publishable form: composed with the existing Lasserre.Relaxation chain
    (lb ≤ val_L ≤ val(d) ≤ C₁ₐ), a verified infeas verdict at t* certifies
    `t* ≤ val_L(d) ≤ val(d) ≤ C₁ₐ`. -/
theorem t_star_le_C1a
    {α β : Type} {P : ℝ → Prop} {t_star : ℝ}
    (v : VerifiedInfeas α β P t_star)
    (feasSet : Set ℝ) (hFeas : feasSet = {t | P t})
    (hFeasNonempty : feasSet.Nonempty)
    (hFeasBddBelow : BddBelow feasSet)
    (val_d C1a : ℝ)
    (h_lasserre_le_valD : sInf feasSet ≤ val_d)
    (h_valD_le_C1a : val_d ≤ C1a) :
    t_star ≤ C1a :=
  le_trans (t_star_le_val v feasSet hFeas hFeasNonempty hFeasBddBelow)
    (le_trans h_lasserre_le_valD h_valD_le_C1a)

end SOSDual.Verdict
