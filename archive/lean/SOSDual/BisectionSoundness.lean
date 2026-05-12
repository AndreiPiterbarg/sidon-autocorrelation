/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# Bisection Soundness for the SOS-dual Driver

This file proves that the bisection loop in
`tests/lasserre_mosek_dual.py::solve_mosek_dual` — and equivalently its
parallel-fan-out variant `solve_mosek_dual_parallel` — produces a provably
valid lower bound on `val_L(d)`.

## Bisection invariant

The driver maintains two bookkeeping variables `lo` and `hi` with the
loop-invariant:

    lo  is the maximum t at which we have SEEN a Farkas certificate (INFEAS),
    hi  is the minimum t at which we have SEEN a primal-feasible solution (FEAS).

On every step, after probing some `mid ∈ (lo, hi)`:

  - verdict `infeas`:  lo ← mid  (because Farkas cert at `mid` certifies val > mid)
  - verdict `feas`:    hi ← mid  (because feasibility at `mid` certifies val ≤ mid)
  - verdict `uncertain`:    (lo, hi) unchanged

The final `lo` is ≤ `val_L(d)` regardless of how many uncertain / feas steps
occur, because every advancement of `lo` is certified by a valid Farkas cert
(via `SOSDual.Verdict.t_star_le_val`).

## What is proved here

The abstract invariant: if a finite list of bisection steps is executed and
the accumulated `lo` is the max over all `infeas`-certified t*, then lo ≤ val.
-/
import Mathlib
import SOSDual.Farkas
import SOSDual.Verdict
import SOSDual.Monotonicity

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace SOSDual.Bisection

open SOSDual.Farkas SOSDual.Verdict SOSDual.Monotonicity

/-- A single bisection step is represented as:
    - a probed t value `t_i`,
    - a verdict: `true` = infeas (advances lo), `false` = feas or uncertain. -/
structure Step where
  t : ℝ
  infeas : Bool

/-- The final `lo` value after a list of steps is the max over all steps
    whose `infeas` flag is true (or a caller-supplied floor `t_lo` if no
    such step exists). -/
def accumulated_lo (t_lo : ℝ) (steps : List Step) : ℝ :=
  steps.foldl (fun acc step => if step.infeas then max acc step.t else acc) t_lo

/-- A "certified step set" is a collection of bisection steps together with a
    mapping from each `infeas`-flagged step to a Farkas-LP certificate
    witnessing primal-empty at that step's t. -/
structure CertifiedBisection (α β : Type) (P : ℝ → Prop) where
  t_lo_init : ℝ
  steps : List Step
  /-- The initial bracket floor is ≤ val (e.g., `0.5` is always a safe floor
      for the Sidon lower-bound problem since val ≥ 1). -/
  t_lo_safe : ∀ {s : ℝ}, P s → t_lo_init ≤ s
  /-- For each infeas-flagged step, we have a verified Farkas cert. -/
  certify : ∀ (step : Step), step ∈ steps → step.infeas = true →
    VerifiedInfeas α β P step.t

/-! ## Core soundness lemma -/

/-- Every infeas-flagged step's t is ≤ every feasible s. -/
theorem infeas_step_le_feasible
    {α β : Type} {P : ℝ → Prop}
    (B : CertifiedBisection α β P)
    (step : Step) (hstep : step ∈ B.steps) (hI : step.infeas = true)
    {s : ℝ} (hs : P s) :
    step.t ≤ s :=
  SOSDual.Verdict.t_star_le_every_feasible (B.certify step hstep hI) hs

/-- A clean form of the fold bound: given any seed t0 ≤ s and a predicate
    that every step in the list either does not advance or has a t ≤ s,
    the fold is ≤ s. -/
private lemma foldl_max_le
    (steps : List Step) (t0 s : ℝ)
    (h_t0_le : t0 ≤ s)
    (h_all_le : ∀ step ∈ steps, step.infeas = true → step.t ≤ s) :
    steps.foldl
      (fun acc step => if step.infeas then max acc step.t else acc) t0 ≤ s := by
  induction steps generalizing t0 with
  | nil =>
      simpa using h_t0_le
  | cons step rest ih =>
      simp only [List.foldl_cons]
      apply ih
      · -- new seed after processing `step` is ≤ s
        by_cases hinf : step.infeas = true
        · simp only [hinf, if_true]
          have h_step_le : step.t ≤ s :=
            h_all_le step List.mem_cons_self hinf
          exact max_le h_t0_le h_step_le
        · simp only [hinf]
          exact h_t0_le
      · -- the tail still satisfies the predicate
        intro s' hs' hinf'
        exact h_all_le s' (List.mem_cons_of_mem _ hs') hinf'

/-- The accumulated `lo` after any list of steps is ≤ every feasible s. -/
theorem accumulated_lo_le_feasible
    {α β : Type} {P : ℝ → Prop}
    (B : CertifiedBisection α β P)
    {s : ℝ} (hs : P s) :
    accumulated_lo B.t_lo_init B.steps ≤ s := by
  unfold accumulated_lo
  apply foldl_max_le
  · exact B.t_lo_safe hs
  · intro step hstep hinf
    exact infeas_step_le_feasible B step hstep hinf hs

/-! ## Headline theorem -/

/-- The final `lo` of a certified bisection run is ≤ val_L(d), taken as the
    infimum of the (non-empty, bounded-below) feasibility set. -/
theorem bisection_lo_le_val
    {α β : Type} {P : ℝ → Prop}
    (B : CertifiedBisection α β P)
    (feasSet : Set ℝ) (hFeas : feasSet = {t | P t})
    (hFeasNonempty : feasSet.Nonempty)
    (_hFeasBddBelow : BddBelow feasSet) :
    accumulated_lo B.t_lo_init B.steps ≤ sInf feasSet := by
  apply le_csInf hFeasNonempty
  intro t ht
  rw [hFeas] at ht
  exact accumulated_lo_le_feasible B ht

end SOSDual.Bisection
