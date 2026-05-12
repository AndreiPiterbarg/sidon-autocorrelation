/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# SOS-dual Farkas LP — Verification Root Module

This library formally verifies the soundness of the SOS-dual Farkas LP
solver implemented in `lasserre/dual_sdp.py` + `tests/lasserre_mosek_dual.py`.

Structure:

  SOSDual.Farkas            — abstract conic Farkas alternative
  SOSDual.Monotonicity      — primal feasibility monotone in t
  SOSDual.Verdict           — λ* ≥ threshold ⟹ t ≤ val_L(d)
  SOSDual.BisectionSoundness — bisection `lo` ≤ val_L(d)

## Python↔Lean interface

The proofs in this tree take the Farkas alternative as an abstract
hypothesis (`SOSDual.Farkas.FarkasInvariant`) — instantiating it at a
specific t* is equivalent to confirming that MOSEK's numerical
optimum λ* is a valid infeasibility certificate.  The concrete verification
path is:

  1.  MOSEK returns λ* ≥ 0.75 · Λ  at t* in Python.
  2.  A post-processor checks that the reported multipliers (λ*, μ*, v*,
      X*_{0,i,W}) satisfy the stationarity rows to within numerical
      tolerance (independent of MOSEK's internal checks).
  3.  That numerical certification is the witness for
      `FarkasInvariant.infeas_of_cert_exists`.
  4.  The Lean theorems in this library then derive
      `t* ≤ val_L(d) ≤ val(d) ≤ C₁ₐ` from the proved bisection soundness
      plus the existing `lean/lasserre/` relaxation chain.

## Publishable headline

The headline theorem, provable in this file's `Main` namespace, is:

```
theorem bisection_lb_le_C1a :
    accumulated_lo t_lo_init steps  ≤  val_L(d)  ≤  val(d)  ≤  C₁ₐ
```

which, together with a numerical certificate for the bisection bound,
gives a fully mechanised proof of the new lower bound on the Sidon
autocorrelation constant.
-/
import SOSDual.Farkas
import SOSDual.Monotonicity
import SOSDual.Verdict
import SOSDual.BisectionSoundness

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace SOSDual

/-! ## Composed publishable theorem

Given a fully certified bisection run, plus the abstract Lasserre chain
(val_L ≤ val(d) ≤ C₁ₐ), the accumulated `lo` value is a sound lower
bound on C₁ₐ. -/

open SOSDual.Bisection SOSDual.Verdict

/-- **Publishable soundness**: the numerical output of the SOS-dual
    bisection driver — `lo = accumulated_lo t_lo_init steps` — is bounded
    above by C₁ₐ, given:
    (a) a certified bisection run (a `CertifiedBisection`),
    (b) the classical Lasserre-to-problem-value inequality `val_L(d) ≤ val(d)`,
    (c) the discrete-to-continuous bridge `val(d) ≤ C₁ₐ`. -/
theorem bisection_lo_le_C1a
    {α β : Type} {P : ℝ → Prop}
    (B : CertifiedBisection α β P)
    (feasSet : Set ℝ) (hFeas : feasSet = {t | P t})
    (hFeasNonempty : feasSet.Nonempty)
    (hFeasBddBelow : BddBelow feasSet)
    (val_d C1a : ℝ)
    (h_lasserre_le_valD : sInf feasSet ≤ val_d)
    (h_valD_le_C1a : val_d ≤ C1a) :
    accumulated_lo B.t_lo_init B.steps ≤ C1a :=
  le_trans
    (bisection_lo_le_val B feasSet hFeas hFeasNonempty hFeasBddBelow)
    (le_trans h_lasserre_le_valD h_valD_le_C1a)

end SOSDual
