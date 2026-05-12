"""Farkas-style certificate for the multifreq MO 2.17 master inequality.

Per the brief: "For the optimal M* found, produce a Farkas-style dual
certificate (multipliers lambda_217, lambda_214_j, lambda_pos) such that
the master inequality becomes a verifiable algebraic identity."

Status (2026-04-29):  the QP optimum produced by ``qp_solver.main_report``
is M* = 1.27484, IDENTICAL to MV's 1.27481 (to 4 decimal places) and
strictly BELOW the CS 2017 threshold 1.2802.  The L2.17 strong
constraint is *not active* at the joint argmax, so the dual multiplier
``lambda_217`` would be 0 in the Farkas certificate; the certificate
reduces to MV's own Farkas-style certificate for eq. (10).

Per the brief's FAILURE MODE clause:

    "If the QP optimum gives M* < 1.28 (no improvement over MV), document
    clearly in derivation.md why ... Do NOT publish a non-improvement as
    a theorem."

We therefore do NOT produce a Farkas certificate that asserts M* > 1.2802;
that would be false.  Instead we produce the **diagnostic certificate**:
a numerical demonstration (validated at dps = 80) that

    *  L2.17 strong constraint slacks > 0 at the joint argmax;
    *  P-side gap > 0 at the joint argmax (Prop 2.11 inactive);
    *  MV-side gap = 0 at the joint argmax (binding constraint = MV).

This is the honest record:  the only certificate that holds is "no
improvement over MV's 1.27481".

If ever the brief is reformulated and a true lift is achieved, the
function ``produce_farkas_certificate`` below should be filled in with
the genuine multipliers; for now it raises NotImplementedError with a
pointer to the failure-mode write-up.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import mpmath as mp
from mpmath import mpf

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qp_solver import (
    MVSetup, kernel_data,
    mv_master_gap, prop211_m3_lower_bound, lemma_214_box,
    bisect_M_lower_bound, _bisect_p211_only,
)


# =============================================================================
# Diagnostic certificate (failure mode)
# =============================================================================


@dataclass
class DiagnosticCertificate:
    """A complete record of the QP-optimum state at the joint argmax.

    Validates at mpmath dps=80 that:
      * L2.17 strong slacks (right and left) are non-negative;
      * P-side gap is non-negative (Prop 2.11 satisfied);
      * MV-side gap is approximately zero (active binding).

    Records mu(M*), the kernel data, and all intermediate sums.
    """
    M_star: mpf
    c1: mpf
    s1: mpf
    c2: mpf
    s2: mpf

    # Lemma 2.14 slacks (mu(M) - z_n^2 >= 0)
    L214_slack_1: mpf
    L214_slack_2: mpf
    # Lemma 2.17 slacks
    L217_R_slack: mpf      # 2 c1 - 1 - c2 >= 0
    L217_L_slack: mpf      # c2 - (2 z1sq - 1) >= 0

    # MV master inequality gap (>= 0 means satisfied; should be ~0 at argmax)
    mv_gap: mpf
    # Prop 2.11 + Hoelder gap (>= 0 means satisfied)
    p_gap: mpf

    # Kernel data used for Prop 2.11
    K_choice: str
    K0: mpf
    K1: mpf
    K2: mpf
    tail_43: mpf

    # MV setup (essentials)
    delta: mpf
    u: mpf
    a_gain: mpf
    k1_mv: mpf
    k2_mv: mpf
    K2_mv: mpf
    Lambda: mpf            # 2/u + a_gain

    def residual(self) -> mpf:
        """A single scalar residual: max(violation across 6 constraints).

        Should be approximately 0 (within numerical bisection tolerance) at
        the QP-optimum M*.  A negative value means a constraint is violated
        (catastrophic; would invalidate the bisection).
        """
        # Active binding: MV gap should be exactly 0 (M* is the smallest M).
        # Other slacks should be >= 0.
        violations = []
        if self.L214_slack_1 < 0:
            violations.append(("L2.14[1]", self.L214_slack_1))
        if self.L214_slack_2 < 0:
            violations.append(("L2.14[2]", self.L214_slack_2))
        if self.L217_R_slack < 0:
            violations.append(("L2.17.R", self.L217_R_slack))
        if self.L217_L_slack < 0:
            violations.append(("L2.17.L", self.L217_L_slack))
        if self.p_gap < 0:
            violations.append(("Prop2.11", self.p_gap))
        # MV gap should be ~0; allow tolerance
        if violations:
            return min(v[1] for v in violations)
        # No violation; report MV gap (the binding active value)
        return abs(self.mv_gap)

    def summary(self) -> str:
        return f"""DiagnosticCertificate (failure-mode):
  M*                        = {mp.nstr(self.M_star, 12)}
  argmax (c1, s1, c2, s2)   = ({mp.nstr(self.c1, 8)}, {mp.nstr(self.s1, 8)},
                                {mp.nstr(self.c2, 8)}, {mp.nstr(self.s2, 8)})
  L2.14 slack [n=1]         = {mp.nstr(self.L214_slack_1, 6)}   (~0 means active)
  L2.14 slack [n=2]         = {mp.nstr(self.L214_slack_2, 6)}
  L2.17 R slack             = {mp.nstr(self.L217_R_slack, 6)}   (>0 means INACTIVE)
  L2.17 L slack             = {mp.nstr(self.L217_L_slack, 6)}
  MV master gap             = {mp.nstr(self.mv_gap, 6)}         (~0 means active binding)
  Prop 2.11 gap             = {mp.nstr(self.p_gap, 6)}          (>0 means INACTIVE)
  Kernel ({self.K_choice})  K0={mp.nstr(self.K0, 6)} K1={mp.nstr(self.K1, 6)} K2={mp.nstr(self.K2, 6)}
                               ||_3 K^||_(4/3) = {mp.nstr(self.tail_43, 6)}
  Residual                  = {mp.nstr(self.residual(), 6)}

Interpretation:
  - The MV master gap is the binding (active) constraint; its slack is ~0.
  - L2.17 strong is INACTIVE: would be active iff c2 = 2 c1 - 1 (slack = 0).
  - Prop 2.11 is INACTIVE: would be binding iff p_gap = 0.
  - L2.14 [n=1] is on the boundary at the MV optimum (z1 = sqrt(mu(M*))).

This certificate confirms the FAILURE MODE: no lift above 1.2802 is
implied by the QP combination; M* = MV's 1.27481.  The certificate
itself only certifies the MV bound, not any improvement.
"""


def build_diagnostic_certificate(
    K_choice: str = "step",
    dps_validate: int = 80,
    M_lo: mpf = mpf("1.20"),
    M_hi: mpf = mpf("1.40"),
    tol: mpf = mpf("1e-14"),
) -> DiagnosticCertificate:
    """Run the QP, find joint M*, validate slacks at dps_validate."""
    # First find M_star at dps=40
    mp.mp.dps = 40
    mv = MVSetup.build()
    K_data = kernel_data(K_choice)
    M_star, info = bisect_M_lower_bound(
        mv, K_data,
        M_lo=M_lo, M_hi=M_hi, tol=tol,
        verbose=False,
    )
    c1, s1, c2, s2 = info["argmax"]

    # Re-validate at higher precision
    mp.mp.dps = dps_validate
    mv = MVSetup.build()
    K_data = kernel_data(K_choice)

    M_star = mpf(M_star)
    c1, s1, c2, s2 = mpf(c1), mpf(s1), mpf(c2), mpf(s2)

    mu = lemma_214_box(M_star)
    z1sq = c1 * c1 + s1 * s1
    z2sq = c2 * c2 + s2 * s2

    L214_slack_1 = mu - z1sq
    L214_slack_2 = mu - z2sq
    L217_R_slack = (2 * c1 - 1) - c2
    L217_L_slack = c2 - (2 * z1sq - 1)

    mv_gap = mv_master_gap(M_star, c1, s1, c2, s2, mv)
    p_gap = M_star - prop211_m3_lower_bound(M_star, c1, s1, c2, s2, K_data)

    return DiagnosticCertificate(
        M_star=M_star, c1=c1, s1=s1, c2=c2, s2=s2,
        L214_slack_1=L214_slack_1,
        L214_slack_2=L214_slack_2,
        L217_R_slack=L217_R_slack,
        L217_L_slack=L217_L_slack,
        mv_gap=mv_gap,
        p_gap=p_gap,
        K_choice=K_choice,
        K0=K_data["K0"], K1=K_data["K1"], K2=K_data["K2"],
        tail_43=K_data["tail_43"],
        delta=mv.delta, u=mv.u, a_gain=mv.a_gain,
        k1_mv=mv.k1_mv, k2_mv=mv.k2_mv, K2_mv=mv.K2_mv,
        Lambda=mv.lhs_target,
    )


# =============================================================================
# True Farkas certificate (NOT POSSIBLE in the failure mode)
# =============================================================================


def produce_farkas_certificate(*args, **kwargs):
    """Stub: a Farkas certificate would assert M* > 1.2802 algebraically.

    Since the QP optimum is M* = 1.27484 < 1.2802, no such certificate
    exists.  See ``derivation.md`` §6 for the proof of why the lift is
    structurally impossible in this framework.

    Calling this raises NotImplementedError with a pointer to the
    failure-mode write-up.
    """
    raise NotImplementedError(
        "M* = 1.27484 < 1.2802 (failure mode).  No Farkas certificate "
        "asserting M* > 1.2802 exists.  See derivation.md §6 for proof "
        "that the lift is impossible by Prop 2.11 m=3 + L2.14 + L2.17 + MV."
    )


# =============================================================================
# Top-level main
# =============================================================================


if __name__ == "__main__":
    print("Building diagnostic certificate at dps=80 ...")
    print()
    for K_choice in ("step", "pm"):
        cert = build_diagnostic_certificate(K_choice=K_choice)
        print(cert.summary())
        print()
