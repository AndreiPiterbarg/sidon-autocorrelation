"""DEPRECATED: superseded by bnb.rigor_replay.

This module's verify_* functions are kept as documentation of the
per-leaf rigor idea but are not invoked by the B&B pipeline. Do not
add new callers. See bnb.py::rigor_replay for the live rigor gate.

Historical description
----------------------
Every leaf accepted by the float64 fast path was re-checked here in
fractions.Fraction arithmetic. A leaf certified by window W at target c
passed iff
    lb_rat(B, W)  >=  c_q
for the exact rational value of lb. Disagreements between the float64
and Fraction bounds indicated a bug (never a rounding issue), and were
reported upstream.

Two bounds were re-evaluated exactly:
  * natural interval:    lb_nat = scale_q * sum_{(i,j) in pairs_all} lo_i lo_j
  * McCormick at mu*:    given a candidate mu* (rational), check
      lb_mcc = scale_q * (g_q . mu* + c0_q)
    AND check mu* is primal-feasible (sum = 1, lo <= mu* <= hi, sym).
    This verifies the *value* of the McCormick function at mu*, which
    is a valid lower bound on mu*^T M_W mu* -- but NOT automatically
    on the LP minimum. To rigorously certify the LP minimum we either
    (a) rely on the natural bound (always a valid LB over the box),
    (b) accept that the McCormick fast path may be over-aggressive and
        split further when Fractions disagree.

Policy: first try the natural bound; if it certified, the leaf was
rigorous. Else, retry with an even tighter box produced by further
splitting. This is ALWAYS rigorous; the cost is only extra splits.
"""
from __future__ import annotations

import warnings
from fractions import Fraction
from typing import List, Sequence, Tuple

from .bound_eval import bound_natural_exact, _mccormick_linear_exact
from .windows import WindowMeta


def verify_leaf_natural(
    lo_q: Sequence[Fraction], hi_q: Sequence[Fraction],
    w: WindowMeta, target_q: Fraction,
) -> Tuple[bool, Fraction]:
    """Return (certified, lb_rat) using the natural interval bound."""
    warnings.warn(
        "rigorous_check is deprecated; use bnb.rigor_replay",
        DeprecationWarning,
        stacklevel=2,
    )
    lb_rat = bound_natural_exact(lo_q, hi_q, w)
    return lb_rat >= target_q, lb_rat


def verify_leaf_mccormick_at_point(
    lo_q: Sequence[Fraction], hi_q: Sequence[Fraction],
    w: WindowMeta, target_q: Fraction, mu_q: Sequence[Fraction],
    sym_cuts: Sequence[Tuple[int, int]] = (),
) -> Tuple[bool, Fraction]:
    """Return (certified, lb_rat) evaluating the McCormick function at
    a CANDIDATE mu_q (rational). The caller is responsible for showing
    mu_q is the LP optimiser; verify_leaf_mccormick_primal_feasible
    checks the easy feasibility part.
    """
    warnings.warn(
        "rigorous_check is deprecated; use bnb.rigor_replay",
        DeprecationWarning,
        stacklevel=2,
    )
    if not verify_mccormick_primal_feasible(lo_q, hi_q, mu_q, sym_cuts):
        return False, Fraction(0)
    g_q, c0_q = _mccormick_linear_exact(lo_q, w)
    lb_rat = w.scale_q * (sum(g_q[i] * mu_q[i] for i in range(len(mu_q))) + c0_q)
    return lb_rat >= target_q, lb_rat


def verify_mccormick_primal_feasible(
    lo_q: Sequence[Fraction], hi_q: Sequence[Fraction],
    mu_q: Sequence[Fraction],
    sym_cuts: Sequence[Tuple[int, int]] = (),
) -> bool:
    """Check that mu_q is feasible: sum = 1, in box, satisfies sym."""
    warnings.warn(
        "rigorous_check is deprecated; use bnb.rigor_replay",
        DeprecationWarning,
        stacklevel=2,
    )
    if sum(mu_q) != Fraction(1):
        return False
    d = len(mu_q)
    for i in range(d):
        if mu_q[i] < lo_q[i] or mu_q[i] > hi_q[i]:
            return False
    for (i, j) in sym_cuts:
        if mu_q[i] > mu_q[j]:
            return False
    return True
