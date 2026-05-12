"""Z/2 symmetry reduction via half-simplex cover (rigorous version).

Let sigma be the time-reversal involution on Delta_d,
    sigma(mu) := (mu_{d-1}, mu_{d-2}, ..., mu_0).
It is an involution (sigma^2 = id) and the window set W_d is
sigma-invariant as a SET, so the objective

    f(mu) := max_{W in W_d}  mu^T M_W mu

satisfies  f(sigma(mu)) = f(mu)  identically in mu (reversal of an
ordered pair (i, j) maps it to (d-1-j, d-1-i), whose sum is preserved
modulo the d -> 2d-1 shift -- which is exactly the map W -> sigma(W)).

====================================================================
ORBIT COVER (proof-verified, does NOT depend on Kakutani / convexity).
====================================================================

Define the HALF-SIMPLEX
    H_d := { mu in Delta_d : mu_0 <= mu_{d-1} }.

PROPOSITION.  min_{mu in H_d} f(mu) = min_{mu in Delta_d} f(mu).

Proof.  For every mu in Delta_d, either (a) mu_0 <= mu_{d-1},
putting mu in H_d, or (b) mu_0 > mu_{d-1}, in which case
sigma(mu)_0 = mu_{d-1} < mu_0 = sigma(mu)_{d-1}, putting sigma(mu)
in H_d. At equality mu_0 = mu_{d-1} both mu and sigma(mu) are in H_d.
Hence H_d meets every orbit of sigma. Since f is constant on orbits,

    min_{mu in H_d} f = min_{orbits hit by H_d} f = min_{Delta_d} f. QED.

====================================================================
Why a FINER cut (mu_i <= mu_{d-1-i} for ALL i) would NOT be rigorous.
====================================================================

Consider d = 4 and mu = (0.3, 0.1, 0.4, 0.2). The pair (0, 3) has
mu_0 = 0.3 > mu_3 = 0.2, and sigma(mu) = (0.2, 0.4, 0.1, 0.3) has
the pair (1, 2) violating sigma(mu)_1 = 0.4 > sigma(mu)_2 = 0.1.
Neither representative satisfies ALL pair inequalities, so the "all
pairs" half-simplex does NOT cover this orbit. Using it would require
the symmetric-minimiser theorem -- which for the DISCRETE val(d) goes
via Kakutani on the minimiser set, whose convexity fails because
max_W (mu)^T M_W mu is a max of INDEFINITE quadratics.

The single-pair cut {mu_0 <= mu_{d-1}} sidesteps this difficulty.
"""
from __future__ import annotations

from typing import List, Tuple


def half_simplex_cuts(d: int) -> List[Tuple[int, int]]:
    """Return the list of (i, j) pairs defining the half-simplex
    H_d := {mu_i <= mu_j for (i, j) in cuts}.

    For rigor we use exactly ONE cut -- the (0, d-1) pair -- so H_d
    contains at least one representative of every sigma-orbit without
    relying on the (non-self-contained) symmetric-minimiser theorem.
    """
    if d < 2:
        return []
    return [(0, d - 1)]


def sigma_pairs(d: int) -> List[Tuple[int, int]]:
    """Return (i, sigma(i)) for i = 0..d-1 (including fixed points)."""
    return [(i, d - 1 - i) for i in range(d)]


def box_outside_hd(box) -> bool:
    """Return True iff the box lies STRICTLY outside the half-simplex
    H_d = {mu : mu_0 <= mu_{d-1}}, i.e. every mu in the box has
    mu_0 > mu_{d-1}.

    The check uses the exact integer endpoints so it is rigorously
    correct at all dyadic depths. A box is strictly outside H_d iff
    its forced-min mu_0 strictly exceeds its forced-max mu_{d-1}, i.e.
        lo_int[0] > hi_int[d-1].

    Boxes whose interval-projection straddles the H_d boundary
    {mu_0 = mu_{d-1}} (i.e. lo_int[0] <= hi_int[d-1]) are KEPT because
    they may contain points in H_d.

    SOUNDNESS: by Lemma 3.4 of THEOREM.md, val(d) = min_{H_d} f. The
    BnB explores a cover of {mu_0 <= 1/2}, which is a SUPERSET of H_d
    (any mu in Delta_d with mu_0 > mu_{d-1} satisfies mu_{d-1} < 1/2,
    so its sigma-image has (sigma mu)_0 = mu_{d-1} < 1/2 and lies in
    {mu_0 <= 1/2}). Hence dropping any box strictly outside H_d is
    sound: no min(H_d) candidate is lost.
    """
    if box.lo_int is None or box.hi_int is None:
        return False  # cannot judge; keep box (conservative)
    d = len(box.lo_int)
    if d < 2:
        return False
    return box.lo_int[0] > box.hi_int[d - 1]
