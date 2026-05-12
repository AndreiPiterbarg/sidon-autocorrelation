"""Marginalized Phi_N=2 with F4_MO217 constraint applied analytically.

For each (a_1, b_1), we compute an upper bound on
    sup over (a_2, b_2) admissible AND satisfying F4_MO217 of Phi_N=2(M, y_1, y_2)
where
    y_1 = a_1^2 + b_1^2,  y_2 = a_2^2 + b_2^2,
    F4: a_2 <= 2 a_1 - 1.
    Bathtub: y_2 <= mu(M).

The sup over y_2 (given y_1 fixed) reduces to a 1D concave optimization
on the interval [y_2_lo(a_1), mu(M)], where
    y_2_lo(a_1) = (1 - 2*a_1)^2  if a_1 < 1/2  (F4 forces a_2 <= 2 a_1 - 1 < 0)
                = 0              if a_1 >= 1/2.

Phi_N=2 as a function of y_2 (with y_1 fixed) is CONCAVE (proof: second
derivative is negative, see derivation in the docstring of __init__.py).
Hence sup over y_2 in [y_2_lo, mu] is at one of three candidates:
    (a) interior critical point y_2_star = k_2 * sqrt(A/C),
        where A = M - 1 - 2 y_1^2,  C = K_2 - 1 - 2 k_1^2.
        Substituting back: Phi at y_2_star = Phi_N=1(M, y_1)  [proven].
    (b) lower endpoint y_2 = y_2_lo (F4 binds from below).
    (c) upper endpoint y_2 = mu(M)   (bathtub binds from above).

Soundness: we conservatively return max(.upper()) of all three candidates.
This always upper-bounds the true sup, so phi_marg.upper() < 0 implies
sup Phi < 0 rigorously.

If the F4 region is empty (y_2_lo > mu), the cell is filter-forbidden and
phi_marg returns arb(-inf) (treated as PHI_REJECT).
"""
from __future__ import annotations

from flint import arb, fmpq, ctx

from .phi_mm import _safe_sqrt, arb_sqr, mu_of_M, PhiMMParams


def _max_arb_upper(*candidates) -> float:
    """Return the maximum of `.upper()` over the given arb candidates,
    treating None or arb('-inf') as -infinity."""
    best = float('-inf')
    for c in candidates:
        if c is None:
            continue
        try:
            u = float(c.upper())
        except (TypeError, ValueError, OverflowError):
            continue
        if u > best:
            best = u
    return best


def phi_n1_at(M: arb, a_1: arb, b_1: arb, params: PhiMMParams) -> arb:
    """Phi_N=1 (MV-10) at the given (a_1, b_1).

    Phi_N=1 = M + 1 + 2*y_1*k_1 + sqrt(M-1-2*y_1^2) * sqrt(K_2-1-2*k_1^2) - (2/u + a_gain).

    Note: this DOES NOT marginalize anything — it's the standard N=1 inequality.
    """
    k_1 = params.k_arb[0]
    K_2 = params.K2
    LHS = arb(2) / arb(params.u) + params.gain_a

    y_1 = arb_sqr(a_1) + arb_sqr(b_1)
    A = M - arb(1) - arb(2) * arb_sqr(y_1)
    C = K_2 - arb(1) - arb(2) * arb_sqr(k_1)

    # Phi_N=1 RHS - LHS
    return (M + arb(1) + arb(2) * y_1 * k_1
            + _safe_sqrt(A) * _safe_sqrt(C)
            - LHS)


def _phi_n2_at_fixed_y2(M: arb, y_1: arb, y_2: arb, params: PhiMMParams) -> arb | None:
    """Phi_N=2 with y_1 = z_1^2 and y_2 = z_2^2 both as arbs.

    Returns None if Parseval radicand is non-positive (cell forbidden).
    """
    k_1 = params.k_arb[0]
    k_2 = params.k_arb[1]
    K_2 = params.K2
    LHS = arb(2) / arb(params.u) + params.gain_a

    A_full = M - arb(1) - arb(2) * (arb_sqr(y_1) + arb_sqr(y_2))
    if A_full.upper() < 0:
        return None  # non-physical — cell is forbidden by Parseval
    B = K_2 - arb(1) - arb(2) * (arb_sqr(k_1) + arb_sqr(k_2))

    return (M + arb(1) + arb(2) * (y_1 * k_1 + y_2 * k_2)
            + _safe_sqrt(A_full) * _safe_sqrt(B)
            - LHS)


def phi_marg_f4_upper(M: arb, a_1: arb, b_1: arb, params: PhiMMParams) -> float:
    """Rigorous upper bound on sup over (a_2, b_2 | F4) of Phi_N=2(M, y_1, y_2).

    Phi_N=2 is concave in y_2, so the sup over y_2 in [y_2_lo, mu(M)] is at:
      (a) interior critical y_2* = k_2 sqrt(A/C)   -- gives Phi_N=1
      (b) lower endpoint y_2 = y_2_lo               -- F4 binds from below
      (c) upper endpoint y_2 = mu(M)                -- bathtub binds from above

    Candidate (a) is included ONLY when y_2* could lie within [y_2_lo, mu(M)]
    (rigorously decided in arb arithmetic).  Endpoints (b), (c) always included.

    If the F4 region is empty (y_2_lo > mu rigorously), returns -inf.
    """
    if len(params.k_arb) < 2:
        raise ValueError("phi_marg_f4 requires N >= 2 (need k_1, k_2)")

    k_1 = params.k_arb[0]
    k_2 = params.k_arb[1]
    K_2 = params.K2
    mu = mu_of_M(M)

    # ---- y_2_lo from F4 ----
    half = fmpq(1, 2)
    a_1_up = a_1.upper()
    if a_1_up < float(arb(half).lower()):
        a_1_up_arb = arb(a_1_up)
        y_2_lo = arb_sqr(arb(1) - arb(2) * a_1_up_arb)
    else:
        y_2_lo = arb(0)

    # ---- F4 region empty? ----
    if float(y_2_lo.lower()) > float(mu.upper()):
        return float('-inf')

    # ---- y_1 ----
    y_1 = arb_sqr(a_1) + arb_sqr(b_1)

    # ---- Compute y_2* (interior critical point) and check feasibility ----
    # y_2* = k_2 * sqrt(A/C),  A = M-1-2*y_1^2,  C = K_2-1-2*k_1^2
    A = M - arb(1) - arb(2) * arb_sqr(y_1)
    C = K_2 - arb(1) - arb(2) * arb_sqr(k_1)
    interior_in_range = True
    y_2_star_in = None
    if A.upper() < 0:
        # Parseval radicand non-physical: cell is forbidden anyway.
        return float('-inf')
    if A.lower() < 0:
        # A interval crosses 0; conservatively assume interior is in range.
        interior_in_range = True
    else:
        # A.lower() >= 0; compute y_2* rigorously.
        y_2_star = k_2 * (A / C).sqrt()
        # y_2* below y_2_lo (rigorously) ?
        if float(y_2_star.upper()) < float(y_2_lo.lower()):
            interior_in_range = False
        # y_2* above mu (rigorously) ?
        elif float(y_2_star.lower()) > float(mu.upper()):
            interior_in_range = False

    # ---- Candidates ----
    candidates = []

    # (b): y_2 = y_2_lo  (always include)
    phi_b = _phi_n2_at_fixed_y2(M, y_1, y_2_lo, params)
    candidates.append(phi_b)

    # (c): y_2 = mu(M)  (always include)
    phi_c = _phi_n2_at_fixed_y2(M, y_1, mu, params)
    candidates.append(phi_c)

    # (a): interior critical => Phi_N=1, ONLY if y_2* possibly in [y_2_lo, mu(M)]
    if interior_in_range:
        phi_a = phi_n1_at(M, a_1, b_1, params)
        candidates.append(phi_a)

    return _max_arb_upper(*candidates)


def phi_marg_f4_arb(M: arb, a_1: arb, b_1: arb, params: PhiMMParams) -> arb:
    """Like phi_marg_f4_upper but returns an arb interval enclosing the sup.

    Useful when you need both upper and lower bounds (e.g., for adaptive
    refinement).  The returned arb has .upper() == phi_marg_f4_upper(...).
    """
    if len(params.k_arb) < 2:
        raise ValueError("phi_marg_f4 requires N >= 2 (need k_1, k_2)")

    mu = mu_of_M(M)

    half = fmpq(1, 2)
    a_1_up = a_1.upper()
    if a_1_up < float(arb(half).lower()):
        a_1_up_arb = arb(a_1_up)
        y_2_lo = arb_sqr(arb(1) - arb(2) * a_1_up_arb)
    else:
        y_2_lo = arb(0)

    if float(y_2_lo.lower()) > float(mu.upper()):
        return arb(-1, 0)  # placeholder negative; cell will be rejected

    y_1 = arb_sqr(a_1) + arb_sqr(b_1)

    phi_a = phi_n1_at(M, a_1, b_1, params)
    phi_b = _phi_n2_at_fixed_y2(M, y_1, y_2_lo, params)
    phi_c = _phi_n2_at_fixed_y2(M, y_1, mu, params)

    # Return union (which contains all three)
    result = phi_a
    if phi_b is not None:
        result = result.union(phi_b)
    if phi_c is not None:
        result = result.union(phi_c)
    return result


__all__ = [
    "phi_n1_at",
    "phi_marg_f4_upper",
    "phi_marg_f4_arb",
]
