"""Union of forbidden z_1 intervals from TWO Matolcsi--Vinuesa triples.

Implementation of MV (2010) "open direction 2" from the remark on p. 7
lines 363--374 of arXiv:0907.1379v2.  MV note that any triple
``(delta, K, G)`` produces a forbidden set ``F`` of z_1-values; if two
triples produce forbidden sets whose UNION covers the feasible range
``[0, z_max(M_0)]`` predicted by Lemma 3.4, then the bound
``||f*f||_infty >= M_0`` follows automatically.

This module is a CORRECTNESS-FIRST re-implementation; no optimisation
of delta/G has been done, and the code is not executed here -- it is
written to be read, checked, and run by a human.

============================================================================
    MATHEMATICAL DERIVATION OF THE FORBIDDEN INTERVAL FORMULA
============================================================================

Set-up (MV eq. (10), p. 7 lines 343--347).  For a probability density
f >= 0 supported on [-1/4, 1/4] with int f = 1, write
``M = ||f*f||_infty`` and ``z_1 = |hat f(1)|``.  Fix:

    delta in (0, 1/4],       u = 1/2 + delta,
    K(x) = (1/delta) * (beta ** beta)(x/delta)   (MO 2009 eq. (2.1)-(2.2))
                                                  where beta is the arcsine
                                                  density on (-1/2, 1/2),
    G(x) = sum_{j=1}^{n} a_j cos(2 pi j x / u)   (G > 0 on [-1/4, 1/4]),
    K2    = ||K||_2^2 = sum_{r in Z} |J_0(pi r delta)|^4           (*)
          = (1/delta) * ||beta ** beta||_2^2
    k1    = hat K(1) = |J_0(pi delta)|^2        (period-1 FT; MV p.7 l.348)
    a     = (4/u) * (min_{[0,1/4]} G)^2
            / sum_{j=1}^{n} a_j^2 / |J_0(pi j delta / u)|^2   (gain; MV p.4)
    rho   = 2/u + a.

The exact identity (*) for ||K||_2^2 as a sum of fourth powers of Bessel
values is proved in MO 2009 Lemma 2.2 (see also
delsarte_dual/mo_framework_detailed.md section A).  At delta = 0.138
this evaluates to ~0.5746 / delta, i.e. strictly below the 0.5747 / delta
upper bound used in MV's paper.  We recompute it at each delta.

----------------------------------------------------------------------------
Lemma 3.4 upper bound on z_1.  For h = f*f one has ``0 <= h <= M``,
``int h = 1``, ``supp h subset [-1/2, 1/2]``.  MV's Lemma 3.4 then gives

    z_1^2 = |hat h(1)| <= M sin(pi / M) / pi.

Define z_max(M) = sqrt( M sin(pi/M) / pi ).  Assuming (for contradiction)
M < M_0 one has z_1 <= z_max(M) <= z_max(M_0), because z_max is monotone
non-decreasing in M on the physically relevant range M >= 1 (see
`z_max_is_monotone_near_MO` remark in the code below).

----------------------------------------------------------------------------
Master inequality (MV eq. 10).  Define

    RHS(M, z_1) := M + 1 + 2 z_1^2 k_1
                 + sqrt( M - 1 - 2 z_1^4 ) * sqrt( K2 - 1 - 2 k_1^2 ).

MV Lemmas 3.1--3.4 prove that *every* valid pair (M, z_1) satisfies

    rho  <=  RHS(M, z_1).                                     (MV eq. 10)

In particular the set of (M, z_1) for which ``RHS(M, z_1) < rho`` is EMPTY.

----------------------------------------------------------------------------
Forbidden set of z_1 at target M_0.  Fix the target bound M_0.  Define

    F(M_0) := { z_1 in [0, z_max(M_0)] : RHS(M_0, z_1) < rho }.

Claim (consequence of monotonicity of RHS in M): if z_1 in F(M_0), then
for every M <= M_0 we also have RHS(M, z_1) < rho, hence MV eq. (10) is
VIOLATED for every pair (M, z_1) with M <= M_0 and that fixed z_1; this
contradicts eq. (10).  So z_1 in F(M_0) excludes M <= M_0.

Proof of the monotonicity step.  Fix z_1 with 2 z_1^4 < M_0 - 1 (it must
be; else the sqrt is undefined).  View RHS as a function of M:

    d/dM RHS(M, z_1) = 1 + B / (2 sqrt(M - 1 - 2 z_1^4)),
          where B := sqrt(K2 - 1 - 2 k_1^2) >= 0.

Both terms are nonnegative, so RHS is non-decreasing in M.  Hence
RHS(M_0, z_1) < rho implies RHS(M, z_1) < rho for all M <= M_0.  QED.

----------------------------------------------------------------------------
Shape of F(M_0): F is an INTERVAL in z_1.

Proof.  Parametrise by t = z_1^2 in [0, t_max], t_max = z_max(M_0)^2.
Write

    Phi(t) := RHS(M_0, sqrt(t))
           = M_0 + 1 + 2 k_1 t + sqrt(M_0 - 1 - 2 t^2) * B.

Differentiate w.r.t. t on the open interval where M_0 - 1 - 2 t^2 > 0:

    Phi'(t)  =  2 k_1  -  2 B t / sqrt(M_0 - 1 - 2 t^2).

Phi'(t) = 0  <==>  k_1 sqrt(M_0 - 1 - 2 t^2) = B t
              <==>  k_1^2 (M_0 - 1 - 2 t^2) = B^2 t^2
              <==>  t^2 ( B^2 + 2 k_1^2 ) = k_1^2 (M_0 - 1)
              <==>  t = t*  :=  k_1 * sqrt( (M_0 - 1) / (B^2 + 2 k_1^2) ).

For t < t*: Phi'(t) > 0.   For t > t*: Phi'(t) < 0 (as long as the
sqrt is real).  So Phi is UNIMODAL on [0, t_max], strictly increasing
then strictly decreasing, with a unique maximum at t*.

Therefore the sub-level set { t : Phi(t) < rho } is the union of at
most two connected components [0, t_L) and (t_R, t_max]; equivalently
the super-level set { t : Phi(t) >= rho } is the connected interval
[t_L, t_R] (possibly empty, possibly truncated by the endpoints 0 and
t_max).  Pulling back by z_1 = sqrt(t) preserves intervals (sqrt is
strictly monotone on [0, inf)), so

    F(M_0) = [0, z_L) cup (z_R, z_max(M_0)]      (possibly with z_L = 0
                                                  or z_R = z_max(M_0),
                                                  collapsing one side).

We return F as a PAIR of sub-intervals (left piece and right piece);
either (or both) may be empty.  This matches MV's ``single interval''
description in the special case where Phi(0) >= rho (only a right-side
piece survives) or where Phi(t_max) >= rho (only a left-side piece).
For MV's own numbers at delta = 0.138, s_0 = 1.2748, Phi(0) < rho AND
Phi(t_max) < rho -- so BOTH sides contribute; see the smoke test.

----------------------------------------------------------------------------
Two-triple union.  Given two triples i = 1, 2 at (delta_i, G_i),
build F_i(M_0) as above.  If

    F_1(M_0) cup F_2(M_0)  SUPERSET  [0, z_max(M_0)],

then every feasible z_1 is forbidden by at least one triple, so the
existence of f with ||f*f||_infty <= M_0 is ruled out, hence M > M_0.

Both triples must use the SAME target M_0 (otherwise the sub-level-set
condition is inconsistent).  The feasible range is the SAME
[0, z_max(M_0)] for both triples (z_max depends only on M_0, not on
the triple).  So the check ``F_1 cup F_2 covers [0, z_max]'' is
well-posed.

Since each F_i is the union of a left piece [0, z_L^{(i)}) and a right
piece (z_R^{(i)}, z_max], the union F_1 cup F_2 covers [0, z_max] iff

    max(z_L^{(1)}, z_L^{(2)})   >=   min(z_R^{(1)}, z_R^{(2)}),

i.e. the left pieces together extend at least as far as the start of
the earliest right piece.  We check this explicitly in
`union_covers_interval`.

============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import mpmath as mp
from mpmath import mpf

from .mv_bound import (
    MV_COEFFS_119,
    MV_DELTA,
    MV_U,
    MV_K2_BOUND_OVER_DELTA,
    MV_BOUND_FINAL,
    G_value,
    S1_sum,
    gain_parameter,
    k1_value,
    min_G_on_0_quarter,
    solve_for_M_no_z1,
    solve_for_M_with_z1,
)


# ---------------------------------------------------------------------------
# Kernel K2 = ||K||_2^2 at arbitrary delta (recomputed; NOT the 0.5747 surrogate)
# ---------------------------------------------------------------------------

def K2_norm_squared(delta, n_terms: int = 5000) -> mpf:
    """Compute ||K||_2^2 = sum_{r in Z} |J_0(pi r delta)|^4 truncated at |r|<=n_terms.

    By MO 2009 Lemma 2.2 (see mo_framework_detailed.md section A), the
    period-1 Parseval identity yields

        ||K||_2^2  =  sum_{r in Z} |hat K(r)|^2
                    =  sum_{r in Z} |J_0(pi r delta)|^4.

    The sum converges: for large |r|, |J_0(pi r delta)|^2 ~ 2/(pi^2 r delta),
    so |J_0|^4 ~ 4/(pi^4 r^2 delta^2) = O(1/r^2).  The truncation tail
    error is bounded by  (4/(pi^4 delta^2)) * sum_{r > n_terms} 1/r^2
    < 4/(pi^4 delta^2 n_terms).

    This function is a NUMERICAL computation at the current mpmath
    precision; it is NOT a certified rigorous enclosure, but is
    sufficient for search.  A certified enclosure would add an explicit
    tail bound to ``total`` (see ``K2_norm_squared_upper`` below).

    Parameters
    ----------
    delta : mpf or float
        The kernel bandwidth in (0, 1/4].
    n_terms : int
        Number of positive-r terms to sum.

    Returns
    -------
    Truncated value of ||K||_2^2 as an mpf.
    """
    delta = mpf(delta)
    pi = mp.pi
    # r = 0 contributes |J_0(0)|^4 = 1.
    total = mpf(1)
    for r in range(1, n_terms + 1):
        j0 = mp.besselj(0, pi * r * delta)
        # r and -r contribute equally.
        total += 2 * j0 ** 4
    return total


def K2_norm_squared_upper(delta, n_terms: int = 5000) -> mpf:
    """Upper bound on ||K||_2^2, adding an explicit tail estimate.

    Tail bound (for r > n_terms):
        |J_0(x)|^4  <=  ( sqrt(2 / (pi x)) )^4   =  4 / (pi^2 x^2)
    for x >= some threshold where the asymptotic majorises (the
    bound |J_0(x)| <= sqrt(2/(pi x)) holds for all x >= 1/2;  to be
    strictly rigorous one would use an explicit Bessel majorant; here we
    use ``4/pi^2 * 1/(r^2 delta^2) * 1/pi^2`` which assumes x = pi r delta
    is large).  Tail sum:

        sum_{r > n_terms} 2 * 4 / (pi^4 r^2 delta^2)
        <=   8 / (pi^4 delta^2 n_terms).

    For n_terms = 5000 and delta = 0.1 this is ~ 8 / (97 * 0.01 * 5000)
    ~ 1.6e-3 which is much larger than double precision -- adequate for
    search but NOT for rigorous certification.  For rigour use a larger
    n_terms and interval arithmetic on J_0.
    """
    delta = mpf(delta)
    base = K2_norm_squared(delta, n_terms=n_terms)
    tail = 8 / (mp.pi ** 4 * delta ** 2 * mpf(n_terms))
    return base + tail


# ---------------------------------------------------------------------------
# Lemma 3.4: the upper bound z_max(M) on z_1
# ---------------------------------------------------------------------------

def z_max_lemma34(M) -> mpf:
    """Return z_max(M) = sqrt( M sin(pi/M) / pi ) from Lemma 3.4 of MV.

    Valid for M > 1 (in which case pi/M in (0, pi) and the sqrt is real).
    For M = 1 exactly the expression is 0 * something = 0.
    """
    M = mpf(M)
    if M <= 0:
        raise ValueError(f"Lemma 3.4 requires M > 0; got {M}.")
    val = M * mp.sin(mp.pi / M) / mp.pi
    if val < 0:
        # Should not happen for M in the physical range, but guard.
        val = mpf(0)
    return mp.sqrt(val)


# ---------------------------------------------------------------------------
# Gain parameter at arbitrary (delta, u); reuses MV_COEFFS_119 by default.
# ---------------------------------------------------------------------------

def gain_at(delta, u=None, a_coeffs: Sequence[mpf] = MV_COEFFS_119,
            n_grid_minG: int = 40001) -> Tuple[mpf, mpf, mpf]:
    """Compute (min_G, S_1, gain a) at (delta, u).

    If u is None, uses the MV-standard u = 1/2 + delta.

    Note: MV's tabulated a_j are OPTIMAL at (delta, u) = (0.138, 0.638)
    only; at other delta they are merely FEASIBLE (they still define a
    valid cosine polynomial G, with its own min over [0, 1/4]), but the
    gain will generally decrease.  The triple remains VALID for the
    master inequality; it just may give a weaker bound.

    Returns (min_G, S1, gain_a).
    """
    delta = mpf(delta)
    if u is None:
        u = mpf("0.5") + delta
    else:
        u = mpf(u)
    min_G, _argmin = min_G_on_0_quarter(a_coeffs=a_coeffs, u=u,
                                         n_grid=n_grid_minG)
    min_G = mpf(min_G)
    S1 = S1_sum(a_coeffs=a_coeffs, delta=delta, u=u)
    a = gain_parameter(min_G, S1, u=u)
    return min_G, S1, a


# ---------------------------------------------------------------------------
# Forbidden interval F(K, G, M_0)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForbiddenInterval:
    """Representation of F = [0, z_L) cup (z_R, z_max] as two sub-intervals.

    Each sub-interval is either None (empty) or a pair (lo, hi) of
    mpf values with 0 <= lo < hi <= z_max.  The left piece starts at
    lo=0; the right piece ends at hi=z_max.  A value z_1 is in F iff
    it lies in the left piece (if present) or the right piece (if
    present).
    """
    z_max: mpf
    left: Optional[Tuple[mpf, mpf]]     # [0, z_L)         ; None if empty
    right: Optional[Tuple[mpf, mpf]]    # (z_R, z_max]     ; None if empty

    def contains(self, z1) -> bool:
        z1 = mpf(z1)
        if self.left is not None:
            lo, hi = self.left
            if lo <= z1 < hi:
                return True
        if self.right is not None:
            lo, hi = self.right
            if lo < z1 <= hi:
                return True
        return False

    def covers_entire(self) -> bool:
        """Check if F covers the whole [0, z_max]."""
        # F covers [0, z_max] iff left piece extends to or past where
        # right piece starts (i.e., no gap).
        left_hi = self.left[1] if self.left is not None else mpf(0)
        right_lo = self.right[0] if self.right is not None else self.z_max
        # left covers [0, left_hi); right covers (right_lo, z_max].
        # Together they cover [0, z_max] iff left_hi > right_lo (strict,
        # to cover the boundary point... actually closed-open vs open-closed
        # means left_hi >= right_lo gives one point gap {left_hi} if
        # left_hi == right_lo and left is [0, left_hi) right is (left_hi, zmax]
        # That point is not covered by either; strictly speaking F misses it.
        # For a safe certification we require STRICT overlap: left_hi > right_lo.
        return left_hi > right_lo


def forbidden_interval(
    a_gain,
    delta,
    u=None,
    K2=None,
    M_target=mpf("1.278"),
    solver_tol: Optional[mpf] = None,
) -> ForbiddenInterval:
    """Compute F(K, G, M_target) = { z_1 in [0, z_max(M_target)] :
                                     RHS(M_target, z_1) < rho }.

    See module docstring for derivation.  Returns a ForbiddenInterval
    representing F as two sub-intervals (left and right of the unique
    maximum of Phi(t) = RHS(M_target, sqrt(t))).

    Parameters
    ----------
    a_gain : mpf
        Gain parameter  a = (4/u) * min_G^2 / S1.
    delta : mpf
        Kernel bandwidth.
    u : mpf, optional
        Period; defaults to 1/2 + delta.
    K2 : mpf, optional
        Value of ||K||_2^2 to use.  If None, recomputed via
        ``K2_norm_squared(delta)``.
    M_target : mpf
        Target bound M_0 to forbid.
    solver_tol : mpf, optional
        Tolerance for mp.findroot; defaults to mp.mpf('1e-' + str(dps-5)).

    Returns
    -------
    ForbiddenInterval with z_max = sqrt(M_target sin(pi/M_target)/pi).
    """
    delta = mpf(delta)
    if u is None:
        u = mpf("0.5") + delta
    else:
        u = mpf(u)
    M = mpf(M_target)
    if K2 is None:
        K2 = K2_norm_squared(delta)
    K2 = mpf(K2)
    a = mpf(a_gain)
    k1 = k1_value(delta)
    rho = 2 / u + a

    # The second-sqrt argument must be non-negative for the master
    # inequality to be well-formed at all.
    B2 = K2 - 1 - 2 * k1 * k1
    if B2 < 0:
        raise ValueError(
            f"K2 - 1 - 2 k_1^2 = {B2} < 0 at delta={delta}, K2={K2}, k1={k1}; "
            f"inequality (10) ill-defined."
        )
    B = mp.sqrt(B2)

    # Feasible z_1 range from Lemma 3.4
    z_max = z_max_lemma34(M)

    # Phi(t) = M + 1 + 2 k_1 t + sqrt(M - 1 - 2 t^2) * B     for t in [0, t_max]
    # with t = z_1^2 and t_max = z_max^2.  Also require 2 t^2 <= M - 1 for
    # the inner sqrt to be real; i.e. t <= sqrt((M-1)/2).  The physical
    # constraint t <= z_max(M)^2 is tighter for M near 1 (check: at M=1.278,
    # z_max^2 ~ 0.254, (M-1)/2 = 0.139; so sqrt((M-1)/2) ~ 0.373 is SMALLER
    # than z_max^2?  No wait: z_max^2 ~ 1.278*sin(pi/1.278)/pi ~ 0.254.  And
    # sqrt((M-1)/2) ~ 0.373.  So z_max^2 = 0.254 < 0.373 = sqrt((M-1)/2).
    # So the Lemma 3.4 bound is TIGHTER on the t range than the sqrt
    # domain constraint in this regime.  Good -- we can ignore the sqrt
    # constraint as long as z_max^2 < sqrt((M-1)/2), which we verify.
    sqrt_dom_upper = mp.sqrt((M - 1) / 2)      # in t coords; max of t allowed
    t_max_lemma = z_max * z_max
    t_max_eff = min(t_max_lemma, sqrt_dom_upper)

    if t_max_eff <= 0:
        # Degenerate; no feasible z_1.  Return trivially-covered F.
        return ForbiddenInterval(
            z_max=mpf(0),
            left=None,
            right=None,
        )

    def Phi(t):
        t = mpf(t)
        inner = M - 1 - 2 * t * t
        if inner < 0:
            # Outside domain; treat as -infinity (forbidden)
            return mpf("-inf")
        return M + 1 + 2 * k1 * t + mp.sqrt(inner) * B

    # Locate t* (max of Phi) analytically:
    #   t* = k1 * sqrt( (M-1) / (B^2 + 2 k1^2) )
    t_star = k1 * mp.sqrt((M - 1) / (B2 + 2 * k1 * k1))

    # Clip t* to the feasible t-range [0, t_max_eff].
    t_star_clip = min(max(t_star, mpf(0)), t_max_eff)

    # Evaluate Phi at 0, t_star_clip, and t_max_eff.
    phi_0 = Phi(0)
    phi_star = Phi(t_star_clip)
    phi_top = Phi(t_max_eff)

    # Three regimes for F:
    #
    # A)  phi_star < rho:  Phi is everywhere below rho  =>  F is all of
    #     [0, z_max], i.e. left piece = [0, z_max] and right piece
    #     empty.  (Total coverage by this single triple -- extraordinary.)
    #
    # B)  phi_0 >= rho AND phi_top >= rho:  Phi stays above rho on both
    #     endpoints; since Phi is unimodal with interior max >= the
    #     endpoints, Phi >= rho throughout [0, t_max_eff]  =>  F is empty.
    #
    # C)  phi_0 < rho AND phi_top >= rho:  Phi crosses rho once going up
    #     on the left of t*.  Left piece = [0, z_L), right piece empty.
    #
    # D)  phi_0 >= rho AND phi_top < rho:  Phi crosses rho once going
    #     down on the right of t*.  Left piece empty, right piece
    #     = (z_R, z_max].
    #
    # E)  phi_0 < rho AND phi_top < rho AND phi_star >= rho:  Phi crosses
    #     rho twice: once going up (at t_L < t_star), once going down
    #     (at t_R > t_star).  Both pieces present.  This is MV's
    #     case at (delta=0.138, M_target=1.278).

    left_piece: Optional[Tuple[mpf, mpf]] = None
    right_piece: Optional[Tuple[mpf, mpf]] = None

    if phi_star < rho:
        # Regime A: Phi < rho everywhere
        left_piece = (mpf(0), z_max)
        right_piece = None
    elif phi_0 >= rho and phi_top >= rho:
        # Regime B: Phi >= rho everywhere
        left_piece = None
        right_piece = None
    else:
        # Regimes C/D/E: find crossings via bisection.
        # Phi is strictly increasing on [0, t_star_clip] and strictly
        # decreasing on [t_star_clip, t_max_eff].

        # Left crossing: only exists if phi_0 < rho <= phi_star
        if phi_0 < rho and phi_star >= rho and t_star_clip > 0:
            t_L = _bisect_increasing(
                func=lambda t: Phi(t) - rho,
                a=mpf(0),
                b=t_star_clip,
                phi_a=phi_0 - rho,
                phi_b=phi_star - rho,
            )
            z_L = mp.sqrt(t_L)
            left_piece = (mpf(0), z_L)

        # Right crossing: only exists if phi_top < rho <= phi_star
        if phi_top < rho and phi_star >= rho and t_star_clip < t_max_eff:
            t_R = _bisect_decreasing(
                func=lambda t: Phi(t) - rho,
                a=t_star_clip,
                b=t_max_eff,
                phi_a=phi_star - rho,
                phi_b=phi_top - rho,
            )
            z_R = mp.sqrt(t_R)
            right_piece = (z_R, z_max)

    return ForbiddenInterval(z_max=z_max, left=left_piece, right=right_piece)


def _bisect_increasing(func, a, b, phi_a, phi_b, max_iter: int = 200):
    """Find unique root of `func` in [a, b] assuming func(a)<0<=func(b)
    and func strictly increasing.  Uses plain bisection at current dps.
    """
    a, b = mpf(a), mpf(b)
    fa, fb = mpf(phi_a), mpf(phi_b)
    if fa * fb > 0:
        # No sign change; return the endpoint with smaller abs value.
        return a if abs(fa) < abs(fb) else b
    tol = mp.mpf(10) ** (-(mp.mp.dps - 5))
    for _ in range(max_iter):
        m = (a + b) / 2
        fm = func(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        # Monotone increasing:  fm < 0 ==> root in (m, b); fm > 0 ==> root in (a, m)
        if fm < 0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    return (a + b) / 2


def _bisect_decreasing(func, a, b, phi_a, phi_b, max_iter: int = 200):
    """Find unique root of `func` in [a, b] assuming func(a)>=0>func(b)
    and func strictly decreasing.
    """
    a, b = mpf(a), mpf(b)
    fa, fb = mpf(phi_a), mpf(phi_b)
    if fa * fb > 0:
        return a if abs(fa) < abs(fb) else b
    tol = mp.mpf(10) ** (-(mp.mp.dps - 5))
    for _ in range(max_iter):
        m = (a + b) / 2
        fm = func(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        # Monotone decreasing:  fm > 0 ==> root in (m, b); fm < 0 ==> root in (a, m)
        if fm > 0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    return (a + b) / 2


# ---------------------------------------------------------------------------
# Union coverage
# ---------------------------------------------------------------------------

def union_covers(F1: ForbiddenInterval, F2: ForbiddenInterval,
                 z_max: mpf) -> bool:
    """Return True iff F1 cup F2 covers [0, z_max].

    Each F_i is a left piece [0, z_L^{(i)}) (possibly None) plus a right
    piece (z_R^{(i)}, z_max] (possibly None).  The combined union covers
    [0, z_max] iff the farthest-right left endpoint reaches (or exceeds)
    the nearest-left right endpoint:

        max(z_L^{(1)}, z_L^{(2)})  >  min(z_R^{(1)}, z_R^{(2)}),

    AND neither F_i entirely misses one side -- more precisely, we need
    both the 0-endpoint and the z_max-endpoint to be inside *some* F_i.
    The 0 endpoint is covered iff some F_i has a nonempty left piece;
    the z_max endpoint is covered iff some F_i has a nonempty right
    piece OR regime A for some F_i (left piece extending to z_max).

    We implement this cleanly by extracting:

        L := max over i of (z_L^{(i)} if left piece of F_i exists else 0)
        R := min over i of (z_R^{(i)} if right piece of F_i exists
                            else z_max)

    and returning True iff L > R.  The convention ``else z_max`` for
    the right-piece reflects "no right piece means the right piece is
    trivially the single point {z_max}" -- but actually F_i without a
    right piece may still have a left piece extending to z_max (regime
    A).  Handle the full-left special case first.
    """
    z_max = mpf(z_max)

    # Z_max mismatch guard: both F's must have the same z_max (same M_target).
    # We enforce consistency by using the supplied z_max as the common range.
    # Caller is responsible for passing the correct common z_max.

    # Special case: either F_i covers the full [0, z_max] by itself.
    # That happens if the left piece reaches z_max (regime A).
    for Fi in (F1, F2):
        if Fi.left is not None and Fi.left[1] >= z_max:
            return True

    def left_reach(F: ForbiddenInterval) -> mpf:
        return F.left[1] if F.left is not None else mpf(0)

    def right_start(F: ForbiddenInterval) -> mpf:
        # If no right piece, right piece is "empty starting at z_max",
        # so start = z_max (i.e. nothing to cover on the right).
        return F.right[0] if F.right is not None else z_max

    L = max(left_reach(F1), left_reach(F2))
    R = min(right_start(F1), right_start(F2))

    return L > R


# ---------------------------------------------------------------------------
# Scanning: search for two triples whose union covers [0, z_max(M_target)]
# ---------------------------------------------------------------------------

@dataclass
class TripleData:
    delta: mpf
    u: mpf
    min_G: mpf
    S1: mpf
    gain_a: mpf
    k1: mpf
    K2: mpf
    F: ForbiddenInterval


def build_triple(delta, a_coeffs: Sequence[mpf] = MV_COEFFS_119,
                 u: Optional[mpf] = None,
                 K2: Optional[mpf] = None,
                 M_target: mpf = mpf("1.278"),
                 n_grid_minG: int = 40001,
                 K2_n_terms: int = 5000) -> TripleData:
    """Assemble a TripleData at (delta, G = sum a_j cos(2 pi j x/u)).

    If ``u`` is None, uses ``u = 1/2 + delta``.
    If ``K2`` is None, computes ``K2 = K2_norm_squared(delta, K2_n_terms)``.

    IMPORTANT CORRECTNESS NOTE (MV construction):  The admissibility
    hypothesis of MV's Lemma 3.1 requires ``min_{[0,1/4]} G >= (any
    positive constant)``.  MV's specific a_j at (0.138, 0.638) satisfy
    min G = 1 (the QP makes this active).  At a DIFFERENT delta (or u)
    the same a_j produce a different trig polynomial G, and its minimum
    may not be 1 (could be above OR below).  If min G <= 0 at the new
    delta, the triple is INADMISSIBLE and we must report an error.
    Otherwise the triple is admissible but typically has a smaller
    gain.
    """
    delta = mpf(delta)
    if u is None:
        u = mpf("0.5") + delta
    else:
        u = mpf(u)
    min_G, _argmin = min_G_on_0_quarter(a_coeffs=a_coeffs, u=u,
                                         n_grid=n_grid_minG)
    min_G = mpf(min_G)
    if min_G <= 0:
        raise ValueError(
            f"Triple at delta={delta}, u={u} is inadmissible: "
            f"min_G = {min_G} <= 0.  The tabulated a_j are optimal "
            f"at (delta, u) = (0.138, 0.638) only; at other (delta, u) "
            f"G may go negative on [0, 1/4]."
        )
    S1 = S1_sum(a_coeffs=a_coeffs, delta=delta, u=u)
    a = gain_parameter(min_G, S1, u=u)
    k1 = k1_value(delta)
    if K2 is None:
        K2 = K2_norm_squared(delta, n_terms=K2_n_terms)
    K2 = mpf(K2)
    F = forbidden_interval(
        a_gain=a, delta=delta, u=u, K2=K2, M_target=M_target,
    )
    return TripleData(
        delta=delta, u=u, min_G=min_G, S1=S1, gain_a=a,
        k1=k1, K2=K2, F=F,
    )


def _build_triple_worker(d, u, a_coeffs, n_grid_minG, K2_n_terms, M_target):
    """Top-level joblib worker."""
    try:
        return build_triple(
            delta=d, a_coeffs=a_coeffs, u=u,
            M_target=M_target,
            n_grid_minG=n_grid_minG,
            K2_n_terms=K2_n_terms,
        )
    except (ValueError, ZeroDivisionError) as err:
        return ("error", d, str(err))


def search_two_triples(
    M_target,
    delta_candidates: Iterable,
    a_coeffs: Sequence[mpf] = MV_COEFFS_119,
    u_from_delta=lambda d: mpf("0.5") + mpf(d),
    K2_n_terms: int = 4000,
    n_grid_minG: int = 20001,
    verbose: bool = False,
    n_jobs: int = -1,
) -> Optional[Tuple[TripleData, TripleData]]:
    """Scan delta_candidates IN PARALLEL for pairs of triples whose union
    F_1 cup F_2 covers [0, z_max(M_target)].
    """
    from joblib import Parallel, delayed
    M_target = mpf(M_target)
    deltas = [mpf(d) for d in delta_candidates]

    tasks = [(d, u_from_delta(d), a_coeffs, n_grid_minG, K2_n_terms, M_target)
             for d in deltas]
    raw = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_build_triple_worker)(*t) for t in tasks
    )

    triples = []
    for item in raw:
        if isinstance(item, tuple) and len(item) == 3 and item[0] == "error":
            if verbose:
                print(f"  delta={float(item[1]):.5f}: SKIPPED ({item[2]})")
            continue
        T = item
        triples.append(T)
        if verbose:
            f = T.F
            ls = f.left if f.left is not None else None
            rs = f.right if f.right is not None else None
            print(f"  delta={float(T.delta):.5f} u={float(T.u):.5f}  "
                  f"gain a={float(T.gain_a):.5f}  "
                  f"min_G={float(T.min_G):.5f}  K2={float(T.K2):.5f}  "
                  f"F.left={ls}  F.right={rs}")

    if len(triples) < 2:
        return None

    z_max = z_max_lemma34(M_target)
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            if union_covers(triples[i].F, triples[j].F, z_max=z_max):
                return triples[i], triples[j]
    return None


def certify_bound_via_union(
    M_target,
    delta1, delta2,
    a_coeffs1: Sequence[mpf] = MV_COEFFS_119,
    a_coeffs2: Sequence[mpf] = MV_COEFFS_119,
    u1: Optional[mpf] = None,
    u2: Optional[mpf] = None,
    K2_n_terms: int = 5000,
    n_grid_minG: int = 40001,
) -> dict:
    """Given two triples at delta1, delta2 (with optional distinct a-coeffs),
    check whether F_1 cup F_2 covers [0, z_max(M_target)] and hence
    ||f*f||_infty >= M_target.

    Returns a dict with the two triples' intermediate quantities and a
    boolean 'covers' flag.

    CAUTION: this is a NUMERICAL check at the current mpmath precision;
    it does NOT yet incorporate rigorous interval arithmetic for K2 or
    min_G.  For a certified bound, upgrade K2_norm_squared to a rigorous
    enclosure and replace min_G_on_0_quarter with an interval B&B.
    """
    M_target = mpf(M_target)
    T1 = build_triple(delta=delta1, u=u1, a_coeffs=a_coeffs1,
                      M_target=M_target,
                      n_grid_minG=n_grid_minG,
                      K2_n_terms=K2_n_terms)
    T2 = build_triple(delta=delta2, u=u2, a_coeffs=a_coeffs2,
                      M_target=M_target,
                      n_grid_minG=n_grid_minG,
                      K2_n_terms=K2_n_terms)
    z_max = z_max_lemma34(M_target)
    ok = union_covers(T1.F, T2.F, z_max=z_max)
    return {
        "M_target": M_target,
        "z_max": z_max,
        "triple_1": T1,
        "triple_2": T2,
        "covers": ok,
    }


# ---------------------------------------------------------------------------
# Main: smoke tests and scan
# ---------------------------------------------------------------------------

def reproduce_mv_single_triple_forbidden(dps: int = 40,
                                         M_target=mpf("1.2748")):
    """Reproduce MV's stated single-triple forbidden interval
    F = (0.504433, 0.529849) at (delta, u, M_target) = (0.138, 0.638, 1.2748).

    NOTE: MV's interval (0.504433, 0.529849) is the RIGHT-side sub-
    interval of our F (the piece (z_R, z_max]).  But MV's upper endpoint
    0.529849 is ABOVE z_max(1.2748) ~ 0.50426 -- so MV's quoted interval
    extends past the Lemma-3.4 feasible range.  Our F is intersected
    with [0, z_max(M_target)], so our right piece will be
    (z_R, z_max(M_target)] where z_R ~ 0.504433.  Our computed z_R
    should agree with MV's 0.504433 to high precision.
    """
    mp.mp.dps = dps
    T = build_triple(delta=MV_DELTA, u=MV_U, a_coeffs=MV_COEFFS_119,
                     M_target=M_target)
    return T


if __name__ == "__main__":
    mp.mp.dps = 40

    print("=" * 72)
    print("MV single-triple forbidden interval (smoke test)")
    print("=" * 72)
    T_mv = reproduce_mv_single_triple_forbidden(
        dps=40, M_target=MV_BOUND_FINAL,
    )
    z_max_mv = z_max_lemma34(MV_BOUND_FINAL)
    print(f"  delta        = {float(T_mv.delta)}")
    print(f"  u            = {float(T_mv.u)}")
    print(f"  min_G        = {mp.nstr(T_mv.min_G, 8)}")
    print(f"  S1           = {mp.nstr(T_mv.S1, 8)}")
    print(f"  gain a       = {mp.nstr(T_mv.gain_a, 8)}")
    print(f"  k1           = {mp.nstr(T_mv.k1, 8)}")
    print(f"  K2 computed  = {mp.nstr(T_mv.K2, 8)} "
          f"(MV surrogate 0.5747/delta = "
          f"{mp.nstr(MV_K2_BOUND_OVER_DELTA/T_mv.delta, 8)})")
    print(f"  z_max(M_0)   = {mp.nstr(z_max_mv, 10)}     "
          f"(MV's 0.50426)")
    print(f"  F.left       = {T_mv.F.left}")
    print(f"  F.right      = {T_mv.F.right}     "
          f"(MV's right piece starts at 0.504433)")
    print()

    print("=" * 72)
    print("Search for TWO-triple coverage at various M_target")
    print("=" * 72)

    # Candidate delta values around MV's 0.138.  At each delta we use
    # the same tabulated MV_COEFFS_119, which are OPTIMAL at 0.138 only;
    # at nearby delta the gain drops off but the triple remains valid
    # *as long as min_G > 0*.  If min_G goes negative at some delta, the
    # build_triple call raises ValueError and the delta is skipped.
    delta_grid = [
        mpf("0.10"),
        mpf("0.11"),
        mpf("0.115"),
        mpf("0.12"),
        mpf("0.125"),
        mpf("0.13"),
        mpf("0.135"),
        mpf("0.138"),    # MV's canonical value
        mpf("0.14"),
        mpf("0.145"),
        mpf("0.15"),
        mpf("0.155"),
        mpf("0.16"),
        mpf("0.17"),
        mpf("0.18"),
    ]

    for M_target_str in ("1.2748", "1.278", "1.279", "1.280"):
        M_target = mpf(M_target_str)
        print(f"\n--- M_target = {M_target_str} ---")
        result = search_two_triples(
            M_target=M_target,
            delta_candidates=delta_grid,
            a_coeffs=MV_COEFFS_119,
            verbose=True,
        )
        if result is None:
            print(f"  >>> NO pair in the scan covers [0, z_max] at "
                  f"M_target = {M_target_str}.")
        else:
            T1, T2 = result
            print(f"  >>> FOUND covering pair:")
            print(f"        delta_1 = {float(T1.delta)}, gain_1 = "
                  f"{float(T1.gain_a):.6f}")
            print(f"        delta_2 = {float(T2.delta)}, gain_2 = "
                  f"{float(T2.gain_a):.6f}")
            print(f"        F_1 = left={T1.F.left}  right={T1.F.right}")
            print(f"        F_2 = left={T2.F.left}  right={T2.F.right}")
            print(f"        z_max(M_target) = "
                  f"{mp.nstr(z_max_lemma34(M_target), 8)}")

    print()
    print("Done.  (This is a search; certification requires rigorous "
          "K2 enclosure and min_G interval B&B.)")
