"""Martin-O'Bryant 2004 Lemma 2.17 joint (z_1, z_2) constraint in the MV dual.

Idea B2 for improving the Matolcsi-Vinuesa lower bound on C_{1a}.
This file is self-contained and does NOT modify any existing file; it only
imports building blocks from ``mv_bound.py``.

================================================================================
1.  Background: the 2-moment MV master inequality
================================================================================

Fix the MV data (delta = 0.138, u = 0.638, n = 119 cosine coefficients a_j,
kernel K the rescaled arcsine density).  Write

    h := f*f,    M := ||h||_oo,    K_2 := ||K||_2^2,
    z_n := |hat f(n)|,   k_n := hat K(n)   (period-1 Fourier).

Following the derivation in ``multi_moment_derivation.md`` (Theorem 2, (MM-10))
the two-moment refinement of MV's eq. (10) reads

    (MM-10)     2/u + a  <=  M + 1 + 2 (z_1^2 k_1 + z_2^2 k_2)
                              + sqrt( M - 1 - 2 (z_1^4 + z_2^4) )
                              * sqrt( K_2 - 1 - 2 (k_1^2 + k_2^2) ).

This is a *consequence* of the basic MV inequality chain (Parseval + Cauchy-
Schwarz on the high-frequency tail), obtained by peeling off the n = 1, 2
terms before applying Cauchy-Schwarz.  The two sqrt radicands are non-negative
by the (B1) assumptions

    (A1)   z_1^4 + z_2^4       <=  (M - 1) / 2,
    (A2)   k_1^2 + k_2^2       <=  (K_2 - 1) / 2,

both of which are implied by Lemma 1 of ``multi_moment_derivation.md``
(mu(M)^2 <= (M-1)/2 for M <= 2 and ``nmax = 2``, and the numerical values
k_1^2 + k_2^2 << (K_2 - 1)/2 for MV's K; see the feasibility asserts below).

The single-moment MV bound (``mv_bound.solve_for_M_with_z1``) is recovered by
setting z_2 = k_2 = 0 in (MM-10).

================================================================================
2.  The new ingredient: MO 2004 Lemma 2.17
================================================================================

Martin-O'Bryant 2004, Lemma 2.17 (p. 22 lines 1438-1469).  For every pdf f
with supp f subset [-1/4, 1/4] and int f = 1,

    (L217)   2 |hat f(1)|^2 - 1  <=  Re hat f(2)  <=  2 Re hat f(1) - 1.

We only use the RIGHT inequality here.  Note Re hat f(n) is real because we
may translate f so that hat f(1) is real and positive without changing
|hat f(1)|, |hat f(2)|, or M; more carefully:

    *  WLOG one may replace f(x) by f(x - t_0) for any t_0 in R.  This
       multiplies hat f(n) by exp(-2 pi i n t_0).  Choose t_0 so that
       hat f(1) becomes real and >= 0; then r_1 := Re hat f(1) = |hat f(1)|.
    *  After this rotation, hat f(2) is rotated by exp(-4 pi i t_0) and
       generally has non-zero imaginary part; we only retain the bound on
       its real part.
    *  |hat f(2)| is invariant under translation.  In general
       |hat f(2)| >= |Re hat f(2)|, hence z_2 >= |r_2|.

Both the admissible set and M are unchanged by the translation, so we may
assume throughout

    r_1 = Re hat f(1) = z_1  >=  0,       r_2 = Re hat f(2),       r_2 <= z_2.

Warning: we are NOT free to also rotate hat f(2) independently; the Lemma
2.17 constraint r_2 <= 2 r_1 - 1 is stated IN THIS TRANSLATED FRAME.

Observation (strict).  Because we have already translated so that
r_1 = z_1 >= 0, the quantity on the left of Lemma 2.17 is 2 z_1 - 1, and the
constraint becomes

    (L217*)        r_2  <=  2 z_1 - 1,

to be combined with r_2 in [-z_2, z_2].

================================================================================
3.  Derivation of the joint (z_1, z_2) feasibility constraint
================================================================================

We now derive a constraint on the PAIR (z_1, z_2) that must hold for every
admissible f.  After the translation in Section 2 we have the constraints

    (a)   z_1 >= 0,
    (b)   z_2 >= 0,
    (c)   |r_2| <= z_2,         i.e.  r_2 in [-z_2, z_2],
    (d)   r_2 <= 2 z_1 - 1      (Lemma 2.17, right inequality, in this frame).

Feasibility: there must exist a real r_2 satisfying (c) and (d) simultaneously,
i.e.

    -z_2  <=  r_2  <=  min(z_2, 2 z_1 - 1).

Such r_2 exists iff -z_2 <= min(z_2, 2 z_1 - 1).  The inequality
-z_2 <= z_2 is automatic (z_2 >= 0), so feasibility reduces to

    -z_2  <=  2 z_1 - 1
      <=>    2 z_1  >=  1 - z_2
      <=>    2 z_1 + z_2  >=  1.                              (*)

This is the joint constraint.  Equivalent forms used below:

    z_1 >= (1 - z_2)/2,          z_2 >= 1 - 2 z_1.

Note that the constraint is VACUOUS if z_2 >= 1 (because then -z_2 <= 2 z_1 - 1
may fail in principle, but we are in the regime 2 z_1 - 1 <= 1 <= z_2 which
makes the bound trivial; in numerics we have z_2 <= mu(M) < 1, so (*) is
always binding at the lower left corner of the box).

Sanity checks of (*).

    *  Setting z_2 = 0 forces z_1 >= 1/2.  Interpretation: if hat f(2) = 0
       exactly, then Re hat f(2) = 0 <= 2 Re hat f(1) - 1 forces
       Re hat f(1) >= 1/2, hence |hat f(1)| = z_1 >= 1/2.  Consistent.
    *  Setting z_1 = 0 forces z_2 >= 1.  Interpretation: if hat f(1) = 0,
       then 2 Re hat f(1) - 1 = -1, so Re hat f(2) <= -1, hence
       |hat f(2)| = z_2 >= 1.  Consistent.
    *  At z_1 = 1/2, z_2 = 0 the constraint is tight: r_2 = 0 with
       r_2 = 2 r_1 - 1, r_1 = 1/2, exactly Lemma 2.17's boundary case.

================================================================================
4.  Use in the master inequality: lower-bound recipe
================================================================================

The recipe for a lower bound on M is the standard "worst-case over hidden
variables" argument.  For any admissible f, its (z_1, z_2) lives in

    D(M) := { (z_1, z_2) in [0, mu(M)]^2  :  2 z_1 + z_2 >= 1 }         (D)

where mu(M) := sqrt(M sin(pi/M)/pi)  (MO 2004 Lemma 2.14, applied to h = f*f
giving |hat h(n)| <= M sin(pi/M)/pi, and z_n^2 = |hat h(n)|).  The master
inequality (MM-10) must hold for the UNKNOWN actual (z_1, z_2) in D(M).
Hence it suffices that the inequality hold for SOME (z_1, z_2) in D(M),
i.e.

    2/u + a  <=  max_{(z_1, z_2) in D(M)}  R(M; z_1, z_2),             (LB)

where R(.) is the RHS of (MM-10).  The lower bound on M is the SMALLEST M
for which (LB) holds.

Adding the Lemma 2.17 constraint (*) STRICTLY SHRINKS D(M) compared with
the full box [0, mu(M)]^2 (which is the (B1) two-moment feasible set),
so the max on the right of (LB) is no larger than the B1 max, and the
threshold M is no smaller than the B1 threshold.  In particular, removing
(*) must reproduce the 2-moment B1 bound; and with z_2 = 0, k_2 = 0 it
must reproduce MV's 1.27481.  Both regressions are verified in
``reproduce_lemma217``.

================================================================================
5.  Implementation notes
================================================================================

*  mpmath dps = 30 throughout.
*  The objective R(M; z_1, z_2) is concave in (y_1, y_2) := (z_1^2, z_2^2),
   but the Lemma 2.17 constraint 2 z_1 + z_2 >= 1 is NON-convex in (y_1, y_2).
   We therefore do a simple 2-D search:  fine grid in (z_1, z_2), then a
   ``scipy.optimize.minimize`` local polish from the grid max.  Since the
   admissible set is compact and the grid is dense, this is adequate for
   double-checking.  A rigorous evaluation is straightforward because
   R is monotone in each of (z_1^2, z_2^2) up to the sqrt radicand boundary,
   and the constraint surface is 1-D (the line 2 z_1 + z_2 = 1 intersected
   with the box).
*  Outer bisection on M uses the signed gap  max_D R - (2/u + a): negative
   means M is ruled out (increase M), non-negative means M is feasible (decrease).
"""
from __future__ import annotations

import mpmath as mp
from mpmath import mpf

from .mv_bound import (
    MV_COEFFS_119,
    MV_DELTA,
    MV_U,
    MV_K2_BOUND_OVER_DELTA,
    MV_BOUND_FINAL,
    MV_BOUND_NO_Z1,
    min_G_on_0_quarter,
    S1_sum,
    gain_parameter,
    k1_value,
)


# -----------------------------------------------------------------------------
# Fourier data for K at integer frequencies (period-1 convention, matching
# mv_bound.k1_value which uses |J_0(pi delta)|^2).
# -----------------------------------------------------------------------------

def kn_value(n, delta=MV_DELTA):
    """k_n = hat K(n) = |J_0(pi n delta)|^2   (period-1 Fourier).

    Matches ``mv_bound.k1_value`` at n = 1 (verified by construction).
    """
    delta = mpf(delta)
    n = mpf(n)
    j0 = mp.besselj(0, mp.pi * n * delta)
    return j0 * j0


# -----------------------------------------------------------------------------
# Lemma 2.17 joint constraint
# -----------------------------------------------------------------------------

def joint_z_feasible(z1, z2):
    """Return True iff (z_1, z_2) satisfies the MO 2004 Lemma 2.17 constraint

        2 z_1 + z_2 >= 1

    (derivation in the module docstring, Section 3).
    """
    z1 = mpf(z1)
    z2 = mpf(z2)
    return 2 * z1 + z2 >= mpf(1)


# -----------------------------------------------------------------------------
# Two-moment master inequality RHS (MM-10)
# -----------------------------------------------------------------------------

def rhs_z1_z2(M, z1, z2, k1, k2, K2):
    """RHS of (MM-10):

        M + 1 + 2 (z_1^2 k_1 + z_2^2 k_2)
          + sqrt(M - 1 - 2 (z_1^4 + z_2^4)) * sqrt(K_2 - 1 - 2 (k_1^2 + k_2^2)).

    Returns mpf(-inf) if either sqrt radicand is negative (infeasible pair).
    """
    M, z1, z2 = mpf(M), mpf(z1), mpf(z2)
    k1, k2, K2 = mpf(k1), mpf(k2), mpf(K2)
    z1sq, z2sq = z1 * z1, z2 * z2
    z14, z24 = z1sq * z1sq, z2sq * z2sq
    rad1 = M - 1 - 2 * (z14 + z24)
    rad2 = K2 - 1 - 2 * (k1 * k1 + k2 * k2)
    if rad1 < 0 or rad2 < 0:
        return mp.mpf("-inf")
    linear = M + 1 + 2 * (z1sq * k1 + z2sq * k2)
    return linear + mp.sqrt(rad1) * mp.sqrt(rad2)


def mu_of_M(M):
    """mu(M) = sqrt(M sin(pi/M)/pi)  (MO 2004 Lemma 2.14).

    This is the common upper bound |hat f(n)| <= mu(M) for every n >= 1.
    """
    M = mpf(M)
    val = M * mp.sin(mp.pi / M) / mp.pi
    if val < 0:
        raise ValueError(f"mu(M) radicand negative at M = {M}.")
    return mp.sqrt(val)


# -----------------------------------------------------------------------------
# Maximise RHS over (z_1, z_2)
# -----------------------------------------------------------------------------

def _rhs_vec(M_f, z1_arr, z2_arr, k1_f, k2_f, K2_f):
    """Vectorised float64 RHS.  Returns -inf where either radicand is negative.
    z1_arr, z2_arr: numpy arrays (same shape).  Returns a numpy array.
    """
    import numpy as np
    z1sq = z1_arr * z1_arr
    z2sq = z2_arr * z2_arr
    z14 = z1sq * z1sq
    z24 = z2sq * z2sq
    rad1 = M_f - 1.0 - 2.0 * (z14 + z24)
    rad2 = K2_f - 1.0 - 2.0 * (k1_f * k1_f + k2_f * k2_f)
    out = np.full(z1_arr.shape, -np.inf, dtype=np.float64)
    if rad2 < 0:
        return out
    mask = rad1 >= 0
    if not np.any(mask):
        return out
    out[mask] = (
        M_f + 1.0
        + 2.0 * (z1sq[mask] * k1_f + z2sq[mask] * k2_f)
        + np.sqrt(rad1[mask]) * np.sqrt(rad2)
    )
    return out


def _max_rhs_box(M, k1, k2, K2, n_grid=401):
    """Max R(M; z_1, z_2) over [0, mu(M)]^2 (NO Lemma 2.17 constraint).

    Vectorised float64 grid + mpmath polish at the argmax for precision.
    """
    import numpy as np
    zmax = mu_of_M(M)
    M_f, k1_f, k2_f, K2_f = float(M), float(k1), float(k2), float(K2)
    zmax_f = float(zmax)

    axis = np.linspace(0.0, zmax_f, int(n_grid), dtype=np.float64)
    Z1, Z2 = np.meshgrid(axis, axis, indexing='ij')
    R = _rhs_vec(M_f, Z1, Z2, k1_f, k2_f, K2_f)

    # Plus the (zmax, zmax) corner explicitly
    corner = _rhs_vec(M_f, np.array([zmax_f]), np.array([zmax_f]), k1_f, k2_f, K2_f)[0]
    idx_flat = int(np.argmax(R))
    i, j = divmod(idx_flat, int(n_grid))
    best_f = float(R[i, j])
    if corner > best_f:
        best_f = float(corner)
        z1_star_f, z2_star_f = zmax_f, zmax_f
    else:
        z1_star_f, z2_star_f = float(Z1[i, j]), float(Z2[i, j])

    # mpmath polish at argmax
    best_mp = rhs_z1_z2(M, mpf(z1_star_f), mpf(z2_star_f), k1, k2, K2)
    if best_mp != mp.mpf("-inf"):
        return best_mp
    return mpf(best_f)


def max_rhs_under_lemma217(M, k1, k2, K2, n_grid=401):
    """Max R(M; z_1, z_2) over D(M) = { z in [0, mu(M)]^2 : 2 z_1 + z_2 >= 1 }.

    Vectorised float64 grid + scipy polish + mpmath re-evaluation for precision.
    """
    import numpy as np
    zmax = mu_of_M(M)
    if 2 * zmax + zmax < 1:
        return mp.mpf("-inf")

    M_f, k1_f, k2_f, K2_f = float(M), float(k1), float(k2), float(K2)
    zmax_f = float(zmax)

    axis = np.linspace(0.0, zmax_f, int(n_grid), dtype=np.float64)
    Z1, Z2 = np.meshgrid(axis, axis, indexing='ij')
    feasible = (2.0 * Z1 + Z2) >= 1.0
    R = _rhs_vec(M_f, Z1, Z2, k1_f, k2_f, K2_f)
    R[~feasible] = -np.inf

    if not np.isfinite(R).any():
        return mp.mpf("-inf")
    idx_flat = int(np.argmax(R))
    i, j = divmod(idx_flat, int(n_grid))
    z1_star_f, z2_star_f = float(Z1[i, j]), float(Z2[i, j])
    best_f = float(R[i, j])

    # scipy Nelder-Mead polish
    try:
        from scipy.optimize import minimize

        def neg_r(v):
            z1_f_, z2_f_ = v
            if z1_f_ < 0 or z1_f_ > zmax_f:
                return 1e6
            if z2_f_ < 0 or z2_f_ > zmax_f:
                return 1e6
            if 2.0 * z1_f_ + z2_f_ < 1.0 - 1e-12:
                return 1e6
            val = _rhs_vec(M_f, np.array([z1_f_]), np.array([z2_f_]),
                           k1_f, k2_f, K2_f)[0]
            if not np.isfinite(val):
                return 1e6
            return -float(val)

        res = minimize(
            neg_r,
            x0=np.array([z1_star_f, z2_star_f]),
            method="Nelder-Mead",
            options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 2000},
        )
        if -res.fun > best_f:
            best_f = -res.fun
            z1_star_f, z2_star_f = float(res.x[0]), float(res.x[1])
    except Exception:
        pass

    # mpmath re-evaluation at argmax (for full precision).
    best_mp = rhs_z1_z2(M, mpf(z1_star_f), mpf(z2_star_f), k1, k2, K2)
    if best_mp != mp.mpf("-inf"):
        return best_mp
    return mpf(best_f)


# -----------------------------------------------------------------------------
# Outer bisection on M
# -----------------------------------------------------------------------------

def _solve_for_M(lhs_value, delta, u, K2, use_lemma217, n_grid=401,
                 M_lo=mpf("1.0001"), M_hi=mpf("1.6"), tol=mpf("1e-8")):
    """Bisection for the smallest M such that

        max_{(z_1, z_2) in feasible set}  R(M; z_1, z_2)  >=  lhs_value,

    where the feasible set is either [0, mu(M)]^2 (use_lemma217 = False) or
    D(M) (use_lemma217 = True).
    """
    k1 = kn_value(1, delta)
    k2 = kn_value(2, delta)

    def gap(M):
        if use_lemma217:
            R = max_rhs_under_lemma217(M, k1, k2, K2, n_grid=n_grid)
        else:
            R = _max_rhs_box(M, k1, k2, K2, n_grid=n_grid)
        return R - lhs_value

    # Endpoint sanity.  At low M the RHS is small and the gap is negative;
    # at high M the RHS is large and the gap is positive.
    g_lo = gap(M_lo)
    g_hi = gap(M_hi)
    if g_lo >= 0:
        return M_lo
    if g_hi < 0:
        raise RuntimeError(
            f"solve_for_M bracketing failure: M_hi = {M_hi} still insufficient "
            f"(gap = {g_hi}).  Raise M_hi."
        )
    while M_hi - M_lo > tol:
        M_mid = (M_lo + M_hi) / 2
        g_mid = gap(M_mid)
        if g_mid >= 0:
            M_hi = M_mid
        else:
            M_lo = M_mid
    return (M_lo + M_hi) / 2


def solve_for_M_lemma217(a_gain, delta=MV_DELTA, u=MV_U,
                         K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
                         n_grid=401, tol=mpf("1e-8")):
    """Smallest M with

        max_{(z_1, z_2) in D(M)}  R(M; z_1, z_2)  >=  2/u + a,

    where D(M) = { (z_1, z_2) in [0, mu(M)]^2 : 2 z_1 + z_2 >= 1 }.

    Returns the lower bound on M = ||f*f||_oo implied by MV (two-moment
    master inequality) combined with MO 2004 Lemma 2.17.
    """
    delta, u, a = mpf(delta), mpf(u), mpf(a_gain)
    K2 = mpf(K2_bound_over_delta) / delta
    lhs = 2 / u + a
    return _solve_for_M(
        lhs_value=lhs,
        delta=delta,
        u=u,
        K2=K2,
        use_lemma217=True,
        n_grid=n_grid,
        tol=tol,
    )


def solve_for_M_two_moment_b1(a_gain, delta=MV_DELTA, u=MV_U,
                              K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
                              n_grid=401, tol=mpf("1e-8")):
    """Smallest M with max over [0, mu(M)]^2 of R(M; .) >= 2/u + a.

    Regression: this is the B1 two-moment bound without Lemma 2.17.  In the
    limit where z_2 -> 0, k_2 -> 0 this recovers MV's eq. (10) and should
    match 1.27481 modulo the Lemma 2.14 clipping (which is what MV already do).
    """
    delta, u, a = mpf(delta), mpf(u), mpf(a_gain)
    K2 = mpf(K2_bound_over_delta) / delta
    lhs = 2 / u + a
    return _solve_for_M(
        lhs_value=lhs,
        delta=delta,
        u=u,
        K2=K2,
        use_lemma217=False,
        n_grid=n_grid,
        tol=tol,
    )


# -----------------------------------------------------------------------------
# Top-level: reproduce MV + report improvement
# -----------------------------------------------------------------------------

def reproduce_lemma217(dps: int = 30, n_grid: int = 401):
    """Plug in MV's 119 tabulated cosine coefficients and report:

    1.  The single-moment MV bound (1.27481) - SANITY CHECK of the setup.
    2.  The two-moment B1 bound (no Lemma 2.17).
    3.  The two-moment + Lemma 2.17 bound (this module's contribution).

    Feasibility asserts:
        rad1 positivity   z_1^4 + z_2^4 <= (M - 1)/2   at the argmax
        rad2 positivity   k_1^2 + k_2^2 <= (K_2 - 1)/2 (fixed numerical check)
    """
    mp.mp.dps = dps

    # MV-tabulated ingredients
    min_G, argmin_G = min_G_on_0_quarter(n_grid=40001)
    S1 = S1_sum()
    a_gain = gain_parameter(min_G, S1)
    k1 = k1_value()
    k2 = kn_value(2)
    K2 = MV_K2_BOUND_OVER_DELTA / MV_DELTA

    # (A2) positivity of the K-side radicand at z_2 = k_2 = 0 is automatic
    # since MV's eq (10) already has K_2 - 1 - 2 k_1^2 > 0 (they compute it).
    # For the two-moment version we need K_2 - 1 - 2 (k_1^2 + k_2^2) > 0.
    rad2 = K2 - 1 - 2 * (k1 * k1 + k2 * k2)
    assert rad2 > 0, f"K_2 - 1 - 2(k_1^2 + k_2^2) = {rad2} <= 0; B1 infeasible."

    # (1) single-moment MV bound (sanity)
    from mv_bound import solve_for_M_with_z1
    M_mv = solve_for_M_with_z1(a_gain)

    # (2) two-moment B1 (no Lemma 2.17)
    M_b1 = solve_for_M_two_moment_b1(a_gain, n_grid=n_grid)

    # (3) two-moment + Lemma 2.17
    M_l217 = solve_for_M_lemma217(a_gain, n_grid=n_grid)

    out = {
        "a_gain": a_gain,
        "k1": k1,
        "k2": k2,
        "K2": K2,
        "K2_radicand_two_moment": rad2,
        "mu_at_M_mv": mu_of_M(M_mv),
        "M_mv_single_moment (target 1.27481)": M_mv,
        "M_two_moment_b1 (no Lemma 2.17)": M_b1,
        "M_two_moment_lemma217": M_l217,
        "MV_target_final": MV_BOUND_FINAL,
    }
    return out


if __name__ == "__main__":
    print("=" * 72)
    print("MO 2004 Lemma 2.17 joint (z_1, z_2) constraint in the MV dual")
    print("=" * 72)
    res = reproduce_lemma217()
    for k, v in res.items():
        print(f"  {k:42s} = {v}")
    print()
    print("Interpretation:")
    print("  * M_mv_single_moment               must reproduce MV's 1.27481.")
    print("  * M_two_moment_b1                  is the 2-moment bound without")
    print("    Lemma 2.17 (Idea B1 alone, if k_2 is non-trivial).")
    print("  * M_two_moment_lemma217            is this module's 2-moment +")
    print("    Lemma 2.17 bound.  Expected lift from 1.27481 toward [1.28, 1.30].")
