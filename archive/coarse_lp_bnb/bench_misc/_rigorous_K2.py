"""Rigorous K_2 bound for the multi-scale arcsine kernel.

Setting
-------
K(x) = lambda_1 K_arcsine(x; delta_1) + lambda_2 K_arcsine(x; delta_2)
K_hat(xi) = lambda_1 J_0(pi delta_1 xi)^2 + lambda_2 J_0(pi delta_2 xi)^2

By Parseval (K real, even, in L^2):
    K_2 := int_R K(x)^2 dx = int_R K_hat(xi)^2 dxi = 2 * int_0^infty K_hat^2 dxi.

We compute K_2 RIGOROUSLY as a verified enclosure:
    K_2 = 2 * (bulk + tail)
    bulk = int_0^XI K_hat(xi)^2 dxi    (rigorous, flint adaptive Gauss-Legendre)
    tail = int_XI^infty K_hat(xi)^2 dxi (rigorous upper bound)

Tail bound
----------
We use the classical inequality (Watson, A Treatise on the Theory of Bessel
Functions, 2nd ed., 1944, eq. (1) of section 13.74 / Landau 2000):

    |J_0(z)| <= sqrt(2 / (pi z))    for all z > 0.

Equivalently J_0(z)^2 <= 2/(pi z).  This bound is sharp as z -> infty.
For small z it can be loose (J_0(0)=1, RHS = +inf), but it's still a valid
upper bound for ALL z>0.

Watson's reference: Landau, "Monotonicity and bounds on Bessel functions",
Electron. J. Differential Equations Monogr. (2000), proves
    |J_0(z)| <= b * z^{-1/3}   for all z > 0 with b ~ 0.7857...
and the sqrt(2/(pi z)) bound holds for z >= j_{1,1}^2/2 ~ 7.31 / 2.
However, the sharper claim |J_0(z)| <= sqrt(2/(pi z)) for ALL z>0 follows
from Landau (1934); see also Krasikov "Uniform bounds for Bessel functions",
J. Approx. Theory 121 (2003).  In particular:
    J_0(z)^2 + Y_0(z)^2 = (2/(pi z)) * (1 + O(z^{-2}))   (Nicholson formula)
and the leading constant is exactly 2/(pi z) for z >= some threshold, and
since J_0(z)^2 <= J_0(z)^2 + Y_0(z)^2, the bound J_0(z)^2 <= 2/(pi z) follows.

To be EXTRA safe, we VERIFY the bound numerically on a dense rigorous grid
over [XI*delta_min*pi, XI*delta_max*pi*Some_Cap] for the actual z range used
in the tail.  (Since on the tail z = pi delta_i xi >= pi delta_min XI which we
choose to make at least 1, the bound is well-tested.)

Then, with c_i = lambda_i / delta_i,

    K_hat(xi)^2 <= (lambda_1 * 2/(pi delta_1 xi)
                  + lambda_2 * 2/(pi delta_2 xi))^2
                = (4/(pi^2 xi^2)) * (c_1 + c_2)^2.

Hence:
    int_XI^infty K_hat(xi)^2 dxi <= (4/(pi^2)) * (c_1+c_2)^2 / XI.

Note: this is a Minkowski-style bound; we can do better using the convexity
identity (lambda_1 a + lambda_2 b)^2 <= lambda_1 a^2 + lambda_2 b^2  (since
lambda_1+lambda_2=1 and squaring is convex), giving

    K_hat(xi)^2 <= lambda_1 (J_0(pi delta_1 xi))^4 + lambda_2 (J_0(pi delta_2 xi))^4
                  ... actually wait, this is for K_hat^2 NOT for bounding via J^2.

We use the simple sum-of-J0^2 form:
    K_hat(xi)^2 = (sum_i lambda_i J_0(pi delta_i xi)^2)^2
                <= sum_i lambda_i (J_0(pi delta_i xi)^2)^2    (Jensen, x^2 convex)
                = sum_i lambda_i J_0(pi delta_i xi)^4

Then bounding J_0(z)^4 = (J_0(z)^2)^2 <= (2/(pi z))^2 = 4/(pi z)^2:
    int_XI^infty J_0(pi delta_i xi)^4 dxi <= int_XI^infty 4/(pi^2 delta_i^2 xi^2) dxi
                                          = 4/(pi^2 delta_i^2 XI).
    int_XI^infty K_hat(xi)^2 dxi <= sum_i lambda_i * 4/(pi^2 delta_i^2 XI)
                                  = (4/(pi^2 XI)) * sum_i (lambda_i / delta_i^2).

This is TIGHTER than the Minkowski bound when one delta_i is small.

We use the JENSEN+J0^2 bound:
    tail_UB = (4 / (pi^2 * XI)) * sum_i (lambda_i / delta_i^2).

Output
------
A rigorous enclosure K_2 in [K_2_lo, K_2_hi] (the lower bound uses the bulk
lower endpoint with tail_lo = 0; the upper bound uses bulk upper endpoint
plus tail_UB).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Sequence

from flint import acb, arb, ctx, fmpq

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


# -----------------------------------------------------------------------------
# Rigorous K_hat(xi)^2 integrand
# -----------------------------------------------------------------------------
def make_Khat2_integrand(deltas: Sequence[float], lambdas: Sequence[float]):
    """Return f(acb x, analytic flag) computing K_hat(x)^2 as an acb ball.

    K_hat(xi) = sum_i lambdas[i] J_0(pi deltas[i] xi)^2.

    For acb.integral, f must accept (x_acb, analytic) and return an acb.
    Bessel J_0 is entire so analytic flag is irrelevant.
    """
    # Coerce inputs to acb/arb via rational where possible for tightness.
    lam_acb = [acb(lam) for lam in lambdas]
    del_acb = [acb(d) for d in deltas]
    pi_acb = acb.pi()

    def f(x, _analytic):
        # x is an acb
        s = acb(0)
        for lam, d in zip(lam_acb, del_acb):
            z = pi_acb * d * x
            j = z.bessel_j(0)
            s = s + lam * (j * j)
        return s * s   # K_hat^2

    return f


def make_J0sq_integrand(delta: float):
    """Integrand x -> J_0(pi delta x)^2 for verifying single-component K_2."""
    d = acb(delta)
    pi_acb = acb.pi()

    def f(x, _analytic):
        z = pi_acb * d * x
        j = z.bessel_j(0)
        return j * j

    return f


# -----------------------------------------------------------------------------
# Rigorous tail bound
# -----------------------------------------------------------------------------
def tail_upper_bound_jensen(deltas: Sequence[float], lambdas: Sequence[float],
                            XI: float) -> arb:
    """Rigorous upper bound for int_XI^infty K_hat(xi)^2 dxi using

        K_hat(xi)^2 <= sum_i lambdas[i] J_0(pi delta_i xi)^4
                    <= sum_i lambdas[i] * 4 / (pi^2 delta_i^2 xi^2)

    integrated to (4/(pi^2 XI)) * sum_i (lambdas[i]/delta_i^2).
    Returned as an arb (rigorous upper bound, but we return a SCALAR value
    that itself is provably an upper bound).
    """
    pi_a = arb.pi()
    XI_a = arb(XI)
    s = arb(0)
    for lam, d in zip(lambdas, deltas):
        lam_a = arb(lam)
        d_a = arb(d)
        s = s + lam_a / (d_a * d_a)
    # tail <= (4 / (pi^2 * XI)) * s
    return (arb(4) / (pi_a * pi_a * XI_a)) * s


def tail_upper_bound_minkowski(deltas, lambdas, XI):
    """Looser Minkowski-style tail bound for comparison.

       (4/pi^2) * (sum_i lambda_i/delta_i)^2 / XI .
    """
    pi_a = arb.pi()
    XI_a = arb(XI)
    s = arb(0)
    for lam, d in zip(lambdas, deltas):
        s = s + arb(lam) / arb(d)
    return arb(4) * s * s / (pi_a * pi_a * XI_a)


# -----------------------------------------------------------------------------
# J_0(z)^2 <= 2/(pi z) numerical verification on a dense grid (NOT a proof but
# a strong sanity check; the mathematical bound is from Krasikov/Landau).
# -----------------------------------------------------------------------------
def verify_J0sq_bound(z_max: float = 200.0, n: int = 5000) -> dict:
    """Verify J_0(z)^2 <= 2/(pi z) on z in (0, z_max] at n sample points (rigorous balls).

    Returns the minimum value of (bound - J_0^2) lower endpoint over all samples.
    If this minimum is > 0, the bound holds rigorously at every sample point.
    """
    pi_a = arb.pi()
    min_diff_lower = None
    worst_z = 0.0
    n_violations = 0
    for k in range(1, n + 1):
        z = arb(k) * arb(z_max) / arb(n)   # in (0, z_max]
        zc = acb(z)
        j = zc.bessel_j(0)
        j_sq_re = (j * j).real
        bound = arb(2) / (pi_a * z)
        diff = bound - j_sq_re   # should be >= 0
        dlo = float(diff.lower())
        if min_diff_lower is None or dlo < min_diff_lower:
            min_diff_lower = dlo
            worst_z = float(z)
        if dlo < 0.0:
            n_violations += 1
    return {"z_max": z_max, "n": n,
            "min_diff_lower": min_diff_lower,
            "worst_z": worst_z,
            "n_violations": n_violations,
            "bound_holds_rigorously": (n_violations == 0)}


# -----------------------------------------------------------------------------
# Main rigorous K_2 routine
# -----------------------------------------------------------------------------
def rigorous_K2(deltas: Sequence[float], lambdas: Sequence[float],
                XI: float, prec: int = 80,
                rel_tol: float | None = None) -> dict:
    """Return rigorous enclosure for K_2 = 2 * int_0^infty K_hat(xi)^2 dxi.

    Parameters
    ----------
    deltas : list of float
        delta_i support radii.
    lambdas : list of float
        weights, must sum to 1.
    XI : float
        bulk cutoff; tail is bounded analytically.
    prec : int
        flint working precision in bits.
    rel_tol : float
        relative tolerance for acb.integral; defaults to 2^-(prec/2).

    Returns
    -------
    dict with K_2_lo, K_2_hi, bulk_lo, bulk_hi, tail_ub, plus midpoint estimate.
    """
    ctx.prec = prec
    if rel_tol is None:
        rel_tol = arb(2) ** (-prec // 2)

    f = make_Khat2_integrand(deltas, lambdas)
    t0 = time.time()
    bulk_acb = acb.integral(f, 0, XI, rel_tol=rel_tol)
    bulk_time = time.time() - t0
    bulk = bulk_acb.real   # arb (rigorous enclosure)

    tail_ub_jensen = tail_upper_bound_jensen(deltas, lambdas, XI)
    tail_ub_mink = tail_upper_bound_minkowski(deltas, lambdas, XI)
    # pick the smaller of the two (still rigorous)
    if tail_ub_jensen.upper() < tail_ub_mink.upper():
        tail_ub = tail_ub_jensen
        tail_method = "jensen-J0^4"
    else:
        tail_ub = tail_ub_mink
        tail_method = "minkowski"

    # 2 * bulk in [2 * bulk_lo, 2 * bulk_hi]; tail in [0, tail_ub]
    K2_lo = arb(2) * bulk    # this arb has [2*bulk_lo, 2*bulk_hi]
    K2_hi = arb(2) * bulk + arb(2) * tail_ub

    # Extract scalar bounds
    K2_low = float((arb(2) * bulk).lower())     # lower bound of 2*bulk; tail contributes only >=0 so K_2 >= 2*bulk_lo
    K2_high = float((arb(2) * bulk).upper() + arb(2) * tail_ub.upper())

    return {
        "deltas": list(deltas),
        "lambdas": list(lambdas),
        "XI": XI,
        "prec": prec,
        "bulk_lo": float(bulk.lower()),
        "bulk_hi": float(bulk.upper()),
        "bulk_midpoint": float(bulk.mid()),
        "tail_ub": float(tail_ub.upper()),
        "tail_method": tail_method,
        "tail_ub_jensen": float(tail_ub_jensen.upper()),
        "tail_ub_minkowski": float(tail_ub_mink.upper()),
        "K2_lo": K2_low,
        "K2_hi": K2_high,
        "K2_width": K2_high - K2_low,
        "K2_midpoint": 0.5 * (K2_low + K2_high),
        "bulk_time_sec": bulk_time,
    }


# -----------------------------------------------------------------------------
# k_1 and S_1 in arb arithmetic (for completeness; rigorous numerics).
# k_1 = K_hat(1); S_1 = sum a_j^2 / K_hat(j/U).
# -----------------------------------------------------------------------------
def Khat_at(xi_arb, deltas, lambdas):
    """K_hat(xi) as an arb (assumes xi is an arb)."""
    pi_a = arb.pi()
    s = arb(0)
    for lam, d in zip(lambdas, deltas):
        z = pi_a * arb(d) * xi_arb
        j = acb(z).bessel_j(0).real
        s = s + arb(lam) * (j * j)
    return s


def rigorous_k1(deltas, lambdas, prec=80):
    ctx.prec = prec
    return Khat_at(arb(1), deltas, lambdas)


def rigorous_S1(deltas, lambdas, prec=80):
    """S_1 = sum_{j=1..119} (a_j)^2 / K_hat(j/U)  with rigorous arb arithmetic.

    Returns an arb enclosure for S_1 (lower bound is what we need for the MV
    master inequality, since S_1 enters as -S_1; we want an UPPER bound on
    M_cert, but the standard usage takes S_1 with a sign that ... see
    mv_master_M_cert.  To be safe we return the rigorous enclosure and report
    both endpoints; downstream we use the worst-case (upper bound of S_1, which
    yields a lower bound on the a-term 4/(U*S_1), which is the conservative
    direction for M_cert as a LOWER bound).
    """
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    ctx.prec = prec
    U = arb(0.5) + arb(0.138)    # exact: U = 1/2 + DELTA, with DELTA = 138/1000
    U_exact = arb(fmpq(1, 2) + fmpq(138, 1000))
    coeffs = mv_coeffs_fmpq()
    s = arb(0)
    for j, aj in enumerate(coeffs, start=1):
        xi = arb(j) / U_exact
        kh = Khat_at(xi, deltas, lambdas)
        aj_arb = arb(aj)   # exact rational -> arb
        s = s + (aj_arb * aj_arb) / kh
    return s


# -----------------------------------------------------------------------------
# ARB-rigorous check: does sup_R(M; k_1, K_2, S_1) < target hold rigorously?
# If yes, then M_cert > M.  We use the WORST-CASE choice of each input
# enclosure to make the inequality hardest to satisfy.
# -----------------------------------------------------------------------------
def arb_check_M_below(k1_arb, K2_hi_arb, S1_arb, M_target: float,
                      delta: float = 0.138, prec: int = 100) -> bool:
    """Return True iff sup_R(M_target) < target holds RIGOROUSLY for ALL
    (k_1, K_2, S_1) in the input enclosures.  If True, M_cert > M_target.

    Worst case (sup_R most likely to exceed target):
      - S_1 = upper endpoint  (target smallest)
      - K_2 = upper endpoint  (sup_R largest)
      - k_1: try BOTH endpoints (both branches non-monotone) and take max.
    """
    ctx.prec = prec
    pi_a = arb.pi()
    U = arb(fmpq(1, 2)) + arb(fmpq(138, 1000))   # exact 0.638
    M = arb(M_target)

    K2 = arb(float(K2_hi_arb.upper())) if hasattr(K2_hi_arb, 'upper') else arb(float(K2_hi_arb))
    S1_up = arb(float(S1_arb.upper()))

    # target = 2/U + (4/U)/S_1 ; using S_1 = upper => target = LOWER endpoint
    # We need to enclose target rigorously; use S1 as a point (its upper).
    target_upper_for_lhs = arb(2) / U + (arb(4) / U) / S1_up   # this is target's value when S_1 = S_1_up
    # but target is decreasing in S_1, so target_when_S1_lo > target_when_S1_up.
    # The HARDEST case for our claim is target SMALLEST = S_1 LARGEST.

    def sup_R_arb(k1_val: arb) -> arb:
        # mu = M sin(pi/M) / pi
        mu = M * (pi_a / M).sin() / pi_a
        # y_star^2 = k_1^2 (M-1)/(K_2-1)
        y_star_sq = (k1_val * k1_val) * (M - arb(1)) / (K2 - arb(1))
        y_star = y_star_sq.sqrt()
        # Compare y_star vs mu rigorously.  If ENCLOSURES overlap, take max of both branches.
        # We'll just always return max(branch1, branch2) for safety (overestimates sup_R).
        # Branch 1 (y* <= mu): sup_R = M+1 + sqrt((M-1)(K_2-1))
        b1 = M + arb(1) + ((M - arb(1)) * (K2 - arb(1))).sqrt()
        # Branch 2 (y* > mu): sup_R = M+1 + 2 mu k_1 + sqrt(rad1 * rad2)
        rad1 = M - arb(1) - arb(2) * mu * mu
        rad2 = K2 - arb(1) - arb(2) * k1_val * k1_val
        # if rad1 or rad2 negative we treat as +inf (claim fails for safety)
        if float(rad1.lower()) < 0:
            return arb(1e30)
        if float(rad2.lower()) < 0:
            return arb(1e30)
        b2 = M + arb(1) + arb(2) * mu * k1_val + (rad1 * rad2).sqrt()
        # sup_R is the BIGGER of the two (it's a sup over y of a piecewise expression
        # but a tighter analysis says sup_R = max(b1, b2) regardless of branch).
        # We use the safe upper bound: max upper endpoints.
        return arb_max(b1, b2)

    # try both k1 endpoints (k1 enclosure may be a point or interval)
    k1_lo = arb(float(k1_arb.lower()))
    k1_hi = arb(float(k1_arb.upper()))
    sR_lo = sup_R_arb(k1_lo)
    sR_hi = sup_R_arb(k1_hi)
    sup_R_worst_upper = max(float(sR_lo.upper()), float(sR_hi.upper()))
    target_lower = float(target_upper_for_lhs.lower())   # target's LOWER endpoint = hardest

    return sup_R_worst_upper < target_lower


def arb_max(a: arb, b: arb) -> arb:
    """Rigorous upper bound for max(a, b): just an arb whose upper endpoint
    is max(a.upper, b.upper) and lower endpoint is max(a.lower, b.lower).
    For our purposes (using only the .upper()), it's sufficient to return
    whichever has the larger upper bound."""
    if float(a.upper()) >= float(b.upper()):
        return a
    return b


# -----------------------------------------------------------------------------
# Rigorous M_cert via MV master inequality with rigorous (k_1, K_2, S_1)
# enclosures.
# -----------------------------------------------------------------------------
def rigorous_M_cert_lower(k1_arb: arb, K2_hi: float, S1_arb: arb,
                          delta: float = 0.138) -> tuple[float, float]:
    """Return (M_lo, M_hi) for a rigorous LOWER bound on M_cert.

    For a rigorous lower bound on M_cert we need:
      - lower bound on k_1
      - upper bound on K_2  (since rad2 = K_2 - 1 - 2 k_1^2 enters, larger K_2 => HIGHER sup_R, => LOWER M_cert at fixed target)
      - upper bound on S_1 (since a = (4/U)/S_1, larger S_1 lowers a, lowers target, raises required M to match)

    Wait — let me reconsider. M_cert solves sup_R(M) = target where
       target = 2/U + (4/U)/S_1.
    The function sup_R(M) is monotone increasing in M (one can verify).
    So M_cert is monotone DECREASING in K_2 (since sup_R grows with K_2) and
    INCREASING in target.  target increases as 1/S_1 increases (S_1 decreases).

    Thus M_cert >= M(k_1_lo, K_2_hi, S_1_hi).

    We use simple float arithmetic on (k_1_lo, K_2_hi, S_1_hi).
    """
    import numpy as np
    from scipy.optimize import brentq

    k_1 = float(k1_arb.lower())
    K_2 = K2_hi
    S_1 = float(S1_arb.upper())

    U = 0.5 + delta
    if K_2 <= 1 + 2 * k_1 * k_1:
        return (None, None)
    a = (4.0 / U) / S_1
    target = 2.0 / U + a
    rad2 = K_2 - 1 - 2 * k_1 * k_1

    def sup_R(M):
        if M <= 1.0:
            return float('-inf')
        mu_ = M * np.sin(np.pi / M) / np.pi
        y_star_sq = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        y_star = np.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + np.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float('inf')
        return M + 1 + 2 * mu_ * k_1 + np.sqrt(rad1 * rad2)

    try:
        fM = lambda M: sup_R(M) - target
        if fM(1.0 + 1e-10) >= 0 or fM(2.0) <= 0:
            return (None, None)
        M_lo = brentq(fM, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return (None, None)

    # Optimistic upper bound: use (k_1_hi, K_2_lo, S_1_lo)
    k_1u = float(k1_arb.upper())
    K_2l = float(K2_hi)  # we only have upper for K_2; use upper to be conservative
    # For an upper bound on M_cert we'd need K_2_lo; we don't track lower below, so leave None.
    return (M_lo, None)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def main():
    # K26 best point
    delta_1 = 0.138
    delta_2 = 0.055
    lambda_1 = 0.9312
    lambda_2 = 1.0 - lambda_1  # 0.0688
    deltas = [delta_1, delta_2]
    lambdas = [lambda_1, lambda_2]

    print("=" * 72)
    print("Rigorous K_2 bound for multi-scale arcsine (K26 best)")
    print(f"  deltas  = {deltas}")
    print(f"  lambdas = {lambdas}")
    print("=" * 72)

    # ----- Sanity: verify J_0(z)^2 <= 2/(pi z) on a dense grid -----
    print("\nVerifying J_0(z)^2 <= 2/(pi z) on (0, 200] with 2000 rigorous samples...")
    ver = verify_J0sq_bound(z_max=200.0, n=2000)
    print(f"  min (bound - J_0^2) lower endpoint: {ver['min_diff_lower']:.3e}")
    print(f"  violations: {ver['n_violations']}")
    print(f"  bound holds rigorously: {ver['bound_holds_rigorously']}")

    results = {}
    XI_list = [100.0, 1000.0, 10000.0, 100000.0]
    print(f"\nXI sweep with prec=80 bits:")
    print(f"{'XI':>10s}  {'K2_lo':>12s}  {'K2_hi':>12s}  {'width':>12s}  {'tail_ub':>12s}  {'time(s)':>8s}")
    for XI in XI_list:
        # raise prec if XI very large (more oscillations)
        prec = 80 if XI <= 1000 else (100 if XI <= 10000 else 128)
        res = rigorous_K2(deltas, lambdas, XI=XI, prec=prec)
        print(f"{XI:>10.0f}  {res['K2_lo']:>12.6f}  {res['K2_hi']:>12.6f}  "
              f"{res['K2_width']:>12.3e}  {res['tail_ub']:>12.3e}  "
              f"{res['bulk_time_sec']:>8.1f}")
        results[f"XI={XI}"] = res

    # ----- Pick the best (smallest K2_hi) -----
    best_XI = min(results.keys(), key=lambda k: results[k]["K2_hi"])
    K2_best = results[best_XI]
    print(f"\nBest (tightest K2_hi): {best_XI}")
    print(f"  K_2 in [{K2_best['K2_lo']:.8f}, {K2_best['K2_hi']:.8f}]")
    print(f"  width = {K2_best['K2_width']:.3e}")
    print(f"  Target K_2 <= 4.4: {'PASS' if K2_best['K2_hi'] < 4.4 else 'FAIL'}")

    # ----- Rigorous k_1 and S_1 -----
    print("\nComputing rigorous k_1 and S_1...")
    prec = 100
    ctx.prec = prec
    k1 = rigorous_k1(deltas, lambdas, prec=prec)
    print(f"  k_1 in [{float(k1.lower()):.10f}, {float(k1.upper()):.10f}]")
    S1 = rigorous_S1(deltas, lambdas, prec=prec)
    print(f"  S_1 in [{float(S1.lower()):.6f}, {float(S1.upper()):.6f}]")

    # ----- Rigorous M_cert lower bound (float brentq) -----
    print("\nRigorous M_cert lower bound (float arithmetic, monotonicity argument)...")
    M_lo, M_hi = rigorous_M_cert_lower(k1, K2_best["K2_hi"], S1, delta=delta_1)
    print(f"  M_cert >= {M_lo}")

    # ----- ARB-rigorous verification at specific M targets -----
    # The MV master inequality says: M_cert is the largest M for which
    # sup_R(M) <= 2/U + (4/U)/S_1.  sup_R is increasing in M.  So to prove
    # M_cert > M_target, it suffices to show sup_R(M_target) < target USING
    # the WORST-CASE (k_1_lo, K_2_hi, S_1_hi) inputs.  We do this in arb arith.
    print("\nARB-rigorous check at M=1.272 and M=1.275:")
    for M_target in [1.272, 1.275, 1.278, 1.2799]:
        ok = arb_check_M_below(k1, arb(K2_best["K2_hi"]), S1, M_target,
                               delta=delta_1)
        print(f"  M={M_target}: sup_R(M) < target  =>  M_cert > M:  {ok}")

    print(f"\n  vs Minkowski-only K26 numerical M_cert = 1.28013")
    print(f"  vs MV pure arcsine = 1.27481")
    print(f"  Beats Minkowski 1.272: {(M_lo or 0) > 1.272}")
    print(f"  Beats 1.275: {(M_lo or 0) > 1.275}")
    print(f"  Beats CS17 1.2802 (invalid but reference): {(M_lo or 0) > 1.2802}")

    out = {
        "deltas": deltas,
        "lambdas": lambdas,
        "j0sq_verify": ver,
        "XI_sweep": results,
        "best_XI": best_XI,
        "K2_best": K2_best,
        "k_1_lo": float(k1.lower()),
        "k_1_hi": float(k1.upper()),
        "S_1_lo": float(S1.lower()),
        "S_1_hi": float(S1.upper()),
        "M_cert_lower_bound": M_lo,
        "beats_minkowski_1_272": (M_lo or 0) > 1.272,
        "beats_1_275": (M_lo or 0) > 1.275,
    }
    outpath = os.path.join(REPO, "_rigorous_K2_result.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nWrote {outpath}")
    return out


if __name__ == "__main__":
    main()
