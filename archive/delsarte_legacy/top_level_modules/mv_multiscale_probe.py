"""Multi-scale arcsine-kernel probe for the MV master inequality.

GOAL: Replace the single kernel K_delta in MV's eq.(7)/(10) by a convex
combination K = sum_i lambda_i K_{delta_i} (same arcsine shape, multiple
scales) and check whether the resulting M-bound exceeds MV's 1.2748.

This is a LIGHT numerical probe (no rigorous LP solve, no QP re-optimization
of G coefficients).  It uses MV's fixed 119 a_j and the existing arcsine
self-conv K, swapping in mixtures of scales.

The relevant MV master inequality with a single K is
    2/u + a  <=  M + 1 + 2 z1^2 k1
                  + sqrt(max(0, M - 1 - 2 z1^4)) * sqrt(max(0, K2 - 1 - 2 k1^2))

For a mixture K = sum_i lambda_i K_{delta_i}:
    K_hat(j)  = sum_i lambda_i * J_0(pi j delta_i / u)^2     (period-u FT)
    k_n       = sum_i lambda_i * J_0(pi n delta_i)^2         (period-1 FT)
    ||K||_2^2 = sum_{i,j} lambda_i lambda_j <K_{d_i}, K_{d_j}>
              <= 0.5747 * (sum_i lambda_i / sqrt(d_i))^2     [Cauchy-Schwarz]
The CS bound on ||K||_2^2 is what MV use at single scale (sharp at d_i = d_j);
for distinct scales it is an upper bound (so phi remains a valid LB on M).

For the gain a, we use the same G(x) (MV's a_j fixed) but recompute S_1
relative to the MIXED kernel.  Because the gain argument uses Parseval applied
to (G * K), and G is a finite Fourier series, the natural per-scale S_1's
combine: integration vs the mixture gives sum_i lambda_i <G K_{d_i}>, and
applying CS once more bounds this by sqrt(sum a_j^2 * (sum_i lambda_i k_{d_i}_hat(j/u))^{-1}).

So the multiscale S_1 (mixed):
    S_1^{mix} = sum_j a_j^2 / [ sum_i lambda_i J_0(pi j delta_i / u)^2 ]
gain a^{mix} = (4/u) * (min G)^2 / S_1^{mix}.

The multi-scale K2 we use:
    K2^{mix} = (sum_i lambda_i / delta_i) * 0.5747       [diagonal sum]
            + 0.5747 * sum_{i!=j} lambda_i lambda_j / sqrt(delta_i delta_j)
i.e., 0.5747 * (sum_i lambda_i / sqrt(delta_i))^2.

We grid-search lambda for each (delta_1, delta_2) pair and report the best M.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpmath as mp
import numpy as np
from mpmath import mpf

from delsarte_dual.mv_bound import (
    MV_COEFFS_119, MV_DELTA, MV_U, MV_BOUND_FINAL,
    min_G_on_0_quarter, k1_value,
)
from delsarte_dual.mv_delta_u_scan import MV_C0  # 0.5747


def k_n_mixture(n, lambdas, deltas):
    """k_n = sum_i lambda_i * J_0(pi n delta_i)^2  (period-1 FT)."""
    total = mpf(0)
    for lam, d in zip(lambdas, deltas):
        j0 = mp.besselj(0, mp.pi * mpf(n) * mpf(d))
        total += mpf(lam) * j0 * j0
    return total


def K_hat_period_u(j, lambdas, deltas, u):
    """tilde_K(j) for the mixture, period-u."""
    total = mpf(0)
    u = mpf(u)
    for lam, d in zip(lambdas, deltas):
        j0 = mp.besselj(0, mp.pi * mpf(j) * mpf(d) / u)
        total += mpf(lam) * j0 * j0
    return total


def S1_mixture(a_coeffs, lambdas, deltas, u):
    """S_1^{mix} = sum a_j^2 / [ sum_i lam_i J_0(pi j d_i / u)^2 ]."""
    total = mpf(0)
    u = mpf(u)
    for j, a in enumerate(a_coeffs, start=1):
        denom = mpf(0)
        for lam, d in zip(lambdas, deltas):
            j0 = mp.besselj(0, mp.pi * mpf(j) * mpf(d) / u)
            denom += mpf(lam) * j0 * j0
        if denom == 0:
            return mp.mpf("+inf")
        total += mpf(a) ** 2 / denom
    return total


def K2_mixture_upper(lambdas, deltas, C0=MV_C0):
    """Upper bound (Cauchy-Schwarz) on ||K||_2^2 for the mixture.

    ||K||_2^2 = int (sum_i lam_i K_i)^2
              = sum_{ij} lam_i lam_j <K_i, K_j>
              <= sum_{ij} lam_i lam_j sqrt(<K_i, K_i> <K_j, K_j>)
              = (sum_i lam_i sqrt(<K_i, K_i>))^2
              = C0 * (sum_i lam_i / sqrt(d_i))^2.
    """
    s = mpf(0)
    for lam, d in zip(lambdas, deltas):
        s += mpf(lam) / mp.sqrt(mpf(d))
    return mpf(C0) * s * s


def evaluate_mixture(lambdas, deltas, u=MV_U, a_coeffs=MV_COEFFS_119,
                     n_grid_minG=40001, dps=30):
    """Compute MV bound M for the multi-scale mixture (lam_i, delta_i)."""
    mp.mp.dps = dps
    lambdas = [mpf(l) for l in lambdas]
    deltas = [mpf(d) for d in deltas]

    # Normalize lambdas
    s = sum(lambdas)
    if s <= 0:
        raise ValueError("sum lambda_i must be > 0")
    lambdas = [l / s for l in lambdas]

    # min_G unchanged (depends only on a_j and u, not on K)
    min_G, _ = min_G_on_0_quarter(a_coeffs=a_coeffs, u=u, n_grid=n_grid_minG)

    if min_G <= 0:
        return {"valid": False, "reason": f"min_G={min_G}<=0", "M": None}

    # S_1 with mixture
    S1 = S1_mixture(a_coeffs, lambdas, deltas, u)
    gain = (mpf(4) / mpf(u)) * mpf(min_G) ** 2 / S1

    # k_1 with mixture
    k1 = k_n_mixture(1, lambdas, deltas)

    # K2 upper via CS
    K2 = K2_mixture_upper(lambdas, deltas)

    # Master inequality (with z_1 refinement, optimized)
    lhs = mpf(2) / mpf(u) + gain

    rad2 = K2 - 1 - 2 * k1 * k1
    if rad2 < 0:
        return {"valid": False, "reason": f"rad2={float(rad2)}<0",
                "M": None, "min_G": min_G, "S1": S1, "gain": gain,
                "k1": k1, "K2": K2}

    A = mp.sqrt(rad2)

    # solve_for_M_with_z1 logic, fixed-point on z1
    def solve_M(z1):
        z1 = mpf(z1)
        z1sq = z1 * z1
        z14 = z1sq * z1sq
        c_eff = lhs - 2 - 2 * z14 - 2 * z1sq * k1
        disc = A * A + 4 * c_eff
        if disc < 0:
            return None
        s_star = (-A + mp.sqrt(disc)) / 2
        if s_star < 0:
            s_star = mpf(0)
        return s_star ** 2 + 1 + 2 * z14

    def best_z1(M_val):
        # interior critical point in t = z1^2:
        # d/dt [ M + 1 + 2 t k1 + sqrt(M-1-2 t^2) A ] = 2 k1 - 2 t A / sqrt(...)
        # set = 0 => t* such that t* / sqrt(M-1-2 t*^2) = k1 / A
        # ratio r = k1/A, then t = r sqrt(M-1) / sqrt(1 + 2 r^2)
        r = k1 / A
        t_star = r * mp.sqrt((M_val - 1)) / mp.sqrt(1 + 2 * r * r)
        # bathtub bound t <= mu(M) = M sin(pi/M) / pi
        t_max = M_val * mp.sin(mp.pi / M_val) / mp.pi
        t_dom = mp.sqrt((M_val - 1) / 2)
        t_opt = min(t_star, t_max, t_dom)
        if t_opt < 0:
            t_opt = mpf(0)
        return mp.sqrt(t_opt)

    M_curr = mpf("1.28")
    z1_curr = best_z1(M_curr)
    for _ in range(50):
        M_new = solve_M(z1_curr)
        if M_new is None:
            return {"valid": False, "reason": "disc<0",
                    "M": None, "min_G": min_G, "S1": S1, "gain": gain,
                    "k1": k1, "K2": K2}
        if abs(M_new - M_curr) < mpf("1e-12"):
            M_curr = M_new
            break
        M_curr = M_new
        z1_curr = best_z1(M_curr)

    return {
        "valid": True,
        "M": M_curr,
        "min_G": min_G,
        "S1": S1,
        "gain": gain,
        "k1": k1,
        "K2": K2,
        "lhs": lhs,
        "z1_opt": z1_curr,
    }


def grid_search_two_scales(delta1, delta2, u=MV_U, n_lam=21, dps=30,
                           a_coeffs=MV_COEFFS_119):
    """Sweep lambda_1 in [0,1] (lambda_2 = 1 - lambda_1) for fixed (d1, d2)."""
    results = []
    best = None
    for lam1 in np.linspace(0.0, 1.0, n_lam):
        lam2 = 1.0 - lam1
        if lam1 == 0:
            r = evaluate_mixture([1.0], [delta2], u=u, dps=dps,
                                 a_coeffs=a_coeffs)
        elif lam2 == 0:
            r = evaluate_mixture([1.0], [delta1], u=u, dps=dps,
                                 a_coeffs=a_coeffs)
        else:
            r = evaluate_mixture([lam1, lam2], [delta1, delta2], u=u,
                                 dps=dps, a_coeffs=a_coeffs)
        r["lam1"] = float(lam1)
        results.append(r)
        if r["valid"] and r["M"] is not None:
            if best is None or float(r["M"]) > float(best["M"]):
                best = r
    return best, results


def main():
    mp.mp.dps = 30
    print("=" * 72)
    print("Multi-scale arcsine kernel probe for MV master inequality")
    print("=" * 72)

    # 1. Sanity: reproduce single-scale at MV's delta=0.138.
    print("\n[1] Sanity: single-scale at MV's (delta=0.138, u=0.638)")
    print("-" * 72)
    r0 = evaluate_mixture([1.0], [MV_DELTA], u=MV_U)
    print(f"  M = {float(r0['M']):.6f}    (MV target: {float(MV_BOUND_FINAL):.5f})")
    print(f"  k1={float(r0['k1']):.6f}  K2={float(r0['K2']):.6f}  "
          f"gain={float(r0['gain']):.6f}")

    # 2. Mixture (0.10, 0.18).
    print("\n[2] Two-scale: (delta_1, delta_2) = (0.10, 0.18), u = 0.638")
    print("-" * 72)
    best, allres = grid_search_two_scales(0.10, 0.18, u=MV_U, n_lam=21)
    print(f"  Best: lam1={best['lam1']:.3f}  M={float(best['M']):.6f}")
    print(f"        k1={float(best['k1']):.6f}  K2={float(best['K2']):.6f}  "
          f"gain={float(best['gain']):.6f}")
    print()
    print("  Full sweep:")
    print(f"  {'lam1':>6}  {'M':>10}  {'k1':>8}  {'K2':>8}  {'gain':>8}")
    for r in allres:
        if r["valid"] and r["M"] is not None:
            print(f"  {r['lam1']:6.3f}  {float(r['M']):10.6f}  "
                  f"{float(r['k1']):8.5f}  {float(r['K2']):8.4f}  "
                  f"{float(r['gain']):8.5f}")
        else:
            print(f"  {r['lam1']:6.3f}  INVALID ({r.get('reason','?')})")

    # 3. Sweep over multiple (d1, d2) pairs to see if any beats single scale.
    print("\n[3] Two-scale sweep: pairs around MV's delta=0.138")
    print("-" * 72)
    delta_pairs = [
        (0.10, 0.18),
        (0.11, 0.17),
        (0.12, 0.16),
        (0.13, 0.15),
        (0.125, 0.155),
        (0.115, 0.165),
        (0.105, 0.175),
        (0.108, 0.168),
        (0.13, 0.146),
        (0.130, 0.150),
        (0.130, 0.140),
        (0.135, 0.142),
        (0.135, 0.140),
        (0.138, 0.138),  # = single-scale baseline
    ]
    print(f"  {'d1':>6} {'d2':>6}  {'best lam1':>9}  {'M':>10}  vs MV")
    overall_best = None
    for d1, d2 in delta_pairs:
        # u must be >= 1/2 + max(d1,d2)
        u_use = max(MV_U, 0.5 + max(d1, d2) + 1e-6)
        try:
            b, _ = grid_search_two_scales(d1, d2, u=u_use, n_lam=21)
            if b is not None and b.get("M") is not None:
                M = float(b["M"])
                diff = M - float(MV_BOUND_FINAL)
                marker = "  ***" if diff > 0 else ""
                print(f"  {d1:6.3f} {d2:6.3f}  {b['lam1']:9.3f}  "
                      f"{M:10.6f}  {diff:+.4f}{marker}")
                if overall_best is None or M > float(overall_best["M"]):
                    overall_best = {**b, "d1": d1, "d2": d2, "u": u_use}
            else:
                print(f"  {d1:6.3f} {d2:6.3f}  no valid result")
        except Exception as e:
            print(f"  {d1:6.3f} {d2:6.3f}  ERROR: {e}")

    print("\n" + "=" * 72)
    if overall_best is not None:
        M = float(overall_best["M"])
        diff = M - float(MV_BOUND_FINAL)
        print(f"OVERALL BEST: d1={overall_best['d1']}, d2={overall_best['d2']}, "
              f"lam1={overall_best['lam1']:.3f}, u={overall_best['u']:.4f}")
        print(f"   M = {M:.6f}    vs MV 1.27481    diff = {diff:+.6f}")
        if diff > 0:
            print(f"   *** MULTI-SCALE BEATS MV by {diff:.5f} ***")
        else:
            print(f"   No improvement; multi-scale is dominated by single scale.")


if __name__ == "__main__":
    main()
