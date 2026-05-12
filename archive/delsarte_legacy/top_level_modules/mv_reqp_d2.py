"""Re-solve MV's semi-infinite QP at D2's best (delta, u) = (0.1632, 0.6632).

D2 gave M_with_z1 = 1.2838 at (0.1632, 0.6632) with MV's FIXED 119 a_j.
Those a_j are QP-optimal only at (0.138, 0.638).  Re-optimising at the new
(delta, u) generally yields:
  - smaller S_1 (MV gain denominator),
  - larger gain a = (4/u) * min_G^2 / S_1,
  - larger lower bound M.

This script:
  1. Re-solves the MV QP at (0.1632, 0.6632) with n in {119, 200, 400, 800}.
  2. Plugs the new a_j into the master inequality (eq 10) with the
     SELF-CONSISTENT z_1 fixed-point iteration (same logic as the fixed D2).
  3. Also runs a tight (delta, u) grid scan AROUND (0.1632, 0.6632) with
     re-optimised coefficients, in parallel on the pod.
  4. Reports the best bound.
"""
from __future__ import annotations

import math
import sys

import mpmath as mp
import numpy as np
from mpmath import mpf

try:
    from delsarte_dual.mv_qp_optimize import solve_mv_qp
    from delsarte_dual.mv_bound import (
        MV_K2_BOUND_OVER_DELTA, k1_value, solve_for_M_with_z1,
    )
except ImportError:
    sys.path.insert(0, ".")
    from mv_qp_optimize import solve_mv_qp
    from mv_bound import (
        MV_K2_BOUND_OVER_DELTA, k1_value, solve_for_M_with_z1,
    )


def best_z1_for_M(M_val, delta, u, K2):
    """z_1 in [0, z_max(M)] that maximises RHS of eq (10).  Interior KKT or
    Lemma 3.4 boundary, whichever is smaller."""
    M_val = mpf(M_val)
    k1 = k1_value(delta)
    B2 = mpf(K2) - 1 - 2 * k1 * k1
    if B2 < 0:
        return mp.mpf(0)
    # Interior critical point in t = z_1^2:  t* = k_1 sqrt((M-1)/(B^2 + 2 k_1^2))
    t_star = k1 * mp.sqrt((M_val - 1) / (B2 + 2 * k1 * k1))
    t_max = M_val * mp.sin(mp.pi / M_val) / mp.pi   # = z_max(M)^2
    # rad1 = M - 1 - 2 t^2 >= 0  <=>  t <= sqrt((M-1)/2)   (t = z_1^2)
    t_sqrt_dom = mp.sqrt((M_val - 1) / 2)
    t_opt = min(t_star, t_max, t_sqrt_dom)
    if t_opt < 0:
        t_opt = mpf(0)
    return mp.sqrt(t_opt)


def solve_M_self_consistent(gain, delta, u, K2_bound_over_delta):
    """Fixed-point iteration for self-consistent (M, z_1)."""
    M_current = mpf("1.28")
    K2 = mpf(K2_bound_over_delta) / mpf(delta)
    z1_current = best_z1_for_M(M_current, delta, u, K2)
    for _ in range(60):
        M_new = solve_for_M_with_z1(
            gain, delta=delta, u=u,
            K2_bound_over_delta=K2_bound_over_delta,
            z1=z1_current,
        )
        if abs(M_new - M_current) < mpf("1e-12"):
            M_current = M_new
            break
        M_current = M_new
        z1_current = best_z1_for_M(M_current, delta, u, K2)
    return M_current


def reqp_at(delta, u, n, n_grid=5001, solver="cvxpy", verbose=False):
    """Solve the QP at (delta, u) with n cosines, then compute the
    self-consistent master-inequality bound."""
    res = solve_mv_qp(n=n, delta=delta, u=u, n_grid=n_grid,
                      solver=solver, verbose=verbose)
    gain = mpf(res["gain"])
    M_best = solve_M_self_consistent(gain, delta, u, MV_K2_BOUND_OVER_DELTA)
    return {
        "delta": delta, "u": u, "n": n,
        "S1": res["S1"],
        "min_G": res["min_G"],
        "gain": float(gain),
        "M_reported_by_qp_opt": res["M_with_z1"],
        "M_self_consistent": float(M_best),
    }


def scan_near_d2(n=119, n_grid=3001, n_jobs=32, verbose=True):
    """Parallel scan of (delta, u) around (0.1632, 0.6632) with re-optimised
    coefficients at each grid point."""
    from joblib import Parallel, delayed

    # Tight grid around D2's best, staying on u >= 1/2 + delta.
    deltas = np.linspace(0.14, 0.20, 13)
    # Multiple u ratios: 1/2 + delta, 1/2 + delta + 0.01, etc.
    pairs = []
    for d in deltas:
        for du in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            u_val = 0.5 + float(d) + du
            pairs.append((float(d), u_val))

    def worker(d, u):
        try:
            return reqp_at(d, u, n=n, n_grid=n_grid, verbose=False)
        except Exception as e:
            return {"delta": d, "u": u, "error": str(e)}

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(worker)(d, u) for (d, u) in pairs
    )
    if verbose:
        for r in results:
            if "error" in r:
                print(f"  delta={r['delta']:.4f} u={r['u']:.4f}: FAILED ({r['error']})")
            else:
                print(
                    f"  delta={r['delta']:.4f} u={r['u']:.4f} n={r['n']} "
                    f"min_G={r['min_G']:+.5f} gain={r['gain']:.5f} "
                    f"M={r['M_self_consistent']:.6f}"
                )
    ok = [r for r in results if "error" not in r]
    if not ok:
        return None, results
    best = max(ok, key=lambda r: r["M_self_consistent"])
    return best, results


if __name__ == "__main__":
    mp.mp.dps = 30

    print("=" * 72)
    print("[1] Baseline: MV at (0.138, 0.638), n=119")
    print("=" * 72)
    res_mv = reqp_at(delta=0.138, u=0.638, n=119, n_grid=5001, verbose=False)
    print(f"  S1   = {res_mv['S1']:.6f}")
    print(f"  min_G= {res_mv['min_G']:.6f}")
    print(f"  gain = {res_mv['gain']:.6f}")
    print(f"  M_qp_opt_reported = {res_mv['M_reported_by_qp_opt']:.6f}")
    print(f"  M_self_consistent  = {res_mv['M_self_consistent']:.6f}")
    print()

    print("=" * 72)
    print("[2] Re-solve QP at D2's best (0.1632, 0.6632) with n in {119, 200, 400, 800}")
    print("=" * 72)
    for n in (119, 200, 400, 800):
        res = reqp_at(delta=0.1632, u=0.6632, n=n, n_grid=5001, verbose=False)
        print(
            f"  n={n:4d}: S1={res['S1']:.4f} min_G={res['min_G']:+.4f} "
            f"gain={res['gain']:.5f} M_qp_opt_reported={res['M_reported_by_qp_opt']:.6f} "
            f"M_self_consistent={res['M_self_consistent']:.6f}"
        )
    print()

    print("=" * 72)
    print("[3] Parallel scan near (0.1632, 0.6632) with re-optimised coefficients")
    print("=" * 72)
    best, _all = scan_near_d2(n=119, n_grid=3001, n_jobs=-1, verbose=True)
    print()
    if best is not None:
        print(
            f"  BEST: delta={best['delta']:.4f} u={best['u']:.4f} n={best['n']} "
            f"M={best['M_self_consistent']:.6f}"
        )
        print(
            f"  Over MV (1.27481):      +{best['M_self_consistent'] - 1.27481:.6f}"
        )
        print(
            f"  Over CS (1.28):         +{best['M_self_consistent'] - 1.28:.6f}"
        )
        print(
            f"  Over D2 (1.283788):     +{best['M_self_consistent'] - 1.283788:.6f}"
        )
    print()
    print("DONE.")
