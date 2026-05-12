"""Re-solve MV's quadratic program in modern tools, possibly with larger n.

MV's QP (from paper p.4):
    minimize    sum_{j=1}^n a_j^2 / |J_0(pi j delta/u)|^2
    subject to  min_{x in [0, 1/4]} sum_{j=1}^n a_j cos(2 pi j x / u)  >= 1

This is semi-infinite (infinitely many linear constraints indexed by x).
We discretise [0, 1/4] into a fine t-grid; the QP becomes a standard
convex QP solvable via scipy/cvxpy.

The gain parameter (MV p. 4):
    a = (4/u) * min_G^2 / S_1
with min_G = active-constraint value (typically exactly 1 after rescaling)
and S_1 = the QP objective. Maximising gain == minimising S_1 subject to
min_G >= 1, which is exactly this QP.

We can then plug a into the master inequality (eq. 10) to get a lower
bound on C_{1a}.

Usage:
    from delsarte_dual.mv_qp_optimize import optimize_qp
    result = optimize_qp(n=119, delta=0.138, u=0.638, n_grid=5000)
    # result['bound_with_z1'] is the lower bound; should reach ~1.2748 at MV's settings.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import linprog
import mpmath as mp
from mpmath import mpf

from .mv_bound import (
    solve_for_M_with_z1, solve_for_M_no_z1, k1_value,
    MV_K2_BOUND_OVER_DELTA,
)


def _J0_weights(n: int, delta: float, u: float) -> np.ndarray:
    """|J_0(pi j delta/u)|^2 for j = 1..n."""
    from scipy.special import j0
    return np.array([j0(math.pi * j * delta / u) ** 2 for j in range(1, n + 1)])


def solve_mv_qp(n: int, delta: float, u: float, n_grid: int = 5001,
                solver: str = "cvxpy", verbose: bool = False) -> dict:
    """Solve MV's semi-infinite QP for given (n, delta, u).

    Returns dict with 'a_opt' (the n coefficients), 'S1' (objective), 'min_G'
    (≈ 1, the active constraint), 'gain' (gain parameter).
    """
    xs = np.linspace(0.0, 0.25, n_grid)
    # Basis: B_{i, j} = cos(2 pi j x_i / u) for i in grid, j in 1..n
    B = np.zeros((n_grid, n))
    for j in range(1, n + 1):
        B[:, j - 1] = np.cos(2 * math.pi * j * xs / u)
    # J_0 weights
    J0_sq = _J0_weights(n, delta, u)   # w_j = |J_0(...)|^2

    # QP:
    #   min  sum a_j^2 / w_j  =  a^T diag(1/w) a
    #   s.t. B @ a >= 1 (for every grid point i)
    # This is strictly convex (diag(1/w) > 0).
    if solver == "cvxpy":
        import cvxpy as cp
        a = cp.Variable(n)
        obj = cp.Minimize(cp.sum(cp.multiply(1.0 / J0_sq, cp.square(a))))
        constraints = [B @ a >= 1]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver="CLARABEL", verbose=verbose)
        if a.value is None:
            raise RuntimeError(f"QP failed: {prob.status}")
        a_opt = np.asarray(a.value).flatten()
        S1 = float(prob.value)
    elif solver == "scipy":
        # Scipy only has LP; we must reformulate QP as LP. Trick: min a^T M a
        # can be reformulated using Lagrangian + Newton but no direct LP.
        raise NotImplementedError("scipy QP solver not set up; use cvxpy.")
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Verify min_G >= 1 on the grid
    G_vals = B @ a_opt
    min_G = float(G_vals.min())

    # Gain parameter (with min_G ≈ 1 on grid)
    gain = (4.0 / u) * min_G ** 2 / S1

    # Compute master-inequality bounds
    mp.mp.dps = 30
    M_no_z1 = solve_for_M_no_z1(gain, delta=delta, u=u)
    M_with_z1 = solve_for_M_with_z1(gain, delta=delta, u=u)

    return {
        "n": n, "delta": delta, "u": u, "n_grid": n_grid,
        "a_opt": a_opt, "S1": S1, "min_G": min_G, "gain": gain,
        "M_no_z1": float(M_no_z1),
        "M_with_z1": float(M_with_z1),
    }


def scan_delta_u(delta_values, u_values, n=119, n_grid=5001, verbose=False):
    """Grid-scan (delta, u) and report the best bound.

    MV used (delta, u) = (0.138, 0.638). Vary around.
    """
    best = {"M_with_z1": -np.inf}
    all_results = []
    for delta in delta_values:
        for u in u_values:
            try:
                res = solve_mv_qp(n=n, delta=delta, u=u, n_grid=n_grid)
                if verbose:
                    print(f"  delta={delta:.4f} u={u:.4f}: gain={res['gain']:.5f} "
                          f"M_z1={res['M_with_z1']:.5f} min_G={res['min_G']:.4f}")
                all_results.append(res)
                if res["M_with_z1"] > best["M_with_z1"]:
                    best = res
            except Exception as e:
                if verbose:
                    print(f"  delta={delta:.4f} u={u:.4f}: FAILED ({e})")
    return best, all_results


def scan_n(n_values, delta=0.138, u=0.638, n_grid=5001, verbose=False):
    """Scan over number of coefficients n at fixed (delta, u)."""
    results = []
    for n in n_values:
        try:
            res = solve_mv_qp(n=n, delta=delta, u=u, n_grid=n_grid)
            if verbose:
                print(f"  n={n:4d}: gain={res['gain']:.6f} M_z1={res['M_with_z1']:.6f}")
            results.append(res)
        except Exception as e:
            if verbose:
                print(f"  n={n}: FAILED ({e})")
    return results


if __name__ == "__main__":
    import sys
    # Step 1: reproduce MV's setting.
    print("=" * 70)
    print("Reproduce MV: n=119, delta=0.138, u=0.638")
    print("=" * 70)
    res = solve_mv_qp(n=119, delta=0.138, u=0.638, n_grid=5001)
    print(f"  gain = {res['gain']:.6f}  (MV: 0.07137)")
    print(f"  M_no_z1 = {res['M_no_z1']:.6f}  (MV: 1.2743)")
    print(f"  M_with_z1 = {res['M_with_z1']:.6f}  (MV: 1.27481)")
    print()

    # Step 2: scan n
    print("=" * 70)
    print("Scan n at (delta, u) = (0.138, 0.638)")
    print("=" * 70)
    scan_n([50, 100, 119, 200, 400, 800], n_grid=5001, verbose=True)
    print()

    # Step 3: scan delta, u
    print("=" * 70)
    print("Scan (delta, u)")
    print("=" * 70)
    ds = np.linspace(0.10, 0.17, 8)
    us = np.linspace(0.60, 0.68, 5)
    best, _ = scan_delta_u(ds, us, n=119, n_grid=5001, verbose=True)
    print()
    print(f"BEST: delta={best['delta']:.4f} u={best['u']:.4f} M_z1={best['M_with_z1']:.6f}")
