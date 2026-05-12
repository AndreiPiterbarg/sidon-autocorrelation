"""Re-optimize the 119-cosine G coefficients for the multi-scale K, then
recompute M_cert. MV's published G was optimal for arcsine — for a different
K the optimal G is different, potentially smaller S_1.

QP: min sum a_j^2 / w_j(K)  subject to  G(x) = sum a_j cos(2 pi j x / u) >= 1 on [0, 1/4].
w_j(K) = K_hat(j/u).
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq
import cvxpy as cp

DELTA = 0.138
U = 0.5 + DELTA
N_QP = 119


def K_hat_ms(xi, deltas, lambdas):
    """K_hat(xi) = sum_i lambdas[i] * J_0(pi * deltas[i] * xi)^2."""
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * np.asarray(xi)) ** 2
    return out


def solve_qp_for_K(deltas, lambdas, n_grid=5001, verbose=False):
    """Solve MV's QP for a given K_hat."""
    # Weights at j/u
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = K_hat_ms(j / U, deltas, lambdas)
    if w.min() <= 0:
        raise RuntimeError(f"min weight <= 0: {w.min()}")
    # Discretize G >= 1 constraint on [0, 1/4]
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, N_QP))
    for j in range(1, N_QP + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / U)
    a = cp.Variable(N_QP)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    # Try solvers in order
    solver_name = None
    for s_name in ("MOSEK", "CLARABEL", "SCS"):
        try:
            prob.solve(solver=s_name, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                solver_name = s_name
                break
        except Exception:
            continue
    if a.value is None:
        raise RuntimeError(f"QP failed for all solvers; status={prob.status}")
    a_opt = np.asarray(a.value).flatten()
    S1 = float(np.sum(a_opt ** 2 / w))
    min_G_grid = float((B @ a_opt).min())
    if verbose:
        print(f"  QP solver={solver_name}, S_1={S1:.4f}, min_G_grid={min_G_grid:.6f}")
    return a_opt, S1, w, min_G_grid


def K_2_via_quad(deltas, lambdas, xi_split=10.0):
    def f(xi):
        return K_hat_ms(xi, deltas, lambdas) ** 2
    v1, _ = quad(f, 0.0, xi_split, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, xi_split, np.inf, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2)


def mv_M_cert(k_1, K_2, S_1, min_G):
    """MV master inequality solver (use min_G in the gain term)."""
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    # gain a = (4/u) * min_G^2 / S_1
    a = (4.0 / U) * (min_G ** 2) / S_1
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
        return brentq(lambda M: sup_R(M) - target, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return None


def eval_with_reopt(deltas, lambdas, label):
    deltas = np.asarray(deltas)
    lambdas = np.asarray(lambdas)
    a_opt, S1_reopt, w, min_G_grid = solve_qp_for_K(deltas, lambdas, verbose=True)
    k_1 = K_hat_ms(1.0, deltas, lambdas)
    K_2 = K_2_via_quad(deltas, lambdas)
    # Use min_G_grid as a proxy for min_G (rigorous arb pipeline would tighten)
    M_cert = mv_M_cert(k_1, K_2, S1_reopt, min_G_grid)
    print(f"  [{label}]  k_1={k_1:.5f}  K_2={K_2:.5f}  S_1(reopt)={S1_reopt:.4f}  "
          f"min_G={min_G_grid:.5f}  M_cert={M_cert:.6f}")
    # For comparison, the MV-fixed-G S_1
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    a_mv = np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])
    S1_mv = float(np.sum(a_mv ** 2 / w))
    G_mv = np.cos(2 * math.pi * np.arange(1, N_QP + 1)[:, None] / U * np.linspace(0, 0.25, 5001)[None, :])
    # MV G(x) at the same grid (same min as well, MV proves min_G >= 1)
    # Use min_G = 1 for MV's published G (rigorous in MV paper)
    M_mv = mv_M_cert(k_1, K_2, S1_mv, 1.0)
    print(f"  [{label}/MV-G]  k_1={k_1:.5f}  K_2={K_2:.5f}  S_1(MV)={S1_mv:.4f}  "
          f"M_cert={M_mv:.6f}")
    return {
        "k_1": float(k_1), "K_2": float(K_2),
        "S_1_reopt": S1_reopt, "S_1_MV": S1_mv,
        "M_cert_reopt": M_cert, "M_cert_MV_G": M_mv,
        "min_G_reopt": min_G_grid,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Re-optimize G for multi-scale arcsine K")
    print("=" * 70)
    print()
    print("Sanity 1: pure arcsine (should ~match MV's S_1)")
    r = eval_with_reopt([DELTA], [1.0], "pure-arcsine")
    print()
    print("Best from K26: d1=0.138, d2=0.055, lam1=0.9312")
    r = eval_with_reopt([DELTA, 0.055], [0.9312, 0.0688], "best-K26")
    print()
    print("Alternative: d1=0.138, d2=0.0525, lam1=0.935")
    r = eval_with_reopt([DELTA, 0.0525], [0.935, 0.065], "alt-K26")
    print()
    print("Finer: d1=0.138, d2=0.050, lam1=0.940")
    r = eval_with_reopt([DELTA, 0.050], [0.940, 0.060], "fine-K26")
