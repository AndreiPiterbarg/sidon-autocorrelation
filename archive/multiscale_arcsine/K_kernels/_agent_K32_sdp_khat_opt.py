"""Agent K32: Convex optimization over K_hat values on a fine xi-grid.

Goal: explore the MV (k_1, K_2, S_1) frontier OUTSIDE closed-form kernels.

Approach
--------
Decision variables: c_k = K_hat(k * Dxi) for k = 0..L, with Dxi = 1/(2*DELTA)
(Shannon-Nyquist sampling, since K is supported in [-DELTA, DELTA] so K_hat is
entire of exponential type 2*pi*DELTA and uniquely determined by its values at
the lattice {k/(2*DELTA)}).

The Shannon series reconstructs K_hat at ANY xi:
    K_hat(xi) = sum_k c_k * sinc(2*DELTA*xi - k)
where sinc(x) = sin(pi x) / (pi x), sinc(0) = 1.

Constraints (convex in c_k):
    (i)   c_0 = 1                                   (K_hat(0) = int K = 1 normalization)
    (ii)  K_hat(xi_grid) >= 0  for a fine xi grid   (Bochner)
    (iii) sum_k c_k^2 / (2*DELTA) <= K_2_target     (Parseval: int K_hat^2 dxi)
    (iv)  K_hat(1) = k_1_target                     (fix k_1)

Objective: minimize S_1 = sum_{j=1..119} a_j^2 / K_hat(j/U)
This is CONVEX since each term is inverse of a non-negative linear functional
(use cp.inv_pos).

Sweep over (k_1_target, K_2_target), solve the convex program, compute M_cert.

Mathematical sanity: a feasible solution with K_2 < 4.254 AND k_1 = 0.909 AND
S_1 < 87.88 would *improve* MV's M_cert = 1.269 -- so checking if any LP/QP cell
yields M_cert > arcsine's 1.270 = goal.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import cvxpy as cp

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import mv_master_M_cert, DELTA, U, N_QP
from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq


# Shannon parameters
DXI = 1.0 / (2.0 * DELTA)             # Nyquist spacing in xi
L_SHANNON = 150                        # number of Shannon coeffs (one-sided after even symmetry)
                                       # c_k for k=0..L_SHANNON; we ENFORCE even symmetry c_{-k}=c_k

# Bochner check grid (positivity of K_hat on R)
N_BOCHNER = 4000
XI_BOCHNER_MAX = 150.0 / DELTA


def _sinc(x: np.ndarray) -> np.ndarray:
    """Unnormalised sinc: sin(pi x) / (pi x), with sinc(0)=1."""
    out = np.ones_like(x)
    mask = np.abs(x) > 1e-12
    px = np.pi * x[mask]
    out[mask] = np.sin(px) / px
    return out


def shannon_design_matrix(xi_vals: np.ndarray, L: int = L_SHANNON,
                          dxi: float = DXI) -> np.ndarray:
    """Return matrix A_{n,k} so that K_hat(xi_n) = sum_k A_{n,k} c_k.

    K_hat is EVEN (real), so we represent c_k for k=0..L only and use the
    identity:
        K_hat(xi) = c_0 sinc(2*DELTA*xi)
                 + sum_{k=1}^{L} c_k * [sinc(2*DELTA*xi - k) + sinc(2*DELTA*xi + k)]
    """
    xi_vals = np.asarray(xi_vals, dtype=float)
    M = xi_vals.size
    A = np.zeros((M, L + 1), dtype=float)
    arg0 = 2.0 * DELTA * xi_vals
    A[:, 0] = _sinc(arg0)
    for k in range(1, L + 1):
        A[:, k] = _sinc(arg0 - k) + _sinc(arg0 + k)
    return A


def mv_coeffs_array() -> np.ndarray:
    return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])


def solve_one_cell(k_1_target: float, K_2_target: float,
                   L: int = L_SHANNON,
                   verbose: bool = False) -> dict:
    """Solve the convex QP for a fixed (k_1, K_2) cell.

    Returns dict with status, c (Shannon coeffs), S_1, etc.
    """
    a = mv_coeffs_array()                  # MV coeffs a_1..a_119

    # Design matrices
    # Bochner: xi grid on (0, XI_BOCHNER_MAX]; include xi=0 separately as c_0=1.
    # Use a dense Cheb-like grid: linear on first interval + extra integer pts.
    xi_bochner_lin = np.linspace(0.0, XI_BOCHNER_MAX, N_BOCHNER + 1)[1:]
    xi_bochner_ints = np.arange(1, 201)  # all integers from 1..200 (matches helper's bochner_check_max)
    xi_bochner = np.unique(np.concatenate([xi_bochner_lin, xi_bochner_ints]))
    A_bochner = shannon_design_matrix(xi_bochner, L=L)

    # k_1 = K_hat(1)
    a_k1 = shannon_design_matrix(np.array([1.0]), L=L)[0, :]

    # MV S_1: K_hat at j/U for j=1..N_QP
    xi_qp = np.arange(1, N_QP + 1) / U
    A_qp = shannon_design_matrix(xi_qp, L=L)

    # Variables
    c = cp.Variable(L + 1)

    # Use ZERO safety margin on the constraint grid -- LP boundary will be
    # numerically slightly violated, but we VALIDATE the solution off-grid.
    constraints = [
        c[0] == 1.0,                       # K_hat(0) = 1
        A_bochner @ c >= 0.0,              # Bochner on grid
        a_k1 @ c == k_1_target,            # k_1 fixed
    ]
    # K_2 budget via Parseval: K_2 = int K_hat^2 dxi
    # Sampling theorem: int K_hat^2 dxi = (1/(2*DELTA)) * (c_0^2 + 2 sum_{k>=1} c_k^2)
    # i.e. = DXI * (c_0^2 + 2 sum_{k>=1} c_k^2)
    # We model 2 sum c_k^2 + c_0^2 via cp.sum_squares with weights.
    w_K2 = np.ones(L + 1) * 2.0
    w_K2[0] = 1.0
    K2_expr = DXI * cp.sum(cp.multiply(w_K2, cp.square(c)))
    constraints.append(K2_expr <= K_2_target)

    # Objective: minimize S_1 = sum_j a_j^2 / K_hat(j/U)
    # We need K_hat(j/U) > 0 for cp.inv_pos. Introduce slack t_j = K_hat(j/U).
    t = cp.Variable(N_QP, nonneg=True)
    constraints.append(A_qp @ c == t)
    constraints.append(t >= 1e-6)          # avoid degeneracy
    objective = cp.Minimize(cp.sum(cp.multiply(a ** 2, cp.inv_pos(t))))

    prob = cp.Problem(objective, constraints)

    t_start = time.time()
    status = None
    val = None
    solver_used = None
    try:
        val = prob.solve(solver=cp.CLARABEL, verbose=False)
        status = prob.status
        solver_used = "CLARABEL"
    except Exception as e:
        if verbose:
            print(f"  CLARABEL failed: {e}")
        try:
            val = prob.solve(solver=cp.SCS, verbose=False)
            status = prob.status
            solver_used = "SCS"
        except Exception as e2:
            return {"status": "solver_error", "error": str(e2),
                    "k_1_target": k_1_target, "K_2_target": K_2_target}

    elapsed = time.time() - t_start
    out = {
        "k_1_target": float(k_1_target),
        "K_2_target": float(K_2_target),
        "status": status,
        "solver": solver_used,
        "solve_time_s": elapsed,
    }

    if status not in ("optimal", "optimal_inaccurate"):
        out["S_1_LP"] = None
        out["M_cert"] = None
        return out

    c_val = np.asarray(c.value, dtype=float)
    t_val = np.asarray(t.value, dtype=float)
    S_1 = float(np.sum((a ** 2) / t_val))

    # Tight upper bound on K_2 actually attained
    K_2_actual = float(DXI * (c_val[0] ** 2 + 2.0 * np.sum(c_val[1:] ** 2)))
    k_1_actual = float(a_k1 @ c_val)

    # Check Bochner numerics: minimum over a denser xi grid (DISJOINT from
    # constraint grid to detect cheating).
    xi_dense = np.linspace(1e-3, XI_BOCHNER_MAX, 30000)
    A_dense = shannon_design_matrix(xi_dense, L=L)
    boch_dense_min = float(np.min(A_dense @ c_val))

    # M_cert is only legitimate if Bochner holds on R.  Tag accordingly.
    # We accept boch >= -2e-5 as numerically valid (5e-4 of the typical
    # K_hat magnitude after solver-tolerance jitter).
    bochner_valid = (boch_dense_min >= -2e-5)
    M_cert = mv_master_M_cert(k_1_actual, K_2_actual, S_1)

    out.update({
        "k_1_actual": k_1_actual,
        "K_2_actual": K_2_actual,
        "S_1": S_1,
        "bochner_dense_min": boch_dense_min,
        "bochner_valid": bool(bochner_valid),
        "M_cert": M_cert,
        "M_cert_valid": (M_cert if bochner_valid else None),
        "beats_127": (bochner_valid and M_cert is not None and M_cert > 1.27),
        "beats_MV": (bochner_valid and M_cert is not None and M_cert > 1.27481),
        "beats_arcsine": (bochner_valid and M_cert is not None and M_cert > 1.26990),
    })
    if verbose:
        print(f"  (k1={k_1_target:.3f}, K2<={K_2_target:.3f}): "
              f"S1={S_1:.3f}, K2_act={K_2_actual:.3f}, M_cert="
              f"{M_cert if M_cert is None else f'{M_cert:.5f}'}")
    return out


def main():
    print(f"Agent K32: K_hat convex optimization")
    print(f"  DELTA={DELTA}, U={U}, Shannon coeffs L={L_SHANNON}")
    print(f"  Reference arcsine: k_1=0.9093, K_2=4.2539, S_1=87.88, M_cert=1.26990")
    print()

    k_1_targets = [0.85, 0.88, 0.909, 0.91, 0.93, 0.95, 0.97]
    K_2_targets = [3.5, 3.8, 4.0, 4.2, 4.254, 4.5, 5.0]

    results = []
    best = {"M_cert": -np.inf}
    best_valid = {"M_cert": -np.inf}

    for k1t in k_1_targets:
        for K2t in K_2_targets:
            res = solve_one_cell(k1t, K2t, verbose=True)
            results.append(res)
            if res.get("M_cert") is not None and res["M_cert"] > best["M_cert"]:
                best = res
            if (res.get("bochner_valid") and res.get("M_cert") is not None
                    and res["M_cert"] > best_valid["M_cert"]):
                best_valid = res

    out_path = os.path.join(REPO, "_agent_K32_sdp_khat_opt_result.json")
    summary = {
        "delta": DELTA,
        "u": U,
        "shannon_L": L_SHANNON,
        "n_bochner_pts": N_BOCHNER,
        "xi_bochner_max_over_delta": 100.0,
        "reference_arcsine": {
            "k_1": 0.9093, "K_2": 4.2539, "S_1": 87.88, "M_cert": 1.26990,
        },
        "results": results,
        "best_any": best,
        "best_bochner_valid": best_valid,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"\nBest cell (Bochner-valid only):")
    print(json.dumps(best_valid, indent=2, default=str))
    print(f"\nBest cell (any, may violate Bochner off-grid):")
    print(json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
