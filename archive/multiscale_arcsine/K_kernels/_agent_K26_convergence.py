"""Convergence check for Agent K26: verify the K_2 integral is converged
for the best two-component combo, and recompute M_cert with high-precision grid.

Best from coarse+refined sweep: delta_2 = 0.055, lambda_1 = 0.9312.
Baseline (pure arcsine): delta_2 irrelevant, lambda_1 = 1.0.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
from scipy.special import j0

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import DELTA, MV_COEFFS, N_QP, U, mv_master_M_cert


def K_hat_multiscale(xi, deltas, lambdas):
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


def evaluate(deltas, lambdas, label, n_xi=40001, xi_max=600.0, verbose=True):
    xi = np.linspace(0.0, xi_max, n_xi)
    dxi = xi[1] - xi[0]
    Kh = K_hat_multiscale(xi, deltas, lambdas)
    K_2 = 2.0 * np.trapezoid(Kh ** 2, dx=dxi)
    k_1 = float(K_hat_multiscale(np.array([1.0]), deltas, lambdas)[0])
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat_multiscale(qp_xi, deltas, lambdas)
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))
    M_cert = mv_master_M_cert(k_1, float(K_2), S_1)
    if verbose:
        print(f"[{label}] N_XI={n_xi}, XI_MAX={xi_max}: K_2={K_2:.6f}, "
              f"S_1={S_1:.4f}, k_1={k_1:.6f}, M_cert={M_cert:.6f}")
    return {"label": label, "K_2": float(K_2), "S_1": S_1, "k_1": k_1,
            "M_cert": M_cert, "n_xi": n_xi, "xi_max": xi_max}


def main():
    print("=" * 78)
    print("K26 Convergence check")
    print("=" * 78)

    # Best two-component combo
    BEST = ([DELTA, 0.055], [0.9312, 1.0 - 0.9312], "best-d2_0.055-l1_0.9312")
    BASELINE = ([DELTA, 0.055], [1.0, 0.0], "baseline-pure-DELTA")

    # Grid resolution sweep.
    grids = [
        (10001, 200.0),
        (20001, 400.0),
        (40001, 600.0),   # default used in main sweep
        (80001, 1200.0),
        (160001, 2400.0),
        (320001, 4800.0),
    ]

    print("\n--- Baseline (pure DELTA arcsine) ---")
    baseline_results = []
    for n_xi, xi_max in grids:
        baseline_results.append(evaluate(*BASELINE, n_xi=n_xi, xi_max=xi_max))

    print("\n--- Multi-scale best (delta_2=0.055, lambda_1=0.9312) ---")
    best_results = []
    for n_xi, xi_max in grids:
        best_results.append(evaluate(*BEST, n_xi=n_xi, xi_max=xi_max))

    # Also a finer refined search near the optimum on the converged grid.
    print("\n--- Fine refined search on converged grid (N_XI=320001, XI_MAX=4800) ---")
    best_M = -np.inf
    best_pt = None
    d2_grid = np.linspace(0.04, 0.075, 15)
    l1_grid = np.linspace(0.88, 0.96, 17)
    fine_results = []
    for d2 in d2_grid:
        for l1 in l1_grid:
            label = f"d2={d2:.4f},l1={l1:.4f}"
            r = evaluate([DELTA, float(d2)], [float(l1), 1.0 - float(l1)],
                         label, n_xi=160001, xi_max=2400.0, verbose=False)
            fine_results.append({"delta_2": float(d2), "lambda_1": float(l1),
                                 **{k: v for k, v in r.items()
                                    if k not in ("label", "n_xi", "xi_max")}})
            if r["M_cert"] is not None and r["M_cert"] > best_M:
                best_M = r["M_cert"]
                best_pt = (float(d2), float(l1))
                print(f"  new best: {label}: M_cert={r['M_cert']:.6f}")

    print(f"\nFine refined best on converged grid: M={best_M:.6f} at "
          f"(delta_2={best_pt[0]}, lambda_1={best_pt[1]})")

    out = {
        "baseline_grid_results": baseline_results,
        "best_grid_results": best_results,
        "fine_search_results": fine_results,
        "fine_best_M_cert": best_M,
        "fine_best_delta_2": best_pt[0],
        "fine_best_lambda_1": best_pt[1],
    }
    outpath = os.path.join(REPO, "_agent_K26_convergence.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outpath}")


if __name__ == "__main__":
    main()
