"""Push the multi-scale arcsine search deeper:
1. Smaller delta_2 (down to 0.005)
2. 3-component combinations
3. Free delta_1 (not fixed at 0.138)
"""
from __future__ import annotations

import json
import math
import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution
import cvxpy as cp

DELTA_MAX = 0.138
U = 0.5 + DELTA_MAX
N_QP = 119


def K_hat_ms(xi, deltas, lambdas):
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * np.asarray(xi)) ** 2
    return out


def solve_QP(deltas, lambdas, n_grid=5001):
    w = np.array([K_hat_ms(j / U, deltas, lambdas) for j in range(1, N_QP + 1)])
    if w.min() <= 0:
        return None, None, None
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.cos(2.0 * math.pi * np.arange(1, N_QP + 1)[None, :] * xs[:, None] / U)
    a = cp.Variable(N_QP)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="MOSEK", verbose=False)
        if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            prob.solve(solver="CLARABEL", verbose=False)
        if a.value is None:
            return None, None, None
    except Exception:
        return None, None, None
    a_opt = np.asarray(a.value).flatten()
    S1 = float(np.sum(a_opt ** 2 / w))
    min_G = float((B @ a_opt).min())
    return a_opt, S1, min_G


def K_2_quad(deltas, lambdas):
    def f(xi):
        return K_hat_ms(xi, deltas, lambdas) ** 2
    v1, _ = quad(f, 0.0, 10.0, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, 10.0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2)


def M_cert(k_1, K_2, S_1, min_G):
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    a_gain = (4.0 / U) * (min_G ** 2) / S_1
    target = 2.0 / U + a_gain
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


def eval_at(deltas, lambdas):
    res = solve_QP(deltas, lambdas)
    if res[0] is None:
        return None
    a_opt, S1, mG = res
    k1 = K_hat_ms(1.0, deltas, lambdas)
    K2 = K_2_quad(deltas, lambdas)
    return M_cert(k1, K2, S1, mG), (float(k1), float(K2), float(S1), float(mG))


def neg_M_for_DE(x, n_components):
    # x = [delta_1, delta_2, ..., lambda_1, lambda_2, ..., lambda_{n-1}]
    # lambda_n = 1 - sum(lambdas)
    deltas = np.array(x[:n_components])
    lambdas_short = np.array(x[n_components:])
    lambda_last = 1.0 - lambdas_short.sum()
    if lambda_last < 0.01 or lambdas_short.min() < 0.01:
        return 0.0
    if (deltas <= 0).any() or (deltas > DELTA_MAX + 1e-9).any():
        return 0.0
    lambdas = np.concatenate([lambdas_short, [lambda_last]])
    out = eval_at(list(deltas), list(lambdas))
    if out is None or out[0] is None:
        return 0.0
    return -float(out[0])


def main():
    print("=" * 80)
    print("Deeper multi-scale arcsine search")
    print("=" * 80)

    # Phase 1: 2-component with smaller delta_2
    print("\n--- Phase 1: 2-component, smaller delta_2 ---")
    print(f"{'d_2':>8} {'lam_1':>7} {'k_1':>8} {'K_2':>7} {'S_1':>8} {'M_cert':>9}")
    best = (-np.inf, None)
    for d2 in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
        for l1 in [0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.90, 0.93, 0.95, 0.97]:
            res = eval_at([DELTA_MAX, d2], [l1, 1.0 - l1])
            if res is None or res[0] is None:
                continue
            M, (k1, K2, S1, mG) = res
            print(f"{d2:>8.4f} {l1:>7.3f} {k1:>8.5f} {K2:>7.4f} {S1:>8.2f} {M:>9.5f}")
            if M > best[0]:
                best = (M, {"d2": d2, "l1": l1, "k_1": k1, "K_2": K2, "S_1": S1, "min_G": mG})

    print(f"\nPhase 1 best: M_cert={best[0]:.5f} at {best[1]}")

    # Phase 2: 3-component combination
    print("\n--- Phase 2: 3-component (DELTA, d2, d3) ---")
    best3 = (best[0], best[1])  # carry over
    if best[1] is not None:
        d2_star = best[1]["d2"]
        l1_star = best[1]["l1"]
        print(f"Starting from 2-comp best (d2={d2_star}, l1={l1_star})")
        for d3 in [0.005, 0.008, 0.012, 0.015, 0.018, 0.022, 0.028, 0.035, 0.06, 0.08, 0.10]:
            if abs(d3 - d2_star) < 0.005:
                continue
            for l3 in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
                l1 = l1_star - l3 * 0.5  # split between l_2 and l_3 roughly
                l2 = 1.0 - l1 - l3
                if l1 < 0.5 or l2 < 0.02 or l3 < 0.02:
                    continue
                res = eval_at([DELTA_MAX, d2_star, d3], [l1, l2, l3])
                if res is None or res[0] is None:
                    continue
                M, (k1, K2, S1, mG) = res
                if M > best3[0]:
                    print(f"  d3={d3:.4f} l3={l3:.3f} l2={l2:.3f} l1={l1:.3f} -> M={M:.5f} *")
                    best3 = (M, {"d1": DELTA_MAX, "d2": d2_star, "d3": d3,
                                 "l1": l1, "l2": l2, "l3": l3,
                                 "k_1": k1, "K_2": K2, "S_1": S1, "min_G": mG})

    print(f"\nPhase 2 best: M_cert={best3[0]:.5f}")
    if best3[1]:
        print(f"  Params: {best3[1]}")

    # Phase 3: DE over (d1, d2, lambda_1) for 2-component
    print("\n--- Phase 3: DE search over (d1, d2, lambda_1) ---")
    bounds = [(0.10, DELTA_MAX), (0.005, 0.10), (0.50, 0.99)]
    result = differential_evolution(
        lambda x: neg_M_for_DE(x, n_components=2),
        bounds=bounds,
        maxiter=50, popsize=20, seed=42,
        tol=1e-8, workers=1, polish=True,
    )
    M_de = -float(result.fun)
    print(f"DE result: M_cert={M_de:.5f} at x={result.x}")
    print(f"  d1={result.x[0]:.5f}, d2={result.x[1]:.5f}, lambda_1={result.x[2]:.5f}")

    # Save
    out = {
        "phase1_best": {"M_cert": float(best[0]), "params": best[1]},
        "phase2_best": {"M_cert": float(best3[0]), "params": best3[1]},
        "phase3_de_best": {"M_cert": M_de,
                          "d1": float(result.x[0]),
                          "d2": float(result.x[1]),
                          "lambda_1": float(result.x[2])},
    }
    with open("_K26_push_deeper_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote _K26_push_deeper_result.json")

    final_best = max(best[0], best3[0], M_de)
    print(f"\n=== FINAL DEEPEST: M_cert={final_best:.5f}  (MV: 1.27481, gap +{final_best-1.27481:.5f}) ===")


if __name__ == "__main__":
    main()
