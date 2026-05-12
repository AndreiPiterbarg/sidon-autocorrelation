"""Full grid sweep with G re-optimization at each (delta_2, lambda_1) point.
Find the joint maximum (delta_2*, lambda_1*) and the resulting numerical M_cert.
"""
from __future__ import annotations

import json
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
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * np.asarray(xi)) ** 2
    return out


def solve_QP(deltas, lambdas, n_grid=5001):
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = K_hat_ms(j / U, deltas, lambdas)
    if w.min() <= 0:
        return None, None, None, "non-positive weight"
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, N_QP))
    for j in range(1, N_QP + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / U)
    a = cp.Variable(N_QP)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="MOSEK", verbose=False)
        if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            prob.solve(solver="CLARABEL", verbose=False)
        if a.value is None:
            return None, None, None, "QP failed"
    except Exception as e:
        return None, None, None, f"QP exception: {e}"
    a_opt = np.asarray(a.value).flatten()
    S1 = float(np.sum(a_opt ** 2 / w))
    min_G = float((B @ a_opt).min())
    return a_opt, S1, min_G, "ok"


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
    a_opt, S1, mG, status = solve_QP(deltas, lambdas)
    if status != "ok":
        return {"status": status}
    k1 = K_hat_ms(1.0, deltas, lambdas)
    K2 = K_2_quad(deltas, lambdas)
    Mc = M_cert(k1, K2, S1, mG)
    return {"k_1": float(k1), "K_2": float(K2), "S_1": S1, "min_G": mG,
            "M_cert": Mc, "status": status}


def main():
    print("Sweep with G re-optimization. d_1 = 0.138 fixed.")
    print("=" * 80)
    print(f"{'d_2':>8} {'lam_1':>7} {'k_1':>8} {'K_2':>7} {'S_1':>9} {'min_G':>7} {'M_cert':>9}")
    print("-" * 80)

    delta_2_list = [0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.10, 0.115, 0.13]
    lambda_1_list = [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]

    all_results = []
    best = {"M_cert": -np.inf, "params": None}
    for d2 in delta_2_list:
        for l1 in lambda_1_list:
            r = eval_at([DELTA, d2], [l1, 1.0 - l1])
            if r.get("M_cert") is not None:
                print(f"{d2:>8.4f} {l1:>7.3f} {r['k_1']:>8.5f} {r['K_2']:>7.4f} "
                      f"{r['S_1']:>9.4f} {r['min_G']:>7.5f} {r['M_cert']:>9.5f}")
                all_results.append({"delta_2": d2, "lambda_1": l1, **r})
                if r["M_cert"] > best["M_cert"]:
                    best = {"M_cert": float(r["M_cert"]),
                            "params": {"delta_2": d2, "lambda_1": l1,
                                       "k_1": r["k_1"], "K_2": r["K_2"],
                                       "S_1": r["S_1"], "min_G": r["min_G"]}}
            else:
                print(f"{d2:>8.4f} {l1:>7.3f}  --- SKIPPED ({r.get('status')}) ---")

    print()
    print(f"BEST: M_cert={best['M_cert']:.5f} at {best['params']}")
    print(f"  vs MV's 1.27481 baseline: {best['M_cert'] - 1.27481:+.5f}")

    # Refined sweep around the best
    print()
    print("--- Refined sweep around best ---")
    print(f"{'d_2':>8} {'lam_1':>7} {'k_1':>8} {'K_2':>7} {'S_1':>9} {'min_G':>7} {'M_cert':>9}")
    print("-" * 80)
    d2_0 = best["params"]["delta_2"]
    l1_0 = best["params"]["lambda_1"]
    refined_d2 = np.linspace(max(0.02, d2_0 - 0.02), min(DELTA - 0.001, d2_0 + 0.02), 11)
    refined_l1 = np.linspace(max(0.80, l1_0 - 0.05), min(0.99, l1_0 + 0.05), 11)
    for d2 in refined_d2:
        for l1 in refined_l1:
            r = eval_at([DELTA, float(d2)], [float(l1), 1.0 - float(l1)])
            if r.get("M_cert") is not None:
                Mc = r["M_cert"]
                marker = " *" if Mc > best["M_cert"] else ""
                print(f"{d2:>8.5f} {l1:>7.4f} {r['k_1']:>8.5f} {r['K_2']:>7.4f} "
                      f"{r['S_1']:>9.4f} {r['min_G']:>7.5f} {r['M_cert']:>9.5f}{marker}")
                all_results.append({"delta_2": float(d2), "lambda_1": float(l1), **r})
                if Mc > best["M_cert"]:
                    best = {"M_cert": float(Mc),
                            "params": {"delta_2": float(d2), "lambda_1": float(l1),
                                       "k_1": r["k_1"], "K_2": r["K_2"],
                                       "S_1": r["S_1"], "min_G": r["min_G"]}}

    print()
    print(f"FINAL BEST: M_cert={best['M_cert']:.5f}")
    print(f"  Params: {best['params']}")
    print(f"  vs MV's 1.27481: {best['M_cert'] - 1.27481:+.5f}")

    out = {"baseline_M_cert_MV": 1.27481,
           "best_M_cert": best["M_cert"],
           "best_params": best["params"],
           "improvement": best["M_cert"] - 1.27481,
           "all_results": all_results}
    with open("_K26_full_sweep_reopt_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote _K26_full_sweep_reopt_result.json")


if __name__ == "__main__":
    main()
