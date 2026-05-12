"""M8: Joint (K, G) alternating optimization for multi-scale arcsine kernel.

The fix-K -> solve-QP -> M_cert pipeline (cf. _K26_full_sweep_reopt.py) is
"alternating" in the sense that for each fixed K we obtain a globally
optimal G via convex QP, but only re-evaluates K on a finite grid.
M_cert is a non-convex function of the joint (K, G), so alternation
that ALSO continuously tunes K parameters (delta_2, lambda_1, ...) for
fixed G coefficients (a_j) can escape local optima the discrete K sweep
gets stuck in.

Pipeline per iteration:
  (Step A) Fix K, solve convex QP for a_j   (gives new G)
  (Step B) Fix a_j, DE-optimize K params for max M_cert
            -- when K changes the weights w_j change, so the QP-derived
               a_j is no longer optimal for the new K; we instead score
               M_cert using S_1 = sum(a_j^2 / w_j(K)) and the new k_1, K_2,
               and ALSO recompute min_G = min_x (B(x) a) (does not depend on K).
            -- This is the key non-convex step where K moves without
               re-solving the QP.
  Convergence: |Delta M_cert| < 1e-5  OR  max iters reached.

We start from the proven optimum (delta_1=0.138, delta_2=0.045, lambda_1=0.85)
with single-shot M_cert = 1.29005, and also try a 3-component variant.

Outputs `_M8_alternating.json`.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution
import cvxpy as cp

DELTA = 0.138       # delta_1 is FIXED at MV value
U = 0.5 + DELTA
N_QP = 119
N_GRID = 5001


# ---------- shared math (copied from _K26_full_sweep_reopt.py) ----------
def K_hat_ms(xi, deltas, lambdas):
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * np.asarray(xi)) ** 2
    return out


def solve_QP(deltas, lambdas, n_grid=N_GRID):
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = K_hat_ms(j / U, deltas, lambdas)
    if w.min() <= 0:
        return None, None, None, "non-positive weight"
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


# ---------- M8 key piece: score K parameters with FROZEN a_j ----------
def precompute_B(n_grid=N_GRID):
    """Cosine design matrix that does NOT depend on K (only on U, N_QP)."""
    xs = np.linspace(0.0, 0.25, n_grid)
    return xs, np.cos(2.0 * math.pi * np.arange(1, N_QP + 1)[None, :] * xs[:, None] / U)


_XS, _B = precompute_B()


def m_cert_with_fixed_G(deltas, lambdas, a_j):
    w = np.array([K_hat_ms(j / U, deltas, lambdas) for j in range(1, N_QP + 1)])
    if w.min() <= 0:
        return None
    S1 = float(np.sum(a_j ** 2 / w))
    min_G = float((_B @ a_j).min())
    if min_G <= 0.0:
        # G no longer dual-feasible -> bound vacuous
        return None
    k1 = float(K_hat_ms(1.0, deltas, lambdas))
    K2 = K_2_quad(deltas, lambdas)
    mc = M_cert(k1, K2, S1, min_G)
    return None if mc is None else (mc, k1, K2, S1, min_G)


# ---------- alternating loop, 2-component ----------
def alternate_2comp(d2_init, l1_init, n_iter=20, de_iter=40, tol=1e-5,
                    bounds=((0.02, 0.13), (0.70, 0.99)), seed=0):
    deltas = [DELTA, d2_init]
    lambdas = [l1_init, 1.0 - l1_init]
    history = []
    a_opt, S1, mG, status = solve_QP(deltas, lambdas)
    if status != "ok":
        return None
    k1 = float(K_hat_ms(1.0, deltas, lambdas))
    K2 = K_2_quad(deltas, lambdas)
    Mc = M_cert(k1, K2, S1, mG)
    history.append({"iter": 0, "step": "init_QP", "delta_2": float(d2_init),
                    "lambda_1": float(l1_init), "k_1": k1, "K_2": K2,
                    "S_1": S1, "min_G": mG, "M_cert": Mc})
    best_Mc = Mc

    for it in range(1, n_iter + 1):
        # ----- Step B: fix a_j, DE on (delta_2, lambda_1) -----
        def negM(x):
            d2, l1 = float(x[0]), float(x[1])
            res = m_cert_with_fixed_G([DELTA, d2], [l1, 1.0 - l1], a_opt)
            if res is None:
                return 10.0
            return -res[0]

        de = differential_evolution(negM, bounds, maxiter=de_iter, tol=1e-9,
                                    seed=seed + it, polish=True,
                                    mutation=(0.5, 1.2), recombination=0.9,
                                    init='sobol', popsize=20)
        d2_new, l1_new = float(de.x[0]), float(de.x[1])
        res = m_cert_with_fixed_G([DELTA, d2_new], [l1_new, 1.0 - l1_new], a_opt)
        if res is None:
            history.append({"iter": it, "step": "DE_K_failed"})
            break
        Mc_K, k1_K, K2_K, S1_K, mG_K = res
        history.append({"iter": it, "step": "DE_K_fixedG",
                        "delta_2": d2_new, "lambda_1": l1_new,
                        "k_1": k1_K, "K_2": K2_K, "S_1": S1_K,
                        "min_G": mG_K, "M_cert": Mc_K})

        # ----- Step A: fix K, re-solve QP -----
        a_new, S1_q, mG_q, status_q = solve_QP([DELTA, d2_new], [l1_new, 1.0 - l1_new])
        if status_q != "ok":
            history.append({"iter": it, "step": "QP_failed", "status": status_q})
            break
        k1_q = float(K_hat_ms(1.0, [DELTA, d2_new], [l1_new, 1.0 - l1_new]))
        K2_q = K_2_quad([DELTA, d2_new], [l1_new, 1.0 - l1_new])
        Mc_q = M_cert(k1_q, K2_q, S1_q, mG_q)
        history.append({"iter": it, "step": "QP_fixedK",
                        "delta_2": d2_new, "lambda_1": l1_new,
                        "k_1": k1_q, "K_2": K2_q, "S_1": S1_q,
                        "min_G": mG_q, "M_cert": Mc_q})

        a_opt = a_new
        new_best = max(Mc_K, Mc_q) if (Mc_K is not None and Mc_q is not None) else (Mc_q or Mc_K)
        delta_M = new_best - best_Mc if new_best is not None else -float('inf')
        print(f"  iter {it:2d}: K-step M={Mc_K:.6f}  QP-step M={Mc_q:.6f}  "
              f"d2={d2_new:.5f} l1={l1_new:.5f}  delta_M={delta_M:+.6e}")
        if new_best > best_Mc:
            best_Mc = new_best
        if abs(delta_M) < tol:
            history.append({"iter": it, "step": "converged"})
            break
    return {"history": history, "best_M_cert": float(best_Mc),
            "final_a_opt_norm": float(np.linalg.norm(a_opt))}


# ---------- alternating loop, 3-component ----------
def alternate_3comp(d2_init, d3_init, l1_init, l2_init, n_iter=15, de_iter=50,
                    tol=1e-5,
                    bounds=((0.03, 0.13), (0.005, 0.06), (0.50, 0.95), (0.03, 0.40)),
                    seed=0):
    deltas = [DELTA, d2_init, d3_init]
    lambdas = [l1_init, l2_init, max(1e-6, 1.0 - l1_init - l2_init)]
    history = []
    a_opt, S1, mG, status = solve_QP(deltas, lambdas)
    if status != "ok":
        return None
    k1 = float(K_hat_ms(1.0, deltas, lambdas))
    K2 = K_2_quad(deltas, lambdas)
    Mc = M_cert(k1, K2, S1, mG)
    history.append({"iter": 0, "step": "init_QP", "deltas": deltas,
                    "lambdas": lambdas, "k_1": k1, "K_2": K2,
                    "S_1": S1, "min_G": mG, "M_cert": Mc})
    best_Mc = Mc

    for it in range(1, n_iter + 1):
        def negM(x):
            d2, d3, l1, l2 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            if l1 + l2 >= 1.0 or l1 + l2 <= 0.0:
                return 10.0
            l3 = 1.0 - l1 - l2
            res = m_cert_with_fixed_G([DELTA, d2, d3], [l1, l2, l3], a_opt)
            if res is None:
                return 10.0
            return -res[0]

        de = differential_evolution(negM, bounds, maxiter=de_iter, tol=1e-9,
                                    seed=seed + it, polish=True,
                                    mutation=(0.5, 1.2), recombination=0.9,
                                    init='sobol', popsize=24)
        d2_new, d3_new, l1_new, l2_new = (float(de.x[0]), float(de.x[1]),
                                          float(de.x[2]), float(de.x[3]))
        l3_new = 1.0 - l1_new - l2_new
        res = m_cert_with_fixed_G([DELTA, d2_new, d3_new],
                                   [l1_new, l2_new, l3_new], a_opt)
        if res is None:
            history.append({"iter": it, "step": "DE_K_failed"})
            break
        Mc_K, k1_K, K2_K, S1_K, mG_K = res
        history.append({"iter": it, "step": "DE_K_fixedG",
                        "deltas": [DELTA, d2_new, d3_new],
                        "lambdas": [l1_new, l2_new, l3_new],
                        "k_1": k1_K, "K_2": K2_K, "S_1": S1_K,
                        "min_G": mG_K, "M_cert": Mc_K})

        a_new, S1_q, mG_q, status_q = solve_QP([DELTA, d2_new, d3_new],
                                                [l1_new, l2_new, l3_new])
        if status_q != "ok":
            history.append({"iter": it, "step": "QP_failed", "status": status_q})
            break
        k1_q = float(K_hat_ms(1.0, [DELTA, d2_new, d3_new], [l1_new, l2_new, l3_new]))
        K2_q = K_2_quad([DELTA, d2_new, d3_new], [l1_new, l2_new, l3_new])
        Mc_q = M_cert(k1_q, K2_q, S1_q, mG_q)
        history.append({"iter": it, "step": "QP_fixedK",
                        "deltas": [DELTA, d2_new, d3_new],
                        "lambdas": [l1_new, l2_new, l3_new],
                        "k_1": k1_q, "K_2": K2_q, "S_1": S1_q,
                        "min_G": mG_q, "M_cert": Mc_q})

        a_opt = a_new
        new_best = max(Mc_K, Mc_q) if (Mc_K is not None and Mc_q is not None) else (Mc_q or Mc_K)
        delta_M = new_best - best_Mc if new_best is not None else -float('inf')
        print(f"  iter {it:2d}: K-step M={Mc_K:.6f}  QP-step M={Mc_q:.6f}  "
              f"d2={d2_new:.5f} d3={d3_new:.5f} l1={l1_new:.4f} l2={l2_new:.4f}  "
              f"delta_M={delta_M:+.6e}")
        if new_best > best_Mc:
            best_Mc = new_best
        if abs(delta_M) < tol:
            history.append({"iter": it, "step": "converged"})
            break
    return {"history": history, "best_M_cert": float(best_Mc)}


def main():
    out = {"baseline_single_shot": 1.2900489969954338,
           "baseline_MV": 1.27481,
           "DELTA_1": DELTA,
           "U": U,
           "N_QP": N_QP}
    print("Single-shot baseline: M_cert = 1.29005 at (d_2, l_1) = (0.045, 0.85)")
    print("=" * 80)

    # ---- 2-component alternating from the known optimum
    print("[2-component] start at (d_2=0.045, l_1=0.85)")
    t0 = time.time()
    r2a = alternate_2comp(d2_init=0.045, l1_init=0.85, n_iter=20, de_iter=40,
                          seed=1)
    out["alt_2comp_from_optimum"] = r2a
    out["alt_2comp_from_optimum"]["walltime_s"] = time.time() - t0
    print(f"  best_M_cert = {r2a['best_M_cert']:.6f}  (delta vs single-shot = "
          f"{r2a['best_M_cert']-1.2900489969954338:+.6e})")

    # ---- 2-component alternating from a perturbed start
    print()
    print("[2-component] start at (d_2=0.07, l_1=0.90)  -- perturbed init")
    t0 = time.time()
    r2b = alternate_2comp(d2_init=0.07, l1_init=0.90, n_iter=20, de_iter=40,
                          seed=2)
    out["alt_2comp_from_perturbed"] = r2b
    out["alt_2comp_from_perturbed"]["walltime_s"] = time.time() - t0
    print(f"  best_M_cert = {r2b['best_M_cert']:.6f}")

    # ---- 2-component alternating from another perturbed start
    print()
    print("[2-component] start at (d_2=0.03, l_1=0.95)  -- another perturbed init")
    t0 = time.time()
    r2c = alternate_2comp(d2_init=0.03, l1_init=0.95, n_iter=20, de_iter=40,
                          seed=3)
    out["alt_2comp_from_perturbed2"] = r2c
    out["alt_2comp_from_perturbed2"]["walltime_s"] = time.time() - t0
    print(f"  best_M_cert = {r2c['best_M_cert']:.6f}")

    # ---- 3-component alternating
    print()
    print("[3-component] start at (d_2=0.06, d_3=0.025, l_1=0.75, l_2=0.15)")
    t0 = time.time()
    r3 = alternate_3comp(d2_init=0.06, d3_init=0.025, l1_init=0.75, l2_init=0.15,
                         n_iter=12, de_iter=50, seed=4)
    out["alt_3comp"] = r3
    out["alt_3comp"]["walltime_s"] = time.time() - t0
    print(f"  best_M_cert = {r3['best_M_cert']:.6f}")

    overall = max(r2a["best_M_cert"], r2b["best_M_cert"], r2c["best_M_cert"],
                  r3["best_M_cert"])
    out["overall_best_M_cert"] = overall
    out["overall_vs_single_shot"] = overall - 1.2900489969954338
    out["overall_vs_MV"] = overall - 1.27481

    print()
    print("=" * 80)
    print(f"OVERALL BEST (alternating) = {overall:.6f}")
    print(f"  vs single-shot 1.29005   = {overall-1.2900489969954338:+.6e}")
    print(f"  vs MV       1.27481      = {overall-1.27481:+.6e}")
    if overall > 1.2900489969954338 + 1e-5:
        print("  >>> alternating IMPROVED over single-shot")
        out["conclusion"] = "alternating_improved"
    else:
        print("  --- alternating did NOT improve; single-shot is near-optimal.")
        out["conclusion"] = "single_shot_near_optimal"

    with open("_M8_alternating.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote _M8_alternating.json")


if __name__ == "__main__":
    main()
