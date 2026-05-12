"""Agent M12: arcsine x B-spline (cardinal) mix sweep.

K(x) = lam * K_arcsine(x; delta_1) + (1 - lam) * K_bspline_n(x; delta_2)

Bochner: K_arcsine_hat = j0(pi d xi)^2 >= 0
         K_bspline_n_hat = sinc(pi d xi / n)^(2n) >= 0
         (each non-negative individually => sum non-negative)

Pipeline reused from _K26_full_sweep_reopt:
  - U = 0.5 + DELTA, N_QP = 119, DELTA = 0.138
  - solve_QP using w_j = K_hat(j/U)
  - K_2 = integral of K_hat(xi)^2 dxi over R
  - M_cert via brentq on sup_R

Phase 1: grid sweep over (n, delta_2, lam)
Phase 2: refined DE around best
Phase 3: 3-component arcsine + 2 B-splines

Baselines:
  - MV (single arcsine):              1.27481
  - 2-scale arcsine (same-family):    1.29005
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

from _K26_full_sweep_reopt import (
    DELTA as DELTA1_DEFAULT,
    U,
    N_QP,
    M_cert,
)

DELTA1 = 0.138  # arcsine scale fixed at MV's optimal

# -----------------------------------------------------------------------------
# Custom K_hat evaluators
# -----------------------------------------------------------------------------

def K_hat_arc(xi, delta):
    return j0(np.pi * delta * np.asarray(xi)) ** 2


def K_hat_bspline(xi, delta, n):
    """K_hat for cardinal B-spline of order n (auto-convolution of indicator).

    K_hat(xi) = sinc(pi delta xi / n)^(2n)  where sinc(z) = sin(z)/z, sinc(0)=1
    """
    xi = np.asarray(xi, dtype=float)
    arg = np.pi * delta * xi / float(n)
    s = np.where(np.abs(arg) < 1e-12, 1.0, np.sin(arg) / np.where(np.abs(arg) < 1e-12, 1.0, arg))
    return s ** (2 * n)


def K_hat_mix_2(xi, delta1, delta2, n, lam):
    """2-component: arcsine + B-spline."""
    return lam * K_hat_arc(xi, delta1) + (1.0 - lam) * K_hat_bspline(xi, delta2, n)


def K_hat_mix_3(xi, delta1, delta2a, na, delta2b, nb, lam, mu):
    """3-component: arcsine (weight lam) + B-spline-na (weight mu*(1-lam))
                  + B-spline-nb (weight (1-mu)*(1-lam))."""
    arc = lam * K_hat_arc(xi, delta1)
    b1 = mu * (1.0 - lam) * K_hat_bspline(xi, delta2a, na)
    b2 = (1.0 - mu) * (1.0 - lam) * K_hat_bspline(xi, delta2b, nb)
    return arc + b1 + b2


# -----------------------------------------------------------------------------
# QP / K_2 / eval
# -----------------------------------------------------------------------------

def _solve_QP_from_weights(w, n_grid=5001):
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


def solve_QP_2(delta1, delta2, n, lam, n_grid=5001):
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = K_hat_mix_2(j / U, delta1, delta2, n, lam)
    return _solve_QP_from_weights(w, n_grid=n_grid)


def K_2_quad_2(delta1, delta2, n, lam):
    def f(xi):
        return K_hat_mix_2(xi, delta1, delta2, n, lam) ** 2
    v1, _ = quad(f, 0.0, 10.0, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, 10.0, 200.0, limit=400, epsabs=1e-14, epsrel=1e-12)
    v3, _ = quad(f, 200.0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2 + v3)


def eval_at_2(delta2, n, lam, delta1=DELTA1):
    a_opt, S1, mG, status = solve_QP_2(delta1, delta2, n, lam)
    if status != "ok":
        return {"status": status}
    k1 = float(K_hat_mix_2(1.0, delta1, delta2, n, lam))
    K2 = float(K_2_quad_2(delta1, delta2, n, lam))
    Mc = M_cert(k1, K2, S1, mG)
    return {"k_1": k1, "K_2": K2, "S_1": S1, "min_G": mG,
            "M_cert": Mc, "status": status}


def solve_QP_3(delta1, delta2a, na, delta2b, nb, lam, mu, n_grid=5001):
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = K_hat_mix_3(j / U, delta1, delta2a, na, delta2b, nb, lam, mu)
    return _solve_QP_from_weights(w, n_grid=n_grid)


def K_2_quad_3(delta1, delta2a, na, delta2b, nb, lam, mu):
    def f(xi):
        return K_hat_mix_3(xi, delta1, delta2a, na, delta2b, nb, lam, mu) ** 2
    v1, _ = quad(f, 0.0, 10.0, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, 10.0, 200.0, limit=400, epsabs=1e-14, epsrel=1e-12)
    v3, _ = quad(f, 200.0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2 + v3)


def eval_at_3(delta2a, na, delta2b, nb, lam, mu, delta1=DELTA1):
    a_opt, S1, mG, status = solve_QP_3(delta1, delta2a, na, delta2b, nb, lam, mu)
    if status != "ok":
        return {"status": status}
    k1 = float(K_hat_mix_3(1.0, delta1, delta2a, na, delta2b, nb, lam, mu))
    K2 = float(K_2_quad_3(delta1, delta2a, na, delta2b, nb, lam, mu))
    Mc = M_cert(k1, K2, S1, mG)
    return {"k_1": k1, "K_2": K2, "S_1": S1, "min_G": mG,
            "M_cert": Mc, "status": status}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    BASELINE_SAMEFAMILY = 1.29005
    BASELINE_MV = 1.27481

    print("Agent M12: arcsine x B-spline (cardinal) mix sweep")
    print("=" * 95)
    print(f"delta_1 = {DELTA1} (arcsine, fixed at MV optimal)")
    print(f"Same-family 2-scale arcsine baseline: M_cert = {BASELINE_SAMEFAMILY}")
    print(f"MV (single arcsine):                  M_cert = {BASELINE_MV}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: grid sweep
    # ------------------------------------------------------------------
    n_list = [1, 2, 3, 4, 5, 6, 8, 10]
    delta_2_list = [0.03, 0.05, 0.07, 0.10, 0.138]
    lambda_list = [0.6, 0.7, 0.8, 0.85, 0.90, 0.93, 0.95]

    print(f"{'n':>3} {'d_2':>7} {'lam':>5} "
          f"{'k_1':>8} {'K_2':>7} {'S_1':>9} {'min_G':>7} {'M_cert':>9}")
    print("-" * 90)

    all_results = []
    best = {"M_cert": -np.inf, "params": None}

    for n in n_list:
        for d2 in delta_2_list:
            for lam in lambda_list:
                r = eval_at_2(d2, n, lam)
                if r.get("M_cert") is not None:
                    Mc = r["M_cert"]
                    marker = ""
                    if Mc > best["M_cert"]:
                        marker = " ***"
                    if Mc > BASELINE_SAMEFAMILY:
                        marker += " <<BEATS SAME-FAMILY>>"
                    print(f"{n:>3d} {d2:>7.4f} {lam:>5.2f} "
                          f"{r['k_1']:>8.5f} {r['K_2']:>7.4f} {r['S_1']:>9.4f} "
                          f"{r['min_G']:>7.5f} {Mc:>9.5f}{marker}")
                    rec = {"n": int(n), "delta_2": float(d2), "lambda": float(lam),
                           **r}
                    all_results.append(rec)
                    if Mc > best["M_cert"]:
                        best = {"M_cert": float(Mc), "params": rec}

    print()
    print(f"PHASE-1 BEST: M_cert = {best['M_cert']:.5f}")
    print(f"  Params: n={best['params']['n']}, "
          f"delta_2={best['params']['delta_2']:.5f}, "
          f"lambda={best['params']['lambda']:.4f}")
    print(f"  vs same-family 1.29005: {best['M_cert'] - BASELINE_SAMEFAMILY:+.5f}")
    print(f"  vs MV 1.27481:          {best['M_cert'] - BASELINE_MV:+.5f}")
    print()

    # ------------------------------------------------------------------
    # Phase 2: refined DE around best n (keep n integer)
    # ------------------------------------------------------------------
    print("--- Phase 2: refined DE (continuous d_2, lam; n integer scan around best) ---")
    n_star = int(best["params"]["n"])
    d2_0 = best["params"]["delta_2"]
    l_0 = best["params"]["lambda"]
    de_best = {"M_cert": best["M_cert"], "params": dict(best["params"])}

    # Scan n in {n_star, n_star+/-1, n_star+/-2} where positive
    n_refine = sorted({max(1, n_star - 2), max(1, n_star - 1), n_star,
                       n_star + 1, n_star + 2})
    for n in n_refine:
        d2_lo = max(0.01, d2_0 * 0.5)
        d2_hi = min(0.20, d2_0 * 2.0 + 0.04)
        l_lo = max(0.40, l_0 - 0.15)
        l_hi = min(0.99, l_0 + 0.08)

        def neg_M_cert(x, n_fixed=n):
            d2, lam = x
            r = eval_at_2(float(d2), int(n_fixed), float(lam))
            Mc = r.get("M_cert")
            if Mc is None or not np.isfinite(Mc):
                return 0.0
            return -float(Mc)

        bounds = [(d2_lo, d2_hi), (l_lo, l_hi)]

        try:
            de_res = differential_evolution(
                neg_M_cert, bounds,
                maxiter=25, popsize=10, tol=1e-7, seed=1 + n,
                polish=True, workers=1,
            )
            Mc = -float(de_res.fun)
            d2_opt, l_opt = float(de_res.x[0]), float(de_res.x[1])
            print(f"  n={n}: best M_cert={Mc:.6f} at d2={d2_opt:.5f}, lam={l_opt:.4f}")
            r = eval_at_2(d2_opt, n, l_opt)
            rec = {"n": int(n), "delta_2": d2_opt, "lambda": l_opt, "phase": "DE",
                   **r}
            all_results.append(rec)
            if Mc > de_best["M_cert"]:
                de_best = {"M_cert": Mc, "params": rec}
        except Exception as e:
            print(f"  n={n}: DE failed: {e}")

    print()
    print(f"PHASE-2 BEST: M_cert = {de_best['M_cert']:.6f}")
    print(f"  Params: {de_best['params']}")
    print(f"  vs same-family 1.29005: {de_best['M_cert'] - BASELINE_SAMEFAMILY:+.6f}")
    print()

    # ------------------------------------------------------------------
    # Phase 3: 3-component (arcsine + 2 B-splines)
    # ------------------------------------------------------------------
    print("--- Phase 3: 3-component (arcsine + 2 B-splines) ---")
    # Try a small set of complementary B-spline orders and scales.
    # Build initial structure around DE best n=n_de, and pair with a different n.
    n_de = int(de_best["params"]["n"])
    d2_de = float(de_best["params"]["delta_2"])
    lam_de = float(de_best["params"]["lambda"])

    # Try second B-spline component with a small set of (n_other, d_other) pairs.
    other_n_list = [1, 2, 3, 4, 6, 8]
    other_d_list = [0.03, 0.05, 0.07, 0.10]
    mu_list = [0.3, 0.5, 0.7]   # weight of "first" B-spline within the (1-lam) mass

    best3 = {"M_cert": -np.inf, "params": None}
    print(f"  base: arcsine + B-spline(n={n_de}, d={d2_de:.4f}) at lam={lam_de:.4f}")
    print(f"{'n_a':>3} {'d_a':>7} {'n_b':>3} {'d_b':>7} {'lam':>5} {'mu':>5} {'M_cert':>9}")
    print("-" * 65)

    for n_b in other_n_list:
        for d_b in other_d_list:
            if n_b == n_de and abs(d_b - d2_de) < 1e-6:
                continue  # same as base
            for mu in mu_list:
                r = eval_at_3(d2_de, n_de, d_b, n_b, lam_de, mu)
                if r.get("M_cert") is not None:
                    Mc = r["M_cert"]
                    marker = " ***" if Mc > best3["M_cert"] else ""
                    if Mc > de_best["M_cert"]:
                        marker += " <<beats 2-comp>>"
                    print(f"{n_de:>3d} {d2_de:>7.4f} {n_b:>3d} {d_b:>7.4f} "
                          f"{lam_de:>5.2f} {mu:>5.2f} {Mc:>9.5f}{marker}")
                    rec = {"phase": "3comp",
                           "n_a": int(n_de), "delta_a": d2_de,
                           "n_b": int(n_b), "delta_b": float(d_b),
                           "lambda": lam_de, "mu": float(mu), **r}
                    all_results.append(rec)
                    if Mc > best3["M_cert"]:
                        best3 = {"M_cert": float(Mc), "params": rec}

    if best3["params"] is not None:
        print()
        print(f"PHASE-3 BEST (grid): M_cert = {best3['M_cert']:.6f}")
        # Now DE-refine the 6-dim continuous problem (with fixed integer n_a, n_b)
        n_a = int(best3["params"]["n_a"])
        n_b = int(best3["params"]["n_b"])
        d_a0 = float(best3["params"]["delta_a"])
        d_b0 = float(best3["params"]["delta_b"])
        l0 = float(best3["params"]["lambda"])
        m0 = float(best3["params"]["mu"])

        def neg3(x):
            d_a, d_b, lam, mu = x
            r = eval_at_3(float(d_a), n_a, float(d_b), n_b, float(lam), float(mu))
            Mc = r.get("M_cert")
            if Mc is None or not np.isfinite(Mc):
                return 0.0
            return -float(Mc)

        bounds3 = [
            (max(0.01, d_a0 * 0.5), min(0.20, d_a0 * 2.0 + 0.04)),
            (max(0.01, d_b0 * 0.5), min(0.20, d_b0 * 2.0 + 0.04)),
            (max(0.40, l0 - 0.15), min(0.99, l0 + 0.08)),
            (0.05, 0.95),
        ]
        try:
            de3 = differential_evolution(
                neg3, bounds3, maxiter=30, popsize=15, tol=1e-7,
                seed=7, polish=True, workers=1,
            )
            Mc3 = -float(de3.fun)
            d_a_opt, d_b_opt, l_opt, m_opt = (float(de3.x[0]), float(de3.x[1]),
                                              float(de3.x[2]), float(de3.x[3]))
            r3 = eval_at_3(d_a_opt, n_a, d_b_opt, n_b, l_opt, m_opt)
            print(f"  Phase-3 DE: M_cert={Mc3:.6f} at "
                  f"n_a={n_a}, d_a={d_a_opt:.5f}, n_b={n_b}, d_b={d_b_opt:.5f}, "
                  f"lam={l_opt:.4f}, mu={m_opt:.4f}")
            rec3 = {"phase": "3comp_DE",
                    "n_a": n_a, "delta_a": d_a_opt,
                    "n_b": n_b, "delta_b": d_b_opt,
                    "lambda": l_opt, "mu": m_opt, **r3}
            all_results.append(rec3)
            if Mc3 > best3["M_cert"]:
                best3 = {"M_cert": Mc3, "params": rec3}
        except Exception as e:
            print(f"  Phase-3 DE failed: {e}")

    # ------------------------------------------------------------------
    # Save and print summary
    # ------------------------------------------------------------------
    overall_best = de_best
    if best3.get("params") is not None and best3["M_cert"] > overall_best["M_cert"]:
        overall_best = best3

    elapsed = time.time() - t0
    print()
    print("=" * 95)
    print(f"OVERALL BEST: M_cert = {overall_best['M_cert']:.6f}")
    print(f"  Params: {overall_best['params']}")
    print(f"  vs same-family 2-scale arcsine 1.29005: "
          f"{overall_best['M_cert'] - BASELINE_SAMEFAMILY:+.6f}")
    print(f"  vs MV 1.27481:                          "
          f"{overall_best['M_cert'] - BASELINE_MV:+.6f}")
    print(f"  elapsed: {elapsed:.1f} s")

    out = {
        "baseline_MV": BASELINE_MV,
        "baseline_samefamily_2scale_arcsine": BASELINE_SAMEFAMILY,
        "phase1_best": best,
        "phase2_best": de_best,
        "phase3_best": best3 if best3.get("params") is not None else None,
        "overall_best": overall_best,
        "all_results": all_results,
        "elapsed_seconds": elapsed,
    }
    with open("_M12_arc_bspline_mix.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("Wrote _M12_arc_bspline_mix.json")


if __name__ == "__main__":
    main()
