"""Agent M3: Cross-family mix of arcsine (delta_1) and Chebyshev-beta (delta_2, beta).

K(x) = lam * K_arcsine(x; delta_1) + (1 - lam) * K_chebyshev_beta(x; delta_2, beta)

For convex combo Bochner: each K_hat >= 0 individually, sum >= 0 trivially.
arcsine = Chebyshev-beta with beta = 1/2, so we require beta != 0.5 to be cross-family.

Reuse the _K26_full_sweep_reopt pipeline:
  - same solve_QP / M_cert / K_2_quad infrastructure
  - override K_hat with K_hat_mix
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.special import j0, gamma as scipy_gamma, jv
from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution
import cvxpy as cp

# Reuse constants & helpers from _K26_full_sweep_reopt
from _K26_full_sweep_reopt import (
    DELTA as DELTA1_DEFAULT,
    U,
    N_QP,
    M_cert,
)

DELTA1 = 0.138  # arcsine scale fixed at MV's optimal


# -----------------------------------------------------------------------------
# Custom K_hat-side evaluators
# -----------------------------------------------------------------------------

def K_hat_arc(xi, delta):
    return j0(np.pi * delta * np.asarray(xi)) ** 2


def K_hat_cheby_beta(xi, delta, beta):
    """K_hat_cheby_beta(xi; delta, beta) = (phi_hat(xi))^2

    phi_hat(xi) = Gamma(beta + 1/2) * (2/(pi*delta*xi))^(beta-1/2) * J_{beta-1/2}(pi*delta*xi)
    With limit phi_hat(0) = 1 (mass preservation).
    """
    xi = np.asarray(xi, dtype=float)
    arg = np.pi * delta * xi
    nu = beta - 0.5
    pref = scipy_gamma(beta + 0.5)

    # Mask near 0 to avoid singularity; use closed-form limit phi_hat(0) = 1.
    near_zero = np.abs(arg) < 1e-10
    safe_arg = np.where(near_zero, 1.0, arg)
    pow_factor = (2.0 / safe_arg) ** nu
    Jnu = jv(nu, safe_arg)
    phi_hat = pref * pow_factor * Jnu
    K_hat = phi_hat ** 2
    return np.where(near_zero, 1.0, K_hat)


def K_hat_mix(xi, delta1, delta2, beta, lam):
    return lam * K_hat_arc(xi, delta1) + (1.0 - lam) * K_hat_cheby_beta(xi, delta2, beta)


# -----------------------------------------------------------------------------
# Re-implemented solve_QP / K_2_quad / eval_at using K_hat_mix
# -----------------------------------------------------------------------------

def solve_QP_mix(delta1, delta2, beta, lam, n_grid=5001):
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = K_hat_mix(j / U, delta1, delta2, beta, lam)
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


def K_2_quad_mix(delta1, delta2, beta, lam):
    def f(xi):
        return K_hat_mix(xi, delta1, delta2, beta, lam) ** 2
    # Split: arcsine piece (delta1=0.138) has compact-ish decay; Cheby-beta tail
    # behaves like xi^{-4 beta}. For very small beta the tail integration needs
    # care; we use scipy quad with adaptive intervals.
    v1, _ = quad(f, 0.0, 10.0, limit=400, epsabs=1e-12, epsrel=1e-10)
    v2, _ = quad(f, 10.0, 100.0, limit=400, epsabs=1e-12, epsrel=1e-10)
    v3, _ = quad(f, 100.0, np.inf, limit=400, epsabs=1e-12, epsrel=1e-10)
    return 2.0 * (v1 + v2 + v3)


def eval_at_mix(delta2, beta, lam, delta1=DELTA1):
    a_opt, S1, mG, status = solve_QP_mix(delta1, delta2, beta, lam)
    if status != "ok":
        return {"status": status}
    k1 = float(K_hat_mix(1.0, delta1, delta2, beta, lam))
    K2 = float(K_2_quad_mix(delta1, delta2, beta, lam))
    Mc = M_cert(k1, K2, S1, mG)
    return {
        "k_1": k1, "K_2": K2, "S_1": S1, "min_G": mG,
        "M_cert": Mc, "status": status,
    }


# -----------------------------------------------------------------------------
# Bochner sanity check (each K_hat >= 0 on its own)
# -----------------------------------------------------------------------------

def check_bochner(delta1, delta2, beta, lam, xi_max=200.0, n_check=20000):
    xs = np.linspace(1e-6, xi_max, n_check)
    Ka = K_hat_arc(xs, delta1)
    Kb = K_hat_cheby_beta(xs, delta2, beta)
    Kmix = lam * Ka + (1 - lam) * Kb
    return {
        "min_K_arc": float(Ka.min()),
        "min_K_cheby": float(Kb.min()),
        "min_K_mix": float(Kmix.min()),
        "all_nonneg": bool(Ka.min() >= -1e-12 and Kb.min() >= -1e-12 and Kmix.min() >= -1e-12),
    }


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    BASELINE_SAMEFAMILY = 1.29005  # multi-scale arcsine
    BASELINE_MV = 1.27481

    print("Agent M3: arcsine x Chebyshev-beta cross-family mix sweep")
    print("=" * 90)
    print(f"delta_1 = {DELTA1} (arcsine, fixed at MV optimal)")
    print(f"Same-family 2-scale arcsine baseline: M_cert = {BASELINE_SAMEFAMILY}")
    print(f"MV baseline:                          M_cert = {BASELINE_MV}")
    print()

    # Phase 1: coarse grid
    delta_2_list = [0.025, 0.045, 0.07, 0.10, 0.138]
    beta_list = [0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
    lambda_list = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95]

    print(f"{'d_2':>7} {'beta':>5} {'lam':>5} "
          f"{'k_1':>8} {'K_2':>7} {'S_1':>9} {'min_G':>7} {'M_cert':>9}")
    print("-" * 90)

    all_results = []
    bochner_warnings = []
    best = {"M_cert": -np.inf, "params": None}

    for d2 in delta_2_list:
        for beta in beta_list:
            for lam in lambda_list:
                bc = check_bochner(DELTA1, d2, beta, lam)
                if not bc["all_nonneg"]:
                    bochner_warnings.append({
                        "delta_2": d2, "beta": beta, "lambda": lam, **bc,
                    })
                r = eval_at_mix(d2, beta, lam)
                if r.get("M_cert") is not None:
                    Mc = r["M_cert"]
                    marker = ""
                    if Mc > best["M_cert"]:
                        marker = " ***"
                    if Mc > BASELINE_SAMEFAMILY:
                        marker += " <<beats same-family>>"
                    print(f"{d2:>7.4f} {beta:>5.2f} {lam:>5.2f} "
                          f"{r['k_1']:>8.5f} {r['K_2']:>7.4f} {r['S_1']:>9.4f} "
                          f"{r['min_G']:>7.5f} {Mc:>9.5f}{marker}")
                    rec = {
                        "delta_2": d2, "beta": beta, "lambda": lam,
                        "bochner_min": bc["min_K_mix"], **r,
                    }
                    all_results.append(rec)
                    if Mc > best["M_cert"]:
                        best = {"M_cert": float(Mc), "params": rec}
                else:
                    pass  # skip print for noisy failures

    print()
    print(f"PHASE-1 BEST: M_cert = {best['M_cert']:.5f}")
    print(f"  Params: delta_2={best['params']['delta_2']}, "
          f"beta={best['params']['beta']}, lambda={best['params']['lambda']}")
    print(f"  vs same-family 1.29005: {best['M_cert'] - BASELINE_SAMEFAMILY:+.5f}")
    print(f"  vs MV 1.27481:          {best['M_cert'] - BASELINE_MV:+.5f}")
    print()

    # Phase 2: Differential evolution around best
    print("--- Phase 2: differential evolution refinement ---")
    d2_0 = best["params"]["delta_2"]
    b_0 = best["params"]["beta"]
    l_0 = best["params"]["lambda"]

    # Define bounds around the phase-1 optimum
    d2_lo, d2_hi = max(0.01, d2_0 / 2), min(DELTA1, d2_0 * 2 + 0.02)
    b_lo, b_hi = max(0.30, b_0 - 0.30), min(2.0, b_0 + 0.30)
    l_lo, l_hi = max(0.40, l_0 - 0.20), min(0.99, l_0 + 0.10)
    print(f"  bounds: d2 in [{d2_lo:.4f}, {d2_hi:.4f}], "
          f"beta in [{b_lo:.3f}, {b_hi:.3f}], lam in [{l_lo:.3f}, {l_hi:.3f}]")

    # Skip beta=0.5 strictly
    def neg_M_cert(x):
        d2, b, lam = x
        if abs(b - 0.5) < 0.01:
            return 0.0  # penalize exact arcsine match
        r = eval_at_mix(float(d2), float(b), float(lam))
        Mc = r.get("M_cert")
        if Mc is None or not np.isfinite(Mc):
            return 0.0
        return -float(Mc)

    bounds = [(d2_lo, d2_hi), (b_lo, b_hi), (l_lo, l_hi)]

    de_best = {"M_cert": best["M_cert"], "x": [d2_0, b_0, l_0]}

    def cb(xk, convergence):
        Mc = -neg_M_cert(xk)
        if Mc > de_best["M_cert"]:
            de_best["M_cert"] = Mc
            de_best["x"] = [float(xk[0]), float(xk[1]), float(xk[2])]
            print(f"  DE: M_cert={Mc:.6f} at d2={xk[0]:.5f}, "
                  f"beta={xk[1]:.4f}, lam={xk[2]:.4f}")
        return False

    try:
        de_res = differential_evolution(
            neg_M_cert, bounds,
            maxiter=30, popsize=12, tol=1e-7, seed=1,
            polish=True, callback=cb, workers=1,
        )
        x_de = de_res.x
        Mc_de = -de_res.fun
        # Evaluate one more time at the DE result
        r_de = eval_at_mix(float(x_de[0]), float(x_de[1]), float(x_de[2]))
        if r_de.get("M_cert") is not None and r_de["M_cert"] > best["M_cert"]:
            best = {"M_cert": float(r_de["M_cert"]),
                    "params": {"delta_2": float(x_de[0]),
                               "beta": float(x_de[1]),
                               "lambda": float(x_de[2]),
                               **r_de}}
            all_results.append({"delta_2": float(x_de[0]),
                                "beta": float(x_de[1]),
                                "lambda": float(x_de[2]),
                                **r_de, "tag": "DE_refined"})
        print(f"  DE final: M_cert={Mc_de:.6f}")
    except Exception as e:
        print(f"  DE failed: {e}")

    print()
    print("=" * 90)
    print(f"FINAL BEST: M_cert = {best['M_cert']:.6f}")
    print(f"  Params: delta_2={best['params'].get('delta_2'):.5f}, "
          f"beta={best['params'].get('beta'):.4f}, "
          f"lambda={best['params'].get('lambda'):.4f}")
    print(f"  vs same-family 1.29005: {best['M_cert'] - BASELINE_SAMEFAMILY:+.5f}")
    print(f"  vs MV 1.27481:          {best['M_cert'] - BASELINE_MV:+.5f}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print(f"Bochner warnings (count): {len(bochner_warnings)}")

    out = {
        "agent": "M3_arc_cheb_mix",
        "delta_1": DELTA1,
        "baseline_same_family_M_cert": BASELINE_SAMEFAMILY,
        "baseline_MV_M_cert": BASELINE_MV,
        "best_M_cert": best["M_cert"],
        "best_params": best["params"],
        "improvement_vs_same_family": best["M_cert"] - BASELINE_SAMEFAMILY,
        "improvement_vs_MV": best["M_cert"] - BASELINE_MV,
        "n_phase1_results": len(all_results),
        "bochner_warnings": bochner_warnings,
        "all_results": all_results,
        "elapsed_s": time.time() - t0,
    }
    out_path = "_M3_arc_cheb_mix.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Wrote {out_path}")

    return best


if __name__ == "__main__":
    main()
