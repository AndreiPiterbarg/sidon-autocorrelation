"""Agent M4: Arcsine x Askey cross-family mix sweep.

K_hat_mix(xi) = lambda * J_0(pi*delta_1*xi)^2 + (1-lambda) * K_hat_askey(xi, delta_2, nu)

Askey kernel K(x) = c*(1 - |x|/delta)_+^nu on [-delta, delta], c = (nu+1)/(2*delta).
K_hat_askey(xi) = (nu+1)/delta * int_0^delta (1 - x/delta)^nu cos(2*pi*xi*x) dx,
which gives K_hat_askey(0) = 1.

Askey 1973: K is positive-definite (K_hat >= 0) whenever nu >= 1 in 1D.
"""
from __future__ import annotations

import json
import math
import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq
import cvxpy as cp

DELTA1 = 0.138  # arcsine width fixed at MV optimum
U = 0.5 + DELTA1
N_QP = 119


# --------------------------------------------------------------------------
# Askey K_hat with frequency-level caching
# --------------------------------------------------------------------------

_ASKEY_CACHE: dict = {}


def _askey_one(xi: float, delta: float, nu: float) -> float:
    if xi == 0.0:
        return 1.0
    key = (round(float(xi), 12), round(float(delta), 10), round(float(nu), 6))
    hit = _ASKEY_CACHE.get(key)
    if hit is not None:
        return hit
    # Use Fourier-cosine weight for oscillatory robustness.
    # int_0^d (1-x/d)^nu cos(omega x) dx with omega = 2*pi*xi
    omega = 2.0 * math.pi * xi
    try:
        v, _ = quad(lambda x: (1.0 - x / delta) ** nu, 0.0, delta,
                    weight='cos', wvar=omega, limit=200, epsabs=1e-13, epsrel=1e-13)
    except Exception:
        v, _ = quad(lambda x: (1.0 - x / delta) ** nu * math.cos(omega * x),
                    0.0, delta, limit=400, epsabs=1e-12, epsrel=1e-12)
    out = (nu + 1.0) * v / delta
    _ASKEY_CACHE[key] = out
    return out


def K_hat_askey(xi_arr, delta: float, nu: float) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(xi_arr, dtype=float))
    out = np.empty_like(arr)
    for i, xi in enumerate(arr):
        out[i] = _askey_one(float(xi), delta, nu)
    return out


def K_hat_mix(xi, delta_1: float, delta_2: float, nu: float, lam: float):
    arc = j0(np.pi * delta_1 * np.asarray(xi, dtype=float)) ** 2
    ask = K_hat_askey(xi, delta_2, nu)
    return lam * arc + (1.0 - lam) * ask


# --------------------------------------------------------------------------
# QP, K_2, M_cert (parallel to _K26_full_sweep_reopt.py)
# --------------------------------------------------------------------------

def solve_QP(delta_1: float, delta_2: float, nu: float, lam: float, n_grid: int = 5001):
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = float(K_hat_mix(j / U, delta_1, delta_2, nu, lam))
    if w.min() <= 0.0:
        return None, None, None, f"non-positive weight (min={w.min():.3e})"
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
        try:
            prob.solve(solver="CLARABEL", verbose=False)
        except Exception:
            return None, None, None, f"QP exception: {e}"
    a_opt = np.asarray(a.value).flatten()
    S1 = float(np.sum(a_opt ** 2 / w))
    min_G = float((B @ a_opt).min())
    return a_opt, S1, min_G, "ok"


def K_2_quad(delta_1: float, delta_2: float, nu: float, lam: float) -> float:
    def f(xi):
        return float(K_hat_mix(xi, delta_1, delta_2, nu, lam)) ** 2
    # Both arcsine J_0(pi d xi)^2 and Askey K_hat decay polynomially; tail integrand ~ xi^{-2}.
    v1, _ = quad(f, 0.0, 10.0, limit=400, epsabs=1e-12, epsrel=1e-10)
    v2, _ = quad(f, 10.0, 200.0, limit=400, epsabs=1e-12, epsrel=1e-10)
    v3, _ = quad(f, 200.0, 5000.0, limit=400, epsabs=1e-12, epsrel=1e-10)
    return 2.0 * (v1 + v2 + v3)


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


def eval_at(delta_1: float, delta_2: float, nu: float, lam: float) -> dict:
    a_opt, S1, mG, status = solve_QP(delta_1, delta_2, nu, lam)
    if status != "ok":
        return {"status": status}
    k1 = float(K_hat_mix(1.0, delta_1, delta_2, nu, lam))
    K2 = float(K_2_quad(delta_1, delta_2, nu, lam))
    Mc = M_cert(k1, K2, S1, mG)
    return {
        "k_1": k1, "K_2": K2, "S_1": S1, "min_G": mG,
        "M_cert": Mc, "status": status,
    }


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def main():
    NU_LIST = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    D2_LIST = [0.04, 0.07, 0.10, 0.138]
    LAM_LIST = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95]

    print("Agent M4: Arcsine x Askey cross-family mix")
    print(f"  delta_1 = {DELTA1} fixed (arcsine)")
    print(f"  baseline 2-arcsine M_cert = 1.29005")
    print("=" * 92)
    print(f"{'nu':>5} {'d_2':>6} {'lam':>6} {'k_1':>9} {'K_2':>8} {'S_1':>9} {'min_G':>8} {'M_cert':>9}")
    print("-" * 92)

    all_results = []
    best = {"M_cert": -np.inf, "params": None}

    for nu in NU_LIST:
        for d2 in D2_LIST:
            for lam in LAM_LIST:
                r = eval_at(DELTA1, d2, nu, lam)
                if r.get("M_cert") is not None:
                    Mc = r["M_cert"]
                    marker = " *" if Mc > best["M_cert"] else ""
                    print(f"{nu:>5.2f} {d2:>6.3f} {lam:>6.3f} {r['k_1']:>9.5f} "
                          f"{r['K_2']:>8.4f} {r['S_1']:>9.4f} {r['min_G']:>8.5f} "
                          f"{Mc:>9.5f}{marker}")
                    rec = {"nu": nu, "delta_2": d2, "lambda": lam, **r}
                    all_results.append(rec)
                    if Mc > best["M_cert"]:
                        best = {"M_cert": float(Mc),
                                "params": {"nu": nu, "delta_2": d2, "lambda": lam,
                                           "k_1": r["k_1"], "K_2": r["K_2"],
                                           "S_1": r["S_1"], "min_G": r["min_G"]}}
                else:
                    print(f"{nu:>5.2f} {d2:>6.3f} {lam:>6.3f} --- SKIP ({r.get('status')}) ---")

    print()
    print(f"COARSE BEST: M_cert={best['M_cert']:.5f} at {best['params']}")
    print(f"  vs MV 1.27481:        {best['M_cert'] - 1.27481:+.5f}")
    print(f"  vs 2-arcsine 1.29005: {best['M_cert'] - 1.29005:+.5f}")

    # Refined sweep around best
    print()
    print("--- Refined sweep around coarse best ---")
    print(f"{'nu':>5} {'d_2':>6} {'lam':>6} {'k_1':>9} {'K_2':>8} {'S_1':>9} {'min_G':>8} {'M_cert':>9}")
    print("-" * 92)
    nu_0 = best["params"]["nu"]
    d2_0 = best["params"]["delta_2"]
    lam_0 = best["params"]["lambda"]
    nu_grid = sorted(set([max(1.05, nu_0 - 0.5), max(1.05, nu_0 - 0.25), nu_0,
                          nu_0 + 0.25, nu_0 + 0.5]))
    d2_grid = sorted(set(np.clip(
        [d2_0 - 0.02, d2_0 - 0.01, d2_0, d2_0 + 0.01, d2_0 + 0.02], 0.01, DELTA1 - 0.001).tolist()))
    lam_grid = sorted(set(np.clip(
        [lam_0 - 0.03, lam_0 - 0.015, lam_0, lam_0 + 0.015, lam_0 + 0.03], 0.5, 0.99).tolist()))
    for nu in nu_grid:
        for d2 in d2_grid:
            for lam in lam_grid:
                r = eval_at(DELTA1, float(d2), float(nu), float(lam))
                if r.get("M_cert") is not None:
                    Mc = r["M_cert"]
                    marker = " *" if Mc > best["M_cert"] else ""
                    print(f"{nu:>5.2f} {d2:>6.4f} {lam:>6.4f} {r['k_1']:>9.5f} "
                          f"{r['K_2']:>8.4f} {r['S_1']:>9.4f} {r['min_G']:>8.5f} "
                          f"{Mc:>9.5f}{marker}")
                    rec = {"nu": float(nu), "delta_2": float(d2),
                           "lambda": float(lam), **r}
                    all_results.append(rec)
                    if Mc > best["M_cert"]:
                        best = {"M_cert": float(Mc),
                                "params": {"nu": float(nu),
                                           "delta_2": float(d2),
                                           "lambda": float(lam),
                                           "k_1": r["k_1"], "K_2": r["K_2"],
                                           "S_1": r["S_1"], "min_G": r["min_G"]}}

    print()
    print(f"FINAL BEST: M_cert={best['M_cert']:.5f}")
    print(f"  Params: {best['params']}")
    print(f"  vs MV 1.27481:        {best['M_cert'] - 1.27481:+.5f}")
    print(f"  vs 2-arcsine 1.29005: {best['M_cert'] - 1.29005:+.5f}")

    out = {
        "agent": "M4",
        "kernel": "arcsine_x_askey_mix",
        "delta_1_fixed": DELTA1,
        "baseline_M_cert_MV": 1.27481,
        "baseline_2_arcsine": 1.29005,
        "best_M_cert": best["M_cert"],
        "best_params": best["params"],
        "improvement_vs_MV": (best["M_cert"] - 1.27481) if best["M_cert"] > -np.inf else None,
        "improvement_vs_2arcsine": (best["M_cert"] - 1.29005) if best["M_cert"] > -np.inf else None,
        "n_results": len(all_results),
        "n_askey_cache_entries": len(_ASKEY_CACHE),
        "all_results": all_results,
    }
    with open("_M4_arc_askey_mix.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote _M4_arc_askey_mix.json")


if __name__ == "__main__":
    main()
