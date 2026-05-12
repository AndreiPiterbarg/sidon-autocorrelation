"""Agent M14: Multi-scale Chebyshev-beta kernel sweep.

K_hat_cheby_beta(xi, delta, beta) = (Gamma(beta+1/2) * (2/(pi*delta*xi))^(beta-1/2)
                                    * J_{beta-1/2}(pi*delta*xi))^2
beta=1/2 recovers arcsine (J_0(pi*delta*xi))^2.

Pipeline reused from _K26_full_sweep_reopt.py: solve_QP, K_2_quad, M_cert.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.special import jv, gamma as gamma_fn
from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution
import cvxpy as cp

DELTA_FIXED = 0.138  # only used as anchor for sanity check
U_DEFAULT = 0.5 + DELTA_FIXED  # we'll let U vary as 0.5 + max(deltas) when needed
N_QP = 119
BASELINE_MV = 1.27481
BASELINE_2SCALE_ARCSINE = 1.29005  # claimed by the parallel-sweep coordinator


# ---------- Kernel ----------

def K_hat_cheby_beta_single(xi, delta, beta):
    """Scalar evaluation of K_hat_cheby_beta. Handles xi -> 0 limit (= 1)."""
    xi = float(xi)
    if xi == 0.0:
        return 1.0
    arg = math.pi * delta * xi
    order = beta - 0.5
    pref = gamma_fn(beta + 0.5) * (2.0 / arg) ** order
    val = pref * jv(order, arg)
    return float(val * val)


def K_hat_cheby_beta(xi, delta, beta):
    """Vectorized K_hat_cheby_beta. K_hat(0)=1 by construction."""
    xi = np.asarray(xi, dtype=float)
    out = np.empty_like(xi)
    mask0 = xi == 0.0
    out[mask0] = 1.0
    nz = ~mask0
    if np.any(nz):
        arg = math.pi * delta * xi[nz]
        order = beta - 0.5
        # (2/arg)^order * J_order(arg) is well-defined and >0 for small arg
        # since J_order(x) ~ (x/2)^order / Gamma(order+1) for small x.
        pref = gamma_fn(beta + 0.5) * (2.0 / arg) ** order
        val = pref * jv(order, arg)
        out[nz] = val * val
    return out


def K_hat_mix(xi, betas, deltas, lambdas):
    """K_hat = sum_i lambda_i * K_hat_cheby_beta(xi, delta_i, beta_i).
    Each component is Bochner-positive (auto-convolution squared), so the
    convex combination is Bochner-positive automatically.
    """
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi) if xi.ndim > 0 else 0.0
    for lam, d, b in zip(lambdas, deltas, betas):
        if lam == 0:
            continue
        out = out + lam * K_hat_cheby_beta(xi, d, b)
    return out


# ---------- QP / certificate ----------

def solve_QP(betas, deltas, lambdas, U, n_grid=5001):
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = float(K_hat_mix(np.array([j / U]), betas, deltas, lambdas)[0])
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
        try:
            prob.solve(solver="CLARABEL", verbose=False)
            if a.value is None:
                return None, None, None, f"QP exception: {e}"
        except Exception as e2:
            return None, None, None, f"QP exception: {e2}"
    a_opt = np.asarray(a.value).flatten()
    S1 = float(np.sum(a_opt ** 2 / w))
    min_G = float((B @ a_opt).min())
    return a_opt, S1, min_G, "ok"


def K_2_quad(betas, deltas, lambdas):
    def f(xi):
        v = K_hat_mix(np.array([xi]), betas, deltas, lambdas)[0]
        return float(v * v)
    v1, _ = quad(f, 0.0, 10.0, limit=400, epsabs=1e-12, epsrel=1e-10)
    v2, _ = quad(f, 10.0, np.inf, limit=400, epsabs=1e-12, epsrel=1e-10)
    return 2.0 * (v1 + v2)


def M_cert(k_1, K_2, S_1, min_G, U):
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


def eval_at(betas, deltas, lambdas, U=None, n_grid=5001):
    if U is None:
        U = U_DEFAULT
    a_opt, S1, mG, status = solve_QP(betas, deltas, lambdas, U, n_grid=n_grid)
    if status != "ok":
        return {"status": status}
    k1 = float(K_hat_mix(np.array([1.0]), betas, deltas, lambdas)[0])
    K2 = K_2_quad(betas, deltas, lambdas)
    Mc = M_cert(k1, K2, S1, mG, U)
    return {"k_1": k1, "K_2": float(K2), "S_1": float(S1), "min_G": float(mG),
            "M_cert": Mc, "status": status, "U": U}


# ---------- Sanity check: arcsine reduction ----------

def sanity_arcsine_reduction():
    """At beta_1=beta_2=0.5 the mixture should be 2-scale arcsine.
    Compare to MV-like single-scale arcsine and look for M_cert ~ 1.29005.
    """
    print("Sanity: beta_1=beta_2=0.5 -> 2-scale arcsine. Expect to find ~1.29 region.")
    # Try a coarse 2D scan over (delta_2, lambda_1) at delta_1=0.138, betas=0.5.
    best = -np.inf
    best_p = None
    for d2 in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        for l1 in [0.88, 0.90, 0.92, 0.94, 0.96]:
            r = eval_at([0.5, 0.5], [DELTA_FIXED, d2], [l1, 1.0 - l1])
            if r.get("M_cert") is not None and r["M_cert"] > best:
                best = r["M_cert"]
                best_p = {"d2": d2, "l1": l1, **{k: r[k] for k in ("k_1","K_2","S_1","min_G")}}
    print(f"  arcsine 2-scale sanity best M_cert = {best:.5f} at {best_p}")
    return best, best_p


# ---------- DE wrappers ----------

def neg_M_phase1(x, beta_fixed):
    """Phase 1: same beta, sweep (delta_1, delta_2, lambda_1)."""
    d1, d2, l1 = x
    if d1 <= 0 or d2 <= 0 or d1 > 0.5 or d2 > 0.5:
        return 0.0
    if l1 < 0.0 or l1 > 1.0:
        return 0.0
    U = 0.5 + max(d1, d2)
    r = eval_at([beta_fixed, beta_fixed], [d1, d2], [l1, 1.0 - l1], U=U, n_grid=4001)
    Mc = r.get("M_cert")
    if Mc is None:
        return 0.0
    return -Mc


def neg_M_phase2(x):
    """Phase 2: (beta_1, beta_2, delta_1, delta_2, lambda_1)."""
    b1, b2, d1, d2, l1 = x
    if d1 <= 0 or d2 <= 0 or d1 > 0.5 or d2 > 0.5:
        return 0.0
    if l1 < 0.0 or l1 > 1.0:
        return 0.0
    if b1 <= 0.0 or b2 <= 0.0:
        return 0.0
    U = 0.5 + max(d1, d2)
    r = eval_at([b1, b2], [d1, d2], [l1, 1.0 - l1], U=U, n_grid=4001)
    Mc = r.get("M_cert")
    if Mc is None:
        return 0.0
    return -Mc


def neg_M_phase3(x):
    """Phase 3: 3-scale (beta_1..3, delta_1..3, l1, l2).  l3 = 1 - l1 - l2."""
    b1, b2, b3, d1, d2, d3, l1, l2 = x
    l3 = 1.0 - l1 - l2
    if l1 < 0 or l2 < 0 or l3 < 0:
        return 0.0
    if d1 <= 0 or d2 <= 0 or d3 <= 0 or max(d1, d2, d3) > 0.5:
        return 0.0
    if min(b1, b2, b3) <= 0.0:
        return 0.0
    U = 0.5 + max(d1, d2, d3)
    r = eval_at([b1, b2, b3], [d1, d2, d3], [l1, l2, l3], U=U, n_grid=4001)
    Mc = r.get("M_cert")
    if Mc is None:
        return 0.0
    return -Mc


# ---------- Phases ----------

def run_phase1():
    print("\n=== Phase 1: same beta (offset from arcsine), 2-scale sweep ===")
    out = {}
    betas_to_try = [0.4, 0.45, 0.50, 0.55, 0.60]
    for b in betas_to_try:
        t0 = time.time()
        # bounds: d1, d2 in (0.02, 0.20); l1 in (0.5, 0.99)
        bounds = [(0.05, 0.20), (0.02, 0.15), (0.70, 0.99)]
        res = differential_evolution(
            neg_M_phase1, bounds, args=(b,),
            maxiter=30, popsize=14, tol=1e-7, seed=42, polish=True,
            workers=1, updating="immediate",
        )
        x = res.x
        d1, d2, l1 = x
        U = 0.5 + max(d1, d2)
        r = eval_at([b, b], [d1, d2], [l1, 1.0 - l1], U=U, n_grid=6001)
        Mc = r.get("M_cert")
        elapsed = time.time() - t0
        print(f"  beta={b:.2f}  M_cert={Mc:.5f}  d1={d1:.4f} d2={d2:.4f} l1={l1:.4f}  ({elapsed:.1f}s)")
        out[f"beta={b:.2f}"] = {
            "beta": b, "delta_1": float(d1), "delta_2": float(d2),
            "lambda_1": float(l1),
            "M_cert": Mc, "k_1": r.get("k_1"), "K_2": r.get("K_2"),
            "S_1": r.get("S_1"), "min_G": r.get("min_G"), "U": U,
        }
    return out


def run_phase2():
    print("\n=== Phase 2: different beta per component, 5D DE ===")
    bounds = [
        (0.30, 0.80),   # beta_1
        (0.30, 0.80),   # beta_2
        (0.05, 0.20),   # delta_1
        (0.02, 0.15),   # delta_2
        (0.70, 0.99),   # lambda_1
    ]
    t0 = time.time()
    res = differential_evolution(
        neg_M_phase2, bounds, maxiter=60, popsize=20, tol=1e-7,
        seed=42, polish=True, workers=1, updating="immediate",
    )
    x = res.x
    b1, b2, d1, d2, l1 = x
    U = 0.5 + max(d1, d2)
    r = eval_at([b1, b2], [d1, d2], [l1, 1.0 - l1], U=U, n_grid=6001)
    Mc = r.get("M_cert")
    elapsed = time.time() - t0
    print(f"  best M_cert={Mc:.5f}  b1={b1:.3f} b2={b2:.3f}  d1={d1:.4f} d2={d2:.4f}  l1={l1:.4f}  ({elapsed:.1f}s)")
    return {
        "beta_1": float(b1), "beta_2": float(b2),
        "delta_1": float(d1), "delta_2": float(d2), "lambda_1": float(l1),
        "M_cert": Mc, "k_1": r.get("k_1"), "K_2": r.get("K_2"),
        "S_1": r.get("S_1"), "min_G": r.get("min_G"), "U": U,
        "elapsed_s": elapsed,
    }


def run_phase3():
    print("\n=== Phase 3: 3-scale, varying beta, 8D DE ===")
    bounds = [
        (0.30, 0.80),   # beta_1
        (0.30, 0.80),   # beta_2
        (0.30, 0.80),   # beta_3
        (0.05, 0.20),   # delta_1
        (0.03, 0.15),   # delta_2
        (0.01, 0.10),   # delta_3
        (0.30, 0.90),   # lambda_1
        (0.05, 0.50),   # lambda_2 (l3 = 1-l1-l2 >= 0 enforced inside)
    ]
    t0 = time.time()
    res = differential_evolution(
        neg_M_phase3, bounds, maxiter=80, popsize=22, tol=1e-7,
        seed=42, polish=True, workers=1, updating="immediate",
    )
    x = res.x
    b1, b2, b3, d1, d2, d3, l1, l2 = x
    l3 = 1.0 - l1 - l2
    U = 0.5 + max(d1, d2, d3)
    r = eval_at([b1, b2, b3], [d1, d2, d3], [l1, l2, l3], U=U, n_grid=6001)
    Mc = r.get("M_cert")
    elapsed = time.time() - t0
    print(f"  best M_cert={Mc:.5f}  betas=({b1:.3f},{b2:.3f},{b3:.3f})  deltas=({d1:.4f},{d2:.4f},{d3:.4f})  lams=({l1:.4f},{l2:.4f},{l3:.4f})  ({elapsed:.1f}s)")
    return {
        "beta_1": float(b1), "beta_2": float(b2), "beta_3": float(b3),
        "delta_1": float(d1), "delta_2": float(d2), "delta_3": float(d3),
        "lambda_1": float(l1), "lambda_2": float(l2), "lambda_3": float(l3),
        "M_cert": Mc, "k_1": r.get("k_1"), "K_2": r.get("K_2"),
        "S_1": r.get("S_1"), "min_G": r.get("min_G"), "U": U,
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 80)
    print("Agent M14: Multi-scale Chebyshev-beta sweep.")
    print(f"Baselines: MV={BASELINE_MV}, 2-scale arcsine={BASELINE_2SCALE_ARCSINE}")
    print("=" * 80)

    sanity_M, sanity_p = sanity_arcsine_reduction()

    phase1 = run_phase1()
    phase2 = run_phase2()
    phase3 = run_phase3()

    # Determine overall best.
    candidates = []
    for k, v in phase1.items():
        if v.get("M_cert") is not None:
            candidates.append(("phase1_" + k, v["M_cert"], v))
    if phase2.get("M_cert") is not None:
        candidates.append(("phase2", phase2["M_cert"], phase2))
    if phase3.get("M_cert") is not None:
        candidates.append(("phase3", phase3["M_cert"], phase3))
    if candidates:
        best_label, best_M, best_p = max(candidates, key=lambda t: t[1])
    else:
        best_label, best_M, best_p = "none", None, None

    print("\n" + "=" * 80)
    print(f"OVERALL BEST: {best_label}  M_cert={best_M}")
    if best_M is not None:
        print(f"  vs MV 1.27481: {best_M - BASELINE_MV:+.5f}")
        print(f"  vs 2-scale arcsine 1.29005: {best_M - BASELINE_2SCALE_ARCSINE:+.5f}")
    print("=" * 80)

    out = {
        "baselines": {"MV": BASELINE_MV, "two_scale_arcsine": BASELINE_2SCALE_ARCSINE},
        "sanity_arcsine_2scale": {"M_cert": sanity_M, "params": sanity_p},
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "best": {"label": best_label, "M_cert": best_M, "params": best_p},
    }
    with open("_M14_multiscale_cheby_results.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else None)
    print("Wrote _M14_multiscale_cheby_results.json")


if __name__ == "__main__":
    main()
