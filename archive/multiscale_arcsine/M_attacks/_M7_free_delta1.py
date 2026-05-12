"""Agent M7: free delta_1 (NOT fixed at MV's 0.138) + delta_2 + lambda_1 sweep + DE.

MV chose delta = 0.138 as the optimal single-scale arcsine. With multi-scale,
the joint optimum may shift. This script:

  1. Parameterises the 2-component arcsine kernel
        K(xi) = lam_1 * j0(pi * d_1 * xi)^2 + (1 - lam_1) * j0(pi * d_2 * xi)^2
     with u = 1/2 + d_1  (the MV coupling).

  2. Re-implements solve_QP / K_2_quad / M_cert as functions of (deltas, lambdas, u),
     so changing d_1 also changes u and the 119 QP frequency points {j/u}.

  3. Phase 1: coarse grid over (d_1, d_2, lam_1).
  4. Phase 2: differential_evolution over [0.10, 0.20] x [0.005, 0.10] x [0.5, 0.99].

Sanity reference: at d_1=0.138, d_2=0.045, lam_1=0.85 should give M_cert ~= 1.290.
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

N_QP = 119
MV_BASELINE = 1.27481
TWO_COMP_BASELINE = 1.29005  # numerical reference at (0.138, 0.045, 0.85)


# ---------------------------------------------------------------------------
# Kernel + pipeline (delta_1-aware: u is computed inside)
# ---------------------------------------------------------------------------
def K_hat_ms(xi, deltas, lambdas):
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * np.asarray(xi)) ** 2
    return out


def solve_QP_u(deltas, lambdas, u, n_grid=5001):
    """QP with frequency grid j/u, j=1..N_QP, and target B@a >= 1 on x in [0, 0.25].

    Returns (a_opt, S_1, min_G, status).
    """
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = K_hat_ms(j / u, deltas, lambdas)
    if w.min() <= 0:
        return None, None, None, "non-positive weight"
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, N_QP))
    for j in range(1, N_QP + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / u)
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


def M_cert_u(k_1, K_2, S_1, min_G, u):
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    a_gain = (4.0 / u) * (min_G ** 2) / S_1
    target = 2.0 / u + a_gain
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


def eval_at_free(delta_1, delta_2, lambda_1):
    """Evaluate M_cert with u = 0.5 + delta_1 and 2-scale kernel."""
    if delta_1 <= 0 or delta_1 >= 0.5:
        return {"status": "delta_1 out of (0, 0.5)"}
    if delta_2 <= 0:
        return {"status": "delta_2 <= 0"}
    if not (0.0 < lambda_1 < 1.0):
        return {"status": "lambda_1 out of (0,1)"}
    u = 0.5 + delta_1
    deltas = [delta_1, delta_2]
    lambdas = [lambda_1, 1.0 - lambda_1]
    a_opt, S1, mG, status = solve_QP_u(deltas, lambdas, u)
    if status != "ok":
        return {"status": status, "u": u}
    k1 = float(K_hat_ms(1.0, deltas, lambdas))
    K2 = float(K_2_quad(deltas, lambdas))
    Mc = M_cert_u(k1, K2, S1, mG, u)
    return {
        "delta_1": float(delta_1),
        "delta_2": float(delta_2),
        "lambda_1": float(lambda_1),
        "u": float(u),
        "k_1": k1,
        "K_2": K2,
        "S_1": S1,
        "min_G": mG,
        "M_cert": Mc,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Phase 1: coarse grid
# ---------------------------------------------------------------------------
def phase1_grid():
    delta_1_list = [0.10, 0.115, 0.125, 0.138, 0.15, 0.16, 0.175, 0.20]
    # delta_2 scaled by d_1/0.138 anchor:
    base_d2_list = [0.02, 0.035, 0.05, 0.07, 0.09]
    lambda_1_list = [0.75, 0.80, 0.85, 0.90, 0.93]

    print("=" * 90)
    print("Phase 1: coarse grid (d_1, d_2, lam_1)")
    print(f"  d_1 in {delta_1_list}")
    print(f"  base d_2 in {base_d2_list} (scaled by d_1/0.138)")
    print(f"  lam_1 in {lambda_1_list}")
    print("=" * 90)
    print(
        f"{'d_1':>7} {'d_2':>7} {'lam1':>6} {'u':>7} {'k_1':>8} {'K_2':>7} "
        f"{'S_1':>9} {'min_G':>7} {'M_cert':>9}"
    )
    print("-" * 90)

    results = []
    best = {"M_cert": -np.inf, "params": None}
    t0 = time.time()
    n_evaluated = 0
    for d1 in delta_1_list:
        for base_d2 in base_d2_list:
            d2 = base_d2 * (d1 / 0.138)
            # Avoid degeneracy with d_2 >= d_1
            if d2 >= d1 - 1e-4:
                continue
            for l1 in lambda_1_list:
                n_evaluated += 1
                r = eval_at_free(d1, d2, l1)
                Mc = r.get("M_cert")
                if Mc is None or not np.isfinite(Mc):
                    print(
                        f"{d1:>7.4f} {d2:>7.4f} {l1:>6.3f}   ---  SKIPPED ({r.get('status')})"
                    )
                    continue
                results.append(r)
                if Mc > best["M_cert"]:
                    best = {"M_cert": float(Mc), "params": dict(r)}
                    marker = " *"
                else:
                    marker = ""
                print(
                    f"{d1:>7.4f} {d2:>7.4f} {l1:>6.3f} {r['u']:>7.4f} "
                    f"{r['k_1']:>8.5f} {r['K_2']:>7.4f} {r['S_1']:>9.4f} "
                    f"{r['min_G']:>7.5f} {Mc:>9.5f}{marker}"
                )

    dt = time.time() - t0
    print(f"\nPhase 1: {len(results)} valid / {n_evaluated} evaluated in {dt:.1f}s")
    if best["params"] is not None:
        p = best["params"]
        print(
            f"BEST: M_cert={best['M_cert']:.5f} at "
            f"d_1={p['delta_1']:.4f}, d_2={p['delta_2']:.5f}, lam_1={p['lambda_1']:.3f}"
        )
        print(f"  vs MV   1.27481: {best['M_cert'] - MV_BASELINE:+.5f}")
        print(f"  vs 2-comp 1.29005: {best['M_cert'] - TWO_COMP_BASELINE:+.5f}")
    return results, best


# ---------------------------------------------------------------------------
# Phase 2: DE
# ---------------------------------------------------------------------------
def neg_M_cert_3param(x):
    """For DE: x = (d_1, d_2, lam_1). Returns -M_cert (DE minimises)."""
    d1, d2, l1 = float(x[0]), float(x[1]), float(x[2])
    if d1 <= 0 or d1 >= 0.4:
        return 100.0
    if d2 <= 0 or d2 >= d1 - 1e-4:
        return 100.0
    if not (0.4 < l1 < 0.995):
        return 100.0
    r = eval_at_free(d1, d2, l1)
    Mc = r.get("M_cert")
    if Mc is None or not np.isfinite(Mc):
        return 100.0
    return -float(Mc)


def phase2_DE():
    bounds = [(0.10, 0.20), (0.005, 0.10), (0.5, 0.99)]
    print("\n" + "=" * 90)
    print("Phase 2: differential evolution")
    print(f"  bounds: {bounds}")
    print("=" * 90)
    t0 = time.time()
    res = differential_evolution(
        neg_M_cert_3param,
        bounds,
        seed=0,
        maxiter=40,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        workers=1,
    )
    dt = time.time() - t0
    d1, d2, l1 = res.x
    Mde = -res.fun
    print(
        f"  DE finished in {dt:.1f}s nit={res.nit} nfev={res.nfev}  ->  M_cert={Mde:.6f}"
    )
    print(f"  best: d_1={d1:.5f} d_2={d2:.5f} lam_1={l1:.5f}")
    final = eval_at_free(float(d1), float(d2), float(l1))
    final["DE_runtime_s"] = dt
    final["DE_nit"] = int(res.nit)
    final["DE_nfev"] = int(res.nfev)
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 90)
    print("AGENT M7: free delta_1 + delta_2 + lambda_1 sweep + DE")
    print(f"  MV baseline = {MV_BASELINE}")
    print(f"  2-comp baseline (d_1=0.138, d_2=0.045, lam_1=0.85) = {TWO_COMP_BASELINE}")
    print("=" * 90)

    # Sanity check: reproduce 2-comp baseline.
    r_chk = eval_at_free(0.138, 0.045, 0.85)
    print(
        f"\nSanity: d_1=0.138 d_2=0.045 lam_1=0.85 -> M_cert={r_chk.get('M_cert')}"
    )
    if r_chk.get("M_cert") is not None:
        print(f"  delta from 1.29005: {r_chk['M_cert'] - TWO_COMP_BASELINE:+.5f}")

    # Phase 1
    results, best_p1 = phase1_grid()

    out = {
        "MV_baseline": MV_BASELINE,
        "two_comp_baseline": TWO_COMP_BASELINE,
        "sanity_check": r_chk,
        "phase1_all": results,
        "phase1_best": best_p1["params"],
    }

    # Phase 2
    best_p2 = phase2_DE()
    out["phase2_DE_best"] = best_p2

    # Overall best
    p1_M = best_p1["M_cert"] if best_p1["params"] is not None else -np.inf
    p2_M = best_p2["M_cert"] if best_p2.get("M_cert") is not None else -np.inf
    if p2_M > p1_M:
        overall = best_p2
    else:
        overall = best_p1["params"]
    out["overall_best"] = overall
    out["improvement_over_MV"] = (
        overall["M_cert"] - MV_BASELINE if overall.get("M_cert") is not None else None
    )
    out["improvement_over_2comp"] = (
        overall["M_cert"] - TWO_COMP_BASELINE
        if overall.get("M_cert") is not None
        else None
    )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print(
        f"  Phase 1 best:  M_cert={p1_M:.6f}  at "
        f"d_1={best_p1['params']['delta_1']:.5f}, "
        f"d_2={best_p1['params']['delta_2']:.5f}, "
        f"lam_1={best_p1['params']['lambda_1']:.5f}"
        if best_p1["params"] is not None
        else "  Phase 1 best: (none)"
    )
    print(
        f"  Phase 2 best:  M_cert={p2_M:.6f}  at "
        f"d_1={best_p2['delta_1']:.5f}, "
        f"d_2={best_p2['delta_2']:.5f}, "
        f"lam_1={best_p2['lambda_1']:.5f}"
        if best_p2.get("M_cert") is not None
        else "  Phase 2 best: (none)"
    )
    print(
        f"  OVERALL:       M_cert={overall.get('M_cert'):.6f}  at "
        f"d_1={overall['delta_1']:.5f}, "
        f"d_2={overall['delta_2']:.5f}, "
        f"lam_1={overall['lambda_1']:.5f}"
        if overall.get("M_cert") is not None
        else "  OVERALL: (none)"
    )
    if overall.get("M_cert") is not None:
        print(f"  vs MV       1.27481: {overall['M_cert'] - MV_BASELINE:+.6f}")
        print(f"  vs 2-comp   1.29005: {overall['M_cert'] - TWO_COMP_BASELINE:+.6f}")
        print(
            f"  Optimum at d_1=0.138? {'YES' if abs(overall['delta_1'] - 0.138) < 0.002 else 'NO (shifted)'}"
        )

    with open("_M7_free_delta1_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Wrote _M7_free_delta1_result.json")


if __name__ == "__main__":
    main()
