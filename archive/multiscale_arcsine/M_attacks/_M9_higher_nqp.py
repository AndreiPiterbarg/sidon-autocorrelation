"""Agent M9: vary N_QP (number of cosines in G) and check M_cert saturation.

Tests both pure arcsine (delta=0.138) and 2-scale (delta_1=0.138, delta_2=0.045,
lambda_1=0.85) at N_QP in {119, 150, 200, 250, 300, 400, 500}.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq
import cvxpy as cp

DELTA = 0.138
U = 0.5 + DELTA


def K_hat_ms(xi, deltas, lambdas):
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * np.asarray(xi)) ** 2
    return out


def solve_QP(deltas, lambdas, N_QP, n_grid=5001):
    """Solve MV-style semi-infinite QP with N_QP cosines.

    min sum a_j^2 / w_j  s.t.  sum a_j cos(2 pi j x / U) >= 1 for x in [0, 1/4]
    where w_j = K_hat(j / U).
    """
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


def eval_at(deltas, lambdas, N_QP, n_grid=5001):
    t0 = time.time()
    a_opt, S1, mG, status = solve_QP(deltas, lambdas, N_QP, n_grid=n_grid)
    if status != "ok":
        return {"status": status, "wall": time.time() - t0}
    k1 = float(K_hat_ms(1.0, deltas, lambdas))
    K2 = K_2_quad(deltas, lambdas)
    Mc = M_cert(k1, K2, S1, mG)
    return {"k_1": k1, "K_2": float(K2), "S_1": S1, "min_G": mG,
            "M_cert": Mc, "status": status, "wall": time.time() - t0,
            "N_QP": N_QP}


def main():
    NQP_LIST = [119, 150, 200, 250, 300, 400, 500]

    # Pure arcsine: single delta = 0.138
    arcsine_deltas = [DELTA]
    arcsine_lams = [1.0]

    # Best 2-scale from M0/M1 sweep
    ms_deltas = [DELTA, 0.045]
    ms_lams = [0.85, 0.15]

    print("=" * 80)
    print("Agent M9: M_cert vs N_QP")
    print("=" * 80)
    print(f"{'N_QP':>6}  {'kind':>10}  {'k_1':>9}  {'K_2':>8}  {'S_1':>10}  "
          f"{'min_G':>8}  {'M_cert':>9}  {'wall_s':>7}")
    print("-" * 80)

    results = {"arcsine": [], "two_scale": []}
    best = {"M_cert": -np.inf, "N_QP": None, "kind": None,
            "deltas": None, "lambdas": None}

    for N_QP in NQP_LIST:
        # Pure arcsine
        r1 = eval_at(arcsine_deltas, arcsine_lams, N_QP)
        if r1.get("M_cert") is not None:
            print(f"{N_QP:>6}  {'arcsine':>10}  {r1['k_1']:>9.5f}  {r1['K_2']:>8.4f}  "
                  f"{r1['S_1']:>10.5f}  {r1['min_G']:>8.5f}  "
                  f"{r1['M_cert']:>9.5f}  {r1['wall']:>7.1f}")
            results["arcsine"].append({"N_QP": N_QP, **r1})
            if r1["M_cert"] > best["M_cert"]:
                best = {"M_cert": float(r1["M_cert"]), "N_QP": N_QP,
                        "kind": "arcsine", "deltas": arcsine_deltas,
                        "lambdas": arcsine_lams}
        else:
            print(f"{N_QP:>6}  {'arcsine':>10}  --- FAILED ({r1.get('status')}) "
                  f"wall={r1.get('wall', 0):.1f}s ---")
            results["arcsine"].append({"N_QP": N_QP, **r1})

        # 2-scale
        r2 = eval_at(ms_deltas, ms_lams, N_QP)
        if r2.get("M_cert") is not None:
            print(f"{N_QP:>6}  {'2-scale':>10}  {r2['k_1']:>9.5f}  {r2['K_2']:>8.4f}  "
                  f"{r2['S_1']:>10.5f}  {r2['min_G']:>8.5f}  "
                  f"{r2['M_cert']:>9.5f}  {r2['wall']:>7.1f}")
            results["two_scale"].append({"N_QP": N_QP, **r2})
            if r2["M_cert"] > best["M_cert"]:
                best = {"M_cert": float(r2["M_cert"]), "N_QP": N_QP,
                        "kind": "2-scale", "deltas": ms_deltas,
                        "lambdas": ms_lams}
        else:
            print(f"{N_QP:>6}  {'2-scale':>10}  --- FAILED ({r2.get('status')}) "
                  f"wall={r2.get('wall', 0):.1f}s ---")
            results["two_scale"].append({"N_QP": N_QP, **r2})

    print("-" * 80)
    print(f"BEST: M_cert={best['M_cert']:.5f} (kind={best['kind']}, N_QP={best['N_QP']})")
    print(f"  vs MV(N_QP=119): improvement TBD from results")
    print(f"  vs baseline 1.27481: {best['M_cert'] - 1.27481:+.5f}")
    print(f"  vs 2-scale @119 (1.29005): {best['M_cert'] - 1.29005:+.5f}")

    # Saturation: compute M_cert differences vs the largest N_QP for each kind
    def trend(rs):
        vals = [r for r in rs if r.get("M_cert") is not None]
        if len(vals) < 2:
            return []
        return [{"N_QP": v["N_QP"], "M_cert": float(v["M_cert"]),
                 "delta_vs_119": float(v["M_cert"]) - float(vals[0]["M_cert"])
                 if vals[0].get("M_cert") is not None else None}
                for v in vals]

    print()
    print("Arcsine trend (M_cert - M_cert@N_QP=119):")
    for t in trend(results["arcsine"]):
        print(f"  N_QP={t['N_QP']:>4}: M_cert={t['M_cert']:.5f}  delta={t['delta_vs_119']:+.6f}")
    print("2-scale trend (M_cert - M_cert@N_QP=119):")
    for t in trend(results["two_scale"]):
        print(f"  N_QP={t['N_QP']:>4}: M_cert={t['M_cert']:.5f}  delta={t['delta_vs_119']:+.6f}")

    out = {
        "NQP_list": NQP_LIST,
        "arcsine_params": {"deltas": arcsine_deltas, "lambdas": arcsine_lams},
        "two_scale_params": {"deltas": ms_deltas, "lambdas": ms_lams},
        "results": results,
        "best": best,
        "arcsine_trend": trend(results["arcsine"]),
        "two_scale_trend": trend(results["two_scale"]),
        "baseline_MV_119": 1.27481,
        "two_scale_119_reference": 1.29005,
        "improvement_vs_MV": best["M_cert"] - 1.27481,
        "improvement_vs_2scale_119": best["M_cert"] - 1.29005,
        "exceeded_1_290": bool(best["M_cert"] > 1.290),
    }
    with open("_M9_higher_nqp.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None if x is None else float(x))
    print()
    print("Wrote _M9_higher_nqp.json")


if __name__ == "__main__":
    main()
