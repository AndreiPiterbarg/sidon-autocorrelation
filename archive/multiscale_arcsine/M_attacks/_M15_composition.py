"""Agent M15: COMPOSITION (CONVOLUTION) KERNELS for the Sidon constant lower bound.

Option A: K = K_1 * K_2 (convolution), so K_hat = K_1_hat * K_2_hat (product).
Each K_i is K_arc(delta_i) with K_i_hat(xi) = J_0(pi * delta_i * xi)^2.
Support: K_i supported on [-delta_i, delta_i], so K = K_1*K_2 supported on
[-(delta_1+delta_2), (delta_1+delta_2)]. We need delta_1 + delta_2 <= DELTA = 0.138.

K_hat(0) = 1 * 1 = 1 (each J_0(0)^2 = 1), so K is already L^1-normalized.

Phase 1: 2-fold conv. K_hat = J_0(pi d1 xi)^2 * J_0(pi d2 xi)^2.
Phase 2: 3-fold conv. K_hat = prod_i J_0(pi d_i xi)^2 with sum d_i <= DELTA.
Phase 3: Convex combo of conv kernels: K = sum_j lambda_j K_j with each K_j a convolution.

Reuses pipeline: solve_QP, K_2_quad, M_cert from _K26_full_sweep_reopt.py.
Baseline to beat: 1.29005 (2-scale LINEAR combo arcsine).
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
BASELINE_LINEAR = 1.29005  # 2-scale linear combo arcsine breakthrough
BASELINE_MV = 1.27481


# ---------- Composition kernels: K_hat as product over scales ----------

def K_hat_conv(xi, deltas):
    """Convolution composition: K_hat(xi) = prod_i J_0(pi*delta_i*xi)^2.
    deltas: list of per-scale supports; sum(deltas) must be <= DELTA.
    """
    xi = np.asarray(xi, dtype=float)
    out = np.ones_like(xi, dtype=float) if xi.ndim else 1.0
    for d in deltas:
        out = out * (j0(np.pi * d * xi) ** 2)
    return out


def K_hat_combo_conv(xi, conv_groups, lambdas):
    """Convex combo of convolution kernels.
    conv_groups: list of lists; each inner list is deltas for one conv kernel.
    lambdas: list of mixing weights (sum to 1, all >= 0).
    K_hat(xi) = sum_j lambdas[j] * prod_i J_0(pi*conv_groups[j][i]*xi)^2.
    """
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi, dtype=float) if xi.ndim else 0.0
    for lam, deltas in zip(lambdas, conv_groups):
        out = out + lam * K_hat_conv(xi, deltas)
    return out


# ---------- QP and M_cert pipeline (reused from _K26_full_sweep_reopt.py) ----------

def solve_QP_from_w(w, n_grid=5001):
    """Solve QP given precomputed weights w[j-1] = K_hat(j/U), j=1..N_QP."""
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


def K_2_quad_from_fn(K_hat_fn):
    """Integrate K_hat^2 over (-inf, inf), exploiting symmetry."""
    def f(xi):
        return K_hat_fn(xi) ** 2
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
        mu_ = M * math.sin(math.pi / M) / math.pi
        y_star_sq = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        y_star = math.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + math.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float('inf')
        return M + 1 + 2 * mu_ * k_1 + math.sqrt(rad1 * rad2)

    try:
        return brentq(lambda M: sup_R(M) - target, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return None


def eval_K_hat_fn(K_hat_fn, tag=""):
    """Evaluate a K_hat function: build weights, solve QP, compute M_cert."""
    w = np.zeros(N_QP)
    for j in range(1, N_QP + 1):
        w[j - 1] = float(K_hat_fn(j / U))
    a_opt, S1, mG, status = solve_QP_from_w(w)
    if status != "ok":
        return {"tag": tag, "status": status}
    k1 = float(K_hat_fn(1.0))
    K2 = float(K_2_quad_from_fn(K_hat_fn))
    Mc = M_cert(k1, K2, S1, mG)
    return {"tag": tag, "k_1": k1, "K_2": K2, "S_1": S1, "min_G": mG,
            "M_cert": Mc, "status": status}


# ---------- Phase 1: 2-fold convolution sweep ----------

def phase1_two_fold():
    print("=" * 80)
    print("PHASE 1: 2-fold convolution K_hat = J_0(pi d1 xi)^2 * J_0(pi d2 xi)^2")
    print(f"Constraint: d1 + d2 <= {DELTA}")
    print("=" * 80)
    print(f"{'d_1':>8} {'d_2':>8} {'sum':>7} {'k_1':>8} {'K_2':>9} {'M_cert':>9} {'vs1.29005':>10}")
    print("-" * 80)

    # As listed in task: pairs with d1+d2 = 0.138 (or less)
    pairs = [
        (0.069, 0.069),
        (0.080, 0.058),
        (0.100, 0.038),
        (0.115, 0.023),
        (0.125, 0.013),
        (0.130, 0.008),
        (0.135, 0.003),
    ]
    # Also add some pairs with smaller sums (slack in support constraint)
    for s in [0.130, 0.120, 0.100, 0.080]:
        pairs.append((s * 0.5, s * 0.5))
    # And unbalanced with slack
    pairs.extend([
        (0.07, 0.05), (0.08, 0.04), (0.10, 0.03),
        (0.06, 0.06), (0.05, 0.05),
    ])

    results = []
    best = {"M_cert": -np.inf, "params": None}
    for (d1, d2) in pairs:
        s = d1 + d2
        if s > DELTA + 1e-12:
            continue
        K_hat_fn = lambda xi, d1=d1, d2=d2: K_hat_conv(xi, [d1, d2])
        r = eval_K_hat_fn(K_hat_fn, tag=f"2conv({d1:.3f},{d2:.3f})")
        Mc = r.get("M_cert")
        if Mc is not None:
            print(f"{d1:>8.4f} {d2:>8.4f} {s:>7.4f} {r['k_1']:>8.5f} {r['K_2']:>9.5f} "
                  f"{Mc:>9.5f} {Mc - BASELINE_LINEAR:>+10.5f}")
            results.append({"d1": d1, "d2": d2, "sum": s, **r})
            if Mc > best["M_cert"]:
                best = {"M_cert": float(Mc),
                        "params": {"d1": d1, "d2": d2, "k_1": r["k_1"],
                                   "K_2": r["K_2"], "S_1": r["S_1"], "min_G": r["min_G"]}}
        else:
            print(f"{d1:>8.4f} {d2:>8.4f} {s:>7.4f}   --- {r.get('status')} ---")
    return results, best


# ---------- Phase 2: 3-fold convolution sweep ----------

def phase2_three_fold():
    print()
    print("=" * 80)
    print("PHASE 2: 3-fold convolution K_hat = prod_i J_0(pi d_i xi)^2 for i=1,2,3")
    print(f"Constraint: d1+d2+d3 <= {DELTA}")
    print("=" * 80)
    print(f"{'d_1':>7} {'d_2':>7} {'d_3':>7} {'sum':>7} {'k_1':>8} {'K_2':>9} {'M_cert':>9}")
    print("-" * 78)

    triples = [
        (DELTA / 3, DELTA / 3, DELTA / 3),                      # equal split (~0.046, 0.046, 0.046)
        (0.05, 0.05, 0.038),
        (0.06, 0.04, 0.038),
        (0.07, 0.04, 0.028),
        (0.08, 0.04, 0.018),
        (0.09, 0.03, 0.018),
        (0.10, 0.03, 0.008),
        (0.05, 0.04, 0.03),
        (0.06, 0.05, 0.02),
        (0.07, 0.05, 0.018),
        (0.05, 0.05, 0.03),
        (0.04, 0.04, 0.04),    # equal split, sub-DELTA
        (0.03, 0.03, 0.03),
        (0.10, 0.025, 0.013),
        (0.115, 0.015, 0.008),
    ]

    results = []
    best = {"M_cert": -np.inf, "params": None}
    for (d1, d2, d3) in triples:
        s = d1 + d2 + d3
        if s > DELTA + 1e-12:
            continue
        K_hat_fn = lambda xi, ds=(d1, d2, d3): K_hat_conv(xi, ds)
        r = eval_K_hat_fn(K_hat_fn, tag=f"3conv({d1:.3f},{d2:.3f},{d3:.3f})")
        Mc = r.get("M_cert")
        if Mc is not None:
            print(f"{d1:>7.4f} {d2:>7.4f} {d3:>7.4f} {s:>7.4f} {r['k_1']:>8.5f} "
                  f"{r['K_2']:>9.5f} {Mc:>9.5f}")
            results.append({"d1": d1, "d2": d2, "d3": d3, "sum": s, **r})
            if Mc > best["M_cert"]:
                best = {"M_cert": float(Mc),
                        "params": {"d1": d1, "d2": d2, "d3": d3,
                                   "k_1": r["k_1"], "K_2": r["K_2"],
                                   "S_1": r["S_1"], "min_G": r["min_G"]}}
        else:
            print(f"{d1:>7.4f} {d2:>7.4f} {d3:>7.4f} {s:>7.4f}  --- {r.get('status')} ---")
    return results, best


# ---------- Phase 3: convex combos of conv kernels ----------

def phase3_combo_conv():
    print()
    print("=" * 80)
    print("PHASE 3: convex combo K = lam*K_conv1 + (1-lam)*K_conv2")
    print("=" * 80)
    print(f"{'group1':>22} {'group2':>22} {'lam':>5} {'M_cert':>9} {'vs1.29005':>10}")
    print("-" * 80)

    # Build a small set of convolution-based building blocks
    blocks = [
        ([DELTA], "1conv(0.138)"),                  # single arcsine (= MV-style)
        ([0.069, 0.069], "2conv(eq)"),              # symmetric 2-conv
        ([0.080, 0.058], "2conv(0.08,0.058)"),
        ([0.10, 0.038], "2conv(0.10,0.038)"),
        ([0.115, 0.023], "2conv(0.115,0.023)"),
        ([0.125, 0.013], "2conv(0.125,0.013)"),
        ([DELTA / 3] * 3, "3conv(eq)"),             # symmetric 3-conv
        ([0.05, 0.05, 0.038], "3conv(0.05,0.05,0.038)"),
    ]

    results = []
    best = {"M_cert": -np.inf, "params": None}
    lam_grid = [0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    for i, (g1, t1) in enumerate(blocks):
        for j, (g2, t2) in enumerate(blocks):
            if i >= j:
                continue
            for lam in lam_grid:
                K_hat_fn = lambda xi, g1=g1, g2=g2, lam=lam: \
                    K_hat_combo_conv(xi, [g1, g2], [lam, 1.0 - lam])
                r = eval_K_hat_fn(K_hat_fn, tag=f"combo({t1},{t2},lam={lam:.2f})")
                Mc = r.get("M_cert")
                if Mc is not None:
                    print(f"{t1:>22} {t2:>22} {lam:>5.2f} {Mc:>9.5f} "
                          f"{Mc - BASELINE_LINEAR:>+10.5f}")
                    results.append({"g1": g1, "g2": g2, "tag1": t1, "tag2": t2,
                                    "lam": lam, **r})
                    if Mc > best["M_cert"]:
                        best = {"M_cert": float(Mc),
                                "params": {"g1": g1, "g2": g2, "tag1": t1, "tag2": t2,
                                           "lam": lam, "k_1": r["k_1"], "K_2": r["K_2"],
                                           "S_1": r["S_1"], "min_G": r["min_G"]}}
    return results, best


# ---------- Refinement around the best 2-fold conv ----------

def refine_two_fold(best_p1):
    if best_p1["params"] is None:
        return [], best_p1
    print()
    print("=" * 80)
    print("REFINEMENT around best 2-fold convolution")
    print("=" * 80)
    print(f"{'d_1':>8} {'d_2':>8} {'sum':>7} {'k_1':>8} {'K_2':>9} {'M_cert':>9}")
    print("-" * 80)
    d1_0 = best_p1["params"]["d1"]
    d2_0 = best_p1["params"]["d2"]
    refined_d1 = np.linspace(max(0.005, d1_0 - 0.02), min(DELTA - 0.005, d1_0 + 0.02), 9)
    results = []
    best = dict(best_p1)
    for d1 in refined_d1:
        d2_max = DELTA - float(d1)
        if d2_max <= 0.001:
            continue
        d2_grid = np.linspace(0.005, d2_max, 9)
        for d2 in d2_grid:
            K_hat_fn = lambda xi, d1=float(d1), d2=float(d2): K_hat_conv(xi, [d1, d2])
            r = eval_K_hat_fn(K_hat_fn, tag=f"2conv-ref")
            Mc = r.get("M_cert")
            if Mc is not None:
                marker = " *" if Mc > best["M_cert"] else ""
                print(f"{d1:>8.5f} {d2:>8.5f} {d1+d2:>7.4f} {r['k_1']:>8.5f} "
                      f"{r['K_2']:>9.5f} {Mc:>9.5f}{marker}")
                results.append({"d1": float(d1), "d2": float(d2),
                                "sum": float(d1+d2), **r})
                if Mc > best["M_cert"]:
                    best = {"M_cert": float(Mc),
                            "params": {"d1": float(d1), "d2": float(d2),
                                       "k_1": r["k_1"], "K_2": r["K_2"],
                                       "S_1": r["S_1"], "min_G": r["min_G"]}}
    return results, best


def main():
    print("Agent M15: COMPOSITION (CONVOLUTION) kernels")
    print(f"Baseline to beat: LINEAR 2-scale = {BASELINE_LINEAR}, MV = {BASELINE_MV}")
    print()

    p1_results, p1_best = phase1_two_fold()
    p1_ref_results, p1_best = refine_two_fold(p1_best)
    p2_results, p2_best = phase2_three_fold()
    p3_results, p3_best = phase3_combo_conv()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Phase 1 (2-fold conv):           best M_cert = {p1_best['M_cert']:.5f}")
    print(f"  params: {p1_best['params']}")
    print(f"Phase 2 (3-fold conv):           best M_cert = {p2_best['M_cert']:.5f}")
    print(f"  params: {p2_best['params']}")
    print(f"Phase 3 (combo of conv kernels): best M_cert = {p3_best['M_cert']:.5f}")
    print(f"  params: {p3_best['params']}")
    overall_M = max(p1_best['M_cert'], p2_best['M_cert'], p3_best['M_cert'])
    print()
    print(f"OVERALL BEST M_cert = {overall_M:.5f}")
    print(f"  vs LINEAR 2-scale baseline 1.29005:  {overall_M - BASELINE_LINEAR:+.5f}")
    print(f"  vs MV baseline           1.27481:    {overall_M - BASELINE_MV:+.5f}")

    out = {
        "baseline_linear": BASELINE_LINEAR,
        "baseline_MV": BASELINE_MV,
        "phase1_best": p1_best,
        "phase2_best": p2_best,
        "phase3_best": p3_best,
        "overall_best_M_cert": overall_M,
        "delta_vs_linear": overall_M - BASELINE_LINEAR,
        "delta_vs_MV": overall_M - BASELINE_MV,
        "phase1_results": p1_results,
        "phase1_refinement": p1_ref_results,
        "phase2_results": p2_results,
        "phase3_results": p3_results,
    }
    with open("_M15_composition_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("Wrote _M15_composition_results.json")


if __name__ == "__main__":
    main()
