"""Master: joint (G, delta_2, lambda_1) optimization for K26 multi-scale arcsine kernel.

The QP (variables a_1,..,a_N, weights w_j = K_hat(j/u)):

    minimize   S_1 = sum_{j=1..N} a_j^2 / w_j
    subject to G(x) = sum a_j cos(2 pi j x / u) >= 1   on  [0, 1/4]
               (discretised on fine grid; verified post hoc).

After solving, M_cert is recovered from MV's master inequality using
(k_1, K_2, S_1, min_G).  The outer loop maximises M_cert over (delta_2, lambda_1)
on a grid (with refinement).

Outputs: _master_k26_reopt_results.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq
import cvxpy as cp

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

DELTA = 0.138
U = 0.5 + DELTA
MV_BASELINE = 1.27481
N_QP_MV = 119


# ---------- multi-scale K_hat ----------

def K_hat_ms(xi, deltas, lambdas):
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


# ---------- K_2 via quad ----------

def K_2_quad(deltas, lambdas, xi_split=10.0):
    def f(xi):
        return float(K_hat_ms(np.array([xi]), deltas, lambdas)[0]) ** 2
    v1, _ = quad(f, 0.0, xi_split, limit=400, epsabs=1e-13, epsrel=1e-11)
    v2, _ = quad(f, xi_split, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)
    return 2.0 * (v1 + v2)


# ---------- MV M_cert ----------

def M_cert_from(k_1, K_2, S_1, min_G):
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
        y_star = math.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + math.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float('inf')
        return M + 1 + 2 * mu_ * k_1 + math.sqrt(rad1 * rad2)

    try:
        f_lo = sup_R(1.0 + 1e-10) - target
        f_hi = sup_R(2.0) - target
        if f_lo >= 0 or f_hi <= 0:
            return None
        return brentq(lambda M: sup_R(M) - target, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return None


# ---------- QP solver ----------

def solve_QP(deltas, lambdas, N=119, n_grid=5001, solver_order=("MOSEK", "CLARABEL", "SCS")):
    """Solve QP for given K (deltas, lambdas) and N coefficients.

    Returns (a_opt, S_1, min_G_grid, w, status)
    """
    qp_xi = np.arange(1, N + 1) / U
    w = K_hat_ms(qp_xi, deltas, lambdas)
    if w.min() <= 1e-30:
        return None, None, None, None, "non-positive weight"

    xs = np.linspace(0.0, 0.25, n_grid)
    # B[i,j] = cos(2 pi (j+1) xs[i] / U)
    B = np.cos(2.0 * math.pi * np.arange(1, N + 1)[None, :] * xs[:, None] / U)

    a = cp.Variable(N)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)

    last_err = None
    for sname in solver_order:
        try:
            prob.solve(solver=sname, verbose=False)
            if a.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                a_opt = np.asarray(a.value).flatten()
                S_1 = float(np.sum(a_opt ** 2 / w))
                min_G_grid = float((B @ a_opt).min())
                return a_opt, S_1, min_G_grid, w, f"ok({sname},{prob.status})"
        except Exception as e:
            last_err = e
            continue
    return None, None, None, None, f"QP failed: {last_err}"


# ---------- Verification on fine grid ----------

def verify_min_G(a_opt, n_grid_fine=200001):
    """Recompute min G(x) on a finer grid for verification.
    """
    N = len(a_opt)
    xs = np.linspace(0.0, 0.25, n_grid_fine)
    G = np.zeros_like(xs)
    # batch to control memory
    batch = 1000
    for s in range(0, N, batch):
        js = np.arange(s + 1, min(s + batch, N) + 1)
        B = np.cos(2.0 * math.pi * js[None, :] * xs[:, None] / U)
        G = G + B @ a_opt[s:s + len(js)]
    return float(G.min()), float(G.max())


# ---------- MV-coeffs baseline ----------

def load_mv_coeffs():
    try:
        from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
        return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])
    except Exception as e:
        print(f"  could not load MV coeffs: {e}")
        return None


# ---------- evaluate at a (deltas, lambdas) ----------

def evaluate_point(deltas, lambdas, N=119, verbose=False):
    a_opt, S_1, min_G_grid, w, status = solve_QP(deltas, lambdas, N=N)
    if "ok" not in status:
        return {"status": status}
    k_1 = float(K_hat_ms(np.array([1.0]), deltas, lambdas)[0])
    K_2 = K_2_quad(deltas, lambdas)
    Mc = M_cert_from(k_1, K_2, S_1, min_G_grid)
    out = {
        "deltas": [float(d) for d in deltas],
        "lambdas": [float(l) for l in lambdas],
        "N": N,
        "k_1": k_1, "K_2": float(K_2),
        "S_1": S_1, "min_G_grid": min_G_grid,
        "M_cert": Mc,
        "status": status,
    }
    if verbose:
        Mtxt = f"{Mc:.5f}" if Mc is not None else "None"
        print(f"  d={list(deltas)} l={list(lambdas)} N={N}: "
              f"k_1={k_1:.5f} K_2={K_2:.4f} S_1={S_1:.3f} "
              f"min_G={min_G_grid:.5f} M_cert={Mtxt}")
    return out, a_opt


# ---------- sanity check: MV at (DELTA, lambda_1=1.0) ----------

def sanity_mv_reproduction():
    print("\n=== SANITY 1: reoptimize at (DELTA, lambda_1=1.0) ===")
    print("(MV's published a_j should be close to reoptimized a_j; M_cert ~ 1.2748)")
    r, a_opt = evaluate_point([DELTA], [1.0], N=119, verbose=True)
    mv = load_mv_coeffs()
    if mv is not None:
        # Compute S_1 from MV's coeffs at this same K
        w = K_hat_ms(np.arange(1, 120) / U, [DELTA], [1.0])
        S1_mv = float(np.sum(mv ** 2 / w))
        # Coefficient agreement
        a_diff = np.linalg.norm(a_opt - mv) / np.linalg.norm(mv)
        print(f"  MV S_1 (their coefs) = {S1_mv:.4f}")
        print(f"  reopt S_1            = {r['S_1']:.4f}")
        print(f"  relative coef diff    = {a_diff:.4f}")
        print(f"  reopt S_1 should be <= MV S_1 (it is the QP optimum)")
        # M_cert from MV coefs
        Mc_mv = M_cert_from(r["k_1"], r["K_2"], S1_mv, 1.0)
        print(f"  M_cert(MV)   = {Mc_mv:.6f}")
        print(f"  M_cert(reopt)= {r['M_cert']:.6f}")
    return r


# ---------- joint sweep ----------

def joint_sweep(N=119, level="full"):
    """Run joint (delta_2, lambda_1) sweep.

    level: 'small' (3x3, quick test), 'full' (broad coarse), 'fine' (refined+broad).
    """
    if level == "small":
        delta_2_list = [0.04, 0.05, 0.07]
        lambda_1_list = [0.80, 0.85, 0.90]
    elif level == "full":
        delta_2_list = [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
                        0.055, 0.060, 0.070, 0.080, 0.090, 0.100, 0.115, 0.130]
        lambda_1_list = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85,
                         0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    elif level == "fine":
        # Around earlier best (0.045, 0.85)
        delta_2_list = list(np.round(np.linspace(0.03, 0.06, 13), 5))
        lambda_1_list = list(np.round(np.linspace(0.70, 0.92, 12), 4))
    else:
        raise ValueError(level)

    results = []
    best = {"M_cert": -np.inf}
    print(f"\n=== Sweep level={level}  N={N}  "
          f"({len(delta_2_list)}x{len(lambda_1_list)}={len(delta_2_list)*len(lambda_1_list)} points) ===")
    t0 = time.time()
    for d2 in delta_2_list:
        for l1 in lambda_1_list:
            t1 = time.time()
            res, _ = evaluate_point([DELTA, float(d2)], [float(l1), 1.0 - float(l1)], N=N)
            dt = time.time() - t1
            Mc = res.get("M_cert")
            if Mc is None:
                continue
            results.append({"delta_2": float(d2), "lambda_1": float(l1),
                            "k_1": res["k_1"], "K_2": res["K_2"],
                            "S_1": res["S_1"], "min_G_grid": res["min_G_grid"],
                            "M_cert": Mc, "dt_sec": dt})
            if Mc > best["M_cert"]:
                best = {"M_cert": float(Mc), "delta_2": float(d2),
                        "lambda_1": float(l1),
                        "k_1": res["k_1"], "K_2": res["K_2"],
                        "S_1": res["S_1"], "min_G_grid": res["min_G_grid"]}
                print(f"  NEW BEST M={Mc:.5f} at d2={d2:.4f} l1={l1:.4f} "
                      f"S1={res['S_1']:.2f} ({dt:.1f}s)")
    print(f"  total {time.time()-t0:.1f}s, best M_cert = {best['M_cert']:.5f}")
    return results, best


# ---------- main ----------

def main():
    out = {"DELTA": DELTA, "MV_BASELINE": MV_BASELINE}

    # 0) Sanity check
    r_sanity = sanity_mv_reproduction()
    out["sanity_DELTA_lambda1_1"] = r_sanity

    # 1) Small sweep first (3x3) to confirm pipeline
    print("\n=== Step 1: small (3x3) test sweep, N=119 ===")
    small_res, small_best = joint_sweep(N=119, level="small")
    out["small_sweep"] = {"results": small_res, "best": small_best}

    # 2) Full sweep N=119
    print("\n=== Step 2: full broad sweep, N=119 ===")
    full_res, full_best = joint_sweep(N=119, level="full")
    out["full_sweep_N119"] = {"results": full_res, "best": full_best}

    # 3) Refined sweep around current best
    d2_0 = full_best.get("delta_2")
    l1_0 = full_best.get("lambda_1")
    if d2_0 is not None and l1_0 is not None:
        print(f"\n=== Step 3: refined sweep around ({d2_0:.4f}, {l1_0:.4f}), N=119 ===")
        d2_grid = list(np.round(np.linspace(max(0.01, d2_0 - 0.015),
                                            min(DELTA - 1e-4, d2_0 + 0.015), 11), 5))
        l1_grid = list(np.round(np.linspace(max(0.50, l1_0 - 0.08),
                                            min(0.99, l1_0 + 0.08), 11), 4))
        refined_res = []
        refined_best = full_best
        for d2 in d2_grid:
            for l1 in l1_grid:
                res, _ = evaluate_point([DELTA, float(d2)], [float(l1), 1.0 - float(l1)],
                                        N=119)
                Mc = res.get("M_cert")
                if Mc is None:
                    continue
                refined_res.append({"delta_2": float(d2), "lambda_1": float(l1),
                                    "k_1": res["k_1"], "K_2": res["K_2"],
                                    "S_1": res["S_1"],
                                    "min_G_grid": res["min_G_grid"],
                                    "M_cert": Mc})
                if Mc > refined_best["M_cert"]:
                    refined_best = {"M_cert": float(Mc), "delta_2": float(d2),
                                    "lambda_1": float(l1),
                                    "k_1": res["k_1"], "K_2": res["K_2"],
                                    "S_1": res["S_1"],
                                    "min_G_grid": res["min_G_grid"]}
                    print(f"  refined NEW BEST M={Mc:.5f} at d2={d2:.4f} l1={l1:.4f}")
        out["refined_N119"] = {"results": refined_res, "best": refined_best}
    else:
        refined_best = full_best

    best_so_far = refined_best
    print(f"\nBest after N=119 sweeps: M_cert = {best_so_far['M_cert']:.5f}")

    # 4) Probe N>119 at best point
    print("\n=== Step 4: increase QP dimension N at best (delta_2, lambda_1) ===")
    d2_best = best_so_far["delta_2"]
    l1_best = best_so_far["lambda_1"]
    N_probe = {}
    for N_try in [119, 150, 200, 300, 500]:
        try:
            res, _ = evaluate_point([DELTA, d2_best], [l1_best, 1.0 - l1_best],
                                    N=N_try, verbose=True)
            Mc = res.get("M_cert")
            if Mc is None:
                print(f"  N={N_try}: failed status={res.get('status')}")
                continue
            N_probe[N_try] = {"k_1": res["k_1"], "K_2": res["K_2"],
                              "S_1": res["S_1"],
                              "min_G_grid": res["min_G_grid"],
                              "M_cert": Mc, "status": res.get("status")}
        except Exception as e:
            print(f"  N={N_try} exception: {e}")
            N_probe[N_try] = {"error": str(e)}
    out["N_probe_at_best"] = N_probe

    # 5) If higher N helps, joint sweep at that N (refined window only)
    Mc_119 = N_probe.get(119, {}).get("M_cert")
    Mc_200 = N_probe.get(200, {}).get("M_cert")
    if Mc_119 is not None and Mc_200 is not None and Mc_200 > Mc_119 + 1e-5:
        print("\n=== Step 5: refined sweep at N=200 ===")
        d2_grid = list(np.round(np.linspace(max(0.01, d2_best - 0.015),
                                            min(DELTA - 1e-4, d2_best + 0.015), 9), 5))
        l1_grid = list(np.round(np.linspace(max(0.50, l1_best - 0.08),
                                            min(0.99, l1_best + 0.08), 9), 4))
        rN200_res = []
        rN200_best = {"M_cert": Mc_200, "delta_2": d2_best, "lambda_1": l1_best}
        for d2 in d2_grid:
            for l1 in l1_grid:
                res, _ = evaluate_point([DELTA, float(d2)],
                                        [float(l1), 1.0 - float(l1)], N=200)
                Mc = res.get("M_cert")
                if Mc is None:
                    continue
                rN200_res.append({"delta_2": float(d2), "lambda_1": float(l1),
                                  "k_1": res["k_1"], "K_2": res["K_2"],
                                  "S_1": res["S_1"],
                                  "min_G_grid": res["min_G_grid"],
                                  "M_cert": Mc})
                if Mc > rN200_best["M_cert"]:
                    rN200_best = {"M_cert": float(Mc), "delta_2": float(d2),
                                  "lambda_1": float(l1),
                                  "k_1": res["k_1"], "K_2": res["K_2"],
                                  "S_1": res["S_1"],
                                  "min_G_grid": res["min_G_grid"]}
                    print(f"  N=200 NEW BEST M={Mc:.5f} d2={d2:.4f} l1={l1:.4f}")
        out["N200_refined_sweep"] = {"results": rN200_res, "best": rN200_best}
        if rN200_best["M_cert"] > best_so_far["M_cert"]:
            best_so_far = rN200_best
    else:
        print("\n  N=200 does not improve over N=119; skipping N=200 refined sweep")

    # 6) Verify min_G on fine grid at the very best
    print("\n=== Step 6: fine-grid verification of min_G at overall best ===")
    res_best, a_best = evaluate_point([DELTA, best_so_far["delta_2"]],
                                      [best_so_far["lambda_1"],
                                       1.0 - best_so_far["lambda_1"]],
                                      N=N_QP_MV)
    min_G_fine, max_G_fine = verify_min_G(a_best, n_grid_fine=200001)
    print(f"  min_G on coarse grid (5001): {res_best['min_G_grid']:.7f}")
    print(f"  min_G on fine   grid (200001): {min_G_fine:.7f}")
    print(f"  max_G on fine grid: {max_G_fine:.5f}")
    out["fine_grid_verification"] = {
        "min_G_grid_5001": res_best["min_G_grid"],
        "min_G_grid_200001": min_G_fine,
        "max_G_grid_200001": max_G_fine,
    }
    # Re-evaluate M_cert with fine-grid min_G
    Mc_finegrid = M_cert_from(res_best["k_1"], res_best["K_2"],
                              res_best["S_1"], min_G_fine)
    print(f"  M_cert with fine-grid min_G  = {Mc_finegrid}")
    out["M_cert_with_fine_min_G"] = Mc_finegrid

    out["final_best"] = best_so_far
    out["improvement_over_MV_1_2748"] = (
        best_so_far["M_cert"] - MV_BASELINE if best_so_far["M_cert"] is not None else None
    )

    outpath = os.path.join(REPO, "_master_k26_reopt_results.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nWrote {outpath}")
    print(f"\n{'='*70}")
    print(f"FINAL M_cert = {best_so_far['M_cert']:.6f}")
    print(f"  delta_2 = {best_so_far['delta_2']:.5f}")
    print(f"  lambda_1 = {best_so_far['lambda_1']:.5f}")
    print(f"  improvement over MV (1.27481) = "
          f"{best_so_far['M_cert'] - MV_BASELINE:+.5f}")
    print(f"{'='*70}")
    return out


if __name__ == "__main__":
    main()
