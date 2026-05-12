"""Master K26 large-N study.

For the K26 multi-scale arcsine kernel
    K_hat(xi) = lam_1 * J_0(pi*d_1*xi)^2 + (1-lam_1) * J_0(pi*d_2*xi)^2,
we re-optimise the cosine polynomial
    G(x) = sum_{j=1..N} a_j cos(2 pi j x / u),
minimising S_1(N) = sum_j a_j^2 / K_hat(j/u) subject to min_{[0,1/4]} G >= 1.

For each N in {119, 200, 500, 1000} we

  (a) solve the QP at the best K26 K (d_2 = 0.045, lam_1 = 0.85) and
      compute M_cert from MV's master inequality (with the gain term
      using the actual min_G grid value).
  (b) optionally re-tune (d_2, lam_1) jointly with the new N to see if
      the optimum shifts.

The objective in the QP is quadratic in `a`, the constraints are linear
on a fine grid in [0, 1/4]; we use cvxpy with MOSEK preferred.

Output: ``_master_k26_large_N.json``.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import cvxpy as cp
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0

DELTA = 0.138
U = 0.5 + DELTA  # 0.638
MV_BASELINE = 1.27481

# Best K26 from the existing full sweep (_K26_full_sweep_reopt_result.json):
#   delta_2 = 0.045, lambda_1 = 0.85
BEST_K26 = {"delta_2": 0.045, "lambda_1": 0.85}


def K_hat_ms(xi, deltas, lambdas):
    """K_hat(xi) = sum_i lambdas[i] * J_0(pi * deltas[i] * xi)^2 >= 0."""
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi) if xi.ndim else np.float64(0.0)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


def K_2_quad(deltas, lambdas):
    """K_2 = int_R K_hat(xi)^2 dxi (Parseval, finite)."""
    def f(xi):
        return float(K_hat_ms(np.array([xi]), deltas, lambdas)[0]) ** 2
    v1, _ = quad(f, 0.0, 20.0, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, 20.0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2)


def M_cert_master(k_1, K_2, S_1, min_G):
    """MV master inequality with the supplied gain term."""
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    a_gain = (4.0 / U) * (min_G ** 2) / S_1
    target = 2.0 / U + a_gain
    rad2 = K_2 - 1 - 2 * k_1 * k_1

    def sup_R(M):
        if M <= 1.0:
            return float("-inf")
        mu_ = M * np.sin(np.pi / M) / np.pi
        y_star_sq = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        y_star = np.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + np.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float("inf")
        return M + 1 + 2 * mu_ * k_1 + np.sqrt(rad1 * rad2)

    try:
        return brentq(lambda M: sup_R(M) - target, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return None


def solve_QP_N(N, deltas, lambdas, n_grid=None, verbose=False):
    """Solve the QP min sum a_j^2/w_j s.t. G(x) >= 1 on grid in [0,1/4].

    Returns (a_opt, S_1, min_G, w, status, time_s).
    The grid resolution scales with N (the highest cosine frequency is
    N/u, so we want at least ~6N grid points per [0,1/4]).
    """
    if n_grid is None:
        # ~10 samples per period at the highest frequency on [0, 1/4]
        # period = u/N, so number of periods in [0, 1/4] is N/(4u) ~ N/2.55.
        # 10x oversample -> ~4 N points.  Cap at 30000 for memory.
        n_grid = max(5001, min(30000, 4 * N + 1))
    t0 = time.time()
    js = np.arange(1, N + 1)
    w = K_hat_ms(js / U, deltas, lambdas)
    if np.min(w) <= 0:
        return None, None, None, None, f"non-positive K_hat (min={float(np.min(w)):.3e})", time.time() - t0
    xs = np.linspace(0.0, 0.25, n_grid)
    # B[i, j-1] = cos(2 pi j x_i / u)
    B = np.cos(2.0 * math.pi * np.outer(xs, js) / U)
    a = cp.Variable(N)
    # Objective: sum a_j^2 / w_j  (convex quadratic)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    solver_used = None
    for s_name in ("MOSEK", "CLARABEL", "SCS"):
        try:
            kwargs = {"verbose": False}
            if s_name == "SCS":
                kwargs["eps"] = 1e-10
                kwargs["max_iters"] = 200000
            prob.solve(solver=s_name, **kwargs)
            if a.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                solver_used = s_name
                break
        except Exception as e:
            if verbose:
                print(f"    [solver {s_name} fail: {e}]")
            continue
    if a.value is None:
        return None, None, None, w, f"QP failed (last status={prob.status})", time.time() - t0
    a_opt = np.asarray(a.value).flatten()
    S_1 = float(np.sum(a_opt ** 2 / w))
    Gvals = B @ a_opt
    min_G = float(Gvals.min())
    elapsed = time.time() - t0
    if verbose:
        print(f"    [N={N}, solver={solver_used}, S_1={S_1:.4f}, "
              f"min_G={min_G:.6f}, time={elapsed:.1f}s]")
    return a_opt, S_1, min_G, w, "ok", elapsed


def refine_min_G(a_opt, N, n_grid_refine=200001):
    """Tighten min_G by a denser grid evaluation (no new QP)."""
    js = np.arange(1, N + 1)
    xs = np.linspace(0.0, 0.25, n_grid_refine)
    # Compute Gvals = sum_j a_j cos(2 pi j x / u) on the fine grid in
    # blocks to limit memory.
    Gvals = np.zeros_like(xs)
    blk = 2000
    for s in range(0, n_grid_refine, blk):
        chunk = xs[s:s + blk]
        Gvals[s:s + blk] = (np.cos(2.0 * math.pi * np.outer(chunk, js) / U) @ a_opt)
    return float(Gvals.min())


def eval_K_at_N(N, deltas, lambdas, label, k1=None, K2=None, verbose=True):
    """Solve QP at this N and return the result dict (including M_cert)."""
    deltas = np.asarray(deltas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    a_opt, S_1, min_G_grid, w, status, t_qp = solve_QP_N(N, deltas, lambdas, verbose=verbose)
    if status != "ok":
        return {"label": label, "N": N, "status": status, "M_cert": None,
                "time_qp": t_qp}
    # Tighten min_G on a finer grid
    min_G_fine = refine_min_G(a_opt, N, n_grid_refine=200001)
    if k1 is None:
        k1 = float(K_hat_ms(np.array([1.0]), deltas, lambdas)[0])
    if K2 is None:
        K2 = K_2_quad(deltas, lambdas)
    Mc = M_cert_master(k1, K2, S_1, min_G_fine)
    res = {
        "label": label,
        "N": N,
        "delta_1": float(deltas[0]),
        "delta_2": float(deltas[1]) if len(deltas) > 1 else None,
        "lambda_1": float(lambdas[0]),
        "k_1": float(k1),
        "K_2": float(K2),
        "S_1": float(S_1),
        "min_G_grid": float(min_G_grid),
        "min_G_fine": float(min_G_fine),
        "M_cert": Mc,
        "time_qp": float(t_qp),
        "beats_MV": Mc is not None and Mc > MV_BASELINE,
        "status": status,
    }
    if verbose:
        Mtxt = f"{Mc:.6f}" if Mc is not None else "None"
        delta_M = (Mc - MV_BASELINE) if Mc is not None else float('nan')
        print(f"  [{label}, N={N}] S_1={S_1:.4f} min_G={min_G_fine:.6f} "
              f"M_cert={Mtxt}  (d vs MV = {delta_M:+.5f}, QP {t_qp:.1f}s)")
    return res


def joint_refine_at_N(N, d2_grid, l1_grid, k_cache=None):
    """Sweep (d_2, lambda_1) at fixed N and find the best M_cert."""
    if k_cache is None:
        k_cache = {}
    results = []
    best = {"M_cert": -np.inf}
    for d2 in d2_grid:
        for l1 in l1_grid:
            key = (round(d2, 6), round(l1, 6))
            if key in k_cache:
                k1, K2 = k_cache[key]
            else:
                k1 = float(K_hat_ms(np.array([1.0]), [DELTA, d2], [l1, 1 - l1])[0])
                K2 = K_2_quad([DELTA, d2], [l1, 1 - l1])
                k_cache[key] = (k1, K2)
            label = f"joint-N{N}-d2{d2:.4f}-l1{l1:.3f}"
            r = eval_K_at_N(N, [DELTA, d2], [l1, 1 - l1], label,
                            k1=k1, K2=K2, verbose=False)
            results.append(r)
            if r["M_cert"] is not None and r["M_cert"] > best["M_cert"]:
                best = {"M_cert": float(r["M_cert"]), "delta_2": float(d2),
                        "lambda_1": float(l1), "N": N,
                        "S_1": r["S_1"], "min_G_fine": r["min_G_fine"],
                        "k_1": r["k_1"], "K_2": r["K_2"]}
    return results, best, k_cache


def main():
    print("=" * 80)
    print("Master K26 large-N study")
    print(f"  K26 best: d_1={DELTA}, d_2={BEST_K26['delta_2']}, "
          f"lambda_1={BEST_K26['lambda_1']}")
    print(f"  MV baseline M_cert (G=MV-119, K=pure arcsine) = {MV_BASELINE}")
    print("=" * 80)

    d2 = BEST_K26["delta_2"]
    l1 = BEST_K26["lambda_1"]
    deltas = [DELTA, d2]
    lambdas = [l1, 1.0 - l1]

    k1 = float(K_hat_ms(np.array([1.0]), deltas, lambdas)[0])
    K2 = K_2_quad(deltas, lambdas)
    print(f"\n  k_1 = {k1:.6f}   K_2 = {K2:.6f}")

    # ---------------------------------------------------------------- (1)
    # Step (1): M_cert vs N at the fixed K26-best K.
    Ns = [119, 200, 500, 1000]
    print("\n--- (1) M_cert vs N at fixed K26-best K ---")
    per_N = []
    for N in Ns:
        r = eval_K_at_N(N, deltas, lambdas, f"K26@N={N}", k1=k1, K2=K2)
        per_N.append(r)

    # Convergence estimate: fit S_1(N) ~ S_inf + C * N^(-p) for the last three.
    s_vals = [r["S_1"] for r in per_N if r["M_cert"] is not None]
    n_vals = [r["N"] for r in per_N if r["M_cert"] is not None]
    s_inf_est = None
    if len(s_vals) >= 3:
        # Use the last two pairs to project (Aitken-like geometric extrapolation)
        # Assume S_1(N) = S_inf + C / N (or N^p); try Richardson on last 3:
        # x1=N1, x2=N2, x3=N3, S1, S2, S3.
        x1, x2, x3 = n_vals[-3:]
        S1n, S2n, S3n = s_vals[-3:]
        # Solve S = S_inf + C / N^p numerically (3 unknowns, 3 eqs).
        # If powers don't matter much we approximate via p = 1 Richardson:
        # S_inf ~ (N2*S2 - N1*S1) / (N2 - N1)  using nearest pair to inf.
        s_inf_richardson_12 = (x2 * S2n - x1 * S1n) / (x2 - x1)
        s_inf_richardson_23 = (x3 * S3n - x2 * S2n) / (x3 - x2)
        s_inf_est = {
            "richardson_12_via_pair": float(s_inf_richardson_12),
            "richardson_23_via_pair": float(s_inf_richardson_23),
            "pair_12": [int(x1), int(x2), float(S1n), float(S2n)],
            "pair_23": [int(x2), int(x3), float(S2n), float(S3n)],
        }

    # Project M_cert at S_1 = s_inf_richardson_23 assuming min_G -> 1 (it
    # already is essentially 1 in MV's QP).  This is a numerical proxy of
    # the "N -> infty" lift.
    M_inf_est = None
    if s_inf_est is not None:
        S_inf_proxy = s_inf_est["richardson_23_via_pair"]
        M_inf_est = M_cert_master(k1, K2, S_inf_proxy, 1.0)
        print(f"  S_1(N->inf) ~ {S_inf_proxy:.4f}  ->  "
              f"M_cert(N->inf, K26) ~ {M_inf_est}")

    # ---------------------------------------------------------------- (5)
    # Step (5): joint (G, K) refinement at the largest tractable N.
    print("\n--- (5) Joint (G, K) refinement at large N ---")
    # Tight grid around (d_2 = 0.045, lambda_1 = 0.85).
    d2_grid = [0.03, 0.04, 0.045, 0.05, 0.055, 0.06]
    l1_grid = [0.80, 0.83, 0.85, 0.87, 0.90, 0.92]
    # Use N = 200 for the joint sweep (good lift, fast).
    N_joint = 200
    print(f"  N = {N_joint},  |d2_grid|x|l1_grid| = "
          f"{len(d2_grid)}x{len(l1_grid)} = {len(d2_grid)*len(l1_grid)}")
    joint_results, joint_best, _ = joint_refine_at_N(N_joint, d2_grid, l1_grid)
    print(f"  joint best at N={N_joint}: M_cert={joint_best['M_cert']:.6f} "
          f"at d_2={joint_best['delta_2']}, lambda_1={joint_best['lambda_1']}")

    # And one tighter pass with the local best at N=500 (one solve).
    print("\n--- (5b) Confirm joint best at N=500 ---")
    r500_at_joint = eval_K_at_N(
        500, [DELTA, joint_best["delta_2"]],
        [joint_best["lambda_1"], 1.0 - joint_best["lambda_1"]],
        f"joint-N500-d2{joint_best['delta_2']:.4f}-l1{joint_best['lambda_1']:.3f}")

    # ---------------------------------------------------------------- summary
    summary = {
        "kernel": {"delta_1": DELTA, "delta_2": BEST_K26["delta_2"],
                   "lambda_1": BEST_K26["lambda_1"], "k_1": k1, "K_2": K2},
        "MV_baseline": MV_BASELINE,
        "per_N_fixed_K": per_N,
        "S_inf_estimates": s_inf_est,
        "M_cert_N_inf_proxy": M_inf_est,
        "joint_grid": {"N": N_joint, "d2_grid": d2_grid, "l1_grid": l1_grid,
                       "results": joint_results, "best": joint_best},
        "joint_best_at_N500": r500_at_joint,
        "best_overall": None,
    }

    # Best overall M_cert across all evaluations.
    best_overall = {"M_cert": -np.inf}
    for r in per_N + joint_results + [r500_at_joint]:
        if r.get("M_cert") is not None and r["M_cert"] > best_overall["M_cert"]:
            best_overall = {
                "M_cert": float(r["M_cert"]),
                "label": r.get("label"),
                "N": r.get("N"),
                "delta_2": r.get("delta_2"),
                "lambda_1": r.get("lambda_1"),
                "S_1": r.get("S_1"),
                "min_G_fine": r.get("min_G_fine"),
                "k_1": r.get("k_1"),
                "K_2": r.get("K_2"),
            }
    summary["best_overall"] = best_overall
    print("\n" + "=" * 80)
    print("BEST OVERALL")
    print(f"  M_cert = {best_overall['M_cert']:.6f}  "
          f"(d vs MV={best_overall['M_cert']-MV_BASELINE:+.5f})")
    print(f"  Params: {best_overall}")
    print("=" * 80)

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_master_k26_large_N.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nWrote {outpath}")


if __name__ == "__main__":
    main()
