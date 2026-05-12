"""Agent M11: Frequency-tuned delta_i selection via min-max LP.

Hypothesis: Choose delta_i to make K_hat(j/U) maximally uniform over j=1..119
(MV's QP frequencies). Solve:

    max_{lambda} min_{j in 1..119} sum_i lambda_i * J_0(pi * delta_i * j/U)^2
    s.t.  sum lambda_i = 1, lambda_i >= 0

over a fine candidate grid of delta values. Compare to the 2-scale baseline
M_cert = 1.29005.

Phase 2: Use the LP-optimal support (active deltas + weights) as a seed for
differential evolution to refine M_cert.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.special import j0
from scipy.optimize import differential_evolution
import cvxpy as cp

from _K26_full_sweep_reopt import (
    eval_at,
    K_hat_ms,
    DELTA,
    U,
    N_QP,
)

DELTA_MAX = DELTA  # 0.138
BASELINE_K26 = 1.29005  # 2-scale arcsine (delta_1=0.138, delta_2=0.045, lambda_1=0.85)
MV_BASE = 1.27481


def build_freq_matrix(d_grid):
    """A[i, j-1] = J_0(pi * d_grid[i] * (j/U))**2 for j=1..N_QP."""
    js = np.arange(1, N_QP + 1, dtype=float)
    xi = js / U  # shape (119,)
    # outer product: rows = deltas, cols = freqs
    arg = np.pi * np.outer(d_grid, xi)  # shape (G, 119)
    return j0(arg) ** 2


def solve_minmax_lp(d_grid, A):
    """Solve max t s.t. sum_i lam_i * A[i, j] >= t for all j, sum lam = 1, lam >= 0.

    Returns (lam_opt (length G), t_opt).
    """
    G = len(d_grid)
    lam = cp.Variable(G, nonneg=True)
    t = cp.Variable()
    cons = [cp.sum(lam) == 1.0]
    # A.T has shape (119, G); A.T @ lam is the min_j argument
    cons.append(A.T @ lam >= t)
    prob = cp.Problem(cp.Maximize(t), cons)
    try:
        prob.solve(solver="MOSEK", verbose=False)
        if lam.value is None:
            prob.solve(solver="CLARABEL", verbose=False)
    except Exception:
        prob.solve(solver="CLARABEL", verbose=False)
    return np.asarray(lam.value).flatten(), float(t.value)


def extract_active(d_grid, lam_opt, tol=1e-6):
    """Extract (deltas, lambdas) where lambda > tol; renormalize."""
    mask = lam_opt > tol
    deltas = list(d_grid[mask])
    lams = list(lam_opt[mask])
    s = sum(lams)
    lams = [l / s for l in lams]
    return deltas, lams


def cluster_deltas(deltas, lams, gap=0.001):
    """Cluster adjacent deltas (within `gap`) and sum their weights."""
    if not deltas:
        return deltas, lams
    order = np.argsort(deltas)
    d_sorted = [deltas[i] for i in order]
    l_sorted = [lams[i] for i in order]
    out_d = [d_sorted[0]]
    out_l = [l_sorted[0]]
    for d, l in zip(d_sorted[1:], l_sorted[1:]):
        if d - out_d[-1] < gap:
            # weighted average for delta
            tot = out_l[-1] + l
            out_d[-1] = (out_d[-1] * out_l[-1] + d * l) / tot
            out_l[-1] = tot
        else:
            out_d.append(d)
            out_l.append(l)
    return out_d, out_l


def de_refine(deltas_seed, lams_seed, maxiter=80, popsize=20, seed=42):
    """DE around the LP-optimal seed. Variables: deltas + free lambdas."""
    N = len(deltas_seed)
    DELTA_MIN_DE = 0.005

    def decode(x):
        ds = list(x[:N])
        free = list(x[N:2 * N - 1])
        s = sum(free)
        last = 1.0 - s
        if last < 0.0 or last > 1.0:
            return None, None
        if any(d < DELTA_MIN_DE or d > DELTA_MAX for d in ds):
            return None, None
        return ds, free + [last]

    history = {"best": -np.inf, "best_x": None}

    def obj(x):
        ds, ls = decode(x)
        if ds is None:
            return 0.0
        try:
            r = eval_at(ds, ls)
        except Exception:
            return 0.0
        Mc = r.get("M_cert")
        if Mc is None or not np.isfinite(Mc):
            return 0.0
        if Mc > history["best"]:
            history["best"] = float(Mc)
            history["best_x"] = list(x)
        return -float(Mc)

    # Build bounds: deltas around seed +- 0.03; lambdas in [0.01, 0.99]
    bounds = []
    for d in deltas_seed:
        lo = max(0.005, d - 0.03)
        hi = min(DELTA_MAX, d + 0.03)
        bounds.append((lo, hi))
    for _ in range(N - 1):
        bounds.append((0.01, 0.99))

    # x0 from seed
    x0 = list(deltas_seed) + list(lams_seed[:-1])
    try:
        differential_evolution(
            obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed,
            tol=1e-8, polish=True, init='sobol', x0=x0,
        )
    except TypeError:
        # x0 not supported in older scipy
        differential_evolution(
            obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed,
            tol=1e-8, polish=True,
        )
    return history["best"], history["best_x"]


def main():
    t0 = time.time()
    out = {
        "baseline_K26": BASELINE_K26,
        "MV_base": MV_BASE,
        "DELTA_MAX": DELTA_MAX,
        "U": U,
        "N_QP": N_QP,
    }

    # ============================================================
    # Phase 1: LP min-max over delta grid
    # ============================================================
    grid_results = []
    best_lp = {"t": -np.inf}
    for G_size, lo, hi in [
        (100, 0.005, DELTA_MAX),
        (200, 0.005, DELTA_MAX),
        (400, 0.005, DELTA_MAX),
        (800, 0.005, DELTA_MAX),
    ]:
        d_grid = np.linspace(lo, hi, G_size)
        A = build_freq_matrix(d_grid)
        lam_opt, t_opt = solve_minmax_lp(d_grid, A)
        n_active = int((lam_opt > 1e-6).sum())
        print(f"[LP] G={G_size:4d} t={t_opt:.6f} active={n_active}")
        deltas_act, lams_act = extract_active(d_grid, lam_opt)
        deltas_cl, lams_cl = cluster_deltas(deltas_act, lams_act, gap=0.001)
        # Evaluate M_cert at the LP solution
        r = eval_at(deltas_cl, lams_cl)
        Mc = r.get("M_cert")
        print(f"      M_cert={Mc} k_1={r.get('k_1')} K_2={r.get('K_2')} "
              f"S_1={r.get('S_1')} min_G={r.get('min_G')}")
        grid_results.append({
            "G_size": G_size, "lp_t": float(t_opt), "n_active": n_active,
            "deltas": [float(d) for d in deltas_cl],
            "lambdas": [float(l) for l in lams_cl],
            "M_cert": float(Mc) if (Mc is not None and np.isfinite(Mc)) else None,
            "k_1": float(r.get("k_1")) if r.get("k_1") is not None else None,
            "K_2": float(r.get("K_2")) if r.get("K_2") is not None else None,
            "S_1": float(r.get("S_1")) if r.get("S_1") is not None else None,
            "min_G": float(r.get("min_G")) if r.get("min_G") is not None else None,
        })
        if t_opt > best_lp["t"]:
            best_lp = {
                "G_size": G_size, "t": float(t_opt),
                "deltas": deltas_cl, "lambdas": lams_cl,
                "M_cert": Mc,
                "n_active": n_active,
            }

    out["lp_phase"] = grid_results

    # ============================================================
    # Phase 2: DE refinement around LP-best with consolidated supports
    # ============================================================
    print()
    print("=== Phase 2: DE refinement ===")
    de_results = []
    # Take the finest grid's solution; try further consolidations to 3, 4, 5 deltas
    finest = grid_results[-1]
    fd = finest["deltas"]
    fl = finest["lambdas"]
    print(f"Finest LP: {len(fd)} active deltas: {fd}")
    print(f"           weights: {fl}")

    # Greedy consolidate to N target deltas by merging adjacent smallest-weight pairs
    def consolidate_to(deltas, lams, target_N):
        ds = list(deltas)
        ls = list(lams)
        while len(ds) > target_N:
            # find pair with min combined-impact; here, merge the two closest
            best_i = None
            best_gap = np.inf
            for i in range(len(ds) - 1):
                g = ds[i + 1] - ds[i]
                if g < best_gap:
                    best_gap = g
                    best_i = i
            tot = ls[best_i] + ls[best_i + 1]
            new_d = (ds[best_i] * ls[best_i] + ds[best_i + 1] * ls[best_i + 1]) / tot
            ds = ds[:best_i] + [new_d] + ds[best_i + 2:]
            ls = ls[:best_i] + [tot] + ls[best_i + 2:]
        return ds, ls

    for target_N in [2, 3, 4, 5, 6]:
        if len(fd) < target_N:
            continue
        ds_c, ls_c = consolidate_to(fd, fl, target_N)
        r_init = eval_at(ds_c, ls_c)
        Mc_init = r_init.get("M_cert")
        print(f"\n[DE] target_N={target_N} initial M_cert={Mc_init}")
        print(f"     deltas={ds_c}")
        print(f"     lambdas={ls_c}")
        best_de, best_x = de_refine(ds_c, ls_c, maxiter=60, popsize=18, seed=7 + target_N)
        # Decode best_x to get final (deltas, lambdas)
        best_ds, best_ls = None, None
        if best_x is not None:
            N = target_N
            best_ds = list(best_x[:N])
            free = list(best_x[N:2 * N - 1])
            last = 1.0 - sum(free)
            best_ls = free + [last]
        print(f"     DE best M_cert={best_de}")
        if best_ds is not None:
            print(f"     deltas={best_ds}")
            print(f"     lambdas={best_ls}")
        de_results.append({
            "target_N": target_N,
            "M_cert_init": float(Mc_init) if (Mc_init is not None and np.isfinite(Mc_init)) else None,
            "M_cert_DE": float(best_de) if np.isfinite(best_de) else None,
            "deltas_init": [float(d) for d in ds_c],
            "lambdas_init": [float(l) for l in ls_c],
            "deltas_DE": [float(d) for d in best_ds] if best_ds else None,
            "lambdas_DE": [float(l) for l in best_ls] if best_ls else None,
        })

    out["de_phase"] = de_results

    # ============================================================
    # Final summary
    # ============================================================
    all_M = []
    for g in grid_results:
        if g["M_cert"] is not None:
            all_M.append(("LP-G%d" % g["G_size"], g["M_cert"]))
    for d in de_results:
        if d["M_cert_DE"] is not None:
            all_M.append(("DE-N%d" % d["target_N"], d["M_cert_DE"]))
    if all_M:
        best_label, best_M = max(all_M, key=lambda p: p[1])
    else:
        best_label, best_M = "none", float("-inf")

    out["best_label"] = best_label
    out["best_M_cert"] = float(best_M) if np.isfinite(best_M) else None
    out["delta_vs_K26"] = float(best_M - BASELINE_K26) if np.isfinite(best_M) else None
    out["delta_vs_MV"] = float(best_M - MV_BASE) if np.isfinite(best_M) else None
    out["time_sec"] = time.time() - t0

    print()
    print("=" * 70)
    print(f"BEST overall: {best_label} -> M_cert = {best_M}")
    print(f"  vs K26 baseline 1.29005: {best_M - BASELINE_K26:+.6f}")
    print(f"  vs MV 1.27481:           {best_M - MV_BASE:+.6f}")
    print(f"Total time: {out['time_sec']:.1f} s")

    with open("_M11_freq_tuned.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote _M11_freq_tuned.json")


if __name__ == "__main__":
    main()
