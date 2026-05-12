"""Master K26: push multi-scale arcsine to N >= 3 components and find theoretical
optimum of M_cert over the simplex of arcsine mixtures.

Setup
-----
K(x) = sum_i lambda_i K_arc(x; delta_i),  lambda in simplex, delta_i <= DELTA.
K_hat(xi) = sum_i lambda_i J_0(pi delta_i xi)^2.

Derived quantities (with delta_i fixed, lambda free):
  k_1(lambda) = sum_i lambda_i * c_i,        c_i = J_0(pi delta_i)^2
  K_2(lambda) = lambda^T A lambda,          A_{ij} = 2 int_0^inf J_0(pi d_i xi)^2 J_0(pi d_j xi)^2 dxi
  S_1(lambda) = sum_{j=1..N_QP} a_j^2 / (sum_i lambda_i B_{ji})
                                            B_{ji} = J_0(pi d_i j/U)^2

M_cert is then computed via mv_master_M_cert(k_1, K_2, S_1).

For fixed deltas, the optimum lambda is found by SLSQP / coordinate descent
(local), and many random restarts (global).

We sweep:
  (i)  N=3 coarse + refined
  (ii) N=4 small lambda_4
  (iii) N=20, 50 (grid over delta, optimize lambda)

Output: _master_k26_n_scale.json with the best (delta, lambda) for each N.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import j0

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import DELTA, MV_COEFFS, N_QP, U, mv_master_M_cert  # noqa: E402

# ----- High-accuracy xi grid (matches converged_best_two_component) -----
N_XI = 80001
XI_MAX = 1200.0
_XI = np.linspace(0.0, XI_MAX, N_XI)
_DXI = _XI[1] - _XI[0]


# ---------- Precomputation utilities ----------

def precompute(deltas):
    """For fixed deltas, precompute c_i, A_{ij}, B_{ji}.

    Returns dict with keys c (N,), A (N,N), B (N_QP, N).
    """
    deltas = np.asarray(deltas, dtype=float)
    N = len(deltas)
    # J_0(pi * d_i * xi) on the dense grid: (N, N_XI)
    Jxi = j0(np.pi * deltas[:, None] * _XI[None, :]) ** 2  # (N, N_XI)
    # c_i = J_0(pi d_i)^2  (xi = 1)
    c = j0(np.pi * deltas) ** 2
    # A_{ij} = 2 * int_0^inf Jxi_i * Jxi_j dxi
    # Use trapezoidal weights: half-weight at endpoints
    w = np.full(_XI.shape, _DXI)
    w[0] = 0.5 * _DXI
    w[-1] = 0.5 * _DXI
    Jxi_w = Jxi * w[None, :]
    A = 2.0 * (Jxi @ Jxi_w.T)

    # B at QP frequencies j/U for j=1..N_QP
    qp_xi = np.arange(1, N_QP + 1) / U
    B = j0(np.pi * qp_xi[:, None] * deltas[None, :]) ** 2  # (N_QP, N)
    return {"c": c, "A": A, "B": B, "deltas": deltas}


def eval_lambda(pre, lam):
    """Given precomputed pre and lambda vector, return (k_1, K_2, S_1, M_cert)."""
    c = pre["c"]; A = pre["A"]; B = pre["B"]
    lam = np.asarray(lam, dtype=float)
    k_1 = float(c @ lam)
    K_2 = float(lam @ A @ lam)
    kh_qp = B @ lam  # (N_QP,)
    if np.any(kh_qp < 1e-18):
        return k_1, K_2, np.inf, None
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))
    M_cert = mv_master_M_cert(k_1, K_2, S_1)
    return k_1, K_2, S_1, M_cert


# ---------- Local optimization on the simplex ----------

def neg_M_of_z(z, pre):
    """Softmax param: lambda = softmax(z), so we don't need a constraint."""
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    lam = e / e.sum()
    _, _, _, M = eval_lambda(pre, lam)
    if M is None or not np.isfinite(M):
        return 1e3
    return -M


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def optimize_lambda(pre, lam0=None, n_restarts=8, seed=0):
    """Find the best lambda for fixed deltas. Returns dict with lam, M_cert, ..."""
    N = len(pre["deltas"])
    rng = np.random.default_rng(seed)
    best = {"M_cert": -np.inf, "lam": None}

    starts = []
    if lam0 is not None:
        lam0 = np.maximum(np.asarray(lam0, dtype=float), 1e-6)
        lam0 = lam0 / lam0.sum()
        starts.append(np.log(lam0))
    # Pure delta_1 start (heavy on first component, which is DELTA=0.138)
    pure = np.full(N, 1e-3); pure[0] = 1.0
    pure = pure / pure.sum()
    starts.append(np.log(pure))
    # Uniform start
    starts.append(np.zeros(N))
    # Random starts (Dirichlet-ish)
    for _ in range(n_restarts):
        z = rng.normal(size=N) * 2.0
        starts.append(z)

    for z0 in starts:
        try:
            res = minimize(
                neg_M_of_z, z0, args=(pre,),
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000},
            )
            lam = softmax(res.x)
            _, _, _, M = eval_lambda(pre, lam)
            if M is not None and M > best["M_cert"]:
                best = {"M_cert": float(M), "lam": lam.tolist()}
        except Exception as e:
            pass
    return best


# ---------- Sweeps ----------

def sweep_3_component_coarse(verbose=True):
    """Task 1: 3-component sweep. delta_1=DELTA fixed."""
    t0 = time.time()
    d1 = DELTA
    d23_list = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    l1_list = [0.85, 0.9, 0.92, 0.94, 0.96]
    l3_list = [0.01, 0.02, 0.05, 0.10]

    results = []
    best = {"M_cert": -np.inf}
    count = 0
    for d2 in d23_list:
        for d3 in d23_list:
            if d3 <= d2:
                continue  # ensure d2 != d3 and d2 < d3 (symmetric)
            pre = precompute([d1, d2, d3])
            for l1 in l1_list:
                for l3 in l3_list:
                    l2 = 1.0 - l1 - l3
                    if l2 < 0.0 or l2 > 1.0:
                        continue
                    lam = np.array([l1, l2, l3])
                    _, _, _, M = eval_lambda(pre, lam)
                    if M is None:
                        continue
                    count += 1
                    rec = {"d2": d2, "d3": d3, "l1": l1, "l2": l2, "l3": l3,
                           "M_cert": float(M)}
                    results.append(rec)
                    if M > best["M_cert"]:
                        best = dict(rec); best["M_cert"] = float(M)
    if verbose:
        print(f"[3c coarse] {count} points, best M={best['M_cert']:.5f} at {best}")
        print(f"  time: {time.time()-t0:.1f}s")
    return {"results": results, "best": best, "n_eval": count}


def refine_3_component(seed_best, verbose=True):
    """Task 2: refine around the best 3-component."""
    t0 = time.time()
    d1 = DELTA
    d2_0 = seed_best["d2"]; d3_0 = seed_best["d3"]
    # Fine grids around the seed
    d2_grid = np.unique(np.round(np.linspace(max(0.02, d2_0 - 0.02),
                                              min(d1 - 0.001, d2_0 + 0.02), 9), 5))
    d3_grid = np.unique(np.round(np.linspace(max(0.02, d3_0 - 0.02),
                                              min(d1 - 0.001, d3_0 + 0.02), 9), 5))
    results = []
    best = {"M_cert": -np.inf}
    count = 0
    for d2 in d2_grid:
        for d3 in d3_grid:
            if d3 == d2:
                continue
            # Ensure ordering for caching: sort the smaller two
            if d3 < d2:
                d2, d3 = d3, d2
            pre = precompute([d1, d2, d3])
            # Use lambda optimization with seed
            seed_lam = [seed_best["l1"], seed_best["l2"], seed_best["l3"]]
            r = optimize_lambda(pre, lam0=seed_lam, n_restarts=4)
            count += 1
            rec = {"d2": float(d2), "d3": float(d3),
                   "lam": r["lam"], "M_cert": r["M_cert"]}
            results.append(rec)
            if r["M_cert"] > best["M_cert"]:
                best = dict(rec)
    if verbose:
        print(f"[3c refined] {count} grid points, best M={best['M_cert']:.5f}")
        print(f"  best: d2={best['d2']}, d3={best['d3']}, lam={best['lam']}")
        print(f"  time: {time.time()-t0:.1f}s")
    return {"results": results, "best": best}


def sweep_4_component(seed_best, verbose=True):
    """Task 3: 4-component, add an intermediate scale."""
    t0 = time.time()
    d1 = DELTA
    d2 = seed_best["d2"]; d3 = seed_best["d3"]
    # Intermediate scale d_4 between min and max of {d2, d3}
    lo = min(d2, d3); hi = max(d2, d3)
    if hi - lo < 0.01:
        d4_grid = [0.03, 0.04, 0.06, 0.08, 0.10]
    else:
        d4_grid = list(np.round(np.linspace(lo + 0.005, hi - 0.005, 7), 5))
        # Also add some scales outside
        d4_grid += [0.03, 0.04, 0.10, 0.11, 0.12, 0.13]
    d4_grid = sorted(set(d4_grid))

    results = []
    best = {"M_cert": -np.inf}
    for d4 in d4_grid:
        if abs(d4 - d1) < 1e-4 or abs(d4 - d2) < 1e-4 or abs(d4 - d3) < 1e-4:
            continue
        pre = precompute([d1, d2, d3, d4])
        # Initial: take seed 3-component lam, give d4 small mass split from d2
        seed_lam = list(seed_best["lam"]) if "lam" in seed_best else \
            [seed_best["l1"], seed_best["l2"], seed_best["l3"]]
        seed4 = seed_lam + [0.02]
        s = sum(seed4); seed4 = [x / s for x in seed4]
        r = optimize_lambda(pre, lam0=seed4, n_restarts=10)
        rec = {"d4": float(d4), "lam": r["lam"], "M_cert": r["M_cert"]}
        results.append(rec)
        if r["M_cert"] > best["M_cert"]:
            best = dict(rec)
            best["deltas"] = [d1, d2, d3, d4]
    if verbose:
        print(f"[4c] {len(results)} d4 values, best M={best['M_cert']:.5f}")
        print(f"  time: {time.time()-t0:.1f}s")
    return {"results": results, "best": best}


def grid_n_component(N, verbose=True):
    """Task 4-6: dense delta grid of size N, optimize lambda via Nelder-Mead."""
    t0 = time.time()
    # delta grid (0.02, DELTA] with N nodes; include DELTA exactly
    # Logarithmic spacing puts more nodes near DELTA (where the best mass lives)
    deltas = np.geomspace(0.02, DELTA, N)
    # Always include DELTA exactly (last node already DELTA)
    pre = precompute(deltas)
    # Seed: heavy on the largest delta (DELTA), tiny mass spread elsewhere
    lam0 = np.full(N, 1.0 / (4 * N))
    lam0[-1] = 1.0 - lam0[:-1].sum()  # heavy on last
    r = optimize_lambda(pre, lam0=lam0, n_restarts=12)
    lam = r["lam"]
    if verbose:
        print(f"[N={N}] best M={r['M_cert']:.5f}  time={time.time()-t0:.1f}s")
        if lam is not None:
            top = sorted([(float(deltas[i]), float(lam[i])) for i in range(N)],
                         key=lambda x: -x[1])[:8]
            print(f"  top atoms (delta, lambda): {top}")
    return {"N": N, "deltas": deltas.tolist(), "lam": lam,
            "M_cert": r["M_cert"]}


def coordinate_descent_n(N=50, n_iter=200, seed=0, verbose=True):
    """Coordinate descent on N-component lambda (delta grid fixed)."""
    t0 = time.time()
    deltas = np.geomspace(0.02, DELTA, N)
    pre = precompute(deltas)
    rng = np.random.default_rng(seed)
    # Init: concentrated on DELTA
    lam = np.full(N, 1e-6); lam[-1] = 1.0 - lam[:-1].sum()
    _, _, _, M_cur = eval_lambda(pre, lam)
    history = [M_cur]
    for it in range(n_iter):
        # Pick a pair (i, j) and find optimal split
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N))
        if i == j:
            continue
        # Total mass on (i, j)
        s = lam[i] + lam[j]
        if s < 1e-10:
            continue
        # Line search on alpha in [0, 1]: lam_i = alpha*s, lam_j = (1-alpha)*s
        best_alpha = lam[i] / s
        best_M = M_cur
        for alpha in np.linspace(0.0, 1.0, 21):
            new_lam = lam.copy()
            new_lam[i] = alpha * s
            new_lam[j] = (1.0 - alpha) * s
            _, _, _, M = eval_lambda(pre, new_lam)
            if M is not None and M > best_M:
                best_M = M
                best_alpha = alpha
        lam[i] = best_alpha * s
        lam[j] = (1.0 - best_alpha) * s
        if best_M > M_cur:
            M_cur = best_M
        history.append(M_cur)
    if verbose:
        print(f"[CD N={N}] final M={M_cur:.5f}  time={time.time()-t0:.1f}s")
    return {"N": N, "deltas": deltas.tolist(), "lam": lam.tolist(),
            "M_cert": float(M_cur), "history_final": float(M_cur)}


def main():
    print("=" * 78)
    print("Master K26: N-scale arcsine mixture optimisation")
    print(f"XI_MAX={XI_MAX}, N_XI={N_XI}, DELTA={DELTA}, U={U}, N_QP={N_QP}")
    print("=" * 78)

    out = {"family": "n-scale-arcsine-mixture",
           "DELTA": DELTA, "U": U, "N_QP": N_QP,
           "XI_MAX": XI_MAX, "N_XI": N_XI}

    # Sanity: pure DELTA arcsine
    pre1 = precompute([DELTA])
    _, _, _, M1 = eval_lambda(pre1, [1.0])
    print(f"\nSanity: pure DELTA arcsine M_cert = {M1:.5f}")
    out["baseline_pure_DELTA"] = float(M1) if M1 is not None else None

    # Sanity: 2-component best from prior run
    pre2 = precompute([DELTA, 0.055])
    _, _, _, M2 = eval_lambda(pre2, [0.9312, 0.0688])
    print(f"Sanity: 2-scale (0.138, 0.055; 0.9312, 0.0688) M_cert = {M2:.5f}")
    out["baseline_2_scale"] = float(M2) if M2 is not None else None

    # Also do a full 2-scale optimization (finer than prior, with our N_XI=80001)
    print("\n--- 2-scale re-optimization (high-accuracy grid) ---")
    best2 = {"M_cert": -np.inf}
    d2_grid = np.round(np.linspace(0.03, 0.10, 15), 5)
    for d2 in d2_grid:
        pre = precompute([DELTA, d2])
        r = optimize_lambda(pre, lam0=[0.93, 0.07], n_restarts=4)
        if r["M_cert"] > best2["M_cert"]:
            best2 = {"M_cert": r["M_cert"], "d2": float(d2), "lam": r["lam"]}
    print(f"2-scale best: M={best2['M_cert']:.5f} at d2={best2['d2']}, lam={best2['lam']}")
    out["two_scale_best"] = best2

    # Task 1: 3-component coarse
    print("\n--- Task 1: 3-component coarse sweep ---")
    three_c = sweep_3_component_coarse(verbose=True)
    out["three_coarse"] = three_c

    # Task 2: 3-component refined around best
    print("\n--- Task 2: 3-component refined ---")
    three_r = refine_3_component(three_c["best"], verbose=True)
    out["three_refined"] = three_r
    best_3 = three_r["best"] if three_r["best"]["M_cert"] > three_c["best"]["M_cert"] \
        else {"M_cert": three_c["best"]["M_cert"], "d2": three_c["best"]["d2"],
              "d3": three_c["best"]["d3"],
              "lam": [three_c["best"]["l1"], three_c["best"]["l2"], three_c["best"]["l3"]]}
    out["best_3_component"] = best_3
    print(f"\nBEST 3c: M={best_3['M_cert']:.5f}")

    # Task 3: 4-component
    print("\n--- Task 3: 4-component sweep ---")
    four = sweep_4_component(best_3, verbose=True)
    out["four_component"] = four
    print(f"\nBEST 4c: M={four['best']['M_cert']:.5f}")

    # Task 4-6: N = 20, 50 (continuous-distribution discretization)
    for N in (10, 20, 50):
        print(f"\n--- N = {N} grid optimization ---")
        rN = grid_n_component(N, verbose=True)
        out[f"N_{N}"] = rN
        print(f"\nBEST N={N}: M={rN['M_cert']:.5f}")

    # Coordinate descent for N=50 as a cross-check (Caratheodory sanity)
    print("\n--- Coordinate descent N=50 cross-check ---")
    cd = coordinate_descent_n(N=50, n_iter=400, seed=1, verbose=True)
    out["N_50_CD"] = cd

    # Plateau analysis
    plateau = {
        "M_2": float(best2["M_cert"]),
        "M_3": float(best_3["M_cert"]),
        "M_4": float(four["best"]["M_cert"]),
        "M_10": float(out["N_10"]["M_cert"]),
        "M_20": float(out["N_20"]["M_cert"]),
        "M_50": float(out["N_50"]["M_cert"]),
        "M_50_CD": float(cd["M_cert"]),
    }
    out["plateau"] = plateau

    final_best = max(plateau.values())
    out["final_best_M_cert"] = float(final_best)
    out["beats_MV_1.2748"] = bool(final_best > 1.27481)
    out["beats_CS_1.2802"] = bool(final_best > 1.2802)

    outpath = os.path.join(REPO, "_master_k26_n_scale.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\n{'='*78}")
    print(f"Final best M_cert = {final_best:.6f}")
    print(f"Plateau: {plateau}")
    print(f"Wrote {outpath}")
    return out


if __name__ == "__main__":
    main()
