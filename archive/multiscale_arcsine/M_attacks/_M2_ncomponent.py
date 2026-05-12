"""Agent M2: N-component multi-scale arcsine kernel search via differential evolution.

K(x) = sum_{i=1..N} lambda_i * K_arcsine(x; delta_i),
constraints: sum(lambda) = 1, 0 < delta_i <= DELTA_MAX = 0.138, lambda_i >= 0.

For each N in {3,4,5,6,8,12}, DE search the (delta, lambda) joint space, then refine
the best N. Goal: surpass N=2's M_cert = 1.29005.
"""
from __future__ import annotations

import json
import time
import numpy as np
from scipy.optimize import differential_evolution

# Re-use the working K26 pipeline (do not redefine).
from _K26_full_sweep_reopt import eval_at, DELTA, U, N_QP

DELTA_MAX = DELTA  # = 0.138
DELTA_MIN = 0.005
LAM_MIN = 0.01
LAM_MAX = 0.99

# Baselines
MV_BASE = 1.27481
N2_BEST = 1.29005


def decode(x, N):
    """Decode DE vector x = [delta_1,...,delta_N, lambda_1,...,lambda_{N-1}].

    Returns (deltas, lambdas) with sum(lambdas)=1. If invalid, returns (None, None).
    """
    deltas = list(x[:N])
    lams_free = list(x[N:2 * N - 1])
    s = sum(lams_free)
    lam_last = 1.0 - s
    if lam_last < LAM_MIN or lam_last > LAM_MAX:
        return None, None
    if any(d < DELTA_MIN or d > DELTA_MAX for d in deltas):
        return None, None
    return deltas, lams_free + [lam_last]


def make_objective(N, history):
    """Return DE objective function: negative M_cert. Side-effect: append to history."""
    def obj(x):
        deltas, lambdas = decode(x, N)
        if deltas is None:
            return 0.0
        try:
            r = eval_at(deltas, lambdas)
        except Exception:
            return 0.0
        Mc = r.get("M_cert")
        if Mc is None or not np.isfinite(Mc):
            return 0.0
        # Track best
        if not history or Mc > history[-1]["M_cert"]:
            history.append({"M_cert": float(Mc), "deltas": deltas,
                            "lambdas": lambdas,
                            "k_1": float(r["k_1"]), "K_2": float(r["K_2"]),
                            "S_1": float(r["S_1"]), "min_G": float(r["min_G"])})
        return -float(Mc)
    return obj


def run_de_for_N(N, seeds=(1, 42, 100), maxiter=80, popsize=30):
    """DE search for fixed N across multiple seeds. Returns best result dict."""
    print(f"\n========== N = {N} ==========")
    # Bounds: deltas in [DELTA_MIN, DELTA_MAX], lambdas (free, N-1) in [LAM_MIN, LAM_MAX]
    bounds = [(DELTA_MIN, DELTA_MAX)] * N + [(LAM_MIN, LAM_MAX)] * (N - 1)

    best = {"M_cert": -np.inf, "seed": None}
    per_seed = []
    for seed in seeds:
        t0 = time.time()
        history = []
        obj = make_objective(N, history)
        result = differential_evolution(
            obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed,
            polish=False, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            init='sobol', updating='deferred', workers=1, disp=False)
        elapsed = time.time() - t0
        Mc = -float(result.fun) if result.fun < 0 else None
        # Get the best from history (in case the DE result wasn't appended)
        if history:
            top = max(history, key=lambda h: h["M_cert"])
            if Mc is None or top["M_cert"] > Mc:
                Mc = top["M_cert"]
                deltas_best = top["deltas"]
                lambdas_best = top["lambdas"]
                extras = {"k_1": top["k_1"], "K_2": top["K_2"], "S_1": top["S_1"], "min_G": top["min_G"]}
            else:
                deltas_best, lambdas_best = decode(result.x, N)
                extras = {}
        else:
            deltas_best, lambdas_best = decode(result.x, N) if result.x is not None else (None, None)
            extras = {}

        seed_rec = {"seed": seed, "M_cert": Mc, "deltas": deltas_best, "lambdas": lambdas_best,
                    "elapsed_s": elapsed, "n_evals": len(history), **extras}
        per_seed.append(seed_rec)
        print(f"  seed={seed}: M_cert={Mc:.5f}  ({elapsed:.1f}s, hist={len(history)})")
        if Mc is not None and Mc > best["M_cert"]:
            best = seed_rec.copy()

    return best, per_seed


def refine_best(N, best, maxiter=40, popsize=30, seeds=(7, 17)):
    """Second DE pass with tighter bounds around best."""
    print(f"\n--- Refine N={N} around M_cert={best['M_cert']:.5f} ---")
    deltas0 = best["deltas"]
    lambdas0 = best["lambdas"]
    if deltas0 is None:
        return best

    # Tight bounds: +/- 0.015 on delta, +/- 0.05 on lambda
    bounds = []
    for d in deltas0:
        lo = max(DELTA_MIN, d - 0.015)
        hi = min(DELTA_MAX, d + 0.015)
        if hi - lo < 1e-4:
            hi = lo + 1e-3
        bounds.append((lo, hi))
    for lam in lambdas0[:-1]:
        lo = max(LAM_MIN, lam - 0.05)
        hi = min(LAM_MAX, lam + 0.05)
        if hi - lo < 1e-4:
            hi = lo + 1e-3
        bounds.append((lo, hi))

    refined_best = best.copy()
    for seed in seeds:
        t0 = time.time()
        history = [{"M_cert": best["M_cert"], "deltas": deltas0, "lambdas": lambdas0,
                    "k_1": best.get("k_1", 0.0), "K_2": best.get("K_2", 0.0),
                    "S_1": best.get("S_1", 0.0), "min_G": best.get("min_G", 0.0)}]
        obj = make_objective(N, history)
        result = differential_evolution(
            obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed,
            polish=False, tol=1e-8, mutation=(0.4, 0.9), recombination=0.8,
            init='sobol', updating='deferred', workers=1, disp=False)
        elapsed = time.time() - t0
        top = max(history, key=lambda h: h["M_cert"])
        if top["M_cert"] > refined_best["M_cert"]:
            refined_best = {"seed": seed, "M_cert": top["M_cert"],
                            "deltas": top["deltas"], "lambdas": top["lambdas"],
                            "k_1": top["k_1"], "K_2": top["K_2"],
                            "S_1": top["S_1"], "min_G": top["min_G"],
                            "elapsed_s": elapsed, "n_evals": len(history)}
        print(f"  refine seed={seed}: top M_cert={top['M_cert']:.5f}  ({elapsed:.1f}s)")
    return refined_best


def sorted_components(best):
    """Return sorted list of (delta, lambda) by delta descending."""
    pairs = list(zip(best["deltas"], best["lambdas"]))
    pairs.sort(key=lambda p: -p[0])
    return pairs


def main():
    t_start = time.time()
    print("=" * 80)
    print("Agent M2: N-component multi-scale arcsine via DE")
    print(f"DELTA_MAX={DELTA_MAX}, DELTA_MIN={DELTA_MIN}, baselines: MV={MV_BASE}, N2={N2_BEST}")
    print("=" * 80)

    # eval_at costs ~3s/call so we right-size budgets. Total DE evals per (N, seed)
    # is approximately popsize * len(x) * (maxiter + 1) where len(x) = 2N - 1.
    # Per-N config: (seeds, maxiter, popsize).
    N_configs = {
        3:  ((1, 42), 12, 5),   # ~330 evals/seed * 2 seeds ~ 34 min  (heavy)
        4:  ((1, 42),  9, 5),   # ~350 evals/seed * 2 seeds ~ 36 min
        5:  ((1,),     9, 5),   # ~450 evals/seed
        6:  ((1,),     8, 5),   # ~495 evals
        8:  ((1,),     6, 4),   # ~420 evals
        12: ((1,),     5, 4),   # ~552 evals
    }
    # Lighter actual config (budget ~30 min total at ~3s/eval).
    # Total ~ 150 evals/N * 6 N = 900 evals = 47 min.
    N_configs = {
        3:  ((1,),  6, 5),   # popsize*len(x)*(maxiter+1) = 5*5*7 = 175 evals
        4:  ((1,),  5, 4),   # 4*7*6 = 168 evals
        5:  ((1,),  4, 4),   # 4*9*5 = 180 evals
        6:  ((1,),  3, 4),   # 4*11*4 = 176 evals
        8:  ((1,),  3, 3),   # 3*15*4 = 180 evals
        12: ((1,),  2, 3),   # 3*23*3 = 207 evals
    }
    N_values = sorted(N_configs.keys())
    results_by_N = {}

    for N in N_values:
        seeds, maxit, popsz = N_configs[N]
        best, per_seed = run_de_for_N(N, seeds=seeds, maxiter=maxit, popsize=popsz)
        results_by_N[N] = {"best": best, "per_seed": per_seed}
        print(f"  >>> N={N} best M_cert = {best['M_cert']:.5f}"
              f"  (vs N=2: {best['M_cert'] - N2_BEST:+.5f})")

    # Find best N
    best_N = max(results_by_N.keys(), key=lambda n: results_by_N[n]["best"]["M_cert"])
    overall_best = results_by_N[best_N]["best"]
    print("\n" + "=" * 80)
    print(f"Best across N: N={best_N}, M_cert={overall_best['M_cert']:.5f}")
    print("=" * 80)

    # Refine the best N
    refined = refine_best(best_N, overall_best, maxiter=6, popsize=5, seeds=(7,))
    if refined["M_cert"] > overall_best["M_cert"]:
        print(f"Refinement helped: {overall_best['M_cert']:.5f} -> {refined['M_cert']:.5f}")
    else:
        print(f"Refinement did not improve: stays at {overall_best['M_cert']:.5f}")
    final_best = refined if refined["M_cert"] > overall_best["M_cert"] else overall_best
    results_by_N[best_N]["refined"] = refined

    # Print sorted components for final best
    print(f"\n--- Sorted components at FINAL BEST (N={best_N}, M_cert={final_best['M_cert']:.5f}) ---")
    print(f"{'i':>3} {'delta_i':>10} {'lambda_i':>10}")
    print("-" * 30)
    pairs = sorted_components(final_best)
    for i, (d, lam) in enumerate(pairs):
        print(f"{i+1:>3} {d:>10.5f} {lam:>10.5f}")

    # Mechanism analysis
    deltas_arr = np.array([p[0] for p in pairs])
    lams_arr = np.array([p[1] for p in pairs])
    big_lam_idx = np.argmax(lams_arr)
    big_delta = deltas_arr[big_lam_idx]
    small_idx = [i for i in range(len(deltas_arr)) if i != big_lam_idx]
    avg_small_delta = float(np.mean(deltas_arr[small_idx])) if small_idx else 0.0
    n_large_delta = int(np.sum(deltas_arr > 0.10))
    n_small_delta = int(np.sum(deltas_arr < 0.03))
    spread = float(np.max(deltas_arr) - np.min(deltas_arr))
    print("\n--- Mechanism analysis ---")
    print(f"  Dominant component: delta={big_delta:.5f}, lambda={lams_arr[big_lam_idx]:.5f}")
    print(f"  Avg delta of non-dominant: {avg_small_delta:.5f}")
    print(f"  # components with delta > 0.10: {n_large_delta} / {len(deltas_arr)}")
    print(f"  # components with delta < 0.03: {n_small_delta} / {len(deltas_arr)}")
    print(f"  delta spread: {spread:.5f}")
    if n_large_delta == 1 and n_small_delta >= len(deltas_arr) - 1:
        mech = "TWO-MODE: one large delta + (N-1) clustered small deltas"
    elif n_large_delta + n_small_delta < len(deltas_arr) - 1:
        mech = "SPREAD: components distributed across [0, 0.138], suggesting continuous density"
    else:
        mech = "MIXED: partial clustering"
    print(f"  Mechanism: {mech}")

    # Saturation analysis
    Mc_by_N = {N: results_by_N[N]["best"]["M_cert"] for N in N_values}
    print("\n--- M_cert by N ---")
    for N in N_values:
        gain_vs_n2 = Mc_by_N[N] - N2_BEST
        print(f"  N={N}: M_cert={Mc_by_N[N]:.5f}  (vs N=2: {gain_vs_n2:+.5f})")
    # Determine saturation
    sorted_Ns = sorted(N_values)
    sat_N = sorted_Ns[-1]
    for i, N in enumerate(sorted_Ns[1:], 1):
        prev_N = sorted_Ns[i - 1]
        if Mc_by_N[N] - Mc_by_N[prev_N] < 1e-4:
            sat_N = prev_N
            break
    print(f"  Saturation appears at N >= {sat_N}")

    total_elapsed = time.time() - t_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s")

    # Persist
    output = {
        "agent": "M2",
        "description": "N-component multi-scale arcsine via differential evolution",
        "baselines": {"MV": MV_BASE, "N2": N2_BEST},
        "DELTA_MAX": DELTA_MAX,
        "DELTA_MIN": DELTA_MIN,
        "N_values": N_values,
        "results_by_N": {
            str(N): {
                "best": {k: v for k, v in results_by_N[N]["best"].items()},
                "per_seed": results_by_N[N]["per_seed"],
                **({"refined": results_by_N[N]["refined"]} if "refined" in results_by_N[N] else {}),
            } for N in N_values
        },
        "best_N": best_N,
        "final_best": final_best,
        "sorted_components": [{"delta": d, "lambda": lam} for d, lam in pairs],
        "mechanism": mech,
        "saturation_N": sat_N,
        "Mc_by_N": Mc_by_N,
        "total_elapsed_s": total_elapsed,
    }
    out_path = "_M2_ncomponent_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
