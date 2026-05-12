"""Parallel LP-based F1 grid search for large K.

Designed for the 192-core pod. Searches over:
  - K in {8, 12, 16, 20, 25, 30, 40}
  - T in [0.3, 2.8] with 30 grid points
  - omega in [0.05, 3.0] with 40 grid points
  - a_box (abs-bound on cosine coefficients) in {1.5, 2.5, 4, 6}

For each (K, a_box) configuration, runs the LP at every (T, omega) grid point
in parallel via joblib. Records the best (float64) score achieved along with
the parameters.

Output: JSON summary to stdout, best-per-K.

Usage:
  python -m delsarte_dual.pod_parallel_search --n_jobs 180
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product

import numpy as np
from joblib import Parallel, delayed

from . import optimise_lp


def _worker(T, omega, K, a_box, n_xi, n_t):
    try:
        s, a_opt, info = optimise_lp.f1_lp_for_fixed_Tomega(
            float(T), float(omega), K=K,
            n_xi=n_xi, n_t=n_t, a_box=a_box,
        )
        return {"T": float(T), "omega": float(omega), "K": int(K),
                "a_box": float(a_box), "score": float(s),
                "a": None if a_opt is None else a_opt.tolist()}
    except Exception as e:
        return {"T": float(T), "omega": float(omega), "K": int(K),
                "a_box": float(a_box), "score": float("-inf"),
                "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--Ks", type=int, nargs="*", default=[8, 12, 16, 20, 25, 30])
    parser.add_argument("--a_boxes", type=float, nargs="*", default=[1.5, 2.5, 4.0, 6.0])
    parser.add_argument("--n_T", type=int, default=26)
    parser.add_argument("--n_omega", type=int, default=35)
    parser.add_argument("--T_lo", type=float, default=0.3)
    parser.add_argument("--T_hi", type=float, default=2.8)
    parser.add_argument("--om_lo", type=float, default=0.05)
    parser.add_argument("--om_hi", type=float, default=3.0)
    parser.add_argument("--n_xi", type=int, default=2001)
    parser.add_argument("--n_t", type=int, default=2001)
    parser.add_argument("--out", type=str, default="/tmp/delsarte_lp_search.json")
    args = parser.parse_args()

    Tg = np.linspace(args.T_lo, args.T_hi, args.n_T)
    Og = np.linspace(args.om_lo, args.om_hi, args.n_omega)

    print(f"# Grid: {len(Tg)} T x {len(Og)} omega = {len(Tg)*len(Og)} per (K, a_box)")
    print(f"# Configs: {len(args.Ks)} K x {len(args.a_boxes)} a_box = {len(args.Ks)*len(args.a_boxes)} configs")
    print(f"# Total LPs: {len(Tg)*len(Og)*len(args.Ks)*len(args.a_boxes)}")

    all_results = []

    for K, a_box in product(args.Ks, args.a_boxes):
        t0 = time.time()
        tasks = [(T, om, K, a_box, args.n_xi, args.n_t) for T in Tg for om in Og]
        results = Parallel(n_jobs=args.n_jobs, verbose=0, backend="loky")(
            delayed(_worker)(*tk) for tk in tasks
        )
        best = max(results, key=lambda r: r["score"] if r["score"] != float("-inf") else -1e18)
        print(f"K={K:3d} a_box={a_box:.1f}: best_score={best['score']:.6f} "
              f"T={best['T']:.3f} omega={best['omega']:.3f} "
              f"({time.time()-t0:.1f}s, {len(tasks)} LPs)")
        all_results.append({
            "K": K, "a_box": a_box, "best": best,
            "time_sec": time.time() - t0,
        })
        sys.stdout.flush()

    # Top overall.
    best_overall = max(all_results, key=lambda r: r["best"]["score"])
    print(f"\n# BEST OVERALL: score={best_overall['best']['score']:.6f}")
    print(f"#   K={best_overall['K']}, a_box={best_overall['a_box']}")
    print(f"#   T={best_overall['best']['T']}, omega={best_overall['best']['omega']}")
    print(f"#   a={best_overall['best']['a']}")

    with open(args.out, "w") as fh:
        json.dump(all_results, fh, indent=2, default=float)
    print(f"# Wrote {args.out}")


if __name__ == "__main__":
    main()
