"""Take a JSON from pod_parallel_search and run rigorous mpmath verify on
the best (K, a_box) per K, with a safety scaling factor to ensure g >= 0.

Usage:
    python -m delsarte_dual.pod_rigorous_verify /tmp/lp_search_wgen.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import family_f1_selberg as f1
from . import optimise_lp as olp
from . import verify as ver


def check_g_nonneg_f64(p: f1.F1Params, n_t: int = 200001) -> tuple[float, float]:
    """Return (min_g, int_g) sampled on fine grid.

    Thin wrapper around optimise_lp._check_g_nonneg_f64 (canonical location).
    """
    return olp._check_g_nonneg_f64(p, n_t=n_t)


def find_safe_scale(p_raw: f1.F1Params, target_min_g: float = 1e-6) -> float:
    """Binary search for scale s such that scaled g has min >= target_min_g.

    Thin wrapper around optimise_lp.find_safe_scale (canonical location).
    """
    return olp.find_safe_scale(p_raw, target_min_g=target_min_g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--precision-bits", type=int, default=120)
    parser.add_argument("--subdiv", type=int, default=4096)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--target-min-g", type=float, default=1e-5)
    args = parser.parse_args()

    with open(args.json_path) as fh:
        data = json.load(fh)

    # Group by K; pick best a_box per K.
    by_K = {}
    for entry in data:
        K = entry["K"]
        if K not in by_K or entry["best"]["score"] > by_K[K]["best"]["score"]:
            by_K[K] = entry

    print("# Rigorous verify of best LP solutions per K (general-f weight)")
    print(f"# precision_bits={args.precision_bits}, subdiv={args.subdiv}, rel_tol={args.rel_tol}")
    print(f"# target_min_g={args.target_min_g}\n")

    for K in sorted(by_K.keys()):
        entry = by_K[K]
        best = entry["best"]
        p_raw = f1.F1Params(
            T=best["T"], omega=best["omega"], a=best["a"],
        )
        t0 = time.time()
        scale = find_safe_scale(p_raw, target_min_g=args.target_min_g)
        a_scaled = [scale * v for v in p_raw.a]
        p = f1.F1Params(T=p_raw.T, omega=p_raw.omega, a=a_scaled)
        min_g, int_g = check_g_nonneg_f64(p)
        print(f"# K={K}  a_box={entry['a_box']}  scale={scale:.4f}  min_g_f64={min_g:.3e}  (LP={best['score']:.4f})")

        # Now rigorous mpmath verify.
        vb = ver.verify_f1(
            p,
            rel_tol=args.rel_tol,
            n_subdiv=args.subdiv,
            precision_bits=args.precision_bits,
            denom_kind="int_g",
        )
        elapsed = time.time() - t0
        print(f"  num   in [{float(vb.numerator_lo):.6f}, {float(vb.numerator_hi):.6f}]")
        print(f"  int_g in [{float(vb.int_g_lo):.6f}, {float(vb.int_g_hi):.6f}]")
        print(f"  max_g in [{float(vb.max_g_lo):.6f}, {float(vb.max_g_hi):.6f}]")
        print(f"  g_nonneg_rigorous={vb.g_nonneg_certified}  pd_certified={vb.pd_certified}")
        print(f"  lb  in [{float(vb.lb_low):.6f}, {float(vb.lb_high):.6f}]  width={float(vb.lb_high - vb.lb_low):.2e}  [{elapsed:.1f}s]")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
