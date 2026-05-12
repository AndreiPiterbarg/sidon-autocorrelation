"""End-to-end orchestrator: optimise -> verify -> report.

Run:
    python -m delsarte_dual.run_all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mpmath import mp

from . import family_f1_selberg as f1
from . import family_f2_gauss_poly as f2
from . import optimise as opt
from . import verify as ver


CURRENT_RECORD = 1.2802   # Matolcsi-Vinuesa


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--restarts-f1", type=int, default=30)
    parser.add_argument("--restarts-f2", type=int, default=30)
    parser.add_argument("--rel-tol", type=float, default=1e-8)
    parser.add_argument("--subdiv", type=int, default=16384)
    parser.add_argument("--precision-bits", type=int, default=200)
    parser.add_argument("--tight", action="store_true",
                        help="Use tighter verification defaults: "
                             "rel_tol=1e-12, subdiv=65536, precision-bits=300.")
    parser.add_argument("--skip-f2", action="store_true")
    parser.add_argument("--skip-optim", action="store_true",
                        help="Use canonical MV-reference params only (fast sanity).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.tight:
        args.rel_tol = 1e-12
        args.subdiv = 65536
        args.precision_bits = 300

    mp.prec = args.precision_bits
    t0 = time.time()

    results = []

    # ------------------------ F1 ------------------------
    print("=" * 70)
    print("Family F1 (Fejer-modulated Selberg-type)")
    print("=" * 70)
    if args.skip_optim:
        f1_best_params = opt.matolcsi_vinuesa_reference_f1()
        f1_score = None
    else:
        best = opt.optimise_f1(K=2, n_restarts=args.restarts_f1, verbose=args.verbose)
        if best["params"] is None:
            print("F1: no feasible point found.")
            f1_best_params = opt.matolcsi_vinuesa_reference_f1()
            f1_score = None
        else:
            f1_best_params = best["params"]
            f1_score = best["score"]
            print(f"F1 float64 best score: {f1_score:.6f}")
            print(f"    params: {f1_best_params}")

    print(">>> Running rigorous verification (F1)...")
    vb1 = ver.verify_f1(
        f1_best_params,
        rel_tol=args.rel_tol,
        n_subdiv=args.subdiv,
        precision_bits=args.precision_bits,
    )
    print(ver.format_ball(vb1))
    results.append(vb1)

    # ------------------------ F2 ------------------------
    vb2 = None
    if not args.skip_f2:
        print()
        print("=" * 70)
        print("Family F2 (Gaussian-polynomial)")
        print("=" * 70)
        if args.skip_optim:
            f2_best_params = f2.F2Params(alpha=4.0, c=[1.0])
            f2_score = None
        else:
            best = opt.optimise_f2(N=2, n_restarts=args.restarts_f2, verbose=args.verbose)
            if best["params"] is None:
                print("F2: no feasible point found.")
                f2_best_params = f2.F2Params(alpha=4.0, c=[1.0])
                f2_score = None
            else:
                f2_best_params = best["params"]
                f2_score = best["score"]
                print(f"F2 float64 best score: {f2_score:.6f}")
                print(f"    params: {f2_best_params}")

        print(">>> Running rigorous verification (F2)...")
        vb2 = ver.verify_f2(
            f2_best_params,
            rel_tol=args.rel_tol,
            n_subdiv=args.subdiv,
            precision_bits=args.precision_bits,
        )
        print(ver.format_ball(vb2))
        results.append(vb2)

    # ------------------------ F3 ------------------------
    print()
    print("=" * 70)
    print("Family F3 (Vaaler/Beurling-Selberg)")
    print("=" * 70)
    print("SKIPPED: F3 not fully implemented in this version.")
    print("See delsarte_dual/family_f3_vaaler.py docstring.")

    # ------------------------ Summary ------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_lb = None
    for r in results:
        if r.pd_certified:
            lb = float(r.lb_low)
            print(f"[{r.family}] pd_certified   lb_low={lb:.10f}  width={float(r.lb_high - r.lb_low):.2e}")
            if best_lb is None or lb > best_lb:
                best_lb = lb
        else:
            print(f"[{r.family}] pd FAILED   reason={r.pd_reason}")

    print()
    if best_lb is None:
        print("No certified bound produced.")
        status = "no_cert"
    elif best_lb > CURRENT_RECORD:
        print(f"NEW RECORD: {best_lb:.10f}  >  {CURRENT_RECORD} (MV)")
        status = "new_record"
    else:
        print(f"Best certified lb_low: {best_lb:.10f}   (did NOT beat {CURRENT_RECORD})")
        status = "below_record"

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")
    return 0 if status != "no_cert" else 1


if __name__ == "__main__":
    sys.exit(main())
