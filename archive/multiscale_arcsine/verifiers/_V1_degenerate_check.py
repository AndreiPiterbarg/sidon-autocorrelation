"""Agent V1: Verify MultiScaleArcsineKernel reproduces ArcsineKernel baseline
in the degenerate single-component case (lambdas=[1], deltas=[0.138]).

PASS: M_cert matches to >= 6 decimal digits, and K_2, k_1, S_1 also match.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound_alt_kernel.kernels import (
    ArcsineKernel,
    MultiScaleArcsineKernel,
)
from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (
    run_single_kernel,
)


def main():
    delta_q = fmpq(138, 1000)
    print("=" * 70)
    print("Agent V1: MultiScale degenerate-case reproduction check")
    print("=" * 70)
    print(f"delta = {delta_q} (= {float(delta_q.p)/float(delta_q.q)})")

    # Baseline: ArcsineKernel
    k_arc = ArcsineKernel(delta=delta_q)
    # Degenerate multiscale: single component, lambda = 1
    k_ms = MultiScaleArcsineKernel(
        deltas=[delta_q],
        lambdas=[fmpq(1, 1)],
    )

    print(f"\nArcsineKernel name        : {k_arc.name}")
    print(f"MultiScaleArcsineKernel   : {k_ms.name}")

    # Common run-kwargs
    run_kwargs = dict(
        u=fmpq(638, 1000),
        n_coeffs=119,
        n_grid_qp=5001,
        n_cells_min_G=4096,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=fmpq(1, 10**4),
        max_cells_per_M=50000,
        initial_splits=32,
        bochner_max=100,
        prec_bits=192,
        verbose=True,
    )

    print("\n" + "-" * 70)
    print("Running ArcsineKernel (baseline)...")
    print("-" * 70)
    t0 = time.time()
    e_arc = run_single_kernel(k_arc, **run_kwargs)
    t_arc = time.time() - t0

    print("\n" + "-" * 70)
    print("Running MultiScaleArcsineKernel (degenerate)...")
    print("-" * 70)
    t0 = time.time()
    e_ms = run_single_kernel(k_ms, **run_kwargs)
    t_ms = time.time() - t0

    def f(x):
        return None if x is None else float(x)

    summary = {
        "delta": str(delta_q),
        "arcsine": {
            "kernel_name": e_arc.kernel_name,
            "admissible": e_arc.admissible,
            "k1": f(e_arc.tilde_K_1),
            "K2": f(e_arc.K_norm_sq),
            "S1": f(e_arc.S1),
            "min_G": f(e_arc.min_G_cert),
            "gain_a": f(e_arc.gain_a),
            "M_cert": f(e_arc.M_cert),
            "M_cert_q": e_arc.M_cert_q,
            "note": e_arc.note,
            "wall_time": t_arc,
        },
        "multiscale_degenerate": {
            "kernel_name": e_ms.kernel_name,
            "admissible": e_ms.admissible,
            "k1": f(e_ms.tilde_K_1),
            "K2": f(e_ms.K_norm_sq),
            "S1": f(e_ms.S1),
            "min_G": f(e_ms.min_G_cert),
            "gain_a": f(e_ms.gain_a),
            "M_cert": f(e_ms.M_cert),
            "M_cert_q": e_ms.M_cert_q,
            "note": e_ms.note,
            "wall_time": t_ms,
        },
    }

    # Diffs
    def diff(a, b):
        if a is None or b is None:
            return None
        return abs(a - b)

    summary["diffs"] = {
        "k1": diff(summary["arcsine"]["k1"], summary["multiscale_degenerate"]["k1"]),
        "K2": diff(summary["arcsine"]["K2"], summary["multiscale_degenerate"]["K2"]),
        "S1": diff(summary["arcsine"]["S1"], summary["multiscale_degenerate"]["S1"]),
        "M_cert": diff(
            summary["arcsine"]["M_cert"], summary["multiscale_degenerate"]["M_cert"]
        ),
    }

    # Determine pass/fail
    tol_Mcert = 1e-5
    tol_other = 1e-6
    passes = True
    reasons = []
    if summary["diffs"]["M_cert"] is None:
        passes = False
        reasons.append("M_cert None for one or both runs")
    elif summary["diffs"]["M_cert"] > tol_Mcert:
        passes = False
        reasons.append(f"M_cert diff {summary['diffs']['M_cert']:.3e} > {tol_Mcert}")
    if summary["diffs"]["k1"] is not None and summary["diffs"]["k1"] > tol_other:
        passes = False
        reasons.append(f"k1 diff {summary['diffs']['k1']:.3e} > {tol_other}")
    if summary["diffs"]["K2"] is not None and summary["diffs"]["K2"] > tol_other:
        passes = False
        reasons.append(f"K2 diff {summary['diffs']['K2']:.3e} > {tol_other}")
    if summary["diffs"]["S1"] is not None and summary["diffs"]["S1"] > 1e-6:
        passes = False
        reasons.append(f"S1 diff {summary['diffs']['S1']:.3e} > 1e-6")

    summary["verdict"] = "CONFIRM" if passes else "FLAG"
    summary["reasons"] = reasons

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))

    out_path = "_V1_degenerate_check_result.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}")
        traceback.print_exc()
        sys.exit(1)
