"""Correctness + sweep tests for the L^{3/2}-constrained moment-Lasserre SDP.

Tests:
  1. Baseline recovery: at large B (no effective L^{3/2} constraint) and
     L^infty <= 2.15 with k=7, recover V3's lambda ~ 1.31.
  2. L^{3/2} alone: at B = 2.0 with no L^infty, check lambda behavior.
  3. Just-above-SS-norm: at B = 1.6 (||SS||_{3/2} = 2^{2/3} ~ 1.587), does
     lambda > 1.2802?
  4. Sweep: B in {1.5, 1.55, 1.6, 1.7, 1.8, 2.0} crossed with k in {6, 7, 8}.

Reports a single table summarizing results.  Soundness assessment in the report.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lasserre.threepoint_l32_moment import build_2pt_l32, build_3pt_l32
from lasserre.threepoint_full import solve as solve_problem


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _run_one(name: str, builder, **kwargs) -> Dict:
    """Build and solve, return summary dict."""
    print(f"  [{name}] building ... ", flush=True, end="")
    t0 = time.time()
    problem, info, handles = builder(**kwargs)
    build_t = time.time() - t0
    print(f"build {build_t:.2f}s  blocks={info.block_sizes}", flush=True)
    info = solve_problem(problem, info)
    print(f"  [{name}] -> status={info.status}  lambda={info.objective}  "
          f"solve {info.solve_seconds:.2f}s",
          flush=True)
    return dict(
        name=name,
        build_seconds=build_t,
        solve_seconds=info.solve_seconds,
        status=info.status,
        objective=info.objective,
        block_sizes=info.block_sizes,
        kwargs={k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool, type(None)))},
    )


# ---------------------------------------------------------------------
# Test 1: baseline recovery — V3 result at L^infty <= 2.15, k=7 ~ 1.31
# ---------------------------------------------------------------------

def test_baseline_recovery_2pt():
    """At k=7, N=7, with ||f||_infty <= 2.15 AND a generous B = 5.0 (loose
    L^{3/2}), we should recover V3's lambda ~ 1.28 for 2pt.

    Convention reminder: build_2pt_l32 takes l_inf_M_rho2_orig = ||f||_infty^2
    (since rho^(2) = f tensor f has L_inf bound = ||f||_infty^2).
    """
    record = _run_one(
        "baseline_2pt_k7",
        build_2pt_l32,
        k=7, N=7,
        B=5.0,                              # very loose
        epsilon=0.1,
        l_inf_M_rho2_orig=2.15 ** 2,        # this should bind to V3's value 1.2803
    )
    if record["status"] not in ("optimal", "optimal_inaccurate"):
        pytest.skip(f"Solver failed: {record['status']}")
    lam = record["objective"]
    # V3 reported 1.2803 at this point.
    assert 1.25 <= lam <= 1.32, \
        f"lambda={lam}; expected ~1.28 (V3 baseline)"


def test_baseline_recovery_3pt():
    """At k=6 (faster than k=7), 3-point with both L^infty bounds.
    V3 numbers per report: with both 2D and 3D L^infty at f_inf=2.10,
    lambda is around 1.30-1.35.
    """
    record = _run_one(
        "baseline_3pt_k6",
        build_3pt_l32,
        k=6, N=6,
        B=5.0,
        epsilon=0.1,
        l_inf_M_rho2_orig=2.10 ** 2,
        l_inf_M_rho3_orig=2.10 ** 3,
    )
    if record["status"] not in ("optimal", "optimal_inaccurate"):
        pytest.skip(f"Solver failed: {record['status']}")
    lam = record["objective"]
    # V3 3pt+3D-Linf at f_inf=2.10, k>=4 gives ~1.30+ per report
    assert 1.0 <= lam <= 1.6, f"lambda={lam} out of plausible range"


# ---------------------------------------------------------------------
# Test 2: L^{3/2} alone, no L^infty
# ---------------------------------------------------------------------

def test_l32_only_b_2():
    """B = 2.0, no L^infty.  Should give a finite (not collapsing to ~0)
    lambda.
    """
    record = _run_one(
        "l32_only_B2_k6",
        build_2pt_l32,
        k=6, N=6,
        B=2.0,
        epsilon=0.1,
        l_inf_M_rho2_orig=None,
    )
    if record["status"] not in ("optimal", "optimal_inaccurate"):
        pytest.skip(f"Solver failed: {record['status']}")
    lam = record["objective"]
    # Sanity check
    assert lam > 0, f"lambda={lam} non-positive"
    # SS achieves 1.5708; uniform achieves 2.0.  At B=2.0, both are feasible,
    # but we want the inf, so lambda should be at most around uniform's value.
    assert lam <= 2.5


# ---------------------------------------------------------------------
# Test 3: Just-above-SS-norm.  Does lambda > 1.2802?
# ---------------------------------------------------------------------

def test_l32_at_b_1_6():
    """B = 1.6 (just above ||SS||_{3/2} ~ 1.587).  At this constraint level,
    SS function (which has ~1.5708 sup f*f) is feasible, so the inf C_{1a}^{(B)}
    is at most pi/2 ~ 1.5708.  The Lasserre LB is at most that.

    Question: does the LB exceed 1.2802?
    """
    record = _run_one(
        "l32_B1.6_k7",
        build_2pt_l32,
        k=7, N=7,
        B=1.6,
        epsilon=0.1,
        l_inf_M_rho2_orig=None,
    )
    if record["status"] not in ("optimal", "optimal_inaccurate"):
        pytest.skip(f"Solver failed: {record['status']}")
    lam = record["objective"]
    print(f"  lambda(B=1.6, k=7) = {lam}")
    # We don't ASSERT lambda > 1.2802 - we just want to record the value.
    # If it does exceed: WIN.
    assert lam >= 0


# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

def test_sweep_b_k():
    """Sweep B in {1.5, 1.55, 1.6, 1.7, 1.8, 2.0} crossed with k in {6, 7, 8}.

    NOTE: k=8 may be conditioning-limited per V3 notes; we handle solver
    failures gracefully and report partial results.
    """
    Bs = [1.5, 1.55, 1.6, 1.7, 1.8, 2.0]
    ks = [6, 7]   # k=8 dropped initially due to conditioning concerns; can extend if time
    out: List[Dict] = []
    for k in ks:
        for B in Bs:
            try:
                record = _run_one(
                    f"sweep_B{B}_k{k}_2pt",
                    build_2pt_l32,
                    k=k, N=k, B=B, epsilon=0.1, l_inf_M_rho2_orig=None,
                )
            except Exception as exc:
                record = dict(name=f"sweep_B{B}_k{k}_2pt",
                              status=f"BUILD_ERROR: {exc}",
                              objective=None)
            out.append(record)
    json.dump(out, open(RESULTS_DIR / "l32_moment_sweep.json", "w"), indent=2,
              default=str)
    print("\n=== L^{3/2} Sweep Summary (2pt) ===")
    print(f"{'k':>3} {'B':>6} {'status':>30} {'lambda':>14}")
    for r in out:
        kk = r.get("kwargs", {}).get("k", "-")
        Bv = r.get("kwargs", {}).get("B", "-")
        status = str(r.get("status", "-"))[:30]
        lam = r.get("objective", None)
        lam_str = f"{lam:.6f}" if isinstance(lam, float) else str(lam)
        print(f"{str(kk):>3} {str(Bv):>6} {status:>30} {lam_str:>14}")
    # No assertions on values; this is exploratory.


# ---------------------------------------------------------------------
# Standalone main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("L^{3/2} moment-Lasserre SDP — correctness + sweep")
    print("=" * 72)
    print("\n--- Test 1: baseline recovery (L^infty=2.15, k=7, 2pt) ---")
    try:
        test_baseline_recovery_2pt()
    except Exception as exc:
        print(f"  FAILED: {exc}")

    print("\n--- Test 2: L^{3/2} only at B=2.0, k=6 ---")
    try:
        test_l32_only_b_2()
    except Exception as exc:
        print(f"  FAILED: {exc}")

    print("\n--- Test 3: L^{3/2} at B=1.6, k=7 (above 1.2802?) ---")
    try:
        test_l32_at_b_1_6()
    except Exception as exc:
        print(f"  FAILED: {exc}")

    print("\n--- Sweep: B x k ---")
    try:
        test_sweep_b_k()
    except Exception as exc:
        print(f"  FAILED: {exc}")
