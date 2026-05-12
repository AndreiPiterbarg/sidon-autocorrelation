"""Unit tests B and C: solve the same d/order with both monomial and
Chebyshev encodings and confirm lb agrees.

  Test B: d = 4 L2  and  d = 6 L2  — tolerance 1e-8.
  Test C: d = 8 L3  and  d = 10 L3 — tolerance 1e-6  (LOAD-BEARING).

This is the load-bearing test for the Chebyshev reformulation.  If any
test disagrees by more than its tolerance, the implementation is wrong
and must not be deployed.
"""
from __future__ import annotations

import os
import sys
import time

# Propagate test envvars for single-threaded BLAS before NumPy import.
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_phys = max(1, (os.cpu_count() or 2) // 2)
os.environ.setdefault('OMP_NUM_THREADS', str(_phys))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from lasserre_mosek_tuned import solve_mosek_tuned as solve_monomial
from lasserre_mosek_cheby import solve_mosek_tuned as solve_cheby


CASES = [
    # (d, order, tolerance, label)
    (4, 2, 1e-8, 'Test B1: d=4 L2'),
    (6, 2, 1e-8, 'Test B2: d=6 L2'),
    (8, 3, 1e-6, 'Test C1: d=8 L3 (load-bearing)'),
    (10, 3, 1e-6, 'Test C2: d=10 L3 (load-bearing)'),
]


def run_case(d, order, tol, label):
    print(f"\n{'#' * 68}")
    print(f"# {label}")
    print(f"{'#' * 68}")
    t0 = time.time()
    r_mono = solve_monomial(
        d, order, mode='tuned', add_upper_loc=True,
        n_bisect=15, primary_tol=1e-7, verbose=False)
    t_mono = time.time() - t0
    print(f"  MONOMIAL  lb = {r_mono['lb']:.10f}  "
          f"({t_mono:.1f}s, {r_mono['n_solves']} solves)")

    t0 = time.time()
    r_cheb = solve_cheby(
        d, order, mode='tuned', add_upper_loc=True,
        n_bisect=15, primary_tol=1e-7, verbose=False)
    t_cheb = time.time() - t0
    print(f"  CHEBYSHEV lb = {r_cheb['lb']:.10f}  "
          f"({t_cheb:.1f}s, {r_cheb['n_solves']} solves)")

    diff = abs(r_mono['lb'] - r_cheb['lb'])
    ok = diff <= tol
    tag = 'PASS' if ok else 'FAIL'
    print(f"  |diff| = {diff:.3e}   tol = {tol:.0e}   [{tag}]")
    return {
        'label': label, 'd': d, 'order': order, 'tol': tol,
        'lb_mono': r_mono['lb'], 'lb_cheb': r_cheb['lb'],
        'diff': diff, 'ok': ok,
        't_mono_s': t_mono, 't_cheb_s': t_cheb,
        'n_mono': r_mono['n_solves'], 'n_cheb': r_cheb['n_solves'],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default='',
                    help='Comma-separated subset of case labels '
                         '(match by substring), e.g. "B1,B2".')
    args = ap.parse_args()

    if args.only:
        keys = [k.strip() for k in args.only.split(',') if k.strip()]
        cases = [c for c in CASES if any(k in c[3] for k in keys)]
    else:
        cases = CASES

    results = []
    for (d, order, tol, label) in cases:
        r = run_case(d, order, tol, label)
        results.append(r)

    print(f"\n{'=' * 68}")
    print("Summary")
    print(f"{'=' * 68}")
    all_ok = True
    for r in results:
        mark = 'PASS' if r['ok'] else 'FAIL'
        print(f"  [{mark}] {r['label']:40s}  "
              f"lb_mono={r['lb_mono']:.10f}  "
              f"lb_cheb={r['lb_cheb']:.10f}  "
              f"|diff|={r['diff']:.2e}")
        if not r['ok']:
            all_ok = False
    print(f"{'=' * 68}")
    if not all_ok:
        print("FAIL — Chebyshev reformulation disagrees with monomial; "
              "DO NOT DEPLOY.")
        sys.exit(1)
    print("ALL PASS — Chebyshev reformulation mathematically agrees with "
          "monomial.")


if __name__ == '__main__':
    main()
