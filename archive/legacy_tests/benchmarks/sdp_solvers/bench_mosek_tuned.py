#!/usr/bin/env python
"""Benchmark driver for lasserre_mosek_tuned.solve_mosek_tuned.

Sweeps d x order x mode and records:
    • final lb, gap closure
    • build time, total solve time, per-solve times
    • speedup ratios relative to baseline

Prints a clean markdown table at the end and writes a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_mosek_tuned import solve_mosek_tuned, val_d_known


def _run_one(d: int, order: int, mode: str, n_bisect: int, verbose: bool
              ) -> Dict[str, Any]:
    t0 = time.time()
    try:
        r = solve_mosek_tuned(
            d, order, mode=mode,
            add_upper_loc=True,
            n_bisect=n_bisect,
            verbose=verbose)
        r['wall_s'] = time.time() - t0
        r['failed'] = False
        return r
    except Exception as exc:
        return {
            'd': d, 'order': order, 'mode': mode, 'failed': True,
            'error': repr(exc), 'wall_s': time.time() - t0,
        }


def _fmt(v: Any, spec: str = '.3f') -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return '-'
    try:
        return format(v, spec)
    except Exception:
        return str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--ds', type=int, nargs='+', default=[4, 6, 8])
    p.add_argument('--orders', type=int, nargs='+', default=[3])
    p.add_argument('--modes', type=str, nargs='+',
                    default=['baseline', 'tuned', 'z2_eq', 'z2_bd'])
    p.add_argument('--n-bisect', type=int, default=12)
    p.add_argument('--out', type=str,
                    default='data/mosek_tuned_bench.json')
    p.add_argument('--skip-d', type=int, nargs='*', default=[],
                    help="Skip baseline at these d's (stress-test only "
                         "the tuned modes).")
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    results: List[Dict[str, Any]] = []

    for order in args.orders:
        for d in args.ds:
            for mode in args.modes:
                if mode == 'baseline' and d in args.skip_d:
                    print(f"-- SKIPPING d={d} L{order} {mode} (skip-d)",
                          flush=True)
                    continue
                print(f"\n>>> d={d} order={order} mode={mode}",
                      flush=True)
                r = _run_one(d, order, mode, args.n_bisect,
                              verbose=not args.quiet)
                results.append(r)
                # Log concise row.
                if r.get('failed'):
                    print(f"    FAILED: {r.get('error')}", flush=True)
                else:
                    print(f"    lb={_fmt(r.get('lb'), '.6f')}  "
                          f"gc={_fmt(r.get('gc_pct'), '.2f')}%  "
                          f"build={_fmt(r.get('build_time_s'), '.2f')}s "
                          f"avg_solve={_fmt(r.get('avg_solve_time_s'), '.3f')}s "
                          f"n_solves={r.get('n_solves')}  "
                          f"total={_fmt(r.get('total_time_s'), '.2f')}s",
                          flush=True)

    # Save.
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'results': [
                {k: v for k, v in r.items()
                 if k not in ('per_solve_times_s', 'history',
                               'build_stats', 'params')}
                for r in results
            ],
            'full_results': results,
        }, f, indent=2, default=str)

    # Print summary table.
    print()
    print('=' * 96)
    print('SUMMARY')
    print('=' * 96)
    header = (
        f"| {'d':>2} | {'ord':>3} | {'mode':<8} | {'lb':>10} | "
        f"{'gc%':>6} | {'build':>7} | {'avg/solve':>9} | "
        f"{'n':>3} | {'total':>8} |"
    )
    sep = '|' + '|'.join('-' * (len(x) + 2)
                           for x in header.strip('|').split('|')) + '|'
    print(header)
    print(sep)
    for r in results:
        if r.get('failed'):
            print(f"| {r['d']:>2} | {r.get('order', '-'):>3} | "
                  f"{r['mode']:<8} | {'FAILED':>10} | {'-':>6} | "
                  f"{'-':>7} | {'-':>9} | {'-':>3} | {'-':>8} |")
            continue
        print(f"| {r['d']:>2} | {r['order']:>3} | {r['mode']:<8} | "
              f"{_fmt(r.get('lb'), '.6f'):>10} | "
              f"{_fmt(r.get('gc_pct'), '.2f'):>6} | "
              f"{_fmt(r.get('build_time_s'), '.2f'):>7}s | "
              f"{_fmt(r.get('avg_solve_time_s'), '.3f'):>8}s | "
              f"{r.get('n_solves', 0):>3} | "
              f"{_fmt(r.get('total_time_s'), '.2f'):>7}s |")

    # Speedup table (tuned/z2_eq/z2_bd vs baseline at same d, order).
    print()
    print('=' * 96)
    print('PER-SOLVE SPEEDUPS vs BASELINE (>1 means faster)')
    print('=' * 96)
    print(f"| {'d':>2} | {'ord':>3} | {'tuned':>9} | {'z2_eq':>9} | "
          f"{'z2_bd':>9} |")
    print('|----|-----|-----------|-----------|-----------|')
    # Organise by (d, order).
    keyed: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for r in results:
        if r.get('failed'):
            continue
        keyed.setdefault(
            (r['d'], r['order']), {})[r['mode']] = r
    for (d, order), by_mode in sorted(keyed.items()):
        base = by_mode.get('baseline')
        if base is None:
            continue
        base_avg = base.get('avg_solve_time_s') or 0.0
        def _sp(m: str) -> str:
            rr = by_mode.get(m)
            if rr is None or not base_avg:
                return '-'
            avg = rr.get('avg_solve_time_s') or 0.0
            if not avg:
                return '-'
            return f"{base_avg / avg:.2f}x"
        print(f"| {d:>2} | {order:>3} | {_sp('tuned'):>9} | "
              f"{_sp('z2_eq'):>9} | {_sp('z2_bd'):>9} |")

    print()
    print(f"JSON saved to {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
