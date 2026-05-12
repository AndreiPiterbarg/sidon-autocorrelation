#!/usr/bin/env python
"""A/B benchmark: monolithic vs clique-decomposed SOS-dual driver.

Measures wall time, peak RSS, bracket midpoint, and — for the clique
path — IPM refines triggered by the knife-edge detector.

Three modes per d:
  (a) ORIG   : solve_mosek_dual,   fixed primary_tol, no cliques
                (matches the pre-change production driver)
  (b) CLIQUE : solve_mosek_dual_cliques, fixed primary_tol, bandwidth=b
                (isolates the clique-decomposition memory win)
  (c) FULL   : solve_mosek_dual_cliques, adaptive tol + refine, bandwidth=b
                (steps 1 + 2 from the design note)

Peak RSS is sampled from /proc/self/status ("VmRSS") or
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` as a fallback, snapshotted
every 0.5s in a background thread during the bisection.

Usage:
    python tests/bench_cliques_vs_monolithic.py \\
        --d 8 --order 3 --bandwidth 4 --n-bisect 12 --json out.json

The per-d benchmark records a single JSON row per mode.  Run across d
values and aggregate for the comparison table.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))


# =====================================================================
# Peak-RSS monitor
# =====================================================================

def _read_vmrss_bytes() -> int:
    """Current process resident-set-size in bytes.

    Prefers /proc/self/status (Linux) for precision; falls back to
    resource.getrusage (portable but in KB on Linux and B on macOS).
    """
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    return int(parts[1]) * 1024  # kB -> B
    except Exception:
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return int(ru)  # macOS: bytes
        return int(ru) * 1024  # Linux: kB
    except Exception:
        return 0


class PeakRSSMonitor:
    """Samples /proc/self/status every interval_s seconds in a thread."""

    def __init__(self, interval_s: float = 0.5):
        self.interval = interval_s
        self._peak = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._peak = _read_vmrss_bytes()

        def _loop():
            while not self._stop.wait(self.interval):
                cur = _read_vmrss_bytes()
                if cur > self._peak:
                    self._peak = cur

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Capture one final sample.
        cur = _read_vmrss_bytes()
        if cur > self._peak:
            self._peak = cur
        return self._peak


# =====================================================================
# Runners
# =====================================================================

def _run_original(d: int, order: int, n_bisect: int, t_hi: Optional[float],
                  primary_tol: float, num_threads: Optional[int],
                  ) -> Dict[str, Any]:
    from lasserre_mosek_dual import solve_mosek_dual
    mon = PeakRSSMonitor()
    mon.start()
    t0 = time.time()
    r = solve_mosek_dual(
        d, order,
        add_upper_loc=False, z2_full=False,
        n_bisect=n_bisect, t_hi=t_hi,
        primary_tol=primary_tol,
        solve_form='dual',
        num_threads=num_threads,
        mosek_log=False,
        reuse_task=False,   # match clique path (rebuild per probe)
        verbose=False,
    )
    wall = time.time() - t0
    peak = mon.stop()
    return {
        'mode': 'orig',
        'd': d, 'order': order,
        'lb': r.get('lb'),
        'bracket_mid': 0.5 * ((r.get('history') or [{'t': r.get('lb')}])[-1]['t'] + r.get('lb'))
                        if r.get('history') else r.get('lb'),
        'n_solves': r.get('n_solves'),
        'total_solve_time_s': r.get('total_solve_time_s'),
        'wall_total_s': wall,
        'peak_rss_bytes': peak,
        'peak_rss_gb': peak / (1024 ** 3),
        'n_uncertain': r.get('n_uncertain'),
        'ok': r.get('ok'),
    }


def _run_clique(d: int, order: int, bandwidth: int, n_bisect: int,
                t_hi: Optional[float], primary_tol: float,
                use_adaptive_tol: bool, refine_on_ambiguous: bool,
                tol_cap: float, tol_rate: float,
                num_threads: Optional[int]) -> Dict[str, Any]:
    from lasserre_mosek_dual_cliques import solve_mosek_dual_cliques
    mon = PeakRSSMonitor()
    mon.start()
    t0 = time.time()
    r = solve_mosek_dual_cliques(
        d, order,
        bandwidth=bandwidth,
        add_upper_loc=False, z2_full=False,
        n_bisect=n_bisect, t_hi=t_hi,
        primary_tol=primary_tol,
        solve_form='dual',
        num_threads=num_threads,
        use_adaptive_tol=use_adaptive_tol,
        tol_cap=tol_cap, tol_rate=tol_rate,
        refine_on_ambiguous=refine_on_ambiguous,
        mosek_log=False,
        verbose=False,
    )
    wall = time.time() - t0
    peak = mon.stop()
    mode = 'full' if (use_adaptive_tol and refine_on_ambiguous) else 'clique'
    return {
        'mode': mode,
        'd': d, 'order': order, 'bandwidth': bandwidth,
        'lb': r.get('lb'),
        'bracket_mid': r.get('bracket_mid'),
        'n_solves': r.get('n_solves'),
        'n_refines': r.get('n_refines'),
        'total_solve_time_s': r.get('total_solve_time_s'),
        'wall_total_s': wall,
        'peak_rss_bytes': peak,
        'peak_rss_gb': peak / (1024 ** 3),
        'n_uncertain': r.get('n_uncertain'),
        'use_adaptive_tol': use_adaptive_tol,
        'refine_on_ambiguous': refine_on_ambiguous,
        'tol_cap': tol_cap,
        'tol_rate': tol_rate,
        'ok': r.get('ok'),
    }


# =====================================================================
# Driver
# =====================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--bandwidth', type=int, required=True,
                   help='Clique bandwidth b (order ≤ b ≤ d-1).')
    p.add_argument('--n-bisect', type=int, default=12)
    p.add_argument('--t-hi', type=float, default=None)
    p.add_argument('--primary-tol', type=float, default=1e-6)
    p.add_argument('--num-threads', type=int, default=None)
    p.add_argument('--tol-cap', type=float, default=1e-4)
    p.add_argument('--tol-rate', type=float, default=0.001)
    p.add_argument('--skip-orig', action='store_true',
                   help='Skip the monolithic baseline (useful when '
                        'monolithic is expected to OOM).')
    p.add_argument('--skip-clique', action='store_true',
                   help='Skip the fixed-tol clique mode; only run FULL.')
    p.add_argument('--skip-full', action='store_true')
    p.add_argument('--json', type=str, default=None)
    p.add_argument('--label', type=str, default='')
    args = p.parse_args()

    results: List[Dict[str, Any]] = []

    def _banner(mode):
        print('\n' + '=' * 72)
        print(f'  [{mode}]  d={args.d} order={args.order} '
              f'bandwidth={args.bandwidth} n_bisect={args.n_bisect}')
        print('=' * 72, flush=True)

    if not args.skip_orig:
        _banner('ORIG  (monolithic, fixed tol)')
        try:
            r = _run_original(
                args.d, args.order, args.n_bisect, args.t_hi,
                args.primary_tol, args.num_threads)
        except Exception as e:
            r = {'mode': 'orig', 'd': args.d, 'order': args.order,
                 'error': f'{type(e).__name__}: {e}'}
            print(f'  ERROR: {r["error"]}', flush=True)
        else:
            print(f'  lb={r["lb"]}  peak={r["peak_rss_gb"]:.2f} GB  '
                  f'wall={r["wall_total_s"]:.1f}s  '
                  f'n_solves={r["n_solves"]}', flush=True)
        results.append(r)
        gc.collect()

    if not args.skip_clique:
        _banner('CLIQUE (fixed tol, bandwidth decomposition only)')
        try:
            r = _run_clique(
                args.d, args.order, args.bandwidth, args.n_bisect,
                args.t_hi, args.primary_tol,
                use_adaptive_tol=False, refine_on_ambiguous=False,
                tol_cap=args.tol_cap, tol_rate=args.tol_rate,
                num_threads=args.num_threads)
        except Exception as e:
            r = {'mode': 'clique', 'd': args.d, 'order': args.order,
                 'bandwidth': args.bandwidth,
                 'error': f'{type(e).__name__}: {e}'}
            print(f'  ERROR: {r["error"]}', flush=True)
        else:
            print(f'  lb={r["lb"]}  peak={r["peak_rss_gb"]:.2f} GB  '
                  f'wall={r["wall_total_s"]:.1f}s  '
                  f'n_solves={r["n_solves"]}', flush=True)
        results.append(r)
        gc.collect()

    if not args.skip_full:
        _banner('FULL   (cliques + adaptive tol + refine)')
        try:
            r = _run_clique(
                args.d, args.order, args.bandwidth, args.n_bisect,
                args.t_hi, args.primary_tol,
                use_adaptive_tol=True, refine_on_ambiguous=True,
                tol_cap=args.tol_cap, tol_rate=args.tol_rate,
                num_threads=args.num_threads)
        except Exception as e:
            r = {'mode': 'full', 'd': args.d, 'order': args.order,
                 'bandwidth': args.bandwidth,
                 'error': f'{type(e).__name__}: {e}'}
            print(f'  ERROR: {r["error"]}', flush=True)
        else:
            print(f'  lb={r["lb"]}  peak={r["peak_rss_gb"]:.2f} GB  '
                  f'wall={r["wall_total_s"]:.1f}s  '
                  f'n_solves={r["n_solves"]}  '
                  f'n_refines={r["n_refines"]}', flush=True)
        results.append(r)
        gc.collect()

    # Compact summary.
    print('\n' + '=' * 72)
    print('  SUMMARY')
    print('=' * 72)
    print(f"{'mode':<8} {'lb':>10} {'wall_s':>8} {'peak_GB':>8} "
          f"{'n_solve':>7} {'n_refine':>8}")
    for r in results:
        if 'error' in r:
            print(f"  {r['mode']:<6} ERROR: {r['error']}")
            continue
        print(f"  {r['mode']:<6} {r.get('lb', 0):>10.6f} "
              f"{r.get('wall_total_s', 0):>8.1f} "
              f"{r.get('peak_rss_gb', 0):>8.2f} "
              f"{r.get('n_solves', 0):>7d} "
              f"{r.get('n_refines', 0):>8}")

    if args.json:
        out = {
            'label': args.label,
            'd': args.d, 'order': args.order,
            'bandwidth': args.bandwidth,
            'n_bisect': args.n_bisect,
            'primary_tol': args.primary_tol,
            'tol_cap': args.tol_cap, 'tol_rate': args.tol_rate,
            'num_threads': args.num_threads,
            'results': results,
            'timestamp': time.time(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.',
                    exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'\n  JSON -> {args.json}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
