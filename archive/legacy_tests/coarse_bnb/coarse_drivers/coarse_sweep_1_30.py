#!/usr/bin/env python
"""Sweep script: find coarse-grid parameters that rigorously prove C_{1a} >= 1.30.

This would BEAT the current best lower bound of 1.2802 (Matolcsi & Vinuesa 2010).

The coarse grid has NO correction term at grid points (Theorem 1), so it only
needs TV >= 1.30 (vs fine grid needing TV >= 1.40).  The cost is box certification.

Strategy from 1.10 success: d0=6 proved everything at L0 with box cert passing.
For 1.30, we need higher d0 and/or larger S since pruning is harder.

Usage:
    python tests/coarse_sweep_1_30.py
    python tests/coarse_sweep_1_30.py --phase 2   # run only phase 2
"""
import argparse
import sys
import os
import time
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

import numpy as np
from pruning import count_compositions


def run_single(d0, S, c_target, max_levels=6, n_workers=None, timeout_s=300):
    """Run one coarse cascade config, return result dict."""
    import signal

    from run_cascade import run_cascade

    t0 = time.time()
    try:
        info = run_cascade(
            n_half=d0 / 2.0,
            m=20,
            c_target=c_target,
            max_levels=max_levels,
            n_workers=n_workers,
            verbose=False,
            output_dir='data_sweep',
            coarse_S=S,
            d0=d0,
        )
    except Exception as e:
        return {
            'd0': d0, 'S': S, 'c_target': c_target,
            'proven': False, 'box_certified': False,
            'error': str(e), 'elapsed': time.time() - t0,
        }
    elapsed = time.time() - t0

    proven = 'proven_at' in info
    box_ok = info.get('box_certified', False)

    l0_net = info.get('l0', {}).get('min_cert_net', None)
    l0_box = info.get('l0', {}).get('box_certified', None)
    l0_surv = info.get('l0_survivors', 0)

    worst_net = l0_net if l0_net is not None else 1e30
    for lv in info.get('levels', []):
        ln = lv.get('min_cert_net')
        if ln is not None and ln < worst_net:
            worst_net = ln

    return {
        'd0': d0, 'S': S, 'c_target': c_target,
        'proven': proven, 'proven_at': info.get('proven_at'),
        'box_certified': box_ok,
        'l0_survivors': l0_surv,
        'l0_min_net': l0_net, 'l0_box_ok': l0_box,
        'worst_net': worst_net,
        'elapsed': elapsed,
        'n_levels': len(info.get('levels', [])),
    }


def clean_checkpoints():
    d = 'data_sweep'
    if not os.path.isdir(d):
        return
    for f in os.listdir(d):
        if f.startswith('checkpoint_') or f == 'checkpoint_meta.json':
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass


def estimate_l0_comps(d0, S):
    """Estimate L0 compositions (canonical ≈ total/2)."""
    n = count_compositions(d0, S)
    return n // 2


def print_row(r):
    l0s = r.get('l0_survivors', '?')
    wn = r.get('worst_net')
    wn_str = f"{wn:>10.6f}" if wn is not None else "       N/A"
    proven_str = "YES" if r.get('proven') else "no"
    box_str = "YES" if r.get('box_certified') else "no"
    at_str = r.get('proven_at', '-') or '-'
    err = r.get('error', '')
    if err:
        at_str = 'ERR'
    print(f"{r['d0']:>4} {r['S']:>5} {r['c_target']:>6.3f} "
          f"{proven_str:>7} {at_str:>4} {box_str:>7} "
          f"{l0s:>8} {wn_str} {r['elapsed']:>8.1f}s"
          f"{'  ' + err[:40] if err else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=0,
                        help='Run specific phase (0=all)')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    os.makedirs('data_sweep', exist_ok=True)
    results = []

    hdr = (f"{'d0':>4} {'S':>5} {'c':>6} {'proven':>7} {'at':>4} "
           f"{'box_ok':>7} {'L0surv':>8} {'worst_net':>10} {'time':>9}")
    sep = "-" * 75

    # ================================================================
    # PHASE 1: Quick feasibility — which (d0, c_target) combos prune at L0?
    # Use small S to keep L0 fast. We want to find the sweet spot.
    # ================================================================
    if args.phase in (0, 1):
        print("=" * 75)
        print("PHASE 1: Quick feasibility scan")
        print("=" * 75)
        print(hdr)
        print(sep)

        phase1_configs = []
        for c in [1.28, 1.29, 1.30, 1.31, 1.32, 1.35]:
            for d0 in [4, 6, 8, 10, 12, 14, 16]:
                for S in [20, 30, 40, 50]:
                    n = estimate_l0_comps(d0, S)
                    if n > 100_000_000:
                        continue
                    phase1_configs.append((d0, S, c))

        for d0, S, c in phase1_configs:
            clean_checkpoints()
            r = run_single(d0, S, c, max_levels=3, n_workers=args.workers)
            results.append(r)
            print_row(r)
            if r.get('box_certified'):
                print(f"\n*** RIGOROUS PROOF: C_{{1a}} >= {c} ***\n")

    # ================================================================
    # PHASE 2: Zoom in on promising combos from Phase 1.
    # For configs that proved at L0 with positive or near-positive
    # box cert, try larger S to push box cert positive.
    # ================================================================
    if args.phase in (0, 2):
        print()
        print("=" * 75)
        print("PHASE 2: Targeted S sweep on promising d0 values")
        print("=" * 75)
        print(hdr)
        print(sep)

        # From 1.10 results: d0=6 was magic. For 1.30, likely need d0=8-16.
        # Focus on configs that proved grid-point but need box cert tuning.
        phase2_configs = []
        for c in [1.28, 1.29, 1.30]:
            for d0 in [6, 8, 10, 12, 14]:
                for S in [25, 30, 35, 40, 50, 60, 75, 100]:
                    n = estimate_l0_comps(d0, S)
                    if n > 200_000_000:
                        continue
                    phase2_configs.append((d0, S, c))

        for d0, S, c in phase2_configs:
            clean_checkpoints()
            r = run_single(d0, S, c, max_levels=4, n_workers=args.workers)
            results.append(r)
            print_row(r)
            if r.get('box_certified'):
                print(f"\n*** RIGOROUS PROOF: C_{{1a}} >= {c} ***\n")

    # ================================================================
    # PHASE 3: Push higher — try c_target up to 1.40 with large d0
    # ================================================================
    if args.phase in (0, 3):
        print()
        print("=" * 75)
        print("PHASE 3: Push for highest provable bound")
        print("=" * 75)
        print(hdr)
        print(sep)

        phase3_configs = []
        for c in [1.30, 1.32, 1.35, 1.38, 1.40]:
            for d0 in [10, 12, 14, 16, 20]:
                for S in [15, 20, 25, 30]:
                    n = estimate_l0_comps(d0, S)
                    if n > 500_000_000:
                        continue
                    phase3_configs.append((d0, S, c))

        for d0, S, c in phase3_configs:
            clean_checkpoints()
            r = run_single(d0, S, c, max_levels=3, n_workers=args.workers)
            results.append(r)
            print_row(r)
            if r.get('box_certified'):
                print(f"\n*** RIGOROUS PROOF: C_{{1a}} >= {c} ***\n")

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)

    box_results = [r for r in results if r.get('box_certified')]
    proven_results = [r for r in results if r.get('proven')]

    if box_results:
        print("\n*** RIGOROUS PROOFS (box cert passes): ***")
        # Group by c_target
        by_c = {}
        for r in box_results:
            by_c.setdefault(r['c_target'], []).append(r)
        for c in sorted(by_c.keys(), reverse=True):
            best = min(by_c[c], key=lambda r: r['elapsed'])
            print(f"  C_{{1a}} >= {c}: d0={best['d0']}, S={best['S']}, "
                  f"time={best['elapsed']:.1f}s, "
                  f"worst_net={best['worst_net']:.6f}")
        best_c = max(r['c_target'] for r in box_results)
        print(f"\n  BEST RIGOROUS BOUND: C_{{1a}} >= {best_c}")
    else:
        print("\nNo rigorous proofs found.")

    # Show closest misses for grid-point proofs
    gp_only = [r for r in proven_results if not r.get('box_certified')]
    if gp_only:
        # Find closest to box cert passing per c_target
        by_c = {}
        for r in gp_only:
            wn = r.get('worst_net', -1e30)
            if wn is not None:
                by_c.setdefault(r['c_target'], []).append(r)
        print("\nClosest grid-point proofs (need box cert):")
        for c in sorted(by_c.keys(), reverse=True)[:5]:
            best = max(by_c[c], key=lambda r: r.get('worst_net', -1e30))
            wn = best.get('worst_net', -1e30)
            print(f"  c={c}: d0={best['d0']}, S={best['S']}, "
                  f"worst_net={wn:.6f} (gap={abs(wn):.6f})")

    # Save
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f'data_sweep/coarse_sweep_1_30_{ts}.json'
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
