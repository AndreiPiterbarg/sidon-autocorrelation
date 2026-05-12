#!/usr/bin/env python
"""Comprehensive experiment to prove C_{1a} >= 1.30 via coarse cascade.

Uses the PARALLELIZED L0 pruning kernel (prange) for fast sweeps,
and the diagnostic kernel only for small configs where we need
the margin/cell_var breakdown.

Usage:
  python tests/prove_1_30.py                    # Run all phases
  python tests/prove_1_30.py --phase 1          # Just L0 sweep
  python tests/prove_1_30.py --phase 2          # S scaling
  python tests/prove_1_30.py --phase 3          # Full proof attempt
  python tests/prove_1_30.py --phase 4          # Stepping stones
  python tests/prove_1_30.py --c_target 1.28    # Different target
"""
import argparse
import math
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

from pruning import count_compositions


def run_l0(d0, S, c_target):
    """Run L0 via the main cascade code (PARALLELIZED _prune_coarse)."""
    from run_cascade import run_level0
    t0 = time.time()
    r = run_level0(d0 / 2.0, 20, c_target, verbose=False, d0=d0, coarse_S=S)
    elapsed = time.time() - t0
    return {
        'd0': d0, 'S': S, 'c_target': c_target,
        'survivors': r['n_survivors'],
        'min_net': r.get('min_cert_net'),
        'box_ok': r.get('box_certified', False),
        'elapsed': elapsed,
        'n_total': count_compositions(d0, S),
    }


def run_cascade_v2(d0, S, c_target, max_levels=6, verbose=True):
    """Run the v2 coarse cascade."""
    from run_cascade_coarse_v2 import run_cascade
    return run_cascade(d0, S, c_target, max_levels=max_levels, verbose=verbose)


# =====================================================================
# PHASE 1: Fast L0 sweep — which (d0, S) prune everything?
# =====================================================================

def phase1(c_target, max_comps=500_000_000):
    """Find which d0 prunes everything at L0 and measure box cert net."""
    print("=" * 80)
    print(f"PHASE 1: L0 sweep for c_target = {c_target}")
    print("  Goal: Find (d0, S) where L0 prunes everything, check box cert")
    print("  Using parallelized _prune_coarse kernel")
    print("=" * 80)

    configs = []
    for d0 in [6, 8, 10, 12, 14, 16, 20, 24, 32]:
        for S in [8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]:
            n = count_compositions(d0, S)
            if n > max_comps:
                continue
            if n < 50:
                continue
            configs.append((d0, S))

    print(f"\n  Testing {len(configs)} configurations\n")
    print(f"  {'d0':>4} {'S':>5} {'comps':>12} {'surv':>8} "
          f"{'min_net':>10} {'box':>5} {'time':>8}")
    print("  " + "-" * 65)

    results = []
    for d0, S in configs:
        r = run_l0(d0, S, c_target)
        results.append(r)

        box_str = "YES" if r['box_ok'] else "no"
        mn = r['min_net']
        mn_str = f"{mn:>10.6f}" if mn is not None else "       N/A"
        tag = ""
        if r['survivors'] == 0 and r['box_ok']:
            tag = " *** RIGOROUS ***"
        elif r['survivors'] == 0:
            tag = " (grid-point)"

        print(f"  {d0:>4} {S:>5} {r['n_total']:>12,} {r['survivors']:>8,} "
              f"{mn_str} {box_str:>5} {r['elapsed']:>7.1f}s{tag}")

    # Summary
    print(f"\n  --- L0-complete configs (0 survivors) ---")
    l0_complete = [r for r in results if r['survivors'] == 0]
    if l0_complete:
        # Group by d0
        by_d0 = {}
        for r in l0_complete:
            d = r['d0']
            if d not in by_d0 or r['min_net'] > by_d0[d]['min_net']:
                by_d0[d] = r
        print(f"  {'d0':>4} {'best_S':>6} {'min_net':>10} {'box':>5}")
        for d in sorted(by_d0.keys()):
            r = by_d0[d]
            box_str = "YES" if r['box_ok'] else "no"
            mn = r['min_net']
            mn_str = f"{mn:>10.6f}" if mn is not None else "       N/A"
            print(f"  {d:>4} {r['S']:>6} {mn_str} {box_str:>5}")
    else:
        print("  None!")

    return results


# =====================================================================
# PHASE 2: S scaling — push box cert toward 0
# =====================================================================

def phase2(c_target, phase1_results=None, max_comps=2_000_000_000,
           max_time_per=3600):
    """For d0 values that prune at L0, push S until box cert passes."""
    print("\n" + "=" * 80)
    print(f"PHASE 2: S scaling for box cert (c_target = {c_target})")
    print("=" * 80)

    # Find d0 values with 0 survivors
    candidates = {}
    if phase1_results:
        for r in phase1_results:
            if r['survivors'] == 0:
                d = r['d0']
                if d not in candidates or r['min_net'] > candidates[d]['min_net']:
                    candidates[d] = r

    if not candidates:
        print("  No L0-complete configs found. Testing defaults.")
        for d0 in [10, 12, 14, 16, 20]:
            candidates[d0] = {'min_net': -1.0, 'S': 10}

    print(f"  Candidates: {sorted(candidates.keys())}")

    all_results = []
    best_proof = None

    for d0 in sorted(candidates.keys()):
        info = candidates[d0]
        best_s = info.get('S', 10)
        best_net = info.get('min_net', -1)

        print(f"\n  --- d0={d0} (best so far: S={best_s}, net={best_net:.6f}) ---")
        print(f"  {'S':>5} {'comps':>14} {'surv':>8} {'min_net':>10} "
              f"{'box':>5} {'time':>8}")
        print("  " + "-" * 60)

        # Sweep S from small to large
        S_values = sorted(set(
            list(range(max(8, d0), d0 * 8, max(2, d0 // 4))) +
            list(range(d0 * 8, d0 * 20, d0)) +
            [best_s]
        ))

        found = False
        for S in S_values:
            n = count_compositions(d0, S)
            if n > max_comps:
                print(f"  {S:>5} {n:>14,} SKIP (>{max_comps:,})")
                continue
            if n < 50:
                continue

            # Time estimate (rough: 2M comps/sec with parallel kernel)
            est = n / 2_000_000
            if est > max_time_per:
                print(f"  {S:>5} {n:>14,} SKIP (est {est:.0f}s)")
                continue

            r = run_l0(d0, S, c_target)
            all_results.append(r)

            box_str = "YES" if r['box_ok'] else "no"
            mn = r['min_net']
            mn_str = f"{mn:>10.6f}" if mn is not None else "       N/A"
            tag = ""
            if r['survivors'] == 0 and r['box_ok']:
                tag = " *** RIGOROUS ***"
                found = True
                best_proof = r
            elif r['survivors'] == 0:
                tag = " (grid-point)"
            elif r['survivors'] > 0:
                tag = f" ({r['survivors']} surv)"

            print(f"  {S:>5} {r['n_total']:>14,} {r['survivors']:>8,} "
                  f"{mn_str} {box_str:>5} {r['elapsed']:>7.1f}s{tag}")

            if found:
                print(f"\n  *** RIGOROUS: C_{{1a}} >= {c_target} at "
                      f"d0={d0}, S={S} ***")
                break

        if found:
            break

    if best_proof:
        print(f"\n{'='*80}")
        print(f"  RIGOROUS PROOF FOUND: C_{{1a}} >= {c_target}")
        print(f"  d0={best_proof['d0']}, S={best_proof['S']}, "
              f"min_net={best_proof['min_net']:.6f}")
        print(f"{'='*80}")

    return all_results


# =====================================================================
# PHASE 3: Cascade for configs with L0 survivors
# =====================================================================

def phase3(c_target, phase1_results=None, max_levels=4):
    """Try cascade for d0 values with few L0 survivors."""
    print("\n" + "=" * 80)
    print(f"PHASE 3: Cascade attempt (c_target = {c_target})")
    print("=" * 80)

    # Find configs with small number of survivors
    candidates = []
    if phase1_results:
        for r in phase1_results:
            if 0 < r['survivors'] <= 100:
                candidates.append(r)

    if not candidates:
        print("  No configs with 1-100 survivors. Trying defaults.")
        candidates = [
            {'d0': 10, 'S': 12, 'survivors': 2},
            {'d0': 10, 'S': 15, 'survivors': 17},
        ]

    candidates.sort(key=lambda r: r['survivors'])

    results = []
    for info in candidates[:5]:
        d0, S = info['d0'], info['S']
        print(f"\n  --- Cascade: d0={d0}, S={S}, "
              f"L0 survivors={info['survivors']} ---")
        try:
            r = run_cascade_v2(d0, S, c_target,
                               max_levels=max_levels, verbose=True)
            proven = 'proven_at' in r
            box_ok = r.get('box_certified', False)
            results.append({
                'd0': d0, 'S': S, 'c_target': c_target,
                'proven': proven, 'box_ok': box_ok,
                'levels': len(r.get('levels', [])),
            })
            if box_ok:
                print(f"\n  *** RIGOROUS (cascade): C_{{1a}} >= {c_target} ***")
                break
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


# =====================================================================
# PHASE 4: Stepping stones
# =====================================================================

def phase4():
    """Prove progressively higher c values."""
    print("\n" + "=" * 80)
    print("PHASE 4: Stepping stones — prove progressively higher bounds")
    print("=" * 80)

    targets = [1.15, 1.18, 1.20, 1.22, 1.24, 1.25, 1.26, 1.27, 1.28,
               1.29, 1.30, 1.32, 1.35]
    proven = {}

    for c in targets:
        print(f"\n  --- Target: C_{{1a}} >= {c} ---")
        found = False

        for d0 in [6, 8, 10, 12, 14, 16, 20, 24, 32]:
            if found:
                break
            for S in [10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]:
                n = count_compositions(d0, S)
                if n > 500_000_000:
                    continue
                if n < 50:
                    continue

                r = run_l0(d0, S, c)
                if r['survivors'] == 0 and r.get('box_ok'):
                    print(f"  PROVED: d0={d0}, S={S}, "
                          f"net={r['min_net']:.6f}, "
                          f"time={r['elapsed']:.1f}s")
                    proven[c] = (d0, S, r['min_net'])
                    found = True
                    break

        if not found:
            print(f"  NOT PROVED (within budget)")

    print(f"\n{'='*80}")
    print("STEPPING STONE SUMMARY:")
    for c in sorted(proven.keys()):
        d0, S, net = proven[c]
        print(f"  C_{{1a}} >= {c:.2f}: d0={d0}, S={S}, net={net:.6f}")
    if proven:
        print(f"\n  BEST: C_{{1a}} >= {max(proven.keys()):.2f}")
    else:
        print("  No proofs found.")
    print("=" * 80)

    return proven


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_target', type=float, default=1.30)
    parser.add_argument('--phase', type=int, default=0,
                        help='0=all, 1=L0 sweep, 2=S scaling, '
                             '3=cascade, 4=stepping stones')
    parser.add_argument('--max_comps', type=int, default=500_000_000)
    args = parser.parse_args()

    c = args.c_target
    t_start = time.time()

    print(f"\n{'#' * 80}")
    print(f"# EXPERIMENT: Prove C_{{1a}} >= {c}")
    print(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n")

    all_results = {}

    if args.phase in (0, 1):
        r1 = phase1(c, max_comps=args.max_comps)
        all_results['phase1'] = r1
        if any(r.get('box_ok') and r['survivors'] == 0 for r in r1):
            print(f"\n*** ALREADY PROVED IN PHASE 1! ***")

    if args.phase in (0, 2):
        p1 = all_results.get('phase1')
        r2 = phase2(c, phase1_results=p1)
        all_results['phase2'] = r2

    if args.phase in (0, 3):
        p1 = all_results.get('phase1')
        r3 = phase3(c, phase1_results=p1)
        all_results['phase3'] = r3

    if args.phase == 4:
        phase4()

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/3600:.2f} hours)")

    # Save results
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f'data/prove_{c:.2f}_{ts}.json'
    os.makedirs('data', exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    flat = []
    for phase_name, results in all_results.items():
        if isinstance(results, list):
            for r in results:
                r2 = dict(r)
                r2['phase'] = phase_name
                flat.append(r2)

    with open(out_path, 'w') as f:
        json.dump(flat, f, indent=2, default=convert)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
