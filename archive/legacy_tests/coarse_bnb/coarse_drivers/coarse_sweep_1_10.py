#!/usr/bin/env python
"""Sweep script: find coarse-grid parameters that rigorously prove C_{1a} >= 1.10.

Tests combinations of (d0, S, c_target) to find configurations where:
  1. The cascade converges (all compositions pruned)
  2. Box certification passes at ALL levels (margin > cell_var + quad_corr)

When box cert passes everywhere, the proof is RIGOROUS for continuous
distributions, not just grid points.

Strategy:
  - Grid-point pruning is easy (works for small S).
  - Box cert is the bottleneck: needs margin > cell_var + quad_corr.
  - cell_var ~ O(d/S), so large S helps.
  - Fewer cascade levels = fewer chances for box cert to fail.
  - Starting at larger d0 means L0 handles the hardest dimension
    (L0 tries ALL windows for best box cert, cascade uses first).
  - Slightly lower c_target gives more margin.

Usage:
    python tests/coarse_sweep_1_10.py           # quick sweep
    python tests/coarse_sweep_1_10.py --full     # exhaustive sweep
"""
import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

import numpy as np


def run_single(d0, S, c_target, max_levels=6, n_workers=None):
    """Run one coarse cascade config, return result dict."""
    # Import here so Numba JIT doesn't slow down the import
    from run_cascade import run_cascade

    t0 = time.time()
    info = run_cascade(
        n_half=d0 / 2.0,
        m=20,  # not used in coarse mode, but required param
        c_target=c_target,
        max_levels=max_levels,
        n_workers=n_workers,
        verbose=False,
        output_dir='data_sweep',
        coarse_S=S,
        d0=d0,
    )
    elapsed = time.time() - t0

    proven = 'proven_at' in info
    box_ok = info.get('box_certified', False)

    # Collect per-level box cert details
    l0_net = info.get('l0', {}).get('min_cert_net', None)
    l0_box = info.get('l0', {}).get('box_certified', None)
    level_nets = []
    for lv in info.get('levels', []):
        level_nets.append({
            'level': lv.get('level'),
            'd_child': lv.get('d_child'),
            'survivors': lv.get('survivors_out'),
            'min_cert_net': lv.get('min_cert_net'),
            'box_certified': lv.get('box_certified'),
        })

    return {
        'd0': d0,
        'S': S,
        'c_target': c_target,
        'proven': proven,
        'proven_at': info.get('proven_at'),
        'box_certified': box_ok,
        'l0_min_net': l0_net,
        'l0_box_ok': l0_box,
        'levels': level_nets,
        'elapsed': elapsed,
        'n_survivors_final': info.get('l0_survivors', 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Sweep coarse-grid parameters to prove C_{1a} >= 1.10')
    parser.add_argument('--full', action='store_true',
                        help='Exhaustive sweep (slower)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: auto)')
    args = parser.parse_args()

    os.makedirs('data_sweep', exist_ok=True)

    print("=" * 75)
    print("COARSE-GRID SWEEP: Finding parameters to prove C_{1a} >= 1.10")
    print("=" * 75)

    # Clean up checkpoint files between runs
    def clean_checkpoints():
        for f in os.listdir('data_sweep'):
            if f.startswith('checkpoint_') or f.endswith('.json'):
                try:
                    os.remove(os.path.join('data_sweep', f))
                except OSError:
                    pass

    # ================================================================
    # PHASE 1: Quick scan — find which (d0, S) combos converge
    # ================================================================
    print("\n--- Phase 1: Quick convergence scan ---")
    print(f"{'d0':>4} {'S':>5} {'c':>6} {'proven':>7} {'at':>4} "
          f"{'box_ok':>7} {'L0 net':>10} {'time':>8}")
    print("-" * 65)

    results = []

    # Test grid: d0 in {2, 3, 4, 6}, S in {50..500}, c in {1.10, 1.09, 1.08}
    if args.full:
        d0_vals = [2, 3, 4, 6, 8]
        S_vals = [50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
        c_vals = [1.10, 1.09, 1.08, 1.05]
    else:
        d0_vals = [2, 3, 4, 6]
        S_vals = [50, 100, 200, 300, 500]
        c_vals = [1.10, 1.09, 1.08]

    for c_target in c_vals:
        for d0 in d0_vals:
            for S in S_vals:
                # Skip obviously slow combos
                from pruning import count_compositions
                n_comps = count_compositions(d0, S)
                if n_comps > 50_000_000:
                    continue  # too many L0 compositions

                clean_checkpoints()
                try:
                    r = run_single(d0, S, c_target, n_workers=args.workers)
                except Exception as e:
                    r = {'d0': d0, 'S': S, 'c_target': c_target,
                         'proven': False, 'box_certified': False,
                         'proven_at': 'ERR', 'l0_min_net': None,
                         'elapsed': 0, 'error': str(e)}

                results.append(r)

                l0_net_str = (f"{r.get('l0_min_net', 0):>10.6f}"
                              if r.get('l0_min_net') is not None
                              else "       N/A")
                proven_str = "YES" if r.get('proven') else "no"
                box_str = "YES" if r.get('box_certified') else "no"
                at_str = r.get('proven_at', '-') or '-'

                print(f"{d0:>4} {S:>5} {c_target:>6.2f} {proven_str:>7} "
                      f"{at_str:>4} {box_str:>7} {l0_net_str} "
                      f"{r['elapsed']:>7.2f}s")

                # Early out: if we found a rigorous proof, celebrate
                if r.get('box_certified'):
                    print(f"\n*** RIGOROUS PROOF FOUND: d0={d0}, S={S}, "
                          f"c_target={c_target} ***")

    # ================================================================
    # PHASE 2: Report best results
    # ================================================================
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)

    proven_results = [r for r in results if r.get('proven')]
    box_results = [r for r in results if r.get('box_certified')]

    if box_results:
        print("\n*** RIGOROUS PROOFS (box cert passes at all levels): ***")
        for r in box_results:
            print(f"  d0={r['d0']}, S={r['S']}, c_target={r['c_target']}, "
                  f"proven at {r['proven_at']}, time={r['elapsed']:.2f}s")
    else:
        print("\nNo rigorous proofs found (box cert failed everywhere).")

    if proven_results:
        print(f"\nGrid-point proofs ({len(proven_results)} configs converged):")
        # Find the one closest to passing box cert
        best = None
        best_net = -1e30
        for r in proven_results:
            # Get worst box cert net across all levels
            worst_net = r.get('l0_min_net', -1e30) or -1e30
            for lv in r.get('levels', []):
                ln = lv.get('min_cert_net')
                if ln is not None and ln < worst_net:
                    worst_net = ln
            if worst_net > best_net:
                best_net = worst_net
                best = r

        if best:
            print(f"\n  Closest to rigorous proof:")
            print(f"    d0={best['d0']}, S={best['S']}, "
                  f"c_target={best['c_target']}")
            print(f"    proven at {best.get('proven_at')}, "
                  f"worst box cert net = {best_net:.6f}")
            print(f"    Need net >= 0 for rigorous proof "
                  f"(gap = {abs(best_net):.6f})")

    # Save results
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f'data_sweep/coarse_sweep_{ts}.json'
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
