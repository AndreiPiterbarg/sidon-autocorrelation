#!/usr/bin/env python
"""High-d0 sweep: find configs where L0 prunes EVERYTHING at c_target >= 1.28.

Key insight from 1.10 proof: d0=6 proved at L0 → box cert passed.
For 1.28+, we need higher d0 so the bin structure is rich enough to
prune all compositions at L0 without cascading.

Strategy: sweep d0=8..30 with small S (fewer compositions), check if
L0 prunes all and if box cert passes.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

import numpy as np
from pruning import count_compositions


def run_l0_only(d0, S, c_target, n_workers=None):
    """Run just L0 and return stats."""
    from run_cascade import run_level0
    t0 = time.time()
    try:
        r = run_level0(d0 / 2.0, 20, c_target, verbose=False,
                       d0=d0, coarse_S=S)
    except Exception as e:
        return {'error': str(e), 'elapsed': time.time() - t0}
    elapsed = time.time() - t0
    return {
        'survivors': r['n_survivors'],
        'min_net': r.get('min_cert_net'),
        'box_ok': r.get('box_certified', False),
        'elapsed': elapsed,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max_comps', type=int, default=50_000_000,
                        help='Max L0 compositions to attempt')
    args = parser.parse_args()

    print("=" * 80)
    print("HIGH-d0 SWEEP: Find L0-complete proofs for C_{1a} >= 1.28+")
    print("=" * 80)
    print(f"{'d0':>4} {'S':>4} {'c':>6} {'comps':>12} {'L0surv':>8} "
          f"{'box_ok':>7} {'min_net':>10} {'time':>8}")
    print("-" * 75)

    best_rigorous = {}  # c_target -> best result

    for c_target in [1.25, 1.26, 1.27, 1.28, 1.29, 1.30, 1.31, 1.32,
                     1.33, 1.34, 1.35, 1.38, 1.40]:
        found_for_c = False
        for d0 in range(4, 36, 2):
            if found_for_c:
                break
            for S in [10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100]:
                n_comps = count_compositions(d0, S)
                if n_comps > args.max_comps:
                    continue
                if n_comps < 100:
                    continue

                r = run_l0_only(d0, S, c_target, n_workers=args.workers)

                if 'error' in r:
                    print(f"{d0:>4} {S:>4} {c_target:>6.2f} "
                          f"{n_comps:>12,} {'ERR':>8} "
                          f"{'':>7} {'':>10} "
                          f"{r['elapsed']:>7.1f}s  {r['error'][:30]}")
                    continue

                surv = r['survivors']
                mn = r.get('min_net')
                mn_str = f"{mn:>10.6f}" if mn is not None else "       N/A"
                box = "YES" if r.get('box_ok') else "no"

                tag = ""
                if surv == 0 and r.get('box_ok'):
                    tag = " *** RIGOROUS ***"
                    best_rigorous[c_target] = {
                        'd0': d0, 'S': S, 'min_net': mn,
                        'time': r['elapsed']
                    }
                    found_for_c = True
                elif surv == 0:
                    tag = " (grid-point)"

                print(f"{d0:>4} {S:>4} {c_target:>6.2f} "
                      f"{n_comps:>12,} {surv:>8,} "
                      f"{box:>7} {mn_str} "
                      f"{r['elapsed']:>7.1f}s{tag}")

                if found_for_c:
                    break

    print()
    print("=" * 80)
    if best_rigorous:
        print("RIGOROUS PROOFS FOUND:")
        for c in sorted(best_rigorous.keys(), reverse=True):
            r = best_rigorous[c]
            print(f"  C_{{1a}} >= {c}: d0={r['d0']}, S={r['S']}, "
                  f"min_net={r['min_net']:.6f}, time={r['time']:.1f}s")
        best_c = max(best_rigorous.keys())
        print(f"\n  BEST: C_{{1a}} >= {best_c}")
    else:
        print("No rigorous proofs found.")
    print("=" * 80)


if __name__ == '__main__':
    main()
