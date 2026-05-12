#!/usr/bin/env python
"""Targeted attempt to prove C_{1a} >= 1.30 with d0=6.

Based on diagnostic: d0=6, c=1.30 has worst-case cv_raw=5.20,
margin=0.030 → S_needed ~ 173.  Try S=175, 200, 250.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

from pruning import count_compositions


def run_test(d0, S, c_target):
    from run_cascade import run_level0
    n = count_compositions(d0, S)
    print(f"\n{'='*70}")
    print(f"ATTEMPT: d0={d0}, S={S}, c_target={c_target}")
    print(f"  Compositions: {n:,}")
    print(f"{'='*70}")

    t0 = time.time()
    r = run_level0(d0 / 2.0, 20, c_target, verbose=True, d0=d0, coarse_S=S)
    elapsed = time.time() - t0

    if r['n_survivors'] == 0:
        box_ok = r.get('box_certified', False)
        mn = r.get('min_cert_net', None)
        print(f"\n  GRID-POINT PROOF: all pruned at L0 in {elapsed:.1f}s")
        if box_ok:
            print(f"  *** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
            print(f"  Box cert margin: {mn:.6f}")
        else:
            print(f"  Box cert FAILED (min_net = {mn})")
            print(f"  Need larger S")
    else:
        print(f"  {r['n_survivors']:,} survivors — need cascade")
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    # From diagnostic: d0=6, c=1.30, S_needed~173
    # Try a range of S values
    configs = [
        # Quick tests first
        (6, 100, 1.30),
        (6, 150, 1.30),
        (6, 175, 1.30),
        (6, 200, 1.30),
        # Also try c=1.28 (easier) at lower S
        (6, 100, 1.28),
        (6, 125, 1.28),
        (6, 150, 1.28),
        # And d0=8 with smaller S for comparison
        (8, 30, 1.30),
        (8, 40, 1.30),
        (8, 50, 1.30),
        # d0=10 very small S
        (10, 20, 1.30),
        (10, 25, 1.30),
        # Try higher c if d0=6 works
        (6, 200, 1.32),
        (6, 200, 1.35),
        (6, 250, 1.35),
    ]

    for d0, S, c in configs:
        n = count_compositions(d0, S)
        if n > 5_000_000_000:
            print(f"\nSKIPPING d0={d0}, S={S}: {n:,} compositions (too many)")
            continue
        run_test(d0, S, c)


if __name__ == '__main__':
    main()
