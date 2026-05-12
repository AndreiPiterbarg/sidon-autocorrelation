#!/usr/bin/env python
"""Find the MAXIMUM c_target where the coarse grid gives a RIGOROUS proof.

We know d0=6, S=50, c=1.10 works. Binary search upward.
Key: need L0 to prune everything AND box cert to pass.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

from pruning import count_compositions


def test_l0(d0, S, c_target):
    from run_cascade import run_level0
    r = run_level0(d0 / 2.0, 20, c_target, verbose=False, d0=d0, coarse_S=S)
    return {
        'survivors': r['n_survivors'],
        'min_net': r.get('min_cert_net'),
        'box_ok': r.get('box_certified', False),
    }


def main():
    print("=" * 80)
    print("MAXIMUM RIGOROUS c_target SEARCH")
    print("=" * 80)

    # For each (d0, S) that we know works at c=1.10, push c upward
    configs_to_try = [
        # (d0, S) pairs that proved 1.10 rigorously
        (4, 300), (4, 400), (4, 500),
        (6, 50), (6, 75),
        # Also try medium configs
        (4, 100), (4, 150), (4, 200),
        (6, 30), (6, 40),
        (8, 20), (8, 25), (8, 30),
        (10, 15), (10, 20),
        (12, 12), (12, 15),
    ]

    best_overall = 0

    for d0, S in configs_to_try:
        n = count_compositions(d0, S)
        if n > 100_000_000:
            print(f"\nSKIP d0={d0}, S={S}: {n:,} comps")
            continue

        print(f"\n--- d0={d0}, S={S} ({n:,} comps) ---")

        # Binary search: find max c where L0 prunes all + box cert passes
        lo, hi = 1.00, 1.50
        best_c = 0
        best_net = None

        # First check if L0 prunes everything at c=1.10 (our baseline)
        r = test_l0(d0, S, 1.10)
        if r['survivors'] > 0:
            print(f"  c=1.10: {r['survivors']} survivors — skip")
            continue

        # Quick scan in 0.01 steps
        for c100 in range(110, 151):
            c = c100 / 100.0
            r = test_l0(d0, S, c)
            if r['survivors'] > 0:
                print(f"  c={c:.2f}: {r['survivors']} survivors — L0 limit")
                break
            box = "YES" if r.get('box_ok') else "no"
            mn = r.get('min_net')
            mn_s = f"{mn:.6f}" if mn is not None else "N/A"
            if r.get('box_ok'):
                best_c = c
                best_net = mn
            print(f"  c={c:.2f}: 0 surv, box={box}, net={mn_s}")

        if best_c > 0:
            print(f"  BEST RIGOROUS: c={best_c:.2f} (net={best_net:.6f})")
            if best_c > best_overall:
                best_overall = best_c

    print(f"\n{'='*80}")
    print(f"OVERALL BEST RIGOROUS: C_{{1a}} >= {best_overall}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
