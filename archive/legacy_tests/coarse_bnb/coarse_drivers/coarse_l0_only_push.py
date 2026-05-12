#!/usr/bin/env python
"""Push for highest c where L0 prunes everything AND box cert passes.

From previous data: d0=6, S=50, c=1.10 works (box cert +0.048).
Now push c upward with d0=6 and find the maximum c where box cert passes.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

from pruning import count_compositions


def run_l0(d0, S, c_target):
    from run_cascade import run_level0
    t0 = time.time()
    r = run_level0(d0 / 2.0, 20, c_target, verbose=False, d0=d0, coarse_S=S)
    elapsed = time.time() - t0
    return {
        'survivors': r['n_survivors'],
        'min_net': r.get('min_cert_net'),
        'box_ok': r.get('box_certified', False),
        'elapsed': elapsed,
    }


def main():
    print("=" * 80)
    print("L0-ONLY PUSH: find max c with L0 pruning + box cert")
    print("=" * 80)
    print(f"{'d0':>4} {'S':>5} {'c':>6} {'comps':>12} {'surv':>8} "
          f"{'box':>5} {'min_net':>10} {'time':>8}")
    print("-" * 70)

    best = {}  # d0 -> best c with box cert

    # Binary-search style: for each d0, find max c
    for d0 in [4, 6, 8, 10, 12, 14]:
        for S in [20, 30, 40, 50, 75, 100]:
            n = count_compositions(d0, S)
            if n > 100_000_000:
                continue

            # Sweep c from high to low
            for c100 in range(150, 100, -1):
                c = c100 / 100.0
                r = run_l0(d0, S, c)
                surv = r['survivors']
                mn = r.get('min_net')
                mn_s = f"{mn:>10.6f}" if mn is not None else "       N/A"
                box = "YES" if r.get('box_ok') else "no"

                # Only print interesting results
                if surv == 0:
                    tag = ""
                    if r.get('box_ok'):
                        tag = " ***"
                        key = d0
                        if key not in best or c > best[key][0]:
                            best[key] = (c, S, mn)
                    print(f"{d0:>4} {S:>5} {c:>6.2f} {n:>12,} {surv:>8} "
                          f"{box:>5} {mn_s} {r['elapsed']:>7.1f}s{tag}")
                    # Found L0-prune: try next S
                    break
                else:
                    # L0 didn't prune everything, stop decreasing c
                    # (lower c is easier, so if this c fails, higher c also fails)
                    break

    print()
    print("=" * 80)
    if best:
        print("BEST RIGOROUS PROOFS PER d0:")
        for d0 in sorted(best.keys()):
            c, S, mn = best[d0]
            print(f"  d0={d0}: C_{{1a}} >= {c} (S={S}, min_net={mn:.6f})")
        overall = max(c for c, _, _ in best.values())
        print(f"\n  OVERALL BEST: C_{{1a}} >= {overall}")
    else:
        print("No rigorous proofs found.")


if __name__ == '__main__':
    main()
