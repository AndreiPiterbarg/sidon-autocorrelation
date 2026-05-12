#!/usr/bin/env python
"""
EXPERIMENT: Prove C_{1a} >= 1.30 using coarse-grid cascade.

Strategy:
  Phase 1: Verify L0 prunes all at c=1.30 for d0=8,10,12,14,16
           (val(d) > 1.30 for these, so should always prune)
  Phase 2: Run diagnostics to find S_needed for box cert
  Phase 3: Run with optimal S for rigorous proof

Key insight: val(d) decreases with d. val(8)~1.5, val(16)~1.4, val(32)~1.336.
Since val(d) > 1.30 for d<=32, L0 prunes everything. We just need large enough
S for box certification (cell_var ~ 1/S -> 0 as S grows).

Lower d0 = bigger margin (val(d0) - 1.30) but more bins to certify.
Higher d0 = smaller margin but certify at L0 directly (no cascade).
"""
import sys
import os
import time
import math
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

import numpy as np
from pruning import count_compositions


def run_l0_test(d0, S, c_target, verbose=True):
    """Run L0 and return results."""
    from run_cascade import run_level0
    n = count_compositions(d0, S)
    print(f"\n{'='*70}")
    print(f"  d0={d0}, S={S}, c_target={c_target}, compositions={n:,}")
    print(f"{'='*70}")

    if n > 10_000_000_000:
        print(f"  SKIP: too many compositions")
        return None

    t0 = time.time()
    r = run_level0(d0 / 2.0, 20, c_target, verbose=verbose,
                   d0=d0, coarse_S=S)
    elapsed = time.time() - t0

    result = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'survivors': r['n_survivors'],
        'min_net': r.get('min_cert_net'),
        'box_ok': r.get('box_certified', False),
        'elapsed': elapsed,
        'comps': n,
    }

    if r['n_survivors'] == 0:
        mn = r.get('min_cert_net')
        if r.get('box_certified'):
            print(f"  >>> RIGOROUS PROOF: C_{{1a}} >= {c_target} <<<")
            print(f"  Box cert net = {mn:.6f}, time = {elapsed:.1f}s")
        else:
            print(f"  Grid-point proof (all pruned at L0)")
            print(f"  Box cert FAIL: min_net = {mn:.6f}")
            print(f"  Need larger S")
    else:
        print(f"  {r['n_survivors']:,} survivors at L0 — "
              f"need cascade or higher d0")

    return result


def run_cascade_test(d0, S, c_target, max_levels=8, verbose=True):
    """Run full cascade and return results."""
    from run_cascade import run_cascade
    n = count_compositions(d0, S)
    print(f"\n{'='*70}")
    print(f"CASCADE: d0={d0}, S={S}, c_target={c_target}")
    print(f"  L0 compositions: {n:,}")
    print(f"{'='*70}")

    if n > 10_000_000_000:
        print(f"  SKIP: too many compositions")
        return None

    t0 = time.time()
    os.makedirs('data_experiment', exist_ok=True)
    info = run_cascade(
        n_half=d0 / 2.0, m=20, c_target=c_target,
        max_levels=max_levels, n_workers=None,
        verbose=verbose, output_dir='data_experiment',
        coarse_S=S, d0=d0,
    )
    elapsed = time.time() - t0

    proven = 'proven_at' in info
    box_ok = info.get('box_certified', False)

    result = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'proven': proven,
        'box_certified': box_ok,
        'elapsed': elapsed,
        'proven_at': info.get('proven_at', None),
        'levels': info.get('levels', []),
    }

    if proven:
        print(f"\n  GRID-POINT PROOF at {info['proven_at']} "
              f"in {elapsed:.1f}s")
        if box_ok:
            print(f"  >>> RIGOROUS PROOF: C_{{1a}} >= {c_target} <<<")
        else:
            # Find worst level
            worst_net = 1e30
            for lv in info.get('levels', []):
                mn = lv.get('min_cert_net', 1e30)
                if mn < worst_net:
                    worst_net = mn
            l0_net = info.get('l0', {}).get('min_cert_net', 1e30)
            if l0_net < worst_net:
                worst_net = l0_net
            print(f"  Box cert FAIL: worst min_net = {worst_net:.6f}")
    else:
        last = info.get('levels', [{}])[-1] if info.get('levels') else {}
        print(f"  NOT PROVEN after {max_levels} levels")
        print(f"  Last: {last.get('survivors_out', '?')} survivors "
              f"at d={last.get('d_child', '?')}")

    return result


def estimate_s_needed(d0, S_test, c_target):
    """Run diagnostic: estimate S needed for box cert at given d0/c."""
    from run_cascade import run_level0
    n = count_compositions(d0, S_test)
    if n > 500_000_000:
        return None

    r = run_level0(d0 / 2.0, 20, c_target, verbose=False,
                   d0=d0, coarse_S=S_test)

    if r['n_survivors'] > 0:
        return {'d0': d0, 'S_test': S_test, 'c': c_target,
                'status': 'survivors', 'n_surv': r['n_survivors']}

    mn = r.get('min_cert_net', 0)
    if mn >= 0:
        return {'d0': d0, 'S_test': S_test, 'c': c_target,
                'status': 'box_pass', 'min_net': mn}

    # Estimate S_needed from scaling: cell_var ~ cv_raw/S
    # net = margin - cv_raw/S - qc_raw/S^2
    # At S_test: net = mn < 0
    # margin = mn + cv_raw/S_test + qc_raw/S_test^2 (unknown decomposition)
    # Rough estimate: if net(S) = a - b/S - c/S^2, and we know net(S_test),
    # try 2x S and see if that's enough.
    # Better: just report the ratio needed.
    # cell_var dominates, so S_needed ~ S_test * |mn| / (margin)
    # But we don't know margin separately. Just report min_net and suggest
    # trying 2x, 3x S.

    return {'d0': d0, 'S_test': S_test, 'c': c_target,
            'status': 'box_fail', 'min_net': mn,
            'suggestion': f"Try S={2*S_test} or S={3*S_test}"}


def phase1():
    """Phase 1: Verify L0 prunes all at c=1.30 for various d0."""
    print("\n" + "=" * 70)
    print("PHASE 1: Verify L0 prunes all at c=1.30")
    print("(val(d) > 1.30 for d <= ~28, so this should always work)")
    print("=" * 70)

    results = []
    # Start with small S (just verify grid-point pruning)
    configs = [
        # d0, S (small S just to verify L0 prunes all)
        (8,  20),   # 735K comps, fast
        (8,  30),   # 10M comps
        (10, 15),   # 817K comps
        (10, 20),   # 10M comps
        (12, 12),   # 1.4M comps
        (12, 15),   # 7.7M comps
        (14, 10),   # 817K comps
        (14, 12),   # 3.3M comps
        (16, 10),   # 5M comps
        (16, 12),   # 17M comps
        (6,  30),   # 170K comps (smaller d0, higher val)
        (6,  50),   # 3.5M comps
    ]

    for d0, S in configs:
        try:
            r = run_l0_test(d0, S, 1.30)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"  {'d0':>4} {'S':>5} {'surv':>8} {'min_net':>12} {'box':>5} "
          f"{'time':>8}")
    for r in results:
        mn = r['min_net']
        mn_s = f"{mn:.6f}" if mn is not None else "N/A"
        box = "YES" if r['box_ok'] else "no"
        print(f"  {r['d0']:>4} {r['S']:>5} {r['survivors']:>8} "
              f"{mn_s:>12} {box:>5} {r['elapsed']:>7.1f}s")

    # Find which d0 values prune all
    pruned_all = [r for r in results if r['survivors'] == 0]
    if pruned_all:
        print(f"\n  L0 prunes all at c=1.30 for: "
              f"{set(r['d0'] for r in pruned_all)}")
        # Best margin
        best = max(pruned_all, key=lambda r: r['min_net'] or -1e30)
        print(f"  Best min_net: d0={best['d0']}, S={best['S']}, "
              f"net={best['min_net']:.6f}")

    return results


def phase2(phase1_results):
    """Phase 2: Find S_needed for box cert at promising d0 values."""
    print("\n" + "=" * 70)
    print("PHASE 2: Estimate S_needed for box cert at c=1.30")
    print("=" * 70)

    # Find d0 values where L0 pruned all
    good_d0s = set()
    for r in phase1_results:
        if r['survivors'] == 0:
            good_d0s.add(r['d0'])

    if not good_d0s:
        print("  No d0 pruned all at L0 — trying cascade instead")
        return []

    results = []
    # For each good d0, try increasing S
    for d0 in sorted(good_d0s):
        print(f"\n--- d0={d0} ---")
        # Try progressively larger S
        for S in [20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300]:
            n = count_compositions(d0, S)
            if n > 2_000_000_000:
                print(f"  S={S}: {n:,} comps — too many, stopping")
                break

            t0 = time.time()
            r = run_l0_test(d0, S, 1.30, verbose=False)
            if r is None:
                continue

            results.append(r)

            if r['box_ok']:
                print(f"  >>> FOUND: d0={d0}, S={S} gives RIGOROUS PROOF <<<")
                break
            elif r['survivors'] > 0:
                print(f"  S={S}: {r['survivors']} survivors — skip larger S")
                break

    return results


def phase3(phase2_results):
    """Phase 3: Run definitive proof with best parameters."""
    print("\n" + "=" * 70)
    print("PHASE 3: Run definitive proof")
    print("=" * 70)

    # Check if we already have a proof
    proofs = [r for r in phase2_results if r.get('box_ok')]
    if proofs:
        best = min(proofs, key=lambda r: r['elapsed'])
        print(f"\n  Already proven in Phase 2!")
        print(f"  d0={best['d0']}, S={best['S']}, "
              f"net={best['min_net']:.6f}, time={best['elapsed']:.1f}s")
        return proofs

    # If no L0 proof, try cascade approach
    print("\n  No L0 rigorous proof found. Trying cascade approach...")
    print("  (Small S, let cascade run to high dimension)")

    results = []
    cascade_configs = [
        # (d0, S, max_levels) — ordered by expected speed
        (2, 30, 8),    # d0=2: 31 comps at L0, cascade to d=512
        (2, 50, 8),    # d0=2: 51 comps, finer grid
        (3, 20, 7),    # d0=3: 231 comps, cascade to d=384
        (3, 30, 6),    # d0=3: 496 comps
        (4, 15, 6),    # d0=4: 816 comps, cascade to d=256
        (4, 20, 6),    # d0=4: 1771 comps
        (4, 30, 5),    # d0=4: 5456 comps, cascade to d=128
        (2, 100, 8),   # d0=2: 101 comps, S=100 for better box cert
        (3, 50, 6),    # d0=3: 1326 comps
    ]

    for d0, S, max_levels in cascade_configs:
        try:
            r = run_cascade_test(d0, S, 1.30, max_levels=max_levels)
            if r is not None:
                results.append(r)
                if r.get('box_certified'):
                    print(f"\n  >>> RIGOROUS PROOF via cascade! <<<")
                    return results
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    return results


def phase_intermediate():
    """Prove intermediate values: 1.20, 1.25, 1.28 to build confidence."""
    print("\n" + "=" * 70)
    print("PHASE 0: Prove intermediate values (building confidence)")
    print("=" * 70)

    results = []
    targets = [
        # (c_target, configs_to_try)
        (1.20, [(6, 75), (6, 100), (8, 30), (8, 50), (10, 20), (10, 30)]),
        (1.25, [(6, 100), (6, 150), (8, 40), (8, 60), (10, 25), (10, 40)]),
        (1.28, [(8, 50), (8, 75), (10, 30), (10, 40), (12, 20), (12, 25)]),
    ]

    for c, configs in targets:
        print(f"\n--- c_target = {c} ---")
        proved = False
        for d0, S in configs:
            n = count_compositions(d0, S)
            if n > 2_000_000_000:
                continue
            try:
                r = run_l0_test(d0, S, c, verbose=False)
                if r is not None:
                    results.append(r)
                    if r['box_ok']:
                        print(f"  PROVED: C_{{1a}} >= {c} at "
                              f"d0={d0}, S={S}")
                        proved = True
                        break
            except Exception as e:
                print(f"  ERROR: {e}")

        if not proved:
            print(f"  Not yet proved rigorously at L0. "
                  f"Try cascade or larger S.")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment to prove C_{1a} >= 1.30")
    parser.add_argument('--phase', type=int, default=None,
                        help="Run specific phase (0,1,2,3)")
    parser.add_argument('--skip-intermediate', action='store_true',
                        help="Skip phase 0 intermediate proofs")
    parser.add_argument('--d0', type=int, default=None,
                        help="Test specific d0 only")
    parser.add_argument('--S', type=int, default=None,
                        help="Test specific S only")
    parser.add_argument('--c', type=float, default=1.30,
                        help="Target c (default 1.30)")
    args = parser.parse_args()

    # Single test mode
    if args.d0 is not None and args.S is not None:
        run_l0_test(args.d0, args.S, args.c)
        return

    all_results = []
    os.makedirs('data_experiment', exist_ok=True)

    if args.phase is None or args.phase == 0:
        if not args.skip_intermediate:
            inter = phase_intermediate()
            all_results.extend(inter)

    if args.phase is None or args.phase == 1:
        p1 = phase1()
        all_results.extend(p1)

        if args.phase is None or args.phase == 2:
            p2 = phase2(p1)
            all_results.extend(p2)

            if args.phase is None or args.phase == 3:
                p3 = phase3(p2)
                all_results.extend(p3)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    rigorous = [r for r in all_results
                if r.get('box_ok') or r.get('box_certified')]
    if rigorous:
        best_c = max(r['c_target'] for r in rigorous)
        print(f"\n  BEST RIGOROUS PROOF: C_{{1a}} >= {best_c}")
        for r in rigorous:
            if r['c_target'] == best_c:
                mn = r.get('min_net', r.get('min_cert_net'))
                print(f"    d0={r['d0']}, S={r['S']}, "
                      f"net={mn}, time={r['elapsed']:.1f}s")
    else:
        print("\n  No rigorous proofs found.")
        grid_proofs = [r for r in all_results
                       if r.get('survivors', 1) == 0 or r.get('proven')]
        if grid_proofs:
            print("  Grid-point proofs available (need larger S for box cert):")
            for r in grid_proofs:
                if r.get('survivors', 1) == 0:
                    print(f"    d0={r['d0']}, S={r['S']}, c={r['c_target']}, "
                          f"net={r.get('min_net'):.6f}")


if __name__ == '__main__':
    main()
