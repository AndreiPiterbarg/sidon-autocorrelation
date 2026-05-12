#!/usr/bin/env python3
"""Complete cascade proof runner.

Runs the FULL cascade (ALL parents at every level) for specified (m, c_target) configs.
Uses multiprocessing to parallelize the bottleneck (L2).

Usage:
    python run_proof.py
    python run_proof.py --m 35 --c_target 1.33
    python run_proof.py --m 35 --c_targets 1.28,1.30,1.33,1.35,1.37,1.40
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_this_dir, 'cloninger-steinerberger', 'cpu'))

from run_cascade import run_level0, process_parent_fused


def process_one_parent(args):
    """Worker function for multiprocessing."""
    parent, m, c_target, n_half_child = args
    ch, _ = process_parent_fused(parent, m, c_target, n_half_child)
    return ch


def full_cascade(m, c_target, n_workers=None):
    """Run COMPLETE cascade with ALL parents at every level.

    Returns list of (level, d, n_survivors, elapsed_seconds).
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 32)

    t0 = time.time()
    results = []

    # L0
    r = run_level0(n_half=2, m=m, c_target=c_target, verbose=False)
    surv = r['survivors']
    results.append((0, 4, len(surv), time.time() - t0))
    print("  L0 (d=4):  %d survivors  [%.1fs]" % (len(surv), time.time()-t0), flush=True)

    level = 0
    while len(surv) > 0:
        level += 1
        dp = surv.shape[1]
        dc = 2 * dp
        n_half_child = dp
        n_parents = len(surv)

        t_lv = time.time()

        all_ch = []
        for pi in range(n_parents):
            ch, _ = process_parent_fused(surv[pi], m, c_target, n_half_child)
            if len(ch) > 0:
                all_ch.append(ch)
            if (pi+1) % max(1, n_parents//10) == 0:
                elapsed_lv = time.time() - t_lv
                rate = (pi+1) / elapsed_lv
                eta = (n_parents - pi - 1) / rate if rate > 0 else 0
                n_ch = sum(len(c) for c in all_ch)
                print("    L%d: %d/%d (%.0f/s ETA %.0fs) %d children" % (
                    level, pi+1, n_parents, rate, eta, n_ch), flush=True)

        if all_ch:
            surv = np.unique(np.vstack(all_ch), axis=0)
        else:
            surv = np.empty((0, dc), dtype=np.int32)

        elapsed = time.time() - t0
        rig = " [d=%d <= m=%d: RIGOROUS]" % (dc, m) if dc <= m else " [d=%d > m=%d: NOT RIGOROUS]" % (dc, m)
        results.append((level, dc, len(surv), elapsed))
        print("  L%d (d=%d): %d survivors  [%.1fs total]%s" % (
            level, dc, len(surv), elapsed, rig if len(surv) == 0 else ""), flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description='Complete cascade proof runner')
    parser.add_argument('--m', type=int, default=35, help='Mass parameter (default: 35)')
    parser.add_argument('--c_targets', type=str, default='1.28,1.30,1.33,1.35,1.37,1.40',
                        help='Comma-separated c_target values')
    parser.add_argument('--c_target', type=float, default=None, help='Single c_target (overrides --c_targets)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    m = args.m
    if args.c_target is not None:
        c_targets = [args.c_target]
    else:
        c_targets = [float(x) for x in args.c_targets.split(',')]

    n_workers = args.workers or min(mp.cpu_count(), 32)

    print("=" * 70)
    print("COMPLETE CASCADE PROOF RUNNER")
    print("m = %d, %d workers" % (m, n_workers))
    print("Rigorous if converges at d <= %d" % m)
    print("=" * 70)

    # JIT warmup
    print("\nJIT warmup...", end="", flush=True)
    _ = run_level0(n_half=2, m=min(m, 20), c_target=1.28, verbose=False)
    r = run_level0(n_half=2, m=m, c_target=1.50, verbose=False)
    if len(r['survivors']) > 0:
        _ = process_parent_fused(r['survivors'][0], m, 1.50, len(r['survivors'][0]))
    print(" done\n", flush=True)

    all_results = {}
    for ct in c_targets:
        print("\n" + "=" * 70)
        print("c_target = %.2f" % ct)
        print("=" * 70)

        results = full_cascade(m, ct, n_workers=n_workers)
        all_results[ct] = results

        # Summary
        final_level, final_d, final_surv, final_t = results[-1]
        if final_surv == 0:
            if final_d <= m:
                print("\n  *** PROVEN: C_1a >= %.2f  (L%d, d=%d <= m=%d)  RIGOROUS  [%.1fs] ***\n" % (
                    ct, final_level, final_d, m, final_t))
            else:
                print("\n  *** PROVEN at L%d (d=%d) but d > m=%d — NOT RIGOROUS  [%.1fs] ***\n" % (
                    final_level, final_d, m, final_t))
        else:
            print("\n  NOT PROVEN: %d survivors remain at L%d (d=%d)  [%.1fs]\n" % (
                final_surv, final_level, final_d, final_t))

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ct in c_targets:
        results = all_results[ct]
        final_level, final_d, final_surv, final_t = results[-1]
        cascade_str = " -> ".join("%d" % n for _, _, n, _ in results)
        if final_surv == 0 and final_d <= m:
            print("  c=%.2f: [%s]  RIGOROUS PROOF  (%.1fs)" % (ct, cascade_str, final_t))
        elif final_surv == 0:
            print("  c=%.2f: [%s]  proven but d=%d>m=%d  (%.1fs)" % (ct, cascade_str, final_d, m, final_t))
        else:
            print("  c=%.2f: [%s]  NOT PROVEN  (%.1fs)" % (ct, cascade_str, final_t))


if __name__ == '__main__':
    main()
