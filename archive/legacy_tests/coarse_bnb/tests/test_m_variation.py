#!/usr/bin/env python3
"""Test Idea 3: varying m parameter - L0 only (no heavy JIT needed).

Reports L0 survivor counts for different m values, plus computes
the theoretical search space expansion per child and correction term.

For L1 survivors, we report projections based on the correction reduction.

Usage:
    python tests/test_m_variation.py
    python tests/test_m_variation.py --c_target 1.30
"""
import argparse
import sys
import os
import time
import math

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_root = os.path.join(_root, 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import run_level0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_target', type=float, default=1.30)
    args = parser.parse_args()

    c_target = args.c_target
    m_values = [20, 30, 40, 50]

    print(f"IDEA 3: Varying Grid Resolution m (L0 analysis)")
    print(f"  c_target={c_target}, d0=2, n_half=1")
    print(f"  m values: {m_values}")
    print()

    results = []
    for m in m_values:
        n_half = 1
        d0 = 2
        corr = correction(m, n_half)
        S = int(2 * d0 * m)
        n_compositions = count_compositions(d0, S)

        # Effective threshold = c_target + correction
        eff_thresh = c_target + corr

        # x_cap at child level (d_child=4, n_half_child=2)
        d_child = 4
        n_half_child = 2
        corr_child = correction(m, n_half_child)
        thresh_child = c_target + corr_child + 1e-9
        x_cap = int(math.floor(m * math.sqrt(4 * d_child * thresh_child)))

        print(f"--- m={m}: S={S}, compositions={n_compositions}, "
              f"correction={corr:.6f} ---")
        print(f"    eff_threshold={eff_thresh:.6f}, x_cap(L1)={x_cap}")

        t0 = time.time()
        l0 = run_level0(n_half, m, c_target, verbose=False, d0=d0)
        elapsed = time.time() - t0

        l0_surv = l0['n_survivors']
        print(f"    L0 survivors: {l0_surv} ({elapsed:.2f}s)")

        # Estimate avg children per parent at L1
        survivors = l0['survivors']
        if l0_surv > 0:
            total_ch = 0
            for i in range(l0_surv):
                p = survivors[i]
                ch_count = 1
                for j in range(len(p)):
                    lo = max(0, 2 * int(p[j]) - x_cap)
                    hi = min(2 * int(p[j]), x_cap)
                    ch_count *= max(0, hi - lo + 1)
                total_ch += ch_count
            avg_ch = total_ch / l0_surv
            print(f"    Avg children per parent at L1: {avg_ch:,.0f}")
            print(f"    Total L1 children (estimate): {total_ch:,}")
        else:
            total_ch = 0
            avg_ch = 0
            print(f"    PROVEN at L0!")

        print()
        results.append({
            'm': m, 'corr': corr, 'S': S,
            'n_comp': n_compositions,
            'l0_surv': l0_surv,
            'total_l1_children': total_ch,
            'avg_ch': avg_ch,
        })

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: c_target={c_target}")
    print(f"{'='*80}")
    print(f"  {'m':>5} {'corr':>10} {'S':>6} {'L0_comps':>10} "
          f"{'L0_surv':>8} {'L1_children':>14} {'avg_ch/par':>12}")
    print(f"  {'-'*5} {'-'*10} {'-'*6} {'-'*10} "
          f"{'-'*8} {'-'*14} {'-'*12}")
    for r in results:
        print(f"  {r['m']:5d} {r['corr']:10.6f} {r['S']:6d} "
              f"{r['n_comp']:10,} {r['l0_surv']:8d} "
              f"{r['total_l1_children']:14,} "
              f"{r['avg_ch']:12,.0f}")

    # Relative comparison
    if len(results) >= 2:
        base = results[0]
        print(f"\n  Relative to m={base['m']}:")
        for r in results[1:]:
            l0_ratio = r['l0_surv'] / max(1, base['l0_surv'])
            ch_ratio = r['total_l1_children'] / max(1, base['total_l1_children'])
            print(f"    m={r['m']}: L0 survivors {l0_ratio:.2f}x, "
                  f"L1 children {ch_ratio:.2f}x")


if __name__ == '__main__':
    main()
