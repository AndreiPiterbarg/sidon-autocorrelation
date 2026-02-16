"""Benchmark pruning power of new bounds.

Two metrics:
1. "Autoconv skip": configs that survive cheap checks (asymmetry + half_sum + max_elem)
   but would be pruned by new bounds BEFORE the expensive INT64 autoconvolution.
   This saves compute even though those configs would eventually be test_pruned.

2. "Survivor reduction": configs that survive the ENTIRE pipeline (including autoconv)
   but are pruned by new bounds. These are the hardest configs.
"""
import numpy as np
import time
import sys
import os

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from cpu.compositions import generate_canonical_compositions_batched
from cpu.pruning import correction, asymmetry_threshold
from test_values import compute_test_value_single
from gpu.pruning.bounds import (
    new_block_sum_bounds,
    two_max_bound,
    cross_term_c0c5_bound,
    sum_of_squares_bound,
    central_conv_ell2_bound,
    adjacent_conv_ell2_bounds,
    ell3_central_bounds,
    partial_ell4_center_bound,
    existing_max_element_bound,
    existing_half_sum_bound,
    existing_asymmetry_bound,
)


def benchmark_pruning_power(n_half, m, c_target, batch_size=100000):
    """Benchmark pruning power of each new bound.

    Measures two things:
    1. How many configs reaching autoconvolution would new bounds skip?
    2. How many true survivors would new bounds additionally eliminate?
    """
    d = 2 * n_half
    S = 4 * n_half * m
    corr = correction(m)
    thresh = c_target + corr

    print(f"\nBenchmarking D={d}, n={n_half}, m={m}, c_target={c_target}")
    print(f"  Correction: {corr:.6f}, Threshold: {thresh:.6f}")

    # Stage counters
    total = 0
    asym_pruned = 0
    halfsm_pruned = 0
    maxel_pruned = 0
    reach_autoconv = 0   # configs that pass all cheap checks
    test_pruned = 0
    survivors = 0

    # New bounds applied at the "reach autoconv" stage (pre-autoconv)
    new_bound_names = [
        'new_block_sum', 'two_max', 'sum_of_squares',
        'central_conv_ell2', 'adjacent_conv_ell2',
        'ell3_central', 'partial_ell4_ctr',
    ]
    if d == 6:
        new_bound_names.append('cross_c0c5')

    # Counts: configs that reach autoconv AND are pruned by each new bound
    pre_autoconv_prunes = {n: 0 for n in new_bound_names}
    pre_autoconv_any = 0

    # Counts: true survivors pruned by new bounds
    survivor_prunes = {n: 0 for n in new_bound_names}
    survivor_any = 0

    t0 = time.time()

    for batch in generate_canonical_compositions_batched(d, S, batch_size):
        for row_idx in range(len(batch)):
            c = batch[row_idx]
            total += 1

            # === Existing cheap checks (FP32-equivalent) ===
            # 1. Asymmetry
            margin = 1.0 / (4.0 * m)
            left_frac = float(np.sum(c[:d//2])) / S
            dom = max(left_frac, 1.0 - left_frac)
            asym_base = max(dom - margin, 0.0)
            asym_val = 2.0 * asym_base * asym_base
            if asym_val >= c_target:
                asym_pruned += 1
                continue

            # 2. Half-sum (ell=D)
            if existing_half_sum_bound(c, n_half, m) >= thresh:
                halfsm_pruned += 1
                continue

            # 3. Max-element (ell=2)
            if existing_max_element_bound(c, n_half, m) >= thresh:
                maxel_pruned += 1
                continue

            # === This config would reach autoconvolution in CUDA ===
            reach_autoconv += 1

            # Check new bounds (these would go BEFORE autoconv)
            new_bound_results = {}
            new_pruned = False
            for name in new_bound_names:
                if name == 'new_block_sum':
                    val = new_block_sum_bounds(c, n_half, m)
                elif name == 'two_max':
                    val = two_max_bound(c, n_half, m)
                elif name == 'sum_of_squares':
                    val = sum_of_squares_bound(c, n_half, m)
                elif name == 'cross_c0c5':
                    val = cross_term_c0c5_bound(c, n_half, m)
                elif name == 'central_conv_ell2':
                    val = central_conv_ell2_bound(c, n_half, m)
                elif name == 'adjacent_conv_ell2':
                    val = adjacent_conv_ell2_bounds(c, n_half, m)
                elif name == 'ell3_central':
                    val = ell3_central_bounds(c, n_half, m)
                elif name == 'partial_ell4_ctr':
                    val = partial_ell4_center_bound(c, n_half, m)
                else:
                    val = 0.0
                new_bound_results[name] = val
                if val >= thresh:
                    pre_autoconv_prunes[name] += 1
                    new_pruned = True
            if new_pruned:
                pre_autoconv_any += 1

            # Full autoconvolution (the expensive step)
            a = c.astype(np.float64) / m
            true_val = compute_test_value_single(a, n_half)
            if true_val >= thresh:
                test_pruned += 1
            else:
                survivors += 1
                # Check new bounds against true survivors
                surv_pruned = False
                for name in new_bound_names:
                    if new_bound_results[name] >= thresh:
                        survivor_prunes[name] += 1
                        surv_pruned = True
                if surv_pruned:
                    survivor_any += 1

        if total % 500000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} processed, {reach_autoconv} reach autoconv, "
                  f"{survivors} survivors, {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0

    # Report
    print(f"\n{'='*70}")
    print(f"BENCHMARK: D={d}, n={n_half}, m={m}, c_target={c_target}")
    print(f"{'='*70}")
    print(f"Total canonical compositions: {total:,}")
    print(f"\nExisting cascade:")
    for name, cnt in [('asymmetry', asym_pruned), ('half_sum', halfsm_pruned),
                      ('max_elem', maxel_pruned)]:
        print(f"  {name:15s}: {cnt:>10,} ({100*cnt/total:.2f}%)")
    print(f"  {'reach autoconv':15s}: {reach_autoconv:>10,} ({100*reach_autoconv/total:.2f}%)")
    print(f"  {'test_pruned':15s}: {test_pruned:>10,} ({100*test_pruned/total:.2f}%)")
    print(f"  {'survivor':15s}: {survivors:>10,} ({100*survivors/total:.2f}%)")

    print(f"\n--- Metric 1: Autoconv skips (saves INT64 compute) ---")
    print(f"  Configs reaching autoconv: {reach_autoconv:,}")
    print(f"  {'Bound':<20s} {'Skips':>10s} {'% of autoconv':>14s}")
    print(f"  {'-'*20} {'-'*10} {'-'*14}")
    for name in sorted(pre_autoconv_prunes, key=lambda x: -pre_autoconv_prunes[x]):
        cnt = pre_autoconv_prunes[name]
        pct = 100.0 * cnt / reach_autoconv if reach_autoconv > 0 else 0
        print(f"  {name:<20s} {cnt:>10,} {pct:>13.2f}%")
    pct = 100.0 * pre_autoconv_any / reach_autoconv if reach_autoconv > 0 else 0
    print(f"  {'ANY new bound':<20s} {pre_autoconv_any:>10,} {pct:>13.2f}%")

    print(f"\n--- Metric 2: True survivor reduction ---")
    print(f"  True survivors: {survivors:,}")
    print(f"  {'Bound':<20s} {'Prunes':>10s} {'% of survivors':>15s}")
    print(f"  {'-'*20} {'-'*10} {'-'*15}")
    for name in sorted(survivor_prunes, key=lambda x: -survivor_prunes[x]):
        cnt = survivor_prunes[name]
        pct = 100.0 * cnt / survivors if survivors > 0 else 0
        print(f"  {name:<20s} {cnt:>10,} {pct:>14.2f}%")
    pct = 100.0 * survivor_any / survivors if survivors > 0 else 0
    print(f"  {'ANY new bound':<20s} {survivor_any:>10,} {pct:>14.2f}%")

    print(f"\nTime: {elapsed:.1f}s")

    return {
        'total': total,
        'reach_autoconv': reach_autoconv,
        'test_pruned': test_pruned,
        'survivors': survivors,
        'pre_autoconv_prunes': pre_autoconv_prunes,
        'pre_autoconv_any': pre_autoconv_any,
        'survivor_prunes': survivor_prunes,
        'survivor_any': survivor_any,
        'time_sec': elapsed,
    }


def run_benchmarks():
    """Run benchmarks at standard parameter sets."""
    configs = [
        (2, 5, 1.20),    # D=4, m=5
        (2, 10, 1.20),   # D=4, m=10
        (2, 20, 1.20),   # D=4, m=20
        (3, 5, 1.20),    # D=6, m=5
    ]

    results = []
    for n_half, m, c_target in configs:
        result = benchmark_pruning_power(n_half, m, c_target)
        results.append((n_half, m, c_target, result))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'D':>3s} {'m':>4s} {'Total':>12s} {'Reach AC':>12s} "
          f"{'AC Skips':>12s} {'Skip%':>8s} {'Survivors':>12s} {'Surv Red':>12s}")
    print(f"{'-'*3} {'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
    for n_half, m, c_target, r in results:
        d = 2 * n_half
        skip_pct = f"{100*r['pre_autoconv_any']/r['reach_autoconv']:.1f}%" if r['reach_autoconv'] > 0 else "N/A"
        print(f"{d:>3d} {m:>4d} {r['total']:>12,} {r['reach_autoconv']:>12,} "
              f"{r['pre_autoconv_any']:>12,} {skip_pct:>8s} "
              f"{r['survivors']:>12,} {r['survivor_any']:>12,}")

    return results


if __name__ == '__main__':
    run_benchmarks()
