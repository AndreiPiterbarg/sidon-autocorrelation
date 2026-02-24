"""Benchmark script: compare baseline, CS pre-filter, and multi-level solvers.

Runs all approaches on trivial targets (c=1.1, c=1.15) and prints a
comparison table with pruning stats and timings.
"""
import sys
import os
import time

# Add both cpu/ and parent cloninger-steinerberger/ to path
_cpu_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_cpu_dir)
sys.path.insert(0, _cpu_dir)
sys.path.insert(0, _parent_dir)

from solvers import run_single_level
from multilevel import run_multi_level, run_multi_level_standard


def run_benchmark():
    results = []

    # ===================================================================
    # Approach 1: Baseline single-level
    # ===================================================================
    print("\n" + "=" * 70)
    print("APPROACH: Baseline single-level (no CS, no multi-level)")
    print("=" * 70)

    for c_target, n_half, m in [
        (1.1, 2, 15), (1.1, 2, 20),
        (1.15, 2, 25), (1.15, 2, 50),
    ]:
        print(f"\n--- c_target={c_target}, n_half={n_half}, m={m} ---")
        r = run_single_level(n_half, m, c_target, verbose=True)
        results.append({
            'approach': 'Baseline',
            'c_target': c_target,
            'n_half': n_half,
            'm': m,
            'proven': r['proven'],
            'n_processed': r['stats']['n_processed'],
            'n_asym': r['stats']['n_pruned_asym'],
            'n_cs': r['stats'].get('n_pruned_cs', 0),
            'n_test': r['stats']['n_pruned_test'],
            'n_survived': r['stats']['n_survived'],
            'elapsed': r['stats']['elapsed'],
        })

    # ===================================================================
    # Approach 2: CS pre-filter (2b)
    # ===================================================================
    print("\n" + "=" * 70)
    print("APPROACH: 2b (Cauchy-Schwarz pre-filter)")
    print("=" * 70)

    for c_target, n_half, m in [
        (1.1, 2, 15), (1.1, 2, 20),
        (1.15, 2, 25), (1.15, 2, 50),
    ]:
        print(f"\n--- c_target={c_target}, n_half={n_half}, m={m} ---")
        r = run_single_level(n_half, m, c_target, verbose=True, use_cs=True)
        results.append({
            'approach': '2b CS',
            'c_target': c_target,
            'n_half': n_half,
            'm': m,
            'proven': r['proven'],
            'n_processed': r['stats']['n_processed'],
            'n_asym': r['stats']['n_pruned_asym'],
            'n_cs': r['stats'].get('n_pruned_cs', 0),
            'n_test': r['stats']['n_pruned_test'],
            'n_survived': r['stats']['n_survived'],
            'elapsed': r['stats']['elapsed'],
        })

    # ===================================================================
    # Approach 3: Multi-level standard (2c, full refinement)
    # ===================================================================
    print("\n" + "=" * 70)
    print("APPROACH: 2c multi-level standard (refine all bins)")
    print("=" * 70)

    for c_target, n_start, n_max, m in [
        (1.1, 2, 4, 10),
        (1.15, 2, 4, 10),
    ]:
        print(f"\n--- c_target={c_target}, n_start={n_start}, "
              f"n_max={n_max}, m={m} ---")
        r = run_multi_level_standard(n_start, n_max, m, c_target, verbose=True)
        lvl_stats = r['level_stats']
        total_tested = sum(ls.get('n_children_tested', 0) for ls in lvl_stats)
        total_time = sum(ls.get('elapsed', 0) for ls in lvl_stats)
        l0_surv = lvl_stats[0]['n_survivors'] if len(lvl_stats) > 0 else 0
        l1_surv = lvl_stats[1]['n_survivors'] if len(lvl_stats) > 1 else 0
        results.append({
            'approach': '2c Standard',
            'c_target': c_target,
            'n_half': f"{n_start}->{n_max}",
            'm': m,
            'proven': r['proven'],
            'n_processed': total_tested,
            'n_asym': sum(ls.get('n_asym', 0) for ls in lvl_stats),
            'n_cs': 0,
            'n_test': sum(ls.get('n_test', 0) for ls in lvl_stats),
            'n_survived': len(r['survivors']),
            'elapsed': total_time,
            'l0_surv': l0_surv,
            'l1_surv': l1_surv,
        })

    # ===================================================================
    # Approach 4: Multi-level adaptive (2c, selective refinement)
    # ===================================================================
    print("\n" + "=" * 70)
    print("APPROACH: 2c multi-level adaptive (refine high-sensitivity bins)")
    print("=" * 70)

    for c_target, n_start, n_max, m in [
        (1.1, 2, 4, 10),
        (1.15, 2, 4, 10),
    ]:
        print(f"\n--- c_target={c_target}, n_start={n_start}, "
              f"n_max={n_max}, m={m}, top_fraction=0.5 ---")
        r = run_multi_level(n_start, n_max, m, c_target,
                            top_fraction=0.5, verbose=True)
        lvl_stats = r['level_stats']
        total_tested = sum(ls.get('n_children_tested', 0) for ls in lvl_stats)
        total_time = sum(ls.get('elapsed', 0) for ls in lvl_stats)
        l0_surv = lvl_stats[0]['n_survivors'] if len(lvl_stats) > 0 else 0
        l1_surv = lvl_stats[1]['n_survivors'] if len(lvl_stats) > 1 else 0
        results.append({
            'approach': '2c Adaptive',
            'c_target': c_target,
            'n_half': f"{n_start}->{n_max}",
            'm': m,
            'proven': r['proven'],
            'n_processed': total_tested,
            'n_asym': sum(ls.get('n_asym', 0) for ls in lvl_stats),
            'n_cs': 0,
            'n_test': sum(ls.get('n_test', 0) for ls in lvl_stats),
            'n_survived': len(r['survivors']),
            'elapsed': total_time,
            'l0_surv': l0_surv,
            'l1_surv': l1_surv,
        })

    # ===================================================================
    # Print comparison table
    # ===================================================================
    print("\n\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    header = (f"{'Approach':<16} {'Target':>6} {'n_half':>8} {'m':>4} "
              f"{'Proven':>6} {'Tested':>10} {'Asym':>8} {'CS':>6} "
              f"{'Test':>8} {'Surv':>6} {'Time':>8}")
    print(header)
    print("-" * 110)
    for r in results:
        n_half_str = str(r['n_half'])
        print(f"{r['approach']:<16} {r['c_target']:>6.2f} {n_half_str:>8} {r['m']:>4} "
              f"{'YES' if r['proven'] else 'NO':>6} {r['n_processed']:>10,} "
              f"{r['n_asym']:>8,} {r['n_cs']:>6,} "
              f"{r['n_test']:>8,} {r['n_survived']:>6,} "
              f"{r['elapsed']:>7.3f}s")
    print("=" * 110)


if __name__ == '__main__':
    run_benchmark()
