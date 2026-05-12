"""Empirical L0 + L1 runs for all non-vacuous (m, n_half) at c_target=1.40.

Runs L0 fully, then samples L1 to measure actual expansion factors.
"""
import math
import os
import sys
import time
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import run_level0, process_parent_fused

C_TARGET = 1.40
C_UPPER_BOUND = 1.5029

# Parameters to test (only non-vacuous for c_target=1.40)
CONFIGS = [
    # (n_half, m)
    (2, 20),  # current main config
    (2, 25),
    (2, 30),
    (2, 40),
    (3, 20),
    (3, 25),
    (3, 30),
]

MAX_L1_PARENTS = 500  # sample size for L1 estimation
L1_TIME_BUDGET = 120  # seconds per L1 run

results = []

for n_half, m in CONFIGS:
    corr = correction(m)
    threshold = C_TARGET + corr
    if threshold >= C_UPPER_BOUND:
        print(f"\n*** SKIP n_half={n_half}, m={m}: VACUOUS (threshold={threshold:.4f})")
        continue

    margin = C_UPPER_BOUND - threshold
    d = 2 * n_half
    n_total = count_compositions(d, m)

    print(f"\n{'='*70}")
    print(f"n_half={n_half}, m={m}, d={d}")
    print(f"  correction={corr:.6f}, threshold={threshold:.6f}, margin={margin:.6f}")
    print(f"  total L0 compositions: {n_total:,}")
    print(f"{'='*70}")

    # --- Run L0 ---
    t0 = time.time()
    l0 = run_level0(n_half, m, C_TARGET, verbose=True)
    l0_time = time.time() - t0

    survivors = l0['survivors']
    n_surv = l0['n_survivors']
    l0_ratio = n_surv / max(n_total, 1)

    print(f"\n  L0 Results:")
    print(f"    survivors: {n_surv:,} / {n_total:,} ({l0_ratio*100:.4f}%)")
    print(f"    asym pruned: {l0['n_pruned_asym']:,}")
    print(f"    test pruned: {l0['n_pruned_test']:,}")
    print(f"    time: {l0_time:.2f}s")

    if n_surv == 0:
        print(f"    ** PROVEN AT L0! **")
        results.append({
            'n_half': n_half, 'm': m, 'd': d,
            'l0_survivors': 0, 'l0_ratio': 0,
            'l1_expansion': 0, 'status': 'PROVEN_L0'
        })
        continue

    # --- Sample L1 ---
    # Take a representative sample of L0 survivors
    n_sample = min(MAX_L1_PARENTS, n_surv)
    if n_sample < n_surv:
        idx = np.random.default_rng(42).choice(n_surv, n_sample, replace=False)
        sample = survivors[idx]
    else:
        sample = survivors

    d_child = 2 * d
    n_half_child = d_child // 2

    print(f"\n  L1 Sample ({n_sample} parents of {n_surv}):")

    total_children = 0
    total_l1_survivors = 0
    t1 = time.time()

    for i, parent in enumerate(sample):
        if time.time() - t1 > L1_TIME_BUDGET:
            print(f"    Time budget exhausted after {i} parents")
            n_sample = i
            break

        surv, n_children = process_parent_fused(parent, m, C_TARGET, n_half_child)
        total_children += n_children
        total_l1_survivors += len(surv)

        if (i + 1) % max(1, n_sample // 5) == 0:
            elapsed = time.time() - t1
            print(f"    {i+1}/{n_sample}: {total_l1_survivors} L1 survivors, "
                  f"{total_children:,} children tested, {elapsed:.1f}s")

    l1_time = time.time() - t1

    if n_sample > 0:
        # Scale to full L0 population
        children_per_parent = total_children / n_sample
        l1_surv_per_parent = total_l1_survivors / n_sample
        projected_l1 = int(l1_surv_per_parent * n_surv)
        expansion = projected_l1 / max(n_surv, 1)
        throughput = total_children / max(l1_time, 0.001)
    else:
        children_per_parent = 0
        l1_surv_per_parent = 0
        projected_l1 = 0
        expansion = 0
        throughput = 0

    print(f"\n  L1 Results:")
    print(f"    sample: {n_sample} parents, {total_children:,} children, "
          f"{total_l1_survivors:,} L1 survivors")
    print(f"    children/parent (avg): {children_per_parent:.1f}")
    print(f"    L1 survivors/parent: {l1_surv_per_parent:.2f}")
    print(f"    projected L1 total: {projected_l1:,}")
    print(f"    expansion factor: {expansion:.1f}x")
    print(f"    throughput: {throughput:,.0f} children/sec")
    print(f"    time: {l1_time:.2f}s")

    results.append({
        'n_half': n_half, 'm': m, 'd': d,
        'l0_survivors': n_surv, 'l0_ratio': l0_ratio,
        'l1_sample_parents': n_sample,
        'l1_sample_children': total_children,
        'l1_sample_survivors': total_l1_survivors,
        'children_per_parent': children_per_parent,
        'l1_expansion': expansion,
        'projected_l1': projected_l1,
        'throughput': throughput,
        'status': 'DIVERGING' if expansion > 1 else 'CONVERGING'
    })

# =========================================================================
# SUMMARY TABLE
# =========================================================================

print("\n\n" + "=" * 90)
print("SUMMARY: Parameter Viability for c_target = 1.40")
print("=" * 90)
print(f"{'n_half':>6} {'m':>4} | {'L0 surv':>10} | {'children/p':>11} | {'L1 expansion':>13} | {'proj L1':>12} | {'status':>12}")
print("-" * 85)

for r in results:
    status = r.get('status', '?')
    l1_exp = r.get('l1_expansion', 0)
    proj_l1 = r.get('projected_l1', 0)
    cpp = r.get('children_per_parent', 0)
    print(f"{r['n_half']:6d} {r['m']:4d} | {r['l0_survivors']:10,} | {cpp:11,.1f} | {l1_exp:13.1f}x | {proj_l1:12,} | {status:>12}")

print("\n\nKEY FINDING:")
print("Configs where L1 expansion >> 1 will NEVER converge (survivors grow exponentially).")
print("Only configs with expansion dropping toward <1 at some level have a chance.")
print("For c_target=1.40, the question is whether ANY config converges.")
