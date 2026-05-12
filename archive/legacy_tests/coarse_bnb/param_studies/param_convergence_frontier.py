"""Find the exact convergence frontier: max c_target that converges.

Tests n_half=2 m=20 and n_half=3 m=15 through multiple cascade levels
at fine-grained c_target values.
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
from run_cascade import (run_level0, process_parent_fused,
                         _canonicalize_inplace, _fast_dedup)


def run_cascade_sample(n_half, m, c_target, max_levels=6, sample_per_level=200,
                       time_per_level=60):
    """Run sampled cascade and return per-level expansion factors."""
    l0 = run_level0(n_half, m, c_target, verbose=False)
    results = [{'level': 0, 'survivors': l0['n_survivors']}]

    if l0['n_survivors'] == 0:
        return results

    current = l0['survivors']

    for level in range(1, max_levels + 1):
        if len(current) == 0:
            results.append({'level': level, 'survivors': 0, 'expansion': 0})
            break

        d_parent = current.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        n_sample = min(sample_per_level, len(current))
        if n_sample < len(current):
            idx = np.random.default_rng(42).choice(len(current), n_sample, replace=False)
            sample = current[idx]
        else:
            sample = current

        all_surv = []
        t0 = time.time()
        completed = 0

        for parent in sample:
            if time.time() - t0 > time_per_level:
                break
            surv, _ = process_parent_fused(parent, m, c_target, n_half_child)
            if len(surv) > 0:
                all_surv.append(surv)
            completed += 1

        if completed == 0:
            results.append({'level': level, 'survivors': -1, 'expansion': -1})
            break

        n_surv_sample = sum(len(s) for s in all_surv)
        expansion = n_surv_sample / completed
        projected = int(expansion * len(current))

        results.append({
            'level': level,
            'survivors': projected,
            'expansion': expansion,
            'sample_completed': completed,
            'sample_survivors': n_surv_sample,
        })

        if projected == 0:
            break

        # Build next level from sample survivors
        if all_surv:
            next_surv = np.vstack(all_surv)
            _canonicalize_inplace(next_surv)
            next_surv = _fast_dedup(next_surv)
            current = next_surv
        else:
            break

    return results


# =========================================================================
# Test 1: n_half=2, m=20 — fine grid of c_target
# =========================================================================

print("=" * 80)
print("CONFIG: n_half=2, m=20 (d_L0=4)")
print("  correction = 0.1025, max valid c_target < 1.4004")
print("=" * 80)

targets_2_20 = [1.15, 1.20, 1.22, 1.24, 1.25, 1.26, 1.27, 1.28, 1.30, 1.33, 1.35, 1.37, 1.40]

for ct in targets_2_20:
    corr = correction(20)
    if ct + corr >= 1.5029:
        continue
    print(f"\nc_target = {ct}:")
    results = run_cascade_sample(2, 20, ct, max_levels=5, sample_per_level=150,
                                  time_per_level=30)
    for r in results:
        exp_str = f"exp={r.get('expansion', '?'):.1f}x" if 'expansion' in r else ""
        print(f"  L{r['level']}: {r['survivors']:,} survivors {exp_str}")

    # Check if converged
    if results[-1]['survivors'] == 0:
        print(f"  ** CONVERGES at L{results[-1]['level']} **")


# =========================================================================
# Test 2: n_half=3, m=15 — fine grid
# =========================================================================

print("\n\n" + "=" * 80)
print("CONFIG: n_half=3, m=15 (d_L0=6)")
print("  correction = 0.1378, max valid c_target < 1.3651")
print("=" * 80)

targets_3_15 = [1.20, 1.25, 1.28, 1.30, 1.33, 1.35, 1.36]

for ct in targets_3_15:
    corr = correction(15)
    if ct + corr >= 1.5029:
        continue
    print(f"\nc_target = {ct}:")
    results = run_cascade_sample(3, 15, ct, max_levels=6, sample_per_level=200,
                                  time_per_level=45)
    for r in results:
        exp_str = f"exp={r.get('expansion', '?'):.1f}x" if 'expansion' in r else ""
        print(f"  L{r['level']}: {r['survivors']:,} survivors {exp_str}")

    if results[-1]['survivors'] == 0:
        print(f"  ** CONVERGES at L{results[-1]['level']} **")


# =========================================================================
# Test 3: n_half=2, m=20 — verify full convergence at c_target=1.30
# =========================================================================

print("\n\n" + "=" * 80)
print("VERIFICATION: n_half=2, m=20, c_target=1.30")
print("  Running full (not sampled) cascade through convergence")
print("=" * 80)

l0 = run_level0(2, 20, 1.30, verbose=True)
current = l0['survivors']
print(f"L0: {len(current)} survivors")

for level in range(1, 8):
    if len(current) == 0:
        print(f"L{level}: PROVEN! 0 survivors")
        break

    d_parent = current.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    all_surv = []
    total_children = 0
    t0 = time.time()

    for i, parent in enumerate(current):
        if time.time() - t0 > 120:
            print(f"L{level}: timeout after {i}/{len(current)} parents, "
                  f"{sum(len(s) for s in all_surv)} survivors so far")
            break
        surv, nc = process_parent_fused(parent, 20, 1.30, n_half_child)
        total_children += nc
        if len(surv) > 0:
            all_surv.append(surv)
    else:
        elapsed = time.time() - t0
        if all_surv:
            next_surv = np.vstack(all_surv)
            _canonicalize_inplace(next_surv)
            next_surv = _fast_dedup(next_surv)
        else:
            next_surv = np.empty((0, d_child), dtype=np.int32)

        n_surv = len(next_surv)
        expansion = n_surv / max(len(current), 1)
        print(f"L{level}: {len(current)} parents -> {n_surv} survivors "
              f"(expansion={expansion:.1f}x), {total_children:,} children, {elapsed:.1f}s")
        current = next_surv
        continue

    # Timeout path
    break
