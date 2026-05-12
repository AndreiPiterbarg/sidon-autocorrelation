"""Quick survivor estimation for cascade with d0=2, m=20, c_target=1.35.

Runs L0 fully, L1 fully, then samples random parents at each subsequent
level to estimate survivor counts. Continues until convergence (0 survivors)
or we hit a max level.
"""
import sys
import os
import math
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))

from cpu.run_cascade import (
    run_level0, process_parent_fused,
    _fast_dedup, correction,
)

# Parameters
M = 20
C_TARGET = 1.35
D0 = 2
N_HALF_INIT = 1  # d0=2 means n_half=1
MAX_LEVELS = 10

# Samples per level — small to stay fast, enough for a rough estimate
SAMPLE_PER_LEVEL = 10


def process_one_parent(parent, n_half_child, m, c_target):
    """Process a single parent. Returns (survivors_array, n_children_tested)."""
    # process_parent_fused signature: (parent_int, m, c_target, n_half_child)
    survivors, n_tested = process_parent_fused(parent, m, c_target, n_half_child)
    return survivors, n_tested


def estimate_level(parents, n_parents_true, n_half_child, m, c_target, level_num, sample_size):
    """Sample random parents and extrapolate survivor count.

    parents: actual array of parents we have (may be a sample from prior level)
    n_parents_true: estimated true number of parents at this level
    """
    d_parent = parents.shape[1]
    d_child = 2 * d_parent

    # Pre-filter infeasible parents
    corr_pf = correction(m, n_half_child)
    thresh_pf = c_target + corr_pf + 1e-9
    x_cap_pf = int(math.floor(m * math.sqrt(4 * d_child * thresh_pf)))
    x_cap_cs_pf = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    x_cap_pf = min(x_cap_pf, x_cap_cs_pf)
    feasible_mask = np.all(parents <= x_cap_pf, axis=1)
    n_infeasible = len(parents) - int(np.sum(feasible_mask))
    parents = np.ascontiguousarray(parents[feasible_mask])
    n_feasible_in_hand = len(parents)

    if n_feasible_in_hand == 0:
        print(f"  L{level_num}: ALL parents infeasible! 0 survivors.", flush=True)
        return 0, np.empty((0, d_child), dtype=np.int32)

    # Adjust true count for infeasibility ratio
    feas_ratio = n_feasible_in_hand / (n_feasible_in_hand + n_infeasible) if (n_feasible_in_hand + n_infeasible) > 0 else 1.0
    n_parents_feasible_true = int(n_parents_true * feas_ratio)

    actual_sample = min(sample_size, n_feasible_in_hand)
    rng = np.random.RandomState(12345 + level_num)
    indices = rng.choice(n_feasible_in_hand, size=actual_sample, replace=False)

    total_children = 0
    total_survivors = 0
    all_survivors = []
    parent_survivor_counts = []

    t0 = time.time()
    for i, idx in enumerate(indices):
        parent = parents[idx]
        t_p = time.time()
        surv, n_tested = process_one_parent(parent, n_half_child, m, c_target)
        dt = time.time() - t_p
        total_children += n_tested
        n_surv = len(surv)
        total_survivors += n_surv
        parent_survivor_counts.append(n_surv)
        if n_surv > 0:
            # Keep only a random subsample of survivors to avoid OOM
            if n_surv > 50000:
                sub_idx = rng.choice(n_surv, size=50000, replace=False)
                all_survivors.append(surv[sub_idx])
            else:
                all_survivors.append(surv)

        elapsed = time.time() - t0
        avg_surv = total_survivors / (i + 1)
        est_total = avg_surv * n_parents_feasible_true
        print(f"  L{level_num} [{i+1}/{actual_sample}] "
              f"this={n_surv:,} surv ({n_tested:,} ch, {dt:.1f}s) | "
              f"avg={avg_surv:,.0f}/parent | "
              f"EST TOTAL={est_total:,.0f} | "
              f"elapsed={elapsed:.0f}s", flush=True)

    avg_survivors_per_parent = total_survivors / actual_sample
    est_survivors = avg_survivors_per_parent * n_parents_feasible_true

    # Collect sample survivors for next level
    if all_survivors:
        sampled_surv = np.vstack(all_survivors)
        # Dedup only if manageable size
        if len(sampled_surv) < 5_000_000:
            sampled_surv = _fast_dedup(sampled_surv)
    else:
        sampled_surv = np.empty((0, d_child), dtype=np.int32)

    surv_counts = np.array(parent_survivor_counts)
    zero_frac = np.mean(surv_counts == 0)

    print(f"\n  L{level_num} RESULT (d={d_parent}->{d_child}):", flush=True)
    print(f"    True parents (est): {n_parents_true:,} -> feasible: {n_parents_feasible_true:,}")
    print(f"    Sampled: {actual_sample} parents")
    print(f"    Avg survivors/parent: {avg_survivors_per_parent:,.1f}")
    print(f"    Zero-survivor parents: {zero_frac*100:.1f}%")
    print(f"    EST TOTAL SURVIVORS: {est_survivors:,.0f}")
    print(f"    Sample survivors in hand: {len(sampled_surv):,}", flush=True)

    return int(est_survivors), sampled_surv


def main():
    print("=" * 70)
    print(f"SURVIVOR ESTIMATION: d0={D0}, m={M}, c_target={C_TARGET}")
    print(f"Will run until convergence (0 survivors) or L{MAX_LEVELS}")
    print("=" * 70)

    # L0: run fully (fast at d0=2)
    print("\n[L0] Running full L0...", flush=True)
    l0 = run_level0(N_HALF_INIT, M, C_TARGET, verbose=True, d0=D0)
    survivors = l0['survivors']
    print(f"L0: {len(survivors):,} survivors at d={D0}")

    # L1: run fully (still fast — only 26 parents)
    print(f"\n[L1] Running full L1 (all {len(survivors)} parents)...", flush=True)
    t1 = time.time()
    all_l1 = []
    n_half_child = 2 * N_HALF_INIT
    for i in range(len(survivors)):
        s, n = process_parent_fused(survivors[i], M, C_TARGET, n_half_child)
        if len(s) > 0:
            all_l1.append(s)
    if all_l1:
        l1_surv = _fast_dedup(np.vstack(all_l1))
    else:
        l1_surv = np.empty((0, 4), dtype=np.int32)
    print(f"L1: {len(l1_surv):,} survivors at d=4 ({time.time()-t1:.1f}s)")

    results = [
        {'level': 0, 'd': D0, 'survivors': len(survivors)},
        {'level': 1, 'd': 4, 'survivors': len(l1_surv)},
    ]

    current = l1_surv
    n_parents_true = len(l1_surv)
    d_parent = 4
    n_half_parent = 2 * N_HALF_INIT

    # Estimate levels 2+
    for level in range(2, MAX_LEVELS + 1):
        d_child = 2 * d_parent
        n_half_child = 2 * n_half_parent

        print(f"\n{'='*70}")
        print(f"[L{level}] d={d_parent} -> d={d_child}, "
              f"est {n_parents_true:,} true parents, "
              f"{len(current):,} in hand")
        print(f"{'='*70}", flush=True)

        if len(current) == 0 or n_parents_true == 0:
            print(f"  0 parents -> PROVEN at L{level}!")
            results.append({'level': level, 'd': d_child, 'survivors': 0})
            break

        est_surv, sample_surv = estimate_level(
            current, n_parents_true, n_half_child, M, C_TARGET, level,
            sample_size=min(SAMPLE_PER_LEVEL, len(current))
        )

        results.append({
            'level': level,
            'd': d_child,
            'survivors': est_surv,
        })

        if est_surv == 0:
            print(f"\n  *** PROVEN at L{level}! ***")
            break

        # For next level
        current = sample_surv
        n_parents_true = est_surv
        d_parent = d_child
        n_half_parent = n_half_child

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: d0=2, m=20, c_target=1.35")
    print(f"{'='*70}")
    print(f"{'Level':<8} {'Dim':<8} {'Est Survivors':<25}")
    print("-" * 45)
    for r in results:
        s = r['survivors']
        if s > 1e12:
            label = f"{s:,.0f} ({s/1e12:.1f}T)"
        elif s > 1e9:
            label = f"{s:,.0f} ({s/1e9:.1f}B)"
        elif s > 1e6:
            label = f"{s:,.0f} ({s/1e6:.1f}M)"
        else:
            label = f"{s:,}"
        print(f"L{r['level']:<7} d={r['d']:<6} {label}")

    # Check convergence
    if len(results) >= 3:
        surv_vals = [r['survivors'] for r in results[1:] if r['survivors'] > 0]
        if len(surv_vals) >= 2:
            ratios = [surv_vals[i+1]/surv_vals[i] for i in range(len(surv_vals)-1)]
            print(f"\nExpansion ratios (level-to-level): {['%.1fx' % r for r in ratios]}")
            if ratios[-1] < 1:
                print("CASCADE IS CONVERGING!")
            else:
                print(f"CASCADE IS DIVERGING (last ratio: {ratios[-1]:.1f}x)")


if __name__ == '__main__':
    main()
