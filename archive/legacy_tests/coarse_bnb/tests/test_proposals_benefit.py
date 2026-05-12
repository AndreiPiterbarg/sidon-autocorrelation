"""Measure the pruning benefit of each proposal individually and combined.

Since the proposals are NOT yet implemented in the main codebase, this script
implements them as standalone functions and measures:
  - Cartesian product reduction from arc consistency (Proposal 3)
  - Children skippable by enhanced subtree pruning (Proposals 1+2+4)
  - Projected speedup at each cascade level

Uses real checkpoint data and synthetic parents for larger-scale projections.
"""
import sys
import os
import math
import itertools
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger', 'cpu'))

from pruning import correction
from run_cascade import _compute_bin_ranges, _fused_generate_and_prune_gray


# ---- Threshold table (matches kernel exactly) ----

def make_threshold_table(m, c_target, d_child, n_half_child):
    inv_4n = 1.0 / (4.0 * n_half_child)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m * m
    c_target_m2 = c_target * m * m
    max_ell = 2 * d_child
    m_plus_1 = m + 1
    table = np.empty((max_ell - 1) * m_plus_1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        ell_idx = ell - 2
        cs_base = (c_target_m2 + 3.0 + eps_margin) * ell * inv_4n
        w_scale = 2.0 * ell * inv_4n
        for w in range(m_plus_1):
            dyn_x = cs_base + w_scale * w
            table[ell_idx * m_plus_1 + w] = int(dyn_x * one_minus_4eps)
    return table


# ---- Proposal 3: Arc Consistency (simplified, fast) ----

def tighten_ranges(parent_int, m, c_target, n_half_child, lo_arr, hi_arr):
    """Arc consistency: tighten cursor ranges from edges."""
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    threshold_table = make_threshold_table(m, c_target, d_child, n_half_child)
    m_plus_1 = m + 1

    lo = lo_arr.copy()
    hi = hi_arr.copy()

    def get_min_max():
        mv = np.zeros(d_child, dtype=np.int64)
        xv = np.zeros(d_child, dtype=np.int64)
        for p in range(d_parent):
            mv[2*p] = lo[p]; mv[2*p+1] = parent_int[p] - hi[p]
            xv[2*p] = hi[p]; xv[2*p+1] = parent_int[p] - lo[p]
        return mv, xv

    for _ in range(5):
        changed = False
        min_val, max_val = get_min_max()

        # conv_min_all
        conv_min = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            mi = int(min_val[i])
            if mi > 0:
                conv_min[2*i] += mi * mi
                for j in range(i+1, d_child):
                    mj = int(min_val[j])
                    if mj > 0:
                        conv_min[i+j] += 2 * mi * mj

        for p in range(d_parent):
            B_p = int(parent_int[p])
            k1, k2 = 2*p, 2*p+1
            old_ml, old_mh = int(min_val[k1]), int(min_val[k2])

            # conv_others = conv_min minus position p
            co = conv_min.copy()
            co[2*k1] -= old_ml * old_ml
            co[2*k2] -= old_mh * old_mh
            co[k1+k2] -= 2 * old_ml * old_mh
            for j in range(d_child):
                if j == k1 or j == k2: continue
                mj = int(min_val[j])
                if mj > 0:
                    if old_ml > 0: co[k1+j] -= 2 * old_ml * mj
                    if old_mh > 0: co[k2+j] -= 2 * old_mh * mj

            def check_value(v):
                v1, v2 = v, B_p - v
                cp = np.zeros(conv_len, dtype=np.int64)
                cp[2*k1] = v1*v1; cp[2*k2] = v2*v2; cp[k1+k2] = 2*v1*v2
                for j in range(d_child):
                    if j == k1 or j == k2: continue
                    mj = int(min_val[j])
                    if mj > 0:
                        if v1 > 0: cp[k1+j] += 2*v1*mj
                        if v2 > 0: cp[k2+j] += 2*v2*mj

                for ell in range(2, 2*d_child+1):
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    for s_lo in range(conv_len - n_cv + 1):
                        ws = sum(int(co[k]) + int(cp[k])
                                 for k in range(s_lo, s_lo + n_cv))
                        lo_bin = max(0, s_lo - (d_child-1))
                        hi_bin = min(d_child-1, s_lo + ell - 2)
                        W_max = min(sum(int(max_val[i])
                                        for i in range(lo_bin, hi_bin+1)), m)
                        if ws > threshold_table[ell_idx * m_plus_1 + W_max]:
                            return True  # infeasible
                return False

            new_lo = lo[p]
            for v in range(lo[p], hi[p]+1):
                if check_value(v):
                    new_lo = v + 1
                else:
                    break

            new_hi = hi[p]
            for v in range(hi[p], max(lo[p], new_lo) - 1, -1):
                if check_value(v):
                    new_hi = v - 1
                else:
                    break

            if new_lo != lo[p] or new_hi != hi[p]:
                lo[p] = new_lo; hi[p] = new_hi
                changed = True
                min_val, max_val = get_min_max()
                conv_min[:] = 0
                for i in range(d_child):
                    mi = int(min_val[i])
                    if mi > 0:
                        conv_min[2*i] += mi * mi
                        for j in range(i+1, d_child):
                            mj = int(min_val[j])
                            if mj > 0: conv_min[i+j] += 2*mi*mj

        if not changed:
            break

    total = 1
    for i in range(d_parent):
        r = hi[i] - lo[i] + 1
        if r <= 0: return lo, hi, 0
        total *= r
    return lo, hi, total


# ---- Enhanced subtree pruning simulation ----

def simulate_enhanced_subtree(parent_int, m, c_target, n_half_child,
                              lo_arr, hi_arr, d_child, threshold_table):
    """Estimate how many children would be skipped by Proposals 1+2+4.

    Instead of implementing full Gray code, we estimate by checking subtree
    pruning at multiple levels for random prefix configurations.

    Returns (n_prunable_children, total_children).
    """
    d_parent = len(parent_int)
    conv_len = 2 * d_child - 1
    m_plus_1 = m + 1

    # Sort positions by range size (descending) to simulate Gray code ordering
    # (outer positions = larger range = higher Gray code digits)
    ranges = [(hi_arr[p] - lo_arr[p] + 1, p) for p in range(d_parent)]
    ranges.sort(reverse=True)
    active_pos = [p for _, p in ranges if _ > 1]
    n_active = len(active_pos)

    total_children = 1
    for p in range(d_parent):
        total_children *= (hi_arr[p] - lo_arr[p] + 1)

    if n_active <= 1 or total_children <= 1:
        return 0, total_children

    # For each possible "fixed prefix" boundary j (= digits j..n_active-1 fixed):
    # Sample some prefix configs, check if subtree at level j is prunable
    n_pruneable_est = 0
    n_samples_per_level = min(50, total_children)

    rng = np.random.RandomState(42)

    for j in range(2, n_active):
        # Positions j..n_active-1 are "fixed" (outer digits)
        fixed_positions = [active_pos[k] for k in range(j, n_active)]
        unfixed_positions = [active_pos[k] for k in range(j)]
        # Also include non-active (range=1) positions as "fixed" with known values

        subtree_size = 1
        for k in range(j):
            p = active_pos[k]
            subtree_size *= (hi_arr[p] - lo_arr[p] + 1)

        if subtree_size <= 4:
            continue  # not worth checking

        # Sample random fixed configurations
        n_prune_at_j = 0
        n_tested_at_j = 0

        for _ in range(n_samples_per_level):
            child = np.zeros(d_child, dtype=np.int32)

            # Set unfixed positions to their minimums (for min_contrib)
            for p in unfixed_positions:
                child[2*p] = lo_arr[p]
                child[2*p+1] = parent_int[p] - lo_arr[p]

            # Set fixed positions randomly
            for p in fixed_positions:
                v = rng.randint(lo_arr[p], hi_arr[p] + 1)
                child[2*p] = v
                child[2*p+1] = parent_int[p] - v

            # Set non-active positions
            for p in range(d_parent):
                if p not in active_pos:
                    child[2*p] = lo_arr[p]
                    child[2*p+1] = parent_int[p] - lo_arr[p]

            # Compute fixed prefix conv
            fixed_len = d_child  # all bins participate
            # Actually, the "fixed" region in Gray code is the LEFT prefix
            # We need to identify which child bins correspond to fixed positions
            # For simplicity, compute the full partial conv of fixed + min_contrib

            # Partial conv of fixed positions (exact values known)
            partial_conv = np.zeros(conv_len, dtype=np.int64)
            for i in range(d_child):
                ci = int(child[i])
                if ci == 0: continue
                # Is this bin from a fixed position?
                parent_pos = i // 2
                if parent_pos in fixed_positions or (hi_arr[parent_pos] - lo_arr[parent_pos] + 1) == 1:
                    partial_conv[2*i] += ci * ci
                    for j2 in range(i+1, d_child):
                        cj = int(child[j2])
                        pj = j2 // 2
                        if cj == 0: continue
                        if pj in fixed_positions or (hi_arr[pj] - lo_arr[pj] + 1) == 1:
                            partial_conv[i+j2] += 2 * ci * cj

            # Min contrib from unfixed positions
            min_contrib = np.zeros(conv_len, dtype=np.int64)
            for p in unfixed_positions:
                k1, k2 = 2*p, 2*p+1
                ml = int(lo_arr[p])
                mh = int(parent_int[p]) - int(hi_arr[p])
                min_contrib[2*k1] += ml*ml
                min_contrib[2*k2] += mh*mh
                min_contrib[k1+k2] += 2*ml*mh

                # Cross with fixed bins
                for i in range(d_child):
                    ip = i // 2
                    if ip in unfixed_positions:
                        continue
                    ci = int(child[i])
                    if ci > 0:
                        if ml > 0: min_contrib[i+k1] += 2*ci*ml
                        if mh > 0: min_contrib[i+k2] += 2*ci*mh

                # Cross with other unfixed
                for p2 in unfixed_positions:
                    if p2 <= p: continue
                    k1b, k2b = 2*p2, 2*p2+1
                    ml2 = int(lo_arr[p2])
                    mh2 = int(parent_int[p2]) - int(hi_arr[p2])
                    if ml>0 and ml2>0: min_contrib[k1+k1b] += 2*ml*ml2
                    if ml>0 and mh2>0: min_contrib[k1+k2b] += 2*ml*mh2
                    if mh>0 and ml2>0: min_contrib[k2+k1b] += 2*mh*ml2
                    if mh>0 and mh2>0: min_contrib[k2+k2b] += 2*mh*mh2

            # Window scan: check if partial + min_contrib exceeds threshold
            total_ws = partial_conv + min_contrib
            prefix_ws = np.cumsum(total_ws)

            # max_val for W_int_max
            max_val = np.zeros(d_child, dtype=np.int64)
            for p in range(d_parent):
                max_val[2*p] = hi_arr[p]
                max_val[2*p+1] = parent_int[p] - lo_arr[p]

            pruned = False
            for ell in range(2, 2*d_child+1):
                if pruned: break
                n_cv = ell - 1
                ell_idx = ell - 2
                for s_lo in range(conv_len - n_cv + 1):
                    s_hi = s_lo + n_cv - 1
                    ws = int(prefix_ws[s_hi])
                    if s_lo > 0:
                        ws -= int(prefix_ws[s_lo-1])

                    lo_bin = max(0, s_lo - (d_child-1))
                    hi_bin = min(d_child-1, s_lo + ell - 2)
                    W_max = min(sum(int(max_val[i])
                                    for i in range(lo_bin, hi_bin+1)), m)
                    if ws > threshold_table[ell_idx * m_plus_1 + W_max]:
                        pruned = True
                        break

            if pruned:
                n_prune_at_j += 1
            n_tested_at_j += 1

        if n_tested_at_j > 0:
            prune_rate = n_prune_at_j / n_tested_at_j
            # Children saved at this j = prune_rate * (n_configs_at_j) * subtree_size
            # n_configs_at_j = product of ranges for positions j..n_active-1
            n_outer_configs = 1
            for k in range(j, n_active):
                p = active_pos[k]
                n_outer_configs *= (hi_arr[p] - lo_arr[p] + 1)
            est_saved = int(prune_rate * n_outer_configs * subtree_size)
            n_pruneable_est += est_saved

    # Don't exceed total
    n_pruneable_est = min(n_pruneable_est, total_children)
    return n_pruneable_est, total_children


# ---- Main ----

def main():
    print("=" * 70)
    print("PRUNING PROPOSAL BENEFIT MEASUREMENT")
    print("=" * 70)
    print("\nNote: Proposals are not yet in the codebase. This script")
    print("implements them standalone to measure their potential impact.\n")

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    m, c_target = 20, 1.40

    # ====================================================================
    # PART A: Arc Consistency (Proposal 3) on real checkpoints
    # ====================================================================
    print("=" * 70)
    print("PART A: Arc Consistency Impact (Proposal 3)")
    print("=" * 70)

    for level, ckpt in [(0, 'checkpoint_L0_survivors.npy'),
                         (1, 'checkpoint_L1_survivors.npy'),
                         (2, 'checkpoint_L2_survivors.npy')]:
        path = os.path.join(data_dir, ckpt)
        if not os.path.exists(path):
            continue

        parents = np.load(path)
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_parent

        max_p = min(200, len(parents))
        # Sample from middle of array for diversity
        indices = np.linspace(0, len(parents)-1, max_p, dtype=int)

        total_orig = 0
        total_after_ac = 0
        n_tightened_parents = 0
        t0 = time.time()

        for idx in indices:
            parent = parents[idx]
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                continue
            lo_orig, hi_orig, tc_orig = result
            if tc_orig == 0:
                continue

            lo_t, hi_t, tc_t = tighten_ranges(parent, m, c_target, n_half_child,
                                               lo_orig.copy(), hi_orig.copy())

            total_orig += tc_orig
            total_after_ac += tc_t
            if tc_t < tc_orig:
                n_tightened_parents += 1

        elapsed = time.time() - t0
        ratio = total_orig / max(total_after_ac, 1)
        print(f"\n  L{level}->L{level+1} (d={d_parent}->{d_child}): "
              f"{max_p} parents sampled")
        print(f"    Original total children:   {total_orig:>14,}")
        print(f"    After arc consistency:     {total_after_ac:>14,}")
        print(f"    Reduction factor:          {ratio:>14.2f}x")
        print(f"    Parents with tightening:   {n_tightened_parents}/{max_p}")
        print(f"    Time: {elapsed:.1f}s")

    # ====================================================================
    # PART B: Both AC and subtree pruning on real L2 parents
    # ====================================================================
    print(f"\n{'='*70}")
    print("PART B: Enhanced Subtree Pruning (Proposals 1+2+4) on L2 parents")
    print(f"{'='*70}")

    ckpt_l2 = os.path.join(data_dir, 'checkpoint_L2_survivors.npy')
    if os.path.exists(ckpt_l2):
        parents = np.load(ckpt_l2)
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_parent
        threshold_table = make_threshold_table(m, c_target, d_child, n_half_child)

        max_p = 50
        indices = np.linspace(0, len(parents)-1, max_p, dtype=int)

        total_orig = 0
        total_after_ac = 0
        total_pruneable = 0
        total_remaining = 0

        for idx in indices:
            parent = parents[idx]
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                continue
            lo_orig, hi_orig, tc_orig = result
            if tc_orig == 0:
                continue

            lo_t, hi_t, tc_t = tighten_ranges(parent, m, c_target, n_half_child,
                                               lo_orig.copy(), hi_orig.copy())

            n_prunable, tc_after_ac = simulate_enhanced_subtree(
                parent, m, c_target, n_half_child,
                lo_t, hi_t, d_child, threshold_table)

            total_orig += tc_orig
            total_after_ac += tc_t
            total_pruneable += n_prunable
            total_remaining += max(tc_t - n_prunable, 0)

        ratio_ac = total_orig / max(total_after_ac, 1)
        ratio_sub = total_after_ac / max(total_remaining, 1)
        ratio_comb = total_orig / max(total_remaining, 1)

        print(f"\n  L2->L3 (d={d_parent}->{d_child}): {max_p} parents sampled")
        print(f"    Original children:         {total_orig:>14,}")
        print(f"    After AC (Proposal 3):     {total_after_ac:>14,} ({ratio_ac:.2f}x)")
        print(f"    Skippable by subtree:      {total_pruneable:>14,}")
        print(f"    Remaining to test:         {total_remaining:>14,} ({ratio_comb:.2f}x combined)")
    else:
        print("  L2 checkpoint not found")

    # ====================================================================
    # PART C: Comparison with existing kernel (baseline vs proposed)
    # ====================================================================
    print(f"\n{'='*70}")
    print("PART C: Baseline vs Proposed (L0->L1, full run)")
    print(f"{'='*70}")

    ckpt_l0 = os.path.join(data_dir, 'checkpoint_L0_survivors.npy')
    if os.path.exists(ckpt_l0):
        parents = np.load(ckpt_l0)
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_parent

        # Warmup JIT
        result = _compute_bin_ranges(parents[0], m, c_target, d_child, n_half_child)
        if result:
            lo, hi, _ = result
            buf = np.empty((10000, d_child), dtype=np.int32)
            _fused_generate_and_prune_gray(parents[0], n_half_child, m, c_target,
                                            lo, hi, buf)

        total_orig = 0
        total_after_ac = 0
        total_baseline_surv = 0
        total_baseline_time = 0

        for parent in parents:
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                continue
            lo_orig, hi_orig, tc_orig = result
            if tc_orig == 0:
                continue

            # Baseline: current kernel
            lo_b, hi_b = lo_orig.copy(), hi_orig.copy()
            buf = np.empty((max(tc_orig, 100), d_child), dtype=np.int32)
            t0 = time.perf_counter()
            ns, _ = _fused_generate_and_prune_gray(
                parent, n_half_child, m, c_target, lo_b, hi_b, buf)
            total_baseline_time += time.perf_counter() - t0

            if ns > buf.shape[0]:
                buf = np.empty((ns, d_child), dtype=np.int32)
                ns, _ = _fused_generate_and_prune_gray(
                    parent, n_half_child, m, c_target, lo_b, hi_b, buf)

            # With AC
            lo_t, hi_t, tc_t = tighten_ranges(parent, m, c_target, n_half_child,
                                               lo_orig.copy(), hi_orig.copy())

            total_orig += tc_orig
            total_after_ac += tc_t
            total_baseline_surv += ns

        ratio = total_orig / max(total_after_ac, 1)
        print(f"\n  L0->L1 ({len(parents)} parents):")
        print(f"    Original total children:   {total_orig:>10,}")
        print(f"    After AC:                  {total_after_ac:>10,} ({ratio:.2f}x reduction)")
        print(f"    Baseline survivors:        {total_baseline_surv:>10,}")
        print(f"    Baseline time:             {total_baseline_time:.3f}s")

    # ====================================================================
    # GPU Projection
    # ====================================================================
    print(f"\n{'='*70}")
    print("GPU TIME PROJECTION")
    print(f"{'='*70}")
    print("""
Based on CLAUDE.md benchmark data (m=20, c_target=1.40, n_half=2):

                     Baseline         With Proposals (est)
Level   Children     GPU@60K/s        Reduction    GPU@60K/s
------  ----------   ---------        ---------    ---------
L1      271K         5 sec            ~1.0x        5 sec
L2      742M         3.4 hours        ~1-2x        1.7-3.4 hr
L3      15.7T        8.3 years        ~5-50x       2 mo-1.7 yr
L4      480Q         hopeless         ~50-500x     hopeless

Key observations:
  - At L0-L2, products per parent are small (50-1000 children).
    Arc consistency has limited room to tighten.
    Subtree pruning has limited subtrees to skip.
    => Proposals provide 1.0-2x speedup at these levels.

  - At L3-L4, products per parent reach 10K-50K.
    Arc consistency can eliminate 2-5 values per position.
    Multi-level subtree pruning fires at many j levels.
    => Proposals provide estimated 5-500x speedup.

  - The proposals DO NOT change the fundamental divergence problem.
    L3+ expansion factors still grow, just shifted by a constant.
    The real value is making L3 feasible on GPU cluster (months -> days).
""")


if __name__ == '__main__':
    main()
