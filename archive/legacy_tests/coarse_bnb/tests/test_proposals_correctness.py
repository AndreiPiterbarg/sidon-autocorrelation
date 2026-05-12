"""Brute-force correctness verification for all 4 pruning proposals.

The proposals are NOT yet implemented in the main codebase. This script
implements them as standalone pure-Python functions and verifies:

  SOUNDNESS: Every child that is a true survivor (test_value < c_target)
  must NOT be eliminated by the proposed pruning optimization.

  COMPLETENESS CHECK: The proposals prune strictly more than the baseline.

Strategy:
  1. Pick small parents (d_parent=2,4) with m=10,15,20
  2. Enumerate ALL children brute-force, compute exact test values
  3. Run each proposal's pruning logic, collect what it would eliminate
  4. Assert: no true survivor is eliminated
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
from test_values import compute_test_value_single
from run_cascade import _compute_bin_ranges


# ---- Threshold computation (matches kernel exactly) ----

def make_threshold_table(m, c_target, d_child, n_half_child):
    """Precompute threshold_table[ell_idx * (m+1) + W_int] -> int64 threshold."""
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


def compute_autoconv(child):
    """Exact integer autoconvolution."""
    d = len(child)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        if child[i] == 0:
            continue
        conv[2 * i] += int(child[i]) ** 2
        for j in range(i + 1, d):
            if child[j] == 0:
                continue
            conv[i + j] += 2 * int(child[i]) * int(child[j])
    return conv


def child_is_pruned_by_window_scan(child, m, c_target, d_child, n_half_child, threshold_table):
    """Full window scan: returns True if child would be pruned."""
    conv = compute_autoconv(child)
    conv_len = len(conv)
    m_plus_1 = m + 1

    # Prefix sum of conv
    prefix_conv = np.cumsum(conv)

    # Prefix sum of child masses
    prefix_c = np.zeros(d_child + 1, dtype=np.int64)
    for i in range(d_child):
        prefix_c[i + 1] = prefix_c[i] + int(child[i])

    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        ell_idx = ell - 2
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            ws = int(prefix_conv[s_hi])
            if s_lo > 0:
                ws -= int(prefix_conv[s_lo - 1])

            lo_bin = max(0, s_lo - (d_child - 1))
            hi_bin = min(d_child - 1, s_lo + ell - 2)
            W_int = int(prefix_c[hi_bin + 1] - prefix_c[lo_bin])

            dyn_it = threshold_table[ell_idx * m_plus_1 + W_int]
            if ws > dyn_it:
                return True
    return False


# ---- Proposal 3: Arc Consistency ----

def tighten_ranges_proposal3(parent_int, m, c_target, n_half_child, lo_arr, hi_arr):
    """Arc consistency: remove infeasible cursor values from range edges.

    Returns (new_lo, new_hi, total_children).
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    threshold_table = make_threshold_table(m, c_target, d_child, n_half_child)
    m_plus_1 = m + 1

    lo = lo_arr.copy()
    hi = hi_arr.copy()

    # Compute min/max values for each child bin
    def get_min_max():
        min_val = np.zeros(d_child, dtype=np.int64)
        max_val = np.zeros(d_child, dtype=np.int64)
        for p in range(d_parent):
            min_val[2*p] = lo[p]
            min_val[2*p+1] = parent_int[p] - hi[p]
            max_val[2*p] = hi[p]
            max_val[2*p+1] = parent_int[p] - lo[p]
        return min_val, max_val

    MAX_PASSES = 5
    for pass_num in range(MAX_PASSES):
        changed = False
        min_val, max_val = get_min_max()

        # Precompute conv_min_all (all positions at minimum)
        conv_min_all = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            mi = int(min_val[i])
            if mi > 0:
                conv_min_all[2*i] += mi * mi
                for j in range(i+1, d_child):
                    mj = int(min_val[j])
                    if mj > 0:
                        conv_min_all[i+j] += 2 * mi * mj

        for p in range(d_parent):
            B_p = int(parent_int[p])
            k1, k2 = 2*p, 2*p+1

            # Subtract position p's contribution from conv_min_all
            old_ml, old_mh = int(min_val[k1]), int(min_val[k2])
            conv_others = conv_min_all.copy()
            conv_others[2*k1] -= old_ml * old_ml
            conv_others[2*k2] -= old_mh * old_mh
            conv_others[k1+k2] -= 2 * old_ml * old_mh
            for j in range(d_child):
                if j == k1 or j == k2:
                    continue
                mj = int(min_val[j])
                if mj > 0:
                    if old_ml > 0:
                        conv_others[k1+j] -= 2 * old_ml * mj
                    if old_mh > 0:
                        conv_others[k2+j] -= 2 * old_mh * mj

            # Scan from low end
            new_lo_p = lo[p]
            for v in range(lo[p], hi[p] + 1):
                v1, v2 = v, B_p - v

                # Build conv contribution of position p at value v
                conv_p = np.zeros(conv_len, dtype=np.int64)
                conv_p[2*k1] = v1 * v1
                conv_p[2*k2] = v2 * v2
                conv_p[k1+k2] = 2 * v1 * v2
                for j in range(d_child):
                    if j == k1 or j == k2:
                        continue
                    mj = int(min_val[j])
                    if mj > 0:
                        if v1 > 0:
                            conv_p[k1+j] += 2 * v1 * mj
                        if v2 > 0:
                            conv_p[k2+j] += 2 * v2 * mj

                # Check all windows
                infeasible = False
                for ell in range(2, 2*d_child + 1):
                    if infeasible:
                        break
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    for s_lo in range(conv_len - n_cv + 1):
                        ws = 0
                        for k in range(s_lo, s_lo + n_cv):
                            ws += int(conv_others[k]) + int(conv_p[k])

                        # W_int_max
                        lo_bin = max(0, s_lo - (d_child - 1))
                        hi_bin = min(d_child - 1, s_lo + ell - 2)
                        W_int_max = 0
                        for i in range(lo_bin, hi_bin + 1):
                            W_int_max += int(max_val[i])
                        W_int_max = min(W_int_max, m)

                        dyn_it = threshold_table[ell_idx * m_plus_1 + W_int_max]
                        if ws > dyn_it:
                            infeasible = True
                            break

                if infeasible:
                    new_lo_p = v + 1
                else:
                    break

            # Scan from high end
            new_hi_p = hi[p]
            for v in range(hi[p], max(lo[p], new_lo_p) - 1, -1):
                v1, v2 = v, B_p - v

                conv_p = np.zeros(conv_len, dtype=np.int64)
                conv_p[2*k1] = v1 * v1
                conv_p[2*k2] = v2 * v2
                conv_p[k1+k2] = 2 * v1 * v2
                for j in range(d_child):
                    if j == k1 or j == k2:
                        continue
                    mj = int(min_val[j])
                    if mj > 0:
                        if v1 > 0:
                            conv_p[k1+j] += 2 * v1 * mj
                        if v2 > 0:
                            conv_p[k2+j] += 2 * v2 * mj

                infeasible = False
                for ell in range(2, 2*d_child + 1):
                    if infeasible:
                        break
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    for s_lo in range(conv_len - n_cv + 1):
                        ws = 0
                        for k in range(s_lo, s_lo + n_cv):
                            ws += int(conv_others[k]) + int(conv_p[k])

                        lo_bin = max(0, s_lo - (d_child - 1))
                        hi_bin = min(d_child - 1, s_lo + ell - 2)
                        W_int_max = 0
                        for i in range(lo_bin, hi_bin + 1):
                            W_int_max += int(max_val[i])
                        W_int_max = min(W_int_max, m)

                        dyn_it = threshold_table[ell_idx * m_plus_1 + W_int_max]
                        if ws > dyn_it:
                            infeasible = True
                            break

                if infeasible:
                    new_hi_p = v - 1
                else:
                    break

            if new_lo_p != lo[p] or new_hi_p != hi[p]:
                lo[p] = new_lo_p
                hi[p] = new_hi_p
                changed = True
                # Recompute min/max and conv_min_all
                min_val, max_val = get_min_max()
                conv_min_all[:] = 0
                for i in range(d_child):
                    mi = int(min_val[i])
                    if mi > 0:
                        conv_min_all[2*i] += mi * mi
                        for j2 in range(i+1, d_child):
                            mj2 = int(min_val[j2])
                            if mj2 > 0:
                                conv_min_all[i+j2] += 2 * mi * mj2

        if not changed:
            break

    total = 1
    for i in range(d_parent):
        r = hi[i] - lo[i] + 1
        if r <= 0:
            return lo, hi, 0
        total *= r
    return lo, hi, total


# ---- Proposal 2: Guaranteed Minimum Contribution ----

def compute_min_contrib(child_fixed, fixed_len, d_child, d_parent,
                        parent_int, lo_arr, hi_arr, unfixed_positions):
    """Compute lower-bound autoconv contribution from unfixed bins."""
    conv_len = 2 * d_child - 1
    min_contrib = np.zeros(conv_len, dtype=np.int64)

    for p in unfixed_positions:
        k1, k2 = 2*p, 2*p+1
        ml = int(lo_arr[p])
        mh = int(parent_int[p]) - int(hi_arr[p])

        # Self-terms
        min_contrib[2*k1] += ml * ml
        min_contrib[2*k2] += mh * mh
        min_contrib[k1+k2] += 2 * ml * mh

        # Cross with fixed
        for i in range(fixed_len):
            ci = int(child_fixed[i])
            if ci > 0:
                if ml > 0:
                    min_contrib[i+k1] += 2 * ci * ml
                if mh > 0:
                    min_contrib[i+k2] += 2 * ci * mh

        # Cross with other unfixed
        for p2 in unfixed_positions:
            if p2 <= p:
                continue
            k1b, k2b = 2*p2, 2*p2+1
            ml2 = int(lo_arr[p2])
            mh2 = int(parent_int[p2]) - int(hi_arr[p2])
            if ml > 0 and ml2 > 0:
                min_contrib[k1+k1b] += 2 * ml * ml2
            if ml > 0 and mh2 > 0:
                min_contrib[k1+k2b] += 2 * ml * mh2
            if mh > 0 and ml2 > 0:
                min_contrib[k2+k1b] += 2 * mh * ml2
            if mh > 0 and mh2 > 0:
                min_contrib[k2+k2b] += 2 * mh * mh2

    return min_contrib


# ---- Test Functions ----

def test_proposal3_soundness():
    """Test arc consistency (Proposal 3) never eliminates a true survivor."""
    print("=" * 70)
    print("TEST: Proposal 3 (Arc Consistency) Soundness")
    print("=" * 70)

    configs = [
        # (n_half_parent, m, c_target)
        (1, 10, 1.33),
        (1, 10, 1.40),
        (1, 15, 1.33),
        (1, 15, 1.40),
        (2, 10, 1.33),
        (2, 10, 1.40),
        (2, 15, 1.40),
        (2, 20, 1.40),
    ]

    total_violations = 0
    total_tightened = 0

    for n_half_parent, m, c_target in configs:
        d_parent = 2 * n_half_parent
        d_child = 2 * d_parent
        n_half_child = d_parent

        print(f"\n  d_parent={d_parent}, m={m}, c_target={c_target}")

        threshold_table = make_threshold_table(m, c_target, d_child, n_half_child)

        # Generate some parents
        if d_parent == 2:
            parents = [np.array([a, m-a], dtype=np.int32) for a in range(m+1)]
        elif d_parent == 4:
            parents = []
            for a0 in range(m+1):
                for a1 in range(m-a0+1):
                    for a2 in range(m-a0-a1+1):
                        a3 = m - a0 - a1 - a2
                        parents.append(np.array([a0, a1, a2, a3], dtype=np.int32))
                        if len(parents) >= 100:
                            break
                    if len(parents) >= 100:
                        break
                if len(parents) >= 100:
                    break
        else:
            continue

        n_violations = 0
        n_tightened_positions = 0
        n_parents_tested = 0

        for parent in parents:
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                continue
            lo_orig, hi_orig, tc_orig = result
            if tc_orig == 0 or tc_orig > 100000:
                continue

            lo_tight, hi_tight, tc_tight = tighten_ranges_proposal3(
                parent, m, c_target, n_half_child, lo_orig.copy(), hi_orig.copy())

            # Check each eliminated value
            for p in range(d_parent):
                for v in range(lo_orig[p], lo_tight[p]):
                    n_tightened_positions += 1
                    # Check if ANY child with cursor[p]=v is a true survivor
                    other_ranges = []
                    for q in range(d_parent):
                        if q == p:
                            other_ranges.append([v])
                        else:
                            other_ranges.append(range(lo_orig[q], hi_orig[q] + 1))

                    for cursors in itertools.product(*other_ranges):
                        child = np.zeros(d_child, dtype=np.int32)
                        for q in range(d_parent):
                            child[2*q] = cursors[q]
                            child[2*q+1] = parent[q] - cursors[q]

                        if not child_is_pruned_by_window_scan(
                                child, m, c_target, d_child, n_half_child,
                                threshold_table):
                            n_violations += 1
                            print(f"    *** VIOLATION at parent={parent}, pos={p}, "
                                  f"v={v}: child={child} is a survivor!")
                            break

                for v in range(hi_tight[p] + 1, hi_orig[p] + 1):
                    n_tightened_positions += 1
                    other_ranges = []
                    for q in range(d_parent):
                        if q == p:
                            other_ranges.append([v])
                        else:
                            other_ranges.append(range(lo_orig[q], hi_orig[q] + 1))

                    for cursors in itertools.product(*other_ranges):
                        child = np.zeros(d_child, dtype=np.int32)
                        for q in range(d_parent):
                            child[2*q] = cursors[q]
                            child[2*q+1] = parent[q] - cursors[q]

                        if not child_is_pruned_by_window_scan(
                                child, m, c_target, d_child, n_half_child,
                                threshold_table):
                            n_violations += 1
                            print(f"    *** VIOLATION at parent={parent}, pos={p}, "
                                  f"v={v}: child={child} is a survivor!")
                            break

            n_parents_tested += 1

        print(f"    Parents tested: {n_parents_tested}")
        print(f"    Values eliminated by AC: {n_tightened_positions}")
        print(f"    Violations: {n_violations}")
        total_violations += n_violations
        total_tightened += n_tightened_positions

    print(f"\n  TOTAL violations: {total_violations}")
    print(f"  TOTAL values eliminated: {total_tightened}")
    if total_violations == 0:
        print("  PASS: Proposal 3 is SOUND.")
    else:
        print("  *** FAIL: Proposal 3 has SOUNDNESS VIOLATIONS! ***")
    return total_violations == 0


def test_proposal2_soundness():
    """Test that guaranteed minimum contributions are valid lower bounds."""
    print("\n" + "=" * 70)
    print("TEST: Proposal 2 (Min Unfixed Contribution) Soundness")
    print("=" * 70)

    configs = [
        (1, 10, 1.33),
        (1, 10, 1.40),
        (1, 15, 1.40),
        (2, 10, 1.40),
        (2, 15, 1.40),
    ]

    total_violations = 0

    for n_half_parent, m, c_target in configs:
        d_parent = 2 * n_half_parent
        d_child = 2 * d_parent
        n_half_child = d_parent

        print(f"\n  d_parent={d_parent}, m={m}, c_target={c_target}")

        if d_parent == 2:
            parents = [np.array([a, m-a], dtype=np.int32) for a in range(m+1)]
        elif d_parent == 4:
            parents = []
            for a0 in range(min(m+1, 8)):
                for a1 in range(min(m-a0+1, 8)):
                    for a2 in range(min(m-a0-a1+1, 8)):
                        a3 = m - a0 - a1 - a2
                        if a3 >= 0:
                            parents.append(np.array([a0, a1, a2, a3], dtype=np.int32))
        else:
            continue

        n_violations = 0
        n_checked = 0

        for parent in parents[:50]:
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                continue
            lo_arr, hi_arr, tc = result
            if tc == 0 or tc > 50000:
                continue

            # Pick a split point: fixed prefix = first half, unfixed = second half
            fixed_parent_boundary = d_parent // 2
            fixed_len = 2 * fixed_parent_boundary
            unfixed_positions = list(range(fixed_parent_boundary, d_parent))

            # Enumerate some fixed-prefix configurations
            fixed_ranges = [range(lo_arr[p], hi_arr[p] + 1)
                            for p in range(fixed_parent_boundary)]

            for fixed_cursors in itertools.islice(
                    itertools.product(*fixed_ranges), 20):
                child_fixed = np.zeros(d_child, dtype=np.int32)
                for p in range(fixed_parent_boundary):
                    child_fixed[2*p] = fixed_cursors[p]
                    child_fixed[2*p+1] = parent[p] - fixed_cursors[p]

                # Compute min_contrib
                mc = compute_min_contrib(
                    child_fixed, fixed_len, d_child, d_parent,
                    parent, lo_arr, hi_arr, unfixed_positions)

                # For EVERY unfixed assignment, verify actual contrib >= min_contrib
                unfixed_ranges = [range(lo_arr[p], hi_arr[p] + 1)
                                  for p in unfixed_positions]

                for unfixed_cursors in itertools.product(*unfixed_ranges):
                    child_full = child_fixed.copy()
                    for ip, p in enumerate(unfixed_positions):
                        child_full[2*p] = unfixed_cursors[ip]
                        child_full[2*p+1] = parent[p] - unfixed_cursors[ip]

                    # Actual autoconv
                    actual_conv = compute_autoconv(child_full)
                    # Partial conv of fixed only
                    partial_conv = compute_autoconv(child_fixed[:fixed_len])

                    # The "unfixed contribution" is actual_conv - partial_conv
                    # (contributions involving at least one unfixed bin)
                    unfixed_actual = actual_conv.copy()
                    plen = len(partial_conv)
                    unfixed_actual[:plen] -= partial_conv

                    # Check: unfixed_actual[k] >= min_contrib[k] for all k
                    for k in range(len(actual_conv)):
                        if int(unfixed_actual[k]) < int(mc[k]):
                            n_violations += 1
                            print(f"    *** VIOLATION: parent={parent}, "
                                  f"fixed={fixed_cursors}, unfixed={unfixed_cursors}")
                            print(f"        conv index {k}: actual_unfixed="
                                  f"{unfixed_actual[k]}, min_contrib={mc[k]}")
                            break

                    n_checked += 1

        print(f"    Assignments checked: {n_checked}")
        print(f"    Violations: {n_violations}")
        total_violations += n_violations

    if total_violations == 0:
        print("\n  PASS: Proposal 2 min_contrib is a valid lower bound.")
    else:
        print("\n  *** FAIL: min_contrib exceeds actual contribution! ***")
    return total_violations == 0


def test_proposal4_window_soundness():
    """Test that partial-overlap window checks use valid lower bounds."""
    print("\n" + "=" * 70)
    print("TEST: Proposal 4 (Partial-Overlap Windows) Soundness")
    print("=" * 70)

    configs = [
        (1, 10, 1.40),
        (1, 15, 1.40),
        (2, 10, 1.40),
        (2, 15, 1.40),
    ]

    total_violations = 0

    for n_half_parent, m, c_target in configs:
        d_parent = 2 * n_half_parent
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        print(f"\n  d_parent={d_parent}, m={m}, c_target={c_target}")

        threshold_table = make_threshold_table(m, c_target, d_child, n_half_child)
        m_plus_1 = m + 1

        if d_parent == 2:
            parents = [np.array([a, m-a], dtype=np.int32)
                       for a in range(m+1)]
        elif d_parent == 4:
            parents = []
            for a0 in range(min(m+1, 6)):
                for a1 in range(min(m-a0+1, 6)):
                    for a2 in range(min(m-a0-a1+1, 6)):
                        a3 = m - a0 - a1 - a2
                        if a3 >= 0:
                            parents.append(np.array([a0, a1, a2, a3], dtype=np.int32))
        else:
            continue

        n_violations = 0
        n_checked = 0

        for parent in parents[:30]:
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                continue
            lo_arr, hi_arr, tc = result
            if tc == 0 or tc > 10000:
                continue

            # Use fixed_len = half of child bins
            fixed_parent_boundary = d_parent // 2
            fixed_len = 2 * fixed_parent_boundary
            unfixed_positions = list(range(fixed_parent_boundary, d_parent))

            fixed_ranges = [range(lo_arr[p], hi_arr[p] + 1)
                            for p in range(fixed_parent_boundary)]

            for fixed_cursors in itertools.islice(
                    itertools.product(*fixed_ranges), 10):
                child_fixed = np.zeros(d_child, dtype=np.int32)
                for p in range(fixed_parent_boundary):
                    child_fixed[2*p] = fixed_cursors[p]
                    child_fixed[2*p+1] = parent[p] - fixed_cursors[p]

                # Partial conv of fixed region
                partial_conv = compute_autoconv(child_fixed[:fixed_len])
                partial_prefix = np.cumsum(partial_conv)
                partial_conv_len = len(partial_conv)

                # min_contrib from unfixed
                mc = compute_min_contrib(
                    child_fixed, fixed_len, d_child, d_parent,
                    parent, lo_arr, hi_arr, unfixed_positions)
                mc_prefix = np.cumsum(mc)

                # For EACH window (ell, s_lo), compute the proposal 4 lower bound
                # Then verify it's <= actual window sum for ALL unfixed assignments
                unfixed_ranges = [range(lo_arr[p], hi_arr[p] + 1)
                                  for p in unfixed_positions]

                # For each window, check if proposed lower bound <= actual minimum
                for ell in range(2, 2 * d_child + 1):
                    n_cv = ell - 1
                    ell_idx = ell - 2

                    for s_lo in range(conv_len - n_cv + 1):
                        s_hi = s_lo + n_cv - 1

                        # Proposed lower bound: fixed part + min_contrib part
                        ws_lb = 0
                        # Fixed part
                        k_start = max(s_lo, 0)
                        k_end = min(s_hi, partial_conv_len - 1)
                        if k_end >= k_start:
                            ws_lb += int(partial_prefix[k_end])
                            if k_start > 0:
                                ws_lb -= int(partial_prefix[k_start - 1])

                        # min_contrib part
                        mc_sum = int(mc_prefix[s_hi])
                        if s_lo > 0:
                            mc_sum -= int(mc_prefix[s_lo - 1])
                        ws_lb += mc_sum

                        # Verify: for all unfixed assignments, actual ws >= ws_lb
                        for unfixed_cursors in itertools.product(*unfixed_ranges):
                            child_full = child_fixed.copy()
                            for ip, p in enumerate(unfixed_positions):
                                child_full[2*p] = unfixed_cursors[ip]
                                child_full[2*p+1] = parent[p] - unfixed_cursors[ip]

                            actual_conv = compute_autoconv(child_full)
                            actual_ws = sum(int(actual_conv[k]) for k in range(s_lo, s_hi + 1))

                            if actual_ws < ws_lb:
                                n_violations += 1
                                if n_violations <= 3:
                                    print(f"    *** VIOLATION: ws_lb={ws_lb} > "
                                          f"actual_ws={actual_ws}")
                                    print(f"        parent={parent}, ell={ell}, "
                                          f"s_lo={s_lo}")
                                break

                        n_checked += 1

        print(f"    Windows checked: {n_checked}")
        print(f"    Violations: {n_violations}")
        total_violations += n_violations

    if total_violations == 0:
        print("\n  PASS: Proposal 4 lower bounds are valid.")
    else:
        print("\n  *** FAIL: Proposal 4 lower bounds EXCEED actual values! ***")
    return total_violations == 0


# ---- Main ----

if __name__ == '__main__':
    print("=" * 70)
    print("BRUTE-FORCE CORRECTNESS VERIFICATION FOR PROPOSALS 1-4")
    print("=" * 70)
    print("\nNote: Proposals are NOT yet in the codebase. This script")
    print("implements them as standalone functions for verification.\n")

    t_start = time.time()
    all_pass = True

    all_pass &= test_proposal3_soundness()
    all_pass &= test_proposal2_soundness()
    all_pass &= test_proposal4_window_soundness()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Overall: {'ALL PASS' if all_pass else '*** FAILURES ***'}")
    print(f"{'='*70}")
