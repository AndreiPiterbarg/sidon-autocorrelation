"""Tests for modifications that could make the bivariate approach viable.

Three modifications tested:

Modification A: Multi-window union check
  Instead of requiring ONE window to prune the entire rectangle, check if
  the UNION of prunable regions from the top-K windows covers the rectangle.
  Split the rectangle into sub-rectangles and assign the best window to each.

Modification B: Bivariate bound tightening
  Use the quadratic structure to TIGHTEN cursor ranges before enumeration.
  For each boundary row/column of the rectangle, check if a window prunes it.
  If yes, shrink the range. This reduces total children to enumerate.

Modification C: Univariate sweep skip (the 1D simplification)
  After the full window scan identifies a killing window, check if this
  window also kills ALL remaining values of the sweeping cursor.
  Skip the rest of the sweep with O(d) conv fast-forward (not O(d^2) reset).
  This is the bivariate idea stripped to 1D where it's most cost-effective.
"""
import sys
import os
import math
import time
import itertools

import pytest
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import (
    _fused_generate_and_prune,
    _fused_generate_and_prune_gray,
    _compute_bin_ranges,
)

# Import helpers from the original bivariate test
from test_bivariate_block_pruning import (
    compute_raw_conv,
    compute_window_sum,
    compute_W_int,
    compute_threshold,
    is_pruned,
    build_child_from_cursors,
    compute_bivariate_coefficients,
    eval_bivariate,
    find_min_on_rect,
)

M = 20
C_TARGET = 1.4


# =====================================================================
# Univariate quadratic helpers
# =====================================================================

def compute_univariate_coefficients(parent, cursors, pos, ell, s_lo):
    """Compute the univariate quadratic D(x) = S_w(x) - T_w(x) coefficients.

    Let x = cursor[pos], all other cursors fixed.
    D(x) = A*x^2 + B*x + C

    Returns dict with keys: A, B, C
    """
    d_parent = len(parent)
    d_child = 2 * d_parent
    n_half_child = d_parent

    k1 = 2 * pos
    k2 = 2 * pos + 1
    a_parent = int(parent[pos])

    n_cv = ell - 1
    s_hi = s_lo + n_cv - 1

    def in_window(k):
        return s_lo <= k <= s_hi

    # Quadratic coefficient of S_w(x)
    A_S = (1 if in_window(2 * k1) else 0) + \
          (1 if in_window(2 * k2) else 0) - \
          (2 if in_window(k1 + k2) else 0)

    # Linear coefficient from variable-variable terms
    B_S = -2 * a_parent * (1 if in_window(2 * k2) else 0) + \
           2 * a_parent * (1 if in_window(k1 + k2) else 0)

    # Linear coefficient from cross-terms with fixed bins
    base_child = build_child_from_cursors(parent, cursors)
    for j in range(d_child):
        if j == k1 or j == k2:
            continue
        cj = int(base_child[j])
        if cj == 0:
            continue
        if in_window(k1 + j):
            B_S += 2 * cj
        if in_window(k2 + j):
            B_S -= 2 * cj

    # Constant: evaluate at reference point
    x0 = int(cursors[pos])
    ws0 = compute_window_sum(compute_raw_conv(base_child), s_lo, ell)
    C_S = ws0 - A_S * x0 * x0 - B_S * x0

    # Threshold linear coefficients
    m_d = float(M)
    inv_4n = 1.0 / (4.0 * float(n_half_child))
    dyn_base = C_TARGET * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    dyn_base_ell = dyn_base * float(ell) * inv_4n
    two_ell_inv_4n = 2.0 * float(ell) * inv_4n

    # W_int linear coefficient in x
    lo_bin = s_lo - (d_child - 1)
    if lo_bin < 0:
        lo_bin = 0
    hi_bin = s_lo + ell - 2
    if hi_bin > d_child - 1:
        hi_bin = d_child - 1

    w_x = 0
    k1_in = (lo_bin <= k1 <= hi_bin)
    k2_in = (lo_bin <= k2 <= hi_bin)
    if k1_in and k2_in:
        w_x = 0  # x + (a-x) = a, constant
    elif k1_in:
        w_x = 1   # W_int increases with x
    elif k2_in:
        w_x = -1  # W_int decreases with x (a-x contribution)

    W_int_ref = compute_W_int(base_child, d_child, s_lo, ell)
    T_const = dyn_base_ell + two_ell_inv_4n * W_int_ref
    T_bx = two_ell_inv_4n * w_x

    return {
        'A': A_S,
        'B': float(B_S) - T_bx,
        'C': float(C_S) - T_const,
        'A_S': A_S,
        'B_S': B_S,
    }


def min_on_interval(A, B, C, x_lo, x_hi):
    """Minimum of A*x^2 + B*x + C on integer interval [x_lo, x_hi]."""
    candidates = [x_lo, x_hi]
    if A > 0:
        x_star = -B / (2.0 * A)
        for xc in [int(math.floor(x_star)), int(math.ceil(x_star))]:
            if x_lo <= xc <= x_hi:
                candidates.append(xc)
    vals = [A * x * x + B * x + C for x in candidates]
    return min(vals)


# =====================================================================
# MODIFICATION A: Multi-window union check
# =====================================================================

class TestModA_MultiWindowUnion:
    """Check if using multiple windows with sub-rectangle partitioning
    increases the 2D block pruning success rate."""

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([3, 3, 7, 7], dtype=np.int32),
        np.array([7, 3, 7, 3], dtype=np.int32),
    ])
    def test_multi_window_success_rate(self, parent):
        """Split rectangle into 2x2 sub-rectangles. For each sub-rectangle,
        independently find the best window. If all 4 sub-rectangles are
        prunable (possibly by different windows), the block is prunable.

        Compare with: single-window 2D check (from original tests).
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        single_window_ok = 0
        multi_window_ok = 0
        total_blocks = 0

        for pos1 in range(d_parent):
            for pos2 in range(pos1 + 1, d_parent):
                x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
                y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])
                if x_hi - x_lo < 1 or y_hi - y_lo < 1:
                    continue
                total_blocks += 1
                cursors_base = lo_arr.copy()

                # Single-window check (from original)
                sw_ok = False
                for ell in range(2, 2 * d_child + 1):
                    if sw_ok:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo in range(n_windows):
                        coeffs = compute_bivariate_coefficients(
                            parent, cursors_base, pos1, pos2, ell, s_lo)
                        min_val, _ = find_min_on_rect(
                            coeffs, x_lo, x_hi, y_lo, y_hi)
                        if min_val > 0:
                            sw_ok = True
                            break
                if sw_ok:
                    single_window_ok += 1

                # Multi-window: split into 2x2 sub-rectangles
                x_mid = (x_lo + x_hi) // 2
                y_mid = (y_lo + y_hi) // 2
                sub_rects = [
                    (x_lo, x_mid, y_lo, y_mid),
                    (x_mid + 1, x_hi, y_lo, y_mid),
                    (x_lo, x_mid, y_mid + 1, y_hi),
                    (x_mid + 1, x_hi, y_mid + 1, y_hi),
                ]
                # Filter empty sub-rectangles
                sub_rects = [(xl, xh, yl, yh) for xl, xh, yl, yh in sub_rects
                             if xl <= xh and yl <= yh]

                all_sub_ok = True
                for (xl, xh, yl, yh) in sub_rects:
                    sub_ok = False
                    for ell in range(2, 2 * d_child + 1):
                        if sub_ok:
                            break
                        n_cv = ell - 1
                        n_windows = conv_len - n_cv + 1
                        for s_lo in range(n_windows):
                            coeffs = compute_bivariate_coefficients(
                                parent, cursors_base, pos1, pos2, ell, s_lo)
                            min_val, _ = find_min_on_rect(
                                coeffs, xl, xh, yl, yh)
                            if min_val > 0:
                                sub_ok = True
                                break
                    if not sub_ok:
                        all_sub_ok = False
                        break

                if all_sub_ok:
                    multi_window_ok += 1

        print(f"\n  Parent: {parent}")
        print(f"  Blocks: {total_blocks}")
        print(f"  Single-window 2D: {single_window_ok} ({single_window_ok}/{total_blocks})")
        print(f"  Multi-window 2x2: {multi_window_ok} ({multi_window_ok}/{total_blocks})")
        extra = multi_window_ok - single_window_ok
        print(f"  Extra from multi-window: {extra}")


# =====================================================================
# MODIFICATION B: Bivariate bound tightening
# =====================================================================

class TestModB_BoundTightening:
    """Use the quadratic structure to tighten cursor ranges.

    For each boundary row/column, check if a window can prune the entire
    row/column. If yes, shrink the range. Iterate until stable.
    """

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([3, 3, 7, 7], dtype=np.int32),
        np.array([7, 3, 7, 3], dtype=np.int32),
    ])
    def test_bound_tightening_per_position(self, parent):
        """For each cursor position, check if the boundary values (lo, hi)
        can be tightened using 1D quadratic range checks.

        Specifically: for cursor at pos, check if x=x_lo is prunable for
        ALL values of other cursors. If yes, tighten x_lo to x_lo+1.
        Similarly for x_hi. Iterate.

        This reduces the total children count.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        total_before = 1
        for i in range(d_parent):
            total_before *= (hi_arr[i] - lo_arr[i] + 1)

        # Attempt per-position tightening using univariate checks
        lo_tight = lo_arr.copy()
        hi_tight = hi_arr.copy()

        changed = True
        iters = 0
        while changed and iters < 20:
            changed = False
            iters += 1
            for pos in range(d_parent):
                if lo_tight[pos] >= hi_tight[pos]:
                    continue

                # Check if x = lo_tight[pos] is prunable for ALL other
                # cursor configs (i.e., for ALL y values in every other pos)
                #
                # This is a hard combinatorial problem. We approximate:
                # check if for this specific x value, the child is prunable
                # for ALL other cursors at their WORST case (which produces
                # the minimum window sum).
                #
                # Conservative approach: check if x = lo_tight[pos] is
                # individually prunable when all other cursors are at lo.
                cursors_test = lo_tight.copy()
                cursors_test[pos] = lo_tight[pos]
                child_test = build_child_from_cursors(parent, cursors_test)
                if is_pruned(child_test, n_half_child, M, C_TARGET):
                    # Also check at all-hi for other cursors
                    cursors_test2 = hi_tight.copy()
                    cursors_test2[pos] = lo_tight[pos]
                    child_test2 = build_child_from_cursors(parent, cursors_test2)
                    if is_pruned(child_test2, n_half_child, M, C_TARGET):
                        # Use univariate range check: for x=lo, does the
                        # child survive for ANY other cursor config?
                        # (harder to check without full enumeration)
                        pass

                # Simpler per-position tightening using Cauchy-Schwarz:
                # If child[k1] = x and x^2/d_child > c_target (scaled),
                # the child is always prunable regardless of other values.
                # This is already captured by x_cap.
                #
                # The bivariate approach adds: for x at boundary,
                # considering cross-terms with one other cursor.
                # But we showed these cross-terms are small.
                pass

        # More practical approach: for each position pair (pos1, pos2),
        # check if the boundary ROW y=y_lo (or y_hi) is entirely prunable.
        tightened_ranges = {}
        for pos in range(d_parent):
            x_lo = int(lo_arr[pos])
            x_hi = int(hi_arr[pos])
            if x_lo >= x_hi:
                continue

            # Check lo boundary: is x=x_lo prunable for ALL other configs?
            # Conservative: check x=x_lo with all OTHER cursors at their
            # most favorable (for survival) position.
            # This requires checking ALL combinations of other cursors —
            # exponential. Instead, use the univariate quadratic check.
            #
            # For each other position pos2, fix pos1=pos at x_lo,
            # check if there's a window that prunes all y in [y_lo, y_hi].
            lo_pruneable = True
            for pos2 in range(d_parent):
                if pos2 == pos:
                    continue
                y_lo2, y_hi2 = int(lo_arr[pos2]), int(hi_arr[pos2])
                if y_lo2 >= y_hi2:
                    continue

                # Check: with cursor[pos]=x_lo, is child pruned for
                # ALL cursor[pos2] in [y_lo2, y_hi2], with other cursors
                # at lo? (Very conservative — other cursors fixed at lo.)
                cursors_check = lo_arr.copy()
                cursors_check[pos] = x_lo

                found_covering_window = False
                for ell in range(2, 2 * d_child + 1):
                    if found_covering_window:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo_w in range(n_windows):
                        # 1D check on pos2 axis
                        coeffs = compute_univariate_coefficients(
                            parent, cursors_check, pos2, ell, s_lo_w)
                        min_d = min_on_interval(
                            coeffs['A'], coeffs['B'], coeffs['C'],
                            y_lo2, y_hi2)
                        if min_d > 0:
                            found_covering_window = True
                            break

                if not found_covering_window:
                    lo_pruneable = False
                    break

            new_lo = x_lo + 1 if lo_pruneable else x_lo

            # Similarly for hi boundary
            hi_pruneable = True
            for pos2 in range(d_parent):
                if pos2 == pos:
                    continue
                y_lo2, y_hi2 = int(lo_arr[pos2]), int(hi_arr[pos2])
                if y_lo2 >= y_hi2:
                    continue

                cursors_check = lo_arr.copy()
                cursors_check[pos] = x_hi

                found_covering_window = False
                for ell in range(2, 2 * d_child + 1):
                    if found_covering_window:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo_w in range(n_windows):
                        coeffs = compute_univariate_coefficients(
                            parent, cursors_check, pos2, ell, s_lo_w)
                        min_d = min_on_interval(
                            coeffs['A'], coeffs['B'], coeffs['C'],
                            y_lo2, y_hi2)
                        if min_d > 0:
                            found_covering_window = True
                            break

                if not found_covering_window:
                    hi_pruneable = False
                    break

            new_hi = x_hi - 1 if hi_pruneable else x_hi

            if new_lo > x_lo or new_hi < x_hi:
                tightened_ranges[pos] = (new_lo, new_hi, x_lo, x_hi)

        total_after = 1
        for i in range(d_parent):
            if i in tightened_ranges:
                new_lo, new_hi, _, _ = tightened_ranges[i]
                total_after *= max(0, new_hi - new_lo + 1)
            else:
                total_after *= max(0, int(hi_arr[i]) - int(lo_arr[i]) + 1)

        print(f"\n  Parent: {parent}")
        print(f"  Children before: {total_before}")
        print(f"  Tightened positions: {len(tightened_ranges)}")
        for pos, (nl, nh, ol, oh) in sorted(tightened_ranges.items()):
            print(f"    pos {pos}: [{ol},{oh}] -> [{nl},{nh}]")
        print(f"  Children after: {total_after}")
        if total_before > 0:
            reduction = 1.0 - total_after / total_before
            print(f"  Reduction: {reduction:.1%}")


# =====================================================================
# MODIFICATION C: Univariate sweep skip
# =====================================================================

class TestModC_UnivariateSweepSkip:
    """After full scan finds a killing window, check if this window kills
    ALL remaining values of the sweeping cursor. If yes, skip the sweep.

    This is the most practical modification — directly reduces the number
    of children tested in the inner loop.
    """

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([3, 3, 7, 7], dtype=np.int32),
        np.array([7, 3, 7, 3], dtype=np.int32),
    ])
    def test_sweep_skip_rate(self, parent):
        """Simulate Gray code traversal. When quick-check fails and full
        scan finds a killing window, check if the 1D quadratic range check
        proves the remaining sweep is prunable.

        Measure:
        - How often the range check succeeds ("sweep skip rate")
        - How many children are skipped per successful check
        - Net savings accounting for the O(d) coefficient cost
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        # Simulate traversal for each cursor position
        total_children_tested = 0
        total_quick_checks = 0
        total_full_scans = 0
        total_range_checks = 0
        range_check_successes = 0
        children_skipped = 0

        for pos in range(d_parent):
            x_lo = int(lo_arr[pos])
            x_hi = int(hi_arr[pos])
            radix = x_hi - x_lo + 1
            if radix <= 1:
                continue

            # For each configuration of OTHER cursors, simulate sweep
            # (We only test a few configurations to keep it tractable)
            other_configs = []
            # Config 1: all others at lo
            other_configs.append(lo_arr.copy())
            # Config 2: all others at hi
            other_configs.append(hi_arr.copy())
            # Config 3: all others at mid
            mid = np.array([(lo_arr[i] + hi_arr[i]) // 2
                            for i in range(d_parent)], dtype=np.int32)
            other_configs.append(mid)

            for base_cursors in other_configs:
                last_kill = None  # (ell, s_lo) of last killing window

                for x in range(x_lo, x_hi + 1):
                    cursors = base_cursors.copy()
                    cursors[pos] = x
                    child = build_child_from_cursors(parent, cursors)
                    total_children_tested += 1

                    # Quick-check
                    if last_kill is not None:
                        ell_qc, s_qc = last_kill
                        conv = compute_raw_conv(child)
                        ws = compute_window_sum(conv, s_qc, ell_qc)
                        W_int = compute_W_int(child, d_child, s_qc, ell_qc)
                        dyn_it = compute_threshold(n_half_child, M, C_TARGET,
                                                   ell_qc, W_int)
                        total_quick_checks += 1
                        if ws > dyn_it:
                            continue  # quick-check killed it

                    # Quick-check failed → full scan
                    total_full_scans += 1
                    pruned = False
                    kill_ell, kill_s = None, None
                    for ell in range(2, 2 * d_child + 1):
                        if pruned:
                            break
                        n_cv = ell - 1
                        n_windows = conv_len - n_cv + 1
                        for s_lo_w in range(n_windows):
                            conv = compute_raw_conv(child)
                            ws = compute_window_sum(conv, s_lo_w, ell)
                            W_int = compute_W_int(child, d_child, s_lo_w, ell)
                            dyn_it = compute_threshold(n_half_child, M,
                                                       C_TARGET, ell, W_int)
                            if ws > dyn_it:
                                kill_ell, kill_s = ell, s_lo_w
                                pruned = True
                                break

                    if pruned:
                        last_kill = (kill_ell, kill_s)

                        # === RANGE CHECK: does this window kill remaining? ===
                        remaining = x_hi - x
                        if remaining >= 1:  # at least 1 child left to skip
                            total_range_checks += 1
                            coeffs = compute_univariate_coefficients(
                                parent, cursors, pos, kill_ell, kill_s)
                            # Check: D(x') > 0 for all x' in [x+1, x_hi]
                            min_d = min_on_interval(
                                coeffs['A'], coeffs['B'], coeffs['C'],
                                x + 1, x_hi)
                            if min_d > 0:
                                range_check_successes += 1
                                children_skipped += remaining
                                # Skip rest of sweep
                                break
                    else:
                        last_kill = None

        print(f"\n  Parent: {parent}, d_child={d_child}")
        print(f"  Total children tested: {total_children_tested}")
        print(f"  Quick-checks: {total_quick_checks}")
        print(f"  Full scans: {total_full_scans}")
        print(f"  Range checks attempted: {total_range_checks}")
        if total_range_checks > 0:
            rate = range_check_successes / total_range_checks
            print(f"  Range check successes: {range_check_successes} ({rate:.1%})")
            print(f"  Children skipped: {children_skipped}")
            avg_skip = children_skipped / range_check_successes if range_check_successes > 0 else 0
            print(f"  Avg children per skip: {avg_skip:.1f}")

            # Cost model
            cost_per_range_check = d_child  # O(d) coefficient computation
            cost_per_child_qc = 8  # O(ell) quick-check
            cost_per_child_full_scan = d_child * 10  # O(d*ells_tried)
            cost_per_child_conv = d_child  # O(d) conv update

            overhead = total_range_checks * cost_per_range_check
            # Children skipped save: conv update + quick-check per child
            saved_per_skip = cost_per_child_conv + cost_per_child_qc
            total_saved = children_skipped * saved_per_skip
            net = total_saved - overhead
            print(f"  Cost: {overhead} ops overhead, {total_saved} ops saved")
            print(f"  Net: {'+' if net > 0 else ''}{net} ops "
                  f"({'POSITIVE' if net > 0 else 'NEGATIVE'})")
        else:
            print(f"  No range checks triggered (all caught by quick-check)")


# =====================================================================
# Sweep skip on realistic L1 parents
# =====================================================================

class TestModC_RealisticSweepSkip:
    """Run the univariate sweep skip on actual L1 survivors."""

    def _get_l1_survivors(self, max_parents=10):
        from cpu.run_cascade import process_parent_fused
        l0_parents = [
            np.array([5, 5, 5, 5], dtype=np.int32),
            np.array([4, 6, 4, 6], dtype=np.int32),
            np.array([3, 3, 7, 7], dtype=np.int32),
            np.array([10, 5, 3, 2], dtype=np.int32),
            np.array([7, 3, 7, 3], dtype=np.int32),
        ]
        l1_survivors = []
        for p in l0_parents:
            surv, _ = process_parent_fused(p, M, C_TARGET, len(p))
            for s in surv[:max_parents // len(l0_parents) + 1]:
                l1_survivors.append(s)
            if len(l1_survivors) >= max_parents:
                break
        return l1_survivors[:max_parents]

    def test_l1_sweep_skip_effectiveness(self):
        """At d_parent=8, measure sweep skip effectiveness."""
        parents = self._get_l1_survivors(5)
        if not parents:
            pytest.skip("No L1 survivors")

        total_range_checks = 0
        total_successes = 0
        total_children_skipped = 0
        total_full_scans = 0
        total_children = 0

        for parent in parents:
            d_parent = len(parent)
            d_child = 2 * d_parent
            n_half_child = d_parent
            conv_len = 2 * d_child - 1

            result = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
            if result is None:
                continue
            lo_arr, hi_arr, _ = result

            for pos in range(d_parent):
                x_lo = int(lo_arr[pos])
                x_hi = int(hi_arr[pos])
                if x_hi - x_lo < 2:  # need radix >= 3 for meaningful skip
                    continue

                # Test with other cursors at lo
                base_cursors = lo_arr.copy()
                last_kill = None

                for x in range(x_lo, x_hi + 1):
                    cursors = base_cursors.copy()
                    cursors[pos] = x
                    child = build_child_from_cursors(parent, cursors)
                    total_children += 1

                    # Quick-check
                    if last_kill is not None:
                        ell_qc, s_qc = last_kill
                        conv = compute_raw_conv(child)
                        ws = compute_window_sum(conv, s_qc, ell_qc)
                        W_int = compute_W_int(child, d_child, s_qc, ell_qc)
                        dyn_it = compute_threshold(n_half_child, M, C_TARGET,
                                                   ell_qc, W_int)
                        if ws > dyn_it:
                            continue

                    # Full scan
                    total_full_scans += 1
                    pruned = False
                    for ell in range(2, 2 * d_child + 1):
                        if pruned:
                            break
                        n_cv = ell - 1
                        n_windows = conv_len - n_cv + 1
                        for s_lo_w in range(n_windows):
                            conv = compute_raw_conv(child)
                            ws = compute_window_sum(conv, s_lo_w, ell)
                            W_int = compute_W_int(child, d_child, s_lo_w, ell)
                            dyn_it = compute_threshold(n_half_child, M,
                                                       C_TARGET, ell, W_int)
                            if ws > dyn_it:
                                last_kill = (ell, s_lo_w)
                                pruned = True
                                break

                    if pruned:
                        remaining = x_hi - x
                        if remaining >= 1:
                            total_range_checks += 1
                            coeffs = compute_univariate_coefficients(
                                parent, cursors, pos, last_kill[0], last_kill[1])
                            min_d = min_on_interval(
                                coeffs['A'], coeffs['B'], coeffs['C'],
                                x + 1, x_hi)
                            if min_d > 0:
                                total_successes += 1
                                total_children_skipped += remaining
                                break
                    else:
                        last_kill = None

        print(f"\n  === L1 parents (d_child=16) sweep skip ===")
        print(f"  Total children: {total_children}")
        print(f"  Full scans: {total_full_scans}")
        print(f"  Range checks: {total_range_checks}")
        if total_range_checks > 0:
            rate = total_successes / total_range_checks
            print(f"  Successes: {total_successes} ({rate:.1%})")
            print(f"  Children skipped: {total_children_skipped}")

            # Net impact
            d = 16
            overhead = total_range_checks * d
            saved = total_children_skipped * (d + 8)
            net = saved - overhead
            baseline = total_children * (d + 8)
            pct = net / baseline * 100 if baseline > 0 else 0
            print(f"  Net ops: {net:+d} ({pct:+.1f}% of baseline)")


# =====================================================================
# Modification C variant: range check BEFORE full scan
# =====================================================================

class TestModC_RangeCheckBeforeFullScan:
    """Instead of range-checking after the full scan, check BEFORE.

    When quick-check fails, immediately do a range check with the OLD
    quick-check window. If the old window proves the remaining sweep
    prunable via the 1D quadratic, skip everything — no full scan needed.

    This saves the O(d^2) full scan cost, which is the most expensive
    per-child operation.
    """

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([3, 3, 7, 7], dtype=np.int32),
        np.array([7, 3, 7, 3], dtype=np.int32),
    ])
    def test_pre_scan_range_check(self, parent):
        """When QC fails at step x, check: does the OLD QC window
        kill all x' in [x, x_hi] via 1D quadratic? If yes, skip
        the full scan AND the remaining sweep.

        Key difference from post-scan: uses OLD window, triggers EARLIER.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        pre_scan_checks = 0
        pre_scan_successes = 0
        pre_scan_children_skipped = 0
        total_full_scans_without = 0
        total_full_scans_with = 0
        total_children = 0

        for pos in range(d_parent):
            x_lo = int(lo_arr[pos])
            x_hi = int(hi_arr[pos])
            if x_hi - x_lo < 1:
                continue

            base_cursors = lo_arr.copy()
            last_kill = None

            for x in range(x_lo, x_hi + 1):
                cursors = base_cursors.copy()
                cursors[pos] = x
                child = build_child_from_cursors(parent, cursors)
                total_children += 1

                # Quick-check
                quick_hit = False
                if last_kill is not None:
                    ell_qc, s_qc = last_kill
                    conv = compute_raw_conv(child)
                    ws = compute_window_sum(conv, s_qc, ell_qc)
                    W_int = compute_W_int(child, d_child, s_qc, ell_qc)
                    dyn_it = compute_threshold(n_half_child, M, C_TARGET,
                                               ell_qc, W_int)
                    if ws > dyn_it:
                        quick_hit = True

                if quick_hit:
                    continue

                # QC failed. Try range check with OLD window BEFORE full scan.
                if last_kill is not None:
                    ell_old, s_old = last_kill
                    remaining = x_hi - x + 1  # includes current
                    pre_scan_checks += 1

                    coeffs = compute_univariate_coefficients(
                        parent, cursors, pos, ell_old, s_old)
                    min_d = min_on_interval(
                        coeffs['A'], coeffs['B'], coeffs['C'],
                        x, x_hi)
                    if min_d > 0:
                        pre_scan_successes += 1
                        pre_scan_children_skipped += remaining
                        break  # skip entire remaining sweep
                    # else: old window can't cover range, must full scan

                # Full scan
                total_full_scans_without += 1
                total_full_scans_with += 1
                pruned = False
                for ell in range(2, 2 * d_child + 1):
                    if pruned:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo_w in range(n_windows):
                        conv = compute_raw_conv(child)
                        ws = compute_window_sum(conv, s_lo_w, ell)
                        W_int = compute_W_int(child, d_child, s_lo_w, ell)
                        dyn_it = compute_threshold(n_half_child, M,
                                                   C_TARGET, ell, W_int)
                        if ws > dyn_it:
                            last_kill = (ell, s_lo_w)
                            pruned = True
                            break
                if not pruned:
                    last_kill = None

        print(f"\n  Parent: {parent}, d_child={d_child}")
        print(f"  Total children: {total_children}")
        print(f"  Full scans (without opt): {total_full_scans_without}")
        print(f"  Pre-scan range checks: {pre_scan_checks}")
        if pre_scan_checks > 0:
            rate = pre_scan_successes / pre_scan_checks
            print(f"  Pre-scan successes: {pre_scan_successes} ({rate:.1%})")
            print(f"  Children skipped (incl. full scans avoided): "
                  f"{pre_scan_children_skipped}")
            full_scans_avoided = pre_scan_successes
            print(f"  Full scans avoided: {full_scans_avoided}")

            # Cost model
            d = d_child
            overhead = pre_scan_checks * d  # range check cost
            saved_full_scans = full_scans_avoided * d * 10  # avg full scan
            saved_children = pre_scan_children_skipped * (d + 8)  # conv + qc
            total_saved = saved_full_scans + saved_children
            net = total_saved - overhead
            print(f"  Overhead: {overhead}, Saved: {total_saved}, Net: {net:+d}")
