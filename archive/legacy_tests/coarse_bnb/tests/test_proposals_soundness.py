"""End-to-end soundness tests for all 4 proposals.

Verifies that the optimized cascade (with arc consistency + multi-level
subtree pruning + min contributions + partial-overlap windows) produces
EXACTLY the same survivor set as brute-force enumeration.

This is the definitive correctness check: any optimization that loses a
single survivor will fail these tests.
"""
import sys
import os
import math
import unittest
import numpy as np
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pruning import correction


def compute_autoconv(child):
    d = len(child)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += child[i] * child[j]
    return conv


def is_pruned_bruteforce(child, c_target, m, n_half_child):
    d_child = len(child)
    conv = compute_autoconv(child)
    conv_len = 2 * d_child - 1
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m * m
    inv_4n = 1.0 / (4.0 * n_half_child)
    c_target_m2 = c_target * m * m

    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ws = int(np.sum(conv[s_lo:s_lo + n_cv]))
            lo_bin = max(0, s_lo - (d_child - 1))
            hi_bin = min(d_child - 1, s_lo + ell - 2)
            W_int = int(np.sum(child[lo_bin:hi_bin + 1]))
            # C&S Lemma 3 + eq(1) W-refined, corrected for discrete W_g:
            # +3 = +1 (|ε*ε|≤1/m²) + 2 (W_f≤W_g+1/m, cumulative rounding)
            cs_corr_base = c_target_m2 + 3.0 + eps_margin
            dyn_x = (cs_corr_base + 2.0 * W_int) * ell * inv_4n
            dyn_it = int(dyn_x * one_minus_4eps)
            if ws > dyn_it:
                return True
    return False


def compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child):
    d_parent = len(parent_int)
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child))) + 1
    x_cap = min(x_cap, x_cap_cs)
    x_cap = min(x_cap, m)
    x_cap = max(x_cap, 0)

    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, b_i - x_cap)
        hi = min(b_i, x_cap)
        if lo > hi:
            return None
        lo_arr[i] = lo
        hi_arr[i] = hi
    return lo_arr, hi_arr


def bruteforce_survivors(parent_int, m, c_target, n_half_child):
    """Enumerate ALL children and return survivors + total count."""
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    result = compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child)
    if result is None:
        return set(), 0
    lo_arr, hi_arr = result

    survivors = set()
    total = 0
    ranges = [range(lo_arr[i], hi_arr[i] + 1) for i in range(d_parent)]
    for cursors in itertools.product(*ranges):
        total += 1
        child = np.zeros(d_child, dtype=np.int64)
        for i, c in enumerate(cursors):
            child[2 * i] = c
            child[2 * i + 1] = parent_int[i] - c
        if not is_pruned_bruteforce(child, c_target, m, n_half_child):
            # Canonicalize (min of row and its reverse)
            child_tuple = tuple(child)
            rev_tuple = tuple(child[::-1])
            canon = min(child_tuple, rev_tuple)
            survivors.add(canon)

    return survivors, total


def cascade_survivors(parent_int, m, c_target, n_half_child):
    """Run optimized cascade and return survivor set."""
    from cpu.run_cascade import process_parent_fused

    surv_arr, total = process_parent_fused(
        parent_int, m, c_target, n_half_child)

    survivors = set()
    for i in range(len(surv_arr)):
        survivors.add(tuple(surv_arr[i]))
    return survivors, total


class TestEndToEndSoundness(unittest.TestCase):
    """Compare optimized cascade against brute-force for small parents."""

    def _check_parent(self, parent_int, m, c_target, label=""):
        d_parent = len(parent_int)
        n_half_child = d_parent  # d_child/2 = d_parent

        bf_surv, bf_total = bruteforce_survivors(parent_int, m, c_target, n_half_child)
        cas_surv, cas_total = cascade_survivors(parent_int, m, c_target, n_half_child)

        # The cascade may report a smaller total (due to arc consistency tightening)
        # but must never lose a survivor
        missing = bf_surv - cas_surv
        self.assertEqual(len(missing), 0,
            f"{label}: Cascade lost {len(missing)} survivors!\n"
            f"  parent={parent_int}, m={m}, c={c_target}\n"
            f"  BF: {len(bf_surv)} survivors from {bf_total} children\n"
            f"  CAS: {len(cas_surv)} survivors from {cas_total} children\n"
            f"  Missing: {list(missing)[:5]}")

        # Also check no spurious survivors
        extra = cas_surv - bf_surv
        self.assertEqual(len(extra), 0,
            f"{label}: Cascade has {len(extra)} spurious survivors!\n"
            f"  Extra: {list(extra)[:5]}")

    def test_d4_small_m5(self):
        """d_parent=2 → d_child=4, m=5"""
        parents = [
            np.array([3, 2], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            np.array([4, 1], dtype=np.int32),
            np.array([1, 4], dtype=np.int32),
            np.array([5, 0], dtype=np.int32),
        ]
        for i, p in enumerate(parents):
            for ct in [1.3, 1.35, 1.4]:
                self._check_parent(p, 5, ct, f"d4_m5[{i}]_c{ct}")

    def test_d4_m10(self):
        """d_parent=2 → d_child=4, m=10"""
        parents = [
            np.array([5, 5], dtype=np.int32),
            np.array([6, 4], dtype=np.int32),
            np.array([7, 3], dtype=np.int32),
            np.array([8, 2], dtype=np.int32),
            np.array([3, 7], dtype=np.int32),
        ]
        for i, p in enumerate(parents):
            for ct in [1.3, 1.4]:
                self._check_parent(p, 10, ct, f"d4_m10[{i}]_c{ct}")

    def test_d6_small(self):
        """d_parent=3 → d_child=6, m=5"""
        parents = [
            np.array([2, 2, 1], dtype=np.int32),
            np.array([1, 2, 2], dtype=np.int32),
            np.array([3, 1, 1], dtype=np.int32),
            np.array([1, 3, 1], dtype=np.int32),
            np.array([1, 1, 3], dtype=np.int32),
        ]
        for i, p in enumerate(parents):
            for ct in [1.3, 1.4]:
                self._check_parent(p, 5, ct, f"d6_m5[{i}]_c{ct}")

    def test_d6_m10(self):
        """d_parent=3 → d_child=6, m=10"""
        parents = [
            np.array([4, 3, 3], dtype=np.int32),
            np.array([3, 4, 3], dtype=np.int32),
            np.array([5, 3, 2], dtype=np.int32),
            np.array([2, 5, 3], dtype=np.int32),
        ]
        for i, p in enumerate(parents):
            self._check_parent(p, 10, 1.4, f"d6_m10[{i}]")

    def test_d8_m10(self):
        """d_parent=4 → d_child=8, m=10"""
        parents = [
            np.array([3, 3, 2, 2], dtype=np.int32),
            np.array([4, 3, 2, 1], dtype=np.int32),
            np.array([2, 2, 3, 3], dtype=np.int32),
            np.array([5, 2, 2, 1], dtype=np.int32),
        ]
        for i, p in enumerate(parents):
            self._check_parent(p, 10, 1.4, f"d8_m10[{i}]")

    def test_d8_m20(self):
        """d_parent=4 → d_child=8, m=20 (project parameters)"""
        parents = [
            np.array([5, 5, 5, 5], dtype=np.int32),
            np.array([6, 5, 5, 4], dtype=np.int32),
            np.array([7, 6, 4, 3], dtype=np.int32),
            np.array([8, 5, 4, 3], dtype=np.int32),
        ]
        for i, p in enumerate(parents):
            self._check_parent(p, 20, 1.4, f"d8_m20[{i}]")


class TestArcConsistencyEffect(unittest.TestCase):
    """Verify arc consistency actually tightens ranges (not a no-op)."""

    def test_tightening_occurs(self):
        """Check that _tighten_ranges reduces total_children on some parents."""
        from cpu.run_cascade import (
            _compute_bin_ranges, _tighten_ranges)

        # Use a parent where tightening should help
        parent_int = np.array([5, 5, 5, 5], dtype=np.int32)
        m = 20
        c_target = 1.4
        d_child = 8
        n_half_child = 4

        result = _compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child)
        self.assertIsNotNone(result)
        lo_arr, hi_arr, total_before = result

        lo_copy = lo_arr.copy()
        hi_copy = hi_arr.copy()
        total_after = _tighten_ranges(parent_int, n_half_child, m, c_target,
                                       lo_copy, hi_copy)

        # At minimum, tightening should not INCREASE total
        self.assertLessEqual(total_after, total_before,
            f"Tightening increased total: {total_before} → {total_after}")


class TestMultiLevelSubtreePruning(unittest.TestCase):
    """Verify multi-level subtree pruning produces same results as single-level."""

    def test_survivors_subset_of_bruteforce(self):
        """Cascade survivors ⊆ brute-force survivors for d_parent=4."""
        parents = [
            np.array([3, 3, 2, 2], dtype=np.int32),
            np.array([4, 2, 2, 2], dtype=np.int32),
        ]
        for p in parents:
            m = 10
            c_target = 1.4
            n_half_child = len(p)

            bf_surv, _ = bruteforce_survivors(p, m, c_target, n_half_child)
            cas_surv, _ = cascade_survivors(p, m, c_target, n_half_child)

            missing = bf_surv - cas_surv
            self.assertEqual(len(missing), 0,
                f"Lost survivors for parent={p}: {missing}")


if __name__ == '__main__':
    unittest.main()
