"""Algorithm equivalence tests: verify GPU logic (reimplemented in Python)
matches CPU source of truth.

Tests the core mathematical components WITHOUT needing a GPU:
  1. Threshold table computation
  2. Autoconvolution (full O(d²) and incremental O(d))
  3. Window scan with sliding W_int
  4. Gray code enumeration (completeness + single-position changes)
  5. Quick-check correctness
  6. Bin range computation
  7. End-to-end: CPU fused kernel vs Python GPU-logic reimplementation

Usage:
    pytest tests/test_algorithm_equivalence.py -v
"""
import math
import os
import sys
import time

import numpy as np
import pytest

# Project layout
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import (
    process_parent_fused,
    _compute_bin_ranges,
)
from pruning import correction


# ════════════════════��═════════════════════════════════════════════
#  GPU-logic reimplementation in Python (matches cascade_kernel.cu)
# ═════════════════════════════════════════════════════��════════════

def gpu_build_threshold_table(d_child, m, c_target):
    """Reimplements build_threshold_table from cascade_host.cu."""
    n_half_child = d_child // 2
    m_d = float(m)
    inv_4n = 1.0 / (4.0 * float(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d
    cs_corr_base = c_target * m_d * m_d + 3.0 + eps_margin

    ell_count = 2 * d_child - 1
    table = np.empty(ell_count * (m + 1), dtype=np.int32)

    for ell in range(2, 2 * d_child + 1):
        ell_idx = ell - 2
        ell_scale = float(ell) * inv_4n
        ct_base_ell = cs_corr_base * ell_scale
        w_scale = 2.0 * ell_scale
        for w in range(m + 1):
            dyn_x = ct_base_ell + w_scale * float(w)
            table[ell_idx * (m + 1) + w] = int(dyn_x * one_minus_4eps)

    return table


def cpu_build_threshold_table(d_child, m, c_target):
    """Reimplements threshold computation from CPU run_cascade.py."""
    n_half_child = d_child // 2
    m_d = np.float64(m)
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d
    cs_corr_base = c_target * m_d * m_d + 3.0 + eps_margin

    ell_count = 2 * d_child - 1
    table = np.empty(ell_count * (m + 1), dtype=np.int64)

    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        ell_f = np.float64(ell)
        ct_base = cs_corr_base * ell_f * inv_4n
        w_scale = 2.0 * ell_f * inv_4n
        for w in range(m + 1):
            dyn_x = ct_base + w_scale * np.float64(w)
            table[idx * (m + 1) + w] = np.int64(dyn_x * one_minus_4eps)

    return table


def full_autoconv(child):
    """O(d²) autoconvolution — matches both CPU and GPU."""
    d = len(child)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int32)
    for i in range(d):
        ci = int(child[i])
        if ci == 0:
            continue
        conv[2 * i] += ci * ci
        for j in range(i + 1, d):
            cj = int(child[j])
            if cj != 0:
                conv[i + j] += 2 * ci * cj
    return conv


def incremental_conv_update_cpu(conv, child, pos, old1, old2, new1, new2, d_child):
    """CPU-style incremental update (run_cascade.py lines 1318-1355)."""
    k1 = 2 * pos
    k2 = k1 + 1
    delta1 = new1 - old1
    delta2 = new2 - old2

    conv[2 * k1] += new1 * new1 - old1 * old1
    conv[2 * k2] += new2 * new2 - old2 * old2
    conv[k1 + k2] += 2 * (new1 * new2 - old1 * old2)

    for j in range(k1):
        cj = int(child[j])
        if cj != 0:
            conv[k1 + j] += 2 * delta1 * cj
            conv[k2 + j] += 2 * delta2 * cj
    for j in range(k2 + 1, d_child):
        cj = int(child[j])
        if cj != 0:
            conv[k1 + j] += 2 * delta1 * cj
            conv[k2 + j] += 2 * delta2 * cj


def incremental_conv_update_gpu(conv, child, pos, old1, old2, new1, new2, d_child):
    """GPU-style incremental update (cascade_kernel.cu lines 91-155).

    Single-phase: each 'lane' writes to conv[k1+lane], combining delta1
    from child[lane] and delta2 from child[lane-1].
    """
    k1 = 2 * pos
    k2 = k1 + 1
    delta1 = new1 - old1
    delta2 = new2 - old2
    conv_len = 2 * d_child - 1

    for lane in range(d_child):  # GPU iterates lane 0..d_child-1
        idx = k1 + lane
        if idx >= conv_len:
            continue

        delta_total = 0

        # Self-terms
        if lane == k1:
            delta_total += new1 * new1 - old1 * old1
        if lane == k2:
            delta_total += 2 * (new1 * new2 - old1 * old2)
        if lane == k1 + 2:
            delta_total += new2 * new2 - old2 * old2

        # delta1 cross-term
        if lane < d_child and lane != k1 and lane != k2:
            cj = int(child[lane])
            if cj != 0:
                delta_total += 2 * delta1 * cj

        # delta2 cross-term (from child[lane-1])
        jm1 = lane - 1
        if 0 <= jm1 < d_child and jm1 != k1 and jm1 != k2:
            cj = int(child[jm1])
            if cj != 0:
                delta_total += 2 * delta2 * cj

        if delta_total != 0:
            conv[idx] += delta_total

    # Extra address: conv[k1+d_child] — not covered by main body (lane < d_child)
    extra_idx = k1 + d_child
    if extra_idx < conv_len:
        extra_delta = 0

        # Self-term conv[2*k2] when k1+2 == d_child (pos == d_parent-1)
        if k1 + 2 == d_child:
            extra_delta += new2 * new2 - old2 * old2

        # Delta2 cross-term from child[d_child-1]
        jlast = d_child - 1
        if jlast != k1 and jlast != k2:
            cj = int(child[jlast])
            if cj != 0:
                extra_delta += 2 * delta2 * cj

        if extra_delta != 0:
            conv[extra_idx] += extra_delta


def window_scan_cpu(conv, child, threshold_table, d_child, m):
    """CPU window scan: sliding window over ell values (sequential order).

    Returns (pruned, ell, s_lo, W_int) or (False, 0, 0, 0).
    """
    conv_len = 2 * d_child - 1
    m_plus_1 = m + 1

    prefix_c = np.zeros(d_child + 1, dtype=np.int64)
    for i in range(d_child):
        prefix_c[i + 1] = prefix_c[i] + int(child[i])

    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        ell_idx = ell - 2
        n_windows = conv_len - n_cv + 1

        ws = np.int64(0)
        for k in range(n_cv):
            ws += np.int64(conv[k])

        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])

            lo_bin = max(0, s_lo - (d_child - 1))
            hi_bin = min(d_child - 1, s_lo + ell - 2)
            W_int = int(prefix_c[hi_bin + 1] - prefix_c[lo_bin])

            dyn_it = threshold_table[ell_idx * m_plus_1 + W_int]
            if ws > dyn_it:
                return True, ell, s_lo, W_int

    return False, 0, 0, 0


def window_scan_gpu(conv, child, threshold_table, d_child, m):
    """GPU window scan: thread-private sliding window (cascade_kernel.cu lines 430-517).

    Uses sliding W_int update instead of prefix sum lookup.
    Returns (pruned, ell, s_lo, W_int) or (False, 0, 0, 0).
    """
    conv_len = 2 * d_child - 1
    m_plus_1 = m + 1

    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue

        # Initial window sum
        ws = np.int32(0)
        for k in range(n_cv):
            ws += np.int32(conv[k])

        # Initial W_int
        hi_bin_0 = min(ell - 2, d_child - 1)
        W_int = np.int32(0)
        for b in range(hi_bin_0 + 1):
            W_int += np.int32(child[b])

        # Check s=0
        ell_idx = ell - 2
        W_cl = max(0, min(m, int(W_int)))
        if ws > threshold_table[ell_idx * m_plus_1 + W_cl]:
            return True, ell, 0, int(W_int)

        # Sliding window s=1..n_windows-1
        for s in range(1, n_windows):
            ws += np.int32(conv[s + n_cv - 1])
            ws -= np.int32(conv[s - 1])

            if s + ell - 2 < d_child:
                W_int += np.int32(child[s + ell - 2])
            if s >= d_child:
                W_int -= np.int32(child[s - d_child])

            W_cl = max(0, min(m, int(W_int)))
            if ws > threshold_table[ell_idx * m_plus_1 + W_cl]:
                return True, ell, s, int(W_int)

    return False, 0, 0, 0


def gray_code_enumerate(lo_arr, hi_arr, parent_int):
    """Mixed-radix Gray code enumeration — matches both CPU and GPU.

    Yields (child, pos_changed, old_k1, old_k2, new_k1, new_k2) for each step.
    First yield is the initial child (pos_changed=-1).
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    # Build active positions right-to-left
    active_pos = []
    radix = []
    for i in range(d_parent - 1, -1, -1):
        r = int(hi_arr[i]) - int(lo_arr[i]) + 1
        if r > 1:
            active_pos.append(i)
            radix.append(r)
    n_active = len(active_pos)

    cursor = np.array([int(lo_arr[i]) for i in range(d_parent)], dtype=np.int32)
    child = np.empty(d_child, dtype=np.int32)
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    yield child.copy(), -1, 0, 0, 0, 0

    if n_active == 0:
        return

    gc_a = [0] * n_active
    gc_dir = [1] * n_active
    gc_focus = list(range(n_active + 1))

    while True:
        j = gc_focus[0]
        if j >= n_active:
            break
        gc_focus[0] = 0

        pos = active_pos[j]
        gc_a[j] += gc_dir[j]
        cursor[pos] = lo_arr[pos] + gc_a[j]

        if gc_a[j] == 0 or gc_a[j] == radix[j] - 1:
            gc_dir[j] = -gc_dir[j]
            gc_focus[j] = gc_focus[j + 1]
            gc_focus[j + 1] = j + 1

        k1 = 2 * pos
        k2 = k1 + 1
        old1 = int(child[k1])
        old2 = int(child[k2])
        child[k1] = cursor[pos]
        child[k2] = parent_int[pos] - cursor[pos]
        new1 = int(child[k1])
        new2 = int(child[k2])

        yield child.copy(), pos, old1, old2, new1, new2


# ══════════════════════════════════════════════════���═══════════════
#  Tests
# ═════════════════════════════════════════════��════════════════════

class TestThresholdTable:
    """Verify GPU and CPU threshold tables are bitwise identical."""

    @pytest.mark.parametrize("m,c_target,d_child", [
        (20, 1.4, 8), (20, 1.4, 16), (20, 1.4, 32), (20, 1.4, 64),
        (20, 1.30, 8), (20, 1.30, 32),
        (15, 1.33, 12), (15, 1.35, 12),
    ])
    def test_tables_match(self, m, c_target, d_child):
        gpu_table = gpu_build_threshold_table(d_child, m, c_target)
        cpu_table = cpu_build_threshold_table(d_child, m, c_target)

        ell_count = 2 * d_child - 1
        mismatches = 0
        for i in range(ell_count * (m + 1)):
            if int(gpu_table[i]) != int(cpu_table[i]):
                mismatches += 1

        assert mismatches == 0, f"{mismatches} threshold mismatches"


class TestAutoconvolution:
    """Verify full and incremental autoconvolution produce identical results."""

    def _random_child(self, d_child, m, rng):
        """Generate a random valid child (bins sum to m)."""
        child = np.zeros(d_child, dtype=np.int32)
        remaining = m
        for i in range(d_child - 1):
            child[i] = rng.integers(0, remaining + 1)
            remaining -= child[i]
        child[d_child - 1] = remaining
        return child

    @pytest.mark.parametrize("d_child", [8, 16, 32])
    def test_full_autoconv(self, d_child):
        """Full autoconv matches numpy reference."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            child = self._random_child(d_child, 20, rng)
            conv = full_autoconv(child)
            # Verify against numpy convolution
            child_f = child.astype(np.float64)
            np_conv = np.convolve(child_f, child_f)
            assert np.allclose(conv, np_conv), f"Mismatch: {conv} vs {np_conv}"

    @pytest.mark.parametrize("d_child", [8, 16, 32])
    def test_incremental_cpu_matches_full(self, d_child):
        """CPU incremental conv update produces same result as full recompute."""
        rng = np.random.default_rng(123)
        m = 20

        for _ in range(20):
            child = self._random_child(d_child, m, rng)
            conv = full_autoconv(child)

            # Pick a random parent position to change
            d_parent = d_child // 2
            pos = rng.integers(0, d_parent)
            k1, k2 = 2 * pos, 2 * pos + 1
            old1, old2 = int(child[k1]), int(child[k2])

            # New split: move ±1 (or random)
            parent_val = old1 + old2
            new1 = rng.integers(0, parent_val + 1)
            new2 = parent_val - new1
            child[k1] = new1
            child[k2] = new2

            # Incremental update
            conv_inc = conv.copy()
            incremental_conv_update_cpu(conv_inc, child, pos,
                                         old1, old2, int(new1), int(new2), d_child)

            # Full recompute
            conv_full = full_autoconv(child)

            assert np.array_equal(conv_inc, conv_full), \
                f"CPU incremental mismatch at pos={pos}"

    @pytest.mark.parametrize("d_child", [8, 16, 32])
    def test_incremental_gpu_matches_full(self, d_child):
        """GPU incremental conv update produces same result as full recompute."""
        rng = np.random.default_rng(456)
        m = 20

        for _ in range(20):
            child = self._random_child(d_child, m, rng)
            conv = full_autoconv(child)

            d_parent = d_child // 2
            pos = rng.integers(0, d_parent)
            k1, k2 = 2 * pos, 2 * pos + 1
            old1, old2 = int(child[k1]), int(child[k2])

            parent_val = old1 + old2
            new1 = rng.integers(0, parent_val + 1)
            new2 = parent_val - new1
            child[k1] = new1
            child[k2] = new2

            conv_inc = conv.copy()
            incremental_conv_update_gpu(conv_inc, child, pos,
                                         old1, old2, int(new1), int(new2), d_child)

            conv_full = full_autoconv(child)

            assert np.array_equal(conv_inc, conv_full), \
                f"GPU incremental mismatch at pos={pos}"

    @pytest.mark.parametrize("d_child", [8, 16, 32])
    def test_cpu_gpu_incremental_match(self, d_child):
        """CPU and GPU incremental updates produce identical results."""
        rng = np.random.default_rng(789)
        m = 20

        for _ in range(20):
            child = self._random_child(d_child, m, rng)
            conv = full_autoconv(child)

            d_parent = d_child // 2
            pos = rng.integers(0, d_parent)
            k1, k2 = 2 * pos, 2 * pos + 1
            old1, old2 = int(child[k1]), int(child[k2])

            parent_val = old1 + old2
            new1 = rng.integers(0, parent_val + 1)
            new2 = parent_val - new1
            child[k1] = new1
            child[k2] = new2

            conv_cpu = conv.copy()
            incremental_conv_update_cpu(conv_cpu, child, pos,
                                         old1, old2, int(new1), int(new2), d_child)

            conv_gpu = conv.copy()
            incremental_conv_update_gpu(conv_gpu, child, pos,
                                         old1, old2, int(new1), int(new2), d_child)

            assert np.array_equal(conv_cpu, conv_gpu), \
                f"CPU vs GPU incremental mismatch at pos={pos}"


class TestWindowScan:
    """Verify GPU and CPU window scans agree on pruning decisions."""

    def _random_child(self, d_child, m, rng):
        child = np.zeros(d_child, dtype=np.int32)
        remaining = m
        for i in range(d_child - 1):
            child[i] = rng.integers(0, remaining + 1)
            remaining -= child[i]
        child[d_child - 1] = remaining
        return child

    @pytest.mark.parametrize("d_child,m,c_target", [
        (8, 20, 1.4), (16, 20, 1.4), (32, 20, 1.4),
        (8, 20, 1.30), (16, 20, 1.30),
    ])
    def test_window_scan_agreement(self, d_child, m, c_target):
        """CPU and GPU window scans produce same pruning decision."""
        rng = np.random.default_rng(42)
        # Use int32 threshold table (GPU style) — values are identical
        table = gpu_build_threshold_table(d_child, m, c_target)

        n_agree = 0
        n_disagree = 0

        for _ in range(500):
            child = self._random_child(d_child, m, rng)
            conv = full_autoconv(child)

            cpu_result = window_scan_cpu(conv, child, table, d_child, m)
            gpu_result = window_scan_gpu(conv, child, table, d_child, m)

            if cpu_result[0] == gpu_result[0]:
                n_agree += 1
            else:
                n_disagree += 1

        assert n_disagree == 0, \
            f"{n_disagree}/{n_agree + n_disagree} window scan disagreements"


class TestGrayCode:
    """Verify Gray code enumeration is complete and changes one position per step."""

    @pytest.mark.parametrize("parent,m", [
        (np.array([5, 5, 5, 5], dtype=np.int32), 20),
        (np.array([10, 10], dtype=np.int32), 20),
        (np.array([3, 7, 4, 6], dtype=np.int32), 20),
        (np.array([1, 2, 3, 4, 5, 3, 1, 1], dtype=np.int32), 20),
    ])
    def test_completeness(self, parent, m):
        """Gray code visits exactly the right number of children."""
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        result = _compute_bin_ranges(parent, m, 1.4, d_child, n_half_child)
        if result is None:
            pytest.skip("No valid bin ranges for this parent")
        lo_arr, hi_arr, total_children = result

        children_seen = []
        for child, pos, *_ in gray_code_enumerate(lo_arr, hi_arr, parent):
            children_seen.append(child.copy())

        assert len(children_seen) == total_children, \
            f"Expected {total_children} children, got {len(children_seen)}"

    @pytest.mark.parametrize("parent,m", [
        (np.array([5, 5, 5, 5], dtype=np.int32), 20),
        (np.array([3, 7, 4, 6], dtype=np.int32), 20),
    ])
    def test_single_position_change(self, parent, m):
        """Each Gray code step changes exactly one parent position."""
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        result = _compute_bin_ranges(parent, m, 1.4, d_child, n_half_child)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        prev_child = None
        for child, pos, old1, old2, new1, new2 in gray_code_enumerate(lo_arr, hi_arr, parent):
            if prev_child is not None:
                # Exactly 2 bins should differ (k1=2*pos, k2=2*pos+1)
                diff_indices = np.where(child != prev_child)[0]
                assert len(diff_indices) == 2, \
                    f"Expected 2 bins to change, got {len(diff_indices)}: {diff_indices}"
                assert diff_indices[0] == 2 * pos
                assert diff_indices[1] == 2 * pos + 1
            prev_child = child.copy()

    @pytest.mark.parametrize("parent,m", [
        (np.array([5, 5, 5, 5], dtype=np.int32), 20),
        (np.array([3, 7, 4, 6], dtype=np.int32), 20),
    ])
    def test_no_duplicates(self, parent, m):
        """Gray code visits each child exactly once."""
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        result = _compute_bin_ranges(parent, m, 1.4, d_child, n_half_child)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, total_children = result

        seen = set()
        for child, *_ in gray_code_enumerate(lo_arr, hi_arr, parent):
            key = tuple(child)
            assert key not in seen, f"Duplicate child: {key}"
            seen.add(key)

        assert len(seen) == total_children


class TestEndToEnd:
    """End-to-end: compare GPU-logic reimplementation against CPU fused kernel."""

    def _gpu_logic_process_parent(self, parent_int, m, c_target, n_half_child):
        """Process one parent using GPU-equivalent logic (Python reimplementation).

        Tests every child with full autoconv + window scan (no quick-check,
        no subtree pruning, no lazy QC). This is the baseline correctness check.
        """
        d_parent = len(parent_int)
        d_child = 2 * d_parent

        result = _compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child)
        if result is None:
            return np.empty((0, d_child), dtype=np.int32), 0

        lo_arr, hi_arr, total_children = result
        table = gpu_build_threshold_table(d_child, m, c_target)

        # Asymmetry filter (matches both CPU and GPU)
        threshold_asym = math.sqrt(c_target / 2.0)
        left_sum = sum(int(parent_int[i]) for i in range(d_parent // 2))
        left_frac = left_sum / float(m)
        if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
            return np.empty((0, d_child), dtype=np.int32), 0

        survivors = []
        n_tested = 0

        for child, pos, old1, old2, new1, new2 in \
                gray_code_enumerate(lo_arr, hi_arr, parent_int):
            n_tested += 1
            conv = full_autoconv(child)
            pruned, *_ = window_scan_gpu(conv, child, table, d_child, m)

            if not pruned:
                # Canonicalize
                rev = child[::-1]
                use_rev = False
                for i in range(d_child):
                    if rev[i] < child[i]:
                        use_rev = True
                        break
                    elif rev[i] > child[i]:
                        break
                if use_rev:
                    survivors.append(rev.copy())
                else:
                    survivors.append(child.copy())

        if survivors:
            result_arr = np.array(survivors, dtype=np.int32)
        else:
            result_arr = np.empty((0, d_child), dtype=np.int32)

        return result_arr, n_tested

    def _sort_dedup(self, arr):
        if arr.shape[0] == 0:
            return arr
        d = arr.shape[1]
        keys = [arr[:, c] for c in reversed(range(d))]
        order = np.lexsort(keys)
        arr = arr[order]
        mask = np.ones(arr.shape[0], dtype=bool)
        mask[1:] = np.any(arr[1:] != arr[:-1], axis=1)
        return arr[mask]

    def _load_parents(self, level):
        ckpt = {1: "checkpoint_L0_survivors.npy",
                2: "checkpoint_L1_survivors.npy",
                3: "checkpoint_L2_survivors.npy"}
        path = os.path.join(_project_dir, "data", ckpt[level])
        if not os.path.exists(path):
            pytest.skip(f"Checkpoint not found: {path}")
        return np.load(path)

    @pytest.mark.parametrize("level,n_parents,c_target", [
        (1, 10, 1.4),
        (1, 10, 1.30),
    ])
    def test_gpu_logic_matches_cpu(self, level, n_parents, c_target):
        """GPU-equivalent logic (no optimizations) matches CPU fused kernel."""
        m = 20
        all_parents = self._load_parents(level)
        parents = all_parents[:n_parents]
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        cpu_survivors = []
        gpu_survivors = []

        for i, parent in enumerate(parents):
            # CPU
            cpu_surv, cpu_count = process_parent_fused(parent, m, c_target, n_half_child)
            cpu_survivors.append(cpu_surv)

            # GPU-logic reimplementation
            gpu_surv, gpu_count = self._gpu_logic_process_parent(
                parent, m, c_target, n_half_child)
            gpu_survivors.append(gpu_surv)

        cpu_all = self._sort_dedup(np.concatenate(cpu_survivors, axis=0)
                                    if cpu_survivors else
                                    np.empty((0, d_child), dtype=np.int32))
        gpu_all = self._sort_dedup(np.concatenate(gpu_survivors, axis=0)
                                    if gpu_survivors else
                                    np.empty((0, d_child), dtype=np.int32))

        assert cpu_all.shape == gpu_all.shape, \
            f"Survivor count mismatch: CPU={cpu_all.shape[0]}, GPU={gpu_all.shape[0]}"
        assert np.array_equal(cpu_all, gpu_all), \
            "Survivor arrays differ"

    @pytest.mark.parametrize("level,n_parents,c_target", [
        (1, None, 1.4),   # Full L0->L1
        (1, None, 1.30),  # Full L0->L1 at different target
    ])
    def test_full_L1_match(self, level, n_parents, c_target):
        """Full L0->L1 cascade: GPU-logic and CPU produce identical survivors."""
        m = 20
        all_parents = self._load_parents(level)
        if n_parents is not None:
            parents = all_parents[:n_parents]
        else:
            parents = all_parents
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        cpu_survivors = []
        gpu_survivors = []

        t0 = time.time()
        for i, parent in enumerate(parents):
            cpu_surv, _ = process_parent_fused(parent, m, c_target, n_half_child)
            cpu_survivors.append(cpu_surv)

            gpu_surv, _ = self._gpu_logic_process_parent(
                parent, m, c_target, n_half_child)
            gpu_survivors.append(gpu_surv)

        elapsed = time.time() - t0

        cpu_all = self._sort_dedup(np.concatenate(cpu_survivors, axis=0)
                                    if any(s.shape[0] > 0 for s in cpu_survivors)
                                    else np.empty((0, d_child), dtype=np.int32))
        gpu_all = self._sort_dedup(np.concatenate(gpu_survivors, axis=0)
                                    if any(s.shape[0] > 0 for s in gpu_survivors)
                                    else np.empty((0, d_child), dtype=np.int32))

        print(f"\n  L{level} c={c_target}: {len(parents)} parents, "
              f"CPU={cpu_all.shape[0]} GPU={gpu_all.shape[0]} survivors, "
              f"{elapsed:.1f}s")

        if cpu_all.shape[0] != gpu_all.shape[0]:
            cpu_set = set(map(tuple, cpu_all))
            gpu_set = set(map(tuple, gpu_all))
            only_cpu = cpu_set - gpu_set
            only_gpu = gpu_set - cpu_set
            print(f"  CPU-only: {len(only_cpu)}, GPU-only: {len(only_gpu)}")
            for s in list(only_cpu)[:3]:
                print(f"    CPU-only: {s}")
            for s in list(only_gpu)[:3]:
                print(f"    GPU-only: {s}")

        assert cpu_all.shape == gpu_all.shape, \
            f"Survivor count mismatch: CPU={cpu_all.shape[0]}, GPU={gpu_all.shape[0]}"
        assert np.array_equal(cpu_all, gpu_all), "Survivor arrays differ"


class TestBinRanges:
    """Verify bin range computation matches between CPU and GPU."""

    @pytest.mark.parametrize("m,c_target,d_child", [
        (20, 1.4, 8), (20, 1.4, 16), (20, 1.4, 32), (20, 1.4, 64),
        (20, 1.30, 8), (20, 1.30, 32),
        (15, 1.33, 12), (15, 1.35, 12),
    ])
    def test_gpu_bin_ranges_match_cpu(self, m, c_target, d_child):
        """GPU bin range formula matches CPU _compute_bin_ranges."""
        # GPU formula (after fix)
        corr_val = 2.0 / m + 1.0 / (m * m)
        thresh = c_target + corr_val + 1e-9
        x_cap_gpu = int(math.floor(m * math.sqrt(thresh / d_child)))
        x_cap_cs_gpu = int(math.floor(m * math.sqrt(c_target / d_child))) + 1
        x_cap_gpu = min(x_cap_gpu, x_cap_cs_gpu, m)
        x_cap_gpu = max(x_cap_gpu, 0)

        # CPU formula
        n_half_child = d_child // 2
        corr_cpu = correction(m, n_half_child)
        thresh_cpu = c_target + corr_cpu + 1e-9
        x_cap_cpu = int(math.floor(m * math.sqrt(thresh_cpu / d_child)))
        x_cap_cs_cpu = int(math.floor(m * math.sqrt(c_target / d_child))) + 1
        x_cap_cpu = min(x_cap_cpu, x_cap_cs_cpu, m)
        x_cap_cpu = max(x_cap_cpu, 0)

        assert x_cap_gpu == x_cap_cpu, \
            f"x_cap mismatch: GPU={x_cap_gpu}, CPU={x_cap_cpu}"
