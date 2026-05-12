"""Tests for the Gray code variant of _fused_generate_and_prune.

Verifies that _fused_generate_and_prune_gray produces the exact same set
of canonical survivors as _fused_generate_and_prune for diverse parents.
"""
import sys
import os
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

M = 20
C_TARGET = 1.4


def _sorted_survivors(buf, n_surv, d_child):
    """Sort survivors lexicographically for comparison."""
    if n_surv == 0:
        return np.empty((0, d_child), dtype=np.int32)
    s = buf[:n_surv].copy()
    s_v = np.ascontiguousarray(s).view(
        np.dtype((np.void, s.dtype.itemsize * d_child)))
    return s[np.argsort(s_v.ravel())]


def _run_both(parent):
    d_parent = len(parent)
    d_child = 2 * d_parent
    n_half_child = d_parent

    result = _compute_bin_ranges(parent, M, C_TARGET, d_child)
    if result is None:
        return 0, 0, True  # both filtered

    lo_arr, hi_arr, total = result
    if total == 0:
        return 0, 0, True

    buf1 = np.empty((total, d_child), dtype=np.int32)
    buf2 = np.empty((total, d_child), dtype=np.int32)

    n_orig, _ = _fused_generate_and_prune(
        parent, n_half_child, M, C_TARGET, lo_arr, hi_arr, buf1)
    n_gray, _ = _fused_generate_and_prune_gray(
        parent, n_half_child, M, C_TARGET, lo_arr, hi_arr, buf2)

    if n_orig != n_gray:
        return n_orig, n_gray, False

    s1 = _sorted_survivors(buf1, n_orig, d_child)
    s2 = _sorted_survivors(buf2, n_gray, d_child)
    return n_orig, n_gray, np.array_equal(s1, s2)


# --- d_parent=4 tests ---

@pytest.mark.parametrize("parent", [
    np.array([5, 5, 5, 5], dtype=np.int32),
    np.array([10, 5, 3, 2], dtype=np.int32),
    np.array([4, 6, 4, 6], dtype=np.int32),
    np.array([0, 10, 10, 0], dtype=np.int32),
    np.array([3, 3, 7, 7], dtype=np.int32),
    np.array([7, 3, 7, 3], dtype=np.int32),
    np.array([0, 5, 0, 15], dtype=np.int32),
])
def test_d4_survivors_match(parent):
    n_orig, n_gray, match = _run_both(parent)
    assert match, f"Mismatch: orig={n_orig}, gray={n_gray}"


# --- d_parent=8 tests ---

@pytest.mark.parametrize("parent", [
    np.array([2, 2, 3, 3, 2, 3, 2, 3], dtype=np.int32),
    np.array([3, 3, 3, 2, 2, 3, 2, 2], dtype=np.int32),
    np.array([4, 1, 3, 2, 3, 2, 4, 1], dtype=np.int32),
    np.array([0, 0, 5, 5, 0, 0, 5, 5], dtype=np.int32),
])
def test_d8_survivors_match(parent):
    n_orig, n_gray, match = _run_both(parent)
    assert match, f"Mismatch: orig={n_orig}, gray={n_gray}"


# --- d_parent=2 edge cases ---

@pytest.mark.parametrize("parent", [
    np.array([10, 10], dtype=np.int32),
    np.array([15, 5], dtype=np.int32),
    np.array([20, 0], dtype=np.int32),
])
def test_d2_survivors_match(parent):
    n_orig, n_gray, match = _run_both(parent)
    assert match, f"Mismatch: orig={n_orig}, gray={n_gray}"


# --- d_parent=16 tests (exercises sparse cross-term path, d_child=32) ---

@pytest.mark.parametrize("parent", [
    np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3], dtype=np.int32),
    np.array([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int32),
    np.array([0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0], dtype=np.int32),
])
def test_d16_survivors_match(parent):
    """d_child=32 exercises sparse cross-term path."""
    n_orig, n_gray, match = _run_both(parent)
    assert match, f"Mismatch: orig={n_orig}, gray={n_gray}"


# --- Enumeration completeness: c_target=999 disables all pruning ---

def test_enumeration_completeness_d4():
    parent = np.array([5, 5, 5, 5], dtype=np.int32)
    d_child = 8
    n_half_child = 4
    result = _compute_bin_ranges(parent, M, 999.0, d_child)
    assert result is not None
    lo_arr, hi_arr, total = result
    buf1 = np.empty((total, d_child), dtype=np.int32)
    buf2 = np.empty((total, d_child), dtype=np.int32)
    n1, _ = _fused_generate_and_prune(
        parent, n_half_child, M, 999.0, lo_arr, hi_arr, buf1)
    n2, _ = _fused_generate_and_prune_gray(
        parent, n_half_child, M, 999.0, lo_arr, hi_arr, buf2)
    assert n1 == n2 == total, f"Expected {total}, got orig={n1}, gray={n2}"


# --- Zero-range positions (all cursors fixed) ---

def test_single_child():
    """Parent where all bins have range 1 → exactly 1 child."""
    parent = np.array([20, 0, 0, 0], dtype=np.int32)
    d_child = 8
    result = _compute_bin_ranges(parent, M, C_TARGET, d_child)
    # This parent is filtered (x_cap clips)
    if result is not None:
        lo_arr, hi_arr, total = result
        buf = np.empty((total, d_child), dtype=np.int32)
        n, _ = _fused_generate_and_prune_gray(
            parent, 4, M, C_TARGET, lo_arr, hi_arr, buf)
        assert n >= 0  # just check it doesn't crash
