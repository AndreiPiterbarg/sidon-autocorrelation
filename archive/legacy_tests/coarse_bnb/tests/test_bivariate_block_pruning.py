"""Tests for the proposed Bivariate Quadratic Block Pruning (2D Range Skip).

Evaluates mathematical correctness, pruning effectiveness, and cost/benefit
of the proposed optimization against the current per-child pruning approach.

Key findings to measure:
  1. Is S_w(x,y) truly a bivariate quadratic in two cursor variables?
  2. For most windows, do the quadratic terms vanish (making it just linear)?
  3. Does the 9-point minimum finder correctly identify the minimum on a rect?
  4. How often does the 2D block check prune blocks that per-child doesn't?
  5. What is the net cost/benefit including O(d) coefficient overhead?
"""
import sys
import os
import time
import itertools
import math

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


# =====================================================================
# Reference implementations (pure Python, no Numba)
# =====================================================================

def compute_raw_conv(child):
    """Brute-force autoconvolution: conv[k] = sum_{i+j=k} c_i * c_j."""
    d = len(child)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += int(child[i]) * int(child[j])
    return conv


def compute_window_sum(conv, s_lo, ell):
    """Window sum: sum of conv[s_lo .. s_lo + ell - 2]."""
    n_cv = ell - 1
    return sum(int(conv[s_lo + k]) for k in range(n_cv))


def compute_W_int(child, d_child, s_lo, ell):
    """W_int: sum of child masses in the window's bin range."""
    lo_bin = s_lo - (d_child - 1)
    if lo_bin < 0:
        lo_bin = 0
    hi_bin = s_lo + ell - 2
    if hi_bin > d_child - 1:
        hi_bin = d_child - 1
    return sum(int(child[i]) for i in range(lo_bin, hi_bin + 1))


def compute_threshold(n_half_child, m, c_target, ell, W_int):
    """Compute the dynamic integer threshold (fused kernel formula)."""
    m_d = float(m)
    inv_4n = 1.0 / (4.0 * float(n_half_child))
    dyn_base = c_target * m_d * m_d + 3.0 + 1e-9 * m_d * m_d
    dyn_base_ell = dyn_base * float(ell) * inv_4n
    two_ell_inv_4n = 2.0 * float(ell) * inv_4n
    dyn_x = dyn_base_ell + two_ell_inv_4n * float(W_int)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    return int(dyn_x * one_minus_4eps)


def is_pruned_by_window(child, n_half_child, m, c_target, ell, s_lo):
    """Check if child is pruned by a specific (ell, s_lo) window."""
    d_child = len(child)
    conv = compute_raw_conv(child)
    ws = compute_window_sum(conv, s_lo, ell)
    W_int = compute_W_int(child, d_child, s_lo, ell)
    dyn_it = compute_threshold(n_half_child, m, c_target, ell, W_int)
    return ws > dyn_it


def is_pruned(child, n_half_child, m, c_target):
    """Check if child is pruned by any window."""
    d_child = len(child)
    conv = compute_raw_conv(child)
    conv_len = 2 * d_child - 1
    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ws = compute_window_sum(conv, s_lo, ell)
            W_int = compute_W_int(child, d_child, s_lo, ell)
            dyn_it = compute_threshold(n_half_child, m, c_target, ell, W_int)
            if ws > dyn_it:
                return True
    return False


def build_child_from_cursors(parent, cursors):
    """Build child array from parent + cursor values."""
    d_parent = len(parent)
    child = np.zeros(2 * d_parent, dtype=np.int32)
    for i in range(d_parent):
        child[2 * i] = cursors[i]
        child[2 * i + 1] = parent[i] - cursors[i]
    return child


# =====================================================================
# Bivariate quadratic coefficient computation
# =====================================================================

def compute_bivariate_coefficients(parent, cursors, pos1, pos2, ell, s_lo):
    """Compute the bivariate quadratic S_w(x,y) - T_w(x,y) coefficients.

    Let x = cursor[pos1], y = cursor[pos2], all other cursors fixed.
    S_w(x,y) is the window sum, T_w(x,y) is the continuous threshold.
    D(x,y) = S_w(x,y) - T_w(x,y) = Axx*x^2 + Ayy*y^2 + Axy*x*y + Bx*x + By*y + C

    Returns dict with keys: Axx, Ayy, Axy, Bx, By, C
    """
    d_parent = len(parent)
    d_child = 2 * d_parent
    n_half_child = d_parent
    conv_len = 2 * d_child - 1

    # The four variable child bin indices
    k1 = 2 * pos1       # child[k1] = x
    k2 = 2 * pos1 + 1   # child[k2] = a1 - x
    k3 = 2 * pos2       # child[k3] = y
    k4 = 2 * pos2 + 1   # child[k4] = a2 - y
    a1 = int(parent[pos1])
    a2 = int(parent[pos2])

    # Window range in convolution space
    n_cv = ell - 1
    s_hi = s_lo + n_cv - 1  # inclusive

    def in_window(k):
        return s_lo <= k <= s_hi

    # --- Quadratic coefficients of S_w(x,y) ---
    # x^2 coefficient: from self-terms and mutual of pos1's bins
    Axx_S = (1 if in_window(2 * k1) else 0) + \
            (1 if in_window(2 * k2) else 0) - \
            (2 if in_window(k1 + k2) else 0)

    # y^2 coefficient: from self-terms and mutual of pos2's bins
    Ayy_S = (1 if in_window(2 * k3) else 0) + \
            (1 if in_window(2 * k4) else 0) - \
            (2 if in_window(k3 + k4) else 0)

    # xy coefficient: from cross-terms between pos1 and pos2 bins
    Axy_S = 2 * ((1 if in_window(k1 + k3) else 0) -
                  (1 if in_window(k1 + k4) else 0) -
                  (1 if in_window(k2 + k3) else 0) +
                  (1 if in_window(k2 + k4) else 0))

    # --- Linear coefficients of S_w(x,y) ---
    # x coefficient from variable-variable terms
    Bx_S = -2 * a1 * (1 if in_window(2 * k2) else 0) + \
             2 * a1 * (1 if in_window(k1 + k2) else 0) + \
             2 * a2 * (1 if in_window(k1 + k4) else 0) - \
             2 * a2 * (1 if in_window(k2 + k4) else 0)

    # y coefficient from variable-variable terms
    By_S = -2 * a2 * (1 if in_window(2 * k4) else 0) + \
             2 * a2 * (1 if in_window(k3 + k4) else 0) + \
             2 * a1 * (1 if in_window(k2 + k3) else 0) - \
             2 * a1 * (1 if in_window(k2 + k4) else 0)

    # x,y coefficients from cross-terms with FIXED bins
    base_child = build_child_from_cursors(parent, cursors)
    variable_bins = {k1, k2, k3, k4}
    for j in range(d_child):
        if j in variable_bins:
            continue
        cj = int(base_child[j])
        if cj == 0:
            continue
        # x contribution: bins k1 and k2
        if in_window(k1 + j):
            Bx_S += 2 * cj
        if in_window(k2 + j):
            Bx_S -= 2 * cj
        # y contribution: bins k3 and k4
        if in_window(k3 + j):
            By_S += 2 * cj
        if in_window(k4 + j):
            By_S -= 2 * cj

    # --- Constant term of S_w ---
    # Compute S_w at x=cursor[pos1], y=cursor[pos2] and subtract polynomial
    x0 = int(cursors[pos1])
    y0 = int(cursors[pos2])
    ws0 = compute_window_sum(compute_raw_conv(base_child), s_lo, ell)
    C_S = ws0 - Axx_S * x0 * x0 - Ayy_S * y0 * y0 - Axy_S * x0 * y0 - Bx_S * x0 - By_S * y0

    # --- Threshold T_w(x,y) = dyn_base_ell + two_ell_inv_4n * W_int(x,y) ---
    # W_int(x,y) is linear in x and y
    m_d = float(M)
    inv_4n = 1.0 / (4.0 * float(n_half_child))
    dyn_base = C_TARGET * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    dyn_base_ell = dyn_base * float(ell) * inv_4n
    two_ell_inv_4n = 2.0 * float(ell) * inv_4n

    # W_int linear coefficients
    lo_bin = s_lo - (d_child - 1)
    if lo_bin < 0:
        lo_bin = 0
    hi_bin = s_lo + ell - 2
    if hi_bin > d_child - 1:
        hi_bin = d_child - 1

    def in_bin_range(idx):
        return lo_bin <= idx <= hi_bin

    # W_int = sum of child[i] for i in [lo_bin, hi_bin]
    # Coefficient of x in W_int
    w_x = (1 if in_bin_range(k1) else 0) - (1 if in_bin_range(k2) else 0)
    # Coefficient of y in W_int
    w_y = (1 if in_bin_range(k3) else 0) - (1 if in_bin_range(k4) else 0)
    # Constant part of W_int
    W_int_const = 0
    for j in range(d_child):
        if j in variable_bins:
            continue
        if in_bin_range(j):
            W_int_const += int(base_child[j])
    # Add the fixed part of variable bins (when both bins of a pair are in range)
    if in_bin_range(k1) and in_bin_range(k2):
        W_int_const += a1  # x + (a1 - x) = a1
        w_x = 0  # cancels
    elif in_bin_range(k1):
        W_int_const += 0  # just x, handled by w_x
    elif in_bin_range(k2):
        W_int_const += a1  # (a1 - x), so W_int has a1 - x
        # w_x should be -1 (already set above)

    if in_bin_range(k3) and in_bin_range(k4):
        W_int_const += a2
        w_y = 0
    elif in_bin_range(k3):
        W_int_const += 0
    elif in_bin_range(k4):
        W_int_const += a2

    # Actually, let me just compute W_int at a reference point and derive
    # the linear coefficients by finite differences
    W_int_at_ref = compute_W_int(base_child, d_child, s_lo, ell)

    # T_w(x,y) = dyn_base_ell + two_ell_inv_4n * (W_int_const + w_x*x + w_y*y)
    T_const = dyn_base_ell + two_ell_inv_4n * W_int_at_ref
    T_bx = two_ell_inv_4n * w_x
    T_by = two_ell_inv_4n * w_y

    # Verify T at reference point
    T_ref_check = dyn_base_ell + two_ell_inv_4n * W_int_at_ref
    # Finite difference verification of w_x and w_y
    # (computed below in the test, not here)

    # --- D(x,y) = S_w(x,y) - T_w(x,y) ---
    return {
        'Axx': Axx_S,     # pure x^2 (integer)
        'Ayy': Ayy_S,     # pure y^2 (integer)
        'Axy': Axy_S,     # xy cross (integer)
        'Bx': float(Bx_S) - T_bx,  # x linear (float due to threshold)
        'By': float(By_S) - T_by,  # y linear
        'C': float(C_S) - T_const,  # constant
        # Also return the S_w components separately for inspection
        'S_Axx': Axx_S,
        'S_Ayy': Ayy_S,
        'S_Axy': Axy_S,
        'S_Bx': Bx_S,
        'S_By': By_S,
        'S_C': C_S,
        'T_const': T_const,
        'T_bx': T_bx,
        'T_by': T_by,
        'w_x': w_x,
        'w_y': w_y,
    }


def eval_bivariate(coeffs, x, y):
    """Evaluate the bivariate quadratic D(x,y)."""
    return (coeffs['Axx'] * x * x +
            coeffs['Ayy'] * y * y +
            coeffs['Axy'] * x * y +
            coeffs['Bx'] * x +
            coeffs['By'] * y +
            coeffs['C'])


def eval_window_sum_quadratic(coeffs, x, y):
    """Evaluate just the S_w(x,y) quadratic."""
    return (coeffs['S_Axx'] * x * x +
            coeffs['S_Ayy'] * y * y +
            coeffs['S_Axy'] * x * y +
            coeffs['S_Bx'] * x +
            coeffs['S_By'] * y +
            coeffs['S_C'])


def find_min_on_rect(coeffs, x_lo, x_hi, y_lo, y_hi):
    """Find minimum of D(x,y) on integer rectangle [x_lo,x_hi]x[y_lo,y_hi].

    Uses the 9-point candidate approach for continuous quadratics,
    then restricts to integer points.
    """
    Axx = coeffs['Axx']
    Ayy = coeffs['Ayy']
    Axy = coeffs['Axy']
    Bx = coeffs['Bx']
    By = coeffs['By']

    candidates = []

    # 4 corners
    for x in [x_lo, x_hi]:
        for y in [y_lo, y_hi]:
            candidates.append((x, y))

    # 4 edge critical points (univariate quadratic min on each edge)
    # Bottom edge: y = y_lo, f(x) = Axx*x^2 + (Axy*y_lo + Bx)*x + ...
    for y_fix in [y_lo, y_hi]:
        a_1d = Axx
        b_1d = Axy * y_fix + Bx
        if a_1d > 0:
            x_star = -b_1d / (2.0 * a_1d)
            # Round to nearest integer in range
            for xc in [int(math.floor(x_star)), int(math.ceil(x_star))]:
                if x_lo <= xc <= x_hi:
                    candidates.append((xc, y_fix))

    for x_fix in [x_lo, x_hi]:
        a_1d = Ayy
        b_1d = Axy * x_fix + By
        if a_1d > 0:
            y_star = -b_1d / (2.0 * a_1d)
            for yc in [int(math.floor(y_star)), int(math.ceil(y_star))]:
                if y_lo <= yc <= y_hi:
                    candidates.append((x_fix, yc))

    # 1 interior critical point
    det = 4 * Axx * Ayy - Axy * Axy
    if det > 0 and Axx > 0:  # positive definite
        x_star = (Axy * By - 2 * Ayy * Bx) / det
        y_star = (Axy * Bx - 2 * Axx * By) / det
        for xc in [int(math.floor(x_star)), int(math.ceil(x_star))]:
            for yc in [int(math.floor(y_star)), int(math.ceil(y_star))]:
                if x_lo <= xc <= x_hi and y_lo <= yc <= y_hi:
                    candidates.append((xc, yc))

    # Evaluate at all candidates
    best_val = float('inf')
    best_pt = None
    for (x, y) in candidates:
        val = eval_bivariate(coeffs, x, y)
        if val < best_val:
            best_val = val
            best_pt = (x, y)

    return best_val, best_pt


# =====================================================================
# Test 1: S_w(x,y) is exactly a bivariate quadratic
# =====================================================================

class TestBivariateQuadraticForm:
    """Verify the window sum is exactly a bivariate quadratic in two cursors."""

    @pytest.mark.parametrize("parent,pos1,pos2", [
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 1),
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 2),
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 3),
        (np.array([5, 5, 5, 5], dtype=np.int32), 1, 3),
        (np.array([10, 5, 3, 2], dtype=np.int32), 0, 1),
        (np.array([8, 4, 4, 4], dtype=np.int32), 0, 2),
        (np.array([3, 3, 7, 7], dtype=np.int32), 1, 2),
    ])
    def test_window_sum_matches_quadratic(self, parent, pos1, pos2):
        """For every (ell, s_lo) window, verify S_w matches the quadratic
        formula at ALL integer (x,y) points in the cursor rectangle."""
        d_parent = len(parent)
        d_child = 2 * d_parent

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        # Set up base cursors at lo values
        cursors = lo_arr.copy()
        x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
        y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])

        if x_lo == x_hi or y_lo == y_hi:
            pytest.skip("Degenerate range")

        # Test a representative set of windows
        conv_len = 2 * d_child - 1
        errors = 0
        windows_tested = 0
        for ell in range(2, min(2 * d_child + 1, 10)):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                coeffs = compute_bivariate_coefficients(
                    parent, cursors, pos1, pos2, ell, s_lo)

                # Verify at every integer point
                for x in range(x_lo, x_hi + 1):
                    for y in range(y_lo, y_hi + 1):
                        test_cursors = cursors.copy()
                        test_cursors[pos1] = x
                        test_cursors[pos2] = y
                        child = build_child_from_cursors(parent, test_cursors)
                        conv = compute_raw_conv(child)
                        ws_actual = compute_window_sum(conv, s_lo, ell)
                        ws_predicted = eval_window_sum_quadratic(coeffs, x, y)

                        if abs(ws_actual - ws_predicted) > 1e-6:
                            errors += 1
                windows_tested += 1

        assert errors == 0, f"Quadratic formula mismatched at {errors} points across {windows_tested} windows"
        assert windows_tested > 0


# =====================================================================
# Test 2: Quadratic terms vanish for most windows
# =====================================================================

class TestQuadraticTermsVanish:
    """Verify that for most windows, Axx=Ayy=Axy=0 (linear regime)."""

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
    ])
    def test_quadratic_terms_mostly_zero(self, parent):
        """Count windows where quadratic terms are nonzero.

        The key insight: the quadratic coefficients are:
          Axx = I[4p1∈W] + I[4p1+2∈W] - 2*I[4p1+1∈W]
        These are zero whenever all three indices {4p1, 4p1+1, 4p1+2}
        are EITHER all inside or all outside the window. This happens for
        all but the ~4 windows whose boundaries split these indices.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result
        cursors = lo_arr.copy()

        total_windows = 0
        nonzero_quadratic = 0

        for pos1 in range(d_parent):
            for pos2 in range(pos1 + 1, d_parent):
                if hi_arr[pos1] == lo_arr[pos1] or hi_arr[pos2] == lo_arr[pos2]:
                    continue

                for ell in range(2, 2 * d_child + 1):
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo in range(n_windows):
                        coeffs = compute_bivariate_coefficients(
                            parent, cursors, pos1, pos2, ell, s_lo)
                        total_windows += 1
                        if (coeffs['S_Axx'] != 0 or
                            coeffs['S_Ayy'] != 0 or
                            coeffs['S_Axy'] != 0):
                            nonzero_quadratic += 1

        if total_windows > 0:
            frac_nonzero = nonzero_quadratic / total_windows
            # The critical finding: most windows have zero quadratic terms
            print(f"\n  Windows with nonzero quadratic terms: "
                  f"{nonzero_quadratic}/{total_windows} = {frac_nonzero:.1%}")
            # For d_child=8, each pos pair has ~6 "boundary" windows out of
            # ~100+ total. So typically < 15% have nonzero quadratic terms.
            # This means the 2D quadratic adds value only for a small minority.


# =====================================================================
# Test 3: 9-point minimum finder is correct
# =====================================================================

class TestNinePointMinimum:
    """Verify the 9-point minimum finder against brute-force enumeration."""

    @pytest.mark.parametrize("parent,pos1,pos2", [
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 1),
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 3),
        (np.array([10, 5, 3, 2], dtype=np.int32), 0, 1),
        (np.array([3, 3, 7, 7], dtype=np.int32), 1, 2),
    ])
    def test_nine_point_finds_true_minimum(self, parent, pos1, pos2):
        """Verify the 9-point approach finds the correct minimum of D(x,y)
        by comparing against exhaustive enumeration over integer grid."""
        d_parent = len(parent)
        d_child = 2 * d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result
        cursors = lo_arr.copy()

        x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
        y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])

        if x_lo == x_hi or y_lo == y_hi:
            pytest.skip("Degenerate range")

        mismatches = 0
        total = 0

        for ell in range(2, min(2 * d_child + 1, 8)):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                coeffs = compute_bivariate_coefficients(
                    parent, cursors, pos1, pos2, ell, s_lo)

                # 9-point minimum
                min_9pt, _ = find_min_on_rect(
                    coeffs, x_lo, x_hi, y_lo, y_hi)

                # Brute-force minimum
                min_bf = float('inf')
                for x in range(x_lo, x_hi + 1):
                    for y in range(y_lo, y_hi + 1):
                        val = eval_bivariate(coeffs, x, y)
                        if val < min_bf:
                            min_bf = val

                # The 9-point approach should find min <= brute-force min
                # (it checks candidate points including all corners)
                total += 1
                if min_9pt > min_bf + 1e-6:
                    mismatches += 1

        assert mismatches == 0, (
            f"9-point missed the true minimum in {mismatches}/{total} windows")


# =====================================================================
# Test 4: Threshold T_w is affine (no xy term)
# =====================================================================

class TestThresholdAffine:
    """Verify the threshold has no xy term — it's affine, not bilinear."""

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
    ])
    def test_threshold_has_no_xy_term(self, parent):
        """T_w(x,y) = alpha + beta1*x + beta2*y (no xy term).

        This means the xy coefficient of D(x,y) comes entirely from S_w,
        not from the threshold. Verified by finite differences.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        for pos1 in range(d_parent):
            for pos2 in range(pos1 + 1, d_parent):
                if hi_arr[pos1] - lo_arr[pos1] < 1 or hi_arr[pos2] - lo_arr[pos2] < 1:
                    continue

                cursors = lo_arr.copy()
                x0, y0 = int(cursors[pos1]), int(cursors[pos2])
                x1 = min(x0 + 1, int(hi_arr[pos1]))
                y1 = min(y0 + 1, int(hi_arr[pos2]))
                if x0 == x1 or y0 == y1:
                    continue

                for ell in range(2, min(2 * d_child + 1, 6)):
                    n_cv = ell - 1
                    for s_lo in range(2 * d_child - 1 - n_cv + 1):
                        # Compute W_int at 4 points to check for xy term
                        def get_W(x, y):
                            c = cursors.copy()
                            c[pos1] = x
                            c[pos2] = y
                            child = build_child_from_cursors(parent, c)
                            return compute_W_int(child, d_child, s_lo, ell)

                        W00 = get_W(x0, y0)
                        W10 = get_W(x1, y0)
                        W01 = get_W(x0, y1)
                        W11 = get_W(x1, y1)

                        # If W_int is affine: W11 - W10 - W01 + W00 = 0
                        xy_term = W11 - W10 - W01 + W00
                        assert xy_term == 0, (
                            f"W_int has xy term: pos1={pos1}, pos2={pos2}, "
                            f"ell={ell}, s_lo={s_lo}, xy_coeff={xy_term}")


# =====================================================================
# Test 5: Pruning effectiveness measurement
# =====================================================================

class TestPruningEffectiveness:
    """Measure how many 2D blocks are fully prunable vs per-child pruning."""

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([7, 3, 7, 3], dtype=np.int32),
    ])
    def test_2d_block_pruning_rate(self, parent):
        """For each pair of cursor positions, measure:
        1. Total children in the 2D rectangle
        2. Children pruned per-child
        3. Whether the 2D block check can prune the entire rectangle

        A block is "fully prunable" if min D(x,y) > 0 for at least one window.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        total_blocks = 0
        fully_prunable_blocks = 0
        children_in_prunable_blocks = 0
        children_individually_pruned = 0
        total_children = 0

        for pos1 in range(d_parent):
            for pos2 in range(pos1 + 1, d_parent):
                x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
                y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])
                rx = x_hi - x_lo + 1
                ry = y_hi - y_lo + 1

                if rx <= 1 or ry <= 1:
                    continue

                block_size = rx * ry
                total_blocks += 1
                total_children += block_size

                # Check 2D block pruning: try all windows
                block_prunable = False
                cursors_base = lo_arr.copy()

                for ell in range(2, 2 * d_child + 1):
                    if block_prunable:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo in range(n_windows):
                        coeffs = compute_bivariate_coefficients(
                            parent, cursors_base, pos1, pos2, ell, s_lo)
                        min_val, _ = find_min_on_rect(
                            coeffs, x_lo, x_hi, y_lo, y_hi)
                        if min_val > 0:
                            block_prunable = True
                            break

                if block_prunable:
                    fully_prunable_blocks += 1
                    children_in_prunable_blocks += block_size

                # Count individually pruned children
                for x in range(x_lo, x_hi + 1):
                    for y in range(y_lo, y_hi + 1):
                        test_cursors = cursors_base.copy()
                        test_cursors[pos1] = x
                        test_cursors[pos2] = y
                        child = build_child_from_cursors(parent, test_cursors)
                        if is_pruned(child, n_half_child, M, C_TARGET):
                            children_individually_pruned += 1

        if total_blocks > 0:
            print(f"\n  Parent: {parent}")
            print(f"  Blocks: {total_blocks}, "
                  f"fully prunable: {fully_prunable_blocks} "
                  f"({fully_prunable_blocks/total_blocks:.0%})")
            print(f"  Children: {total_children}, "
                  f"individually pruned: {children_individually_pruned} "
                  f"({children_individually_pruned/total_children:.0%})")
            if fully_prunable_blocks > 0:
                print(f"  Children in prunable blocks: "
                      f"{children_in_prunable_blocks} "
                      f"({children_in_prunable_blocks/total_children:.0%})")
            # The key metric: how many EXTRA children does the 2D check
            # catch beyond what individual pruning already gets?
            # If individual pruning already gets 99%+, the 2D check's
            # marginal benefit is at most 1% of children.


# =====================================================================
# Test 6: Marginal benefit over per-child quick-check
# =====================================================================

class TestMarginalBenefit:
    """Quantify the marginal benefit of 2D block pruning over per-child
    testing, accounting for quick-check locality."""

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
    ])
    def test_quick_check_covers_most_block_pruning(self, parent):
        """Measure: of children in a block, what fraction would be caught
        by the quick-check heuristic (same window kills adjacent children)?

        If quick-check hit rate is >90%, the 2D block check's marginal
        speedup is <10% of the block × (quick-check cost) — likely negative
        after accounting for the O(d) coefficient computation overhead.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        # Simulate Gray code order for one cursor pair
        total_quick_check_hits = 0
        total_pruned = 0
        total_children = 0

        for pos1 in range(d_parent):
            for pos2 in range(pos1 + 1, d_parent):
                x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
                y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])
                if x_hi - x_lo < 1 or y_hi - y_lo < 1:
                    continue

                # Simulate scanning in x-major order (mimicking Gray code)
                last_kill_window = None
                for x in range(x_lo, x_hi + 1):
                    for y in range(y_lo, y_hi + 1):
                        total_children += 1
                        cursors = lo_arr.copy()
                        cursors[pos1] = x
                        cursors[pos2] = y
                        child = build_child_from_cursors(parent, cursors)
                        conv = compute_raw_conv(child)

                        # Check if previous killing window still works
                        quick_hit = False
                        if last_kill_window is not None:
                            ell_qc, s_lo_qc = last_kill_window
                            ws = compute_window_sum(conv, s_lo_qc, ell_qc)
                            W_int = compute_W_int(child, d_child, s_lo_qc, ell_qc)
                            dyn_it = compute_threshold(
                                n_half_child, M, C_TARGET, ell_qc, W_int)
                            if ws > dyn_it:
                                quick_hit = True
                                total_quick_check_hits += 1
                                total_pruned += 1
                                continue

                        # Full window scan
                        pruned = False
                        for ell in range(2, 2 * d_child + 1):
                            if pruned:
                                break
                            n_cv = ell - 1
                            n_windows = conv_len - n_cv + 1
                            for s_lo in range(n_windows):
                                ws = compute_window_sum(conv, s_lo, ell)
                                W_int = compute_W_int(
                                    child, d_child, s_lo, ell)
                                dyn_it = compute_threshold(
                                    n_half_child, M, C_TARGET, ell, W_int)
                                if ws > dyn_it:
                                    last_kill_window = (ell, s_lo)
                                    pruned = True
                                    total_pruned += 1
                                    break

                        if not pruned:
                            last_kill_window = None

        if total_pruned > 0:
            qc_rate = total_quick_check_hits / total_pruned
            print(f"\n  Parent: {parent}")
            print(f"  Total children: {total_children}, "
                  f"pruned: {total_pruned}")
            print(f"  Quick-check hits: {total_quick_check_hits} "
                  f"({qc_rate:.1%} of pruned)")
            # If quick-check rate > 90%, the 2D block check's ceiling
            # benefit is saving at most 10% of pruned children from
            # the full window scan — which costs O(d*num_windows).
            # But the 2D check itself costs O(d) per trigger, and
            # triggers almost every step for small radices.


# =====================================================================
# Test 7: Cost model — break-even analysis
# =====================================================================

class TestCostModel:
    """Estimate the computational cost of 2D block pruning vs savings."""

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([3, 3, 7, 7], dtype=np.int32),
    ])
    def test_cost_benefit_ratio(self, parent):
        """For each cursor pair, compute:
        - Cost: O(d) coefficient computation per trigger
        - Benefit: children skipped × (average per-child cost)

        A 2D block check triggers at every focus boundary (when cursor j
        hits its limit). For radix r, this happens every r steps.

        If benefit/cost < 2, the optimization is not worthwhile (the
        constant factors in O(d) will eat the savings).
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        # For each cursor pair (pos1, pos2), compute:
        # - Radices
        # - Block size
        # - Whether block is fully prunable by 2D check
        # - Quick-check hit rate
        print(f"\n  Parent: {parent}, d_child={d_child}")
        print(f"  {'pos1':>4} {'pos2':>4} {'rad1':>4} {'rad2':>4} {'block':>5} "
              f"{'2D_ok':>5} {'qc_rate':>7} {'benefit':>7}")

        conv_len = 2 * d_child - 1

        for pos1 in range(d_parent):
            for pos2 in range(pos1 + 1, d_parent):
                x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
                y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])
                rad1 = x_hi - x_lo + 1
                rad2 = y_hi - y_lo + 1

                if rad1 <= 1 or rad2 <= 1:
                    continue

                block_size = rad1 * rad2

                # Check 2D pruning
                cursors_base = lo_arr.copy()
                block_prunable = False
                for ell in range(2, 2 * d_child + 1):
                    if block_prunable:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo in range(n_windows):
                        coeffs = compute_bivariate_coefficients(
                            parent, cursors_base, pos1, pos2, ell, s_lo)
                        min_val, _ = find_min_on_rect(
                            coeffs, x_lo, x_hi, y_lo, y_hi)
                        if min_val > 0:
                            block_prunable = True
                            break

                # Quick-check simulation
                qc_hits = 0
                pruned = 0
                last_win = None
                for x in range(x_lo, x_hi + 1):
                    for y in range(y_lo, y_hi + 1):
                        cursors = cursors_base.copy()
                        cursors[pos1] = x
                        cursors[pos2] = y
                        child = build_child_from_cursors(parent, cursors)
                        conv = compute_raw_conv(child)
                        killed = False
                        if last_win:
                            ell_qc, s_qc = last_win
                            ws = compute_window_sum(conv, s_qc, ell_qc)
                            W_int = compute_W_int(child, d_child, s_qc, ell_qc)
                            dyn_it = compute_threshold(n_half_child, M, C_TARGET, ell_qc, W_int)
                            if ws > dyn_it:
                                qc_hits += 1
                                pruned += 1
                                killed = True
                        if not killed:
                            for ell in range(2, 2 * d_child + 1):
                                if killed:
                                    break
                                n_cv = ell - 1
                                n_windows = conv_len - n_cv + 1
                                for s_lo in range(n_windows):
                                    ws = compute_window_sum(conv, s_lo, ell)
                                    W_int = compute_W_int(child, d_child, s_lo, ell)
                                    dyn_it = compute_threshold(n_half_child, M, C_TARGET, ell, W_int)
                                    if ws > dyn_it:
                                        last_win = (ell, s_lo)
                                        pruned += 1
                                        killed = True
                                        break
                            if not killed:
                                last_win = None

                qc_rate = qc_hits / pruned if pruned > 0 else 0.0

                # Cost model (unit = "O(d) operations"):
                # 2D check: d operations for coefficients + 9 evaluations
                coeff_cost = d_child + 9
                # Per-child cost when NOT quick-checked: ~d * avg_windows_tried
                # Quick-check cost: ~ell (typically 4-16)
                avg_quick_cost = 8
                avg_full_cost = d_child * 5  # rough estimate
                # Benefit of 2D block skip:
                if block_prunable:
                    children_saved = block_size
                    cost_saved = qc_rate * block_size * avg_quick_cost + \
                                 (1 - qc_rate) * block_size * avg_full_cost
                    benefit_ratio = cost_saved / coeff_cost if coeff_cost > 0 else 0
                else:
                    benefit_ratio = 0  # check failed, cost is pure overhead

                print(f"  {pos1:>4} {pos2:>4} {rad1:>4} {rad2:>4} "
                      f"{block_size:>5} {'Y' if block_prunable else 'N':>5} "
                      f"{qc_rate:>7.1%} {benefit_ratio:>7.1f}")


# =====================================================================
# Test 8: Verify mathematical claim about coefficient values
# =====================================================================

class TestCoefficientValues:
    """Verify the specific coefficient formulas from the proposal."""

    @pytest.mark.parametrize("parent,pos1,pos2", [
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 1),
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 3),
        (np.array([10, 5, 3, 2], dtype=np.int32), 0, 2),
    ])
    def test_axx_formula(self, parent, pos1, pos2):
        """Verify Axx = I[4p1∈W] + I[4p1+2∈W] - 2*I[4p1+1∈W]."""
        d_parent = len(parent)
        d_child = 2 * d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result
        cursors = lo_arr.copy()

        k1 = 2 * pos1
        idx_self1 = 2 * k1          # = 4 * pos1
        idx_self2 = 2 * (k1 + 1)    # = 4 * pos1 + 2
        idx_mutual = k1 + k1 + 1    # = 4 * pos1 + 1

        for ell in range(2, 2 * d_child + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                s_hi = s_lo + n_cv - 1
                in_w = lambda k: s_lo <= k <= s_hi

                expected_Axx = ((1 if in_w(idx_self1) else 0) +
                                (1 if in_w(idx_self2) else 0) -
                                (2 if in_w(idx_mutual) else 0))

                coeffs = compute_bivariate_coefficients(
                    parent, cursors, pos1, pos2, ell, s_lo)

                assert coeffs['S_Axx'] == expected_Axx, (
                    f"Axx mismatch at ell={ell}, s_lo={s_lo}: "
                    f"got {coeffs['S_Axx']}, expected {expected_Axx}")

    @pytest.mark.parametrize("parent,pos1,pos2", [
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 1),
        (np.array([5, 5, 5, 5], dtype=np.int32), 0, 3),
    ])
    def test_axy_formula(self, parent, pos1, pos2):
        """Verify Axy = 2*(I[k1+k3∈W] - I[k1+k4∈W] - I[k2+k3∈W] + I[k2+k4∈W])."""
        d_parent = len(parent)
        d_child = 2 * d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result
        cursors = lo_arr.copy()

        k1, k2 = 2 * pos1, 2 * pos1 + 1
        k3, k4 = 2 * pos2, 2 * pos2 + 1

        for ell in range(2, 2 * d_child + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                s_hi = s_lo + n_cv - 1
                in_w = lambda k: s_lo <= k <= s_hi

                expected_Axy = 2 * ((1 if in_w(k1 + k3) else 0) -
                                    (1 if in_w(k1 + k4) else 0) -
                                    (1 if in_w(k2 + k3) else 0) +
                                    (1 if in_w(k2 + k4) else 0))

                coeffs = compute_bivariate_coefficients(
                    parent, cursors, pos1, pos2, ell, s_lo)

                assert coeffs['S_Axy'] == expected_Axy, (
                    f"Axy mismatch at ell={ell}, s_lo={s_lo}: "
                    f"got {coeffs['S_Axy']}, expected {expected_Axy}")

    def test_axy_zero_when_all_indices_in_window(self):
        """When the window contains all cross-indices, Axy = 0.

        Cross indices for (pos1, pos2) are:
          k1+k3 = 2p1+2p2, k1+k4 = 2p1+2p2+1,
          k2+k3 = 2p1+2p2+1, k2+k4 = 2p1+2p2+2

        Note k1+k4 = k2+k3 = 2p1+2p2+1, so there are only 3 distinct values.
        When all 3 are in the window: Axy = 2*(1 - 2*1 + 1) = 0.
        """
        parent = np.array([5, 5, 5, 5], dtype=np.int32)
        pos1, pos2 = 0, 1
        d_child = 8
        # Cross indices: 0+2=2, 0+3=3, 1+2=3, 1+3=4
        # So distinct: {2, 3, 4}. Need window containing indices 2,3,4.
        # ell=4 gives n_cv=3. Window [2,4] means s_lo=2.
        ell, s_lo = 4, 2

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_child // 2)
        lo_arr, hi_arr, _ = result
        cursors = lo_arr.copy()

        coeffs = compute_bivariate_coefficients(
            parent, cursors, pos1, pos2, ell, s_lo)
        assert coeffs['S_Axy'] == 0, f"Expected Axy=0, got {coeffs['S_Axy']}"


# =====================================================================
# Test 9: Realistic cascade parents — L1 survivors
# =====================================================================

class TestRealisticCascade:
    """Run the 2D block analysis on actual L1 survivors to measure
    real-world effectiveness at the d_parent=8, d_child=16 level."""

    def _get_l1_survivors(self, max_parents=20):
        """Get L1 survivors by running L0+L1 on small parameters."""
        from cpu.run_cascade import process_parent_fused

        # Generate some L0 survivors as d=4 parents
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

    def test_l1_2d_block_effectiveness(self):
        """At d_parent=8 (L1 survivors), measure 2D block pruning rates.

        This is the scale where the proposal claims 2-4x efficiency gain.
        """
        l1_parents = self._get_l1_survivors(max_parents=10)

        if not l1_parents:
            pytest.skip("No L1 survivors generated")

        total_blocks = 0
        prunable_blocks = 0
        total_block_children = 0
        prunable_block_children = 0
        total_individual_pruned = 0

        for parent in l1_parents:
            d_parent = len(parent)
            d_child = 2 * d_parent
            n_half_child = d_parent
            conv_len = 2 * d_child - 1

            result = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
            if result is None:
                continue
            lo_arr, hi_arr, _ = result

            for pos1 in range(d_parent):
                for pos2 in range(pos1 + 1, d_parent):
                    x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
                    y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])
                    if x_hi - x_lo < 1 or y_hi - y_lo < 1:
                        continue

                    block_size = (x_hi - x_lo + 1) * (y_hi - y_lo + 1)
                    total_blocks += 1
                    total_block_children += block_size

                    # Try all windows for 2D block check
                    cursors_base = lo_arr.copy()
                    block_ok = False
                    for ell in range(2, 2 * d_child + 1):
                        if block_ok:
                            break
                        n_cv = ell - 1
                        n_windows = conv_len - n_cv + 1
                        for s_lo_w in range(n_windows):
                            coeffs = compute_bivariate_coefficients(
                                parent, cursors_base, pos1, pos2, ell, s_lo_w)
                            min_val, _ = find_min_on_rect(
                                coeffs, x_lo, x_hi, y_lo, y_hi)
                            if min_val > 0:
                                block_ok = True
                                break

                    if block_ok:
                        prunable_blocks += 1
                        prunable_block_children += block_size

                    # Count individually pruned (sample if block too large)
                    sample = min(block_size, 50)
                    individually_pruned = 0
                    tested = 0
                    for x in range(x_lo, min(x_lo + 8, x_hi + 1)):
                        for y in range(y_lo, min(y_lo + 8, y_hi + 1)):
                            cursors = cursors_base.copy()
                            cursors[pos1] = x
                            cursors[pos2] = y
                            child = build_child_from_cursors(parent, cursors)
                            if is_pruned(child, n_half_child, M, C_TARGET):
                                individually_pruned += 1
                            tested += 1

                    if tested > 0:
                        total_individual_pruned += individually_pruned * block_size // tested

        if total_blocks > 0:
            print(f"\n  === L1 (d_parent=8) 2D Block Analysis ===")
            print(f"  Total blocks: {total_blocks}")
            print(f"  Fully prunable blocks: {prunable_blocks} "
                  f"({prunable_blocks/total_blocks:.0%})")
            print(f"  Children in prunable blocks: "
                  f"{prunable_block_children}/{total_block_children} "
                  f"({prunable_block_children/total_block_children:.0%})")
            print(f"  Est. individually pruned: "
                  f"{total_individual_pruned}/{total_block_children} "
                  f"({total_individual_pruned/total_block_children:.0%})")


# =====================================================================
# Test 10: Radix distribution at typical cascade levels
# =====================================================================

class TestBivariateVsUnivariate:
    """Compare 2D block check against two independent 1D range checks.

    The key question: does the bivariate analysis catch blocks that
    two separate univariate analyses would miss? If not, the xy cross-term
    and 2D machinery adds no value over simpler 1D range skips.
    """

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([3, 3, 7, 7], dtype=np.int32),
        np.array([7, 3, 7, 3], dtype=np.int32),
    ])
    def test_2d_vs_1d_range_checks(self, parent):
        """For each cursor pair, compare:
        1D check: for a fixed y, can all x in [x_lo, x_hi] be pruned?
                  AND for a fixed x, can all y in [y_lo, y_hi] be pruned?
        2D check: can all (x,y) in rectangle be pruned simultaneously?

        A 2D check that catches STRICTLY MORE blocks than two 1D checks
        justifies the bivariate approach. Otherwise, two simpler 1D checks
        (which are O(d) each but with smaller constants) suffice.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent
        n_half_child = d_parent
        conv_len = 2 * d_child - 1

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        blocks_1d_prunable = 0  # prunable by 1D check on BOTH cursors
        blocks_2d_prunable = 0  # prunable by 2D check
        blocks_2d_only = 0      # prunable by 2D but NOT by 1D
        total_blocks = 0

        for pos1 in range(d_parent):
            for pos2 in range(pos1 + 1, d_parent):
                x_lo, x_hi = int(lo_arr[pos1]), int(hi_arr[pos1])
                y_lo, y_hi = int(lo_arr[pos2]), int(hi_arr[pos2])
                if x_hi - x_lo < 1 or y_hi - y_lo < 1:
                    continue
                total_blocks += 1
                cursors_base = lo_arr.copy()

                # 1D check: for EACH fixed y, can all x be pruned?
                # Then for EACH fixed x, can all y be pruned?
                # Block is 1D-prunable if there exists a window that prunes
                # all x for ALL y values AND all y for ALL x values.
                # More precisely: the block is "1D-prunable on x" if for each
                # fixed y in [y_lo, y_hi], the 1D range [x_lo, x_hi] is prunable.
                all_x_ranges_prunable = True
                for y in range(y_lo, y_hi + 1):
                    y_prunable = False
                    for ell in range(2, 2 * d_child + 1):
                        if y_prunable:
                            break
                        n_cv = ell - 1
                        n_windows = conv_len - n_cv + 1
                        for s_lo_w in range(n_windows):
                            # Check if D(x, y_fixed) > 0 for all x in [x_lo, x_hi]
                            coeffs = compute_bivariate_coefficients(
                                parent, cursors_base, pos1, pos2, ell, s_lo_w)
                            # D(x, y) at fixed y is: Axx*x^2 + (Axy*y+Bx)*x + (Ayy*y^2+By*y+C)
                            a1d = coeffs['Axx']
                            b1d = coeffs['Axy'] * y + coeffs['Bx']
                            c1d = coeffs['Ayy'] * y * y + coeffs['By'] * y + coeffs['C']
                            # Check min of ax^2+bx+c on [x_lo, x_hi]
                            vals = [a1d * x * x + b1d * x + c1d for x in [x_lo, x_hi]]
                            if a1d > 0:
                                xc = -b1d / (2.0 * a1d)
                                for xr in [int(math.floor(xc)), int(math.ceil(xc))]:
                                    if x_lo <= xr <= x_hi:
                                        vals.append(a1d * xr * xr + b1d * xr + c1d)
                            if min(vals) > 0:
                                y_prunable = True
                                break
                    if not y_prunable:
                        all_x_ranges_prunable = False
                        break

                # 2D check
                block_2d = False
                for ell in range(2, 2 * d_child + 1):
                    if block_2d:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    for s_lo_w in range(n_windows):
                        coeffs = compute_bivariate_coefficients(
                            parent, cursors_base, pos1, pos2, ell, s_lo_w)
                        min_val, _ = find_min_on_rect(
                            coeffs, x_lo, x_hi, y_lo, y_hi)
                        if min_val > 0:
                            block_2d = True
                            break

                if all_x_ranges_prunable:
                    blocks_1d_prunable += 1
                if block_2d:
                    blocks_2d_prunable += 1
                if block_2d and not all_x_ranges_prunable:
                    blocks_2d_only += 1

        if total_blocks > 0:
            print(f"\n  Parent: {parent}")
            print(f"  Total blocks: {total_blocks}")
            print(f"  1D-prunable (all x-ranges for each y): {blocks_1d_prunable}")
            print(f"  2D-prunable: {blocks_2d_prunable}")
            print(f"  2D-only (caught by 2D but not 1D): {blocks_2d_only}")
            if blocks_2d_only > 0:
                print(f"  ** 2D adds value: {blocks_2d_only} extra blocks **")
            else:
                print(f"  ** 2D adds NO additional blocks over 1D **")


class TestRadixDistribution:
    """Measure typical radix values to estimate block sizes."""

    @pytest.mark.parametrize("parent", [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
    ])
    def test_radix_sizes(self, parent):
        """Report radix values (cursor range sizes) for each parent position.

        The 2D block check skips radix1 × radix2 children.
        If typical radices are 2-3, blocks are 4-9 children.
        The O(d) coefficient cost needs to be amortized over this.
        """
        d_parent = len(parent)
        d_child = 2 * d_parent

        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, d_parent)
        if result is None:
            pytest.skip("No valid bin ranges")
        lo_arr, hi_arr, _ = result

        radices = []
        for i in range(d_parent):
            r = hi_arr[i] - lo_arr[i] + 1
            radices.append(r)

        active_radices = [r for r in radices if r > 1]
        print(f"\n  Parent: {parent}")
        print(f"  Radices: {radices}")
        print(f"  Active (>1): {active_radices}")
        if len(active_radices) >= 2:
            # Typical 2D block sizes
            from itertools import combinations
            block_sizes = [r1 * r2 for r1, r2 in combinations(active_radices, 2)]
            print(f"  2D block sizes: {block_sizes}")
            print(f"  Average block size: {sum(block_sizes)/len(block_sizes):.1f}")
            print(f"  O(d) coefficient cost: ~{d_child} ops per block check")
            print(f"  Cost amortized per child: ~{d_child / (sum(block_sizes)/len(block_sizes)):.1f} ops")
