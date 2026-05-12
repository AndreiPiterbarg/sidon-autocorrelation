"""Benchmark and validate arc consistency (_tighten_ranges) for the cascade prover.

Sections:
  1. Fast statistics pass -- run _tighten_ranges on ALL L0 parents, collect stats
  2. Aggregate statistics -- tabular summary of reduction ratios, timings, etc.
  3. Correctness validation -- tiny subset with Gray code kernel verification
  4. Improvement: profile-guided ell ordering
  5. Improvement: batched cross-term update (avoids O(conv_len) copy)
  6. Projection to deeper cascade levels (L1->L2, L2->L3)

Usage:
    python tests/test_arc_consistency.py
"""

import os
import sys
import time
import math
import numpy as np
from numba import njit

# Path setup
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
_cs_dir = os.path.join(_root, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)

from pruning import correction
# Import directly from the cpu module
sys.path.insert(0, os.path.join(_cs_dir, "cpu"))
import run_cascade as rc

# -- Parameters --------------------------------------------------------
M = 20
C_TARGET = 1.35


# ======================================================================
# Improvement variant 1: profile-guided ell ordering in _tighten_ranges
# ======================================================================

@njit(cache=True)
def _tighten_ranges_fast_ell(parent_int, lo_arr, hi_arr, m, c_target,
                              n_half_child, use_flat_threshold=False):
    """Arc consistency with profile-guided ell scan order (early exit).

    Identical logic to _tighten_ranges except ell values are scanned in
    empirically-best order (matching the Gray code kernel) instead of
    sequential 2..2*d_child.
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half_child)
    n_half_d = np.float64(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    S_child_plus_1 = int(4 * n_half_child * m + 1)
    ell_count = conv_len

    # Threshold table (same as original)
    threshold_table = np.empty(ell_count * S_child_plus_1, dtype=np.int64)
    flat_corr = 2.0 * m_d + 1.0
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        scale_ell = np.float64(ell) * four_n
        if use_flat_threshold:
            dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
            flat_val = np.int64(dyn_x)
            for w in range(S_child_plus_1):
                threshold_table[idx * S_child_plus_1 + w] = flat_val
        else:
            for w in range(S_child_plus_1):
                corr_w = 1.0 + np.float64(w) / (2.0 * n_half_d)
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                threshold_table[idx * S_child_plus_1 + w] = np.int64(dyn_x)

    # --- Build profile-guided ell order (same as Gray code kernel) ---
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    if d_child >= 20:
        hc = d_child // 2
        for ell in (hc + 1, hc + 2, hc + 3, hc, hc - 1, hc + 4, hc + 5,
                    hc - 2, hc + 6, hc - 3, hc + 7, hc + 8):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                    d_child * 2, d_child + d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
    else:
        phase1_end = min(16, 2 * d_child)
        for ell in range(2, phase1_end + 1):
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = np.int32(1)
            oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                    d_child * 2, d_child + d_child // 2, d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1
    n_ell = oi

    test_conv = np.empty(conv_len, dtype=np.int64)

    max_rounds = d_parent + 1
    for _round in range(max_rounds):
        any_changed = False

        child_min = np.empty(d_child, dtype=np.int32)
        for q in range(d_parent):
            child_min[2 * q] = lo_arr[q]
            child_min[2 * q + 1] = 2 * parent_int[q] - hi_arr[q]

        conv_min = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            ci = np.int64(child_min[i])
            if ci == 0:
                continue
            conv_min[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int64(child_min[j])
                if cj != 0:
                    conv_min[i + j] += np.int64(2) * ci * cj

        max_child_prefix = np.empty(d_child + 1, dtype=np.int64)
        max_child_prefix[0] = np.int64(0)
        for q in range(d_parent):
            max_child_prefix[2 * q + 1] = (max_child_prefix[2 * q]
                                            + np.int64(hi_arr[q]))
            max_child_prefix[2 * q + 2] = (max_child_prefix[2 * q + 1]
                                            + np.int64(2 * parent_int[q] - lo_arr[q]))

        for p in range(d_parent):
            if lo_arr[p] == hi_arr[p]:
                continue

            B_p = parent_int[p]
            k1 = 2 * p
            k2 = 2 * p + 1
            old1 = np.int64(child_min[k1])
            old2 = np.int64(child_min[k2])

            new_lo = lo_arr[p]
            new_hi = hi_arr[p]

            # --- Tighten from low end ---
            for v in range(lo_arr[p], hi_arr[p] + 1):
                new1 = np.int64(v)
                new2 = np.int64(2 * B_p - v)
                delta1 = new1 - old1
                delta2 = new2 - old2

                for kk in range(conv_len):
                    test_conv[kk] = conv_min[kk]
                test_conv[2 * k1] += new1 * new1 - old1 * old1
                test_conv[2 * k2] += new2 * new2 - old2 * old2
                test_conv[k1 + k2] += np.int64(2) * (new1 * new2 - old1 * old2)
                for j in range(d_child):
                    if j == k1 or j == k2:
                        continue
                    cj = np.int64(child_min[j])
                    if cj != 0:
                        test_conv[k1 + j] += np.int64(2) * delta1 * cj
                        test_conv[k2 + j] += np.int64(2) * delta2 * cj

                # Profile-guided ell scan with early exit
                infeasible = False
                for ei in range(n_ell):
                    ell = ell_order[ei]
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    if n_windows <= 0:
                        continue
                    ell_idx = ell - 2

                    ws = np.int64(0)
                    for kk in range(n_cv):
                        ws += test_conv[kk]

                    hb = ell - 2
                    if hb > d_child - 1:
                        hb = d_child - 1
                    W_max = max_child_prefix[hb + 1]
                    if W_max >= np.int64(S_child_plus_1):
                        W_max = np.int64(S_child_plus_1 - 1)
                    if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                        infeasible = True
                        break

                    for s in range(1, n_windows):
                        ws += test_conv[s + n_cv - 1] - test_conv[s - 1]
                        lb = s - (d_child - 1)
                        if lb < 0:
                            lb = 0
                        hb = s + ell - 2
                        if hb > d_child - 1:
                            hb = d_child - 1
                        W_max = max_child_prefix[hb + 1] - max_child_prefix[lb]
                        if W_max >= np.int64(S_child_plus_1):
                            W_max = np.int64(S_child_plus_1 - 1)
                        if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                            infeasible = True
                            break
                    if infeasible:
                        break

                if infeasible:
                    if v == new_lo:
                        new_lo = v + 1
                    else:
                        break
                else:
                    break

            # --- Tighten from high end ---
            for v in range(hi_arr[p], new_lo - 1, -1):
                new1 = np.int64(v)
                new2 = np.int64(2 * B_p - v)
                delta1 = new1 - old1
                delta2 = new2 - old2

                for kk in range(conv_len):
                    test_conv[kk] = conv_min[kk]
                test_conv[2 * k1] += new1 * new1 - old1 * old1
                test_conv[2 * k2] += new2 * new2 - old2 * old2
                test_conv[k1 + k2] += np.int64(2) * (new1 * new2 - old1 * old2)
                for j in range(d_child):
                    if j == k1 or j == k2:
                        continue
                    cj = np.int64(child_min[j])
                    if cj != 0:
                        test_conv[k1 + j] += np.int64(2) * delta1 * cj
                        test_conv[k2 + j] += np.int64(2) * delta2 * cj

                infeasible = False
                for ei in range(n_ell):
                    ell = ell_order[ei]
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    if n_windows <= 0:
                        continue
                    ell_idx = ell - 2

                    ws = np.int64(0)
                    for kk in range(n_cv):
                        ws += test_conv[kk]

                    hb = ell - 2
                    if hb > d_child - 1:
                        hb = d_child - 1
                    W_max = max_child_prefix[hb + 1]
                    if W_max >= np.int64(S_child_plus_1):
                        W_max = np.int64(S_child_plus_1 - 1)
                    if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                        infeasible = True
                        break

                    for s in range(1, n_windows):
                        ws += test_conv[s + n_cv - 1] - test_conv[s - 1]
                        lb = s - (d_child - 1)
                        if lb < 0:
                            lb = 0
                        hb = s + ell - 2
                        if hb > d_child - 1:
                            hb = d_child - 1
                        W_max = max_child_prefix[hb + 1] - max_child_prefix[lb]
                        if W_max >= np.int64(S_child_plus_1):
                            W_max = np.int64(S_child_plus_1 - 1)
                        if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                            infeasible = True
                            break
                    if infeasible:
                        break

                if infeasible:
                    if v == new_hi:
                        new_hi = v - 1
                    else:
                        break
                else:
                    break

            if new_lo != lo_arr[p] or new_hi != hi_arr[p]:
                lo_arr[p] = new_lo
                hi_arr[p] = new_hi
                any_changed = True
                if new_lo > new_hi:
                    return 0

        if not any_changed:
            break

    total = np.int64(1)
    for i in range(d_parent):
        r = hi_arr[i] - lo_arr[i] + 1
        if r <= 0:
            return 0
        total *= np.int64(r)
    return total


# ======================================================================
# Improvement variant 2: batched cross-term (no O(conv_len) copy)
# ======================================================================

@njit(cache=True)
def _tighten_ranges_batched_cross(parent_int, lo_arr, hi_arr, m, c_target,
                                   n_half_child, use_flat_threshold=False):
    """Arc consistency with precomputed cross-term coefficients.

    Instead of copying conv_min[] for each candidate v then applying deltas,
    precompute the cross-term coefficient vector for position p so that:
        test_conv[k] = conv_min[k] + cross1[k]*delta1 + cross2[k]*delta2
                       + self_term_correction(k, new1, new2, old1, old2)
    This avoids the O(conv_len) array copy per candidate.
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half_child)
    n_half_d = np.float64(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    S_child_plus_1 = int(4 * n_half_child * m + 1)
    ell_count = conv_len

    threshold_table = np.empty(ell_count * S_child_plus_1, dtype=np.int64)
    flat_corr = 2.0 * m_d + 1.0
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        scale_ell = np.float64(ell) * four_n
        if use_flat_threshold:
            dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
            flat_val = np.int64(dyn_x)
            for w in range(S_child_plus_1):
                threshold_table[idx * S_child_plus_1 + w] = flat_val
        else:
            for w in range(S_child_plus_1):
                corr_w = 1.0 + np.float64(w) / (2.0 * n_half_d)
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                threshold_table[idx * S_child_plus_1 + w] = np.int64(dyn_x)

    # Precomputed cross-term coefficient arrays (reused per position p)
    cross1 = np.zeros(conv_len, dtype=np.int64)
    cross2 = np.zeros(conv_len, dtype=np.int64)

    max_rounds = d_parent + 1
    for _round in range(max_rounds):
        any_changed = False

        child_min = np.empty(d_child, dtype=np.int32)
        for q in range(d_parent):
            child_min[2 * q] = lo_arr[q]
            child_min[2 * q + 1] = 2 * parent_int[q] - hi_arr[q]

        conv_min = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            ci = np.int64(child_min[i])
            if ci == 0:
                continue
            conv_min[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int64(child_min[j])
                if cj != 0:
                    conv_min[i + j] += np.int64(2) * ci * cj

        max_child_prefix = np.empty(d_child + 1, dtype=np.int64)
        max_child_prefix[0] = np.int64(0)
        for q in range(d_parent):
            max_child_prefix[2 * q + 1] = (max_child_prefix[2 * q]
                                            + np.int64(hi_arr[q]))
            max_child_prefix[2 * q + 2] = (max_child_prefix[2 * q + 1]
                                            + np.int64(2 * parent_int[q] - lo_arr[q]))

        for p in range(d_parent):
            if lo_arr[p] == hi_arr[p]:
                continue

            B_p = parent_int[p]
            k1 = 2 * p
            k2 = 2 * p + 1
            old1 = np.int64(child_min[k1])
            old2 = np.int64(child_min[k2])

            # Precompute cross-term coefficients for position p
            # cross1[k] = coefficient of delta1 in test_conv[k]
            # cross2[k] = coefficient of delta2 in test_conv[k]
            for kk in range(conv_len):
                cross1[kk] = np.int64(0)
                cross2[kk] = np.int64(0)
            for j in range(d_child):
                if j == k1 or j == k2:
                    continue
                cj = np.int64(child_min[j])
                if cj != 0:
                    cross1[k1 + j] = np.int64(2) * cj
                    cross2[k2 + j] = np.int64(2) * cj

            new_lo = lo_arr[p]
            new_hi = hi_arr[p]

            # --- Tighten from low end ---
            for v in range(lo_arr[p], hi_arr[p] + 1):
                new1 = np.int64(v)
                new2 = np.int64(2 * B_p - v)
                delta1 = new1 - old1
                delta2 = new2 - old2

                # Self-term corrections (only 3 conv indices affected)
                self_2k1 = new1 * new1 - old1 * old1
                self_2k2 = new2 * new2 - old2 * old2
                self_k1k2 = np.int64(2) * (new1 * new2 - old1 * old2)

                # Check all windows -- compute test_conv on the fly
                infeasible = False
                for ell in range(2, 2 * d_child + 1):
                    if infeasible:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    if n_windows <= 0:
                        continue
                    ell_idx = ell - 2

                    # Build initial window sum [0..n_cv-1]
                    ws = np.int64(0)
                    for kk in range(n_cv):
                        val = conv_min[kk] + cross1[kk] * delta1 + cross2[kk] * delta2
                        if kk == 2 * k1:
                            val += self_2k1
                        if kk == 2 * k2:
                            val += self_2k2
                        if kk == k1 + k2:
                            val += self_k1k2
                        ws += val

                    hb = ell - 2
                    if hb > d_child - 1:
                        hb = d_child - 1
                    W_max = max_child_prefix[hb + 1]
                    if W_max >= np.int64(S_child_plus_1):
                        W_max = np.int64(S_child_plus_1 - 1)
                    if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                        infeasible = True
                        break

                    for s in range(1, n_windows):
                        # Slide: add new element, drop old
                        kk_add = s + n_cv - 1
                        val_add = conv_min[kk_add] + cross1[kk_add] * delta1 + cross2[kk_add] * delta2
                        if kk_add == 2 * k1:
                            val_add += self_2k1
                        if kk_add == 2 * k2:
                            val_add += self_2k2
                        if kk_add == k1 + k2:
                            val_add += self_k1k2

                        kk_sub = s - 1
                        val_sub = conv_min[kk_sub] + cross1[kk_sub] * delta1 + cross2[kk_sub] * delta2
                        if kk_sub == 2 * k1:
                            val_sub += self_2k1
                        if kk_sub == 2 * k2:
                            val_sub += self_2k2
                        if kk_sub == k1 + k2:
                            val_sub += self_k1k2

                        ws += val_add - val_sub
                        lb = s - (d_child - 1)
                        if lb < 0:
                            lb = 0
                        hb = s + ell - 2
                        if hb > d_child - 1:
                            hb = d_child - 1
                        W_max = max_child_prefix[hb + 1] - max_child_prefix[lb]
                        if W_max >= np.int64(S_child_plus_1):
                            W_max = np.int64(S_child_plus_1 - 1)
                        if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                            infeasible = True
                            break

                if infeasible:
                    if v == new_lo:
                        new_lo = v + 1
                    else:
                        break
                else:
                    break

            # --- Tighten from high end ---
            for v in range(hi_arr[p], new_lo - 1, -1):
                new1 = np.int64(v)
                new2 = np.int64(2 * B_p - v)
                delta1 = new1 - old1
                delta2 = new2 - old2

                self_2k1 = new1 * new1 - old1 * old1
                self_2k2 = new2 * new2 - old2 * old2
                self_k1k2 = np.int64(2) * (new1 * new2 - old1 * old2)

                infeasible = False
                for ell in range(2, 2 * d_child + 1):
                    if infeasible:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    if n_windows <= 0:
                        continue
                    ell_idx = ell - 2

                    ws = np.int64(0)
                    for kk in range(n_cv):
                        val = conv_min[kk] + cross1[kk] * delta1 + cross2[kk] * delta2
                        if kk == 2 * k1:
                            val += self_2k1
                        if kk == 2 * k2:
                            val += self_2k2
                        if kk == k1 + k2:
                            val += self_k1k2
                        ws += val

                    hb = ell - 2
                    if hb > d_child - 1:
                        hb = d_child - 1
                    W_max = max_child_prefix[hb + 1]
                    if W_max >= np.int64(S_child_plus_1):
                        W_max = np.int64(S_child_plus_1 - 1)
                    if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                        infeasible = True
                        break

                    for s in range(1, n_windows):
                        kk_add = s + n_cv - 1
                        val_add = conv_min[kk_add] + cross1[kk_add] * delta1 + cross2[kk_add] * delta2
                        if kk_add == 2 * k1:
                            val_add += self_2k1
                        if kk_add == 2 * k2:
                            val_add += self_2k2
                        if kk_add == k1 + k2:
                            val_add += self_k1k2

                        kk_sub = s - 1
                        val_sub = conv_min[kk_sub] + cross1[kk_sub] * delta1 + cross2[kk_sub] * delta2
                        if kk_sub == 2 * k1:
                            val_sub += self_2k1
                        if kk_sub == 2 * k2:
                            val_sub += self_2k2
                        if kk_sub == k1 + k2:
                            val_sub += self_k1k2

                        ws += val_add - val_sub
                        lb = s - (d_child - 1)
                        if lb < 0:
                            lb = 0
                        hb = s + ell - 2
                        if hb > d_child - 1:
                            hb = d_child - 1
                        W_max = max_child_prefix[hb + 1] - max_child_prefix[lb]
                        if W_max >= np.int64(S_child_plus_1):
                            W_max = np.int64(S_child_plus_1 - 1)
                        if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                            infeasible = True
                            break

                if infeasible:
                    if v == new_hi:
                        new_hi = v - 1
                    else:
                        break
                else:
                    break

            if new_lo != lo_arr[p] or new_hi != hi_arr[p]:
                lo_arr[p] = new_lo
                hi_arr[p] = new_hi
                any_changed = True
                if new_lo > new_hi:
                    return 0

        if not any_changed:
            break

    total = np.int64(1)
    for i in range(d_parent):
        r = hi_arr[i] - lo_arr[i] + 1
        if r <= 0:
            return 0
        total *= np.int64(r)
    return total


# ======================================================================
# Helper: compute ranges + tighten, return stats
# ======================================================================

def _get_ranges_and_tighten(parent_int, m, c_target, n_half_child,
                             tighten_fn=None):
    """Returns (lo_before, hi_before, total_before,
               lo_after, hi_after, total_after, elapsed_us)
    or None if bin ranges are empty.
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    result = rc._compute_bin_ranges(parent_int, m, c_target, d_child,
                                     n_half_child)
    if result is None:
        return None
    lo_arr, hi_arr, total_before = result
    lo_before = lo_arr.copy()
    hi_before = hi_arr.copy()

    if tighten_fn is None:
        tighten_fn = rc._tighten_ranges

    t0 = time.perf_counter()
    total_after = tighten_fn(parent_int, lo_arr, hi_arr, m, c_target,
                              n_half_child)
    elapsed_us = (time.perf_counter() - t0) * 1e6

    return (lo_before, hi_before, int(total_before),
            lo_arr.copy(), hi_arr.copy(), int(total_after), elapsed_us)


def _run_kernel_on_ranges(parent_int, n_half_child, m, c_target,
                           lo_arr, hi_arr, total_children):
    """Run Gray code kernel on given ranges, return sorted canonical survivors."""
    d_child = 2 * len(parent_int)
    buf_cap = min(total_children, 10_000_000)
    out_buf = np.empty((max(buf_cap, 1), d_child), dtype=np.int32)

    n_surv, _ = rc._fused_generate_and_prune_gray(
        parent_int, n_half_child, m, c_target,
        lo_arr, hi_arr, out_buf)

    if n_surv > buf_cap:
        out_buf = np.empty((n_surv, d_child), dtype=np.int32)
        n_surv, _ = rc._fused_generate_and_prune_gray(
            parent_int, n_half_child, m, c_target,
            lo_arr, hi_arr, out_buf)

    survivors = out_buf[:n_surv].copy()
    # Sort for comparison
    if len(survivors) > 0:
        # Lexicographic sort
        keys = tuple(survivors[:, i] for i in range(d_child - 1, -1, -1))
        order = np.lexsort(keys)
        survivors = survivors[order]
    return survivors


# ======================================================================
# MAIN
# ======================================================================

def main():
    D0 = 2  # Starting dimension

    print("=" * 72)
    print("ARC CONSISTENCY BENCHMARK")
    print("=" * 72)
    print(f"  Parameters: d0={D0}, m={M}, c_target={C_TARGET}")
    print()

    # -- Generate L0 fresh with d0=2 ----------------------------------
    print("Generating L0 survivors (d0=2)...")
    l0_result = rc.run_level0(n_half=D0/2.0, m=M, c_target=C_TARGET,
                               d0=D0, verbose=True)
    parents = l0_result['survivors']
    n_parents = len(parents)
    d_parent = parents.shape[1]  # = D0 = 2
    d_child = 2 * d_parent       # = 4
    n_half_child = d_parent      # = 2

    print(f"\n  L0 survivors: {n_parents:,}  (d={d_parent})")
    print()

    # -- JIT warmup ----------------------------------------------------
    print("Warming up JIT (first call compiles)...")
    warmup_parent = parents[0].copy()
    _ = _get_ranges_and_tighten(warmup_parent, M, C_TARGET, n_half_child)
    _ = _get_ranges_and_tighten(warmup_parent, M, C_TARGET, n_half_child,
                                 tighten_fn=_tighten_ranges_fast_ell)
    _ = _get_ranges_and_tighten(warmup_parent, M, C_TARGET, n_half_child,
                                 tighten_fn=_tighten_ranges_batched_cross)
    print("  JIT warmup done.\n")

    # ==================================================================
    # SECTION 1: Fast statistics pass (ALL parents)
    # ==================================================================
    print("=" * 72)
    print("SECTION 1: Fast Statistics Pass (ALL parents, original _tighten_ranges)")
    print("=" * 72)

    totals_before = []
    totals_after = []
    reduction_ratios = []
    tighten_times_us = []
    per_pos_shrink = [[] for _ in range(d_parent)]  # shrinkage per position
    fully_pruned_indices = []
    tightened_indices = []   # indices where at least one range shrank

    t_section = time.perf_counter()
    for idx in range(n_parents):
        parent_int = parents[idx]
        res = _get_ranges_and_tighten(parent_int, M, C_TARGET, n_half_child)
        if res is None:
            # Empty bin ranges -> fully pruned before tightening
            totals_before.append(0)
            totals_after.append(0)
            reduction_ratios.append(float('inf'))
            tighten_times_us.append(0.0)
            fully_pruned_indices.append(idx)
            continue

        (lo_b, hi_b, tb, lo_a, hi_a, ta, elapsed) = res
        totals_before.append(tb)
        totals_after.append(ta)
        tighten_times_us.append(elapsed)

        if ta == 0:
            fully_pruned_indices.append(idx)
            reduction_ratios.append(float('inf'))
        else:
            ratio = tb / ta if ta > 0 else float('inf')
            reduction_ratios.append(ratio)

        # Track which parents had tightening
        any_shrunk = False
        for p in range(d_parent):
            range_before = hi_b[p] - lo_b[p] + 1
            range_after = hi_a[p] - lo_a[p] + 1
            shrink = range_before - range_after
            per_pos_shrink[p].append(shrink)
            if shrink > 0:
                any_shrunk = True

        if any_shrunk and ta > 0:
            tightened_indices.append(idx)

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx+1:,}/{n_parents:,}...")

    t_section_elapsed = time.perf_counter() - t_section

    # ==================================================================
    # SECTION 2: Aggregate Statistics
    # ==================================================================
    print()
    print("=" * 72)
    print("SECTION 2: Aggregate Statistics")
    print("=" * 72)

    total_before_sum = sum(totals_before)
    total_after_sum = sum(totals_after)
    n_fully_pruned = len(fully_pruned_indices)

    # Filter finite ratios for stats
    finite_ratios = [r for r in reduction_ratios if r != float('inf')]
    finite_ratios_arr = np.array(finite_ratios) if finite_ratios else np.array([1.0])
    times_arr = np.array(tighten_times_us)

    print(f"\n  Total parents processed:          {n_parents:,}")
    print(f"  Parents fully pruned by AC:       {n_fully_pruned:,} "
          f"({100.0*n_fully_pruned/n_parents:.2f}%)")
    print(f"  Parents with range tightening:    {len(tightened_indices):,} "
          f"({100.0*len(tightened_indices)/n_parents:.2f}%)")
    print()
    print(f"  Total children BEFORE tightening: {total_before_sum:,}")
    print(f"  Total children AFTER  tightening: {total_after_sum:,}")
    if total_before_sum > 0:
        pct_reduction = 100.0 * (1 - total_after_sum / total_before_sum)
        print(f"  Overall reduction:                {pct_reduction:.2f}%")
    print(f"  Children eliminated:              {total_before_sum - total_after_sum:,}")
    est_saved_s = (total_before_sum - total_after_sum) / 7_000_000
    print(f"  Estimated time saved:             {est_saved_s:,.1f} sec "
          f"({est_saved_s/3600:.2f} hr)")

    print(f"\n  -- Reduction ratio (before/after) for non-fully-pruned parents --")
    if len(finite_ratios) > 0:
        print(f"    Min:    {np.min(finite_ratios_arr):.4f}")
        print(f"    Max:    {np.max(finite_ratios_arr):.4f}")
        print(f"    Mean:   {np.mean(finite_ratios_arr):.4f}")
        print(f"    Median: {np.median(finite_ratios_arr):.4f}")
        print(f"    p90:    {np.percentile(finite_ratios_arr, 90):.4f}")
        print(f"    p99:    {np.percentile(finite_ratios_arr, 99):.4f}")
    else:
        print(f"    (all parents fully pruned)")

    print(f"\n  -- _tighten_ranges timing (microseconds) --")
    print(f"    Min:    {np.min(times_arr):.1f}")
    print(f"    Max:    {np.max(times_arr):.1f}")
    print(f"    Mean:   {np.mean(times_arr):.1f}")
    print(f"    Median: {np.median(times_arr):.1f}")
    print(f"    Total:  {np.sum(times_arr)/1e6:.3f} sec")
    print(f"    Wall:   {t_section_elapsed:.3f} sec (includes overhead)")

    print(f"\n  -- Per-position range shrinkage distribution --")
    print(f"    {'Pos':>4s}  {'shrink=0':>10s}  {'shrink=1':>10s}  "
          f"{'shrink=2':>10s}  {'shrink>=3':>10s}")
    for p in range(d_parent):
        arr = np.array(per_pos_shrink[p])
        n0 = int(np.sum(arr == 0))
        n1 = int(np.sum(arr == 1))
        n2 = int(np.sum(arr == 2))
        n3 = int(np.sum(arr >= 3))
        print(f"    {p:4d}  {n0:10,}  {n1:10,}  {n2:10,}  {n3:10,}")

    # ==================================================================
    # SECTION 3: Correctness Validation (TINY subset)
    # ==================================================================
    print()
    print("=" * 72)
    print("SECTION 3: Correctness Validation (tiny subset)")
    print("=" * 72)

    # 3a: Parents where AC tightened -- pick smallest total_after
    MAX_CHILDREN_VALIDATION = 500_000  # ~0.07s per kernel run
    tightened_by_size = sorted(tightened_indices, key=lambda i: totals_after[i])
    small_tightened = []
    for idx in tightened_by_size:
        if totals_after[idx] > 0 and totals_after[idx] <= MAX_CHILDREN_VALIDATION:
            small_tightened.append(idx)
            if len(small_tightened) >= 5:
                break

    if len(small_tightened) == 0 and len(tightened_by_size) > 0:
        # Fall back to smallest available even if large
        small_tightened = tightened_by_size[:3]
        MAX_CHILDREN_VALIDATION = max(totals_after[i] for i in small_tightened) + 1

    print(f"\n  Testing {len(small_tightened)} tightened parents "
          f"(smallest by total_after, limit {MAX_CHILDREN_VALIDATION:,}):")
    for idx in small_tightened:
        parent_int = parents[idx]

        # Get original ranges
        res_orig = rc._compute_bin_ranges(parent_int, M, C_TARGET, d_child,
                                           n_half_child)
        lo_orig, hi_orig, tc_orig = res_orig

        # Get tightened ranges
        lo_tight = lo_orig.copy()
        hi_tight = hi_orig.copy()
        tc_tight = int(rc._tighten_ranges(parent_int, lo_tight, hi_tight,
                                            M, C_TARGET, n_half_child))

        est_time_orig = tc_orig / 7_000_000
        est_time_tight = tc_tight / 7_000_000

        print(f"\n  Parent #{idx}: {parent_int}")
        print(f"    Original:  total={tc_orig:,}  "
              f"(est {est_time_orig:.2f}s)")
        print(f"    Tightened: total={tc_tight:,}  "
              f"(est {est_time_tight:.2f}s)")

        if est_time_orig > 60.0:
            print(f"    SKIP original kernel (>60s estimated)")
            surv_tight = _run_kernel_on_ranges(
                parent_int, n_half_child, M, C_TARGET,
                lo_tight, hi_tight, tc_tight)
            print(f"    Tightened survivors: {len(surv_tight)}")
            print(f"    (cannot compare - original too large)")
            continue

        surv_orig = _run_kernel_on_ranges(
            parent_int, n_half_child, M, C_TARGET,
            lo_orig, hi_orig, tc_orig)
        surv_tight = _run_kernel_on_ranges(
            parent_int, n_half_child, M, C_TARGET,
            lo_tight, hi_tight, tc_tight)

        match = (len(surv_orig) == len(surv_tight) and
                 (len(surv_orig) == 0 or np.array_equal(surv_orig, surv_tight)))
        status = "PASS" if match else "FAIL"
        print(f"    Original survivors:  {len(surv_orig)}")
        print(f"    Tightened survivors: {len(surv_tight)}")
        print(f"    -> {status}")

    # 3b: Parents that AC fully pruned with small original total
    small_pruned = []
    for idx in fully_pruned_indices:
        if totals_before[idx] > 0 and totals_before[idx] < 50_000:
            small_pruned.append(idx)
            if len(small_pruned) >= 3:
                break

    print(f"\n  Testing {len(small_pruned)} fully-pruned parents "
          f"(total_before < 50,000):")
    for idx in small_pruned:
        parent_int = parents[idx]

        res_orig = rc._compute_bin_ranges(parent_int, M, C_TARGET, d_child,
                                           n_half_child)
        if res_orig is None:
            print(f"\n  Parent #{idx}: {parent_int}")
            print(f"    Bin ranges empty (trivially pruned)")
            print(f"    -> PASS")
            continue

        lo_orig, hi_orig, tc_orig = res_orig
        est_time = tc_orig / 7_000_000
        print(f"\n  Parent #{idx}: {parent_int}")
        print(f"    Original total: {tc_orig:,} (est {est_time:.2f}s)")

        if est_time > 10.0:
            print(f"    SKIP (>{10}s estimated)")
            continue

        surv = _run_kernel_on_ranges(
            parent_int, n_half_child, M, C_TARGET,
            lo_orig, hi_orig, tc_orig)
        status = "PASS" if len(surv) == 0 else "FAIL"
        print(f"    Kernel survivors: {len(surv)}")
        print(f"    -> {status}")

    # ==================================================================
    # SECTION 4: Improvement -- Profile-guided ell ordering
    # ==================================================================
    print()
    print("=" * 72)
    print("SECTION 4: Profile-Guided Ell Ordering")
    print("=" * 72)

    times_original = []
    times_fast_ell = []
    mismatches = 0

    t_sec4 = time.perf_counter()
    for idx in range(n_parents):
        parent_int = parents[idx]

        # Original
        res_o = _get_ranges_and_tighten(parent_int, M, C_TARGET, n_half_child)
        # Fast ell
        res_f = _get_ranges_and_tighten(parent_int, M, C_TARGET, n_half_child,
                                         tighten_fn=_tighten_ranges_fast_ell)

        if res_o is None and res_f is None:
            continue
        if res_o is None or res_f is None:
            mismatches += 1
            continue

        times_original.append(res_o[6])
        times_fast_ell.append(res_f[6])

        # Verify identical ranges
        if (not np.array_equal(res_o[3], res_f[3]) or
                not np.array_equal(res_o[4], res_f[4])):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH at parent #{idx}: {parents[idx]}")
                print(f"    original lo={res_o[3]} hi={res_o[4]}")
                print(f"    fast_ell lo={res_f[3]} hi={res_f[4]}")

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx+1:,}/{n_parents:,}...")

    t_sec4_elapsed = time.perf_counter() - t_sec4
    times_o_arr = np.array(times_original)
    times_f_arr = np.array(times_fast_ell)

    print(f"\n  Range mismatches:   {mismatches} / {n_parents}")
    print(f"  Wall time:          {t_sec4_elapsed:.2f} sec")
    print(f"\n  -- Timing comparison (microseconds) --")
    print(f"    {'':20s}  {'Original':>10s}  {'FastEll':>10s}  {'Speedup':>8s}")
    print(f"    {'Mean':20s}  {np.mean(times_o_arr):10.1f}  "
          f"{np.mean(times_f_arr):10.1f}  "
          f"{np.mean(times_o_arr)/np.mean(times_f_arr):8.2f}x")
    print(f"    {'Median':20s}  {np.median(times_o_arr):10.1f}  "
          f"{np.median(times_f_arr):10.1f}  "
          f"{np.median(times_o_arr)/np.median(times_f_arr):8.2f}x")
    print(f"    {'Total':20s}  {np.sum(times_o_arr)/1e6:10.3f}s  "
          f"{np.sum(times_f_arr)/1e6:10.3f}s  "
          f"{np.sum(times_o_arr)/np.sum(times_f_arr):8.2f}x")

    # ==================================================================
    # SECTION 5: Improvement -- Batched cross-term update
    # ==================================================================
    print()
    print("=" * 72)
    print("SECTION 5: Batched Cross-Term Update")
    print("=" * 72)

    times_batched = []
    mismatches_b = 0

    t_sec5 = time.perf_counter()
    for idx in range(n_parents):
        parent_int = parents[idx]

        res_o = _get_ranges_and_tighten(parent_int, M, C_TARGET, n_half_child)
        res_b = _get_ranges_and_tighten(parent_int, M, C_TARGET, n_half_child,
                                         tighten_fn=_tighten_ranges_batched_cross)

        if res_o is None and res_b is None:
            continue
        if res_o is None or res_b is None:
            mismatches_b += 1
            continue

        times_batched.append(res_b[6])

        if (not np.array_equal(res_o[3], res_b[3]) or
                not np.array_equal(res_o[4], res_b[4])):
            mismatches_b += 1
            if mismatches_b <= 3:
                print(f"  MISMATCH at parent #{idx}: {parents[idx]}")
                print(f"    original lo={res_o[3]} hi={res_o[4]}")
                print(f"    batched  lo={res_b[3]} hi={res_b[4]}")

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx+1:,}/{n_parents:,}...")

    t_sec5_elapsed = time.perf_counter() - t_sec5
    times_b_arr = np.array(times_batched)
    # Reuse times_original from section 4
    times_o_arr2 = np.array(times_original[:len(times_batched)])

    print(f"\n  Range mismatches:   {mismatches_b} / {n_parents}")
    print(f"  Wall time:          {t_sec5_elapsed:.2f} sec")
    print(f"\n  -- Timing comparison (microseconds) --")
    print(f"    {'':20s}  {'Original':>10s}  {'Batched':>10s}  {'Speedup':>8s}")
    if len(times_b_arr) > 0 and len(times_o_arr2) > 0:
        print(f"    {'Mean':20s}  {np.mean(times_o_arr2):10.1f}  "
              f"{np.mean(times_b_arr):10.1f}  "
              f"{np.mean(times_o_arr2)/np.mean(times_b_arr):8.2f}x")
        print(f"    {'Median':20s}  {np.median(times_o_arr2):10.1f}  "
              f"{np.median(times_b_arr):10.1f}  "
              f"{np.median(times_o_arr2)/np.median(times_b_arr):8.2f}x")
        print(f"    {'Total':20s}  {np.sum(times_o_arr2)/1e6:10.3f}s  "
              f"{np.sum(times_b_arr)/1e6:10.3f}s  "
              f"{np.sum(times_o_arr2)/np.sum(times_b_arr):8.2f}x")

    # ==================================================================
    # SECTION 6: Real Cascade Through Levels
    # ==================================================================
    print()
    print("=" * 72)
    print("SECTION 6: Real Cascade -- AC Effectiveness at Each Level")
    print("=" * 72)
    print()
    print("  Strategy: start from L0 survivors, run the actual kernel on a")
    print("  random sample to get REAL survivors at each level, then measure")
    print("  arc consistency on those real survivors as parents for the next level.")
    print("  Continue until no survivors remain or time budget exhausted.")

    # We'll carry forward real survivors level by level.
    # At each level: measure AC stats on ALL parents at that level,
    # then run the kernel on a small sample to produce next-level parents.
    current_parents = parents.copy()  # L0 survivors, d=4
    level = 0  # L0 survivors -> L1 children (this is the L0->L1 transition)

    # Time budget per level for kernel runs (seconds)
    KERNEL_TIME_BUDGET = 300.0
    # Max parents to run kernel on per level
    MAX_KERNEL_PARENTS = 30
    # Max children per parent before skipping kernel
    MAX_CHILDREN_PER_PARENT = 50_000_000  # ~7s
    # Max parents for AC stats pass (sample if more)
    MAX_AC_STATS_PARENTS = 10_000

    while len(current_parents) > 0:
        d_par = current_parents.shape[1]
        d_ch = 2 * d_par
        nhc = d_par  # n_half_child = d_parent in the cascade

        print(f"\n  {'='*60}")
        print(f"  LEVEL L{level} -> L{level+1}: "
              f"d_parent={d_par}, d_child={d_ch}, n_half_child={nhc}")
        print(f"  Parents at this level: {len(current_parents):,}")
        print(f"  {'='*60}")

        # -- AC statistics (sample if too many parents) --
        n_total_parents = len(current_parents)
        if n_total_parents > MAX_AC_STATS_PARENTS:
            rng = np.random.RandomState(42)
            stats_indices = rng.choice(n_total_parents, MAX_AC_STATS_PARENTS,
                                        replace=False)
            stats_indices.sort()
            print(f"  Sampling {MAX_AC_STATS_PARENTS:,} of "
                  f"{n_total_parents:,} parents for AC stats")
        else:
            stats_indices = np.arange(n_total_parents)

        ac_ratios = []
        ac_times = []
        ac_fully_pruned = 0
        ac_tightened = 0
        total_before_sum_lv = 0
        total_after_sum_lv = 0
        per_pos_shrink_lv = [[] for _ in range(d_par)]

        # Track parents suitable for kernel runs (scan ALL parents for this)
        kernel_candidates = []  # (idx, total_before, total_after)

        # First pass: AC stats on sample
        for si, pidx in enumerate(stats_indices):
            pidx = int(pidx)
            parent_int = current_parents[pidx]

            result = rc._compute_bin_ranges(parent_int, M, C_TARGET, d_ch, nhc)
            if result is None:
                ac_fully_pruned += 1
                continue

            lo_b, hi_b, tb = result
            lo_a = lo_b.copy()
            hi_a = hi_b.copy()

            t0 = time.perf_counter()
            ta = int(rc._tighten_ranges(parent_int, lo_a, hi_a, M, C_TARGET, nhc))
            elapsed = (time.perf_counter() - t0) * 1e6
            ac_times.append(elapsed)

            total_before_sum_lv += tb
            total_after_sum_lv += ta

            if ta == 0:
                ac_fully_pruned += 1
                ac_ratios.append(float('inf'))
            else:
                ratio = tb / ta
                ac_ratios.append(ratio)
                if ratio > 1.0001:
                    ac_tightened += 1

            # Per-position shrinkage (lo_b/hi_b are the originals)
            for p in range(d_par):
                rb = hi_b[p] - lo_b[p] + 1
                ra = hi_a[p] - lo_a[p] + 1
                per_pos_shrink_lv[p].append(rb - ra)

            # Track all non-pruned parents as kernel candidates
            if ta > 0:
                kernel_candidates.append((pidx, tb, ta))

            if (si + 1) % 2000 == 0:
                print(f"    AC stats: {si+1:,}/{len(stats_indices):,}...")

        # If we sampled, scan a random subset of remaining parents for
        # more kernel candidates (pick up to 50K more to find small ones)
        if n_total_parents > MAX_AC_STATS_PARENTS:
            stats_set = set(int(x) for x in stats_indices)
            extra_scan = min(50_000, n_total_parents - MAX_AC_STATS_PARENTS)
            rng2 = np.random.RandomState(123)
            extra_indices = rng2.choice(n_total_parents, extra_scan,
                                         replace=False)
            print(f"  Scanning {extra_scan:,} more parents for "
                  f"kernel candidates...")
            for pidx in extra_indices:
                pidx = int(pidx)
                if pidx in stats_set:
                    continue
                parent_int = current_parents[pidx]
                result = rc._compute_bin_ranges(parent_int, M, C_TARGET,
                                                 d_ch, nhc)
                if result is None:
                    continue
                lo_arr, hi_arr, tb = result
                ta = int(rc._tighten_ranges(parent_int, lo_arr, hi_arr,
                                              M, C_TARGET, nhc))
                if ta > 0:
                    kernel_candidates.append((pidx, tb, ta))

        # -- Print AC statistics --
        n_stats = len(stats_indices)
        finite_ratios = [r for r in ac_ratios if r != float('inf')]
        finite_arr = np.array(finite_ratios) if finite_ratios else np.array([1.0])
        times_arr_lv = np.array(ac_times) if ac_times else np.array([0.0])

        print(f"\n  AC Statistics (sample={n_stats:,} of {n_total_parents:,}):")
        print(f"    Parents in sample:            {n_stats:,}")
        print(f"    Fully pruned by AC:           {ac_fully_pruned:,} "
              f"({100.0*ac_fully_pruned/max(n_stats,1):.2f}%)")
        print(f"    Parents with range shrinkage: {ac_tightened:,} "
              f"({100.0*ac_tightened/max(n_stats,1):.2f}%)")
        print(f"    Total children BEFORE:        {total_before_sum_lv:,}")
        print(f"    Total children AFTER:         {total_after_sum_lv:,}")
        if total_before_sum_lv > 0:
            pct = 100.0 * (1 - total_after_sum_lv / total_before_sum_lv)
            print(f"    Reduction:                    {pct:.4f}%")
            print(f"    Children eliminated:          "
                  f"{total_before_sum_lv - total_after_sum_lv:,}")
            est_s = (total_before_sum_lv - total_after_sum_lv) / 7_000_000
            print(f"    Est. time saved:              {est_s:,.1f}s ({est_s/3600:.2f}hr)")

        if len(finite_ratios) > 0:
            print(f"\n    Reduction ratio (before/after), non-pruned parents:")
            print(f"      Min:    {np.min(finite_arr):.4f}")
            print(f"      Max:    {np.max(finite_arr):.4f}")
            print(f"      Mean:   {np.mean(finite_arr):.4f}")
            print(f"      Median: {np.median(finite_arr):.4f}")
            if len(finite_arr) >= 10:
                print(f"      p90:    {np.percentile(finite_arr, 90):.4f}")
                print(f"      p99:    {np.percentile(finite_arr, 99):.4f}")

        print(f"\n    _tighten_ranges timing (us):")
        print(f"      Mean:   {np.mean(times_arr_lv):.1f}")
        print(f"      Median: {np.median(times_arr_lv):.1f}")
        print(f"      Max:    {np.max(times_arr_lv):.1f}")

        print(f"\n    Per-position range shrinkage:")
        print(f"      {'Pos':>4s}  {'shrink=0':>10s}  {'shrink=1':>10s}  "
              f"{'shrink=2':>10s}  {'shrink>=3':>10s}")
        for p in range(d_par):
            if len(per_pos_shrink_lv[p]) == 0:
                continue
            arr = np.array(per_pos_shrink_lv[p])
            n0 = int(np.sum(arr == 0))
            n1 = int(np.sum(arr == 1))
            n2 = int(np.sum(arr == 2))
            n3 = int(np.sum(arr >= 3))
            print(f"      {p:4d}  {n0:10,}  {n1:10,}  {n2:10,}  {n3:10,}")

        # -- Run kernel on sample to get next-level survivors --
        if len(kernel_candidates) == 0:
            print(f"\n  No kernel candidates (all fully pruned). Stopping.")
            break

        # Sort by total_after (smallest first) — run smallest parents
        # to maximize count within time budget
        kernel_candidates.sort(key=lambda x: x[2])
        smallest_ta = kernel_candidates[0][2]
        largest_ta = kernel_candidates[-1][2]
        n_to_run = min(MAX_KERNEL_PARENTS, len(kernel_candidates))

        print(f"\n  Kernel candidates: {len(kernel_candidates):,}")
        print(f"    Smallest total_after: {smallest_ta:,} "
              f"(est {smallest_ta/7_000_000:.1f}s)")
        print(f"    Largest total_after:  {largest_ta:,}")
        print(f"  Running kernel on up to {n_to_run} smallest parents "
              f"(time budget {KERNEL_TIME_BUDGET:.0f}s)...")

        all_survivors = []
        kernel_time_total = 0.0
        n_kernel_run = 0

        for cidx, (pidx, tb, ta) in enumerate(kernel_candidates[:n_to_run]):
            if kernel_time_total > KERNEL_TIME_BUDGET:
                print(f"    Time budget exhausted after {n_kernel_run} parents")
                break

            parent_int = current_parents[pidx]
            result = rc._compute_bin_ranges(parent_int, M, C_TARGET, d_ch, nhc)
            if result is None:
                continue
            lo_arr, hi_arr, _ = result
            # Apply tightening
            ta2 = int(rc._tighten_ranges(parent_int, lo_arr, hi_arr,
                                           M, C_TARGET, nhc))
            if ta2 == 0:
                continue

            t0 = time.perf_counter()
            surv = _run_kernel_on_ranges(parent_int, nhc, M, C_TARGET,
                                          lo_arr, hi_arr, ta2)
            elapsed = time.perf_counter() - t0
            kernel_time_total += elapsed
            n_kernel_run += 1

            if len(surv) > 0:
                all_survivors.append(surv)

            if cidx < 5 or (cidx + 1) % 5 == 0:
                print(f"    Parent {cidx+1}/{n_to_run}: "
                      f"{ta2:,} children -> {len(surv)} survivors "
                      f"({elapsed:.1f}s, total {kernel_time_total:.1f}s)")

        if len(all_survivors) > 0:
            next_parents = np.concatenate(all_survivors, axis=0)
            # Deduplicate
            if len(next_parents) > 1:
                keys = tuple(next_parents[:, i]
                             for i in range(next_parents.shape[1] - 1, -1, -1))
                order = np.lexsort(keys)
                next_parents = next_parents[order]
                mask = np.ones(len(next_parents), dtype=bool)
                for i in range(1, len(next_parents)):
                    if np.array_equal(next_parents[i], next_parents[i-1]):
                        mask[i] = False
                next_parents = next_parents[mask]
        else:
            next_parents = np.empty((0, d_ch), dtype=np.int32)

        print(f"\n  Kernel summary:")
        print(f"    Parents run:     {n_kernel_run}")
        print(f"    Kernel time:     {kernel_time_total:.2f}s")
        print(f"    Total survivors: {len(next_parents):,} (d={d_ch})")

        if len(next_parents) == 0:
            print(f"    No survivors -- cascade terminates at L{level+1}.")
            break

        current_parents = next_parents
        level += 1

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
