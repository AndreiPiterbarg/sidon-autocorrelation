"""CPU-only cascade prover — no GPU, no dimension limits.

Runs L0 (composition generation + pruning) then cascades through
refinement levels until all survivors are eliminated or a max
dimension is reached.

Optimizations (integrated from parallel agent work):
  - Fused generate+prune kernel: generates children on-the-fly and prunes
    inline, avoiding 50M+ row intermediate arrays (10-18x speedup on L1-L3)
  - _prune_dynamic with int32/int64 dispatch, pre-computed per-ell constants
  - Numba-parallel canonicalization (replaces Python tuple comparison)
  - Sort-based deduplication (replaces set-of-tuples)
  - JIT warmup at module load

Usage:
    python -m cloninger-steinerberger.cpu.run_cascade
    python -m cloninger-steinerberger.cpu.run_cascade --n_half 2 --m 20 --c_target 1.30
    python -m cloninger-steinerberger.cpu.run_cascade --n_half 3 --m 50 --c_target 1.30 --max_levels 5
"""
import argparse
import json
import math
import multiprocessing as mp
import tempfile
import os
import sys
import time
import itertools

import numpy as np
import numba
from numba import njit, prange

# Path setup — import from parent cloninger-steinerberger/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cs_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _cs_dir)

from compositions import (generate_canonical_compositions_batched,
                         generate_compositions_batched)
from pruning import (correction, asymmetry_threshold, count_compositions,
                     asymmetry_prune_mask, _canonical_mask,
                     block_mass_prune_mask)
from test_values import compute_test_values_batch


# =====================================================================
# Dynamic per-window threshold — int32 path (m <= 200)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_dynamic_int32(batch_int, n_half, m, c_target,
                          use_flat_threshold=False, use_F=False):
    """int32 path: halves memory bandwidth in autoconvolution inner loop.

    Safe when m <= 200 because max prefix sum of conv = m^2 = 40000,
    which fits comfortably in int32 (max 2,147,483,647).
    Values are widened to int64 only at the threshold comparison point.

    Three correction modes (priority: flat > F > W-refined):
      use_flat_threshold=True: flat C&S Lemma 3 (2/m + 1/m^2).  Required
        for the Lean axiom cascade_all_pruned.
      use_F=True: variant F — LP-tight linear bound via sort+extremes
        plus the standard δ² bound.  Empirically 25-65% additional
        pruning over W-refined; sound under |a-b|_∞ ≤ 1/m, Σa = 4n,
        a ≥ 0.  See _M1_bench.py:prune_F.
      else: W-refined (1 + W_int/(2n)) — the prior default.
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    d_minus_1 = d - 1
    eps_margin = 1e-9 * m_d * m_d

    # Pre-compute per-ell scale factors.
    # Fine grid: heights = c_i/m, TV = window_sum/(4n*ell*m^2).
    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    # Flat C&S Lemma 3 correction: (2/m + 1/m^2)*m^2 = 2m + 1.
    flat_corr = 2.0 * m_d + 1.0
    flat_threshold_arr = np.empty(max_ell + 1, dtype=np.int64)
    if use_flat_threshold:
        for ell in range(2, max_ell + 1):
            dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_arr[ell]
            flat_threshold_arr[ell] = np.int64(dyn_x)

    # Variant F: precompute ell_prefix where ell_int_arr[k] = #{(i,j): i+j=k}.
    # ell_int_arr[k] = max(0, 2n - |k+1 - 2n|), in [0, 2n].
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    if use_F and not use_flat_threshold:
        two_n = 2 * n_half
        for k in range(conv_len):
            d_idx = (k + 1) - two_n
            if d_idx < 0:
                d_idx = -d_idx
            v = two_n - d_idx
            if v < 0:
                v = 0
            ell_prefix[k + 1] = ell_prefix[k] + v

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        if not use_flat_threshold:
            prefix_c = np.zeros(d + 1, dtype=np.int64)
            for i in range(d):
                prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            # Sliding window: initialize sum for s_lo=0
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])

            if use_flat_threshold:
                dyn_it = flat_threshold_arr[ell]
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                    if ws > dyn_it:
                        pruned = True
                        break
            else:
                scale_ell = scale_arr[ell]
                ell_f = np.float64(ell)
                half = d // 2
                BB = np.empty(d, dtype=np.int64)
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                    if use_F:
                        # F: corr = Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)
                        # BB_j = Σ_{i: i+j ∈ window, i ∈ [0,d-1]} c_i
                        for j in range(d):
                            lo_i = s_lo - j
                            if lo_i < 0:
                                lo_i = 0
                            hi_i = s_lo + ell - 2 - j
                            if hi_i > d_minus_1:
                                hi_i = d_minus_1
                            if lo_i > hi_i:
                                BB[j] = np.int64(0)
                            else:
                                BB[j] = prefix_c[hi_i + 1] - prefix_c[lo_i]
                        BB_sorted = np.sort(BB)
                        Delta_BB = np.int64(0)
                        for jj in range(half):
                            Delta_BB += BB_sorted[d - 1 - jj] - BB_sorted[jj]
                        ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                        corr_w = (np.float64(Delta_BB) / (2.0 * n_half_d * ell_f)
                                  + np.float64(ell_int_sum)
                                  / (4.0 * n_half_d * ell_f))
                    else:
                        # W-refined (legacy default)
                        lo_bin = s_lo - d_minus_1
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_minus_1:
                            hi_bin = d_minus_1
                        W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                        corr_w = 1.0 + np.float64(W_int) / (2.0 * n_half_d)
                    dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                    dyn_it = np.int64(dyn_x)
                    if ws > dyn_it:
                        pruned = True
                        break

        if pruned:
            survived[b] = False

    return survived


# =====================================================================
# Dynamic per-window threshold — int64 path (m > 200)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_dynamic_int64(batch_int, n_half, m, c_target,
                          use_flat_threshold=False, use_F=False):
    """int64 path for large m values where int32 conv may overflow.

    See _prune_dynamic_int32 for use_flat_threshold and use_F docs.
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    d_minus_1 = d - 1
    eps_margin = 1e-9 * m_d * m_d

    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    # Flat C&S Lemma 3 correction: (2/m + 1/m^2)*m^2 = 2m + 1.
    flat_corr = 2.0 * m_d + 1.0
    flat_threshold_arr = np.empty(max_ell + 1, dtype=np.int64)
    if use_flat_threshold:
        for ell in range(2, max_ell + 1):
            dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_arr[ell]
            flat_threshold_arr[ell] = np.int64(dyn_x)

    # Variant F: ell_int_arr prefix sum.
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    if use_F and not use_flat_threshold:
        two_n = 2 * n_half
        for k in range(conv_len):
            d_idx = (k + 1) - two_n
            if d_idx < 0:
                d_idx = -d_idx
            v = two_n - d_idx
            if v < 0:
                v = 0
            ell_prefix[k + 1] = ell_prefix[k] + v

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        if not use_flat_threshold:
            prefix_c = np.zeros(d + 1, dtype=np.int64)
            for i in range(d):
                prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])

            if use_flat_threshold:
                dyn_it = flat_threshold_arr[ell]
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                    if ws > dyn_it:
                        pruned = True
                        break
            else:
                scale_ell = scale_arr[ell]
                ell_f = np.float64(ell)
                half = d // 2
                BB = np.empty(d, dtype=np.int64)
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                    if use_F:
                        for j in range(d):
                            lo_i = s_lo - j
                            if lo_i < 0:
                                lo_i = 0
                            hi_i = s_lo + ell - 2 - j
                            if hi_i > d_minus_1:
                                hi_i = d_minus_1
                            if lo_i > hi_i:
                                BB[j] = np.int64(0)
                            else:
                                BB[j] = prefix_c[hi_i + 1] - prefix_c[lo_i]
                        BB_sorted = np.sort(BB)
                        Delta_BB = np.int64(0)
                        for jj in range(half):
                            Delta_BB += BB_sorted[d - 1 - jj] - BB_sorted[jj]
                        ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                        corr_w = (np.float64(Delta_BB) / (2.0 * n_half_d * ell_f)
                                  + np.float64(ell_int_sum)
                                  / (4.0 * n_half_d * ell_f))
                    else:
                        lo_bin = s_lo - d_minus_1
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_minus_1:
                            hi_bin = d_minus_1
                        W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                        corr_w = 1.0 + np.float64(W_int) / (2.0 * n_half_d)
                    dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                    dyn_it = np.int64(dyn_x)
                    if ws > dyn_it:
                        pruned = True
                        break

        if pruned:
            survived[b] = False

    return survived


def _prune_dynamic(batch_int, n_half, m, c_target, use_flat_threshold=False,
                    use_F=False):
    """Per-window dynamic threshold — dispatches int32/int64 based on m.

    Works in integer convolution space (fine grid: c_i sum to S = 4nm).

    When use_flat_threshold=False (default): uses the W-refined correction
        dyn_it = floor((c_target*m^2 + 1 + W_int/(2n) + eps) * 4n*ell)
    which is tighter (prunes more) but does NOT verify the Lean axiom.

    When use_flat_threshold=True: uses the flat C&S Lemma 3 correction
        dyn_it = floor((c_target*m^2 + 2m + 1 + eps) * 4n*ell)
    which matches the Lean axiom cascade_all_pruned threshold
    (c_target + 2/m + 1/m^2).  Required for formal verification.

    Returns boolean mask: True = survived (not pruned).
    """
    if m <= 200:
        return _prune_dynamic_int32(batch_int, n_half, m, c_target,
                                     use_flat_threshold, use_F)
    else:
        return _prune_dynamic_int64(batch_int, n_half, m, c_target,
                                     use_flat_threshold, use_F)


# =====================================================================
# Coarse-grid helpers: pair-count prefix sums for second-order box cert
# =====================================================================

@njit(cache=True)
def _build_pair_prefix(d):
    """Build prefix sums for n_k (pair counts) and m_k (self-term indicators).

    n_k = #{(i,j): 0<=i,j<d, i+j=k} = min(k+1, d, 2d-1-k)
    m_k = 1 if k even and k//2 < d, else 0

    Returns (prefix_nk, prefix_mk) each of length conv_len+1 = 2d.
    """
    conv_len = 2 * d - 1
    prefix_nk = np.zeros(conv_len + 1, dtype=np.int64)
    prefix_mk = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        nk = min(k + 1, d, conv_len - k)
        mk = np.int64(1) if (k % 2 == 0 and k // 2 < d) else np.int64(0)
        prefix_nk[k + 1] = prefix_nk[k] + np.int64(nk)
        prefix_mk[k + 1] = prefix_mk[k] + mk
    return prefix_nk, prefix_mk


# =====================================================================
# Coarse-grid L0 batch pruning (Theorem 1 + sound box cert)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_coarse(batch_int, d, S, c_target):
    """Prune coarse-grid compositions by Theorem 1 (no correction).

    Prune if exists window (ell, s) with TV_W >= c_target.
    Integer: ws > floor(c_target * ell * S^2 / (2*d) - eps).

    Returns (survived_mask, min_cert_net).
    min_cert_net: smallest (margin - cell_var - quad_corr).
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d
                                - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    n_threads = numba.config.NUMBA_NUM_THREADS
    min_net_arr = np.full(n_threads, 1e30, dtype=np.float64)

    for b in prange(B):
        tid = numba.get_thread_id()

        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        pruned = False
        best_net = np.float64(-1e30)
        grad_arr = np.empty(d, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(
                        conv[s_lo - 1])
                if ws > dyn_it:
                    pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g
                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key
                    cell_var = 0.0
                    for k in range(d // 2):
                        cell_var += grad_arr[d - 1 - k] - grad_arr[k]
                    cell_var /= (2.0 * S_d)

                    hi_idx = s_lo + ell - 1
                    N_W = prefix_nk[hi_idx] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi_idx] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    compl_bound = d_sq - N_W
                    pb = min(cross_W, compl_bound)
                    qc = (2.0 * d_d / ell_f) * np.float64(
                        max(pb, np.int64(0))) * inv_4S2

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net

        if pruned:
            survived[b] = False
            if best_net < min_net_arr[tid]:
                min_net_arr[tid] = best_net

    global_min_net = 1e30
    for t in range(n_threads):
        if min_net_arr[t] < global_min_net:
            global_min_net = min_net_arr[t]

    return survived, global_min_net


# =====================================================================
# L0 Branch-and-Bound kernel (parallelized over first bin)
# =====================================================================
#
# Soundness of partial pruning:
#   - All c_i >= 0, so all conv cross-terms are non-negative.
#     Therefore partial_ws (from assigned bins) <= final_ws.
#   - The flat threshold is W-independent (doesn't grow as more bins
#     are assigned), so partial_ws > flat_thr implies final_ws > flat_thr.
#   - At leaves, uses the tighter W-refined threshold for maximum pruning.

@njit(cache=True)
def _l0_bnb_inner(c0, d, S, n_half_d, cs_base_m2, eps_margin,
                   flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                   max_ell, four_n, use_flat_threshold,
                   out_buf, count_only):
    """B&B subtree with bins[0]=c0 fixed.

    When count_only=True, counts survivors without writing to buffer.
    Returns (n_survivors, n_leaves_tested).
    """
    conv_len = 2 * d - 1
    d_m1 = d - 1
    conv = np.zeros(conv_len, dtype=np.int32)
    bins = np.zeros(d, dtype=np.int32)
    rem_arr = np.zeros(d, dtype=np.int32)

    bins[0] = c0
    if c0 > 0:
        conv[0] = c0 * c0
    rem_arr[0] = S
    rem_arr[1] = S - c0

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    buf_cap = np.int64(0)
    if not count_only:
        buf_cap = np.int64(out_buf.shape[0])

    # d=1: single bin, only c0=S is valid
    if d == 1:
        if c0 == S:
            n_tested = 1
            # Leaf check (trivially c0^2 vs threshold)
            # handled by caller
        return n_surv, n_tested

    # d=2: bins = (c0, S-c0)
    if d == 2:
        forced = S - c0
        if 0 <= forced <= x_cap:
            n_tested = 1
            bins[1] = forced
            conv[0] = c0 * c0
            conv[1] = np.int32(2) * c0 * np.int32(forced)
            conv[2] = forced * forced

            # Asymmetry
            left_s = np.float64(bins[0])
            left_frac = left_s / np.float64(S)
            asym_cov = (left_frac <= 1.0 - asym_thr) or (left_frac >= asym_thr)

            if not asym_cov:
                pruned_leaf = False
                if use_flat_threshold:
                    for ell in range(2, max_ell + 1):
                        if pruned_leaf:
                            break
                        n_cv = ell - 1
                        nw = conv_len - n_cv + 1
                        ws = np.int64(0)
                        for k in range(n_cv):
                            ws += np.int64(conv[k])
                        for s_lo in range(nw):
                            if s_lo > 0:
                                ws += (np.int64(conv[s_lo + n_cv - 1])
                                       - np.int64(conv[s_lo - 1]))
                            if ws > flat_thr[ell]:
                                pruned_leaf = True
                                break
                else:
                    prefix_c = np.zeros(d + 1, dtype=np.int64)
                    for i in range(d):
                        prefix_c[i + 1] = prefix_c[i] + np.int64(bins[i])
                    for ell in range(2, max_ell + 1):
                        if pruned_leaf:
                            break
                        n_cv = ell - 1
                        nw = conv_len - n_cv + 1
                        ws = np.int64(0)
                        for k in range(n_cv):
                            ws += np.int64(conv[k])
                        sc = scale_arr[ell]
                        for s_lo in range(nw):
                            if s_lo > 0:
                                ws += (np.int64(conv[s_lo + n_cv - 1])
                                       - np.int64(conv[s_lo - 1]))
                            lo_b = s_lo - d_m1
                            if lo_b < 0:
                                lo_b = 0
                            hi_b = s_lo + ell - 2
                            if hi_b > d_m1:
                                hi_b = d_m1
                            W_int = prefix_c[hi_b + 1] - prefix_c[lo_b]
                            cw = 1.0 + np.float64(W_int) / (2.0 * n_half_d)
                            dyn_it = np.int64(
                                (cs_base_m2 + cw + eps_margin) * sc)
                            if ws > dyn_it:
                                pruned_leaf = True
                                break
                if not pruned_leaf:
                    if not count_only and n_surv < buf_cap:
                        out_buf[n_surv, 0] = c0
                        out_buf[n_surv, 1] = forced
                    n_surv += 1
        return n_surv, n_tested

    # General case: d >= 3, process positions 1..d-1
    pos = 1
    bins[1] = 0

    while True:
        c_val = bins[pos]
        rem = rem_arr[pos]

        if pos == d_m1:
            # --- Last bin: forced value ---
            forced = rem
            if 0 <= forced <= x_cap:
                n_tested += 1
                bins[pos] = forced

                conv[2 * pos] += forced * forced
                for j in range(pos):
                    conv[pos + j] += np.int32(2) * forced * bins[j]

                # Asymmetry check
                left_s = np.float64(0)
                for i in range(left_bins):
                    left_s += np.float64(bins[i])
                left_frac = left_s / np.float64(S)
                asym_cov = ((left_frac <= 1.0 - asym_thr)
                            or (left_frac >= asym_thr))

                if not asym_cov:
                    pruned_leaf = False
                    if use_flat_threshold:
                        for ell in range(2, max_ell + 1):
                            if pruned_leaf:
                                break
                            n_cv = ell - 1
                            nw = conv_len - n_cv + 1
                            ws = np.int64(0)
                            for k in range(n_cv):
                                ws += np.int64(conv[k])
                            thr_val = flat_thr[ell]
                            for s_lo in range(nw):
                                if s_lo > 0:
                                    ws += (np.int64(conv[s_lo + n_cv - 1])
                                           - np.int64(conv[s_lo - 1]))
                                if ws > thr_val:
                                    pruned_leaf = True
                                    break
                    else:
                        prefix_c = np.zeros(d + 1, dtype=np.int64)
                        for i in range(d):
                            prefix_c[i + 1] = (prefix_c[i]
                                               + np.int64(bins[i]))
                        for ell in range(2, max_ell + 1):
                            if pruned_leaf:
                                break
                            n_cv = ell - 1
                            nw = conv_len - n_cv + 1
                            ws = np.int64(0)
                            for k in range(n_cv):
                                ws += np.int64(conv[k])
                            sc = scale_arr[ell]
                            for s_lo in range(nw):
                                if s_lo > 0:
                                    ws += (
                                        np.int64(conv[s_lo + n_cv - 1])
                                        - np.int64(conv[s_lo - 1]))
                                lo_b = s_lo - d_m1
                                if lo_b < 0:
                                    lo_b = 0
                                hi_b = s_lo + ell - 2
                                if hi_b > d_m1:
                                    hi_b = d_m1
                                W_int = (prefix_c[hi_b + 1]
                                         - prefix_c[lo_b])
                                cw = (1.0 + np.float64(W_int)
                                      / (2.0 * n_half_d))
                                dyn_it = np.int64(
                                    (cs_base_m2 + cw + eps_margin) * sc)
                                if ws > dyn_it:
                                    pruned_leaf = True
                                    break

                    if not pruned_leaf:
                        if not count_only and n_surv < buf_cap:
                            for i in range(d):
                                out_buf[n_surv, i] = bins[i]
                        n_surv += 1

                conv[2 * pos] -= forced * forced
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * forced * bins[j]

            # Backtrack from last position
            pos -= 1
            if pos < 1:
                break
            c_old = bins[pos]
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * c_old * bins[j]
            bins[pos] += 1
            continue

        # --- Non-last bin ---
        max_v = min(rem, x_cap)
        min_v = rem - (d_m1 - pos) * x_cap
        if min_v < 0:
            min_v = 0
        if c_val < min_v:
            bins[pos] = min_v
            c_val = min_v

        if c_val > max_v:
            if pos <= 1:
                break
            pos -= 1
            c_old = bins[pos]
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * c_old * bins[j]
            bins[pos] += 1
            continue

        # Add conv contribution
        if c_val > 0:
            conv[2 * pos] += c_val * c_val
            for j in range(pos):
                conv[pos + j] += np.int32(2) * c_val * bins[j]

        # Partial prune: restricted to windows overlapping [0, 2*pos]
        pruned_partial = False
        max_ci = 2 * pos
        for ell in range(2, max_ell + 1):
            if pruned_partial:
                break
            n_cv = ell - 1
            max_s = min(max_ci, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = n_cv
            if init_end > max_ci + 1:
                init_end = max_ci + 1
            for k in range(init_end):
                ws += np.int64(conv[k])
            thr_val = flat_thr[ell]
            if ws > thr_val:
                pruned_partial = True
                break
            for s_lo in range(1, max_s + 1):
                new_k = s_lo + n_cv - 1
                if new_k <= max_ci:
                    ws += np.int64(conv[new_k])
                ws -= np.int64(conv[s_lo - 1])
                if ws > thr_val:
                    pruned_partial = True
                    break

        if pruned_partial:
            if c_val > 0:
                conv[2 * pos] -= c_val * c_val
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * c_val * bins[j]
            bins[pos] += 1
            continue

        # Descend
        rem_arr[pos + 1] = rem - c_val
        pos += 1
        bins[pos] = 0

    return n_surv, n_tested


@njit(parallel=True, cache=True)
def _l0_bnb_count(d, S, n_half_d, cs_base_m2, eps_margin,
                   flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                   max_ell, four_n, use_flat_threshold,
                   min_c0, n_c0, counts, tested_arr):
    """Pass 1: count survivors per c0 in parallel."""
    dummy = np.empty((0, d), dtype=np.int32)
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        ns, nt = _l0_bnb_inner(
            c0, d, S, n_half_d, cs_base_m2, eps_margin,
            flat_thr, scale_arr, x_cap, asym_thr, left_bins,
            max_ell, four_n, use_flat_threshold, dummy, True)
        counts[idx] = ns
        tested_arr[idx] = nt


@njit(parallel=True, cache=True)
def _l0_bnb_fill(d, S, n_half_d, cs_base_m2, eps_margin,
                  flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                  max_ell, four_n, use_flat_threshold,
                  min_c0, n_c0, counts, offsets, out_buf):
    """Pass 2: fill output buffer per c0 in parallel."""
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        cnt = counts[idx]
        if cnt == 0:
            continue
        off = offsets[idx]
        _l0_bnb_inner(
            c0, d, S, n_half_d, cs_base_m2, eps_margin,
            flat_thr, scale_arr, x_cap, asym_thr, left_bins,
            max_ell, four_n, use_flat_threshold,
            out_buf[off:off + cnt], False)


def _l0_bnb_run(d, S, n_half, m, c_target, use_flat_threshold):
    """Run parallel branch-and-bound L0.

    Returns (survivors_array, n_survivors, n_tested).
    """
    m_d = np.float64(m)
    n_half_d = np.float64(n_half)
    four_n = 4.0 * n_half_d
    cs_base_m2 = c_target * m_d * m_d
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    d_m1 = d - 1

    flat_corr = 2.0 * m_d + 1.0
    flat_thr = np.empty(max_ell + 1, dtype=np.int64)
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n
        flat_thr[ell] = np.int64(
            (cs_base_m2 + flat_corr + eps_margin) * scale_arr[ell])

    thresh_xcap = c_target + 2.0 / m_d + 1.0 / (m_d * m_d) + 1e-9
    x_cap = np.int32(np.floor(
        m_d * np.sqrt(4.0 * np.float64(d) * thresh_xcap)))
    asym_thr = np.sqrt(c_target / 2.0)
    left_bins = d // 2

    min_c0 = max(0, S - d_m1 * int(x_cap))
    max_c0 = min(S, int(x_cap))

    # Quick prune: c0^2 > flat_thr[2] means entire subtree pruned
    max_c0_flat = int(np.floor(np.sqrt(float(flat_thr[2]))))
    if max_c0_flat < max_c0:
        max_c0 = max_c0_flat

    n_c0 = max_c0 - min_c0 + 1
    if n_c0 <= 0:
        return np.empty((0, d), dtype=np.int32), 0, 0

    counts = np.zeros(n_c0, dtype=np.int64)
    tested_arr = np.zeros(n_c0, dtype=np.int64)

    # Pass 1: count
    _l0_bnb_count(d, S, n_half_d, cs_base_m2, eps_margin,
                   flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                   max_ell, four_n, use_flat_threshold,
                   min_c0, n_c0, counts, tested_arr)

    # Compute offsets (prefix sum)
    offsets = np.zeros(n_c0 + 1, dtype=np.int64)
    for i in range(n_c0):
        offsets[i + 1] = offsets[i] + counts[i]
    total_surv = int(offsets[n_c0])
    total_tested = int(np.sum(tested_arr))

    if total_surv == 0:
        return np.empty((0, d), dtype=np.int32), 0, total_tested

    out_buf = np.empty((total_surv, d), dtype=np.int32)

    # Pass 2: fill
    _l0_bnb_fill(d, S, n_half_d, cs_base_m2, eps_margin,
                  flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                  max_ell, four_n, use_flat_threshold,
                  min_c0, n_c0, counts, offsets, out_buf)

    return out_buf, total_surv, total_tested


# =====================================================================
# Numba-parallel canonicalization
# =====================================================================

@njit(parallel=True, cache=True)
def _canonicalize_inplace(arr):
    """Replace each row with min(row, rev(row)) lexicographically, in-place.

    Much faster than Python-level tuple comparisons: uses Numba prange
    over survivors and an early-exit lexicographic comparison.
    """
    B = arr.shape[0]
    d = arr.shape[1]
    half = d // 2
    for b in prange(B):
        swap = False
        for i in range(half):
            j = d - 1 - i
            if arr[b, j] < arr[b, i]:
                swap = True
                break
            elif arr[b, j] > arr[b, i]:
                break
        if swap:
            for i in range(half):
                j = d - 1 - i
                tmp = arr[b, i]
                arr[b, i] = arr[b, j]
                arr[b, j] = tmp


# =====================================================================
# Sort-based deduplication (Numba)
# =====================================================================

@njit(cache=True)
def _dedup_sorted(arr, sort_idx):
    """Given a sorted array (via sort_idx), return indices of unique rows."""
    n = len(sort_idx)
    d = arr.shape[1]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    keep = np.empty(n, dtype=np.int64)
    keep[0] = sort_idx[0]
    count = 1
    for i in range(1, n):
        curr = sort_idx[i]
        prev = sort_idx[i - 1]
        is_same = True
        for j in range(d):
            if arr[curr, j] != arr[prev, j]:
                is_same = False
                break
        if not is_same:
            keep[count] = curr
            count += 1
    return keep[:count]


def _fast_dedup(arr):
    """Deduplicate rows using lexsort + Numba scan.

    Much faster than set-of-tuples for large arrays because it avoids
    creating Python tuple objects for each row.
    """
    if len(arr) == 0:
        return arr
    d = arr.shape[1]
    keys = tuple(arr[:, d - 1 - i] for i in range(d))
    sort_idx = np.lexsort(keys).astype(np.int64)
    unique_idx = _dedup_sorted(arr, sort_idx)
    return arr[unique_idx]


# =====================================================================
# Sorted merge for large shards (avoids 3x RAM of load+lexsort+dedup)
# =====================================================================

@njit(cache=True)
def _sorted_merge_dedup_kernel(a, b, out):
    """Two-pointer merge of two sorted, deduped 2D int32 arrays.

    Both inputs must be in lexicographic order with no duplicate rows
    (as produced by _fast_dedup).  Output is sorted and deduplicated
    (cross-shard duplicates removed).

    Uses memory-mapped inputs so peak RAM is only the output buffer,
    not 3x total like load+vstack+lexsort.

    Returns number of rows written to out.
    """
    na = a.shape[0]
    nb = b.shape[0]
    d = a.shape[1]
    i = 0
    j = 0
    k = 0

    while i < na and j < nb:
        # Lexicographic compare a[i] vs b[j]
        cmp = 0
        for c in range(d):
            if a[i, c] < b[j, c]:
                cmp = -1
                break
            elif a[i, c] > b[j, c]:
                cmp = 1
                break

        if cmp < 0:
            for c in range(d):
                out[k, c] = a[i, c]
            k += 1
            i += 1
        elif cmp > 0:
            for c in range(d):
                out[k, c] = b[j, c]
            k += 1
            j += 1
        else:
            # Equal row — take one copy, advance both
            for c in range(d):
                out[k, c] = a[i, c]
            k += 1
            i += 1
            j += 1

    while i < na:
        for c in range(d):
            out[k, c] = a[i, c]
        k += 1
        i += 1

    while j < nb:
        for c in range(d):
            out[k, c] = b[j, c]
        k += 1
        j += 1

    return k


def _merge_dedup_shards(shard_paths, d, verbose=False):
    """Merge and deduplicate disk shards using pairwise reduction.

    Uses a tournament-style merge: pairs of shards are merged and
    deduped, results written back to disk.  Peak memory is ~3x the
    size of the two shards being merged.

    Returns (array_or_None, remaining_shard_paths).
    - If everything merges into one shard that fits in RAM:
      returns (array, [])
    - If shards are too large to merge in RAM:
      returns (None, [list of shard file paths])
    """
    if not shard_paths:
        return np.empty((0, d), dtype=np.int32), []

    if len(shard_paths) == 1:
        arr = np.load(shard_paths[0])
        try:
            os.remove(shard_paths[0])
        except OSError:
            pass
        return arr, []

    current = list(shard_paths)
    merge_round = 0

    while len(current) > 1:
        merge_round += 1
        next_round = []
        hit_mem_limit = False
        if verbose:
            _log(f"       Merge round {merge_round}: {len(current)} shards")

        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                if hit_mem_limit:
                    # Can't merge any more — carry remaining shards forward
                    next_round.append(current[i])
                    next_round.append(current[i + 1])
                    continue

                # Check RAM before attempting merge
                a_size = os.path.getsize(current[i])
                b_size = os.path.getsize(current[i + 1])
                need_bytes = (a_size + b_size) * 3
                try:
                    import psutil
                    avail = psutil.virtual_memory().available
                except ImportError:
                    avail = int(50e9)
                if need_bytes > avail * 0.80:
                    # 3x RAM too expensive — try sorted merge (1x RAM).
                    # Shards are already lexicographically sorted by
                    # _fast_dedup, so a two-pointer merge works.
                    out_max_bytes = a_size + b_size
                    if out_max_bytes <= avail * 0.85:
                        a_mm = np.load(current[i], mmap_mode='r')
                        b_mm = np.load(current[i + 1], mmap_mode='r')
                        out = np.empty((len(a_mm) + len(b_mm), d),
                                       dtype=np.int32)
                        n_out = _sorted_merge_dedup_kernel(a_mm, b_mm, out)
                        del a_mm, b_mm
                        out_path = current[i] + f'.m{merge_round}.npy'
                        np.save(out_path, out[:n_out])
                        if verbose:
                            _log(f"         Sorted merge ({i},{i+1}): "
                                 f"{n_out:,} unique rows "
                                 f"({n_out * d * 4 / 1e9:.2f} GB)")
                        del out
                        next_round.append(out_path)
                        for p in [current[i], current[i + 1]]:
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                        continue

                    if verbose:
                        _log(f"       Memory limit: merge needs "
                             f"{need_bytes/1e9:.1f} GB, "
                             f"available {avail/1e9:.1f} GB")
                    hit_mem_limit = True
                    next_round.append(current[i])
                    next_round.append(current[i + 1])
                    continue

                a = np.load(current[i])
                b = np.load(current[i + 1])
                combined = np.vstack([a, b])
                del a, b
                merged = _fast_dedup(combined)
                del combined
                out_path = current[i] + f'.m{merge_round}.npy'
                np.save(out_path, merged)
                if verbose:
                    _log(f"         Pair ({i},{i+1}): "
                         f"{len(merged):,} unique rows "
                         f"({merged.nbytes/1e9:.2f} GB)")
                del merged
                next_round.append(out_path)
                for p in [current[i], current[i + 1]]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            else:
                next_round.append(current[i])

        # If no progress was made (all pairs hit mem limit), stop
        if len(next_round) >= len(current):
            if verbose:
                _log(f"       Cannot reduce further — "
                     f"{len(current)} shards remain on disk")
            break
        current = next_round

    if len(current) == 1:
        result = np.load(current[0])
        try:
            os.remove(current[0])
        except OSError:
            pass
        return result, []
    else:
        # Multiple shards remain — too large for RAM
        # Count total rows across shards
        total = 0
        for p in current:
            # Quick row count from file size: file has 128-byte npy header
            # then rows * d * 4 bytes
            sz = os.path.getsize(p) - 128
            total += sz // (d * 4)
        if verbose:
            _log(f"       {len(current)} unmerged shards, ~{total:,} rows total")
        return None, current


# =====================================================================
# Fused generate + prune kernel (highest-impact optimization)
# =====================================================================

@njit(cache=True)
def _fused_generate_and_prune(parent_int, n_half_child, m, c_target,
                               lo_arr, hi_arr, out_buf):
    """Generate children of one parent and immediately prune each one.

    Replaces the pipeline of generate_children_uniform() + test_children()
    by never materializing the full children array.  Each child is built
    on-the-fly via a stack-based Cartesian-product iterator, subjected to
    asymmetry + autoconvolution pruning, and only stored if it survives.

    Optimization: maintains the autoconvolution incrementally.  Consecutive
    children in the odometer differ in only 2 bins (~67% of steps), so we
    update raw_conv in O(d) instead of recomputing in O(d^2).

    Parameters
    ----------
    parent_int : (d_parent,) int32 array
    n_half_child : int  (= 2 * n_half_parent = d_parent in the cascade)
    m : int
    c_target : float
    lo_arr : (d_parent,) int32 — per-bin cursor lower bounds
    hi_arr : (d_parent,) int32 — per-bin cursor upper bounds
    out_buf : (max_survivors, d_child) int32 array (pre-allocated)

    Returns
    -------
    (n_survivors, n_subtree_pruned) : (int, int)
        n_survivors: number of rows written to out_buf
        n_subtree_pruned: number of subtrees skipped by partial-autoconv check
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent

    # --- Safety check: int32 conv values require m <= 200 ---
    # Max raw_conv entry: m^2 = 40000 for m=200; max mutual cross-term 2*m*m = 80000.
    # Incremental deltas bounded by ±2*m^2.  All well within int32 range (2^31-1).
    assert m <= 200, f"int32 conv requires m <= 200, got m={m}"

    # --- Asymmetry filter constants ---
    # No discretization margin needed: left_frac is exact for step functions
    # and preserved under refinement. See docs/verification_part1_framework.md §8.
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    # --- Hoisted asymmetry check (constant across all children) ---
    # sum(child[0:n_half_child]) = sum(parent_int[0:d_parent//2])
    # because child[2k]+child[2k+1] = 2*parent_int[k] and n_half_child = d_parent
    S_parent = np.int64(0)
    for i in range(d_parent):
        S_parent += np.int64(parent_int[i])
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / np.float64(S_parent)
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0

    # --- Dynamic pruning constants (C&S Lemma 3 + W-refinement) ---
    # Fine grid (S = 4nm): compositions sum to S, conv values are int32.
    # threshold = floor((c_target*m^2 + 1 + W_int/(2n) + eps) * 4n*ell)
    # where W_int ranges 0..S (fine-grid mass in overlapping bins).
    four_n = 4.0 * np.float64(n_half_child)
    n_half_d = np.float64(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1
    carry_threshold = d_parent // 4

    # --- Prefix sum of parent bins (for W_int_max in subtree pruning) ---
    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])

    # --- Allocate arrays ---
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    child = np.empty(d_child, dtype=np.int32)
    prev_child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    n_subtree_pruned = 0

    # Quick-check state: track the (ell, s_lo) that killed the previous child.
    # When qc_ell > 0, we try that same window first on the next child,
    # computing the window sum directly from raw_conv (O(ell) instead of O(conv_len)).
    qc_ell = np.int32(0)       # 0 = not yet tracking (first child)
    qc_s = np.int32(0)         # s_lo of tracked window
    qc_W_int = np.int64(0)     # W_int for tracked window on current child

    # --- Build initial child ---
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = 2 * parent_int[i] - cursor[i]

    # --- Precompute per-ell scale factors (C&S Lemma 3 + W-refinement) ---
    # Fine grid (S = 4nm): threshold = floor((c_target*m^2 + 1 + W_int/(2n) + eps) * 4n*ell)
    # where W_int ranges 0..S (fine-grid mass in overlapping bins).
    ell_count = 2 * d_child - 1  # ell ranges 2..2*d_child, count = 2*d_child - 1
    scale_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        scale_arr[idx] = np.float64(ell) * four_n

    # --- Optimized ell scan order ---
    # Most children are pruned by narrow windows (ell=2..16) or wide windows
    # (ell near d_child). Scanning these first reduces the average number of
    # ell values checked before pruning.
    # Phase 1: ell=2..min(16, 2*d_child)  (narrow windows catch peaked configs)
    # Phase 2: ell=d_child, d_child+1, d_child-1, ...  (wide windows catch spread)
    # Phase 3: remaining values
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)  # boolean flags
    oi = 0
    # Phase 1: narrow (ell=2..16)
    phase1_end = min(16, 2 * d_child)
    for ell in range(2, phase1_end + 1):
        ell_order[oi] = np.int32(ell)
        ell_used[ell - 2] = np.int32(1)
        oi += 1
    # Phase 2: wide windows around d_child
    for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                d_child * 2, d_child + d_child // 2, d_child // 2):
        if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = np.int32(1)
            oi += 1
    # Phase 3: everything else in order
    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1

    # --- Compute full raw_conv for initial child ---
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj

    while True:
        # --- Quick check: re-try previous killing window on raw_conv ---
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell - 2
            dyn_it_qc = threshold_table[ell_idx_qc * S_child_plus_1 + qc_W_int]
            if ws_qc > dyn_it_qc:
                quick_killed = True

        if not quick_killed:
            # --- Compute prefix_c ---
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

            # --- Window scan (dynamic pruning, optimized ell order) ---
            pruned = False
            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                ell_idx = ell - 2
                n_windows = conv_len - n_cv + 1
                # Sliding window: initialize sum for s_lo=0
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(raw_conv[s_lo + n_cv - 1]) - np.int64(raw_conv[s_lo - 1])
                    lo_bin = s_lo - (d_child - 1)
                    if lo_bin < 0:
                        lo_bin = 0
                    hi_bin = s_lo + ell - 2
                    if hi_bin > d_child - 1:
                        hi_bin = d_child - 1
                    W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                    dyn_it = threshold_table[ell_idx * S_child_plus_1 + W_int]
                    if ws > dyn_it:
                        pruned = True
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        break

            if not pruned:
                # --- Survivor! Canonicalize: min(child, rev(child)) lex ---
                use_rev = False
                for i in range(d_child // 2):
                    j = d_child - 1 - i
                    if child[j] < child[i]:
                        use_rev = True
                        break
                    elif child[j] > child[i]:
                        break

                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                n_surv += 1

        # --- Advance cursor (odometer increment) ---
        carry = d_parent - 1
        while carry >= 0:
            cursor[carry] += 1
            if cursor[carry] <= hi_arr[carry]:
                break
            cursor[carry] = lo_arr[carry]
            carry -= 1

        if carry < 0:
            break

        # --- Build new child for changed positions ---
        n_changed = d_parent - carry

        if n_changed == 1:
            # === FAST PATH: only last position changed (~67% of steps) ===
            pos = d_parent - 1
            k1 = 2 * pos
            k2 = k1 + 1
            old1 = np.int32(child[k1])
            old2 = np.int32(child[k2])
            child[k1] = cursor[pos]
            child[k2] = 2 * parent_int[pos] - cursor[pos]
            new1 = np.int32(child[k1])
            new2 = np.int32(child[k2])
            delta1 = new1 - old1
            delta2 = new2 - old2

            # Self-terms
            raw_conv[2 * k1] += new1 * new1 - old1 * old1
            raw_conv[2 * k2] += new2 * new2 - old2 * old2
            # Mutual term
            raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)
            # Cross-terms with all unchanged bins (j < k1)
            for j in range(k1):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[k1 + j] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + j] += np.int32(2) * delta2 * cj

            # Quick-check: O(1) W_int update (only bins k1, k2 changed)
            if qc_ell > 0:
                qc_lo = qc_s - (d_child - 1)
                if qc_lo < 0:
                    qc_lo = 0
                qc_hi = qc_s + qc_ell - 2
                if qc_hi > d_child - 1:
                    qc_hi = d_child - 1
                if qc_lo <= k1 and k1 <= qc_hi:
                    qc_W_int += np.int64(delta1)
                if qc_lo <= k2 and k2 <= qc_hi:
                    qc_W_int += np.int64(delta2)

        elif n_changed <= carry_threshold:
            # === SHORT CARRY: incremental update for 2..threshold positions ===
            first_changed_bin = 2 * carry

            # Save prev_child (only needed for incremental path)
            for i in range(d_child):
                prev_child[i] = child[i]

            # Rebuild changed child bins
            for pos in range(carry, d_parent):
                child[2 * pos] = cursor[pos]
                child[2 * pos + 1] = 2 * parent_int[pos] - cursor[pos]

            # Self + mutual terms for each changed position pair
            for pos in range(carry, d_parent):
                k1 = 2 * pos
                k2 = k1 + 1
                old1 = np.int32(prev_child[k1])
                old2 = np.int32(prev_child[k2])
                new1 = np.int32(child[k1])
                new2 = np.int32(child[k2])
                raw_conv[2 * k1] += new1 * new1 - old1 * old1
                raw_conv[2 * k2] += new2 * new2 - old2 * old2
                raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)

            # Cross-terms between different changed position pairs
            for pa in range(carry, d_parent):
                a1 = 2 * pa
                a2 = a1 + 1
                new_a1 = np.int32(child[a1])
                new_a2 = np.int32(child[a2])
                old_a1 = np.int32(prev_child[a1])
                old_a2 = np.int32(prev_child[a2])
                for pb in range(pa + 1, d_parent):
                    b1 = 2 * pb
                    b2 = b1 + 1
                    new_b1 = np.int32(child[b1])
                    new_b2 = np.int32(child[b2])
                    old_b1 = np.int32(prev_child[b1])
                    old_b2 = np.int32(prev_child[b2])
                    raw_conv[a1 + b1] += np.int32(2) * (new_a1 * new_b1 - old_a1 * old_b1)
                    raw_conv[a1 + b2] += np.int32(2) * (new_a1 * new_b2 - old_a1 * old_b2)
                    raw_conv[a2 + b1] += np.int32(2) * (new_a2 * new_b1 - old_a2 * old_b1)
                    raw_conv[a2 + b2] += np.int32(2) * (new_a2 * new_b2 - old_a2 * old_b2)

            # Cross-terms between changed bins and unchanged bins
            for pos in range(carry, d_parent):
                k1 = 2 * pos
                k2 = k1 + 1
                delta1 = np.int32(child[k1]) - np.int32(prev_child[k1])
                delta2 = np.int32(child[k2]) - np.int32(prev_child[k2])
                for j in range(first_changed_bin):
                    cj = np.int32(child[j])
                    if cj != 0:
                        raw_conv[k1 + j] += np.int32(2) * delta1 * cj
                        raw_conv[k2 + j] += np.int32(2) * delta2 * cj

            # Quick-check: recompute W_int (multiple bins changed)
            if qc_ell > 0:
                qc_lo = qc_s - (d_child - 1)
                if qc_lo < 0:
                    qc_lo = 0
                qc_hi = qc_s + qc_ell - 2
                if qc_hi > d_child - 1:
                    qc_hi = d_child - 1
                qc_W_int = np.int64(0)
                for i in range(qc_lo, qc_hi + 1):
                    qc_W_int += np.int64(child[i])

        else:
            # === DEEP CARRY: attempt subtree prune before full recompute ===
            fixed_len = 2 * carry          # number of fixed child bins

            if fixed_len >= 4:  # need at least 4 bins for a meaningful check
                # Compute partial autoconvolution (fixed bins only)
                partial_conv_len = 2 * fixed_len - 1
                for k in range(partial_conv_len):
                    conv[k] = np.int32(0)
                for i in range(fixed_len):
                    ci = np.int32(child[i])
                    if ci != 0:
                        conv[2 * i] += ci * ci
                        for j in range(i + 1, fixed_len):
                            cj = np.int32(child[j])
                            if cj != 0:
                                conv[i + j] += np.int32(2) * ci * cj
                # Prefix sum
                for k in range(1, partial_conv_len):
                    conv[k] += conv[k - 1]

                # Compute fixed-region prefix_c for W_int
                prefix_c[0] = 0
                for i in range(fixed_len):
                    prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

                # Window scan with W_int_max thresholds
                subtree_pruned = False
                first_unfixed_parent = carry

                for ell_oi in range(ell_count):
                    if subtree_pruned:
                        break
                    ell = ell_order[ell_oi]
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    scale_ell = scale_arr[ell_idx]

                    # Only check windows fully contained in partial conv
                    n_windows_partial = partial_conv_len - n_cv + 1
                    if n_windows_partial <= 0:
                        continue

                    for s_lo in range(n_windows_partial):
                        s_hi = s_lo + n_cv - 1
                        ws = np.int64(conv[s_hi])
                        if s_lo > 0:
                            ws -= np.int64(conv[s_lo - 1])

                        # W_int_max: fixed child bins + unfixed parent bins
                        lo_bin = s_lo - (d_child - 1)
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_child - 1:
                            hi_bin = d_child - 1

                        # Fixed part
                        fixed_hi = hi_bin
                        if fixed_hi > fixed_len - 1:
                            fixed_hi = fixed_len - 1
                        if fixed_hi >= lo_bin:
                            lo_clamp = lo_bin
                            if lo_clamp < 0:
                                lo_clamp = 0
                            W_int_fixed = prefix_c[fixed_hi + 1] - prefix_c[lo_clamp]
                        else:
                            W_int_fixed = np.int64(0)

                        # Unfixed part (child bins sum to 2*parent)
                        unfixed_lo_bin = lo_bin
                        if unfixed_lo_bin < fixed_len:
                            unfixed_lo_bin = fixed_len
                        if unfixed_lo_bin <= hi_bin:
                            p_lo = unfixed_lo_bin // 2
                            p_hi = hi_bin // 2
                            if p_lo < first_unfixed_parent:
                                p_lo = first_unfixed_parent
                            if p_hi >= d_parent:
                                p_hi = d_parent - 1
                            if p_lo <= p_hi:
                                W_int_unfixed = np.int64(2) * (parent_prefix[p_hi + 1] - parent_prefix[p_lo])
                            else:
                                W_int_unfixed = np.int64(0)
                        else:
                            W_int_unfixed = np.int64(0)

                        W_int_max = W_int_fixed + W_int_unfixed
                        corr_w = 1.0 + np.float64(W_int_max) / (2.0 * n_half_d)
                        dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                        dyn_it = np.int64(dyn_x)
                        if ws > dyn_it:
                            subtree_pruned = True
                            break

                if subtree_pruned:
                    n_subtree_pruned += 1
                    # Skip entire subtree: fast-forward trailing cursors
                    for i in range(carry + 1, d_parent):
                        cursor[i] = hi_arr[i]
                    # Rebuild child for current cursor
                    for pos in range(carry, d_parent):
                        child[2 * pos] = cursor[pos]
                        child[2 * pos + 1] = 2 * parent_int[pos] - cursor[pos]
                    # Full recompute of raw_conv
                    for k in range(conv_len):
                        raw_conv[k] = np.int32(0)
                    for i in range(d_child):
                        ci = np.int32(child[i])
                        if ci != 0:
                            raw_conv[2 * i] += ci * ci
                            for j in range(i + 1, d_child):
                                cj = np.int32(child[j])
                                if cj != 0:
                                    raw_conv[i + j] += np.int32(2) * ci * cj
                    # Quick-check: recompute W_int after subtree recompute
                    if qc_ell > 0:
                        qc_lo = qc_s - (d_child - 1)
                        if qc_lo < 0:
                            qc_lo = 0
                        qc_hi = qc_s + qc_ell - 2
                        if qc_hi > d_child - 1:
                            qc_hi = d_child - 1
                        qc_W_int = np.int64(0)
                        for i in range(qc_lo, qc_hi + 1):
                            qc_W_int += np.int64(child[i])
                    continue

            # === Not subtree-pruned: original full recompute path ===
            for pos in range(carry, d_parent):
                child[2 * pos] = cursor[pos]
                child[2 * pos + 1] = 2 * parent_int[pos] - cursor[pos]

            for k in range(conv_len):
                raw_conv[k] = np.int32(0)
            for i in range(d_child):
                ci = np.int32(child[i])
                if ci != 0:
                    raw_conv[2 * i] += ci * ci
                    for j in range(i + 1, d_child):
                        cj = np.int32(child[j])
                        if cj != 0:
                            raw_conv[i + j] += np.int32(2) * ci * cj

            # Quick-check: recompute W_int after full recompute
            if qc_ell > 0:
                qc_lo = qc_s - (d_child - 1)
                if qc_lo < 0:
                    qc_lo = 0
                qc_hi = qc_s + qc_ell - 2
                if qc_hi > d_child - 1:
                    qc_hi = d_child - 1
                qc_W_int = np.int64(0)
                for i in range(qc_lo, qc_hi + 1):
                    qc_W_int += np.int64(child[i])

    return n_surv, n_subtree_pruned



# =====================================================================
# Gray code variant — O(d) per step, no deep carries
# =====================================================================

@njit(cache=True)
def _fused_generate_and_prune_gray(parent_int, n_half_child, m, c_target,
                                    lo_arr, hi_arr, out_buf,
                                    use_flat_threshold=False):
    """Gray code variant: visits same Cartesian product, O(d) per step.

    Replaces the lexicographic odometer with a mixed-radix Gray code
    (Knuth TAOCP 7.2.1.1).  Every step changes exactly one cursor position
    by ±1, so the incremental autoconvolution update is always O(d_child).
    Includes subtree pruning at J_MIN level for inner-sweep skip.

    When use_flat_threshold=True, uses the flat C&S Lemma 3 correction
    (2m+1)/m^2 instead of the W-refined (3+W_int/(2n))/m^2.

    Same signature and return type as _fused_generate_and_prune.
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    assert m <= 200, f"int32 conv requires m <= 200, got m={m}"

    # --- Asymmetry filter (identical to original) ---
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    S_parent = np.int64(0)
    for i in range(d_parent):
        S_parent += np.int64(parent_int[i])
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / np.float64(S_parent)
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0

    # --- Dynamic pruning constants (C&S Lemma 3 + W-refinement) ---
    # Fine grid (S = 4nm): compositions sum to S, conv values are int32.
    # threshold = floor((c_target*m^2 + 1 + W_int/(2n) + eps) * 4n*ell)
    # where W_int ranges 0..S (fine-grid mass in overlapping bins).
    four_n = 4.0 * np.float64(n_half_child)
    n_half_d = np.float64(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1

    # --- Subtree pruning constants ---
    J_MIN = 7
    n_subtree_pruned = 0
    partial_conv = np.empty(conv_len, dtype=np.int32)
    min_contrib = np.empty(conv_len, dtype=np.int64)
    min_contrib_prefix = np.empty(conv_len, dtype=np.int64)

    # Prefix sum of parent bins (for W_int_unfixed in subtree pruning)
    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])

    # --- Allocate arrays ---
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)

    # --- Sparse cross-term optimization (d_child >= 32) ---
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0

    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

    # --- L1-resident staging buffer for survivor writes ---
    # Benchmark (Section 5): AMD EPYC 9354 L1d = 32KB.  Previous 64KB
    # staging buffer spilled into L2, defeating the purpose.  Halved to
    # fit entirely in L1 alongside the kernel's hot working set (~2KB).
    if d_child <= 32:
        _STAGE_CAP = 256    # 256 * 32 * 4 = 32KB, fits L1 (32KB)
    else:
        _STAGE_CAP = 128    # 128 * 64 * 4 = 32KB, fits L1 (32KB)
    stage_buf = np.empty((_STAGE_CAP, d_child), dtype=np.int32)
    n_staged = 0

    # --- Build initial child ---
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = 2 * parent_int[i] - cursor[i]

    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i
                nz_pos[i] = nz_count
                nz_count += 1

    # --- Precompute 2D threshold table ---
    # Fine grid (S = 4nm): W_int ranges 0..S (fine-grid mass in window bins).
    # threshold_table[ell_idx * S_child_plus_1 + W_int] =
    #   floor((c_target*m^2 + corr + eps) * 4n*ell)
    # where corr = 2m+1 (flat C&S) or 0 (Theorem 1, no correction).
    #
    # DEFAULT (use_flat_threshold=False): Theorem 1 threshold — no correction.
    # Theorem 1 is EXACT for bin masses (no step-function approximation).
    # This is ~10% tighter than the W-refined threshold at m=16, giving
    # ~30-50% fewer survivors per level.  Sound because Theorem 1 guarantees
    # max(f*f) >= TV_W(mu) for ANY f with bin masses mu.
    # Box certification is required at the final cascade level.
    #
    # use_flat_threshold=True: flat C&S Lemma 3 correction (2m+1)/m^2.
    # Required for the Lean axiom cascade_all_pruned.
    ell_count = 2 * d_child - 1
    S_child_plus_1 = int(4 * n_half_child * m + 1)
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
            # Theorem 1 + box certification (per-window proven bound):
            #
            # correction_int(ell) = min(n, ell-1, 2d-ell) * B
            # where B = n*(8m+1)/2, n = n_half_child, d = d_child.
            #
            # Three proven ingredients:
            # 1. Per-index lemma: |Δconv[k]| ≤ S + d/4 = B
            # 2. Complement trick: Σ Δconv[k] = 0 → complement bound
            # 3. Gradient-pairing: constant n*B for all ell
            #
            # Combined: min(n, ell-1, 2d-ell) * B
            # 3-5x tighter than C&S at typical killing windows (ell ≈ d/2).
            B_corr = n_half_d * (8.0 * m_d + 1.0) / 2.0
            n_int = int(n_half_child)
            mult = ell - 1
            comp = 2 * d_child - ell
            if comp < mult:
                mult = comp
            if n_int < mult:
                mult = n_int
            box_corr = np.float64(mult) * B_corr
            th1_val = np.int64(cs_base_m2 * scale_ell + box_corr)
            for w in range(S_child_plus_1):
                threshold_table[idx * S_child_plus_1 + w] = th1_val

    # --- Optimized ell scan order ---
    # Empirically tuned: at d_child=32, ell=7-13 account for 92% of prunes.
    # Start with those in decreasing-kill-rate order, then widen outward.
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    if d_child >= 20:
        # Profile-guided order: ell values sorted by kill rate at d_child=32
        # ell=9(27%), 10(16%), 11(13%), 8(12%), 7(10%), 12(9%), 13(5%),
        # 6(3%), 14(2%), 5(1%), 15, 16, then widen
        hc = d_child // 2  # half_child = center of killing range
        for ell in (hc + 1, hc + 2, hc + 3, hc, hc - 1, hc + 4, hc + 5,
                    hc - 2, hc + 6, hc - 3, hc + 7, hc + 8):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
        # Phase 2: wide windows around d_child
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                    d_child * 2, d_child + d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
    else:
        # Original Phase 1 for small d_child
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
    # Phase 3: everything else in order
    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1

    # --- Compute full raw_conv for initial child ---
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj

    # --- Gray code setup: active positions (range > 1) ---
    # Build right-to-left so inner (fast) digits are rightmost parent bins.
    # This makes the fixed region for subtree pruning a left prefix.
    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent - 1, -1, -1):
        r = hi_arr[i] - lo_arr[i] + 1
        if r > 1:
            active_pos[n_active] = i
            radix[n_active] = r
            n_active += 1

    # Knuth TAOCP 7.2.1.1 — mixed-radix Gray code state
    gc_a = np.zeros(n_active, dtype=np.int32)       # relative digits
    gc_dir = np.ones(n_active, dtype=np.int32)       # +1 or -1
    gc_focus = np.empty(n_active + 1, dtype=np.int32)
    for i in range(n_active + 1):
        gc_focus[i] = i

    # --- Main loop ---
    while True:
        # === TEST current child (identical to original lines 642-718) ===
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell - 2
            dyn_it_qc = threshold_table[ell_idx_qc * S_child_plus_1 + qc_W_int]
            if ws_qc > dyn_it_qc:
                quick_killed = True

        if not quick_killed:
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

            pruned = False
            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                ell_idx = ell - 2
                n_windows = conv_len - n_cv + 1
                # Sliding window: initialize sum for s_lo=0
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(raw_conv[s_lo + n_cv - 1]) - np.int64(raw_conv[s_lo - 1])
                    lo_bin = s_lo - (d_child - 1)
                    if lo_bin < 0:
                        lo_bin = 0
                    hi_bin = s_lo + ell - 2
                    if hi_bin > d_child - 1:
                        hi_bin = d_child - 1
                    W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                    dyn_it = threshold_table[ell_idx * S_child_plus_1 + W_int]
                    if ws > dyn_it:
                        pruned = True
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        break

            if not pruned:
                use_rev = False
                half_d = d_child // 2
                for i in range(half_d):
                    j = d_child - 1 - i
                    if child[j] < child[i]:
                        use_rev = True
                        break
                    elif child[j] > child[i]:
                        break

                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            stage_buf[n_staged, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            stage_buf[n_staged, i] = child[i]
                    n_staged += 1
                    if n_staged == _STAGE_CAP:
                        flush_base = n_surv + 1 - _STAGE_CAP
                        for fi in range(_STAGE_CAP):
                            for ci in range(d_child):
                                out_buf[flush_base + fi, ci] = stage_buf[fi, ci]
                        n_staged = 0
                n_surv += 1

        # === GRAY CODE ADVANCE ===
        j = gc_focus[0]
        if j == n_active:
            break                             # all children visited
        gc_focus[0] = 0

        pos = active_pos[j]                   # physical parent position
        gc_a[j] += gc_dir[j]
        cursor[pos] = lo_arr[pos] + gc_a[j]

        # Boundary: reverse direction, update focus chain
        if gc_a[j] == 0 or gc_a[j] == radix[j] - 1:
            gc_dir[j] = -gc_dir[j]
            gc_focus[j] = gc_focus[j + 1]
            gc_focus[j + 1] = j + 1

        # === INCREMENTAL UPDATE (always single-position, O(d_child)) ===
        k1 = 2 * pos
        k2 = k1 + 1
        old1 = np.int32(child[k1])
        old2 = np.int32(child[k2])
        child[k1] = cursor[pos]
        child[k2] = 2 * parent_int[pos] - cursor[pos]
        new1 = np.int32(child[k1])
        new2 = np.int32(child[k2])
        delta1 = new1 - old1
        delta2 = new2 - old2

        # Self-terms
        raw_conv[2 * k1] += new1 * new1 - old1 * old1
        raw_conv[2 * k2] += new2 * new2 - old2 * old2
        # Mutual term (k2 = k1 + 1, always adjacent)
        raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)
        # Cross-terms with unchanged bins
        if use_sparse:
            # Update nz_list for changed bins k1, k2
            if old1 != 0 and new1 == 0:
                p = nz_pos[k1]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k1] = -1
            elif old1 == 0 and new1 != 0:
                nz_list[nz_count] = k1; nz_pos[k1] = nz_count; nz_count += 1
            if old2 != 0 and new2 == 0:
                p = nz_pos[k2]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k2] = -1
            elif old2 == 0 and new2 != 0:
                nz_list[nz_count] = k2; nz_pos[k2] = nz_count; nz_count += 1
            # Iterate only nonzero bins, skip k1 and k2
            for idx in range(nz_count):
                jj = nz_list[idx]
                if jj != k1 and jj != k2:
                    cj = np.int32(child[jj])
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
        else:
            for jj in range(k1):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
            for jj in range(k2 + 1, d_child):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj

        # Quick-check W_int update (O(1))
        if qc_ell > 0:
            qc_lo = qc_s - (d_child - 1)
            if qc_lo < 0:
                qc_lo = 0
            qc_hi = qc_s + qc_ell - 2
            if qc_hi > d_child - 1:
                qc_hi = d_child - 1
            if qc_lo <= k1 and k1 <= qc_hi:
                qc_W_int += np.int64(delta1)
            if qc_lo <= k2 and k2 <= qc_hi:
                qc_W_int += np.int64(delta2)

        # === SUBTREE PRUNING CHECK ===
        # When digit J_MIN just advanced, inner digits 0..J_MIN-1 are about
        # to sweep.  Check if the partial autoconvolution of the fixed left
        # prefix already exceeds the threshold for all possible inner values.
        if j == J_MIN and n_active > J_MIN:
            fixed_parent_boundary = active_pos[J_MIN - 1]
            fixed_len = 2 * fixed_parent_boundary

            if fixed_len >= 4:
                # Compute partial autoconvolution of fixed prefix
                partial_conv_len = 2 * fixed_len - 1
                for kk in range(partial_conv_len):
                    partial_conv[kk] = np.int32(0)
                for ii in range(fixed_len):
                    ci = np.int32(child[ii])
                    if ci != 0:
                        partial_conv[2 * ii] += ci * ci
                        for jj2 in range(ii + 1, fixed_len):
                            cj2 = np.int32(child[jj2])
                            if cj2 != 0:
                                partial_conv[ii + jj2] += np.int32(2) * ci * cj2
                # Prefix sum for sliding window
                for kk in range(1, partial_conv_len):
                    partial_conv[kk] += partial_conv[kk - 1]

                # Fixed-region child prefix for W_int_fixed
                prefix_c[0] = 0
                for ii in range(fixed_len):
                    prefix_c[ii + 1] = prefix_c[ii] + np.int64(child[ii])

                first_unfixed_parent = fixed_parent_boundary
                subtree_pruned = False

                # --- Idea 02: guaranteed minimum contributions from unfixed bins ---
                for kk in range(conv_len):
                    min_contrib[kk] = np.int64(0)

                # (A) Inner active positions (digits 0..J_MIN-1)
                for kk in range(J_MIN):
                    p_unf = active_pos[kk]
                    k1u = 2 * p_unf
                    k2u = 2 * p_unf + 1
                    ml = np.int64(lo_arr[p_unf])
                    mh = np.int64(2) * np.int64(parent_int[p_unf]) - np.int64(hi_arr[p_unf])
                    # Self-terms
                    min_contrib[2 * k1u] += ml * ml
                    min_contrib[2 * k2u] += mh * mh
                    # Mutual term: min of concave 2*c1*(2P-c1) on [lo,hi]
                    # is at an endpoint, NOT at independent mins ml*mh.
                    lo_val = np.int64(lo_arr[p_unf])
                    hi_val = np.int64(hi_arr[p_unf])
                    two_P = np.int64(2) * np.int64(parent_int[p_unf])
                    mut_lo = np.int64(2) * lo_val * (two_P - lo_val)
                    mut_hi = np.int64(2) * hi_val * (two_P - hi_val)
                    min_contrib[k1u + k2u] += mut_lo if mut_lo < mut_hi else mut_hi
                    # Cross with fixed bins
                    for ii in range(fixed_len):
                        ci64 = np.int64(child[ii])
                        if ci64 > 0:
                            if ml > 0:
                                min_contrib[ii + k1u] += np.int64(2) * ci64 * ml
                            if mh > 0:
                                min_contrib[ii + k2u] += np.int64(2) * ci64 * mh
                    # Cross with other inner unfixed
                    for kk2 in range(kk + 1, J_MIN):
                        p2 = active_pos[kk2]
                        k1u2 = 2 * p2
                        k2u2 = 2 * p2 + 1
                        ml2 = np.int64(lo_arr[p2])
                        mh2 = np.int64(2) * np.int64(parent_int[p2]) - np.int64(hi_arr[p2])
                        if ml > 0 and ml2 > 0:
                            min_contrib[k1u + k1u2] += np.int64(2) * ml * ml2
                        if ml > 0 and mh2 > 0:
                            min_contrib[k1u + k2u2] += np.int64(2) * ml * mh2
                        if mh > 0 and ml2 > 0:
                            min_contrib[k2u + k1u2] += np.int64(2) * mh * ml2
                        if mh > 0 and mh2 > 0:
                            min_contrib[k2u + k2u2] += np.int64(2) * mh * mh2

                # (B) Non-active unfixed parents beyond fixed prefix
                for pp in range(first_unfixed_parent, d_parent):
                    is_inner = False
                    for kk in range(J_MIN):
                        if active_pos[kk] == pp:
                            is_inner = True
                            break
                    if is_inner:
                        continue
                    k1na = 2 * pp
                    k2na = 2 * pp + 1
                    cv1 = np.int64(lo_arr[pp])
                    cv2 = np.int64(2) * np.int64(parent_int[pp]) - cv1
                    # Self-terms
                    min_contrib[2 * k1na] += cv1 * cv1
                    min_contrib[2 * k2na] += cv2 * cv2
                    min_contrib[k1na + k2na] += np.int64(2) * cv1 * cv2
                    # Cross with fixed prefix
                    for ii in range(fixed_len):
                        ci64 = np.int64(child[ii])
                        if ci64 > 0:
                            if cv1 > 0:
                                min_contrib[ii + k1na] += np.int64(2) * ci64 * cv1
                            if cv2 > 0:
                                min_contrib[ii + k2na] += np.int64(2) * ci64 * cv2
                    # Cross with inner unfixed
                    for kk in range(J_MIN):
                        p_unf = active_pos[kk]
                        k1u = 2 * p_unf
                        k2u = 2 * p_unf + 1
                        ml = np.int64(lo_arr[p_unf])
                        mh = np.int64(2) * np.int64(parent_int[p_unf]) - np.int64(hi_arr[p_unf])
                        if cv1 > 0 and ml > 0:
                            min_contrib[k1na + k1u] += np.int64(2) * cv1 * ml
                        if cv1 > 0 and mh > 0:
                            min_contrib[k1na + k2u] += np.int64(2) * cv1 * mh
                        if cv2 > 0 and ml > 0:
                            min_contrib[k2na + k1u] += np.int64(2) * cv2 * ml
                        if cv2 > 0 and mh > 0:
                            min_contrib[k2na + k2u] += np.int64(2) * cv2 * mh
                    # Cross with other non-active unfixed
                    for pp2 in range(pp + 1, d_parent):
                        is_inner2 = False
                        for kk2b in range(J_MIN):
                            if active_pos[kk2b] == pp2:
                                is_inner2 = True
                                break
                        if is_inner2:
                            continue
                        k1na2 = 2 * pp2
                        k2na2 = 2 * pp2 + 1
                        cv12 = np.int64(lo_arr[pp2])
                        cv22 = np.int64(2) * np.int64(parent_int[pp2]) - cv12
                        if cv1 > 0 and cv12 > 0:
                            min_contrib[k1na + k1na2] += np.int64(2) * cv1 * cv12
                        if cv1 > 0 and cv22 > 0:
                            min_contrib[k1na + k2na2] += np.int64(2) * cv1 * cv22
                        if cv2 > 0 and cv12 > 0:
                            min_contrib[k2na + k1na2] += np.int64(2) * cv2 * cv12
                        if cv2 > 0 and cv22 > 0:
                            min_contrib[k2na + k2na2] += np.int64(2) * cv2 * cv22

                # Build prefix sum of min_contrib
                min_contrib_prefix[0] = min_contrib[0]
                for kk in range(1, conv_len):
                    min_contrib_prefix[kk] = min_contrib_prefix[kk - 1] + min_contrib[kk]

                # Window scan: ALL positions (not just within partial_conv_len)
                for ell_oi in range(ell_count):
                    if subtree_pruned:
                        break
                    ell = ell_order[ell_oi]
                    n_cv = ell - 1
                    ell_idx = ell - 2

                    n_windows_total = conv_len - n_cv + 1
                    if n_windows_total <= 0:
                        continue

                    for s_lo in range(n_windows_total):
                        s_hi = s_lo + n_cv - 1

                        # Fixed part: partial_conv in window ∩ [0, partial_conv_len)
                        ws = np.int64(0)
                        k_start = s_lo
                        k_end = s_hi
                        if k_end >= partial_conv_len:
                            k_end = partial_conv_len - 1
                        if k_end >= k_start:
                            ws = np.int64(partial_conv[k_end])
                            if k_start > 0:
                                ws -= np.int64(partial_conv[k_start - 1])

                        # Unfixed part: guaranteed minimum contributions
                        mc_sum = min_contrib_prefix[s_hi]
                        if s_lo > 0:
                            mc_sum -= min_contrib_prefix[s_lo - 1]
                        ws += mc_sum

                        lo_bin = s_lo - (d_child - 1)
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_child - 1:
                            hi_bin = d_child - 1

                        # W_int_fixed: actual child masses in fixed prefix
                        fixed_hi = hi_bin
                        if fixed_hi > fixed_len - 1:
                            fixed_hi = fixed_len - 1
                        if fixed_hi >= lo_bin:
                            lo_clamp = lo_bin
                            if lo_clamp < 0:
                                lo_clamp = 0
                            W_int_fixed = prefix_c[fixed_hi + 1] - prefix_c[lo_clamp]
                        else:
                            W_int_fixed = np.int64(0)

                        # W_int_unfixed: parent upper bound for bins
                        # right of fixed prefix (child bins sum to 2*parent)
                        unfixed_lo_bin = lo_bin
                        if unfixed_lo_bin < fixed_len:
                            unfixed_lo_bin = fixed_len
                        if unfixed_lo_bin <= hi_bin:
                            p_lo = unfixed_lo_bin // 2
                            p_hi = hi_bin // 2
                            if p_lo < first_unfixed_parent:
                                p_lo = first_unfixed_parent
                            if p_hi >= d_parent:
                                p_hi = d_parent - 1
                            if p_lo <= p_hi:
                                W_int_unfixed = np.int64(2) * (parent_prefix[p_hi + 1] - parent_prefix[p_lo])
                            else:
                                W_int_unfixed = np.int64(0)
                        else:
                            W_int_unfixed = np.int64(0)

                        W_int_max = W_int_fixed + W_int_unfixed
                        if W_int_max >= np.int64(S_child_plus_1):
                            W_int_max = np.int64(S_child_plus_1 - 1)
                        dyn_it = threshold_table[ell_idx * S_child_plus_1 + W_int_max]
                        if ws > dyn_it:
                            subtree_pruned = True
                            break

                if subtree_pruned:
                    n_subtree_pruned += 1

                    # Save where focus should go after skip
                    next_focus = gc_focus[J_MIN]

                    # Reset inner Gray code digits to fresh state
                    for kk in range(J_MIN):
                        gc_a[kk] = 0
                        gc_dir[kk] = 1
                        gc_focus[kk] = kk

                    # Wire focus to skip inner sweep
                    gc_focus[0] = next_focus
                    gc_focus[J_MIN] = J_MIN

                    # Reset cursor and child for inner positions
                    for kk in range(J_MIN):
                        p = active_pos[kk]
                        cursor[p] = lo_arr[p]
                        child[2 * p] = lo_arr[p]
                        child[2 * p + 1] = 2 * parent_int[p] - lo_arr[p]

                    # Full recompute raw_conv (O(d²))
                    for kk in range(conv_len):
                        raw_conv[kk] = np.int32(0)
                    for ii in range(d_child):
                        ci = np.int32(child[ii])
                        if ci != 0:
                            raw_conv[2 * ii] += ci * ci
                            for jj2 in range(ii + 1, d_child):
                                cj2 = np.int32(child[jj2])
                                if cj2 != 0:
                                    raw_conv[ii + jj2] += np.int32(2) * ci * cj2

                    # Rebuild nz_list after subtree prune reset
                    if use_sparse:
                        nz_count = 0
                        for ii in range(d_child):
                            if child[ii] != 0:
                                nz_list[nz_count] = ii
                                nz_pos[ii] = nz_count
                                nz_count += 1
                            else:
                                nz_pos[ii] = -1

                    # Recompute qc_W_int
                    if qc_ell > 0:
                        qc_lo2 = qc_s - (d_child - 1)
                        if qc_lo2 < 0:
                            qc_lo2 = 0
                        qc_hi2 = qc_s + qc_ell - 2
                        if qc_hi2 > d_child - 1:
                            qc_hi2 = d_child - 1
                        qc_W_int = np.int64(0)
                        for ii in range(qc_lo2, qc_hi2 + 1):
                            qc_W_int += np.int64(child[ii])

                    continue  # skip to TEST at top of loop

    # --- Final flush of remaining staged survivors ---
    if n_staged > 0:
        flush_base = min(n_surv, max_survivors) - n_staged
        for fi in range(n_staged):
            for ci in range(d_child):
                out_buf[flush_base + fi, ci] = stage_buf[fi, ci]

    return n_surv, n_subtree_pruned


# =====================================================================
# Coarse-grid Gray code kernel (Theorem 1 + sound box cert)
# =====================================================================

@njit(cache=True)
def _fused_coarse_gray(parent_int, d_child, S, c_target,
                        lo_arr, hi_arr, out_buf,
                        prefix_nk, prefix_mk):
    """Coarse-grid Gray code kernel with full optimizations + box cert.

    Theorem 1: no correction at grid points.
    Child: child[2i]+child[2i+1] = parent[i] (constant S, no factor 2).
    Threshold: ws > floor(c_target * ell * S^2/(2d) - eps) (1D, no W_int).
    Box cert: tracks min(margin - cell_var - quad_corr).

    Optimizations: Gray code O(d) steps, incremental conv, quick-check,
    optimized ell scan, sparse cross-terms, subtree pruning, staging buffer.

    Returns (n_survivors, n_subtree_pruned, min_cert_net).
    """
    d_parent = parent_int.shape[0]

    # --- Asymmetry filter ---
    threshold_asym = math.sqrt(c_target / 2.0)
    S_d = np.float64(S)
    left_sum = np.int64(0)
    for i in range(d_parent // 2):
        left_sum += np.int64(parent_int[i])
    left_frac = np.float64(left_sum) / S_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0, np.float64(1e30)

    # --- Threshold constants (coarse: 1D, no W_int) ---
    S_sq = S_d * S_d
    d_d = np.float64(d_child)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d_child
    conv_len = 2 * d_child - 1
    ell_count = conv_len

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d
                                - eps)

    max_survivors = out_buf.shape[0]
    n_surv = 0
    n_subtree_pruned = 0
    local_min_net = np.float64(1e30)

    # --- Sparse cross-term optimization ---
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0

    # Quick-check state (no W_int for coarse)
    qc_ell = np.int32(0)
    qc_s = np.int32(0)

    # --- Allocate arrays ---
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    grad_buf = np.empty(d_child, dtype=np.float64)

    # Subtree pruning arrays
    J_MIN = 7
    partial_conv = np.empty(conv_len, dtype=np.int32)
    min_contrib = np.empty(conv_len, dtype=np.int64)
    min_contrib_prefix = np.empty(conv_len, dtype=np.int64)

    # L1-resident staging buffer
    if d_child <= 32:
        _STAGE_CAP = 256
    else:
        _STAGE_CAP = 128
    stage_buf = np.empty((_STAGE_CAP, d_child), dtype=np.int32)
    n_staged = 0

    # --- Build initial child (COARSE: no factor 2) ---
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i
                nz_pos[i] = nz_count
                nz_count += 1

    # --- Optimized ell scan order ---
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    if d_child >= 20:
        hc = d_child // 2
        for ell in (hc + 1, hc + 2, hc + 3, hc, hc - 1, hc + 4, hc + 5,
                    hc - 2, hc + 6, hc - 3, hc + 7, hc + 8):
            if 2 <= ell <= max_ell and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2,
                    d_child - 2, max_ell, d_child + d_child // 2):
            if 2 <= ell <= max_ell and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
    else:
        phase1_end = min(16, max_ell)
        for ell in range(2, phase1_end + 1):
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = np.int32(1)
            oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2,
                    d_child - 2, max_ell, d_child + d_child // 2,
                    d_child // 2):
            if 2 <= ell <= max_ell and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
    for ell in range(2, max_ell + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1

    # --- Compute full raw_conv for initial child ---
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj

    # --- Gray code setup ---
    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent - 1, -1, -1):
        r = hi_arr[i] - lo_arr[i] + 1
        if r > 1:
            active_pos[n_active] = i
            radix[n_active] = r
            n_active += 1

    gc_a = np.zeros(n_active, dtype=np.int32)
    gc_dir = np.ones(n_active, dtype=np.int32)
    gc_focus = np.empty(n_active + 1, dtype=np.int32)
    for i in range(n_active + 1):
        gc_focus[i] = i

    # --- Main loop ---
    while True:
        # === TEST current child ===
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            if ws_qc > thr_arr[qc_ell]:
                quick_killed = True

        if not quick_killed:
            pruned = False

            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                n_windows = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                thr_val = thr_arr[ell]
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += (np.int64(raw_conv[s_lo + n_cv - 1])
                               - np.int64(raw_conv[s_lo - 1]))
                    if ws > thr_val:
                        pruned = True
                        break
        else:
            pruned = True

        if pruned:
            # --- Multi-window box cert: scan ALL killing windows,
            #     pick the one with best net = margin - cell_var - qc.
            #     Each window gives an independent lower bound on
            #     min_{mu in cell} TV_W(mu); taking the max is sound. ---
            best_net = np.float64(-1e30)
            best_kill_ell = np.int32(2)
            best_kill_s = np.int32(0)

            for ell in range(2, max_ell + 1):
                n_cv = ell - 1
                n_windows = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                thr_val = thr_arr[ell]
                ell_f = np.float64(ell)
                scale_g = 4.0 * d_d / ell_f

                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += (np.int64(raw_conv[s_lo + n_cv - 1])
                               - np.int64(raw_conv[s_lo - 1]))
                    if ws > thr_val:
                        tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                        margin = tv - c_target

                        for i in range(d_child):
                            g = 0.0
                            for j_g in range(d_child):
                                kk = i + j_g
                                if s_lo <= kk <= s_lo + ell - 2:
                                    g += np.float64(child[j_g]) / S_d
                            grad_buf[i] = g * scale_g
                        for i in range(1, d_child):
                            key = grad_buf[i]
                            jj = i - 1
                            while jj >= 0 and grad_buf[jj] > key:
                                grad_buf[jj + 1] = grad_buf[jj]
                                jj -= 1
                            grad_buf[jj + 1] = key
                        cell_var = 0.0
                        for kk in range(d_child // 2):
                            cell_var += (grad_buf[d_child - 1 - kk]
                                         - grad_buf[kk])
                        cell_var /= (2.0 * S_d)

                        hi_idx = s_lo + ell - 1
                        N_W = prefix_nk[hi_idx] - prefix_nk[s_lo]
                        M_W = prefix_mk[hi_idx] - prefix_mk[s_lo]
                        cross_W = N_W - M_W
                        d_sq = np.int64(d_child) * np.int64(d_child)
                        compl_bound = d_sq - N_W
                        pb = min(cross_W, compl_bound)
                        if pb > 0:
                            qc_val = ((2.0 * d_d / ell_f)
                                      * np.float64(pb) * inv_4S2)
                        else:
                            qc_val = 0.0
                        net = margin - cell_var - qc_val
                        if net > best_net:
                            best_net = net
                            best_kill_ell = np.int32(ell)
                            best_kill_s = np.int32(s_lo)

            qc_ell = best_kill_ell
            qc_s = best_kill_s
            if best_net < local_min_net:
                local_min_net = best_net
        else:
            # --- Survivor: inline canonicalize + store ---
            use_rev = False
            half_d = d_child // 2
            for i in range(half_d):
                j_r = d_child - 1 - i
                if child[j_r] < child[i]:
                    use_rev = True
                    break
                elif child[j_r] > child[i]:
                    break
            if n_surv < max_survivors:
                if use_rev:
                    for i in range(d_child):
                        stage_buf[n_staged, i] = child[d_child - 1 - i]
                else:
                    for i in range(d_child):
                        stage_buf[n_staged, i] = child[i]
                n_staged += 1
                if n_staged == _STAGE_CAP:
                    flush_base = n_surv + 1 - _STAGE_CAP
                    for fi in range(_STAGE_CAP):
                        for ci_f in range(d_child):
                            out_buf[flush_base + fi, ci_f] = (
                                stage_buf[fi, ci_f])
                    n_staged = 0
            n_surv += 1

        # === GRAY CODE ADVANCE ===
        j = gc_focus[0]
        if j == n_active:
            break
        gc_focus[0] = 0

        pos = active_pos[j]
        gc_a[j] += gc_dir[j]
        cursor[pos] = lo_arr[pos] + gc_a[j]

        if gc_a[j] == 0 or gc_a[j] == radix[j] - 1:
            gc_dir[j] = -gc_dir[j]
            gc_focus[j] = gc_focus[j + 1]
            gc_focus[j + 1] = j + 1

        # === INCREMENTAL UPDATE (COARSE: child[2p+1] = parent[p] - cursor[p]) ===
        k1 = 2 * pos
        k2 = k1 + 1
        old1 = np.int32(child[k1])
        old2 = np.int32(child[k2])
        child[k1] = cursor[pos]
        child[k2] = parent_int[pos] - cursor[pos]
        new1 = np.int32(child[k1])
        new2 = np.int32(child[k2])
        delta1 = new1 - old1
        delta2 = new2 - old2

        raw_conv[2 * k1] += new1 * new1 - old1 * old1
        raw_conv[2 * k2] += new2 * new2 - old2 * old2
        raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)

        if use_sparse:
            if old1 != 0 and new1 == 0:
                p_nz = nz_pos[k1]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p_nz] = last
                nz_pos[last] = p_nz; nz_pos[k1] = -1
            elif old1 == 0 and new1 != 0:
                nz_list[nz_count] = k1; nz_pos[k1] = nz_count; nz_count += 1
            if old2 != 0 and new2 == 0:
                p_nz = nz_pos[k2]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p_nz] = last
                nz_pos[last] = p_nz; nz_pos[k2] = -1
            elif old2 == 0 and new2 != 0:
                nz_list[nz_count] = k2; nz_pos[k2] = nz_count; nz_count += 1
            for idx_nz in range(nz_count):
                jj = nz_list[idx_nz]
                if jj != k1 and jj != k2:
                    cj = np.int32(child[jj])
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
        else:
            for jj in range(k1):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
            for jj in range(k2 + 1, d_child):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj

        # === SUBTREE PRUNING (coarse: 1D threshold, no W_int) ===
        if j == J_MIN and n_active > J_MIN:
            fixed_parent_boundary = active_pos[J_MIN - 1]
            fixed_len = 2 * fixed_parent_boundary

            if fixed_len >= 4:
                partial_conv_len = 2 * fixed_len - 1
                for kk in range(partial_conv_len):
                    partial_conv[kk] = np.int32(0)
                for ii in range(fixed_len):
                    ci_s = np.int32(child[ii])
                    if ci_s != 0:
                        partial_conv[2 * ii] += ci_s * ci_s
                        for jj2 in range(ii + 1, fixed_len):
                            cj2 = np.int32(child[jj2])
                            if cj2 != 0:
                                partial_conv[ii + jj2] += (
                                    np.int32(2) * ci_s * cj2)
                for kk in range(1, partial_conv_len):
                    partial_conv[kk] += partial_conv[kk - 1]

                for kk in range(conv_len):
                    min_contrib[kk] = np.int64(0)

                # Inner active positions — COARSE child relationship
                for kk in range(J_MIN):
                    p_unf = active_pos[kk]
                    k1u = 2 * p_unf
                    k2u = 2 * p_unf + 1
                    ml = np.int64(lo_arr[p_unf])
                    P_unf = np.int64(parent_int[p_unf])
                    mh = P_unf - np.int64(hi_arr[p_unf])
                    if mh < 0:
                        mh = np.int64(0)
                    min_contrib[2 * k1u] += ml * ml
                    min_contrib[2 * k2u] += mh * mh
                    lo_val = np.int64(lo_arr[p_unf])
                    hi_val = np.int64(hi_arr[p_unf])
                    mut_lo = np.int64(2) * lo_val * (P_unf - lo_val)
                    mut_hi = np.int64(2) * hi_val * (P_unf - hi_val)
                    min_contrib[k1u + k2u] += (
                        mut_lo if mut_lo < mut_hi else mut_hi)
                    for ii in range(fixed_len):
                        ci64 = np.int64(child[ii])
                        if ci64 > 0:
                            if ml > 0:
                                min_contrib[ii + k1u] += (
                                    np.int64(2) * ci64 * ml)
                            if mh > 0:
                                min_contrib[ii + k2u] += (
                                    np.int64(2) * ci64 * mh)
                    for kk2 in range(kk + 1, J_MIN):
                        p2 = active_pos[kk2]
                        k1u2 = 2 * p2
                        k2u2 = 2 * p2 + 1
                        ml2 = np.int64(lo_arr[p2])
                        P2 = np.int64(parent_int[p2])
                        mh2 = P2 - np.int64(hi_arr[p2])
                        if mh2 < 0:
                            mh2 = np.int64(0)
                        if ml > 0 and ml2 > 0:
                            min_contrib[k1u + k1u2] += (
                                np.int64(2) * ml * ml2)
                        if ml > 0 and mh2 > 0:
                            min_contrib[k1u + k2u2] += (
                                np.int64(2) * ml * mh2)
                        if mh > 0 and ml2 > 0:
                            min_contrib[k2u + k1u2] += (
                                np.int64(2) * mh * ml2)
                        if mh > 0 and mh2 > 0:
                            min_contrib[k2u + k2u2] += (
                                np.int64(2) * mh * mh2)

                # Non-active unfixed parents — COARSE
                first_unfixed_parent = fixed_parent_boundary
                for pp in range(first_unfixed_parent, d_parent):
                    is_inner = False
                    for kk in range(J_MIN):
                        if active_pos[kk] == pp:
                            is_inner = True
                            break
                    if is_inner:
                        continue
                    k1na = 2 * pp
                    k2na = 2 * pp + 1
                    cv1 = np.int64(lo_arr[pp])
                    cv2 = np.int64(parent_int[pp]) - cv1
                    min_contrib[2 * k1na] += cv1 * cv1
                    min_contrib[2 * k2na] += cv2 * cv2
                    min_contrib[k1na + k2na] += np.int64(2) * cv1 * cv2
                    for ii in range(fixed_len):
                        ci64 = np.int64(child[ii])
                        if ci64 > 0:
                            if cv1 > 0:
                                min_contrib[ii + k1na] += (
                                    np.int64(2) * ci64 * cv1)
                            if cv2 > 0:
                                min_contrib[ii + k2na] += (
                                    np.int64(2) * ci64 * cv2)
                    for kk in range(J_MIN):
                        p_unf3 = active_pos[kk]
                        k1u3 = 2 * p_unf3
                        k2u3 = 2 * p_unf3 + 1
                        ml3 = np.int64(lo_arr[p_unf3])
                        P3 = np.int64(parent_int[p_unf3])
                        mh3 = P3 - np.int64(hi_arr[p_unf3])
                        if mh3 < 0:
                            mh3 = np.int64(0)
                        if cv1 > 0 and ml3 > 0:
                            min_contrib[k1na + k1u3] += (
                                np.int64(2) * cv1 * ml3)
                        if cv1 > 0 and mh3 > 0:
                            min_contrib[k1na + k2u3] += (
                                np.int64(2) * cv1 * mh3)
                        if cv2 > 0 and ml3 > 0:
                            min_contrib[k2na + k1u3] += (
                                np.int64(2) * cv2 * ml3)
                        if cv2 > 0 and mh3 > 0:
                            min_contrib[k2na + k2u3] += (
                                np.int64(2) * cv2 * mh3)
                    for pp2 in range(pp + 1, d_parent):
                        is_inner2 = False
                        for kk2b in range(J_MIN):
                            if active_pos[kk2b] == pp2:
                                is_inner2 = True
                                break
                        if is_inner2:
                            continue
                        k1na2 = 2 * pp2
                        k2na2 = 2 * pp2 + 1
                        cv12 = np.int64(lo_arr[pp2])
                        cv22 = np.int64(parent_int[pp2]) - cv12
                        if cv1 > 0 and cv12 > 0:
                            min_contrib[k1na + k1na2] += (
                                np.int64(2) * cv1 * cv12)
                        if cv1 > 0 and cv22 > 0:
                            min_contrib[k1na + k2na2] += (
                                np.int64(2) * cv1 * cv22)
                        if cv2 > 0 and cv12 > 0:
                            min_contrib[k2na + k1na2] += (
                                np.int64(2) * cv2 * cv12)
                        if cv2 > 0 and cv22 > 0:
                            min_contrib[k2na + k2na2] += (
                                np.int64(2) * cv2 * cv22)

                min_contrib_prefix[0] = min_contrib[0]
                for kk in range(1, conv_len):
                    min_contrib_prefix[kk] = (min_contrib_prefix[kk - 1]
                                              + min_contrib[kk])

                subtree_pruned = False
                for ell_oi in range(ell_count):
                    if subtree_pruned:
                        break
                    ell = ell_order[ell_oi]
                    n_cv = ell - 1
                    n_windows_total = conv_len - n_cv + 1
                    if n_windows_total <= 0:
                        continue
                    thr_val = thr_arr[ell]
                    for s_lo in range(n_windows_total):
                        s_hi = s_lo + n_cv - 1
                        ws_st = np.int64(0)
                        k_end_st = s_hi
                        if k_end_st >= partial_conv_len:
                            k_end_st = partial_conv_len - 1
                        if k_end_st >= s_lo:
                            ws_st = np.int64(partial_conv[k_end_st])
                            if s_lo > 0:
                                ws_st -= np.int64(
                                    partial_conv[s_lo - 1])
                        mc_sum = min_contrib_prefix[s_hi]
                        if s_lo > 0:
                            mc_sum -= min_contrib_prefix[s_lo - 1]
                        ws_st += mc_sum
                        if ws_st > thr_val:
                            subtree_pruned = True
                            break

                if subtree_pruned:
                    n_subtree_pruned += 1
                    next_focus = gc_focus[J_MIN]
                    for kk in range(J_MIN):
                        gc_a[kk] = 0
                        gc_dir[kk] = 1
                        gc_focus[kk] = kk
                    gc_focus[0] = next_focus
                    gc_focus[J_MIN] = J_MIN
                    for kk in range(J_MIN):
                        p_r = active_pos[kk]
                        cursor[p_r] = lo_arr[p_r]
                        child[2 * p_r] = lo_arr[p_r]
                        child[2 * p_r + 1] = (parent_int[p_r]
                                               - lo_arr[p_r])
                    for kk in range(conv_len):
                        raw_conv[kk] = np.int32(0)
                    for ii in range(d_child):
                        ci_r = np.int32(child[ii])
                        if ci_r != 0:
                            raw_conv[2 * ii] += ci_r * ci_r
                            for jj2 in range(ii + 1, d_child):
                                cj2 = np.int32(child[jj2])
                                if cj2 != 0:
                                    raw_conv[ii + jj2] += (
                                        np.int32(2) * ci_r * cj2)
                    if use_sparse:
                        nz_count = 0
                        for ii in range(d_child):
                            if child[ii] != 0:
                                nz_list[nz_count] = ii
                                nz_pos[ii] = nz_count
                                nz_count += 1
                            else:
                                nz_pos[ii] = -1
                    continue

    # --- Final flush ---
    if n_staged > 0:
        flush_base = min(n_surv, max_survivors) - n_staged
        for fi in range(n_staged):
            for ci_f in range(d_child):
                out_buf[flush_base + fi, ci_f] = stage_buf[fi, ci_f]

    return n_surv, n_subtree_pruned, local_min_net


def _compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child=None):
    """Compute per-bin lo/hi cursor ranges and total children count.

    Parameters
    ----------
    n_half_child : int or None
        Half-dimension of child.  When provided, uses the corrected
        correction term ``correction(m, n_half_child)``.  When None,
        falls back to the legacy flat correction for backward
        compatibility with external callers.

    Returns (lo_arr, hi_arr, total_children) or None if any bin has empty range.
    """
    d_parent = len(parent_int)
    # Theorem 1 x_cap: based on c_target directly (no correction).
    # A child bin with value v has self-conv v^2.  Pruned when
    # v^2 > c_target * 4n * m^2 * 2, i.e. v > m*sqrt(8n*c_target).
    # +1 for rounding safety (canonical adjustment).
    x_cap = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    x_cap = max(x_cap, 0)

    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total_children = 1
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, 2 * b_i - x_cap)
        hi = min(2 * b_i, x_cap)
        if lo > hi:
            return None
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_children *= (hi - lo + 1)

    return lo_arr, hi_arr, total_children


@njit(cache=True)
def _fix_conv_min_mutual_cross(conv_min, child_min, lo_arr, hi_arr,
                                parent_int, d_parent):
    """Tighten conv_min by correcting within-parent mutual terms and
    cross-parent middle terms from loose independent-min to tight
    4-corner bounds.

    Within-parent mutual: conv[4q+1] has 2*lo_q*(2P_q - hi_q) from
    child_min, but the tight minimum of 2*x*(2P-x) over [lo,hi] is
    min(2*lo*(2P-lo), 2*hi*(2P-hi)).

    Cross-parent middle: conv[2i+2j+1] has the sum of per-term mins
    from child_min, but the tight value is the 4-corner minimum of
    f(xi,xj) = 2*xi*(2Pj-xj) + 2*(2Pi-xi)*xj.

    Modifies conv_min in place.
    """
    # --- Fix 1: within-parent mutual terms ---
    for q in range(d_parent):
        k_mut = 4 * q + 1  # conv index of mutual term for parent q
        # Current value from child_min: 2*lo*(2P-hi) [too low]
        old_mut = (np.int64(2) * np.int64(child_min[2 * q])
                   * np.int64(child_min[2 * q + 1]))
        lo_q = np.int64(lo_arr[q])
        hi_q = np.int64(hi_arr[q])
        P2_q = np.int64(2) * np.int64(parent_int[q])
        tight_a = np.int64(2) * lo_q * (P2_q - lo_q)
        tight_b = np.int64(2) * hi_q * (P2_q - hi_q)
        if tight_a < tight_b:
            tight_mut = tight_a
        else:
            tight_mut = tight_b
        conv_min[k_mut] += tight_mut - old_mut

    # --- Fix 2: cross-parent middle terms (4-corner bound) ---
    for i in range(d_parent):
        lo_i = np.int64(lo_arr[i])
        hi_i = np.int64(hi_arr[i])
        P2_i = np.int64(2) * np.int64(parent_int[i])
        for j in range(i + 1, d_parent):
            k_mid = 2 * i + 2 * j + 1
            lo_j = np.int64(lo_arr[j])
            hi_j = np.int64(hi_arr[j])
            P2_j = np.int64(2) * np.int64(parent_int[j])
            # Current value from child_min (sum of per-term mins):
            old_cross = (np.int64(2) * np.int64(child_min[2 * i])
                         * np.int64(child_min[2 * j + 1])
                         + np.int64(2) * np.int64(child_min[2 * i + 1])
                         * np.int64(child_min[2 * j]))
            # 4-corner tight minimum:
            # f(xi,xj) = 2*xi*(2Pj-xj) + 2*(2Pi-xi)*xj
            tight_cross = np.int64(1152921504606846976)  # 2^60, large sentinel
            for ci in range(2):
                xi = lo_i if ci == 0 else hi_i
                for cj in range(2):
                    xj = lo_j if cj == 0 else hi_j
                    val = (np.int64(2) * xi * (P2_j - xj)
                           + np.int64(2) * (P2_i - xi) * xj)
                    if val < tight_cross:
                        tight_cross = val
            conv_min[k_mid] += tight_cross - old_cross


@njit(cache=True)
def _tighten_ranges(parent_int, lo_arr, hi_arr, m, c_target, n_half_child,
                     use_flat_threshold=False):
    """Arc consistency: tighten cursor ranges by removing provably-infeasible
    edge values.

    For each position p and each edge value v, computes a lower bound on the
    window sum when all other positions take their minimum-contribution values
    (full autoconvolution including cross-terms).  If this lower bound exceeds
    the pruning threshold for ANY window, v cannot produce a surviving child
    and is removed from the range.

    Soundness: uses W_int_max (maximum possible mass in window) for the
    threshold, which is the highest (hardest-to-exceed) threshold.  Since
    min_ws > threshold(W_int_max) >= threshold(actual_W_int), any child with
    cursor[p] = v will be pruned by the kernel.

    Optimization: before building the expensive O(ell_count * S_child)
    threshold table, a cheap pre-check tests only the extreme values (lo, hi)
    of each position against the minimum possible threshold (W_int=0).
    If no extreme exceeds even this easiest-to-exceed threshold, AC cannot
    tighten any range and we return immediately.  This avoids the table
    construction cost (dominant at large d_child) for the common case where
    AC finds nothing.

    Correctness of pre-check: min_threshold(ell) uses W_int=0 (corr=1.0),
    giving the smallest threshold.  actual_threshold >= min_threshold always.
    So ws <= min_threshold => ws <= actual_threshold => value is NOT
    infeasible => AC won't tighten from that end.  If all extremes pass,
    round 1 changes nothing, so no further rounds are needed.

    Modifies lo_arr and hi_arr in place.
    Returns the new total_children (product of range sizes), 0 if any range
    becomes empty.
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    # --- Threshold constants (must match _fused_generate_and_prune_gray) ---
    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half_child)
    n_half_d = np.float64(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    S_child_plus_1 = int(4 * n_half_child * m + 1)
    ell_count = conv_len

    # =================================================================
    # PHASE 0: Cheap pre-check — can AC possibly help this parent?
    #
    # Uses min-threshold (W_int=0, one value per ell) instead of the
    # full threshold table (ell_count * S_child_plus_1 entries).
    # Tests ONLY the two extreme values (lo, hi) per position.
    # If no extreme of any position exceeds min-threshold for any
    # window, AC cannot tighten and we skip entirely.
    # =================================================================

    # Check if any position has range > 1
    any_free = False
    for i in range(d_parent):
        if lo_arr[i] != hi_arr[i]:
            any_free = True
            break
    if not any_free:
        total = np.int64(1)
        for i in range(d_parent):
            total *= np.int64(hi_arr[i] - lo_arr[i] + 1)
        return total

    # Build child_min: each bin at its independent minimum
    child_min = np.empty(d_child, dtype=np.int32)
    for q in range(d_parent):
        child_min[2 * q] = lo_arr[q]
        child_min[2 * q + 1] = 2 * parent_int[q] - hi_arr[q]

    # Full autoconvolution of child_min
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

    # Tighten conv_min: fix within-parent mutual and cross-parent middle terms
    _fix_conv_min_mutual_cross(conv_min, child_min, lo_arr, hi_arr,
                               parent_int, d_parent)

    # Min-threshold per ell.
    # Flat: correction = (2m+1)/m^2, threshold is W_int-independent.
    # Theorem 1 + box cert (non-flat): correction = n^2*(8m+1)/2 (constant).
    # C&S flat: correction = (2m+1)*4n*ell (grows with ell).
    min_thresh = np.empty(ell_count, dtype=np.int64)
    if use_flat_threshold:
        for ell in range(2, 2 * d_child + 1):
            scale_ell = np.float64(ell) * four_n
            dyn_x = (cs_base_m2 + 2.0 * m_d + 1.0 + eps_margin) * scale_ell
            min_thresh[ell - 2] = np.int64(dyn_x)
    else:
        # Per-window proven bound: min(n, ell-1, 2d-ell) * B
        B_corr = n_half_d * (8.0 * m_d + 1.0) / 2.0
        n_int = int(n_half_child)
        for ell in range(2, 2 * d_child + 1):
            scale_ell = np.float64(ell) * four_n
            mult = ell - 1
            comp = 2 * d_child - ell
            if comp < mult:
                mult = comp
            if n_int < mult:
                mult = n_int
            min_thresh[ell - 2] = np.int64(
                cs_base_m2 * scale_ell + np.float64(mult) * B_corr)

    # Test extreme values of each free position against min-threshold
    test_conv = np.empty(conv_len, dtype=np.int64)
    any_might_tighten = False

    for p in range(d_parent):
        if any_might_tighten:
            break
        if lo_arr[p] == hi_arr[p]:
            continue

        B_p = parent_int[p]
        k1 = 2 * p
        k2 = 2 * p + 1
        old1 = np.int64(child_min[k1])
        old2 = np.int64(child_min[k2])

        # Test v=lo (lo-end tightening check: child[2p]=lo, child[2p+1]=max)
        # and v=hi (hi-end tightening check: child[2p]=max, child[2p+1]=min)
        for vi in range(2):
            if any_might_tighten:
                break
            if vi == 0:
                v = lo_arr[p]
            else:
                v = hi_arr[p]
            new1 = np.int64(v)
            new2 = np.int64(2 * B_p - v)
            delta1 = new1 - old1
            delta2 = new2 - old2

            # Build test_conv = conv_min + deltas for position p
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

            # Check all windows against min-threshold
            for ell in range(2, 2 * d_child + 1):
                if any_might_tighten:
                    break
                n_cv = ell - 1
                n_windows = conv_len - n_cv + 1
                if n_windows <= 0:
                    continue
                thresh_val = min_thresh[ell - 2]

                ws = np.int64(0)
                for kk in range(n_cv):
                    ws += test_conv[kk]
                if ws > thresh_val:
                    any_might_tighten = True
                    break

                for s in range(1, n_windows):
                    ws += test_conv[s + n_cv - 1] - test_conv[s - 1]
                    if ws > thresh_val:
                        any_might_tighten = True
                        break

    if not any_might_tighten:
        # No extreme value of any position exceeds even the minimum
        # threshold — AC cannot tighten any range.
        total = np.int64(1)
        for i in range(d_parent):
            r = hi_arr[i] - lo_arr[i] + 1
            if r <= 0:
                return 0
            total *= np.int64(r)
        return total

    # =================================================================
    # PHASE 1: Full AC — pre-check found potential, run proper tightening
    # =================================================================

    # Build full threshold table (W_int-dependent)
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
            # Per-window proven bound: min(n, ell-1, 2d-ell) * B
            B_corr = n_half_d * (8.0 * m_d + 1.0) / 2.0
            n_int = int(n_half_child)
            mult = ell - 1
            comp = 2 * d_child - ell
            if comp < mult:
                mult = comp
            if n_int < mult:
                mult = n_int
            th1_val = np.int64(
                cs_base_m2 * scale_ell + np.float64(mult) * B_corr)
            for w in range(S_child_plus_1):
                threshold_table[idx * S_child_plus_1 + w] = th1_val

    max_rounds = d_parent + 1
    for _round in range(max_rounds):
        any_changed = False

        # Rebuild child_min (ranges may have changed in prior rounds)
        for q in range(d_parent):
            child_min[2 * q] = lo_arr[q]
            child_min[2 * q + 1] = 2 * parent_int[q] - hi_arr[q]

        # Rebuild conv_min
        for kk in range(conv_len):
            conv_min[kk] = np.int64(0)
        for i in range(d_child):
            ci = np.int64(child_min[i])
            if ci == 0:
                continue
            conv_min[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int64(child_min[j])
                if cj != 0:
                    conv_min[i + j] += np.int64(2) * ci * cj

        # Tighten conv_min: fix within-parent mutual and cross-parent middle terms
        _fix_conv_min_mutual_cross(conv_min, child_min, lo_arr, hi_arr,
                                   parent_int, d_parent)

        # Max child prefix sum for W_int_max queries
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

                # test_conv = conv_min + incremental deltas for position p
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

                # Check all windows via sliding window
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


# =====================================================================
# Mass-based grid + palindrome symmetry (MATLAB-style C&S algorithm)
#
# Fix A: Only d_parent/2 free cursor variables (palindrome symmetry).
# Fix B: Mass-based split: child[2i]+child[2i+1] = parent[i] (not 2*parent[i]).
#        S = 2*m (constant across all levels).
#
# Threshold formula (mass-conv space):
#   TV = n_half_current * ws_mass / (m^2 * ell)
#   Prune if ws_mass > floor((c_target*m^2 + corr_int + eps) * ell / n_half_current)
#   where corr_int = 2*m + 1 (flat C&S correction with epsilon=1/m).
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_mass_flat(batch_int, n_half_current, m, c_target):
    """Mass-based flat threshold pruning for L0.

    batch_int: (B, d) int32, palindromic mass vectors summing to 2*m.
    n_half_current: d/2 at the current level.

    Threshold: ws_mass > floor((c_target*m^2 + 2m+1 + eps) * ell / n_half_current)
    Returns boolean mask: True = survived.
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    n_d = np.float64(n_half_current)
    eps_margin = 1e-9
    base = c_target * m_d * m_d + 2.0 * m_d + 1.0 + eps_margin

    max_ell = 2 * d
    threshold_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        threshold_arr[ell] = np.int64(base * np.float64(ell) / n_d)

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            dyn_it = threshold_arr[ell]
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(
                        conv[s_lo - 1])
                if ws > dyn_it:
                    pruned = True
                    break

        if pruned:
            survived[b] = False

    return survived


@njit(cache=True)
def _fused_mass_palindrome(parent_int, n_half_child, m, c_target, out_buf):
    """Mass-based palindrome cascade kernel.

    Enumerates palindromic children via mass-based split.
    Only n_half_parent = d_parent/2 free cursor variables.

    parent_int: (d_parent,) palindromic int32 mass vector (sum = 2*m).
    n_half_child: d_child/2 = d_parent (n_half at the child level).
    out_buf: (max_survivors, d_child) int32 pre-allocated.

    Returns (n_survivors, total_children_tested).
    """
    d_parent = parent_int.shape[0]
    n_half_parent = d_parent // 2
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    m_d = np.float64(m)
    n_d = np.float64(n_half_child)
    eps_margin = 1e-9
    base = c_target * m_d * m_d + 2.0 * m_d + 1.0 + eps_margin

    max_ell = 2 * d_child
    threshold_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        threshold_arr[ell] = np.int64(base * np.float64(ell) / n_d)

    # x_cap: max child bin mass (from ell=2 single-bin energy bound)
    thresh_phys = c_target + 2.0 / m_d + 1.0 / (m_d * m_d) + eps_margin
    x_cap = np.int32(m_d * math.sqrt(2.0 * thresh_phys / n_d))

    # Cursor ranges for free variables (first half of parent)
    lo_arr = np.empty(n_half_parent, dtype=np.int32)
    hi_arr = np.empty(n_half_parent, dtype=np.int32)
    total_product = np.int64(1)
    for i in range(n_half_parent):
        p = parent_int[i]
        lo = np.int32(0)
        hi = p
        if p > x_cap:
            lo = p - x_cap
        if hi > x_cap:
            hi = x_cap
        if lo > hi:
            return 0, 0
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_product *= np.int64(hi - lo + 1)

    max_survivors = out_buf.shape[0]
    n_surv = 0
    child = np.empty(d_child, dtype=np.int32)
    conv = np.empty(conv_len, dtype=np.int32)
    cursor = np.empty(n_half_parent, dtype=np.int32)
    for i in range(n_half_parent):
        cursor[i] = lo_arr[i]

    n_tested = np.int64(0)

    while True:
        # Build palindromic child from cursor
        for i in range(n_half_parent):
            j = d_parent - 1 - i
            c_val = cursor[i]
            p_val = parent_int[i]
            child[2 * i] = c_val
            child[2 * i + 1] = p_val - c_val
            child[2 * j] = p_val - c_val
            child[2 * j + 1] = c_val

        n_tested += 1

        # Full autoconvolution (O(d_child^2))
        for k in range(conv_len):
            conv[k] = np.int32(0)
        for ii in range(d_child):
            ci = np.int32(child[ii])
            if ci != 0:
                conv[2 * ii] += ci * ci
                for jj in range(ii + 1, d_child):
                    cj = np.int32(child[jj])
                    if cj != 0:
                        conv[ii + jj] += np.int32(2) * ci * cj

        # Window scan with flat threshold
        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            dyn_it = threshold_arr[ell]
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(
                        conv[s_lo - 1])
                if ws > dyn_it:
                    pruned = True
                    break

        if not pruned:
            if n_surv < max_survivors:
                for ii in range(d_child):
                    out_buf[n_surv, ii] = child[ii]
            n_surv += 1

        # Advance odometer (rightmost = fastest)
        carry = n_half_parent - 1
        while carry >= 0:
            cursor[carry] += 1
            if cursor[carry] <= hi_arr[carry]:
                break
            cursor[carry] = lo_arr[carry]
            carry -= 1

        if carry < 0:
            break

    return n_surv, n_tested


def run_level0_mass(n_half, m, c_target, verbose=True):
    """L0 with mass-based grid: palindromic compositions summing to 2*m.

    Generates compositions of n_half elements summing to m (half-domain),
    mirrors to create palindromic 2*n_half element vectors (full domain).
    """
    d = 2 * n_half
    S_half = m
    n_total_half = count_compositions(n_half, S_half)
    corr = correction(m)

    if verbose:
        _log(f"\n[L0 mass-grid] d={d}, m={m}, S=2m={2*m}, "
             f"palindromic comps={n_total_half:,}")
        _log(f"     correction(1/m)={corr:.6f}, "
             f"threshold={c_target+corr:.6f}")

    t0 = time.time()
    all_survivors = []
    n_pruned = 0
    n_processed = 0
    last_report = t0

    for half_batch in generate_compositions_batched(n_half, S_half,
                                                     batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]

        n_processed += len(batch)

        survived_mask = _prune_mass_flat(batch, n_half, m, c_target)
        n_pruned += int(np.sum(~survived_mask))

        survivors = batch[survived_mask]
        if len(survivors) > 0:
            all_survivors.append(survivors)

        now = time.time()
        if verbose and (now - last_report >= 2.0):
            pct = n_processed / n_total_half * 100
            n_surv_so_far = sum(len(s) for s in all_survivors)
            _log(f"     [L0] {n_processed:,}/{n_total_half:,} ({pct:.1f}%) "
                 f"| {n_surv_so_far:,} survivors")
            last_report = now

    elapsed = time.time() - t0

    if all_survivors:
        all_survivors = np.vstack(all_survivors)
    else:
        all_survivors = np.empty((0, d), dtype=np.int32)

    n_survivors = len(all_survivors)
    proven = n_survivors == 0

    if verbose:
        _log(f"     {elapsed:.2f}s: {n_processed:,} palindromic compositions")
        _log(f"     pruned: {n_pruned:,}, survivors: {n_survivors:,}")
        if proven:
            _log(f"     PROVEN at L0!")

    return {
        'survivors': all_survivors,
        'n_survivors': n_survivors,
        'n_pruned_asym': 0,
        'n_pruned_test': n_pruned,
        'n_processed': n_processed,
        'elapsed': elapsed,
        'proven': proven,
    }


def process_parent_mass(parent_int, m, c_target, n_half_child, buf_cap=None):
    """Process one palindromic parent with mass-based split + palindrome.

    Returns (survivors, total_children_tested).
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    if buf_cap is None:
        buf_cap = 100_000
    out_buf = np.empty((buf_cap, d_child), dtype=np.int32)

    n_surv, n_tested = _fused_mass_palindrome(
        parent_int, n_half_child, m, c_target, out_buf)

    if n_surv > buf_cap:
        out_buf = np.empty((n_surv, d_child), dtype=np.int32)
        n2, _ = _fused_mass_palindrome(
            parent_int, n_half_child, m, c_target, out_buf)
        assert n2 == n_surv
        n_surv = n2

    return out_buf[:n_surv].copy(), n_tested


def _default_buf_cap(d_child):
    """Default survivor buffer capacity, scaled by dimension.

    Benchmark (Section 4): 100K is the smallest cap with zero re-runs
    at L3 (d=32, max survivor = 78,262). Using 5M wastes 640MB/worker
    for no benefit. At L4+ survival rate is near zero so 100K is safe.
    """
    if d_child <= 16:
        return 1_000_000    # 1M rows; ~64 MB (was 10M / 640MB)
    elif d_child <= 32:
        return 200_000      # 200K rows; ~25 MB (was 5M / 640MB)
    else:
        return 100_000      # 100K rows; ~25 MB at d=64


def process_parent_fused(parent_int, m, c_target, n_half_child, buf_cap=None,
                          use_flat_threshold=False, use_F=False,
                          skip_sdp_cert=False,
                          use_Q=False, use_L=False):
    """Wrapper: compute x_cap, allocate buffer, call fused kernel.

    Parameters
    ----------
    buf_cap : int or None
        Max rows for the output buffer.  *None* → ``_default_buf_cap(d_child)``.
        If the kernel reports more survivors than fit, the buffer is
        re-allocated at the exact size and the kernel re-run.
    use_flat_threshold : bool
        When True, uses flat C&S Lemma 3 correction for Lean axiom soundness.
    use_F : bool
        When True, after the fused W-refined kernel returns survivors,
        apply variant F (LP-tight linear correction Δ_BB + ell_int_sum)
        as an additional filter.  F is a strictly tighter rule than
        W-refined, so any F-survivor is also a W-survivor.  Empirically
        prunes 25-65% additional survivors at L0 and beyond.
    skip_sdp_cert : bool
        When True, skip the CVXPY-based SDP parent cert (which has been
        observed to clear <1% of parents at typical configs while costing
        50-500ms per call).  Theorem-1 interval cert and LP dual cert
        still run.  Soundness preserved (we just prune fewer parents
        before enumeration).

    Returns
    -------
    survivors : (K, d_child) int32 array
    total_children : int  (total Cartesian product size, for stats)
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    result = _compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child)
    if result is None:
        return np.empty((0, d_child), dtype=np.int32), 0
    lo_arr, hi_arr, total_children = result

    if total_children == 0:
        return np.empty((0, d_child), dtype=np.int32), 0

    # Arc consistency: tighten cursor ranges before enumeration
    total_children = _tighten_ranges(parent_int, lo_arr, hi_arr,
                                     m, c_target, n_half_child,
                                     use_flat_threshold)
    if total_children == 0:
        return np.empty((0, d_child), dtype=np.int32), 0

    # Whole-parent pre-pruning: if ALL children are provably pruned by
    # Theorem 1 interval arithmetic, skip enumeration entirely.
    # Only when NOT using flat threshold (Theorem 1 is tighter than flat).
    if not use_flat_threshold:
        from cascade_opts import (_whole_parent_prune_theorem1,
                                  lp_dual_certificate, sdp_certify_parent)
        if _whole_parent_prune_theorem1(parent_int, lo_arr, hi_arr,
                                         int(n_half_child), int(m),
                                         c_target):
            return np.empty((0, d_child), dtype=np.int32), total_children

        # LP dual certificate: find window weights proving all children pruned.
        if d_parent <= 10 and lp_dual_certificate(
                parent_int, lo_arr, hi_arr,
                int(n_half_child), int(m), c_target):
            return np.empty((0, d_child), dtype=np.int32), total_children

        # SDP relaxation: tests global feasibility of survivor region.
        # Handles cases where LP fails (balanced children) by exploiting
        # joint quadratic structure across ALL windows simultaneously.
        if (not skip_sdp_cert) and d_parent <= 12 and sdp_certify_parent(
                parent_int, lo_arr, hi_arr,
                int(n_half_child), int(m), c_target):
            return np.empty((0, d_child), dtype=np.int32), total_children

    if buf_cap is None:
        buf_cap = _default_buf_cap(d_child)
    max_buf = min(total_children, buf_cap)
    out_buf = np.empty((max_buf, d_child), dtype=np.int32)

    _kernel = _fused_generate_and_prune_gray

    n_survivors, _ = _kernel(
        parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf,
        use_flat_threshold)

    if n_survivors > max_buf:
        # Overflow: re-allocate exact-size buffer and re-run.
        max_buf = n_survivors
        out_buf = np.empty((max_buf, d_child), dtype=np.int32)
        n2, _ = _kernel(
            parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf,
            use_flat_threshold)
        assert n2 == n_survivors, (
            f"Non-deterministic kernel: first run {n_survivors}, "
            f"retry {n2}")
        n_survivors = n2

    # Per-composition post-filter chain F → Q → L.  Each is a strict
    # subset of the previous; soundness is by construction.
    survivors_view = out_buf[:n_survivors]
    if use_F and n_survivors > 0 and not use_flat_threshold:
        f_mask = _prune_dynamic(survivors_view, n_half_child, m, c_target,
                                 use_flat_threshold=False, use_F=True)
        survivors_view = survivors_view[f_mask]
    if (use_Q or use_L) and len(survivors_view) > 0 and not use_flat_threshold:
        from post_filters import apply_post_filter_chain
        survivors_view = apply_post_filter_chain(
            survivors_view, n_half_child, m, c_target,
            use_Q=use_Q, use_L=use_L)
    return survivors_view.copy(), total_children


# =====================================================================
# Coarse-grid parent processing
# =====================================================================

def _compute_bin_ranges_coarse(parent_int, S, c_target, d_child):
    """Coarse-grid cursor ranges.  child[2i]+child[2i+1] = parent[i]."""
    d_parent = len(parent_int)
    x_cap = int(math.floor(S * math.sqrt(c_target / d_child)))

    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total_children = 1
    for i in range(d_parent):
        p = int(parent_int[i])
        lo = max(0, p - x_cap)
        hi = min(p, x_cap)
        if lo > hi:
            return None
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_children *= (hi - lo + 1)
    return lo_arr, hi_arr, total_children


def process_parent_coarse(parent_int, S, c_target, d_child, buf_cap=None):
    """Process one parent in coarse-grid mode.

    Returns (survivors, total_children, min_cert_net).
    """
    d_parent = len(parent_int)

    result = _compute_bin_ranges_coarse(parent_int, S, c_target, d_child)
    if result is None:
        return np.empty((0, d_child), dtype=np.int32), 0, 1e30
    lo_arr, hi_arr, total_children = result

    if total_children == 0:
        return np.empty((0, d_child), dtype=np.int32), 0, 1e30

    prefix_nk, prefix_mk = _build_pair_prefix(d_child)

    if buf_cap is None:
        buf_cap = _default_buf_cap(d_child)
    max_buf = min(total_children, buf_cap)
    out_buf = np.empty((max_buf, d_child), dtype=np.int32)

    n_survivors, _, min_net = _fused_coarse_gray(
        parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf,
        prefix_nk, prefix_mk)

    if n_survivors > max_buf:
        max_buf = n_survivors
        out_buf = np.empty((max_buf, d_child), dtype=np.int32)
        n2, _, min_net = _fused_coarse_gray(
            parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf,
            prefix_nk, prefix_mk)
        assert n2 == n_survivors
        n_survivors = n2

    return out_buf[:n_survivors].copy(), total_children, min_net


def process_parent_verbose(parent_int, m, c_target, n_half_child,
                            parent_idx, n_parents,
                            use_flat_threshold=False, use_F=False,
                            skip_sdp_cert=False,
                            use_Q=False, use_L=False):
    """Like process_parent_fused but logs intra-parent progress.

    Splits the Cartesian product along cursor[0]'s range so we can
    log between slices.  Falls back to single-shot for small parents.

    See process_parent_fused for use_F documentation.

    Returns
    -------
    survivors : (K, d_child) int32 array
    total_children : int
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    label = f"parent {parent_idx+1}/{n_parents}"

    result = _compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child)
    if result is None:
        _log(f"       {label}: empty range, skipped")
        return np.empty((0, d_child), dtype=np.int32), 0
    lo_arr, hi_arr, total_children = result

    if total_children == 0:
        _log(f"       {label}: 0 children")
        return np.empty((0, d_child), dtype=np.int32), 0

    n_slices = int(hi_arr[0]) - int(lo_arr[0]) + 1

    # Small parent or only 1 slice → single-shot
    if total_children < 500_000 or n_slices <= 1:
        _log(f"       {label}: {total_children:,} children (single pass)...")
        surv, tc = process_parent_fused(parent_int, m, c_target, n_half_child,
                                         use_flat_threshold=use_flat_threshold,
                                         use_F=use_F,
                                         skip_sdp_cert=skip_sdp_cert,
                                         use_Q=use_Q, use_L=use_L)
        _log(f"       {label}: done, {len(surv):,} survivors")
        return surv, tc

    # Arc consistency: tighten ranges (also done inside process_parent_fused,
    # but we need it here for the slice path)
    total_children = _tighten_ranges(parent_int, lo_arr, hi_arr,
                                     m, c_target, n_half_child,
                                     use_flat_threshold)
    if total_children == 0:
        _log(f"       {label}: tightened to empty")
        return np.empty((0, d_child), dtype=np.int32), 0

    # Whole-parent pre-pruning (Idea 4)
    if not use_flat_threshold:
        from cascade_opts import _whole_parent_prune_theorem1
        if _whole_parent_prune_theorem1(parent_int, lo_arr, hi_arr,
                                         int(n_half_child), int(m),
                                         c_target):
            _log(f"       {label}: whole-parent pruned by Theorem 1")
            return np.empty((0, d_child), dtype=np.int32), total_children

    n_slices = int(hi_arr[0]) - int(lo_arr[0]) + 1

    # Large parent → split by cursor[0] value for progress
    _log(f"       {label}: {total_children:,} children, "
         f"{n_slices} slices on bin[0]")

    children_per_slice = total_children // n_slices
    all_survivors = []
    total_survived = 0
    t_start = time.time()

    slice_lo = lo_arr.copy()
    slice_hi = hi_arr.copy()

    for si, v0 in enumerate(range(int(lo_arr[0]), int(hi_arr[0]) + 1)):
        slice_lo[0] = np.int32(v0)
        slice_hi[0] = np.int32(v0)

        slice_buf_cap = _default_buf_cap(d_child)
        max_buf = min(children_per_slice, slice_buf_cap)
        out_buf = np.empty((max_buf, d_child), dtype=np.int32)

        _kernel = _fused_generate_and_prune_gray

        n_surv, _ = _kernel(
            parent_int, n_half_child, m, c_target,
            slice_lo, slice_hi, out_buf, use_flat_threshold)

        if n_surv > max_buf:
            # Overflow: re-allocate and re-run
            max_buf = n_surv
            out_buf = np.empty((max_buf, d_child), dtype=np.int32)
            n2, _ = _kernel(
                parent_int, n_half_child, m, c_target,
                slice_lo, slice_hi, out_buf, use_flat_threshold)
            assert n2 == n_surv, (
                f"Non-deterministic kernel: first run {n_surv}, retry {n2}")
            n_surv = n2
        if n_surv > 0:
            all_survivors.append(out_buf[:n_surv].copy())
            total_survived += n_surv

        # Log every slice, or at least every 5 seconds
        done_slices = si + 1
        elapsed = time.time() - t_start
        if done_slices == n_slices or done_slices % max(1, n_slices // 20) == 0:
            rate = done_slices / elapsed if elapsed > 0 else 0
            eta = (n_slices - done_slices) / rate if rate > 0 else 0
            pct = done_slices / n_slices * 100
            _log(f"       {label}: slice {done_slices}/{n_slices} "
                 f"({pct:.0f}%) {total_survived:,} surv, "
                 f"ETA {_fmt_time(eta)}")

    if all_survivors:
        survivors = np.vstack(all_survivors)
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)

    # F → Q → L post-filter chain (see process_parent_fused).
    if use_F and len(survivors) > 0 and not use_flat_threshold:
        f_mask = _prune_dynamic(survivors, n_half_child, m, c_target,
                                 use_flat_threshold=False, use_F=True)
        survivors = survivors[f_mask]
    if (use_Q or use_L) and len(survivors) > 0 and not use_flat_threshold:
        from post_filters import apply_post_filter_chain
        survivors = apply_post_filter_chain(
            survivors, n_half_child, m, c_target,
            use_Q=use_Q, use_L=use_L)

    elapsed = time.time() - t_start
    _log(f"       {label}: done in {_fmt_time(elapsed)}, "
         f"{len(survivors):,} survivors")
    return survivors, total_children


# =====================================================================
# JIT warmup
# =====================================================================

def _warmup_jit():
    """Warm up Numba JIT for common array dimensions at import time."""
    for d in (4, 8):
        dummy = np.zeros((1, d), dtype=np.int32)
        _prune_dynamic_int32(dummy, d // 2, 20, 1.3)
        _prune_dynamic_int64(dummy, d // 2, 300, 1.3)
        _canonical_mask(dummy)
        _canonicalize_inplace(dummy.copy())
    # Warm up sorted merge kernel
    _dm = np.zeros((1, 4), dtype=np.int32)
    _sorted_merge_dedup_kernel(_dm, _dm, np.zeros((2, 4), dtype=np.int32))

_warmup_jit()


# =====================================================================
# Level 0: generate all compositions, prune, collect survivors
# =====================================================================

def run_level0(n_half, m, c_target, verbose=True, use_flat_threshold=False,
               d0=None, use_bnb=True, coarse_S=None, use_F=False,
               skip_sdp_cert=False, use_Q=False, use_L=False):
    """Run Level 0: enumerate compositions, prune, collect survivors.

    Parameters
    ----------
    n_half : int or float
        Half-dimension.  For even d: integer, d = 2*n_half.
        For odd d: pass d0 instead, n_half is computed as d0/2.
    coarse_S : int or None
        When set, use coarse-grid mode (Theorem 1, no correction).
        S is the constant total mass sum.
    d0 : int, optional
        Starting dimension (number of bins).  Overrides n_half if given.
        Allows odd starting dimensions like d=3 (C&S original).
    use_bnb : bool
        When True (default), use branch-and-bound L0 that prunes during
        composition enumeration.  Falls back to batch generation if d > 8.

    Returns
    -------
    dict with: survivors (N, d) int32, n_survivors, n_pruned_asym,
               n_pruned_test, elapsed, proven
    """
    if d0 is not None:
        d = d0
        n_half = d / 2.0  # float for odd d
    else:
        d = 2 * n_half

    # === COARSE-GRID L0 PATH ===
    if coarse_S is not None:
        S = coarse_S
        n_total = count_compositions(d, S)
        if verbose:
            x_cap_c = int(math.floor(S * math.sqrt(c_target / d)))
            _log(f"\n[L0 coarse] d={d}, S={S}, c_target={c_target}")
            _log(f"     x_cap={x_cap_c}, "
                 f"compositions={n_total:,}")
        t0 = time.time()
        all_survivors = []
        n_pruned_asym = 0
        n_pruned_test = 0
        n_processed = 0
        global_min_net = 1e30
        last_report = t0

        gen = generate_canonical_compositions_batched(d, S, batch_size=500_000)
        for batch in gen:
            n_processed += len(batch)

            # Asymmetry pruning (coarse)
            threshold_a = asymmetry_threshold(c_target)
            left_bins = d // 2
            left = batch[:, :left_bins].sum(axis=1).astype(np.float64)
            left_frac = left / float(S)
            asym_mask = ((left_frac > 1 - threshold_a)
                         & (left_frac < threshold_a))
            n_pruned_asym += int(np.sum(~asym_mask))
            batch = batch[asym_mask]
            if len(batch) == 0:
                continue

            survived_mask, min_net = _prune_coarse(
                batch, d, S, c_target)
            n_pruned_test += int(np.sum(~survived_mask))
            if min_net < global_min_net:
                global_min_net = min_net
            survivors = batch[survived_mask]
            if len(survivors) > 0:
                all_survivors.append(survivors)

            now = time.time()
            if verbose and (now - last_report >= 2.0):
                n_surv = sum(len(s) for s in all_survivors)
                _log(f"     [{n_processed:,} processed] "
                     f"{n_surv:,} survivors")
                last_report = now

        elapsed = time.time() - t0
        if all_survivors:
            all_survivors = np.vstack(all_survivors)
        else:
            all_survivors = np.empty((0, d), dtype=np.int32)
        n_survivors = len(all_survivors)
        proven = n_survivors == 0
        box_ok = global_min_net >= 0.0

        if verbose:
            _log(f"     {elapsed:.2f}s: pruned asym={n_pruned_asym:,} "
                 f"test={n_pruned_test:,} survivors={n_survivors:,}")
            if n_pruned_test > 0:
                _log(f"     min(margin - cell_var - quad_corr) = "
                     f"{global_min_net:.6f}")
                _log(f"     BOX CERT: "
                     f"{'PASS' if box_ok else 'FAIL (increase S)'}")
            if proven:
                _log(f"     PROVEN at L0!")

        return {
            'survivors': all_survivors,
            'n_survivors': n_survivors,
            'n_pruned_asym': n_pruned_asym,
            'n_pruned_test': n_pruned_test,
            'n_processed': n_processed,
            'elapsed': elapsed,
            'proven': proven,
            'min_cert_net': global_min_net,
            'box_certified': box_ok,
        }

    S = int(2 * d * m)  # Fine grid: integer coords sum to 2*d*m
    n_total = count_compositions(d, S)
    corr = correction(m, n_half)

    # --- Branch-and-bound path ---
    if use_bnb and d <= 8:
        if verbose:
            _log(f"\n[L0 B&B] d={d}, m={m}, n_half={n_half}, "
                 f"compositions={n_total:,}")
            _log(f"     correction={corr:.6f}, "
                 f"threshold={c_target+corr:.6f}")

        t0 = time.time()

        survivors, n_surv_raw, n_tested = _l0_bnb_run(
            d, S, n_half, m, c_target, use_flat_threshold)

        # Canonical filter (keep only c <= rev(c) lexicographically)
        n_non_canonical = 0
        if len(survivors) > 0:
            canon_mask = _canonical_mask(survivors)
            n_non_canonical = int(np.sum(~canon_mask))
            survivors = survivors[canon_mask]

        # F → Q → L post-filter chain on B&B output.  Each is strictly
        # tighter and sound by monotonicity (F ⊇ Q ⊇ L).
        n_F_pruned = 0
        if use_F and len(survivors) > 0 and not use_flat_threshold:
            f_mask = _prune_dynamic(survivors, n_half, m, c_target,
                                     use_flat_threshold=False, use_F=True)
            n_F_pruned = int(np.sum(~f_mask))
            survivors = survivors[f_mask]
        if (use_Q or use_L) and len(survivors) > 0 and not use_flat_threshold:
            from post_filters import apply_post_filter_chain
            n_before = len(survivors)
            survivors = apply_post_filter_chain(
                survivors, n_half, m, c_target,
                use_Q=use_Q, use_L=use_L)
            n_F_pruned += (n_before - len(survivors))

        elapsed = time.time() - t0
        n_survivors = len(survivors)
        proven = n_survivors == 0

        if verbose:
            _log(f"     {elapsed:.2f}s: {n_tested:,} leaves tested "
                 f"(of {n_total:,} total compositions)")
            _log(f"     B&B pruned: {n_total - n_tested:,} branches, "
                 f"leaf pruned: {n_tested - n_surv_raw:,}, "
                 f"non-canonical: {n_non_canonical:,}")
            if use_F:
                _log(f"     F post-filter pruned: {n_F_pruned:,}")
            _log(f"     survivors: {n_survivors:,}")
            if proven:
                _log(f"     PROVEN at L0!")

        return {
            'survivors': survivors,
            'n_survivors': n_survivors,
            'n_pruned_asym': 0,  # asymmetry handled inside B&B kernel
            'n_pruned_test': n_tested - n_surv_raw + n_F_pruned,
            'n_processed': n_tested,
            'elapsed': elapsed,
            'proven': proven,
        }

    # --- Original batch path (fallback for large d) ---
    if verbose:
        _log(f"\n[L0] d={d}, m={m}, n_half={n_half}, compositions={n_total:,}")
        _log(f"     correction={corr:.6f}, threshold={c_target+corr:.6f}")

    t0 = time.time()
    all_survivors = []
    n_pruned_asym = 0
    n_pruned_test = 0
    n_processed = 0
    n_non_canonical = 0
    n_batches = 0
    last_report = t0

    for batch in generate_compositions_batched(d, S, batch_size=200_000):
        n_processed += len(batch)
        n_batches += 1

        # Canonical filter (match GPU: only c <= rev(c))
        canon = _canonical_mask(batch)
        n_non_canonical += int(np.sum(~canon))
        batch = batch[canon]
        if len(batch) == 0:
            continue

        # Asymmetry filter
        needs_check = asymmetry_prune_mask(batch, n_half, m, c_target)
        n_asym_batch = int(np.sum(~needs_check))
        n_pruned_asym += n_asym_batch

        candidates = batch[needs_check]
        if len(candidates) == 0:
            continue

        # Dynamic per-window threshold
        survived_mask = _prune_dynamic(candidates, n_half, m, c_target,
                                        use_flat_threshold, use_F)
        n_pruned_test += int(np.sum(~survived_mask))

        survivors = candidates[survived_mask]
        # F → Q → L per-batch post-filter chain at L0.
        if (use_Q or use_L) and len(survivors) > 0 and not use_flat_threshold:
            from post_filters import apply_post_filter_chain
            n_before = len(survivors)
            survivors = apply_post_filter_chain(
                survivors, n_half, m, c_target,
                use_Q=use_Q, use_L=use_L)
            n_pruned_test += (n_before - len(survivors))
        if len(survivors) > 0:
            all_survivors.append(survivors)

        # Progress: report every 2 seconds or every batch if slow
        now = time.time()
        if verbose and (now - last_report >= 2.0):
            pct = n_processed / n_total * 100 if n_total > 0 else 0
            n_surv_so_far = sum(len(s) for s in all_survivors)
            elapsed_so_far = now - t0
            rate = n_processed / elapsed_so_far if elapsed_so_far > 0 else 0
            eta = (n_total - n_processed) / rate if rate > 0 else 0
            _log(f"     [L0] {n_processed:,}/{n_total:,} ({pct:.1f}%) "
                 f"| {n_surv_so_far:,} survivors | "
                 f"ETA {_fmt_time(eta)}")
            last_report = now

    elapsed = time.time() - t0

    if all_survivors:
        all_survivors = np.vstack(all_survivors)
    else:
        all_survivors = np.empty((0, d), dtype=np.int32)

    n_survivors = len(all_survivors)
    proven = n_survivors == 0

    if verbose:
        _log(f"     {elapsed:.2f}s: {n_processed:,} compositions processed")
        _log(f"     asym pruned: {n_pruned_asym:,}, "
             f"test pruned: {n_pruned_test:,}, "
             f"survivors: {n_survivors:,}")
        if proven:
            _log(f"     PROVEN at L0!")

    return {
        'survivors': all_survivors,
        'n_survivors': n_survivors,
        'n_pruned_asym': n_pruned_asym,
        'n_pruned_test': n_pruned_test,
        'n_processed': n_processed,
        'elapsed': elapsed,
        'proven': proven,
    }


# =====================================================================
# Refinement: uniform full-bin split (legacy — kept as fallback)
# =====================================================================

def generate_children_uniform(parent_int, m, c_target, n_half_child=None):
    """Generate all child compositions from a parent via uniform 2-split.

    Each parent bin c_i is split into (a, b) where a + b = c_i.
    Both sub-bins are capped at x_cap (energy bound) for efficiency.

    Parameters
    ----------
    parent_int : (d_parent,) int array
    m : int
    c_target : float
    n_half_child : int or None
        Half-dimension of child.  When provided, uses the corrected
        correction term ``correction(m, n_half_child)``.

    Returns
    -------
    (N_children, d_child) int32 array where d_child = 2 * d_parent
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    # x_cap: single-bin energy cap (fine grid)
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(4 * d_child * thresh)))
    # Cauchy-Schwarz bound: +1 for canonical rounding adjustment
    x_cap_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    x_cap = min(x_cap, x_cap_cs)
    x_cap = max(x_cap, 0)

    # Build per-bin split options
    per_bin_choices = []
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, 2 * b_i - x_cap)
        hi = min(2 * b_i, x_cap)
        if lo > hi:
            # This parent can't produce valid children
            return np.empty((0, d_child), dtype=np.int32)
        per_bin_choices.append(list(range(lo, hi + 1)))

    # Total children = product of choice counts
    total = 1
    for choices in per_bin_choices:
        total *= len(choices)

    if total == 0:
        return np.empty((0, d_child), dtype=np.int32)

    # For very large expansions, use chunked generation to avoid OOM
    if total > 50_000_000:
        return _generate_children_chunked(parent_int, per_bin_choices,
                                          d_parent, d_child, total)

    children = np.empty((total, d_child), dtype=np.int32)
    idx = 0
    for combo in itertools.product(*per_bin_choices):
        for i in range(d_parent):
            children[idx, 2 * i] = combo[i]
            children[idx, 2 * i + 1] = 2 * int(parent_int[i]) - combo[i]
        idx += 1

    return children


def _generate_children_chunked(parent_int, per_bin_choices, d_parent,
                                d_child, total):
    """Generate children in chunks to avoid memory blowup."""
    chunk_size = 10_000_000
    chunks = []
    buf = np.empty((chunk_size, d_child), dtype=np.int32)
    idx = 0

    for combo in itertools.product(*per_bin_choices):
        for i in range(d_parent):
            buf[idx, 2 * i] = combo[i]
            buf[idx, 2 * i + 1] = 2 * int(parent_int[i]) - combo[i]
        idx += 1
        if idx == chunk_size:
            chunks.append(buf[:idx].copy())
            idx = 0

    if idx > 0:
        chunks.append(buf[:idx].copy())

    return np.vstack(chunks) if chunks else np.empty((0, d_child), dtype=np.int32)


def test_children(children_int, n_half_child, m, c_target):
    """Prune children via asymmetry + dynamic threshold.

    NO canonical filter at refinement levels — applying it here would
    silently drop canonical children whose parent is non-canonical
    (rev(P) for canonical P), since rev(P) is never in our parent list.
    Instead, survivors are canonicalized and deduped after testing.

    Returns
    -------
    (survivors, stats) where survivors is (K, d_child) int32
    """
    if len(children_int) == 0:
        d_child = children_int.shape[1] if children_int.ndim == 2 else 2
        return np.empty((0, d_child), dtype=np.int32), {
            'n_tested': 0, 'n_canonical': 0, 'n_asym': 0, 'n_test': 0,
            'n_survived': 0
        }

    N, d_child = children_int.shape

    # Asymmetry filter
    needs_check = asymmetry_prune_mask(children_int, n_half_child, m, c_target)
    n_asym = int(np.sum(~needs_check))
    candidates = np.ascontiguousarray(children_int[needs_check])

    if len(candidates) > 0:
        # Dynamic per-window threshold
        survived_mask = _prune_dynamic(candidates, n_half_child, m, c_target)
        survivors = np.ascontiguousarray(candidates[survived_mask])
        n_test = int(np.sum(~survived_mask))
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)
        n_test = 0

    # Canonicalize survivors using Numba parallel kernel
    if len(survivors) > 0:
        _canonicalize_inplace(survivors)

    return survivors, {
        'n_tested': N,
        'n_canonical': 0,
        'n_asym': n_asym,
        'n_test': n_test,
        'n_survived': len(survivors),
    }


# =====================================================================
# Multiprocessing helpers
# =====================================================================

def _init_worker_threads(n_threads):
    """Limit Numba parallelism in each worker to avoid oversubscription."""
    numba.set_num_threads(n_threads)


def _process_single_parent_fused(args):
    """Worker: generate + prune children for one parent using fused kernel.

    Avoids materializing the full children array — generates each child
    on-the-fly and prunes inline.

    Accepts 5-element tuple (parent, m, c_target, n_half_child, batch_size),
    6-element tuple with buf_cap, or 7-element tuple with use_flat_threshold.
    """
    use_flat = False
    buf_cap = None
    if len(args) >= 7:
        parent, m, c_target, n_half_child, batch_size, buf_cap, use_flat = args[:7]
    elif len(args) >= 6:
        parent, m, c_target, n_half_child, batch_size, buf_cap = args[:6]
    else:
        parent, m, c_target, n_half_child, batch_size = args

    survivors, total_children = process_parent_fused(
        parent, m, c_target, n_half_child, buf_cap=buf_cap,
        use_flat_threshold=use_flat)

    n_survived = len(survivors)

    if n_survived > 0:
        result = survivors
    else:
        result = None

    return result, {
        'children': total_children,
        'asym': 0,
        'test': 0,
        'survived': n_survived,
    }


def _process_single_parent_legacy(args):
    """Legacy worker: generate children for one parent, prune, return survivors.

    Kept as fallback — the fused version is the default.
    """
    parent, m, c_target, n_half_child, batch_size = args

    children = generate_children_uniform(parent, m, c_target, n_half_child)
    n_children = len(children)

    if n_children == 0:
        return None, {'children': 0, 'asym': 0, 'test': 0, 'survived': 0}

    parent_survivors = []
    total_asym = 0
    total_test = 0
    total_survived = 0

    for start in range(0, n_children, batch_size):
        end = min(start + batch_size, n_children)
        batch = children[start:end]

        survivors, stats = test_children(batch, n_half_child, m, c_target)
        total_asym += stats['n_asym']
        total_test += stats['n_test']
        total_survived += stats['n_survived']

        if len(survivors) > 0:
            parent_survivors.append(survivors)

    if parent_survivors:
        result = np.vstack(parent_survivors)
    else:
        result = None

    return result, {
        'children': n_children,
        'asym': total_asym,
        'test': total_test,
        'survived': total_survived,
    }


def _process_single_parent(args):
    """Worker: generate + prune children for one parent.

    Uses the fused kernel by default.
    """
    return _process_single_parent_fused(args)


# =====================================================================
# Shared-memory multiprocessing helpers
# =====================================================================

def _init_worker_shm(mmap_path, shape, dtype_str, m, c_target, n_half_child,
                     numba_threads, use_flat_threshold=False, use_F=False,
                     skip_sdp_cert=False, use_Q=False, use_L=False):
    """Pool initializer: open mmap of parent array and store params in globals."""
    numba.set_num_threads(numba_threads)
    global _shared_parents, _shm_m, _shm_c_target, _shm_n_half_child
    global _shm_use_flat_threshold, _shm_use_F, _shm_skip_sdp
    global _shm_use_Q, _shm_use_L
    _shared_parents = np.memmap(mmap_path, dtype=np.dtype(dtype_str),
                                mode='r', shape=shape)
    _shm_m = m
    _shm_c_target = c_target
    _shm_n_half_child = n_half_child
    _shm_use_flat_threshold = use_flat_threshold
    _shm_use_F = use_F
    _shm_skip_sdp = skip_sdp_cert
    _shm_use_Q = use_Q
    _shm_use_L = use_L


def _process_parent_shm(idx):
    """Worker: process parent at index idx from shared memory array."""
    parent = _shared_parents[idx].copy()  # local copy from shared mem
    survivors, total_children = process_parent_fused(
        parent, _shm_m, _shm_c_target, _shm_n_half_child,
        use_flat_threshold=_shm_use_flat_threshold,
        use_F=_shm_use_F,
        skip_sdp_cert=_shm_skip_sdp,
        use_Q=_shm_use_Q, use_L=_shm_use_L)
    n_survived = len(survivors)
    result = survivors if n_survived > 0 else None
    return result, {
        'children': total_children,
        'asym': 0,
        'test': 0,
        'survived': n_survived,
    }


# =====================================================================
# Coarse-grid shared-memory multiprocessing helpers
# =====================================================================

def _init_worker_coarse(mmap_path, shape, dtype_str, S, c_target,
                        d_child, numba_threads):
    """Pool initializer for coarse-grid workers."""
    numba.set_num_threads(numba_threads)
    global _shared_parents, _shm_S_coarse, _shm_c_target
    global _shm_d_child_coarse
    _shared_parents = np.memmap(mmap_path, dtype=np.dtype(dtype_str),
                                mode='r', shape=shape)
    _shm_S_coarse = S
    _shm_c_target = c_target
    _shm_d_child_coarse = d_child


def _process_parent_coarse_shm(idx):
    """Worker: process parent at index idx from shared memory (coarse)."""
    parent = _shared_parents[idx].copy()
    survivors, total_children, min_net = process_parent_coarse(
        parent, _shm_S_coarse, _shm_c_target, _shm_d_child_coarse)
    n_survived = len(survivors)
    result = survivors if n_survived > 0 else None
    return result, {
        'children': total_children,
        'survived': n_survived,
        'min_cert_net': min_net,
    }


# =====================================================================
# Checkpoint helpers
# =====================================================================

def _save_checkpoint(output_dir, level, survivors, meta):
    """Save survivors array and metadata after a completed level."""
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f'checkpoint_L{level}_survivors.npy')
    meta_path = os.path.join(output_dir, 'checkpoint_meta.json')

    np.save(npy_path, survivors)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=_convert)

    _log(f"     Checkpoint saved: {npy_path} "
         f"({survivors.nbytes / 1e9:.2f} GB, {len(survivors):,} rows)")


def _load_checkpoint(resume_dir, n_half, m, c_target):
    """Load checkpoint if it exists and parameters match.

    Returns (survivors, level_num, info) or None.
    """
    meta_path = os.path.join(resume_dir, 'checkpoint_meta.json')
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Validate parameters match
    if (meta['n_half'] != n_half or meta['m'] != m
            or meta['c_target'] != c_target):
        _log(f"     Checkpoint found but parameters don't match:")
        _log(f"       checkpoint: n_half={meta['n_half']}, m={meta['m']}, "
             f"c_target={meta['c_target']}")
        _log(f"       requested:  n_half={n_half}, m={m}, "
             f"c_target={c_target}")
        return None

    level = meta['level_completed']
    npy_path = os.path.join(resume_dir,
                            f'checkpoint_L{level}_survivors.npy')
    if not os.path.exists(npy_path):
        _log(f"     Checkpoint meta found but {npy_path} missing")
        return None

    survivors = np.load(npy_path, mmap_mode='r')
    _log(f"     Loaded checkpoint: L{level} complete, "
         f"{len(survivors):,} survivors (d={survivors.shape[1]})")

    info = meta.get('info', {})
    return survivors, level, info


# =====================================================================
# CPU detection (cgroup-aware for containers)
# =====================================================================

def _effective_cpu_count():
    """Detect actual usable CPUs, accounting for cgroup limits in containers.

    On bare metal, returns mp.cpu_count().  In Docker/RunPod containers,
    mp.cpu_count() returns the host CPU count (e.g. 192) even though the
    container may be cgroup-limited to 32 vCPUs.  This reads the cgroup
    quota to return the correct value.
    """
    logical = mp.cpu_count()
    # Try cgroup v1
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read().strip())
        if quota > 0:
            cgroup_cpus = max(1, int(quota / period))
            return min(logical, cgroup_cpus)
    except (FileNotFoundError, ValueError, OSError):
        pass
    # Try cgroup v2
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            parts = f.read().strip().split()
            if parts[0] != 'max':
                cgroup_cpus = max(1, int(int(parts[0]) / int(parts[1])))
                return min(logical, cgroup_cpus)
    except (FileNotFoundError, ValueError, OSError):
        pass
    return logical


# =====================================================================
# Cascade runner
# =====================================================================

def run_cascade(n_half, m, c_target, max_levels=10, n_workers=None,
                verbose=True, output_dir='data', resume_dir=None,
                use_flat_threshold=False, mass_grid=False, d0=None,
                use_bnb=True, coarse_S=None, use_F=False,
                skip_sdp_cert=False, use_Q=False, use_L=False):
    """Run the full CPU cascade: L0 + refinement levels.

    Parameters
    ----------
    n_half : int or float
        Initial n_half (d0 = 2 * n_half).  Default 1 → d0=2.
        Can be float for odd d0.  Ignored when d0 is given.
    d0 : int, optional
        Starting dimension (overrides n_half).  Must be >= 2.
        Default behaviour (d0=None, n_half=1) gives d0=2, which is
        optimal: d0=2 produces ~2x fewer survivors than d0=4 at the
        d=8 bottleneck, at the cost of one extra trivial cascade level.
    m : int
        Grid resolution.
    c_target : float
        Target lower bound.
    max_levels : int
        Max refinement levels after L0.
    n_workers : int or None
        Number of parallel workers for refinement levels.
        None = auto-detect CPU count.
    verbose : bool
        Print progress.
    output_dir : str
        Directory for checkpoints and results.
    resume_dir : str or None
        Directory to look for checkpoint files.  If None, uses output_dir.
    use_flat_threshold : bool
        When True, uses the flat C&S Lemma 3 correction (2/m + 1/m^2)
        instead of the W-refined correction (1 + W_int/(2n))/m^2.
        Required for verifying the Lean axiom cascade_all_pruned.
        The flat threshold is higher (harder to prune), so the cascade
        may produce more survivors and take longer.

    Returns
    -------
    dict with cascade results.
    """
    if n_workers is None:
        # Benchmark on AMD EPYC 9354 (64 vCPU) showed optimal throughput
        # at ~cpus/5 workers: the fused Gray code kernel is sequential
        # (not prange-parallel), so Numba threads add nothing — maximize
        # worker-level parallelism with moderate process count to avoid
        # IPC overhead and memory duplication.
        eff = _effective_cpu_count()
        n_workers = max(1, eff // 5) if eff >= 10 else max(1, eff)
    n_workers = max(1, n_workers)

    # Ensure enough file descriptors for spawn-based multiprocessing
    # (each worker needs ~6 fds for pipes/semaphores)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        needed = n_workers * 8 + 256  # generous headroom
        if soft < needed:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (min(needed, hard), hard))
    except (ImportError, ValueError, OSError):
        pass  # Windows or insufficient permissions — proceed anyway
    if d0 is not None:
        n_half = d0 / 2.0  # float for odd d
    else:
        d0 = 2 * n_half
    if d0 < 2:
        raise ValueError(
            f'd0={d0} is invalid (must be >= 2).  d0=1 has degenerate '
            'asymmetry pruning: left_bins = d//2 = 0 makes left_frac '
            'always 0, so ALL compositions are spuriously pruned and '
            'the cascade falsely claims proof for any c_target < 2.0.')
    corr = correction(m, n_half)
    S0 = int(2 * d0 * m)
    if coarse_S is not None:
        # Coarse-grid mode: constant S, Theorem 1 (no correction)
        n_total = count_compositions(d0, coarse_S)
        if verbose:
            _log(f"\n{'='*70}")
            _log(f"CPU CASCADE PROVER (COARSE — Theorem 1, no correction)")
            _log(f"  d0={d0}, S={coarse_S}, c_target={c_target}")
            _log(f"  Grid: integer masses sum to S={coarse_S} (constant)")
            _log(f"  Threshold: TV >= {c_target} (exact, no correction)")
            _log(f"  Box cert: margin > cell_var + quad_corr (2nd-order)")
            _log(f"  L0 compositions: {n_total:,}")
            _log(f"  workers: {n_workers}")
            _log(f"  max refinement levels: {max_levels}")
            _log(f"{'='*70}")
    elif mass_grid:
        n_total = count_compositions(n_half, m)
    else:
        n_total = count_compositions(d0, S0)

    if coarse_S is None and verbose:
        _log(f"\n{'='*70}")
        _log(f"CPU CASCADE PROVER{' (MASS-GRID)' if mass_grid else ''}")
        _log(f"  n_half={n_half}, m={m}, d0={d0}, c_target={c_target}")
        _log(f"  correction(m, n_half={n_half})={corr:.6f}, "
             f"effective threshold={c_target+corr:.6f}")
        _log(f"  L0 compositions: {n_total:,}"
             f"{' (palindromic)' if mass_grid else ''}")
        _log(f"  workers: {n_workers} (logical CPUs: {mp.cpu_count()}, "
             f"effective: {_effective_cpu_count()})")
        _log(f"  max refinement levels: {max_levels}")
        _log(f"{'='*70}")

    if resume_dir is None:
        resume_dir = output_dir

    t_total = time.time()

    # --- Try to resume from checkpoint ---
    resume_result = _load_checkpoint(resume_dir, n_half, m, c_target)
    start_level = 0  # 0 means run L0 fresh

    if resume_result is not None:
        current_configs, last_completed, saved_info = resume_result
        start_level = last_completed + 1
        d_parent = current_configs.shape[1]
        # n_half doubles each level: at level L, n_half_parent = n_half * 2^L
        n_half_parent = n_half * (2 ** last_completed)

        info = saved_info if isinstance(saved_info, dict) else {}
        # Ensure required keys exist
        info.setdefault('n_half', n_half)
        info.setdefault('m', m)
        info.setdefault('d0', d0)
        info.setdefault('c_target', c_target)
        info.setdefault('correction', corr)
        info.setdefault('levels', [])

        if verbose:
            _log(f"\n  RESUMING from L{last_completed} checkpoint")
            _log(f"  {len(current_configs):,} survivors at d={d_parent}")
            _log(f"  Skipping L0 through L{last_completed}")

    if start_level == 0:
        # --- Level 0 (fresh run) ---
        if mass_grid:
            l0 = run_level0_mass(n_half, m, c_target, verbose=verbose)
        elif coarse_S is not None:
            l0 = run_level0(n_half, m, c_target, verbose=verbose,
                             d0=d0, coarse_S=coarse_S)
        else:
            l0 = run_level0(n_half, m, c_target, verbose=verbose,
                             use_flat_threshold=use_flat_threshold,
                             d0=d0, use_bnb=use_bnb, use_F=use_F,
                             skip_sdp_cert=skip_sdp_cert,
                             use_Q=use_Q, use_L=use_L)

        info = {
            'n_half': n_half, 'm': m, 'd0': d0, 'c_target': c_target,
            'correction': corr,
            'l0_time': l0['elapsed'],
            'l0_survivors': l0['n_survivors'],
            'l0_pruned_asym': l0['n_pruned_asym'],
            'l0_pruned_test': l0['n_pruned_test'],
            'levels': [],
        }
        if coarse_S is not None:
            info['grid'] = 'coarse'
            info['S'] = coarse_S
            info['l0'] = {
                'min_cert_net': l0.get('min_cert_net', 1e30),
                'box_certified': l0.get('box_certified', False),
            }

        if l0['proven']:
            info['proven_at'] = 'L0'
            info['total_time'] = time.time() - t_total
            if coarse_S is not None:
                info['box_certified'] = l0.get('box_certified', False)
            return info

        current_configs = l0['survivors']
        d_parent = d0
        n_half_parent = n_half

        # Checkpoint L0 survivors
        _save_checkpoint(output_dir, 0, current_configs, {
            'n_half': n_half, 'm': m, 'c_target': c_target,
            'level_completed': 0,
            'd_survivors': d_parent,
            'n_survived': len(current_configs),
            'info': info,
        })

    # --- Refinement levels ---
    for level_num in range(max(1, start_level), max_levels + 1):
        d_child = 2 * d_parent
        n_half_child = 2 * n_half_parent
        n_parents = len(current_configs)

        if n_parents == 0:
            break

        # === COARSE CASCADE PATH ===
        if coarse_S is not None:
            x_cap_c = int(math.floor(
                coarse_S * math.sqrt(c_target / d_child)))
            d2_over_2S2 = float(d_child ** 2) / (2.0 * coarse_S ** 2)

            if verbose:
                _log(f"\n[L{level_num}] d={d_parent} -> {d_child}, "
                     f"parents={n_parents:,}, x_cap={x_cap_c}")
                _log(f"    quad_corr scale: d^2/(2S^2) = {d2_over_2S2:.6f}")

            # Pre-filter infeasible parents: in coarse grid, S is constant,
            # so child splits parent bin p into (c, p-c) with both <= x_cap.
            # Feasible iff p <= 2*x_cap (not p <= x_cap as in fine grid).
            feasible = np.all(current_configs <= 2 * x_cap_c, axis=1)
            n_inf = n_parents - int(feasible.sum())
            if n_inf > 0:
                current_configs = np.ascontiguousarray(
                    current_configs[feasible])
                n_parents = len(current_configs)
                if verbose:
                    _log(f"    Pre-filtered {n_inf:,} infeasible parents")

            if n_parents == 0:
                if verbose:
                    _log(f"    All parents infeasible -> 0 survivors")
                info['levels'].append({
                    'level': level_num, 'd_child': d_child,
                    'parents_in': 0, 'total_children': 0,
                    'survivors_out': 0,
                })
                d_parent = d_child
                n_half_parent = n_half_child
                continue

            t_level = time.time()
            all_survivors_c = []
            total_children_c = 0
            level_min_net = 1e30
            n_done = 0
            last_report_c = time.time()

            if n_workers > 1 and n_parents > n_workers:
                # Parallel coarse path
                parents_shape = current_configs.shape
                parents_dtype_str = current_configs.dtype.str
                fd, mmap_path = tempfile.mkstemp(
                    suffix=f'_coarse_L{level_num}.dat', dir=output_dir)
                os.close(fd)
                current_configs.tofile(mmap_path)

                ctx = mp.get_context("spawn")
                try:
                    with ctx.Pool(
                            n_workers,
                            initializer=_init_worker_coarse,
                            initargs=(mmap_path, parents_shape,
                                      parents_dtype_str,
                                      coarse_S, c_target, d_child,
                                      1)) as pool:
                        for surv, stats in pool.imap_unordered(
                                _process_parent_coarse_shm,
                                range(n_parents), chunksize=1):
                            total_children_c += stats['children']
                            mn = stats.get('min_cert_net', 1e30)
                            if mn < level_min_net:
                                level_min_net = mn
                            if surv is not None:
                                all_survivors_c.append(surv)
                            n_done += 1
                            now = time.time()
                            if verbose and (now - last_report_c >= 5.0):
                                ns = sum(len(s) for s in all_survivors_c)
                                pct = n_done / n_parents * 100
                                _log(f"    [{n_done}/{n_parents}] "
                                     f"({pct:.1f}%) "
                                     f"{ns:,} survivors so far")
                                last_report_c = now
                finally:
                    try:
                        os.remove(mmap_path)
                    except OSError:
                        pass
            else:
                # Sequential coarse path
                for p_idx in range(n_parents):
                    parent = current_configs[p_idx]
                    surv, n_ch, mn = process_parent_coarse(
                        parent, coarse_S, c_target, d_child)
                    total_children_c += n_ch
                    if mn < level_min_net:
                        level_min_net = mn
                    if len(surv) > 0:
                        all_survivors_c.append(surv)
                    n_done += 1
                    now = time.time()
                    if verbose and (now - last_report_c >= 5.0):
                        ns = sum(len(s) for s in all_survivors_c)
                        _log(f"    [{n_done}/{n_parents}] "
                             f"{ns:,} survivors so far")
                        last_report_c = now

            elapsed_level = time.time() - t_level

            if all_survivors_c:
                current_configs = np.vstack(all_survivors_c)
                current_configs = _fast_dedup(current_configs)
            else:
                current_configs = np.empty((0, d_child), dtype=np.int32)

            n_survived = len(current_configs)
            box_ok = level_min_net >= 0.0

            if verbose:
                rate = (n_survived / max(1, total_children_c) * 100)
                _log(f"    {elapsed_level:.1f}s: {total_children_c:,} "
                     f"children, {n_survived:,} survivors "
                     f"({rate:.4f}%)")
                if total_children_c > 0:
                    _log(f"    min(margin - cell_var - quad_corr) = "
                         f"{level_min_net:.6f}")
                    _log(f"    BOX CERT: "
                         f"{'PASS' if box_ok else 'FAIL'}")
                if n_survived == 0:
                    _log(f"    *** ALL PRUNED ***")

            info['levels'].append({
                'level': level_num,
                'd_child': d_child,
                'parents_in': n_parents,
                'total_children': total_children_c,
                'survivors_out': n_survived,
                'elapsed': elapsed_level,
                'min_cert_net': level_min_net,
                'box_certified': box_ok,
            })

            if n_survived == 0:
                info['proven_at'] = f'L{level_num}'
                all_cert = info.get('l0', {}).get(
                    'box_certified', True)
                if all_cert:
                    for lv in info['levels']:
                        if not lv.get('box_certified', True):
                            all_cert = False
                            break
                info['box_certified'] = all_cert
                break

            _save_checkpoint(output_dir, level_num, current_configs, {
                'n_half': n_half, 'm': m, 'c_target': c_target,
                'level_completed': level_num,
                'd_survivors': d_child,
                'n_survived': n_survived,
                'info': info,
            })

            d_parent = d_child
            n_half_parent = n_half_child
            continue
        # === END COARSE CASCADE PATH ===

        # --- Pre-filter: skip parents where any bin produces empty range ---
        # A parent bin b_i has cursor range [max(0, 2*b_i - x_cap), min(2*b_i, x_cap)].
        # Empty when 2*b_i - x_cap > min(2*b_i, x_cap), i.e. b_i > x_cap.
        # (Since lo = max(0, 2*b_i - x_cap) and hi = min(2*b_i, x_cap),
        # lo > hi iff 2*b_i > 2*x_cap, i.e. b_i > x_cap.)
        corr_pf = correction(m, n_half_child)
        thresh_pf = c_target + corr_pf + 1e-9
        x_cap_pf = int(math.floor(m * math.sqrt(4 * d_child * thresh_pf)))
        # Cauchy-Schwarz bound: +1 for canonical rounding adjustment
        x_cap_cs_pf = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
        x_cap_pf = min(x_cap_pf, x_cap_cs_pf)
        max_bin_val = x_cap_pf
        feasible_mask = np.all(current_configs <= max_bin_val, axis=1)
        n_infeasible = n_parents - int(np.sum(feasible_mask))
        if n_infeasible > 0:
            current_configs = np.ascontiguousarray(current_configs[feasible_mask])
            n_parents = len(current_configs)
            if verbose:
                _log(f"     Pre-filtered {n_infeasible:,} infeasible parents "
                     f"(bin > {max_bin_val})")

        if n_parents == 0:
            break

        # --- Pre-filter: block mass invariant pruning ---
        # For any contiguous block of k parent bins with total mass M,
        # the child autoconvolution sum over that block's conv range is
        # >= 4M^2 (invariant under all cursor assignments).  If this
        # exceeds the threshold, ALL children of this parent are pruned.
        bm_mask = block_mass_prune_mask(
            current_configs, n_half_child, m, c_target,
            use_flat_threshold=use_flat_threshold)
        n_block_pruned = n_parents - int(np.sum(bm_mask))
        if n_block_pruned > 0:
            current_configs = np.ascontiguousarray(current_configs[bm_mask])
            n_parents = len(current_configs)
            if verbose:
                _log(f"     Block-mass pruned {n_block_pruned:,} parents "
                     f"(contiguous block self-conv exceeds threshold)")

        if n_parents == 0:
            break

        # Shuffle parents for unbiased ETA estimation and better load balance.
        # Lex order from dedup correlates with per-parent cost (bin values
        # determine child count, varying ~1000x).  Fixed seed for reproducibility.
        # Ensure writable (checkpoint loads as read-only mmap).
        if not current_configs.flags.writeable:
            current_configs = np.array(current_configs)
        rng = np.random.RandomState(42)
        rng.shuffle(current_configs)

        if verbose:
            _log(f"\n[L{level_num}] d_parent={d_parent} -> d_child={d_child}, "
                 f"{n_parents:,} parents")

        t_level = time.time()
        total_children = 0
        total_survived = 0
        report_interval = _progress_interval(n_parents)

        if mass_grid:
            # --- Mass-grid palindrome path (serial, small search space) ---
            all_survivors_mg = []
            for pi in range(n_parents):
                parent = current_configs[pi]
                surv, n_ch = process_parent_mass(
                    parent, m, c_target, n_half_child)
                total_children += n_ch
                total_survived += len(surv)
                if len(surv) > 0:
                    all_survivors_mg.append(surv)
                if verbose and (pi + 1) % max(1, n_parents // 10) == 0:
                    elapsed_so_far = time.time() - t_level
                    n_surv = sum(len(s) for s in all_survivors_mg)
                    _log(f"     {pi+1:,}/{n_parents:,} parents "
                         f"| {n_surv:,} survivors | "
                         f"{elapsed_so_far:.1f}s")

            elapsed_level = time.time() - t_level
            if all_survivors_mg:
                all_surv = np.vstack(all_survivors_mg)
                all_surv = _fast_dedup(all_surv)
            else:
                all_surv = np.empty((0, d_child), dtype=np.int32)

            n_survived_level = len(all_surv)
            expansion = n_survived_level / n_parents if n_parents > 0 else 0

            level_info = {
                'level': level_num,
                'd_parent': d_parent,
                'd_child': d_child,
                'parents_in': n_parents,
                'survivors': n_survived_level,
                'total_children': total_children,
                'expansion': expansion,
                'elapsed': elapsed_level,
            }
            info['levels'].append(level_info)

            if verbose:
                _log(f"     L{level_num}: {n_survived_level:,} survivors "
                     f"(expansion {expansion:.1f}x) in {elapsed_level:.2f}s")

            if n_survived_level == 0:
                info['proven_at'] = f'L{level_num}'
                info['total_time'] = time.time() - t_total
                if verbose:
                    _log(f"\n  PROVEN at L{level_num}!")
                return info

            # Checkpoint
            _save_checkpoint(output_dir, level_num, all_surv, {
                'n_half': n_half, 'm': m, 'c_target': c_target,
                'level_completed': level_num,
                'd_survivors': d_child,
                'n_survived': n_survived_level,
                'info': info,
            })

            current_configs = all_surv
            d_parent = d_child
            n_half_parent = n_half_child
            continue
        # --- End mass_grid path ---

        # Memory-safe survivor collection: accumulate in RAM up to a
        # budget, then spill to disk shards.  Final dedup merges shards.
        bytes_per_row = d_child * 4
        try:
            import psutil
            avail_bytes = psutil.virtual_memory().available
        except ImportError:
            avail_bytes = int(64e9 * 0.80)
        # Reserve memory for shared array + workers + OS.
        # _fast_dedup needs ~3x the array size (input + sort index + output),
        # so the safe in-RAM batch is 1/4 of available budget.
        shm_bytes = 0  # mmap: parent array lives in OS page cache, not process RSS
        survivor_mem_budget = max(int(1e9),
            (avail_bytes - shm_bytes - int(10e9)) // 4)
        shard_threshold = max(100_000, survivor_mem_budget // bytes_per_row)
        if verbose:
            _log(f"     Survivor spool: {survivor_mem_budget/1e9:.1f} GB "
                 f"in-RAM ({shard_threshold:,} rows), then disk shards")

        all_survivors = []
        all_survivors_rows = 0
        shard_dir = os.path.join(output_dir, f'_shards_L{level_num}')
        shard_paths = []
        n_shards = 0

        def _flush_to_shard():
            nonlocal all_survivors, all_survivors_rows, n_shards
            if not all_survivors:
                return
            batch = np.vstack(all_survivors)
            batch = _fast_dedup(batch)
            os.makedirs(shard_dir, exist_ok=True)
            path = os.path.join(shard_dir, f'shard_{n_shards:04d}.npy')
            np.save(path, batch)
            shard_paths.append(path)
            n_shards += 1
            if verbose:
                _log(f"     Flushed shard {n_shards}: {len(batch):,} unique rows "
                     f"({batch.nbytes/1e9:.2f} GB)")
            all_survivors = []
            all_survivors_rows = 0

        if n_workers > 1 and n_parents > n_workers:
            # --- Parallel path: shared memory + index dispatch ---

            # Memory-aware worker cap.
            # Workers and dedup don't peak concurrently: dedup runs in the
            # main process between imap_unordered batches, and the shard
            # flush frees the accumulator before the next fill cycle.
            # Reserve: 1x survivor spool (pre-flush peak) + 4 GB OS/main.
            _buf_cap = _default_buf_cap(d_child)
            per_worker_bytes = _buf_cap * d_child * 4 + 150 * 1024 * 1024
            reserved = shm_bytes + survivor_mem_budget + int(4e9)
            worker_mem_budget = max(int(1e9), avail_bytes - reserved)
            max_by_mem = max(1, int(worker_mem_budget / per_worker_bytes))
            if n_workers > max_by_mem:
                if verbose:
                    _log(f"     Memory cap: {n_workers} -> {max_by_mem} workers "
                         f"(avail={avail_bytes/1e9:.1f}GB, "
                         f"per_worker={per_worker_bytes/1e9:.2f}GB, "
                         f"shm={shm_bytes/1e9:.2f}GB)")
                n_workers_level = max_by_mem
            else:
                n_workers_level = n_workers

            # Benchmark: Numba threads have zero effect on the fused Gray
            # code kernel (it is sequential, not prange). Set to 1 to avoid
            # thread-pool overhead inside workers.
            numba_threads = 1
            # Benchmark: chunksize=1 is optimal — parent costs are highly
            # heterogeneous (1000x variation), so fine-grained dispatch
            # gives the best load balance.
            chunksize = 1

            # Write parent array to a temp file; workers mmap it read-only
            parents_shape = current_configs.shape
            parents_dtype_str = current_configs.dtype.str
            parents_nbytes = current_configs.nbytes
            fd, mmap_path = tempfile.mkstemp(
                suffix=f'_L{level_num}_parents.dat', dir=output_dir)
            os.close(fd)
            current_configs.tofile(mmap_path)
            del current_configs  # free RAM; workers mmap from disk

            if verbose:
                mmap_gb = parents_nbytes / 1e9
                _log(f"     (parallel: {n_workers_level} workers, "
                     f"chunksize={chunksize}, "
                     f"numba_threads={numba_threads}, "
                     f"mmap={mmap_gb:.2f} GB)")

            completed = 0
            last_report = time.time()
            last_checkpoint_time = time.time()
            checkpoint_interval = 1800  # 30 minutes
            ctx = mp.get_context("spawn")
            try:
                with ctx.Pool(
                        n_workers_level,
                        initializer=_init_worker_shm,
                        initargs=(mmap_path, parents_shape,
                                  parents_dtype_str,
                                  m, c_target, n_half_child,
                                  numba_threads,
                                  use_flat_threshold,
                                  use_F,
                                  skip_sdp_cert,
                                  use_Q, use_L)) as pool:
                    for surv, stats in pool.imap_unordered(
                            _process_parent_shm, range(n_parents),
                            chunksize=chunksize):
                        total_children += stats['children']
                        total_survived += stats['survived']
                        completed += 1

                        if surv is not None:
                            all_survivors.append(surv)
                            all_survivors_rows += len(surv)

                        # Flush to disk when in-RAM budget exceeded
                        if all_survivors_rows >= shard_threshold:
                            _flush_to_shard()

                        now = time.time()
                        if verbose and (completed % report_interval == 0
                                        or now - last_report >= 5.0):
                            elapsed_so_far = now - t_level
                            rate = completed / elapsed_so_far if elapsed_so_far > 0 else 0
                            eta = (n_parents - completed) / rate if rate > 0 else 0
                            pct = completed / n_parents * 100
                            _log(f"     [{completed}/{n_parents}] ({pct:.1f}%) "
                                 f"{total_survived:,} survivors so far, "
                                 f"shards={n_shards}, "
                                 f"ETA {_fmt_time(eta)}")
                            last_report = now

                        # Intra-level checkpoint: save progress every 30 min
                        if now - last_checkpoint_time >= checkpoint_interval:
                            progress_path = os.path.join(
                                output_dir,
                                f'_progress_L{level_num}.json')
                            progress = {
                                'level': level_num,
                                'completed': completed,
                                'total_parents': n_parents,
                                'total_children': total_children,
                                'total_survived': total_survived,
                                'elapsed_seconds': now - t_level,
                                'timestamp': time.strftime(
                                    '%Y-%m-%d %H:%M:%S'),
                            }
                            try:
                                with open(progress_path, 'w') as pf:
                                    json.dump(progress, pf, indent=2)
                                if verbose:
                                    _log(f"     Checkpoint: {completed:,}/"
                                         f"{n_parents:,} completed "
                                         f"({progress_path})")
                            except OSError:
                                pass
                            last_checkpoint_time = now
            finally:
                try:
                    os.remove(mmap_path)
                except OSError:
                    pass  # Windows: file may still be held if worker crashed

        else:
            # --- Sequential path: use verbose per-parent progress ---
            for p_idx in range(n_parents):
                parent = current_configs[p_idx]

                if verbose:
                    survivors, n_children = process_parent_verbose(
                        parent, m, c_target, n_half_child,
                        p_idx, n_parents,
                        use_flat_threshold=use_flat_threshold,
                        use_F=use_F,
                        skip_sdp_cert=skip_sdp_cert,
                        use_Q=use_Q, use_L=use_L)
                else:
                    survivors, n_children = process_parent_fused(
                        parent, m, c_target, n_half_child,
                        use_flat_threshold=use_flat_threshold,
                        use_F=use_F,
                        skip_sdp_cert=skip_sdp_cert,
                        use_Q=use_Q, use_L=use_L)
                total_children += n_children
                n_survived_this = len(survivors)
                total_survived += n_survived_this

                if n_survived_this > 0:
                    all_survivors.append(survivors)
                    all_survivors_rows += n_survived_this

                if all_survivors_rows >= shard_threshold:
                    _flush_to_shard()

        elapsed_level = time.time() - t_level

        # --- Merge shards + remaining in-RAM survivors ---
        if all_survivors:
            # Flush last batch
            _flush_to_shard()

        remaining_shards = []
        if shard_paths:
            # Multi-shard: load and merge-dedup incrementally
            if verbose:
                _log(f"     Merging {len(shard_paths)} shards...")
            merged, remaining_shards = _merge_dedup_shards(
                shard_paths, d_child, verbose)
            if merged is not None:
                all_survivors = merged
                try:
                    os.rmdir(shard_dir)
                except OSError:
                    pass
            else:
                # Survivors too large for RAM — save as sharded checkpoint
                all_survivors = np.empty((0, d_child), dtype=np.int32)
        elif all_survivors:
            all_survivors = np.vstack(all_survivors)
            all_survivors = _fast_dedup(all_survivors)
        else:
            all_survivors = np.empty((0, d_child), dtype=np.int32)

        if remaining_shards:
            # Count rows across shards (approximate — may have cross-shard dupes)
            n_survived = 0
            for sp in remaining_shards:
                sz = os.path.getsize(sp) - 128
                n_survived += sz // (d_child * 4)
            if verbose:
                total_gb = sum(os.path.getsize(p) for p in remaining_shards) / 1e9
                _log(f"     Survivors on disk: {n_survived:,} rows in "
                     f"{len(remaining_shards)} shards ({total_gb:.1f} GB)")
                _log(f"     TOO LARGE for RAM — cannot continue cascade.")
                _log(f"     Shards saved in: {shard_dir}")
        else:
            n_survived = len(all_survivors)

        if n_parents > 0:
            factor = n_survived / n_parents
        else:
            factor = 0

        lvl_info = {
            'level': level_num,
            'd_parent': d_parent,
            'd_child': d_child,
            'parents_in': n_parents,
            'total_children': total_children,
            'children_per_parent': total_children / max(1, n_parents),
            'survivors_out': n_survived,
            'expansion_factor': factor,
            'elapsed': elapsed_level,
        }
        info['levels'].append(lvl_info)

        if verbose:
            _log(f"     {elapsed_level:.2f}s: {total_children:,} children "
                 f"({total_children/max(1,n_parents):.1f}/parent)")
            _log(f"     survivors: {n_survived:,} (factor={factor:.4f}x)")
            if n_survived == 0:
                _log(f"     PROVEN at L{level_num}!")

        if n_survived == 0:
            info['proven_at'] = f'L{level_num}'
            break

        if remaining_shards:
            # Survivors too large to fit in RAM — stop cascade here.
            # Shards are already on disk for manual inspection/resume.
            info['stopped_at'] = f'L{level_num}'
            info['stopped_reason'] = 'survivors_exceed_ram'
            info['shard_paths'] = remaining_shards
            break

        # Prepare next level
        current_configs = all_survivors
        d_parent = d_child
        n_half_parent = n_half_child

        # Checkpoint survivors after each completed level
        _save_checkpoint(output_dir, level_num, current_configs, {
            'n_half': n_half, 'm': m, 'c_target': c_target,
            'level_completed': level_num,
            'd_survivors': d_child,
            'n_survived': n_survived,
            'info': info,
        })

    info['total_time'] = time.time() - t_total

    if verbose:
        _log(f"\n{'='*70}")
        if 'proven_at' in info:
            _log(f"PROVEN: c >= {c_target} (cascade converges at "
                 f"{info['proven_at']})")
        else:
            n_remain = len(current_configs) if len(current_configs) > 0 else 0
            _log(f"NOT PROVEN: {n_remain:,} survivors remain at "
                 f"d={d_parent}")
        _log(f"Total time: {_fmt_time(info['total_time'])}")
        _log(f"{'='*70}")

    return info


# =====================================================================
# Progress helpers
# =====================================================================

def _log(msg):
    """Print with immediate flush so remote/piped output is visible."""
    print(msg, flush=True)


def _progress_interval(n_total):
    """Choose a sensible progress reporting interval based on total count."""
    if n_total <= 10:
        return 1
    if n_total <= 100:
        return 10
    if n_total <= 1000:
        return 100
    if n_total <= 10_000:
        return 500
    return 1000


# =====================================================================
# Formatting helpers
# =====================================================================

def _fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.2f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}m'
    return f'{seconds/3600:.2f}h'


def print_summary(info):
    """Print a compact summary table."""
    print(f"\nCASCADE SUMMARY: n_half={info['n_half']}, m={info['m']}, "
          f"d0={info['d0']}, c_target={info['c_target']}")
    print(f"  L0: {_fmt_time(info['l0_time'])}, "
          f"{info['l0_survivors']:,} survivors")

    if info.get('levels'):
        print(f"\n  {'Level':>5} | {'Parents':>10} | {'Children':>12} | "
              f"{'Ch/Par':>8} | {'Survivors':>10} | {'Factor':>10} | "
              f"{'Time':>10}")
        print(f"  {'-'*75}")

        for lvl in info['levels']:
            factor = lvl.get('expansion_factor', 0)
            if lvl.get('parents_in', 0) > 0 and 'expansion_factor' not in lvl:
                factor = lvl.get('survivors_out', 0) / lvl['parents_in']
            if factor == 0:
                fstr = '0x'
            elif factor < 0.01:
                fstr = f'{factor:.6f}x'
            else:
                fstr = f'{factor:.4f}x'

            ch_per = lvl.get('children_per_parent', 0)
            if ch_per == 0 and lvl.get('parents_in', 0) > 0:
                ch_per = lvl.get('total_children', 0) / lvl['parents_in']

            print(f"  L{lvl['level']:>4} | {lvl.get('parents_in', 0):>10,} | "
                  f"{lvl.get('total_children', 0):>12,} | "
                  f"{ch_per:>8.1f} | "
                  f"{lvl.get('survivors_out', 0):>10,} | "
                  f"{fstr:>10} | "
                  f"{_fmt_time(lvl.get('elapsed', 0)):>10}")

    proven_at = info.get('proven_at')
    if proven_at:
        print(f"\n  PROVEN at {proven_at} "
              f"(total: {_fmt_time(info['total_time'])})")
    else:
        last_lvl = info['levels'][-1] if info.get('levels') else None
        remain = last_lvl['survivors_out'] if last_lvl else info['l0_survivors']
        print(f"\n  NOT PROVEN — {remain:,} survivors remain "
              f"(total: {_fmt_time(info['total_time'])})")


# =====================================================================
# Relaxed child verification (for Lean axiom soundness)
# =====================================================================

def _generate_delta_vectors(d):
    """Generate all delta vectors in {-1,0,1}^d that sum to 0.

    For d=4: 19 vectors.  For d=8: 1107 vectors.
    These represent the +/-1 floor-rounding deviations that can occur
    when canonical_discretization at doubled resolution doesn't exactly
    match the cascade's exact child constraint (child[2i]+child[2i+1] = 2*parent[i]).
    """
    deltas = []
    for combo in itertools.product([-1, 0, 1], repeat=d):
        if sum(combo) == 0:
            deltas.append(combo)
    return deltas


def verify_relaxed_children(parents, n_half_parent, m, c_target,
                            use_flat_threshold=True, verbose=True):
    """Verify that ALL +/-1 rounding variants of children are also pruned.

    For each parent at the current level, the cascade checks children with
    child[2i]+child[2i+1] = 2*parent[i] exactly.  But canonical_discretization
    at doubled resolution can produce child[2i]+child[2i+1] = 2*parent[i] + delta_i
    where delta_i in {-1, 0, 1} and sum(delta_i) = 0.

    This function verifies that the cascade ALSO prunes all such +/-1 variants,
    ensuring the CascadePruned axiom with relaxed is_valid_child is sound.

    Parameters
    ----------
    parents : (N, d_parent) int32 array
        Parent compositions that SURVIVED at this level (need children checked).
    n_half_parent : int
        Half-dimension of parent (n_half at the parent level).
    m, c_target : cascade parameters
    use_flat_threshold : bool
        Must be True for Lean axiom verification.

    Returns
    -------
    dict with 'all_pruned' (bool), 'n_delta_variants', 'n_unpruned_variants',
    'unpruned_examples' (list of dicts with parent, delta, survivors).
    """
    d_parent = parents.shape[1]
    n_half_child = 2 * n_half_parent
    d_child = 2 * d_parent

    # Generate non-zero delta vectors (skip all-zeros — that's the standard cascade)
    all_deltas = _generate_delta_vectors(d_parent)
    nonzero_deltas = [d for d in all_deltas if any(di != 0 for di in d)]

    if verbose:
        _log(f"  Relaxed verification: {len(nonzero_deltas)} non-zero delta "
             f"vectors for d_parent={d_parent}")

    n_parents = len(parents)
    total_variants = 0
    unpruned_variants = 0
    unpruned_examples = []

    for pi in range(n_parents):
        parent = parents[pi]

        for delta in nonzero_deltas:
            # Create modified parent: each bin i has pair sum 2*parent[i] + delta[i]
            # instead of 2*parent[i].
            # We need to check if ALL compositions with these modified pair sums
            # are pruned.
            #
            # Approach: create a virtual parent where pair_sum_i = 2*parent[i] + delta[i].
            # The cursor range for bin i is [0, pair_sum_i], and
            # child[2i] = cursor[i], child[2i+1] = pair_sum_i - cursor[i].
            # Skip if any pair_sum_i < 0.
            pair_sums = [2 * int(parent[i]) + delta[i] for i in range(d_parent)]
            if any(ps < 0 for ps in pair_sums):
                continue

            total_variants += 1

            # Generate all children for this delta variant
            # Total children = product of (pair_sum_i + 1) for each bin
            total_children = 1
            for ps in pair_sums:
                total_children *= (ps + 1)

            if total_children == 0:
                continue

            # For small child counts, generate explicitly and batch-prune
            # For large child counts, use the fused kernel with modified ranges
            if total_children <= 100_000:
                # Generate all children explicitly
                children = np.empty((total_children, d_child), dtype=np.int32)
                idx = 0
                ranges = [range(ps + 1) for ps in pair_sums]
                for combo in itertools.product(*ranges):
                    for i in range(d_parent):
                        children[idx, 2 * i] = combo[i]
                        children[idx, 2 * i + 1] = pair_sums[i] - combo[i]
                    idx += 1

                # Apply Cauchy-Schwarz pre-filter (x_cap)
                from pruning import correction as _correction
                corr = _correction(m, n_half_child)
                thresh = c_target + corr + 1e-9
                x_cap = int(math.floor(m * math.sqrt(4 * d_child * thresh)))
                x_cap_cs = int(math.floor(
                    m * math.sqrt(4 * d_child * c_target))) + 1
                x_cap = min(x_cap, x_cap_cs)
                feasible = np.all(children <= x_cap, axis=1)
                children = children[feasible]

                if len(children) == 0:
                    continue

                # Batch prune
                survived = _prune_dynamic_int32(
                    children, n_half_child, m, c_target,
                    use_flat_threshold=use_flat_threshold)
                n_survived = int(np.sum(survived))

                if n_survived > 0:
                    unpruned_variants += 1
                    if len(unpruned_examples) < 5:
                        surv_arr = children[survived]
                        unpruned_examples.append({
                            'parent': parent.tolist(),
                            'delta': list(delta),
                            'n_survived': n_survived,
                            'example_survivor': surv_arr[0].tolist(),
                        })
            else:
                # For very large child counts, use fused kernel approach
                # Create a "virtual parent" with modified pair sums
                virtual_parent = np.array(pair_sums, dtype=np.int32)
                # Divide by 2 (rounding) to create effective parent for the kernel
                # The kernel generates child[2i]+child[2i+1] = 2*vp[i]
                # We need pair_sum = 2*vp[i], so vp[i] = pair_sum/2
                # But pair_sum may be odd! In that case, split differently.
                # For odd pair_sum ps: one child has ceil(ps/2), other has floor(ps/2)
                # We can handle this by running the kernel with vp[i] = (ps+1)//2
                # and adjusting, but this gets complex.
                #
                # Simpler: just skip this and warn. For d_parent=4, total_children
                # is at most ~(2*160+1)^4 which is huge, but the delta shifts are small
                # (+/-1 from 2*parent[i]) so the actual pair sums are close to even.
                # In practice with m=20 and d_parent=4, max parent bin ~160, so
                # pair_sum max ~321, giving ~321^4 ~ 10^10 children. Too many to enumerate.
                #
                # However: the x_cap pre-filter typically reduces this dramatically.
                # And for the Lean axiom, we only care about the case n_half=2, m=20.
                # At L0 (d_parent=4), the surviving parents have small bin values,
                # so total_children after x_cap filtering should be manageable.
                _log(f"  WARNING: delta variant for parent {pi} has {total_children:,} "
                     f"children — too many to enumerate. Skipping.")
                unpruned_variants += 1  # conservative: count as unpruned

        if verbose and (pi + 1) % max(1, n_parents // 10) == 0:
            _log(f"     Relaxed verify: {pi+1}/{n_parents} parents checked, "
                 f"{unpruned_variants} unpruned variants so far")

    all_pruned = unpruned_variants == 0

    if verbose:
        if all_pruned:
            _log(f"  RELAXED VERIFICATION PASSED: all {total_variants} "
                 f"delta variants pruned across {n_parents} parents")
        else:
            _log(f"  RELAXED VERIFICATION FAILED: {unpruned_variants}/"
                 f"{total_variants} variants have survivors")
            for ex in unpruned_examples[:3]:
                _log(f"    parent={ex['parent']}, delta={ex['delta']}, "
                     f"survivors={ex['n_survived']}")

    return {
        'all_pruned': all_pruned,
        'n_delta_variants': total_variants,
        'n_unpruned_variants': unpruned_variants,
        'unpruned_examples': unpruned_examples,
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CPU-only cascade prover (no GPU, no dimension limits)')
    parser.add_argument('--n_half', type=int, default=1,
                        help='Initial n_half (d0 = 2*n_half, default: 1 → d0=2). '
                             'Ignored when --d0 is given.')
    parser.add_argument('--d0', type=int, default=None,
                        help='Starting dimension (overrides n_half, must be >= 2). '
                             'Default is d0=2 (optimal: fewer survivors propagate '
                             'through the cascade than larger starting dimensions).')
    parser.add_argument('--m', type=int, default=20,
                        help='Grid resolution (default: 20)')
    parser.add_argument('--c_target', type=float, default=1.30,
                        help='Target lower bound (default: 1.30)')
    parser.add_argument('--max_levels', type=int, default=10,
                        help='Max refinement levels after L0 (default: 10)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers (default: CPU count)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--resume', nargs='?', const='data', default=None,
                        help='Resume from checkpoint (optionally specify dir, '
                             'default: data)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--use_flat_threshold', action='store_true',
                        help='Use flat C&S Lemma 3 correction (2/m + 1/m^2) '
                             'instead of W-refined.  Required for verifying '
                             'the Lean axiom cascade_all_pruned.  Higher '
                             'threshold = fewer prunes = more survivors.')
    parser.add_argument('--use_F', action='store_true',
                        help='Use variant F pruning (LP-tight linear Δ_BB '
                             'plus standard δ²): empirically prunes 25-65%% '
                             'additional survivors over W-refined.  Sound '
                             'under |a-b|_∞ ≤ 1/m, Σa = 4n, a ≥ 0.  Applied '
                             'as a two-stage post-filter on W-survivors so '
                             'the hot Gray-code kernel is unchanged.  '
                             'Mutually exclusive with --use_flat_threshold.')
    parser.add_argument('--skip_sdp', action='store_true',
                        help='Skip the CVXPY SDP parent cert (50-500ms per '
                             'parent at d_parent ≤ 12, clears <1%% in '
                             'practice).  Theorem-1 + LP cert still run, '
                             'so soundness is preserved.')
    parser.add_argument('--use_Q', action='store_true',
                        help='Apply variant Q (multi-window joint LP) as a '
                             'post-filter on F-survivors.  Sound: F ⊇ Q.  '
                             'Decisive at d ≥ 10 (57-92%% additional pruning '
                             'over F).  Cost ~5-30 ms per LP.  Requires F.')
    parser.add_argument('--use_L', action='store_true',
                        help='Apply variant L (Lasserre/Shor SDP) as a '
                             'post-filter on F→Q survivors.  Sound: F ⊇ Q ⊇ L.  '
                             'Theoretical SDP ceiling: at d=10 prunes 94%% of '
                             'Q-survivors.  Cost 0.5-2 s per SDP.  Requires F.')
    parser.add_argument('--mass_grid', action='store_true',
                        help='Use MATLAB-style mass-based grid with palindrome '
                             'symmetry.  S=2*m (constant), only d_parent/2 '
                             'free cursors.  Dramatically reduces search space.')
    parser.add_argument('--no_l0_bnb', action='store_true',
                        help='Disable L0 branch-and-bound optimization. '
                             'Use the original batch enumeration path.')
    parser.add_argument('--coarse', action='store_true',
                        help='Use coarse-grid mode (Theorem 1, no correction). '
                             'Constant S grid, exact TV lower bound at grid '
                             'points, sound box certification with 2nd-order '
                             'quadratic bound.  Requires --S.')
    parser.add_argument('--S', type=int, default=None,
                        help='Grid resolution for coarse mode (total mass sum). '
                             'Required when --coarse is set.')
    parser.add_argument('--verify_relaxed', action='store_true',
                        help='After cascade completes, verify that +/-1 floor '
                             'rounding variants of children are also pruned. '
                             'Required for soundness of the Lean CascadePruned '
                             'axiom with relaxed is_valid_child.')
    args = parser.parse_args()

    # Validate d0 >= 2.  d0=1 has a degenerate asymmetry prune: left_bins =
    # d//2 = 0, so left_frac is always 0 and ALL compositions are spuriously
    # "asymmetry-pruned," falsely claiming proof for any c_target < 2.
    effective_d0 = args.d0 if args.d0 is not None else 2 * args.n_half
    if effective_d0 < 2:
        parser.error(f'--d0 must be >= 2 (got {effective_d0}).  d0=1 has '
                     'degenerate asymmetry pruning that falsely proves any '
                     'c_target < 2.0.')

    coarse_S = None
    if args.coarse:
        if args.S is None:
            parser.error('--coarse requires --S (grid resolution)')
        coarse_S = args.S

    if args.use_F and args.use_flat_threshold:
        parser.error('--use_F and --use_flat_threshold are mutually exclusive. '
                     'flat is required for the Lean axiom; F is for performance.')
    if args.use_F and coarse_S is not None:
        print('WARNING: --use_F has no effect with --coarse; coarse mode '
              'uses Theorem 1 directly with no m-discretization correction.',
              flush=True)

    if (args.use_Q or args.use_L) and not args.use_F:
        parser.error('--use_Q and --use_L require --use_F (they are '
                     'post-filters on F-survivors).')

    info = run_cascade(
        n_half=args.n_half,
        m=args.m,
        c_target=args.c_target,
        max_levels=args.max_levels,
        n_workers=args.workers,
        verbose=not args.quiet,
        output_dir=args.output_dir,
        resume_dir=args.resume,
        use_flat_threshold=args.use_flat_threshold,
        mass_grid=args.mass_grid,
        d0=args.d0,
        use_bnb=not args.no_l0_bnb,
        coarse_S=coarse_S,
        use_F=args.use_F,
        skip_sdp_cert=args.skip_sdp,
        use_Q=args.use_Q,
        use_L=args.use_L,
    )

    print_summary(info)

    # --- Coarse-grid result summary ---
    if coarse_S is not None and 'proven_at' in info:
        box_ok = info.get('box_certified', False)
        if box_ok:
            _log(f"\nRIGOROUS PROOF: C_{{1a}} >= {args.c_target}")
            _log(f"  (Sound: Theorem 1 + 2nd-order quadratic bound)")
        else:
            _log(f"\nGRID-POINT PROOF: TV >= {args.c_target} at all "
                 f"compositions (box cert incomplete — increase S)")

    # --- Relaxed verification (optional) ---
    if args.verify_relaxed:
        print(f"\n{'='*70}")
        print("RELAXED CHILD VERIFICATION (+/-1 floor rounding)")
        print(f"{'='*70}")
        if not args.use_flat_threshold:
            print("WARNING: --verify_relaxed should be used with --use_flat_threshold "
                  "for Lean axiom soundness.")

        # Load checkpoints for each level and verify relaxed children
        all_relaxed_ok = True
        # Compute effective n_half accounting for --d0 flag
        if args.d0 is not None:
            effective_n_half = args.d0 / 2.0
        else:
            effective_n_half = args.n_half

        # L0 survivors need relaxed verification at the L1 child level
        for level in range(20):  # up to max levels
            ckpt_path = os.path.join(args.output_dir,
                                     f'checkpoint_L{level}_survivors.npy')
            if not os.path.exists(ckpt_path):
                break

            survivors = np.load(ckpt_path)
            if len(survivors) == 0:
                continue

            n_half_parent = effective_n_half * (2 ** level)
            print(f"\n[L{level}] Verifying {len(survivors)} survivors "
                  f"(d_parent={2*n_half_parent})")

            result = verify_relaxed_children(
                survivors, n_half_parent, args.m, args.c_target,
                use_flat_threshold=args.use_flat_threshold,
                verbose=True)

            if not result['all_pruned']:
                all_relaxed_ok = False
                print(f"  FAILED at L{level}!")

        if all_relaxed_ok:
            print(f"\n{'='*70}")
            print("ALL RELAXED VERIFICATIONS PASSED")
            print(f"The CascadePruned axiom with is_valid_child (+/-1 tolerance) is sound.")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print("RELAXED VERIFICATION FAILED — some delta variants have survivors")
            print(f"{'='*70}")

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(args.output_dir,
                        f'cpu_cascade_{ts}.json')

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(info, f, indent=2, default=convert)
    print(f"\nResult saved to {path}")


if __name__ == '__main__':
    main()
