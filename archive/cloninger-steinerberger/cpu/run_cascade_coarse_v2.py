"""Coarse-grid cascade prover — Theorem 1 (no correction at grid points).

v2: SOUND box certification with exact second-order quadratic correction.

=======================================================================
MATHEMATICAL BASIS
=======================================================================
Theorem 1 (Test Value Lower Bound — No Correction):
  For any nonneg f on [-1/4,1/4] with integral 1, partition into d equal
  bins with masses mu_i = integral_{B_i} f.  Then for any window (ell, s):

    max_{|t|<=1/2} (f*f)(t) >= TV_W(mu) = (2d/ell) * sum_{k=s..s+ell-2}
                                                         sum_{i+j=k} mu_i*mu_j

  This is EXACT — no correction term, no step-function approximation.

Refinement Monotonicity (PROVEN):
  For parent mu at d bins and any child nu at 2d bins with
  nu_{2i} + nu_{2i+1} = mu_i:  max_W TV(nu; 2d) >= max_W TV(mu; d).

  Proof: The child window (2*ell, 2*s) covers the same physical interval
  as parent window (ell, s).  Every parent mass product mu_i*mu_j expands
  to four child products ALL within [2s, 2s+2ell-2].  Additional non-neg
  cross-terms only increase the child sum.  QED.

=======================================================================
GRID
=======================================================================
  Integer mass coordinates: k_i >= 0,  sum(k_i) = S  (CONSTANT)
  Physical mass:  mu_i = k_i / S
  S is fixed across all cascade levels (unlike the fine grid where S=2dm).

=======================================================================
INTEGER THRESHOLD (no correction)
=======================================================================
  TV_W = (2d / (ell * S^2)) * ws_int
  where ws_int = sum_{k=s..s+ell-2} sum_{i+j=k} k_i * k_j

  Prune if TV >= c_target:
    ws_int >= c_target * ell * S^2 / (2d)

  Integer threshold:
    thr[ell] = floor(c_target * ell * S^2 / (2d) - eps)
    Prune if ws_int > thr[ell]

  This is a SIMPLE 1D ARRAY indexed by ell.  No per-window W_int needed.

=======================================================================
BOX CERTIFICATION (v2 — with second-order correction)
=======================================================================
  The cascade proves TV >= c at grid points.  For continuous mu between
  grid points, we need: min_{mu in cell} max_W TV_W(mu) >= c.

  Cell: each mu_i in [k_i/S - 1/(2S), k_i/S + 1/(2S)], sum mu_i = 1.
  Write mu = mu* + delta, |delta_i| <= h = 1/(2S), sum delta_i = 0.

  Taylor expansion (EXACT for quadratic TV_W):
    TV_W(mu* + delta) = TV_W(mu*) + grad(TV_W) . delta + (2d/ell) Q(delta)

  where Q(delta) = delta^T A delta, A_{ij} = 1_{s <= i+j <= s+ell-2}.

  IMPORTANT: A is NOT positive semidefinite for all windows.
  (Counterexample: d=2, ell=2, s=1 gives A = [[0,1],[1,0]], eigenvalues +-1.)
  Therefore TV_W is NOT convex, and the second-order term can be NEGATIVE,
  making TV decrease MORE than the first-order prediction.

  --- First-order bound (cell_var) ---

  cell_var = max_{delta in cell} |grad . delta|
           = (1/(2S)) * sum_{k=0..d/2-1} (grad_sorted[d-1-k] - grad_sorted[k])

  where grad_i = (4d/ell) * sum_{j: s <= i+j <= s+ell-2} mu*_j.

  --- Second-order bound (quad_corr) ---

  Decompose c_k = sum_{i+j=k} delta_i delta_j into:
    s_k = delta_{k/2}^2 >= 0   (self-term, exists when k even & k/2 < d)
    x_k = c_k - s_k            (cross-terms)

  Direct bound:  Q = sum_{k in W} (s_k + x_k) >= 0 + (-cross_W * h^2)
  Complement:    Q = -sum_{k not in W} c_k >= -(d^2 - N_W) * h^2

  where:
    n_k = min(k+1, d, 2d-1-k)           (ordered pairs with i+j=k)
    N_W = sum_{k in W} n_k               (total pairs in window)
    M_W = #{k in W : k even, k/2 < d}   (self-terms in window)
    cross_W = N_W - M_W                  (cross-pair count in window)

  Tight bound:  -Q(delta) <= min(cross_W, d^2 - N_W) * h^2

  TV-space second-order correction:
    quad_corr = (2d/ell) * min(cross_W, d^2 - N_W) / (4 S^2)

  --- Sound certification ---

  By triangle inequality:
    max_{delta} |grad.delta + (2d/ell)Q(delta)|
        <= max_{delta} |grad.delta| + max_{delta} |(2d/ell)Q(delta)|
        = cell_var + quad_corr

  Certification:  margin = TV - c_target > cell_var + quad_corr

Usage:
    python -m cloninger-steinerberger.cpu.run_cascade_coarse_v2
    python -m cloninger-steinerberger.cpu.run_cascade_coarse_v2 --d0 3 --S 50 --c_target 1.30
    python -m cloninger-steinerberger.cpu.run_cascade_coarse_v2 --d0 2 --S 200 --c_target 1.20
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import numba
from numba import njit, prange

_this_dir = os.path.dirname(os.path.abspath(__file__))
_cs_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _cs_dir)

from compositions import (generate_compositions_batched,
                          generate_canonical_compositions_batched)
from pruning import count_compositions, asymmetry_threshold, _canonical_mask


def _log(msg):
    print(msg, flush=True)


# =====================================================================
# Pair-count prefix sums for second-order bound (depends only on d)
# =====================================================================

@njit(cache=True)
def _build_pair_prefix(d):
    """Build prefix sums for n_k (pair counts) and m_k (self-term indicators).

    n_k = #{(i,j): 0<=i,j<d, i+j=k} = min(k+1, d, 2d-1-k)
    m_k = 1 if k even and k//2 < d, else 0

    Returns (prefix_nk, prefix_mk) each of length conv_len+1 = 2d.
    prefix_nk[i] = sum(n_k for k in 0..i-1).
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


@njit(cache=True)
def _quad_corr_for_window(s_lo, ell, d, prefix_nk, prefix_mk, inv_4S2):
    """Compute the second-order quadratic correction for window (ell, s_lo).

    quad_corr = (2d/ell) * min(cross_W, d^2 - N_W) / (4*S^2)

    where cross_W = N_W - M_W (cross-term pairs in window).
    """
    # Window covers conv indices s_lo to s_lo + ell - 2 (inclusive, ell-1 values)
    hi = s_lo + ell - 1  # exclusive upper bound for prefix sum
    N_W = prefix_nk[hi] - prefix_nk[s_lo]
    M_W = prefix_mk[hi] - prefix_mk[s_lo]
    cross_W = N_W - M_W
    d_sq = np.int64(d) * np.int64(d)
    compl_bound = d_sq - N_W

    pair_bound = min(cross_W, compl_bound)
    if pair_bound <= 0:
        return 0.0

    ell_f = np.float64(ell)
    d_f = np.float64(d)
    return (2.0 * d_f / ell_f) * np.float64(pair_bound) * inv_4S2


# =====================================================================
# Pruning kernel — NO correction at grid points, SOUND box cert
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_no_correction(batch_int, d, S, c_target):
    """Prune coarse-grid compositions by Theorem 1 (no correction).

    Prune if exists window (ell, s) with TV_W >= c_target.
    Integer: ws > floor(c_target * ell * S^2 / (2*d) - eps).

    Returns (survived_mask, min_cert_net).
    min_cert_net: smallest (margin - cell_var - quad_corr) across pruned comps.
    If min_cert_net >= 0: all pruned cells are box-certified.
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
    d_minus_1 = d - 1

    # Precompute threshold per ell (no correction — simple 1D array)
    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d
                                - eps)

    # Precompute pair-count prefix sums for second-order bound
    prefix_nk, prefix_mk = _build_pair_prefix(d)

    # Per-thread tracking of min(margin - cell_var - quad_corr)
    n_threads = numba.config.NUMBA_NUM_THREADS
    min_net_arr = np.full(n_threads, 1e30, dtype=np.float64)

    for b in prange(B):
        tid = numba.get_thread_id()

        # Autoconvolution
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        # Window scan — check ALL windows, find best (margin - cell_var - quad_corr)
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
                    # Compute margin
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    # Compute gradient for this window
                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    # Sort (insertion sort, fine for d<=64)
                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key

                    # Cell var: first-order bound (pair extremes)
                    cell_var = 0.0
                    for k in range(d // 2):
                        cell_var += grad_arr[d - 1 - k] - grad_arr[k]
                    cell_var /= (2.0 * S_d)

                    # Second-order quadratic correction
                    qc = _quad_corr_for_window(
                        s_lo, ell, d, prefix_nk, prefix_mk, inv_4S2)

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net

        if pruned:
            survived[b] = False
            if best_net < min_net_arr[tid]:
                min_net_arr[tid] = best_net

    # Reduce across threads
    global_min_net = 1e30
    for t in range(n_threads):
        if min_net_arr[t] < global_min_net:
            global_min_net = min_net_arr[t]

    return survived, global_min_net


# =====================================================================
# Asymmetry pruning (exact on coarse grid, no correction needed)
# =====================================================================

def asymmetry_prune_mask_coarse(batch_int, S, c_target):
    """Asymmetry covers: if left_frac >= sqrt(c/2) or <= 1-sqrt(c/2)."""
    d = batch_int.shape[1]
    threshold = asymmetry_threshold(c_target)
    left_bins = d // 2
    left = batch_int[:, :left_bins].sum(axis=1).astype(np.float64)
    left_frac = left / float(S)
    needs_check = (left_frac > 1 - threshold) & (left_frac < threshold)
    return needs_check


# =====================================================================
# Per-bin mass cap
# =====================================================================

def coarse_x_cap(d, S, c_target):
    """Max integer mass per bin from the TV ell=2 self-convolution bound.

    TV(ell=2, self-conv of bin i) = d * k_i^2 / S^2.
    Prune if d * k_i^2 / S^2 >= c_target.
    x_cap = floor(S * sqrt(c_target / d)).
    """
    return int(math.floor(S * math.sqrt(c_target / d)))


# =====================================================================
# Fused generate-and-prune (cascade levels)
# =====================================================================

@njit(cache=True)
def _fused_coarse(parent_int, d_child, S, c_target, lo_arr, hi_arr,
                  out_buf):
    """Generate children of one parent and prune inline.  No correction
    at grid points; sound second-order box certification.

    Returns (n_survivors, n_tested, min_cert_net).
    """
    d_parent = parent_int.shape[0]
    conv_len = 2 * d_child - 1

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d_child)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d_child
    d_minus_1 = d_child - 1

    # Precompute threshold per ell
    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d
                                - eps)

    # Precompute pair-count prefix sums for second-order bound
    prefix_nk, prefix_mk = _build_pair_prefix(d_child)

    max_survivors = out_buf.shape[0]
    n_surv = 0
    n_tested = np.int64(0)
    local_min_net = np.float64(1e30)

    cursor = np.empty(d_parent, dtype=np.int32)
    child = np.empty(d_child, dtype=np.int32)
    conv = np.empty(conv_len, dtype=np.int32)

    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    while True:
        # Build child: child[2i] = cursor[i], child[2i+1] = parent[i] - cursor[i]
        for i in range(d_parent):
            child[2 * i] = cursor[i]
            child[2 * i + 1] = parent_int[i] - cursor[i]

        n_tested += 1

        # Full autoconvolution
        for k in range(conv_len):
            conv[k] = np.int32(0)
        for i in range(d_child):
            ci = np.int32(child[i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d_child):
                    cj = np.int32(child[j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        # Window scan — check ALL windows, find best cert net
        pruned = False
        best_net = np.float64(-1e30)
        grad_buf = np.empty(d_child, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            if pruned:
                break
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

                    for i in range(d_child):
                        g = 0.0
                        for j in range(d_child):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(child[j]) / S_d
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
                        cell_var += grad_buf[d_child - 1 - kk] - grad_buf[kk]
                    cell_var /= (2.0 * S_d)

                    # Second-order quadratic correction
                    qc = _quad_corr_for_window(
                        s_lo, ell, d_child, prefix_nk, prefix_mk, inv_4S2)

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net
                    break

        if pruned:
            if best_net < local_min_net:
                local_min_net = best_net
        else:
            if n_surv < max_survivors:
                for i in range(d_child):
                    out_buf[n_surv, i] = child[i]
            n_surv += 1

        # Advance odometer
        carry = d_parent - 1
        while carry >= 0:
            cursor[carry] += 1
            if cursor[carry] <= hi_arr[carry]:
                break
            cursor[carry] = lo_arr[carry]
            carry -= 1
        if carry < 0:
            break

    return n_surv, n_tested, local_min_net


# =====================================================================
# Level 0
# =====================================================================

def run_level0(d0, S, c_target, verbose=True):
    """L0: enumerate compositions of S into d0 bins, prune by TV >= c."""
    n_total = count_compositions(d0, S)

    if verbose:
        _log(f"\n[L0] d={d0}, S={S}, compositions={n_total:,}")
        _log(f"     x_cap = {coarse_x_cap(d0, S, c_target)}")

    t0 = time.time()
    all_survivors = []
    n_pruned_asym = 0
    n_pruned_test = 0
    n_processed = 0
    global_min_net = 1e30
    last_report = t0

    gen = generate_canonical_compositions_batched(d0, S, batch_size=500_000)

    for batch in gen:
        n_processed += len(batch)

        # Asymmetry pruning
        asym_mask = asymmetry_prune_mask_coarse(batch, S, c_target)
        n_pruned_asym += int(np.sum(~asym_mask))
        batch = batch[asym_mask]

        if len(batch) == 0:
            continue

        # Window scan (no correction)
        survived_mask, min_net = _prune_no_correction(
            batch, d0, S, c_target)
        n_pruned_test += int(np.sum(~survived_mask))

        if min_net < global_min_net:
            global_min_net = min_net

        survivors = batch[survived_mask]
        if len(survivors) > 0:
            all_survivors.append(survivors)

        now = time.time()
        if verbose and (now - last_report >= 2.0):
            pct = n_processed / max(1, n_total) * 100
            n_surv = sum(len(s) for s in all_survivors)
            _log(f"     [{n_processed:,} processed] {n_surv:,} survivors")
            last_report = now

    elapsed = time.time() - t0

    if all_survivors:
        all_survivors = np.vstack(all_survivors)
    else:
        all_survivors = np.empty((0, d0), dtype=np.int32)

    n_survivors = len(all_survivors)
    proven = n_survivors == 0

    box_ok = global_min_net >= 0.0

    if verbose:
        _log(f"     {elapsed:.2f}s: pruned asym={n_pruned_asym:,} "
             f"test={n_pruned_test:,} survivors={n_survivors:,}")
        if n_pruned_test > 0:
            _log(f"     min(margin - cell_var - quad_corr) = "
                 f"{global_min_net:.6f}")
            if box_ok:
                _log(f"     BOX CERT: PASS (sound, with 2nd-order bound)")
            else:
                _log(f"     BOX CERT: FAIL (increase S)")
        if proven:
            _log(f"     PROVEN at L0!")

    return {
        'survivors': all_survivors,
        'n_survivors': n_survivors,
        'n_pruned_asym': n_pruned_asym,
        'n_pruned_test': n_pruned_test,
        'elapsed': elapsed,
        'proven': proven,
        'min_cert_net': global_min_net,
        'box_certified': box_ok,
    }


def _suggest_S(d, min_margin):
    """Suggest S needed for cell_var + quad_corr < margin."""
    if min_margin <= 0:
        return 99999
    return int(math.ceil(d / min_margin)) * 2


# =====================================================================
# Process one parent
# =====================================================================

def process_parent(parent_int, d_child, S, c_target, buf_cap=100_000):
    """Generate children of one parent, prune inline."""
    d_parent = len(parent_int)
    x_cap = coarse_x_cap(d_child, S, c_target)

    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total_product = 1

    for i in range(d_parent):
        p = int(parent_int[i])
        lo = max(0, p - x_cap)
        hi = min(p, x_cap)
        if lo > hi:
            return np.empty((0, d_child), dtype=np.int32), 0, 1e30
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_product *= (hi - lo + 1)

    if total_product == 0:
        return np.empty((0, d_child), dtype=np.int32), 0, 1e30

    out_buf = np.empty((min(total_product, buf_cap), d_child), dtype=np.int32)

    n_surv, n_tested, min_net = _fused_coarse(
        parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf)

    if n_surv > buf_cap:
        out_buf = np.empty((n_surv, d_child), dtype=np.int32)
        n2, _, min_net = _fused_coarse(
            parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf)
        assert n2 == n_surv
        n_surv = n2

    return out_buf[:n_surv].copy(), n_tested, min_net


def _worker(args):
    parent, d_child, S, c_target, buf_cap = args
    surv, n_t, m_net = process_parent(parent, d_child, S, c_target, buf_cap)
    return surv, n_t, m_net


# =====================================================================
# Full cascade
# =====================================================================

def run_cascade(d0, S, c_target, max_levels=10, n_workers=None,
                verbose=True):
    """Run the coarse-grid cascade with no correction (Theorem 1).

    Box certification uses sound second-order quadratic bound.
    Reports box certification status at each level.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)

    n_total_comps = count_compositions(d0, S)

    if verbose:
        _log(f"\n{'='*70}")
        _log(f"COARSE CASCADE PROVER v2 (Theorem 1 + sound box cert)")
        _log(f"  d0={d0}, S={S}, c_target={c_target}")
        _log(f"  Grid: integer masses sum to S={S} (constant)")
        _log(f"  Threshold: TV >= {c_target} (exact, no correction)")
        _log(f"  Box cert: margin > cell_var + quad_corr (2nd-order bound)")
        _log(f"  L0 compositions (canonical): ~{n_total_comps // 2:,}")
        _log(f"  workers: {n_workers}")
        _log(f"{'='*70}")

    t_total = time.time()
    info = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'grid': 'coarse', 'correction': 'none (grid-point)',
        'box_cert_method': 'first_order + second_order_quad_bound',
        'levels': [],
    }

    # --- Level 0 ---
    l0 = run_level0(d0, S, c_target, verbose=verbose)
    info['l0'] = {
        'survivors': l0['n_survivors'],
        'time': l0['elapsed'],
        'min_cert_net': l0['min_cert_net'],
        'box_certified': l0['box_certified'],
    }

    if l0['proven']:
        info['proven_at'] = 'L0'
        info['box_certified'] = l0['box_certified']
        info['total_time'] = time.time() - t_total
        if verbose:
            cert = l0['box_certified']
            _log(f"\n*** GRID-POINT PROOF at L0 ***")
            _log(f"    Box certification: {'PASS' if cert else 'FAIL'}")
        return info

    current = l0['survivors']
    d_parent = d0

    # --- Cascade ---
    for level in range(1, max_levels + 1):
        d_child = 2 * d_parent
        n_parents = len(current)

        if n_parents == 0:
            break

        x_cap = coarse_x_cap(d_child, S, c_target)

        # Report expected quad_corr scale at this level
        d2_over_2S2 = float(d_child * d_child) / (2.0 * S * S)

        if verbose:
            _log(f"\n--- Level {level}: d={d_parent} -> {d_child}, "
                 f"parents={n_parents:,}, x_cap={x_cap} ---")
            _log(f"    quad_corr scale: d^2/(2S^2) = {d2_over_2S2:.6f}")

        # Pre-filter infeasible parents: parent bin p needs p <= 2*x_cap
        # (child splits into (c, p-c) with both in [0, x_cap], so p <= 2*x_cap)
        feasible = np.all(current <= 2 * x_cap, axis=1)
        n_infeasible = n_parents - int(feasible.sum())
        if n_infeasible > 0:
            current = np.ascontiguousarray(current[feasible])
            n_parents = len(current)
            if verbose:
                _log(f"    Pre-filtered {n_infeasible:,} infeasible parents")

        if n_parents == 0:
            if verbose:
                _log(f"    All parents infeasible -> 0 survivors")
            current = np.empty((0, d_child), dtype=np.int32)
            d_parent = d_child
            info['levels'].append({
                'level': level, 'd_child': d_child,
                'parents': 0, 'children': 0, 'survivors': 0,
            })
            continue

        t_level = time.time()
        all_survivors = []
        total_children = 0
        level_min_net = 1e30
        n_done = 0
        last_report = time.time()

        buf_cap = max(100_000, 2_000_000 // max(1, d_child))

        if n_workers > 1 and n_parents > 4:
            args = [(current[i], d_child, S, c_target, buf_cap)
                    for i in range(n_parents)]
            ctx = mp.get_context('spawn')
            chunk = max(1, n_parents // (n_workers * 4))
            with ctx.Pool(n_workers) as pool:
                for surv, n_t, m_net in pool.imap_unordered(
                        _worker, args, chunksize=chunk):
                    total_children += n_t
                    if len(surv) > 0:
                        all_survivors.append(surv)
                    if m_net < level_min_net:
                        level_min_net = m_net
                    n_done += 1
                    now = time.time()
                    if verbose and (now - last_report >= 5.0):
                        ns = sum(len(s) for s in all_survivors)
                        _log(f"    [{n_done}/{n_parents}] "
                             f"children={total_children:,} "
                             f"survivors={ns:,}")
                        last_report = now
        else:
            for i in range(n_parents):
                surv, n_t, m_net = process_parent(
                    current[i], d_child, S, c_target, buf_cap)
                total_children += n_t
                if len(surv) > 0:
                    all_survivors.append(surv)
                if m_net < level_min_net:
                    level_min_net = m_net
                n_done += 1
                now = time.time()
                if verbose and (now - last_report >= 5.0):
                    ns = sum(len(s) for s in all_survivors)
                    _log(f"    [{n_done}/{n_parents}] "
                         f"children={total_children:,} "
                         f"survivors={ns:,}")
                    last_report = now

        elapsed_level = time.time() - t_level

        if all_survivors:
            current = np.vstack(all_survivors)
            # Map non-canonical survivors to their canonical form (reverse)
            # instead of discarding them — they are children of the unexpanded
            # reverse parent and would otherwise be permanently lost.
            canon = _canonical_mask(current)
            non_canon = ~canon
            current[non_canon] = current[non_canon, ::-1]
            # Deduplicate: a canonical child and a non-canonical child that
            # maps to the same canonical form can both appear.
            current = np.unique(current, axis=0)
        else:
            current = np.empty((0, d_child), dtype=np.int32)

        n_survivors = len(current)
        rate = n_survivors / max(1, total_children) * 100
        box_ok = level_min_net >= 0.0

        if verbose:
            _log(f"    {elapsed_level:.1f}s: {total_children:,} children, "
                 f"{n_survivors:,} survivors ({rate:.4f}%)")
            if total_children > 0:
                _log(f"    min(margin - cell_var - quad_corr) = "
                     f"{level_min_net:.6f}")
                if box_ok:
                    _log(f"    BOX CERT: PASS (sound)")
                else:
                    _log(f"    BOX CERT: FAIL (min net = {level_min_net:.6f})")
                    sugg_S = _suggest_S(d_child, level_min_net)
                    _log(f"    Suggested S >= {sugg_S} for this level")
            if n_survivors == 0:
                _log(f"    *** ALL PRUNED ***")

        info['levels'].append({
            'level': level, 'd_child': d_child,
            'parents': n_parents,
            'children': int(total_children),
            'survivors': n_survivors,
            'rate': rate,
            'time': elapsed_level,
            'min_cert_net': level_min_net,
            'box_certified': box_ok,
        })

        if n_survivors == 0:
            info['proven_at'] = f'L{level}'
            info['total_time'] = time.time() - t_total

            # Check if ALL levels (including L0) have box certification
            all_cert = info['l0'].get('box_certified', False)
            if all_cert:
                for lv in info['levels']:
                    if not lv.get('box_certified', False):
                        all_cert = False
                        break
            info['box_certified'] = all_cert

            if verbose:
                _log(f"\n*** GRID-POINT PROOF COMPLETE: "
                     f"TV >= {c_target} for all compositions ***")
                if all_cert:
                    _log(f"*** BOX CERTIFICATION: ALL LEVELS PASS ***")
                    _log(f"*** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
                    _log(f"    (Sound: includes 2nd-order quadratic bound)")
                else:
                    _log(f"    Box certification incomplete at some levels.")
                    _log(f"    Increase S or use two-phase approach.")
                _log(f"    Total time: {info['total_time']:.1f}s")
            return info

        d_parent = d_child

    info['total_time'] = time.time() - t_total
    info['final_survivors'] = len(current)
    if verbose:
        _log(f"\nCascade did not converge ({len(current):,} survivors "
             f"at d={d_parent})")
    return info


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Coarse-grid cascade prover v2 '
                    '(Theorem 1, sound box cert with 2nd-order bound)')
    parser.add_argument('--d0', type=int, default=2,
                        help='Starting dimension (default: 2)')
    parser.add_argument('--S', type=int, default=50,
                        help='Grid resolution / mass sum (default: 50)')
    parser.add_argument('--c_target', type=float, default=1.30,
                        help='Target lower bound (default: 1.30)')
    parser.add_argument('--max_levels', type=int, default=10,
                        help='Max cascade levels (default: 10)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Parallel workers (default: auto)')
    args = parser.parse_args()

    result = run_cascade(
        d0=args.d0, S=args.S, c_target=args.c_target,
        max_levels=args.max_levels, n_workers=args.n_workers)

    if 'proven_at' in result:
        if result.get('box_certified'):
            _log(f"\nRIGOROUS PROOF: C_{{1a}} >= {args.c_target}")
            _log(f"  (Sound box cert with 2nd-order quadratic bound)")
        else:
            _log(f"\nGRID-POINT PROOF: TV >= {args.c_target} at all "
                 f"compositions (box cert incomplete)")
    else:
        _log(f"\nNOT PROVEN (cascade did not converge)")


if __name__ == '__main__':
    main()
