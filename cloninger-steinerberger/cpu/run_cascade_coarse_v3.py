"""Coarse-grid cascade prover v3 — Theorem 1 + N+O sound box cert.

v3 = v2 with two SOUND tightenings of the box-cert bound:
  - Variant N (spectral δ²-bound):   quad_corr ← min(cross_W, d²-N_W, op_rest·d) · h²
                                     where op_rest = ‖A_W − (N_W/d²)·11ᵀ‖_op.
                                     Sound: 1ᵀδ=0 ⇒ δᵀA_Wδ = δᵀ(A_W−α11ᵀ)δ.
  - Variant O (one-sided LP):        cell_var ← LP over Cell ∩ {δ_j≥0 if k_j=0}.
                                     Sound: smaller polytope ⇒ smaller LP optimum.

Both bounds are taken as the *min* with the v2 baseline so v3 ⊆ v2 (no regression).

Empirical pruning gain (vs v2 triangle baseline) at fixed grid:
  d=10/S=15/c=1.20:  22.34% extra cells closed
  d=10/S=12/c=1.20:  18.41%
  d=8/S=20/c=1.20:   11.48%
  d=8/S=12/c=1.20:   14.65% (combined N+O orthogonality bonus)

Usage:
    python -m cloninger-steinerberger.cpu.run_cascade_coarse_v3 --d0 2 --S 200 --c_target 1.20
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

# Reuse helpers we don't change
from run_cascade_coarse_v2 import (asymmetry_prune_mask_coarse,
                                     coarse_x_cap, _suggest_S, _build_pair_prefix)


def _log(msg):
    print(msg, flush=True)


# =====================================================================
# Variant N pre-compute: op_rest_d table
# =====================================================================

def precompute_op_rest_d(d):
    """For each (ell, s_lo), op_rest = ‖A_W − (N_W/d²)·11ᵀ‖_op.

    Sound: 1ᵀδ=0 ⇒ |δᵀA_Wδ| ≤ op_rest · ‖δ‖² ≤ op_rest · d · h².
    Returned table is op_rest * d (the bound on Σ pair contributions).
    """
    conv_len = 2 * d - 1
    max_ell = 2 * d
    op_rest_d = np.zeros((max_ell + 1, conv_len), dtype=np.float64)
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            A = np.zeros((d, d), dtype=np.float64)
            for i in range(d):
                for j in range(d):
                    if s_lo <= i + j <= s_lo + ell - 2:
                        A[i, j] = 1.0
            n = int(A.sum())
            if n == 0:
                continue
            alpha = n / float(d * d)
            eigs = np.linalg.eigvalsh(A - alpha)
            op_rest_d[ell, s_lo] = float(np.abs(eigs).max()) * d
    return op_rest_d


# =====================================================================
# Variant O helper: one-sided LP closed form (vertex enumeration).
# =====================================================================

@njit(cache=True, inline='always')
def _one_sided_lp(grad, k_int, d, h, work_buf):
    """cell_var_O = max{ Σ δ_j · grad_j : |δ_j|≤h, δ_j≥0 if k_j=0, Σδ=0 }.
    work_buf: int64 scratch length 4*d.  We sort indices by signed grad;
    at level k=1..min(|T|, d/2) pair top-k positives with bottom-k of T.
    """
    best = 0.0
    for sign_iter in range(2):
        # work_buf [0..d): T_asc; [d..2d): A_desc; [2*d..3d): used flag
        T_count = 0
        for j in range(d):
            if k_int[j] > 0:
                key = grad[j] if sign_iter == 0 else -grad[j]
                pos = T_count
                while pos > 0:
                    prev_idx = work_buf[pos - 1]
                    prev_key = grad[prev_idx] if sign_iter == 0 else -grad[prev_idx]
                    if prev_key > key:
                        work_buf[pos] = work_buf[pos - 1]
                        pos -= 1
                    else:
                        break
                work_buf[pos] = j
                T_count += 1
        A_off = d
        for j in range(d):
            key = grad[j] if sign_iter == 0 else -grad[j]
            pos = j
            while pos > 0:
                prev_idx = work_buf[A_off + pos - 1]
                prev_key = grad[prev_idx] if sign_iter == 0 else -grad[prev_idx]
                if prev_key < key:
                    work_buf[A_off + pos] = work_buf[A_off + pos - 1]
                    pos -= 1
                else:
                    break
            work_buf[A_off + pos] = j

        k_max = T_count if T_count < (d // 2) else (d // 2)
        used_off = 2 * d
        for j in range(d):
            work_buf[used_off + j] = 0

        neg_sum = 0.0
        for k in range(1, k_max + 1):
            new_neg = work_buf[k - 1]
            new_neg_signed = grad[new_neg] if sign_iter == 0 else -grad[new_neg]
            neg_sum += new_neg_signed
            work_buf[used_off + new_neg] = 1

            pos_sum = 0.0
            found = 0
            for jj in range(d):
                idx = work_buf[A_off + jj]
                if work_buf[used_off + idx] == 0:
                    pos_signed = grad[idx] if sign_iter == 0 else -grad[idx]
                    pos_sum += pos_signed
                    found += 1
                    if found == k:
                        break
            if found < k:
                break
            val = h * (pos_sum - neg_sum)
            if val > best:
                best = val
    return best


# =====================================================================
# v3 quad_corr: min(cross_W, d²-N_W, op_rest_d) · h² · (2d/ell)
# =====================================================================

@njit(cache=True, inline='always')
def _quad_corr_v3(s_lo, ell, d, prefix_nk, prefix_mk, op_rest_d_arr, inv_4S2):
    hi = s_lo + ell - 1
    N_W = prefix_nk[hi] - prefix_nk[s_lo]
    M_W = prefix_mk[hi] - prefix_mk[s_lo]
    cross_W = N_W - M_W
    d_sq = np.int64(d) * np.int64(d)
    compl_bound = d_sq - N_W
    pair_bound = cross_W
    if compl_bound < pair_bound:
        pair_bound = compl_bound
    spec_bound_int = np.int64(np.ceil(op_rest_d_arr[ell, s_lo]))
    if spec_bound_int < pair_bound:
        pair_bound = spec_bound_int
    if pair_bound <= 0:
        return 0.0
    return (2.0 * np.float64(d) / np.float64(ell)
            * np.float64(pair_bound) * inv_4S2)


# =====================================================================
# v3 L0 prune kernel: N+O combined, returns survived mask + min_net.
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_no_correction_v3(batch_int, d, S, c_target, op_rest_d_arr):
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
    h = 1.0 / (2.0 * S_d)

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    n_threads = numba.config.NUMBA_NUM_THREADS
    min_net_arr = np.full(n_threads, 1e30, dtype=np.float64)

    for b in prange(B):
        tid = numba.get_thread_id()

        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        pruned = False
        best_net = np.float64(-1e30)
        grad_arr = np.empty(d, dtype=np.float64)
        k_local = np.empty(d, dtype=np.int64)
        for j in range(d):
            k_local[j] = np.int64(batch_int[b, j])
        work_buf = np.empty(4 * d, dtype=np.int64)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
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

                    # cell_var_O (variant O sparsity-aware LP)
                    cell_var_O = _one_sided_lp(grad_arr, k_local, d, h, work_buf)

                    # Compute baseline cell_var (sort + extreme pairs); FP-safety clip.
                    grad_sorted = np.empty(d, dtype=np.float64)
                    for i in range(d):
                        grad_sorted[i] = grad_arr[i]
                    for i in range(1, d):
                        key = grad_sorted[i]
                        jj = i - 1
                        while jj >= 0 and grad_sorted[jj] > key:
                            grad_sorted[jj + 1] = grad_sorted[jj]
                            jj -= 1
                        grad_sorted[jj + 1] = key
                    cell_var_BL = 0.0
                    for kk in range(d // 2):
                        cell_var_BL += grad_sorted[d - 1 - kk] - grad_sorted[kk]
                    cell_var_BL *= h
                    if cell_var_O > cell_var_BL:
                        cell_var_O = cell_var_BL

                    # quad_corr_v3 (variant N spectral floor)
                    qc = _quad_corr_v3(s_lo, ell, d, prefix_nk, prefix_mk,
                                        op_rest_d_arr, inv_4S2)

                    net = margin - cell_var_O - qc
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
# v3 fused gen-and-prune (cascade levels)
# =====================================================================

@njit(cache=True)
def _fused_coarse_v3(parent_int, d_child, S, c_target, lo_arr, hi_arr,
                       out_buf, op_rest_d_arr):
    d_parent = parent_int.shape[0]
    conv_len = 2 * d_child - 1

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d_child)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d_child
    h = 1.0 / (2.0 * S_d)

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d_child)

    max_survivors = out_buf.shape[0]
    n_surv = 0
    n_tested = np.int64(0)
    local_min_net = np.float64(1e30)

    cursor = np.empty(d_parent, dtype=np.int32)
    child = np.empty(d_child, dtype=np.int32)
    conv = np.empty(conv_len, dtype=np.int32)
    grad_buf = np.empty(d_child, dtype=np.float64)
    grad_sorted = np.empty(d_child, dtype=np.float64)
    k_local = np.empty(d_child, dtype=np.int64)
    work_buf = np.empty(4 * d_child, dtype=np.int64)

    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    while True:
        for i in range(d_parent):
            child[2 * i] = cursor[i]
            child[2 * i + 1] = parent_int[i] - cursor[i]
        n_tested += 1

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

        for j in range(d_child):
            k_local[j] = np.int64(child[j])

        pruned = False
        best_net = np.float64(-1e30)

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
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
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

                    cell_var_O = _one_sided_lp(grad_buf, k_local, d_child, h, work_buf)

                    for i in range(d_child):
                        grad_sorted[i] = grad_buf[i]
                    for i in range(1, d_child):
                        key = grad_sorted[i]
                        jj = i - 1
                        while jj >= 0 and grad_sorted[jj] > key:
                            grad_sorted[jj + 1] = grad_sorted[jj]
                            jj -= 1
                        grad_sorted[jj + 1] = key
                    cell_var_BL = 0.0
                    for kk in range(d_child // 2):
                        cell_var_BL += grad_sorted[d_child - 1 - kk] - grad_sorted[kk]
                    cell_var_BL *= h
                    if cell_var_O > cell_var_BL:
                        cell_var_O = cell_var_BL

                    qc = _quad_corr_v3(s_lo, ell, d_child, prefix_nk, prefix_mk,
                                        op_rest_d_arr, inv_4S2)

                    net = margin - cell_var_O - qc
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

def run_level0(d0, S, c_target, op_rest_d_arr, verbose=True):
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
        asym_mask = asymmetry_prune_mask_coarse(batch, S, c_target)
        n_pruned_asym += int(np.sum(~asym_mask))
        batch = batch[asym_mask]
        if len(batch) == 0:
            continue

        survived_mask, min_net = _prune_no_correction_v3(
            batch, d0, S, c_target, op_rest_d_arr)
        n_pruned_test += int(np.sum(~survived_mask))
        if min_net < global_min_net:
            global_min_net = min_net
        survivors = batch[survived_mask]
        if len(survivors) > 0:
            all_survivors.append(survivors)
        now = time.time()
        if verbose and (now - last_report >= 2.0):
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
            _log(f"     min(margin - cell_var_O - quad_corr_v3) = {global_min_net:.6f}")
            if box_ok:
                _log(f"     BOX CERT v3: PASS (sound, N+O combined bound)")
            else:
                _log(f"     BOX CERT v3: FAIL (increase S)")
        if proven:
            _log(f"     PROVEN at L0!")
    return {
        'survivors': all_survivors, 'n_survivors': n_survivors,
        'n_pruned_asym': n_pruned_asym, 'n_pruned_test': n_pruned_test,
        'elapsed': elapsed, 'proven': proven,
        'min_cert_net': global_min_net, 'box_certified': box_ok,
    }


def process_parent(parent_int, d_child, S, c_target, op_rest_d_arr,
                   buf_cap=100_000):
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
    n_surv, n_tested, min_net = _fused_coarse_v3(
        parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf, op_rest_d_arr)
    if n_surv > buf_cap:
        out_buf = np.empty((n_surv, d_child), dtype=np.int32)
        n2, _, min_net = _fused_coarse_v3(
            parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf, op_rest_d_arr)
        assert n2 == n_surv
        n_surv = n2
    return out_buf[:n_surv].copy(), n_tested, min_net


def _worker(args):
    parent, d_child, S, c_target, op_rest_d_arr, buf_cap = args
    surv, n_t, m_net = process_parent(
        parent, d_child, S, c_target, op_rest_d_arr, buf_cap)
    return surv, n_t, m_net


# =====================================================================
# Full cascade v3
# =====================================================================

def run_cascade(d0, S, c_target, max_levels=10, n_workers=None, verbose=True):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)

    n_total_comps = count_compositions(d0, S)
    if verbose:
        _log(f"\n{'='*70}")
        _log(f"COARSE CASCADE PROVER v3 (Theorem 1 + N+O sound box cert)")
        _log(f"  d0={d0}, S={S}, c_target={c_target}")
        _log(f"  Box cert: margin > cell_var_O + quad_corr_v3 (N spectral + O sparsity)")
        _log(f"  L0 compositions (canonical): ~{n_total_comps // 2:,}")
        _log(f"  workers: {n_workers}")
        _log(f"{'='*70}")

    t_total = time.time()
    info = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'grid': 'coarse', 'correction': 'none (grid-point)',
        'box_cert_method': 'v3_N_spectral_plus_O_sparsity',
        'levels': [],
    }

    # Pre-compute op_rest_d for d0 and each cascade dim
    op_rest_d_cache = {d0: precompute_op_rest_d(d0)}

    l0 = run_level0(d0, S, c_target, op_rest_d_cache[d0], verbose=verbose)
    info['l0'] = {
        'survivors': l0['n_survivors'], 'time': l0['elapsed'],
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
    for level in range(1, max_levels + 1):
        d_child = 2 * d_parent
        n_parents = len(current)
        if n_parents == 0:
            break
        x_cap = coarse_x_cap(d_child, S, c_target)
        d2_over_2S2 = float(d_child * d_child) / (2.0 * S * S)
        if verbose:
            _log(f"\n--- Level {level}: d={d_parent} -> {d_child}, "
                 f"parents={n_parents:,}, x_cap={x_cap} ---")
            _log(f"    quad_corr scale: d^2/(2S^2) = {d2_over_2S2:.6f}")

        if d_child not in op_rest_d_cache:
            t_pre = time.time()
            op_rest_d_cache[d_child] = precompute_op_rest_d(d_child)
            if verbose:
                _log(f"    op_rest_d table for d={d_child}: "
                     f"{time.time()-t_pre:.2f}s")
        ord_d = op_rest_d_cache[d_child]

        feasible = np.all(current <= 2 * x_cap, axis=1)
        n_infeasible = n_parents - int(feasible.sum())
        if n_infeasible > 0:
            current = np.ascontiguousarray(current[feasible])
            n_parents = len(current)
            if verbose:
                _log(f"    Pre-filtered {n_infeasible:,} infeasible parents")
        if n_parents == 0:
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
            args = [(current[i], d_child, S, c_target, ord_d, buf_cap)
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
                             f"children={total_children:,} survivors={ns:,}")
                        last_report = now
        else:
            for i in range(n_parents):
                surv, n_t, m_net = process_parent(
                    current[i], d_child, S, c_target, ord_d, buf_cap)
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
                         f"children={total_children:,} survivors={ns:,}")
                    last_report = now

        elapsed_level = time.time() - t_level
        if all_survivors:
            current = np.vstack(all_survivors)
            canon = _canonical_mask(current)
            non_canon = ~canon
            current[non_canon] = current[non_canon, ::-1]
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
                _log(f"    min(margin - cell_var_O - quad_corr_v3) = "
                     f"{level_min_net:.6f}")
                if box_ok:
                    _log(f"    BOX CERT v3: PASS (sound)")
                else:
                    _log(f"    BOX CERT v3: FAIL (min net = {level_min_net:.6f})")
                    sugg_S = _suggest_S(d_child, level_min_net)
                    _log(f"    Suggested S >= {sugg_S} for this level")
            if n_survivors == 0:
                _log(f"    *** ALL PRUNED ***")
        info['levels'].append({
            'level': level, 'd_child': d_child,
            'parents': n_parents, 'children': int(total_children),
            'survivors': n_survivors, 'rate': rate,
            'time': elapsed_level, 'min_cert_net': level_min_net,
            'box_certified': box_ok,
        })
        if n_survivors == 0:
            info['proven_at'] = f'L{level}'
            info['total_time'] = time.time() - t_total
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
                    _log(f"*** BOX CERTIFICATION v3: ALL LEVELS PASS ***")
                    _log(f"*** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
                    _log(f"    (Sound: N spectral + O sparsity)")
                _log(f"    Total time: {info['total_time']:.1f}s")
            return info
        d_parent = d_child

    info['total_time'] = time.time() - t_total
    info['final_survivors'] = len(current)
    if verbose:
        _log(f"\nCascade did not converge ({len(current):,} survivors "
             f"at d={d_parent})")
    return info


def main():
    parser = argparse.ArgumentParser(
        description='Coarse-grid cascade prover v3 (Theorem 1 + N+O box cert)')
    parser.add_argument('--d0', type=int, default=2)
    parser.add_argument('--S', type=int, default=50)
    parser.add_argument('--c_target', type=float, default=1.30)
    parser.add_argument('--max_levels', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=None)
    args = parser.parse_args()
    result = run_cascade(d0=args.d0, S=args.S, c_target=args.c_target,
                          max_levels=args.max_levels, n_workers=args.n_workers)
    if 'proven_at' in result:
        if result.get('box_certified'):
            _log(f"\nRIGOROUS PROOF v3: C_{{1a}} >= {args.c_target}")
            _log(f"  (Sound: N spectral + O sparsity bound)")
        else:
            _log(f"\nGRID-POINT PROOF: TV >= {args.c_target} (box cert incomplete)")
    else:
        _log(f"\nNOT PROVEN (cascade did not converge)")


if __name__ == '__main__':
    main()
