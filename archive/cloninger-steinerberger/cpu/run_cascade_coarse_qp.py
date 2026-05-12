"""Coarse-grid cascade prover with JOINT QP box certification.

Identical to run_cascade_coarse_v2.py except the box-certification step
uses the EXACT joint QP (vertex enumeration) instead of the triangle bound
cell_var + quad_corr.

For d <= QP_MAX_D (default 16) the QP is used. For d > QP_MAX_D, falls back
to the v2 triangle bound (vertex enumeration becomes infeasible).

Soundness: QP <= triangle always (proved). Substituting QP for triangle
in the certification condition `margin > bound` is therefore at least as
sound as v2 — it certifies a SUPERSET of v2's certified compositions.
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
sys.path.insert(0, _this_dir)

from compositions import (generate_compositions_batched,
                          generate_canonical_compositions_batched)
from pruning import count_compositions, asymmetry_threshold, _canonical_mask

QP_MAX_D = 16  # Use QP only when d <= 16; fall back to triangle above.


def _log(msg):
    print(msg, flush=True)


# =====================================================================
# Joint QP bound (vertex enumeration), Numba-jitted
# =====================================================================

@njit(cache=True)
def qp_bound_window(grad, ell, s_lo, scale, h, d):
    """Exact max over Cell of -grad.delta - scale * delta^T A_W delta.

    A_W is implicit: (A_W)_{ij} = 1 if s_lo <= i+j <= s_lo+ell-2 else 0.

    Vertex enumeration: d * 2^(d-1) candidates.
    Cost: O(d^3 * 2^(d-1)) per call.
    """
    s_hi = s_lo + ell - 2
    best = 0.0  # f(0) = 0 always feasible
    delta = np.zeros(d, dtype=np.float64)
    tol = h * 1e-9
    n_pat = 1 << (d - 1)

    for free_idx in range(d):
        for mask in range(n_pat):
            sum_others = 0.0
            bit_pos = 0
            for i in range(d):
                if i == free_idx:
                    continue
                bit = (mask >> bit_pos) & 1
                if bit == 0:
                    delta[i] = -h
                else:
                    delta[i] = h
                sum_others += delta[i]
                bit_pos += 1
            free_val = -sum_others
            if free_val < -h - tol or free_val > h + tol:
                continue
            if free_val > h:
                free_val = h
            elif free_val < -h:
                free_val = -h
            delta[free_idx] = free_val

            lin = 0.0
            for i in range(d):
                lin += grad[i] * delta[i]
            quad = 0.0
            for i in range(d):
                di = delta[i]
                if di == 0.0:
                    continue
                # Sum_j A_W[i,j] * delta[j] = sum over j with s_lo <= i+j <= s_hi
                j_lo = max(0, s_lo - i)
                j_hi = min(d - 1, s_hi - i)
                if j_lo > j_hi:
                    continue
                row_sum = 0.0
                for j in range(j_lo, j_hi + 1):
                    row_sum += delta[j]
                quad += di * row_sum
            val = -lin - scale * quad
            if val > best:
                best = val
    return best


# =====================================================================
# Pair-count prefix sums for the v2 fallback (d > QP_MAX_D)
# =====================================================================

@njit(cache=True)
def _build_pair_prefix(d):
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
def _quad_corr_v2(s_lo, ell, d, prefix_nk, prefix_mk, inv_4S2):
    hi = s_lo + ell - 1
    N_W = prefix_nk[hi] - prefix_nk[s_lo]
    M_W = prefix_mk[hi] - prefix_mk[s_lo]
    cross_W = N_W - M_W
    d_sq = np.int64(d) * np.int64(d)
    compl_bound = d_sq - N_W
    pair_bound = min(cross_W, compl_bound)
    if pair_bound <= 0:
        return 0.0
    return (2.0 * d / ell) * np.float64(pair_bound) * inv_4S2


# =====================================================================
# Pruning kernel — Theorem 1 grid-point + QP box cert
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_qp(batch_int, d, S, c_target, use_qp):
    """Same as v2 _prune_no_correction but with QP box-cert when use_qp."""
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
            scale_qp = 2.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                if ws > dyn_it:
                    pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    s_hi = s_lo + ell - 2
                    for i in range(d):
                        g_acc = 0.0
                        j_lo_g = max(0, s_lo - i)
                        j_hi_g = min(d - 1, s_hi - i)
                        for j in range(j_lo_g, j_hi_g + 1):
                            g_acc += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g_acc * scale_g

                    if use_qp:
                        bound = qp_bound_window(grad_arr, ell, s_lo, scale_qp, h, d)
                    else:
                        # v2 triangle: cell_var + quad_corr
                        # Sort gradient (insertion sort)
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
                        qc = _quad_corr_v2(s_lo, ell, d, prefix_nk, prefix_mk, inv_4S2)
                        bound = cell_var + qc

                    net = margin - bound
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


def asymmetry_prune_mask_coarse(batch_int, S, c_target):
    d = batch_int.shape[1]
    threshold = asymmetry_threshold(c_target)
    left_bins = d // 2
    left = batch_int[:, :left_bins].sum(axis=1).astype(np.float64)
    left_frac = left / float(S)
    needs_check = (left_frac > 1 - threshold) & (left_frac < threshold)
    return needs_check


def coarse_x_cap(d, S, c_target):
    return int(math.floor(S * math.sqrt(c_target / d)))


# =====================================================================
# Fused generate-and-prune (cascade levels)
# =====================================================================

@njit(cache=True)
def _fused_coarse_qp(parent_int, d_child, S, c_target, lo_arr, hi_arr,
                     out_buf, use_qp):
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
            scale_qp = 2.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                if ws > dyn_it:
                    pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    s_hi = s_lo + ell - 2
                    for i in range(d_child):
                        g_acc = 0.0
                        j_lo_g = max(0, s_lo - i)
                        j_hi_g = min(d_child - 1, s_hi - i)
                        for j in range(j_lo_g, j_hi_g + 1):
                            g_acc += np.float64(child[j]) / S_d
                        grad_buf[i] = g_acc * scale_g

                    if use_qp:
                        bound = qp_bound_window(grad_buf, ell, s_lo, scale_qp,
                                                h, d_child)
                    else:
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
                        qc = _quad_corr_v2(s_lo, ell, d_child, prefix_nk,
                                           prefix_mk, inv_4S2)
                        bound = cell_var + qc

                    net = margin - bound
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

def run_level0(d0, S, c_target, verbose=True):
    n_total = count_compositions(d0, S)
    use_qp = d0 <= QP_MAX_D
    bound_label = 'QP' if use_qp else 'v2 triangle'

    if verbose:
        _log(f"\n[L0] d={d0}, S={S}, compositions={n_total:,} "
             f"(box cert: {bound_label})")
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
        survived_mask, min_net = _prune_qp(batch, d0, S, c_target, use_qp)
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
            _log(f"     min(margin - {bound_label}) = {global_min_net:.6f}")
            if box_ok:
                _log(f"     BOX CERT: PASS")
            else:
                _log(f"     BOX CERT: FAIL")
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


# =====================================================================
# Process one parent
# =====================================================================

def process_parent(parent_int, d_child, S, c_target, buf_cap=100_000):
    d_parent = len(parent_int)
    x_cap = coarse_x_cap(d_child, S, c_target)
    use_qp = d_child <= QP_MAX_D

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

    n_surv, n_tested, min_net = _fused_coarse_qp(
        parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf, use_qp)

    if n_surv > buf_cap:
        out_buf = np.empty((n_surv, d_child), dtype=np.int32)
        n2, _, min_net = _fused_coarse_qp(
            parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf, use_qp)
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

def run_cascade(d0, S, c_target, max_levels=10, n_workers=None, verbose=True):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)

    if verbose:
        _log(f"\n{'=' * 70}")
        _log(f"COARSE CASCADE PROVER (QP box cert, fallback to v2 at d>{QP_MAX_D})")
        _log(f"  d0={d0}, S={S}, c_target={c_target}")
        _log(f"  workers: {n_workers}")
        _log(f"{'=' * 70}")

    t_total = time.time()
    info = {'d0': d0, 'S': S, 'c_target': c_target, 'levels': []}

    l0 = run_level0(d0, S, c_target, verbose=verbose)
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
            _log(f"\n*** GRID-POINT PROOF at L0 ***")
            _log(f"    Box certification: {'PASS' if l0['box_certified'] else 'FAIL'}")
        return info

    current = l0['survivors']
    d_parent = d0

    for level in range(1, max_levels + 1):
        d_child = 2 * d_parent
        n_parents = len(current)
        if n_parents == 0:
            break

        x_cap = coarse_x_cap(d_child, S, c_target)
        bound_label = 'QP' if d_child <= QP_MAX_D else 'v2 triangle'

        if verbose:
            _log(f"\n--- Level {level}: d={d_parent} -> {d_child}, "
                 f"parents={n_parents:,}, x_cap={x_cap} (box cert: {bound_label}) ---")

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
                             f"children={total_children:,} survivors={ns:,}")
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
                _log(f"    min(margin - {bound_label}) = {level_min_net:.6f}")
                if box_ok:
                    _log(f"    BOX CERT: PASS")
                else:
                    _log(f"    BOX CERT: FAIL")
            if n_survivors == 0:
                _log(f"    *** ALL PRUNED ***")

        info['levels'].append({
            'level': level, 'd_child': d_child, 'parents': n_parents,
            'children': int(total_children), 'survivors': n_survivors,
            'rate': rate, 'time': elapsed_level,
            'min_cert_net': level_min_net, 'box_certified': box_ok,
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
                _log(f"\n*** GRID-POINT PROOF COMPLETE: TV >= {c_target} ***")
                if all_cert:
                    _log(f"*** RIGOROUS PROOF (QP box cert): C_{{1a}} >= {c_target} ***")
                else:
                    _log(f"    Box certification incomplete at some levels.")
                _log(f"    Total time: {info['total_time']:.1f}s")
            return info

        d_parent = d_child

    info['total_time'] = time.time() - t_total
    info['final_survivors'] = len(current)
    if verbose:
        _log(f"\nCascade did not converge ({len(current):,} survivors at d={d_parent})")
    return info


def main():
    parser = argparse.ArgumentParser(
        description='Coarse-grid cascade prover with QP box certification')
    parser.add_argument('--d0', type=int, default=2)
    parser.add_argument('--S', type=int, default=50)
    parser.add_argument('--c_target', type=float, default=1.28)
    parser.add_argument('--max_levels', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=None)
    args = parser.parse_args()

    result = run_cascade(d0=args.d0, S=args.S, c_target=args.c_target,
                         max_levels=args.max_levels, n_workers=args.n_workers)

    if 'proven_at' in result:
        if result.get('box_certified'):
            _log(f"\nRIGOROUS PROOF (QP): C_{{1a}} >= {args.c_target}")
        else:
            _log(f"\nGRID-POINT PROOF: TV >= {args.c_target} "
                 f"(box cert incomplete at some levels)")
    else:
        _log(f"\nNOT PROVEN")


if __name__ == '__main__':
    main()
