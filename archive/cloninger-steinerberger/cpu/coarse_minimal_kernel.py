"""Minimal coarse-grid Gray-code kernel — strip ALL box-cert math.

Purpose: at deep cascade levels (L3+ at d=16, etc.) where box-cert fails
anyway because the corrections d²/(2S²) dominate any margin, we don't want
to pay the per-pruned-cell box-cert cost (sort + LP + spectral lookup).

This kernel does the absolute minimum:
  - Gray-code child enumeration (incremental conv updates, sparse
    cross-terms at d>=32)
  - TV-at-grid prune check (`ws > floor(c_target * ell * S^2 / (2d) - eps)`)
  - Asymmetry pre-filter (cheap, applied once per parent)
  - Survivor storage to out_buf

NO box-cert.  NO cell_var.  NO quad_corr.  NO Joint dual.

Returns: (n_survivors, n_subtree_pruned).  No min_cert_net (caller must
treat box-cert as DEFERRED at this level).

Soundness: this kernel produces the IDENTICAL survivor set as the full
N+O kernel or the Gray-code triangle kernel — the prune criterion is the
same TV-at-grid threshold.  Box-cert quality differs but only as an
informational metric, not a correctness one.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _fused_coarse_minimal_gray(parent_int, d_child, S, c_target,
                                  lo_arr, hi_arr, out_buf):
    """Coarse-grid Gray-code minimal kernel: no box-cert, just survivors.

    Args mirror _fused_coarse_gray from run_cascade.py (so callers can
    swap in/out easily).
    """
    d_parent = parent_int.shape[0]

    # ---- Asymmetry pre-filter (parent-level, sound) ----
    threshold_asym = math.sqrt(c_target / 2.0)
    S_d = np.float64(S)
    left_sum = np.int64(0)
    for i in range(d_parent // 2):
        left_sum += np.int64(parent_int[i])
    left_frac = np.float64(left_sum) / S_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0

    # ---- Constants ----
    S_sq = S_d * S_d
    d_d = np.float64(d_child)
    inv_2d = 1.0 / (2.0 * d_d)
    eps = 1e-9
    max_ell = 2 * d_child
    conv_len = 2 * d_child - 1

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d
                                 - eps)

    max_survivors = out_buf.shape[0]
    n_surv = 0

    # ---- Sparse cross-term tracking (only when d>=32) ----
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0

    # ---- Allocate ----
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)

    # Initial child
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    # Initial conv
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

    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i
                nz_pos[i] = nz_count
                nz_count += 1

    # ---- Gray-code setup ----
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

    # ---- Main loop ----
    while True:
        # === TEST current child: TV-at-grid prune check ===
        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
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

        if not pruned:
            # Survivor: canonicalize (palindrome rep) + store
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
                        out_buf[n_surv, i] = child[d_child - 1 - i]
                else:
                    for i in range(d_child):
                        out_buf[n_surv, i] = child[i]
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

        # === INCREMENTAL CONV UPDATE ===
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
                nz_list[nz_count] = k1; nz_pos[k1] = nz_count
                nz_count += 1
            if old2 != 0 and new2 == 0:
                p_nz = nz_pos[k2]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p_nz] = last
                nz_pos[last] = p_nz; nz_pos[k2] = -1
            elif old2 == 0 and new2 != 0:
                nz_list[nz_count] = k2; nz_pos[k2] = nz_count
                nz_count += 1
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

    return n_surv, 0


def process_parent_minimal(parent_int, S, c_target, d_child,
                              op_rest_d_arr=None, **kwargs):
    """Process one parent with the minimal kernel (no box-cert).

    Returns (survivors, total_children, counts) — counts has 0 for all
    cert layers (we don't certify, just prune).
    """
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
            return (np.empty((0, d_child), dtype=np.int32), 0,
                    {'n_certified_NO': 0, 'n_certified_J': 0,
                     'n_certified_L': 0, 'n_uncertified': 0,
                     'time_NO': 0.0, 'time_J': 0.0, 'time_L': 0.0,
                     'min_cert_net': -1e30})
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_children *= (hi - lo + 1)
    if total_children == 0:
        return (np.empty((0, d_child), dtype=np.int32), 0,
                {'n_certified_NO': 0, 'n_certified_J': 0,
                 'n_certified_L': 0, 'n_uncertified': 0,
                 'time_NO': 0.0, 'time_J': 0.0, 'time_L': 0.0,
                 'min_cert_net': -1e30})

    # Output buffer sizing — we don't know how many will survive.
    buf_cap = min(total_children, max(100_000, 2_000_000 // max(1, d_child)))
    out_buf = np.empty((buf_cap, d_child), dtype=np.int32)

    import time as _time
    t0 = _time.time()
    n_surv, _ = _fused_coarse_minimal_gray(
        parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf)
    if n_surv > buf_cap:
        # Resize and re-run
        out_buf = np.empty((n_surv, d_child), dtype=np.int32)
        n_surv, _ = _fused_coarse_minimal_gray(
            parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf)
    t_kernel = _time.time() - t0

    survivors = out_buf[:n_surv].copy()
    counts = {
        'n_certified_NO': int(total_children - n_surv),  # implicit prunes
        'n_certified_J': 0, 'n_certified_L': 0,
        'n_uncertified': 0,
        'time_NO': float(t_kernel), 'time_J': 0.0, 'time_L': 0.0,
        'min_cert_net': -1e30,  # box-cert deferred at this level
    }
    return survivors, total_children, counts
