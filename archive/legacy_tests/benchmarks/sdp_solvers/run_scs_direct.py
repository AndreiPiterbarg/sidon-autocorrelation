#!/usr/bin/env python
r"""Direct SCS solver for Lasserre SDP — fully vectorized, GPU-capable.

Builds SCS problem data directly in numpy/scipy. Zero Python loops in
hot paths. All COO construction is vectorized.

For bisection: the A matrix is split into t-independent (A_base) and
t-dependent (A_t) parts. Each bisection step computes A = A_base + t * A_t
as a single sparse addition — no per-element Python loops.

GPU: SCS 3.x supports GPU via compilation flag. On H100, the sparse
matvec and PSD projections run on GPU. Install: pip install scs[gpu].

Usage:
    python tests/run_scs_direct.py --d 16 --order 3 --bw 15
    python tests/run_scs_direct.py --d 14 --order 3 --bw 13 --gpu
"""
import sys
import os
import time
import json
import argparse
import numpy as np
from scipy import sparse as sp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, enum_monomials, val_d_known,
)

SQRT2 = np.sqrt(2.0)


# =====================================================================
# Vectorized PSD cone COO builder
# =====================================================================

def _psd_lower_tri_map(n):
    """Precompute (scs_row, matrix_i, matrix_j, scale) for PSD cone of size n.

    SCS vectorizes PSD cones as column-major lower triangle:
      k = j*n - j*(j-1)/2 + (i - j)  for i >= j
    Off-diag scaled by sqrt(2).

    Returns arrays: (scs_row_offsets, flat_indices, scales) all of length n*(n+1)/2.
    flat_indices[k] = i*n + j (row-major index into the n×n pick array).
    """
    dim = n * (n + 1) // 2
    scs_rows = np.empty(dim, dtype=np.int64)
    flat_idx = np.empty(dim, dtype=np.int64)
    scales = np.empty(dim)
    k = 0
    for j in range(n):
        block_len = n - j
        scs_rows[k:k+block_len] = np.arange(k, k+block_len)
        flat_idx[k:k+block_len] = np.arange(j, n) * n + j  # column j, rows j..n-1
        scales[k] = 1.0  # diagonal
        if block_len > 1:
            scales[k+1:k+block_len] = SQRT2  # off-diagonal
        k += block_len
    return scs_rows, flat_idx, scales


# Cache these — same n reused across many cones
_psd_cache = {}

def _get_psd_map(n):
    if n not in _psd_cache:
        _psd_cache[n] = _psd_lower_tri_map(n)
    return _psd_cache[n]


def _vectorized_psd_coo(pick_flat, n, row_offset, sign=-1.0):
    """Build COO entries for a PSD cone constraint: sign * x[pick] in PSD.

    pick_flat: int64 array of length n*n (row-major y-indices).
    Returns (rows, cols, vals) arrays — all numpy, no Python loops.
    """
    scs_rows, flat_idx, scales = _get_psd_map(n)
    y_indices = pick_flat[flat_idx]  # look up which x-variable each entry maps to
    valid = y_indices >= 0
    r = (row_offset + scs_rows[valid]).astype(np.int64)
    c = y_indices[valid].astype(np.int64)
    v = (sign * scales[valid])
    return r, c, v


def _vectorized_diff_psd_coo(pick_a, pick_b, n, row_offset):
    """COO for (A - B) >> 0: -A + B in SCS form."""
    scs_rows, flat_idx, scales = _get_psd_map(n)
    a_idx = pick_a[flat_idx]
    b_idx = pick_b[flat_idx]

    # A entries: -scale * x[a_idx]
    valid_a = a_idx >= 0
    r_a = (row_offset + scs_rows[valid_a]).astype(np.int64)
    c_a = a_idx[valid_a].astype(np.int64)
    v_a = -scales[valid_a]

    # B entries: +scale * x[b_idx]
    valid_b = b_idx >= 0
    r_b = (row_offset + scs_rows[valid_b]).astype(np.int64)
    c_b = b_idx[valid_b].astype(np.int64)
    v_b = scales[valid_b].copy()

    return (np.concatenate([r_a, r_b]),
            np.concatenate([c_a, c_b]),
            np.concatenate([v_a, v_b]))


# =====================================================================
# Build base SCS problem (vectorized)
# =====================================================================

def build_base_problem(P, add_upper_loc=True):
    """Build base SCS data. All construction is vectorized numpy."""
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    idx = P['idx']
    n_x = n_y + 1
    t_col = n_y

    c = np.zeros(n_x)
    c[t_col] = 1.0

    all_r, all_c, all_v = [], [], []
    b_parts = []
    row_offset = 0

    # ---- ZERO CONE ----
    # y_0 = 1
    zero_tup = tuple(0 for _ in range(d))
    all_r.append(np.array([row_offset], dtype=np.int64))
    all_c.append(np.array([idx[zero_tup]], dtype=np.int64))
    all_v.append(np.array([1.0]))
    b_parts.append(1.0)
    row_offset += 1

    # Consistency equalities
    if P['consist_eq_lists'] is not None:
        n_eq, er, ec, ev = P['consist_eq_lists']
        all_r.append(np.asarray(er, dtype=np.int64) + row_offset)
        all_c.append(np.asarray(ec, dtype=np.int64))
        all_v.append(np.asarray(ev, dtype=np.float64))
        b_parts.extend([0.0] * n_eq)
        row_offset += n_eq

    n_zero = row_offset

    # ---- NONNEG CONE ----
    nonneg_start = row_offset

    # Consistency inequalities (negated for SCS)
    if P['consist_iq_lists'] is not None:
        n_iq, ir, ic, iv = P['consist_iq_lists']
        all_r.append(np.asarray(ir, dtype=np.int64) + row_offset)
        all_c.append(np.asarray(ic, dtype=np.int64))
        all_v.append(-np.asarray(iv, dtype=np.float64))
        b_parts.extend([0.0] * n_iq)
        row_offset += n_iq

    # Scalar windows: f_W(y) - t <= 0
    F_row, F_col, F_val = P['F_coo_lists']
    n_win = P['n_win']
    fr = np.asarray(F_row, dtype=np.int64) + row_offset
    fc = np.asarray(F_col, dtype=np.int64)
    fv = np.asarray(F_val, dtype=np.float64)
    # t coefficient
    tr = np.arange(n_win, dtype=np.int64) + row_offset
    tc = np.full(n_win, t_col, dtype=np.int64)
    tv = np.full(n_win, -1.0)
    all_r.extend([fr, tr])
    all_c.extend([fc, tc])
    all_v.extend([fv, tv])
    b_parts.extend([0.0] * n_win)
    row_offset += n_win

    # y >= 0
    yr = np.arange(n_y, dtype=np.int64) + row_offset
    yc = np.arange(n_y, dtype=np.int64)
    yv = np.full(n_y, -1.0)
    all_r.append(yr)
    all_c.append(yc)
    all_v.append(yv)
    b_parts.extend([0.0] * n_y)
    row_offset += n_y

    n_nonneg = row_offset - nonneg_start

    # ---- PSD CONES ----
    psd_sizes = []

    # Full M_{k-1}
    if P['m1_valid']:
        n_m = P['m1_size']
        r, c_arr, v = _vectorized_psd_coo(P['m1_pick'], n_m, row_offset)
        all_r.append(r); all_c.append(c_arr); all_v.append(v)
        dim = n_m * (n_m + 1) // 2
        b_parts.extend([0.0] * dim)
        row_offset += dim
        psd_sizes.append(n_m)

    # Clique moment PSD
    for cd in P['clique_data']:
        pick = cd['mom_pick']
        if np.any(pick < 0):
            continue
        n_cb = cd['mom_size']
        r, c_arr, v = _vectorized_psd_coo(pick, n_cb, row_offset)
        all_r.append(r); all_c.append(c_arr); all_v.append(v)
        dim = n_cb * (n_cb + 1) // 2
        b_parts.extend([0.0] * dim)
        row_offset += dim
        psd_sizes.append(n_cb)

    # Clique localizing PSD
    if order >= 2:
        for i_var in range(d):
            c_idx = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx]
            picks = cd['loc_picks'].get(i_var)
            if picks is None or np.any(picks < 0):
                continue
            n_cb = cd['loc_size']
            r, c_arr, v = _vectorized_psd_coo(picks, n_cb, row_offset)
            all_r.append(r); all_c.append(c_arr); all_v.append(v)
            dim = n_cb * (n_cb + 1) // 2
            b_parts.extend([0.0] * dim)
            row_offset += dim
            psd_sizes.append(n_cb)

    # Upper localizing PSD
    if add_upper_loc and order >= 2:
        for i_var in range(d):
            c_idx = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx]
            t_pick = cd.get('t_pick')
            loc_pick = cd['loc_picks'].get(i_var)
            if t_pick is None or loc_pick is None:
                continue
            if np.any(t_pick < 0) or np.any(loc_pick < 0):
                continue
            n_cb = cd['loc_size']
            r, c_arr, v = _vectorized_diff_psd_coo(t_pick, loc_pick, n_cb, row_offset)
            all_r.append(r); all_c.append(c_arr); all_v.append(v)
            dim = n_cb * (n_cb + 1) // 2
            b_parts.extend([0.0] * dim)
            row_offset += dim
            psd_sizes.append(n_cb)

    # Assemble
    n_rows = row_offset
    R = np.concatenate(all_r)
    C = np.concatenate(all_c)
    V = np.concatenate(all_v)
    A = sp.csc_matrix((V, (R, C)), shape=(n_rows, n_x))
    b = np.array(b_parts)
    cone = {'z': n_zero, 'l': n_nonneg, 's': psd_sizes}

    meta = {
        'n_x': n_x, 'n_y': n_y, 't_col': t_col,
        'n_zero': n_zero, 'n_nonneg': n_nonneg,
        'n_rows_base': n_rows,
    }

    print(f"  Base problem: {n_rows:,} rows x {n_x:,} cols, nnz={A.nnz:,}")
    print(f"  Cones: z={n_zero}, l={n_nonneg}, PSD={len(psd_sizes)}")
    print(f"  A memory: {A.data.nbytes/1e6:.1f} MB", flush=True)

    return A, b, c, cone, meta


# =====================================================================
# Vectorized window PSD construction
# =====================================================================

def _build_window_psd_block(P, windows, t_val):
    """Build the sparse block for ALL window PSD cones at once.

    Returns (A_block, b_block, psd_sizes) for the window PSD rows.
    All construction is vectorized — no per-window Python loops in the
    inner PSD entry loop.
    """
    decomp = _precompute_window_psd_decomposition(P, windows)
    if decomp is None:
        return None, None, []
    return _assemble_window_psd(decomp, t_val)


def _precompute_window_psd_decomposition(P, windows):
    """Precompute t-independent and t-dependent parts of window PSD block.

    Returns a decomposition dict that can cheaply assemble A for any t_val.
    Build once per CG round; assemble cheaply per bisection step.
    """
    d = P['d']
    n_y = P['n_y']
    n_x = n_y + 1

    base_r, base_c, base_v = [], [], []
    t_r, t_c, t_v = [], [], []
    psd_sizes = []
    row_offset = 0

    for w in windows:
        c_idx_w = int(P['window_covering'][w])
        if c_idx_w < 0:
            continue

        cd = P['clique_data'][c_idx_w]
        n_cb = cd['loc_size']
        t_pick = cd['t_pick']
        if t_pick is None or np.any(t_pick < 0):
            continue

        ab_eiej = cd['viol_ab_eiej']
        gi_grid = cd['viol_gi_grid']
        ell, s_lo = P['windows'][w]
        coeff = 2.0 * d / ell
        mask = (gi_grid >= s_lo) & (gi_grid <= s_lo + ell - 2)
        nz_li, nz_lj = np.nonzero(mask)

        scs_rows, flat_idx, scales = _get_psd_map(n_cb)
        dim = n_cb * (n_cb + 1) // 2

        # T-part (t-dependent): -scale * x[t_pick[flat_idx]]
        t_indices = t_pick[flat_idx]
        valid_t = t_indices >= 0
        if np.any(valid_t):
            t_r.append((row_offset + scs_rows[valid_t]).astype(np.int64))
            t_c.append(t_indices[valid_t].astype(np.int64))
            t_v.append(-scales[valid_t])  # multiply by t_val at assembly

        # Q-part (t-independent): +coeff * scale * x[ab_eiej[...]]
        if len(nz_li) > 0:
            ij_i = flat_idx // n_cb
            ij_j = flat_idx % n_cb
            q_indices = ab_eiej[ij_i[:, None], ij_j[:, None],
                                nz_li[None, :], nz_lj[None, :]]
            valid_q = q_indices >= 0
            scs_rows_exp = np.broadcast_to(
                (row_offset + scs_rows)[:, None], q_indices.shape)
            scales_exp = np.broadcast_to(scales[:, None], q_indices.shape)

            base_r.append(scs_rows_exp[valid_q].astype(np.int64))
            base_c.append(q_indices[valid_q].astype(np.int64))
            base_v.append(coeff * scales_exp[valid_q])

        row_offset += dim
        psd_sizes.append(n_cb)

    if not psd_sizes:
        return None

    n_new_rows = row_offset
    has_t = bool(t_v)

    # Build base sparse matrix (t-independent Q-part)
    if base_v:
        BR = np.concatenate(base_r)
        BC = np.concatenate(base_c)
        BV = np.concatenate(base_v)
    else:
        BR = np.array([], dtype=np.int64)
        BC = np.array([], dtype=np.int64)
        BV = np.array([], dtype=np.float64)
    A_base_win = sp.csc_matrix((BV, (BR, BC)), shape=(n_new_rows, n_x))

    if has_t:
        TR = np.concatenate(t_r)
        TC = np.concatenate(t_c)
        TV = np.concatenate(t_v)
        A_t_win = sp.csc_matrix((TV, (TR, TC)), shape=(n_new_rows, n_x))
    else:
        A_t_win = sp.csc_matrix((n_new_rows, n_x), dtype=np.float64)

    # Precompute union sparsity pattern for in-place A(t) = base + t * t_part
    if has_t:
        A_template = (A_base_win + A_t_win).tocsc()
        A_template.sort_indices()
        A_ref_2t = (A_base_win + 2.0 * A_t_win).tocsc()
        A_ref_2t.sort_indices()
        t_data = A_ref_2t.data - A_template.data
        base_data = A_template.data - t_data
    else:
        A_template = A_base_win.tocsc()
        A_template.sort_indices()
        t_data = None
        base_data = A_template.data.copy()

    return {
        'A_template': A_template,
        'base_data': base_data,
        't_data': t_data,
        'has_t': has_t,
        'n_rows': n_new_rows,
        'n_x': n_x,
        'psd_sizes': psd_sizes,
    }


def _assemble_window_psd(decomp, t_val):
    """Cheaply assemble window PSD block for a given t_val.

    In-place data array update — no new sparse matrix allocation.
    """
    A = decomp['A_template']
    if decomp['has_t']:
        np.add(decomp['base_data'], t_val * decomp['t_data'], out=A.data)
    b = np.zeros(decomp['n_rows'])
    return A, b, decomp['psd_sizes']


# =====================================================================
# Main solver
# =====================================================================

def solve_scs_direct(d, order, bandwidth, c_target=1.28,
                     add_upper_loc=True, max_cg_rounds=10, n_bisect=12,
                     use_gpu=False, scs_max_iters=100000, scs_eps=1e-5,
                     scs_scale=None, use_indirect=False,
                     verbose=True, hook=None):
    """Full CG Lasserre solver using direct SCS.

    hook: optional GapAccelHook (see lasserre.gap_accelerator). When
      provided, per-round diagnostic and cut-reordering hooks are called.
      Passing hook=None reproduces exactly the prior behavior.
    """
    import scs

    t_total = time.time()

    print(f"{'='*70}")
    print(f"DIRECT SCS LASSERRE: L{order} d={d} bw={bandwidth}")
    print(f"  GPU={use_gpu}, max_iters={scs_max_iters}, eps={scs_eps}")
    print(f"{'='*70}\n", flush=True)

    cliques = _build_banded_cliques(d, bandwidth)
    print(f"Cliques: {len(cliques)} of size {len(cliques[0])}\n", flush=True)
    P = _precompute_highd(d, order, cliques, verbose=verbose)
    n_y = P['n_y']

    print("\nBuilding base SCS problem...", flush=True)
    t_build = time.time()
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, add_upper_loc)
    print(f"  Built in {time.time()-t_build:.1f}s\n", flush=True)

    scs_kwargs = dict(max_iters=scs_max_iters, eps_abs=scs_eps, eps_rel=scs_eps,
                      verbose=verbose)
    if scs_scale is not None:
        scs_kwargs['scale'] = scs_scale
    if use_indirect:
        scs_kwargs['use_indirect'] = True

    # Round 0: minimize t
    print("  [Round 0] Minimize t...", flush=True)
    t0 = time.time()

    if use_gpu:
        from admm_gpu_solver import admm_solve
        # Round 0 only needs a rough scalar bound — use loose eps.
        # Tight eps (1e-7) causes adaptive rho oscillation in admm_solve.
        r0_eps = max(scs_eps, 1e-5)
        sol = admm_solve(A_base, b_base, c_obj, cone_base,
                         max_iters=scs_max_iters, eps_abs=r0_eps,
                         eps_rel=r0_eps, device='cuda', verbose=verbose)
    else:
        data = {'A': A_base, 'b': b_base, 'c': c_obj}
        solver = scs.SCS(data, cone_base, **scs_kwargs)
        sol = solver.solve()

    scalar_lb = 0.5
    y_vals = np.zeros(n_y)
    if sol['info']['status'] in ('solved', 'solved_inaccurate'):
        scalar_lb = float(sol['x'][meta['t_col']])
        y_vals = sol['x'][:n_y].copy()

    best_lb = scalar_lb
    active_windows = set()
    violations = _check_violations_highd(y_vals, scalar_lb, P, active_windows)

    print(f"    Scalar bound = {scalar_lb:.10f} "
          f"({time.time()-t0:.1f}s, {sol['info']['iter']} iters)",
          flush=True)
    print(f"    {len(violations)} violations", flush=True)

    if not violations:
        elapsed = time.time() - t_total
        return _result(best_lb, d, order, bandwidth, n_y, 0, elapsed, y_vals)

    # CG rounds
    last_x = sol['x'].copy() if sol['x'] is not None else None
    last_y_dual = sol['y'].copy() if sol['y'] is not None else None
    last_s = sol['s'].copy() if sol['s'] is not None else None

    # Keep last feasible solution across rounds so post-loop fallback
    # (line ~869) has a bound name even if the first round aborts early.
    last_feas_y = None
    last_feas_t = None

    for cg_round in range(1, max_cg_rounds + 1):
        n_add = min(100, len(violations))

        if hook is not None:
            violations = hook.reorder_violations(
                violations, y_vals, P, list(active_windows), n_add)

        for item in violations[:n_add]:
            w = item[0]
            active_windows.add(w)

        print(f"\n  [CG round {cg_round}] {len(active_windows)} windows",
              flush=True)

        lo = max(0.5, best_lb - 1e-3)
        hi = best_lb + 0.02 if best_lb > 0.5 else 5.0

        c_feas = np.zeros(meta['n_x'])  # feasibility objective

        # ── Precompute window PSD decomposition ONCE per CG round ──
        t_decomp = time.time()
        win_decomp = _precompute_window_psd_decomposition(P, active_windows)

        # Precompute full problem template (base + window PSD union pattern)
        if win_decomp is not None:
            # Build full A template at t=1 to get union sparsity
            A_win_t1, b_win_t1, psd_win = _assemble_window_psd(win_decomp, 1.0)
            A_full_t1 = sp.vstack([A_base, A_win_t1], format='csc')
            A_full_t1.sort_indices()

            # Build at t=2 to extract t-dependent part
            A_win_t2, _, _ = _assemble_window_psd(win_decomp, 2.0)
            A_full_t2 = sp.vstack([A_base, A_win_t2], format='csc')
            A_full_t2.sort_indices()

            # Extract base and t-coefficient in union pattern
            full_t_data = A_full_t2.data - A_full_t1.data
            full_base_data = A_full_t1.data - full_t_data
            has_t_full = np.any(full_t_data != 0)

            b_full_base = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
            cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                         's': list(cone_base['s']) + psd_win}
            n_rows_full = A_full_t1.shape[0]
        else:
            A_full_t1 = A_base.tocsc().copy()
            A_full_t1.sort_indices()
            full_t_data = None
            full_base_data = A_full_t1.data.copy()
            has_t_full = False
            b_full_base = b_base.copy()
            cone_full = cone_base
            n_rows_full = A_base.shape[0]

        print(f"    Decomposition precomputed in {time.time()-t_decomp:.1f}s",
              flush=True)

        # Pre-allocate padded warm-start arrays to avoid per-step allocation
        ws_x = np.zeros(meta['n_x'])
        ws_y = np.zeros(n_rows_full)
        ws_s = np.zeros(n_rows_full)
        has_warm = [False]

        def _prepare_warm(sol_t):
            """Copy solution into pre-allocated warm-start arrays."""
            if sol_t['x'] is not None:
                n = min(len(sol_t['x']), len(ws_x))
                ws_x[:n] = sol_t['x'][:n]
                if n < len(ws_x):
                    ws_x[n:] = 0
            if sol_t['y'] is not None:
                n = min(len(sol_t['y']), len(ws_y))
                ws_y[:n] = sol_t['y'][:n]
                if n < len(ws_y):
                    ws_y[n:] = 0
            if sol_t['s'] is not None:
                n = min(len(sol_t['s']), len(ws_s))
                ws_s[:n] = sol_t['s'][:n]
                if n < len(ws_s):
                    ws_s[n:] = 0
            has_warm[0] = True

        # GPU: persistent solver — reuse factorization + CSR pattern cache
        gpu_solver = [None]  # mutable ref for closure
        gpu_tau_col = [None]

        # Pre-build the fix_t + phase-1 augmented pattern ONCE per CG round.
        # Only A.data and b[0] change per bisection step.
        from admm_gpu_solver import ADMMSolver, augment_phase1

        n_cols_full = A_full_t1.shape[1]
        fix_t_row = sp.csc_matrix(
            ([1.0], ([0], [meta['t_col']])),
            shape=(1, n_cols_full))

        # Build augmented pattern at t=1 and t=2 to extract base/t decomposition
        A_fixed_t1 = sp.vstack([fix_t_row, A_full_t1], format='csc')
        A_fixed_t1.sort_indices()
        b_fixed_t1 = np.insert(b_full_base, 0, 1.0)
        cone_fixed = {'z': cone_full['z'] + 1,
                      'l': cone_full['l'],
                      's': cone_full['s']}

        A_p1_t1, b_p1_t1, c_p1_cached, cone_p1_cached, tau_col_cached = \
            augment_phase1(A_fixed_t1, b_fixed_t1, cone_fixed)

        # Build at t=2 to extract t-dependent part of the full augmented matrix
        if has_t_full:
            _tmp_data = A_full_t1.data.copy()
            np.add(full_base_data, 2.0 * full_t_data, out=A_full_t1.data)
            A_fixed_t2 = sp.vstack([fix_t_row, A_full_t1], format='csc')
            A_fixed_t2.sort_indices()
            b_fixed_t2 = np.insert(b_full_base, 0, 2.0)
            A_p1_t2, b_p1_t2, _, _, _ = augment_phase1(
                A_fixed_t2, b_fixed_t2, cone_fixed)
            # Extract decomposition in augmented space
            aug_t_data = A_p1_t2.data - A_p1_t1.data
            aug_base_data = A_p1_t1.data - aug_t_data
            aug_b_t_data = b_p1_t2 - b_p1_t1
            aug_b_base_data = b_p1_t1 - aug_b_t_data
            # Restore A_full_t1.data
            np.copyto(A_full_t1.data, _tmp_data)
        else:
            aug_t_data = None
            aug_base_data = A_p1_t1.data.copy()
            aug_b_t_data = None
            aug_b_base_data = b_p1_t1.copy()

        gpu_tau_col[0] = tau_col_cached

        # Reusable template — only .data and b change per step
        A_p1_template = A_p1_t1.copy()
        b_p1_template = b_p1_t1.copy()

        def check_feasible(t_val, max_iters_override=None,
                           eps_override=None):
            t_b = time.time()

            # In-place update of augmented A and b for this t_val
            if aug_t_data is not None:
                np.add(aug_base_data, t_val * aug_t_data,
                       out=A_p1_template.data)
                np.add(aug_b_base_data, t_val * aug_b_t_data,
                       out=b_p1_template)
            else:
                np.copyto(A_p1_template.data, aug_base_data)
                np.copyto(b_p1_template, aug_b_base_data)

            build_time = time.time() - t_b

            cur_iters = max_iters_override or scs_max_iters
            cur_eps = eps_override or scs_eps

            if use_gpu:
                if gpu_solver[0] is None:
                    # rho=0.1 proven 10-20x faster than rho=0.5
                    # for phase-1 problems (sweep_bisection.py)
                    gpu_solver[0] = ADMMSolver(
                        A_p1_template, b_p1_template, c_p1_cached,
                        cone_p1_cached, rho=0.1,
                        device='cuda', verbose=False)
                    # Warm-up: 100 iters at initial t to seed workspace
                    gpu_solver[0].solve(
                        max_iters=100, eps_abs=1.0, eps_rel=1.0,
                        tau_col=tau_col_cached)
                else:
                    gpu_solver[0]._update_A(A_p1_template)
                    gpu_solver[0].update_b(b_p1_template)

                tau_col = gpu_tau_col[0]
                tau_tol = max(cur_eps * 10, 1e-4)

                sol_t = gpu_solver[0].solve(
                    max_iters=cur_iters, eps_abs=cur_eps, eps_rel=cur_eps,
                    tau_col=tau_col, tau_tol=tau_tol)

                # Extract tau and original x (without tau variable)
                tau_val = sol_t['x'][tau_col] if sol_t['x'] is not None \
                    else float('inf')

                # Feasibility: tau* <= threshold means PSD constraints are
                # satisfiable without slack
                feasible = (sol_t['info']['status'] in
                            ('solved', 'solved_inaccurate')
                            and tau_val <= tau_tol)

                # Strip tau + fix_t row from x for warm-start compatibility
                sol_orig = dict(sol_t)
                sol_orig['x'] = sol_t['x'][:meta['n_x']].copy() \
                    if sol_t['x'] is not None else None
                _prepare_warm(sol_orig)

                return feasible, sol_orig, build_time

            # CPU SCS path (unchanged)
            kw = dict(max_iters=cur_iters, eps_abs=cur_eps,
                      eps_rel=cur_eps, verbose=False)
            if scs_scale is not None:
                kw['scale'] = scs_scale
            if use_indirect:
                kw['use_indirect'] = True
            data_t = {'A': A_full_t1, 'b': b_full_base, 'c': c_feas}
            s = scs.SCS(data_t, cone_full, **kw)

            try:
                if has_warm[0]:
                    sol_t = s.solve(warm_start=True,
                                   x=ws_x, y=ws_y, s=ws_s)
                else:
                    sol_t = s.solve()
            except (ValueError, Exception):
                sol_t = s.solve()

            _prepare_warm(sol_t)

            feasible = sol_t['info']['status'] in ('solved', 'solved_inaccurate')
            return feasible, sol_t, build_time

        # Copy last warm-start from round 0 / previous round
        if last_x is not None:
            ws_x[:min(len(last_x), len(ws_x))] = last_x[:min(len(last_x), len(ws_x))]
        if last_y_dual is not None:
            n = min(len(last_y_dual), len(ws_y))
            ws_y[:n] = last_y_dual[:n]
        if last_s is not None:
            n = min(len(last_s), len(ws_s))
            ws_s[:n] = last_s[:n]
        has_warm[0] = last_x is not None

        # Ensure hi feasible — coarse budget only.
        # Bug fix 2026-04-16: without overrides, check_feasible uses
        # the full scs_max_iters (20K) and scs_eps (1e-7). If the
        # solver doesn't early-exit, this takes 20K × 80ms = 27 min
        # PER CALL. Coarse budget is sufficient: we only need
        # feas/infeas classification, not precision.
        for _ in range(10):
            feas, sol_hi, bt = check_feasible(
                hi, max_iters_override=800, eps_override=1e-5)
            if feas:
                break
            hi *= 1.5
        else:
            print("    WARNING: infeasible up to t=100")
            break

        # Adaptive bisection with graduated iteration budgets +
        # tau-interpolation (ported from lasserre_scs.py's proven approach).
        #
        # CUDA kernel insight: quick-check with 85% hit rate before full scan.
        # SCS analogue: early bisection steps only need coarse feas/infeas
        # classification. Non-convergence → classify as infeasible → lo side →
        # conservative (never unsound). Late steps use full iterations.
        tau_lo, tau_hi = None, None
        last_feas_y = None
        last_feas_t = None

        for step in range(n_bisect):
            # Graduated iteration budget — tuned via sweep_bisection.py.
            # With rho=0.1 + Ruiz, most steps converge in 100-500 iters.
            # Early tau classification handles the rest.
            if step < 4:
                adaptive_iters = min(scs_max_iters, 800)
                adaptive_eps = max(scs_eps, 1e-5)
            elif step < 10:
                adaptive_iters = min(scs_max_iters, 2000)
                adaptive_eps = max(scs_eps, 1e-6)
            else:
                adaptive_iters = min(scs_max_iters, 4000)
                adaptive_eps = scs_eps

            # Tau-interpolation: jump closer to boundary once we have
            # tau from both sides (same technique as lasserre_scs.py)
            use_interp = False
            if (step >= 3 and tau_lo is not None and tau_hi is not None
                    and tau_lo > 0 and tau_hi < 0
                    and (hi - lo) > 1e-12):
                frac = tau_lo / (tau_lo - tau_hi)
                mid_interp = lo + frac * (hi - lo)
                mid = max(lo + 0.1 * (hi - lo),
                          min(hi - 0.1 * (hi - lo), mid_interp))
                use_interp = True
            else:
                mid = (lo + hi) / 2

            t_step = time.time()
            feas, sol_mid, bt = check_feasible(
                mid, max_iters_override=adaptive_iters,
                eps_override=adaptive_eps)
            dt = time.time() - t_step

            tau_val = float(sol_mid['info'].get('pobj', 0))
            if feas:
                hi = mid
                tau_hi = tau_val if np.isfinite(tau_val) else -1e-6
                tag = "feas"
                # Save y* from feasible solve for violation check
                if sol_mid['x'] is not None:
                    last_feas_y = sol_mid['x'][:n_y].copy()
                    last_feas_t = mid
            else:
                lo = mid
                tau_lo = tau_val if (np.isfinite(tau_val) and tau_val > 0) else 1.0
                tag = "infeas"

            interp_tag = " interp" if use_interp else ""
            print(f"    [{step+1}/{n_bisect}] t={mid:.8f} {tag} "
                  f"({dt:.1f}s, build={bt:.1f}s, "
                  f"{sol_mid['info']['iter']} iters){interp_tag}",
                  flush=True)

            # Early termination: interval converged
            if hi - lo < 5e-4 and step >= 4:
                print(f"    Converged: interval {hi-lo:.2e} < 1e-8",
                      flush=True)
                break

        lb = lo
        improvement = lb - best_lb
        best_lb = max(best_lb, lb)

        # Propagate warm-start to next CG round
        last_x = ws_x.copy()
        last_y_dual = ws_y.copy()
        last_s = ws_s.copy()

        v = val_d_known.get(d, 0)
        gc = (best_lb - 1) / (v - 1) * 100 if v > 1 else 0
        print(f"    lb={lb:.10f} (+{improvement:.2e}) gc={gc:.1f}%", flush=True)

        if hook is not None:
            _best_y_for_hook = last_feas_y if last_feas_y is not None else ws_x[:n_y].copy()
            hook.on_round_end(
                cg_round, best_lb, gc, _best_y_for_hook, P,
                list(active_windows), sol_dict=None)

        # ── Save checkpoint after each CG round ──
        _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data')
        os.makedirs(_data_dir, exist_ok=True)
        _tag = f"d{d}_o{order}_bw{bandwidth}_scs"
        _best_y = last_feas_y if last_feas_y is not None else ws_x[:n_y].copy()
        _ckpt = _result(best_lb, d, order, bandwidth, n_y,
                        len(active_windows), time.time() - t_total, _best_y)
        _ckpt['cg_round'] = cg_round
        _y_sol = _ckpt.pop('y_solution', None)
        if _y_sol is not None:
            np.save(os.path.join(_data_dir, f'solution_{_tag}_cg{cg_round}.npy'),
                    _y_sol)
        with open(os.path.join(_data_dir, f'result_{_tag}_cg{cg_round}.json'), 'w') as _f:
            json.dump(_ckpt, _f, indent=2, default=str)
        print(f"    Checkpoint saved: cg{cg_round} lb={best_lb:.10f}", flush=True)

        # Use saved y* from bisection (avoids extra solve, like CUDA's
        # incremental approach — reuse existing work instead of recomputing)
        y_vals = last_feas_y if last_feas_y is not None else ws_x[:n_y].copy()
        hi_for_viol = last_feas_t if last_feas_t is not None else hi
        violations = _check_violations_highd(y_vals, hi_for_viol, P, active_windows)

        if not violations:
            print(f"    No violations — converged.", flush=True)
            break
        if improvement < 1e-5 and cg_round >= 3:
            print(f"    Improvement < 1e-5 — stopping.", flush=True)
            break

        if hook is not None:
            should_abort, abort_reason = hook.should_abort()
            if should_abort:
                print(f"    [gap_accel] ABORT: {abort_reason}", flush=True)
                break

    elapsed = time.time() - t_total
    # Keep best feasible y for proof/verification
    best_y = last_feas_y if last_feas_y is not None else y_vals
    result = _result(best_lb, d, order, bandwidth, n_y,
                     len(active_windows), elapsed, best_y)
    if hook is not None:
        tag = f"d{d}_o{order}_bw{bandwidth}_scs"
        result['gap_accel'] = hook.finalize(tag=tag)
    return result


def _result(lb, d, order, bw, n_y, n_active, elapsed, y_solution=None):
    v = val_d_known.get(d, 0)
    gc = (lb - 1) / (v - 1) * 100 if v > 1 else 0
    sound = lb <= v + 1e-4 if v > 0 else True
    return {
        'lb': lb, 'd': d, 'order': order, 'bw': bw,
        'n_y': n_y, 'n_active_windows': n_active,
        'gap_closure': gc, 'elapsed': elapsed, 'sound': sound,
        'solver': 'scs_direct',
        'y_solution': y_solution,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--bw', type=int, required=True)
    parser.add_argument('--cg-rounds', type=int, default=10)
    parser.add_argument('--bisect', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--scs-iters', type=int, default=100000)
    parser.add_argument('--scs-eps', type=float, default=1e-5)
    parser.add_argument('--scs-scale', type=float, default=None,
                        help='SCS scale param. scale=1 gives ~7x for d<=12')
    parser.add_argument('--use-indirect', action='store_true',
                        help='Use SCS indirect (CG) backend for d>=48')
    parser.add_argument('--use-gap-accel', action='store_true',
                        help='Enable lasserre.gap_accelerator hook for '
                        'diagnostics and cut diversification')
    parser.add_argument('--gap-target', type=float, default=1.2802,
                        help='Target lb for gap-accel diagnostic comparison')
    parser.add_argument('--gap-abort', action='store_true',
                        help='Allow gap-accel to abort the run on '
                        'high-confidence Layer-1 failure')
    args = parser.parse_args()

    print(f"Direct SCS: d={args.d} O{args.order} bw={args.bw} GPU={args.gpu}")
    print(f"Started: {datetime.now().isoformat()}")
    import scs as _scs
    print(f"SCS {_scs.__version__}")

    # Memory monitor
    import threading
    def _mon():
        while True:
            try:
                with open('/sys/fs/cgroup/memory.current') as f:
                    u = int(f.read().strip()) / 1e9
                with open('/sys/fs/cgroup/memory.max') as f:
                    mx = f.read().strip()
                    lim = int(mx) / 1e9 if mx != 'max' else 0
                print(f"  [MEM] {u:.1f}/{lim:.0f}GB" if lim else f"  [MEM] {u:.1f}GB",
                      flush=True)
            except Exception:
                pass
            time.sleep(60)
    threading.Thread(target=_mon, daemon=True).start()

    hook = None
    if args.use_gap_accel:
        # Added 2026-04-16: gap-convergence accelerator.
        # Package path import assumes repo root on sys.path.
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from lasserre.gap_accelerator import GapAccelHook
        cfg = {'abort_on_unfavorable_diagnostic': args.gap_abort}
        hook = GapAccelHook(
            config=cfg, target_lb=args.gap_target,
            tag=f"d{args.d}_o{args.order}_bw{args.bw}_scs")

    r = solve_scs_direct(
        args.d, args.order, args.bw,
        max_cg_rounds=args.cg_rounds, n_bisect=args.bisect,
        use_gpu=args.gpu, scs_max_iters=args.scs_iters, scs_eps=args.scs_eps,
        scs_scale=args.scs_scale, use_indirect=args.use_indirect,
        hook=hook,
    )

    print(f"\n{'='*70}")
    print(f"FINAL: d={args.d} O{args.order} bw={args.bw}")
    print(f"  lb = {r['lb']:.10f}")
    print(f"  val({args.d}) = {val_d_known.get(args.d, '?')}")
    print(f"  gc = {r['gap_closure']:.2f}%")
    print(f"  time = {r['elapsed']:.1f}s = {r['elapsed']/3600:.2f}hr")
    print(f"  sound = {r['sound']}")
    if r['lb'] > 1.2802:
        print(f"  *** NEW RECORD: lb={r['lb']:.6f} > 1.2802 ***")
    print(f"{'='*70}")

    tag = f"d{args.d}_o{args.order}_bw{args.bw}_scs"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save solution vector (proof artifact)
    y_sol = r.pop('y_solution', None)
    if y_sol is not None:
        y_path = os.path.join(data_dir, f'solution_{tag}.npy')
        np.save(y_path, y_sol)
        print(f"Solution vector saved to {y_path} ({len(y_sol)} entries)")

    # Save result JSON
    out = os.path.join(data_dir, f'result_{tag}.json')
    with open(out, 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Result saved to {out}")


if __name__ == '__main__':
    main()
