#!/usr/bin/env python
r"""
Lasserre SDP via SCS — first-order solver for high-dimensional val(d).

Mathematically identical to lasserre_scalable.py (CG mode), but uses SCS
(Splitting Conic Solver) instead of MOSEK's interior-point method.

SCS uses O(nnz) memory instead of O(n_y^2), enabling configurations far
beyond MOSEK's Schur complement memory wall:

  L2 d=32:   MOSEK Schur ~26GB   |  SCS ~4MB,   <1 min
  L2 d=64:   MOSEK Schur ~5TB    |  SCS ~50MB,  ~6 min
  L2 d=128:  MOSEK Schur ~1PB    |  SCS ~650MB, ~8 h

The same _precompute() infrastructure from lasserre_scalable.py is reused.
The mathematical formulation is identical — same cones, same constraints.

SCS 3.x backends:
  direct:   sparse LDL factorization (QDLDL), best for d <= 32
  indirect: CG-based, no factorization, best for d >= 48

Usage:
  python tests/lasserre_scs.py --d 8 --order 2
  python tests/lasserre_scs.py --d 16 --order 3 --cg-rounds 5
  python tests/lasserre_scs.py --d 64 --order 2 --use-indirect
  python tests/lasserre_scs.py --d 128 --order 2 --use-indirect --max-iters 20000
"""

import numpy as np
from scipy import sparse as sp
import scs
import time
import sys
import os
import gc
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_fusion import (
    enum_monomials, _make_hash_bases, _hash_monos,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
)
# Reuse the optimized precompute + violation checker from lasserre_scalable
from lasserre_scalable import _precompute, _check_window_violations


# =====================================================================
# SCS svec helpers
# =====================================================================

def _svec_idx(i, j, n):
    """Svec index for entry (i,j) of an n×n symmetric matrix, i >= j.

    Column-major lower triangle (SCS 3.x convention).
    """
    # Entries before column j: sum_{k=0}^{j-1} (n - k) = j*n - j*(j-1)//2
    return j * n - j * (j - 1) // 2 + (i - j)


_SQRT2 = np.sqrt(2.0)


def _svec_scale(i, j):
    """Scale factor: sqrt(2) for off-diagonal, 1 for diagonal."""
    return _SQRT2 if i != j else 1.0


def _svec_dim(n):
    """Number of entries in svec of n×n symmetric matrix."""
    return n * (n + 1) // 2


# --- Precomputed lower-triangle svec arrays (cached per matrix size) ---
_svec_cache = {}

def _get_svec_arrays(n):
    """Return (a_rows, b_cols, k_svec, sc_scale, flat_idx) for lower triangle.

    a_rows >= b_cols (lower triangle), matching the original loop order:
      for b_col in range(n): for a_row in range(b_col, n).
    """
    if n in _svec_cache:
        return _svec_cache[n]
    # np.tril_indices returns (row_idx, col_idx) with row >= col
    a_rows, b_cols = np.tril_indices(n)
    # svec index: j*n - j*(j-1)//2 + (i-j) where i=a_row, j=b_col
    k_svec = b_cols * n - b_cols * (b_cols - 1) // 2 + (a_rows - b_cols)
    sc = np.where(a_rows != b_cols, _SQRT2, 1.0)
    flat = a_rows * n + b_cols
    _svec_cache[n] = (a_rows, b_cols, k_svec, sc, flat)
    return _svec_cache[n]


def _build_psd_block(n, pick_list, row_offset, sign=-1.0):
    """Build COO entries for a PSD cone (vectorized). Returns numpy arrays."""
    a_rows, b_cols, k_svec, sc, flat = _get_svec_arrays(n)
    pick = np.asarray(pick_list)
    y_idx = pick[flat]
    mask = y_idx >= 0
    rows = (row_offset + k_svec[mask]).astype(np.intp)
    cols = y_idx[mask].astype(np.intp)
    vals = sign * sc[mask]
    return rows, cols, vals


def _build_psd_block_diff(n, pick_plus, pick_minus, row_offset):
    """Build COO for PSD cone: svec(Y+ - Y-) (vectorized). Returns numpy arrays."""
    a_rows, b_cols, k_svec, sc, flat = _get_svec_arrays(n)
    pp = np.asarray(pick_plus)
    pm = np.asarray(pick_minus)
    y_plus = pp[flat]
    y_minus = pm[flat]

    r_parts, c_parts, v_parts = [], [], []
    mask_p = y_plus >= 0
    if mask_p.any():
        r_parts.append((row_offset + k_svec[mask_p]).astype(np.intp))
        c_parts.append(y_plus[mask_p].astype(np.intp))
        v_parts.append(-sc[mask_p])
    mask_m = y_minus >= 0
    if mask_m.any():
        r_parts.append((row_offset + k_svec[mask_m]).astype(np.intp))
        c_parts.append(y_minus[mask_m].astype(np.intp))
        v_parts.append(sc[mask_m].copy())
    if r_parts:
        return np.concatenate(r_parts), np.concatenate(c_parts), np.concatenate(v_parts)
    return np.array([], dtype=np.intp), np.array([], dtype=np.intp), np.array([], dtype=np.float64)


def _build_window_psd_block(n_loc, t_val, t_pick, ab_eiej_idx, Mw,
                            row_offset, n_y):
    """Build COO for window PSD cone (vectorized). Returns numpy arrays."""
    a_rows, b_cols, k_svec, sc, flat = _get_svec_arrays(n_loc)

    r_parts, c_parts, v_parts = [], [], []

    # t * y[t_pick[ab]] term  => A contribution: -t * sc
    tp = np.asarray(t_pick)
    t_idx = tp[flat]
    mask_t = (t_idx >= 0) & (t_val != 0)
    if mask_t.any():
        r_parts.append((row_offset + k_svec[mask_t]).astype(np.intp))
        c_parts.append(t_idx[mask_t].astype(np.intp))
        v_parts.append(-t_val * sc[mask_t])

    # - sum_ij M_W[i,j] * y[abij[a,b,i,j]]  => A contribution: +sc * M_W[i,j]
    if ab_eiej_idx is not None:
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) > 0:
            nz_coeffs = Mw[nz_i, nz_j]
            y_cols = ab_eiej_idx[a_rows, b_cols][:, nz_i, nz_j]
            mask_valid = y_cols >= 0

            r_base = (row_offset + k_svec)[:, None]
            sc_2d = sc[:, None]
            coeff_2d = nz_coeffs[None, :]

            r_parts.append(np.broadcast_to(r_base, mask_valid.shape)[mask_valid].astype(np.intp))
            c_parts.append(y_cols[mask_valid].astype(np.intp))
            v_parts.append((sc_2d * coeff_2d)[mask_valid])

    if r_parts:
        return np.concatenate(r_parts), np.concatenate(c_parts), np.concatenate(v_parts)
    return np.array([], dtype=np.intp), np.array([], dtype=np.intp), np.array([], dtype=np.float64)


def _build_window_psd_block_split(n_loc, t_pick, ab_eiej_idx, Mw,
                                  row_offset):
    """Build COO for window PSD cone, split into t-dependent and t-independent.

    Returns (base_rows, base_cols, base_vals,
             t_rows, t_cols, t_vals) — all numpy arrays.
    where: full A entries = base + t_val * t_part.
    """
    a_rows, b_cols, k_svec, sc, flat = _get_svec_arrays(n_loc)

    base_r_parts, base_c_parts, base_v_parts = [], [], []
    t_r_parts, t_c_parts, t_v_parts = [], [], []

    # t * y[t_pick[ab]] => t-dependent part: coefficient is -sc (multiply by t later)
    tp = np.asarray(t_pick)
    t_idx = tp[flat]
    mask_t = t_idx >= 0
    if mask_t.any():
        t_r_parts.append((row_offset + k_svec[mask_t]).astype(np.intp))
        t_c_parts.append(t_idx[mask_t].astype(np.intp))
        t_v_parts.append(-sc[mask_t])

    # M_W terms => t-independent (base)
    if ab_eiej_idx is not None:
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) > 0:
            nz_coeffs = Mw[nz_i, nz_j]
            y_cols = ab_eiej_idx[a_rows, b_cols][:, nz_i, nz_j]
            mask_valid = y_cols >= 0

            r_base = (row_offset + k_svec)[:, None]
            sc_2d = sc[:, None]
            coeff_2d = nz_coeffs[None, :]

            base_r_parts.append(np.broadcast_to(r_base, mask_valid.shape)[mask_valid].astype(np.intp))
            base_c_parts.append(y_cols[mask_valid].astype(np.intp))
            base_v_parts.append((sc_2d * coeff_2d)[mask_valid])

    _cat = np.concatenate
    _empty_i = np.array([], dtype=np.intp)
    _empty_f = np.array([], dtype=np.float64)
    br = _cat(base_r_parts) if base_r_parts else _empty_i
    bc = _cat(base_c_parts) if base_c_parts else _empty_i
    bv = _cat(base_v_parts) if base_v_parts else _empty_f
    tr = _cat(t_r_parts) if t_r_parts else _empty_i
    tc = _cat(t_c_parts) if t_c_parts else _empty_i
    tv = _cat(t_v_parts) if t_v_parts else _empty_f
    return (br, bc, bv, tr, tc, tv)


# =====================================================================
# Build the complete SCS problem
# =====================================================================

def _precompute_scs_decomposition(P, add_upper_loc, active_windows):
    """Precompute the t-independent and t-dependent parts of the SCS problem.

    A = A_base + t_val * A_t,  b = b_base (with t_val in scalar window slots).
    Build once per CG round; assemble cheaply per bisection step.
    """
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    n_win = P['n_win']
    idx = P['idx']

    TAU = n_y
    n_x = n_y + 1

    # 1. Zero cone (equalities)
    zero_eq = tuple(0 for _ in range(d))
    zero_idx = idx[zero_eq]

    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    consist_rows_data = []
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child = consist_ei_idx[r]
        c_cols, c_vals = [], []
        has_child = False
        for ci in range(d):
            if child[ci] >= 0:
                c_cols.append(int(child[ci]))
                c_vals.append(1.0)
                has_child = True
        if not has_child:
            continue
        c_cols.append(ai)
        c_vals.append(-1.0)
        consist_rows_data.append((c_cols, c_vals))
    n_consist = len(consist_rows_data)
    n_z = 1 + n_consist

    # 2. Nonneg cone
    n_l = n_y + n_win + 1

    # 3. PSD cones
    psd_sizes = [n_basis]
    if order >= 2:
        psd_sizes.extend([n_loc] * d)
        if add_upper_loc:
            psd_sizes.extend([n_loc] * d)
    for _ in active_windows:
        psd_sizes.append(n_loc)

    psd_svec_total = sum(_svec_dim(s) for s in psd_sizes)
    m = n_z + n_l + psd_svec_total

    # ── Build base and t-coeff COO arrays (numpy accumulation) ──
    base_r_parts, base_c_parts, base_v_parts = [], [], []
    t_r_parts, t_c_parts, t_v_parts = [], [], []
    b_base = np.zeros(m)

    row = 0

    # --- Zero cone: y_0 = 1 ---
    base_r_parts.append(np.array([row], dtype=np.intp))
    base_c_parts.append(np.array([zero_idx], dtype=np.intp))
    base_v_parts.append(np.array([1.0]))
    b_base[row] = 1.0
    row += 1

    # --- Zero cone: consistency (small, Python loop is fine) ---
    consist_r, consist_c, consist_v = [], [], []
    for c_cols, c_vals in consist_rows_data:
        for c, v in zip(c_cols, c_vals):
            consist_r.append(row)
            consist_c.append(c)
            consist_v.append(v)
        row += 1
    if consist_r:
        base_r_parts.append(np.array(consist_r, dtype=np.intp))
        base_c_parts.append(np.array(consist_c, dtype=np.intp))
        base_v_parts.append(np.array(consist_v))
    assert row == n_z

    # --- Nonneg cone: y >= 0 ---
    y_range = np.arange(n_y, dtype=np.intp)
    base_r_parts.append(row + y_range)
    base_c_parts.append(y_range)
    base_v_parts.append(np.full(n_y, -1.0))
    row += n_y

    # --- Nonneg cone: scalar windows (b = t_val, A is fixed) ---
    scalar_win_start = row
    F_coo = P['F_scipy'].tocoo()
    base_r_parts.append((row + F_coo.row).astype(np.intp))
    base_c_parts.append(F_coo.col.astype(np.intp))
    base_v_parts.append(F_coo.data.astype(np.float64))
    row += n_win

    # --- Nonneg cone: tau >= 0 ---
    base_r_parts.append(np.array([row], dtype=np.intp))
    base_c_parts.append(np.array([TAU], dtype=np.intp))
    base_v_parts.append(np.array([-1.0]))
    row += 1
    assert row == n_z + n_l

    # --- PSD cone: moment matrix ---
    r2, c2, v2 = _build_psd_block(n_basis, P['moment_pick'], row)
    base_r_parts.append(r2); base_c_parts.append(c2); base_v_parts.append(v2)
    row += _svec_dim(n_basis)

    # --- PSD cones: mu_i localizing ---
    if order >= 2:
        for i_var in range(d):
            r2, c2, v2 = _build_psd_block(n_loc, P['loc_picks'][i_var], row)
            base_r_parts.append(r2); base_c_parts.append(c2); base_v_parts.append(v2)
            row += _svec_dim(n_loc)

        if add_upper_loc:
            for i_var in range(d):
                r2, c2, v2 = _build_psd_block_diff(
                    n_loc, P['t_pick'], P['loc_picks'][i_var], row)
                base_r_parts.append(r2); base_c_parts.append(c2); base_v_parts.append(v2)
                row += _svec_dim(n_loc)

    # --- PSD cones: window localizing (split into base + t-coeff) ---
    diag_k = np.array([_svec_idx(a, a, n_loc) for a in range(n_loc)])

    for w in sorted(active_windows):
        br, bc, bv, tr, tc, tv = _build_window_psd_block_split(
            n_loc, P['t_pick'], P['ab_eiej_idx'],
            P['M_mats'][w], row)
        base_r_parts.append(br); base_c_parts.append(bc); base_v_parts.append(bv)
        if len(tr) > 0:
            t_r_parts.append(tr); t_c_parts.append(tc); t_v_parts.append(tv)

        # tau*I contribution (base, not t-dependent)
        base_r_parts.append((row + diag_k).astype(np.intp))
        base_c_parts.append(np.full(n_loc, TAU, dtype=np.intp))
        base_v_parts.append(np.full(n_loc, -1.0))
        row += _svec_dim(n_loc)

    assert row == m, f"row mismatch: {row} != {m}"

    # ── Assemble sparse matrices (single concatenation) ──
    base_rows = np.concatenate(base_r_parts)
    base_cols = np.concatenate(base_c_parts)
    base_vals = np.concatenate(base_v_parts)
    A_base_csc = sp.csc_matrix(
        (base_vals, (base_rows, base_cols)), shape=(m, n_x), dtype=np.float64)

    has_t = bool(t_r_parts)
    if has_t:
        t_rows = np.concatenate(t_r_parts)
        t_cols = np.concatenate(t_c_parts)
        t_vals = np.concatenate(t_v_parts)
        A_t_csc = sp.csc_matrix(
            (t_vals, (t_rows, t_cols)), shape=(m, n_x), dtype=np.float64)
    else:
        A_t_csc = sp.csc_matrix((m, n_x), dtype=np.float64)

    c_vec = np.zeros(n_x)
    c_vec[TAU] = 1.0

    cone = {'z': n_z, 'l': n_l, 's': psd_sizes}

    # ── Pre-compute aligned data arrays for in-place A updates ──
    # A(t) = A_base + t * A_t. Instead of scipy sparse addition per step,
    # compute the union sparsity pattern once and update .data in-place.
    if has_t:
        # Union pattern via A_base + A_t and A_base + 2*A_t
        A_template = (A_base_csc + A_t_csc).tocsc()
        A_template.sort_indices()
        A_ref_2t = (A_base_csc + 2.0 * A_t_csc).tocsc()
        A_ref_2t.sort_indices()
        t_data = A_ref_2t.data - A_template.data   # A_t at union positions
        base_data = A_template.data - t_data        # A_base at union positions
    else:
        A_template = A_base_csc.tocsc()
        A_template.sort_indices()
        t_data = None
        base_data = A_template.data.copy()

    return {
        'A_base': A_base_csc,
        'A_t': A_t_csc,
        'A_template': A_template,
        'A_base_data': base_data,
        'A_t_data': t_data,
        'has_t': has_t,
        'b_base': b_base,
        'c': c_vec,
        'cone': cone,
        'm': m,
        'n_x': n_x,
        'scalar_win_start': scalar_win_start,
        'n_win': n_win,
    }


def _assemble_scs_problem(decomp, t_val):
    """Cheaply assemble A and b for a given t_val from precomputed decomposition.

    Uses in-place data array update on the pre-allocated A_template to avoid
    scipy sparse matrix creation overhead per bisection step.
    """
    A = decomp['A_template']
    if decomp['has_t']:
        # In-place update: A.data = base + t * t_part (no new sparse matrix)
        np.add(decomp['A_base_data'], t_val * decomp['A_t_data'],
               out=A.data)
    # else: A is already A_base (constant), no update needed
    b = decomp['b_base'].copy()
    s = decomp['scalar_win_start']
    b[s:s + decomp['n_win']] = t_val
    return A, b, decomp['c'], decomp['cone']


def _build_scs_problem(P, t_val, add_upper_loc, active_windows):
    """Build the SCS standard-form data (A, b, c, cone) for fixed t.

    SCS form: min c^T x  s.t.  Ax + s = b,  s in K

    Variables:  x = [y (n_y), tau (1)]
      y:   moment variables
      tau: phase-1 slack (measures infeasibility of window PSD constraints)

    Objective: min tau  (tau=0 iff feasible at t, tau>0 iff infeasible)

    Cones (in order):
      z (zero/equality): y_0 = 1, consistency
      l (nonneg):        y >= 0, scalar window t >= TV_W(y), tau >= 0
      s (PSD):           moment matrix, localizing mu_i, upper-loc, window PSD
    """
    decomp = _precompute_scs_decomposition(P, add_upper_loc, active_windows)
    return _assemble_scs_problem(decomp, t_val)


# =====================================================================
# SCS solver with constraint generation
# =====================================================================

def solve_scs_cg(d, c_target=1.28, order=2, n_bisect=15,
                 add_upper_loc=True, cg_rounds=5, cg_add_per_round=10,
                 conv_tol=1e-7, use_indirect=False,
                 max_iters=10000, eps_abs=1e-5, eps_rel=1e-5,
                 scs_scale=None,
                 verbose=True):
    """Lasserre SDP via SCS with constraint generation.

    Mathematically identical to lasserre_scalable.solve_cg(), but uses SCS
    instead of MOSEK.  SCS uses O(nnz) memory, enabling d=64-128 at L2.

    Parameters
    ----------
    d : int
        Number of bins (dimension).
    c_target : float
        Target lower bound.
    order : int
        Lasserre order (2 = L2, 3 = L3).
    n_bisect : int
        Binary search steps per CG round.
    add_upper_loc : bool
        Add (1-mu_i) >= 0 localizing constraints.
    cg_rounds : int
        Maximum constraint generation rounds.
    cg_add_per_round : int
        Windows to add per CG round.
    conv_tol : float
        Stop CG if improvement < conv_tol.
    use_indirect : bool
        Use SCS indirect (CG) backend instead of direct (LDL).
    max_iters : int
        SCS max iterations per solve.
    eps_abs, eps_rel : float
        SCS convergence tolerances.
    scs_scale : float or None
        SCS scale parameter. scale=1.0 gives ~7x speedup for d<=12 but
        may diverge for larger d. Default None uses SCS default (0.1).
    verbose : bool
        Print progress.
    """
    t_wall = time.time()

    # ── Precompute (reuses lasserre_scalable infrastructure) ──
    P = _precompute(d, order, verbose)
    n_win = P['n_win']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']

    if verbose:
        backend = "indirect (CG)" if use_indirect else "direct (QDLDL)"
        print(f"  SCS backend: {backend}")
        print(f"  SCS settings: max_iters={max_iters}, "
              f"eps_abs={eps_abs:.0e}, eps_rel={eps_rel:.0e}")
        svec_moment = _svec_dim(n_basis)
        svec_loc = d * _svec_dim(n_loc) if order >= 2 else 0
        svec_upper = d * _svec_dim(n_loc) if (order >= 2 and add_upper_loc) else 0
        print(f"  Moment PSD svec: {svec_moment:,}")
        print(f"  Localizing svec: {svec_loc:,}")
        if add_upper_loc:
            print(f"  Upper-loc svec:  {svec_upper:,}")
        print(flush=True)

    # ── SCS solve helper ──
    # decomp_cache: reuse precomputed A decomposition within a CG round
    decomp_cache = [None]  # mutable container for closure
    # solver_cache: reuse SCS solver when A is constant (only b changes)
    # Format: (active_ws_frozenset, solver, decomp)
    solver_cache = [None]

    def check_feasible(t_val, active_ws, warm=None, scs_max_iters=None,
                       scs_eps_abs=None, scs_eps_rel=None):
        """Build SCS problem for threshold t_val, solve via phase-1.

        Uses cached A decomposition when active_ws hasn't changed.
        When A doesn't depend on t (no window PSD constraints), reuses the
        SCS solver via update(b=...) to avoid refactorization.
        Supports per-call SCS parameter overrides for adaptive tolerances.
        """
        cur_eps_abs = scs_eps_abs if scs_eps_abs is not None else eps_abs
        cur_eps_rel = scs_eps_rel if scs_eps_rel is not None else eps_rel
        cur_max_iters = scs_max_iters if scs_max_iters is not None else max_iters

        t0 = time.time()

        # Use cached decomposition if active windows haven't changed
        cached = decomp_cache[0]
        if cached is not None and cached[0] == active_ws:
            decomp = cached[1]
        else:
            decomp = _precompute_scs_decomposition(
                P, add_upper_loc, active_ws)
            decomp_cache[0] = (set(active_ws), decomp)
            solver_cache[0] = None  # invalidate solver cache on new decomp

        A, b, c_vec, cone = _assemble_scs_problem(decomp, t_val)
        build_t = time.time() - t0

        # When A is constant (no t-dependent PSD entries), reuse SCS solver
        # via update(b=...) — avoids costly LDL refactorization.
        ws_key = frozenset(active_ws)
        sc = solver_cache[0]
        if (not decomp['has_t'] and sc is not None
                and sc[0] == ws_key
                and sc[1] is not None):
            solver = sc[1]
            solver.update(b=b)
        else:
            scs_kwargs = dict(
                use_indirect=use_indirect,
                max_iters=cur_max_iters,
                eps_abs=cur_eps_abs,
                eps_rel=cur_eps_rel,
                verbose=False,
            )
            if scs_scale is not None:
                scs_kwargs['scale'] = scs_scale
            solver = scs.SCS(
                {'A': A, 'b': b, 'c': c_vec}, cone, **scs_kwargs)
            if not decomp['has_t']:
                solver_cache[0] = (ws_key, solver)

        # Warm start if available (must match dimensions)
        if warm is not None and warm['x'].shape[0] == A.shape[1]:
            sol = solver.solve(
                warm_start=True,
                x=warm['x'], y=warm['y'], s=warm['s'])
        else:
            sol = solver.solve()

        solve_t = time.time() - t0 - build_t
        status = sol['info']['status']
        iters = sol['info']['iter']

        # Extract tau (last variable in x)
        TAU_idx = P['n_y']
        tau_val = sol['x'][TAU_idx] if sol['x'] is not None else float('inf')

        # Feasible iff tau <= small threshold (tau=0 is exact feasibility)
        tau_tol = max(cur_eps_abs * 10, 1e-4)
        feasible = (status in ('solved', 'solved_inaccurate') and
                    tau_val <= tau_tol)

        y_vals = sol['x'][:P['n_y']] if feasible else None

        # Save warm-start data
        warm_out = {'x': sol['x'], 'y': sol['y'], 's': sol['s']}

        return feasible, y_vals, warm_out, {
            'status': status, 'iters': iters, 'tau': tau_val,
            'build_s': build_t, 'solve_s': solve_t,
        }

    # ── Round 0: seed violations ──
    active_windows = set()
    best_lb = 0.0
    warm_data = None

    if verbose:
        print(f"\n  [Round 0] Seeding initial violations...", flush=True)

    # Quick 5-step bisection to find scalar-only boundary (~1.0)
    lo_seed, hi_seed = 0.5, 3.0
    feas, _, warm_data, info = check_feasible(hi_seed, active_windows)
    if not feas:
        hi_seed = 10.0
        feas, _, warm_data, info = check_feasible(hi_seed, active_windows)

    for _ in range(5):
        mid = (lo_seed + hi_seed) / 2
        feas, y_vals, warm_data, info = check_feasible(
            mid, active_windows, warm_data)
        if feas:
            hi_seed = mid
        else:
            lo_seed = mid

    # Get y* at feasible boundary to find initial violations
    feas, y_vals, warm_data, info = check_feasible(
        hi_seed, active_windows, warm_data)
    if not feas or y_vals is None:
        raise RuntimeError(f"SCS infeasible at t={hi_seed:.4f}: {info['status']}")

    violations = _check_window_violations(y_vals, hi_seed, P, active_windows)

    if verbose:
        print(f"    Scalar boundary ~ {lo_seed:.6f}, "
              f"{len(violations)} violations found "
              f"(SCS: {info['iters']} iters, tau={info['tau']:.2e}, "
              f"{info['solve_s']:.1f}s)", flush=True)

    if len(violations) == 0:
        best_lb = lo_seed
        elapsed = time.time() - t_wall
        if verbose:
            print(f"    No violations — scalar bound is exact.")
        return {'lb': best_lb, 'proven': best_lb >= c_target - 1e-6,
                'elapsed': elapsed, 'd': d, 'order': order, 'mode': 'scs_cg',
                'n_active_windows': 0}

    n_add = min(cg_add_per_round, len(violations))
    for w, min_eig in violations[:n_add]:
        active_windows.add(w)
    if verbose:
        print(f"    Added {n_add} PSD windows "
              f"(worst eig: {violations[0][1]:.6e})")

    # ── CG rounds 1+: binary search + violation check ──
    for cg_round in range(1, cg_rounds + 1):
        if verbose:
            print(f"\n  [CG round {cg_round}] "
                  f"{len(active_windows)} PSD window constraints",
                  flush=True)

        # Tight binary search range
        lo = max(0.5, best_lb - 0.01)
        hi = best_lb * 1.05 + 0.15 if best_lb > 0.5 else 5.0

        # Invalidate decomp cache since active_windows changed
        decomp_cache[0] = None

        # Ensure hi is feasible
        feas, _, warm_data, info = check_feasible(hi, active_windows)
        while not feas:
            hi *= 1.5
            if hi > 100:
                break
            feas, _, warm_data, info = check_feasible(hi, active_windows)
        if not feas:
            if verbose:
                print(f"    WARNING: infeasible up to t={hi:.2f}")
            break

        # Adaptive bisection with graduated iteration budgets and
        # tau-interpolation.
        #
        # Key insight: binary search only needs feasible/infeasible
        # classification, not precise tau values. Capping SCS iterations
        # is safe: non-convergence → classify as infeasible → lo side →
        # conservative lower bound (never wrong).
        #
        # Graduated budgets: early steps use very few iters (coarse
        # classification), late steps use more (fine-tuning near boundary).
        # This typically halves total SCS iterations vs. uniform 2000 cap.
        #
        # Tau-interpolation: once we have tau from both sides, interpolate
        # to jump closer to the boundary, reducing wasted probes.
        tau_lo, tau_hi = None, None
        last_feas_y = None    # save y* from last feasible step
        last_feas_t = None    # and its t value
        for step in range(n_bisect):
            # Graduated iteration budget: coarse early, fine late.
            # Binary search precision comes from more steps, not per-step
            # accuracy. Early steps probe far from boundary where SCS
            # converges quickly. Loose tolerance lets SCS terminate sooner.
            if step < 3:
                adaptive_iters = min(max_iters, 800)
                adaptive_eps = eps_abs * 5
            elif step < 8:
                adaptive_iters = min(max_iters, 1500)
                adaptive_eps = eps_abs * 3
            else:
                adaptive_iters = min(max_iters, 2000)
                adaptive_eps = eps_abs * 2

            # Choose probe point: interpolation or bisection
            use_interp = False
            if (step >= 3 and tau_lo is not None and tau_hi is not None
                    and tau_lo > 0 and tau_hi < 0
                    and (hi - lo) > 1e-12):
                frac = tau_lo / (tau_lo - tau_hi)
                mid_interp = lo + frac * (hi - lo)
                mid_clamped = max(lo + 0.1 * (hi - lo),
                                  min(hi - 0.1 * (hi - lo), mid_interp))
                mid = mid_clamped
                use_interp = True
            else:
                mid = (lo + hi) / 2

            t0 = time.time()
            feas, y_mid, warm_data, info = check_feasible(
                mid, active_windows, warm_data,
                scs_max_iters=adaptive_iters,
                scs_eps_abs=adaptive_eps,
                scs_eps_rel=adaptive_eps)
            dt = time.time() - t0

            tau_val = info['tau']

            # Classify: solved/solved_inaccurate with tau<=tol → feasible
            # Everything else (infeasible, max_iters, nan tau) → infeasible
            if feas:
                hi = mid
                tau_hi = tau_val if np.isfinite(tau_val) else -1e-6
                tag = "feas"
                # Save y* from feasible solve for later violation check
                if y_mid is not None:
                    last_feas_y = y_mid
                    last_feas_t = mid
            else:
                lo = mid
                tau_lo = tau_val if (np.isfinite(tau_val) and tau_val > 0) else 1.0
                tag = "inf "

            if verbose and (step < 2 or step == n_bisect - 1
                            or (step + 1) % 5 == 0):
                interp_tag = " interp" if use_interp else ""
                print(f"    [{step+1:2d}/{n_bisect}] t={mid:.10f} {tag} "
                      f"(tau={info['tau']:.2e}, {info['iters']} iters, "
                      f"{dt:.1f}s){interp_tag}", flush=True)

            # Early termination: interval small enough
            if hi - lo < 1e-8 and step >= 5:
                if verbose:
                    print(f"    Converged: interval {hi-lo:.2e} < 1e-8",
                          flush=True)
                break

        lb = lo
        improvement = lb - best_lb
        best_lb = max(best_lb, lb)

        if verbose:
            print(f"    lb={lb:.10f} (+{improvement:.2e})", flush=True)

        # Use saved y* from bisection instead of an extra solve.
        # Falls back to a fresh solve only if no feasible y was saved.
        y_vals = last_feas_y
        hi_for_viol = last_feas_t if last_feas_t is not None else hi
        if y_vals is None:
            feas, y_vals, warm_data, info = check_feasible(
                hi, active_windows, warm_data)
            hi_for_viol = hi
            if not feas or y_vals is None:
                if verbose:
                    print(f"    WARNING: could not extract y* at t={hi:.6f}")
                break

        violations = _check_window_violations(
            y_vals, hi_for_viol, P, active_windows)

        if len(violations) == 0:
            if verbose:
                print(f"    No violations — converged.", flush=True)
            break

        # Adaptive termination
        if improvement < conv_tol and cg_round >= 2:
            if verbose:
                print(f"    Improvement {improvement:.2e} < tol "
                      f"{conv_tol:.2e} — stopping.", flush=True)
            break

        # Add most violated windows
        n_add = min(cg_add_per_round, len(violations))
        for w, min_eig in violations[:n_add]:
            active_windows.add(w)
        warm_data = None  # invalidate warm start (new cones added)
        decomp_cache[0] = None  # invalidate A decomposition (new cones)
        solver_cache[0] = None  # invalidate solver (new A structure)

        if verbose:
            print(f"    Added {n_add} PSD windows "
                  f"(worst eig: {violations[0][1]:.6e})")
            for w, eig in violations[:min(5, n_add)]:
                ell, s = P['windows'][w]
                print(f"      window ({ell},{s}): min_eig={eig:.6e}")

    gc.collect()

    elapsed = time.time() - t_wall
    proven = best_lb >= c_target - 1e-6

    if verbose:
        print(f"\n  Total wall time: {elapsed:.1f}s")
        print(f"  Best lower bound: {best_lb:.10f}")
        print(f"  Active windows: {len(active_windows)}/{n_win}")
        print(f"  Margin over {c_target}: {best_lb - c_target:.10f}")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")

    return {
        'lb': best_lb, 'proven': proven,
        'elapsed': elapsed,
        'd': d, 'order': order, 'mode': 'scs_cg',
        'n_active_windows': len(active_windows),
        'n_win_total': n_win,
        'n_y': n_y, 'n_basis': n_basis, 'n_loc': n_loc,
        'use_indirect': use_indirect,
        'max_iters': max_iters,
    }


# =====================================================================
# Sweep across dimensions
# =====================================================================

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
    32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
}


def run_sweep(dims, order=2, c_target=1.28, add_upper_loc=True,
              use_indirect=False, max_iters=10000,
              eps_abs=1e-5, eps_rel=1e-5, scs_scale=None,
              cg_rounds=5, cg_add=10, n_bisect=15,
              outdir='data', max_hours=8.0):
    """Run SCS Lasserre sweep across multiple dimensions."""
    print("=" * 72)
    print("SCS LASSERRE SWEEP")
    print(f"  Dimensions: {dims}")
    print(f"  Order: L{order} (degree {2*order})")
    print(f"  Target: {c_target}")
    print(f"  Upper-loc: {add_upper_loc}")
    print(f"  Backend: {'indirect' if use_indirect else 'direct'}")
    print(f"  Budget: {max_hours} h")
    print("=" * 72, flush=True)

    os.makedirs(outdir, exist_ok=True)
    results = []
    t_start = time.time()

    for d in dims:
        elapsed_h = (time.time() - t_start) / 3600
        remain_h = max_hours - elapsed_h

        print(f"\n{'#' * 72}")
        print(f"# d={d}, L{order}")
        print(f"# Elapsed {elapsed_h:.2f}h | Remaining {remain_h:.2f}h")
        print(f"{'#' * 72}", flush=True)

        if remain_h < 0.05:
            print("  TIME EXHAUSTED — stopping.", flush=True)
            break

        try:
            r = solve_scs_cg(
                d, c_target, order=order, n_bisect=n_bisect,
                add_upper_loc=add_upper_loc,
                cg_rounds=cg_rounds, cg_add_per_round=cg_add,
                use_indirect=use_indirect,
                max_iters=max_iters, eps_abs=eps_abs, eps_rel=eps_rel,
                scs_scale=scs_scale,
                verbose=True)

            lb = r['lb']
            v = val_d_known.get(d, 0)
            if v > 1:
                closure = (lb - 1) / (v - 1) * 100
                r['val_d'] = v
                r['gap_closure_pct'] = closure
                print(f"\n  >>> lb={lb:.8f}  val({d})~{v:.3f}  "
                      f"closure={closure:.1f}%", flush=True)
            else:
                r['val_d'] = 0
                r['gap_closure_pct'] = 0
                print(f"\n  >>> lb={lb:.8f}", flush=True)

            results.append(r)

            # Incremental save
            fname = (f"scs_L{order}_d{d}_"
                     f"{time.strftime('%Y%m%d_%H%M%S')}.json")
            outpath = os.path.join(outdir, fname)
            with open(outpath, 'w') as f:
                json.dump(r, f, indent=2, default=str)
            print(f"  Saved: {outpath}", flush=True)

        except Exception as exc:
            print(f"\n  FAILED: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            gc.collect()

    # ── Summary ──
    total_h = (time.time() - t_start) / 3600
    print(f"\n{'=' * 72}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 72}")
    hdr = (f"{'Level':<8}{'d':<6}{'lb':<16}{'val(d)':<10}"
           f"{'Closure':<12}{'Windows':<12}{'Wall':<12}")
    print(hdr)
    print("-" * 72)
    for r in results:
        level = f"L{r['order']}"
        dd = r['d']
        lb = r['lb']
        v = r.get('val_d', 0)
        cl = r.get('gap_closure_pct', 0)
        nw = r.get('n_active_windows', 0)
        wall_m = r.get('elapsed', 0) / 60
        print(f"{level:<8}{dd:<6}{lb:<16.10f}{v:<10.3f}"
              f"{cl:<12.1f}%{nw:<12}{wall_m:<12.1f}m")

    if results:
        best = max(results, key=lambda r: r['lb'])
        print(f"\n  Best lower bound: {best['lb']:.10f}  "
              f"(L{best['order']} d={best['d']})")
    else:
        print("  No results produced.")

    print(f"\n  Total wall time: {total_h:.2f}h / {max_hours}h budget")
    print(f"{'=' * 72}", flush=True)

    return results


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Lasserre SDP via SCS (first-order solver)")
    p.add_argument('--d', type=int, nargs='+', default=[8],
                   help='Dimension(s) to run. E.g. --d 8 16 32 64')
    p.add_argument('--order', type=int, default=2,
                   help='Lasserre order (default: 2)')
    p.add_argument('--c_target', type=float, default=1.28)
    p.add_argument('--bisect', type=int, default=15)
    p.add_argument('--cg-rounds', type=int, default=5)
    p.add_argument('--cg-add', type=int, default=10)
    p.add_argument('--upper-loc', dest='upper_loc',
                   action='store_true', default=True)
    p.add_argument('--no-upper-loc', dest='upper_loc',
                   action='store_false')
    p.add_argument('--use-indirect', action='store_true', default=False,
                   help='Use SCS indirect (CG) backend for large d')
    p.add_argument('--max-iters', type=int, default=10000)
    p.add_argument('--eps', type=float, default=1e-5,
                   help='SCS convergence tolerance (abs and rel)')
    p.add_argument('--scs-scale', type=float, default=None,
                   help='SCS scale parameter. 1.0 gives ~7x speedup for '
                        'd<=12 but may diverge for larger d.')
    p.add_argument('--max-hours', type=float, default=8.0)
    p.add_argument('--outdir', type=str, default='data')
    args = p.parse_args()

    if len(args.d) == 1:
        # Single dimension — run directly
        print(f"{'=' * 60}")
        print(f"SCS LASSERRE L{args.order} (degree {2*args.order})")
        print(f"Target: val({args.d[0]}) >= {args.c_target}")
        v = val_d_known.get(args.d[0])
        if v:
            print(f"Known: val({args.d[0]}) ~ {v}")
        print(f"{'=' * 60}\n", flush=True)

        r = solve_scs_cg(
            args.d[0], args.c_target, order=args.order,
            n_bisect=args.bisect, add_upper_loc=args.upper_loc,
            cg_rounds=args.cg_rounds, cg_add_per_round=args.cg_add,
            use_indirect=args.use_indirect,
            max_iters=args.max_iters,
            eps_abs=args.eps, eps_rel=args.eps,
            scs_scale=args.scs_scale,
            verbose=True)

        print(f"\n{'=' * 60}")
        if r['proven']:
            print(f"*** PROVEN: C_{{1a}} >= {args.c_target} ***")
        else:
            print(f"NOT PROVEN: lb={r['lb']:.8f} < {args.c_target}")
        print(f"{'=' * 60}")
    else:
        # Multi-dimension sweep
        run_sweep(
            args.d, order=args.order, c_target=args.c_target,
            add_upper_loc=args.upper_loc,
            use_indirect=args.use_indirect,
            max_iters=args.max_iters,
            eps_abs=args.eps, eps_rel=args.eps,
            scs_scale=args.scs_scale,
            cg_rounds=args.cg_rounds, cg_add=args.cg_add,
            n_bisect=args.bisect,
            outdir=args.outdir, max_hours=args.max_hours)


if __name__ == '__main__':
    main()
