"""Precompute index arrays, base constraints, and window violation checks.

Shared infrastructure used by all Lasserre solvers (CG, enhanced, high-d).
"""
import numpy as np
from scipy import sparse as sp
try:
    from mosek.fusion import Model, Domain, Expr, Matrix
except ImportError:
    Model = Domain = Expr = Matrix = None
import time

from lasserre.core import (
    enum_monomials, _make_hash_bases, _hash_monos, _hash_add,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
    get_ab_eiej_slice,
)

import os as _os
_LAZY_DEFAULT = _os.environ.get('SIDON_LAZY_ABEIEJ', '1') == '1'


# =====================================================================
# Precompute all index arrays + window COO data
# =====================================================================

def _precompute(d, order, verbose=True, lazy_ab_eiej=None):
    """Precompute monomials, hash tables, index arrays, and window COO.

    lazy_ab_eiej: if True, skip the full (n_loc, n_loc, d, d) ab_eiej_idx
    materialisation.  Downstream callers must use lasserre.core.
    get_ab_eiej_slice / get_ab_eiej_ij for access.  Default follows env
    SIDON_LAZY_ABEIEJ (default "1" → lazy).
    """
    t0 = time.time()
    if lazy_ab_eiej is None:
        lazy_ab_eiej = _LAZY_DEFAULT

    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)

    basis = enum_monomials(d, order)
    n_basis = len(basis)
    loc_basis = enum_monomials(d, order - 1) if order >= 2 else []
    n_loc = len(loc_basis)
    consist_mono = enum_monomials(d, 2 * order - 1)

    mono_list, idx = collect_moments(d, order, basis, loc_basis, consist_mono)
    n_y = len(mono_list)

    max_comp = 2 * order
    bases, prime = _make_hash_bases(d, max_comp)
    sorted_h, sort_o = _build_hash_table(mono_list, bases, prime)

    B_arr = np.array(basis, dtype=np.int64)
    E_arr = np.eye(d, dtype=np.int64)

    B_hash = _hash_monos(B_arr, bases, prime)
    AB_hash = _hash_add(B_hash[:, None], B_hash[None, :], prime)
    moment_pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()
    moment_pick_list = moment_pick.tolist()  # for MOSEK y.pick()

    loc_picks = []
    t_pick = None
    t_pick_list = None
    AB_loc_hash = None
    ab_eiej_idx = None
    ab_flat = None
    if loc_basis:
        LB_arr = np.array(loc_basis, dtype=np.int64)
        AB_loc = LB_arr[:, None, :] + LB_arr[None, :, :]
        AB_loc_hash = _hash_monos(AB_loc, bases, prime)
        t_pick = _hash_lookup(AB_loc_hash, sorted_h, sort_o).ravel()
        t_pick_list = t_pick.tolist()

        for i_var in range(d):
            h = _hash_add(AB_loc_hash, bases[i_var], prime)
            picks = _hash_lookup(h, sorted_h, sort_o)
            assert np.all(picks >= 0), f"Missing moments in mu_{i_var} loc"
            loc_picks.append(picks.ravel())

        if lazy_ab_eiej:
            ab_eiej_idx = None
            if verbose:
                sz_gb = (n_loc * n_loc * d * d) * 8 / 1e9
                print(f"  ab_eiej_idx: LAZY (would have been "
                      f"({n_loc},{n_loc},{d},{d}) = {sz_gb:.2f}GB)",
                      flush=True)
        else:
            EE_hash = _hash_add(bases[:, None], bases[None, :], prime)
            ABIJ_hash = _hash_add(AB_loc_hash[:, :, None, None],
                                  EE_hash[None, None, :, :], prime)
            ab_eiej_idx = _hash_lookup(ABIJ_hash, sorted_h, sort_o)

        ab_flat = (np.arange(n_loc)[:, None] * n_loc
                   + np.arange(n_loc)[None, :])

    EE_deg2 = E_arr[:, None, :] + E_arr[None, :, :]
    EE_deg2_hash = _hash_monos(EE_deg2, bases, prime)
    idx_ij = _hash_lookup(EE_deg2_hash, sorted_h, sort_o)

    consist_arr = np.array(consist_mono, dtype=np.int64)
    consist_hash = _hash_monos(consist_arr, bases, prime)
    consist_idx = _hash_lookup(consist_hash, sorted_h, sort_o)
    consist_ei_hash = _hash_add(consist_hash[:, None], bases[None, :], prime)
    consist_ei_idx = _hash_lookup(consist_ei_hash, sorted_h, sort_o)

    # Vectorized F_scipy construction (eliminates per-window Python loop)
    f_r_parts, f_c_parts, f_v_parts = [], [], []
    for w in range(n_win):
        Mw = M_mats[w]
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0:
            continue
        mi = idx_ij[nz_i, nz_j]
        valid = mi >= 0
        if not valid.any():
            continue
        f_r_parts.append(np.full(int(valid.sum()), w, dtype=np.int32))
        f_c_parts.append(mi[valid].astype(np.int32))
        f_v_parts.append(Mw[nz_i[valid], nz_j[valid]])

    if f_r_parts:
        f_r_all = np.concatenate(f_r_parts)
        f_c_all = np.concatenate(f_c_parts)
        f_v_all = np.concatenate(f_v_parts)
    else:
        f_r_all = np.array([], dtype=np.int32)
        f_c_all = np.array([], dtype=np.int32)
        f_v_all = np.array([], dtype=np.float64)

    F_scipy = sp.csr_matrix(
        (f_v_all, (f_r_all, f_c_all)), shape=(n_win, n_y), dtype=np.float64)

    nontrivial_windows = [w for w in range(n_win)
                          if np.any(M_mats[w] != 0)]

    elapsed = time.time() - t0
    if verbose:
        print(f"  d={d}, order={order} (degree {2*order})")
        print(f"  windows={n_win}, basis={n_basis}, "
              f"loc_basis={n_loc}, moments={n_y}")
        n_psd_base = 1 + (d if loc_basis else 0)
        print(f"  Base PSD cones: {n_psd_base} "
              f"(1x{n_basis}^2 + {d}x{n_loc}^2)")
        if ab_eiej_idx is not None:
            valid_ct = int((ab_eiej_idx >= 0).sum())
            print(f"  ab_eiej_idx: {ab_eiej_idx.shape}, "
                  f"{valid_ct}/{ab_eiej_idx.size} valid")
        print(f"  Precompute: {elapsed:.2f}s", flush=True)

    return {
        'd': d, 'order': order,
        'windows': windows, 'M_mats': M_mats, 'n_win': n_win,
        'basis': basis, 'n_basis': n_basis,
        'loc_basis': loc_basis, 'n_loc': n_loc,
        'mono_list': mono_list, 'idx': idx, 'n_y': n_y,
        'bases': bases, 'prime': prime,
        'sorted_h': sorted_h, 'sort_o': sort_o,
        'moment_pick': moment_pick_list,
        'moment_pick_np': moment_pick,
        'loc_picks': [lp.tolist() if isinstance(lp, np.ndarray) else lp for lp in loc_picks],
        'loc_picks_np': loc_picks,
        't_pick': t_pick_list if t_pick is not None else t_pick,
        't_pick_np': t_pick,
        'AB_loc_hash': AB_loc_hash, 'ab_eiej_idx': ab_eiej_idx,
        'ab_flat': ab_flat,
        'lazy_ab_eiej': bool(lazy_ab_eiej),
        'idx_ij': idx_ij,
        'consist_mono': consist_mono,
        'consist_idx': consist_idx, 'consist_ei_idx': consist_ei_idx,
        'f_r': f_r_all, 'f_c': f_c_all, 'f_v': f_v_all,
        'F_coo_lists': (f_r_all, f_c_all, f_v_all),
        'F_scipy': F_scipy,
        'nontrivial_windows': nontrivial_windows,
    }


# =====================================================================
# Build base MOSEK model (moment + localizing + consistency)
# =====================================================================

def _build_base_constraints(mdl, y, P, add_upper_loc, verbose):
    """Add moment matrix PSD, mu_i localizing PSD, consistency, y_0=1."""
    d = P['d']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    idx = P['idx']

    zero = tuple(0 for _ in range(d))
    mdl.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    c_rows, c_cols, c_vals = [], [], []
    n_added = 0
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        has_child = False
        for ci in range(d):
            if child_idx[ci] >= 0:
                c_rows.append(n_added)
                c_cols.append(int(child_idx[ci]))
                c_vals.append(1.0)
                has_child = True
        if not has_child:
            continue
        c_rows.append(n_added)
        c_cols.append(ai)
        c_vals.append(-1.0)
        n_added += 1

    if n_added > 0:
        A_con = Matrix.sparse(n_added, n_y, c_rows, c_cols, c_vals)
        mdl.constraint("consist", Expr.mul(A_con, y), Domain.equalsTo(0.0))

    M_mat = Expr.reshape(y.pick(P['moment_pick']), n_basis, n_basis)
    mdl.constraint("moment_psd", M_mat, Domain.inPSDCone(n_basis))

    if P['order'] >= 2:
        for i_var in range(d):
            Li = Expr.reshape(y.pick(P['loc_picks'][i_var]), n_loc, n_loc)
            mdl.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_loc))

        if add_upper_loc:
            for i_var in range(d):
                sub_moment = y.pick(P['t_pick'])
                mu_i_loc = y.pick(P['loc_picks'][i_var])
                diff_i = Expr.sub(sub_moment, mu_i_loc)
                L_upper = Expr.reshape(diff_i, n_loc, n_loc)
                mdl.constraint(f"loc_upper_{i_var}",
                               L_upper, Domain.inPSDCone(n_loc))
            if verbose:
                print(f"  Added {d} upper-bound localizing "
                      f"(1-mu_i) >= 0 constraints")

    n_psd = 1 + (d if P['order'] >= 2 else 0)
    if add_upper_loc and P['order'] >= 2:
        n_psd += d
    if verbose:
        print(f"  Consistency constraints: {n_added}")
        print(f"  Total PSD cones (base): {n_psd}", flush=True)

    return n_added


# =====================================================================
# Vectorized helpers
# =====================================================================

def _eval_all_windows(y_vals, P):
    """f_W(y) for all W via sparse matvec."""
    return P['F_scipy'].dot(y_vals)


def _check_window_violations(y_vals, t_val, P, active_windows, tol=1e-6):
    """Batch-check all non-active windows for localizing matrix violations.

    Lazy-aware: when P['ab_eiej_idx'] is None, computes per-window slices
    on demand from AB_loc_hash.
    """
    n_loc = P['n_loc']
    ab_eiej_idx = P['ab_eiej_idx']

    if n_loc == 0:
        return []
    if ab_eiej_idx is None and P.get('AB_loc_hash') is None:
        return []

    t_pick_arr = np.array(P['t_pick'])
    L_t = t_val * y_vals[t_pick_arr].reshape(n_loc, n_loc)

    check_ws = [w for w in P['nontrivial_windows'] if w not in active_windows]
    if not check_ws:
        return []

    violations = []
    if ab_eiej_idx is not None:
        safe_idx = np.clip(ab_eiej_idx, 0, len(y_vals) - 1)
        y_abij = y_vals[safe_idx]
        y_abij[ab_eiej_idx < 0] = 0.0
        for w in check_ws:
            Mw = P['M_mats'][w]
            L_q = np.einsum('ij,abij->ab', Mw, y_abij)
            L_w = L_t - L_q
            L_w = 0.5 * (L_w + L_w.T)
            min_eig = np.linalg.eigvalsh(L_w)[0]
            if min_eig < -tol:
                violations.append((w, float(min_eig)))
    else:
        for w in check_ws:
            Mw = P['M_mats'][w]
            nz_i, nz_j = np.nonzero(Mw)
            if len(nz_i) == 0:
                continue
            y_idx = get_ab_eiej_slice(P, nz_i, nz_j)
            safe = np.clip(y_idx, 0, len(y_vals) - 1)
            y_abk = y_vals[safe]
            y_abk[y_idx < 0] = 0.0
            mw_vals = Mw[nz_i, nz_j]
            L_q = np.tensordot(y_abk, mw_vals, axes=([2], [0]))
            L_w = L_t - L_q
            L_w = 0.5 * (L_w + L_w.T)
            min_eig = np.linalg.eigvalsh(L_w)[0]
            if min_eig < -tol:
                violations.append((w, float(min_eig)))

    violations.sort(key=lambda x: x[1])
    return violations


def _add_psd_window(mdl, y, t_param, w, P):
    """Add PSD localizing constraint for window w (on-demand COO).

    Lazy-aware: uses get_ab_eiej_slice so neither the full 4D array
    nor a materialised slice hierarchy is required.
    """
    n_loc = P['n_loc']
    n_y = P['n_y']
    ab_flat = P['ab_flat']
    Mw = P['M_mats'][w]

    t_y = Expr.mul(t_param, y.pick(P['t_pick']))

    nz_i, nz_j = np.nonzero(Mw)
    if len(nz_i) == 0 or (
            P.get('ab_eiej_idx') is None and P.get('AB_loc_hash') is None):
        Lw_mat = Expr.reshape(t_y, n_loc, n_loc)
        mdl.constraint(f"w_psd_{w}", Lw_mat, Domain.inPSDCone(n_loc))
        return

    y_idx = get_ab_eiej_slice(P, nz_i, nz_j)
    valid = y_idx >= 0
    if not np.any(valid):
        Lw_mat = Expr.reshape(t_y, n_loc, n_loc)
        mdl.constraint(f"w_psd_{w}", Lw_mat, Domain.inPSDCone(n_loc))
        return

    ab_exp = np.broadcast_to(ab_flat[:, :, None], y_idx.shape)
    mw_vals = Mw[nz_i, nz_j]
    mw_exp = np.broadcast_to(mw_vals[None, None, :], y_idx.shape)

    rows = ab_exp[valid].ravel().tolist()
    cols = y_idx[valid].ravel().tolist()
    vals = mw_exp[valid].ravel().tolist()

    flat_size = n_loc * n_loc
    Cw_mosek = Matrix.sparse(flat_size, n_y, rows, cols, vals)
    cw_expr = Expr.mul(Cw_mosek, y)
    Lw_flat = Expr.sub(t_y, cw_expr)
    Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)
    mdl.constraint(f"w_psd_{w}", Lw_mat, Domain.inPSDCone(n_loc))
