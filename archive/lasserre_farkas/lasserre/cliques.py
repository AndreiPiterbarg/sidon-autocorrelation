"""Correlative sparsity via clique decomposition (Waki et al. 2006).

Provides banded clique construction, clique-restricted basis enumeration,
and sparse PSD constraint builders for moment, localizing, and window
localizing matrices.
"""
import numpy as np
try:
    from mosek.fusion import Domain, Expr, Matrix
except ImportError:
    Domain = Expr = Matrix = None

from lasserre.core import (
    enum_monomials, _hash_monos, _hash_add, _hash_lookup,
)
from lasserre.precompute import _add_psd_window


# =====================================================================
# Clique construction
# =====================================================================

def _build_banded_cliques(d, bandwidth):
    """Build overlapping cliques for a banded coupling graph.

    For bandwidth b, the chordal extension of the banded graph on d
    vertices has maximal cliques:
        I_c = {c, c+1, ..., c+b}  for c = 0, 1, ..., d-b-1

    These satisfy the running intersection property (RIP).
    """
    b = min(bandwidth, d - 1)
    cliques = []
    for c in range(d - b):
        cliques.append(list(range(c, c + b + 1)))
    if not cliques:
        cliques = [list(range(d))]
    return cliques


def _build_clique_basis(clique, order, d):
    """Enumerate monomials of degree <= order supported on a clique.

    Returns np.ndarray of shape (n_cb, d) with d-dimensional multi-indices.
    Vectorized: no Python loops in the embedding step.
    """
    s = len(clique)
    local_monos = enum_monomials(s, order)
    local_arr = np.array(local_monos, dtype=np.int64)  # (n_cb, s)
    n_cb = len(local_arr)
    embedded = np.zeros((n_cb, d), dtype=np.int64)
    clique_arr = np.array(clique)  # (s,)
    # Vectorized scatter: embedded[:, clique[j]] = local_arr[:, j] for all j
    embedded[:, clique_arr] = local_arr
    return embedded


# =====================================================================
# Sparse moment PSD constraints
# =====================================================================

def _add_sparse_moment_constraints(mdl, y, P, cliques, verbose=True):
    """Replace the full moment PSD cone with clique-restricted PSD cones.

    Each clique PSD is a NECESSARY condition for M_k(y) >> 0.
    Together they form a valid relaxation: lb_sparse <= lb_full <= val(d).
    Exact when RIP holds (Waki et al. 2006).
    """
    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    order = P['order']
    d = P['d']

    total_psd_size = 0
    n_added = 0

    for c_idx, clique in enumerate(cliques):
        cb_arr = _build_clique_basis(clique, order, d)
        n_cb = len(cb_arr)

        AB_hash = _hash_monos(
            cb_arr[:, np.newaxis, :] + cb_arr[np.newaxis, :, :], bases, prime)
        pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()

        if np.any(pick < 0):
            if verbose:
                print(f"    WARNING: clique {c_idx} has missing moments — skipping")
            continue

        M_c = Expr.reshape(y.pick(pick.tolist()), n_cb, n_cb)
        mdl.constraint(f"sparse_moment_{c_idx}", M_c, Domain.inPSDCone(n_cb))
        total_psd_size += n_cb
        n_added += 1

    if verbose:
        print(f"  Sparse moment PSD: {n_added} cliques, "
              f"total PSD dim {total_psd_size}", flush=True)


def _add_sparse_localizing_constraints(mdl, y, P, cliques, verbose=True):
    """Add clique-restricted mu_i >= 0 localizing PSD constraints."""
    if P['order'] < 2:
        return

    d = P['d']
    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    order = P['order']

    bin_to_clique = {}
    for c_idx, clique in enumerate(cliques):
        mid = (clique[0] + clique[-1]) / 2.0
        for i in clique:
            if i not in bin_to_clique:
                bin_to_clique[i] = (c_idx, abs(i - mid))
            else:
                _, prev_dist = bin_to_clique[i]
                if abs(i - mid) < prev_dist:
                    bin_to_clique[i] = (c_idx, abs(i - mid))
    bin_to_clique = {i: v[0] for i, v in bin_to_clique.items()}

    n_added = 0
    for i_var in range(d):
        c_idx = bin_to_clique.get(i_var, 0)
        clique = cliques[c_idx]

        cb_arr = _build_clique_basis(clique, order - 1, d)
        n_cb = len(cb_arr)

        e_i = np.zeros(d, dtype=np.int64)
        e_i[i_var] = 1
        AB_ei_hash = _hash_monos(
            cb_arr[:, np.newaxis, :] + cb_arr[np.newaxis, :, :] +
            e_i[np.newaxis, np.newaxis, :], bases, prime)
        pick = _hash_lookup(AB_ei_hash, sorted_h, sort_o).ravel()

        if np.any(pick < 0):
            continue

        Li = Expr.reshape(y.pick(pick.tolist()), n_cb, n_cb)
        mdl.constraint(f"sparse_loc_mu_{i_var}", Li, Domain.inPSDCone(n_cb))
        n_added += 1

    if verbose:
        clique_nloc = len(_build_clique_basis(cliques[0], order - 1, d))
        print(f"  Sparse localizing: {n_added}/{d} bins, "
              f"PSD size {clique_nloc} (was {P['n_loc']})", flush=True)


# =====================================================================
# Sparse window PSD constraints
# =====================================================================

def _add_sparse_window_psd(mdl, y, t_param, w, P, cliques):
    """Add a clique-restricted PSD window localizing constraint.

    If no single clique covers all active bins, falls back to the full
    localizing basis via _add_psd_window.
    """
    d = P['d']
    order = P['order']
    if order < 2:
        return
    n_y = P['n_y']
    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']

    ell, s_lo = P['windows'][w]
    Mw = P['M_mats'][w]

    nz_i, nz_j = np.nonzero(Mw)
    if len(nz_i) == 0:
        return
    active_bins = sorted(set(nz_i.tolist()) | set(nz_j.tolist()))

    covering_clique = None
    for clique in cliques:
        clique_set = set(clique)
        if all(b in clique_set for b in active_bins):
            covering_clique = clique
            break

    if covering_clique is None:
        _add_psd_window(mdl, y, t_param, w, P)
        return

    cb_arr = _build_clique_basis(covering_clique, order - 1, d)
    n_cb = len(cb_arr)

    AB_hash = _hash_monos(
        cb_arr[:, np.newaxis, :] + cb_arr[np.newaxis, :, :], bases, prime)
    t_pick_local = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()
    if np.any(t_pick_local < 0):
        _add_psd_window(mdl, y, t_param, w, P)
        return

    t_y = Expr.mul(t_param, y.pick(t_pick_local.tolist()))

    EE_hash = _hash_add(bases[:, np.newaxis], bases[np.newaxis, :], prime)
    ABIJ_hash = _hash_add(AB_hash[:, :, np.newaxis, np.newaxis],
                          EE_hash[np.newaxis, np.newaxis, :, :], prime)
    ab_eiej_local = _hash_lookup(ABIJ_hash, sorted_h, sort_o)

    y_idx = ab_eiej_local[:, :, nz_i, nz_j]
    valid = y_idx >= 0
    if not np.any(valid):
        Lw_mat = Expr.reshape(t_y, n_cb, n_cb)
        mdl.constraint(f"sw_psd_{w}", Lw_mat, Domain.inPSDCone(n_cb))
        return

    flat = (np.arange(n_cb)[:, np.newaxis] * n_cb +
            np.arange(n_cb)[np.newaxis, :])
    ab_exp = np.broadcast_to(flat[:, :, np.newaxis], y_idx.shape)
    mw_vals = Mw[nz_i, nz_j]
    mw_exp = np.broadcast_to(mw_vals[np.newaxis, np.newaxis, :], y_idx.shape)

    rows = ab_exp[valid].ravel().tolist()
    cols = y_idx[valid].ravel().tolist()
    vals = mw_exp[valid].ravel().tolist()

    Cw = Matrix.sparse(n_cb * n_cb, n_y, rows, cols, vals)
    cw_expr = Expr.mul(Cw, y)
    Lw_flat = Expr.sub(t_y, cw_expr)
    Lw_mat = Expr.reshape(Lw_flat, n_cb, n_cb)
    mdl.constraint(f"sw_psd_{w}", Lw_mat, Domain.inPSDCone(n_cb))


# =====================================================================
# Base constraints without PSD (consistency + y_0=1 only)
# =====================================================================

def _build_base_constraints_no_psd(mdl, y, P, add_upper_loc, verbose):
    """Add consistency and y_0=1 only (no moment/localizing PSD).

    Used when PSD constraints are added separately (sparse or DSOS mode).
    """
    d = P['d']
    n_y = P['n_y']
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

    if verbose:
        print(f"  Consistency constraints: {n_added}", flush=True)


# =====================================================================
# Batch violation checking
# =====================================================================

def _batch_check_violations(y_vals, t_val, P, active_windows, tol=1e-6):
    """Check ALL non-active windows for PSD localizing violations in batch."""
    n_loc = P['n_loc']
    ab_eiej_idx = P['ab_eiej_idx']
    M_mats = P['M_mats']
    n_win = P['n_win']

    if n_loc == 0 or ab_eiej_idx is None:
        return []

    t_pick_arr = np.array(P['t_pick'])
    L_t = t_val * y_vals[t_pick_arr].reshape(n_loc, n_loc)

    safe_idx = np.clip(ab_eiej_idx, 0, len(y_vals) - 1)
    y_abij = y_vals[safe_idx]
    y_abij[ab_eiej_idx < 0] = 0.0

    check_indices = []
    M_stack_list = []
    for w in range(n_win):
        if w in active_windows:
            continue
        Mw = M_mats[w]
        if np.all(Mw == 0):
            continue
        check_indices.append(w)
        M_stack_list.append(Mw)

    if not check_indices:
        return []

    M_stack = np.stack(M_stack_list)
    L_q_all = np.einsum('abij,wij->wab', y_abij, M_stack)
    L_all = L_t[np.newaxis, :, :] - L_q_all
    L_all = 0.5 * (L_all + np.swapaxes(L_all, -2, -1))

    all_eigs = np.linalg.eigvalsh(L_all)
    min_eigs = all_eigs[:, 0]

    violated_mask = min_eigs < -tol
    violations = [
        (check_indices[i], float(min_eigs[i]))
        for i in np.where(violated_mask)[0]
    ]
    violations.sort(key=lambda x: x[1])
    return violations
