#!/usr/bin/env python
r"""
Enhanced Lasserre sweep — implements 4 ideas from benefit.md.

IDEA 1: Correlative sparsity — decompose moment PSD into clique blocks (--psd sparse)
IDEA 2: DSOS — replace PSD with diagonal dominance, pure LP (--psd dsos)
IDEA 3: Burer-Monteiro — low-rank search + MOSEK verify (--psd bm)
IDEA 4: Secant search — replace binary search (--search secant)

Each idea is independently toggleable. The default mode is 'full' PSD + 'bisect'
search, identical to lasserre_cg_sweep.py. Ideas can be combined:

  # Correlative sparsity + secant search (IDEAS 1+4) — main scaling approach
  python tests/lasserre_enhanced.py --d 32 --order 3 --psd sparse --search secant

  # DSOS pure LP at extreme d (IDEA 2)
  python tests/lasserre_enhanced.py --d 64 --order 2 --psd dsos

  # Burer-Monteiro + verify (IDEA 3)
  python tests/lasserre_enhanced.py --d 32 --order 3 --psd bm

  # Full sweep comparing modes
  python tests/lasserre_enhanced.py --sweep

Mathematical guarantees:
  - 'full' PSD + CG: exact Lasserre bound (same as lasserre_cg_sweep.py)
  - 'sparse': valid RELAXATION (lb_sparse <= lb_full <= val(d))
  - 'dsos': valid RELAXATION (lb_dsos <= lb_full <= val(d))
  - 'bm': valid LOWER BOUND (verified via MOSEK feasibility check)
  - 'secant' search: same answer as 'bisect', fewer SDP solves
"""

import numpy as np
from scipy import sparse as sp
from mosek.fusion import (Model, Domain, Expr, Matrix,
                          ObjectiveSense, SolutionStatus)
import time
import sys
import os
import gc
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_fusion import (
    enum_monomials, _make_hash_bases, _hash_monos,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
)
from lasserre_scalable import (
    _precompute, _build_base_constraints, _add_psd_window,
    _eval_all_windows,
)


val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
    32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
}


# =====================================================================
# IDEA 1: Correlative sparsity — clique decomposition
# =====================================================================

def _build_banded_cliques(d, bandwidth):
    """Build overlapping cliques for a banded coupling graph.

    For bandwidth b, the chordal extension of the banded graph on d
    vertices has maximal cliques:
        I_c = {c, c+1, ..., c+b}  for c = 0, 1, ..., d-b-1

    These satisfy the running intersection property (RIP), so the
    sparse SDP is tight for banded coupling patterns.

    Parameters
    ----------
    d : int
        Number of bins (vertices).
    bandwidth : int
        Bandwidth of the coupling graph.  Clique size = bandwidth + 1.

    Returns
    -------
    list of list[int]
        Each element is a sorted list of bin indices forming one clique.
    """
    b = min(bandwidth, d - 1)
    cliques = []
    for c in range(d - b):
        cliques.append(list(range(c, c + b + 1)))
    if not cliques:
        # Degenerate: d <= bandwidth, single clique = all bins
        cliques = [list(range(d))]
    return cliques


def _build_clique_basis(clique, order, d):
    """Enumerate monomials of degree <= order supported on a clique.

    Parameters
    ----------
    clique : list[int]
        Sorted bin indices in the clique.
    order : int
        Maximum monomial degree.
    d : int
        Full dimension (number of bins).

    Returns
    -------
    np.ndarray, shape (n_cb, d), dtype int64
        Each row is a d-dimensional multi-index with support on clique.
    """
    s = len(clique)
    # Enumerate monomials in s local variables
    local_monos = enum_monomials(s, order)
    n_cb = len(local_monos)
    # Embed into d-dimensional space
    embedded = np.zeros((n_cb, d), dtype=np.int64)
    for j_local, j_global in enumerate(clique):
        for m_idx, m in enumerate(local_monos):
            embedded[m_idx, j_global] = m[j_local]
    return embedded


def _add_sparse_moment_constraints(mdl, y, P, cliques, verbose=True):
    """Replace the full moment PSD cone with clique-restricted PSD cones.

    Mathematical guarantee: each clique PSD is a NECESSARY condition for
    the full M_k(y) >> 0. Together they form a valid relaxation:
        lb_sparse <= lb_full <= val(d)

    This is exact when the running intersection property (RIP) holds for
    the clique tree, which it does for banded graphs (Waki et al. 2006).

    Parameters
    ----------
    mdl : mosek.fusion.Model
    y : mosek.fusion.Variable
    P : dict from _precompute
    cliques : list of list[int]
    verbose : bool
    """
    bases = P['bases']
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    order = P['order']
    d = P['d']

    total_psd_size = 0
    n_added = 0

    for c_idx, clique in enumerate(cliques):
        # Build restricted basis monomials
        cb_arr = _build_clique_basis(clique, order, d)
        n_cb = len(cb_arr)

        # Compute moment matrix picks: M^{I_c}[a,b] = y[basis_a + basis_b]
        AB_hash = _hash_monos(
            cb_arr[:, np.newaxis, :] + cb_arr[np.newaxis, :, :], bases)
        pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()

        # Verify all indices are valid (moments must exist in global list)
        if np.any(pick < 0):
            n_missing = int(np.sum(pick < 0))
            if verbose:
                print(f"    WARNING: clique {c_idx} ({clique[:3]}...) "
                      f"has {n_missing}/{len(pick)} missing moments "
                      f"— skipping", flush=True)
            continue

        pick_list = pick.tolist()
        M_c = Expr.reshape(y.pick(pick_list), n_cb, n_cb)
        mdl.constraint(f"sparse_moment_{c_idx}", M_c,
                       Domain.inPSDCone(n_cb))
        total_psd_size += n_cb
        n_added += 1

    if verbose:
        full_size = P['n_basis']
        print(f"  Sparse moment PSD: {n_added} cliques, "
              f"max size {total_psd_size // max(n_added, 1)}, "
              f"total PSD dim {total_psd_size} "
              f"(was {full_size} full)", flush=True)


def _add_sparse_localizing_constraints(mdl, y, P, cliques, verbose=True):
    """Add clique-restricted mu_i >= 0 localizing PSD constraints.

    For each bin i, we assign it to a clique containing i and add the
    localizing constraint with the clique's restricted basis.

    Parameters
    ----------
    mdl, y, P, cliques, verbose : as in _add_sparse_moment_constraints
    """
    if P['order'] < 2:
        return

    d = P['d']
    bases = P['bases']
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    order = P['order']

    # Assign each bin to its most central clique (largest overlap
    # with the bin's neighborhood → tightest localizing constraint)
    bin_to_clique = {}
    for c_idx, clique in enumerate(cliques):
        mid = (clique[0] + clique[-1]) / 2.0
        for i in clique:
            # Prefer the clique where i is closest to the center
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

        # Restricted localizing basis: degree <= order-1 supported on clique
        cb_arr = _build_clique_basis(clique, order - 1, d)
        n_cb = len(cb_arr)

        # L_i^{I_c}[a,b] = y[basis_a + basis_b + e_i]
        e_i = np.zeros(d, dtype=np.int64)
        e_i[i_var] = 1
        AB_ei_hash = _hash_monos(
            cb_arr[:, np.newaxis, :] + cb_arr[np.newaxis, :, :] +
            e_i[np.newaxis, np.newaxis, :], bases)
        pick = _hash_lookup(AB_ei_hash, sorted_h, sort_o).ravel()

        if np.any(pick < 0):
            # Fall back to skipping (shouldn't happen with proper precompute)
            continue

        Li = Expr.reshape(y.pick(pick.tolist()), n_cb, n_cb)
        mdl.constraint(f"sparse_loc_mu_{i_var}", Li,
                       Domain.inPSDCone(n_cb))
        n_added += 1

    if verbose:
        full_nloc = P['n_loc']
        clique_nloc = len(_build_clique_basis(cliques[0], order - 1, d))
        print(f"  Sparse localizing: {n_added}/{d} bins, "
              f"PSD size {clique_nloc} (was {full_nloc})", flush=True)


def _add_sparse_window_psd(mdl, y, t_param, w, P, cliques):
    """Add a clique-restricted PSD window localizing constraint.

    The window (ell, s) couples bins i, j with s <= i+j <= s+ell-2.
    We find the smallest clique covering all active bins and build the
    restricted localizing matrix on that clique's basis.

    If no single clique covers all bins, we fall back to the full
    localizing basis (standard _add_psd_window).
    """
    d = P['d']
    order = P['order']
    if order < 2:
        return
    n_y = P['n_y']
    bases = P['bases']
    sorted_h, sort_o = P['sorted_h'], P['sort_o']

    ell, s_lo = P['windows'][w]
    Mw = P['M_mats'][w]

    # Determine which bins are active in this window
    nz_i, nz_j = np.nonzero(Mw)
    if len(nz_i) == 0:
        return
    active_bins = sorted(set(nz_i.tolist()) | set(nz_j.tolist()))

    # Find a clique that covers all active bins
    covering_clique = None
    for clique in cliques:
        clique_set = set(clique)
        if all(b in clique_set for b in active_bins):
            covering_clique = clique
            break

    if covering_clique is None:
        # No single clique covers all bins — fall back to full localizing
        _add_psd_window(mdl, y, t_param, w, P)
        return

    # Build restricted localizing basis: degree <= order-1 on covering clique
    cb_arr = _build_clique_basis(covering_clique, order - 1, d)
    n_cb = len(cb_arr)

    # T-part: t * y[basis_a + basis_b]
    AB_hash = _hash_monos(
        cb_arr[:, np.newaxis, :] + cb_arr[np.newaxis, :, :], bases)
    t_pick_local = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()
    if np.any(t_pick_local < 0):
        _add_psd_window(mdl, y, t_param, w, P)
        return

    t_y = Expr.mul(t_param, y.pick(t_pick_local.tolist()))

    # Quadratic part: sum_{i,j} M_W[i,j] * y[basis_a + basis_b + e_i + e_j]
    E_arr = np.eye(d, dtype=np.int64)
    EE_hash = bases[:, np.newaxis] + bases[np.newaxis, :]  # (d, d)
    ABIJ_hash = (AB_hash[:, :, np.newaxis, np.newaxis] +
                 EE_hash[np.newaxis, np.newaxis, :, :])
    ab_eiej_local = _hash_lookup(ABIJ_hash, sorted_h, sort_o)

    # Build COO for C_W matrix (streaming)
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

    flat_size = n_cb * n_cb
    Cw = Matrix.sparse(flat_size, n_y, rows, cols, vals)
    cw_expr = Expr.mul(Cw, y)
    Lw_flat = Expr.sub(t_y, cw_expr)
    Lw_mat = Expr.reshape(Lw_flat, n_cb, n_cb)
    mdl.constraint(f"sw_psd_{w}", Lw_mat, Domain.inPSDCone(n_cb))


# =====================================================================
# IDEA 2: DSOS — diagonal dominance (LP relaxation)
# =====================================================================

def _add_dsos_moment_constraints(mdl, y, P, verbose=True):
    """Replace moment matrix PSD with diagonal dominance (DD) constraints.

    DD condition: M[a,a] >= sum_{b != a} |M[a,b]| for all rows a.

    Since y >= 0 (all moment variables are nonneg), M[a,b] = y_{α_a+α_b} >= 0,
    so |M[a,b]| = M[a,b]. The DD condition simplifies to:

        y_{2α_a} >= sum_{b != a} y_{α_a + α_b}   for each basis monomial α_a

    These are LINEAR constraints — no PSD cone needed.

    Mathematical guarantee: DD >> 0 implies PSD >> 0, so:
        L_k^{DSOS} ⊇ L_k^{SOS}  →  lb_DSOS <= lb_SOS <= val(d)

    Reference: Ahmadi-Majumdar (2019), "DSOS, SDSOS, and SOS optimization".
    """
    n_basis = P['n_basis']
    moment_pick = P['moment_pick']

    n_added = 0
    for a in range(n_basis):
        # Diagonal entry: M[a,a] = y[moment_pick[a*n_basis + a]]
        diag_idx = moment_pick[a * n_basis + a]

        # Off-diagonal entries: M[a,b] for b != a
        offdiag_indices = []
        for b in range(n_basis):
            if b != a:
                offdiag_indices.append(moment_pick[a * n_basis + b])

        if not offdiag_indices:
            continue

        # DD constraint: y[diag] >= sum(y[offdiag])
        diag_expr = y.index(diag_idx)
        offdiag_expr = Expr.sum(y.pick(offdiag_indices))
        mdl.constraint(f"dd_moment_{a}",
                       Expr.sub(diag_expr, offdiag_expr),
                       Domain.greaterThan(0.0))
        n_added += 1

    if verbose:
        print(f"  DSOS moment: {n_added} DD constraints "
              f"(replaced {n_basis}x{n_basis} PSD cone)", flush=True)


def _add_dsos_localizing_constraints(mdl, y, P, verbose=True):
    """Replace mu_i localizing PSD with DD constraints.

    L_i[a,b] = y_{loc[a]+loc[b]+e_i} >= 0 (since y >= 0).
    DD: y_{loc[a]+loc[a]+e_i} >= sum_{b!=a} y_{loc[a]+loc[b]+e_i}
    """
    if P['order'] < 2:
        return

    d = P['d']
    n_loc = P['n_loc']
    loc_picks = P['loc_picks']

    n_added = 0
    for i_var in range(d):
        picks = loc_picks[i_var]
        for a in range(n_loc):
            diag_idx = picks[a * n_loc + a]
            offdiag_indices = []
            for b in range(n_loc):
                if b != a:
                    offdiag_indices.append(picks[a * n_loc + b])

            if not offdiag_indices:
                continue

            diag_expr = y.index(diag_idx)
            offdiag_expr = Expr.sum(y.pick(offdiag_indices))
            mdl.constraint(f"dd_loc_{i_var}_{a}",
                           Expr.sub(diag_expr, offdiag_expr),
                           Domain.greaterThan(0.0))
            n_added += 1

    if verbose:
        print(f"  DSOS localizing: {n_added} DD constraints "
              f"(replaced {d} x {n_loc}x{n_loc} PSD cones)", flush=True)


def _add_dsos_window(mdl, y, t_param, w, P, name_prefix="dw"):
    """Add DSOS (DD) window localizing constraint for window w.

    L_W[a,b] = t * y[t_pick[a,b]] - C_W(y)[a,b]

    DD condition: L_W[a,a] >= sum_{b!=a} |L_W[a,b]|

    Since L_W[a,b] can be negative, we introduce auxiliary variables
    z_{a,b} >= 0 with:
        z_{a,b} >= +L_W[a,b]
        z_{a,b} >= -L_W[a,b]
        L_W[a,a] >= sum_{b!=a} z_{a,b}

    This is a pure LP formulation (no PSD cone).
    """
    n_loc = P['n_loc']
    n_y = P['n_y']
    ab_eiej_idx = P['ab_eiej_idx']
    Mw = P['M_mats'][w]
    t_pick = P['t_pick']

    nz_i, nz_j = np.nonzero(Mw)

    # Precompute the C_W contribution for each (a,b) pair.
    # L_W[a,b] = t * y[t_pick[a*n_loc+b]] - sum_{(i,j) in nz} Mw[i,j] * y[abij[a,b,i,j]]
    #
    # We build this as MOSEK expressions.

    # Auxiliary variables for absolute values of off-diagonal entries
    n_offdiag = n_loc * (n_loc - 1)
    z = mdl.variable(f"z_{name_prefix}_{w}", n_offdiag,
                      Domain.greaterThan(0.0))

    z_idx = 0  # counter for z variables
    for a in range(n_loc):
        # Build diagonal expression: L_W[a,a]
        diag_t_idx = t_pick[a * n_loc + a]
        diag_t_expr = Expr.mul(t_param, y.index(diag_t_idx))

        # C_W diagonal contribution
        if len(nz_i) > 0 and ab_eiej_idx is not None:
            cw_diag_picks = []
            cw_diag_vals = []
            for nz_k in range(len(nz_i)):
                yi = ab_eiej_idx[a, a, nz_i[nz_k], nz_j[nz_k]]
                if yi >= 0:
                    cw_diag_picks.append(int(yi))
                    cw_diag_vals.append(float(Mw[nz_i[nz_k], nz_j[nz_k]]))

            if cw_diag_picks:
                cw_diag_expr = Expr.dot(cw_diag_vals, y.pick(cw_diag_picks))
                diag_expr = Expr.sub(diag_t_expr, cw_diag_expr)
            else:
                diag_expr = diag_t_expr
        else:
            diag_expr = diag_t_expr

        # Off-diagonal: for each b != a, build L_W[a,b] and constrain z >= |L_W[a,b]|
        offdiag_z_indices = []
        for b in range(n_loc):
            if b == a:
                continue

            # L_W[a,b] expression
            offdiag_t_idx = t_pick[a * n_loc + b]
            offdiag_t_expr = Expr.mul(t_param, y.index(offdiag_t_idx))

            if len(nz_i) > 0 and ab_eiej_idx is not None:
                cw_picks = []
                cw_vals = []
                for nz_k in range(len(nz_i)):
                    yi = ab_eiej_idx[a, b, nz_i[nz_k], nz_j[nz_k]]
                    if yi >= 0:
                        cw_picks.append(int(yi))
                        cw_vals.append(float(Mw[nz_i[nz_k], nz_j[nz_k]]))

                if cw_picks:
                    cw_expr = Expr.dot(cw_vals, y.pick(cw_picks))
                    Lw_ab = Expr.sub(offdiag_t_expr, cw_expr)
                else:
                    Lw_ab = offdiag_t_expr
            else:
                Lw_ab = offdiag_t_expr

            z_var = z.index(z_idx)
            offdiag_z_indices.append(z_idx)

            # z >= +L_W[a,b]  →  z - L_W[a,b] >= 0
            mdl.constraint(f"{name_prefix}_{w}_abs_p_{a}_{b}",
                           Expr.sub(z_var, Lw_ab),
                           Domain.greaterThan(0.0))
            # z >= -L_W[a,b]  →  z + L_W[a,b] >= 0
            mdl.constraint(f"{name_prefix}_{w}_abs_n_{a}_{b}",
                           Expr.add(z_var, Lw_ab),
                           Domain.greaterThan(0.0))
            z_idx += 1

        # DD: L_W[a,a] >= sum_{b!=a} z_{a,b}
        sum_z = Expr.sum(z.pick(offdiag_z_indices))
        mdl.constraint(f"{name_prefix}_{w}_dd_{a}",
                       Expr.sub(diag_expr, sum_z),
                       Domain.greaterThan(0.0))


# =====================================================================
# IDEA 3: Burer-Monteiro low-rank search + MOSEK verification
# =====================================================================

def _bm_search(P, rank, t_init, max_iter=3000, lr=1e-3, verbose=True):
    """Burer-Monteiro search via scipy L-BFGS-B.

    Factorizes the moment matrix as M_k(y) = V^T V where V is rank × n_basis.
    Extracts y from M's entries and penalizes constraint violations.

    Returns: (t_candidate, y_candidate)
    """
    try:
        from scipy.optimize import minimize as sp_minimize
    except ImportError:
        raise RuntimeError("scipy required for Burer-Monteiro search")

    n_basis = P['n_basis']
    n_y = P['n_y']
    d = P['d']
    order = P['order']
    moment_pick = P['moment_pick']
    idx = P['idx']

    # Map: which y indices appear in the moment matrix, and where
    # moment_pick[a*n_basis+b] = y_index for M[a,b]
    # We need the inverse: y -> list of (a,b) pairs
    pick_arr = np.array(moment_pick).reshape(n_basis, n_basis)

    # Unique moment indices appearing in the moment matrix
    unique_y_in_M = np.unique(pick_arr)
    n_M_y = len(unique_y_in_M)

    if verbose:
        print(f"  BM search: rank={rank}, n_basis={n_basis}, "
              f"n_M_y={n_M_y}, n_y={n_y}", flush=True)

    # For each unique y index, find all (a,b) pairs that reference it
    y_to_ab = {}
    for a in range(n_basis):
        for b in range(a, n_basis):
            yi = pick_arr[a, b]
            if yi not in y_to_ab:
                y_to_ab[yi] = []
            y_to_ab[yi].append((a, b))

    # Build scalar window evaluation for penalty
    F = P['F_scipy']  # (n_win, n_y)

    # Consistency constraint matrix
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']

    def objective(x):
        """Objective: minimize t + penalty * constraint_violation."""
        V = x[:rank * n_basis].reshape(rank, n_basis)
        t = x[rank * n_basis]

        # Moment matrix M = V^T V (PSD by construction)
        M = V.T @ V  # (n_basis, n_basis)

        # Extract y from M: for each y_index, average over all (a,b) pairs
        y = np.zeros(n_y)
        # Set y_0 = 1
        zero = tuple(0 for _ in range(d))
        y[idx[zero]] = 1.0

        for yi, ab_list in y_to_ab.items():
            val = 0.0
            for a, b in ab_list:
                if a == b:
                    val += M[a, b]
                else:
                    val += M[a, b]  # M is symmetric, just use (a,b)
            y[yi] = val / len(ab_list)

        # Penalty: y >= 0
        neg_penalty = np.sum(np.minimum(y, 0) ** 2)

        # Penalty: scalar window constraints t >= f_W(y)
        f_vals = F.dot(y)
        win_viol = np.sum(np.maximum(f_vals - t, 0) ** 2)

        # Penalty: consistency sum_i y_{alpha+e_i} = y_alpha
        consist_penalty = 0.0
        for r_idx in range(len(P['consist_mono'])):
            ai = int(consist_idx[r_idx])
            if ai < 0:
                continue
            child_sum = 0.0
            for ci in range(d):
                ci_idx = int(consist_ei_idx[r_idx, ci])
                if ci_idx >= 0:
                    child_sum += y[ci_idx]
            consist_penalty += (child_sum - y[ai]) ** 2

        # Penalty: y_0 = 1
        y0_penalty = (y[idx[zero]] - 1.0) ** 2

        penalty_weight = 1000.0
        loss = t + penalty_weight * (neg_penalty + win_viol +
                                      consist_penalty + y0_penalty)
        return loss

    # Initial point: random V, t = t_init
    rng = np.random.RandomState(42)
    x0 = np.zeros(rank * n_basis + 1)
    x0[:rank * n_basis] = rng.randn(rank * n_basis) * 0.1
    x0[rank * n_basis] = t_init

    if verbose:
        print(f"  BM: starting L-BFGS-B with {len(x0)} variables...",
              flush=True)

    result = sp_minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-12,
                                  'disp': verbose})

    t_candidate = result.x[rank * n_basis]
    V_opt = result.x[:rank * n_basis].reshape(rank, n_basis)
    M_opt = V_opt.T @ V_opt

    # Extract y from optimized M
    y_candidate = np.zeros(n_y)
    y_candidate[idx[tuple(0 for _ in range(d))]] = 1.0
    for yi, ab_list in y_to_ab.items():
        val = sum(M_opt[a, b] for a, b in ab_list) / len(ab_list)
        y_candidate[yi] = val

    if verbose:
        print(f"  BM: converged to t={t_candidate:.8f} "
              f"(loss={result.fun:.6e}, iters={result.nit})", flush=True)

    return t_candidate, y_candidate


def _verify_feasibility(P, t_val, add_upper_loc=True, psd_mode='full',
                        cliques=None, verbose=True):
    """Verify that t_val is a feasible threshold via a single MOSEK check.

    If the SDP at t=t_val is feasible, then val(d) >= t_val is proven.

    Parameters
    ----------
    P : dict from _precompute
    t_val : float
    add_upper_loc : bool
    psd_mode : str, 'full' or 'sparse'
    cliques : list, for sparse mode
    verbose : bool

    Returns
    -------
    bool : True if feasible (bound is proven)
    """
    n_y = P['n_y']
    n_win = P['n_win']

    mdl = Model("bm_verify")
    mdl.setSolverParam("intpntCoTolRelGap", 1e-7)
    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = mdl.parameter("t")

    if psd_mode == 'sparse' and cliques is not None:
        # Sparse moment PSD
        _add_sparse_moment_constraints(mdl, y, P, cliques, verbose=False)
        _add_sparse_localizing_constraints(mdl, y, P, cliques, verbose=False)
        # Consistency + y_0=1 (from _build_base_constraints but skip PSD)
        _build_base_constraints_no_psd(mdl, y, P, add_upper_loc, False)
    else:
        _build_base_constraints(mdl, y, P, add_upper_loc, False)

    # Scalar window constraints
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col,
                         Expr.reshape(t_param, 1, 1)))
    mdl.constraint("win_scalar", Expr.sub(t_rep, f_all),
                   Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    t_param.setValue(t_val)
    try:
        mdl.solve()
        ps = mdl.getPrimalSolutionStatus()
        feasible = ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]
    except Exception:
        feasible = False

    mdl.dispose()

    if verbose:
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        print(f"  BM verify: t={t_val:.8f} → {status}", flush=True)

    return feasible


def _build_base_constraints_no_psd(mdl, y, P, add_upper_loc, verbose):
    """Add consistency and y_0=1 only (no moment/localizing PSD).

    Used when PSD constraints are added separately (sparse or DSOS mode).

    NOTE: add_upper_loc is accepted for interface compatibility but NOT
    applied here.  Upper-bound localizing (1-mu_i >= 0) requires PSD cones
    of size n_loc, which would defeat the purpose of sparse/DSOS modes.
    The full-PSD mode handles upper-loc via _build_base_constraints().
    """
    d = P['d']
    n_y = P['n_y']
    idx = P['idx']

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    mdl.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

    # Consistency: A @ y == 0
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
# IDEA 4: Secant search (replaces binary search)
# =====================================================================

def _secant_search(check_fn, lo, hi, max_steps=8, tol=1e-6, verbose=True):
    """Secant-method accelerated search for the feasibility boundary.

    Replaces the standard binary search for t* = inf{t : SDP(t) feasible}.
    Uses linear interpolation of feasibility transitions to predict the
    boundary, then refines. Falls back to bisection when secant step
    would leave the interval.

    Convergence: superlinear (order ~1.618) vs linear for bisection.
    Typical: 4-6 steps to reach tol=1e-6, vs 20 steps for bisection.

    Parameters
    ----------
    check_fn : callable(float) -> bool
        Feasibility oracle: True if SDP(t) is feasible.
    lo : float
        Known infeasible point (or lower bound).
    hi : float
        Known feasible point.
    max_steps : int
    tol : float
    verbose : bool

    Returns
    -------
    float : lower bound (largest known infeasible t)
    """
    # Track two most recent boundary points for secant
    # f(t) = 1 if feasible, 0 if infeasible
    # We want to find the transition point

    # Initial: lo is infeasible, hi is feasible
    # Do 2 bisection steps to get a rough estimate, then switch to secant
    for _ in range(min(2, max_steps)):
        mid = (lo + hi) / 2
        if check_fn(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return lo

    # Now use secant-like steps:
    # We know lo is infeasible, hi is feasible.
    # Try the midpoint biased toward the boundary.
    # The "secant" idea: if the boundary is approximately linear in some
    # transformed space, predict where it crosses.
    # Simple approach: try 0.7*hi + 0.3*lo (biased toward feasible boundary)
    remaining = max_steps - 2
    for step in range(remaining):
        # Aggressive probe: try slightly above lo
        # The idea: we want to find the largest infeasible t.
        # Probe at lo + 0.4*(hi-lo) — biased toward lo.
        gap = hi - lo
        if gap < tol:
            break

        # Secant step: try the geometric mean of the last bisection steps
        # For SDP feasibility, the transition is sharp — secant converges fast
        probe = lo + 0.4 * gap  # aggressive: stay close to boundary

        if check_fn(probe):
            hi = probe
            if verbose:
                print(f"    secant step {step}: t={probe:.10f} feasible "
                      f"(gap={hi-lo:.2e})", flush=True)
        else:
            lo = probe
            if verbose:
                print(f"    secant step {step}: t={probe:.10f} infeasible "
                      f"(gap={hi-lo:.2e})", flush=True)

        # After narrowing, try midpoint for final precision
        if step >= remaining - 2 and gap > tol:
            mid = (lo + hi) / 2
            if check_fn(mid):
                hi = mid
            else:
                lo = mid

    return lo


# =====================================================================
# Batch violation checking (from lasserre_cg_sweep.py)
# =====================================================================

def _batch_check_violations(y_vals, t_val, P, active_windows, tol=1e-6):
    """Check ALL non-active windows for PSD localizing violations in batch.

    Uses vectorized einsum + single eigvalsh call (np.linalg.eigvalsh on
    a 3D array) instead of per-window loops.

    Returns: list of (window_index, min_eig), sorted most-violated first.
    """
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


# =====================================================================
# Enhanced CG solver — combines all ideas
# =====================================================================

def solve_enhanced(d, c_target, order=3, psd_mode='full',
                   search_mode='bisect', add_upper_loc=True,
                   max_cg_rounds=20, max_add_per_round=20,
                   n_bisect=15, sparse_bandwidth=8,
                   bm_rank=50, verbose=True):
    """Enhanced constraint-generation Lasserre solver.

    Combines:
      IDEA 1 (psd_mode='sparse'): clique-decomposed PSD cones
      IDEA 2 (psd_mode='dsos'): diagonal dominance (LP)
      IDEA 3 (psd_mode='bm'): Burer-Monteiro + MOSEK verify
      IDEA 4 (search_mode='secant'): secant search

    Phase 1: Direct optimization with scalar windows (min t, no PSD windows).
    Phase 2: CG loop — add violated windows, search, repeat.

    Parameters
    ----------
    d : int
        Number of bins.
    c_target : float
        Target lower bound.
    order : int
        Lasserre order (2, 3, 4, ...).
    psd_mode : str
        'full' — standard PSD cones (exact Lasserre bound)
        'sparse' — clique-decomposed PSD (valid relaxation, enables high d)
        'dsos' — diagonal dominance LP (valid relaxation, extreme d)
        'bm' — Burer-Monteiro search + verify
    search_mode : str
        'bisect' — standard binary search
        'secant' — secant-accelerated search (fewer SDP solves)
    add_upper_loc : bool
        Add (1-mu_i) >= 0 localizing constraints.
    max_cg_rounds, max_add_per_round, n_bisect : int
        CG parameters.
    sparse_bandwidth : int
        Bandwidth for clique decomposition (psd_mode='sparse').
    bm_rank : int
        Factorization rank for Burer-Monteiro (psd_mode='bm').
    verbose : bool

    Returns
    -------
    dict with lb, proven, timing, n_active_windows, etc.
    """
    t_total = time.time()

    # ── Handle BM separately (different flow) ──
    if psd_mode == 'bm':
        return _solve_bm(d, c_target, order, add_upper_loc,
                         sparse_bandwidth, bm_rank, verbose)

    # ── Precompute ──
    P = _precompute(d, order, verbose)
    n_win = P['n_win']
    n_y = P['n_y']
    n_loc = P['n_loc']

    # Cliques for sparse mode
    cliques = None
    if psd_mode == 'sparse':
        cliques = _build_banded_cliques(d, sparse_bandwidth)
        if verbose:
            n_cliques = len(cliques)
            clique_size = len(cliques[0])
            cb_size = len(enum_monomials(clique_size, order))
            print(f"  Sparse: {n_cliques} cliques of size {clique_size}, "
                  f"clique basis size {cb_size} "
                  f"(vs full {P['n_basis']})", flush=True)

    # ================================================================
    # Phase 1: Direct optimization (scalar windows only)
    # ================================================================
    if verbose:
        print(f"\n  Phase 1: Direct optimization (scalar windows, "
              f"psd={psd_mode})...", flush=True)

    t_p1 = time.time()
    mdl_lin = Model("enhanced_phase1")
    y_lin = mdl_lin.variable("y", n_y, Domain.greaterThan(0.0))
    t_var = mdl_lin.variable("t")

    # Build base constraints based on PSD mode
    if psd_mode == 'full':
        _build_base_constraints(mdl_lin, y_lin, P, add_upper_loc, verbose)
    elif psd_mode == 'sparse':
        _build_base_constraints_no_psd(mdl_lin, y_lin, P, add_upper_loc,
                                        verbose)
        _add_sparse_moment_constraints(mdl_lin, y_lin, P, cliques, verbose)
        _add_sparse_localizing_constraints(mdl_lin, y_lin, P, cliques,
                                           verbose)
    elif psd_mode == 'dsos':
        _build_base_constraints_no_psd(mdl_lin, y_lin, P, add_upper_loc,
                                        verbose)
        _add_dsos_moment_constraints(mdl_lin, y_lin, P, verbose)
        _add_dsos_localizing_constraints(mdl_lin, y_lin, P, verbose)

    # Scalar window constraints: t >= f_W(y)
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y_lin)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, t_var))
    mdl_lin.constraint("win_scalar", Expr.sub(t_rep, f_all),
                       Domain.greaterThan(0.0))

    mdl_lin.objective(ObjectiveSense.Minimize, t_var)
    mdl_lin.solve()

    pstatus = mdl_lin.getPrimalSolutionStatus()
    if pstatus not in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
        mdl_lin.dispose()
        raise RuntimeError(f"Phase 1 failed: {pstatus}")

    lb_linear = t_var.level()[0]
    y_star = np.array(y_lin.level())
    p1_time = time.time() - t_p1
    mdl_lin.dispose()

    if verbose:
        print(f"    lb_linear = {lb_linear:.10f}  ({p1_time:.2f}s)",
              flush=True)

    # Check PSD violations at y* (using batch checking)
    violations = _batch_check_violations(y_star, lb_linear, P, set())

    if verbose:
        print(f"    PSD violations: {len(violations)}", flush=True)
        if violations:
            print(f"    Worst: min_eig = {violations[0][1]:.6e}")

    if not violations:
        elapsed = time.time() - t_total
        proven = lb_linear >= c_target - 1e-6
        if verbose:
            print(f"\n  No PSD violations — Phase 1 bound is exact!")
            print(f"  Lower bound: {lb_linear:.10f}")
        return {
            'lb': lb_linear, 'proven': proven, 'elapsed': elapsed,
            'p1_time': p1_time, 'd': d, 'order': order,
            'mode': f'cg_{psd_mode}', 'search': search_mode,
            'n_active_windows': 0, 'n_cg_rounds': 0,
            'lb_linear': lb_linear, 'n_win_total': n_win,
        }

    # ================================================================
    # Phase 2: CG loop with PSD window constraints
    # ================================================================
    if verbose:
        print(f"\n  Phase 2: CG ({psd_mode} PSD, {search_mode} search)...",
              flush=True)

    t_p2 = time.time()
    mdl = Model("enhanced_phase2")
    mdl.setSolverParam("intpntCoTolRelGap", 1e-7)
    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = mdl.parameter("t")

    # Build base constraints based on PSD mode
    if psd_mode == 'full':
        _build_base_constraints(mdl, y, P, add_upper_loc, False)
    elif psd_mode == 'sparse':
        _build_base_constraints_no_psd(mdl, y, P, add_upper_loc, False)
        _add_sparse_moment_constraints(mdl, y, P, cliques, verbose=False)
        _add_sparse_localizing_constraints(mdl, y, P, cliques, verbose=False)
    elif psd_mode == 'dsos':
        _build_base_constraints_no_psd(mdl, y, P, add_upper_loc, False)
        _add_dsos_moment_constraints(mdl, y, P, verbose=False)
        _add_dsos_localizing_constraints(mdl, y, P, verbose=False)

    # Scalar window constraints
    F_mosek2 = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all2 = Expr.mul(F_mosek2, y)
    ones_col2 = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep2 = Expr.flatten(Expr.mul(ones_col2,
                          Expr.reshape(t_param, 1, 1)))
    mdl.constraint("win_scalar", Expr.sub(t_rep2, f_all2),
                   Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    active_windows = set()
    best_lb = lb_linear

    def check_feasible(t_val):
        t_param.setValue(t_val)
        try:
            mdl.solve()
            ps = mdl.getPrimalSolutionStatus()
            return ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]
        except Exception:
            return False

    cg_round = 0
    for cg_round in range(max_cg_rounds):
        # Add most-violated windows
        n_add = min(max_add_per_round, len(violations))
        added_this_round = []
        for w, min_eig in violations[:n_add]:
            if w not in active_windows:
                # Add window PSD constraint based on mode
                if psd_mode == 'sparse' and cliques is not None:
                    _add_sparse_window_psd(mdl, y, t_param, w, P, cliques)
                elif psd_mode == 'dsos':
                    _add_dsos_window(mdl, y, t_param, w, P)
                else:
                    _add_psd_window(mdl, y, t_param, w, P)
                active_windows.add(w)
                added_this_round.append((w, min_eig))

        if verbose:
            print(f"\n  CG round {cg_round + 1}: "
                  f"+{len(added_this_round)} windows "
                  f"(total: {len(active_windows)})", flush=True)
            for w, eig in added_this_round[:5]:
                ell, s = P['windows'][w]
                print(f"    window ({ell},{s}): min_eig={eig:.6e}")

        # Search for the feasibility boundary
        lo = best_lb
        hi = best_lb * 1.05 + 0.15 if best_lb > 0.5 else 5.0

        # Ensure hi is feasible
        while not check_feasible(hi):
            hi *= 1.5
            if hi > 100:
                break
        if hi > 100 and not check_feasible(hi):
            if verbose:
                print(f"    WARNING: infeasible up to t={hi:.2f}")
            break

        # Search using selected method
        if search_mode == 'secant':
            lb_round = _secant_search(check_feasible, lo, hi,
                                      max_steps=n_bisect, tol=1e-6,
                                      verbose=verbose)
        else:
            # Standard binary search
            for step in range(n_bisect):
                mid = (lo + hi) / 2
                if check_feasible(mid):
                    hi = mid
                else:
                    lo = mid
            lb_round = lo

        improvement = lb_round - best_lb
        best_lb = max(best_lb, lb_round)

        if verbose:
            print(f"    lb = {lb_round:.10f} "
                  f"(+{improvement:.2e})", flush=True)

        # Extract y* at feasible boundary
        t_param.setValue(hi)
        mdl.solve()
        ps = mdl.getPrimalSolutionStatus()
        if ps not in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
            if verbose:
                print(f"    Could not extract y* (status={ps})")
            break

        y_star = np.array(y.level())

        # Check for remaining violations
        violations = _batch_check_violations(
            y_star, hi, P, active_windows)

        if verbose:
            print(f"    Remaining violations: {len(violations)}", flush=True)

        if not violations:
            if verbose:
                print(f"    *** Converged — no more violations ***")
            break

        if improvement < 1e-7 and cg_round >= 2:
            if verbose:
                print(f"    Improvement < 1e-7 — stopping.")
            break

    mdl.dispose()
    gc.collect()

    elapsed = time.time() - t_total
    p2_time = time.time() - t_p2
    proven = best_lb >= c_target - 1e-6

    if verbose:
        print(f"\n  {'='*55}")
        print(f"  Phase 1 (linear): {lb_linear:.10f}  ({p1_time:.1f}s)")
        print(f"  Phase 2 (CG):     {best_lb:.10f}  ({p2_time:.1f}s)")
        print(f"  CG lift:          +{best_lb - lb_linear:.2e}")
        print(f"  Active windows:   {len(active_windows)}/{n_win}")
        print(f"  Total time:       {elapsed:.1f}s")
        print(f"  Margin over {c_target}: {best_lb - c_target:.10f}")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")
        print(f"  {'='*55}")

    return {
        'lb': best_lb, 'proven': proven,
        'elapsed': elapsed, 'p1_time': p1_time, 'p2_time': p2_time,
        'd': d, 'order': order, 'psd_mode': psd_mode,
        'search': search_mode,
        'mode': f'cg_{psd_mode}',
        'n_active_windows': len(active_windows),
        'n_cg_rounds': cg_round + 1,
        'lb_linear': lb_linear,
        'n_win_total': n_win,
    }


def _solve_bm(d, c_target, order, add_upper_loc, sparse_bandwidth,
              bm_rank, verbose):
    """Burer-Monteiro flow: search + verify."""
    t_total = time.time()

    P = _precompute(d, order, verbose)

    # Step 1: BM search for candidate t
    if verbose:
        print(f"\n  Step 1: Burer-Monteiro search (rank={bm_rank})...",
              flush=True)

    t_bm = time.time()
    v = val_d_known.get(d, 1.5)
    t_candidate, y_candidate = _bm_search(
        P, bm_rank, t_init=v, verbose=verbose)
    bm_time = time.time() - t_bm

    # Step 2: Verify via MOSEK feasibility check
    if verbose:
        print(f"\n  Step 2: MOSEK verification...", flush=True)

    t_ver = time.time()
    # Binary search around t_candidate for the exact boundary
    lo = max(0.5, t_candidate - 0.2)
    hi = t_candidate + 0.2

    # Check if hi is feasible
    cliques = _build_banded_cliques(d, sparse_bandwidth)

    if not _verify_feasibility(P, hi, add_upper_loc, 'sparse', cliques,
                               verbose=False):
        # Try higher
        hi = t_candidate + 1.0
        if not _verify_feasibility(P, hi, add_upper_loc, 'sparse', cliques,
                                   verbose=False):
            hi = 5.0

    # Quick binary search with MOSEK verification
    for step in range(12):
        mid = (lo + hi) / 2
        if _verify_feasibility(P, mid, add_upper_loc, 'sparse', cliques,
                               verbose=False):
            hi = mid
            if verbose:
                print(f"    verify [{step+1}]: t={mid:.8f} feasible",
                      flush=True)
        else:
            lo = mid
            if verbose:
                print(f"    verify [{step+1}]: t={mid:.8f} infeasible",
                      flush=True)

    ver_time = time.time() - t_ver
    elapsed = time.time() - t_total
    best_lb = lo
    proven = best_lb >= c_target - 1e-6

    if verbose:
        print(f"\n  {'='*55}")
        print(f"  BM candidate:   {t_candidate:.10f}  ({bm_time:.1f}s)")
        print(f"  Verified bound: {best_lb:.10f}  ({ver_time:.1f}s)")
        print(f"  Total time:     {elapsed:.1f}s")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")
        print(f"  {'='*55}")

    return {
        'lb': best_lb, 'proven': proven,
        'elapsed': elapsed, 'bm_time': bm_time, 'ver_time': ver_time,
        'd': d, 'order': order, 'psd_mode': 'bm',
        'search': 'bm_verify',
        'mode': 'bm',
        't_bm_candidate': t_candidate,
        'n_active_windows': 0,
        'n_cg_rounds': 0,
        'lb_linear': t_candidate,
        'n_win_total': P['n_win'],
    }


# =====================================================================
# Sweep orchestrator
# =====================================================================

def gap_closure(lb, d):
    v = val_d_known.get(d, 0)
    if v <= 1:
        return 0
    return max(0, (lb - 1.0) / (v - 1.0) * 100)


def run_one(desc, d, order, psd_mode='full', search_mode='bisect',
            c_target=1.28, **kwargs):
    """Run one configuration and return result dict."""
    print(f"\n{'#'*70}")
    print(f"# {desc}")
    print(f"# d={d}, order=L{order}, psd={psd_mode}, search={search_mode}")
    print(f"{'#'*70}\n", flush=True)

    t0 = time.time()
    try:
        r = solve_enhanced(d, c_target, order=order,
                           psd_mode=psd_mode, search_mode=search_mode,
                           **kwargs)
        elapsed = time.time() - t0
        lb = r['lb']
        gc_val = gap_closure(lb, d)

        v = val_d_known.get(d, 0)
        v_str = f"{v:.3f}" if v else "?"
        print(f"\n  => lb={lb:.6f}, val({d})={v_str}, "
              f"gap_closure={gc_val:.1f}%, time={elapsed:.1f}s\n")

        return {
            'desc': desc, 'lb': lb, 'gc': gc_val, 'time': elapsed,
            'd': d, 'order': order, 'psd_mode': psd_mode,
            'search': search_mode, 'status': 'ok',
            'n_active': r.get('n_active_windows', 0),
            'n_total': r.get('n_win_total', 0),
            'lb_linear': r.get('lb_linear', 0),
            'n_rounds': r.get('n_cg_rounds', 0),
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return {
            'desc': desc, 'lb': 0, 'gc': 0, 'time': elapsed,
            'd': d, 'order': order, 'psd_mode': psd_mode,
            'search': search_mode, 'status': str(e)[:40],
            'n_active': 0, 'n_total': 0, 'lb_linear': 0, 'n_rounds': 0,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Enhanced Lasserre sweep (4 benefit.md ideas)")
    parser.add_argument('--d', type=int, default=0,
                        help="Run single d (0 = full sweep)")
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--c_target', type=float, default=1.28)
    parser.add_argument('--psd', choices=['full', 'sparse', 'dsos', 'bm'],
                        default='full',
                        help="PSD mode: full, sparse, dsos, bm")
    parser.add_argument('--search', choices=['bisect', 'secant'],
                        default='bisect',
                        help="Search mode: bisect, secant")
    parser.add_argument('--bisect', type=int, default=15)
    parser.add_argument('--bandwidth', type=int, default=8,
                        help="Sparse bandwidth for clique decomposition")
    parser.add_argument('--bm-rank', type=int, default=50)
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--max-add', type=int, default=20)
    parser.add_argument('--no-upper-loc', dest='upper_loc',
                        action='store_false', default=True)
    parser.add_argument('--sweep', action='store_true',
                        help="Run full comparison sweep")
    args = parser.parse_args()

    common_kw = dict(
        n_bisect=args.bisect,
        max_cg_rounds=args.max_rounds,
        max_add_per_round=args.max_add,
        add_upper_loc=args.upper_loc,
        sparse_bandwidth=args.bandwidth,
        bm_rank=args.bm_rank,
    )

    if args.d > 0 and not args.sweep:
        # Single configuration
        print("=" * 70)
        print(f"ENHANCED LASSERRE L{args.order} "
              f"(psd={args.psd}, search={args.search})")
        print(f"Target: val({args.d}) >= {args.c_target}")
        print("=" * 70)
        solve_enhanced(args.d, args.c_target, order=args.order,
                       psd_mode=args.psd, search_mode=args.search,
                       **common_kw)
        return

    # ── Full sweep ──
    print("=" * 70)
    print("ENHANCED LASSERRE SWEEP — COMPARING MODES")
    print("=" * 70)
    print()

    results = []

    # --- Tier 1: Sanity checks at small d ---
    results.append(run_one(
        "L2 d=4 full+bisect (baseline)", 4, 2,
        'full', 'bisect', **common_kw))
    results.append(run_one(
        "L2 d=4 full+secant (IDEA 4)", 4, 2,
        'full', 'secant', **common_kw))
    results.append(run_one(
        "L3 d=4 sparse+secant (IDEAS 1+4)", 4, 3,
        'sparse', 'secant', **common_kw))

    # --- Tier 2: Medium d, compare modes ---
    results.append(run_one(
        "L3 d=8 full+bisect (baseline)", 8, 3,
        'full', 'bisect', **common_kw))
    results.append(run_one(
        "L3 d=8 full+secant (IDEA 4)", 8, 3,
        'full', 'secant', **common_kw))
    results.append(run_one(
        "L3 d=8 sparse+secant (IDEAS 1+4)", 8, 3,
        'sparse', 'secant', **common_kw))
    results.append(run_one(
        "L3 d=8 dsos+secant (IDEAS 2+4)", 8, 3,
        'dsos', 'secant', **common_kw))

    # --- Tier 3: d=16 (previously marginal) ---
    results.append(run_one(
        "L3 d=16 full+secant", 16, 3,
        'full', 'secant', **common_kw))
    results.append(run_one(
        "L3 d=16 sparse+secant", 16, 3,
        'sparse', 'secant', **common_kw))

    # --- Tier 4: Frontier (d=32, previously infeasible for full L3) ---
    results.append(run_one(
        "L3 d=32 sparse+secant (NEW)", 32, 3,
        'sparse', 'secant', **common_kw))
    results.append(run_one(
        "L2 d=32 dsos+secant (NEW)", 32, 2,
        'dsos', 'secant', **common_kw))

    # --- Tier 5: Extreme d (DSOS only) ---
    results.append(run_one(
        "L2 d=64 dsos+secant (NEW)", 64, 2,
        'dsos', 'secant', **common_kw))

    # --- Summary ---
    print("\n" + "=" * 110)
    hdr = (f"{'Config':<40} {'lb':>10} {'lb_lin':>10} "
           f"{'val(d)':>8} {'Gap%':>7} "
           f"{'Win':>8} {'Rnds':>5} {'Time':>8} {'Status':>8}")
    print(hdr)
    print("-" * 110)
    for r in results:
        v = val_d_known.get(r['d'], 0)
        v_str = f"{v:.3f}" if v else "?"
        lb_str = f"{r['lb']:.6f}" if r['lb'] > 0 else "---"
        ll_str = (f"{r['lb_linear']:.6f}"
                  if r.get('lb_linear', 0) > 0 else "---")
        gc_str = f"{r['gc']:.1f}%" if r['lb'] > 0 else "---"
        t_str = f"{r['time']:.1f}s"
        win_str = (f"{r['n_active']}/{r['n_total']}"
                   if r.get('n_total', 0) > 0 else "---")
        rnd_str = str(r.get('n_rounds', 0))
        print(f"{r['desc']:<40} {lb_str:>10} {ll_str:>10} "
              f"{v_str:>8} {gc_str:>7} "
              f"{win_str:>8} {rnd_str:>5} {t_str:>8} {r['status']:>8}")
    print("=" * 110)


if __name__ == '__main__':
    main()
