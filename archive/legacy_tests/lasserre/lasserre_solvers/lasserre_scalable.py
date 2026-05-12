#!/usr/bin/env python
r"""
Scalable Lasserre hierarchy — constraint generation + upper-bound localizing.

The standard Lasserre SDP at order k has O(d^2) window localizing PSD cones,
each of size C(d+k-1, k-1)^2.  This dominates memory:

  L3 d=16 full:  513 PSD cones (969^2 + 512*153^2)  -> ~60GB  -> OOM
  L4 d=10 full:  ~290 PSD cones (each ~1001^2)      -> >1TB   -> OOM

CONSTRAINT GENERATION (mode='cg') — the only useful mode:
  Start with 0 PSD window constraints, lazily add violated ones.
  Converges to the exact full Lasserre bound with typically ~20-50 active
  windows instead of ~500.  The SDP at each iteration is much smaller.

  Additional tightening: --upper-loc adds (1 - mu_i) >= 0 localizing
  matrices (d extra PSD cones).  These are absent from standard Lasserre
  but valid on the simplex.  Cheap and yield ~5% better gap closure.

NOTE ON SCALAR-ONLY MODES:
  Replacing PSD window localizing with scalar t >= f_W(y) gives lb = 1.0
  regardless of order.  This is because the Lasserre feasible set L_k
  (without window PSD) admits a pseudo-distribution with constant
  convolution conv[k] = 1/(2d-1), making max_W f_W(y) = 1.  The PSD
  window constraints are essential, not just tightening — they certify
  that (t - f_W) is nonneg on the support of any representing measure.
  The 'linear' and 'fw' modes are retained only for diagnostic use.

Usage:
  python tests/lasserre_scalable.py --d 8 --order 2 --mode cg
  python tests/lasserre_scalable.py --d 16 --order 3 --mode cg
  python tests/lasserre_scalable.py --d 16 --order 3 --mode cg --cg-add 20
"""

import numpy as np
from scipy import sparse as sp
from mosek.fusion import (Model, Domain, Expr, Matrix,
                          ObjectiveSense, SolutionStatus)
import time
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
from lasserre_fusion import (
    enum_monomials, _make_hash_bases, _hash_monos,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
)
# Lazy ab_eiej slicing — avoids materialising the (n_loc^2, d^2) index array.
# Critical at d=32 L=3 where the full array is ~351 GB and triggers swap.
from lasserre.core import (
    ab_eiej_slice_batch as _ab_eiej_slice_batch,
    ab_eiej_slice_ij as _ab_eiej_slice_ij,
    get_ab_eiej_slice, get_ab_eiej_ij,
)

# Default: lazy mode (no 4D materialisation).  Can be forced off via env
# SIDON_LAZY_ABEIEJ=0 for regression comparison.
_LAZY_DEFAULT = os.environ.get('SIDON_LAZY_ABEIEJ', '1') == '1'


# =====================================================================
# Precompute all index arrays + window COO data
# =====================================================================

def _precompute(d, order, verbose=True, lazy_ab_eiej=None):
    """Precompute monomials, hash tables, index arrays, and window COO.

    lazy_ab_eiej: if True, skip building the full (n_loc, n_loc, d, d)
    ab_eiej_idx array; downstream consumers must call
    lasserre.core.get_ab_eiej_slice / get_ab_eiej_ij to read slices on
    demand.  At d=32 L=3 the full array is ~351 GB — lazy mode is the
    only way that size is feasible.  Default follows env SIDON_LAZY_ABEIEJ
    (default "1" → lazy).  Pass False to force eager materialisation.
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
    bases = _make_hash_bases(d, max_comp)
    sorted_h, sort_o = _build_hash_table(mono_list, bases)

    B_arr = np.array(basis, dtype=np.int64)
    E_arr = np.eye(d, dtype=np.int64)

    # Moment matrix: M[a,b] = y[basis[a]+basis[b]]
    # Use hash linearity: h(a+b) = h(a)+h(b) to avoid (n_basis,n_basis,d) array.
    # At d=128 L2: n_basis=8385, saving 72GB of temp memory.
    B_hash = _hash_monos(B_arr, bases)                      # (n_basis,)
    AB_hash = B_hash[:, None] + B_hash[None, :]             # (n_basis, n_basis)
    moment_pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel().tolist()

    # Localizing mu_i and sub-moment matrix
    loc_picks = []
    t_pick = None
    AB_loc_hash = None
    ab_eiej_idx = None
    ab_flat = None
    if loc_basis:
        LB_arr = np.array(loc_basis, dtype=np.int64)
        AB_loc = LB_arr[:, None, :] + LB_arr[None, :, :]
        AB_loc_hash = _hash_monos(AB_loc, bases)
        t_pick = _hash_lookup(AB_loc_hash, sorted_h, sort_o).ravel().tolist()

        for i_var in range(d):
            h = AB_loc_hash + bases[i_var]
            picks = _hash_lookup(h, sorted_h, sort_o)
            assert np.all(picks >= 0), f"Missing moments in mu_{i_var} loc"
            loc_picks.append(picks.ravel().tolist())

        # ab_eiej_idx[a,b,i,j] = idx[loc[a]+loc[b]+e_i+e_j]
        # via hash linearity: hash(ab+ei+ej) = AB_loc_hash[a,b] + EE_hash[i,j].
        # Lazy mode: skip materialising the full (n_loc, n_loc, d, d) array.
        # Consumers compute slices on demand via get_ab_eiej_slice /
        # get_ab_eiej_ij using AB_loc_hash + bases from P.
        if lazy_ab_eiej:
            ab_eiej_idx = None
            if verbose:
                n_loc_sq_d_sq = n_loc * n_loc * d * d
                sz_gb = n_loc_sq_d_sq * 8 / 1e9
                print(f"  ab_eiej_idx: LAZY (would have been "
                      f"({n_loc},{n_loc},{d},{d}) = {sz_gb:.2f}GB)",
                      flush=True)
        else:
            EE_hash = bases[:, None] + bases[None, :]  # (d, d)
            ABIJ_hash = (AB_loc_hash[:, :, None, None]
                         + EE_hash[None, None, :, :])   # (n_loc, n_loc, d, d)
            ab_eiej_idx = _hash_lookup(ABIJ_hash, sorted_h, sort_o)

        # Precompute flat (a,b) indices for COO — reused by all windows
        ab_flat = (np.arange(n_loc)[:, None] * n_loc
                   + np.arange(n_loc)[None, :])  # (n_loc, n_loc)

    # Degree-2 moment indices: idx_ij[i,j] = index of y_{e_i+e_j}
    EE_deg2 = E_arr[:, None, :] + E_arr[None, :, :]
    EE_deg2_hash = _hash_monos(EE_deg2, bases)
    idx_ij = _hash_lookup(EE_deg2_hash, sorted_h, sort_o)  # (d, d)

    # Consistency: alpha + e_i indices
    consist_arr = np.array(consist_mono, dtype=np.int64)
    consist_hash = _hash_monos(consist_arr, bases)
    consist_idx = _hash_lookup(consist_hash, sorted_h, sort_o)
    consist_ei_hash = consist_hash[:, None] + bases[None, :]
    consist_ei_idx = _hash_lookup(consist_ei_hash, sorted_h, sort_o)

    # --- Scalar window constraint: sparse F s.t. F @ y = [f_W(y)]_W ---
    f_r, f_c, f_v = [], [], []
    for w in range(n_win):
        Mw = M_mats[w]
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0:
            continue
        mi = idx_ij[nz_i, nz_j]
        valid = mi >= 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        f_r.extend([w] * n_valid)
        f_c.extend(mi[valid].tolist())
        f_v.extend(Mw[nz_i[valid], nz_j[valid]].tolist())

    # scipy sparse F for fast _eval_all_windows
    F_scipy = sp.csr_matrix(
        (f_v, (f_r, f_c)), shape=(n_win, n_y), dtype=np.float64)

    # Precompute set of nontrivial windows (M_w not all-zero) — filtered once
    nontrivial_windows = [w for w in range(n_win)
                          if np.any(M_mats[w] != 0)]

    # NOTE: window COO data is computed on-demand in _add_psd_window
    # (streaming approach from lasserre_sweep_large.py).  CG only adds
    # ~20-50 windows, so precomputing all ~500 is wasted memory.

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
        print(f"  Precompute: {elapsed:.2f}s"
              + (f"  (lazy_ab_eiej={lazy_ab_eiej})"
                 if loc_basis else ""),
              flush=True)

    return {
        'd': d, 'order': order,
        'windows': windows, 'M_mats': M_mats, 'n_win': n_win,
        'basis': basis, 'n_basis': n_basis,
        'loc_basis': loc_basis, 'n_loc': n_loc,
        'mono_list': mono_list, 'idx': idx, 'n_y': n_y,
        'bases': bases, 'sorted_h': sorted_h, 'sort_o': sort_o,
        'prime': None,  # lasserre_fusion pipeline uses non-modular hashing
        'lazy_ab_eiej': bool(lazy_ab_eiej),
        'moment_pick': moment_pick,
        'loc_picks': loc_picks, 't_pick': t_pick,
        'AB_loc_hash': AB_loc_hash, 'ab_eiej_idx': ab_eiej_idx,
        'ab_flat': ab_flat,
        'idx_ij': idx_ij,
        'consist_mono': consist_mono,
        'consist_idx': consist_idx, 'consist_ei_idx': consist_ei_idx,
        'f_r': f_r, 'f_c': f_c, 'f_v': f_v,
        'F_scipy': F_scipy,
        'nontrivial_windows': nontrivial_windows,
    }


# =====================================================================
# Build base MOSEK model (moment + localizing + consistency)
# =====================================================================

def _build_base_constraints(mdl, y, P, add_upper_loc, verbose):
    """Add moment matrix PSD, mu_i localizing PSD, consistency, y_0=1.

    Optionally adds (1-mu_i) >= 0 localizing matrices.
    """
    d = P['d']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    idx = P['idx']

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    mdl.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

    # Moment consistency: A @ y == 0
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

    # Moment matrix M_k(y) >> 0
    M_mat = Expr.reshape(y.pick(P['moment_pick']), n_basis, n_basis)
    mdl.constraint("moment_psd", M_mat, Domain.inPSDCone(n_basis))

    # Localizing mu_i >= 0
    if P['order'] >= 2:
        for i_var in range(d):
            Li = Expr.reshape(y.pick(P['loc_picks'][i_var]), n_loc, n_loc)
            mdl.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_loc))

        # (1 - mu_i) >= 0 localizing: M_{k-1}(y) - M_{k-1}(mu_i * y) >> 0
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
    """f_W(y) for all W via sparse matvec. O(nnz) instead of O(n_win*d^2)."""
    return P['F_scipy'].dot(y_vals)


def _check_window_violations(y_vals, t_val, P, active_windows, tol=1e-6):
    """Batch-check all non-active windows for localizing matrix violations.

    Uses precomputed ab_eiej_idx when eager, or on-demand lazy slices
    when P['ab_eiej_idx'] is None.  Lazy path computes per-window slices
    of shape (n_loc, n_loc, k) for k = nnz(Mw), bounding peak memory at
    O(n_loc^2 * max_nnz) instead of O(n_loc^2 * d^2).

    Returns sorted list of (window_index, min_eigenvalue) for violated windows.
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

    if ab_eiej_idx is not None:
        # Eager: precompute y at all (a,b,i,j) positions once — O(n_loc^2*d^2).
        safe_idx = np.clip(ab_eiej_idx, 0, len(y_vals) - 1)
        y_abij = y_vals[safe_idx]
        y_abij[ab_eiej_idx < 0] = 0.0

        violations = []
        for w in check_ws:
            Mw = P['M_mats'][w]
            L_q = np.einsum('ij,abij->ab', Mw, y_abij)
            L_w = L_t - L_q
            L_w = 0.5 * (L_w + L_w.T)
            min_eig = np.linalg.eigvalsh(L_w)[0]
            if min_eig < -tol:
                violations.append((w, float(min_eig)))
    else:
        # Lazy: per-window (n_loc, n_loc, k) slice with k = nnz(Mw).
        violations = []
        for w in check_ws:
            Mw = P['M_mats'][w]
            nz_i, nz_j = np.nonzero(Mw)
            if len(nz_i) == 0:
                continue
            y_idx = get_ab_eiej_slice(P, nz_i, nz_j)  # (n_loc, n_loc, k)
            safe = np.clip(y_idx, 0, len(y_vals) - 1)
            y_abk = y_vals[safe]
            y_abk[y_idx < 0] = 0.0
            mw_vals = Mw[nz_i, nz_j]
            # L_q[a, b] = sum_k mw_vals[k] * y_abk[a, b, k]
            L_q = np.tensordot(y_abk, mw_vals, axes=([2], [0]))
            L_w = L_t - L_q
            L_w = 0.5 * (L_w + L_w.T)
            min_eig = np.linalg.eigvalsh(L_w)[0]
            if min_eig < -tol:
                violations.append((w, float(min_eig)))

    violations.sort(key=lambda x: x[1])  # most negative first
    return violations


def _add_psd_window(mdl, y, t_param, w, P):
    """Add PSD localizing constraint for window w.

    COO data is computed on-demand (streaming), not read from precomputed
    storage.  MOSEK copies the data internally, so the COO arrays can be
    discarded immediately.  This saves ~50MB+ at d=16 L3.
    """
    n_loc = P['n_loc']
    n_y = P['n_y']
    ab_flat = P['ab_flat']
    Mw = P['M_mats'][w]

    t_y = Expr.mul(t_param, y.pick(P['t_pick']))

    nz_i, nz_j = np.nonzero(Mw)
    # Lazy-aware: either eager full array exists, or we need AB_loc_hash.
    if len(nz_i) == 0 or (
            P.get('ab_eiej_idx') is None and P.get('AB_loc_hash') is None):
        Lw_mat = Expr.reshape(t_y, n_loc, n_loc)
        mdl.constraint(f"w_psd_{w}", Lw_mat, Domain.inPSDCone(n_loc))
        return

    # Streaming COO construction (same math as lasserre_fusion.py)
    y_idx = get_ab_eiej_slice(P, nz_i, nz_j)   # (n_loc, n_loc, n_nz)
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


# =====================================================================
# Mode: CG — constraint generation (exact full Lasserre bound)
# =====================================================================

def solve_cg(d, c_target, order=3, n_bisect=15,
             add_upper_loc=True, cg_rounds=5, cg_add_per_round=10,
             conv_tol=1e-7, verbose=True):
    """Constraint generation Lasserre: lazily add PSD window constraints.

    Converges to the exact full Lasserre bound with far fewer PSD cones.
    The (1-mu_i) upper-bound localizing matrices add ~5% gap closure.

    Flow:
      Round 0: solve once at generous t to get y*, find initial violations,
               add them — NO wasted binary search for lb=1.0.
      Rounds 1+: binary search with tight range, check violations, add.
      Stop when: no violations, or improvement < conv_tol.
    """
    P = _precompute(d, order, verbose)
    n_win = P['n_win']
    n_y = P['n_y']
    n_loc = P['n_loc']

    t_build = time.time()
    mdl = Model("lasserre_cg")

    # Relaxed MOSEK tolerance — still far tighter than bisection resolution
    # (~1e-4) but ~30% faster per interior-point solve.
    mdl.setSolverParam("intpntCoTolRelGap", 1e-7)

    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = mdl.parameter("t")

    _build_base_constraints(mdl, y, P, add_upper_loc, verbose)

    # Scalar window constraints: t_param >= f_W(y) for all W
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col,
                         Expr.reshape(t_param, 1, 1)))
    mdl.constraint("win_scalar", Expr.sub(t_rep, f_all),
                   Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    build_time = time.time() - t_build
    if verbose:
        print(f"  Model built in {build_time:.1f}s", flush=True)

    def check_feasible(t_val):
        t_param.setValue(t_val)
        try:
            mdl.solve()
            ps = mdl.getPrimalSolutionStatus()
            return ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]
        except Exception:
            return False

    # --- Round 0: seed violations from the scalar-only boundary ---
    # The scalar-only bound is exactly 1.0 (flat-convolution pseudo-distribution).
    # Solve at t just above 1.0 to get the tightest y*, then check which
    # window localizing matrices are violated at that y*.
    # This replaces a full 15-step binary search that would converge to 1.0.
    active_windows = set()
    best_lb = 0.0

    if verbose:
        print(f"\n  [Round 0] Seeding initial violations...", flush=True)

    # Quick 5-step bisection to find scalar-only boundary tightly
    lo_seed, hi_seed = 0.5, 3.0
    if not check_feasible(hi_seed):
        hi_seed = 10.0
        check_feasible(hi_seed)
    for _ in range(5):
        mid = (lo_seed + hi_seed) / 2
        if check_feasible(mid):
            hi_seed = mid
        else:
            lo_seed = mid
    # hi_seed is now just above the scalar-only boundary (~1.0)
    t_param.setValue(hi_seed)
    mdl.solve()
    y_vals = np.array(y.level())
    violations = _check_window_violations(y_vals, hi_seed, P, active_windows)

    if verbose:
        print(f"    Scalar boundary ~ {lo_seed:.6f}, "
              f"{len(violations)} violations found", flush=True)

    if len(violations) == 0:
        mdl.dispose()
        elapsed = time.time() - t_build
        best_lb = lo_seed
        if verbose:
            print(f"    No violations -- scalar bound is exact.")
        return {'lb': best_lb, 'proven': best_lb >= c_target - 1e-6,
                'elapsed': elapsed, 'build_time': build_time,
                'd': d, 'order': order, 'mode': 'cg',
                'n_active_windows': 0}

    n_add = min(cg_add_per_round, len(violations))
    for w, min_eig in violations[:n_add]:
        _add_psd_window(mdl, y, t_param, w, P)
        active_windows.add(w)
    if verbose:
        print(f"    {len(violations)} violations found, "
              f"added {n_add} PSD windows "
              f"(worst eig: {violations[0][1]:.6e})")

    # --- Rounds 1+: binary search with tight range + adaptive termination ---
    for cg_round in range(1, cg_rounds + 1):
        if verbose:
            print(f"\n  [CG round {cg_round}] "
                  f"{len(active_windows)} PSD window constraints",
                  flush=True)

        # Tight binary search range
        lo = max(0.5, best_lb - 0.01)
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

        for step in range(n_bisect):
            mid = (lo + hi) / 2
            if check_feasible(mid):
                hi = mid
            else:
                lo = mid

        lb = lo
        improvement = lb - best_lb
        best_lb = max(best_lb, lb)

        if verbose:
            print(f"    lb={lb:.10f} (+{improvement:.2e})", flush=True)

        # Extract y* at feasible boundary and check violations
        t_param.setValue(hi)
        mdl.solve()
        y_vals = np.array(y.level())

        violations = _check_window_violations(
            y_vals, hi, P, active_windows)

        if len(violations) == 0:
            if verbose:
                print(f"    No violations -- converged.", flush=True)
            break

        # Adaptive termination: stop if improvement is negligible
        if improvement < conv_tol and cg_round >= 2:
            if verbose:
                print(f"    Improvement {improvement:.2e} < tol "
                      f"{conv_tol:.2e} -- stopping.", flush=True)
            break

        # Add most violated windows
        n_add = min(cg_add_per_round, len(violations))
        for w, min_eig in violations[:n_add]:
            _add_psd_window(mdl, y, t_param, w, P)
            active_windows.add(w)

        if verbose:
            print(f"    Added {n_add} PSD windows "
                  f"(worst eig: {violations[0][1]:.6e})")
            for w, eig in violations[:min(5, n_add)]:
                ell, s = P['windows'][w]
                print(f"      window ({ell},{s}): min_eig={eig:.6e}")

    mdl.dispose()
    gc.collect()

    proven = best_lb >= c_target - 1e-6
    elapsed = time.time() - t_build

    if verbose:
        print(f"\n  Total time: {elapsed:.1f}s")
        print(f"  Best lower bound: {best_lb:.10f}")
        print(f"  Active windows: {len(active_windows)}/{n_win}")
        print(f"  Margin over {c_target}: {best_lb - c_target:.10f}")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")

    return {
        'lb': best_lb, 'proven': proven,
        'elapsed': elapsed, 'build_time': build_time,
        'd': d, 'order': order, 'mode': 'cg',
        'n_active_windows': len(active_windows),
        'n_win_total': n_win,
    }


# =====================================================================
# Mode: LINEAR — diagnostic only (always gives lb=1.0)
# =====================================================================

def solve_linear(d, c_target, order=3, add_upper_loc=True, verbose=True):
    """Scalar windows only.  WARNING: always gives lb=1.0 (see docstring)."""
    P = _precompute(d, order, verbose)
    n_win = P['n_win']
    n_y = P['n_y']

    if verbose:
        print("  WARNING: linear mode gives lb=1.0 for all orders.")
        print("  Use --mode cg for useful bounds.", flush=True)

    t_build = time.time()
    mdl = Model("lasserre_linear")
    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_var = mdl.variable("t")

    _build_base_constraints(mdl, y, P, add_upper_loc, verbose)

    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, t_var))
    mdl.constraint("win_scalar", Expr.sub(t_rep, f_all),
                   Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, t_var)

    t0 = time.time()
    mdl.solve()

    pstatus = mdl.getPrimalSolutionStatus()
    if pstatus not in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
        mdl.dispose()
        raise RuntimeError(f"Solver failed: {pstatus}")

    lb = t_var.level()[0]
    solve_time = time.time() - t0
    build_time = time.time() - t_build
    mdl.dispose()

    if verbose:
        print(f"  Lower bound: {lb:.10f} (expected ~1.0)")

    return {
        'lb': lb, 'proven': lb >= c_target - 1e-6,
        'solve_time': solve_time, 'build_time': build_time,
        'd': d, 'order': order, 'mode': 'linear',
    }


# =====================================================================
# Unified entry point
# =====================================================================

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
}


def solve_lasserre_scalable(d, c_target, order=3, n_bisect=15,
                             mode='cg', add_upper_loc=True,
                             cg_rounds=5, cg_add_per_round=10,
                             verbose=True):
    """Unified entry point for scalable Lasserre hierarchy."""
    print(f"{'='*60}")
    print(f"SCALABLE LASSERRE L{order} (degree {2*order}), mode={mode}")
    print(f"Target: val({d}) >= {c_target}")
    if d in val_d_known:
        print(f"Known: val({d}) = {val_d_known[d]}")
    print(f"Upper-bound localizing: {'yes' if add_upper_loc else 'no'}")
    print(f"{'='*60}\n", flush=True)

    if mode == 'cg':
        r = solve_cg(d, c_target, order, n_bisect, add_upper_loc,
                     cg_rounds, cg_add_per_round, verbose)
    elif mode == 'linear':
        r = solve_linear(d, c_target, order, add_upper_loc, verbose)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'cg' or 'linear'.")

    # Report gap closure
    v = val_d_known.get(d)
    if v and v > 1 and r['lb'] > 0:
        gap_remain = (v - r['lb']) / (v - 1.0) * 100
        print(f"\n  Gap closure: {100 - gap_remain:.1f}% "
              f"(lb={r['lb']:.6f}, val({d})={v})")

    print(f"\n{'='*60}")
    if r['proven']:
        print(f"*** PROVEN: C_{{1a}} >= {c_target} ***")
    else:
        print(f"NOT PROVEN: lb={r['lb']:.8f} < {c_target}")
    print(f"{'='*60}")

    return r


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scalable Lasserre SDP via constraint generation")
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--c_target', type=float, default=1.28)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--mode', choices=['cg', 'linear'], default='cg')
    parser.add_argument('--bisect', type=int, default=15)
    parser.add_argument('--upper-loc', dest='upper_loc',
                        action='store_true', default=True)
    parser.add_argument('--no-upper-loc', dest='upper_loc',
                        action='store_false')
    parser.add_argument('--cg-rounds', type=int, default=5)
    parser.add_argument('--cg-add', type=int, default=10)
    args = parser.parse_args()

    solve_lasserre_scalable(
        args.d, args.c_target, args.order, args.bisect,
        args.mode, args.upper_loc,
        args.cg_rounds, args.cg_add)


if __name__ == '__main__':
    main()
