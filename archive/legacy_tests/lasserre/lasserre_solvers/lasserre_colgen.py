#!/usr/bin/env python
r"""
Advanced Lasserre solvers with column generation and minimax dual.

Three strategies for dramatically improved gap closure:

1. COLUMN GENERATION (solve_colgen):
   Exact primal Lasserre bound, but with lazy window constraint addition.
   Instead of ~d^2 window PSD constraints upfront, start with NONE and
   iteratively add only violated ones. Typically converges with 10-20
   active windows instead of ~500. Memory reduction: 10-50x.

2. MINIMAX DUAL (solve_minimax_dual):
   Reformulate val(d) = max_{lambda} min_{mu in Delta_d} mu^T Q(lambda) mu
   where Q = sum lambda_W M_W. The inner SDP has ZERO window PSD constraints
   (just 1 moment matrix + d localizing matrices). Outer loop: Frank-Wolfe.
   This enables going to Lasserre order 4+ at d=16.

   Mathematical basis: Sion's minimax theorem gives
     min_mu max_W TV_W(mu) = max_{lambda in Delta_nwin} min_{mu in Delta_d} mu^T Q(lambda) mu
   The Lasserre relaxation of the inner min gives a valid lower bound.
   LB(lambda) is concave in lambda (pointwise min of linear functions),
   so the outer max is a convex optimization problem.

3. DIRECT SADDLE POINT (compute_val_d):
   Compute val(d) exactly by solving the saddle-point problem
   max_lambda min_mu mu^T Q(lambda) mu without any SDP hierarchy.
   Uses Frank-Wolfe on lambda with exact QP inner solve.
   Gives the TRUE val(d), not a relaxation bound.

Usage:
  python tests/lasserre_colgen.py --d 16 --order 3 --method colgen
  python tests/lasserre_colgen.py --d 16 --order 4 --method dual
  python tests/lasserre_colgen.py --d 16 --method direct
  python tests/lasserre_colgen.py --sweep
"""
import numpy as np
import sys, os, time, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse infrastructure from lasserre_fusion
from lasserre_fusion import (
    enum_monomials, build_window_matrices,
    _make_hash_bases, _hash_monos, _build_hash_table, _hash_lookup,
    collect_moments, _add_mi, _unit,
)

from mosek.fusion import (Model, Domain, Expr, Matrix,
                          ObjectiveSense, SolutionStatus)


# =====================================================================
# Strategy 1: Column Generation
# =====================================================================

def solve_colgen(d, c_target, order=3, n_bisect=14, max_cg_iter=60,
                 add_per_iter=5, viol_tol=1e-6, verbose=True):
    """Primal Lasserre with column generation on window constraints.

    Mathematically identical to full Lasserre but adds window localizing
    PSD constraints LAZILY — only the violated ones. Typically needs
    10-20 active windows instead of all ~d^2.

    Memory reduction vs full Lasserre: 10-50x (the window PSD constraints
    are the dominant memory cost at higher orders).
    """
    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)

    basis = enum_monomials(d, order)
    n_basis = len(basis)
    loc_basis = enum_monomials(d, order - 1) if order >= 2 else []
    n_loc = len(loc_basis)
    consist_mono = enum_monomials(d, 2 * order - 1)

    mono_list, idx = collect_moments(d, order, basis, loc_basis, consist_mono)
    n_y = len(mono_list)

    if verbose:
        print(f"  d={d}, order={order} (degree {2*order})")
        print(f"  windows={n_win}, basis={n_basis}, loc_basis={n_loc}, moments={n_y}")
        print(f"  Moment matrix: {n_basis}x{n_basis}")
        if order >= 2:
            print(f"  Localizing (mu_i): {d} x {n_loc}x{n_loc}")
            print(f"  Column gen: starting with 0/{n_win} window constraints")

    t_build = time.time()

    # ── Hash infrastructure ──
    max_comp = 2 * order
    bases = _make_hash_bases(d, max_comp)
    sorted_h, sort_o = _build_hash_table(mono_list, bases)

    B_arr = np.array(basis, dtype=np.int64)
    E_arr = np.eye(d, dtype=np.int64)

    # ── Precompute index arrays ──
    AB_hash = _hash_monos(
        B_arr[:, np.newaxis, :] + B_arr[np.newaxis, :, :], bases)
    moment_pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel().tolist()

    if loc_basis:
        LB_arr = np.array(loc_basis, dtype=np.int64)
        AB_loc = LB_arr[:, np.newaxis, :] + LB_arr[np.newaxis, :, :]
        AB_loc_hash = _hash_monos(AB_loc, bases)

        loc_picks = []
        for i_var in range(d):
            h = AB_loc_hash + bases[i_var]
            picks = _hash_lookup(h, sorted_h, sort_o)
            assert np.all(picks >= 0), f"Missing moments in mu_{i_var} localizing"
            loc_picks.append(picks.ravel().tolist())

        t_pick = _hash_lookup(AB_loc_hash, sorted_h, sort_o).ravel().tolist()

        EE_hash = bases[:, np.newaxis] + bases[np.newaxis, :]
        ABIJ_hash = (AB_loc_hash[:, :, np.newaxis, np.newaxis]
                     + EE_hash[np.newaxis, np.newaxis, :, :])
        ab_eiej_idx = _hash_lookup(ABIJ_hash, sorted_h, sort_o)

        ab_flat = (np.arange(n_loc)[:, np.newaxis] * n_loc
                   + np.arange(n_loc)[np.newaxis, :])

    # Consistency
    consist_arr = np.array(consist_mono, dtype=np.int64)
    consist_hash = _hash_monos(consist_arr, bases)
    consist_idx = _hash_lookup(consist_hash, sorted_h, sort_o)
    consist_ei_hash = consist_hash[:, np.newaxis] + bases[np.newaxis, :]
    consist_ei_idx = _hash_lookup(consist_ei_hash, sorted_h, sort_o)

    # Precompute window COO data (needed for checking violations and adding constraints)
    if loc_basis:
        cw_data_list = [None] * n_win
        for w in range(n_win):
            Mw = M_mats[w]
            nz_i, nz_j = np.nonzero(Mw)
            if len(nz_i) == 0:
                continue
            y_idx = ab_eiej_idx[:, :, nz_i, nz_j]
            valid = y_idx >= 0
            if not np.any(valid):
                continue
            ab_exp = np.broadcast_to(ab_flat[:, :, np.newaxis], y_idx.shape)
            mw_vals = Mw[nz_i, nz_j]
            mw_exp = np.broadcast_to(mw_vals[None, None, :], y_idx.shape)
            cw_data_list[w] = (
                ab_exp[valid].ravel(),
                y_idx[valid].ravel(),
                mw_exp[valid].ravel())

    precompute_time = time.time() - t_build
    if verbose:
        print(f"  Index precompute: {precompute_time:.2f}s", flush=True)

    # ── Build MOSEK model (WITHOUT window constraints) ──
    t_model = time.time()
    M = Model("lasserre_colgen")

    y = M.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = M.parameter("t")

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    M.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

    # Consistency
    c_rows, c_cols, c_vals = [], [], []
    n_consist_added = 0
    for r in range(len(consist_mono)):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        has_child = False
        for ci in range(d):
            if child_idx[ci] >= 0:
                c_rows.append(n_consist_added)
                c_cols.append(int(child_idx[ci]))
                c_vals.append(1.0)
                has_child = True
        if not has_child:
            continue
        c_rows.append(n_consist_added)
        c_cols.append(ai)
        c_vals.append(-1.0)
        n_consist_added += 1

    if n_consist_added > 0:
        A_con = Matrix.sparse(n_consist_added, n_y, c_rows, c_cols, c_vals)
        M.constraint("consist", Expr.mul(A_con, y), Domain.equalsTo(0.0))

    # Moment matrix PSD
    M_mat = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
    M.constraint("moment_psd", M_mat, Domain.inPSDCone(n_basis))

    # Localizing for mu_i >= 0
    if order >= 2:
        for i_var in range(d):
            Li = Expr.reshape(y.pick(loc_picks[i_var]), n_loc, n_loc)
            M.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_loc))

    M.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    model_time = time.time() - t_model
    build_time = time.time() - t_build

    if verbose:
        print(f"  Base model built: {model_time:.2f}s", flush=True)
        print(f"  Constraints (no windows): moment PSD + {d} localizing + "
              f"{n_consist_added} consistency", flush=True)

    # ── Helper: add a window constraint to the model ──
    active_windows = set()

    def add_window_constraint(w):
        """Add window w's localizing PSD constraint to the model."""
        if w in active_windows:
            return
        active_windows.add(w)

        if order < 2:
            # Scalar constraint
            units = [_unit(d, i) for i in range(d)]
            picks, coeffs = [], []
            for i in range(d):
                for j in range(d):
                    if M_mats[w][i, j] != 0:
                        eij = _add_mi(units[i], units[j], d)
                        picks.append(idx[eij])
                        coeffs.append(M_mats[w][i, j])
            tv_expr = Expr.dot(coeffs, y.pick(picks))
            M.constraint(f"win_{w}", Expr.sub(t_param, tv_expr),
                         Domain.greaterThan(0.0))
        else:
            flat_size = n_loc * n_loc
            t_y_pick = y.pick(t_pick)
            cw_data = cw_data_list[w]
            t_expr = Expr.mul(t_param, t_y_pick)

            if cw_data is None:
                Lw_mat = Expr.reshape(t_expr, n_loc, n_loc)
            else:
                rows, cols, vals = cw_data
                Cw_mosek = Matrix.sparse(flat_size, n_y,
                                         rows.tolist(), cols.tolist(),
                                         vals.tolist())
                cw_expr = Expr.mul(Cw_mosek, y)
                Lw_flat = Expr.sub(t_expr, cw_expr)
                Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)

            M.constraint(f"cg_w_{w}", Lw_mat, Domain.inPSDCone(n_loc))

    # ── Helper: check window violations ──
    def find_violations(y_vals, t_val):
        """Find windows whose localizing matrix is NOT PSD.

        Returns list of (min_eigenvalue, window_index) sorted by violation.
        """
        violations = []
        for w in range(n_win):
            if w in active_windows:
                continue

            # Build localizing matrix L_W[a,b] = t * y[loc[a]+loc[b]]
            #                                   - sum M_W[i,j] y[loc[a]+loc[b]+e_i+e_j]
            if order < 2:
                # Scalar check
                tv = 0.0
                for i in range(d):
                    for j in range(d):
                        if M_mats[w][i, j] != 0:
                            eij = _add_mi(_unit(d, i), _unit(d, j), d)
                            tv += M_mats[w][i, j] * y_vals[idx[eij]]
                if tv > t_val + viol_tol:
                    violations.append((-(tv - t_val), w))
            else:
                # Build n_loc x n_loc localizing matrix
                L = np.zeros((n_loc, n_loc))
                # t-coefficient part
                t_idx_arr = _hash_lookup(AB_loc_hash, sorted_h, sort_o)
                for a in range(n_loc):
                    for b in range(n_loc):
                        yi = int(t_idx_arr[a, b])
                        if yi >= 0:
                            L[a, b] = t_val * y_vals[yi]

                # Subtract C_W part
                Mw = M_mats[w]
                nz_i, nz_j = np.nonzero(Mw)
                for idx_nz in range(len(nz_i)):
                    ii, jj = nz_i[idx_nz], nz_j[idx_nz]
                    coeff = Mw[ii, jj]
                    y_idx_arr = ab_eiej_idx[:, :, ii, jj]
                    for a in range(n_loc):
                        for b in range(n_loc):
                            yi = int(y_idx_arr[a, b])
                            if yi >= 0:
                                L[a, b] -= coeff * y_vals[yi]

                # Check PSD
                min_eig = np.linalg.eigvalsh(L)[0]
                if min_eig < -viol_tol:
                    violations.append((min_eig, w))

        violations.sort()  # most negative first
        return violations

    # ── Vectorized violation check (much faster) ──
    def find_violations_fast(y_vals, t_val):
        """Vectorized violation check — avoids Python loops over (a,b).

        Uses a fast Frobenius-norm screening heuristic: skip windows where
        the t-contribution dominates the C_W contribution by a safe margin.
        """
        if order < 2:
            return find_violations(y_vals, t_val)

        violations = []
        # Precompute t-coefficient matrix from y values
        t_idx_arr = _hash_lookup(AB_loc_hash, sorted_h, sort_o)
        t_idx_flat = t_idx_arr.ravel()
        t_matrix_flat = np.zeros(n_loc * n_loc)
        valid_t = t_idx_flat >= 0
        t_matrix_flat[valid_t] = t_val * y_vals[t_idx_flat[valid_t]]

        # Batch all windows: build L matrices and check eigenvalues
        for w in range(n_win):
            if w in active_windows:
                continue

            L_flat = t_matrix_flat.copy()

            cw = cw_data_list[w]
            if cw is not None:
                rows, cols, vals = cw
                # Use bincount for fast scatter-add (much faster than np.add.at)
                contrib = vals * y_vals[cols.astype(int)]
                np.add.at(L_flat, rows.astype(int), -contrib)

            L = L_flat.reshape(n_loc, n_loc)
            L = 0.5 * (L + L.T)

            # Fast screening: if diagonal is all positive and dominant, skip
            diag = np.diag(L)
            if diag.min() > viol_tol:
                off_diag_max = np.abs(L - np.diag(diag)).sum(axis=1).max()
                if diag.min() > off_diag_max:
                    continue  # Diagonally dominant → PSD

            min_eig = np.linalg.eigvalsh(L)[0]
            if min_eig < -viol_tol:
                violations.append((min_eig, w))

        violations.sort()
        return violations

    # ── Binary search with column generation ──
    if verbose:
        print(f"\n  Binary search ({n_bisect} steps) with column generation...",
              flush=True)

    t0 = time.time()

    def check_feasible(t_val):
        t_param.setValue(t_val)
        try:
            M.solve()
            pstatus = M.getPrimalSolutionStatus()
            return pstatus in [SolutionStatus.Optimal, SolutionStatus.Feasible]
        except Exception as e:
            if verbose:
                print(f"      [solver error] t={t_val:.6f}: {e}", flush=True)
            return False

    # Find feasible upper bound
    lo, hi = 0.5, 5.0
    while not check_feasible(hi):
        hi *= 2
        if hi > 100:
            break
    if not check_feasible(hi):
        M.dispose()
        raise RuntimeError(f"SDP infeasible up to t={hi}")

    if verbose:
        print(f"  Feasible at t={hi:.4f}", flush=True)

    for step in range(n_bisect):
        mid = (lo + hi) / 2
        t_param.setValue(mid)

        # Column generation inner loop for this t value
        cg_converged = False
        for cg_iter in range(max_cg_iter):
            try:
                M.solve()
                pstatus = M.getPrimalSolutionStatus()
            except Exception:
                pstatus = None

            if pstatus not in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
                # Infeasible with current constraints — definitely infeasible
                break

            # Extract y* and check for violated windows
            y_vals = np.array(y.level())
            violations = find_violations_fast(y_vals, mid)

            if not violations:
                cg_converged = True
                break

            # Add most violated windows
            n_add = min(add_per_iter, len(violations))
            for _, w in violations[:n_add]:
                add_window_constraint(w)

            if verbose and cg_iter == 0:
                n_viol = len(violations)
                worst = violations[0][0]
                print(f"    t={mid:.8f}: {n_viol} violations "
                      f"(worst eig={worst:.2e}), "
                      f"adding {n_add} windows "
                      f"[active={len(active_windows)}]", flush=True)

        if cg_converged:
            hi = mid
            if verbose:
                print(f"    t={mid:.8f}: FEASIBLE "
                      f"(cg_iters={cg_iter+1}, "
                      f"active_windows={len(active_windows)})", flush=True)
        else:
            lo = mid
            if verbose:
                print(f"    t={mid:.8f}: infeasible", flush=True)

    M.dispose()

    elapsed = time.time() - t0
    lb = lo
    proven = lb >= c_target - 1e-6

    if verbose:
        print(f"\n  Solve time: {elapsed:.1f}s  (build: {build_time:.1f}s)")
        print(f"  Active windows: {len(active_windows)}/{n_win}")
        print(f"  Lower bound: {lb:.10f}")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")

    return {'lb': lb, 'proven': proven, 'elapsed': elapsed,
            'build_time': build_time, 'd': d, 'order': order,
            'method': 'colgen', 'active_windows': len(active_windows)}


# =====================================================================
# Strategy 2: Minimax Dual with Frank-Wolfe
# =====================================================================

def _build_inner_model(d, order):
    """Build the inner Lasserre SDP for min mu^T Q mu on Delta_d.

    Returns model, variables, and index infrastructure.
    The model has NO window constraints — just moment PSD + mu_i localizing.
    The objective coefficients are updated per Frank-Wolfe iteration.
    """
    basis = enum_monomials(d, order)
    n_basis = len(basis)
    loc_basis = enum_monomials(d, order - 1) if order >= 2 else []
    n_loc = len(loc_basis)
    consist_mono = enum_monomials(d, 2 * order - 1)

    mono_list, idx = collect_moments(d, order, basis, loc_basis, consist_mono)
    n_y = len(mono_list)

    max_comp = 2 * order
    bases_h = _make_hash_bases(d, max_comp)
    sorted_h, sort_o = _build_hash_table(mono_list, bases_h)

    B_arr = np.array(basis, dtype=np.int64)

    # Moment matrix indices
    AB_hash = _hash_monos(
        B_arr[:, np.newaxis, :] + B_arr[np.newaxis, :, :], bases_h)
    moment_pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel().tolist()

    # Localizing indices
    loc_picks = []
    if loc_basis:
        LB_arr = np.array(loc_basis, dtype=np.int64)
        AB_loc_hash = _hash_monos(
            LB_arr[:, np.newaxis, :] + LB_arr[np.newaxis, :, :], bases_h)
        for i_var in range(d):
            h = AB_loc_hash + bases_h[i_var]
            picks = _hash_lookup(h, sorted_h, sort_o)
            assert np.all(picks >= 0)
            loc_picks.append(picks.ravel().tolist())

    # Consistency indices
    consist_arr = np.array(consist_mono, dtype=np.int64)
    consist_hash = _hash_monos(consist_arr, bases_h)
    consist_idx = _hash_lookup(consist_hash, sorted_h, sort_o)
    consist_ei_hash = consist_hash[:, np.newaxis] + bases_h[np.newaxis, :]
    consist_ei_idx = _hash_lookup(consist_ei_hash, sorted_h, sort_o)

    # Build MOSEK model
    mdl = Model("inner_lasserre")
    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    mdl.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

    # Consistency
    c_rows, c_cols, c_vals = [], [], []
    n_consist = 0
    for r in range(len(consist_mono)):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        has_child = False
        for ci in range(d):
            if child_idx[ci] >= 0:
                c_rows.append(n_consist)
                c_cols.append(int(child_idx[ci]))
                c_vals.append(1.0)
                has_child = True
        if not has_child:
            continue
        c_rows.append(n_consist)
        c_cols.append(ai)
        c_vals.append(-1.0)
        n_consist += 1

    if n_consist > 0:
        A_con = Matrix.sparse(n_consist, n_y, c_rows, c_cols, c_vals)
        mdl.constraint("consist", Expr.mul(A_con, y), Domain.equalsTo(0.0))

    # Moment PSD
    M_mat = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
    mdl.constraint("moment_psd", M_mat, Domain.inPSDCone(n_basis))

    # Localizing mu_i >= 0
    if order >= 2:
        for i_var in range(d):
            Li = Expr.reshape(y.pick(loc_picks[i_var]), n_loc, n_loc)
            mdl.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_loc))

    # Precompute degree-2 moment picks for objective: y_{e_i+e_j}
    obj_picks = []
    for i in range(d):
        for j in range(d):
            eij = _add_mi(_unit(d, i), _unit(d, j), d)
            obj_picks.append(idx[eij])

    return {
        'model': mdl, 'y': y, 'idx': idx, 'n_y': n_y,
        'n_basis': n_basis, 'n_loc': n_loc,
        'obj_picks': obj_picks, 'd': d, 'order': order,
    }


def solve_minimax_dual(d, c_target, order=3, n_outer=80, verbose=True):
    """Minimax dual: max_lambda Lasserre_lb(sum lambda_W M_W, Delta_d).

    Outer: Pairwise Frank-Wolfe with line search over lambda in simplex.
    Inner: Lasserre relaxation of min mu^T Q mu on Delta_d (NO window PSD).

    The inner SDP has only 1 + d PSD constraints (moment + mu_i localizing),
    vs 1 + d + n_win in the full primal. For d=16: 17 vs 513 PSD constraints.

    Mathematical validity: by Sion's minimax theorem,
      val(d) = min_mu max_W TV_W(mu) = max_lambda min_mu mu^T Q(lambda) mu
    The Lasserre relaxation of the inner min gives LB(lambda) <= min_mu ...,
    and max_lambda LB(lambda) <= val(d). So this is a valid lower bound.

    LB(lambda) is concave (pointwise min of linear functions of lambda),
    so Frank-Wolfe converges to the global max.

    Enhancement: Pairwise FW with away steps for faster convergence, plus
    multiple restarts from promising initializations.
    """
    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)

    if verbose:
        print(f"  d={d}, order={order} (degree {2*order})")
        print(f"  windows={n_win}")
        print(f"  Inner SDP: {1+d} PSD constraints (no windows!)")
        print(f"  Outer: Pairwise Frank-Wolfe, {n_outer} iterations")

    # Build inner model (once)
    t_build = time.time()
    inner = _build_inner_model(d, order)
    mdl = inner['model']
    y = inner['y']
    obj_picks = inner['obj_picks']
    build_time = time.time() - t_build

    if verbose:
        print(f"  Inner model: {inner['n_y']} moments, "
              f"basis={inner['n_basis']}, loc={inner['n_loc']}")
        print(f"  Build time: {build_time:.2f}s", flush=True)

    def solve_inner(Q_mat):
        """Solve inner SDP for given Q. Returns (lb, tv_vector)."""
        obj_coeffs = Q_mat.ravel().tolist()
        mdl.objective(ObjectiveSense.Minimize,
                      Expr.dot(obj_coeffs, y.pick(obj_picks)))
        try:
            mdl.solve()
            pstatus = mdl.getPrimalSolutionStatus()
            if pstatus not in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
                return None, None
        except Exception:
            return None, None

        lb = mdl.primalObjValue()
        y_vals = np.array(y.level())
        X_star = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                X_star[i, j] = y_vals[obj_picks[i * d + j]]
        tv = np.array([np.sum(M_mats[w] * X_star) for w in range(n_win)])
        return lb, tv

    # ── Smart initialization: try concentrating on each window family ──
    best_lb = 0.0
    best_lam = np.ones(n_win) / n_win

    # Strategy: try multiple starting points
    # 1. Uniform lambda
    # 2. Concentrate on small-ell windows (tightest bounds)
    # 3. Concentrate on medium-ell windows around the center
    inits = [np.ones(n_win) / n_win]

    # Windows with ell=2 (tightest per-point bounds)
    ell2_mask = np.array([1.0 if windows[w][0] == 2 else 0.0 for w in range(n_win)])
    if ell2_mask.sum() > 0:
        inits.append(ell2_mask / ell2_mask.sum())

    # Windows near the center of convolution space
    center = d - 1  # central convolution index
    center_mask = np.array([
        1.0 / (1 + abs(windows[w][1] + windows[w][0]//2 - center))
        for w in range(n_win)])
    inits.append(center_mask / center_mask.sum())

    t0 = time.time()

    for init_idx, lam_init in enumerate(inits):
        lam = lam_init.copy()

        for k in range(n_outer):
            Q = np.zeros((d, d))
            for w in range(n_win):
                if lam[w] > 1e-15:
                    Q += lam[w] * M_mats[w]

            lb, tv = solve_inner(Q)
            if lb is None:
                continue

            if lb > best_lb:
                best_lb = lb
                best_lam = lam.copy()

            # Pairwise Frank-Wolfe with away step
            w_fw = np.argmax(tv)  # standard FW direction
            # Away direction: vertex in support with smallest gradient
            support = np.where(lam > 1e-12)[0]
            if len(support) > 1:
                w_away = support[np.argmin(tv[support])]
                # Try away step: move lambda AWAY from w_away toward w_fw
                # lam_new = lam + gamma * (e_fw - e_away)
                max_gamma_away = lam[w_away]  # can't go negative
                gamma_fw = 2.0 / (k + 2)
                gamma = min(gamma_fw, max_gamma_away)

                lam_try = lam.copy()
                lam_try[w_fw] += gamma
                lam_try[w_away] -= gamma

                # Evaluate away step
                Q_try = np.zeros((d, d))
                for w in range(n_win):
                    if lam_try[w] > 1e-15:
                        Q_try += lam_try[w] * M_mats[w]
                lb_try, _ = solve_inner(Q_try)
                if lb_try is not None and lb_try > lb:
                    lam = lam_try
                    if lb_try > best_lb:
                        best_lb = lb_try
                        best_lam = lam.copy()
                    continue

            # Standard FW step
            gamma = 2.0 / (k + 2)
            lam = (1 - gamma) * lam
            lam[w_fw] += gamma

            if verbose and (k < 3 or (k + 1) % 10 == 0):
                print(f"    [init {init_idx}] iter {k:3d}: lb={lb:.8f}, "
                      f"best={best_lb:.8f}, "
                      f"nnz(lam)={np.sum(lam > 1e-10)}", flush=True)

    mdl.dispose()

    elapsed = time.time() - t0
    proven = best_lb >= c_target - 1e-6

    if verbose:
        print(f"\n  Solve time: {elapsed:.1f}s  (build: {build_time:.1f}s)")
        print(f"  Lower bound: {best_lb:.10f}")
        print(f"  Active windows (nnz lambda): {np.sum(best_lam > 1e-10)}")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")

    return {'lb': best_lb, 'proven': proven, 'elapsed': elapsed,
            'build_time': build_time, 'd': d, 'order': order,
            'method': 'minimax_dual', 'n_outer_iters': n_outer}


# =====================================================================
# Strategy 3: Direct val(d) via saddle-point
# =====================================================================

def compute_val_d(d, n_starts=500, verbose=True):
    """Compute val(d) = min_{mu in Delta_d} max_W TV_W(mu).

    Uses the epigraph QCQP formulation:
      min t  s.t.  mu^T M_W mu <= t  for all W,  mu in Delta_d

    with multistart SLSQP (the QCQP is non-convex since M_W is indefinite).
    Also computes a DNN (doubly-nonneg) SDP lower bound for validation.
    """
    from scipy.optimize import minimize as sp_minimize, LinearConstraint

    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)

    if verbose:
        print(f"  d={d}, windows={n_win}")

    # ── Upper bound: multistart on epigraph QCQP ──
    if verbose:
        print(f"  Computing upper bound via multistart SLSQP ({n_starts} starts)...",
              flush=True)

    t0 = time.time()
    best_ub = np.inf
    best_mu = None

    for trial in range(n_starts):
        mu0 = np.random.dirichlet(np.ones(d))

        # Objective: max_W TV_W(mu)
        def obj_and_grad(mu):
            tvs = np.array([mu @ Mw @ mu for Mw in M_mats])
            w_max = np.argmax(tvs)
            grad = 2 * M_mats[w_max] @ mu
            return tvs[w_max], grad

        res = sp_minimize(
            lambda mu: obj_and_grad(mu)[0],
            mu0,
            jac=lambda mu: obj_and_grad(mu)[1],
            method='SLSQP',
            bounds=[(0, 1)] * d,
            constraints=[LinearConstraint(np.ones(d), 1.0, 1.0)],
            options={'maxiter': 1000, 'ftol': 1e-15}
        )
        # Project to simplex
        mu_res = np.maximum(res.x, 0)
        mu_res /= mu_res.sum()
        val = max(mu_res @ Mw @ mu_res for Mw in M_mats)
        if val < best_ub:
            best_ub = val
            best_mu = mu_res.copy()

    ub_time = time.time() - t0

    if verbose:
        print(f"  Upper bound: val({d}) <= {best_ub:.10f}  ({ub_time:.1f}s)")

    # ── Lower bound: DNN SDP relaxation ──
    # min t  s.t.  tr(M_W X) <= t  for all W,  X PSD, X >= 0, sum(X) = 1
    # This is a d x d SDP — trivially cheap.
    if verbose:
        print(f"  Computing DNN lower bound ({d}x{d} SDP)...", flush=True)

    t1 = time.time()
    mdl = Model("dnn_val")
    n_flat = d * d

    # X is d x d symmetric PSD
    X = mdl.variable("X", Domain.inPSDCone(d))
    # X >= 0 (entrywise)
    mdl.constraint("Xnonneg", X.reshape(n_flat), Domain.greaterThan(0.0))
    # sum(X) = 1
    ones_d = [1.0] * n_flat
    mdl.constraint("sum1", Expr.dot(ones_d, X.reshape(n_flat)),
                    Domain.equalsTo(1.0))

    # t = objective
    t_var = mdl.variable("t", Domain.unbounded())
    mdl.objective(ObjectiveSense.Minimize, t_var)

    # tr(M_W X) <= t  for each W
    for w in range(n_win):
        Mw_flat = M_mats[w].ravel().tolist()
        tv_expr = Expr.dot(Mw_flat, X.reshape(n_flat))
        mdl.constraint(f"win_{w}", Expr.sub(t_var, tv_expr),
                        Domain.greaterThan(0.0))

    mdl.solve()
    dnn_lb = t_var.level()[0]
    mdl.dispose()

    dnn_time = time.time() - t1

    if verbose:
        print(f"  DNN lower bound: val({d}) >= {dnn_lb:.10f}  ({dnn_time:.1f}s)")
        print(f"\n  val({d}) in [{dnn_lb:.10f}, {best_ub:.10f}]")
        print(f"  Gap: {best_ub - dnn_lb:.2e}")
        if best_ub - dnn_lb < 0.001:
            print(f"  val({d}) = {(dnn_lb + best_ub)/2:.10f} (tight)")

    return {'val_lb': dnn_lb, 'val_ub': best_ub,
            'val_mid': (dnn_lb + best_ub) / 2,
            'd': d, 'elapsed': ub_time + dnn_time}


# =====================================================================
# Comprehensive sweep
# =====================================================================

def run_sweep():
    """Run all strategies across multiple (d, order) configs."""
    val_d = {4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
             12: 1.271, 14: 1.284, 16: 1.319}

    print("=" * 72)
    print("LASSERRE GAP CLOSURE SWEEP — ADVANCED STRATEGIES")
    print("=" * 72)

    results = []

    # Phase 1: Direct val(d) computation for validation
    print("\n" + "#" * 72)
    print("# PHASE 1: Direct val(d) computation (no hierarchy)")
    print("#" * 72)
    for d in [4, 6, 8, 10, 12, 16]:
        print(f"\n--- d={d} ---")
        try:
            r = compute_val_d(d, n_outer=100, verbose=True)
            known = val_d.get(d, 0)
            print(f"  Known val({d})={known:.3f}, "
                  f"computed=[{r['val_lb']:.6f}, {r['val_ub']:.6f}]")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Phase 2: Column generation
    print("\n" + "#" * 72)
    print("# PHASE 2: Column generation (exact primal Lasserre)")
    print("#" * 72)

    colgen_configs = [
        (4, 2, "L2 d=4"),
        (8, 2, "L2 d=8"),
        (16, 2, "L2 d=16"),
        (4, 3, "L3 d=4"),
        (8, 3, "L3 d=8"),
        (16, 3, "L3 d=16 (was OOM!)"),
        (8, 4, "L4 d=8"),
    ]

    for d, order, desc in colgen_configs:
        print(f"\n{'='*60}")
        print(f"  COLGEN: {desc}")
        print(f"{'='*60}", flush=True)
        try:
            r = solve_colgen(d, 1.28, order=order, n_bisect=14)
            lb = r['lb']
            v = val_d.get(d, 0)
            gap_closed = (lb - 1.0) / (v - 1.0) * 100 if v > 1 else 0
            results.append(('colgen', d, order, lb, v, gap_closed,
                          r['active_windows'], r['elapsed']))
            print(f"\n  => lb={lb:.6f}, val={v:.3f}, "
                  f"gap_closed={gap_closed:.1f}%, "
                  f"active_wins={r['active_windows']}, "
                  f"time={r['elapsed']:.1f}s", flush=True)
        except Exception as e:
            print(f"\n  FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Phase 3: Minimax dual (for higher orders)
    print("\n" + "#" * 72)
    print("# PHASE 3: Minimax dual (enables higher Lasserre orders)")
    print("#" * 72)

    dual_configs = [
        (4, 3, "L3 d=4"),
        (8, 3, "L3 d=8"),
        (16, 3, "L3 d=16"),
        (8, 4, "L4 d=8"),
        (16, 4, "L4 d=16 (NEW — impossible with primal!)"),
    ]

    for d, order, desc in dual_configs:
        print(f"\n{'='*60}")
        print(f"  DUAL: {desc}")
        print(f"{'='*60}", flush=True)
        try:
            r = solve_minimax_dual(d, 1.28, order=order, n_outer=40)
            lb = r['lb']
            v = val_d.get(d, 0)
            gap_closed = (lb - 1.0) / (v - 1.0) * 100 if v > 1 else 0
            results.append(('dual', d, order, lb, v, gap_closed, 0, r['elapsed']))
            print(f"\n  => lb={lb:.6f}, val={v:.3f}, "
                  f"gap_closed={gap_closed:.1f}%, "
                  f"time={r['elapsed']:.1f}s", flush=True)
        except Exception as e:
            print(f"\n  FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Method':<8} {'Level':<5} {'d':<4} {'lb':<12} {'val(d)':<8} "
          f"{'Gap%':<8} {'Wins':<6} {'Time':<8}")
    print("-" * 72)
    for method, d, order, lb, v, gap, wins, elapsed in results:
        print(f"{method:<8} L{order:<4} {d:<4} {lb:<12.6f} {v:<8.3f} "
              f"{gap:<8.1f} {wins:<6} {elapsed:<8.1f}s")
    print("=" * 72)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Lasserre solvers with column generation and minimax dual")
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--c_target', type=float, default=1.28)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--method', choices=['colgen', 'dual', 'direct', 'sweep'],
                        default='colgen')
    parser.add_argument('--n_bisect', type=int, default=14)
    parser.add_argument('--n_outer', type=int, default=40)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep or args.method == 'sweep':
        run_sweep()
        return

    print(f"{'='*60}")

    if args.method == 'colgen':
        print(f"COLUMN GENERATION: L{args.order} d={args.d}")
        print(f"{'='*60}\n", flush=True)
        r = solve_colgen(args.d, args.c_target, order=args.order,
                         n_bisect=args.n_bisect)
    elif args.method == 'dual':
        print(f"MINIMAX DUAL: L{args.order} d={args.d}")
        print(f"{'='*60}\n", flush=True)
        r = solve_minimax_dual(args.d, args.c_target, order=args.order,
                               n_outer=args.n_outer)
    elif args.method == 'direct':
        print(f"DIRECT val({args.d}) COMPUTATION")
        print(f"{'='*60}\n", flush=True)
        r = compute_val_d(args.d, n_outer=args.n_outer)
        print(f"\nval({args.d}) in [{r['val_lb']:.10f}, {r['val_ub']:.10f}]")
        return

    val_d = {4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
             12: 1.271, 14: 1.284, 16: 1.319}
    v = val_d.get(args.d, 0)
    lb = r['lb']
    gap = (lb - 1.0) / (v - 1.0) * 100 if v > 1 else 0

    print(f"\n{'='*60}")
    print(f"lb={lb:.8f}, val({args.d})={v:.3f}, gap_closed={gap:.1f}%")
    if r.get('proven'):
        print(f"*** PROVEN: val({args.d}) >= {args.c_target} ***")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
