#!/usr/bin/env python
r"""
Lasserre sweep v2 — baseline CG vs CG + strengthened constraints.

Implements and benchmarks the valid ideas from benefit.md:
  IDEA 2: Product localizing  mu_i(1-mu_i) >= 0
  IDEA 5: Cross-moment localizing  (1 - sum mu_i^2) >= 0
  IDEA 4: Branch-and-bound (depth 1)

IDEA 1 (symmetry reduction) requires full basis repartitioning and is
not implemented here — it is a scaling optimization, not a tightening.

IDEA 3 was found to be REDUNDANT with L2 (see benefit.md audit) and
is excluded.

Usage:
  python tests/lasserre_sweep_v2.py
  python tests/lasserre_sweep_v2.py --d 8 --order 3
  python tests/lasserre_sweep_v2.py --d 4 --order 3 --no-branch
"""
import numpy as np
from mosek.fusion import (Model, Domain, Expr, Matrix,
                          ObjectiveSense, SolutionStatus)
import time
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_fusion import (
    enum_monomials, _make_hash_bases, _hash_monos,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
)
from lasserre_scalable import (
    _precompute, _check_window_violations, _add_psd_window,
)

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
}


# =====================================================================
# IDEA 5: Cross-moment localizing  (1 - sum mu_i^2) >= 0
# =====================================================================

def _add_cross_moment_loc(mdl, y, P, verbose=False):
    """Add M_1((1 - sum mu_i^2) * y) >> 0.

    On the simplex, 1 - sum mu_i^2 = 2*sum_{i<j} mu_i*mu_j >= 0.
    This is NOT implied by L2 (Putinar) — it lives in the Schmudgen
    preorder at level 2.

    Entries: M_1(g*y)[a,b] = y_{loc[a]+loc[b]} - sum_i y_{loc[a]+loc[b]+2e_i}
    """
    d = P['d']
    n_loc = P['n_loc']

    if n_loc == 0:
        return

    # Sub-moment matrix pick: y_{loc[a]+loc[b]}
    t_pick = P['t_pick']

    # For each i, compute pick list for y_{loc[a]+loc[b]+2*e_i}
    AB_loc_hash = P.get('AB_loc_hash')
    if AB_loc_hash is None:
        if verbose:
            print("    [cross-moment] No AB_loc_hash, skipping")
        return

    bases = P['bases']
    sorted_h = P['sorted_h']
    sort_o = P['sort_o']

    # Build the sum: sum_i y_{loc[a]+loc[b]+2e_i}
    # All picks at once via vectorized hash lookup
    # hash(loc[a]+loc[b]+2e_i) = AB_loc_hash[a,b] + 2*bases[i]
    two_e_hashes = 2 * bases  # (d,)
    # (n_loc, n_loc, d) hash array
    all_hashes = AB_loc_hash[:, :, np.newaxis] + two_e_hashes[np.newaxis, np.newaxis, :]
    all_picks = _hash_lookup(all_hashes, sorted_h, sort_o)  # (n_loc, n_loc, d)

    # Check for missing moments
    valid_mask = all_picks >= 0
    n_valid = int(valid_mask.sum())
    n_total = all_picks.size
    if verbose:
        print(f"    [cross-moment] {n_valid}/{n_total} valid picks "
              f"({100*n_valid/n_total:.1f}%)")

    # Build expression: y_{t_pick} - sum_i y_{picks_i}
    sub_moment_expr = y.pick(t_pick)

    # For missing moments, use index 0 (will contribute y_0 = 1, then subtract)
    # Actually safer: build sum only over valid entries per (a,b) pair
    # But for efficiency, clamp invalid to 0 and use a correction mask
    safe_picks = np.where(valid_mask, all_picks, 0)  # (n_loc, n_loc, d)

    # Sum over i axis: for each (a,b), sum y_{loc[a]+loc[b]+2e_i} for valid i
    # Build as: pick all d*n_loc^2 entries, multiply by valid_mask, reshape+sum
    flat_picks = safe_picks.reshape(n_loc * n_loc, d)  # (n_loc^2, d)
    flat_valid = valid_mask.reshape(n_loc * n_loc, d)   # (n_loc^2, d)

    # Build sparse COO for the sum: rows = flat (a,b) index, cols = y index
    coo_rows = []
    coo_cols = []
    coo_vals = []
    for i_var in range(d):
        col_picks = flat_picks[:, i_var]  # (n_loc^2,)
        v_mask = flat_valid[:, i_var]     # (n_loc^2,)
        valid_indices = np.where(v_mask)[0]
        if len(valid_indices) == 0:
            continue
        coo_rows.extend(valid_indices.tolist())
        coo_cols.extend(col_picks[valid_indices].tolist())
        coo_vals.extend([1.0] * len(valid_indices))

    if len(coo_rows) == 0:
        # No valid degree-4 moments — constraint is just M_1(y) >> 0
        if verbose:
            print("    [cross-moment] No valid picks, skipping")
        return

    n_y = P['n_y']
    flat_size = n_loc * n_loc
    S_mat = Matrix.sparse(flat_size, n_y, coo_rows, coo_cols, coo_vals)
    sum_sq_expr = Expr.mul(S_mat, y)

    cross_flat = Expr.sub(sub_moment_expr, sum_sq_expr)
    L_cross = Expr.reshape(cross_flat, n_loc, n_loc)
    mdl.constraint("cross_moment_loc", L_cross, Domain.inPSDCone(n_loc))

    if verbose:
        print(f"    [cross-moment] Added 1 PSD cone ({n_loc}x{n_loc})")


# =====================================================================
# IDEA 2: Product localizing  mu_i(1-mu_i) >= 0
# =====================================================================

def _add_product_loc(mdl, y, P, verbose=False):
    """Add M_1(mu_i(1-mu_i) * y) >> 0 for each i.

    On the simplex, mu_i in [0,1] so mu_i(1-mu_i) >= 0.
    Entries: M_1(g_i*y)[a,b] = y_{loc[a]+loc[b]+e_i} - y_{loc[a]+loc[b]+2e_i}

    NOT implied by L2 Putinar — the product mu_i*(1-mu_i) is in the
    Schmudgen preorder, not the quadratic module.
    """
    d = P['d']
    n_loc = P['n_loc']

    if n_loc == 0:
        return

    AB_loc_hash = P.get('AB_loc_hash')
    if AB_loc_hash is None:
        return

    bases = P['bases']
    sorted_h = P['sorted_h']
    sort_o = P['sort_o']
    loc_picks = P['loc_picks']  # loc_picks[i] = picks for y_{loc[a]+loc[b]+e_i}

    n_added = 0
    for i_var in range(d):
        # y_{loc[a]+loc[b]+e_i} — already computed in loc_picks
        picks_ei = loc_picks[i_var]  # list of length n_loc^2

        # y_{loc[a]+loc[b]+2*e_i}
        h_2ei = AB_loc_hash + 2 * bases[i_var]
        picks_2ei = _hash_lookup(h_2ei, sorted_h, sort_o)  # (n_loc, n_loc)
        picks_2ei_flat = picks_2ei.ravel()

        # Check all valid
        if np.any(picks_2ei_flat < 0):
            # Some degree-4 moments missing — skip this variable
            n_missing = int((picks_2ei_flat < 0).sum())
            if verbose and i_var == 0:
                print(f"    [product-loc] mu_{i_var}: {n_missing} missing "
                      f"degree-4 moments, skipping")
            continue

        picks_2ei_list = picks_2ei_flat.tolist()

        # Constraint: y.pick(picks_ei) - y.pick(picks_2ei) in PSDCone
        expr_ei = y.pick(picks_ei)
        expr_2ei = y.pick(picks_2ei_list)
        diff = Expr.sub(expr_ei, expr_2ei)
        L_prod = Expr.reshape(diff, n_loc, n_loc)
        mdl.constraint(f"prod_loc_{i_var}", L_prod, Domain.inPSDCone(n_loc))
        n_added += 1

    if verbose:
        print(f"    [product-loc] Added {n_added}/{d} PSD cones "
              f"({n_loc}x{n_loc} each)")


# =====================================================================
# Core CG solver with optional strengthening
# =====================================================================

def _build_base_constraints_v2(mdl, y, P, add_upper_loc, skip_moment_psd,
                               verbose):
    """Build base constraints with optional moment matrix PSD skip.

    When skip_moment_psd=True, the expensive M_k(y) >> 0 cone is omitted.
    This makes the relaxation WEAKER (larger feasible set, lower bound)
    but dramatically cheaper — enabling d=32-64 at L2.

    The remaining constraints (mu_i localizing, upper-loc, consistency, y>=0)
    still provide significant structure.
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

    # Moment matrix M_k(y) >> 0 — optionally skipped
    if skip_moment_psd:
        if verbose:
            print(f"  *** SKIPPING moment matrix PSD ({n_basis}x{n_basis}) ***")
    else:
        M_mat = Expr.reshape(y.pick(P['moment_pick']), n_basis, n_basis)
        mdl.constraint("moment_psd", M_mat, Domain.inPSDCone(n_basis))

    # Localizing mu_i >= 0
    n_psd = 0 if skip_moment_psd else 1
    if P['order'] >= 2:
        for i_var in range(d):
            Li = Expr.reshape(y.pick(P['loc_picks'][i_var]), n_loc, n_loc)
            mdl.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_loc))
        n_psd += d

        # (1 - mu_i) >= 0 localizing
        if add_upper_loc:
            for i_var in range(d):
                sub_moment = y.pick(P['t_pick'])
                mu_i_loc = y.pick(P['loc_picks'][i_var])
                diff_i = Expr.sub(sub_moment, mu_i_loc)
                L_upper = Expr.reshape(diff_i, n_loc, n_loc)
                mdl.constraint(f"loc_upper_{i_var}",
                               L_upper, Domain.inPSDCone(n_loc))
            n_psd += d
            if verbose:
                print(f"  Added {d} upper-bound localizing "
                      f"(1-mu_i) >= 0 constraints")

    if verbose:
        print(f"  Consistency constraints: {n_added}")
        print(f"  Total PSD cones (base): {n_psd}"
              f"{' (NO moment PSD)' if skip_moment_psd else ''}", flush=True)

    return n_added


def solve_cg_v2(d, c_target, order=3, n_bisect=15,
                add_upper_loc=True,
                add_product_loc=False,
                add_cross_moment_loc=False,
                skip_moment_psd=False,
                cg_rounds=5, cg_add_per_round=10,
                conv_tol=1e-7, verbose=True):
    """CG Lasserre with optional strengthened constraints.

    Parameters
    ----------
    add_product_loc : bool
        IDEA 2: Add mu_i(1-mu_i) >= 0 localizing for each i.
    add_cross_moment_loc : bool
        IDEA 5: Add (1 - sum mu_i^2) >= 0 localizing (single cone).
    skip_moment_psd : bool
        Drop the moment matrix PSD cone entirely. Weaker relaxation but
        dramatically cheaper — enables d=32-64 at L2.
    """
    P = _precompute(d, order, verbose)
    n_win = P['n_win']
    n_y = P['n_y']
    n_loc = P['n_loc']

    t_build = time.time()
    mdl = Model("lasserre_cg_v2")
    mdl.setSolverParam("intpntCoTolRelGap", 1e-7)

    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = mdl.parameter("t")

    # --- Base constraints ---
    _build_base_constraints_v2(mdl, y, P, add_upper_loc, skip_moment_psd,
                               verbose)

    # --- IDEA 5: Cross-moment localizing ---
    if add_cross_moment_loc:
        _add_cross_moment_loc(mdl, y, P, verbose)

    # --- IDEA 2: Product localizing ---
    if add_product_loc:
        _add_product_loc(mdl, y, P, verbose)

    # --- Scalar window constraints ---
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, Expr.reshape(t_param, 1, 1)))
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

    # --- Round 0: seed violations ---
    active_windows = set()
    best_lb = 0.0

    if verbose:
        print(f"\n  [Round 0] Seeding initial violations...", flush=True)

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
        return {'lb': best_lb, 'proven': best_lb >= c_target - 1e-6,
                'elapsed': elapsed, 'd': d, 'order': order,
                'n_active_windows': 0}

    n_add = min(cg_add_per_round, len(violations))
    for w, min_eig in violations[:n_add]:
        _add_psd_window(mdl, y, t_param, w, P)
        active_windows.add(w)
    if verbose:
        print(f"    Added {n_add} PSD windows "
              f"(worst eig: {violations[0][1]:.6e})")

    # --- CG rounds ---
    for cg_round in range(1, cg_rounds + 1):
        if verbose:
            print(f"\n  [CG round {cg_round}] "
                  f"{len(active_windows)} PSD windows", flush=True)

        lo = max(0.5, best_lb - 0.01)
        hi = best_lb * 1.05 + 0.15 if best_lb > 0.5 else 5.0

        while not check_feasible(hi):
            hi *= 1.5
            if hi > 100:
                break
        if hi > 100 and not check_feasible(hi):
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

        t_param.setValue(hi)
        mdl.solve()
        y_vals = np.array(y.level())

        violations = _check_window_violations(
            y_vals, hi, P, active_windows)

        if len(violations) == 0:
            if verbose:
                print(f"    No violations -- converged.", flush=True)
            break

        if improvement < conv_tol and cg_round >= 2:
            if verbose:
                print(f"    Improvement {improvement:.2e} < tol -- stopping.",
                      flush=True)
            break

        n_add = min(cg_add_per_round, len(violations))
        for w, min_eig in violations[:n_add]:
            _add_psd_window(mdl, y, t_param, w, P)
            active_windows.add(w)

        if verbose:
            print(f"    Added {n_add} PSD windows "
                  f"(worst eig: {violations[0][1]:.6e})")

    mdl.dispose()
    gc.collect()

    proven = best_lb >= c_target - 1e-6
    elapsed = time.time() - t_build

    if verbose:
        print(f"\n  Total time: {elapsed:.1f}s")
        print(f"  Best lower bound: {best_lb:.10f}")
        print(f"  Active windows: {len(active_windows)}/{n_win}")

    return {
        'lb': best_lb, 'proven': proven,
        'elapsed': elapsed, 'build_time': build_time,
        'd': d, 'order': order,
        'n_active_windows': len(active_windows),
        'n_win_total': n_win,
    }


# =====================================================================
# IDEA 4: Branch-and-bound (depth 1)
# =====================================================================

def solve_branched(d, c_target, order=3, n_bisect=15,
                   add_upper_loc=True,
                   add_product_loc=False,
                   add_cross_moment_loc=False,
                   cg_rounds=5, cg_add_per_round=10,
                   verbose=True):
    """Depth-1 branch-and-bound: branch on the highest-variance bin.

    1. Solve the base CG problem to get y* and lb_base.
    2. Identify branching variable i = argmax Var(mu_i) in y*.
    3. Solve two sub-problems with mu_i <= tau and mu_i >= tau.
    4. Return min(lb_left, lb_right) as the tightened lower bound.
    """
    if verbose:
        print(f"\n  === Branch-and-bound depth 1 ===", flush=True)
        print(f"  Phase 1: Solve base problem to find branching variable...",
              flush=True)

    # Phase 1: solve base to get y* for branching selection
    P = _precompute(d, order, verbose=False)
    n_y = P['n_y']
    n_loc = P['n_loc']
    n_win = P['n_win']

    # Quick base solve (fewer CG rounds, just to get y*)
    base_result = solve_cg_v2(
        d, c_target, order, n_bisect,
        add_upper_loc=add_upper_loc,
        add_product_loc=add_product_loc,
        add_cross_moment_loc=add_cross_moment_loc,
        cg_rounds=cg_rounds, cg_add_per_round=cg_add_per_round,
        verbose=verbose)

    lb_base = base_result['lb']

    if verbose:
        print(f"\n  Base lb = {lb_base:.10f}", flush=True)

    # We need y* to choose the branching variable.
    # Re-solve once more at the feasible boundary to extract y*.
    P2 = _precompute(d, order, verbose=False)
    mdl_extract = Model("extract_y")
    mdl_extract.setSolverParam("intpntCoTolRelGap", 1e-7)
    y_ext = mdl_extract.variable("y", n_y, Domain.greaterThan(0.0))
    t_ext = mdl_extract.parameter("t")

    from lasserre_scalable import _build_base_constraints
    _build_base_constraints(mdl_extract, y_ext, P2, add_upper_loc, False)

    if add_cross_moment_loc:
        _add_cross_moment_loc(mdl_extract, y_ext, P2, False)
    if add_product_loc:
        _add_product_loc(mdl_extract, y_ext, P2, False)

    F_mosek = Matrix.sparse(n_win, n_y, P2['f_r'], P2['f_c'], P2['f_v'])
    f_all = Expr.mul(F_mosek, y_ext)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, Expr.reshape(t_ext, 1, 1)))
    mdl_extract.constraint("win_scalar", Expr.sub(t_rep, f_all),
                           Domain.greaterThan(0.0))
    mdl_extract.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    # Solve at just-feasible t to get optimal y*
    t_feas = lb_base * 1.01 + 0.01
    t_ext.setValue(t_feas)
    try:
        mdl_extract.solve()
        ps = mdl_extract.getPrimalSolutionStatus()
        if ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
            y_vals = np.array(y_ext.level())
        else:
            y_vals = None
    except Exception:
        y_vals = None
    mdl_extract.dispose()

    if y_vals is None:
        if verbose:
            print("  Could not extract y* — returning base lb", flush=True)
        return base_result

    # Phase 2: choose branching variable = highest variance bin
    idx = P2['idx']
    d_val = P2['d']
    E_arr = np.eye(d_val, dtype=np.int64)

    y_ei = np.array([y_vals[idx[tuple(E_arr[i])]] for i in range(d_val)])
    y_2ei = np.array([y_vals[idx[tuple(2 * E_arr[i])]] for i in range(d_val)])
    var_i = y_2ei - y_ei ** 2  # Var(mu_i) in pseudo-distribution

    i_branch = int(np.argmax(var_i))
    tau = float(y_ei[i_branch])

    if verbose:
        print(f"\n  Phase 2: Branching on mu_{i_branch}")
        print(f"    E[mu_{i_branch}] = {tau:.6f}, "
              f"Var = {var_i[i_branch]:.6e}")
        print(f"    Threshold tau = {tau:.6f}")

    # Phase 3: solve left branch (mu_i <= tau)
    if verbose:
        print(f"\n  --- Left branch: mu_{i_branch} <= {tau:.6f} ---",
              flush=True)

    lb_left = _solve_branch(
        d, c_target, order, n_bisect, add_upper_loc,
        add_product_loc, add_cross_moment_loc,
        cg_rounds, cg_add_per_round,
        i_branch, tau, 'left', verbose)

    # Phase 4: solve right branch (mu_i >= tau)
    if verbose:
        print(f"\n  --- Right branch: mu_{i_branch} >= {tau:.6f} ---",
              flush=True)

    lb_right = _solve_branch(
        d, c_target, order, n_bisect, add_upper_loc,
        add_product_loc, add_cross_moment_loc,
        cg_rounds, cg_add_per_round,
        i_branch, tau, 'right', verbose)

    lb_branched = min(lb_left, lb_right)

    if verbose:
        print(f"\n  === Branch-and-bound results ===")
        print(f"  lb_base   = {lb_base:.10f}")
        print(f"  lb_left   = {lb_left:.10f}")
        print(f"  lb_right  = {lb_right:.10f}")
        print(f"  lb_branch = {lb_branched:.10f} "
              f"(+{lb_branched - lb_base:.6e} over base)")

    return {
        'lb': lb_branched, 'lb_base': lb_base,
        'lb_left': lb_left, 'lb_right': lb_right,
        'proven': lb_branched >= c_target - 1e-6,
        'elapsed': base_result['elapsed'],
        'd': d, 'order': order,
        'i_branch': i_branch, 'tau': tau,
    }


def _solve_branch(d, c_target, order, n_bisect, add_upper_loc,
                   add_product_loc, add_cross_moment_loc,
                   cg_rounds, cg_add_per_round,
                   i_branch, tau, direction, verbose):
    """Solve one branch of the B&B tree.

    Adds M_1((tau - mu_i)*y) >> 0 (left) or M_1((mu_i - tau)*y) >> 0 (right).
    """
    P = _precompute(d, order, verbose=False)
    n_y = P['n_y']
    n_loc = P['n_loc']
    n_win = P['n_win']

    mdl = Model(f"branch_{direction}")
    mdl.setSolverParam("intpntCoTolRelGap", 1e-7)
    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = mdl.parameter("t")

    from lasserre_scalable import _build_base_constraints
    _build_base_constraints(mdl, y, P, add_upper_loc, False)

    if add_cross_moment_loc:
        _add_cross_moment_loc(mdl, y, P, False)
    if add_product_loc:
        _add_product_loc(mdl, y, P, False)

    # --- Branching constraint ---
    # M_1((tau - mu_i)*y)[a,b] = tau*y_{loc[a]+loc[b]} - y_{loc[a]+loc[b]+e_i}
    # M_1((mu_i - tau)*y)[a,b] = y_{loc[a]+loc[b]+e_i} - tau*y_{loc[a]+loc[b]}
    t_pick = P['t_pick']  # y_{loc[a]+loc[b]}
    loc_pick_i = P['loc_picks'][i_branch]  # y_{loc[a]+loc[b]+e_i}

    sub_moment = y.pick(t_pick)
    mu_i_moment = y.pick(loc_pick_i)

    if direction == 'left':
        # tau * M_1(y) - M_1(mu_i * y) >> 0
        branch_expr = Expr.sub(Expr.mul(tau, sub_moment), mu_i_moment)
    else:
        # M_1(mu_i * y) - tau * M_1(y) >> 0
        branch_expr = Expr.sub(mu_i_moment, Expr.mul(tau, sub_moment))

    L_branch = Expr.reshape(branch_expr, n_loc, n_loc)
    mdl.constraint(f"branch_{direction}", L_branch, Domain.inPSDCone(n_loc))

    # --- Scalar window constraints ---
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, Expr.reshape(t_param, 1, 1)))
    mdl.constraint("win_scalar", Expr.sub(t_rep, f_all),
                   Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    def check_feasible(t_val):
        t_param.setValue(t_val)
        try:
            mdl.solve()
            ps = mdl.getPrimalSolutionStatus()
            return ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]
        except Exception:
            return False

    # --- CG loop ---
    active_windows = set()
    best_lb = 0.0

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

    t_param.setValue(hi_seed)
    mdl.solve()
    y_vals = np.array(y.level())
    violations = _check_window_violations(y_vals, hi_seed, P, active_windows)

    if len(violations) == 0:
        mdl.dispose()
        return lo_seed

    n_add = min(cg_add_per_round, len(violations))
    for w, min_eig in violations[:n_add]:
        _add_psd_window(mdl, y, t_param, w, P)
        active_windows.add(w)

    for cg_round in range(1, cg_rounds + 1):
        lo = max(0.5, best_lb - 0.01)
        hi = best_lb * 1.05 + 0.15 if best_lb > 0.5 else 5.0

        while not check_feasible(hi):
            hi *= 1.5
            if hi > 100:
                break
        if hi > 100 and not check_feasible(hi):
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
            print(f"    [{direction} CG {cg_round}] lb={lb:.10f} "
                  f"(+{improvement:.2e}), {len(active_windows)} windows",
                  flush=True)

        t_param.setValue(hi)
        mdl.solve()
        y_vals = np.array(y.level())

        violations = _check_window_violations(
            y_vals, hi, P, active_windows)

        if len(violations) == 0:
            break

        if improvement < 1e-7 and cg_round >= 2:
            break

        n_add = min(cg_add_per_round, len(violations))
        for w, min_eig in violations[:n_add]:
            _add_psd_window(mdl, y, t_param, w, P)
            active_windows.add(w)

    mdl.dispose()
    gc.collect()
    return best_lb


# =====================================================================
# Sweep: compare baseline vs strengthened
# =====================================================================

def run_config(desc, d, order, c_target=1.28, branch=False, **kwargs):
    """Run one config and return result dict."""
    print(f"\n{'#'*60}")
    print(f"# {desc}")
    print(f"{'#'*60}\n", flush=True)

    t0 = time.time()
    try:
        if branch:
            r = solve_branched(d, c_target, order, **kwargs)
        else:
            r = solve_cg_v2(d, c_target, order, **kwargs)
        elapsed = time.time() - t0
        lb = r['lb']
        v = val_d_known.get(d, 0)
        gc_pct = (lb - 1.0) / (v - 1.0) * 100 if v > 1 else 0
        print(f"\n  => lb={lb:.8f}, val({d})={v}, "
              f"gap_closure={gc_pct:.1f}%, time={elapsed:.1f}s\n")
        r['gap_closure'] = gc_pct
        r['time'] = elapsed
        r['desc'] = desc
        r['status'] = 'ok'
        return r
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return {'desc': desc, 'lb': 0, 'gap_closure': 0, 'time': elapsed,
                'd': d, 'order': order, 'status': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Lasserre sweep v2: baseline vs strengthened")
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--c_target', type=float, default=1.28)
    parser.add_argument('--bisect', type=int, default=15)
    parser.add_argument('--cg-rounds', type=int, default=5)
    parser.add_argument('--cg-add', type=int, default=10)
    parser.add_argument('--no-branch', action='store_true',
                        help='Skip branch-and-bound (faster)')
    args = parser.parse_args()

    d = args.d
    order = args.order
    c_target = args.c_target
    common = dict(
        n_bisect=args.bisect,
        cg_rounds=args.cg_rounds,
        cg_add_per_round=args.cg_add,
    )

    print("=" * 60)
    print("LASSERRE SWEEP V2 — BASELINE VS STRENGTHENED")
    print(f"  d={d}, order=L{order}, c_target={c_target}")
    print("=" * 60, flush=True)

    results = []

    # A. Baseline CG (full Lasserre with moment PSD)
    results.append(run_config(
        f"L{order} d={d} full CG",
        d, order, c_target,
        add_upper_loc=True,
        skip_moment_psd=False,
        **common))

    # B. NO moment matrix PSD (localizing + window PSD only)
    results.append(run_config(
        f"L{order} d={d} NO moment PSD",
        d, order, c_target,
        add_upper_loc=True,
        skip_moment_psd=True,
        **common))

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"{'Config':<45} {'lb':>12} {'val(d)':>8} "
          f"{'Gap%':>7} {'Time':>8}")
    print("-" * 80)
    for r in results:
        v = val_d_known.get(r.get('d', 0), 0)
        v_str = f"{v:.3f}" if v else "?"
        lb = r.get('lb', 0)
        lb_str = f"{lb:.8f}" if lb > 0 else "---"
        gc_str = f"{r.get('gap_closure', 0):.1f}%" if lb > 0 else "---"
        t_str = f"{r.get('time', 0):.1f}s"
        print(f"{r.get('desc', '?'):<45} {lb_str:>12} {v_str:>8} "
              f"{gc_str:>7} {t_str:>8}")
    print("=" * 80)

    # Delta analysis
    if len(results) >= 2 and results[0].get('lb', 0) > 0:
        base_lb = results[0]['lb']
        print(f"\nDelta from baseline (lb={base_lb:.8f}):")
        for r in results[1:]:
            lb = r.get('lb', 0)
            if lb > 0:
                delta = lb - base_lb
                print(f"  {r.get('desc', '?')}: {delta:+.8f}")


if __name__ == '__main__':
    main()
