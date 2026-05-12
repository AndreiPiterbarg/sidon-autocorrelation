#!/usr/bin/env python
r"""
Sparse Lasserre via Clarabel — PSD-capable solver for d=64-128.

Clarabel is an interior-point solver (Rust backend, Python API) that
handles nonneg + PSD cones natively. Unlike MOSEK, it uses a different
IPM formulation that may handle 12M variables.

Clarabel standard form:
  min  (1/2)x'Px + q'x
  s.t. Ax + s = b,  s in K

where K = product of cones (zero, nonneg, PSD, SOC, ...).

For our feasibility check at threshold t:
  - x = y (moment variables), nonneg
  - Zero cone: y_0 = 1, consistency equalities
  - Nonneg cone: scalar windows t >= TV_W(y)
  - PSD cones: sparse clique moment PSD, sparse localizing PSD

Usage:
  python tests/lasserre_clarabel.py --d 8 --bw 6      # quick test
  python tests/lasserre_clarabel.py --d 128 --bw 16   # the target
"""
import numpy as np
from scipy import sparse as sp
import math
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
from lasserre_scalable import _precompute
from lasserre_enhanced import (
    _build_banded_cliques, _build_clique_basis,
    _batch_check_violations, val_d_known,
)
import clarabel

val_d_known.update({32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448})

SQRT2 = math.sqrt(2.0)


# =====================================================================
# Build Clarabel problem data for a fixed threshold t
# =====================================================================

def _svec_idx(i, j, n):
    """Column-major lower-triangle svec index for (i,j) with i >= j."""
    return j * n - j * (j - 1) // 2 + (i - j)


def _build_clarabel_data(P, cliques, t_val, bin_to_clique):
    """Build Clarabel standard form: Ax + s = b, s in K, x >= 0.

    Clarabel requires s = b - Ax in K. We encode:
      1. Zero cone: equalities (y_0=1, consistency)
      2. Nonneg cone: y >= 0, scalar windows t >= TV_W(y)
      3. PSD cones: sparse moment PSD, sparse localizing PSD

    Variables x = y (moment vector).

    Returns (P_obj, q, A, b, cones).
    """
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    n_win = P['n_win']
    bases = P['bases']
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    idx = P['idx']

    A_rows, A_cols, A_vals = [], [], []
    b_list = []
    cones = []
    row = 0

    # ── ZERO CONE: y_0 = 1 ──
    # s = b - Ax = 0  =>  Ax = b  =>  y[idx_zero] = 1
    zero_tuple = tuple(0 for _ in range(d))
    A_rows.append(row)
    A_cols.append(idx[zero_tuple])
    A_vals.append(1.0)
    b_list.append(1.0)
    row += 1

    # ── ZERO CONE: Consistency  sum_i y_{alpha+e_i} - y_alpha = 0 ──
    consist_idx_arr = P['consist_idx']
    consist_ei_idx_arr = P['consist_ei_idx']
    consist_mono = P['consist_mono']

    for r_c in range(len(consist_mono)):
        ai = int(consist_idx_arr[r_c])
        if ai < 0:
            continue
        children = []
        for ci in range(d):
            ci_idx = int(consist_ei_idx_arr[r_c, ci])
            if ci_idx >= 0:
                children.append(ci_idx)
        if not children:
            continue
        # Constraint: sum(children) - parent = 0
        # => A row: [+1 for children, -1 for parent], b = 0
        for c in children:
            A_rows.append(row)
            A_cols.append(c)
            A_vals.append(1.0)
        A_rows.append(row)
        A_cols.append(ai)
        A_vals.append(-1.0)
        b_list.append(0.0)
        row += 1

    n_zero = row
    cones.append(clarabel.ZeroConeT(n_zero))

    # ── NONNEG CONE: y >= 0 ──
    # s = b - Ax >= 0  with b=0, A = -I  =>  s = y >= 0
    for i in range(n_y):
        A_rows.append(row)
        A_cols.append(i)
        A_vals.append(-1.0)
        b_list.append(0.0)
        row += 1
    cones.append(clarabel.NonnegativeConeT(n_y))

    # ── NONNEG CONE: scalar windows  t - TV_W(y) >= 0 ──
    # s_w = t - sum M_W[ij] y_{e_i+e_j} >= 0
    # => A row: [+M_W[ij] at col y_{e_i+e_j}], b = t
    F_coo = P['F_scipy'].tocoo()
    for k in range(len(F_coo.data)):
        A_rows.append(row + F_coo.row[k])
        A_cols.append(F_coo.col[k])
        A_vals.append(F_coo.data[k])  # positive: s = t - Fy >= 0 => Ay = Fy, b = t
    for w in range(n_win):
        b_list.append(t_val)
    row += n_win
    cones.append(clarabel.NonnegativeConeT(n_win))

    # ── PSD CONES: sparse clique moment PSD ──
    # For each clique, M_clique[a,b] = y[picks[a,b]].
    # Clarabel PSD uses svec (lower triangle, column-major, off-diag scaled by sqrt(2)).
    # s = b - Ax in PSD cone  =>  svec(M) = -A_psd @ y + 0, so b_psd = 0, A_psd = -svec_map.
    for c_idx, clique in enumerate(cliques):
        cb_arr = _build_clique_basis(clique, order, d)
        n_cb = len(cb_arr)
        cb_hash = _hash_monos(cb_arr, bases)
        AB_hash = cb_hash[:, None] + cb_hash[None, :]
        picks = _hash_lookup(AB_hash, sorted_h, sort_o)  # (n_cb, n_cb)

        svec_dim = n_cb * (n_cb + 1) // 2
        for j_col in range(n_cb):
            for i_row in range(j_col, n_cb):
                k = _svec_idx(i_row, j_col, n_cb)
                y_idx = int(picks[i_row, j_col])
                if y_idx < 0:
                    continue
                sc = SQRT2 if i_row != j_col else 1.0
                # s[row+k] = 0 - (-sc * y[y_idx]) = sc * y[y_idx]
                # => A[row+k, y_idx] = -sc  (so s = b - Ax = 0 - (-sc*y) = sc*y)
                A_rows.append(row + k)
                A_cols.append(y_idx)
                A_vals.append(-sc)
        for _ in range(svec_dim):
            b_list.append(0.0)
        row += svec_dim
        cones.append(clarabel.PSDTriangleConeT(n_cb))

    # ── PSD CONES: sparse localizing mu_i >= 0 ──
    if order >= 2:
        for i_var in range(d):
            c_idx = bin_to_clique.get(i_var, 0)
            clique = cliques[c_idx]
            cb_arr = _build_clique_basis(clique, order - 1, d)
            n_cb = len(cb_arr)
            cb_hash = _hash_monos(cb_arr, bases)
            AB_ei_hash = cb_hash[:, None] + cb_hash[None, :] + bases[i_var]
            picks = _hash_lookup(AB_ei_hash, sorted_h, sort_o)

            svec_dim = n_cb * (n_cb + 1) // 2
            for j_col in range(n_cb):
                for i_row in range(j_col, n_cb):
                    k = _svec_idx(i_row, j_col, n_cb)
                    y_idx = int(picks[i_row, j_col])
                    if y_idx < 0:
                        continue
                    sc = SQRT2 if i_row != j_col else 1.0
                    A_rows.append(row + k)
                    A_cols.append(y_idx)
                    A_vals.append(-sc)
            for _ in range(svec_dim):
                b_list.append(0.0)
            row += svec_dim
            cones.append(clarabel.PSDTriangleConeT(n_cb))

    # Assemble sparse A (CSC format for Clarabel)
    m = row
    A = sp.csc_matrix((A_vals, (A_rows, A_cols)), shape=(m, n_y),
                       dtype=np.float64)
    b = np.array(b_list, dtype=np.float64)

    # Objective: feasibility check — minimize 0
    P_obj = sp.csc_matrix((n_y, n_y), dtype=np.float64)
    q = np.zeros(n_y, dtype=np.float64)

    return P_obj, q, A, b, cones


# =====================================================================
# Solve one feasibility check
# =====================================================================

def _check_feasible_clarabel(P, cliques, bin_to_clique, t_val,
                              settings=None):
    """Check if the DSOS/sparse Lasserre SDP is feasible at threshold t.

    Returns (feasible: bool, y_vals: np.ndarray or None).
    """
    P_obj, q, A, b, cones = _build_clarabel_data(
        P, cliques, t_val, bin_to_clique)

    if settings is None:
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 500
        settings.time_limit = 300.0  # 5 min per solve
        settings.tol_gap_abs = 1e-6
        settings.tol_gap_rel = 1e-6
        settings.tol_feas = 1e-6

    solver = clarabel.DefaultSolver(P_obj, q, A, b, cones, settings)
    sol = solver.solve()

    feasible = str(sol.status) in ('Solved', 'AlmostSolved')
    y_vals = np.array(sol.x) if feasible else None

    return feasible, y_vals, str(sol.status)


# =====================================================================
# Main solver: binary search + CG
# =====================================================================

def solve_clarabel_sparse(d, c_target=1.28, order=2, bandwidth=16,
                          n_bisect=15, max_cg_rounds=10,
                          max_add_per_round=20, verbose=True):
    """Sparse Lasserre via Clarabel interior-point solver.

    Binary search on t with CG for window constraints.
    """
    t_total = time.time()

    print(f"{'='*70}")
    print(f"CLARABEL SPARSE LASSERRE: L{order} d={d} bw={bandwidth}")
    print(f"  n_bisect={n_bisect}, cg_rounds={max_cg_rounds}")
    print(f"{'='*70}\n", flush=True)

    # Step 1: Precompute
    print("Step 1: Precompute...", flush=True)
    P = _precompute(d, order, verbose)
    t_pre = time.time() - t_total
    print(f"  Precompute: {t_pre:.1f}s\n", flush=True)

    # Step 2: Build cliques
    cliques = _build_banded_cliques(d, bandwidth)
    n_cliques = len(cliques)
    clique_size = len(cliques[0])
    cb_size = len(enum_monomials(clique_size, order))
    cb_loc_size = len(enum_monomials(clique_size, order - 1)) if order >= 2 else 0

    # Bin-to-clique mapping
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

    # PSD size estimates
    n_mom_psd = n_cliques * cb_size * (cb_size + 1) // 2
    n_loc_psd = d * cb_loc_size * (cb_loc_size + 1) // 2
    print(f"Step 2: {n_cliques} cliques of size {clique_size}")
    print(f"  Moment PSD svec total: {n_mom_psd:,}")
    print(f"  Localizing PSD svec total: {n_loc_psd:,}")
    print(f"  Variables: {P['n_y']:,}\n", flush=True)

    # Step 3: Clarabel settings
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.time_limit = 600.0
    settings.tol_gap_abs = 1e-5
    settings.tol_gap_rel = 1e-5
    settings.tol_feas = 1e-5

    # Step 4: Binary search
    print("Step 3: Binary search...\n", flush=True)

    best_lb = 0.0

    # Find feasible upper bound
    lo, hi = 0.5, 5.0
    print(f"  Testing t={hi}...", flush=True)
    feas, _, status = _check_feasible_clarabel(
        P, cliques, bin_to_clique, hi, settings)
    if not feas:
        print(f"  t={hi}: {status}. Trying t=20...", flush=True)
        hi = 20.0
        feas, _, status = _check_feasible_clarabel(
            P, cliques, bin_to_clique, hi, settings)
    if not feas:
        print(f"  INFEASIBLE at t={hi}: {status}. Aborting.", flush=True)
        return {'lb': 0, 'd': d, 'order': order, 'elapsed': 0,
                'status': status}

    print(f"  Feasible at t={hi}", flush=True)

    # Binary search
    for step in range(n_bisect):
        mid = (lo + hi) / 2
        t_step = time.time()
        feas, y_vals, status = _check_feasible_clarabel(
            P, cliques, bin_to_clique, mid, settings)
        dt = time.time() - t_step
        if feas:
            hi = mid
            tag = "feas"
        else:
            lo = mid
            tag = "inf "
        v = val_d_known.get(d, 0)
        gc_pct = (lo - 1) / (v - 1) * 100 if v > 1 and lo > 1 else 0
        if verbose and (step < 3 or step == n_bisect - 1 or (step+1) % 3 == 0):
            print(f"    [{step+1:2d}/{n_bisect}] t={mid:.10f} {tag} "
                  f"({dt:.1f}s) lb={lo:.6f} gap={gc_pct:.1f}%", flush=True)

    best_lb = lo
    elapsed = time.time() - t_total
    v = val_d_known.get(d, 0)
    gc_pct = (best_lb - 1) / (v - 1) * 100 if v > 1 and best_lb > 1 else 0

    print(f"\n{'='*70}")
    print(f"RESULT: L{order} d={d} Clarabel sparse bw={bandwidth}")
    print(f"  lb = {best_lb:.10f}")
    print(f"  val({d}) = {v}")
    print(f"  gap_closure = {gc_pct:.1f}%")
    print(f"  elapsed = {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  variables = {P['n_y']:,}")
    print(f"{'='*70}")

    return {
        'lb': best_lb, 'd': d, 'order': order,
        'gap_closure': gc_pct, 'elapsed': elapsed,
        'bandwidth': bandwidth,
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse Lasserre via Clarabel")
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--bw', type=int, default=6)
    parser.add_argument('--bisect', type=int, default=15)
    parser.add_argument('--cg-rounds', type=int, default=10)
    parser.add_argument('--cg-add', type=int, default=20)
    parser.add_argument('--c_target', type=float, default=1.28)
    args = parser.parse_args()

    solve_clarabel_sparse(
        args.d, args.c_target, order=args.order, bandwidth=args.bw,
        n_bisect=args.bisect, max_cg_rounds=args.cg_rounds,
        max_add_per_round=args.cg_add)


if __name__ == '__main__':
    main()
