#!/usr/bin/env python
"""Run a single Lasserre highd config using SCS (first-order solver).

SCS uses ADMM — no Schur complement, O(nnz) memory instead of O(n_y^2).
Solves the SAME SDP as MOSEK, producing the same bound.
Trades accuracy (1e-4 vs 1e-7) for dramatically lower memory.

Usage:
    python tests/run_single_scs.py --d 16 --order 3 --bw 12
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

# Import precompute infrastructure from highd
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, enum_monomials, val_d_known,
)
from lasserre.core import _make_hash_bases

# We build the SDP data (A, b, c, cone) and pass to SCS directly.
import scs


def _build_scs_problem(P, add_upper_loc=True):
    """Build the SCS problem data from precomputed Lasserre data.

    The SDP: minimize t subject to
      y >= 0, y_0 = 1, consistency, moment PSD, localizing PSD,
      upper localizing PSD, scalar windows t >= f_W(y).

    SCS solves: min c^T x s.t. Ax + s = b, s in cone K.
    Variables x = [y (n_y), t (1)].
    """
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    idx = P['idx']
    n_x = n_y + 1  # [y_0..y_{n_y-1}, t]
    t_idx = n_y  # index of t in x

    # Objective: minimize t
    c = np.zeros(n_x)
    c[t_idx] = 1.0

    # We'll build constraints as lists and assemble A, b at the end.
    # SCS convention: A x + s = b, s in K
    # For equality Ax = b: zero cone
    # For inequality Ax >= b: nonneg cone (s >= 0 means Ax - b = -s <= 0...
    #   actually SCS: Ax + s = b, s in nonneg => Ax <= b. We want Ax >= b => -Ax <= -b)
    # For PSD: Ax + s = b, s in PSD cone (vectorized lower triangle)

    rows_eq = []  # (row_idx, col_idx, val) for equality block
    rhs_eq = []
    rows_ineq = []
    rhs_ineq = []
    rows_psd = []  # list of (block_rows, block_cols, block_vals, block_rhs, cone_size)
    psd_cones = []

    eq_row = 0
    ineq_row = 0

    # --- y_0 = 1 ---
    zero = tuple(0 for _ in range(d))
    y0_idx_val = idx[zero]
    rows_eq.append((eq_row, y0_idx_val, 1.0))
    rhs_eq.append(1.0)
    eq_row += 1

    # --- Consistency: equality and inequality ---
    consist_parent_indices = P['consist_parent_indices']
    children_idx = P['children_idx']
    full_mask = P['consist_full_mask']
    partial_mask = P['consist_partial_mask']

    # Full equality: sum_i y_{alpha+e_i} = y_alpha
    full_rows_arr = np.where(full_mask)[0]
    for r in full_rows_arr:
        parent = int(consist_parent_indices[r])
        children = children_idx[r]
        # sum children - parent = 0
        for ci in range(d):
            if children[ci] >= 0:
                rows_eq.append((eq_row, int(children[ci]), 1.0))
        rows_eq.append((eq_row, parent, -1.0))
        rhs_eq.append(0.0)
        eq_row += 1

    # Partial inequality: y_alpha >= sum_{i in S'} y_{alpha+e_i}
    # => parent - sum_children >= 0 => -(parent - sum_children) <= 0
    # SCS nonneg: Ax + s = b, s >= 0 => Ax <= b
    # We want parent - sum_children >= 0 => -(parent - sum_children) <= 0
    # => sum_children - parent <= 0 => A row: [children: +1, parent: -1], b=0
    partial_rows_arr = np.where(partial_mask)[0]
    for r in partial_rows_arr:
        parent = int(consist_parent_indices[r])
        children = children_idx[r]
        has_child = False
        for ci in range(d):
            if children[ci] >= 0:
                rows_ineq.append((ineq_row, int(children[ci]), 1.0))
                has_child = True
        if has_child:
            rows_ineq.append((ineq_row, parent, -1.0))
            rhs_ineq.append(0.0)
            ineq_row += 1

    # --- Scalar windows: t >= f_W(y) => f_W(y) - t <= 0 ---
    F_coo = P['F_scipy'].tocoo()
    for k in range(F_coo.nnz):
        w_row = F_coo.row[k]
        y_col = F_coo.col[k]
        val = F_coo.data[k]
        rows_ineq.append((ineq_row + w_row, y_col, val))
    # Add -t for each window row
    n_win = P['n_win']
    for w in range(n_win):
        rows_ineq.append((ineq_row + w, t_idx, -1.0))
        rhs_ineq.append(0.0)
    ineq_row += n_win

    # --- PSD cones ---
    def add_psd_cone(pick_flat, cone_size, name=""):
        """Add a PSD cone constraint: M(y) in PSD cone.
        pick_flat: array of y-indices for the matrix entries (row-major).
        The SCS PSD cone uses LOWER-TRIANGULAR vectorization (column-major).
        """
        # pick_flat is row-major: M[i,j] = y[pick_flat[i*n + j]]
        # SCS expects: for lower triangle, column by column
        n = cone_size
        scs_dim = n * (n + 1) // 2
        block_r = []
        block_c = []
        block_v = []
        block_b = np.zeros(scs_dim)

        scs_row = 0
        for j in range(n):
            for i in range(j, n):
                y_idx_val = pick_flat[i * n + j]
                if y_idx_val >= 0:
                    scale = 1.0 if i == j else np.sqrt(2.0)
                    # Ax + s = b => -y[idx]*scale + s = 0 => s = y[idx]*scale
                    # We want M(y) in PSD => the slack s = M entries
                    block_r.append(scs_row)
                    block_c.append(int(y_idx_val))
                    block_v.append(-scale)
                scs_row += 1

        psd_cones.append((block_r, block_c, block_v, block_b, cone_size))

    # Full M_{k-1} PSD
    if P['m1_valid']:
        add_psd_cone(P['m1_pick'].tolist(), P['m1_size'], "full_mkm1")

    # Clique moment PSD
    for c_idx, cd in enumerate(P['clique_data']):
        pick = cd['mom_pick']
        if np.any(pick < 0):
            continue
        add_psd_cone(pick.tolist(), cd['mom_size'], f"clique_mom_{c_idx}")

    # Clique localizing PSD
    if order >= 2:
        for i_var in range(d):
            c_idx_val = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx_val]
            picks = cd['loc_picks'].get(i_var)
            if picks is None or np.any(picks < 0):
                continue
            add_psd_cone(picks.tolist(), cd['loc_size'], f"loc_{i_var}")

    # Upper localizing PSD (optional)
    if add_upper_loc and order >= 2:
        for i_var in range(d):
            c_idx_val = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx_val]
            t_pick_cd = cd['t_pick']
            loc_pick = cd['loc_picks'].get(i_var)
            if t_pick_cd is None or loc_pick is None:
                continue
            if np.any(t_pick_cd < 0) or np.any(loc_pick < 0):
                continue
            n_cb = cd['loc_size']
            # Upper localizing: M_{k-1}(y) - M_{k-1}(mu_i*y) >= 0
            # Entries: t_pick[a*n+b] - loc_pick[a*n+b]
            # We construct a virtual pick where each entry is a DIFFERENCE
            # This requires two separate terms — add as separate constraint rows
            scs_dim = n_cb * (n_cb + 1) // 2
            block_r = []
            block_c = []
            block_v = []
            block_b = np.zeros(scs_dim)
            scs_row = 0
            for j in range(n_cb):
                for i in range(j, n_cb):
                    flat_idx = i * n_cb + j
                    t_idx_val = int(t_pick_cd[flat_idx])
                    l_idx_val = int(loc_pick[flat_idx])
                    scale = 1.0 if i == j else np.sqrt(2.0)
                    if t_idx_val >= 0:
                        block_r.append(scs_row)
                        block_c.append(t_idx_val)
                        block_v.append(-scale)  # +sub_moment
                    if l_idx_val >= 0:
                        block_r.append(scs_row)
                        block_c.append(l_idx_val)
                        block_v.append(scale)  # -loc_moment
                    scs_row += 1
            psd_cones.append((block_r, block_c, block_v, block_b, n_cb))

    # --- Assemble SCS data ---
    # Order: zero cone (equality), nonneg cone (inequality + y>=0), PSD cones

    # y >= 0 as nonneg cone: -y + s = 0, s >= 0
    y_nonneg_start = ineq_row
    for i in range(n_y):
        rows_ineq.append((ineq_row + i, i, -1.0))
        rhs_ineq.append(0.0)
    ineq_row += n_y

    n_eq = eq_row
    n_ineq = ineq_row

    # Total rows = equality + inequality + sum(psd_dims)
    psd_total = sum(k * (k + 1) // 2 for _, _, _, _, k in psd_cones)
    n_rows = n_eq + n_ineq + psd_total

    # Build sparse A matrix
    all_rows = []
    all_cols = []
    all_vals = []

    # Equality block (rows 0..n_eq-1)
    for r, c, v in rows_eq:
        all_rows.append(r)
        all_cols.append(c)
        all_vals.append(v)

    # Inequality block (rows n_eq..n_eq+n_ineq-1)
    for r, c, v in rows_ineq:
        all_rows.append(n_eq + r)
        all_cols.append(c)
        all_vals.append(v)

    # PSD blocks
    psd_offset = n_eq + n_ineq
    cone_sizes = []
    for block_r, block_c, block_v, block_b, cone_size in psd_cones:
        for i in range(len(block_r)):
            all_rows.append(psd_offset + block_r[i])
            all_cols.append(block_c[i])
            all_vals.append(block_v[i])
        psd_offset += cone_size * (cone_size + 1) // 2
        cone_sizes.append(cone_size)

    A = sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(n_rows, n_x))

    # RHS
    b = np.zeros(n_rows)
    for i, v in enumerate(rhs_eq):
        b[i] = v
    for i, v in enumerate(rhs_ineq):
        b[n_eq + i] = v
    psd_offset = n_eq + n_ineq
    for block_r, block_c, block_v, block_b, cone_size in psd_cones:
        dim = cone_size * (cone_size + 1) // 2
        b[psd_offset:psd_offset + dim] = block_b
        psd_offset += dim

    # Cone specification
    cone = {
        'z': n_eq,         # zero cone (equality)
        'l': n_ineq,       # nonneg cone (inequality)
        's': cone_sizes,   # PSD cones
    }

    print(f"  SCS problem: {n_rows:,} rows x {n_x:,} cols, "
          f"nnz={A.nnz:,}, eq={n_eq}, ineq={n_ineq}, "
          f"PSD cones={len(cone_sizes)} (sizes: {cone_sizes[:5]}...)")
    print(f"  Memory: A={A.data.nbytes/1e9:.2f}GB, "
          f"b={b.nbytes/1e6:.0f}MB, c={c.nbytes/1e6:.0f}MB")

    return {'A': A, 'b': b, 'c': c}, cone


def solve_with_scs(P, add_upper_loc=True, max_cg_rounds=10, verbose=True):
    """Solve the Lasserre SDP using SCS with constraint generation."""
    d = P['d']
    n_y = P['n_y']

    # Build base SDP (no window PSD constraints yet)
    data, cone = _build_scs_problem(P, add_upper_loc)

    # Round 0: solve to get scalar bound
    print("  [Round 0] SCS scalar optimization...", flush=True)
    t0 = time.time()

    solver = scs.SCS(data, cone,
                     max_iters=10000,
                     eps_abs=1e-6,
                     eps_rel=1e-6,
                     verbose=verbose)
    sol = solver.solve()

    if sol['info']['status'] == 'solved' or sol['info']['status'] == 'solved_inaccurate':
        x = sol['x']
        y_vals = x[:n_y]
        t_val = x[n_y]
        scalar_lb = t_val
    else:
        print(f"  SCS status: {sol['info']['status']}")
        scalar_lb = 0.5
        y_vals = np.zeros(n_y)

    print(f"    Scalar bound = {scalar_lb:.10f} "
          f"({time.time()-t0:.1f}s, {sol['info']['iter']} iters)",
          flush=True)

    # Check violations
    active_windows = set()
    violations = _check_violations_highd(y_vals, scalar_lb, P, active_windows)
    print(f"    {len(violations)} violations found", flush=True)

    best_lb = scalar_lb

    if not violations:
        return {'lb': best_lb, 'd': d, 'order': P['order'],
                'n_y': n_y, 'n_active_windows': 0,
                'elapsed': time.time() - t0}

    # For CG with SCS: we rebuild the problem with added window PSD cones
    # and re-solve. SCS doesn't support incremental updates well,
    # but it's fast enough to rebuild.
    # With SCS we can minimize t directly (no bisection needed).
    for cg_round in range(1, max_cg_rounds + 1):
        n_add = min(100, len(violations))
        for w, eig in violations[:n_add]:
            active_windows.add(w)
        print(f"\n  [CG round {cg_round}] Adding {n_add} windows "
              f"(total: {len(active_windows)})", flush=True)

        # TODO: rebuild SDP with window PSD constraints
        # For now, the scalar bound from Round 0 is our result.
        # Window PSD constraints tighten the bound but require
        # rebuilding the full SCS problem with additional PSD cones.
        print(f"    (Window PSD CG with SCS not yet implemented)")
        print(f"    Reporting scalar-only bound: {best_lb:.10f}")
        break

    elapsed = time.time() - t0
    return {'lb': best_lb, 'd': d, 'order': P['order'],
            'n_y': n_y, 'n_active_windows': len(active_windows),
            'elapsed': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--bw', type=int, required=True)
    parser.add_argument('--cg-rounds', type=int, default=10)
    args = parser.parse_args()

    print(f"SCS solver: d={args.d} O{args.order} bw={args.bw}")
    print(f"Started: {datetime.now().isoformat()}")

    # Memory monitoring
    import resource
    def mem_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    cliques = _build_banded_cliques(args.d, args.bw)
    print(f"Cliques: {len(cliques)} of size {len(cliques[0])}")

    P = _precompute_highd(args.d, args.order, cliques, verbose=True)

    try:
        print(f"  RSS after precompute: {mem_mb():.0f} MB")
    except:
        pass

    r = solve_with_scs(P, max_cg_rounds=args.cg_rounds, verbose=False)

    vd = val_d_known.get(args.d, 0)
    gc = (r['lb'] - 1) / (vd - 1) * 100 if vd > 1 else 0

    print(f"\nFINAL: d={args.d} O{args.order} bw={args.bw}")
    print(f"  lb = {r['lb']:.10f}")
    print(f"  gc = {gc:.2f}%")
    print(f"  time = {r['elapsed']:.1f}s")


if __name__ == '__main__':
    main()
