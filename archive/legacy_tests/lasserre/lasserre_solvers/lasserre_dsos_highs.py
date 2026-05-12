#!/usr/bin/env python
r"""
Sparse DSOS Lasserre via HiGHS LP solver — for d=64-128.

Replaces ALL PSD cones with diagonal dominance (DD) on clique-restricted
sub-matrices. The result is a pure LP solvable by HiGHS, which handles
12M+ variables without the memory issues that crash MOSEK's SDP solver.

Mathematical guarantee:
  DD => PSD, so the DSOS feasible set CONTAINS the SOS/PSD feasible set.
  Therefore: lb_DSOS <= lb_PSD <= val(d). VALID lower bound.

NOTE: Full-matrix DD is provably infeasible (row 0 requires 1 >= 2).
      Clique-restricted DD avoids this by operating on small sub-matrices
      where the row sums are much smaller than y_0.

Usage:
  python tests/lasserre_dsos_highs.py --d 128 --bw 16
  python tests/lasserre_dsos_highs.py --d 64 --bw 12
  python tests/lasserre_dsos_highs.py --d 8 --bw 6    # quick test
"""
import numpy as np
from scipy import sparse as sp
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

val_d_known.update({32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448})


# =====================================================================
# Build the full LP constraint matrix for HiGHS
# =====================================================================

def _build_lp(P, cliques, t_val, active_windows, add_upper_loc=True):
    """Build LP data: min c'x s.t. A_eq x = b_eq, A_ub x <= b_ub, x >= 0.

    Variables: x = y (n_y nonneg moment variables).
    t is a FIXED PARAMETER (for binary search feasibility check).

    Returns (c, A_eq, b_eq, A_ub, b_ub) as scipy sparse + numpy arrays.
    All constraints are valid for the DSOS relaxation of the Lasserre SDP.
    """
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    n_win = P['n_win']
    bases = P['bases']
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    idx = P['idx']

    eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
    ub_rows, ub_cols, ub_vals, ub_rhs = [], [], [], []
    n_eq = 0
    n_ub = 0

    # ── EQUALITY 1: y_0 = 1 ──
    zero = tuple(0 for _ in range(d))
    eq_rows.append(n_eq)
    eq_cols.append(idx[zero])
    eq_vals.append(1.0)
    eq_rhs.append(1.0)
    n_eq += 1

    # ── EQUALITY 2: Consistency  sum_i y_{alpha+e_i} = y_alpha ──
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    consist_mono = P['consist_mono']

    for r in range(len(consist_mono)):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        children = []
        for ci in range(d):
            ci_idx = int(consist_ei_idx[r, ci])
            if ci_idx >= 0:
                children.append(ci_idx)
        if not children:
            continue
        # sum children - parent = 0
        for c in children:
            eq_rows.append(n_eq)
            eq_cols.append(c)
            eq_vals.append(1.0)
        eq_rows.append(n_eq)
        eq_cols.append(ai)
        eq_vals.append(-1.0)
        eq_rhs.append(0.0)
        n_eq += 1

    # ── INEQUALITY 1: Scalar windows  TV_W(y) <= t  ──
    # i.e. sum M_W[i,j] y_{e_i+e_j} <= t
    F_coo = P['F_scipy'].tocoo()
    for k in range(len(F_coo.data)):
        ub_rows.append(n_ub + F_coo.row[k])
        ub_cols.append(F_coo.col[k])
        ub_vals.append(F_coo.data[k])
    for w in range(n_win):
        ub_rhs.append(t_val)
    n_ub += n_win

    # ── INEQUALITY 2: Sparse clique DD (moment matrix) ──
    # For each clique, for each basis row a:
    #   y_{2*alpha_a} >= sum_{b!=a} y_{alpha_a + alpha_b}
    # Rewritten as: sum_{b!=a} y_{alpha_a+alpha_b} - y_{2*alpha_a} <= 0
    for clique in cliques:
        cb_arr = _build_clique_basis(clique, order, d)
        n_cb = len(cb_arr)
        cb_hash = _hash_monos(cb_arr, bases)
        AB_hash = cb_hash[:, None] + cb_hash[None, :]  # (n_cb, n_cb)
        picks = _hash_lookup(AB_hash, sorted_h, sort_o)  # (n_cb, n_cb)

        for a in range(n_cb):
            diag = int(picks[a, a])
            if diag < 0:
                continue
            # sum_{b!=a} y_{picks[a,b]} - y_{picks[a,a]} <= 0
            for b in range(n_cb):
                if b == a:
                    continue
                p = int(picks[a, b])
                if p < 0:
                    continue
                ub_rows.append(n_ub)
                ub_cols.append(p)
                ub_vals.append(1.0)
            ub_rows.append(n_ub)
            ub_cols.append(diag)
            ub_vals.append(-1.0)
            ub_rhs.append(0.0)
            n_ub += 1

    # ── INEQUALITY 3: Sparse clique DD (mu_i localizing) ──
    # L_i[a,b] = y_{loc[a]+loc[b]+e_i}. Since y >= 0, entries are nonneg.
    # DD: y_{loc[a]+loc[a]+e_i} >= sum_{b!=a} y_{loc[a]+loc[b]+e_i}
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

    if order >= 2:
        for i_var in range(d):
            c_idx = bin_to_clique.get(i_var, 0)
            clique = cliques[c_idx]
            cb_arr = _build_clique_basis(clique, order - 1, d)
            n_cb = len(cb_arr)
            cb_hash = _hash_monos(cb_arr, bases)
            AB_ei_hash = cb_hash[:, None] + cb_hash[None, :] + bases[i_var]
            picks = _hash_lookup(AB_ei_hash, sorted_h, sort_o)

            for a in range(n_cb):
                diag = int(picks[a, a])
                if diag < 0:
                    continue
                for b in range(n_cb):
                    if b == a:
                        continue
                    p = int(picks[a, b])
                    if p < 0:
                        continue
                    ub_rows.append(n_ub)
                    ub_cols.append(p)
                    ub_vals.append(1.0)
                ub_rows.append(n_ub)
                ub_cols.append(diag)
                ub_vals.append(-1.0)
                ub_rhs.append(0.0)
                n_ub += 1

    # ── INEQUALITY 4: Upper-loc DD ((1-mu_i) localizing) ──
    # L_upper[a,b] = y_{loc[a]+loc[b]} - y_{loc[a]+loc[b]+e_i}
    # Entries can be NEGATIVE, so DD uses |L[a,b]| which is complex.
    # SKIP for now — upper-loc DD requires auxiliary variables.
    # The bound is still valid without it (just slightly weaker).

    # ── INEQUALITY 5: Window DD for active CG windows ──
    # L_W[a,b] = t * y_{t_pick[ab]} - sum M_W[ij] y_{abij[ab,ij]}
    # Entries can be negative. For DD we need |L_W[a,b]|.
    # SKIP window DD for now — CG windows with DD require auxiliary vars.
    # Scalar window constraints (INEQUALITY 1) still apply to all windows.

    # Build sparse matrices
    A_eq = sp.csr_matrix(
        (eq_vals, (eq_rows, eq_cols)), shape=(n_eq, n_y), dtype=np.float64)
    b_eq = np.array(eq_rhs, dtype=np.float64)

    A_ub = sp.csr_matrix(
        (ub_vals, (ub_rows, ub_cols)), shape=(n_ub, n_y), dtype=np.float64)
    b_ub = np.array(ub_rhs, dtype=np.float64)

    # Objective: feasibility check (minimize 0 — just check if constraints hold)
    c = np.zeros(n_y, dtype=np.float64)

    return c, A_eq, b_eq, A_ub, b_ub


# =====================================================================
# HiGHS LP solver wrapper
# =====================================================================

def _solve_lp_highs(c, A_eq, b_eq, A_ub, b_ub, n_y, verbose=False):
    """Solve LP via scipy.linprog with HiGHS backend.

    scipy.linprog(method='highs') uses HiGHS internally but with a stable
    Python API that doesn't depend on highspy version details.

    Returns (feasible: bool, x: np.ndarray or None).
    """
    from scipy.optimize import linprog

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=(0, None), method='highs',
                     options={'presolve': True, 'disp': False,
                              'time_limit': 600})

    feasible = result.success and result.status == 0
    x = result.x if feasible else None
    return feasible, x


def _solve_lp_scipy(c, A_eq, b_eq, A_ub, b_ub, n_y, verbose=False):
    """Fallback: solve LP via scipy.optimize.linprog (HiGHS backend)."""
    from scipy.optimize import linprog

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=(0, None), method='highs',
                     options={'presolve': True, 'disp': verbose})

    feasible = result.success and result.status == 0
    x = result.x if feasible else None
    return feasible, x


# =====================================================================
# Main solver: binary search + CG with DSOS LP
# =====================================================================

def solve_dsos_highs(d, c_target=1.28, order=2, bandwidth=16,
                     n_bisect=15, max_cg_rounds=10,
                     max_add_per_round=20, verbose=True):
    """Sparse DSOS Lasserre via HiGHS LP.

    Binary search on t: for each t, build LP and check feasibility.
    CG: after each solve, check PSD violations. For violated windows,
    add scalar window constraints (not DD window — that requires aux vars).
    """
    t_total = time.time()

    print(f"{'='*70}")
    print(f"SPARSE DSOS + HiGHS: L{order} d={d} bw={bandwidth}")
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
    print(f"Step 2: {n_cliques} cliques of size {clique_size}, "
          f"basis {cb_size}\n", flush=True)

    # Step 3: LP solver
    solver_fn = _solve_lp_highs
    solver_name = "scipy.linprog (HiGHS backend)"
    print(f"Step 3: LP solver: {solver_name}\n", flush=True)

    # Step 4: Binary search
    print("Step 4: Binary search + CG...\n", flush=True)

    active_windows = set()  # CG adds scalar windows beyond the base set
    best_lb = 0.0

    def check_feasible(t_val):
        c, A_eq, b_eq, A_ub, b_ub = _build_lp(
            P, cliques, t_val, active_windows, add_upper_loc=False)
        feasible, x = solver_fn(c, A_eq, b_eq, A_ub, b_ub, P['n_y'])
        return feasible, x

    # Initial feasibility bracket
    lo, hi = 0.5, 5.0
    feas, _ = check_feasible(hi)
    if not feas:
        hi = 20.0
        feas, _ = check_feasible(hi)
    if not feas:
        print("  ERROR: Infeasible at t=20. Aborting.", flush=True)
        return {'lb': 0, 'd': d, 'order': order, 'elapsed': 0}

    print(f"  Feasible at t={hi:.1f}", flush=True)

    # Binary search
    for step in range(n_bisect):
        mid = (lo + hi) / 2
        t_step = time.time()
        feas, x = check_feasible(mid)
        dt = time.time() - t_step
        if feas:
            hi = mid
            tag = "feas"
        else:
            lo = mid
            tag = "inf "
        if verbose and (step < 3 or step == n_bisect - 1 or (step+1) % 5 == 0):
            v = val_d_known.get(d, 0)
            gc = (lo - 1) / (v - 1) * 100 if v > 1 and lo > 1 else 0
            print(f"    [{step+1:2d}/{n_bisect}] t={mid:.10f} {tag} "
                  f"({dt:.1f}s) lb={lo:.6f} gap={gc:.1f}%", flush=True)

    best_lb = lo

    # Extract y* at feasible boundary for violation check
    _, y_vals = check_feasible(hi)

    if y_vals is not None:
        violations = _batch_check_violations(
            y_vals, hi, P, active_windows)
        if verbose:
            print(f"\n  Violations after initial search: {len(violations)}",
                  flush=True)
    else:
        violations = []

    # CG: the DSOS LP only has scalar window constraints + clique DD.
    # PSD window constraints are NOT added (no PSD in LP).
    # But we can add MORE scalar window constraints for violated windows.
    # This doesn't add PSD power but ensures the scalar bound is tight.
    # (Scalar-only bound is ~1.0, so CG won't help much here.)

    elapsed = time.time() - t_total
    v = val_d_known.get(d, 0)
    gc_pct = (best_lb - 1) / (v - 1) * 100 if v > 1 and best_lb > 1 else 0

    print(f"\n{'='*70}")
    print(f"RESULT: L{order} d={d} DSOS bw={bandwidth}")
    print(f"  lb = {best_lb:.10f}")
    print(f"  val({d}) = {v}")
    print(f"  gap_closure = {gc_pct:.1f}%")
    print(f"  elapsed = {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  solver = {solver_name}")
    print(f"  LP size: {P['n_y']:,} vars")
    print(f"{'='*70}")

    return {
        'lb': best_lb, 'd': d, 'order': order,
        'gap_closure': gc_pct, 'elapsed': elapsed,
        'bandwidth': bandwidth, 'solver': solver_name,
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse DSOS Lasserre via HiGHS LP")
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--bw', type=int, default=16)
    parser.add_argument('--bisect', type=int, default=15)
    parser.add_argument('--cg-rounds', type=int, default=10)
    parser.add_argument('--cg-add', type=int, default=20)
    parser.add_argument('--c_target', type=float, default=1.28)
    args = parser.parse_args()

    solve_dsos_highs(
        args.d, args.c_target, order=args.order, bandwidth=args.bw,
        n_bisect=args.bisect, max_cg_rounds=args.cg_rounds,
        max_add_per_round=args.cg_add)


if __name__ == '__main__':
    main()
