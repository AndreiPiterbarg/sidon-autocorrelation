"""Coarse-grid L3-bench: per-cell Lasserre ORDER-3 SDP for the COARSE-GRID cascade.

Goal
====
Strictly tighter than `_coarse_L_bench.py` (Shor / order-1) and
`_coarse_L2_bench.py` (Lasserre order-2), at higher per-cell cost.

The Lasserre hierarchy is sound at every order; order-3 lifts the moment
matrix to include all order-3 monomials up to total degree 3, giving a
PSD constraint on the (1 + d + d(d+1)/2 + d(d+1)(d+2)/6)-dim cone.

For d=4 the moment matrix size is:
    1 + 4 + 10 + 20 = 35
and we need moments up to degree 6 (since M_3[a,b] with |a|=|b|=3 gives
|a+b|=6).  Number of moments: C(d+6, 6) = C(10, 6) = 210.

Mathematical setup
==================
Same cell as in `_coarse_L_bench.py`:
    Cell(c) = { mu = c/S + delta : |delta_i| <= h, sum delta = 0 }, h = 1/(2S)

Inner objective per window W = (ell, s_lo):
    f_W(delta) = grad_W . delta + (2d/ell) * delta^T A_W delta

Order-3 Lasserre relaxation
---------------------------
Basis B_3 = (1, delta_i, delta_i*delta_j (i<=j), delta_i*delta_j*delta_k (i<=j<=k))
    |B_3| = 1 + d + d(d+1)/2 + d(d+1)(d+2)/6

Moment vector y indexed by multi-indices alpha in N^d with |alpha| <= 6.

Constraints (all sound):
  (i)   y_0 = 1                                (probability normalization)
  (ii)  M_3 = M_3(y) >= 0                      (PSD moment matrix on B_3)
  (iii) M_2((h - delta_i) y) >= 0  for all i   (localizing PSD on B_2 basis)
  (iv)  M_2((h + delta_i) y) >= 0  for all i
  (v)   sum_i y_{e_i + alpha} = 0  for all alpha with |alpha| <= 5
        (this is the "lifted" sum=0 constraint by every monomial of degree <=5)

Soundness
=========
By Putinar/Lasserre, the order-r relaxation is a sound lower bound on
inf f_W over the semialgebraic feasible set.  Order-3 dominates order-2
(every order-2 PSD constraint is a principal submatrix of M_3, and the
order-2 localizers are principal submatrices of order-3 localizers when
those exist; the box localizers M_2 here are tighter than M_1 of L2).
Therefore L3_LB >= L2_LB >= Shor_LB always (up to numerical solver tol).

API
===
- `cell_cert_lasserre3(c_int, S, d, c_target, window, ...)`
- `cell_cert_lasserre3_max(...)` for best LB across windows.

Tests
=====
Hard cells at (d=4, S=200, c=1.281) — reflects the Lean-verifiable target.
"""
from __future__ import annotations
import os, sys, time, json, argparse, math, warnings
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# Filter cvxpy noise
warnings.filterwarnings('ignore')
os.environ.setdefault('CVXPY_VERBOSE', '0')

# Pull cvxpy lazily
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

# Import the Shor baseline + helpers from the existing bench
from _coarse_L_bench import (
    all_windows, build_A_matrix, tv_at, grad_at,
    cell_vertices, qp_min_vertex_eval,
    cell_cert_shor, cell_cert_shor_max,
    triangle_cert, find_hard_cells,
)

# Import L2 implementation for direct comparison
from _coarse_L2_bench import (
    cell_cert_lasserre2,
    _make_alpha_index, _make_basis2, _make_basis1, _add_alpha, _e,
)


# =====================================================================
# Multi-index machinery for order-3 Lasserre
# =====================================================================

def _make_basis3(d: int) -> List[Tuple[int, ...]]:
    """Order-3 basis: 1, delta_i, delta_i*delta_j (i<=j), delta_i*delta_j*delta_k (i<=j<=k).
    Each basis element is a multi-index alpha with sum <= 3.
    """
    basis = []
    # constant
    basis.append(tuple([0] * d))
    # order-1
    for i in range(d):
        a = [0] * d
        a[i] = 1
        basis.append(tuple(a))
    # order-2
    for i in range(d):
        for j in range(i, d):
            a = [0] * d
            if i == j:
                a[i] = 2
            else:
                a[i] = 1
                a[j] = 1
            basis.append(tuple(a))
    # order-3
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                a = [0] * d
                a[i] += 1
                a[j] += 1
                a[k] += 1
                basis.append(tuple(a))
    return basis


def _enum_alphas_upto(d: int, max_deg: int) -> List[Tuple[int, ...]]:
    """All multi-indices with sum <= max_deg, sorted by (deg, lex)."""
    out = []

    def _rec(remaining: int, current: List[int]):
        if len(current) == d:
            out.append(tuple(current))
            return
        for a in range(remaining + 1):
            current.append(a)
            _rec(remaining - a, current)
            current.pop()

    _rec(max_deg, [])
    out.sort(key=lambda a: (sum(a), a))
    return out


# =====================================================================
# Lasserre order-3 SDP cell certificate (single window)
# =====================================================================

def cell_cert_lasserre3(c_int: np.ndarray, S: int, d: int, c_target: float,
                        window: Tuple[int, int],
                        solver: str = 'auto',
                        tol: float = 1e-9,
                        enforce_box_localizers: bool = True,
                        enforce_diag_box: bool = True,
                        verbose: bool = False) -> Tuple[float, str]:
    """Order-3 Lasserre SDP LB on  min_{delta in Cell} TV_W(c/S + delta).

    Returns (lb, status).  `lb` is TV_W(c/S) + SDP_optimum.

    Sound: at each Lasserre order, the relaxation contains the true
    measure-feasible set, so SDP optimum LOWER-BOUNDS the QP minimum.

    For d=4, M_3 is 35x35; with 4 box localizers M_2 of size 15x15.
    """
    if not _HAS_CVXPY:
        return float('-inf'), 'NO_CVXPY'

    ell, s_lo = window
    c = np.asarray(c_int, dtype=np.float64)
    h = 1.0 / (2.0 * S)
    A = build_A_matrix(d, ell, s_lo)
    g = grad_at(c, S, d, ell, s_lo)
    tv0 = tv_at(c, S, d, ell, s_lo)
    scale = 2.0 * d / ell

    # Build moment-vector index (multi-indices up to degree 6)
    alphas = _enum_alphas_upto(d, 6)
    idx = {a: k for k, a in enumerate(alphas)}
    n_alpha = len(alphas)

    # cvxpy variable: y[k] for each alpha
    y = cp.Variable(n_alpha)

    cons = []
    # (i) y_0 = 1
    cons.append(y[idx[tuple([0] * d)]] == 1.0)

    # (ii) Moment matrix M_3: rows/cols indexed by basis3
    basis3 = _make_basis3(d)
    nB3 = len(basis3)
    M3 = cp.Variable((nB3, nB3), symmetric=True)
    for a in range(nB3):
        for b in range(a, nB3):
            ab = _add_alpha(basis3[a], basis3[b])
            cons.append(M3[a, b] == y[idx[ab]])
    cons.append(M3 >> 0)

    # (iii)+(iv) Localizing matrices for (h ± delta_i) on basis B_2 (degree <= 2).
    # The product (h ± delta_i) * B_2[a] * B_2[b] has degree at most 1 + 2 + 2 = 5.
    # L_2((h - delta_i) y)[a, b] = h*y_{a+b} - y_{a+b+e_i}.
    if enforce_box_localizers:
        basis2 = _make_basis2(d)
        nB2 = len(basis2)
        for i in range(d):
            ei = _e(d, i)
            # (h - delta_i)
            L_minus = cp.Variable((nB2, nB2), symmetric=True)
            for a in range(nB2):
                for b in range(a, nB2):
                    ab = _add_alpha(basis2[a], basis2[b])
                    abi = _add_alpha(ab, ei)
                    cons.append(L_minus[a, b] == h * y[idx[ab]] - y[idx[abi]])
            cons.append(L_minus >> 0)
            # (h + delta_i)
            L_plus = cp.Variable((nB2, nB2), symmetric=True)
            for a in range(nB2):
                for b in range(a, nB2):
                    ab = _add_alpha(basis2[a], basis2[b])
                    abi = _add_alpha(ab, ei)
                    cons.append(L_plus[a, b] == h * y[idx[ab]] + y[idx[abi]])
            cons.append(L_plus >> 0)

    # (v) Lifted sum=0 constraint by every monomial of degree <= 5:
    #     sum_i y_{e_i + alpha} = 0 for all alpha with |alpha| <= 5
    for alpha in alphas:
        if sum(alpha) > 5:
            continue
        cons.append(sum(y[idx[_add_alpha(_e(d, i), alpha)]] for i in range(d)) == 0.0)

    # Optional: tighten with redundant box constraints on diagonal moments.
    # delta_i^2k >= 0, delta_i^{2k} <= h^{2k}.  Helps numerics.
    if enforce_diag_box:
        for i in range(d):
            cons.append(y[idx[_e(d, i, 2)]] <= h * h)
            cons.append(y[idx[_e(d, i, 2)]] >= 0.0)
            cons.append(y[idx[_e(d, i, 4)]] <= h * h * h * h)
            cons.append(y[idx[_e(d, i, 4)]] >= 0.0)
            cons.append(y[idx[_e(d, i, 6)]] <= h ** 6)
            cons.append(y[idx[_e(d, i, 6)]] >= 0.0)

    # Objective: minimize sum_i g_i y_{e_i} + scale * sum_{i,j} A[i,j] y_{e_i + e_j}
    obj_lin = sum(float(g[i]) * y[idx[_e(d, i)]] for i in range(d))
    obj_quad = 0.0
    for i in range(d):
        for j in range(d):
            if A[i, j] != 0.0:
                obj_quad = obj_quad + scale * float(A[i, j]) * y[idx[_add_alpha(_e(d, i), _e(d, j))]]
    obj = obj_lin + obj_quad
    prob = cp.Problem(cp.Minimize(obj), cons)

    actual_solver = solver
    if solver == 'auto':
        avail = set(cp.installed_solvers())
        for s in ('MOSEK', 'CLARABEL', 'SCS'):
            if s in avail:
                actual_solver = s
                break

    try:
        if actual_solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=verbose,
                       mosek_params={
                           'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                           'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                           'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                       })
        elif actual_solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=verbose,
                       eps_abs=tol, eps_rel=tol, max_iter=400)
        else:
            prob.solve(solver='SCS', verbose=verbose, eps=tol, max_iters=20000)
    except Exception as e:
        return float('-inf'), f'EXC:{type(e).__name__}'

    status = prob.status
    if prob.value is None or status not in ('optimal', 'optimal_inaccurate'):
        return float('-inf'), status

    return tv0 + float(prob.value), status


def cell_cert_lasserre3_max(c_int: np.ndarray, S: int, d: int, c_target: float,
                            windows: Sequence[Tuple[int, int]],
                            solver: str = 'auto',
                            tol: float = 1e-9,
                            verbose: bool = False) -> Tuple[float, Tuple[int, int]]:
    """Best (= largest) single-window L3 LB.  Sound LB on min_delta max_W TV_W."""
    best_lb = float('-inf')
    best_W = (-1, -1)
    for W in windows:
        lb, status = cell_cert_lasserre3(c_int, S, d, c_target, W,
                                          solver=solver, tol=tol, verbose=verbose)
        if lb > best_lb:
            best_lb = lb
            best_W = W
    return best_lb, best_W


# =====================================================================
# Driver: compare Shor (order-1), L2, and L3 on hard cells
# =====================================================================

def run_compare(d: int, S: int, c_target: float, max_cells: int = 30,
                solver: str = 'auto', tol: float = 1e-9,
                also_l2: bool = True,
                verbose: bool = True):
    """Run Shor, L2, and L3 on the K hardest triangle-failing cells."""
    print(f"\n=== _lasserre3_cell_cert: d={d}, S={S}, c_target={c_target} ===")
    print(f"    cell width h = 1/(2S) = {1.0 / (2.0 * S):.6f}")
    print(f"    solver: {solver}")

    # M_3 size
    nB3 = 1 + d + d * (d + 1) // 2 + d * (d + 1) * (d + 2) // 6
    nB2 = 1 + d + d * (d + 1) // 2
    n_alpha = math.comb(d + 6, 6)
    print(f"    M_3 size: {nB3}x{nB3}, M_2 (loc) size: {nB2}x{nB2}, n_moments: {n_alpha}")

    hard, n_grid_pass, n_tri_cert, n_total = find_hard_cells(d, S, c_target)
    print(f"    grid-point passers  : {n_grid_pass:,}")
    print(f"    triangle certified  : {n_tri_cert:,}")
    print(f"    HARD cells (failing): {len(hard):,}")
    if not hard:
        print("    No hard cells — triangle certifies everything.")
        return {'d': d, 'S': S, 'c_target': c_target, 'n_hard': 0}

    hard.sort(key=lambda kv: -kv[1]['net'])
    hard = hard[:max_cells]

    print(f"    Running Shor + L2 + L3 SDP on {len(hard)} hardest cells.\n")
    print(f"    triangle net range: [{hard[0][1]['net']:+.6f}, "
          f"{hard[-1][1]['net']:+.6f}]")

    n_shor_cert = 0
    n_l2_cert = 0
    n_l3_cert = 0
    n_l3_strict_better = 0
    n_l3_violation = 0
    max_l3_violation = 0.0
    times_shor = []
    times_l2 = []
    times_l3 = []
    rows = []

    n_truly_hard_shor = 0  # triangle fails AND Shor fails
    n_l3_rescue_shor = 0   # Shor fails BUT L3 certifies
    n_l3_rescue_l2 = 0     # L2 fails BUT L3 certifies

    for k, (c, tri) in enumerate(hard):
        # Shor (best-only on triangle's W*)
        t0 = time.time()
        shor_lb, shor_status = cell_cert_shor(c, S, d, c_target, tri['W'],
                                                solver=solver, tol=tol)
        dt_shor = time.time() - t0

        # L2 on the same window
        if also_l2:
            t1 = time.time()
            l2_lb, l2_status = cell_cert_lasserre2(c, S, d, c_target, tri['W'],
                                                     solver=solver, tol=tol)
            dt_l2 = time.time() - t1
        else:
            l2_lb, l2_status, dt_l2 = float('-inf'), 'SKIP', 0.0

        # L3 on the same window
        t2 = time.time()
        l3_lb, l3_status = cell_cert_lasserre3(c, S, d, c_target, tri['W'],
                                                solver=solver, tol=tol)
        dt_l3 = time.time() - t2

        # Soundness check: L3_LB >= L2_LB >= Shor_LB - tol
        if l3_lb < shor_lb - 1e-6 and shor_lb > float('-inf') and l3_lb > float('-inf'):
            n_l3_violation += 1
            max_l3_violation = max(max_l3_violation, shor_lb - l3_lb)
        if also_l2 and l3_lb < l2_lb - 1e-6 and l2_lb > float('-inf') and l3_lb > float('-inf'):
            n_l3_violation += 1
            max_l3_violation = max(max_l3_violation, l2_lb - l3_lb)

        shor_cert = shor_lb >= c_target - 1e-9
        l2_cert = l2_lb >= c_target - 1e-9
        l3_cert = l3_lb >= c_target - 1e-9

        if shor_cert:
            n_shor_cert += 1
        if l2_cert:
            n_l2_cert += 1
        if l3_cert:
            n_l3_cert += 1

        if l3_lb > shor_lb + 1e-9 and shor_lb > float('-inf'):
            n_l3_strict_better += 1

        if not shor_cert:
            n_truly_hard_shor += 1
            if l3_cert:
                n_l3_rescue_shor += 1
        if also_l2 and not l2_cert and l3_cert:
            n_l3_rescue_l2 += 1

        times_shor.append(dt_shor)
        times_l2.append(dt_l2)
        times_l3.append(dt_l3)

        rows.append({
            'k': k, 'c': c.tolist(),
            'tri_net': float(tri['net']),
            'tri_W': list(tri['W']),
            'tri_tv': float(tri['tv']),
            'shor_lb': float(shor_lb) if shor_lb != float('-inf') else None,
            'l2_lb': float(l2_lb) if l2_lb != float('-inf') else None,
            'l3_lb': float(l3_lb) if l3_lb != float('-inf') else None,
            'shor_cert': bool(shor_cert),
            'l2_cert': bool(l2_cert),
            'l3_cert': bool(l3_cert),
            'l3_strict_better_than_shor': l3_lb > shor_lb + 1e-9,
            'gap_l3_minus_shor': float(l3_lb - shor_lb) if (
                l3_lb != float('-inf') and shor_lb != float('-inf')) else None,
            'gap_l3_minus_l2': float(l3_lb - l2_lb) if (
                l3_lb != float('-inf') and l2_lb != float('-inf')) else None,
            'time_shor_s': float(dt_shor),
            'time_l2_s': float(dt_l2),
            'time_l3_s': float(dt_l3),
            'shor_status': shor_status,
            'l2_status': l2_status,
            'l3_status': l3_status,
        })

        if verbose and k < 12:
            gap_sl3 = l3_lb - shor_lb if (l3_lb != float('-inf') and shor_lb != float('-inf')) else float('nan')
            gap_l2l3 = l3_lb - l2_lb if (l3_lb != float('-inf') and l2_lb != float('-inf')) else float('nan')
            print(f"    [{k:3d}] c={c.tolist()}  tri={tri['net']:+.5f}  "
                  f"Sh={shor_lb:.5f}({'C' if shor_cert else 'f'}) "
                  f"L2={l2_lb:.5f}({'C' if l2_cert else 'f'}) "
                  f"L3={l3_lb:.5f}({'C' if l3_cert else 'f'}) "
                  f"L3-Sh=+{gap_sl3:.1e} L3-L2=+{gap_l2l3:.1e}  "
                  f"T_S={dt_shor*1000:.0f}/T_L2={dt_l2*1000:.0f}/T_L3={dt_l3*1000:.0f}ms")

    times_shor = np.asarray(times_shor)
    times_l2 = np.asarray(times_l2)
    times_l3 = np.asarray(times_l3)

    pct = lambda x: 100.0 * x / max(1, len(hard))
    print(f"\n    --- Summary ---")
    print(f"    Hard cells tested      : {len(hard)}")
    print(f"    Shor certified         : {n_shor_cert:>4}  ({pct(n_shor_cert):.1f}%)")
    print(f"    Lasserre-2 certified   : {n_l2_cert:>4}  ({pct(n_l2_cert):.1f}%)")
    print(f"    Lasserre-3 certified   : {n_l3_cert:>4}  ({pct(n_l3_cert):.1f}%)")
    print(f"    L3 strictly > Shor LB  : {n_l3_strict_better}")
    print(f"    Soundness violations   : {n_l3_violation}  (max diff: {max_l3_violation:.2e})")
    print(f"    Hard for Shor (Shor fail): {n_truly_hard_shor}")
    print(f"    L3 rescues Shor-fail   : {n_l3_rescue_shor}")
    if also_l2:
        print(f"    L3 rescues L2-fail     : {n_l3_rescue_l2}")
    if len(times_shor):
        print(f"    Shor t/cell (ms): med={1000*np.median(times_shor):.1f}  "
              f"p95={1000*np.percentile(times_shor,95):.1f}  "
              f"max={1000*np.max(times_shor):.1f}")
        if also_l2:
            print(f"    L2   t/cell (ms): med={1000*np.median(times_l2):.1f}  "
                  f"p95={1000*np.percentile(times_l2,95):.1f}  "
                  f"max={1000*np.max(times_l2):.1f}")
        print(f"    L3   t/cell (ms): med={1000*np.median(times_l3):.1f}  "
              f"p95={1000*np.percentile(times_l3,95):.1f}  "
              f"max={1000*np.max(times_l3):.1f}")
        print(f"    Time ratio (L3/Shor): med="
              f"{np.median(times_l3)/max(1e-9,np.median(times_shor)):.2f}x")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_hard_total': len(hard),
        'n_hard_tested': len(hard),
        'n_shor_cert': n_shor_cert,
        'n_l2_cert': n_l2_cert,
        'n_l3_cert': n_l3_cert,
        'n_l3_strict_better': n_l3_strict_better,
        'n_truly_hard_shor': n_truly_hard_shor,
        'n_l3_rescue_shor': n_l3_rescue_shor,
        'n_l3_rescue_l2': n_l3_rescue_l2,
        'soundness_viol': n_l3_violation,
        'max_violation': max_l3_violation,
        'time_shor_med_ms': float(1000 * np.median(times_shor)) if len(times_shor) else None,
        'time_l2_med_ms': float(1000 * np.median(times_l2)) if len(times_l2) else None,
        'time_l3_med_ms': float(1000 * np.median(times_l3)) if len(times_l3) else None,
        'time_shor_p95_ms': float(1000 * np.percentile(times_shor, 95)) if len(times_shor) else None,
        'time_l3_p95_ms': float(1000 * np.percentile(times_l3, 95)) if len(times_l3) else None,
        'M3_size': nB3, 'M2_loc_size': nB2, 'n_moments': n_alpha,
        'rows': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--S', type=int, default=200)
    ap.add_argument('--c_target', type=float, default=1.281)
    ap.add_argument('--max_cells', type=int, default=30)
    ap.add_argument('--solver', default='auto')
    ap.add_argument('--also_l2', action='store_true', default=True)
    ap.add_argument('--no_l2', dest='also_l2', action='store_false')
    ap.add_argument('--out', default='_lasserre3_cell_cert_results.json')
    args = ap.parse_args()

    if not _HAS_CVXPY:
        print("ERROR: cvxpy not available — cannot run SDP.")
        sys.exit(1)

    r = run_compare(args.d, args.S, args.c_target,
                     max_cells=args.max_cells, solver=args.solver,
                     also_l2=args.also_l2)

    rcopy = dict(r)
    if 'rows' in rcopy and len(rcopy['rows']) > 30:
        rcopy['rows'] = rcopy['rows'][:30]
    with open(args.out, 'w') as fp:
        json.dump([rcopy], fp, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
