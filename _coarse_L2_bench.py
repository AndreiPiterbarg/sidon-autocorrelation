"""Coarse-grid L2-bench: per-cell Lasserre ORDER-2 SDP for the COARSE-GRID cascade.

Goal
====
Strictly tighter than `_coarse_L_bench.py` (Shor / order-1), at higher per-cell
cost.  The Lasserre hierarchy is sound at every order; order-2 lifts the
moment matrix to include all order-2 monomials  (1, delta_i, delta_i*delta_j),
giving a PSD constraint on a ((d^2+d)/2 + d + 1)-dim cone.

Mathematical setup
==================
Same cell as in `_coarse_L_bench.py`:
    Cell(c) = { mu = c/S + delta : |delta_i| <= h, sum delta = 0 }, h = 1/(2S)

Inner objective per window W = (ell, s_lo):
    f_W(delta) = grad_W . delta + (2d/ell) * delta^T A_W delta

Order-2 Lasserre relaxation
---------------------------
Basis B_2 = (1, delta_1, ..., delta_d, delta_1^2, delta_1 delta_2, ..., delta_d^2)
    |B_2| = 1 + d + d*(d+1)/2

Moment vector y indexed by monomials alpha in {0..2}^d with |alpha| <= 4.
For our problem we need y_alpha for |alpha| <= 4 (since localizing matrices of
degree-1 polys * order-1 PSD lift involve degree-3 moments, and the
degree-2 moment matrix M_2 itself involves degree-4 moments).

Define multi-index encoding: alpha = (a_1, ..., a_d) with sum a_i <= 4.
The moment matrix M_2 has rows/cols indexed by B_2.  Entry M_2[i, j] is
the moment of basis_i * basis_j (a degree <= 4 monomial).

Constraints (all sound — relaxations of the true measure-feasible problem):
  (i)   y_0 = 1                               (probability normalization)
  (ii)  M_2 = M_2(y) >= 0                     (PSD moment matrix)
  (iii) M_1((h - delta_i) y) >= 0  forall i   (localizing PSD: degree-2 lift
        of (h - delta_i) on order-1 basis (1, delta_1, ..., delta_d))
  (iv)  M_1((h + delta_i) y) >= 0  forall i
  (v)   sum_i y_{e_i} = 0                     (linear: sum delta = 0)
  (vi)  sum_i y_{e_i + e_j} = 0  for all j    (lifted sum=0 = 0 multiplied by delta_j)
  (vii) sum_i y_{e_i + e_j + e_k} = 0  for j <= k  (lifted sum=0 = 0 by delta_j delta_k)

We additionally include all order-1 RLT cuts that the Shor implementation
already had, expressed as direct linear constraints on y_alpha.

Objective: minimize  sum_i grad_i * y_{e_i} + (2d/ell) * sum_{i,j} A_W[i,j] * y_{e_i+e_j}.

Soundness
=========
By Putinar/Lasserre, the order-r relaxation is a sound lower bound on
inf f_W over the semialgebraic feasible set.  Order-2 dominates order-1
(Shor) because every order-1 PSD constraint is implied by a (1+d) x (1+d)
principal submatrix of M_2, plus order-2 RLT is dominated by order-2 PSD
on M_2.  Therefore L2_LB >= Shor_LB always (up to numerical solver tol).

API
===
- `cell_cert_lasserre2(c_int, S, d, c_target, window, ...)` returns
  (lb, status) for a single window — same signature as `cell_cert_shor`.
- `cell_cert_lasserre2_max(...)` returns best LB across windows.

Tests
=====
Same hard-cell setup as `_coarse_L_bench.py`, plus a per-cell comparison
with Shor showing L2_LB >= Shor_LB and time/cert-rate tradeoff.
"""
from __future__ import annotations
import os, sys, time, json, argparse, math
from itertools import product, combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

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


# =====================================================================
# Multi-index machinery for order-2 Lasserre
# =====================================================================

def _make_alpha_index(d: int, max_deg: int) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int]]:
    """Enumerate all multi-indices alpha in N^d with sum(alpha) <= max_deg.

    Returns (alpha_list, alpha_to_idx) where alpha_list[k] is a tuple of length d
    and alpha_to_idx maps that tuple to k.
    """
    alphas = []

    def _rec(remaining: int, current: List[int]):
        if len(current) == d:
            if remaining >= 0:
                alphas.append(tuple(current))
            return
        for a in range(remaining + 1):
            current.append(a)
            _rec(remaining - a, current)
            current.pop()

    _rec(max_deg, [])
    # sort by total degree, then lex
    alphas.sort(key=lambda a: (sum(a), a))
    alpha_to_idx = {a: k for k, a in enumerate(alphas)}
    return alphas, alpha_to_idx


def _make_basis2(d: int) -> List[Tuple[int, ...]]:
    """Order-2 basis: 1, delta_i, delta_i*delta_j (i<=j).
    Each basis element is a multi-index alpha with sum <= 2.
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
    return basis


def _make_basis1(d: int) -> List[Tuple[int, ...]]:
    """Order-1 basis: 1, delta_1, ..., delta_d."""
    basis = [tuple([0] * d)]
    for i in range(d):
        a = [0] * d
        a[i] = 1
        basis.append(tuple(a))
    return basis


def _add_alpha(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _e(d: int, i: int, k: int = 1) -> Tuple[int, ...]:
    """Multi-index with k in slot i and 0 elsewhere."""
    a = [0] * d
    a[i] = k
    return tuple(a)


# =====================================================================
# Lasserre order-2 SDP cell certificate (single window)
# =====================================================================

def cell_cert_lasserre2(c_int: np.ndarray, S: int, d: int, c_target: float,
                         window: Tuple[int, int],
                         solver: str = 'auto',
                         tol: float = 1e-9,
                         enforce_box_localizers: bool = True,
                         enforce_diag_box: bool = True,
                         verbose: bool = False) -> Tuple[float, str]:
    """Order-2 Lasserre SDP LB on  min_{delta in Cell} TV_W(c/S + delta).

    Returns (lb, status).  `lb` is TV_W(c/S) + SDP_optimum (objective is the
    perturbation, then we add the constant TV_W(c/S)).

    Sound: at each Lasserre order, the relaxation contains the true measure-
    feasible set, so SDP optimum LOWER-BOUNDS the QP minimum.
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

    # Build moment-vector index (multi-indices up to degree 4)
    alphas, idx = _make_alpha_index(d, 4)
    n_alpha = len(alphas)

    # cvxpy variable: y[k] for each alpha
    # We'll parameterize by alpha-index, then build M_2 as a symbolic matrix.
    y = cp.Variable(n_alpha)

    cons = []
    # (i) y_0 = 1
    cons.append(y[idx[tuple([0] * d)]] == 1.0)

    # (ii) Moment matrix M_2: rows/cols indexed by basis2
    basis2 = _make_basis2(d)
    nB2 = len(basis2)
    # Build M_2 entries: M_2[a, b] = y_{a+b}
    # Use cp.bmat or a placeholder matrix and equate entries.
    M2_entries = [[None] * nB2 for _ in range(nB2)]
    for a in range(nB2):
        for b in range(nB2):
            ab = _add_alpha(basis2[a], basis2[b])
            M2_entries[a][b] = y[idx[ab]]
    # Construct M_2 using cp.bmat — each entry is a scalar Variable element.
    # cp.bmat needs a list of lists of expressions.  Wrap each scalar in cp.reshape(...,(1,1)) is overkill;
    # we can use cp.vstack/hstack of 1-vectors, but the simpler path is to declare
    # an auxiliary symmetric Variable and impose y-equality.
    M2 = cp.Variable((nB2, nB2), symmetric=True)
    for a in range(nB2):
        for b in range(a, nB2):
            ab = _add_alpha(basis2[a], basis2[b])
            cons.append(M2[a, b] == y[idx[ab]])
    cons.append(M2 >> 0)

    # (iii)+(iv) Localizing matrices for (h - delta_i) and (h + delta_i),
    # of degree 1.  L_1((h ± delta_i) y)[a, b] = h*y_{a+b} ± y_{a+b+e_i}.
    # PSD on B_1 basis (size 1+d).
    if enforce_box_localizers:
        basis1 = _make_basis1(d)
        nB1 = len(basis1)
        for i in range(d):
            ei = _e(d, i)
            # (h - delta_i)
            L_minus = cp.Variable((nB1, nB1), symmetric=True)
            for a in range(nB1):
                for b in range(a, nB1):
                    ab = _add_alpha(basis1[a], basis1[b])
                    abi = _add_alpha(ab, ei)
                    cons.append(L_minus[a, b] == h * y[idx[ab]] - y[idx[abi]])
            cons.append(L_minus >> 0)
            # (h + delta_i)
            L_plus = cp.Variable((nB1, nB1), symmetric=True)
            for a in range(nB1):
                for b in range(a, nB1):
                    ab = _add_alpha(basis1[a], basis1[b])
                    abi = _add_alpha(ab, ei)
                    cons.append(L_plus[a, b] == h * y[idx[ab]] + y[idx[abi]])
            cons.append(L_plus >> 0)

    # (v) sum_i y_{e_i} = 0
    cons.append(sum(y[idx[_e(d, i)]] for i in range(d)) == 0.0)

    # (vi) sum_i y_{e_i + e_j} = 0 for all j  (lifted sum=0 by delta_j)
    for j in range(d):
        ej = _e(d, j)
        cons.append(sum(y[idx[_add_alpha(_e(d, i), ej)]] for i in range(d)) == 0.0)

    # (vii) sum_i y_{e_i + e_j + e_k} = 0 for j <= k  (lifted by delta_j delta_k)
    # All triples (j, k) with j <= k.
    for j in range(d):
        for k in range(j, d):
            ejk = _add_alpha(_e(d, j), _e(d, k))
            cons.append(sum(y[idx[_add_alpha(_e(d, i), ejk)]] for i in range(d)) == 0.0)

    # Optional: tighten diagonal of M_2 with redundant box constraints.
    # delta_i^2 <= h^2 in expectation: y_{2 e_i} <= h^2.  And >= 0 (since square).
    # These follow from box localizer M_1((h-d_i)*(h+d_i) ~ ...) but adding them
    # explicitly is harmless and helps numerically.
    if enforce_diag_box:
        for i in range(d):
            cons.append(y[idx[_e(d, i, 2)]] <= h * h)
            cons.append(y[idx[_e(d, i, 2)]] >= 0.0)

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


def cell_cert_lasserre2_max(c_int: np.ndarray, S: int, d: int, c_target: float,
                              windows: Sequence[Tuple[int, int]],
                              solver: str = 'auto',
                              tol: float = 1e-9,
                              verbose: bool = False) -> Tuple[float, Tuple[int, int]]:
    """Best (= largest) single-window L2 LB.  Sound LB on min_delta max_W TV_W."""
    best_lb = float('-inf')
    best_W = (-1, -1)
    for W in windows:
        lb, status = cell_cert_lasserre2(c_int, S, d, c_target, W,
                                          solver=solver, tol=tol, verbose=verbose)
        if lb > best_lb:
            best_lb = lb
            best_W = W
    return best_lb, best_W


# =====================================================================
# Driver: compare Shor (order-1) vs Lasserre order-2 on hard cells
# =====================================================================

def run_compare(d: int, S: int, c_target: float, max_cells: int = 30,
                solver: str = 'auto', tol: float = 1e-9,
                verbose: bool = True):
    """Run both Shor and Lasserre-2 on the K hardest triangle-failing cells
    (best-only window mode: use the W* that maximized triangle margin).
    """
    print(f"\n=== _coarse_L2_bench: d={d}, S={S}, c_target={c_target} ===")
    print(f"    cell width h = 1/(2S) = {1.0 / (2.0 * S):.6f}")
    print(f"    solver: {solver}")

    hard, n_grid_pass, n_tri_cert, n_total = find_hard_cells(d, S, c_target)
    print(f"    grid-point passers  : {n_grid_pass:,}")
    print(f"    triangle certified  : {n_tri_cert:,}")
    print(f"    HARD cells (failing): {len(hard):,}")
    if not hard:
        print("    No hard cells — triangle certifies everything.")
        return {'d': d, 'S': S, 'c_target': c_target, 'n_hard': 0}

    hard.sort(key=lambda kv: -kv[1]['net'])
    hard = hard[:max_cells]

    print(f"    Running Shor + L2 SDP on {len(hard)} hardest cells.\n")
    print(f"    triangle net range: [{hard[0][1]['net']:+.6f}, "
          f"{hard[-1][1]['net']:+.6f}]")

    n_shor_cert = 0
    n_l2_cert = 0
    n_l2_strict_better = 0
    n_l2_violation = 0
    max_l2_violation = 0.0
    times_shor = []
    times_l2 = []
    rows = []

    n_truly_hard = 0  # triangle fails AND Shor fails
    n_l2_rescue = 0   # triangle fails AND Shor fails BUT L2 certifies

    for k, (c, tri) in enumerate(hard):
        # Shor (best-only on triangle's W*)
        t0 = time.time()
        shor_lb, shor_status = cell_cert_shor(c, S, d, c_target, tri['W'],
                                                solver=solver, tol=tol)
        dt_shor = time.time() - t0

        # L2 on the same window
        t1 = time.time()
        l2_lb, l2_status = cell_cert_lasserre2(c, S, d, c_target, tri['W'],
                                                 solver=solver, tol=tol)
        dt_l2 = time.time() - t1

        # Soundness check: L2_LB >= Shor_LB - tol
        if l2_lb < shor_lb - 1e-6 and shor_lb > float('-inf') and l2_lb > float('-inf'):
            n_l2_violation += 1
            max_l2_violation = max(max_l2_violation, shor_lb - l2_lb)

        shor_cert = shor_lb >= c_target - 1e-9
        l2_cert = l2_lb >= c_target - 1e-9

        if shor_cert:
            n_shor_cert += 1
        if l2_cert:
            n_l2_cert += 1

        if l2_lb > shor_lb + 1e-9:
            n_l2_strict_better += 1

        # Truly hard: triangle fails AND Shor fails
        if not shor_cert:
            n_truly_hard += 1
            if l2_cert:
                n_l2_rescue += 1

        times_shor.append(dt_shor)
        times_l2.append(dt_l2)

        rows.append({
            'k': k, 'c': c.tolist(),
            'tri_net': float(tri['net']),
            'tri_W': list(tri['W']),
            'tri_tv': float(tri['tv']),
            'shor_lb': float(shor_lb) if shor_lb != float('-inf') else None,
            'l2_lb': float(l2_lb) if l2_lb != float('-inf') else None,
            'shor_cert': bool(shor_cert),
            'l2_cert': bool(l2_cert),
            'l2_strict_better': l2_lb > shor_lb + 1e-9,
            'gap_l2_minus_shor': float(l2_lb - shor_lb) if (
                l2_lb != float('-inf') and shor_lb != float('-inf')) else None,
            'time_shor_s': float(dt_shor),
            'time_l2_s': float(dt_l2),
            'shor_status': shor_status,
            'l2_status': l2_status,
        })

        if verbose and k < 8:
            gap = l2_lb - shor_lb if (l2_lb != float('-inf') and shor_lb != float('-inf')) else float('nan')
            print(f"    [{k:3d}] c={c.tolist()}  tri_net={tri['net']:+.5f}  "
                  f"shor={shor_lb:.5f}({'C' if shor_cert else 'f'}) "
                  f"L2={l2_lb:.5f}({'C' if l2_cert else 'f'}) "
                  f"gap=+{gap:.2e}  T_shor={dt_shor*1000:.0f}ms  T_L2={dt_l2*1000:.0f}ms")

    times_shor = np.asarray(times_shor)
    times_l2 = np.asarray(times_l2)

    pct = lambda x: 100.0 * x / max(1, len(hard))
    print(f"\n    --- Summary ---")
    print(f"    Hard cells tested      : {len(hard)}")
    print(f"    Shor certified         : {n_shor_cert:>4}  ({pct(n_shor_cert):.1f}%)")
    print(f"    Lasserre-2 certified   : {n_l2_cert:>4}  ({pct(n_l2_cert):.1f}%)")
    print(f"    L2 strictly > Shor LB  : {n_l2_strict_better}")
    print(f"    Soundness violations   : {n_l2_violation}  (max diff: {max_l2_violation:.2e})")
    print(f"    Truly hard (Shor fail) : {n_truly_hard}")
    print(f"    L2 rescues (Shor fail->L2 cert): {n_l2_rescue}")
    if len(times_shor):
        print(f"    Shor time/cell (ms)    : "
              f"med={1000*np.median(times_shor):.1f}  "
              f"p95={1000*np.percentile(times_shor,95):.1f}  "
              f"max={1000*np.max(times_shor):.1f}")
        print(f"    L2  time/cell (ms)    : "
              f"med={1000*np.median(times_l2):.1f}  "
              f"p95={1000*np.percentile(times_l2,95):.1f}  "
              f"max={1000*np.max(times_l2):.1f}")
        print(f"    Time ratio (L2/Shor)   : "
              f"med={np.median(times_l2)/max(1e-9,np.median(times_shor)):.2f}x")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_hard_total': len(hard),
        'n_hard_tested': len(hard),
        'n_shor_cert': n_shor_cert,
        'n_l2_cert': n_l2_cert,
        'n_l2_strict_better': n_l2_strict_better,
        'n_truly_hard': n_truly_hard,
        'n_l2_rescue': n_l2_rescue,
        'soundness_viol': n_l2_violation,
        'max_violation': max_l2_violation,
        'time_shor_med_ms': float(1000 * np.median(times_shor)) if len(times_shor) else None,
        'time_l2_med_ms': float(1000 * np.median(times_l2)) if len(times_l2) else None,
        'time_shor_p95_ms': float(1000 * np.percentile(times_shor, 95)) if len(times_shor) else None,
        'time_l2_p95_ms': float(1000 * np.percentile(times_l2, 95)) if len(times_l2) else None,
        'rows': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=None)
    ap.add_argument('--S', type=int, default=None)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--max_cells', type=int, default=20)
    ap.add_argument('--solver', default='auto')
    ap.add_argument('--out', default='_coarse_L2_results.json')
    args = ap.parse_args()

    if not _HAS_CVXPY:
        print("ERROR: cvxpy not available — cannot run SDP.")
        sys.exit(1)

    results = []
    if args.d is None:
        # Default sweep — same hard-cell setup as _coarse_L_bench.py
        for d, S, c in [(4, 20, 1.20), (6, 15, 1.20), (8, 12, 1.20)]:
            r = run_compare(d, S, c, max_cells=args.max_cells, solver=args.solver)
            results.append(r)
    else:
        r = run_compare(args.d, args.S, args.c_target,
                         max_cells=args.max_cells, solver=args.solver)
        results.append(r)

    # Strip 'rows' to top 30 for JSON brevity
    out_results = []
    for r in results:
        rcopy = dict(r)
        if 'rows' in rcopy and len(rcopy['rows']) > 30:
            rcopy['rows'] = rcopy['rows'][:30]
        out_results.append(rcopy)
    with open(args.out, 'w') as fp:
        json.dump(out_results, fp, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
