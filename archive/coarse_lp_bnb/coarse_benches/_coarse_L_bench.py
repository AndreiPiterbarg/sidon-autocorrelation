"""Coarse-grid L-bench: per-cell Shor SDP for the COARSE-GRID cascade.

Goal
====
Push the box certification of `cloninger-steinerberger/cpu/run_cascade_coarse_v2.py`
beyond the triangle-inequality bound (`margin > cell_var + quad_corr`).
The triangle bound is what's currently used; for hard cells (margin tiny but
positive) we need a tighter cert.  This file implements a per-cell Shor SDP.

Mathematical setup
==================
Coarse grid: integer composition `c` with `sum c = S`, physical mass
`mu* = c/S` satisfying `mu*_i = c_i/S`.  The cell at `c` is

    Cell(c) = { mu = c/S + delta : |delta_i| <= h, sum delta = 0 }

with `h = 1/(2S)`.  For each window W = (ell, s_lo) we have the EXACT
quadratic identity (since TV is a quadratic form in mu):

    TV_W(c/S + delta) = TV_W(c/S) + grad_W . delta + (2d/ell) * delta^T A_W delta

where
    A_W[i,j] = 1_{s_lo <= i+j <= s_lo+ell-2}      (symmetric d x d)
    grad_W,i = (4d/ell) * sum_{j : (i,j) in W} mu*_j
             = (4d/(ell*S)) * sum_{j : s_lo <= i+j <= s_lo+ell-2} c_j

Cell-cert objective for one window
----------------------------------
    SDP_LB(W) = TV_W(c/S) + min_{delta in Cell} { grad_W . delta + (2d/ell) delta^T A_W delta }

Joint cert (max over windows)
-----------------------------
Sound max-min weak duality:

    min_{delta in Cell} max_W TV_W(c/S + delta) >= max_W SDP_LB(W).

So the cell is certified iff `max_W SDP_LB(W) >= c_target`.
We additionally implement an outer dual-averaging variant:

    min_{delta in Cell} max_W TV_W(c+delta) >= max_{lambda in Delta} SDP_LB_combined(lambda)

where SDP_LB_combined(lambda) = sum_W lambda_W * TV_W(c/S) + min_delta sum_W lambda_W (...).

Shor SDP relaxation
===================
Lift `Y = [[1, delta^T], [delta, D]] >> 0` with `D = delta delta^T` linearized.
Constraints (all sound relaxations):
    Y >> 0
    -h <= delta_i <= h
    sum_i delta_i = 0
    D[i,i] <= h^2
    D[i,i] >= 0
    RLT: from (h ± delta_i)(h ± delta_j) >= 0 (4 sign patterns)
         => h^2 ± h(delta_i + delta_j) + D[i,j] >= 0
            h^2 ± h(delta_i - delta_j) - D[i,j] >= 0

Objective: minimize  grad . delta + (2d/ell) * trace(A_W * D).
SDP optimum is a SOUND LOWER BOUND on the true QP min (PSD relaxation
contains the true feasible set).

Caveat: A_W is NOT PSD in general (counterexample d=2, ell=2: eigenvals ±1).
The Shor bound is loose in proportion to the negative spectrum of A_W;
it is NEVER unsound.

Soundness check (vs. vertex enumeration)
========================================
The vertex enumeration over {±h}^d ∩ {sum=0} gives an UPPER BOUND on the
true min (it's the max over vertices of a non-convex QP, and we are taking
the min via vertex enumeration of the unique balanced extremes — no, vertex
enum gives max only if the QP is convex; for our case we directly evaluate
the QP at all extreme points and take the MIN, which is a sound UPPER bound
on the actual min over the cell, since the cell is the convex hull of those
vertices and min of a quadratic over a polytope occurs at a vertex IF the
quadratic is convex... but A_W is indefinite, so the min could occur in
the interior or on a face.  Vertex-enum gives the CELL VERTEX MIN — a sound
upper bound on the global min only when restricted to vertices.  We use it
for relative comparison.).

Empirical sanity: SDP_LB <= vertex-enum value.

API
===
- `cell_cert_shor(c_int, S, d, c_target, window, ...)`
    SDP LB for one window.
- `cell_cert_shor_max(c_int, S, d, c_target, windows, ...)`
    Best SDP LB over a list of windows.
- `cell_cert_shor_combined(c_int, S, d, c_target, windows, lambdas, ...)`
    SDP LB for a fixed convex combination of windows.

Tests
=====
Hardest cells of `(d=4, S=20, c=1.20)` and `(d=6, S=15, c=1.20)`.
"""
from __future__ import annotations
import os, sys, time, json, argparse, math
from itertools import product
from typing import List, Optional, Sequence, Tuple
import numpy as np

# Pull cvxpy lazily — file should still import without cvxpy for analysis
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


# =====================================================================
# Window enumeration / matrices (coarse-grid version)
# =====================================================================

def all_windows(d: int) -> List[Tuple[int, int]]:
    """All windows (ell, s_lo) with ell in [2, 2d], s_lo in [0, conv_len-(ell-1)]."""
    conv_len = 2 * d - 1
    out = []
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            out.append((ell, s_lo))
    return out


def build_A_matrix(d: int, ell: int, s_lo: int) -> np.ndarray:
    """A_W[i,j] = 1 iff s_lo <= i+j <= s_lo + ell - 2."""
    A = np.zeros((d, d), dtype=np.float64)
    s_hi = s_lo + ell - 2
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                A[i, j] = 1.0
    return A


def tv_at(c: np.ndarray, S: int, d: int, ell: int, s_lo: int) -> float:
    """TV_W(c/S) = (2d/ell) * (c/S)^T A_W (c/S)."""
    A = build_A_matrix(d, ell, s_lo)
    mu = c.astype(np.float64) / float(S)
    return (2.0 * d / ell) * float(mu @ A @ mu)


def grad_at(c: np.ndarray, S: int, d: int, ell: int, s_lo: int) -> np.ndarray:
    """grad TV_W at mu* = c/S.  grad_i = (4d/(ell*S)) * sum_{j: i+j in W} c_j."""
    A = build_A_matrix(d, ell, s_lo)
    g = (4.0 * d / (ell * S)) * (A @ c.astype(np.float64))
    return g


# =====================================================================
# Vertex enumeration (sanity reference UPPER bound on the min)
# =====================================================================

def cell_vertices(d: int, h: float) -> np.ndarray:
    """Enumerate {±h}^d ∩ {sum = 0}.  Only feasible when d even."""
    if d % 2 != 0:
        return np.empty((0, d), dtype=np.float64)
    from itertools import combinations
    out = []
    for chosen in combinations(range(d), d // 2):
        v = np.full(d, h, dtype=np.float64)
        for k in chosen:
            v[k] = -h
        out.append(v)
    return np.asarray(out, dtype=np.float64)


def qp_min_vertex_eval(c: np.ndarray, S: int, d: int, ell: int, s_lo: int,
                       h: float) -> float:
    """Evaluate min_v over balanced ±h vertices of grad.v + (2d/ell)*v^T A v.
    A sound UPPER BOUND on the min over the cell (cell vertices are inside)
    only useful as a sanity check upper-bounding the SDP LB.
    """
    A = build_A_matrix(d, ell, s_lo)
    g = grad_at(c, S, d, ell, s_lo)
    verts = cell_vertices(d, h)
    if len(verts) == 0:
        return float('nan')
    scale = 2.0 * d / ell
    best = float('inf')
    for v in verts:
        val = float(g @ v + scale * v @ A @ v)
        if val < best:
            best = val
    return best


# =====================================================================
# Shor SDP LB (single window)
# =====================================================================

def cell_cert_shor(c_int: np.ndarray, S: int, d: int, c_target: float,
                   window: Tuple[int, int],
                   solver: str = 'auto',
                   tol: float = 1e-9,
                   verbose: bool = False) -> Tuple[float, str]:
    """Shor-relaxed SDP lower bound on  min_{delta in Cell} TV_W(c/S + delta).

    Returns (lb, status).  `lb` is the SDP optimum + TV_W(c/S) (so it's
    directly comparable with c_target).  Status is the cvxpy solver status.

    Sound: PSD relaxation contains the true feasible set, so the SDP optimum
    LOWER-BOUNDS the QP minimum.  Adding the constant TV_W(c/S) keeps it sound.
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

    # Variables
    delta = cp.Variable(d)
    D = cp.Variable((d, d), symmetric=True)

    # Moment matrix Y = [[1, delta^T], [delta, D]]
    one11 = np.ones((1, 1))
    Y = cp.bmat([[one11, cp.reshape(delta, (1, d), order='C')],
                 [cp.reshape(delta, (d, 1), order='C'), D]])

    cons = [Y >> 0]
    cons += [delta >= -h, delta <= h]
    cons += [cp.sum(delta) == 0]

    # D[i,i] in [0, h^2] (delta_i^2 <= h^2 and >= 0)
    for i in range(d):
        cons += [D[i, i] >= 0]
        cons += [D[i, i] <= h * h]
        # McCormick scalar: D[i,i] >= (delta_i)^2 is non-convex; we keep
        # the box and the PSD lift handles tightness.
        # Also: D[i,i] >= 2*lo*delta_i - lo^2 with lo = -h => D[i,i] >= -2h delta_i - h^2
        cons += [D[i, i] >= -2 * h * delta[i] - h * h]
        cons += [D[i, i] >= 2 * h * delta[i] - h * h]

    # RLT for off-diagonals (symmetric)
    for i in range(d):
        for j in range(i + 1, d):
            # (h - di)(h - dj) >= 0
            cons += [h * h - h * (delta[i] + delta[j]) + D[i, j] >= 0]
            # (h + di)(h + dj) >= 0
            cons += [h * h + h * (delta[i] + delta[j]) + D[i, j] >= 0]
            # (h - di)(h + dj) >= 0  =>  h^2 + h(dj - di) - D[i,j] >= 0
            cons += [h * h + h * (delta[j] - delta[i]) - D[i, j] >= 0]
            # (h + di)(h - dj) >= 0  =>  h^2 + h(di - dj) - D[i,j] >= 0
            cons += [h * h + h * (delta[i] - delta[j]) - D[i, j] >= 0]

    # Objective: minimize grad.delta + scale * trace(A * D)
    obj = g @ delta + scale * cp.trace(A @ D)
    prob = cp.Problem(cp.Minimize(obj), cons)

    # Solver
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


# =====================================================================
# Multi-window Shor LB: max_W SDP_LB(W)
# =====================================================================

def cell_cert_shor_max(c_int: np.ndarray, S: int, d: int, c_target: float,
                       windows: Sequence[Tuple[int, int]],
                       solver: str = 'auto',
                       tol: float = 1e-9,
                       verbose: bool = False) -> Tuple[float, Tuple[int, int]]:
    """Best (= largest) single-window SDP LB.  Sound LB on min_delta max_W TV_W."""
    best_lb = float('-inf')
    best_W = (-1, -1)
    for W in windows:
        lb, status = cell_cert_shor(c_int, S, d, c_target, W,
                                     solver=solver, tol=tol, verbose=verbose)
        if lb > best_lb:
            best_lb = lb
            best_W = W
    return best_lb, best_W


# =====================================================================
# Multi-window Shor LB with dual averaging (single SDP, fixed lambda)
# =====================================================================

def cell_cert_shor_combined(c_int: np.ndarray, S: int, d: int, c_target: float,
                            windows: Sequence[Tuple[int, int]],
                            lambdas: np.ndarray,
                            solver: str = 'auto',
                            tol: float = 1e-9,
                            verbose: bool = False) -> Tuple[float, str]:
    """SDP LB on min_delta sum_W lambda_W TV_W(c/S + delta).

    For nonneg lambda summing to 1, this is a SOUND lower bound on
    min_delta max_W TV_W(c/S + delta) (since max >= convex comb).
    """
    if not _HAS_CVXPY:
        return float('-inf'), 'NO_CVXPY'

    c = np.asarray(c_int, dtype=np.float64)
    h = 1.0 / (2.0 * S)

    delta = cp.Variable(d)
    D = cp.Variable((d, d), symmetric=True)
    one11 = np.ones((1, 1))
    Y = cp.bmat([[one11, cp.reshape(delta, (1, d), order='C')],
                 [cp.reshape(delta, (d, 1), order='C'), D]])

    cons = [Y >> 0]
    cons += [delta >= -h, delta <= h]
    cons += [cp.sum(delta) == 0]
    for i in range(d):
        cons += [D[i, i] >= 0, D[i, i] <= h * h]
        cons += [D[i, i] >= -2 * h * delta[i] - h * h]
        cons += [D[i, i] >= 2 * h * delta[i] - h * h]
    for i in range(d):
        for j in range(i + 1, d):
            cons += [h * h - h * (delta[i] + delta[j]) + D[i, j] >= 0]
            cons += [h * h + h * (delta[i] + delta[j]) + D[i, j] >= 0]
            cons += [h * h + h * (delta[j] - delta[i]) - D[i, j] >= 0]
            cons += [h * h + h * (delta[i] - delta[j]) - D[i, j] >= 0]

    # Combined: sum_W lambda_W * [tv0_W + g_W . delta + scale_W * tr(A_W D)]
    A_combo = np.zeros((d, d), dtype=np.float64)
    g_combo = np.zeros(d, dtype=np.float64)
    tv_const = 0.0
    for w_idx, (ell, s_lo) in enumerate(windows):
        lam = float(lambdas[w_idx])
        if lam == 0.0:
            continue
        A = build_A_matrix(d, ell, s_lo)
        g = grad_at(c, S, d, ell, s_lo)
        tv0 = tv_at(c, S, d, ell, s_lo)
        scale = 2.0 * d / ell
        A_combo += lam * scale * A
        g_combo += lam * g
        tv_const += lam * tv0

    obj = g_combo @ delta + cp.trace(A_combo @ D)
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
    if prob.value is None or prob.status not in ('optimal', 'optimal_inaccurate'):
        return float('-inf'), prob.status
    return tv_const + float(prob.value), prob.status


# =====================================================================
# Triangle baseline reference (mirrors run_cascade_coarse_v2)
# =====================================================================

def _build_pair_prefix(d):
    conv_len = 2 * d - 1
    prefix_nk = np.zeros(conv_len + 1, dtype=np.int64)
    prefix_mk = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        nk = min(k + 1, d, conv_len - k)
        mk = 1 if (k % 2 == 0 and k // 2 < d) else 0
        prefix_nk[k + 1] = prefix_nk[k] + nk
        prefix_mk[k + 1] = prefix_mk[k] + mk
    return prefix_nk, prefix_mk


def _quad_corr_for_window(s_lo, ell, d, prefix_nk, prefix_mk, S):
    inv_4S2 = 1.0 / (4.0 * S * S)
    hi = s_lo + ell - 1
    N_W = int(prefix_nk[hi] - prefix_nk[s_lo])
    M_W = int(prefix_mk[hi] - prefix_mk[s_lo])
    cross_W = N_W - M_W
    compl_bound = d * d - N_W
    pair_bound = min(cross_W, compl_bound)
    if pair_bound <= 0:
        return 0.0
    return (2.0 * d / ell) * pair_bound * inv_4S2


def cell_var_for_window(c, S, d, ell, s_lo):
    """First-order cell variation max_{|delta|<=h, sum=0} |grad . delta|.

    Closed-form: pair extremes.  cell_var = (1/(2S)) * sum_{k<d/2}
    (g_sorted[d-1-k] - g_sorted[k]).
    """
    g = grad_at(c, S, d, ell, s_lo)
    g_sorted = np.sort(g)
    h = 1.0 / (2.0 * S)
    cell_var = 0.0
    for k in range(d // 2):
        cell_var += g_sorted[d - 1 - k] - g_sorted[k]
    return h * cell_var


def triangle_cert(c, S, d, c_target):
    """Return (best_W, tv_at_W, margin, cell_var, quad_corr, net) for the W
    that maximizes net = TV - c - cell_var - quad_corr.

    `c` certified by triangle iff net > 0.
    """
    prefix_nk, prefix_mk = _build_pair_prefix(d)
    best = None
    for ell, s_lo in all_windows(d):
        tv = tv_at(c, S, d, ell, s_lo)
        if tv <= c_target:
            continue
        margin = tv - c_target
        cv = cell_var_for_window(c, S, d, ell, s_lo)
        qc = _quad_corr_for_window(s_lo, ell, d, prefix_nk, prefix_mk, S)
        net = margin - cv - qc
        if best is None or net > best['net']:
            best = dict(W=(ell, s_lo), tv=tv, margin=margin,
                        cell_var=cv, quad_corr=qc, net=net)
    return best


# =====================================================================
# Diagnostic: enumerate compositions and find triangle-failing cells
# =====================================================================

def enum_compositions(d: int, S: int):
    """Yield all weak compositions of S into d parts."""
    c = np.zeros(d, dtype=np.int32)
    c[0] = S
    yield c.copy()
    while True:
        # Standard "next composition" (lex order over balanced)
        if c[-1] == S:
            return
        # Find rightmost nonzero before last
        i = d - 2
        while i >= 0 and c[i] == 0:
            i -= 1
        if i < 0:
            return
        c[i] -= 1
        carry = c[d - 1] + 1
        c[d - 1] = 0
        c[i + 1] = carry
        yield c.copy()


def find_hard_cells(d: int, S: int, c_target: float, max_eval: int = 10**7):
    """Return list of compositions c with:
        - exists W with TV_W(c) > c_target  (passes Theorem 1 prune)
        - triangle net <= 0                 (NOT certified by triangle)
    These are the "hard cells" we want the SDP to certify.
    """
    hard = []
    grid_passers = 0
    grid_pruned_triangle = 0
    n = 0
    for c in enum_compositions(d, S):
        n += 1
        if n > max_eval:
            break
        tri = triangle_cert(c, S, d, c_target)
        if tri is None:
            # No window beats c_target at the grid point => doesn't even pass
            # Theorem 1.  Skip (this is a "survivor" not a hard prunee).
            continue
        grid_passers += 1
        if tri['net'] > 0:
            grid_pruned_triangle += 1
        else:
            hard.append((c.copy(), tri))
    return hard, grid_passers, grid_pruned_triangle, n


# =====================================================================
# Driver: run on (d=4, S=20, c=1.20) and (d=6, S=15, c=1.20)
# =====================================================================

def run_test(d: int, S: int, c_target: float, max_cells: int = 50,
             solver: str = 'auto', tol: float = 1e-9,
             multi_window_mode: str = 'max',
             verbose: bool = True):
    """Run the SDP on the K hardest triangle-failing cells.

    multi_window_mode:
      'max' — try max_W SDP_LB(W) (multiple SDPs)
      'best_only' — only the W that maximized triangle margin (1 SDP)
      'combined' — single SDP with lambda equal to 1/k uniform
    """
    print(f"\n=== _coarse_L_bench: d={d}, S={S}, c_target={c_target} ===")
    print(f"    cell width h = 1/(2S) = {1.0 / (2.0 * S):.6f}")
    print(f"    multi-window mode: {multi_window_mode}")

    hard, n_grid_pass, n_tri_cert, n_total = find_hard_cells(d, S, c_target)
    print(f"    grid-point passers  : {n_grid_pass:,}  (TV>c_target at grid)")
    print(f"    triangle certified  : {n_tri_cert:,}")
    print(f"    HARD cells (failing): {len(hard):,}")
    if not hard:
        print("    No hard cells — triangle certifies everything.")
        return {'d': d, 'S': S, 'c_target': c_target, 'n_hard': 0,
                'n_grid_pass': n_grid_pass}

    # Sort by triangle net DESCENDING (closest-to-cert first = least negative net).
    # These are the cells MOST LIKELY savable by a tighter cert.
    # The motivation: triangle is loose (cell_var + quad_corr is sum of two
    # one-sided bounds; the actual joint deviation is smaller).  Cells with
    # tri_net just below 0 are exactly where SDP slack helps most.
    hard.sort(key=lambda kv: -kv[1]['net'])
    hard = hard[:max_cells]

    print(f"    Running SDP on {len(hard)} hardest cells.\n")
    print(f"    triangle net range: [{hard[0][1]['net']:.6f}, "
          f"{hard[-1][1]['net']:.6f}]")

    n_sdp_cert = 0
    n_sdp_loose = 0
    n_sdp_fail = 0
    times = []
    sound_violations = 0
    sound_max_violation = 0.0
    rows = []

    for k, (c, tri) in enumerate(hard):
        t0 = time.time()
        if multi_window_mode == 'best_only':
            lb, status = cell_cert_shor(c, S, d, c_target, tri['W'],
                                         solver=solver, tol=tol)
            tested_W = (tri['W'],)
            best_W = tri['W']
        elif multi_window_mode == 'combined':
            # Use uniform lambda over the windows that pass theorem 1
            ws = [W for W in all_windows(d) if tv_at(c, S, d, *W) > c_target]
            lambdas = np.full(len(ws), 1.0 / max(1, len(ws)))
            lb, status = cell_cert_shor_combined(c, S, d, c_target, ws,
                                                   lambdas, solver=solver,
                                                   tol=tol)
            best_W = ws[0] if ws else (-1, -1)
            tested_W = tuple(ws)
        else:  # 'max'
            ws = [W for W in all_windows(d) if tv_at(c, S, d, *W) > c_target]
            lb = float('-inf')
            best_W = (-1, -1)
            for W in ws:
                lb_w, _ = cell_cert_shor(c, S, d, c_target, W,
                                          solver=solver, tol=tol)
                if lb_w > lb:
                    lb = lb_w
                    best_W = W
            tested_W = tuple(ws)
            status = 'optimal'
        dt = time.time() - t0

        # Sanity: vertex-enum upper bound on the min for the W* that gave best LB
        v_ub = float('-inf')
        if best_W[0] > 0:
            ell, s_lo = best_W
            v_min = qp_min_vertex_eval(c, S, d, ell, s_lo, 1.0 / (2.0 * S))
            tv0 = tv_at(c, S, d, ell, s_lo)
            v_ub = tv0 + v_min  # this UB on min_delta TV_W(c/S+delta)
            # SDP_LB <= v_ub must hold up to tol.
            if not math.isnan(v_ub) and lb > v_ub + 1e-6:
                sound_violations += 1
                sound_max_violation = max(sound_max_violation, lb - v_ub)

        certified = lb >= c_target - 1e-9
        if certified:
            n_sdp_cert += 1
        elif lb > float('-inf'):
            n_sdp_loose += 1
        else:
            n_sdp_fail += 1
        times.append(dt)

        rows.append({
            'k': k, 'c': c.tolist(),
            'tri_W': list(tri['W']),
            'tri_net': float(tri['net']),
            'tri_tv': float(tri['tv']),
            'sdp_W': list(best_W),
            'sdp_lb': float(lb) if lb != float('-inf') else None,
            'vertex_ub': float(v_ub) if v_ub != float('-inf') and not math.isnan(v_ub) else None,
            'sdp_cert': bool(certified),
            'time_s': float(dt),
            'status': status,
            'n_W_tested': len(tested_W),
        })

        if verbose and k < 6:
            v_str = f"{v_ub:.6f}" if (v_ub != float('-inf') and not math.isnan(v_ub)) else "n/a"
            print(f"    [{k:3d}] c={c.tolist()}  tri_net={tri['net']:+.4f}  "
                  f"sdp_lb={lb:.6f}  vert_ub={v_str}  "
                  f"{'CERT' if certified else 'fail'}  {dt*1000:.1f}ms")

    times = np.asarray(times)
    print(f"\n    --- Summary ---")
    print(f"    Hard cells tested      : {len(hard):,}")
    print(f"    SDP certified          : {n_sdp_cert:,}  ({100*n_sdp_cert/max(1,len(hard)):.1f}%)")
    print(f"    SDP loose (lb<c_target): {n_sdp_loose:,}")
    print(f"    SDP failed (status)    : {n_sdp_fail:,}")
    print(f"    Soundness violations   : {sound_violations}  "
          f"(max diff: {sound_max_violation:.2e})")
    if len(times):
        print(f"    Time per cell (ms)     : "
              f"med={1000*np.median(times):.1f}  "
              f"p95={1000*np.percentile(times,95):.1f}  "
              f"max={1000*np.max(times):.1f}  "
              f"sum={times.sum():.2f}s")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_grid_pass': n_grid_pass,
        'n_tri_cert': n_tri_cert,
        'n_hard_total': len(hard),
        'n_hard_tested': len(hard),
        'n_sdp_cert': n_sdp_cert,
        'n_sdp_loose': n_sdp_loose,
        'n_sdp_fail': n_sdp_fail,
        'sound_violations': sound_violations,
        'sound_max_violation': sound_max_violation,
        'time_med_ms': float(1000 * np.median(times)) if len(times) else None,
        'time_p95_ms': float(1000 * np.percentile(times, 95)) if len(times) else None,
        'time_max_ms': float(1000 * np.max(times)) if len(times) else None,
        'time_total_s': float(times.sum()) if len(times) else 0.0,
        'mode': multi_window_mode,
        'rows': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=None)
    ap.add_argument('--S', type=int, default=None)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--max_cells', type=int, default=30)
    ap.add_argument('--mode', choices=['max', 'best_only', 'combined'],
                     default='best_only')
    ap.add_argument('--solver', default='auto')
    ap.add_argument('--out', default='_coarse_L_results.json')
    args = ap.parse_args()

    if not _HAS_CVXPY:
        print("ERROR: cvxpy not available — cannot run SDP.")
        sys.exit(1)

    results = []
    if args.d is None:
        for d, S in [(4, 20), (6, 15)]:
            r = run_test(d, S, args.c_target, max_cells=args.max_cells,
                          solver=args.solver, multi_window_mode=args.mode)
            results.append(r)
    else:
        r = run_test(args.d, args.S, args.c_target,
                      max_cells=args.max_cells, solver=args.solver,
                      multi_window_mode=args.mode)
        results.append(r)

    # Strip 'rows' for JSON if too big
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
