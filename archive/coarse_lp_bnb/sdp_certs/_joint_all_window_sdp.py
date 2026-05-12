"""Joint multi-window MIN-MAX cell-cert via a SINGLE Shor SDP across ALL
windows.  Strictly tighter than per-window Shor and joint-dual approaches
that sum windows with weights.

================================================================
MATHEMATICAL DERIVATION
================================================================
Goal: certify the cell  Cell(c) = { mu = c/S + delta : |delta|_inf <= h,
                                    sum delta = 0,  mu >= 0 }
                                    h = 1/(2S)
satisfies, for EVERY delta in Cell, max_W TV_W(c/S + delta) >= c_target,
where for each window W = (ell, s_lo) we have the EXACT quadratic identity

    TV_W(c/S + delta) = TV_W(c/S) + grad_W . delta + scale_W * delta^T A_W delta
                      =: T_W + g_W . delta + s_W * delta^T A_W delta

with T_W = TV_W(c/S), g_W = grad_W, s_W = 2d/ell, A_W[i,j] = 1{s<=i+j<=s+ell-2}.

Cert problem (TRUE):
    cert_box(c) := min_{delta in Cell} max_{W in WSet} [T_W + g_W.delta + s_W delta^T A_W delta]
                 >= c_target?

Joint Shor lift
---------------
Introduce Y = [[1, delta^T], [delta, D]] PSD with D = delta delta^T (relaxed
to D - delta delta^T >= 0 implied by Y >> 0).  Then for each W,

    delta^T A_W delta = trace(A_W * D)

and the per-window value becomes a LINEAR function of (delta, D):

    LHS_W(delta, D) := T_W + g_W . delta + s_W * trace(A_W * D)

Min-max over Cell with the lift becomes:

    cert_box(c) >= max  t
                  s.t.  LHS_W(delta, D) >= t                  for each W
                        Y = [[1, delta^T], [delta, D]] >> 0   (Shor)
                        -h <= delta_i <= h
                        sum delta = 0
                        D[i,i] >= 0
                        D[i,i] <= h^2
                        D[i,i] >= 2 h delta_i - h^2
                        D[i,i] >= -2 h delta_i - h^2          (box squared)
                        h^2 - h(d_i + d_j) + D[i,j] >= 0      (RLT)
                        h^2 + h(d_i + d_j) + D[i,j] >= 0
                        h^2 + h(d_j - d_i) - D[i,j] >= 0
                        h^2 + h(d_i - d_j) - D[i,j] >= 0      (i < j)

This is a single SDP (linear objective max t; linear+PSD constraints).
The optimum is t*.  By PSD relaxation soundness:

    cert_box(c) >= t*

so c is certified iff t* >= c_target - tol.

================================================================
WHY THIS BEATS PER-WINDOW SHOR AND JOINT-DUAL
================================================================
Per-window Shor: max_W [min_{delta in Cell-PSD-lift} LHS_W] -- weak duality:
    cert_box >= max_W min_{(delta,D)} LHS_W(delta,D)
              <= max_W min_delta TV_W (the actual per-window cert)

Joint-dual K=ALL: max_lambda in simplex [min_{(delta,D)} sum lambda_W LHS_W]
LP weak duality: cert_box >= max_lambda min ... <= cert_box.

Joint-Shor-min-max (THIS FILE): single SDP enforcing LHS_W >= t for all W
SIMULTANEOUSLY at the SAME (delta, D).  The constraint set is:
    intersection over W of { (delta,D,t) : LHS_W >= t }
which is the same as { LHS_W >= t for all W } -- equivalently
    t <= min_W LHS_W.
So the SDP solves
    max  t  s.t.  t <= min_W LHS_W(delta, D)  AND  Shor box.
           equivalent
    max  min_W LHS_W(delta, D)  s.t.  Shor box.
           equivalent (by the lift)
    max  min_W [TV_W(c/S+delta) -- relaxed by PSD]  s.t.  delta in Cell-lift.

This is a SOUND LOWER BOUND on min_{delta in Cell} max_W TV_W (the cert),
because for ANY feasible delta the lifted value lower-bounds the true value
at each window (Shor); hence the worst-case W's lifted value lower-bounds
the worst-case true value, and we are MAXIMIZING over (delta,D).  Wait --
this is backwards.  Let me redo it.

Actually we want a LB on  min_delta max_W TV_W(c/S+delta).  This is a
saddle problem.  Minimax-Shor:

    min_{(delta,D) Shor-feasible} max_W LHS_W(delta,D)            (X)

Claim: cert_box >= X.

Proof:  Take any delta in Cell.  Set D = delta delta^T (rank-1).  The pair
(delta, D) is Shor-feasible.  For this pair, LHS_W(delta,D) = TV_W(c/S+delta)
exactly.  So max_W LHS_W = max_W TV_W = the true objective at delta.  Now,
the Shor-feasible set is LARGER than the rank-1 set (it contains all
rank-r lifts), so

    min_{(delta,D) Shor-feasible} max_W LHS_W
        <= min_{rank-1 lifts (delta,D)} max_W LHS_W
        =  min_delta max_W TV_W = cert_box.

Hmm, this gives X <= cert_box, NOT X >= cert_box.

Right -- the Shor-relaxed feasible set is LARGER, so the min over it is
SMALLER.  X is a sound LOWER BOUND on cert_box.  Cell certified iff
X >= c_target.

The single SDP solves min_{(delta,D)} max_W LHS_W via the epigraph trick:

    max  t
    s.t. LHS_W(delta,D) >= t for all W
         (delta,D) Shor-feasible

is equivalent to  max_{(delta,D),t} t = max min_W LHS_W ... but we want MIN
over (delta,D) of max_W LHS_W = min max LHS_W.  These are DIFFERENT!

    max_{delta,D} min_W LHS_W   <=   min_W max_{delta,D} LHS_W

Maximin lower-bounds minimax (von Neumann is not given here -- D is constr).
So the SDP computes max-min, which is a LB on the per-window min-max.  But
we want a LB on the Shor-feasible MIN-MAX, which is itself a LB on the true
cert.  Is max-min still a sound LB on the true cert?

Let me re-examine.  We want to LOWER bound

    cert_box = min_{delta in Cell} max_W TV_W(c/S+delta).

The per-window value TV_W(c/S+delta) at any delta in Cell is exactly equal
to LHS_W(delta, delta delta^T).  Restrict to rank-1 lifts:

    cert_box = min_{(delta, D=delta delta^T), delta in Cell} max_W LHS_W(delta, D).

Drop the rank-1 constraint -- larger feasible set -- so the min DECREASES:

    cert_box >= min_{(delta,D) Shor-feasible} max_W LHS_W(delta, D).        (I)

Now (I) is a min-max problem over (delta, D, W).  By minimax inequality:

    min_{(delta,D)} max_W LHS_W >= max_W min_{(delta,D)} LHS_W              (II)

The RHS of (II) is what per-window Shor gives.  So per-window Shor LB is
weaker than (I).  Equality holds in (II) only under saddle conditions
(here usually NOT met because Shor-feasible set + W-set are not jointly
concave/convex saddle structure).

The SDP

    max  t
    s.t. LHS_W >= t  for all W
         (delta, D) Shor-feasible,   t free

computes... let's see.  For fixed (delta,D), the largest t satisfying
LHS_W >= t for all W is t = min_W LHS_W(delta,D).  Maximizing over
(delta,D) gives  max_{(delta,D)} min_W LHS_W(delta,D) = RHS of (II) above
(per-window relax SDP for the ARGMAX_W that ties).

That's NOT what we want!  We need a MIN over (delta,D), not max.

================================================================
THE CORRECT FORMULATION
================================================================

To compute (I) -- which is min_{(delta,D)} max_W LHS_W -- via SDP, use
the EPIGRAPH form with max as a NEW variable:

    min_{(delta,D), t}  t
    s.t.  t >= LHS_W  for all W
          (delta,D) Shor-feasible

Now t >= max_W LHS_W => t = max_W LHS_W at optimum, and we minimize this
over (delta,D).  The optimum of THIS SDP equals (I) -- a sound LB on
cert_box.

The "max t" formulation in the original task spec is WRONG; it gives a UB
on per-window Shor, not a LB on the joint cert.  We use min t instead.

================================================================
FINAL SDP (correct sign)
================================================================

    min  t
    s.t.  t >= T_W + g_W . delta + s_W * trace(A_W * D)        for each W
                    [t epigraph of max_W LHS_W]
          Y = [[1, delta^T], [delta, D]] >> 0
          -h <= delta_i <= h
          sum delta = 0
          D[i,i] in [0, h^2]
          D[i,i] >= 2 h delta_i - h^2
          D[i,i] >= -2 h delta_i - h^2
          h^2 +/- h(delta_i +/- delta_j) +/- D[i,j] >= 0  (4 RLT cuts per i<j)

Optimal t* = min_{(delta,D)} max_W LHS_W -- a sound LB on cert_box(c).
Cell certified iff t* >= c_target.

================================================================
SOUNDNESS RECAP
================================================================
1. For any rank-1 (delta, D = delta delta^T) with delta in Cell:
       max_W LHS_W(delta, D) = max_W TV_W(c/S+delta).
2. Shor-feasible set strictly contains rank-1 set => min_{Shor} max_W LHS_W
   <= min_{rank-1} max_W LHS_W = cert_box.
3. The SDP min_{(delta,D) Shor} max_W LHS_W (via epigraph) gives a sound
   LB on cert_box.  No relaxation gap is INTRODUCED (Shor is a sound
   relaxation; epigraph is exact reformulation of max).
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import math
from typing import List, Sequence, Tuple, Optional

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

# Reuse helpers from the existing per-window L bench
from _coarse_L_bench import (  # noqa: E402
    all_windows, build_A_matrix, tv_at, grad_at,
    cell_cert_shor, triangle_cert, find_hard_cells,
)


# =====================================================================
# Joint min-max SDP across ALL windows (single SDP)
# =====================================================================

def joint_all_window_sdp_LB(
    c_int: np.ndarray,
    S: int,
    d: int,
    windows: Sequence[Tuple[int, int]],
    solver: str = 'auto',
    tol: float = 1e-9,
    verbose: bool = False,
) -> Tuple[float, str, float]:
    """Single Shor SDP enforcing LHS_W(delta,D) >= t for all windows W.

    Returns (lb, status, solve_time_s).  `lb` is the optimal t -- a sound
    lower bound on  cert_box(c) := min_{delta in Cell} max_W TV_W(c/S+delta).

    Soundness: see file docstring.  Sketch:
      - For rank-1 lifts (delta, delta delta^T), LHS_W = TV_W.
      - Shor-feasible set contains rank-1 set; min_{Shor} max_W LHS_W
        <= min_{rank-1} max_W LHS_W = cert_box(c).
      - Epigraph form (min t s.t. t >= LHS_W) is exact for max.
    """
    if not _HAS_CVXPY:
        return float('-inf'), 'NO_CVXPY', 0.0

    if len(windows) == 0:
        return float('-inf'), 'NO_WINDOWS', 0.0

    h = 1.0 / (2.0 * S)
    c = np.asarray(c_int, dtype=np.float64)

    # Precompute per-window data
    n_W = len(windows)
    A_list = np.zeros((n_W, d, d), dtype=np.float64)
    g_list = np.zeros((n_W, d), dtype=np.float64)
    tv_list = np.zeros(n_W, dtype=np.float64)
    s_list = np.zeros(n_W, dtype=np.float64)
    for w, (ell, s_lo) in enumerate(windows):
        A_list[w] = build_A_matrix(d, ell, s_lo)
        g_list[w] = grad_at(c, S, d, ell, s_lo)
        tv_list[w] = tv_at(c, S, d, ell, s_lo)
        s_list[w] = 2.0 * d / ell

    # Variables
    delta = cp.Variable(d)
    D = cp.Variable((d, d), symmetric=True)
    t = cp.Variable()

    one11 = np.ones((1, 1))
    Y = cp.bmat([
        [one11, cp.reshape(delta, (1, d), order='C')],
        [cp.reshape(delta, (d, 1), order='C'), D],
    ])

    cons = [Y >> 0]
    cons += [delta >= -h, delta <= h]
    cons += [cp.sum(delta) == 0]
    for i in range(d):
        cons += [D[i, i] >= 0, D[i, i] <= h * h]
        cons += [D[i, i] >= 2 * h * delta[i] - h * h]
        cons += [D[i, i] >= -2 * h * delta[i] - h * h]
    for i in range(d):
        for j in range(i + 1, d):
            cons += [h * h - h * (delta[i] + delta[j]) + D[i, j] >= 0]
            cons += [h * h + h * (delta[i] + delta[j]) + D[i, j] >= 0]
            cons += [h * h + h * (delta[j] - delta[i]) - D[i, j] >= 0]
            cons += [h * h + h * (delta[i] - delta[j]) - D[i, j] >= 0]

    # Per-window epigraph constraints for max:  t >= LHS_W  for each W,
    # so that t = max_W LHS_W at optimum.  We then MINIMIZE t over (delta,D).
    for w in range(n_W):
        lhs = tv_list[w] + g_list[w] @ delta + s_list[w] * cp.trace(A_list[w] @ D)
        cons.append(t >= lhs)

    # Objective: min t = min over (delta,D) of max_W LHS_W = sound LB on cert_box.
    prob = cp.Problem(cp.Minimize(t), cons)

    actual_solver = solver
    if solver == 'auto':
        avail = set(cp.installed_solvers())
        for s in ('MOSEK', 'CLARABEL', 'SCS'):
            if s in avail:
                actual_solver = s
                break

    t0 = time.time()
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
        return float('-inf'), f'EXC:{type(e).__name__}', time.time() - t0

    dt = time.time() - t0
    if prob.value is None or prob.status not in ('optimal', 'optimal_inaccurate'):
        return float('-inf'), prob.status, dt
    return float(prob.value), prob.status, dt


# =====================================================================
# Window selection helpers
# =====================================================================

def all_windows_d(d: int) -> List[Tuple[int, int]]:
    """All (ell, s_lo) windows for the cascade at dimension d."""
    return all_windows(d)


def pruning_windows(c_int: np.ndarray, S: int, d: int, c_target: float
                    ) -> List[Tuple[int, int, float]]:
    """List (ell, s_lo, TV_W) for windows where TV_W(c/S) > c_target."""
    out = []
    for ell, s_lo in all_windows(d):
        tv = tv_at(c_int, S, d, ell, s_lo)
        if tv > c_target:
            out.append((ell, s_lo, tv))
    return out


# =====================================================================
# Joint dual K=ALL baseline (LP duality, mirrors _coarse_J_bench)
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


def joint_dual_K_LB(c, S, d, c_target, K=None):
    """Joint dual subgradient ascent on K highest-TV pruning windows.

    K=None means use ALL pruning windows.  Sound: LP weak duality + per-window
    triangle bounds (mirrors _coarse_J_bench).
    """
    h = 1.0 / (2.0 * S)
    prefix_nk, prefix_mk = _build_pair_prefix(d)
    windows = []
    for ell, s_lo in all_windows(d):
        tv = tv_at(c, S, d, ell, s_lo)
        if tv > c_target:
            windows.append((ell, s_lo, tv))
    if not windows:
        return float('-inf')
    windows.sort(key=lambda w: -w[2])
    if K is not None:
        windows = windows[:K]

    n_W = len(windows)
    A_list = np.zeros((n_W, d, d))
    grad_list = np.zeros((n_W, d))
    scale_list = np.zeros(n_W)
    pb_list = np.zeros(n_W)
    tv_list = np.zeros(n_W)
    for w_idx, (ell, s_lo, tv) in enumerate(windows):
        A_list[w_idx] = build_A_matrix(d, ell, s_lo)
        grad_list[w_idx] = grad_at(c, S, d, ell, s_lo)
        scale_list[w_idx] = 2.0 * d / ell
        hi = s_lo + ell - 1
        N_W = int(prefix_nk[hi] - prefix_nk[s_lo])
        M_W = int(prefix_mk[hi] - prefix_mk[s_lo])
        cross_W = N_W - M_W
        pb_list[w_idx] = float(min(cross_W, d * d - N_W))
        tv_list[w_idx] = tv

    def project_simplex(v):
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = -1
        for i in range(n):
            if u[i] - (cssv[i] - 1.0) / (i + 1) > 0:
                rho = i
        if rho < 0:
            return np.ones(n) / n
        theta = (cssv[rho] - 1.0) / (rho + 1)
        return np.maximum(v - theta, 0.0)

    def lb_at(lam):
        G_lam = (lam[:, None] * grad_list).sum(axis=0)
        order = np.argsort(G_lam)
        half = d // 2
        sigma = np.zeros(d)
        sigma[order[:half]] = +1.0
        sigma[order[d - half:]] = -1.0
        ub_lin_v = h * (G_lam[order[d - half:]].sum() - G_lam[order[:half]].sum())
        ub_quad_v = float(np.sum(lam * scale_list * pb_list)) * h * h
        val = float(np.dot(lam, tv_list)) - ub_lin_v - ub_quad_v
        return val, sigma

    lam = np.full(n_W, 1.0 / n_W)
    best_LB, _ = lb_at(lam)
    for it in range(30):
        val, sigma = lb_at(lam)
        if val > best_LB:
            best_LB = val
        sub = (tv_list + h * (grad_list @ sigma)
               - scale_list * pb_list * h * h)
        step = 1.0 / math.sqrt(it + 1)
        lam = project_simplex(lam + step * sub)
    return best_LB


# =====================================================================
# Per-window Shor (best across pruning windows)
# =====================================================================

def per_window_shor_best(c, S, d, c_target, K=None,
                         solver='auto', tol=1e-9):
    """Best Shor SDP LB across (top-K) pruning windows.

    Sound: max over individual sound LBs.
    """
    windows = []
    for ell, s_lo in all_windows(d):
        tv = tv_at(c, S, d, ell, s_lo)
        if tv > c_target:
            windows.append((ell, s_lo, tv))
    if not windows:
        return float('-inf'), (-1, -1)
    windows.sort(key=lambda w: -w[2])
    if K is not None:
        windows = windows[:K]
    best = float('-inf')
    best_W = (-1, -1)
    for ell, s_lo, _tv in windows:
        lb, _ = cell_cert_shor(c, S, d, c_target, (ell, s_lo),
                                solver=solver, tol=tol)
        if lb > best:
            best = lb
            best_W = (ell, s_lo)
    return best, best_W


# =====================================================================
# Soundness check via fine grid
# =====================================================================

def true_min_max_TV_grid(c_int, S, d, windows, n_grid=9):
    """Fine-grid UPPER bound on cert_box(c) by direct sampling Cell.

    Empirical 'true' min-max on a discrete grid -- no LB should exceed this
    (else SDP unsound).
    """
    from itertools import product as _product
    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / float(S)
    A_list = [build_A_matrix(d, ell, s) for ell, s in windows]
    s_list = [2.0 * d / ell for ell, _ in windows]
    if d <= 6:
        ng_use = n_grid
    elif d == 7:
        ng_use = min(n_grid, 5)
    else:
        ng_use = min(n_grid, 4)
    grid_use = np.linspace(-h, h, ng_use)
    best = math.inf
    for tup in _product(grid_use, repeat=d - 1):
        last = -sum(tup)
        if abs(last) > h + 1e-12:
            continue
        delta = np.array(list(tup) + [last])
        mu = mu_star + delta
        if (mu < -1e-12).any():
            continue
        max_tv = -math.inf
        for A_W, s_W in zip(A_list, s_list):
            tv = s_W * float(mu @ A_W @ mu)
            if tv > max_tv:
                max_tv = tv
        if max_tv < best:
            best = max_tv
    return best


# =====================================================================
# Composition enumerator + hard-cell finder
# =====================================================================

def enum_compositions(d: int, S: int):
    c = np.zeros(d, dtype=np.int32)
    c[0] = S
    yield c.copy()
    while True:
        if c[-1] == S:
            return
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


def find_uncertified_after_jdK_pwShor(d, S, c_target,
                                       jd_K_max=None,
                                       max_eval: int = 10**7,
                                       verbose=False):
    """Return list of compositions that:
      (a) are 'hard' (triangle fails)
      (b) joint-dual K=ALL fails
      (c) per-window Shor best fails
    These are the cells the joint-all-window SDP is meant to rescue.
    """
    out = []
    n_total = 0
    n_hard = 0
    n_jd_cert = 0
    n_pw_cert = 0
    for c in enum_compositions(d, S):
        n_total += 1
        if n_total > max_eval:
            break
        tri = triangle_cert(c, S, d, c_target)
        if tri is None:
            continue
        if tri['net'] > 0:
            continue  # triangle certified
        n_hard += 1
        # Joint-dual K=ALL
        jd_lb = joint_dual_K_LB(c, S, d, c_target, K=jd_K_max)
        jd_cert = jd_lb >= c_target - 1e-9
        if jd_cert:
            n_jd_cert += 1
            continue
        # Per-window Shor (top-K pruning windows; K=None = all pruning)
        pw_lb, _ = per_window_shor_best(c, S, d, c_target, K=8)
        pw_cert = pw_lb >= c_target - 1e-9
        if pw_cert:
            n_pw_cert += 1
            continue
        out.append((c.copy(), tri, jd_lb, pw_lb))
    if verbose:
        print(f"  enum {n_total} total comps, {n_hard} hard, "
              f"{n_jd_cert} jd-cert, {n_pw_cert} pw-cert, "
              f"{len(out)} residue.")
    return out, n_total, n_hard, n_jd_cert, n_pw_cert


# =====================================================================
# Bench driver
# =====================================================================

def find_residue_random_sample(d, S, c_target, N_target=50,
                               max_trials=200000, seed=42):
    """Random near-balanced sampling of compositions to gather N_target
    residue cells (hard, jd-K=4 fail, pw-Shor fail).  For when full
    enumeration is too slow (d=4 S=200 has 1.37M comps).
    """
    np.random.seed(seed)
    seen = set()
    residue = []
    n_trial = 0
    n_hard = 0
    n_jd_cert = 0
    n_pw_cert = 0
    while len(residue) < N_target and n_trial < max_trials:
        n_trial += 1
        a = np.random.dirichlet([1] * d) * S
        c = np.maximum(0, np.round(a)).astype(np.int32)
        diff = S - c.sum()
        c[0] += diff
        if c.sum() != S or any(c < 0):
            continue
        key = tuple(c.tolist())
        if key in seen:
            continue
        seen.add(key)
        tri = triangle_cert(c, S, d, c_target)
        if tri is None or tri['net'] > 0:
            continue
        n_hard += 1
        jd_lb = joint_dual_K_LB(c, S, d, c_target, K=4)
        if jd_lb >= c_target - 1e-9:
            n_jd_cert += 1
            continue
        pw_lb, _ = per_window_shor_best(c, S, d, c_target, K=8)
        if pw_lb >= c_target - 1e-9:
            n_pw_cert += 1
            continue
        residue.append((c.copy(), tri, jd_lb, pw_lb))
    return residue, n_trial, n_hard, n_jd_cert, n_pw_cert


def bench_one_config(d: int, S: int, c_target: float,
                     max_cells: int = 50,
                     soundness_n: int = 20,
                     mode: str = 'all_pruning',
                     solver: str = 'auto',
                     sample_mode: str = 'enum',
                     verbose: bool = True):
    """Bench joint-all-window SDP on the residue cells (hard, not jd-cert,
    not pw-Shor-cert).  Compare cert rate, time, soundness vs fine grid.

    mode:
      'all_pruning' -- use all pruning windows in the joint SDP
      'all_d'       -- use ALL d-windows (also non-pruning) -- never less sound
                       and may be tighter via cell-feasibility constraints.
                       But pruning-only is the relevant one (only those drive
                       cert_box to be > c_target at the local point; for non-
                       pruning windows the constraint LHS_W >= t is slack).
    """
    if not _HAS_CVXPY:
        print("ERROR: cvxpy not installed.")
        return None

    print(f"\n=== d={d}, S={S}, c_target={c_target}, mode={mode}, "
          f"sample_mode={sample_mode} ===")
    t0 = time.time()
    if sample_mode == 'random':
        residue, n_total, n_hard, n_jd_cert, n_pw_cert = (
            find_residue_random_sample(d, S, c_target, N_target=max_cells)
        )
        print(f"  random sample: {n_total} trials, {n_hard} hard, "
              f"jd-cert={n_jd_cert}, pw-Shor-cert={n_pw_cert}, "
              f"residue={len(residue)} ({time.time()-t0:.1f}s)")
    else:
        residue, n_total, n_hard, n_jd_cert, n_pw_cert = (
            find_uncertified_after_jdK_pwShor(d, S, c_target, verbose=True)
        )
        t_enum = time.time() - t0
        print(f"  enum {n_total:,} comps, {n_hard} hard, "
              f"jd-cert={n_jd_cert}, pw-Shor-cert={n_pw_cert}, "
              f"residue={len(residue)} ({t_enum:.1f}s)")
    if not residue:
        print("  No residue cells -- joint+pwShor already cert everything.")
        return {
            'd': d, 'S': S, 'c_target': c_target,
            'n_total': n_total, 'n_hard': n_hard,
            'n_jd_cert': n_jd_cert, 'n_pw_cert': n_pw_cert,
            'residue': 0,
            'sdp_extra_cert': 0,
            'sdp_extra_pct': 0.0,
        }

    # Sort residue by closest-to-cert (max of jd_lb, pw_lb)
    residue.sort(key=lambda r: -max(r[2], r[3]))
    residue_use = residue[:max_cells]
    # Sample mode 'random' already capped at max_cells -- this is a no-op then.

    n_sdp_cert = 0
    n_sdp_better_than_pw = 0
    n_sdp_strictly_lower_than_pw = 0
    times = []
    rows = []
    sdp_violations = 0
    sdp_max_excess = 0.0

    print(f"  Running joint-all-window SDP on {len(residue_use)} residue cells...")
    for k, (c, tri, jd_lb, pw_lb) in enumerate(residue_use):
        # Get pruning windows for this composition
        pwins = pruning_windows(c, S, d, c_target)
        if mode == 'all_d':
            ws_in = all_windows(d)
        else:
            ws_in = [(e, s) for (e, s, _t) in pwins]
        if not ws_in:
            continue

        sdp_lb, status, dt = joint_all_window_sdp_LB(
            c, S, d, ws_in, solver=solver)
        sdp_cert = sdp_lb >= c_target - 1e-9
        times.append(dt)

        if sdp_cert:
            n_sdp_cert += 1
        if sdp_lb > pw_lb + 1e-9:
            n_sdp_better_than_pw += 1
        if sdp_lb < pw_lb - 1e-7:
            n_sdp_strictly_lower_than_pw += 1  # would indicate solver/numerical issue

        # Soundness: SDP LB must be <= true min-max (over a fine grid).
        # Grid is a discrete subset of Cell -> grid min-max is sound UB on
        # cert_box, so SDP LB - grid min-max <= 1e-7 (allow solver tol).
        # Skip in large d where grid is intractable.
        if k < soundness_n and d <= 8:
            ng = 9 if d <= 6 else (5 if d == 7 else 4)
            tm = true_min_max_TV_grid(c, S, d, ws_in, n_grid=ng)
            excess = sdp_lb - tm
            if excess > 1e-6:
                sdp_violations += 1
                if excess > sdp_max_excess:
                    sdp_max_excess = excess

        rows.append({
            'c': c.tolist(),
            'tri_net': float(tri['net']),
            'jd_lb': float(jd_lb), 'pw_lb': float(pw_lb),
            'sdp_lb': float(sdp_lb), 'sdp_cert': bool(sdp_cert),
            'n_W': len(ws_in), 'time_s': float(dt),
        })

        if verbose and k < 5:
            print(f"  [{k:3d}] c={c.tolist()} tri_net={tri['net']:+.4f} "
                  f"jd={jd_lb:.4f} pw={pw_lb:.4f} "
                  f"sdp={sdp_lb:.4f}{'C' if sdp_cert else ' '} "
                  f"({len(ws_in)}W, {dt*1000:.0f}ms)")

    times_a = np.asarray(times)
    print(f"\n  --- Joint-all-window SDP results ---")
    print(f"  residue cells tested: {len(rows)}")
    print(f"  SDP certified       : {n_sdp_cert} "
          f"({100*n_sdp_cert/max(1,len(rows)):.1f}%)")
    print(f"  SDP > per-window    : {n_sdp_better_than_pw} "
          f"({100*n_sdp_better_than_pw/max(1,len(rows)):.1f}%)  "
          f"(joint adds bound over per-window Shor)")
    if n_sdp_strictly_lower_than_pw > 0:
        print(f"  WARN: SDP < per-window in {n_sdp_strictly_lower_than_pw} "
              f"cells (numerical/solver?)")
    print(f"  Time per cell (ms)  : "
          f"med={1000*np.median(times_a):.1f}, "
          f"p95={1000*np.percentile(times_a, 95):.1f}, "
          f"max={1000*np.max(times_a):.1f}")
    print(f"  Soundness check     : {sdp_violations} violations, "
          f"max excess={sdp_max_excess:.2e}")

    return {
        'd': d, 'S': S, 'c_target': c_target, 'mode': mode,
        'sample_mode': sample_mode,
        'n_total': n_total, 'n_hard': n_hard,
        'n_jd_cert': n_jd_cert, 'n_pw_cert': n_pw_cert,
        'residue': len(residue),
        'residue_tested': len(rows),
        'sdp_extra_cert': n_sdp_cert,
        'sdp_extra_pct': 100.0 * n_sdp_cert / max(1, len(rows)),
        'sdp_better_than_pw': n_sdp_better_than_pw,
        'sdp_below_pw_warn': n_sdp_strictly_lower_than_pw,
        'time_med_ms': float(1000 * np.median(times_a)) if len(times_a) else None,
        'time_p95_ms': float(1000 * np.percentile(times_a, 95)) if len(times_a) else None,
        'time_max_ms': float(1000 * np.max(times_a)) if len(times_a) else None,
        'sound_violations': sdp_violations,
        'sound_max_excess': sdp_max_excess,
        'rows': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--S', type=int, default=200)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--max_cells', type=int, default=50)
    ap.add_argument('--soundness_n', type=int, default=20)
    ap.add_argument('--mode', choices=['all_pruning', 'all_d'],
                    default='all_pruning')
    ap.add_argument('--sample_mode', choices=['enum', 'random'], default='enum',
                    help='enum: full composition enumeration; random: random '
                         'near-balanced sampling (use for d=4 S>=100).')
    ap.add_argument('--solver', default='auto')
    ap.add_argument('--out', default='_joint_all_window_sdp_results.json')
    ap.add_argument('--smoke', action='store_true',
                    help='Run small d=4 S=20 c=1.20 smoke config only.')
    args = ap.parse_args()

    if not _HAS_CVXPY:
        print("ERROR: cvxpy not installed.")
        sys.exit(1)

    if args.smoke:
        configs = [(4, 20, 1.20)]
    else:
        configs = [(args.d, args.S, args.c_target)]

    results = []
    print("=" * 64)
    print("Joint multi-window MIN-MAX SDP cell-cert bench")
    print("=" * 64)
    for d, S, c in configs:
        r = bench_one_config(
            d, S, c,
            max_cells=args.max_cells,
            soundness_n=args.soundness_n,
            mode=args.mode,
            solver=args.solver,
            sample_mode=args.sample_mode,
        )
        if r is not None:
            results.append(r)

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"{'d':>3} {'S':>4} {'c':>5} {'hard':>5} {'jd_cert':>7} "
          f"{'pwS_cert':>8} {'residue':>7} {'sdp_cert':>8} {'%':>6} "
          f"{'med_ms':>7} {'viol':>5}")
    for r in results:
        print(f"{r['d']:>3} {r['S']:>4} {r['c_target']:>5.2f} "
              f"{r['n_hard']:>5} {r['n_jd_cert']:>7} "
              f"{r['n_pw_cert']:>8} {r['residue']:>7} "
              f"{r['sdp_extra_cert']:>8} {r['sdp_extra_pct']:>6.1f} "
              f"{r['time_med_ms'] or 0:>7.1f} "
              f"{r['sound_violations']:>5}")

    out_path = os.path.join(os.path.dirname(__file__) or '.', args.out)
    # Strip rows for JSON (keep first 30)
    out_results = []
    for r in results:
        rcopy = dict(r)
        if 'rows' in rcopy and len(rcopy['rows']) > 30:
            rcopy['rows'] = rcopy['rows'][:30]
        out_results.append(rcopy)
    with open(out_path, 'w') as f:
        json.dump(out_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
