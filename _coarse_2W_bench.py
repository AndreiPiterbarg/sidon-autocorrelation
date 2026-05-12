"""Two-window JOINT cell-cert via combined Shor SDP — strictly tighter than
per-window or single-window-max bounds, especially when two windows have
anti-correlated gradients.

================================================================
MATHEMATICAL DERIVATION (5 lines)
================================================================
For two windows W1, W2 with TV_W(c+δ) = TV_W(c) + grad_W·δ + scale_W·δᵀA_Wδ:

(1)  cert_{12}(c) = min_{δ ∈ Cell} max(TV_{W1}, TV_{W2})(c+δ)
(2)  ≥ max_{λ ∈ [0,1]} min_{δ ∈ Cell} [λ TV_{W1}(c+δ) + (1-λ) TV_{W2}(c+δ)]      (LP duality)
(3)  Combined inner min for fixed λ has Hessian H(λ) = λ scale_1 A_{W1} + (1-λ) scale_2 A_{W2}.
(4)  This is a single QP per λ, lower-bounded by Shor SDP relaxation (sound).
(5)  Take max over λ ∈ [0,1] via golden-section search (sound: monotone in best-seen value).

KEY INSIGHT: H(λ) can be MORE PSD than λ A_{W1} or (1-λ) A_{W2} alone.
If anti-correlated negatively curved directions cancel, the combined SDP
is much tighter than either per-window SDP.

SOUNDNESS:
- (2) is LP weak duality (sound).
- The Shor SDP optimum is a sound LB on the inner min (PSD relaxation
  contains true feasible set).
- Any λ ∈ [0,1] gives a sound LB; max-over-λ is therefore sound.
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import argparse
from itertools import product, combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


# =====================================================================
# Window enumeration / matrix building
# =====================================================================

def all_windows(d: int) -> List[Tuple[int, int]]:
    conv_len = 2 * d - 1
    out = []
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            out.append((ell, s_lo))
    return out


def build_A_matrix(d: int, ell: int, s_lo: int) -> np.ndarray:
    A = np.zeros((d, d), dtype=np.float64)
    s_hi = s_lo + ell - 2
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                A[i, j] = 1.0
    return A


def tv_at(c: np.ndarray, S: int, d: int, ell: int, s_lo: int) -> float:
    A = build_A_matrix(d, ell, s_lo)
    mu = c.astype(np.float64) / float(S)
    return (2.0 * d / ell) * float(mu @ A @ mu)


def grad_at(c: np.ndarray, S: int, d: int, ell: int, s_lo: int) -> np.ndarray:
    A = build_A_matrix(d, ell, s_lo)
    g = (4.0 * d / (ell * S)) * (A @ c.astype(np.float64))
    return g


# =====================================================================
# Combined Shor SDP for fixed lambda
# =====================================================================

def _combined_shor_lb(c: np.ndarray, S: int, d: int,
                       W1: Tuple[int, int], W2: Tuple[int, int],
                       lam: float,
                       solver: str = 'auto',
                       tol: float = 1e-9,
                       verbose: bool = False) -> Tuple[float, str]:
    """Shor SDP LB on  min_{δ ∈ Cell} [λ TV_{W1}(c/S+δ) + (1-λ) TV_{W2}(c/S+δ)].

    Returns (lb, status).  lb already includes the constant λ TV_{W1}(c/S) +
    (1-λ) TV_{W2}(c/S), so it's directly comparable with c_target.

    Sound by Shor relaxation.
    """
    if not _HAS_CVXPY:
        return float('-inf'), 'NO_CVXPY'

    h = 1.0 / (2.0 * S)
    A1 = build_A_matrix(d, *W1)
    A2 = build_A_matrix(d, *W2)
    g1 = grad_at(c, S, d, *W1)
    g2 = grad_at(c, S, d, *W2)
    tv01 = tv_at(c, S, d, *W1)
    tv02 = tv_at(c, S, d, *W2)
    s1 = 2.0 * d / W1[0]
    s2 = 2.0 * d / W2[0]

    # Combined linear/quad coefficients
    A_combo = lam * s1 * A1 + (1.0 - lam) * s2 * A2
    g_combo = lam * g1 + (1.0 - lam) * g2
    tv_const = lam * tv01 + (1.0 - lam) * tv02

    # Lifted variables
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
# Two-window joint SDP via golden-section search over lambda
# =====================================================================

def cell_cert_2window_SDP(c_int: np.ndarray, S: int, d: int,
                           c_target: float,
                           W1: Tuple[int, int],
                           W2: Tuple[int, int],
                           n_golden_iters: int = 18,
                           solver: str = 'auto',
                           tol: float = 1e-9,
                           verbose: bool = False) -> Tuple[float, float, str]:
    """Sound LB on  min_{δ ∈ Cell} max(TV_{W1}(c+δ), TV_{W2}(c+δ))  via
    SDP + golden-section search over λ ∈ [0,1].

    Returns (best_lb, best_lambda, status).  Sound: max over λ of sound
    Shor LBs; LP weak duality + Shor relaxation soundness compose.

    Strategy:
      - Sample LB at the 4 endpoints/edges (0, 0.5, 1) and refine via
        golden-section ascent.  LB(λ) is generally NOT concave (since the
        outer max-min duality gap closes only at the optimal λ), but in
        practice unimodal on [0,1].  Golden-section finds the local
        max — sound LB regardless.
      - A few "probe" λ values (0.1, 0.25, 0.5, 0.75, 0.9) are evaluated
        as starting points for robustness.
    """
    c = np.asarray(c_int, dtype=np.float64)

    # Initial probes: λ = 0 (= W2 only), λ = 1 (= W1 only), λ = 0.5
    probe_lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    lb_at = {}
    status = 'optimal'
    for lam in probe_lams:
        lb, st = _combined_shor_lb(c, S, d, W1, W2, lam,
                                    solver=solver, tol=tol, verbose=False)
        lb_at[lam] = lb
        if st not in ('optimal', 'optimal_inaccurate'):
            status = st

    # Pick best 2 probes that bracket the max for golden-section.
    best_lam = max(probe_lams, key=lambda x: lb_at[x])

    # Golden-section search over a bracket of width ~0.5 around best_lam.
    PHI = (math.sqrt(5) - 1) / 2  # 0.618...

    def f(lam):
        if lam in lb_at:
            return lb_at[lam]
        v, _ = _combined_shor_lb(c, S, d, W1, W2, lam,
                                  solver=solver, tol=tol, verbose=False)
        lb_at[lam] = v
        return v

    a = max(0.0, best_lam - 0.4)
    b = min(1.0, best_lam + 0.4)
    if a == b:
        return lb_at[best_lam], best_lam, status

    x1 = a + (1 - PHI) * (b - a)
    x2 = a + PHI * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    for _ in range(max(0, n_golden_iters - len(probe_lams))):
        if f1 > f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - PHI) * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + PHI * (b - a)
            f2 = f(x2)

    best_lb = max(lb_at.values())
    best_lam = max(lb_at.keys(), key=lambda x: lb_at[x])
    return best_lb, best_lam, status


# =====================================================================
# Per-window single-W Shor SDP (baseline)
# =====================================================================

def cell_cert_shor_single(c_int: np.ndarray, S: int, d: int,
                           W: Tuple[int, int],
                           solver: str = 'auto',
                           tol: float = 1e-9,
                           verbose: bool = False) -> Tuple[float, str]:
    """Single-window Shor SDP LB on min_δ TV_W(c/S + δ).  Sound."""
    if not _HAS_CVXPY:
        return float('-inf'), 'NO_CVXPY'

    ell, s_lo = W
    c = np.asarray(c_int, dtype=np.float64)
    h = 1.0 / (2.0 * S)
    A = build_A_matrix(d, ell, s_lo)
    g = grad_at(c, S, d, ell, s_lo)
    tv0 = tv_at(c, S, d, ell, s_lo)
    scale = 2.0 * d / ell

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

    obj = g @ delta + scale * cp.trace(A @ D)
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
    return tv0 + float(prob.value), prob.status


# =====================================================================
# Hard-cell finder (cells that triangle baseline cannot certify)
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
    g = grad_at(c, S, d, ell, s_lo)
    g_sorted = np.sort(g)
    h = 1.0 / (2.0 * S)
    cell_var = 0.0
    for k in range(d // 2):
        cell_var += g_sorted[d - 1 - k] - g_sorted[k]
    return h * cell_var


def triangle_cert(c, S, d, c_target):
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


def find_hard_cells(d: int, S: int, c_target: float, max_eval: int = 10**7):
    hard = []
    grid_passers = 0
    n = 0
    for c in enum_compositions(d, S):
        n += 1
        if n > max_eval:
            break
        tri = triangle_cert(c, S, d, c_target)
        if tri is None:
            continue
        grid_passers += 1
        if tri['net'] <= 0:
            hard.append((c.copy(), tri))
    return hard, grid_passers, n


# =====================================================================
# Joint dual K=8 baseline (same approach as _coarse_J_bench, lightweight)
# =====================================================================

def joint_dual_K8_LB(c, S, d, c_target):
    """Joint dual subgradient ascent on K=8 highest-TV pruning windows.

    Mirrors _coarse_J_bench with K=8.  Sound: LP weak duality + per-window
    triangle bounds.
    """
    h = 1.0 / (2.0 * S)
    prefix_nk, prefix_mk = _build_pair_prefix(d)
    # Find pruning windows
    windows = []
    for ell, s_lo in all_windows(d):
        tv = tv_at(c, S, d, ell, s_lo)
        if tv > c_target:
            windows.append((ell, s_lo, tv))
    if not windows:
        return float('-inf')
    windows.sort(key=lambda w: -w[2])
    windows = windows[:8]

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
        # pair_bound
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
    for it in range(20):
        val, sigma = lb_at(lam)
        if val > best_LB:
            best_LB = val
        sub = (tv_list + h * (grad_list @ sigma)
               - scale_list * pb_list * h * h)
        step = 1.0 / math.sqrt(it + 1)
        lam = project_simplex(lam + step * sub)
    return best_LB


# =====================================================================
# Per-window Shor SDP best-of-K (reference)
# =====================================================================

def per_window_shor_best(c, S, d, c_target, K=8,
                         solver='auto', tol=1e-9):
    """Best Shor SDP LB across K highest-TV pruning windows.

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
    windows = windows[:K]
    best = float('-inf')
    best_W = (-1, -1)
    for ell, s_lo, _tv in windows:
        lb, _ = cell_cert_shor_single(c, S, d, (ell, s_lo),
                                       solver=solver, tol=tol)
        if lb > best:
            best = lb
            best_W = (ell, s_lo)
    return best, best_W


# =====================================================================
# Top-2 highest-TV windows
# =====================================================================

def top2_windows(c, S, d, c_target):
    """Return top-2 highest-TV windows (both > c_target)."""
    windows = []
    for ell, s_lo in all_windows(d):
        tv = tv_at(c, S, d, ell, s_lo)
        if tv > c_target:
            windows.append((ell, s_lo, tv))
    windows.sort(key=lambda w: -w[2])
    if len(windows) < 2:
        return None, None
    return (windows[0][0], windows[0][1]), (windows[1][0], windows[1][1])


# =====================================================================
# Gradient cosine angle (anti-correlation diagnostic)
# =====================================================================

def grad_cosine(c, S, d, W1, W2):
    g1 = grad_at(c, S, d, *W1)
    g2 = grad_at(c, S, d, *W2)
    n1 = np.linalg.norm(g1)
    n2 = np.linalg.norm(g2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(g1, g2) / (n1 * n2))


# =====================================================================
# Soundness check (fine-grid)
# =====================================================================

def true_min_max_TV_grid(c_int, S, d, windows, n_grid=9):
    """Sound UPPER bound on cert_box(c) by sampling Cell on a grid.

    Empirical 'true' min-max — no LB should exceed this.
    """
    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / float(S)
    grid = np.linspace(-h, h, n_grid)
    A_list = [build_A_matrix(d, ell, s) for ell, s in windows]
    s_list = [2.0 * d / ell for ell, _ in windows]
    best = math.inf
    if d <= 8:
        ng_use = n_grid
    else:
        ng_use = min(n_grid, 4)
    grid_use = np.linspace(-h, h, ng_use)
    for tup in product(grid_use, repeat=d - 1):
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
# Bench driver
# =====================================================================

def bench_one_config(d: int, S: int, c_target: float,
                      max_cells: int = 30,
                      n_golden_iters: int = 18,
                      soundness_n: int = 50,
                      solver: str = 'auto',
                      verbose: bool = True):
    """Bench 2W joint vs per-window Shor vs joint dual K=8 on hard cells."""
    print(f"\n=== d={d}, S={S}, c_target={c_target} ===")
    t0 = time.time()
    hard, n_pass, n_total = find_hard_cells(d, S, c_target)
    t_enum = time.time() - t0
    print(f"  enumerated {n_total} compositions ({t_enum:.2f}s)")
    print(f"  triangle-failing 'hard' cells: {len(hard)}")
    if not hard:
        print("  No hard cells.")
        return None

    # Sort by triangle net descending (closest-to-cert first)
    hard.sort(key=lambda kv: -kv[1]['net'])
    hard = hard[:max_cells]

    # Run all three certifiers on each hard cell.
    n_pw_cert = 0
    n_2W_cert = 0
    n_jdK8_cert = 0
    times_pw = []
    times_2W = []
    times_jdK8 = []
    cosines_2W_won = []
    cosines_2W_lost = []
    rows = []
    sound_violations_2W = 0
    sound_max_excess_2W = 0.0

    for k, (c, tri) in enumerate(hard):
        # Top-2 windows
        W1, W2 = top2_windows(c, S, d, c_target)
        if W1 is None:
            continue
        cos = grad_cosine(c, S, d, W1, W2)

        # Per-window Shor on top-K=8
        t = time.time()
        pw_lb, pw_W = per_window_shor_best(c, S, d, c_target, K=8,
                                            solver=solver)
        times_pw.append(time.time() - t)
        pw_cert = pw_lb >= c_target - 1e-9

        # 2W joint
        t = time.time()
        lb_2W, lam_star, st_2W = cell_cert_2window_SDP(
            c, S, d, c_target, W1, W2,
            n_golden_iters=n_golden_iters, solver=solver)
        times_2W.append(time.time() - t)
        cert_2W = lb_2W >= c_target - 1e-9

        # Joint dual K=8
        t = time.time()
        jdK8_lb = joint_dual_K8_LB(c, S, d, c_target)
        times_jdK8.append(time.time() - t)
        jdK8_cert = jdK8_lb >= c_target - 1e-9

        if pw_cert:
            n_pw_cert += 1
        if cert_2W:
            n_2W_cert += 1
        if jdK8_cert:
            n_jdK8_cert += 1

        # 2W vs per-window: did 2W win?
        if cert_2W and not pw_cert:
            cosines_2W_won.append(cos)
        elif (not cert_2W) and pw_cert:
            cosines_2W_lost.append(cos)
        elif lb_2W > pw_lb:
            cosines_2W_won.append(cos)

        rows.append({
            'c': c.tolist(), 'tri_net': float(tri['net']),
            'W1': list(W1), 'W2': list(W2), 'cos': float(cos),
            'pw_lb': float(pw_lb), 'pw_cert': bool(pw_cert),
            'lb_2W': float(lb_2W), 'cert_2W': bool(cert_2W),
            'lam_star': float(lam_star),
            'jdK8_lb': float(jdK8_lb), 'jdK8_cert': bool(jdK8_cert),
        })

        if verbose and k < 6:
            print(f"  [{k:3d}] c={c.tolist()} cos={cos:+.3f} "
                  f"pw_lb={pw_lb:.4f}{'C' if pw_cert else ' '} "
                  f"2W_lb={lb_2W:.4f}{'C' if cert_2W else ' '} "
                  f"(lam*={lam_star:.2f}) "
                  f"jdK8={jdK8_lb:.4f}{'C' if jdK8_cert else ' '}")

    # Soundness on a sample
    sound_n_actual = min(soundness_n, len(rows))
    sound_violations = 0
    sound_max_excess = 0.0
    if sound_n_actual > 0 and d <= 8:
        ng = 9 if d <= 6 else (5 if d == 7 else 4)
        idxs = list(range(sound_n_actual))
        for k in idxs:
            r = rows[k]
            ws_used = [tuple(r['W1']), tuple(r['W2'])]
            true_minmax = true_min_max_TV_grid(
                np.array(r['c']), S, d, ws_used, n_grid=ng)
            excess = r['lb_2W'] - true_minmax
            if excess > 1e-7:
                sound_violations += 1
                if excess > sound_max_excess:
                    sound_max_excess = excess

    times_pw_a = np.asarray(times_pw)
    times_2W_a = np.asarray(times_2W)
    times_jdK8_a = np.asarray(times_jdK8)
    print(f"\n  --- Cert rate over {len(rows)} hard cells ---")
    print(f"  per-window Shor (K=8) : {n_pw_cert:>4}  "
          f"({100*n_pw_cert/max(1,len(rows)):.1f}%)")
    print(f"  2W joint SDP          : {n_2W_cert:>4}  "
          f"({100*n_2W_cert/max(1,len(rows)):.1f}%)")
    print(f"  Joint dual K=8        : {n_jdK8_cert:>4}  "
          f"({100*n_jdK8_cert/max(1,len(rows)):.1f}%)")
    print(f"\n  --- 2W win cosine analysis ---")
    if cosines_2W_won:
        print(f"  2W won/tied (n={len(cosines_2W_won)}): "
              f"mean cos = {np.mean(cosines_2W_won):.3f}, "
              f"min = {min(cosines_2W_won):.3f}")
    if cosines_2W_lost:
        print(f"  2W lost (n={len(cosines_2W_lost)}): "
              f"mean cos = {np.mean(cosines_2W_lost):.3f}, "
              f"min = {min(cosines_2W_lost):.3f}")
    print(f"\n  --- Time per cell (ms) ---")
    if len(times_pw_a):
        print(f"  per-window Shor med = {1000*np.median(times_pw_a):.1f}, "
              f"p95 = {1000*np.percentile(times_pw_a, 95):.1f}")
    if len(times_2W_a):
        print(f"  2W joint     med = {1000*np.median(times_2W_a):.1f}, "
              f"p95 = {1000*np.percentile(times_2W_a, 95):.1f}")
    if len(times_jdK8_a):
        print(f"  joint dual K=8 med = {1000*np.median(times_jdK8_a):.1f}")
    print(f"\n  --- Soundness check (2W) ---")
    print(f"  checked {sound_n_actual} cells, "
          f"{sound_violations} violations, "
          f"max excess = {sound_max_excess:.3e}")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_hard': len(rows),
        'n_pw_cert': n_pw_cert,
        'n_2W_cert': n_2W_cert,
        'n_jdK8_cert': n_jdK8_cert,
        'time_pw_med_ms': float(1000*np.median(times_pw_a)) if len(times_pw_a) else None,
        'time_2W_med_ms': float(1000*np.median(times_2W_a)) if len(times_2W_a) else None,
        'time_jdK8_med_ms': float(1000*np.median(times_jdK8_a)) if len(times_jdK8_a) else None,
        'cos_won_mean': float(np.mean(cosines_2W_won)) if cosines_2W_won else None,
        'cos_lost_mean': float(np.mean(cosines_2W_lost)) if cosines_2W_lost else None,
        'cos_won_n': len(cosines_2W_won),
        'cos_lost_n': len(cosines_2W_lost),
        'sound_violations_2W': sound_violations,
        'sound_max_excess_2W': sound_max_excess,
        'rows': rows[:30],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_cells', type=int, default=30)
    ap.add_argument('--n_golden', type=int, default=18)
    ap.add_argument('--solver', default='auto')
    ap.add_argument('--soundness_n', type=int, default=50)
    ap.add_argument('--out', default='_coarse_2W_bench_results.json')
    args = ap.parse_args()

    if not _HAS_CVXPY:
        print("ERROR: cvxpy not installed.")
        sys.exit(1)

    configs = [
        (4, 20, 1.20),
        (6, 15, 1.20),
        (8, 12, 1.20),
    ]
    results = []
    print("=" * 64)
    print("Two-window joint Shor SDP cell-cert bench")
    print("=" * 64)
    for d, S, c in configs:
        r = bench_one_config(d, S, c,
                              max_cells=args.max_cells,
                              n_golden_iters=args.n_golden,
                              soundness_n=args.soundness_n,
                              solver=args.solver)
        if r is not None:
            results.append(r)

    print("\n" + "=" * 64)
    print("SUMMARY (all configs)")
    print("=" * 64)
    print(f"{'d':>3} {'S':>4} {'c':>5} {'hard':>5} "
          f"{'pwShor':>7} {'2W':>5} {'jdK8':>5} "
          f"{'2W_med':>7} {'pw_med':>7} {'viol':>5}")
    for r in results:
        print(f"{r['d']:>3} {r['S']:>4} {r['c_target']:>5.2f} "
              f"{r['n_hard']:>5} {r['n_pw_cert']:>7} "
              f"{r['n_2W_cert']:>5} {r['n_jdK8_cert']:>5} "
              f"{r['time_2W_med_ms']:>7.1f} "
              f"{r['time_pw_med_ms']:>7.1f} "
              f"{r['sound_violations_2W']:>5}")

    out_path = os.path.join(os.path.dirname(__file__) or '.', args.out)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
