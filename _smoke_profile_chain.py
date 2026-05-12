"""Smoke profile of the full cascade chain F -> FN -> Q -> QN -> L.

Per-phase timing on representative configs (n_half=3, m=10, d=6) and
(n_half=4, m=10, d=8). For Q/QN: per-LP breakdown (window-data prep,
LP-array build, linprog).  For L: per-SDP breakdown (cvxpy build,
prob.solve(), MOSEK solver-stats).

Constraint: do NOT modify any existing files. Sequential paths only
(profiling per-LP/per-SDP cost, not parallel scaling).

Output: prints per-phase summary; saves _smoke_profile_chain.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations

import numpy as np

# --- repo paths
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'cloninger-steinerberger', 'cpu'))


# --- imports (these must succeed)
from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _FN_bench import prune_FN
from _N_bench import precompute_op_norm_restricted
from _Q_bench import (
    _build_windows,
    _composition_window_data,
    _enum_balanced_signs,
)
from _QN_bench import precompute_window_data
from _L_bench import _build_A_matrices, _detect_solver

from scipy.optimize import linprog
import cvxpy as cp


# ----------------------------------------------------------------------
# Q-LP profiled variant: replicates _Q_bench._q_bound_lp internals but with
# fine-grained timing. Soundness/return are identical (we just instrument).
# ----------------------------------------------------------------------
def _q_bound_lp_profiled(c_int, windows, ell_int_sums, sigmas,
                         n_half, m, c_target):
    """Profiled clone of _Q_bench._q_bound_lp.

    Returns (t_opt, t_winddata_s, t_buildlp_s, t_linprog_s, t_total_s).
    """
    t_total_0 = time.perf_counter()

    # window data
    t0 = time.perf_counter()
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    t_winddata = time.perf_counter() - t0

    # build LP arrays
    t0 = time.perf_counter()
    n_win = len(windows)
    n_sigma = len(sigmas)
    n_d = float(n_half)
    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m
    V = (ws.astype(np.float64) * inv_4nl
         - ell_int_sums.astype(np.float64) * inv_4nl - cs_m2)
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    M = sigmas.astype(np.float64) @ BB_over_ell.T
    A = V[None, :] - M / (2.0 * n_d)
    nvar = n_win + 1
    c_obj = np.zeros(nvar)
    c_obj[-1] = -1.0
    A_ub = np.zeros((n_sigma, nvar))
    A_ub[:, :n_win] = -A
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_sigma)
    A_eq = np.zeros((1, nvar))
    A_eq[0, :n_win] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n_win + [(None, None)]
    t_build = time.perf_counter() - t0

    # linprog
    t0 = time.perf_counter()
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs',
                      options={'presolve': True})
        t_lp = time.perf_counter() - t0
        t_opt = float(-res.fun) if res.success else float('-inf')
    except Exception:
        t_lp = time.perf_counter() - t0
        t_opt = float('-inf')
    t_total = time.perf_counter() - t_total_0
    return t_opt, t_winddata, t_build, t_lp, t_total


# Same instrumentation for QN-fast LP
def _qn_bound_lp_profiled(c_int, windows, ell_int_sums, sigmas,
                          n_half, m, c_target, m_W_arr):
    t_total_0 = time.perf_counter()

    t0 = time.perf_counter()
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    t_winddata = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_win = len(windows)
    n_sigma = len(sigmas)
    n_d = float(n_half)
    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m
    V = ws.astype(np.float64) * inv_4nl - m_W_arr * inv_4nl - cs_m2
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    M = sigmas.astype(np.float64) @ BB_over_ell.T
    A = V[None, :] - M / (2.0 * n_d)
    nvar = n_win + 1
    c_obj = np.zeros(nvar); c_obj[-1] = -1.0
    A_ub = np.zeros((n_sigma, nvar))
    A_ub[:, :n_win] = -A
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_sigma)
    A_eq = np.zeros((1, nvar))
    A_eq[0, :n_win] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n_win + [(None, None)]
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs',
                      options={'presolve': True})
        t_lp = time.perf_counter() - t0
        t_opt = float(-res.fun) if res.success else float('-inf')
    except Exception:
        t_lp = time.perf_counter() - t0
        t_opt = float('-inf')
    t_total = time.perf_counter() - t_total_0
    return t_opt, t_winddata, t_build, t_lp, t_total


# ----------------------------------------------------------------------
# L SDP profiled (Shor / order-1) — clones _L_bench._shor_feasibility
# but separates "build cvxpy problem" from "prob.solve()" and reports
# MOSEK solver_stats.
# ----------------------------------------------------------------------
def _shor_feasibility_profiled(c_int, A_mats, windows, n_half, m, c_target,
                               solver='MOSEK', tol=1e-9, eps_margin=1e-9):
    t_total_0 = time.perf_counter()

    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m
    c = np.asarray(c_int, dtype=np.float64)
    lo = np.maximum(0.0, c - 1.0)
    hi = c + 1.0

    # ------ build cvxpy problem
    t0 = time.perf_counter()
    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)
    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                 [cp.reshape(x, (d, 1), order='C'), X]])
    cons = [Y >> 0]
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]
    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]
    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]
    prob = cp.Problem(cp.Minimize(0), cons)
    t_build = time.perf_counter() - t0

    # ------ solve
    t0 = time.perf_counter()
    try:
        if solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=False,
                       mosek_params={
                           'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                           'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                           'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                       })
        elif solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=False,
                       eps_abs=tol, eps_rel=tol, max_iter=300)
        else:
            prob.solve(verbose=False)
    except Exception:
        pass
    t_solve = time.perf_counter() - t0

    # MOSEK solver-stats split
    stats = getattr(prob, 'solver_stats', None)
    t_mosek = float(stats.solve_time) if stats and stats.solve_time else 0.0
    t_setup = float(stats.setup_time) if stats and stats.setup_time else 0.0

    t_total = time.perf_counter() - t_total_0
    status = prob.status if prob.status is not None else 'UNKNOWN'
    pruned = (status == 'infeasible')
    return pruned, status, t_build, t_solve, t_mosek, t_setup, t_total


# ----------------------------------------------------------------------
# Run one config: profile F, FN, Q, QN, L sequentially with breakdowns.
# ----------------------------------------------------------------------
def profile_config(n_half, m, c_target, max_l_solves=120, max_qn_solves=None,
                    max_q_solves=None):
    """Profile one (n_half, m, c_target) config.

    Returns dict with phase-level breakdowns.
    """
    d = 2 * n_half
    S_half = 2 * n_half * m
    print(f"\n{'='*72}")
    print(f"  CONFIG: n_half={n_half}, m={m}, c_target={c_target}, d={d}")
    print(f"{'='*72}")

    out = {
        'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
        'S_half': S_half,
        'phases': {},
    }

    # ---- generate ALL palindromic comps (small d, fits in memory)
    t0 = time.perf_counter()
    half_batches = list(generate_compositions_batched(n_half, S_half,
                                                       batch_size=2_000_000))
    half = np.concatenate(half_batches, axis=0) if half_batches else np.empty((0, n_half), dtype=np.int32)
    batch = np.empty((len(half), d), dtype=np.int32)
    batch[:, :n_half] = half
    batch[:, n_half:] = half[:, ::-1]
    n_all = len(batch)
    out['n_processed'] = n_all
    out['t_gen_s'] = time.perf_counter() - t0
    print(f"  comps={n_all:,}  generated in {out['t_gen_s']:.2f}s")

    # =================================================================
    # F (Numba prange)
    # =================================================================
    # warmup
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)
    t0 = time.perf_counter()
    sF = prune_F(batch, n_half, m, c_target)
    t_F = time.perf_counter() - t0
    n_surv_F = int(np.sum(sF))
    out['phases']['F'] = {
        'wall_s': t_F,
        'ms_per_comp': 1000.0 * t_F / max(1, n_all),
        'n_input': n_all,
        'n_survivors': n_surv_F,
        'frac_kept': n_surv_F / max(1, n_all),
    }
    print(f"  [F]  {t_F*1000:.1f} ms  ({1000.0*t_F/n_all:.4f} ms/comp)  "
          f"surv {n_surv_F:,}/{n_all:,}")

    # =================================================================
    # FN (Numba prange) — needs ell_prefix + op_rest_d
    # =================================================================
    conv_len = 2 * d - 1
    max_ell = 2 * d
    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = abs((k + 1) - two_n)
        v = max(0, two_n - d_idx)
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]
    op_rest, _ = precompute_op_norm_restricted(d, max_ell, conv_len)
    op_rest_d = op_rest * d

    # warmup
    prune_FN(warm.astype(np.int32), n_half, m, c_target, ell_prefix, op_rest_d)
    t0 = time.perf_counter()
    sFN = prune_FN(batch.astype(np.int32, copy=False), n_half, m, c_target,
                    ell_prefix, op_rest_d)
    t_FN = time.perf_counter() - t0
    n_surv_FN = int(np.sum(sFN))
    out['phases']['FN'] = {
        'wall_s': t_FN,
        'ms_per_comp': 1000.0 * t_FN / max(1, n_all),
        'n_input': n_all,
        'n_survivors': n_surv_FN,
        'frac_kept': n_surv_FN / max(1, n_all),
    }
    print(f"  [FN] {t_FN*1000:.1f} ms  ({1000.0*t_FN/n_all:.4f} ms/comp)  "
          f"surv {n_surv_FN:,}/{n_all:,}")

    # =================================================================
    # Q (scipy linprog HiGHS) — runs ONLY on FN-survivors
    # =================================================================
    surv_FN = batch[sFN]
    t0 = time.perf_counter()
    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    n_win = len(windows)
    n_sigma = len(sigmas)
    out['n_win'] = n_win
    out['n_sigma'] = n_sigma
    print(f"  Q-setup {time.perf_counter()-t0:.3f}s  n_win={n_win} "
          f"n_sigma={n_sigma}")

    # Profile Q on a sample (cap so wall < 30s for d=8)
    if max_q_solves is None:
        max_q_solves = min(len(surv_FN), 800 if d <= 6 else 200)
    q_sample = surv_FN[:max_q_solves]
    q_total_times = []
    q_wd_times = []
    q_build_times = []
    q_lp_times = []
    q_pruned = 0
    t0 = time.perf_counter()
    for c_int in q_sample:
        t_opt, twd, tb, tlp, ttot = _q_bound_lp_profiled(
            c_int, windows, ell_int_sums, sigmas,
            n_half, m, c_target)
        q_wd_times.append(twd)
        q_build_times.append(tb)
        q_lp_times.append(tlp)
        q_total_times.append(ttot)
        if t_opt > 1e-9 * m * m:
            q_pruned += 1
    t_Q_wall = time.perf_counter() - t0

    if q_total_times:
        qa = np.array(q_total_times)
        qwd = np.array(q_wd_times)
        qb = np.array(q_build_times)
        qlp = np.array(q_lp_times)
        out['phases']['Q'] = {
            'n_input': len(surv_FN),
            'n_solved_for_profile': len(q_sample),
            'n_pruned_in_sample': q_pruned,
            'wall_sample_s': t_Q_wall,
            'ms_per_lp_mean': 1000.0 * float(qa.mean()),
            'ms_per_lp_median': 1000.0 * float(np.median(qa)),
            'ms_per_lp_std': 1000.0 * float(qa.std()),
            'ms_per_lp_p95': 1000.0 * float(np.percentile(qa, 95)),
            'breakdown': {
                'winddata_ms_mean': 1000.0 * float(qwd.mean()),
                'build_ms_mean': 1000.0 * float(qb.mean()),
                'linprog_ms_mean': 1000.0 * float(qlp.mean()),
                'frac_winddata': float(qwd.sum() / qa.sum()),
                'frac_build': float(qb.sum() / qa.sum()),
                'frac_linprog': float(qlp.sum() / qa.sum()),
            },
            'projected_total_s': float(qa.mean() * len(surv_FN)),
        }
        print(f"  [Q]  {len(q_sample)} LPs in {t_Q_wall:.2f}s  "
              f"mean {qa.mean()*1000:.2f} ms/LP  med {np.median(qa)*1000:.2f}")
        print(f"        winddata {qwd.mean()*1000:.2f} ms ({100*qwd.sum()/qa.sum():.1f}%)  "
              f"build {qb.mean()*1000:.2f} ms ({100*qb.sum()/qa.sum():.1f}%)  "
              f"linprog {qlp.mean()*1000:.2f} ms ({100*qlp.sum()/qa.sum():.1f}%)")
        print(f"        projected total ({len(surv_FN):,} LPs) = "
              f"{qa.mean()*len(surv_FN):.1f}s")
    else:
        out['phases']['Q'] = {'n_input': 0, 'note': 'no FN-survivors'}

    # Build Q-survivor set for downstream profiling.
    # We cannot afford to run Q on ALL FN-survivors at d=8; instead, do a
    # FAST RUN over the full FN-survivors using the original prune_Q_one
    # to get the actual Q-survivor set used by QN/L.
    from _Q_bench import prune_Q_one as _q_one
    keep_Q = np.ones(len(surv_FN), dtype=bool)
    n_q_full = min(len(surv_FN), 5000 if d <= 6 else 1500)
    for i in range(n_q_full):
        if _q_one(surv_FN[i], windows, ell_int_sums, sigmas,
                   n_half, m, c_target):
            keep_Q[i] = False
    surv_Q = surv_FN[keep_Q]
    print(f"  Q (fast over {n_q_full}): "
          f"surv {keep_Q.sum():,}/{len(surv_FN):,}")

    # =================================================================
    # QN (scipy linprog HiGHS, same shape as Q but with m_W_arr)
    # =================================================================
    t0 = time.perf_counter()
    windows_qn, ell_qn, sigmas_qn, m_W_arr, _ = precompute_window_data(
        d, n_half)
    print(f"  QN-setup {time.perf_counter()-t0:.3f}s")

    if max_qn_solves is None:
        max_qn_solves = min(len(surv_Q), 800 if d <= 6 else 200)
    qn_sample = surv_Q[:max_qn_solves]
    qn_total = []; qn_wd = []; qn_b = []; qn_lp = []
    qn_pruned = 0
    t0 = time.perf_counter()
    for c_int in qn_sample:
        t_opt, twd, tb, tlp, ttot = _qn_bound_lp_profiled(
            c_int, windows_qn, ell_qn, sigmas_qn,
            n_half, m, c_target, m_W_arr)
        qn_wd.append(twd); qn_b.append(tb); qn_lp.append(tlp); qn_total.append(ttot)
        if t_opt > 1e-9 * m * m:
            qn_pruned += 1
    t_QN_wall = time.perf_counter() - t0
    if qn_total:
        qa = np.array(qn_total); qwd = np.array(qn_wd)
        qb = np.array(qn_b); qlp = np.array(qn_lp)
        out['phases']['QN'] = {
            'n_input': len(surv_Q),
            'n_solved_for_profile': len(qn_sample),
            'n_pruned_in_sample': qn_pruned,
            'wall_sample_s': t_QN_wall,
            'ms_per_lp_mean': 1000.0 * float(qa.mean()),
            'ms_per_lp_median': 1000.0 * float(np.median(qa)),
            'ms_per_lp_std': 1000.0 * float(qa.std()),
            'ms_per_lp_p95': 1000.0 * float(np.percentile(qa, 95)),
            'breakdown': {
                'winddata_ms_mean': 1000.0 * float(qwd.mean()),
                'build_ms_mean': 1000.0 * float(qb.mean()),
                'linprog_ms_mean': 1000.0 * float(qlp.mean()),
                'frac_winddata': float(qwd.sum() / qa.sum()),
                'frac_build': float(qb.sum() / qa.sum()),
                'frac_linprog': float(qlp.sum() / qa.sum()),
            },
            'projected_total_s': float(qa.mean() * len(surv_Q)),
        }
        print(f"  [QN] {len(qn_sample)} LPs in {t_QN_wall:.2f}s  "
              f"mean {qa.mean()*1000:.2f} ms/LP  med {np.median(qa)*1000:.2f}")
        print(f"        winddata {qwd.mean()*1000:.2f} ms ({100*qwd.sum()/qa.sum():.1f}%)  "
              f"build {qb.mean()*1000:.2f} ms ({100*qb.sum()/qa.sum():.1f}%)  "
              f"linprog {qlp.mean()*1000:.2f} ms ({100*qlp.sum()/qa.sum():.1f}%)")
        print(f"        projected total ({len(surv_Q):,} LPs) = "
              f"{qa.mean()*len(surv_Q):.1f}s")
    else:
        out['phases']['QN'] = {'n_input': 0, 'note': 'no Q-survivors'}

    # =================================================================
    # L (cvxpy + MOSEK SDP) — most expensive, profile a small batch
    # =================================================================
    A_mats = _build_A_matrices(d, windows)
    solver = _detect_solver(prefer='MOSEK')
    print(f"  L solver = {solver}")

    if len(surv_Q) == 0:
        out['phases']['L'] = {'n_input': 0, 'note': 'no Q-survivors'}
    else:
        l_sample = surv_Q[:max_l_solves]
        l_total = []; l_build = []; l_solve = []; l_mosek = []
        l_pruned = 0
        t0 = time.perf_counter()
        for c_int in l_sample:
            pruned, status, tb, tsol, tmsk, tset, ttot = (
                _shor_feasibility_profiled(
                    c_int, A_mats, windows, n_half, m, c_target,
                    solver=solver))
            l_build.append(tb); l_solve.append(tsol)
            l_mosek.append(tmsk); l_total.append(ttot)
            if pruned:
                l_pruned += 1
        t_L_wall = time.perf_counter() - t0
        la = np.array(l_total); lb = np.array(l_build)
        ls = np.array(l_solve); lmsk = np.array(l_mosek)
        out['phases']['L'] = {
            'n_input': len(surv_Q),
            'n_solved_for_profile': len(l_sample),
            'n_pruned_in_sample': l_pruned,
            'wall_sample_s': t_L_wall,
            'ms_per_sdp_mean': 1000.0 * float(la.mean()),
            'ms_per_sdp_median': 1000.0 * float(np.median(la)),
            'ms_per_sdp_std': 1000.0 * float(la.std()),
            'ms_per_sdp_p95': 1000.0 * float(np.percentile(la, 95)),
            'breakdown': {
                'cvxpy_build_ms_mean': 1000.0 * float(lb.mean()),
                'solve_call_ms_mean': 1000.0 * float(ls.mean()),
                'mosek_internal_ms_mean': 1000.0 * float(lmsk.mean()),
                'frac_cvxpy_build': float(lb.sum() / la.sum()),
                'frac_solve_call': float(ls.sum() / la.sum()),
                'frac_mosek_internal': float(lmsk.sum() / la.sum()) if lmsk.sum() > 0 else None,
                'frac_canonicalize_estimate': (
                    float((ls.sum() - lmsk.sum()) / la.sum()) if lmsk.sum() > 0 else None
                ),
            },
            'projected_total_s': float(la.mean() * len(surv_Q)),
        }
        print(f"  [L]  {len(l_sample)} SDPs in {t_L_wall:.2f}s  "
              f"mean {la.mean()*1000:.1f} ms/SDP  med {np.median(la)*1000:.1f}")
        print(f"        cvxpy-build {lb.mean()*1000:.1f} ms "
              f"({100*lb.sum()/la.sum():.1f}%)  "
              f"prob.solve {ls.mean()*1000:.1f} ms "
              f"({100*ls.sum()/la.sum():.1f}%)  "
              f"MOSEK {lmsk.mean()*1000:.1f} ms "
              f"({100*lmsk.sum()/la.sum():.1f}%)")
        print(f"        projected total ({len(surv_Q):,} SDPs) = "
              f"{la.mean()*len(surv_Q):.1f}s")

    # =================================================================
    # Phase-aggregated wall projection
    # =================================================================
    aggregate_s = {
        'F': out['phases']['F']['wall_s'],
        'FN': out['phases']['FN']['wall_s'],
        'Q': out['phases'].get('Q', {}).get('projected_total_s', 0.0),
        'QN': out['phases'].get('QN', {}).get('projected_total_s', 0.0),
        'L': out['phases'].get('L', {}).get('projected_total_s', 0.0),
    }
    total = sum(aggregate_s.values())
    pct = {k: 100.0 * v / max(1e-9, total) for k, v in aggregate_s.items()}
    out['aggregate_s'] = aggregate_s
    out['aggregate_pct'] = pct
    out['aggregate_total_s'] = total
    print(f"\n  AGGREGATE (projected to all surviving inputs):")
    for k, v in sorted(aggregate_s.items(), key=lambda x: -x[1]):
        print(f"      {k:3s}: {v:9.2f} s  ({pct[k]:5.1f}%)")
    print(f"      TOT: {total:9.2f} s")

    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    configs = [
        (3, 10, 1.28, dict(max_l_solves=120, max_q_solves=300, max_qn_solves=300)),
        (4, 10, 1.28, dict(max_l_solves=80, max_q_solves=120, max_qn_solves=120)),
    ]
    out = {'configs': []}
    for nh, m, c, kw in configs:
        try:
            r = profile_config(nh, m, c, **kw)
            out['configs'].append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out['configs'].append({
                'n_half': nh, 'm': m, 'c_target': c,
                'error': f'{type(e).__name__}: {e}',
            })

    # ------------------------------------------------------------------
    # Top-3 bottlenecks summary across both configs
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  GLOBAL BOTTLENECK SUMMARY")
    print(f"{'='*72}")
    total_phase = {'F': 0.0, 'FN': 0.0, 'Q': 0.0, 'QN': 0.0, 'L': 0.0}
    for c in out['configs']:
        if 'aggregate_s' not in c:
            continue
        for k, v in c['aggregate_s'].items():
            total_phase[k] = total_phase.get(k, 0.0) + v
    grand = sum(total_phase.values())
    out['global_aggregate_s'] = total_phase
    out['global_pct'] = {k: 100.0 * v / max(1e-9, grand)
                         for k, v in total_phase.items()}
    sorted_phases = sorted(total_phase.items(), key=lambda x: -x[1])
    print(f"\n  Phase totals (across both configs, projected):")
    for k, v in sorted_phases:
        pct = 100.0 * v / max(1e-9, grand)
        print(f"      {k:3s}: {v:9.2f} s  ({pct:5.1f}%)")
    print(f"      TOT: {grand:9.2f} s")
    print(f"\n  TOP-3 BOTTLENECKS:")
    for i, (k, v) in enumerate(sorted_phases[:3]):
        pct = 100.0 * v / max(1e-9, grand)
        print(f"   {i+1}. {k}: {v:.2f}s ({pct:.1f}%)")

    out_path = os.path.join(_REPO_ROOT, '_smoke_profile_chain.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
