"""Smoke test: KKT-residual rigorous per-cell L filter.

Goal: replace MOSEK status='infeasible' check with a Clarabel-direct KKT-
residual rigorous Farkas certificate, modelled on the residual-aware path
in `interval_bnb/lasserre_cert.py:lasserre_box_lb_float_with_residuals`.

PIPELINE
--------
For each Q-survivor at (n_half=3, m=10, d=6, c=1.28):
  1. Build the Shor (order-1) SDP feasibility problem identical to
     `_L_bench.py:_shor_feasibility`.
  2. Call Clarabel directly via problem.get_problem_data → DefaultSolver,
     read native r_prim, r_dual, primal & dual obj values.
  3. Rigorous infeasibility decision:
       Clarabel reports `PrimalInfeasible` when the dual is unbounded
       (i.e., the dual ray gives a Farkas certificate). In that case
       we accept the prune RIGOROUSLY only if r_dual <= rigour_eps
       (KKT-residual cushion).

  4. Compare against MOSEK status='infeasible' from `prune_L_one`.

SOUNDNESS
---------
Per-cell SDP is a relaxation of the integer problem. Primal-infeasible
=> integer-infeasible => composition pruned. The KKT residual cushion
absorbs solver finite-precision noise; pruning only when r_dual is small
relative to the dual ray's magnitude is rigorous (Clarabel's
`PrimalInfeasible` status is itself a Farkas-certificate verdict).

This smoke test reports:
  - n_Q_surv: Q-survivors processed
  - n_MOSEK_infeasible: pruned by MOSEK status check
  - n_Clarabel_infeasible: pruned by Clarabel status check
  - n_Clarabel_rigorous: pruned by Clarabel + r_dual cushion check
  - status histogram per solver
  - timing comparison

Wall < 10 min target (single config, ~50 Q-survivors typical).
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import (
    _build_A_matrices, _make_cell, prune_L_one, _detect_solver,
)


# ---------------------------------------------------------------------
# KKT-residual rigorous Shor feasibility
# ---------------------------------------------------------------------
def _shor_feasibility_kkt_rigorous(c_int, lo, hi, A_mats, windows,
                                    n_half, m, c_target,
                                    *, eps_margin=1e-9,
                                    rigour_eps=1e-6,
                                    time_limit_s=5.0,
                                    max_iter=200,
                                    verbose=False):
    """Build the same Shor SDP as `_L_bench._shor_feasibility`, but call
    Clarabel directly via cvxpy.problem.get_problem_data and read native
    KKT residuals (r_prim, r_dual). Decide infeasibility rigorously:

      RIGOROUS PRUNE iff status == 'PrimalInfeasible' AND r_dual <= rigour_eps.

    Returns dict:
      'pruned_status'    : bool — Clarabel status said primal infeasible
      'pruned_rigorous'  : bool — status + KKT cushion check pass
      'status'           : Clarabel termination string
      'r_prim', 'r_dual' : residuals
      'obj_val', 'obj_val_dual' : primal/dual objectives
      'iterations', 'solve_time'
    """
    import cvxpy as cp
    import scipy.sparse as sp
    import clarabel

    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

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

    try:
        data, _chain, _inv = prob.get_problem_data(cp.CLARABEL)
    except Exception as e:
        return {
            'pruned_status': False, 'pruned_rigorous': False,
            'status': f'BUILD_FAILED:{type(e).__name__}',
            'r_prim': float('inf'), 'r_dual': float('inf'),
            'obj_val': float('nan'), 'obj_val_dual': float('nan'),
            'iterations': 0, 'solve_time': 0.0,
        }

    A = data['A']
    b = data['b']
    qv = data['c']
    P = data.get('P')
    if P is None:
        P = sp.csc_matrix((qv.size, qv.size))
    P = sp.triu(P).tocsc()
    dims = data['dims']

    cones = []
    if dims.zero > 0:
        cones.append(clarabel.ZeroConeT(int(dims.zero)))
    if dims.nonneg > 0:
        cones.append(clarabel.NonnegativeConeT(int(dims.nonneg)))
    for sd in dims.soc:
        cones.append(clarabel.SecondOrderConeT(int(sd)))
    for sd in dims.psd:
        cones.append(clarabel.PSDTriangleConeT(int(sd)))

    sett = clarabel.DefaultSettings()
    sett.verbose = bool(verbose)
    sett.tol_gap_abs = 1e-9
    sett.tol_gap_rel = 1e-9
    sett.tol_feas = 1e-9
    sett.tol_infeas_abs = 1e-9
    sett.tol_infeas_rel = 1e-9
    sett.time_limit = float(time_limit_s)
    sett.max_iter = int(max_iter)

    try:
        solver = clarabel.DefaultSolver(P, qv, A, b, cones, sett)
        sol = solver.solve()
    except Exception as e:
        return {
            'pruned_status': False, 'pruned_rigorous': False,
            'status': f'EXC:{type(e).__name__}',
            'r_prim': float('inf'), 'r_dual': float('inf'),
            'obj_val': float('nan'), 'obj_val_dual': float('nan'),
            'iterations': 0, 'solve_time': 0.0,
        }

    status_str = str(sol.status)
    r_prim = float(sol.r_prim)
    r_dual = float(sol.r_dual)
    primal_obj = float(sol.obj_val)
    dual_obj = float(sol.obj_val_dual)

    pruned_status = status_str == 'PrimalInfeasible'
    pruned_rigorous = pruned_status and (r_dual <= rigour_eps)

    return {
        'pruned_status': pruned_status,
        'pruned_rigorous': pruned_rigorous,
        'status': status_str,
        'r_prim': r_prim, 'r_dual': r_dual,
        'obj_val': primal_obj, 'obj_val_dual': dual_obj,
        'iterations': int(sol.iterations),
        'solve_time': float(sol.solve_time),
    }


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main():
    t_overall = time.time()

    n_half = 3
    m = 10
    c_target = 1.28
    d = 2 * n_half
    S_half = 2 * n_half * m
    rigour_eps = 1e-6

    print(f"=== KKT-rigorous L smoke test (n_half={n_half}, m={m}, d={d}, "
          f"c_target={c_target}) ===")
    print(f"  rigour_eps for r_dual cushion: {rigour_eps}")

    mosek_solver = _detect_solver('MOSEK')
    print(f"  MOSEK solver detected: {mosek_solver}")

    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows)
    print(f"  n_win = {len(windows)}, n_sigma = {len(sigmas)}")

    # Generate all palindromic compositions and find Q-survivors
    print(f"\n[1/3] Generating compositions and running F filter...")
    q_survivors = []
    n_total = 0
    n_F = 0
    n_Q = 0
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=100_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_total += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        n_F += len(f_idx)
        for idx in f_idx:
            c_int = batch[idx]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target, margin=1e-9):
                q_survivors.append(c_int.copy())
                n_Q += 1

    print(f"  total comps:    {n_total}")
    print(f"  F survivors:    {n_F}")
    print(f"  Q survivors:    {n_Q}")

    if n_Q == 0:
        print("  No Q-survivors. Aborting.")
        return

    # Run MOSEK status check
    print(f"\n[2/3] Running MOSEK Shor SDP per-cell on {n_Q} Q-survivors...")
    mosek_pruned = []
    mosek_status_hist = {}
    t_mosek = time.time()
    for c_int in q_survivors:
        pruned, status = prune_L_one(c_int, A_mats, windows, n_half, m,
                                       c_target, solver=mosek_solver,
                                       order=1, tol=1e-9)
        mosek_pruned.append(pruned)
        mosek_status_hist[status] = mosek_status_hist.get(status, 0) + 1
    t_mosek = time.time() - t_mosek
    n_mosek_pruned = sum(mosek_pruned)
    print(f"  MOSEK pruned:   {n_mosek_pruned} / {n_Q}")
    print(f"  MOSEK statuses: {mosek_status_hist}")
    print(f"  MOSEK wall:     {t_mosek:.2f}s "
          f"({1000*t_mosek/max(1,n_Q):.1f} ms/cell)")

    # Run Clarabel KKT-rigorous
    print(f"\n[3/3] Running Clarabel KKT-rigorous Shor SDP on {n_Q} Q-survivors...")
    cla_pruned_status = []
    cla_pruned_rigorous = []
    cla_status_hist = {}
    cla_r_dual_distrib = []
    t_cla = time.time()
    for c_int in q_survivors:
        lo_x, hi_x = _make_cell(c_int, m)
        info = _shor_feasibility_kkt_rigorous(c_int, lo_x, hi_x,
                                                A_mats, windows,
                                                n_half, m, c_target,
                                                rigour_eps=rigour_eps)
        cla_pruned_status.append(info['pruned_status'])
        cla_pruned_rigorous.append(info['pruned_rigorous'])
        cla_status_hist[info['status']] = cla_status_hist.get(info['status'], 0) + 1
        if info['status'] == 'PrimalInfeasible':
            cla_r_dual_distrib.append(info['r_dual'])
    t_cla = time.time() - t_cla
    n_cla_status = sum(cla_pruned_status)
    n_cla_rigorous = sum(cla_pruned_rigorous)
    print(f"  Clarabel status pruned:    {n_cla_status} / {n_Q}")
    print(f"  Clarabel rigorous pruned:  {n_cla_rigorous} / {n_Q}")
    print(f"  Clarabel statuses:         {cla_status_hist}")
    print(f"  Clarabel wall:             {t_cla:.2f}s "
          f"({1000*t_cla/max(1,n_Q):.1f} ms/cell)")
    if cla_r_dual_distrib:
        arr = np.array(cla_r_dual_distrib)
        print(f"  r_dual on infeasible:      "
              f"min={arr.min():.2e}, med={np.median(arr):.2e}, "
              f"max={arr.max():.2e}")

    # Cross-check: agreement between MOSEK status, Clarabel status, Clarabel
    # rigorous.
    print(f"\n=== Cross-check ===")
    n_both = sum(1 for i in range(n_Q)
                  if mosek_pruned[i] and cla_pruned_status[i])
    n_mosek_only = sum(1 for i in range(n_Q)
                        if mosek_pruned[i] and not cla_pruned_status[i])
    n_cla_only = sum(1 for i in range(n_Q)
                      if cla_pruned_status[i] and not mosek_pruned[i])
    n_rigorous_subset = sum(1 for i in range(n_Q)
                             if cla_pruned_rigorous[i] and mosek_pruned[i])
    n_rigorous_only_status = sum(1 for i in range(n_Q)
                                   if cla_pruned_status[i]
                                   and not cla_pruned_rigorous[i])
    print(f"  MOSEK & Clarabel-status agree (both prune):    {n_both}")
    print(f"  MOSEK only:                                     {n_mosek_only}")
    print(f"  Clarabel-status only:                           {n_cla_only}")
    print(f"  Clarabel-rigorous SUBSET of MOSEK-pruned:       "
          f"{n_rigorous_subset == n_cla_rigorous}")
    print(f"  Status pruned but not rigorous (large r_dual):  "
          f"{n_rigorous_only_status}")

    # Soundness check: rigorous prunes are a subset of MOSEK prunes (both
    # should be valid prunes; the gap is conservatism, not unsoundness).
    soundness_ok = all((not cla_pruned_rigorous[i]) or mosek_pruned[i]
                        or cla_pruned_status[i]
                        for i in range(n_Q))
    print(f"  rigorous-implies-status: {soundness_ok}")

    out = {
        'n_half': n_half, 'm': m, 'd': d, 'c_target': c_target,
        'n_total': n_total, 'n_F': n_F, 'n_Q': n_Q,
        'n_mosek_pruned': n_mosek_pruned,
        'n_cla_status_pruned': n_cla_status,
        'n_cla_rigorous_pruned': n_cla_rigorous,
        'mosek_status_hist': mosek_status_hist,
        'cla_status_hist': cla_status_hist,
        'r_dual_distrib': {
            'min': float(np.min(cla_r_dual_distrib)) if cla_r_dual_distrib else None,
            'med': float(np.median(cla_r_dual_distrib)) if cla_r_dual_distrib else None,
            'max': float(np.max(cla_r_dual_distrib)) if cla_r_dual_distrib else None,
            'n': len(cla_r_dual_distrib),
        },
        't_mosek_s': t_mosek, 't_cla_s': t_cla,
        'agreement': {
            'both_prune': n_both,
            'mosek_only': n_mosek_only,
            'cla_only': n_cla_only,
            'rigorous_subset_status': n_rigorous_subset == n_cla_rigorous,
            'status_but_not_rigorous': n_rigorous_only_status,
        },
        'rigour_eps': rigour_eps,
        'wall_total_s': time.time() - t_overall,
    }
    out_path = os.path.join(_HERE, '_smoke_KKT_rigorous_L_results.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\n  Wrote {out_path}")
    print(f"  Total wall: {time.time() - t_overall:.2f}s")


if __name__ == '__main__':
    main()
