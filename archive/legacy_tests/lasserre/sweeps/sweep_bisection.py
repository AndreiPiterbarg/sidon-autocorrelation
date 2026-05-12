#!/usr/bin/env python
"""Comprehensive sweep of bisection hyperparameters at d=32 bw=31 order=2.

Builds the problem ONCE, then runs multiple configurations measuring
per-step time and convergence behavior.

Sweeps:
  1. Iteration budget: how many iters does it take to actually converge?
  2. Rho: does rho=0.1 beat rho=0.5 for the phase-1 problem?
  3. Early tau thresholds: can we classify infeasible earlier?
  4. Eps tolerance: does looser eps help without hurting accuracy?
"""
import sys, os, time, json
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, enum_monomials, val_d_known,
)
from run_scs_direct import (
    build_base_problem, _precompute_window_psd_decomposition,
    _assemble_window_psd,
)
from admm_gpu_solver import ADMMSolver, augment_phase1, admm_solve
import torch


def build_problem():
    """Build the full augmented problem. Returns everything needed."""
    d, order, bw = 32, 2, 31

    print(f"Building problem d={d} O{order} bw={bw}...")
    t0 = time.time()
    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, order, cliques, verbose=False)
    n_y = P['n_y']
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, add_upper_loc=True)

    # Quick Round 0
    sol = admm_solve(A_base, b_base, c_obj, cone_base,
                     max_iters=5000, eps_abs=1e-5, eps_rel=1e-5,
                     device='cuda', verbose=False)
    scalar_lb = float(sol['x'][meta['t_col']])
    y_vals = sol['x'][:n_y].copy()

    active_windows = set()
    violations = _check_violations_highd(y_vals, scalar_lb, P, active_windows)
    for w, eig in violations[:30]:
        active_windows.add(w)

    # Build augmented phase-1 problem template
    win_decomp = _precompute_window_psd_decomposition(P, active_windows)
    A_win_t1, _, psd_win = _assemble_window_psd(win_decomp, 1.0)
    A_full_t1 = sp.vstack([A_base, A_win_t1], format='csc')
    A_full_t1.sort_indices()

    A_win_t2, _, _ = _assemble_window_psd(win_decomp, 2.0)
    A_full_t2 = sp.vstack([A_base, A_win_t2], format='csc')
    A_full_t2.sort_indices()

    full_t_data = A_full_t2.data - A_full_t1.data
    full_base_data = A_full_t1.data - full_t_data
    b_full_base = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}

    n_cols_full = A_full_t1.shape[1]
    fix_t_row = sp.csc_matrix(([1.0], ([0], [meta['t_col']])),
                               shape=(1, n_cols_full))
    cone_fixed = {'z': cone_full['z'] + 1, 'l': cone_full['l'],
                  's': cone_full['s']}

    np.add(full_base_data, 1.0 * full_t_data, out=A_full_t1.data)
    A_fixed_t1 = sp.vstack([fix_t_row, A_full_t1], format='csc')
    A_fixed_t1.sort_indices()
    b_fixed_t1 = np.insert(b_full_base, 0, 1.0)
    A_p1_t1, b_p1_t1, c_p1, cone_p1, tau_col = augment_phase1(
        A_fixed_t1, b_fixed_t1, cone_fixed)

    np.add(full_base_data, 2.0 * full_t_data, out=A_full_t1.data)
    A_fixed_t2 = sp.vstack([fix_t_row, A_full_t1], format='csc')
    A_fixed_t2.sort_indices()
    b_fixed_t2 = np.insert(b_full_base, 0, 2.0)
    A_p1_t2, b_p1_t2, _, _, _ = augment_phase1(
        A_fixed_t2, b_fixed_t2, cone_fixed)

    aug_t_data = A_p1_t2.data - A_p1_t1.data
    aug_base_data = A_p1_t1.data - aug_t_data
    aug_b_t_data = b_p1_t2 - b_p1_t1
    aug_b_base_data = b_p1_t1 - aug_b_t_data

    A_template = A_p1_t1.copy()
    b_template = b_p1_t1.copy()

    print(f"Problem built in {time.time()-t0:.1f}s")
    print(f"  n={A_template.shape[1]:,}, m={A_template.shape[0]:,}")
    print(f"  scalar_lb={scalar_lb:.6f}, {len(active_windows)} windows")
    print(f"  tau_col={tau_col}\n")

    return {
        'A_template': A_template, 'b_template': b_template,
        'c_p1': c_p1, 'cone_p1': cone_p1, 'tau_col': tau_col,
        'aug_base_data': aug_base_data, 'aug_t_data': aug_t_data,
        'aug_b_base_data': aug_b_base_data, 'aug_b_t_data': aug_b_t_data,
        'scalar_lb': scalar_lb,
    }


def setup_t(prob, t_val):
    """Set up A and b for a given t_val."""
    np.add(prob['aug_base_data'], t_val * prob['aug_t_data'],
           out=prob['A_template'].data)
    np.add(prob['aug_b_base_data'], t_val * prob['aug_b_t_data'],
           out=prob['b_template'])


def single_solve(prob, t_val, *, rho=0.5, max_iters=800, eps=5e-7,
                 tau_infeas_mult=100, solver_cache=None):
    """Run a single bisection step. Returns result dict."""
    setup_t(prob, t_val)
    tau_col = prob['tau_col']
    tau_tol = max(eps * 10, 1e-4)

    if solver_cache[0] is None:
        solver_cache[0] = ADMMSolver(
            prob['A_template'], prob['b_template'], prob['c_p1'],
            prob['cone_p1'], rho=rho, device='cuda', verbose=False)
        # Warm-up
        solver_cache[0].solve(max_iters=100, eps_abs=1.0, eps_rel=1.0,
                              tau_col=tau_col)
    else:
        solver_cache[0]._update_A(prob['A_template'])
        solver_cache[0].update_b(prob['b_template'])

    t0 = time.time()
    sol = solver_cache[0].solve(
        max_iters=max_iters, eps_abs=eps, eps_rel=eps,
        tau_col=tau_col, tau_tol=tau_tol)
    dt = time.time() - t0

    tau_val = sol['x'][tau_col]
    feasible = (sol['info']['status'] in ('solved', 'solved_inaccurate')
                and tau_val <= tau_tol)

    return {
        't': t_val, 'feasible': feasible, 'tau': tau_val,
        'iters': sol['info']['iter'], 'time': dt,
        'ms_per_iter': dt / max(sol['info']['iter'], 1) * 1000,
        'status': sol['info']['status'],
    }


def sweep_convergence(prob):
    """SWEEP 1: How many iters to converge at different t values?

    Test 3 t-values: clearly infeasible, near boundary, clearly feasible.
    Run each with high max_iters to find true convergence point.
    """
    lb = prob['scalar_lb']
    # Test points at different distances from boundary
    test_points = [
        ('far_infeas', lb - 0.02),
        ('near_infeas', lb + 0.02),
        ('boundary', lb + 0.035),
        ('near_feas', lb + 0.04),
        ('far_feas', lb + 0.06),
    ]

    print(f"{'='*70}")
    print("SWEEP 1: Convergence vs iteration budget")
    print(f"  scalar_lb={lb:.6f}")
    print(f"{'='*70}")

    for label, t_val in test_points:
        for max_it in [200, 400, 800, 1500, 3000, 6000]:
            cache = [None]
            r = single_solve(prob, t_val, max_iters=max_it, eps=5e-7,
                             rho=0.5, solver_cache=cache)
            tag = "FEAS" if r['feasible'] else "INFEAS"
            conv = "CONV" if r['status'] == 'solved' else "cap"
            print(f"  {label:12s} t={t_val:.4f} budget={max_it:5d}: "
                  f"{r['iters']:5d} iters {r['time']:6.1f}s "
                  f"tau={r['tau']:+.6f} {tag:6s} {conv}")
        print()


def sweep_rho(prob):
    """SWEEP 2: Rho sensitivity for phase-1 problem."""
    lb = prob['scalar_lb']
    t_val = lb + 0.02  # near boundary — hardest case

    print(f"{'='*70}")
    print("SWEEP 2: Rho sensitivity")
    print(f"  t={t_val:.6f} (near boundary)")
    print(f"{'='*70}")

    for rho in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        cache = [None]
        r = single_solve(prob, t_val, max_iters=3000, eps=5e-7,
                         rho=rho, solver_cache=cache)
        tag = "FEAS" if r['feasible'] else "INFEAS"
        conv = "CONV" if r['status'] == 'solved' else "cap"
        print(f"  rho={rho:5.2f}: {r['iters']:5d} iters {r['time']:6.1f}s "
              f"tau={r['tau']:+.6f} {tag:6s} {conv} "
              f"({r['ms_per_iter']:.1f}ms/it)")
    print()


def sweep_eps(prob):
    """SWEEP 3: Eps tolerance — looser eps = faster convergence."""
    lb = prob['scalar_lb']
    t_val = lb + 0.02

    print(f"{'='*70}")
    print("SWEEP 3: Eps tolerance")
    print(f"  t={t_val:.6f}, max_iters=3000")
    print(f"{'='*70}")

    for eps in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
        cache = [None]
        r = single_solve(prob, t_val, max_iters=3000, eps=eps,
                         rho=0.5, solver_cache=cache)
        tag = "FEAS" if r['feasible'] else "INFEAS"
        conv = "CONV" if r['status'] == 'solved' else "cap"
        print(f"  eps={eps:.0e}: {r['iters']:5d} iters {r['time']:6.1f}s "
              f"tau={r['tau']:+.6f} {tag:6s} {conv}")
    print()


def sweep_tau_threshold(prob):
    """SWEEP 4: Early tau infeasible threshold."""
    lb = prob['scalar_lb']

    print(f"{'='*70}")
    print("SWEEP 4: Early tau classification thresholds")
    print(f"  Testing at multiple t-values, max_iters=3000")
    print(f"{'='*70}")

    # Test several t-values and see when early tau triggers
    for label, t_val in [('far_infeas', lb - 0.02),
                          ('near_infeas', lb + 0.02),
                          ('near_feas', lb + 0.04)]:
        print(f"\n  {label} (t={t_val:.4f}):")
        # Run with verbose tau tracking — use a custom solve
        setup_t(prob, t_val)
        tau_col = prob['tau_col']
        solver = ADMMSolver(
            prob['A_template'], prob['b_template'], prob['c_p1'],
            prob['cone_p1'], rho=0.5, device='cuda', verbose=False)
        solver.solve(max_iters=100, eps_abs=1.0, eps_rel=1.0)

        # Now run with tau monitoring every 50 iters
        setup_t(prob, t_val)
        solver._update_A(prob['A_template'])
        solver.update_b(prob['b_template'])

        # Manual ADMM loop to track tau at each check
        ws = solver.ws
        n, m = solver.n, solver.m
        sigma, rho_v, alpha = solver.sigma, solver.rho, solver.alpha
        A_gpu, AT_gpu = solver.A_gpu, solver.AT_gpu
        b, c = solver.b, solver.c
        cone_info = solver.cone_info
        from admm_gpu_solver import _project_cones_gpu, _torch_cg

        def matvec(v):
            return sigma * v + rho_v * torch.mv(AT_gpu, torch.mv(A_gpu, v))
        cg_prev = torch.zeros(n, dtype=torch.float64, device='cuda')

        for k in range(1500):
            v = rho_v * (b - ws.s) - ws.y
            ATv = torch.mv(AT_gpu, v)
            rhs_v = sigma * ws.x + ATv - c
            _cg_maxiter = max(25, min(n // 1000, 100))
            x_new = _torch_cg(matvec, rhs_v, cg_prev, maxiter=_cg_maxiter,
                               tol=1e-10)
            cg_prev.copy_(x_new)

            Ax_new = torch.mv(A_gpu, x_new)
            v_hat = alpha * (b - Ax_new) + (1.0 - alpha) * ws.s
            s_input = v_hat - ws.y / rho_v
            _project_cones_gpu(s_input, cone_info)
            ws.y = ws.y + rho_v * (s_input - v_hat)
            ws.x = x_new
            ws.s = s_input

            if (k + 1) % 50 == 0:
                tau_now = ws.x[tau_col].item()
                Ax = torch.mv(A_gpu, ws.x)
                pri = torch.norm(Ax + ws.s - b).item()
                print(f"    iter {k+1:5d}: tau={tau_now:+.8f} pri={pri:.2e}")
    print()


def sweep_combined(prob):
    """SWEEP 5: Best config — combine best rho + eps + budget."""
    lb = prob['scalar_lb']

    print(f"{'='*70}")
    print("SWEEP 5: Combined best configs — 3-step bisection")
    print(f"{'='*70}")

    configs = [
        ('baseline',        {'rho': 0.5,  'eps': 5e-7, 'budgets': [800, 800, 800]}),
        ('rho=0.1',         {'rho': 0.1,  'eps': 5e-7, 'budgets': [800, 800, 800]}),
        ('rho=0.1+2K',      {'rho': 0.1,  'eps': 5e-7, 'budgets': [2000, 2000, 2000]}),
        ('rho=0.1+eps=1e-5', {'rho': 0.1, 'eps': 1e-5, 'budgets': [800, 800, 800]}),
        ('rho=0.1+eps=1e-5+2K', {'rho': 0.1, 'eps': 1e-5, 'budgets': [2000, 2000, 2000]}),
        ('rho=0.05+eps=1e-5+2K', {'rho': 0.05, 'eps': 1e-5, 'budgets': [2000, 2000, 2000]}),
    ]

    lo_init = lb - 0.01
    hi_init = lb + 0.05

    for name, cfg in configs:
        lo, hi = lo_init, hi_init
        cache = [None]
        total_t = 0
        total_it = 0
        results = []
        for step in range(3):
            mid = (lo + hi) / 2.0
            budget = cfg['budgets'][step]
            r = single_solve(prob, mid, rho=cfg['rho'],
                             max_iters=budget, eps=cfg['eps'],
                             solver_cache=cache)
            total_t += r['time']
            total_it += r['iters']
            results.append(r)
            if r['feasible']:
                hi = mid
            else:
                lo = mid

        step_strs = [f"{'F' if r['feasible'] else 'I'}({r['iters']})" for r in results]
        print(f"  {name:30s}: {total_t:6.1f}s {total_it:5d}it "
              f"[{lo:.6f},{hi:.6f}] steps={' '.join(step_strs)}")
    print()


if __name__ == '__main__':
    prob = build_problem()

    sweep_convergence(prob)
    sweep_rho(prob)
    sweep_eps(prob)
    sweep_tau_threshold(prob)
    sweep_combined(prob)

    print("ALL SWEEPS DONE")
