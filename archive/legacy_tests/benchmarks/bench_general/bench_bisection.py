#!/usr/bin/env python
"""Benchmark bisection speed at d=32 bw=31 order=2.

Runs precompute once, adds a small batch of window violations,
then times exactly 3 bisection steps of a single CG round.
Reports per-step time, iterations, and total wall clock.

Usage:
    python tests/bench_bisection.py          # full benchmark
    python tests/bench_bisection.py --steps 1  # just 1 step (fastest)
"""
import sys, os, time, argparse
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
from admm_gpu_solver import ADMMSolver, augment_phase1


def run_benchmark(n_bisect_steps=3, max_iters=20000, eps=1e-7,
                   use_early_tau=True, use_warmup=True):
    d, order, bw = 32, 2, 31

    print(f"{'='*60}")
    print(f"BISECTION BENCHMARK: d={d} O{order} bw={bw}")
    print(f"  Steps={n_bisect_steps}, max_iters={max_iters}, eps={eps}")
    print(f"{'='*60}")

    # ── Phase 1: Precompute (timed separately) ──
    t0 = time.time()
    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, order, cliques, verbose=True)
    n_y = P['n_y']
    print(f"\nPrecompute: {time.time()-t0:.1f}s\n")

    # ── Phase 2: Build base problem ──
    t0 = time.time()
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, add_upper_loc=True)
    print(f"Base build: {time.time()-t0:.1f}s\n")

    # ── Phase 3: Get initial bound + violations via a quick Round 0 ──
    # Use ADMM on base problem to get scalar lb and violations
    from admm_gpu_solver import admm_solve
    t0 = time.time()
    sol = admm_solve(A_base, b_base, c_obj, cone_base,
                     max_iters=5000, eps_abs=1e-5, eps_rel=1e-5,
                     device='cuda', verbose=True)
    scalar_lb = float(sol['x'][meta['t_col']])
    y_vals = sol['x'][:n_y].copy()
    print(f"Round 0: lb={scalar_lb:.6f} ({time.time()-t0:.1f}s)\n")

    # Get violations
    active_windows = set()
    violations = _check_violations_highd(y_vals, scalar_lb, P, active_windows)
    n_add = min(30, len(violations))
    for w, eig in violations[:n_add]:
        active_windows.add(w)
    print(f"Adding {len(active_windows)} window PSD constraints\n")

    if not active_windows:
        print("No violations — nothing to bisect. Done.")
        return

    # ── Phase 4: Build augmented problem for bisection ──
    t0 = time.time()
    win_decomp = _precompute_window_psd_decomposition(P, active_windows)
    A_win_t1, b_win_t1, psd_win = _assemble_window_psd(win_decomp, 1.0)
    A_full_t1 = sp.vstack([A_base, A_win_t1], format='csc')
    A_full_t1.sort_indices()

    A_win_t2, _, _ = _assemble_window_psd(win_decomp, 2.0)
    A_full_t2 = sp.vstack([A_base, A_win_t2], format='csc')
    A_full_t2.sort_indices()

    full_t_data = A_full_t2.data - A_full_t1.data
    full_base_data = A_full_t1.data - full_t_data
    has_t_full = np.any(full_t_data != 0)

    b_full_base = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}

    # Build phase-1 augmented template
    n_cols_full = A_full_t1.shape[1]
    fix_t_row = sp.csc_matrix(([1.0], ([0], [meta['t_col']])),
                               shape=(1, n_cols_full))
    cone_fixed = {'z': cone_full['z'] + 1, 'l': cone_full['l'],
                  's': cone_full['s']}

    # Build at t=1 and t=2 to get decomposition
    np.add(full_base_data, 1.0 * full_t_data, out=A_full_t1.data)
    A_fixed_t1 = sp.vstack([fix_t_row, A_full_t1], format='csc')
    A_fixed_t1.sort_indices()
    b_fixed_t1 = np.insert(b_full_base, 0, 1.0)
    A_p1_t1, b_p1_t1, c_p1, cone_p1, tau_col = augment_phase1(A_fixed_t1, b_fixed_t1, cone_fixed)

    np.add(full_base_data, 2.0 * full_t_data, out=A_full_t1.data)
    A_fixed_t2 = sp.vstack([fix_t_row, A_full_t1], format='csc')
    A_fixed_t2.sort_indices()
    b_fixed_t2 = np.insert(b_full_base, 0, 2.0)
    A_p1_t2, b_p1_t2, _, _, _ = augment_phase1(A_fixed_t2, b_fixed_t2, cone_fixed)

    aug_t_data = A_p1_t2.data - A_p1_t1.data
    aug_base_data = A_p1_t1.data - aug_t_data
    aug_b_t_data = b_p1_t2 - b_p1_t1
    aug_b_base_data = b_p1_t1 - aug_b_t_data

    A_p1_template = A_p1_t1.copy()
    b_p1_template = b_p1_t1.copy()

    print(f"Problem build: {time.time()-t0:.1f}s")
    print(f"  n={A_p1_template.shape[1]:,}, m={A_p1_template.shape[0]:,}, nnz={A_p1_template.nnz:,}")
    print(f"  PSD cones: {cone_p1['s'][:5]}... ({len(cone_p1['s'])} total)")
    print(f"  tau_col={tau_col}\n")

    # ── Phase 5: BENCHMARK — time exactly n_bisect_steps ──
    lo = max(0.5, scalar_lb - 0.01)
    hi = scalar_lb + 0.05

    # Set up for bisection step at t=hi first (feasibility check)
    def setup_problem(t_val):
        np.add(aug_base_data, t_val * aug_t_data, out=A_p1_template.data)
        np.add(aug_b_base_data, t_val * aug_b_t_data, out=b_p1_template)

    import torch
    gpu_solver = None

    step_results = []
    print(f"{'='*60}")
    print(f"BISECTION BENCHMARK ({n_bisect_steps} steps)")
    print(f"  lo={lo:.6f}, hi={hi:.6f}")
    print(f"  max_iters={max_iters}, eps={eps}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # ── Warm-up solve at t=hi (clearly feasible) to seed workspace ──
    setup_problem(hi)
    gpu_solver = ADMMSolver(A_p1_template, b_p1_template, c_p1,
                            cone_p1, device='cuda', verbose=False)
    if use_warmup:
        t_wu = time.time()
        gpu_solver.solve(max_iters=100, eps_abs=1.0, eps_rel=1.0,
                         tau_col=tau_col if use_early_tau else None)
        print(f"  Warm-up solve (100 iters at t=hi): {time.time()-t_wu:.1f}s\n")

    for step in range(n_bisect_steps):
        mid = (lo + hi) / 2.0

        # Graduated budget (current production settings)
        if step < 3:
            cur_iters = min(max_iters, 800)
            cur_eps = eps * 5
        elif step < 6:
            cur_iters = min(max_iters, 2000)
            cur_eps = eps * 3
        else:
            cur_iters = min(max_iters, 5000)
            cur_eps = eps * 2

        t_step = time.time()

        # Update problem for this t_val
        setup_problem(mid)

        gpu_solver._update_A(A_p1_template)
        gpu_solver.update_b(b_p1_template)

        cur_tau_tol = max(cur_eps * 10, 1e-4)
        sol = gpu_solver.solve(max_iters=cur_iters, eps_abs=cur_eps,
                               eps_rel=cur_eps,
                               tau_col=tau_col if use_early_tau else None,
                               tau_tol=cur_tau_tol)

        tau_val = sol['x'][tau_col]
        feasible = (sol['info']['status'] in ('solved', 'solved_inaccurate')
                    and tau_val <= cur_tau_tol)

        dt = time.time() - t_step
        iters = sol['info']['iter']

        tag = "FEAS" if feasible else "INFEAS"
        print(f"  [{step+1}/{n_bisect_steps}] t={mid:.8f} {tag} "
              f"tau={tau_val:+.6f} ({iters} iters, {dt:.1f}s, "
              f"{dt/max(iters,1)*1000:.1f}ms/iter)")

        step_results.append({
            'step': step, 't': mid, 'feasible': feasible,
            'tau': tau_val, 'iters': iters, 'time': dt,
            'ms_per_iter': dt / max(iters, 1) * 1000,
        })

        if feasible:
            hi = mid
        else:
            lo = mid

    total_time = time.time() - t_total
    total_iters = sum(r['iters'] for r in step_results)
    avg_ms = total_time / max(total_iters, 1) * 1000

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Total: {total_time:.1f}s for {n_bisect_steps} steps")
    print(f"  Total iters: {total_iters:,}")
    print(f"  Avg ms/iter: {avg_ms:.1f}")
    print(f"  Avg time/step: {total_time/n_bisect_steps:.1f}s")
    print(f"  Interval: [{lo:.8f}, {hi:.8f}]")
    print(f"{'='*60}")

    return step_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--max-iters', type=int, default=20000)
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--no-early-tau', action='store_true',
                        help='Disable early tau classification (baseline)')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Disable warm-up solve (baseline)')
    args = parser.parse_args()
    run_benchmark(args.steps, args.max_iters, args.eps,
                  use_early_tau=not args.no_early_tau,
                  use_warmup=not args.no_warmup)
