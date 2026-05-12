#!/usr/bin/env python
"""Fast benchmark for GPU ADMM pipeline — measures setup + solve overhead.

Uses d=8 order=2 bw=6 (builds in <1s, solves in <1s per step).
Exercises the EXACT same code path as run_scs_direct.py --gpu:
  admm_solve() -> _scipy_to_torch_csr -> Cholesky -> ADMM loop -> cone projection

What we measure (per bisection step):
  - setup_ms: CSR transfer + A^T A + Cholesky factorization
  - solve_ms: ADMM iteration loop
  - aug_ms:   augment_phase1 sparse rebuild
  - total_ms: wall time for the step
"""
import sys, os, time
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, val_d_known,
)
from run_scs_direct import build_base_problem, _precompute_window_psd_decomposition, _assemble_window_psd


def run_benchmark(d=8, bandwidth=6, order=2, n_bisect=10, max_iters=1000,
                  eps=1e-4):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}"
          f" ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"d={d} bw={bandwidth} order={order} bisect={n_bisect} "
          f"iters={max_iters} eps={eps}\n")

    # Build
    t0 = time.time()
    cliques = _build_banded_cliques(d, bandwidth)
    P = _precompute_highd(d, order, cliques, verbose=False)
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, True)
    print(f"Build: {time.time()-t0:.2f}s  n={meta['n_x']} m={A_base.shape[0]} "
          f"nnz={A_base.nnz}")

    from admm_gpu_solver import admm_solve, augment_phase1

    # Round 0
    t0 = time.time()
    sol = admm_solve(A_base, b_base, c_obj, cone_base,
                     max_iters=max_iters, eps_abs=eps, eps_rel=eps,
                     device=device)
    r0_time = time.time() - t0
    lb = float(sol['x'][meta['t_col']]) if sol['info']['status'] in ('solved','solved_inaccurate') else 0.5
    y_vals = sol['x'][:P['n_y']].copy()
    active = set()
    viols = _check_violations_highd(y_vals, lb, P, active)
    print(f"Round0: lb={lb:.6f} viols={len(viols)} iters={sol['info']['iter']} "
          f"time={r0_time:.2f}s")

    if not viols:
        print("No violations."); return

    for w, _ in viols[:30]:
        active.add(w)

    # Precompute window decomposition
    win_decomp = _precompute_window_psd_decomposition(P, active)
    A_win_t1, b_win_t1, psd_win = _assemble_window_psd(win_decomp, 1.0)
    A_full_t1 = sp.vstack([A_base, A_win_t1], format='csc')
    A_full_t1.sort_indices()
    A_win_t2, _, _ = _assemble_window_psd(win_decomp, 2.0)
    A_full_t2 = sp.vstack([A_base, A_win_t2], format='csc')
    A_full_t2.sort_indices()
    full_t_data = A_full_t2.data - A_full_t1.data
    full_base_data = A_full_t1.data - full_t_data
    b_full = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}

    print(f"\nFull A: {A_full_t1.shape} nnz={A_full_t1.nnz}  "
          f"PSD cones: {len(cone_full['s'])}")

    # Bisection
    lo, hi = max(0.5, lb - 0.01), lb + 0.05
    setup_times, solve_times, aug_times, step_times, iters_list = [],[],[],[],[]

    print(f"\n{'step':>4} {'t':>10} {'tag':>6} {'it':>5} "
          f"{'aug_ms':>8} {'setup_ms':>9} {'solve_ms':>9} {'total_ms':>9}")
    print("-" * 72)

    for step in range(n_bisect):
        mid = (lo + hi) / 2
        t_all = time.time()

        np.add(full_base_data, mid * full_t_data, out=A_full_t1.data)

        t_aug = time.time()
        A_p1, b_p1, c_p1, cone_p1, tau_col = augment_phase1(
            A_full_t1, b_full, cone_full)
        aug_t = time.time() - t_aug

        sol_t = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                           max_iters=max_iters, eps_abs=eps, eps_rel=eps,
                           device=device)

        setup_t = sol_t['info'].get('setup_time', 0)
        solve_t = sol_t['info'].get('solve_time', 0)
        it = sol_t['info']['iter']
        total_t = time.time() - t_all

        setup_times.append(setup_t); solve_times.append(solve_t)
        aug_times.append(aug_t); step_times.append(total_t)
        iters_list.append(it)

        tau = sol_t['x'][tau_col] if sol_t['x'] is not None else 999
        feas = sol_t['info']['status'] in ('solved','solved_inaccurate') and tau <= max(eps*10, 1e-4)
        tag = "feas" if feas else "infeas"
        if feas: hi = mid
        else: lo = mid

        print(f"{step+1:4d} {mid:10.6f} {tag:>6} {it:5d} "
              f"{aug_t*1000:8.1f} {setup_t*1000:9.1f} "
              f"{solve_t*1000:9.1f} {total_t*1000:9.1f}")

    print(f"\n{'='*72}")
    print(f"SUMMARY d={d} bw={bandwidth}")
    print(f"  Avg aug:   {np.mean(aug_times)*1000:8.1f} ms")
    print(f"  Avg setup: {np.mean(setup_times)*1000:8.1f} ms")
    print(f"  Avg solve: {np.mean(solve_times)*1000:8.1f} ms")
    print(f"  Avg step:  {np.mean(step_times)*1000:8.1f} ms")
    print(f"  Avg iters: {np.mean(iters_list):8.0f}")
    print(f"  Total bisection: {sum(step_times):.2f}s")
    print(f"{'='*72}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--bw', type=int, default=6)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--bisect', type=int, default=10)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--eps', type=float, default=1e-4)
    args = parser.parse_args()
    run_benchmark(d=args.d, bandwidth=args.bw, order=args.order,
                  n_bisect=args.bisect, max_iters=args.max_iters, eps=args.eps)
