"""Profile SCS GPU (cuDSS) bottlenecks in detail.

Measures:
1. Problem construction time (scipy sparse matrix building)
2. SCS object creation time (factorization / GPU memory transfer)
3. Per-iteration solve time
4. Cone projection breakdown (PSD eigendecomp vs linear system)
5. Scaling with problem size: n_y, PSD cone size, number of cones
6. Warm-start effectiveness
7. Memory usage (CPU RAM + GPU VRAM)

Usage:
    LD_LIBRARY_PATH=/tmp/libcudss-linux-x86_64-0.7.1.4_cuda12-archive/lib \
    python3 tests/profile_scs_gpu.py
"""
import sys
import os
import time
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem, _build_window_psd_block

import scs


def get_gpu_mem():
    """Get GPU memory usage in MB."""
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        used, total = r.stdout.strip().split(', ')
        return int(used), int(total)
    except Exception:
        return 0, 0


def get_cpu_mem_mb():
    """Get process RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0


def profile_config(d, order, bw, label, solvers=None):
    """Profile a single (d, order, bw) config across all stages."""
    if solvers is None:
        solvers = ['direct', 'cudss']

    print(f"\n{'='*70}")
    print(f"PROFILE: {label} (d={d} O{order} bw={bw})")
    print(f"{'='*70}")

    # 1. Precompute
    t0 = time.time()
    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, order, cliques, verbose=False)
    t_precompute = time.time() - t0
    print(f"\n[1] Precompute: {t_precompute:.3f}s, n_y={P['n_y']:,}")

    # 2. Build base SCS problem
    t0 = time.time()
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, add_upper_loc=True)
    t_build = time.time() - t0
    n_rows = A_base.shape[0]
    n_cols = A_base.shape[1]
    nnz = A_base.nnz
    psd_sizes = cone_base['s']
    max_psd = max(psd_sizes) if psd_sizes else 0
    total_psd_dim = sum(s * (s + 1) // 2 for s in psd_sizes)
    print(f"\n[2] Build A: {t_build:.3f}s")
    print(f"    Shape: {n_rows:,} x {n_cols:,}, nnz={nnz:,}")
    print(f"    PSD cones: {len(psd_sizes)} (max={max_psd}, "
          f"total_psd_entries={total_psd_dim:,})")
    print(f"    A memory: {A_base.data.nbytes/1e6:.1f}MB")

    gpu_used_before, gpu_total = get_gpu_mem()

    for solver_name in solvers:
        print(f"\n  --- Solver: {solver_name} ---")
        data = {'A': A_base, 'b': b_base, 'c': c_obj}
        kw = dict(max_iters=50000, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
        if solver_name != 'direct':
            kw['linear_solver'] = solver_name

        # 3. SCS object creation (includes factorization / GPU transfer)
        t0 = time.time()
        try:
            solver = scs.SCS(data, cone_base, **kw)
        except Exception as e:
            print(f"    SCS init FAILED: {e}")
            continue
        t_init = time.time() - t0

        gpu_used_after, _ = get_gpu_mem()
        gpu_delta = gpu_used_after - gpu_used_before

        print(f"  [3] SCS init: {t_init:.3f}s")
        if gpu_delta > 0:
            print(f"      GPU mem delta: +{gpu_delta}MB "
                  f"({gpu_used_after}/{gpu_total}MB)")

        # 4. First solve (cold)
        t0 = time.time()
        sol = solver.solve()
        t_solve_cold = time.time() - t0
        iters_cold = sol['info']['iter']
        status = sol['info']['status']
        obj = float(sol['x'][meta['t_col']]) if sol['x'] is not None else None

        print(f"  [4] Cold solve: {t_solve_cold:.3f}s, {iters_cold} iters, "
              f"status={status}")
        if iters_cold > 0:
            print(f"      Per-iter: {t_solve_cold/iters_cold*1000:.2f}ms")
        if obj is not None:
            print(f"      Objective: {obj:.8f}")

        # 5. Warm-start solve (same problem)
        if sol['x'] is not None:
            t0 = time.time()
            sol2 = solver.solve(warm_start=True,
                                x=sol['x'], y=sol['y'], s=sol['s'])
            t_solve_warm = time.time() - t0
            iters_warm = sol2['info']['iter']
            print(f"  [5] Warm solve (same): {t_solve_warm:.3f}s, "
                  f"{iters_warm} iters")

        # 6. Solve with slightly modified b (simulates bisection t change)
        b_mod = b_base.copy()
        b_mod[0] = 1.001  # slightly perturb y_0 constraint
        data_mod = {'A': A_base, 'b': b_mod, 'c': c_obj}
        try:
            solver_mod = scs.SCS(data_mod, cone_base, **kw)
            t0 = time.time()
            sol_mod = solver_mod.solve(warm_start=True,
                                       x=sol['x'], y=sol['y'], s=sol['s'])
            t_solve_mod = time.time() - t0
            iters_mod = sol_mod['info']['iter']
            print(f"  [6] Modified b solve: {t_solve_mod:.3f}s, "
                  f"{iters_mod} iters (warm from prev)")
        except Exception as e:
            print(f"  [6] Modified b FAILED: {e}")

    # 7. Window PSD cone scaling test
    print(f"\n  --- Window PSD scaling ---")
    active = set()
    # Find some covered windows
    for w in range(P['n_win']):
        if int(P['window_covering'][w]) >= 0:
            active.add(w)
            if len(active) >= 50:
                break

    for n_win_test in [10, 25, 50]:
        wins = set(list(active)[:n_win_test])
        if len(wins) < n_win_test:
            break

        t0 = time.time()
        A_win, b_win, psd_win = _build_window_psd_block(P, wins, t_val=1.0)
        t_build_win = time.time() - t0

        if A_win is not None:
            A_full = sp.vstack([A_base, A_win], format='csc')
            b_full = np.concatenate([b_base, b_win])
            cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                         's': list(cone_base['s']) + psd_win}

            total_rows = A_full.shape[0]
            total_nnz = A_full.nnz
            total_psd = len(cone_full['s'])

            data_win = {'A': A_full, 'b': b_full,
                        'c': np.zeros(meta['n_x'])}

            for solver_name in solvers:
                kw_w = dict(max_iters=50000, eps_abs=1e-5, eps_rel=1e-5,
                            verbose=False)
                if solver_name != 'direct':
                    kw_w['linear_solver'] = solver_name

                try:
                    t0 = time.time()
                    s_w = scs.SCS(data_win, cone_full, **kw_w)
                    t_init_w = time.time() - t0

                    t0 = time.time()
                    sol_w = s_w.solve()
                    t_solve_w = time.time() - t0

                    print(f"  {n_win_test} windows, {solver_name}: "
                          f"init={t_init_w:.2f}s, solve={t_solve_w:.2f}s, "
                          f"{sol_w['info']['iter']} iters, "
                          f"rows={total_rows:,}, nnz={total_nnz:,}, "
                          f"PSD={total_psd}")
                except Exception as e:
                    print(f"  {n_win_test} windows, {solver_name}: FAILED {e}")

    cpu_mem = get_cpu_mem_mb()
    gpu_used_final, _ = get_gpu_mem()
    print(f"\n  Final memory: CPU RSS={cpu_mem:.0f}MB, "
          f"GPU={gpu_used_final}MB/{gpu_total}MB")


def main():
    print(f"SCS {scs.__version__} GPU Profiler")
    print(f"Available modules: {[x for x in dir(scs) if x.startswith('_scs')]}")

    gpu_used, gpu_total = get_gpu_mem()
    print(f"GPU: {gpu_used}/{gpu_total}MB used")
    print()

    configs = [
        (4, 2, 3, "d=4 O2 (tiny)"),
        (4, 3, 3, "d=4 O3 (small)"),
        (8, 2, 4, "d=8 O2 (small)"),
        (8, 3, 4, "d=8 O3 (medium)"),
        (10, 3, 4, "d=10 O3 (medium-large)"),
        (12, 3, 4, "d=12 O3 (large)"),
    ]

    for d, order, bw, label in configs:
        profile_config(d, order, bw, label)

    # Summary
    print(f"\n\n{'='*70}")
    print("KEY FINDINGS:")
    print("- If GPU init time >> solve time: GPU transfer overhead dominates")
    print("- If per-iter GPU ~ per-iter CPU: GPU not helping (too small)")
    print("- If warm-start iters << cold iters: warm-start is effective")
    print("- If window PSD solve time grows linearly with n_windows:")
    print("  cone projection is the bottleneck (scales as sum(k^3))")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
