#!/usr/bin/env python
r"""Bottleneck discovery benchmark for d=16 L3 bw=15 GPU ADMM solver.

Runs a SHORT version of the actual solver pipeline and times each major
component to identify where optimization effort should go.

Components timed:
  1. Precompute (_precompute_highd)
  2. Base SCS problem build (build_base_problem)
  3. Window PSD decomposition (_precompute_window_psd_decomposition)
  4. Phase-1 augmentation (augment_phase1)
  5. Ruiz equilibration (inside ADMMSolver.__init__)
  6. GPU transfer (_update_A: sparse scale + tocsr + transfer)
  7. ATA dense computation + Cholesky factorization
  8. ADMM iteration (per-iter breakdown: linear solve, PSD proj, dual update)
  9. Convergence check overhead

Usage:
    python tests/bench_bottleneck.py
"""
import sys
import os
import time
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, enum_monomials, val_d_known,
)

# Import from the main solver
from run_d16_l3 import (
    build_base_problem, _precompute_window_psd_decomposition,
    _assemble_window_psd,
)

SQRT2 = np.sqrt(2.0)

D = 16
ORDER = 3
BW = 15
N_ADMM_ITERS = 200  # enough to see per-iter cost
N_BISECT_STEPS = 3  # just a few to time the loop


def fmt(t):
    if t < 0.001:
        return f"{t*1e6:.0f}us"
    if t < 1.0:
        return f"{t*1e3:.1f}ms"
    return f"{t:.2f}s"


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"\nBenchmark: d={D} O{ORDER} bw={BW}")
    print(f"{'='*70}\n")

    timings = {}

    # ── 1. Precompute ──
    print("[1] Precompute...", flush=True)
    t0 = time.time()
    cliques = _build_banded_cliques(D, BW)
    P = _precompute_highd(D, ORDER, cliques, verbose=False)
    n_y = P['n_y']
    timings['precompute'] = time.time() - t0
    print(f"    n_y = {n_y:,}, time = {fmt(timings['precompute'])}")

    # ── 2. Base problem build ──
    print("[2] Base problem build...", flush=True)
    t0 = time.time()
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, add_upper_loc=True)
    timings['base_build'] = time.time() - t0
    print(f"    A: {A_base.shape}, nnz={A_base.nnz:,}, time = {fmt(timings['base_build'])}")

    # ── 3. Round 0 solve (short) ──
    print("[3] Round 0 ADMM solve (200 iters)...", flush=True)
    from admm_gpu_solver import admm_solve, ADMMSolver, augment_phase1
    t0 = time.time()
    sol0 = admm_solve(A_base, b_base, c_obj, cone_base,
                      max_iters=N_ADMM_ITERS, eps_abs=1e-4, eps_rel=1e-4,
                      device=device, verbose=False)
    timings['round0_solve'] = time.time() - t0
    scalar_lb = float(sol0['x'][meta['t_col']]) if sol0['info']['status'] in ('solved', 'solved_inaccurate') else 0.5
    print(f"    lb={scalar_lb:.6f}, {sol0['info']['iter']} iters, time = {fmt(timings['round0_solve'])}")

    # ── Get violations for CG ──
    y_vals = sol0['x'][:n_y].copy()
    active_windows = set()
    violations = _check_violations_highd(y_vals, scalar_lb, P, active_windows)
    n_add = min(50, len(violations))
    for w, eig in violations[:n_add]:
        active_windows.add(w)
    print(f"    {len(violations)} violations, adding {n_add} windows")

    # ── 4. Window PSD decomposition ──
    print("[4] Window PSD decomposition...", flush=True)
    t0 = time.time()
    win_decomp = _precompute_window_psd_decomposition(P, active_windows)
    timings['win_psd_decomp'] = time.time() - t0
    print(f"    time = {fmt(timings['win_psd_decomp'])}")

    # ── 5. Full problem assembly ──
    print("[5] Full problem assembly (vstack + phase1)...", flush=True)
    t0 = time.time()
    A_win_t1, b_win_t1, psd_win = _assemble_window_psd(win_decomp, 1.0)
    A_full_t1 = sp.vstack([A_base, A_win_t1], format='csc')
    A_full_t1.sort_indices()
    timings['vstack'] = time.time() - t0
    print(f"    A_full: {A_full_t1.shape}, nnz={A_full_t1.nnz:,}, time = {fmt(timings['vstack'])}")

    # Phase-1 augmentation
    n_cols_full = A_full_t1.shape[1]
    fix_t_row = sp.csc_matrix(([1.0], ([0], [meta['t_col']])), shape=(1, n_cols_full))
    A_fixed = sp.vstack([fix_t_row, A_full_t1], format='csc')
    A_fixed.sort_indices()
    b_fixed = np.insert(np.concatenate([b_base, np.zeros(win_decomp['n_rows'])]), 0, 1.0)
    cone_full = {'z': cone_base['z'] + 1, 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}

    t0 = time.time()
    A_p1, b_p1, c_p1, cone_p1, tau_col = augment_phase1(A_fixed, b_fixed, cone_full)
    timings['phase1_augment'] = time.time() - t0
    print(f"    A_p1: {A_p1.shape}, nnz={A_p1.nnz:,}, time = {fmt(timings['phase1_augment'])}")

    # ── 6. ADMMSolver init (Ruiz + GPU transfer + ATA + Cholesky) ──
    print("[6] ADMMSolver.__init__ (Ruiz + GPU xfer + ATA + Cholesky)...", flush=True)
    t0 = time.time()
    solver = ADMMSolver(A_p1, b_p1, c_p1, cone_p1, rho=0.1, device=device, verbose=False)
    timings['solver_init'] = time.time() - t0
    print(f"    solver_type={solver._solver_type}, time = {fmt(timings['solver_init'])}")

    # ── 7. Detailed per-component timing inside ADMMSolver ──
    print("\n[7] Detailed component timing...", flush=True)

    # 7a. Ruiz scaling only
    t0 = time.time()
    D_ruiz, E_ruiz = solver._compute_ruiz(A_p1, cone_p1)
    timings['ruiz_scaling'] = time.time() - t0
    print(f"    Ruiz scaling: {fmt(timings['ruiz_scaling'])}")

    # 7b. sp.diags @ A @ sp.diags (the per-step sparse multiply)
    t0 = time.time()
    A_scaled = sp.diags(D_ruiz) @ A_p1 @ sp.diags(E_ruiz)
    A_scaled = A_scaled.tocsc()
    timings['sparse_scale'] = time.time() - t0
    print(f"    sp.diags(D)@A@sp.diags(E) + tocsc: {fmt(timings['sparse_scale'])}")

    # 7c. tocsr + GPU transfer
    t0 = time.time()
    A_csr = A_scaled.tocsr()
    timings['tocsr'] = time.time() - t0
    print(f"    tocsr: {fmt(timings['tocsr'])}")

    t0 = time.time()
    crow = torch.tensor(A_csr.indptr, dtype=torch.int64, device=device)
    col = torch.tensor(A_csr.indices, dtype=torch.int64, device=device)
    vals = torch.tensor(A_csr.data, dtype=torch.float64, device=device)
    A_gpu = torch.sparse_csr_tensor(crow, col, vals, size=A_csr.shape,
                                     dtype=torch.float64, device=device)
    torch.cuda.synchronize()
    timings['gpu_transfer'] = time.time() - t0
    print(f"    GPU transfer (A): {fmt(timings['gpu_transfer'])}")

    # 7d. Transpose
    t0 = time.time()
    AT_csr = A_scaled.T.tocsc().tocsr()
    timings['transpose_cpu'] = time.time() - t0
    print(f"    Transpose (CPU): {fmt(timings['transpose_cpu'])}")

    t0 = time.time()
    crow_t = torch.tensor(AT_csr.indptr, dtype=torch.int64, device=device)
    col_t = torch.tensor(AT_csr.indices, dtype=torch.int64, device=device)
    vals_t = torch.tensor(AT_csr.data, dtype=torch.float64, device=device)
    AT_gpu = torch.sparse_csr_tensor(crow_t, col_t, vals_t, size=AT_csr.shape,
                                      dtype=torch.float64, device=device)
    torch.cuda.synchronize()
    timings['gpu_transfer_AT'] = time.time() - t0
    print(f"    GPU transfer (AT): {fmt(timings['gpu_transfer_AT'])}")

    # 7e. ATA dense + Cholesky (only for small n — Cholesky path)
    n = A_p1.shape[1]
    if n < 5000:
        t0 = time.time()
        ATA = (AT_gpu @ A_gpu).to_dense()
        torch.cuda.synchronize()
        timings['ata_dense'] = time.time() - t0
        print(f"    ATA dense (sparse@sparse→dense): {fmt(timings['ata_dense'])}")

        sigma = 1e-6
        sigI = sigma * torch.eye(n, dtype=torch.float64, device=device)
        t0 = time.time()
        M = sigI + 0.1 * ATA
        L = torch.linalg.cholesky(M)
        torch.cuda.synchronize()
        timings['cholesky'] = time.time() - t0
        print(f"    Cholesky factorization: {fmt(timings['cholesky'])}")
    else:
        print(f"    [SKIP] ATA/Cholesky — n={n:,} uses CG path")

    # ── 8. Per-iteration ADMM breakdown (DETAILED per-component) ──
    print(f"\n[8] Per-iteration ADMM breakdown ({N_ADMM_ITERS} iters)...", flush=True)

    # First: total timing
    t0 = time.time()
    sol_bench = solver.solve(max_iters=N_ADMM_ITERS, eps_abs=1e-8, eps_rel=1e-8,
                              check_interval=1000)
    torch.cuda.synchronize()
    timings['admm_pure_iters'] = time.time() - t0
    ms_per_iter = timings['admm_pure_iters'] / N_ADMM_ITERS * 1000
    print(f"    {N_ADMM_ITERS} iters in {fmt(timings['admm_pure_iters'])}")
    print(f"    = {ms_per_iter:.2f} ms/iter")

    # Now instrument individual components
    from admm_gpu_solver import _project_cones_gpu, _torch_cg, AndersonAccelerator
    ws = solver.ws
    A_g = solver.A_gpu
    AT_g = solver.AT_gpu
    b_g = solver.b
    c_g = solver.c
    sig = solver.sigma
    rh = solver.rho
    al = solver.alpha
    ci = solver.cone_info
    n_s = solver.n
    m_s = solver.m

    # Matvec function for CG
    def matvec_s(v):
        return sig * v + rh * torch.mv(AT_g, torch.mv(A_g, v))

    precond = getattr(solver, '_precond_inv', None)

    N_COMP = 50  # fewer iters for component timing (sync is expensive)

    # Time each component separately
    torch.cuda.synchronize()
    t_xupdate = 0
    t_spmv_Ax = 0
    t_psd = 0
    t_dual = 0
    t_aa = 0
    cg_max = max(25, min(n_s // 1000, 100))
    cg_iters_total = 0

    s_prev_c = ws.s.clone()
    aa_c = AndersonAccelerator(5, n_s + m_s + m_s, device)

    for k in range(N_COMP):
        # x-update (CG solve)
        torch.cuda.synchronize()
        t_s = time.time()
        v = rh * (b_g - ws.s) - ws.y
        ATv = torch.mv(AT_g, v)
        rhs = sig * ws.x + ATv - c_g

        # Instrument CG iteration count
        x_cg = solver._cg_x_prev.clone()
        r_cg = rhs - matvec_s(x_cg)
        if precond is not None:
            z_cg = r_cg * precond
        else:
            z_cg = r_cg
        p_cg = z_cg.clone()
        rz = torch.dot(r_cg, z_cg)
        cg_count = 0
        for _ in range(cg_max):
            Ap = matvec_s(p_cg)
            pAp = torch.dot(p_cg, Ap)
            if pAp.abs() < 1e-30:
                break
            alpha_cg = rz / pAp
            x_cg = x_cg + alpha_cg * p_cg
            r_cg = r_cg - alpha_cg * Ap
            cg_count += 1
            if r_cg.norm() < 1e-10:
                break
            if precond is not None:
                z_cg = r_cg * precond
            else:
                z_cg = r_cg
            rz_new = torch.dot(r_cg, z_cg)
            p_cg = z_cg + (rz_new / rz) * p_cg
            rz = rz_new
        x_new = x_cg
        solver._cg_x_prev.copy_(x_new)
        torch.cuda.synchronize()
        t_xupdate += time.time() - t_s
        cg_iters_total += cg_count

        # s-update: A@x + PSD projection
        torch.cuda.synchronize()
        t_s = time.time()
        Ax_new = torch.mv(A_g, x_new)
        torch.cuda.synchronize()
        t_spmv_Ax += time.time() - t_s

        torch.cuda.synchronize()
        t_s = time.time()
        v_hat = al * (b_g - Ax_new) + (1.0 - al) * ws.s
        s_input = v_hat - ws.y / rh
        _project_cones_gpu(s_input, ci)
        s_new = s_input
        torch.cuda.synchronize()
        t_psd += time.time() - t_s

        # Dual update
        torch.cuda.synchronize()
        t_s = time.time()
        y_new = ws.y + rh * (s_new - v_hat)
        torch.cuda.synchronize()
        t_dual += time.time() - t_s

        # Anderson (every 5)
        if (k + 1) % 5 == 0 and k > 25:
            torch.cuda.synchronize()
            t_s = time.time()
            u_old = torch.cat([ws.x, ws.s, ws.y])
            u_new = torch.cat([x_new, s_new, y_new])
            u_acc = aa_c.step(u_old, u_new)
            torch.cuda.synchronize()
            t_aa += time.time() - t_s

        s_prev_c.copy_(ws.s)
        ws.x = x_new
        ws.s = s_new
        ws.y = y_new

    print(f"\n    Per-component ({N_COMP} iters, CG maxiter={cg_max}):")
    total_comp = t_xupdate + t_spmv_Ax + t_psd + t_dual + t_aa
    avg_cg = cg_iters_total / N_COMP
    print(f"      x-update (CG solve):  {t_xupdate/N_COMP*1000:.2f} ms/iter ({t_xupdate/total_comp*100:.0f}%), avg {avg_cg:.0f} CG iters")
    print(f"      SpMV (A@x):           {t_spmv_Ax/N_COMP*1000:.2f} ms/iter ({t_spmv_Ax/total_comp*100:.0f}%)")
    print(f"      PSD projection:       {t_psd/N_COMP*1000:.2f} ms/iter ({t_psd/total_comp*100:.0f}%)")
    print(f"      Dual update:          {t_dual/N_COMP*1000:.2f} ms/iter ({t_dual/total_comp*100:.0f}%)")
    print(f"      Anderson accel:       {t_aa/N_COMP*1000:.2f} ms/iter ({t_aa/total_comp*100:.0f}%)")
    print(f"      TOTAL:                {total_comp/N_COMP*1000:.2f} ms/iter")

    # ── 9. Simulated bisection step (the full _update_A path) ──
    print(f"\n[9] Simulated bisection step (_update_A)...", flush=True)
    # This is what happens EACH bisection step
    t0 = time.time()
    solver._update_A(A_p1)  # re-scales, transfers, refactors
    torch.cuda.synchronize()
    timings['update_A_full'] = time.time() - t0
    print(f"    _update_A (full): {fmt(timings['update_A_full'])}")

    # Break it down: update_b is cheap
    t0 = time.time()
    solver.update_b(b_p1)
    torch.cuda.synchronize()
    timings['update_b'] = time.time() - t0
    print(f"    update_b: {fmt(timings['update_b'])}")

    # ── 10. Full bisection step simulation ──
    print(f"\n[10] Full bisection step (update_A + 800 iter solve)...", flush=True)
    t0 = time.time()
    solver._update_A(A_p1)
    solver.update_b(b_p1)
    sol_bisect = solver.solve(max_iters=800, eps_abs=1e-5, eps_rel=1e-5,
                               tau_col=tau_col, tau_tol=1e-4)
    torch.cuda.synchronize()
    timings['full_bisect_step'] = time.time() - t0
    print(f"    Total: {fmt(timings['full_bisect_step'])}")
    print(f"    Status: {sol_bisect['info']['status']}, iters: {sol_bisect['info']['iter']}")

    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print("BOTTLENECK SUMMARY")
    print(f"{'='*70}")

    # Sort by time
    sorted_t = sorted(timings.items(), key=lambda x: -x[1])
    total = sum(v for k, v in timings.items()
                if k not in ('admm_with_checks', 'full_bisect_step'))

    for name, t in sorted_t:
        pct = t / total * 100 if total > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {name:30s} {fmt(t):>10s} {pct:5.1f}% {bar}")

    print(f"\n  PER BISECTION STEP BREAKDOWN:")
    step_components = ['update_A_full', 'admm_pure_iters']
    step_total = sum(timings.get(k, 0) for k in step_components)
    for k in step_components:
        t = timings.get(k, 0)
        pct = t / step_total * 100 if step_total > 0 else 0
        print(f"    {k:30s} {fmt(t):>10s} ({pct:.0f}%)")
    print(f"    {'TOTAL':30s} {fmt(step_total):>10s}")

    # Key ratios
    print(f"\n  KEY RATIOS:")
    if 'sparse_scale' in timings and 'admm_pure_iters' in timings:
        ratio = timings['sparse_scale'] / (timings['admm_pure_iters'] / N_ADMM_ITERS)
        print(f"    sparse_scale / per_iter = {ratio:.0f}x")
    if 'update_A_full' in timings and 'admm_pure_iters' in timings:
        ratio = timings['update_A_full'] / (timings['admm_pure_iters'] / N_ADMM_ITERS)
        print(f"    update_A / per_iter = {ratio:.0f}x (equiv to N extra iters)")
    if 'ata_dense' in timings and 'cholesky' in timings:
        print(f"    ATA_dense / cholesky = {timings['ata_dense']/timings['cholesky']:.1f}x")

    # Estimate: what a 12-step bisection costs
    n_bisect = 12
    bisect_cost = n_bisect * timings.get('update_A_full', 0)
    solve_cost = n_bisect * 800 * (timings.get('admm_pure_iters', 0) / N_ADMM_ITERS)
    print(f"\n  ESTIMATED 12-STEP BISECTION:")
    print(f"    update_A overhead: {fmt(bisect_cost)} ({bisect_cost/(bisect_cost+solve_cost)*100:.0f}%)")
    print(f"    solve time:        {fmt(solve_cost)} ({solve_cost/(bisect_cost+solve_cost)*100:.0f}%)")
    print(f"    total:             {fmt(bisect_cost + solve_cost)}")


if __name__ == '__main__':
    main()
