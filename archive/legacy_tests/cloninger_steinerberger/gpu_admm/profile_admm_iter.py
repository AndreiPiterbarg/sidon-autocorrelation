#!/usr/bin/env python
"""Profile per-iteration breakdown of GPU ADMM solver.

Instruments the inner loop to measure:
  - SpMV time (A@x, AT@v)
  - Linear system solve (Cholesky or CG)
  - PSD cone projection (torch.linalg.eigh)
  - AA overhead (concat + lstsq)
  - Convergence check overhead
  - Total iteration time

Run: python tests/profile_admm_iter.py --d 16 --bw 12
     python tests/profile_admm_iter.py --d 32 --bw 16
"""
import sys, os, time, argparse
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, val_d_known,
)
from run_scs_direct import build_base_problem, _precompute_window_psd_decomposition, _assemble_window_psd
from admm_gpu_solver import (
    _scipy_to_torch_csr, ConeInfo, _project_cones_gpu,
    AndersonAccelerator, _torch_cg, augment_phase1,
)

import torch


def profile_iteration(d, bandwidth, order=2, n_iters=200, eps=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}"
          f" ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"d={d} bw={bandwidth} order={order} n_iters={n_iters}\n")

    # Build problem
    t0 = time.time()
    cliques = _build_banded_cliques(d, bandwidth)
    P = _precompute_highd(d, order, cliques, verbose=False)
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, True)
    print(f"Build: {time.time()-t0:.2f}s")

    # Add some window PSD cones to make it realistic
    y_dummy = np.zeros(P['n_y'])
    viols = _check_violations_highd(y_dummy, 1.0, P, set())
    active = set()
    for w, _ in viols[:50]:
        active.add(w)
    if active:
        win_decomp = _precompute_window_psd_decomposition(P, active)
        if win_decomp:
            A_win, _, psd_win = _assemble_window_psd(win_decomp, 1.0)
            A_full = sp.vstack([A_base, A_win], format='csc')
            A_full.sort_indices()
            b_full = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
            cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                         's': list(cone_base['s']) + psd_win}
        else:
            A_full, b_full, cone_full = A_base, b_base, cone_base
    else:
        A_full, b_full, cone_full = A_base, b_base, cone_base

    # Phase-1 augment
    A_p1, b_p1, c_p1, cone_p1, tau_col = augment_phase1(A_full, b_full, cone_full)

    m, n = A_p1.shape
    print(f"\nProblem: {m:,} rows x {n:,} cols, nnz={A_p1.nnz:,}")
    print(f"Cones: z={cone_p1['z']}, l={cone_p1['l']}, "
          f"PSD={len(cone_p1['s'])} (sizes: {sorted(set(cone_p1['s']), reverse=True)[:5]}...)")
    print(f"Solver path: {'cholesky' if n < 5000 else 'CG'}")

    # Setup on GPU
    sigma, rho, alpha = 1e-6, 1.0, 1.6

    A_gpu = _scipy_to_torch_csr(A_p1.tocsc(), device)
    A_T_csc = A_p1.T.tocsc()
    AT_gpu = _scipy_to_torch_csr(A_T_csc, device)
    b = torch.tensor(b_p1, dtype=torch.float64, device=device)
    c = torch.tensor(c_p1, dtype=torch.float64, device=device)
    cone_info = ConeInfo(cone_p1, device)

    use_dense = n < 5000
    if use_dense:
        ATA_dense = (AT_gpu @ A_gpu).to_dense()
        sigI = sigma * torch.eye(n, dtype=torch.float64, device=device)
        M = sigI + rho * ATA_dense
        L = torch.linalg.cholesky(M)

    # Workspace
    x = torch.zeros(n, dtype=torch.float64, device=device)
    s = torch.zeros(m, dtype=torch.float64, device=device)
    y = torch.zeros(m, dtype=torch.float64, device=device)

    aa_mem = 5
    aa = AndersonAccelerator(aa_mem, n + m + m, device)
    aa_interval = 5

    # Warm up GPU
    for _ in range(5):
        _project_cones_gpu(s.clone(), cone_info)
        torch.mv(A_gpu, x)
    torch.cuda.synchronize()

    # Profile n_iters iterations
    t_spmv = 0.0
    t_solve = 0.0
    t_psd = 0.0
    t_aa = 0.0
    t_check = 0.0
    t_other = 0.0
    check_interval = 25

    print(f"\nProfiling {n_iters} iterations...")
    torch.cuda.synchronize()
    t_total_start = time.time()

    for k in range(n_iters):
        # --- SpMV + RHS build ---
        torch.cuda.synchronize()
        t0 = time.time()
        v = rho * (b - s) - y
        ATv = torch.mv(AT_gpu, v)
        rhs = sigma * x + ATv - c
        torch.cuda.synchronize()
        t_spmv += time.time() - t0

        # --- Linear solve ---
        torch.cuda.synchronize()
        t0 = time.time()
        if use_dense:
            x_new = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        else:
            def matvec(v):
                return sigma * v + rho * torch.mv(AT_gpu, torch.mv(A_gpu, v))
            x_new = _torch_cg(matvec, rhs, x, maxiter=100, tol=1e-12)
        torch.cuda.synchronize()
        t_solve += time.time() - t0

        # --- s-update SpMV ---
        torch.cuda.synchronize()
        t0 = time.time()
        Ax_new = torch.mv(A_gpu, x_new)
        v_hat = alpha * (b - Ax_new) + (1.0 - alpha) * s
        s_input = v_hat - y / rho
        torch.cuda.synchronize()
        t1 = time.time()
        t_spmv += t1 - t0

        # --- PSD projection ---
        _project_cones_gpu(s_input, cone_info)
        torch.cuda.synchronize()
        t_psd += time.time() - t1

        s_new = s_input

        # --- Dual update ---
        torch.cuda.synchronize()
        t0 = time.time()
        y_new = y + rho * (s_new - v_hat)
        torch.cuda.synchronize()
        t_other += time.time() - t0

        # --- AA ---
        torch.cuda.synchronize()
        t0 = time.time()
        if (k + 1) % aa_interval == 0 and k > aa_mem * aa_interval:
            u_old = torch.cat([x, s, y])
            u_new = torch.cat([x_new, s_new, y_new])
            u_acc = aa.step(u_old, u_new)
            res_std = torch.norm(u_new - u_old)
            res_acc = torch.norm(u_acc - u_old)
            if res_acc <= 2.0 * res_std:
                x_new = u_acc[:n]
                s_new = u_acc[n:n + m]
                _project_cones_gpu(s_new, cone_info)
                y_new = u_acc[n + m:]
        torch.cuda.synchronize()
        t_aa += time.time() - t0

        x = x_new
        s = s_new
        y = y_new

        # --- Convergence check ---
        if (k + 1) % check_interval == 0:
            torch.cuda.synchronize()
            t0 = time.time()
            Ax = torch.mv(A_gpu, x)
            pri_res = torch.norm(Ax + s - b).item()
            AT_sdiff = torch.mv(AT_gpu, s)
            dual_res = (rho * torch.norm(AT_sdiff)).item()
            Ax_norm = torch.norm(Ax).item()
            s_norm = torch.norm(s).item()
            b_norm = torch.norm(b).item()
            ATy_norm = torch.norm(torch.mv(AT_gpu, y)).item()
            c_norm = torch.norm(c).item()
            torch.cuda.synchronize()
            t_check += time.time() - t0

    torch.cuda.synchronize()
    t_total = time.time() - t_total_start

    per_iter_ms = t_total / n_iters * 1000
    print(f"\n{'='*60}")
    print(f"PROFILE: d={d} bw={bandwidth} ({n_iters} iters)")
    print(f"{'='*60}")
    print(f"  Total:     {t_total*1000:8.1f} ms  ({per_iter_ms:.2f} ms/iter)")
    print(f"  SpMV:      {t_spmv*1000:8.1f} ms  ({t_spmv/t_total*100:5.1f}%)")
    print(f"  LinSolve:  {t_solve*1000:8.1f} ms  ({t_solve/t_total*100:5.1f}%)")
    print(f"  PSD proj:  {t_psd*1000:8.1f} ms  ({t_psd/t_total*100:5.1f}%)")
    print(f"  AA:        {t_aa*1000:8.1f} ms  ({t_aa/t_total*100:5.1f}%)")
    print(f"  ConvCheck: {t_check*1000:8.1f} ms  ({t_check/t_total*100:5.1f}%)")
    print(f"  Other:     {t_other*1000:8.1f} ms  ({t_other/t_total*100:5.1f}%)")
    print(f"{'='*60}")

    # PSD cone breakdown
    print(f"\nPSD cone details:")
    for mat_dim, group in sorted(cone_info.size_groups.items(), reverse=True):
        print(f"  {len(group):3d} cones of size {mat_dim:3d}x{mat_dim} "
              f"(svec_dim={mat_dim*(mat_dim+1)//2})")

    # Per-cone projection microbenchmark
    print(f"\nPSD projection microbenchmark (100 repeats):")
    s_test = torch.randn(m, dtype=torch.float64, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _project_cones_gpu(s_test.clone(), cone_info)
    torch.cuda.synchronize()
    psd_micro = (time.time() - t0) / 100 * 1000
    print(f"  PSD proj per call: {psd_micro:.3f} ms")

    # torch.linalg.eigh microbenchmark for the largest PSD size
    if cone_info.size_groups:
        largest = max(cone_info.size_groups.keys())
        n_cones = len(cone_info.size_groups[largest])
        batch = torch.randn(n_cones, largest, largest,
                            dtype=torch.float64, device=device)
        batch = (batch + batch.transpose(-1, -2)) / 2
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            torch.linalg.eigh(batch)
        torch.cuda.synchronize()
        eigh_micro = (time.time() - t0) / 100 * 1000
        print(f"  eigh({largest}x{largest}, batch={n_cones}): {eigh_micro:.3f} ms")

        # Also test FP32
        batch32 = batch.float()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            torch.linalg.eigh(batch32)
        torch.cuda.synchronize()
        eigh32_micro = (time.time() - t0) / 100 * 1000
        print(f"  eigh FP32({largest}x{largest}, batch={n_cones}): {eigh32_micro:.3f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--bw', type=int, default=12)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--iters', type=int, default=200)
    args = parser.parse_args()
    profile_iteration(d=args.d, bandwidth=args.bw, order=args.order,
                      n_iters=args.iters)
