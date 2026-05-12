#!/usr/bin/env python
r"""Comprehensive hyperparameter sweep for d=32 bw=31 (full Lasserre L2).

Target: find constants that maximize gap closure (>90%) at d=32.
val(32) = 1.336, so lb > 1 + 0.9*0.336 = 1.302 needed for 90% gc.
lb > 1.28 (current record) needs gc > 83.3%.

Structure:
  TEST 1: Build problem, measure dimensions and per-iteration costs
  TEST 2: Rho sweep (convergence vs accuracy tradeoff)
  TEST 3: Alpha sweep (over-relaxation)
  TEST 4: Sigma sweep
  TEST 5: CG maxiter sweep (at this n~59K scale)
  TEST 6: Check interval + AA parameters
  TEST 7: Best combo — short production run (2 CG rounds)
  TEST 8: Ruiz equilibration passes

Each test does SHORT solves (200-500 ADMM iters) to measure convergence
rate and solution quality, not full production runs.

Total target: <15 min on H100.
"""
import torch
import numpy as np
import sys
import os
import time
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, val_d_known,
)
from run_scs_direct import (
    build_base_problem, _precompute_window_psd_decomposition,
    _assemble_window_psd,
)
from admm_gpu_solver import (
    admm_solve, augment_phase1, ConeInfo, _project_cones_gpu,
    _scipy_to_torch_csr, _torch_cg,
)

device = 'cuda'
D, BW, ORDER = 32, 31, 2
VAL_D = val_d_known[D]  # 1.336

print("=" * 72)
print(f"HYPERPARAMETER SWEEP: d={D} bw={BW} L{ORDER} (full Lasserre)")
print(f"val({D}) = {VAL_D}, need lb > 1.28 (gc > 83.3%)")
print("=" * 72)

# ================================================================
# TEST 0: Build problem, report dimensions
# ================================================================
print(f"\n{'='*72}")
print("TEST 0: Problem construction + dimensions")
print("=" * 72)

t0 = time.time()
cliques = _build_banded_cliques(D, BW)
P = _precompute_highd(D, ORDER, cliques, verbose=True)
A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, True)
build_time = time.time() - t0

n_y = P['n_y']
n_x = meta['n_x']
m_base = A_base.shape[0]
psd_sizes = cone_base['s']
unique_psd = sorted(set(psd_sizes), reverse=True)

print(f"\nDimensions:")
print(f"  n_y = {n_y:,} moment variables")
print(f"  n_x = {n_x:,} (n_y + t)")
print(f"  m   = {m_base:,} constraint rows")
print(f"  nnz = {A_base.nnz:,}")
print(f"  PSD cones: {len(psd_sizes)} total")
for s in unique_psd[:5]:
    cnt = psd_sizes.count(s)
    print(f"    {cnt} x {s}x{s} (svec={s*(s+1)//2})")
print(f"  Build time: {build_time:.1f}s")
print(f"  Solver path: {'cholesky' if n_x < 5000 else 'CG'}")

# Phase-1 augment for testing
A_p1, b_p1, c_p1, cone_p1, tau_col = augment_phase1(A_base, b_base, cone_base)
m_p1, n_p1 = A_p1.shape
print(f"  After phase-1: {m_p1:,} x {n_p1:,}")

# ================================================================
# TEST 1: Per-iteration cost breakdown
# ================================================================
print(f"\n{'='*72}")
print("TEST 1: Per-iteration cost (50 iters, baseline constants)")
print("=" * 72)

# Warmup
sol_warmup = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                        max_iters=10, device=device)

torch.cuda.synchronize()
t0 = time.time()
sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                 max_iters=50, eps_abs=1e-8, eps_rel=1e-8,
                 device=device, verbose=False)
torch.cuda.synchronize()
dt = time.time() - t0
setup = sol['info'].get('setup_time', 0)
solve = sol['info'].get('solve_time', 0)
print(f"  50 iters: {dt:.2f}s total, setup={setup:.2f}s, solve={solve:.2f}s")
print(f"  Per-iter: {solve/50*1000:.1f}ms")

# eigh microbenchmark at the actual cone sizes
print(f"\n  eigh microbenchmarks:")
for s in unique_psd[:3]:
    cnt = psd_sizes.count(s)
    batch = torch.randn(cnt, s, s, dtype=torch.float64, device=device)
    batch = (batch + batch.transpose(-1, -2)) / 2
    torch.cuda.synchronize()
    for _ in range(3):
        torch.linalg.eigh(batch)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        torch.linalg.eigh(batch)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 50 * 1000
    print(f"    eigh({s}x{s}, batch={cnt}): {ms:.2f}ms")

    # cholesky_ex for comparison
    batch_psd = batch @ batch.transpose(-1, -2) + 0.01 * torch.eye(
        s, dtype=torch.float64, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        torch.linalg.cholesky_ex(batch_psd)
    torch.cuda.synchronize()
    ms_chol = (time.time() - t0) / 50 * 1000
    print(f"    cholesky_ex({s}x{s}, batch={cnt}): {ms_chol:.2f}ms "
          f"(skip ratio: {ms/max(ms_chol,0.001):.0f}x)")

# ================================================================
# TEST 2: Rho sweep
# ================================================================
print(f"\n{'='*72}")
print("TEST 2: Rho sweep (300 ADMM iters, eps=1e-5)")
print("  Key question: what rho gives convergence WITHOUT inaccuracy?")
print("=" * 72)

SWEEP_ITERS = 300
SWEEP_EPS = 1e-5

print(f"\n{'rho':>8} {'iters':>6} {'time':>7} {'ms/it':>7} "
      f"{'obj':>12} {'pri_r':>10} {'dual_r':>10} {'status':>20}")
print("-" * 95)

for rho in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=SWEEP_ITERS, eps_abs=SWEEP_EPS,
                     eps_rel=SWEEP_EPS, rho=rho, alpha=1.0, sigma=1e-6,
                     device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    obj = sol['info']['pobj']
    status = sol['info']['status']

    # Compute residuals for quality assessment
    x_t = torch.tensor(sol['x'], dtype=torch.float64, device=device)
    s_t = torch.tensor(sol['s'], dtype=torch.float64, device=device)
    y_t = torch.tensor(sol['y'], dtype=torch.float64, device=device)
    A_gpu = _scipy_to_torch_csr(A_p1.tocsc(), device)
    b_t = torch.tensor(b_p1, dtype=torch.float64, device=device)
    c_t = torch.tensor(c_p1, dtype=torch.float64, device=device)
    pri = torch.norm(torch.mv(A_gpu, x_t) + s_t - b_t).item()
    dual = torch.norm(torch.mv(
        _scipy_to_torch_csr(A_p1.T.tocsc(), device), y_t) + c_t).item()

    print(f"{rho:8.2f} {it:6d} {dt:7.1f} {ms:7.1f} "
          f"{obj:12.6f} {pri:10.2e} {dual:10.2e} {status:>20}")

# ================================================================
# TEST 3: Alpha sweep
# ================================================================
print(f"\n{'='*72}")
print("TEST 3: Alpha sweep (300 iters, rho from TEST 2 winner)")
print("=" * 72)

# Use rho=0.5 as middle ground for alpha test (adjust after seeing TEST 2)
TEST_RHO = 0.5

print(f"\n  Using rho={TEST_RHO}")
print(f"\n{'alpha':>8} {'iters':>6} {'time':>7} {'ms/it':>7} "
      f"{'obj':>12} {'status':>20}")
print("-" * 75)

for alpha in [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=SWEEP_ITERS, eps_abs=SWEEP_EPS,
                     eps_rel=SWEEP_EPS, rho=TEST_RHO, alpha=alpha,
                     sigma=1e-6, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"{alpha:8.2f} {it:6d} {dt:7.1f} {ms:7.1f} "
          f"{sol['info']['pobj']:12.6f} {sol['info']['status']:>20}")

# ================================================================
# TEST 4: Sigma sweep
# ================================================================
print(f"\n{'='*72}")
print("TEST 4: Sigma sweep (300 iters, rho=0.5, alpha=1.0)")
print("=" * 72)

print(f"\n{'sigma':>10} {'iters':>6} {'time':>7} {'ms/it':>7} "
      f"{'obj':>12} {'status':>20}")
print("-" * 75)

for sigma in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=SWEEP_ITERS, eps_abs=SWEEP_EPS,
                     eps_rel=SWEEP_EPS, rho=TEST_RHO, alpha=1.0,
                     sigma=sigma, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"{sigma:10.0e} {it:6d} {dt:7.1f} {ms:7.1f} "
          f"{sol['info']['pobj']:12.6f} {sol['info']['status']:>20}")

# ================================================================
# TEST 5: CG maxiter sweep (critical at n=59K)
# ================================================================
print(f"\n{'='*72}")
print("TEST 5: CG maxiter sweep (200 ADMM iters, rho=0.5, alpha=1.0)")
print("  n={:,} → default CG maxiter = {}".format(n_p1, max(25, n_p1 // 1000)))
print("=" * 72)

import admm_gpu_solver as mod
orig_cg = mod._torch_cg

print(f"\n{'cg_max':>8} {'iters':>6} {'time':>7} {'ms/it':>7} "
      f"{'obj':>12} {'status':>20}")
print("-" * 75)

for cg_max in [10, 20, 30, 40, 50, 60, 80, 100]:
    def patched_cg(matvec_fn, b, x0, maxiter=100, tol=1e-10, _cm=cg_max):
        return orig_cg(matvec_fn, b, x0, maxiter=_cm, tol=tol)
    mod._torch_cg = patched_cg

    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=200, eps_abs=SWEEP_EPS,
                     eps_rel=SWEEP_EPS, rho=TEST_RHO, alpha=1.0,
                     sigma=1e-6, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"{cg_max:8d} {it:6d} {dt:7.1f} {ms:7.1f} "
          f"{sol['info']['pobj']:12.6f} {sol['info']['status']:>20}")

mod._torch_cg = orig_cg

# ================================================================
# TEST 6: Check interval + AA parameters
# ================================================================
print(f"\n{'='*72}")
print("TEST 6: Check interval sweep (300 iters, best constants so far)")
print("=" * 72)

print(f"\n{'check_int':>10} {'iters':>6} {'time':>7} {'ms/it':>7} "
      f"{'obj':>12} {'status':>20}")
print("-" * 75)

for ci in [10, 25, 50, 100]:
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=SWEEP_ITERS, eps_abs=SWEEP_EPS,
                     eps_rel=SWEEP_EPS, rho=TEST_RHO, alpha=1.0,
                     sigma=1e-6, device=device, verbose=False,
                     check_interval=ci)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"{ci:10d} {it:6d} {dt:7.1f} {ms:7.1f} "
          f"{sol['info']['pobj']:12.6f} {sol['info']['status']:>20}")

# ================================================================
# TEST 7: Ruiz equilibration passes
# ================================================================
print(f"\n{'='*72}")
print("TEST 7: Ruiz equilibration passes (300 iters)")
print("  Currently hardcoded to 10 in admm_solve. Testing effect.")
print("=" * 72)

# We can't easily sweep Ruiz passes without modifying admm_solve,
# so test the effect indirectly by checking condition number proxy
A_csc = A_p1.tocsc().copy()
A_abs = A_csc.copy()
A_abs.data = np.abs(A_abs.data)
row_norms = np.array(A_abs.max(axis=1).todense()).ravel()
col_norms = np.array(A_abs.max(axis=0).todense()).ravel()
print(f"  Raw A: row norms [{row_norms.min():.2e}, {row_norms.max():.2e}], "
      f"ratio={row_norms.max()/max(row_norms.min(),1e-30):.1e}")
print(f"  Raw A: col norms [{col_norms.min():.2e}, {col_norms.max():.2e}], "
      f"ratio={col_norms.max()/max(col_norms.min(),1e-30):.1e}")
print("  (Ruiz aims to make these ratios close to 1)")

# ================================================================
# TEST 8: Best combo — short production run (2 CG rounds, 8 bisect)
# ================================================================
print(f"\n{'='*72}")
print("TEST 8: Best combo — 2 CG round production run")
print("  Testing if the solver can actually find the feasibility boundary")
print("=" * 72)

# For this test, we run the actual production pipeline with 2 CG rounds
# to verify the constants work end-to-end, not just on the base problem.
#
# We test 3 combos:
combos = [
    ("baseline",    dict(rho=1.0, alpha=1.6, sigma=1e-6)),
    ("rho=0.5/a=1", dict(rho=0.5, alpha=1.0, sigma=1e-6)),
    ("rho=0.1/a=1", dict(rho=0.1, alpha=1.0, sigma=1e-6)),
]

# We need to modify admm_solve defaults per combo.
# Simplest: call admm_solve directly from a mini bisection loop.

y_vals = np.zeros(n_y)
active = set()
viols = _check_violations_highd(y_vals, 1.0, P, active)
for w, _ in viols[:50]:
    active.add(w)

# Build problem with 50 window PSD cones
win_decomp = _precompute_window_psd_decomposition(P, active)
if win_decomp:
    A_win, _, psd_win = _assemble_window_psd(win_decomp, 1.0)
    A_full = sp.vstack([A_base, A_win], format='csc')
    A_full.sort_indices()
    b_full = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}
else:
    A_full = A_base
    b_full = b_base
    cone_full = cone_base

A_full_p1, b_full_p1, c_full_p1, cone_full_p1, tau_full = augment_phase1(
    A_full, b_full, cone_full)

print(f"  Full problem with 50 windows: {A_full_p1.shape[0]:,} x "
      f"{A_full_p1.shape[1]:,}, PSD cones={len(cone_full_p1['s'])}")
print()

for label, kwargs in combos:
    print(f"  --- {label} ---")

    # Test: can it distinguish feasible (t=2.0) from infeasible (t=0.5)?
    for t_val, expected in [(2.0, "feas"), (0.5, "infeas")]:
        # Update A for this t value
        if win_decomp and win_decomp['has_t']:
            A_t, _, _ = _assemble_window_psd(win_decomp, t_val)
            A_test = sp.vstack([A_base, A_t], format='csc')
            A_test.sort_indices()
        else:
            A_test = A_full
        A_t_p1, b_t_p1, c_t_p1, cone_t_p1, tau_t = augment_phase1(
            A_test, b_full, cone_full)

        torch.cuda.synchronize()
        t0 = time.time()
        sol = admm_solve(A_t_p1, b_t_p1, c_t_p1, cone_t_p1,
                         max_iters=500, eps_abs=1e-5, eps_rel=1e-5,
                         device=device, verbose=False, **kwargs)
        torch.cuda.synchronize()
        dt = time.time() - t0

        tau_val = sol['x'][tau_t] if sol['x'] is not None else float('inf')
        tau_tol = max(1e-5 * 10, 1e-4)
        is_feas = (sol['info']['status'] in ('solved', 'solved_inaccurate')
                   and tau_val <= tau_tol)
        tag = "feas" if is_feas else "infeas"
        correct = "OK" if tag == expected else "WRONG"

        print(f"    t={t_val}: tau={tau_val:+.6f} → {tag} (expected {expected}) "
              f"[{correct}] {sol['info']['iter']} iters, {dt:.1f}s")

    # Test at boundary (t ≈ val(32) ≈ 1.336)
    t_val = 1.2
    if win_decomp and win_decomp['has_t']:
        A_t, _, _ = _assemble_window_psd(win_decomp, t_val)
        A_test = sp.vstack([A_base, A_t], format='csc')
        A_test.sort_indices()
    else:
        A_test = A_full
    A_t_p1, b_t_p1, c_t_p1, cone_t_p1, tau_t = augment_phase1(
        A_test, b_full, cone_full)

    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_t_p1, b_t_p1, c_t_p1, cone_t_p1,
                     max_iters=1000, eps_abs=1e-5, eps_rel=1e-5,
                     device=device, verbose=False, **kwargs)
    torch.cuda.synchronize()
    dt = time.time() - t0

    tau_val = sol['x'][tau_t] if sol['x'] is not None else float('inf')
    is_feas = (sol['info']['status'] in ('solved', 'solved_inaccurate')
               and tau_val <= max(1e-5 * 10, 1e-4))
    tag = "feas" if is_feas else "infeas"
    print(f"    t={t_val}: tau={tau_val:+.6f} → {tag} "
          f"{sol['info']['iter']} iters, {dt:.1f}s")
    print()

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*72}")
print("SUMMARY: Recommended constants for d=32 bw=31 production run")
print("=" * 72)
print("""
Review the results above and pick:
  rho:   lowest value where TEST 2 shows 'solved' AND TEST 8 gives correct feas/infeas
  alpha: value from TEST 3 with fewest iters
  sigma: value from TEST 4 with fewest iters (usually no effect)
  cg_max: from TEST 5, lowest value where ADMM still converges
  check_interval: from TEST 6, largest value that doesn't miss convergence

For the production run:
  python tests/run_scs_direct.py --d 32 --order 2 --bw 31 \\
    --cg-rounds 10 --bisect 15 --gpu --scs-iters 5000 --scs-eps 1e-6
""")

total_time = time.time() - t0
print(f"Total sweep time: {total_time:.0f}s")
