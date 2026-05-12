#!/usr/bin/env python
"""Test feasibility detection fix: constrain x[t_col]=0 when t is baked into A.

The base problem has scalar window constraints f_W(y) - t <= 0 in the nonneg
cone. When we build A(t_val) = A_base + t_val * A_t, the t_val goes into
the PSD window entries but x[t_col] is STILL free and enters the scalar
nonneg rows. So the solver picks x[t_col] big -> always feasible.

Fix: add x[t_col] = 0 as an equality constraint. Then the scalar window
constraints become f_W(y) <= 0, which combined with y on simplex makes
the problem correctly infeasible when t_val is too small.

We test on the BASE problem only (no window PSD constraints yet — just
the scalar windows + base PSD). This is Round 0. min t* ~ 0.98 for d=32.
So t=0.5 should be infeasible, t=2.0 should be feasible.
"""
import torch
import numpy as np
import sys
import os
import time
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem
from admm_gpu_solver import admm_solve, augment_phase1

D, BW = 32, 31
device = 'cuda'

cliques = _build_banded_cliques(D, BW)
P = _precompute_highd(D, 2, cliques, verbose=False)
A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, True)
t_col = meta['n_x'] - 1
n_cols = A_base.shape[1]

print(f"d={D} bw={BW}: n={n_cols}, m={A_base.shape[0]}, t_col={t_col}")
print(f"min t* ~ 0.98 (from INV 2)")
print()

# Test 1: WITHOUT fix (t free) — should always say feasible
print("WITHOUT FIX (t free, phase-1):")
print(f"  {'t_val':>6} {'tau':>10} {'result':>8} {'iters':>6} {'time':>6}")
print("  " + "-" * 50)

for t_val in [0.5, 0.95, 1.5]:
    A_p1, b_p1, c_p1, cone_p1, tau_idx = augment_phase1(
        A_base, b_base, cone_base)
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=500, eps_abs=1e-5, eps_rel=1e-5,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    tau = sol['x'][tau_idx]
    tag = "FEAS" if tau <= 1e-4 else "INFEAS"
    print(f"  {t_val:6.2f} {tau:+10.6f} {tag:>8} {sol['info']['iter']:6d} "
          f"{sol['info']['solve_time']:6.1f}s")

# Test 2: WITH fix (x[t_col] = t_val via equality)
# The base problem has the t column. When we fix x[t_col] = t_val,
# the scalar window constraints f_W(y) - t <= 0 become f_W(y) - t_val <= 0.
# So we should fix x[t_col] = t_val, NOT 0.
print()
print("WITH FIX (x[t_col] = t_val via equality, phase-1):")
print(f"  {'t_val':>6} {'tau':>10} {'result':>8} {'iters':>6} {'time':>6}")
print("  " + "-" * 50)

for t_val in [0.5, 0.8, 0.95, 0.98, 1.0, 1.05, 1.2, 1.5, 2.0]:
    # Add equality: x[t_col] = t_val
    fix_row = sp.csc_matrix(([1.0], ([0], [t_col])), shape=(1, n_cols))
    A_fixed = sp.vstack([fix_row, A_base], format='csc')
    A_fixed.sort_indices()
    b_fixed = np.concatenate([[t_val], b_base])
    cone_fixed = {'z': cone_base['z'] + 1, 'l': cone_base['l'],
                  's': cone_base['s']}

    A_p1, b_p1, c_p1, cone_p1, tau_idx = augment_phase1(
        A_fixed, b_fixed, cone_fixed)

    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=1000, eps_abs=1e-5, eps_rel=1e-5,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    tau = sol['x'][tau_idx]
    tag = "FEAS" if tau <= 1e-4 else "INFEAS"
    print(f"  {t_val:6.2f} {tau:+10.6f} {tag:>8} {sol['info']['iter']:6d} "
          f"{dt:6.1f}s")

print()
print("Expected: t < ~0.98 should be INFEAS, t > ~0.98 should be FEAS")
