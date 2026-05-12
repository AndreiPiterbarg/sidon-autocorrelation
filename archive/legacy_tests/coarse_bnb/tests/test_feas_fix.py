#!/usr/bin/env python
"""Definitive test: does fixing x[t_col]=t_val correctly detect infeasibility?"""
import torch, numpy as np, sys, os, time
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

print("DEFINITIVE FEASIBILITY TEST")
print("Fix: x[t_col] = t_val via equality constraint")
print("Expected: t=0.5 INFEAS, t=2.0 FEAS (min t* ~ 0.98)")
print()

print("WITH FIX (x[t_col] = t_val):")
hdr = f"{'t_val':>6} {'tau':>12} {'result':>8} {'iters':>6} {'time':>6}"
print(hdr)
print("-" * 50)

for t_val in [0.5, 0.8, 0.98, 1.0, 1.5, 2.0]:
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
    print(f"{t_val:6.2f} {tau:+12.8f} {tag:>8} {sol['info']['iter']:6d} {dt:6.1f}s")

print()
print("WITHOUT FIX (t free — should ALWAYS say FEAS, proving the bug):")
print(hdr)
print("-" * 50)

for t_val in [0.5, 2.0]:
    A_p1, b_p1, c_p1, cone_p1, tau_idx = augment_phase1(
        A_base, b_base, cone_base)
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=500, eps_abs=1e-5, eps_rel=1e-5,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    tau = sol['x'][tau_idx]
    tag = "FEAS" if tau <= 1e-4 else "INFEAS"
    print(f"{t_val:6.2f} {tau:+12.8f} {tag:>8} {sol['info']['iter']:6d}")
