#!/usr/bin/env python
"""Sweep CG maxiter at d=64 to find optimum."""
import torch, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem
from admm_gpu_solver import admm_solve, augment_phase1
import admm_gpu_solver as mod

cliques = _build_banded_cliques(64, 16)
P = _precompute_highd(64, 2, cliques, verbose=False)
A, b, c, cone, meta = build_base_problem(P, True)
A_p1, b_p1, c_p1, cone_p1, tau = augment_phase1(A, b, cone)

print(f"d=64: n={A_p1.shape[1]}, m={A_p1.shape[0]}, PSD cones={len(cone_p1['s'])}")
print(f"Cone sizes: {sorted(set(cone_p1['s']), reverse=True)[:5]}")
print()
print(f"{'CG_max':>7} {'ADMM_it':>8} {'time_s':>8} {'ms/it':>8} {'obj':>12} {'status':>20}")
print("-" * 70)

orig_cg = mod._torch_cg
for cg_max in [10, 20, 25, 30, 40, 50, 75, 100]:
    def patched_cg(matvec_fn, b, x0, maxiter=100, tol=1e-10, _cm=cg_max):
        return orig_cg(matvec_fn, b, x0, maxiter=_cm, tol=tol)
    mod._torch_cg = patched_cg

    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=1000, eps_abs=1e-4, eps_rel=1e-4,
                     device='cuda', verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    obj = sol['info']['pobj']
    status = sol['info']['status']
    print(f"{cg_max:7d} {it:8d} {dt:8.2f} {ms:8.1f} {obj:12.6f} {status:>20}")

mod._torch_cg = orig_cg
