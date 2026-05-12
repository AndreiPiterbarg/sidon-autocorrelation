#!/usr/bin/env python
"""Tune AA hyperparameters and test diagonal preconditioning at d=32 bw=31.

Tests:
  1. AA on vs off in ADMMSolver (does it reduce iterations?)
  2. AA memory size (5, 7, 10)
  3. AA interval (3, 5, 10)
  4. Preconditioner on vs off (does it help CG convergence?)
  5. Best combo

Each test does a short ADMM solve (500-1000 iters) to measure convergence.
Total target: <10 min on H100.
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
import admm_gpu_solver as mod

D, BW = 32, 31
device = 'cuda'

print("=" * 72)
print(f"AA + PRECONDITIONER TUNING: d={D} bw={BW}")
print("=" * 72)

# Build problem
cliques = _build_banded_cliques(D, BW)
P = _precompute_highd(D, 2, cliques, verbose=False)
A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, True)
A_p1, b_p1, c_p1, cone_p1, tau_col = augment_phase1(A_base, b_base, cone_base)
print(f"n={A_p1.shape[1]}, m={A_p1.shape[0]}")

# Warmup
sol = admm_solve(A_p1, b_p1, c_p1, cone_p1, max_iters=10, device=device)

# ================================================================
# TEST 1: Preconditioner effect on CG convergence
# ================================================================
print(f"\n{'='*72}")
print("TEST 1: Preconditioner ON vs OFF (500 ADMM iters, rho=0.5)")
print("=" * 72)

# Test without preconditioner
orig_cg = mod._torch_cg

def cg_no_precond(matvec_fn, b, x0, maxiter=100, tol=1e-10, precond_inv=None):
    """CG ignoring preconditioner."""
    return orig_cg(matvec_fn, b, x0, maxiter=maxiter, tol=tol, precond_inv=None)

def cg_with_precond(matvec_fn, b, x0, maxiter=100, tol=1e-10, precond_inv=None):
    """CG using preconditioner."""
    return orig_cg(matvec_fn, b, x0, maxiter=maxiter, tol=tol,
                   precond_inv=precond_inv)

for label, cg_fn in [("NO precond", cg_no_precond),
                      ("WITH precond", cg_with_precond)]:
    mod._torch_cg = cg_fn
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=500, eps_abs=1e-6, eps_rel=1e-6,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"  {label:>15}: {it:4d} iters, {dt:6.1f}s, {ms:6.1f}ms/iter, "
          f"obj={sol['info']['pobj']:+.6f}, {sol['info']['status']}")

mod._torch_cg = orig_cg

# ================================================================
# TEST 2: AA on vs off in admm_solve (which uses AA)
# ================================================================
print(f"\n{'='*72}")
print("TEST 2: Anderson Acceleration ON vs OFF (1000 ADMM iters)")
print("=" * 72)

# admm_solve already has AA built in. Test by comparing:
# - Default (AA on, mem=5, interval=5)
# - AA disabled (patch to skip)

for label, aa_on in [("AA OFF", False), ("AA ON (mem=5,int=5)", True)]:
    # Patch the AA section by manipulating the interval
    if not aa_on:
        # Make aa_interval huge so it never fires
        orig_code = mod.admm_solve.__code__
        # Simpler: just run with huge aa_interval by modifying globals
        # Actually, cleanest: compare two separate runs
        pass

    # Since we can't easily disable AA inside admm_solve without refactoring,
    # let's test via the ADMMSolver class which NOW has AA (our change).
    # We can compare: ADMMSolver with AA vs without by temporarily removing it.
    pass

# Direct approach: run admm_solve twice - it has AA built in.
# For "AA OFF", we'll temporarily patch the AndersonAccelerator to be a no-op.
class NoopAA:
    """Dummy AA that always returns g_new (no acceleration)."""
    def __init__(self, *a, **kw):
        pass
    def step(self, x_new, g_new):
        return g_new

orig_AA = mod.AndersonAccelerator

for label, aa_cls in [("AA OFF", NoopAA),
                      ("AA ON (mem=5)", orig_AA)]:
    mod.AndersonAccelerator = aa_cls
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=1000, eps_abs=1e-6, eps_rel=1e-6,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"  {label:>20}: {it:4d} iters, {dt:6.1f}s, {ms:6.1f}ms/iter, "
          f"obj={sol['info']['pobj']:+.6f}, {sol['info']['status']}")

mod.AndersonAccelerator = orig_AA

# ================================================================
# TEST 3: AA memory size sweep
# ================================================================
print(f"\n{'='*72}")
print("TEST 3: AA memory size sweep (1000 ADMM iters, interval=5)")
print("=" * 72)

# To sweep AA memory, we need to patch the hardcoded aa_mem=5 in admm_solve.
# Cleanest: monkeypatch AndersonAccelerator.__init__ to override m.
for mem in [3, 5, 7, 10, 15]:
    class PatchedAA(orig_AA):
        _mem_override = mem
        def __init__(self, m, dim, device, dtype=torch.float64):
            super().__init__(self._mem_override, dim, device, dtype)

    mod.AndersonAccelerator = PatchedAA
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=1000, eps_abs=1e-6, eps_rel=1e-6,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"  mem={mem:2d}: {it:4d} iters, {dt:6.1f}s, {ms:6.1f}ms/iter, "
          f"obj={sol['info']['pobj']:+.6f}, {sol['info']['status']}")

mod.AndersonAccelerator = orig_AA

# ================================================================
# TEST 4: Combined — best AA + precond + rho
# ================================================================
print(f"\n{'='*72}")
print("TEST 4: Best combo (2000 ADMM iters, eps=1e-6)")
print("=" * 72)

combos = [
    ("baseline (rho=0.5, no AA patch)", dict(rho=0.5, alpha=1.0)),
    ("rho=0.1", dict(rho=0.1, alpha=1.0)),
    ("rho=0.2", dict(rho=0.2, alpha=1.0)),
    ("rho=0.5", dict(rho=0.5, alpha=1.0)),
    ("rho=1.0", dict(rho=1.0, alpha=1.0)),
]

for label, kwargs in combos:
    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=2000, eps_abs=1e-6, eps_rel=1e-6,
                     device=device, verbose=False, **kwargs)
    torch.cuda.synchronize()
    dt = time.time() - t0
    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    print(f"  {label:>35}: {it:4d} iters, {dt:6.1f}s, {ms:6.1f}ms/iter, "
          f"{sol['info']['status']}")

print(f"\n{'='*72}")
print("DONE")
print("=" * 72)
