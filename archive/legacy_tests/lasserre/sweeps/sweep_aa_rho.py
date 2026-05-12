#!/usr/bin/env python
"""Sweep AA and rho parameters to find optimal convergence settings.

Tests at d=32 bw=16 (CG path) and d=16 bw=12 (Cholesky path).
"""
import torch, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem
from admm_gpu_solver import admm_solve, augment_phase1


def build_problem(d, bw):
    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, 2, cliques, verbose=False)
    A, b, c, cone, meta = build_base_problem(P, True)
    return augment_phase1(A, b, cone)


def test_config(A_p1, b_p1, c_p1, cone_p1, max_iters, eps,
                aa_mem, aa_interval, rho, alpha, sigma, adapt_mu, adapt_tau,
                adapt_interval, label=""):
    """Run one ADMM solve with specific hyperparameters."""
    # We need to monkey-patch the admm_solve internals.
    # Instead, call admm_solve with the parameters it exposes (rho, alpha, sigma)
    # and patch the AA/rho-adapt parameters.
    import admm_gpu_solver as mod

    # Save originals
    orig_aa_section = None

    # Patch: we'll modify the constants inside admm_solve by editing the source
    # at runtime. Cleaner: just call with the exposed params and accept we can't
    # change AA internals without refactoring. Let's test what we CAN change first.

    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=max_iters, eps_abs=eps, eps_rel=eps,
                     sigma=sigma, rho=rho, alpha=alpha,
                     device='cuda', verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    it = sol['info']['iter']
    ms = dt / max(it, 1) * 1000
    obj = sol['info']['pobj']
    status = sol['info']['status']
    print(f"  {label:>30s}: {it:5d} iters, {dt:6.2f}s, {ms:6.1f}ms/it, "
          f"obj={obj:+.6f}, {status}")
    return it, dt, status


def main():
    # ============================================================
    # d=32 bw=16
    # ============================================================
    print("=" * 70)
    print("d=32 bw=16 (CG path, 2000 max iters, eps=1e-4)")
    print("=" * 70)
    A, b, c, cone, tau = build_problem(32, 16)
    MI, EPS = 2000, 1e-4

    # Baseline
    print("\n--- Baseline (sigma=1e-6, rho=1.0, alpha=1.6) ---")
    test_config(A, b, c, cone, MI, EPS,
                aa_mem=5, aa_interval=5, rho=1.0, alpha=1.6,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="baseline")

    # Sweep rho
    print("\n--- Sweep initial rho (sigma=1e-6, alpha=1.6) ---")
    for rho in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        test_config(A, b, c, cone, MI, EPS,
                    aa_mem=5, aa_interval=5, rho=rho, alpha=1.6,
                    sigma=1e-6, adapt_mu=10, adapt_tau=2,
                    adapt_interval=100, label=f"rho={rho}")

    # Sweep alpha (over-relaxation)
    print("\n--- Sweep alpha (sigma=1e-6, rho=1.0) ---")
    for alpha in [1.0, 1.2, 1.4, 1.6, 1.8, 1.95]:
        test_config(A, b, c, cone, MI, EPS,
                    aa_mem=5, aa_interval=5, rho=1.0, alpha=alpha,
                    sigma=1e-6, adapt_mu=10, adapt_tau=2,
                    adapt_interval=100, label=f"alpha={alpha}")

    # Sweep sigma
    print("\n--- Sweep sigma (rho=1.0, alpha=1.6) ---")
    for sigma in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
        test_config(A, b, c, cone, MI, EPS,
                    aa_mem=5, aa_interval=5, rho=1.0, alpha=1.6,
                    sigma=sigma, adapt_mu=10, adapt_tau=2,
                    adapt_interval=100, label=f"sigma={sigma:.0e}")

    # Best combo candidates
    print("\n--- Combo candidates ---")
    test_config(A, b, c, cone, MI, EPS,
                aa_mem=5, aa_interval=5, rho=0.1, alpha=1.8,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="rho=0.1,alpha=1.8")
    test_config(A, b, c, cone, MI, EPS,
                aa_mem=5, aa_interval=5, rho=0.5, alpha=1.8,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="rho=0.5,alpha=1.8")
    test_config(A, b, c, cone, MI, EPS,
                aa_mem=5, aa_interval=5, rho=2.0, alpha=1.8,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="rho=2.0,alpha=1.8")
    test_config(A, b, c, cone, MI, EPS,
                aa_mem=5, aa_interval=5, rho=5.0, alpha=1.8,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="rho=5.0,alpha=1.8")

    # ============================================================
    # d=16 bw=12
    # ============================================================
    print("\n" + "=" * 70)
    print("d=16 bw=12 (Cholesky path, 3000 max iters, eps=1e-4)")
    print("=" * 70)
    A2, b2, c2, cone2, tau2 = build_problem(16, 12)
    MI2, EPS2 = 3000, 1e-4

    print("\n--- Baseline ---")
    test_config(A2, b2, c2, cone2, MI2, EPS2,
                aa_mem=5, aa_interval=5, rho=1.0, alpha=1.6,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="baseline")

    print("\n--- Sweep rho ---")
    for rho in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        test_config(A2, b2, c2, cone2, MI2, EPS2,
                    aa_mem=5, aa_interval=5, rho=rho, alpha=1.6,
                    sigma=1e-6, adapt_mu=10, adapt_tau=2,
                    adapt_interval=100, label=f"rho={rho}")

    print("\n--- Sweep alpha ---")
    for alpha in [1.0, 1.2, 1.4, 1.6, 1.8, 1.95]:
        test_config(A2, b2, c2, cone2, MI2, EPS2,
                    aa_mem=5, aa_interval=5, rho=1.0, alpha=alpha,
                    sigma=1e-6, adapt_mu=10, adapt_tau=2,
                    adapt_interval=100, label=f"alpha={alpha}")

    print("\n--- Best combos ---")
    test_config(A2, b2, c2, cone2, MI2, EPS2,
                aa_mem=5, aa_interval=5, rho=0.1, alpha=1.8,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="rho=0.1,alpha=1.8")
    test_config(A2, b2, c2, cone2, MI2, EPS2,
                aa_mem=5, aa_interval=5, rho=5.0, alpha=1.8,
                sigma=1e-6, adapt_mu=10, adapt_tau=2,
                adapt_interval=100, label="rho=5.0,alpha=1.8")


if __name__ == '__main__':
    main()
