#!/usr/bin/env python
"""Profile CG residual reduction at each iteration, mid-ADMM-solve.

Shows exactly how much residual reduction each CG iteration buys,
using the actual RHS that arises during a real ADMM solve (not random).
This tells us where diminishing returns kick in for each problem size.
"""
import torch, numpy as np, sys, os, time
from scipy import sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem
from admm_gpu_solver import (
    _scipy_to_torch_csr, augment_phase1, ConeInfo,
    _project_cones_gpu, _torch_cg
)

device = 'cuda'

for d, bw in [(32, 16), (64, 16)]:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"d={d} bw={bw}")
    print(sep)

    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, 2, cliques, verbose=False)
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, True)
    A_p1, b_p1, c_p1, cone_p1, tau = augment_phase1(A_base, b_base, cone_base)

    m, n = A_p1.shape
    sigma, rho, alpha = 1e-6, 1.0, 1.6

    # Equilibrate
    A_work = A_p1.tocsc().copy()
    D = np.ones(m); E = np.ones(n)
    for _ in range(10):
        A_abs = A_work.copy(); A_abs.data = np.abs(A_abs.data)
        rn = np.array(A_abs.max(axis=1).todense()).ravel()
        dd = 1.0 / np.sqrt(np.maximum(rn, 1e-10)); dd = np.clip(dd, 1e-4, 1e4)
        A_work = sp.diags(dd) @ A_work; D *= dd
        A_abs = A_work.copy(); A_abs.data = np.abs(A_abs.data)
        cn = np.array(A_abs.max(axis=0).todense()).ravel()
        ee = 1.0 / np.sqrt(np.maximum(cn, 1e-10)); ee = np.clip(ee, 1e-4, 1e4)
        A_work = A_work @ sp.diags(ee); E *= ee

    b_sc = torch.tensor(D * b_p1, dtype=torch.float64, device=device)
    c_sc = torch.tensor(E * c_p1, dtype=torch.float64, device=device)
    A_gpu = _scipy_to_torch_csr(A_work.tocsc(), device)
    AT_gpu = _scipy_to_torch_csr(A_work.T.tocsc(), device)
    cone_info = ConeInfo(cone_p1, device)

    def mv(v):
        return sigma * v + rho * torch.mv(AT_gpu, torch.mv(A_gpu, v))

    # Run 50 ADMM iters to get realistic state
    x = torch.zeros(n, dtype=torch.float64, device=device)
    s = torch.zeros(m, dtype=torch.float64, device=device)
    y = torch.zeros(m, dtype=torch.float64, device=device)

    for k in range(50):
        v = rho * (b_sc - s) - y
        ATv = torch.mv(AT_gpu, v)
        rhs = sigma * x + ATv - c_sc
        x_new = _torch_cg(mv, rhs, x, maxiter=50, tol=1e-10)
        Ax = torch.mv(A_gpu, x_new)
        v_hat = alpha * (b_sc - Ax) + (1 - alpha) * s
        s_in = v_hat - y / rho
        _project_cones_gpu(s_in, cone_info)
        y = y + rho * (s_in - v_hat)
        x = x_new
        s = s_in

    # Now measure CG residual at each iteration on the CURRENT rhs
    v = rho * (b_sc - s) - y
    ATv = torch.mv(AT_gpu, v)
    rhs = sigma * x + ATv - c_sc

    x_cg = x.clone()
    r = rhs - mv(x_cg)
    p = r.clone()
    rs = torch.dot(r, r)
    r0 = rs.sqrt().item()

    print(f"CG initial residual: {r0:.2e}")
    print(f"{'iter':>5} {'resid':>12} {'rel_resid':>12}")
    checkpoints = {1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100}
    for k in range(100):
        Ap = mv(p)
        pAp = torch.dot(p, Ap)
        if pAp.abs() < 1e-30:
            break
        al = rs / pAp
        x_cg = x_cg + al * p
        r = r - al * Ap
        rs_new = torch.dot(r, r)
        cur = rs_new.sqrt().item()
        if (k + 1) in checkpoints:
            print(f"{k+1:5d} {cur:12.4e} {cur/r0:12.4e}")
        if cur < 1e-12:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
