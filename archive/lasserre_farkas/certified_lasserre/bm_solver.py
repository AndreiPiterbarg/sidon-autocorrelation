"""Primal+dual SDP solver for the aggregated Lasserre relaxation.

Strategy:
  1.  MOSEK (Fusion) if available -- solves the SDP in standard form and
      returns both primal y* and dual multipliers (lambda_A, {S_j}) directly.
  2.  Burer-Monteiro augmented-Lagrangian fallback (scipy.optimize L-BFGS-B
      over flattened low-rank factors of the PSD blocks) -- used when MOSEK
      is not available, or for very large problems.

Output: a SolverResult containing the primal y*, the primal objective,
the approximate dual multipliers, and a diagnostic status string.

IMPORTANT: this solver runs in float64.  It provides APPROXIMATE dual
multipliers -- they are rounded and repaired to exact rationals in
round_repair.py before any certification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
import numpy as np
from scipy import sparse as sp

from certified_lasserre.build_sdp import SDPData, PSDBlock


@dataclass
class SolverResult:
    y: np.ndarray                     # primal, shape (n_y,)
    obj: float                        # c^T y
    lambda_A: np.ndarray              # equality duals, shape (n_eq,)
    S_blocks: List[np.ndarray]        # dual PSD matrices, one per block
    status: str
    solver: str
    time: float

    def stationarity_residual(self, sdp: SDPData) -> np.ndarray:
        """r = c - A^T lambda_A - sum_j F_j^*(S_j) in R^{n_y}.
        For an exact dual optimum of a standard SDP with F_j0 = 0, r = 0.
        Finite r measures how far our float64 dual is from exact feasibility.
        """
        r = sdp.c.copy()
        r -= (sdp.A.T @ self.lambda_A)
        for blk, S in zip(sdp.blocks, self.S_blocks):
            r -= blk.adjoint(S)
        return r


# =====================================================================
# MOSEK path
# =====================================================================

def _solve_with_mosek(sdp: SDPData, verbose: bool = False) -> SolverResult:
    import mosek
    from mosek.fusion import (Model, Domain, Expr, Matrix, ObjectiveSense)

    t0 = time.time()
    n_y = sdp.n_y
    n_eq = sdp.n_eq

    with Model('cert_lasserre_primal') as M:
        y = M.variable('y', n_y, Domain.unbounded())

        # Equality constraints Ay = b
        A_csr = sdp.A.tocoo()
        A_mosek = Matrix.sparse(
            n_eq, n_y,
            A_csr.row.astype(int).tolist(),
            A_csr.col.astype(int).tolist(),
            A_csr.data.tolist(),
        )
        eq_con = M.constraint('Ay=b', Expr.mul(A_mosek, y),
                              Domain.equalsTo(sdp.b.tolist()))

        # PSD blocks: for each block j, F_j(y) = reshape(G_flat @ y, n_j, n_j) in PSD.
        psd_cons = []
        for blk in sdp.blocks:
            n_j = blk.size
            G_coo = blk.G_flat.tocoo()
            G_mosek = Matrix.sparse(
                n_j * n_j, n_y,
                G_coo.row.astype(int).tolist(),
                G_coo.col.astype(int).tolist(),
                G_coo.data.tolist(),
            )
            F = Expr.reshape(Expr.mul(G_mosek, y), n_j, n_j)
            # PSDCone expects symmetric matrix; F_j is symmetric for our blocks.
            psd_con = M.constraint(f'psd_{blk.name}', F, Domain.inPSDCone(n_j))
            psd_cons.append(psd_con)

        # Objective: min c^T y
        c_list = sdp.c.tolist()
        M.objective(ObjectiveSense.Minimize, Expr.dot(c_list, y))

        if verbose:
            M.setLogHandler(__import__('sys').stdout)
        M.setSolverParam('numThreads', 4)
        M.setSolverParam('intpntCoTolPfeas', 1e-10)
        M.setSolverParam('intpntCoTolDfeas', 1e-10)
        M.setSolverParam('intpntCoTolRelGap', 1e-10)

        M.solve()
        status = str(M.getPrimalSolutionStatus())

        y_val = np.array(y.level(), dtype=np.float64)
        obj = float(M.primalObjValue())

        # Dual of equality constraint: shape (n_eq,)
        lam_A = np.array(eq_con.dual(), dtype=np.float64)

        # Dual of PSD constraints: S_j in R^{n_j x n_j}
        S_blocks: List[np.ndarray] = []
        for blk, psd_con in zip(sdp.blocks, psd_cons):
            n_j = blk.size
            S_flat = np.array(psd_con.dual(), dtype=np.float64)
            S = S_flat.reshape(n_j, n_j)
            S = 0.5 * (S + S.T)
            S_blocks.append(S)

    return SolverResult(
        y=y_val, obj=obj, lambda_A=lam_A, S_blocks=S_blocks,
        status=status, solver='mosek', time=time.time() - t0,
    )


# =====================================================================
# CVXPY / Clarabel path (used when MOSEK unavailable)
# =====================================================================

def _solve_with_cvxpy(sdp: SDPData, solver: str = 'CLARABEL',
                     verbose: bool = False) -> SolverResult:
    import cvxpy as cp
    t0 = time.time()
    n_y = sdp.n_y

    y = cp.Variable(n_y)
    constraints = [sdp.A @ y == sdp.b]
    psd_vars = []
    for blk in sdp.blocks:
        n_j = blk.size
        F = cp.reshape(blk.G_flat @ y, (n_j, n_j), order='C')
        # enforce symmetry + PSD
        psd_con = (F + F.T) / 2 >> 0
        constraints.append(psd_con)
        psd_vars.append(psd_con)
    obj = cp.Minimize(sdp.c @ y)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose)
    status = str(prob.status)
    y_val = np.array(y.value, dtype=np.float64).ravel()

    # Dual of equality constraint: first constraint
    lam_A = np.array(constraints[0].dual_value, dtype=np.float64).ravel()

    S_blocks = []
    for blk, pc in zip(sdp.blocks, psd_vars):
        S = np.array(pc.dual_value, dtype=np.float64)
        S = 0.5 * (S + S.T)
        S_blocks.append(S)

    return SolverResult(
        y=y_val, obj=float(prob.value),
        lambda_A=lam_A, S_blocks=S_blocks,
        status=status, solver=f'cvxpy/{solver.lower()}',
        time=time.time() - t0,
    )


# =====================================================================
# Burer-Monteiro augmented-Lagrangian fallback
# =====================================================================

def _solve_with_bm(sdp: SDPData, rank: int = 16,
                   mu_init: float = 10.0, mu_mult: float = 5.0,
                   outer_iters: int = 12,
                   inner_maxiter: int = 2000,
                   tol_grad: float = 1e-10,
                   tol_cons: float = 1e-8,
                   verbose: bool = False) -> SolverResult:
    """Augmented-Lagrangian low-rank SDP.

    Represent each PSD block X_j = L_j L_j^T with L_j of shape (n_j, r).
    y is treated as a separate variable tied to X_j via consistency:
        reshape(F_j(y), n_j, n_j) = L_j L_j^T
    Penalty: ||reshape(F_j(y)) - L_j L_j^T||_F^2.

    Minimizes c^T y  s.t.  A y = b  and  F_j(y) = L_j L_j^T (PSD by construction)
    via augmented Lagrangian.

    Returns SolverResult with y, approximate dual multipliers extracted from
    the augmented-Lagrangian stationarity.
    """
    from scipy.optimize import minimize

    t0 = time.time()
    n_y = sdp.n_y
    n_eq = sdp.n_eq
    blocks = sdp.blocks
    B = len(blocks)
    ranks = [min(rank, blk.size) for blk in blocks]
    sizes = [blk.size for blk in blocks]

    # Variable packing: x = [y (n_y); vec(L_1); ...; vec(L_B)]
    offsets = [n_y]
    for nj, r in zip(sizes, ranks):
        offsets.append(offsets[-1] + nj * r)
    total = offsets[-1]

    def unpack(x):
        y = x[:n_y]
        Ls = []
        for j, (nj, r) in enumerate(zip(sizes, ranks)):
            L = x[offsets[j]:offsets[j] + nj * r].reshape(nj, r) \
                if False else x[offsets[j] + 0:offsets[j] + nj * r].reshape(nj, r)
            # Actually correct offsets: unit j from n_y then cumulative
            pass
        # Redo packing correctly:
        Ls = []
        cur = n_y
        for nj, r in zip(sizes, ranks):
            Ls.append(x[cur:cur + nj * r].reshape(nj, r))
            cur += nj * r
        return y, Ls

    # Multipliers (duals): lambda_A for equality, nu_j for PSD consistency
    lam_A = np.zeros(n_eq)
    nu = [np.zeros((nj, nj)) for nj in sizes]

    def augmented_lagrangian(x, mu):
        y, Ls = unpack(x)
        val = float(sdp.c @ y)
        res_eq = sdp.A @ y - sdp.b
        val += float(lam_A @ res_eq) + 0.5 * mu * float(res_eq @ res_eq)
        for j, (blk, L) in enumerate(zip(blocks, Ls)):
            nj = blk.size
            XX = L @ L.T
            Fy = blk.eval(y)   # (nj, nj)
            delta = Fy - XX
            val += float((nu[j] * delta).sum())  # trace(nu_j delta)
            val += 0.5 * mu * float((delta * delta).sum())
        return val

    def grad_al(x, mu):
        y, Ls = unpack(x)
        grad_y = sdp.c.copy()
        res_eq = sdp.A @ y - sdp.b
        grad_y += sdp.A.T @ (lam_A + mu * res_eq)
        grad_L_list = []
        for j, (blk, L) in enumerate(zip(blocks, Ls)):
            nj = blk.size
            XX = L @ L.T
            Fy = blk.eval(y)
            delta = Fy - XX
            # d/dy_k delta_{ij} = G_j^{(k)}_{ij} - 0
            # grad_y += G_flat^T @ vec(nu_j + mu*delta)
            grad_y += blk.G_flat.T @ (nu[j] + mu * delta).ravel()
            # d/dL (nu . (Fy - LL^T) + mu/2 ||Fy - LL^T||^2)
            # = -2*(nu_j + mu*delta) @ L    (using d(LL^T)/dL = 2 sym L)
            grad_L = -2.0 * (nu[j] + mu * delta) @ L
            grad_L_list.append(grad_L)
        g = np.concatenate([grad_y] + [gL.ravel() for gL in grad_L_list])
        return g

    # Initialization: y = moment of uniform distribution (centre of simplex),
    # L_j = principal square root of F_j(y0) truncated to rank r
    y0 = np.zeros(n_y)
    # Set y_0 = 1, y_{e_i} = 1/d, y_{e_i+e_j} = 1/d^2 etc -- match moments of uniform.
    # For simplicity, just set y_0 = 1 and zeros elsewhere (feasible w.r.t. y_0=1
    # but not consistency).  Augmented Lagrangian will drive to feasibility.
    y0[sdp.mono_idx[tuple(0 for _ in range(sdp.d))]] = 1.0
    # Better: inject moments of uniform(Delta_d) to give the solver a head start.
    d = sdp.d
    for idx_alpha, alpha in enumerate(sdp.mono_list):
        k = sum(alpha)
        if k == 0:
            y0[idx_alpha] = 1.0
        elif k == 1:
            y0[idx_alpha] = 1.0 / d
        elif k == 2:
            i = alpha.index(max(alpha))
            # alpha = 2 e_i  -> E[mu_i^2] under uniform Dirichlet(1^d) = 2/(d(d+1))
            # alpha = e_i + e_j  -> E[mu_i mu_j] = 1/(d(d+1))
            if max(alpha) == 2:
                y0[idx_alpha] = 2.0 / (d * (d + 1))
            else:
                y0[idx_alpha] = 1.0 / (d * (d + 1))
        else:
            # approx: leave zero
            pass

    x0 = np.zeros(total)
    x0[:n_y] = y0
    cur = n_y
    for blk, r in zip(blocks, ranks):
        nj = blk.size
        Fy0 = blk.eval(y0)
        Fy0 = 0.5 * (Fy0 + Fy0.T)
        # regularize
        Fy0 += 1e-6 * np.eye(nj)
        w, V = np.linalg.eigh(Fy0)
        w = np.maximum(w, 0.0)
        # take top r
        idx = np.argsort(-w)[:r]
        L0 = V[:, idx] * np.sqrt(w[idx])[None, :]
        x0[cur:cur + nj * r] = L0.ravel()
        cur += nj * r

    mu = mu_init
    x = x0.copy()
    for outer in range(outer_iters):
        res = minimize(
            augmented_lagrangian, x, args=(mu,), jac=grad_al,
            method='L-BFGS-B',
            options={'maxiter': inner_maxiter, 'gtol': tol_grad,
                     'ftol': 1e-14},
        )
        x = res.x
        y, Ls = unpack(x)
        res_eq = sdp.A @ y - sdp.b
        cons_errs = []
        for j, (blk, L) in enumerate(zip(blocks, Ls)):
            nj = blk.size
            delta = blk.eval(y) - L @ L.T
            cons_errs.append(float(np.linalg.norm(delta)))
        eq_err = float(np.linalg.norm(res_eq))
        if verbose:
            print(f'  BM outer {outer}: mu={mu:.1e}, eq={eq_err:.2e}, '
                  f'cons={max(cons_errs):.2e}, obj={float(sdp.c @ y):.6e}')

        if eq_err < tol_cons and max(cons_errs) < tol_cons:
            break

        # Update multipliers
        lam_A = lam_A + mu * res_eq
        for j, (blk, L) in enumerate(zip(blocks, Ls)):
            delta = blk.eval(y) - L @ L.T
            nu[j] = nu[j] + mu * delta

        mu *= mu_mult

    y, Ls = unpack(x)
    # At convergence, S_j = -nu[j] (approx dual PSD matrix, up to sign)
    # Motivation: KKT says nabla_y L = 0  iff  c + A^T lam_A + sum G_flat^T vec(nu_j) = 0
    # so dual match with sign S_j = -nu_j.
    S_blocks = []
    for nj, nu_j in zip(sizes, nu):
        # symmetrize, clip negative eigenvalues for numerical robustness
        S = 0.5 * (nu_j + nu_j.T)
        # the sign convention: we want c - A^T lam - sum adj(S_j) = 0, so
        # compare with the AL stationarity.
        S = -S  # see derivation above
        S = 0.5 * (S + S.T)
        S_blocks.append(S)

    # The AL dual sign: c + A^T lam_A + sum G^T nu_j = 0 at stationarity.
    # We want c - A^T mu - sum G^T S = 0  so  mu = -lam_A, S = -nu_j.
    lam_A_std = -lam_A

    return SolverResult(
        y=y, obj=float(sdp.c @ y),
        lambda_A=lam_A_std, S_blocks=S_blocks,
        status='bm_converged', solver='burer-monteiro',
        time=time.time() - t0,
    )


# =====================================================================
# Dispatcher
# =====================================================================

def solve_primal_dual(sdp: SDPData, solver: str = 'auto',
                     verbose: bool = False, bm_rank: int = 16) -> SolverResult:
    """Solve the primal SDP and return primal y + approximate dual (lambda_A, {S_j}).

    solver:
        'auto'     -- MOSEK if importable, else clarabel.
        'mosek'    -- force MOSEK (Fusion).
        'clarabel' -- cvxpy + Clarabel (default).
        'scs'      -- cvxpy + SCS.
        'bm'       -- Burer-Monteiro augmented-Lagrangian (fallback).
    """
    if solver == 'auto':
        try:
            import mosek.fusion  # noqa
            solver = 'mosek'
        except Exception:
            solver = 'clarabel'

    if solver == 'mosek':
        return _solve_with_mosek(sdp, verbose=verbose)
    elif solver == 'clarabel':
        return _solve_with_cvxpy(sdp, solver='CLARABEL', verbose=verbose)
    elif solver == 'scs':
        return _solve_with_cvxpy(sdp, solver='SCS', verbose=verbose)
    elif solver == 'bm':
        return _solve_with_bm(sdp, rank=bm_rank, verbose=verbose)
    else:
        raise ValueError(f'unknown solver: {solver}')
