"""flint-accelerated version of safe_certify.

Same mathematical pipeline as safe_certify.py, but rational arithmetic
uses python-flint's fmpq / fmpq_mat (C-backed) instead of Python's
Fraction. Benchmarked ~50-100x speedup on n=50 matmul, scaling better
at larger n.

Interface is identical to safe_certify.py:
    safe_certify_flint(sdp, ...) -> (SafeCertifyResult, SafeCertificate)

The CertifiedDual and SafeCertificate data containers use Fraction on
the outside (for Python-stdlib round-trip), but internal computation
goes through fmpq.

Soundness is the same as safe_certify: lb_safe = lambda[0] - ||r||_1 *
moment_l1_bound, where r is the exact rational residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple
import time
import numpy as np
from scipy import sparse as sp

try:
    import flint  # type: ignore
    _HAS_FLINT = True
except ImportError:
    _HAS_FLINT = False

from certified_lasserre.build_sdp import SDPData, PSDBlock, build_sdp_data
from certified_lasserre.bm_solver import solve_primal_dual
from certified_lasserre.dual_extract import extract_dual
from certified_lasserre.safe_certify import (
    SafeCertifyResult, SafeCertificate, _moment_l1_bound, _decimal_str,
)


def _fl(x):
    """Float -> fmpq with reasonable denominator bound."""
    f = Fraction(float(x)).limit_denominator(10**15)
    return flint.fmpq(f.numerator, f.denominator)


def _frac_to_fmpq(f: Fraction):
    return flint.fmpq(f.numerator, f.denominator)


def _fmpq_to_frac(q) -> Fraction:
    return Fraction(int(q.numer()), int(q.denom()))


def _fmpq_to_float(q) -> float:
    return int(q.numer()) / int(q.denom())


# =====================================================================
# Vectorized rational operations with fmpq
# =====================================================================

def _round_vec_fmpq(v: np.ndarray, max_denom: int) -> list:
    out = []
    for x in v:
        f = Fraction(float(x)).limit_denominator(max_denom)
        out.append(flint.fmpq(f.numerator, f.denominator))
    return out


def _round_mat_fmpq(M: np.ndarray, max_denom: int):
    """Return an fmpq_mat built from float matrix M via bounded-denom rounding."""
    n, r = M.shape
    flat = []
    for i in range(n):
        for j in range(r):
            f = Fraction(float(M[i, j])).limit_denominator(max_denom)
            flat.append(flint.fmpq(f.numerator, f.denominator))
    return flint.fmpq_mat(n, r, flat)


def _fmpq_mat_to_list(M) -> list:
    """Extract fmpq_mat to a flat list of fmpq."""
    n, m = M.nrows(), M.ncols()
    return [M[i, j] for i in range(n) for j in range(m)]


def _sparse_matvec_fmpq(A_csr: sp.csr_matrix, v_fmpq: list) -> list:
    """Sparse CSR matvec where v is a list of fmpq. A entries assumed integer."""
    n_rows = A_csr.shape[0]
    out = [flint.fmpq(0) for _ in range(n_rows)]
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data
    for i in range(n_rows):
        s = flint.fmpq(0)
        for p in range(indptr[i], indptr[i + 1]):
            coef = int(round(data[p]))
            if coef != 0:
                s += coef * v_fmpq[indices[p]]
        out[i] = s
    return out


def _adjoint_block_fmpq(blk: PSDBlock, S_mat) -> list:
    """F_j^*(S) as list of fmpq in length n_y. S_mat is fmpq_mat (n_j x n_j).

    (F_j^* S)_alpha = sum_{i,k} G_j^{(alpha)}_{ik} * S_{ik}
                    = G_flat^T @ vec(S)
    G_flat entries are in {0, 1}.
    """
    G_coo = blk.G_flat.tocoo()
    n_y = blk.G_flat.shape[1]
    nj = blk.size
    out = [flint.fmpq(0) for _ in range(n_y)]
    for r, cidx, val in zip(G_coo.row, G_coo.col, G_coo.data):
        v = int(round(float(val)))
        if v == 0:
            continue
        i = int(r) // nj
        k = int(r) % nj
        out[int(cidx)] += v * S_mat[i, k]
    return out


def _rational_c_fmpq(sdp: SDPData, lam_win_fmpq: list) -> list:
    """Rebuild c in fmpq given rational (fmpq) window weights."""
    from lasserre.core import build_window_matrices
    d = sdp.d
    n_y = sdp.n_y
    windows, M_mats = build_window_matrices(d)
    n_win = len(M_mats)
    assert len(lam_win_fmpq) == n_win

    # M_lambda in fmpq: d x d
    M_lam = [[flint.fmpq(0) for _ in range(d)] for _ in range(d)]
    for w in range(n_win):
        lw = lam_win_fmpq[w]
        if lw == 0:
            continue
        ell, s_lo = windows[w]
        coeff = flint.fmpq(2 * d, ell)
        Mw = M_mats[w]
        for i in range(d):
            for j in range(d):
                if Mw[i, j] != 0:
                    M_lam[i][j] += lw * coeff

    c_fmpq = [flint.fmpq(0) for _ in range(n_y)]
    for i in range(d):
        a = tuple((2 if k == i else 0) for k in range(d))
        c_fmpq[sdp.mono_idx[a]] += M_lam[i][i]
    for i in range(d):
        for j in range(i + 1, d):
            a = tuple((1 if k == i or k == j else 0) for k in range(d))
            c_fmpq[sdp.mono_idx[a]] += 2 * M_lam[i][j]
    return c_fmpq


# =====================================================================
# Main pipeline (flint-accelerated)
# =====================================================================

def safe_certify_flint(sdp: SDPData,
                       solver: str = 'auto',
                       max_denom_L: int = 10**10,
                       max_denom_lam: int = 10**12,
                       max_denom_win: int = 10**10,
                       eig_margin: float = 1e-9,
                       verbose: bool = True
                       ) -> Tuple[SafeCertifyResult, SafeCertificate]:
    if not _HAS_FLINT:
        raise ImportError("python-flint not available; use safe_certify instead")

    t_start = time.time()
    d, order = sdp.d, sdp.order

    if verbose:
        print(f'[safe_certify_flint] d={d}, order={order}, n_y={sdp.n_y}, '
              f'n_eq={sdp.n_eq}, n_blocks={len(sdp.blocks)}')

    # Step 1: solve (float)
    t0 = time.time()
    sol = solve_primal_dual(sdp, solver=solver, verbose=False)
    solver_time = time.time() - t0
    if verbose:
        print(f'  solver={sol.solver} status={sol.status} '
              f'primal={sol.obj:.9f} time={solver_time:.2f}s')

    # Step 2: polish dual (float PSD projection)
    polished = extract_dual(sdp, sol, verbose=verbose)

    # Step 3: round to fmpq
    t0 = time.time()

    # Window weights
    lam_win_fl = [_fl(v) for v in sdp.lam]
    total = flint.fmpq(0)
    for v in lam_win_fl:
        total += v
    assert total != 0
    lam_win_fl = [v / total for v in lam_win_fl]

    # Rebuild c in fmpq
    c_fmpq = _rational_c_fmpq(sdp, lam_win_fl)

    # Round lambda_A
    lam_A_fl = []
    for x in polished.lambda_A:
        f = Fraction(float(x)).limit_denominator(max_denom_lam)
        lam_A_fl.append(flint.fmpq(f.numerator, f.denominator))

    # For each block: rational Cholesky with eig margin
    L_mats = []          # list of fmpq_mat (n_j, n_j)
    S_mats = []          # S = L L^T as fmpq_mat
    slack_list = []      # list of list[fmpq] (all zero in this version; margin baked into L)

    for blk, S_float in zip(sdp.blocks, polished.S_blocks):
        n_j = blk.size
        S_sym = 0.5 * (S_float + S_float.T)
        w, V = np.linalg.eigh(S_sym)
        w_pos = np.maximum(w, 0.0) + float(eig_margin)
        L_float = V * np.sqrt(w_pos)[None, :]

        L_fm = _round_mat_fmpq(L_float, max_denom_L)
        L_mats.append(L_fm)

        # S = L @ L.T  (fmpq_mat * fmpq_mat is fast native)
        LT = L_fm.transpose()
        S_fm = L_fm * LT
        S_mats.append(S_fm)

        slack_list.append([flint.fmpq(0) for _ in range(n_j)])

    # Compute exact residual r = c - A^T lam - sum adj(S_j)
    AT_csr = sdp.A.T.tocsr()
    AT_lam = _sparse_matvec_fmpq(AT_csr, lam_A_fl)
    r = [c_fmpq[k] - AT_lam[k] for k in range(sdp.n_y)]
    for blk, S_fm in zip(sdp.blocks, S_mats):
        adj = _adjoint_block_fmpq(blk, S_fm)
        for k in range(sdp.n_y):
            r[k] -= adj[k]

    # ||r||_1 as fmpq
    res_l1 = flint.fmpq(0)
    for v in r:
        res_l1 += abs(v)
    res_l1_float = _fmpq_to_float(res_l1)

    # Moment bound
    M_y = _moment_l1_bound(sdp)  # this returns Fraction(2*order+1)
    M_y_fmpq = flint.fmpq(M_y.numerator, M_y.denominator)

    # Safe bound
    safety_loss = res_l1 * M_y_fmpq
    lam0 = lam_A_fl[0]
    lb_safe = lam0 - safety_loss

    round_time = time.time() - t0
    total_time = time.time() - t_start

    if verbose:
        print(f'  lam0 rational         = {_decimal_str(_fmpq_to_frac(lam0), 15)}')
        print(f'  ||r||_1 (exact)       = {res_l1_float:.3e}')
        print(f'  moment_l1_bound       = {M_y}')
        print(f'  safety_loss           = {_fmpq_to_float(safety_loss):.3e}')
        print(f'  lb_safe (RIGOROUS)    = {_decimal_str(_fmpq_to_frac(lb_safe), 15)}')

    # Convert back to Fraction + np.object arrays for the dataclass
    lam_win_frac = np.array([_fmpq_to_frac(v) for v in lam_win_fl], dtype=object)
    lam_A_frac = np.array([_fmpq_to_frac(v) for v in lam_A_fl], dtype=object)
    L_blocks_frac: List[np.ndarray] = []
    for L_fm in L_mats:
        n, r_ = L_fm.nrows(), L_fm.ncols()
        arr = np.empty((n, r_), dtype=object)
        for i in range(n):
            for j in range(r_):
                arr[i, j] = _fmpq_to_frac(L_fm[i, j])
        L_blocks_frac.append(arr)
    slack_frac = [np.array([_fmpq_to_frac(v) for v in sl], dtype=object)
                  for sl in slack_list]

    lb_safe_frac = _fmpq_to_frac(lb_safe)
    lam0_frac = _fmpq_to_frac(lam0)
    res_l1_frac = _fmpq_to_frac(res_l1)
    safety_loss_frac = _fmpq_to_frac(safety_loss)

    cert = SafeCertificate(
        d=d, order=order,
        lam_windows=lam_win_frac,
        lambda_A=lam_A_frac,
        L_blocks=L_blocks_frac,
        slack=slack_frac,
        residual_abs_sum=res_l1_frac,
        moment_l1_bound=M_y,
        lam0=lam0_frac,
        lb_safe=lb_safe_frac,
    )
    result = SafeCertifyResult(
        d=d, order=order,
        lb_rig=lb_safe_frac,
        lb_rig_decimal=_decimal_str(lb_safe_frac, 15),
        primal_obj_float=sol.obj,
        lam0_rat=lam0_frac,
        residual_l1=res_l1_frac,
        residual_l1_float=res_l1_float,
        moment_l1_bound=M_y,
        safety_loss=safety_loss_frac,
        solver=sol.solver,
        solver_status=str(sol.status),
        solver_time=solver_time,
        round_time=round_time,
        total_time=total_time,
    )
    return result, cert


# =====================================================================
# Wrapper that picks flint if available
# =====================================================================

def safe_certify_best(sdp: SDPData, **kwargs) -> Tuple[SafeCertifyResult, SafeCertificate]:
    if _HAS_FLINT:
        return safe_certify_flint(sdp, **kwargs)
    from certified_lasserre.safe_certify import safe_certify
    return safe_certify(sdp, **kwargs)
