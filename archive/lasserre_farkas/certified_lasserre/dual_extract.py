"""Extract and polish approximate dual multipliers from solver output.

Given a SolverResult (lambda_A, {S_j}) from a float64 SDP solver, this
module:

  1. Symmetrizes each S_j and projects onto the PSD cone via eigen-clipping
     (at float precision) to remove small negative eigenvalues that come
     from solver tolerance.

  2. Reports the stationarity residual  r = c - A^T lambda_A - sum F_j^*(S_j)
     so the caller can decide whether further rounding will succeed.

These steps are PRE-RATIONAL.  The rational rounding + exact feasibility
repair is done in round_repair.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from certified_lasserre.build_sdp import SDPData
from certified_lasserre.bm_solver import SolverResult


@dataclass
class PolishedDual:
    lambda_A: np.ndarray
    S_blocks: List[np.ndarray]             # PSD-projected
    L_blocks: List[np.ndarray]             # S_j = L_j L_j^T (float, full-rank subspace)
    ranks: List[int]                       # numerical rank used
    residual_norm: float                   # ||c - A^T lambda_A - sum F_j^*(S_j)||
    min_eig_before: List[float]
    min_eig_after: List[float]


def extract_dual(sdp: SDPData, sol: SolverResult,
                 eig_drop_tol: float = 1e-9,
                 verbose: bool = False) -> PolishedDual:
    """Symmetrize, PSD-project, and factorize each dual block.

    Args:
        sdp:          SDPData (used for residual computation and shape info).
        sol:          raw SolverResult from bm_solver.solve_primal_dual.
        eig_drop_tol: eigenvalues below this are set to zero (float-level).
                      Rounding to rationals happens later.

    Returns:
        PolishedDual with S_j PSD and factorization S_j = L_j L_j^T.
    """
    S_clean = []
    L_clean = []
    ranks = []
    min_before = []
    min_after = []

    for blk, S in zip(sdp.blocks, sol.S_blocks):
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        min_before.append(float(w[0]))
        # Clip small/negative eigenvalues
        w_cl = np.where(w > eig_drop_tol, w, 0.0)
        nrank = int((w_cl > 0).sum())
        ranks.append(nrank)
        # Reconstruct PSD S
        S_psd = (V * w_cl[None, :]) @ V.T
        S_psd = 0.5 * (S_psd + S_psd.T)
        # Factorize: L = V[:, positive] * sqrt(w)
        if nrank > 0:
            w_pos = w_cl[w_cl > 0]
            V_pos = V[:, w_cl > 0]
            L = V_pos * np.sqrt(w_pos)[None, :]
        else:
            L = np.zeros((blk.size, 0), dtype=np.float64)
        min_after.append(float(np.linalg.eigvalsh(S_psd)[0]))
        S_clean.append(S_psd)
        L_clean.append(L)

    # Residual: c - A^T lam_A - sum F_j^*(S_j)
    r = sdp.c.copy()
    r -= sdp.A.T @ sol.lambda_A
    for blk, S in zip(sdp.blocks, S_clean):
        r -= blk.adjoint(S)
    res_norm = float(np.linalg.norm(r))

    if verbose:
        print('[extract_dual]')
        for blk, mb, ma, rk in zip(sdp.blocks, min_before, min_after, ranks):
            print(f'  {blk.name}: n={blk.size}  min_eig {mb:+.2e} -> {ma:+.2e}  rank={rk}')
        print(f'  stationarity residual ||c - A^T lam - Sum adj(S_j)||_2 = {res_norm:.3e}')
        print(f'  lower bound b^T lambda_A       = {float(sdp.b @ sol.lambda_A):.8f}')

    return PolishedDual(
        lambda_A=sol.lambda_A.copy(),
        S_blocks=S_clean, L_blocks=L_clean, ranks=ranks,
        residual_norm=res_norm,
        min_eig_before=min_before, min_eig_after=min_after,
    )
