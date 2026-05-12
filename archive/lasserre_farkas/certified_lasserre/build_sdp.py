"""Standard-form SDP data for the aggregated Lasserre relaxation.

Given probability weights lambda_W (>=0, sum=1) and M_lambda = sum lambda_W M_W,
build the SDP

    p* = min   c^T y
         s.t.  A y = b                (equality constraints)
               F_j(y) succeq 0        (PSD blocks)

where
    y in R^{n_y}       -- pseudo-moments y_alpha for |alpha| <= 2k
    c^T y              -- sum_{i,j} (M_lambda)_{ij} * y_{e_i + e_j}
    A y = b            -- y_0 = 1 (row 0) + consistency Sum_i y_{alpha+e_i} = y_alpha
    F_0(y)             -- M_k(y)                          (n_basis x n_basis)
    F_{1..d}(y)        -- M_{k-1}(mu_i * y)               (n_loc x n_loc)     [order k>=2]

Each F_j is a LINEAR operator y -> symmetric matrix:
    F_j(y) = sum_alpha y_alpha * G_j^{(alpha)}
with G_j^{(alpha)} encoded as a sparse (row, col, val) coordinate list.

Reuses lasserre.precompute._precompute for monomial indexing (do NOT modify
that module).  This file is ONLY a thin adapter that repackages the indices
into (c, A, b, F_blocks) form suitable for generic SDP + exact-arithmetic
certification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import sparse as sp

from lasserre.precompute import _precompute


# =====================================================================
# Data container
# =====================================================================

@dataclass
class PSDBlock:
    """One PSD cone X_j succeq 0 with X_j = sum_alpha y_alpha * G_j^{(alpha)}.

    Stored as a sparse matrix G_flat: (n_j^2, n_y).  Column k of G_flat
    is the VEC of the matrix derivative dF_j/dy_k.
    Row index `r = i*n_j + j` corresponds to entry (i,j) of the matrix.
    """
    name: str
    size: int           # matrix dimension n_j
    G_flat: sp.csr_matrix  # shape (n_j^2, n_y), sparse

    def eval(self, y: np.ndarray) -> np.ndarray:
        """F_j(y) as a dense symmetric matrix."""
        flat = self.G_flat @ y
        return flat.reshape(self.size, self.size)

    def adjoint(self, S: np.ndarray) -> np.ndarray:
        """F_j^*(S) in R^{n_y}: coefficients with y_alpha coefficient
        equal to <S, G_j^{(alpha)}> = trace(S^T @ G_j^{(alpha)}).

        Since F_j(y) = sum_alpha y_alpha G_j^{(alpha)}, the adjoint w.r.t.
        the inner product <A,B> = trace(A^T B) is
            (F_j^* S)_alpha = trace(G_j^{(alpha)T} S) = sum_{i,j} G_j^{(alpha)}_{ij} S_{ij}
        Vectorized:  (F_j^* S) = G_flat^T @ vec(S).
        """
        return self.G_flat.T @ S.ravel()


@dataclass
class SDPData:
    """Standard-form SDP data for the aggregated Lasserre relaxation.

    Variables y in R^{n_y}.  Constraints:
        Ay = b                              (equality; y_0 = 1 + consistency)
        F_j(y) succeq 0                     (PSD blocks; moment + localizing)
        y >= 0   [if add_L2=True]           (Lasserre L2: moments nonneg)

    With add_L2=True plus consistency + y_0=1, every feasible y satisfies
    |y_alpha| <= 1 (y_alpha >= 0 by L2, y_alpha = y_{alpha-e_i} - sum of
    nonneg terms <= y_{alpha-e_i}, iterating down to y_0 = 1).
    """
    d: int
    order: int
    lam: np.ndarray                       # window weights, shape (n_win,)
    M_lambda: np.ndarray                  # d x d, sum lambda_W M_W
    n_y: int                              # number of moment variables
    c: np.ndarray                         # shape (n_y,)
    A: sp.csr_matrix                      # shape (n_eq, n_y)
    b: np.ndarray                         # shape (n_eq,)
    eq_names: List[str]                   # length n_eq
    blocks: List[PSDBlock]                # PSD cones
    add_L2: bool                          # if True, y >= 0 is also enforced
    mono_list: List[Tuple[int, ...]]      # alpha-index -> multi-index
    mono_idx: Dict[Tuple[int, ...], int]  # multi-index -> alpha-index

    @property
    def n_eq(self) -> int:
        return self.A.shape[0]

    @property
    def y0_row(self) -> int:
        """Row index in A of the y_0 = 1 equation."""
        return 0  # by construction

    def check_consistency(self, y: np.ndarray, tol: float = 1e-6) -> dict:
        """Diagnostic: check Ay=b and F_j(y) eigmin for a candidate y."""
        eq_res = np.linalg.norm(self.A @ y - self.b)
        obj = float(self.c @ y)
        block_eigs = []
        for blk in self.blocks:
            M = blk.eval(y)
            M = 0.5 * (M + M.T)
            eigmin = float(np.linalg.eigvalsh(M)[0])
            block_eigs.append((blk.name, blk.size, eigmin))
        return {'eq_res': float(eq_res), 'obj': obj, 'blocks': block_eigs}


# =====================================================================
# Builder
# =====================================================================

def _build_moment_block(P: dict) -> PSDBlock:
    """M_k(y): (n_basis x n_basis).  Entry (a,b) = y_{basis[a]+basis[b]}.
    Equivalently, G^{(alpha)}_{a,b} = 1 if basis[a]+basis[b] = alpha else 0.
    """
    n_basis = P['n_basis']
    n_y = P['n_y']
    # moment_pick_np[a*n_basis+b] = alpha-index for entry (a,b)
    pick = P['moment_pick_np']
    assert pick.shape == (n_basis * n_basis,), pick.shape
    rows = np.arange(n_basis * n_basis, dtype=np.int64)
    cols = pick.astype(np.int64)
    vals = np.ones_like(rows, dtype=np.float64)
    G_flat = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_basis * n_basis, n_y),
    )
    return PSDBlock(name='moment', size=n_basis, G_flat=G_flat)


def _build_loc_blocks(P: dict) -> List[PSDBlock]:
    """M_{k-1}(mu_i * y) for i = 0..d-1.  Entry (a,b) = y_{loc[a]+loc[b]+e_i}.
    """
    order = P['order']
    if order < 2:
        return []
    d = P['d']
    n_loc = P['n_loc']
    n_y = P['n_y']
    blocks = []
    # loc_picks_np[i] is an array of length n_loc*n_loc giving alpha-indices.
    for i_var in range(d):
        pick = P['loc_picks_np'][i_var]
        assert pick.shape == (n_loc * n_loc,), pick.shape
        rows = np.arange(n_loc * n_loc, dtype=np.int64)
        cols = pick.astype(np.int64)
        vals = np.ones_like(rows, dtype=np.float64)
        G_flat = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_loc * n_loc, n_y),
        )
        blocks.append(PSDBlock(name=f'loc_mu_{i_var}', size=n_loc, G_flat=G_flat))
    return blocks


def _build_equality_constraints(P: dict) -> Tuple[sp.csr_matrix, np.ndarray, List[str]]:
    """A y = b encoding:
        row 0 : y_0 = 1
        next  : Sum_i y_{alpha+e_i} - y_alpha = 0, for each alpha with
                |alpha| <= 2k-1 and valid children in the moment set.

    Returns (A_csr, b, names).
    """
    d = P['d']
    n_y = P['n_y']
    idx = P['idx']
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']

    zero = tuple(0 for _ in range(d))
    y0_idx = idx[zero]

    rows, cols, vals = [], [], []
    names = []

    # Row 0: y_0 = 1
    rows.append(0)
    cols.append(y0_idx)
    vals.append(1.0)
    names.append('y0')

    # Consistency rows
    r_next = 1
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        row_rows_local, row_cols_local, row_vals_local = [], [], []
        has_child = False
        for ci in range(d):
            ch = int(child_idx[ci])
            if ch >= 0:
                row_rows_local.append(r_next)
                row_cols_local.append(ch)
                row_vals_local.append(1.0)
                has_child = True
        if not has_child:
            continue
        row_rows_local.append(r_next)
        row_cols_local.append(ai)
        row_vals_local.append(-1.0)
        rows.extend(row_rows_local)
        cols.extend(row_cols_local)
        vals.extend(row_vals_local)
        names.append(f'consist[{P["consist_mono"][r]}]')
        r_next += 1

    n_eq = r_next
    A = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64),
          np.asarray(cols, dtype=np.int64))),
        shape=(n_eq, n_y),
    )
    b = np.zeros(n_eq, dtype=np.float64)
    b[0] = 1.0
    return A, b, names


def _build_objective(P: dict, M_lambda: np.ndarray) -> np.ndarray:
    """c[alpha] = (M_lambda)_{ii}   if alpha = 2*e_i
                   (M_lambda)_{ij} + (M_lambda)_{ji} = 2*(M_lambda)_{ij}  if alpha = e_i + e_j, i!=j
                   0                   otherwise.

    Objective c^T y = sum_{i,j} (M_lambda)_{ij} y_{e_i+e_j}
                    = sum_i (M_lambda)_{ii} y_{2 e_i}
                      + 2 * sum_{i<j} (M_lambda)_{ij} y_{e_i+e_j}.
    """
    d = P['d']
    n_y = P['n_y']
    idx = P['idx']
    c = np.zeros(n_y, dtype=np.float64)
    # Diagonal:  alpha = 2 e_i
    for i in range(d):
        a = tuple((2 if k == i else 0) for k in range(d))
        c[idx[a]] += float(M_lambda[i, i])
    # Off-diagonal:  alpha = e_i + e_j, i<j
    for i in range(d):
        for j in range(i + 1, d):
            a = tuple((1 if k == i or k == j else 0) for k in range(d))
            c[idx[a]] += 2.0 * float(M_lambda[i, j])
    return c


def build_sdp_data(d: int, order: int, lam: Optional[np.ndarray] = None,
                   verbose: bool = False) -> SDPData:
    """Build standard-form SDP data for the aggregated Lasserre relaxation.

    Args:
        d:       number of bins.
        order:   Lasserre order k (>=1).  For val(d) bounds we use k>=2.
        lam:     window weights, shape (n_win,), >=0, sum=1.  If None,
                 uniform over windows (fallback).
        verbose: print precompute summary.

    Returns:
        SDPData.  Primal:  min c^T y  s.t.  Ay=b, F_j(y) succeq 0 for each block.
    """
    from lasserre.core import build_window_matrices

    P = _precompute(d, order, verbose=verbose)

    windows = P['windows']
    M_mats = P['M_mats']
    n_win = P['n_win']

    if lam is None:
        lam = np.ones(n_win, dtype=np.float64) / n_win
    else:
        lam = np.asarray(lam, dtype=np.float64)
        assert lam.shape == (n_win,), f"lam shape {lam.shape} != ({n_win},)"
        assert np.all(lam >= -1e-15), "lambda must be nonneg"
        lam = np.maximum(lam, 0.0)
        s = lam.sum()
        if s <= 0:
            raise ValueError("lambda sums to zero")
        lam = lam / s

    # M_lambda = sum_W lam_W M_W
    M_lambda = np.zeros((d, d), dtype=np.float64)
    for w in range(n_win):
        if lam[w] == 0.0:
            continue
        M_lambda += lam[w] * M_mats[w]
    # Symmetrize (defensive; M_W already symmetric)
    M_lambda = 0.5 * (M_lambda + M_lambda.T)

    c_obj = _build_objective(P, M_lambda)
    A, b, eq_names = _build_equality_constraints(P)
    blocks: List[PSDBlock] = [_build_moment_block(P)]
    blocks.extend(_build_loc_blocks(P))

    return SDPData(
        d=d, order=order, lam=lam, M_lambda=M_lambda,
        n_y=P['n_y'], c=c_obj,
        A=A.tocsr(), b=b, eq_names=eq_names,
        blocks=blocks,
        add_L2=False,
        mono_list=P['mono_list'], mono_idx=P['idx'],
    )
