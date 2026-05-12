"""Rational rounding + exact feasibility repair for the dual certificate.

Input:    PolishedDual (lambda_A, {L_j}) in float64, plus SDPData.
Output:   CertifiedDual (lambda_rat, {L_rat_j}) in EXACT RATIONALS such that

              c - A^T lambda_rat - sum_j F_j^*(L_rat_j L_rat_j^T) = 0   (exactly)

    where c, A are the rational SDP data (by construction all entries of A are
    in {-1, 0, 1} and c is a sum of doubled / unit entries of M_lambda, which is
    rational whenever lambda is rational).

Strategy:
  1.  Round each L_j entry to a rational with bounded denominator (default 10^8).
      S_j = L_j L_j^T is then an EXACT rational PSD matrix (PSD by construction
      of the product, independent of floats).

  2.  Round lambda_A to rationals with bounded denominator (default 10^10).

  3.  Compute the exact rational residual  r = c - A^T lambda_rat
      - sum_j F_j^*(L_rat_j L_rat_j^T)  using Fraction arithmetic.

  4.  Solve  A^T * delta_lambda = r  in exact rationals.  If A has full row
      rank and r in col(A^T), the (least-norm) solution can be computed via
      sympy on a reduced system.  Otherwise, ABSORB the residual by adding
      a small scalar slack  s_j >= 0  to each block:
          S_j -> S_j + s_j * I    (still PSD, still stored as L_j L_j^T
                                   after re-factorization, or as a sum).

  5.  Output CertifiedDual with the final exact (lambda, {L_j}, {s_j}).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple
import numpy as np
from scipy import sparse as sp
import mpmath

import sympy
from sympy import Rational, Matrix as SyMatrix, zeros as sy_zeros

from certified_lasserre.build_sdp import SDPData, PSDBlock
from certified_lasserre.dual_extract import PolishedDual


# =====================================================================
# Utilities: float -> Fraction with bounded denominator
# =====================================================================

def _float_to_fraction(x: float, max_denom: int) -> Fraction:
    # mpmath nstr rounds correctly
    return Fraction(float(x)).limit_denominator(max_denom)


def _vec_to_fractions(v: np.ndarray, max_denom: int) -> np.ndarray:
    out = np.empty(v.shape, dtype=object)
    flat_out = out.ravel()
    flat_in = v.ravel()
    for k in range(flat_in.size):
        flat_out[k] = _float_to_fraction(float(flat_in[k]), max_denom)
    return out


def _mat_to_fractions(M: np.ndarray, max_denom: int) -> np.ndarray:
    return _vec_to_fractions(M, max_denom)


def _fraction_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Object-dtype matrix product with Fraction entries (exact)."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    out = np.empty((m, n), dtype=object)
    # Naive triple loop in Python; fine for the sizes we need.
    for i in range(m):
        for j in range(n):
            s = Fraction(0)
            for l in range(k):
                s += A[i, l] * B[l, j]
            out[i, j] = s
    return out


def _frac_sparse_matvec(A_csr: sp.csr_matrix, v_frac: np.ndarray) -> np.ndarray:
    """Sparse matvec A @ v with v having object dtype (Fractions)."""
    n_rows = A_csr.shape[0]
    out = np.empty(n_rows, dtype=object)
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data
    for i in range(n_rows):
        s = Fraction(0)
        for p in range(indptr[i], indptr[i + 1]):
            s += Fraction(int(round(data[p]))) * v_frac[indices[p]]
        out[i] = s
    return out


def _adjoint_block_rational(blk: PSDBlock, S_frac: np.ndarray) -> np.ndarray:
    """F_j^*(S) in Q^{n_y}, using exact arithmetic.

    (F_j^* S)_alpha = sum_{i,k} G_j^{(alpha)}_{ik} * S_{ik}
                    = G_flat^T * vec(S)
    where G_flat has entries in {0, 1} for our blocks (Moment and Localizing).
    """
    G_csr = blk.G_flat  # shape (n_j^2, n_y)
    # G_flat^T @ vec(S) => column by column; use sparse.
    G_coo = G_csr.tocoo()
    # Entries are 0/1; cast to int to be sure.
    n_y = G_csr.shape[1]
    out = np.empty(n_y, dtype=object)
    for k in range(n_y):
        out[k] = Fraction(0)
    S_flat = S_frac.ravel()
    # iterate over nonzeros
    for r, cidx, val in zip(G_coo.row, G_coo.col, G_coo.data):
        # val is 1.0 in our construction; cast to integer safely.
        out[int(cidx)] += Fraction(int(round(float(val)))) * S_flat[int(r)]
    return out


# =====================================================================
# Build rational c from SDPData (given rational lambda)
# =====================================================================

def _rational_c(sdp: SDPData, lam_frac: np.ndarray) -> np.ndarray:
    """Rebuild c in exact rationals given a rational lambda.

    c_alpha = sum_W lambda_W * (M_W)_{ij} for alpha = e_i + e_j, with
    appropriate factor of 2 for i != j.
    """
    from lasserre.core import build_window_matrices
    d = sdp.d
    n_y = sdp.n_y
    windows, M_mats = build_window_matrices(d)
    n_win = len(M_mats)
    assert lam_frac.shape == (n_win,)

    # M_lambda in Q
    M_lam = np.empty((d, d), dtype=object)
    for i in range(d):
        for j in range(d):
            M_lam[i, j] = Fraction(0)
    for w in range(n_win):
        lw = lam_frac[w]
        if lw == 0:
            continue
        Mw = M_mats[w]  # float, but entries are (2d/ell)*0/1
        # Convert to rationals exactly:  (2d/ell) with ell integer
        ell, s_lo = windows[w]
        coeff = Fraction(2 * d, ell)
        for i in range(d):
            for j in range(d):
                if Mw[i, j] != 0:
                    M_lam[i, j] += lw * coeff * Fraction(1)

    c_frac = np.empty(n_y, dtype=object)
    for k in range(n_y):
        c_frac[k] = Fraction(0)
    for i in range(d):
        a = tuple((2 if kk == i else 0) for kk in range(d))
        c_frac[sdp.mono_idx[a]] += M_lam[i, i]
    for i in range(d):
        for j in range(i + 1, d):
            a = tuple((1 if kk == i or kk == j else 0) for kk in range(d))
            c_frac[sdp.mono_idx[a]] += Fraction(2) * M_lam[i, j]
    return c_frac


# =====================================================================
# Exact linear solve for the repair step
# =====================================================================

def _solve_AT_lambda_eq_r(A_csr: sp.csr_matrix, r_frac: np.ndarray,
                           max_denom_check: int = 10**15,
                           verbose: bool = False) -> Optional[np.ndarray]:
    """Solve A^T * lambda = r in exact rationals.

    A has integer entries (-1, 0, 1); r has Fraction entries; lambda will
    be Fraction.

    Approach: convert to a sympy Matrix; use sympy.solve_linear_system
    (fraction-free Gauss-Jordan).  Falls back to returning None if infeasible.

    For n_eq up to ~500 (d<=6), this is fast (seconds).  For n_eq >~2000
    (d>=12), this function times out and the caller should switch to
    slack-absorption mode.
    """
    n_eq, n_y = A_csr.shape
    AT = A_csr.T  # (n_y, n_eq)

    # Build augmented sympy system AT @ x = r with x in Q^{n_eq}.
    # We use sympy.Rational for exact rationals.
    # For efficiency we first do a float LSQ to see if the system is
    # consistent; sympy is only called when the float sanity check passes.

    # Float feasibility check first:
    r_float = np.array([float(x) for x in r_frac], dtype=np.float64)
    try:
        AT_dense = AT.toarray()
    except MemoryError:
        raise RuntimeError(
            '_solve_AT_lambda_eq_r: AT too large for dense solve; '
            'use slack mode')
    # Least-squares check
    lam_lsq, *_ = np.linalg.lstsq(AT_dense, r_float, rcond=None)
    r_proj = AT_dense @ lam_lsq
    proj_err = np.linalg.norm(r_proj - r_float)
    if verbose:
        print(f'  _solve_AT_lam: float lstsq proj_err = {proj_err:.3e}')
    if proj_err > 1e-7:
        if verbose:
            print('  _solve_AT_lam: residual NOT in col(A^T) at float level; '
                  'returning None')
        return None

    # Now exact sympy solve.
    AT_sym = SyMatrix(n_y, n_eq,
                      lambda i, j: Rational(int(round(AT_dense[i, j]))))
    r_sym = SyMatrix(n_y, 1, lambda i, j: sympy.Rational(r_frac[i].numerator,
                                                         r_frac[i].denominator))
    # Solve via augmented LU.  sympy.Matrix.solve raises if inconsistent.
    aug = AT_sym.row_join(r_sym)
    rref_mat, pivots = aug.rref()
    # Check: any pivot in the last column => inconsistent
    if (n_eq) in pivots:
        if verbose:
            print('  _solve_AT_lam: sympy detects inconsistent system')
        return None

    # Back-substitute
    x = [Rational(0)] * n_eq
    # pivots are sorted
    for pi, pc in enumerate(pivots):
        if pc == n_eq:
            # shouldn't reach (handled above)
            return None
        x[pc] = rref_mat[pi, n_eq]

    # Convert back to fractions
    out = np.empty(n_eq, dtype=object)
    for k in range(n_eq):
        q = x[k]
        if isinstance(q, sympy.Rational):
            out[k] = Fraction(int(q.p), int(q.q))
        else:
            out[k] = Fraction(int(q), 1)
    return out


# =====================================================================
# Certificate data class
# =====================================================================

@dataclass
class CertifiedDual:
    """Exact-rational dual certificate.

        S_j = L_j L_j^T + diag(D_j)              (PSD by construction, D_j >= 0)
        c - A^T lambda - sum_j F_j^*(S_j) = 0    (EXACTLY in rationals)
    """
    d: int
    order: int
    lam_windows: np.ndarray      # (n_win,) of Fraction -- the window weights
    lambda_A: np.ndarray         # (n_eq,) of Fraction -- equality multipliers
    L_blocks: List[np.ndarray]   # list of (n_j, r_j) Fraction matrices
    slack: List[np.ndarray]      # list of (n_j,) diagonal nonneg slack arrays
                                 # each entry a Fraction >= 0
    exact_residual: np.ndarray   # (n_y,) of Fraction -- should be all zeros
    lb_rig: Fraction             # = b^T lambda_A = lambda_A[0]

    def residual_norm_squared(self) -> Fraction:
        s = Fraction(0)
        for v in self.exact_residual:
            s += v * v
        return s


# =====================================================================
# Main entry point
# =====================================================================

def round_and_repair(sdp: SDPData, dual: PolishedDual,
                     max_denom_L: int = 10**6,
                     max_denom_lam: int = 10**10,
                     max_denom_win: int = 10**8,
                     try_slack: bool = True,
                     verbose: bool = False) -> CertifiedDual:
    """Round a float dual (lambda, {L_j}) to exact rationals and repair
    feasibility of the stationarity equation.

    Args:
        sdp:           problem data (uses sdp.lam for the window weights --
                       these are rounded to rationals too).
        dual:          PolishedDual from dual_extract.
        max_denom_L:   denominator cap when rounding each L_j entry.
        max_denom_lam: denominator cap when rounding lambda_A entries.
        max_denom_win: denominator cap when rounding window weights.
        try_slack:     if True, add scalar slack to each block when the
                       exact linear system is inconsistent.
        verbose:       print progress.

    Returns:
        CertifiedDual with EXACTLY ZERO stationarity residual (Fraction 0
        in every y_alpha coordinate) on success.  Raises ValueError on
        failure (e.g., float dual too noisy to round).
    """
    from lasserre.core import build_window_matrices
    d = sdp.d
    order = sdp.order

    # Round window weights lambda_W first, so they form a rational prob
    # distribution.
    lam_win_float = np.asarray(sdp.lam, dtype=np.float64)
    # Round and renormalize
    lam_win_f = _vec_to_fractions(lam_win_float, max_denom_win)
    total = Fraction(0)
    for v in lam_win_f:
        total += v
    assert total != 0
    for k in range(len(lam_win_f)):
        lam_win_f[k] = lam_win_f[k] / total

    # Rebuild c in rationals from the rounded lam_win
    c_rat = _rational_c(sdp, lam_win_f)

    # Round each L_j
    L_rat_list: List[np.ndarray] = []
    S_rat_list: List[np.ndarray] = []
    for L in dual.L_blocks:
        if L.shape[1] == 0:
            # rank-0 block; S = 0
            n_j = L.shape[0]
            L_rat = np.empty((n_j, 0), dtype=object)
            S_rat = np.empty((n_j, n_j), dtype=object)
            for ii in range(n_j):
                for jj in range(n_j):
                    S_rat[ii, jj] = Fraction(0)
        else:
            L_rat = _mat_to_fractions(L, max_denom_L)
            S_rat = _fraction_matmul(L_rat, L_rat.T)
        L_rat_list.append(L_rat)
        S_rat_list.append(S_rat)

    # Round lambda_A
    lam_A_rat = _vec_to_fractions(dual.lambda_A, max_denom_lam)

    # Exact rational residual
    # r_frac = c - A^T lambda_A - sum_j adj(S_j)
    r_frac = np.empty(sdp.n_y, dtype=object)
    for k in range(sdp.n_y):
        r_frac[k] = c_rat[k]
    # subtract A^T @ lambda_A (exact)
    AT_csr = sdp.A.T.tocsr()
    # AT @ lambda_A
    AT_lambda = _frac_sparse_matvec(AT_csr, lam_A_rat)
    for k in range(sdp.n_y):
        r_frac[k] -= AT_lambda[k]
    # subtract adj(S_j) for each block
    for blk, S_frac in zip(sdp.blocks, S_rat_list):
        adj = _adjoint_block_rational(blk, S_frac)
        for k in range(sdp.n_y):
            r_frac[k] -= adj[k]

    res_fl = np.array([float(x) for x in r_frac])
    res_norm = float(np.linalg.norm(res_fl))
    if verbose:
        print(f'[round_and_repair] after initial rounding: '
              f'||residual||={res_norm:.3e}, '
              f'max|res|={np.max(np.abs(res_fl)):.3e}')

    # Solve A^T delta_lambda = r in exact rationals
    delta = _solve_AT_lambda_eq_r(sdp.A, r_frac, verbose=verbose)

    slack_list: List[np.ndarray] = [
        np.array([Fraction(0) for _ in range(blk.size)], dtype=object)
        for blk in sdp.blocks
    ]

    if delta is None:
        if not try_slack:
            raise ValueError('rounding residual not in col(A^T); '
                             'enable try_slack=True or reduce rank')
        # Absorb residual via DIAGONAL slacks (one nonneg slack per diagonal
        # entry of each block).  Total free params: n_eq + sum_j n_j.
        return _repair_with_diag_slack(sdp, lam_A_rat, L_rat_list, S_rat_list,
                                       r_frac, c_rat, lam_win_f,
                                       verbose=verbose)

    # Apply delta: lambda_final = lambda_A_rat + delta_lambda
    lam_final = np.empty(sdp.n_eq, dtype=object)
    for k in range(sdp.n_eq):
        lam_final[k] = lam_A_rat[k] + delta[k]

    # Verify exact residual = 0
    AT_lambda_new = _frac_sparse_matvec(AT_csr, lam_final)
    r_final = np.empty(sdp.n_y, dtype=object)
    for k in range(sdp.n_y):
        r_final[k] = c_rat[k] - AT_lambda_new[k]
    for blk, S_frac in zip(sdp.blocks, S_rat_list):
        adj = _adjoint_block_rational(blk, S_frac)
        for k in range(sdp.n_y):
            r_final[k] -= adj[k]

    # Check zero
    if verbose:
        nzcount = sum(1 for x in r_final if x != 0)
        maxabs = max((abs(x) for x in r_final), default=Fraction(0))
        print(f'[round_and_repair] final exact residual: '
              f'nonzero entries = {nzcount}, max|entry| = {maxabs}')
    for k in range(sdp.n_y):
        if r_final[k] != 0:
            raise ValueError(
                f'rational stationarity failed at y_index {k}: '
                f'r = {r_final[k]}')

    # Bound = b^T lambda = lambda[0] (only b[0] = 1; rest = 0)
    lb_rig = lam_final[0] * Fraction(1)  # b[0] = 1
    # all other b[i] = 0 so dropped

    return CertifiedDual(
        d=d, order=order,
        lam_windows=lam_win_f,
        lambda_A=lam_final,
        L_blocks=L_rat_list,
        slack=slack_list,
        exact_residual=r_final,
        lb_rig=lb_rig,
    )


def _repair_with_diag_slack(sdp: SDPData, lam_A_rat: np.ndarray,
                             L_rat_list: List[np.ndarray],
                             S_rat_list: List[np.ndarray],
                             r_frac: np.ndarray,
                             c_rat: np.ndarray,
                             lam_win_f: np.ndarray,
                             verbose: bool = False) -> CertifiedDual:
    """Absorb residual via NONNEG DIAGONAL slacks per block.

    For each block j of size n_j, introduce n_j slack variables d_{j,k} >= 0,
    one per diagonal entry.  The augmented PSD matrix is
        S_j  =  L_j L_j^T  +  diag(d_{j,1}, ..., d_{j,n_j})                 (*)
    which is PSD by construction (sum of PSD + PSD-diagonal).

    The residual-repair linear system then becomes
        A^T * delta_lambda  +  sum_j sum_k d_{j,k} * w_{j,k}  =  r
    where w_{j,k} = adj(e_k e_k^T) is the vector with 1 at y-index of
    (2 * basis_j[k]) and 0 elsewhere.  Total free variables:
        n_free = n_eq + sum_j n_j
    For our sizes (d<=16, order<=3) this is substantially larger than n_y,
    so the system is underdetermined and we can satisfy it with nonneg d.

    Strategy:
      Phase 1 (float LP): find a float solution (delta, d) with d >= 0 that
                          satisfies the equation approximately.
      Phase 2 (rational): round to rationals and solve the resulting exact
                          system (which is guaranteed linearly consistent
                          if Phase 1 succeeded).
    """
    import cvxpy as cp
    from sympy import Matrix as SyMatrix, Rational as R

    n_eq = sdp.n_eq
    n_y = sdp.n_y
    B = len(sdp.blocks)
    block_sizes = [blk.size for blk in sdp.blocks]
    total_slacks = sum(block_sizes)

    # Build augmented matrix columns.
    # A^T is n_y x n_eq.
    AT = sdp.A.T.toarray()  # int entries
    # W[:, col] where col iterates over (block j, diag index k) = adj_j(e_k e_k^T).
    W_cols = []
    w_list_rat: List[List[np.ndarray]] = []  # store rational columns per block
    for blk in sdp.blocks:
        nj = blk.size
        per_block_cols = []
        for k in range(nj):
            E = np.zeros((nj, nj), dtype=np.float64)
            E[k, k] = 1.0
            # adj in floats (ints are fine)
            v_float = (blk.G_flat.T @ E.ravel())
            W_cols.append(v_float)
            # adj rational (same, since entries are int)
            E_frac = np.empty((nj, nj), dtype=object)
            for a in range(nj):
                for b in range(nj):
                    E_frac[a, b] = Fraction(1 if (a == b and a == k) else 0)
            v_rat = _adjoint_block_rational(blk, E_frac)
            per_block_cols.append(v_rat)
        w_list_rat.append(per_block_cols)
    W = np.stack(W_cols, axis=1) if W_cols else np.zeros((n_y, 0))
    # Full matrix M_aug: (n_y) x (n_eq + total_slacks)
    M_aug = np.concatenate([AT, W], axis=1)

    # Phase 1: float LP (feasibility w/ nonneg slack)
    r_float = np.array([float(x) for x in r_frac], dtype=np.float64)
    delta_var = cp.Variable(n_eq)
    d_var = cp.Variable(total_slacks, nonneg=True)
    eq = M_aug[:, :n_eq] @ delta_var + M_aug[:, n_eq:] @ d_var == r_float
    prob = cp.Problem(cp.Minimize(cp.sum(d_var)), [eq])
    try:
        prob.solve(solver='CLARABEL', verbose=False)
    except Exception:
        prob.solve(solver='SCS', verbose=False)
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise ValueError(
            f'[_repair_with_diag_slack] float LP failed: status={prob.status}. '
            f'Try higher rounding precision or enlarge slack dimension.')
    delta_fl = np.array(delta_var.value, dtype=np.float64).ravel()
    d_fl = np.array(d_var.value, dtype=np.float64).ravel()
    d_fl = np.maximum(d_fl, 0.0)
    if verbose:
        print(f'[_repair_with_diag_slack] float LP OK: '
              f'||delta||={np.linalg.norm(delta_fl):.3e}, '
              f'sum d = {d_fl.sum():.3e}, max d = {d_fl.max():.3e}')

    # Phase 2: round d_fl to rationals (nonneg, bounded denom),
    # then solve the linear equation (A^T | W_reduced) * delta = r - W @ d
    # in exact rationals, where "W_reduced" is empty (we've fixed d).
    # If the solve fails, we will bump the rounding denominator until it works.
    # But typically rounding d to small denom + solving A^T delta = (r - W @ d)
    # works because r - W @ d is close to col(A^T) after we absorb the
    # orthogonal-complement part via d.

    # Strategy: round d to 10^8 then fine-tune d so that (r - W @ d)
    # is EXACTLY in col(A^T).

    # Best: do both delta AND d rational via a single large sympy solve.
    # Build the augmented system [AT | W] * x = r in rationals.
    # AT: int entries.  W: int entries (adj of 0/1 matrices, all ints).
    # r: Fraction entries.
    n_cols = n_eq + total_slacks

    # Use sympy linsolve.  For n_y ~ 70 and n_cols ~ 71, this is fast.
    AT_sym_rows = []
    for i in range(n_y):
        row = []
        for j in range(n_eq):
            row.append(R(int(round(AT[i, j]))))
        for j in range(total_slacks):
            # W entries are integers (0/1 sums from adj of integer matrix)
            row.append(R(int(round(W[i, j]))))
        AT_sym_rows.append(row)
    M_sym = SyMatrix(AT_sym_rows)
    rhs_sym = SyMatrix([[R(r_frac[i].numerator, r_frac[i].denominator)] for i in range(n_y)])
    aug_sym = M_sym.row_join(rhs_sym)
    rref_mat, pivots = aug_sym.rref()
    if n_cols in pivots:
        raise ValueError('[_repair_with_diag_slack] sympy inconsistent '
                         'even with diagonal slacks')

    # Parametric solution: non-pivot columns are free parameters.
    # To pick a SPECIFIC solution, set all free parameters equal to their
    # float-LP values (rounded to rationals).
    pivot_cols = set(pivots)
    free_cols = [c for c in range(n_cols) if c not in pivot_cols]

    # Initial float-LP guesses for free variables
    free_vals = {}
    for c in free_cols:
        if c < n_eq:
            free_vals[c] = _float_to_fraction(float(delta_fl[c]), 10**8)
        else:
            free_vals[c] = _float_to_fraction(float(d_fl[c - n_eq]), 10**8)

    # Build particular solution
    x = [Fraction(0)] * n_cols
    # Determine pivot variables from rref
    # rref: n_pivots rows give: x[pivot_cols[i]] + sum_{c free} rref[i,c]*x[c] = rref[i, n_cols]
    n_piv = len(pivots)
    for i in range(n_piv):
        pc = pivots[i]
        # x[pc] = rhs - sum_{c free, c > pc} rref[i,c] * x[c]
        val_sym = rref_mat[i, n_cols]
        for c in free_cols:
            coef_sym = rref_mat[i, c]
            if coef_sym == 0:
                continue
            # Convert symbolic rationals
            coef_frac = Fraction(int(coef_sym.p), int(coef_sym.q)) \
                if coef_sym != 0 else Fraction(0)
            val_sym -= coef_sym * R(free_vals[c].numerator, free_vals[c].denominator)
        # Convert val_sym to fraction
        val_rat = (Fraction(int(val_sym.p), int(val_sym.q))
                   if val_sym != 0 else Fraction(0))
        x[pc] = val_rat

    for c in free_cols:
        x[c] = free_vals[c]

    # Verify nonneg slacks
    slack_list: List[np.ndarray] = []
    idx = n_eq
    for j, nj in enumerate(block_sizes):
        slack_j = np.empty(nj, dtype=object)
        for k in range(nj):
            v = x[idx + k]
            if v < 0:
                # Try again: snap this free variable to zero and re-solve
                # For simplicity, if any slack is negative, bump to zero and
                # redistribute -- too complex.  Fail and report.
                if verbose:
                    print(f'[_repair_with_diag_slack] slack d[{j},{k}]={v} '
                          f'negative; retrying with snap.')
                # Retry: set this free var to 0 in free_vals, recompute
                return _repair_retry_zero_negatives(
                    sdp, lam_A_rat, L_rat_list, S_rat_list,
                    r_frac, c_rat, lam_win_f, x,
                    rref_mat, pivots, free_cols, free_vals,
                    n_eq, total_slacks, block_sizes, verbose=verbose,
                )
            slack_j[k] = v
        slack_list.append(slack_j)
        idx += nj

    delta_rat = np.empty(n_eq, dtype=object)
    for k in range(n_eq):
        delta_rat[k] = x[k]

    # Apply
    lam_final = np.empty(n_eq, dtype=object)
    for k in range(n_eq):
        lam_final[k] = lam_A_rat[k] + delta_rat[k]

    # Verify exact residual = 0
    AT_csr = sdp.A.T.tocsr()
    AT_lambda_new = _frac_sparse_matvec(AT_csr, lam_final)
    r_final = np.empty(n_y, dtype=object)
    for k in range(n_y):
        r_final[k] = c_rat[k] - AT_lambda_new[k]
    for blk, S_frac in zip(sdp.blocks, S_rat_list):
        adj_S = _adjoint_block_rational(blk, S_frac)
        for k in range(n_y):
            r_final[k] -= adj_S[k]
    for j, (blk, slack_j) in enumerate(zip(sdp.blocks, slack_list)):
        for k in range(blk.size):
            if slack_j[k] == 0:
                continue
            for iy in range(n_y):
                r_final[iy] -= slack_j[k] * w_list_rat[j][k][iy]

    nz = sum(1 for x in r_final if x != 0)
    max_abs = max((abs(x) for x in r_final), default=Fraction(0))
    if verbose:
        total_slack_val = Fraction(0)
        for sl in slack_list:
            for v in sl:
                total_slack_val += v
        print(f'[_repair_with_diag_slack] exact residual nonzeros={nz}, '
              f'max|entry|={max_abs}, total slack = {float(total_slack_val):.3e}')
    if nz > 0:
        raise ValueError(
            f'[_repair_with_diag_slack] exact residual has {nz} nonzero entries '
            f'after diag-slack repair; max|entry|={max_abs}')

    lb_rig = lam_final[0]
    return CertifiedDual(
        d=sdp.d, order=sdp.order,
        lam_windows=lam_win_f,
        lambda_A=lam_final,
        L_blocks=L_rat_list,
        slack=slack_list,
        exact_residual=r_final,
        lb_rig=lb_rig,
    )


def _repair_retry_zero_negatives(
        sdp, lam_A_rat, L_rat_list, S_rat_list,
        r_frac, c_rat, lam_win_f, x,
        rref_mat, pivots, free_cols, free_vals,
        n_eq, total_slacks, block_sizes, verbose=False):
    """When a diagonal slack comes out negative, snap it to zero and retry.

    This is a simple greedy fix: iterate, snap the most-negative slack to
    zero (make it a pivot variable by removing it from free set).  Not
    globally optimal, but usually sufficient when initial LP was feasible.
    """
    from sympy import Matrix as SyMatrix, Rational as R

    n_y = sdp.n_y
    n_cols = n_eq + total_slacks
    pivot_cols = list(pivots)

    # Naive retry: set ALL slacks with negative values in x to 0, re-solve
    # the system with remaining free parameters.
    # Rebuild from rref:
    all_free = free_cols[:]
    attempts = 0
    while attempts < 10:
        attempts += 1
        # Identify negative slacks
        neg_slacks = []
        idx = n_eq
        for j, nj in enumerate(block_sizes):
            for k in range(nj):
                if x[idx + k] < 0:
                    neg_slacks.append(idx + k)
                idx += nj
        if not neg_slacks:
            break
        for c in neg_slacks:
            free_vals[c] = Fraction(0)
        # Recompute pivot vars
        for i, pc in enumerate(pivot_cols):
            val_sym = rref_mat[i, n_cols]
            for c in all_free:
                coef_sym = rref_mat[i, c]
                if coef_sym == 0:
                    continue
                val_sym -= coef_sym * R(free_vals[c].numerator,
                                        free_vals[c].denominator)
            x[pc] = (Fraction(int(val_sym.p), int(val_sym.q))
                     if val_sym != 0 else Fraction(0))
        for c in all_free:
            x[c] = free_vals[c]

    # Final check
    slack_list = []
    idx = n_eq
    for j, nj in enumerate(block_sizes):
        slack_j = np.empty(nj, dtype=object)
        for k in range(nj):
            v = x[idx + k]
            if v < 0:
                raise ValueError(
                    f'[_repair_retry_zero_negatives] slack [{j},{k}]={v} '
                    f'still negative after {attempts} attempts')
            slack_j[k] = v
        slack_list.append(slack_j)
        idx += nj

    delta_rat = np.empty(n_eq, dtype=object)
    for k in range(n_eq):
        delta_rat[k] = x[k]
    lam_final = np.empty(n_eq, dtype=object)
    for k in range(n_eq):
        lam_final[k] = lam_A_rat[k] + delta_rat[k]

    # Verify residual
    AT_csr = sdp.A.T.tocsr()
    AT_lambda_new = _frac_sparse_matvec(AT_csr, lam_final)
    r_final = np.empty(n_y, dtype=object)
    for k in range(n_y):
        r_final[k] = c_rat[k] - AT_lambda_new[k]
    for blk, S_frac in zip(sdp.blocks, S_rat_list):
        adj_S = _adjoint_block_rational(blk, S_frac)
        for k in range(n_y):
            r_final[k] -= adj_S[k]
    # Recompute w_list_rat inline
    for j, (blk, slack_j) in enumerate(zip(sdp.blocks, slack_list)):
        nj = blk.size
        for k in range(nj):
            if slack_j[k] == 0:
                continue
            E_frac = np.empty((nj, nj), dtype=object)
            for a in range(nj):
                for b in range(nj):
                    E_frac[a, b] = Fraction(1 if (a == b and a == k) else 0)
            v_rat = _adjoint_block_rational(blk, E_frac)
            for iy in range(n_y):
                r_final[iy] -= slack_j[k] * v_rat[iy]

    nz = sum(1 for x in r_final if x != 0)
    if nz > 0:
        raise ValueError(
            f'[_repair_retry_zero_negatives] residual still has {nz} nonzero '
            f'entries after snapping negatives')

    lb_rig = lam_final[0]
    return CertifiedDual(
        d=sdp.d, order=sdp.order,
        lam_windows=lam_win_f,
        lambda_A=lam_final,
        L_blocks=L_rat_list,
        slack=slack_list,
        exact_residual=r_final,
        lb_rig=lb_rig,
    )
