"""Construct the Handelman + variable-lambda LP for the Sidon problem.

LP (standard form, all variables in a single x vector):

  Maximize alpha
  Subject to:
    sum_W lambda_W = 1
    lambda_W >= 0
    alpha free, q_K free
    c_beta >= 0
    For each |beta| <= R:
      sum_W lambda_W * coeff_W(beta)
        - alpha * delta_{beta = 0}
        + q_beta - sum_j q_{beta - e_j}
        - c_beta = 0

Here:
  coeff_W(beta) = [mu^T M_W mu]_beta   (the coefficient of mu^beta in
                                       the homogeneous quadratic
                                       mu^T M_W mu)
  = M_W[i,i]      if beta = 2 e_i
  = 2 M_W[i,j]    if beta = e_i + e_j with i < j
  = 0             if |beta| != 2
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.poly import (
    enum_monomials_le,
    index_map,
    shift_minus,
    multinomial,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class BuildOptions:
    """Build-time switches for the Handelman LP."""
    R: int = 4                      # total degree of the certificate
    use_z2: bool = True             # apply Z/2 symmetry reduction
    fixed_lambda: Optional[np.ndarray] = None  # if not None, lambda is fixed
    use_q_polynomial: bool = True   # include q*(1 - sum mu) multiplier
    drop_zero_constraints: bool = False  # drop equalities where coeff matrix has only c_beta and trivial RHS
    verbose: bool = False
    # NEW (Phase 1): Newton-polytope / term-sparsity restriction.
    # If both are provided, the build enumerates equality constraints only
    # for beta in restricted_Sigma_R and q-variables only for K in
    # restricted_B_R. Defaults None preserve the old behavior exactly.
    restricted_Sigma_R: Optional[List[Tuple[int, ...]]] = None
    restricted_B_R: Optional[List[Tuple[int, ...]]] = None
    # NEW (Q2.a from agent audit): eliminate c_beta slacks. Convert
    # equalities (LHS - c_beta = 0, c_beta >= 0) to inequalities
    # (LHS <= 0). Saves n_le_R variables and n_le_R nonzeros. At
    # d=64 R=8 this is 76M variable reduction. The LP then has
    # inequality rows (handled natively by MOSEK / cuOpt).
    eliminate_c_slacks: bool = False
    # NEW (Q3 from agent audit): apply box bounds to free variables (alpha, q)
    # to help PDLP. Set to None to leave free; set to e.g. 1e6 for soft box.
    free_var_box: Optional[float] = None
    # NEW (Q7 from agent audit): scale M_W by 1/scale to bring entries to
    # O(1). The optimal alpha then needs to be multiplied by `scale` to
    # recover the original-LP alpha. If None, no scaling is applied.
    objective_scale: Optional[float] = None


@dataclass
class BuildResult:
    """Output of build_handelman_lp."""
    # LP data (scipy sparse, HiGHS-friendly)
    A_eq: sp.csr_matrix         # equality constraint matrix (or inequality A_ub if c_slacks eliminated)
    b_eq: np.ndarray            # equality RHS (or b_ub if c_slacks eliminated)
    A_ub: Optional[sp.csr_matrix]  # inequality (None if all are equalities)
    b_ub: Optional[np.ndarray]
    c: np.ndarray               # objective (we MINIMIZE c^T x, so c[alpha_idx] = -1)
    bounds: List[Tuple[Optional[float], Optional[float]]]  # (lo, hi) per variable
    # Variable layout
    n_vars: int
    alpha_idx: int
    lambda_idx: slice           # range of lambda variables (empty if fixed)
    q_idx: slice                # range of q polynomial variables
    c_idx: slice                # range of c_beta slack variables (empty if eliminated)
    # Bookkeeping
    monos_le_R: List[Tuple[int, ...]]
    monos_le_Rm1: List[Tuple[int, ...]]
    n_windows: int
    fixed_lambda: Optional[np.ndarray]
    options: BuildOptions
    build_wall_s: float
    n_nonzero_A: int
    # NEW: track if c_slacks were eliminated (for solver to use ineq form)
    c_slacks_eliminated: bool = False
    # NEW: when objective_scale was applied; alpha must be multiplied back.
    objective_scale_applied: float = 1.0


# ---------------------------------------------------------------------
# Window matrices (re-implementation, decoupled from lasserre.core)
# ---------------------------------------------------------------------

def build_window_matrices(d: int) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """Build all Sidon window matrices for d bins.

    Mirrors lasserre.core.build_window_matrices: M_W[i,j] = (2d/ell)
    if s_lo <= i+j <= s_lo + ell - 2, else 0.
    Returns (window_specs, M_mats).
    """
    conv_len = 2 * d - 1
    windows = [(ell, s) for ell in range(2, 2 * d + 1)
               for s in range(conv_len - ell + 2)]
    ii, jj = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
    sums = ii + jj
    M_mats: List[np.ndarray] = []
    for ell, s_lo in windows:
        mask = (sums >= s_lo) & (sums <= s_lo + ell - 2)
        M_mats.append((2.0 * d / ell) * mask.astype(np.float64))
    return windows, M_mats


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

def _coeff_W_at_beta(M_W: np.ndarray, beta: Tuple[int, ...]) -> float:
    """Coefficient of mu^beta in the polynomial mu^T M_W mu (M_W symmetric).

    mu^T M_W mu = sum_i M_W[i,i] mu_i^2 + sum_{i<j} 2 M_W[i,j] mu_i mu_j.
    """
    if sum(beta) != 2:
        return 0.0
    nz = [(i, b) for i, b in enumerate(beta) if b > 0]
    if len(nz) == 1:
        i, b = nz[0]
        if b == 2:
            return float(M_W[i, i])
        return 0.0
    if len(nz) == 2:
        (i, _), (j, _) = nz
        return 2.0 * float(M_W[i, j])
    return 0.0


def _build_void_lookup(
    monos_le_R: List[Tuple[int, ...]],
    n_vars_poly: int,
) -> Dict[bytes, int]:
    """Build a {bytes -> row-index} dict for fast batched monomial lookups.

    Each monomial is encoded as int8 row bytes, which is faster to hash
    than Python tuples (no per-element hashing).
    """
    if not monos_le_R:
        return {}
    base_arr = np.asarray(monos_le_R, dtype=np.int8)
    base_arr_c = np.ascontiguousarray(base_arr)
    void_dt = np.dtype((np.void, base_arr_c.dtype.itemsize * n_vars_poly))
    base_void = base_arr_c.view(void_dt).ravel()
    return {bytes(v): i for i, v in enumerate(base_void.tolist())}


def _batch_lookup(
    queries: np.ndarray,
    void_lookup: Dict[bytes, int],
) -> np.ndarray:
    """Vectorized monomial -> row-index lookup using a precomputed dict.

    queries: (N, d) int array of multi-indices.
    Returns: (N,) int64 array of row indices, with -1 for missing.
    """
    if queries.size == 0:
        return np.zeros(0, dtype=np.int64)
    n_vars_poly = queries.shape[1]
    queries_c = np.ascontiguousarray(queries.astype(np.int8, copy=False))
    void_dt = np.dtype((np.void, queries_c.dtype.itemsize * n_vars_poly))
    q_void = queries_c.view(void_dt).ravel()
    return np.fromiter(
        (void_lookup.get(bytes(v), -1) for v in q_void.tolist()),
        dtype=np.int64, count=queries.shape[0],
    )


def _coeff_W_vector(M_W: np.ndarray, beta_to_idx: Dict[Tuple[int, ...], int]) -> Dict[int, float]:
    """Sparse {beta_idx: coeff} of the quadratic mu^T M_W mu over all betas.

    Only the |beta|=2 entries are nonzero. Uses the upper-triangular nonzeros
    of M_W to avoid O(d^2) work when M_W is sparse.
    """
    out: Dict[int, float] = {}
    d = M_W.shape[0]
    nz_i, nz_j = np.where(np.triu(M_W != 0))
    for i, j in zip(nz_i, nz_j):
        if i == j:
            beta = tuple(2 if k == i else 0 for k in range(d))
            v = float(M_W[i, i])
        else:
            beta = tuple(1 if (k == i or k == j) else 0 for k in range(d))
            v = 2.0 * float(M_W[i, j])
        idx = beta_to_idx.get(beta)
        if idx is not None and v != 0.0:
            out[idx] = out.get(idx, 0.0) + v
    return out


def build_handelman_lp(
    d: int,
    M_mats: Sequence[np.ndarray],
    options: BuildOptions,
) -> BuildResult:
    """Build the Handelman LP for given d, list of window matrices.

    The ambient variable count is d (after Z/2 reduction the caller is
    expected to have already projected M_mats to the symmetric basis;
    see lasserre.polya_lp.symmetry.project_window_set_to_z2). This
    function does NOT apply Z/2 internally to keep the math transparent.
    """
    t0 = time.time()
    R = options.R
    n_vars_poly = d  # ambient polynomial dimension

    # ---------------------------------------------------------------
    # Monomial enumerations
    #
    # If options.restricted_Sigma_R is set, use it as the constraint set
    # (Newton-polytope / term-sparsity restriction). Soundness: per
    # term_sparsity.py module docstring, the restricted LP has the same
    # max alpha as the full LP (Mai-Magron-Lasserre-Toh 2022 §3.2).
    # ---------------------------------------------------------------
    if options.restricted_Sigma_R is not None:
        monos_le_R = list(options.restricted_Sigma_R)
        # Ensure beta=0 is present (alpha lives in row beta=0).
        zero = tuple([0] * n_vars_poly)
        if zero not in set(monos_le_R):
            monos_le_R = [zero] + monos_le_R
    else:
        monos_le_R = enum_monomials_le(n_vars_poly, R)
    n_le_R = len(monos_le_R)
    beta_to_idx = index_map(monos_le_R)

    if options.use_q_polynomial:
        if options.restricted_B_R is not None:
            monos_le_Rm1 = list(options.restricted_B_R)
        else:
            monos_le_Rm1 = enum_monomials_le(n_vars_poly, R - 1)
    else:
        monos_le_Rm1 = []
    q_to_idx = index_map(monos_le_Rm1)
    n_q = len(monos_le_Rm1)

    # ---------------------------------------------------------------
    # Lambda layout
    # ---------------------------------------------------------------
    n_W = len(M_mats)
    if options.fixed_lambda is not None:
        assert len(options.fixed_lambda) == n_W, \
            f"fixed_lambda length {len(options.fixed_lambda)} != n_W {n_W}"
        n_lambda_var = 0
    else:
        n_lambda_var = n_W

    # ---------------------------------------------------------------
    # Variable layout: x = [alpha | lambda | q | c]
    # If eliminate_c_slacks=True, the c block is OMITTED and the rows
    # become inequality rows (LHS <= 0) instead of equality.
    # ---------------------------------------------------------------
    alpha_idx = 0
    lambda_start = 1
    lambda_stop = lambda_start + n_lambda_var
    q_start = lambda_stop
    q_stop = q_start + n_q
    c_start = q_stop
    n_c_var = 0 if options.eliminate_c_slacks else n_le_R
    c_stop = c_start + n_c_var
    n_vars = c_stop

    if options.verbose:
        print(f"  LP layout: alpha=1, lambda={n_lambda_var}, "
              f"q={n_q}, c={n_le_R}; total n_vars={n_vars}")
        print(f"  Equality constraints (per beta with |beta| <= R): {n_le_R}")
        if n_lambda_var > 0:
            print(f"  Plus 1 simplex equality for lambda. Grand total {n_le_R + 1}.")

    # ---------------------------------------------------------------
    # Build A_eq via COO triplets (VECTORIZED).
    #
    # Equality (per beta):
    #   sum_W lambda_W coeff_W(beta)
    #       - alpha * delta_{beta=0}
    #       + q_beta - sum_j q_{beta - e_j}
    #       - c_beta
    #     = (- coeff_fixed(beta))   if lambda fixed
    #       0                        otherwise
    #
    # We compute the COO triplets as numpy arrays in 4 blocks (alpha,
    # lambda, q, c) and concatenate. Hot loops are vectorized via
    # np.void byte-views for monomial->index lookups.
    # ---------------------------------------------------------------
    triplet_chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rhs = np.zeros(n_le_R, dtype=np.float64)

    # alpha: -alpha in row beta=0
    zero_beta = tuple([0] * n_vars_poly)
    zero_idx = beta_to_idx[zero_beta]
    triplet_chunks.append((
        np.array([zero_idx], dtype=np.int64),
        np.array([alpha_idx], dtype=np.int64),
        np.array([-1.0], dtype=np.float64),
    ))

    # ---------------------------------------------------------------
    # Build a single lookup dict for all monomial -> row-index queries.
    # ---------------------------------------------------------------
    void_lookup = _build_void_lookup(monos_le_R, n_vars_poly)

    # ---------------------------------------------------------------
    # Vectorized helpers shared by lambda and q blocks
    # ---------------------------------------------------------------
    # diag_idx[i] = beta_to_idx[2*e_i] (or -1 if not in monos_le_R)
    diag_betas = np.zeros((n_vars_poly, n_vars_poly), dtype=np.int8)
    for i in range(n_vars_poly):
        diag_betas[i, i] = 2
    diag_idx = _batch_lookup(diag_betas, void_lookup)

    # cross_idx[k] = beta_to_idx[e_i + e_j] for k = packed (i<j)
    if n_vars_poly >= 2:
        ii_grid, jj_grid = np.meshgrid(
            np.arange(n_vars_poly, dtype=np.int64),
            np.arange(n_vars_poly, dtype=np.int64),
            indexing='ij',
        )
        triu = ii_grid < jj_grid
        i_arr = ii_grid[triu]
        j_arr = jj_grid[triu]
        n_cross = i_arr.size
        cross_betas = np.zeros((n_cross, n_vars_poly), dtype=np.int8)
        cross_betas[np.arange(n_cross), i_arr] = 1
        cross_betas[np.arange(n_cross), j_arr] = 1
        cross_idx = _batch_lookup(cross_betas, void_lookup)
    else:
        i_arr = np.zeros(0, dtype=np.int64)
        j_arr = np.zeros(0, dtype=np.int64)
        cross_idx = np.zeros(0, dtype=np.int64)
        n_cross = 0

    # ---------------------------------------------------------------
    # Lambda block: lambda_W * coeff_W(beta)
    # coeff_W(2*e_i)     = M_W[i,i]
    # coeff_W(e_i + e_j) = 2 * M_W[i,j]   (M_W symmetric)
    # ---------------------------------------------------------------
    if n_W > 0:
        M_stack = np.stack([np.asarray(M, dtype=np.float64) for M in M_mats], axis=0)
        # diag_coefs[w, i] = M_stack[w, i, i]
        diag_coefs = np.diagonal(M_stack, axis1=1, axis2=2)  # (n_W, d)
        # cross_coefs[w, k] = 2 * M_stack[w, i_arr[k], j_arr[k]]
        if n_cross > 0:
            cross_coefs = 2.0 * M_stack[:, i_arr, j_arr]  # (n_W, n_cross)
        else:
            cross_coefs = np.zeros((n_W, 0), dtype=np.float64)

        if options.fixed_lambda is None:
            w_col = lambda_start + np.arange(n_W, dtype=np.int64)  # (n_W,)
            # diag triplets
            d_rows_b = np.broadcast_to(diag_idx[None, :], (n_W, n_vars_poly)).ravel()
            d_cols_b = np.broadcast_to(w_col[:, None], (n_W, n_vars_poly)).ravel()
            d_vals_b = diag_coefs.ravel()
            keep = (d_rows_b >= 0) & (d_vals_b != 0)
            triplet_chunks.append((d_rows_b[keep].copy(),
                                   d_cols_b[keep].copy(),
                                   d_vals_b[keep].copy()))
            # cross triplets
            if n_cross > 0:
                c_rows_b = np.broadcast_to(cross_idx[None, :], (n_W, n_cross)).ravel()
                c_cols_b = np.broadcast_to(w_col[:, None], (n_W, n_cross)).ravel()
                c_vals_b = cross_coefs.ravel()
                keep = (c_rows_b >= 0) & (c_vals_b != 0)
                triplet_chunks.append((c_rows_b[keep].copy(),
                                       c_cols_b[keep].copy(),
                                       c_vals_b[keep].copy()))
        else:
            # Fixed lambda: aggregate into rhs.
            lam = np.asarray(options.fixed_lambda, dtype=np.float64)
            # sum_W lam_W * diag_coefs[w, i] -> per-i contribution
            agg_diag = (lam[:, None] * diag_coefs).sum(axis=0)  # (d,)
            keep = (diag_idx >= 0) & (agg_diag != 0)
            np.subtract.at(rhs, diag_idx[keep], agg_diag[keep])
            if n_cross > 0:
                agg_cross = (lam[:, None] * cross_coefs).sum(axis=0)  # (n_cross,)
                keep = (cross_idx >= 0) & (agg_cross != 0)
                np.subtract.at(rhs, cross_idx[keep], agg_cross[keep])

    # ---------------------------------------------------------------
    # q polynomial block (vectorized monomial->index lookup)
    #
    #   +q_K  in row K
    #   -q_K  in row K + e_j  for each j
    # ---------------------------------------------------------------
    if options.use_q_polynomial and n_q > 0:
        monos_Rm1_arr = np.asarray(monos_le_Rm1, dtype=np.int8)
        # +q_K: row index = beta_to_idx[K]
        plus_rows = _batch_lookup(monos_Rm1_arr, void_lookup)
        keep = plus_rows >= 0
        if keep.any():
            triplet_chunks.append((
                plus_rows[keep].copy(),
                (q_start + np.arange(n_q, dtype=np.int64)[keep]).copy(),
                np.ones(int(keep.sum()), dtype=np.float64),
            ))
        # -q_K: row index = beta_to_idx[K + e_j], for all j
        e_j_mat = np.eye(n_vars_poly, dtype=np.int8)
        # shifts[k, j, :] = K_arr[k] + e_j[j]
        shifts = monos_Rm1_arr[:, None, :] + e_j_mat[None, :, :]
        shifts = shifts.reshape(n_q * n_vars_poly, n_vars_poly)
        minus_rows = _batch_lookup(shifts, void_lookup)
        keep = minus_rows >= 0
        if keep.any():
            k_idx_arr = np.repeat(np.arange(n_q, dtype=np.int64), n_vars_poly)
            triplet_chunks.append((
                minus_rows[keep].copy(),
                (q_start + k_idx_arr[keep]).copy(),
                -np.ones(int(keep.sum()), dtype=np.float64),
            ))

    # ---------------------------------------------------------------
    # c_beta block: -c_beta in row beta (only if NOT eliminating)
    # ---------------------------------------------------------------
    if not options.eliminate_c_slacks:
        triplet_chunks.append((
            np.arange(n_le_R, dtype=np.int64),
            c_start + np.arange(n_le_R, dtype=np.int64),
            -np.ones(n_le_R, dtype=np.float64),
        ))

    # ---------------------------------------------------------------
    # Concatenate triplets and build the sparse matrix.
    # ---------------------------------------------------------------
    all_rows = np.concatenate([c[0] for c in triplet_chunks])
    all_cols = np.concatenate([c[1] for c in triplet_chunks])
    all_vals = np.concatenate([c[2] for c in triplet_chunks])
    A_eq = sp.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_le_R, n_vars),
    )
    b_eq = rhs

    # ---------------------------------------------------------------
    # Add the simplex constraint for lambda
    # ---------------------------------------------------------------
    if n_lambda_var > 0:
        sim_row = sp.csr_matrix(
            (np.ones(n_lambda_var),
             (np.zeros(n_lambda_var, dtype=np.int64),
              np.arange(lambda_start, lambda_stop, dtype=np.int64))),
            shape=(1, n_vars),
        )
        A_eq = sp.vstack([A_eq, sim_row], format="csr")
        b_eq = np.concatenate([b_eq, np.array([1.0])])

    # ---------------------------------------------------------------
    # Variable bounds
    # ---------------------------------------------------------------
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    fb = options.free_var_box
    bounds.append((-fb, fb) if fb else (None, None))  # alpha free or boxed
    for _ in range(n_lambda_var):
        bounds.append((0.0, None))  # lambda >= 0
    for _ in range(n_q):
        bounds.append((-fb, fb) if fb else (None, None))  # q free or boxed
    if not options.eliminate_c_slacks:
        for _ in range(n_le_R):
            bounds.append((0.0, None))  # c >= 0

    # ---------------------------------------------------------------
    # Objective (minimize -alpha)
    # ---------------------------------------------------------------
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[alpha_idx] = -1.0

    # If c_slacks eliminated, the rows for beta are inequalities.
    # Original equality:  LHS - c_beta = b_eq  with  c_beta >= 0
    # => c_beta = LHS - b_eq >= 0  =>  LHS >= b_eq
    # In scipy/MOSEK convention  A_ub x <= b_ub, so negate both sides:
    #   -LHS <= -b_eq
    # The simplex equality (sum lambda = 1) remains an equality.
    if options.eliminate_c_slacks:
        A_ub_only = -A_eq[:n_le_R, :]               # negated for A_ub x <= b_ub
        b_ub_only = -b_eq[:n_le_R]
        A_eq_only = A_eq[n_le_R:, :] if A_eq.shape[0] > n_le_R else None
        b_eq_only = b_eq[n_le_R:] if A_eq.shape[0] > n_le_R else None
        return BuildResult(
            A_eq=A_eq_only if A_eq_only is not None and A_eq_only.shape[0] > 0
                  else sp.csr_matrix((0, n_vars)),
            b_eq=b_eq_only if b_eq_only is not None
                  else np.zeros(0),
            A_ub=A_ub_only,
            b_ub=b_ub_only,
            c=c_obj,
            bounds=bounds,
            n_vars=n_vars,
            alpha_idx=alpha_idx,
            lambda_idx=slice(lambda_start, lambda_stop),
            q_idx=slice(q_start, q_stop),
            c_idx=slice(c_start, c_stop),
            monos_le_R=monos_le_R,
            monos_le_Rm1=monos_le_Rm1,
            n_windows=n_W,
            fixed_lambda=options.fixed_lambda,
            options=options,
            build_wall_s=time.time() - t0,
            n_nonzero_A=A_eq.nnz,
            c_slacks_eliminated=True,
        )

    return BuildResult(
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=None,
        b_ub=None,
        c=c_obj,
        bounds=bounds,
        n_vars=n_vars,
        alpha_idx=alpha_idx,
        lambda_idx=slice(lambda_start, lambda_stop),
        q_idx=slice(q_start, q_stop),
        c_idx=slice(c_start, c_stop),
        monos_le_R=monos_le_R,
        monos_le_Rm1=monos_le_Rm1,
        n_windows=n_W,
        fixed_lambda=options.fixed_lambda,
        options=options,
        build_wall_s=time.time() - t0,
        n_nonzero_A=A_eq.nnz,
        c_slacks_eliminated=False,
    )
