"""Fast (vectorized) build of the dual-epigraph LP for high-d / high-R.

The reference build_dual_epi.py uses Python loops over monomials and
windows -- fine for d <= 16, R <= 8 (~ms), but unworkable at d = 24, R = 20
where n_le_R reaches ~225M. This module rebuilds the SAME LP using
numpy vectorized monomial enumeration + np.void byte-key batch lookups
(the same trick build.py uses for the primal lambda/q blocks).

Targets:
  - d=16 R=20 (~50M monomials): build in seconds, RSS < 4 GB
  - d=24 R=20 (~225M monomials): build in ~minute, RSS ~ 32 GB
  - d=32 R=15 (~300M monomials): build in ~minutes, RSS ~ 50 GB

Output: a DualEpiBuildResult IDENTICAL to build_dual_epi_lp's output
(same monomial ordering, same row/col indices, same A_eq / A_ub /
b / c / bounds). MOSEK / cuOpt / ortools all consume it the same way.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.tier_dual.build_dual_epi import DualEpiBuildResult


# =====================================================================
# Vectorized monomial enumeration in graded-lex order
# =====================================================================

def enum_monomials_le_np(d: int, R: int) -> np.ndarray:
    """All alpha in N^d with |alpha| <= R, in graded-lex order.

    Returned as (n_le_R, d) int8 array. Matches the order of
    poly.enum_monomials_le but constructed without per-element python.

    Implementation: for each total degree k = 0, 1, ..., R, generate
    all alpha with sum exactly k; concatenate. Within a fixed k, we use
    the standard recursive pattern but only at the OUTER loop level
    (with vectorized inner loops via cumulative sums).
    """
    if d <= 0 or R < 0:
        return np.zeros((0, max(d, 0)), dtype=np.int8)

    # Match the order of poly.enum_monomials_le exactly: that function
    # is recursive lex with the FIRST coordinate as the OUTER loop.
    # Equivalent: sorted by (alpha_0, alpha_1, ..., alpha_{d-1}) ascending,
    # restricted to sum <= R. Equivalent to lexicographic order of the
    # tuple (alpha_0, alpha_1, ..., alpha_{d-1}).
    #
    # We build it recursively but with numpy slicing.
    chunks: List[np.ndarray] = []
    cur = np.zeros(d, dtype=np.int8)
    _enum_lex(d, R, 0, R, cur, chunks)
    if not chunks:
        return np.zeros((0, d), dtype=np.int8)
    return np.concatenate(chunks, axis=0)


def _enum_lex(d: int, R: int, pos: int, remaining: int,
              cur: np.ndarray, chunks: List[np.ndarray]) -> None:
    if pos == d - 1:
        # last coordinate: a row per value 0..remaining
        n = remaining + 1
        block = np.broadcast_to(cur, (n, d)).copy()
        block[:, pos] = np.arange(n, dtype=np.int8)
        chunks.append(block)
        cur[pos] = 0
        return
    for v in range(remaining + 1):
        cur[pos] = v
        _enum_lex(d, R, pos + 1, remaining - v, cur, chunks)
    cur[pos] = 0


# =====================================================================
# Void-key batch lookup helpers
# =====================================================================

def _make_void_lookup(monos_arr: np.ndarray) -> dict:
    """Return {bytes_key: row_index} mapping for fast batch lookup."""
    n, d = monos_arr.shape
    arr_c = np.ascontiguousarray(monos_arr.astype(np.int8, copy=False))
    void_dt = np.dtype((np.void, d * np.dtype(np.int8).itemsize))
    arr_void = arr_c.view(void_dt).ravel()
    return {bytes(v): i for i, v in enumerate(arr_void.tolist())}


def _batch_lookup(queries: np.ndarray, lookup: dict) -> np.ndarray:
    """queries: (N, d) int array.  Return (N,) int64 indices, -1 if missing."""
    if queries.size == 0:
        return np.zeros(0, dtype=np.int64)
    n, d = queries.shape
    arr_c = np.ascontiguousarray(queries.astype(np.int8, copy=False))
    void_dt = np.dtype((np.void, d * np.dtype(np.int8).itemsize))
    q_void = arr_c.view(void_dt).ravel()
    return np.fromiter(
        (lookup.get(bytes(v), -1) for v in q_void.tolist()),
        dtype=np.int64, count=n,
    )


# =====================================================================
# Fast dual-epigraph build
# =====================================================================

def build_dual_epi_fast(
    d: int,
    M_mats: Sequence[np.ndarray],
    R: int,
    y_upper: float = 1.0,
    tau_upper: float = 10.0,
    verbose: bool = False,
) -> DualEpiBuildResult:
    """Vectorized epigraph dual build.  Drop-in for build_dual_epi_lp."""
    t0 = time.time()
    n_W = len(M_mats)

    # ----------------------------------------------------------------
    # 1. Enumerate ALL monos |alpha| <= R as a (n_le_R, d) int8 array.
    # ----------------------------------------------------------------
    monos_arr = enum_monomials_le_np(d, R)
    n_le_R = monos_arr.shape[0]
    sums = monos_arr.sum(axis=1, dtype=np.int64)

    if verbose:
        print(f"  [build_fast] d={d} R={R} n_le_R={n_le_R} ({(time.time()-t0)*1000:.0f}ms enum)",
              flush=True)

    void_lookup = _make_void_lookup(monos_arr)

    # ----------------------------------------------------------------
    # 2. Identify K's for moment recursion: |K| <= R-1
    # ----------------------------------------------------------------
    is_K_row = sums <= R - 1
    n_q = int(is_K_row.sum())
    K_arr = monos_arr[is_K_row]              # (n_q, d), the K rows
    K_self_col = np.flatnonzero(is_K_row)    # (n_q,) col index of K in monos_arr

    # zero index for y_0 = 1
    zero_void = np.zeros((1, d), dtype=np.int8)
    zero_idx_arr = _batch_lookup(zero_void, void_lookup)
    if zero_idx_arr[0] < 0:
        raise RuntimeError("zero monomial not in monos_arr; build is broken")
    zero_idx = int(zero_idx_arr[0])

    if verbose:
        print(f"  [build_fast] n_q (recursion rows)={n_q}", flush=True)

    # ----------------------------------------------------------------
    # 3. Variable layout
    # ----------------------------------------------------------------
    tau_idx = n_le_R
    n_vars = n_le_R + 1

    # ----------------------------------------------------------------
    # 4. Equality rows  (1 + n_q rows)
    #    Row 0: y_0 = 1   (one nonzero at col zero_idx, value +1; b=1)
    #    Rows 1..n_q: y_K - sum_{j} y_{K+e_j} = 0
    # ----------------------------------------------------------------
    n_eq_rows = 1 + n_q

    # Row 0
    eq_rows0 = np.array([0], dtype=np.int64)
    eq_cols0 = np.array([zero_idx], dtype=np.int64)
    eq_vals0 = np.array([1.0], dtype=np.float64)

    # Rows 1..n_q : the +1 self entries
    eq_rows_self = np.arange(1, n_q + 1, dtype=np.int64)
    eq_cols_self = K_self_col.astype(np.int64)
    eq_vals_self = np.ones(n_q, dtype=np.float64)

    # Rows 1..n_q : the -1 shifted entries
    e_mat = np.eye(d, dtype=np.int8)
    # shifted[r, j, :] = K_arr[r] + e_j   shape (n_q, d, d)
    shifted = K_arr[:, None, :] + e_mat[None, :, :]
    shifted_flat = shifted.reshape(n_q * d, d)
    shifted_sum = shifted_flat.sum(axis=1, dtype=np.int64)
    keep_inrange = shifted_sum <= R
    cols_minus_full = _batch_lookup(shifted_flat, void_lookup)
    keep = keep_inrange & (cols_minus_full >= 0)

    r_idx_full = np.repeat(np.arange(n_q, dtype=np.int64), d)
    eq_rows_minus = (1 + r_idx_full[keep])
    eq_cols_minus = cols_minus_full[keep].astype(np.int64)
    eq_vals_minus = -np.ones(int(keep.sum()), dtype=np.float64)

    eq_rows = np.concatenate([eq_rows0, eq_rows_self, eq_rows_minus])
    eq_cols = np.concatenate([eq_cols0, eq_cols_self, eq_cols_minus])
    eq_vals = np.concatenate([eq_vals0, eq_vals_self, eq_vals_minus])

    A_eq = sp.csr_matrix(
        (eq_vals, (eq_rows, eq_cols)),
        shape=(n_eq_rows, n_vars),
    )
    b_eq = np.zeros(n_eq_rows, dtype=np.float64)
    b_eq[0] = 1.0

    if verbose:
        print(f"  [build_fast] A_eq: {A_eq.shape} nnz={A_eq.nnz} "
              f"({(time.time()-t0)*1000:.0f}ms cum)",
              flush=True)

    # ----------------------------------------------------------------
    # 5. Inequality rows  (n_W rows)
    #    For each W:  sum_b coeff_W(b) y_b - tau <= 0
    #
    #    coeff_W(2 e_i)     = M_W[i, i]
    #    coeff_W(e_i + e_j) = 2 M_W[i, j]    (i < j)
    # ----------------------------------------------------------------
    if n_W > 0:
        # diag_idx[i] = beta_to_idx[2 e_i]
        diag_betas = (2 * np.eye(d, dtype=np.int8))
        diag_idx = _batch_lookup(diag_betas, void_lookup)

        # cross_idx[k] = beta_to_idx[e_i + e_j] for k = packed (i<j)
        if d >= 2:
            ii_grid, jj_grid = np.meshgrid(
                np.arange(d, dtype=np.int64),
                np.arange(d, dtype=np.int64),
                indexing='ij',
            )
            triu = ii_grid < jj_grid
            i_arr = ii_grid[triu]
            j_arr = jj_grid[triu]
            n_cross = i_arr.size
            cross_betas = np.zeros((n_cross, d), dtype=np.int8)
            cross_betas[np.arange(n_cross), i_arr] = 1
            cross_betas[np.arange(n_cross), j_arr] = 1
            cross_idx = _batch_lookup(cross_betas, void_lookup)
        else:
            i_arr = np.zeros(0, dtype=np.int64)
            j_arr = np.zeros(0, dtype=np.int64)
            cross_idx = np.zeros(0, dtype=np.int64)
            n_cross = 0

        # Stack windows
        M_stack = np.stack([np.asarray(M, dtype=np.float64) for M in M_mats], axis=0)  # (n_W, d, d)
        diag_coefs = np.diagonal(M_stack, axis1=1, axis2=2).copy()                     # (n_W, d)
        if n_cross > 0:
            cross_coefs = 2.0 * M_stack[:, i_arr, j_arr]                               # (n_W, n_cross)
        else:
            cross_coefs = np.zeros((n_W, 0), dtype=np.float64)

        w_arr = np.arange(n_W, dtype=np.int64)

        # diag triplets: row=w, col=diag_idx[i], val=M_W[i,i]
        d_rows = np.broadcast_to(w_arr[:, None], (n_W, d)).ravel()
        d_cols = np.broadcast_to(diag_idx[None, :], (n_W, d)).ravel()
        d_vals = diag_coefs.ravel()
        d_keep = (d_cols >= 0) & (d_vals != 0)

        # cross triplets: row=w, col=cross_idx[k], val=2*M_W[i,j]
        c_rows = np.broadcast_to(w_arr[:, None], (n_W, n_cross)).ravel()
        c_cols = np.broadcast_to(cross_idx[None, :], (n_W, n_cross)).ravel()
        c_vals = cross_coefs.ravel()
        c_keep = (c_cols >= 0) & (c_vals != 0)

        # tau triplets: row=w, col=tau_idx, val=-1
        t_rows = w_arr.copy()
        t_cols = np.full(n_W, tau_idx, dtype=np.int64)
        t_vals = -np.ones(n_W, dtype=np.float64)

        ub_rows = np.concatenate([d_rows[d_keep], c_rows[c_keep], t_rows])
        ub_cols = np.concatenate([d_cols[d_keep], c_cols[c_keep], t_cols])
        ub_vals = np.concatenate([d_vals[d_keep], c_vals[c_keep], t_vals])
    else:
        ub_rows = np.zeros(0, dtype=np.int64)
        ub_cols = np.zeros(0, dtype=np.int64)
        ub_vals = np.zeros(0, dtype=np.float64)

    A_ub = sp.csr_matrix(
        (ub_vals, (ub_rows, ub_cols)),
        shape=(n_W, n_vars),
    )
    b_ub = np.zeros(n_W, dtype=np.float64)

    if verbose:
        print(f"  [build_fast] A_ub: {A_ub.shape} nnz={A_ub.nnz} "
              f"({(time.time()-t0)*1000:.0f}ms cum)",
              flush=True)

    # ----------------------------------------------------------------
    # 6. Objective and bounds
    # ----------------------------------------------------------------
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[tau_idx] = 1.0

    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    bounds.extend([(0.0, y_upper)] * n_le_R)
    bounds.append((0.0, tau_upper))

    # ----------------------------------------------------------------
    # 7. Reconstruct monos_le_Rm1 (only needed if downstream code reads it)
    # ----------------------------------------------------------------
    monos_le_R_list = [tuple(int(x) for x in row) for row in monos_arr]
    monos_le_Rm1_list = [tuple(int(x) for x in row) for row in K_arr]
    beta_to_idx = {m: i for i, m in enumerate(monos_le_R_list)}

    return DualEpiBuildResult(
        A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
        c=c_obj, bounds=bounds,
        n_vars=n_vars, y_idx=slice(0, n_le_R), tau_idx=tau_idx,
        monos_le_R=monos_le_R_list, monos_le_Rm1=monos_le_Rm1_list,
        beta_to_idx=beta_to_idx,
        n_W=n_W, n_q_recursion_rows=n_q,
        d=d, R=R,
        build_wall_s=time.time() - t0,
    )


# =====================================================================
# Self-test: parity with the reference build at small sizes
# =====================================================================

def _self_check(d: int, R: int, M_mats):
    from lasserre.polya_lp.tier_dual.build_dual_epi import build_dual_epi_lp
    ref = build_dual_epi_lp(d, M_mats, R)
    fst = build_dual_epi_fast(d, M_mats, R)
    assert ref.n_vars == fst.n_vars, f"n_vars mismatch {ref.n_vars} vs {fst.n_vars}"
    assert ref.A_eq.shape == fst.A_eq.shape, "A_eq shape"
    assert ref.A_ub.shape == fst.A_ub.shape, "A_ub shape"
    diff_eq = (ref.A_eq - fst.A_eq).max() if ref.A_eq.nnz else 0
    diff_ub = (ref.A_ub - fst.A_ub).max() if ref.A_ub.nnz else 0
    assert abs(diff_eq) < 1e-12, f"A_eq differs: max={diff_eq}"
    assert abs(diff_ub) < 1e-12, f"A_ub differs: max={diff_ub}"
    assert np.array_equal(ref.b_eq, fst.b_eq), "b_eq differs"
    assert np.array_equal(ref.b_ub, fst.b_ub), "b_ub differs"
    assert np.array_equal(ref.c, fst.c), "c differs"
    return True
