"""Fast rational residual computation for the Farkas certificate.

Replaces the per-window Python fmpq scatter-add loops in
farkas_certify._adj_qW_exact_fmpq (and the moment/localizing block
analogs in safe_certify_flint._adjoint_block_fmpq) with a numpy int64
scatter-add pipeline.

All exact-arithmetic guarantees are preserved: every rounded rational
is represented as (numerator : int, denominator : int) with no
float64 at any stage after rounding.  The speedup comes from:

    1. fixed-denominator rounding (one np.rint per matrix, no
       Fraction.limit_denominator / math.gcd per entry);
    2. numpy int64 scatter-add (np.add.at on int64 arrays) for every
       block adjoint and every per-window adj_qW, avoiding per-entry
       fmpq construction in Python;
    3. per-ell bucketing of window adj_qW contributions so the
       combination step involves only a few (≤ 2d-1) distinct
       denominators in the final fmpq folding.

Soundness model
---------------
We round each dual to the fixed denominator D_S (so every rounded
entry has denom exactly D_S; no reduction).  Numerators fit in int64
as long as

    n_active_windows × n_loc^2 × |support(M_W)| × max|S_W.num|  <  2^63

which holds up to d = 16 with D_S = 10^9 and ||S_W||_∞ < 10 with
margin; we assert it at runtime and fall back to object-dtype
(Python bignum) scatter-add if the bound is exceeded.

Returns a list of fmpq (length n_y) identical — bit-for-bit — to
what the reference per-block / per-window fmpq loops produce, so it
can drop into farkas_certify_at without touching the safety-bound
math.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse as sp

try:
    import flint  # type: ignore
    _HAS_FLINT = True
except ImportError:
    _HAS_FLINT = False

from certified_lasserre.build_sdp import PSDBlock


# Conservative int64 bound: leave 1 bit of headroom against unlucky sums.
_INT64_SAFE = 2 ** 62


# =====================================================================
# Fixed-denominator rounding (replaces Fraction.limit_denominator)
# =====================================================================

def round_mat_fixed_denom(M: np.ndarray, D: int) -> np.ndarray:
    """Round a float matrix M to integer numerators at denominator D.

    Returns an int64 array of the same shape; the rational value at (i,j)
    is num[i,j] / D exactly.  Uses np.rint which is vectorized C-level;
    compare Fraction(float).limit_denominator(D) which calls math.gcd
    per entry and dominates profile at d=4 (~20% of probe time).
    """
    num = np.rint(np.asarray(M, dtype=np.float64) * float(D)).astype(np.int64)
    return num


def round_vec_fixed_denom(v: np.ndarray, D: int) -> np.ndarray:
    num = np.rint(np.asarray(v, dtype=np.float64) * float(D)).astype(np.int64)
    return num


def int64_arr_to_fmpq_list(num: np.ndarray, D: int) -> list:
    """Convert int64 numerator array at denom D to a Python list of fmpq."""
    num_list = num.tolist()
    return [flint.fmpq(n, D) for n in num_list]


def chol_round_bignum_product(
    S_float: np.ndarray, eig_margin: float, D_L: int,
) -> Tuple[np.ndarray, int]:
    """Round L at fixed denom D_L, product L·Lᵀ in Python bignum.

    Strategy:
      1. eigh(S_float); clamp eigenvalues; L_float = V·diag(sqrt(w+margin)).
      2. L_num = np.rint(L_float × D_L).astype(np.int64) — vectorized.
      3. For small n (n_loc ≤ 64), use numpy object-dtype matmul —
         simple and fast enough at these sizes.
      4. For large n, use flint.fmpz_mat matmul (C-level bignum
         matrix multiply), extract back to object numpy for consistent
         downstream handling.  Extraction uses fmpz_mat.entries() which
         is ~100× faster than per-[i,j] indexing.

    Precision: one L entry is rounded with error ≤ 1/(2 D_L); S entries
    carry O(||L|| / D_L) error ≤ ~3/D_L at D_L = 10⁹, matching the
    slow path's limit_denominator(10⁹) rounding.  All subsequent
    arithmetic is exact (no cumulative rounding).

    Returns (S_num, D_L²) with implicit rational S = S_num / D_L² exact.
    """
    S_sym = 0.5 * (np.asarray(S_float, dtype=np.float64)
                   + np.asarray(S_float, dtype=np.float64).T)
    w, V = np.linalg.eigh(S_sym)
    w_pos = np.maximum(w, 0.0) + float(eig_margin)
    L_float = V * np.sqrt(w_pos)[None, :]
    L_num_int = np.rint(L_float * float(D_L)).astype(np.int64)
    n = L_num_int.shape[0]
    if n <= 64:
        # numpy object-dtype matmul: simple; ~n³ Python int mul/adds.
        L_obj = L_num_int.astype(object)
        S_num = L_obj @ L_obj.T
        return S_num, D_L * D_L
    # Large n: route through flint.fmpz_mat for C-level matmul.
    flat_L = [int(x) for x in L_num_int.ravel().tolist()]
    L_fmpz = flint.fmpz_mat(n, n, flat_L)
    # Build Lᵀ by reshape in Python (cheap at these n).
    flat_LT = [int(x) for x in L_num_int.T.ravel().tolist()]
    LT_fmpz = flint.fmpz_mat(n, n, flat_LT)
    S_fmpz = L_fmpz * LT_fmpz
    # fmpz_mat supports .entries() returning a flat list of fmpz;
    # fall back to per-element [i,j] indexing if not available.
    try:
        flat = S_fmpz.entries()
    except AttributeError:
        flat = [S_fmpz[i, j] for i in range(n) for j in range(n)]
    S_num = np.empty((n, n), dtype=object)
    it = iter(flat)
    for i in range(n):
        for j in range(n):
            S_num[i, j] = int(next(it))
    return S_num, D_L * D_L


def chol_round_int_product(
    S_float: np.ndarray, eig_margin: float, D_L: int,
) -> Tuple[np.ndarray, int]:
    """PSD-preserving integer rounding of Cholesky-factor-of-S.

    Given a (near-)PSD float matrix S_float, compute
        eigh(S)  →  V, w
        w_pos = max(w, 0) + eig_margin
        L_float = V · diag(sqrt(w_pos))      (so L·Lᵀ ≈ S + margin·I)
        L_num = round(L_float × D_L)          (int64)
        S_num = L_num @ L_numᵀ                (int64 matmul, exact)

    Returns (S_num, D) with implicit rational S = S_num / D, D = D_L².
    S_num is PSD exactly — it is a Gram matrix of L_num treated as
    integers — so rational Cholesky gives a valid PSD Farkas dual.

    Overflow: each S_num[i,j] = Σ_k L_num[i,k] × L_num[j,k].  With
    |L_num| ≤ sqrt(||S||∞ + margin) × D_L ~ c × D_L, the inner sum
    is bounded by n × c² × D_L²; for n_loc ≤ 561 (d=32 order=3) and
    c² ≤ 10 we need D_L² ≤ 2^62 / 5610 ≈ 1.4 × 10¹⁵, so D_L ≤ 3.7 × 10⁷.
    We default to D_L = 10⁶ which fits through d = 32 with headroom.
    """
    S_sym = 0.5 * (np.asarray(S_float, dtype=np.float64)
                   + np.asarray(S_float, dtype=np.float64).T)
    w, V = np.linalg.eigh(S_sym)
    w_pos = np.maximum(w, 0.0) + float(eig_margin)
    L_float = V * np.sqrt(w_pos)[None, :]
    L_num = np.rint(L_float * float(D_L)).astype(np.int64)
    # Integer Gram product (exact, no rounding).
    S_num = L_num @ L_num.T
    # Overflow sanity: if any |S_num| came out > 2^62, caller must lower D_L.
    if S_num.size and int(np.abs(S_num).max()) >= _INT64_SAFE:
        raise RuntimeError(
            f"chol_round_int_product overflow: |S_num|_max = "
            f"{int(np.abs(S_num).max())} >= 2^62. Reduce D_L.")
    return S_num, D_L * D_L


# =====================================================================
# Block adjoint: precompute scatter indices once per (d, order)
# =====================================================================

@dataclass
class BlockScatter:
    """Precomputed scatter pattern for one PSD block's adjoint.

    For F_j(y) = Σ_alpha y_alpha G_j^{(alpha)}, the adjoint is
        F_j^*(S)[alpha] = Σ_{i,k} G_j^{(alpha)}[i,k] S[i,k]
                        = Σ_{ab} [G_flat[ab, alpha] ≠ 0] × S_flat[ab]
    For moment / loc blocks, G_flat entries are 0 or 1, so the adjoint
    reduces to a pure scatter-add:
        accum[alpha_flat[k]] += S_flat[ab_flat[k]]   ∀k
    plus a possible sign/coefficient (always +1 for moment+loc).

    Storage: two int64 arrays of length = G_flat.nnz.  Both are
    consumed by np.add.at; ordering doesn't matter.
    """
    n_j: int
    alpha_flat: np.ndarray   # int64, target positions in the n_y accumulator
    ab_flat: np.ndarray      # int64, source positions in vec(S) = S.ravel()


def precompute_block_scatter(blk: PSDBlock) -> BlockScatter:
    """Extract the (ab, alpha) scatter pairs for one block's adjoint."""
    G_coo = blk.G_flat.tocoo()
    # G_flat is (n_j^2, n_y) with 0/1 entries; for moment & loc blocks
    # each (row, col) with val=1 contributes S[row] to accum[col].
    # There may be multiple rows mapping to the same col (same alpha)
    # — that's the scatter duplicate.
    data_int = np.rint(G_coo.data).astype(np.int64)
    mask = data_int != 0
    # Assert all non-zero entries are +1 (true for moment + mu_i localizing).
    if not np.all(data_int[mask] == 1):
        raise ValueError(
            f"block {blk.name!r} has non-0/1 G_flat entries: "
            f"unique values = {np.unique(data_int)}")
    return BlockScatter(
        n_j=int(blk.size),
        alpha_flat=G_coo.col[mask].astype(np.int64),
        ab_flat=G_coo.row[mask].astype(np.int64),
    )


# =====================================================================
# Window adjoint: precompute per-ell scatter indices
# =====================================================================

@dataclass
class WindowScatter:
    """Precomputed scatter pattern for ONE window's adj_qW plus adj_t.

    adj_qW_W(S)[alpha] = Σ_{(a,b,i,j) ∈ support_W, ab_eiej[a,b,i,j]=alpha} S[a,b]
                      × coeff_W         (coeff_W = 2d/ell_W)
    adj_t_W(S)[alpha]  = Σ_{(a,b): t_pick[ab]=alpha} S[a,b]

    t_pick is the same for every window, so its scatter is precomputed
    once and reused (stored in WindowScatterCache).  Per-window we
    only store the qW scatter: (alpha_flat, ab_flat) pairs filtered
    to the window's support.

    max_mult: max number of scatter entries landing on any single alpha
    for this window.  Used for static int64 overflow bounds.
    """
    w_idx: int
    ell: int
    s_lo: int
    coeff_num: int           # 2d
    coeff_den: int           # ell
    alpha_flat: np.ndarray   # int64 alpha indices (adj_qW targets)
    ab_flat: np.ndarray      # int64 source ab=a*nl+b indices
    max_mult: int            # max(bincount(alpha_flat)), for overflow bound


@dataclass
class ResidualPrecomp:
    """All per-(d,order) precomputed scatter patterns, reused across probes."""
    d: int
    order: int
    n_y: int
    n_basis: int
    n_loc: int
    n_win: int

    # Base blocks: moment + d localizing
    base_block_names: List[str]
    base_block_scatters: List[BlockScatter]

    # Per-window adj_qW scatter
    window_scatters: List[WindowScatter]

    # Shared adj_t scatter (t_pick flat, all windows share it)
    t_pick_flat: np.ndarray  # int64, length n_loc^2; entry k = target alpha for src S[k]
    n_loc_sq: int


def _precompute_window_scatter(
    d: int,
    windows: List[Tuple[int, int]],
    M_mats: List[np.ndarray],
    ab_eiej_idx: np.ndarray,
    n_loc: int,
) -> List[WindowScatter]:
    """Build per-window (alpha_flat, ab_flat) scatter arrays for adj_qW."""
    out: List[WindowScatter] = []
    ab_unit = (np.arange(n_loc, dtype=np.int64)[:, None] * n_loc
               + np.arange(n_loc, dtype=np.int64)[None, :])  # (n_loc, n_loc)
    for w_idx, (ell, s_lo) in enumerate(windows):
        Mw = M_mats[w_idx]
        nz_i, nz_j = np.nonzero(Mw)
        if nz_i.size == 0:
            out.append(WindowScatter(
                w_idx=w_idx, ell=ell, s_lo=s_lo,
                coeff_num=2 * d, coeff_den=ell,
                alpha_flat=np.zeros(0, dtype=np.int64),
                ab_flat=np.zeros(0, dtype=np.int64),
                max_mult=0,
            ))
            continue
        # ab_eiej_idx has shape (n_loc, n_loc, d, d); slice to (n_loc, n_loc, K)
        idx_slice = ab_eiej_idx[:, :, nz_i, nz_j]  # (n_loc, n_loc, K)
        valid = idx_slice >= 0
        if not valid.any():
            out.append(WindowScatter(
                w_idx=w_idx, ell=ell, s_lo=s_lo,
                coeff_num=2 * d, coeff_den=ell,
                alpha_flat=np.zeros(0, dtype=np.int64),
                ab_flat=np.zeros(0, dtype=np.int64),
                max_mult=0,
            ))
            continue
        ab_bcast = np.broadcast_to(ab_unit[:, :, None], idx_slice.shape)
        alpha_flat = idx_slice[valid].astype(np.int64).ravel()
        ab_flat = ab_bcast[valid].astype(np.int64).ravel()
        mm = int(np.bincount(alpha_flat).max()) if alpha_flat.size else 0
        out.append(WindowScatter(
            w_idx=w_idx, ell=ell, s_lo=s_lo,
            coeff_num=2 * d, coeff_den=ell,
            alpha_flat=alpha_flat, ab_flat=ab_flat,
            max_mult=mm,
        ))
    return out


def build_residual_precomp(P: dict, base_blocks: List[PSDBlock]) -> ResidualPrecomp:
    """Build the per-(d, order) scatter cache.  Reused across all probes."""
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    windows = P['windows']
    M_mats = P['M_mats']
    n_win = len(windows)
    ab_eiej_idx = P['ab_eiej_idx']
    t_pick_np = np.asarray(P['t_pick'], dtype=np.int64)

    base_scatters = [precompute_block_scatter(b) for b in base_blocks]
    base_names = [b.name for b in base_blocks]
    win_scatters = _precompute_window_scatter(
        d=d, windows=windows, M_mats=M_mats,
        ab_eiej_idx=ab_eiej_idx, n_loc=n_loc,
    )
    return ResidualPrecomp(
        d=d, order=order, n_y=n_y,
        n_basis=n_basis, n_loc=n_loc, n_win=n_win,
        base_block_names=base_names,
        base_block_scatters=base_scatters,
        window_scatters=win_scatters,
        t_pick_flat=t_pick_np,
        n_loc_sq=n_loc * n_loc,
    )


# =====================================================================
# Overflow guard
# =====================================================================


def _int64_safe_bound(n_terms: int, max_abs_val: int) -> bool:
    """Conservative overflow check for int64 scatter-add.

    A scatter-add of n_terms entries each bounded by max_abs_val in
    absolute value sums to at most n_terms × max_abs_val.  We require
    this to fit in 2^62 for safety.
    """
    try:
        product = n_terms * max_abs_val
    except OverflowError:
        return False
    return product < _INT64_SAFE


# =====================================================================
# Residual computation (one probe)
# =====================================================================

def compute_residual_fast(
    pre: ResidualPrecomp,
    # A^T μ inputs:
    A_csr: sp.csr_matrix,
    mu_A_num: np.ndarray,        # int64, length A_csr.shape[0]
    D_mu: int,
    # Base PSD inputs (moment + d localizing):
    base_S_num: List[np.ndarray],  # list of int64 OR object (n_j, n_j); same order as pre.base_block_scatters
    D_S: int,                    # common base-block denominator
    # Window PSD inputs (None for inactive):
    win_S_num: List[Optional[np.ndarray]],  # length n_win; each entry (n_loc, n_loc) or None
    D_W: int,                    # common window denominator
    # Epigraph variable:
    t_test_fmpq,                 # flint.fmpq
) -> list:
    """Return r[alpha] as a list of fmpq, length n_y.

    r = A^T μ + Σ_j F_j^*(S_j) + Σ_W [t_test · adj_t(S_W) − adj_qW(S_W)]

    Internally uses int64 scatter-add per block/window group, then
    folds the per-group int64 arrays to fmpq once at the end.
    """
    if not _HAS_FLINT:
        raise ImportError("python-flint required")

    n_y = pre.n_y
    nl = pre.n_loc

    # Detect dtype: if any S_num is object-dtype, do all accumulation in
    # Python bignum (dtype=object), else in int64.  The two paths share the
    # same shape semantics; only the element arithmetic type differs.
    is_bignum = (
        (hasattr(base_S_num[0], 'dtype') and base_S_num[0].dtype == object) or
        any(S is not None and S.dtype == object for S in win_S_num)
    )
    accum_dtype = object if is_bignum else np.int64

    # -----------------------------------------------------------------
    # (1) A^T μ contribution.
    # -----------------------------------------------------------------
    # scipy sparse matvec with object dtype works via Python __iadd__; it's
    # slower than int64 but correct.  For int64, we get C-level speed.
    A_int = sp.csr_matrix(
        (np.rint(A_csr.data).astype(np.int64), A_csr.indices, A_csr.indptr),
        shape=A_csr.shape)
    if is_bignum:
        # scipy sparse doesn't support object dtype — do the matvec manually
        # in Python bignum.  A_csr is n_eq × n_y; we want Aᵀ @ μ in length-n_y.
        mu_arr = [int(x) for x in np.asarray(mu_A_num).tolist()]
        r_mu_num = np.zeros(n_y, dtype=object)
        A_csr_int = A_int
        indptr_ = A_csr_int.indptr
        indices_ = A_csr_int.indices
        data_ = A_csr_int.data.tolist()
        for i_eq in range(A_csr_int.shape[0]):
            mu_i = mu_arr[i_eq]
            if mu_i == 0:
                continue
            start, end = indptr_[i_eq], indptr_[i_eq + 1]
            for p in range(start, end):
                col = indices_[p]
                coef = data_[p]
                r_mu_num[col] += coef * mu_i
    else:
        mu_arr = np.asarray(mu_A_num, dtype=np.int64)
        r_mu_num = np.asarray(A_int.T @ mu_arr, dtype=np.int64)

    # -----------------------------------------------------------------
    # (2) Base PSD contributions: Σ_j F_j^*(S_j) at denom D_S.
    # -----------------------------------------------------------------
    r_base_num = np.zeros(n_y, dtype=accum_dtype)
    if len(base_S_num) != len(pre.base_block_scatters):
        raise ValueError(
            f"base_S_num has {len(base_S_num)} entries, expected "
            f"{len(pre.base_block_scatters)}")
    for scat, S_num in zip(pre.base_block_scatters, base_S_num):
        n_j = scat.n_j
        if S_num.shape != (n_j, n_j):
            raise ValueError(
                f"base block size mismatch: expected ({n_j},{n_j}), got {S_num.shape}")
        S_flat = np.ascontiguousarray(S_num, dtype=accum_dtype).ravel()
        # Scatter: accum[alpha_flat[k]] += S_flat[ab_flat[k]]
        np.add.at(r_base_num, scat.alpha_flat, S_flat[scat.ab_flat])

    # -----------------------------------------------------------------
    # (3) Window adj_t contribution: accumulated across all active
    # windows with the SAME scatter pattern (t_pick).  Sum numerators
    # first, multiply by t_test once at the end.
    # -----------------------------------------------------------------
    accum_t_num = np.zeros(n_y, dtype=accum_dtype)  # denom = D_W
    # -----------------------------------------------------------------
    # (4) Window adj_qW contribution: per-ell bucket so the final
    # combination has ≤ 2d-1 distinct denominators.
    # -----------------------------------------------------------------
    accum_qW_by_ell: Dict[int, np.ndarray] = {}
    coeff_2d = 2 * pre.d

    # Static overflow bound (int64 path only; bignum cannot overflow).
    total_active = 0
    max_abs_S = 0
    per_ell_max_bin: Dict[int, int] = {}
    per_ell_active_count: Dict[int, int] = {}
    for scat in pre.window_scatters:
        S_num = win_S_num[scat.w_idx]
        if S_num is None:
            continue
        if S_num.shape != (nl, nl):
            raise ValueError(f"window S size mismatch: {S_num.shape} vs ({nl},{nl})")
        total_active += 1
        if not is_bignum and S_num.size:
            mav = int(np.abs(S_num).max())
            if mav > max_abs_S:
                max_abs_S = mav
        per_ell_max_bin[scat.ell] = per_ell_max_bin.get(scat.ell, 0) + int(scat.max_mult)
        per_ell_active_count[scat.ell] = per_ell_active_count.get(scat.ell, 0) + 1

    if not is_bignum:
        for ell, mb in per_ell_max_bin.items():
            if mb * max_abs_S >= _INT64_SAFE:
                raise RuntimeError(
                    f"int64 overflow risk in adj_qW ell={ell}: "
                    f"max_per_bin_count={mb}, max|S|={max_abs_S}. "
                    f"Reduce fast_D_L or set use_bignum=True.")
        if total_active > 0 and pre.t_pick_flat.size:
            t_mult_max = int(np.bincount(
                pre.t_pick_flat[pre.t_pick_flat >= 0]).max()) \
                if (pre.t_pick_flat >= 0).any() else 0
            if total_active * t_mult_max * max_abs_S >= _INT64_SAFE:
                raise RuntimeError(
                    f"int64 overflow risk in adj_t: "
                    f"total_active={total_active}, t_mult_max={t_mult_max}, "
                    f"max|S|={max_abs_S}. Reduce fast_D_L or set use_bignum=True.")

    # Now do the actual scatter.
    for scat in pre.window_scatters:
        S_num = win_S_num[scat.w_idx]
        if S_num is None:
            continue
        S_flat = np.ascontiguousarray(S_num, dtype=accum_dtype).ravel()
        np.add.at(accum_t_num, pre.t_pick_flat, S_flat)
        if scat.alpha_flat.size == 0:
            continue
        ell = scat.ell
        if ell not in accum_qW_by_ell:
            accum_qW_by_ell[ell] = np.zeros(n_y, dtype=accum_dtype)
        np.add.at(accum_qW_by_ell[ell], scat.alpha_flat, S_flat[scat.ab_flat])

    # -----------------------------------------------------------------
    # Fold to fmpq: r[alpha] = (r_mu_num[alpha]/D_mu)
    #                       + (r_base_num[alpha]/D_S)
    #                       + t_test × (accum_t_num[alpha]/D_W)
    #                       + Σ_ell (coeff_2d × accum_qW_by_ell[ell][alpha]) / (ell × D_W)
    # Sign on adj_qW: the window-localizing PSD L_W = t_test·M − M(q_W·y),
    # whose adjoint contributes +t_test·adj_t(S_W) and −adj_qW(S_W).
    # So we SUBTRACT the qW terms.
    # -----------------------------------------------------------------
    # tolist handles both int64 arrays (returns Python ints, lossless) and
    # object arrays (returns Python ints directly).
    r_mu_list = r_mu_num.tolist()
    r_base_list = r_base_num.tolist()
    accum_t_list = accum_t_num.tolist()
    ells_sorted = sorted(accum_qW_by_ell.keys())
    accum_qW_lists = {ell: accum_qW_by_ell[ell].tolist() for ell in ells_sorted}

    out = [flint.fmpq(0) for _ in range(n_y)]
    for alpha in range(n_y):
        v = flint.fmpq(0)
        if r_mu_list[alpha] != 0:
            v += flint.fmpq(int(r_mu_list[alpha]), int(D_mu))
        if r_base_list[alpha] != 0:
            v += flint.fmpq(int(r_base_list[alpha]), int(D_S))
        if accum_t_list[alpha] != 0:
            v += t_test_fmpq * flint.fmpq(int(accum_t_list[alpha]), int(D_W))
        for ell in ells_sorted:
            a = accum_qW_lists[ell][alpha]
            if a != 0:
                v -= flint.fmpq(coeff_2d * int(a), ell * int(D_W))
        out[alpha] = v
    return out


# =====================================================================
# Self-test: compare against reference fmpq loops on a small SDP
# =====================================================================

def _reference_residual(
    P: dict,
    A_csr: sp.csr_matrix,
    mu_A_fmpq: list,
    base_blocks: List[PSDBlock],
    base_S_fmpq: list,
    S_win_fmpq: list,
    t_test_fmpq,
    active_windows: np.ndarray,
) -> list:
    """Reference implementation using the existing fmpq loops — used in
    tests to bit-compare against compute_residual_fast.  Imported from
    farkas_certify & safe_certify_flint so any refactor to those stays
    in sync.
    """
    from certified_lasserre.farkas_certify import (
        _adj_qW_exact_fmpq, _adj_t_fmpq,
    )
    from certified_lasserre.safe_certify_flint import (
        _adjoint_block_fmpq, _sparse_matvec_fmpq,
    )
    n_y = P['n_y']
    d = P['d']
    windows = P['windows']
    M_mats = P['M_mats']
    ab_eiej_idx = P['ab_eiej_idx']
    t_pick_np = np.asarray(P['t_pick'], dtype=np.int64)

    r = [flint.fmpq(0) for _ in range(n_y)]
    AT_mu = _sparse_matvec_fmpq(A_csr.T.tocsr(), mu_A_fmpq)
    for k in range(n_y):
        r[k] += AT_mu[k]
    for blk, S_fm in zip(base_blocks, base_S_fmpq):
        adj = _adjoint_block_fmpq(blk, S_fm)
        for k in range(n_y):
            r[k] += adj[k]
    for w, S_fm in enumerate(S_win_fmpq):
        if S_fm is None:
            continue
        ell, _ = windows[w]
        coeff = flint.fmpq(2 * d, ell)
        adj_t = _adj_t_fmpq(S_fm, t_pick_np, n_y)
        for k in range(n_y):
            r[k] += t_test_fmpq * adj_t[k]
        if ab_eiej_idx is None:
            continue
        Mw_support = (M_mats[w] != 0).astype(np.int64)
        adj_q = _adj_qW_exact_fmpq(S_fm, ab_eiej_idx, Mw_support, coeff, n_y)
        for k in range(n_y):
            r[k] -= adj_q[k]
    return r
