"""Pre-elim-reduced dual Lasserre SDP (Task API, Farkas form).

================================================================================
WHAT THIS FILE IS
================================================================================

This is the pre-elimination analogue of ``lasserre/dual_sdp.py``.  Given the
moment-primal substitution y = T·ỹ + c (from ``lasserre.preelim``), the dual
Farkas LP is reduced by **row-contracting the stationarity matrix**:

    stationarity_α̃_reduced = Σ_{α} T[α, α̃] · stationarity_α_original

This shrinks n_cons from n_y to n_y_red (≈70% at d=14 L=3), and reduces the
scalar μ-multiplier count from n_consist_kept to n_resid (one per residual
equality that survived Gauss–Jordan).

Mathematical validity
---------------------
The reduced dual is the Farkas LP of the reduced primal.  With
``forbidden_cols = {alpha_zero_row}`` passed to ``build_preelim_transform``,
the simplex row ``y_0 = 1`` is guaranteed to be a residual equality (not
pivoted).  Its dual multiplier is λ, bounded in [0, Λ].  All other residual
equalities have free-signed μ̃ multipliers.

Verdict semantics are IDENTICAL to the monolithic dual:
  λ* ≈ 1  ⟹  primal infeasible  ⟹  val_L(d) > t.
  λ* ≈ 0  ⟹  primal feasible    ⟹  val_L(d) ≤ t.

================================================================================
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

import mosek

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'tests'))

from lasserre_fusion import _hash_monos  # noqa: E402

from lasserre.dual_sdp import (  # noqa: E402
    _alpha_lookup, _aggregate_bar_triplet, _aggregate_scalar_triplet,
)
from lasserre.preelim import (  # noqa: E402
    PreElimTransform, build_preelim_transform,
)


__all__ = [
    'build_dual_task_preelim',
    'update_task_t_preelim',
]


# ---------------------------------------------------------------------------
# Row-contraction helper: project bar-triplets through T.
# ---------------------------------------------------------------------------

def _contract_bar_triplets_via_T(
    subi: np.ndarray, subj: np.ndarray,
    subk: np.ndarray, subl: np.ndarray,
    val:  np.ndarray, tcoef: np.ndarray,
    T_csr: sp.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Row-contract bar triplets via the pre-elim transform T.

    Given an original bar triplet (subi=α, subj, subk, subl, val, tcoef)
    and T ∈ R^{n_y × n_y_red}, expand each entry into
        (subi=α̃, subj, subk, subl, val * T[α,α̃], tcoef * T[α,α̃])
    for every α̃ ∈ free_cols with T[α, α̃] ≠ 0.

    Fast paths:
      * Most α's are free columns — T[α, :] has exactly one nonzero at
        column new_idx[α] with value 1.  These are handled by direct
        gather (vectorised).
      * Pivot α's have |T[α, :].nnz| potentially > 1; handled via CSR
        row slicing and explicit per-entry expansion.

    Returns six contiguous int32/float64 arrays (subi_out, ...) with the
    same semantics as the input, but with subi now in [0..n_y_red).
    """
    if subi.size == 0:
        e32 = np.array([], dtype=np.int32)
        ef = np.array([], dtype=np.float64)
        return e32, e32, e32, e32, ef, ef

    n_y = int(T_csr.shape[0])
    n_y_red = int(T_csr.shape[1])

    indptr = T_csr.indptr
    indices = T_csr.indices
    data = T_csr.data

    # For each triplet t, count = indptr[subi[t]+1] - indptr[subi[t]].
    subi_i64 = subi.astype(np.int64, copy=False)
    counts = (indptr[subi_i64 + 1] - indptr[subi_i64]).astype(np.int64)
    n_expanded = int(counts.sum())

    if n_expanded == 0:
        # All rows in subi mapped to rows of T with no nonzeros -> all
        # originally-emitted stationarity contributions got absorbed into
        # the constant shift.  The reduced dual has no bar-triplet at all.
        e32 = np.array([], dtype=np.int32)
        ef = np.array([], dtype=np.float64)
        return e32, e32, e32, e32, ef, ef

    # Output arrays of size n_expanded.
    out_subi = np.empty(n_expanded, dtype=np.int32)
    out_subj = np.empty(n_expanded, dtype=np.int32)
    out_subk = np.empty(n_expanded, dtype=np.int32)
    out_subl = np.empty(n_expanded, dtype=np.int32)
    out_val = np.empty(n_expanded, dtype=np.float64)
    out_tcoef = np.empty(n_expanded, dtype=np.float64)

    # offsets[t] is the cumulative starting position for triplet t.
    offsets = np.empty(subi.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    # Build a mapping triplet_index -> T-entry-index via "repeat by counts".
    # triplet_for_entry[k] ∈ [0, num_triplets) gives which input triplet
    # the k-th expanded entry came from.
    triplet_for_entry = np.repeat(np.arange(subi.size, dtype=np.int64), counts)

    # For each expanded entry k, offset within its source triplet.
    within = np.arange(n_expanded, dtype=np.int64) - offsets[triplet_for_entry]

    # Source T row for entry k:
    T_row_start = indptr[subi_i64[triplet_for_entry]]
    T_entry_index = T_row_start + within

    # subi <- T.indices[T_entry_index] (the α̃ column of T)
    out_subi[:] = indices[T_entry_index].astype(np.int32, copy=False)
    # val <- original val * T.data
    weights = data[T_entry_index]
    out_val[:] = val[triplet_for_entry] * weights
    out_tcoef[:] = tcoef[triplet_for_entry] * weights
    # The (subj, subk, subl) are replicated from the source triplet.
    out_subj[:] = subj[triplet_for_entry]
    out_subk[:] = subk[triplet_for_entry]
    out_subl[:] = subl[triplet_for_entry]

    return out_subi, out_subj, out_subk, out_subl, out_val, out_tcoef


# =========================================================================
# Top-level builder
# =========================================================================

def build_dual_task_preelim(
    P: Dict[str, Any], *,
    t_val: float,
    env: Optional[mosek.Env] = None,
    xf: Optional[PreElimTransform] = None,
    include_upper_loc: bool = False,
    z2_blockdiag_map: Optional[Dict[str, Any]] = None,
    active_loc: Optional[List[int]] = None,
    active_windows: Optional[List[int]] = None,
    lambda_upper_bound: float = 1.0,
    preelim_max_fill_ratio: float = 10.0,
    preelim_protect_degrees: Optional[set] = None,
    cache_for_reuse: bool = True,
    verbose: bool = True,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Build the Farkas-infeasibility dual SDP, row-contracted via pre-elim.

    The reduced dual has:
      * n_y_red stationarity rows (vs n_y monolithic),
      * n_resid + n_y_red scalar variables (λ absorbed into μ̃[0]; rest
        are μ̃_{1..n_resid−1} + v̂_α̃),
      * Same bar-variable structure as build_dual_task (moment + loc +
        optional uloc + windows).

    ``xf`` is the pre-elim transform.  If None, it's built from P with
    ``forbidden_cols = {alpha_zero_row}`` to guarantee the simplex row
    remains as a residual equality so λ retains its [0, Λ] bound.

    Protecting degree-1 and degree-2 monomials (the default) is critical
    for conditioning; see preelim.py:465.

    Other parameters match ``build_dual_task`` exactly.
    """
    d = int(P['d'])
    n_y = int(P['n_y'])
    basis = P['basis']
    n_basis = int(P['n_basis'])
    loc_basis = P['loc_basis']
    n_loc = int(P['n_loc'])
    mono_idx = P['idx']
    M_mats = P['M_mats']
    bases_arr = np.asarray(P['bases'], dtype=np.int64)
    sorted_h = np.asarray(P['sorted_h'])
    sort_o = np.asarray(P['sort_o'])
    old_to_new_arr = P.get('old_to_new')
    if old_to_new_arr is not None:
        old_to_new_arr = np.asarray(old_to_new_arr, dtype=np.int64)

    z2_canon = old_to_new_arr is not None

    if z2_blockdiag_map is not None and not z2_canon:
        raise ValueError(
            "z2_blockdiag_map requires the precompute to be canonicalized "
            "via lasserre.z2_elim.canonicalize_z2.")

    alpha_zero = tuple(0 for _ in range(d))
    if alpha_zero not in mono_idx:
        raise RuntimeError(
            "Zero monomial (0,...,0) missing from P['idx'] — required by "
            "Lasserre normalisation y_0 = 1.")
    alpha_zero_row = int(mono_idx[alpha_zero])

    # Build (or take) the pre-elim transform.  Forbidding alpha_zero_row
    # ensures the simplex y_0 = 1 stays as a residual equality.  Protect
    # deg 1 and 2 for conditioning.
    if xf is None:
        forbidden = {alpha_zero_row}
        protect_deg = preelim_protect_degrees if preelim_protect_degrees is not None else {1, 2}
        xf = build_preelim_transform(
            P,
            max_fill_ratio=preelim_max_fill_ratio,
            forbidden_cols=forbidden,
            protect_degrees=protect_deg,
            verbose=verbose,
        )

    T_csr = xf.T  # already csr
    offset = xf.offset
    residual_A = xf.residual_A.tocsr()
    residual_b = xf.residual_b
    n_y_red = xf.n_y_red
    n_resid = int(residual_A.shape[0])

    # Identify which residual row is the simplex equation (y_0 = 1).  By
    # convention, assemble_consistency_equalities emits it as row 0 IFF
    # include_simplex=True (the default, which preelim uses).  BUT
    # Gauss-Jordan may reorder residuals.  We identify by pattern: the
    # simplex row has exactly one nonzero (at col alpha_zero_row = 1)
    # and residual_b = 1.
    simplex_row_idx: Optional[int] = None
    for k in range(n_resid):
        row = residual_A.getrow(k)
        if row.nnz == 1 and int(row.indices[0]) == alpha_zero_row and abs(float(row.data[0]) - 1.0) < 1e-12:
            if abs(float(residual_b[k]) - 1.0) < 1e-12:
                simplex_row_idx = k
                break

    if simplex_row_idx is None:
        raise RuntimeError(
            "preelim did not preserve the simplex row y_0 = 1 as a "
            "residual equality.  Check that alpha_zero_row="
            f"{alpha_zero_row} is in forbidden_cols.")

    # Assert alpha_zero is a free column (necessary consequence of
    # forbidding it as a pivot).
    if alpha_zero_row not in xf.new_idx:
        raise RuntimeError(
            f"alpha_zero_row={alpha_zero_row} is not a free column after "
            "preelim — this violates the forbidden_cols guarantee.")

    if n_loc == 0:
        active_loc = []
        active_windows = []
    else:
        if active_loc is None:
            active_loc = list(range(d))
        if active_windows is None:
            active_windows = list(P['nontrivial_windows'])

    if env is None:
        env = mosek.Env()
    task = env.Task()

    if verbose:
        task.set_Stream(
            mosek.streamtype.log, lambda s: print(s, end=''))

    t0 = time.time()

    # -----------------------------------------------------------------
    # 1. Bar-variable layout — IDENTICAL to build_dual_task.
    # -----------------------------------------------------------------
    bar_sizes: List[int] = []
    moment_bar_ids: List[int] = []

    if z2_blockdiag_map is None:
        moment_bar_ids.append(0)
        bar_sizes.append(n_basis)
    else:
        n_sym = int(z2_blockdiag_map['n_sym'])
        n_anti = int(z2_blockdiag_map['n_anti'])
        moment_bar_ids.append(0)
        bar_sizes.append(n_sym)
        if n_anti > 0:
            moment_bar_ids.append(1)
            bar_sizes.append(n_anti)

    loc_bar_start = len(bar_sizes)
    for _ in active_loc:
        bar_sizes.append(n_loc)
    loc_bar_end = len(bar_sizes)

    uloc_bar_start = loc_bar_end
    if include_upper_loc:
        for _ in active_loc:
            bar_sizes.append(n_loc)
    uloc_bar_end = len(bar_sizes)

    win_bar_start = uloc_bar_end
    for _ in active_windows:
        bar_sizes.append(n_loc)
    win_bar_end = len(bar_sizes)

    n_bar = len(bar_sizes)
    task.appendbarvars(bar_sizes)

    # -----------------------------------------------------------------
    # 2. Scalar variables: [λ = μ̃[simplex] | μ̃_other | v̂_α̃].
    #
    # We reorder residuals so that simplex_row_idx → position 0 in μ̃.
    # -----------------------------------------------------------------
    # Reorder residual_A rows: simplex first.
    row_order = np.arange(n_resid, dtype=np.int64)
    if simplex_row_idx != 0:
        row_order[0], row_order[simplex_row_idx] = (
            int(simplex_row_idx), 0)
    if n_resid > 0:
        residual_A_reord = residual_A[row_order].tocsr()
        residual_b_reord = residual_b[row_order].copy()
    else:
        residual_A_reord = residual_A
        residual_b_reord = residual_b

    LAMBDA_IDX = 0
    MU_START = 1
    MU_END = n_resid                    # μ̃_{1..n_resid-1} (others)
    V_START = n_resid
    V_END = n_resid + n_y_red
    n_scalar = V_END

    task.appendvars(n_scalar)

    # λ ∈ [0, Λ].  Others free.
    task.putvarbound(
        LAMBDA_IDX, mosek.boundkey.ra, 0.0, float(lambda_upper_bound))
    if MU_END > MU_START:
        n_free = MU_END - MU_START
        task.putvarboundslice(
            MU_START, MU_END,
            [mosek.boundkey.fr] * n_free,
            np.full(n_free, -np.inf, dtype=np.float64),
            np.full(n_free, +np.inf, dtype=np.float64),
        )
    task.putvarboundslice(
        V_START, V_END,
        [mosek.boundkey.lo] * n_y_red,
        np.zeros(n_y_red, dtype=np.float64),
        np.full(n_y_red, +np.inf, dtype=np.float64),
    )

    # -----------------------------------------------------------------
    # 3. Constraint rows: one per α̃ ∈ [0..n_y_red), all = 0.
    # -----------------------------------------------------------------
    task.appendcons(n_y_red)
    task.putconboundslice(
        0, n_y_red,
        [mosek.boundkey.fx] * n_y_red,
        np.zeros(n_y_red, dtype=np.float64),
        np.zeros(n_y_red, dtype=np.float64),
    )

    # -----------------------------------------------------------------
    # 4. Scalar coefficients (A matrix) — already reduced.
    #
    #    (a) μ̃_k couplings: (residual_A @ T)[k, α̃] at (α̃, MU_start+k-1).
    #        Or for k=0 (λ): coupling at (α̃, LAMBDA_IDX).
    #    (b) v̂_α̃ diagonal: +1 at (α̃, V_START+α̃).
    #
    # We skip the bar-triplet reduction for now — that's step 5.
    # -----------------------------------------------------------------
    if n_resid > 0:
        A_T = (residual_A_reord @ T_csr).tocoo()
        scalar_rows = A_T.col.astype(np.int64, copy=False)     # α̃
        scalar_cols = A_T.row.astype(np.int64, copy=False)     # μ̃ index
        scalar_vals = A_T.data.astype(np.float64, copy=False)
    else:
        scalar_rows = np.zeros(0, dtype=np.int64)
        scalar_cols = np.zeros(0, dtype=np.int64)
        scalar_vals = np.zeros(0, dtype=np.float64)

    # v̂_α̃: diagonal +1.
    v_rows = np.arange(n_y_red, dtype=np.int64)
    v_cols = (V_START + v_rows).astype(np.int64)
    v_vals = np.ones(n_y_red, dtype=np.float64)

    all_scalar_rows = np.concatenate([scalar_rows, v_rows])
    all_scalar_cols = np.concatenate([scalar_cols, v_cols])
    all_scalar_vals = np.concatenate([scalar_vals, v_vals])

    r_arr, c_arr, v_arr = _aggregate_scalar_triplet(
        all_scalar_rows, all_scalar_cols, all_scalar_vals,
    )
    if r_arr.size:
        task.putaijlist(r_arr, c_arr, v_arr)

    # -----------------------------------------------------------------
    # 5. Bar-matrix sensitivity coefficients, row-contracted via T,
    #    STREAMED per-cone.
    #
    # Per-cone streaming eliminates the transient peak from concatenating
    # all bar triplets across 200+ cones before the T-contraction.  At
    # d=14 preelim (~10M entries, 32 B/entry) the concat held ~3× the
    # aggregated size as working memory; at d=20 this transient was
    # multi-GB.  Streaming aggregates within each cone, T-contracts,
    # re-aggregates, submits to MOSEK, then drops the cone buffers.
    # -----------------------------------------------------------------
    # Per-cone cache for update_task_t_preelim when cache_for_reuse.
    cache_arrays: List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray, np.ndarray]] = []
    cone_entry_counts: List[int] = []

    def _submit_cone(subi, subj_val: int, subk, subl, vals, tcoefs=None):
        """Aggregate one cone's triplet, contract via T, submit to MOSEK.

        subj_val : the single bar-index for this cone (moment/loc/uloc/win).
        """
        if subi.size == 0:
            return
        subi_i32 = np.ascontiguousarray(subi, dtype=np.int32)
        subj_i32 = np.full(subi.size, subj_val, dtype=np.int32)
        subk_i32 = np.ascontiguousarray(subk, dtype=np.int32)
        subl_i32 = np.ascontiguousarray(subl, dtype=np.int32)
        val_f64 = np.ascontiguousarray(vals, dtype=np.float64)
        if tcoefs is None:
            tcoef_f64 = np.zeros(subi.size, dtype=np.float64)
        else:
            tcoef_f64 = np.ascontiguousarray(tcoefs, dtype=np.float64)

        # (1) Aggregate within cone — dedupe duplicates from z2 collapse.
        s_i, s_j, s_k, s_l, s_v, s_tc = _aggregate_bar_triplet(
            subi_i32, subj_i32, subk_i32, subl_i32, val_f64, tcoef_f64,
        )
        # (2) Row-contract via T: subi=α → T[α,:].nonzero() with weights.
        c_i, c_j, c_k, c_l, c_v, c_tc = _contract_bar_triplets_via_T(
            s_i, s_j, s_k, s_l, s_v, s_tc, T_csr,
        )
        # (3) Re-aggregate after T-expansion.
        r_i, r_j, r_k, r_l, r_v, r_tc = _aggregate_bar_triplet(
            c_i, c_j, c_k, c_l, c_v, c_tc,
        )
        if r_i.size == 0:
            return
        # (4) Submit to MOSEK (cumulative over cones — distinct subj).
        init_vals = r_v + float(t_val) * r_tc
        task.putbarablocktriplet(r_i, r_j, r_k, r_l, init_vals)
        cone_entry_counts.append(int(r_i.size))
        # (5) Cache for update_task_t if requested.
        if cache_for_reuse:
            cache_arrays.append((r_i, r_j, r_k, r_l, r_v, r_tc))

    # ---- Moment cone ----
    if z2_blockdiag_map is None:
        B_arr = np.asarray(basis, dtype=np.int64)
        B_hash = _hash_monos(B_arr, bases_arr)
        ks_m, ls_m = np.tril_indices(n_basis)
        alpha_hash_m = B_hash[ks_m] + B_hash[ls_m]
        alpha_idx_m = _alpha_lookup(
            alpha_hash_m, sorted_h, sort_o, old_to_new_arr)
        if np.any(alpha_idx_m < 0):
            raise RuntimeError(
                "Moment sensitivity lookup produced -1; precompute P is "
                "inconsistent.")
        _submit_cone(
            alpha_idx_m, moment_bar_ids[0],
            ks_m, ls_m,
            np.full(ks_m.shape, +1.0, dtype=np.float64),
        )
    else:
        T_sym = z2_blockdiag_map['T_sym'].tocoo()
        n_sym_here = int(z2_blockdiag_map['n_sym'])
        if T_sym.nnz:
            u = T_sym.row // n_sym_here
            v = T_sym.row % n_sym_here
            alpha_col = T_sym.col
            val = T_sym.data
            mask = u >= v
            _submit_cone(
                alpha_col[mask].astype(np.int32), moment_bar_ids[0],
                u[mask].astype(np.int32), v[mask].astype(np.int32),
                val[mask].astype(np.float64),
            )

        if len(moment_bar_ids) > 1:
            T_anti = z2_blockdiag_map['T_anti'].tocoo()
            n_anti_here = int(z2_blockdiag_map['n_anti'])
            if T_anti.nnz:
                u = T_anti.row // n_anti_here
                v = T_anti.row % n_anti_here
                alpha_col = T_anti.col
                val = T_anti.data
                mask = u >= v
                _submit_cone(
                    alpha_col[mask].astype(np.int32), moment_bar_ids[1],
                    u[mask].astype(np.int32), v[mask].astype(np.int32),
                    val[mask].astype(np.float64),
                )

    # ---- Localizing cones ----
    if n_loc > 0:
        L_arr = np.asarray(loc_basis, dtype=np.int64)
        L_hash = _hash_monos(L_arr, bases_arr)
        ks_l, ls_l = np.tril_indices(n_loc)
        base_hash_loc = L_hash[ks_l] + L_hash[ls_l]
        alpha_idx_loc0 = _alpha_lookup(
            base_hash_loc, sorted_h, sort_o, old_to_new_arr)
    else:
        L_arr = np.zeros((0, d), dtype=np.int64)
        L_hash = np.zeros(0, dtype=np.int64)
        ks_l = np.zeros(0, dtype=np.int64)
        ls_l = np.zeros(0, dtype=np.int64)
        base_hash_loc = np.zeros(0, dtype=np.int64)
        alpha_idx_loc0 = np.zeros(0, dtype=np.int64)

    for j, i in enumerate(active_loc):
        alpha_hash_li = base_hash_loc + bases_arr[i]
        alpha_idx_li = _alpha_lookup(
            alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
        mask = alpha_idx_li >= 0
        n_m = int(mask.sum())
        if n_m == 0:
            continue
        _submit_cone(
            alpha_idx_li[mask], loc_bar_start + j,
            ks_l[mask], ls_l[mask],
            np.full(n_m, +1.0, dtype=np.float64),
        )

    # ---- Upper-localizing ----
    # Each uloc cone is the SUM of a +1 (loc+loc) and a −1 (loc+loc+e_i)
    # component — we submit both in ONE call so intra-cone aggregation
    # can collapse entries that land on the same (α, k, l) across the
    # two components.
    if include_upper_loc:
        mask_t0 = alpha_idx_loc0 >= 0
        for j, i in enumerate(active_loc):
            bar_idx_here = uloc_bar_start + j
            pieces_subi: List[np.ndarray] = []
            pieces_subk: List[np.ndarray] = []
            pieces_subl: List[np.ndarray] = []
            pieces_val: List[np.ndarray] = []
            if np.any(mask_t0):
                n_m = int(mask_t0.sum())
                pieces_subi.append(alpha_idx_loc0[mask_t0])
                pieces_subk.append(ks_l[mask_t0])
                pieces_subl.append(ls_l[mask_t0])
                pieces_val.append(np.full(n_m, +1.0, dtype=np.float64))
            alpha_hash_li = base_hash_loc + bases_arr[i]
            alpha_idx_li = _alpha_lookup(
                alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
            mask = alpha_idx_li >= 0
            n_m = int(mask.sum())
            if n_m > 0:
                pieces_subi.append(alpha_idx_li[mask])
                pieces_subk.append(ks_l[mask])
                pieces_subl.append(ls_l[mask])
                pieces_val.append(np.full(n_m, -1.0, dtype=np.float64))
            if not pieces_subi:
                continue
            _submit_cone(
                np.concatenate(pieces_subi), bar_idx_here,
                np.concatenate(pieces_subk),
                np.concatenate(pieces_subl),
                np.concatenate(pieces_val),
            )

    # ---- Window cones ----
    # Each window cone combines a +t·(loc+loc) piece and a −M_W·(loc+loc+e_i+e_j)
    # piece.  Submit both together per window so intra-cone aggregation
    # collapses (α, k, l) duplicates from z2 and M_W symmetry.
    for w_j, w in enumerate(active_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_bar_start + w_j

        pieces_subi: List[np.ndarray] = []
        pieces_subk: List[np.ndarray] = []
        pieces_subl: List[np.ndarray] = []
        pieces_val: List[np.ndarray] = []
        pieces_tcoef: List[np.ndarray] = []

        mask_t = alpha_idx_loc0 >= 0
        n_m = int(mask_t.sum())
        if n_m:
            pieces_subi.append(alpha_idx_loc0[mask_t])
            pieces_subk.append(ks_l[mask_t])
            pieces_subl.append(ls_l[mask_t])
            pieces_val.append(np.zeros(n_m, dtype=np.float64))
            pieces_tcoef.append(np.ones(n_m, dtype=np.float64))

        for ii, jj in zip(nz_i.tolist(), nz_j.tolist()):
            if ii < jj:
                continue
            raw = float(Mw[ii, jj])
            if raw == 0.0:
                continue
            coef = -raw if ii == jj else -2.0 * raw
            alpha_hash_q = (base_hash_loc
                            + bases_arr[ii] + bases_arr[jj])
            alpha_idx_q = _alpha_lookup(
                alpha_hash_q, sorted_h, sort_o, old_to_new_arr)
            mask = alpha_idx_q >= 0
            n_m = int(mask.sum())
            if n_m == 0:
                continue
            pieces_subi.append(alpha_idx_q[mask])
            pieces_subk.append(ks_l[mask])
            pieces_subl.append(ls_l[mask])
            pieces_val.append(np.full(n_m, coef, dtype=np.float64))
            pieces_tcoef.append(np.zeros(n_m, dtype=np.float64))

        if not pieces_subi:
            continue
        _submit_cone(
            np.concatenate(pieces_subi), W_bar_idx,
            np.concatenate(pieces_subk),
            np.concatenate(pieces_subl),
            np.concatenate(pieces_val),
            tcoefs=np.concatenate(pieces_tcoef),
        )

    # ---- Consolidate per-cone cache (streamed submissions already done) ----
    # Each _submit_cone() call has already aggregated + T-contracted +
    # submitted its triplet to MOSEK.  For update_task_t_preelim we
    # concatenate the per-cone caches into flat _all_* arrays; distinct
    # cones have distinct subj, so no cross-cone aggregation needed.
    n_bar_entries = int(sum(cone_entry_counts))
    if cache_for_reuse and cache_arrays:
        all_subi = np.concatenate([c[0] for c in cache_arrays])
        all_subj = np.concatenate([c[1] for c in cache_arrays])
        all_subk = np.concatenate([c[2] for c in cache_arrays])
        all_subl = np.concatenate([c[3] for c in cache_arrays])
        all_val = np.concatenate([c[4] for c in cache_arrays])
        all_tcoef = np.concatenate([c[5] for c in cache_arrays])
        # Drop per-cone arrays — free memory after concat.
        cache_arrays.clear()
    else:
        all_subi = all_subj = all_subk = all_subl = None
        all_val = all_tcoef = None

    # -----------------------------------------------------------------
    # 6. Objective: max λ.
    # -----------------------------------------------------------------
    task.putobjsense(mosek.objsense.maximize)
    task.putcj(LAMBDA_IDX, 1.0)

    build_time = time.time() - t0

    info = {
        'build_time_s': build_time,
        'bar_sizes': bar_sizes,
        'n_bar': n_bar,
        'moment_bar_ids': moment_bar_ids,
        'loc_bar_start': loc_bar_start,
        'loc_bar_end': loc_bar_end,
        'uloc_bar_start': uloc_bar_start,
        'uloc_bar_end': uloc_bar_end,
        'win_bar_start': win_bar_start,
        'win_bar_end': win_bar_end,
        'active_loc': list(active_loc),
        'active_windows': list(active_windows),
        'n_scalar': n_scalar,
        'n_consist_kept': n_resid,
        'n_cons': n_y_red,
        'n_y': n_y,
        'n_y_red': n_y_red,
        'n_resid': n_resid,
        'simplex_row_idx': int(simplex_row_idx),
        'n_bar_entries': n_bar_entries,
        't_val': float(t_val),
        'LAMBDA_IDX': LAMBDA_IDX,
        'MU_START': MU_START,
        'V_START': V_START,
        'lambda_upper_bound': float(lambda_upper_bound),
        'z2_canonicalized': z2_canon,
        'z2_blockdiag': z2_blockdiag_map is not None,
        'include_upper_loc': bool(include_upper_loc),
        '_all_subi': all_subi,
        '_all_subj': all_subj,
        '_all_subk': all_subk,
        '_all_subl': all_subl,
        '_all_static': all_val,
        '_all_tcoef': all_tcoef,
        '_dyn_count': 0 if all_tcoef is None else int(np.count_nonzero(all_tcoef)),
        'n_bar_entries_total': n_bar_entries,
        'n_dynamic_entries': 0 if all_tcoef is None else int(np.count_nonzero(all_tcoef)),
        'xf': xf,
    }

    if verbose:
        reduction_pct = 100.0 * (1.0 - n_y_red / max(n_y, 1))
        print(f"  [dual-preelim] n_y {n_y} -> n_y_red {n_y_red} "
              f"({reduction_pct:.1f}% reduction)  n_resid={n_resid}  "
              f"n_scalar={n_scalar:,}  n_bar_entries={n_bar_entries:,}  "
              f"t={t_val:.6f}  build={build_time:.2f}s",
              flush=True)

    return task, info


# =====================================================================
# Task-reuse update (same logic as dual_sdp.update_task_t)
# =====================================================================

def update_task_t_preelim(task: mosek.Task, info: Dict[str, Any],
                           t_val: float) -> None:
    """Update t-dependent bar entries of a preelim-built task.

    Identical to ``lasserre.dual_sdp.update_task_t`` — the reduced-task
    triplet structure is the same shape (aggregated (subi, subj, subk,
    subl, val, tcoef)) and MOSEK's update API is unchanged.
    """
    all_subi = info.get('_all_subi')
    if all_subi is None or all_subi.size == 0:
        info['t_val'] = float(t_val)
        return
    new_vals = info['_all_static'] + float(t_val) * info['_all_tcoef']
    task.putbarablocktriplet(
        info['_all_subi'], info['_all_subj'],
        info['_all_subk'], info['_all_subl'], new_vals)
    info['t_val'] = float(t_val)

    try:
        task.putintparam(mosek.iparam.intpnt_hotstart,
                          mosek.intpnthotstart.none)
        task.putintparam(mosek.iparam.intpnt_starting_point,
                          mosek.startpointtype.free)
    except Exception:
        pass
    for sol_kind in (mosek.soltype.itr, mosek.soltype.bas, mosek.soltype.itg):
        try:
            task.deletesolution(sol_kind)
        except Exception:
            pass


# =====================================================================
# Post-solve window multiplier audit  (#3 — provably-slack detection)
# =====================================================================

def audit_window_multipliers(
    task: mosek.Task,
    info: Dict[str, Any],
    *,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute Frobenius norm of each window cone's dual X_W matrix and
    classify windows as slack vs active.

    By complementary slackness on the PSD cone, a window constraint
    t·M_{k-1}(y) − M_{k-1}(q_W y) ⪰ 0 that is strictly satisfied at the
    optimum has dual X_W = 0 ⇒ ‖X_W‖_F = 0.  Windows with ‖X_W‖_F below
    ``tol`` can be dropped without changing the bound (at this t).

    Parameters
    ----------
    task : solved MOSEK task (post-optimize()).
    info : dict from build_dual_task_preelim (needs win_bar_start,
           active_windows, bar_sizes).
    tol  : Frobenius-norm threshold below which a window is declared slack.

    Returns
    -------
    dict with:
      'win_norms'     : dict window_index -> ‖X_W‖_F
      'slack_windows' : list of window indices with ‖X_W‖_F < tol
      'active_windows_kept' : list to pass as active_windows on a
                              follow-up build to test bound preservation.
    """
    win_bar_start = int(info['win_bar_start'])
    active_windows = list(info['active_windows'])
    bar_sizes = info['bar_sizes']

    win_norms: Dict[int, float] = {}
    slack: List[int] = []
    active: List[int] = []

    for i, w in enumerate(active_windows):
        bar_idx = win_bar_start + i
        size = int(bar_sizes[bar_idx])
        if size == 0:
            win_norms[w] = 0.0
            slack.append(w)
            continue
        packed = task.getbarsj(mosek.soltype.itr, bar_idx)
        # MOSEK returns the lower triangle in column-major form — length
        # size*(size+1)/2.  Frobenius norm = sqrt(sum diag^2 + 2·sum off-diag^2).
        packed_arr = np.asarray(packed, dtype=np.float64)
        # Build a mask for diagonal entries in the packed vector.
        # Column-major lower-triangle index mapping: for column j, entries
        # at positions [offset(j), offset(j)+size-j); within-column offset
        # 0 is the diagonal element X[j,j].
        diag_positions = np.zeros(size, dtype=np.int64)
        off = 0
        for j in range(size):
            diag_positions[j] = off
            off += (size - j)
        is_diag = np.zeros(packed_arr.size, dtype=bool)
        is_diag[diag_positions] = True
        fnorm_sq = (packed_arr[is_diag] ** 2).sum() + \
                   2.0 * (packed_arr[~is_diag] ** 2).sum()
        fnorm = float(np.sqrt(fnorm_sq))
        win_norms[w] = fnorm
        if fnorm < tol:
            slack.append(w)
        else:
            active.append(w)

    if verbose:
        sorted_pairs = sorted(win_norms.items(), key=lambda x: x[1])
        n_slack = len(slack)
        n_total = len(active_windows)
        print(f"  [window-audit] {n_slack}/{n_total} windows slack "
              f"(||X_W||_F < {tol:.0e})", flush=True)
        if sorted_pairs:
            # Print 5 smallest and 5 largest norms for sanity.
            lo = sorted_pairs[:5]
            hi = sorted_pairs[-5:]
            print(f"    smallest ‖X_W‖_F: " +
                  "  ".join(f"W{w}:{n:.2e}" for w, n in lo), flush=True)
            print(f"    largest  ‖X_W‖_F: " +
                  "  ".join(f"W{w}:{n:.2e}" for w, n in hi), flush=True)

    return {
        'win_norms': win_norms,
        'slack_windows': slack,
        'active_windows_kept': active,
    }


# =====================================================================
# Flat-extension tightness check  (#6 — Curto–Fialkow early-exit)
# =====================================================================

def flat_extension_check(
    task: mosek.Task,
    info: Dict[str, Any],
    P: Dict[str, Any],
    *,
    rank_tol: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test Curto–Fialkow flat-extension condition on the primal moment
    matrices recovered from the SOS-dual solve.

    rank(M_L(y*)) == rank(M_{L-1}(y*))  ⟹  relaxation is TIGHT at this t*,
    and val_L(d) = t*.

    The primal moment vector ỹ* is extracted from MOSEK's dual variables
    (``task.gety``), then lifted to full y* via xf.reconstruct.  M_L and
    M_{L-1} are built by hash-lookup from the basis / loc_basis arrays.

    Parameters
    ----------
    task     : solved MOSEK task (post-optimize, FEAS verdict).
    info     : dict from build_dual_task_preelim (must include 'xf').
    P        : canonicalized precompute dict.  Provides basis, loc_basis,
               idx (α tuple → canonical row).
    rank_tol : relative SVD tolerance for rank determination
               (σ_i ≤ rank_tol · σ_max treated as zero).

    Returns
    -------
    dict:
      'rank_ML'     : int
      'rank_MLm1'   : int
      'flat'        : bool (ranks match)
      'sigma_ML'    : np.ndarray of singular values of M_L
      'sigma_MLm1'  : np.ndarray of singular values of M_{L-1}
      'y_red_norm'  : float, ‖ỹ*‖_2 (sanity check)
      'primal_feasible' : bool (True if extraction succeeded)
    """
    xf: PreElimTransform = info['xf']
    n_y_red = int(info['n_y_red'])
    # Extract the primal ỹ* from MOSEK's dual variables (one per
    # stationarity constraint).  Use `gety` which corresponds to the
    # Lagrange duals of our n_cons equality rows.
    try:
        y_red_full = np.zeros(n_y_red, dtype=np.float64)
        task.gety(mosek.soltype.itr, y_red_full)
    except Exception as exc:
        if verbose:
            print(f"  [flat-ext] dual extraction failed: {exc}", flush=True)
        return {'flat': False, 'primal_feasible': False,
                'error': str(exc)}

    # Reconstruct full y from reduced ỹ.
    y_full = xf.reconstruct(y_red_full)  # shape (n_y,) in canonical space.

    basis = P['basis']
    loc_basis = P['loc_basis']
    mono_idx = P['idx']
    d = int(P['d'])
    n_basis = int(P['n_basis'])
    n_loc = int(P['n_loc'])

    if n_basis == 0:
        return {'flat': False, 'primal_feasible': True,
                'rank_ML': 0, 'rank_MLm1': 0}

    # Build M_L(y) via vectorized hash-lookup.  basis[k] + basis[l] =
    # pointwise monomial-index sum (multi-index addition).
    def _build_moment_matrix(b_arr):
        nb = len(b_arr)
        M = np.zeros((nb, nb), dtype=np.float64)
        # Iterate lower-triangle + diagonal; fill symmetric.
        for k in range(nb):
            for l in range(k + 1):
                alpha = tuple(int(b_arr[k][i]) + int(b_arr[l][i])
                              for i in range(d))
                if alpha in mono_idx:
                    M[k, l] = y_full[int(mono_idx[alpha])]
                    if k != l:
                        M[l, k] = M[k, l]
        return M

    M_L = _build_moment_matrix(basis)
    if n_loc > 0:
        M_Lm1 = _build_moment_matrix(loc_basis)
    else:
        M_Lm1 = np.zeros((0, 0))

    # Numerical rank via SVD + relative threshold.
    if M_L.size:
        sigma_L = np.linalg.svd(M_L, compute_uv=False)
        smax_L = float(sigma_L.max()) if sigma_L.size else 0.0
        r_L = int(np.sum(sigma_L > rank_tol * smax_L))
    else:
        sigma_L = np.zeros(0)
        r_L = 0

    if M_Lm1.size:
        sigma_Lm1 = np.linalg.svd(M_Lm1, compute_uv=False)
        smax_Lm1 = float(sigma_Lm1.max()) if sigma_Lm1.size else 0.0
        r_Lm1 = int(np.sum(sigma_Lm1 > rank_tol * smax_Lm1))
    else:
        sigma_Lm1 = np.zeros(0)
        r_Lm1 = 0

    flat = (r_L == r_Lm1) and (r_L > 0)

    if verbose:
        msg = "FLAT" if flat else "not flat"
        print(f"  [flat-ext] rank(M_L)={r_L}  rank(M_{{L-1}})={r_Lm1}  "
              f"→ {msg}  (rank_tol={rank_tol:.0e})", flush=True)

    return {
        'rank_ML': r_L,
        'rank_MLm1': r_Lm1,
        'flat': bool(flat),
        'sigma_ML': sigma_L,
        'sigma_MLm1': sigma_Lm1,
        'y_red_norm': float(np.linalg.norm(y_red_full)),
        'primal_feasible': True,
    }
