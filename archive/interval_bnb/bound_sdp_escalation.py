"""Lasserre order-2 SDP ESCALATION via the existing dual-Farkas Task API.

Wraps `lasserre/dual_sdp.py` for per-box use. The original
`build_dual_task` does Lasserre val(d) Farkas at fixed t over the FULL
simplex (μ_i ≥ 0, optionally 1 − μ_i ≥ 0). We need it over a BOX
(lo_i ≤ μ_i ≤ hi_i). The math:

  Lower box: M_{k-1}((μ_i − lo_i) y) ⪰ 0
    sensitivity: E_lo_i[α]_{β,γ} = 𝟙[loc[β]+loc[γ]+e_i = α]
                              − lo_i · 𝟙[loc[β]+loc[γ] = α]
  Upper box: M_{k-1}((hi_i − μ_i) y) ⪰ 0
    sensitivity: E_hi_i[α]_{β,γ} = hi_i · 𝟙[loc[β]+loc[γ] = α]
                              − 𝟙[loc[β]+loc[γ]+e_i = α]

Both are linear shifts of the existing `μ_i ≥ 0` and `(1 − μ_i) ≥ 0`
sensitivity coefficients. The bar-matrix index structure (subi, subj,
subk, subl) is IDENTICAL — only the values change. We can therefore
copy the entire `build_dual_task` body and just substitute the
localising-block coefficients.

Speed (per `lasserre/dual_sdp.py` docstring): Task API +
putbarablocktriplet bulk submission avoids Fusion's O(n) build cost.
At d=22..30 with task reuse via `update_task_box`, repeated per-box
solves are ~3-15s each (build the task ONCE per worker, then just
re-submit the bar triplet on each box).


SOUNDNESS
---------
Farkas duality (Lasserre 2001 Thm 4.2 + standard conic Farkas):

  Primal feasibility SDP at fixed t:  ∃ y satisfying all PSD blocks
  (moment, lower-box, upper-box, window) AND the equalities
  (y_0 = 1, simplex consistency). Equivalent to: ∃ μ in box ∩ Δ_d
  with max_W μ^T M_W μ ≤ t.

  Dual Farkas LP: max λ s.t. (stationarity rows) under PSD multipliers
  for each cone. λ* > 0 ↔ no such y exists ↔ val_B > t.

So: solve the dual Farkas LP at t = target.
  λ* ≥ infeas_threshold * λ_ub  →  val_B > target  →  CERT.
  λ* ≤ feas_threshold  * λ_ub  →  val_B ≤ target  →  no cert.
  intermediate λ*                  →  ambiguous, no cert.

The decision is RIGOROUS modulo MOSEK's reported residuals + threshold
margin. We use 0.25 / 0.75 thresholds (default in `solve_dual_task`)
which leaves a 0.5 gap absorbing solver tolerance.


EMPTY BOX
---------
sum(lo) > 1+ε OR sum(hi) < 1−ε → vacuous cert (returns True).
"""
from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import mosek

from .box import SCALE as _SCALE


# ---------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------

_DUAL_SDP_API = None


def _import_dual_sdp_api():
    """Vendored / proxy access to lasserre.dual_sdp helpers + precompute.

    Hash utilities come from `lasserre.core` (NOT `tests/lasserre_fusion`)
    because `lasserre.precompute` uses the core variant — at d>=22 the
    core impl switches to a Mersenne-prime split-base hash to avoid
    int64 overflow, while the lasserre_fusion impl uses naive int64
    tensordot which gives DIFFERENT hash values. Mismatch causes the
    moment-matrix lookup to return -1 spuriously at d=22+.
    """
    global _DUAL_SDP_API
    if _DUAL_SDP_API is None:
        from lasserre.precompute import _precompute
        from lasserre.dual_sdp import (
            _aggregate_bar_triplet, _aggregate_scalar_triplet,
            solve_dual_task, update_task_t,
        )
        from lasserre.core import _hash_monos, _hash_lookup

        def _alpha_lookup(alpha_hash, sorted_h, sort_o, old_to_new=None):
            raw = _hash_lookup(alpha_hash, sorted_h, sort_o)
            if old_to_new is None:
                return raw
            out = np.where(raw < 0, -1, old_to_new[np.maximum(raw, 0)])
            return out

        _DUAL_SDP_API = {
            '_precompute': _precompute,
            '_aggregate_bar_triplet': _aggregate_bar_triplet,
            '_aggregate_scalar_triplet': _aggregate_scalar_triplet,
            '_alpha_lookup': _alpha_lookup,
            'solve_dual_task': solve_dual_task,
            'update_task_t': update_task_t,
            '_hash_monos': _hash_monos,
            '_hash_lookup': _hash_lookup,
        }
    return _DUAL_SDP_API


# ---------------------------------------------------------------------
# Box-localizing dual Farkas Task builder
# (vendored from lasserre/dual_sdp.py with localizing block modified
#  to use box endpoints lo, hi instead of μ_i ≥ 0, 1 - μ_i ≥ 0.)
# ---------------------------------------------------------------------

def _build_dual_task_box(
    P: Dict[str, Any], lo: np.ndarray, hi: np.ndarray, t_val: float,
    env: mosek.Env, *,
    active_windows: Optional[List[int]] = None,
    lambda_upper_bound: float = 1.0,
    verbose: bool = False,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Build the dual Farkas LP for the per-box Lasserre order-2 SDP at
    fixed t = t_val with BOX localizing constraints lo ≤ μ ≤ hi.

    Returns (task, info). task.optimize() then `solve_dual_task` to read
    the verdict. info contains the bar-triplet caches needed for
    `update_task_box` to re-submit on box / t change without rebuild.
    """
    api = _import_dual_sdp_api()
    _hash_monos = api['_hash_monos']
    _alpha_lookup = api['_alpha_lookup']
    _aggregate_bar_triplet = api['_aggregate_bar_triplet']
    _aggregate_scalar_triplet = api['_aggregate_scalar_triplet']
    from lasserre.core import _hash_add

    d = int(P['d'])
    n_y = int(P['n_y'])
    basis = P['basis']
    n_basis = int(P['n_basis'])
    loc_basis = P['loc_basis']
    n_loc = int(P['n_loc'])
    mono_idx = P['idx']
    M_mats = P['M_mats']
    bases_arr = np.asarray(P['bases'], dtype=np.int64)
    prime = P.get('prime')
    sorted_h = np.asarray(P['sorted_h'])
    sort_o = np.asarray(P['sort_o'])
    consist_mono = P['consist_mono']
    consist_idx = np.asarray(P['consist_idx'], dtype=np.int64)
    consist_ei_idx = np.asarray(P['consist_ei_idx'], dtype=np.int64)
    old_to_new_arr = P.get('old_to_new')
    if old_to_new_arr is not None:
        old_to_new_arr = np.asarray(old_to_new_arr, dtype=np.int64)

    if n_loc == 0:
        raise RuntimeError("Order-2 Lasserre requires n_loc > 0")
    active_loc = list(range(d))
    if active_windows is None:
        active_windows = list(P['nontrivial_windows'])

    task = env.Task()
    if verbose:
        task.set_Stream(mosek.streamtype.log, lambda s: print(s, end=''))

    t0 = time.time()

    # ----- Bar-variable layout -----
    # [moment X_0] + [lower-box X_lo_i for i=0..d-1] + [upper-box X_hi_i] + [window X_W]
    bar_sizes: List[int] = [n_basis]
    moment_bar_id = 0
    lo_bar_start = 1
    for _ in active_loc:
        bar_sizes.append(n_loc)
    lo_bar_end = len(bar_sizes)
    hi_bar_start = lo_bar_end
    for _ in active_loc:
        bar_sizes.append(n_loc)
    hi_bar_end = len(bar_sizes)
    win_bar_start = hi_bar_end
    for _ in active_windows:
        bar_sizes.append(n_loc)
    win_bar_end = len(bar_sizes)
    n_bar = len(bar_sizes)
    task.appendbarvars(bar_sizes)

    # ----- Scalar variables: [λ | μ_k | v_α] -----
    kept_k = [k for k in range(len(consist_mono)) if int(consist_idx[k]) >= 0]
    n_consist = len(kept_k)
    LAMBDA_IDX = 0
    MU_START = 1
    MU_END = 1 + n_consist
    V_START = MU_END
    V_END = V_START + n_y
    n_scalar = V_END
    task.appendvars(n_scalar)

    task.putvarbound(LAMBDA_IDX, mosek.boundkey.ra, 0.0,
                     float(lambda_upper_bound))
    if n_consist > 0:
        task.putvarboundslice(
            MU_START, MU_END,
            [mosek.boundkey.fr] * n_consist,
            np.full(n_consist, -np.inf, dtype=np.float64),
            np.full(n_consist, +np.inf, dtype=np.float64),
        )
    task.putvarboundslice(
        V_START, V_END,
        [mosek.boundkey.lo] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.full(n_y, +np.inf, dtype=np.float64),
    )

    # ----- Constraint rows: one per α (stationarity = 0) -----
    task.appendcons(n_y)
    task.putconboundslice(
        0, n_y, [mosek.boundkey.fx] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.zeros(n_y, dtype=np.float64),
    )

    # ----- Scalar coefficients (A matrix) -----
    scalar_rows: List[int] = []
    scalar_cols: List[int] = []
    scalar_vals: List[float] = []

    alpha_zero = tuple(0 for _ in range(d))
    alpha_zero_row = int(mono_idx[alpha_zero])
    scalar_rows.append(alpha_zero_row)
    scalar_cols.append(LAMBDA_IDX)
    scalar_vals.append(1.0)

    for j, k in enumerate(kept_k):
        a_k = int(consist_idx[k])
        child = consist_ei_idx[k]
        coef_by_row: Dict[int, float] = {}
        for i in range(d):
            c = int(child[i])
            if c >= 0:
                coef_by_row[c] = coef_by_row.get(c, 0.0) + 1.0
        coef_by_row[a_k] = coef_by_row.get(a_k, 0.0) - 1.0
        col_j = MU_START + j
        for row, val in coef_by_row.items():
            if val != 0.0:
                scalar_rows.append(row)
                scalar_cols.append(col_j)
                scalar_vals.append(float(val))

    alpha_rows = np.arange(n_y, dtype=np.int64)
    scalar_rows.extend(alpha_rows.tolist())
    scalar_cols.extend((V_START + alpha_rows).tolist())
    scalar_vals.extend([1.0] * n_y)

    r_arr, c_arr, v_arr = _aggregate_scalar_triplet(
        np.asarray(scalar_rows, dtype=np.int64),
        np.asarray(scalar_cols, dtype=np.int64),
        np.asarray(scalar_vals, dtype=np.float64),
    )
    if r_arr.size:
        task.putaijlist(r_arr, c_arr, v_arr)

    # ----- Bar-matrix sensitivity coefficients -----
    bar_subi_list: List[np.ndarray] = []
    bar_subj_list: List[np.ndarray] = []
    bar_subk_list: List[np.ndarray] = []
    bar_subl_list: List[np.ndarray] = []
    bar_val_list: List[np.ndarray] = []
    bar_tcoef_list: List[np.ndarray] = []  # parallel coefficient-of-t array
    # Box-coefficient arrays: parallel arrays carrying for each entry
    # (lo_idx, lo_coef, hi_idx, hi_coef) — used by update_task_box to
    # re-evaluate without rebuild. -1 means "static (no box dependence)".
    bar_box_lo_idx_list: List[np.ndarray] = []
    bar_box_lo_coef_list: List[np.ndarray] = []
    bar_box_hi_idx_list: List[np.ndarray] = []
    bar_box_hi_coef_list: List[np.ndarray] = []

    def _append(subi, subj, subk, subl, vals,
                tcoefs=None, lo_idx=None, lo_coef=None,
                hi_idx=None, hi_coef=None):
        if subi.size == 0:
            return
        bar_subi_list.append(np.ascontiguousarray(subi, dtype=np.int32))
        bar_subj_list.append(np.ascontiguousarray(subj, dtype=np.int32))
        bar_subk_list.append(np.ascontiguousarray(subk, dtype=np.int32))
        bar_subl_list.append(np.ascontiguousarray(subl, dtype=np.int32))
        bar_val_list.append(np.ascontiguousarray(vals, dtype=np.float64))
        n = subi.size
        bar_tcoef_list.append(
            np.ascontiguousarray(tcoefs, dtype=np.float64)
            if tcoefs is not None else np.zeros(n, dtype=np.float64))
        bar_box_lo_idx_list.append(
            np.ascontiguousarray(lo_idx, dtype=np.int32)
            if lo_idx is not None else np.full(n, -1, dtype=np.int32))
        bar_box_lo_coef_list.append(
            np.ascontiguousarray(lo_coef, dtype=np.float64)
            if lo_coef is not None else np.zeros(n, dtype=np.float64))
        bar_box_hi_idx_list.append(
            np.ascontiguousarray(hi_idx, dtype=np.int32)
            if hi_idx is not None else np.full(n, -1, dtype=np.int32))
        bar_box_hi_coef_list.append(
            np.ascontiguousarray(hi_coef, dtype=np.float64)
            if hi_coef is not None else np.zeros(n, dtype=np.float64))

    # Moment cone: single n_basis × n_basis BAR; coef = +1 at α=loc[k]+loc[l].
    B_arr = np.asarray(basis, dtype=np.int64)
    B_hash = _hash_monos(B_arr, bases_arr, prime)
    ks_m, ls_m = np.tril_indices(n_basis)
    alpha_hash_m = _hash_add(B_hash[ks_m], B_hash[ls_m], prime)
    alpha_idx_m = _alpha_lookup(alpha_hash_m, sorted_h, sort_o, old_to_new_arr)
    if np.any(alpha_idx_m < 0):
        raise RuntimeError("Moment sensitivity lookup -1; precompute broken")
    _append(
        alpha_idx_m,
        np.full(ks_m.shape, moment_bar_id, dtype=np.int32),
        ks_m, ls_m,
        np.full(ks_m.shape, +1.0, dtype=np.float64),
    )

    # Localizing prep: hashes for loc_basis tabulation.
    L_arr = np.asarray(loc_basis, dtype=np.int64)
    L_hash = _hash_monos(L_arr, bases_arr, prime)
    ks_l, ls_l = np.tril_indices(n_loc)
    base_hash_loc = _hash_add(L_hash[ks_l], L_hash[ls_l], prime)  # α = loc[k] + loc[l]
    alpha_idx_loc0 = _alpha_lookup(
        base_hash_loc, sorted_h, sort_o, old_to_new_arr)

    # ---- LOWER BOX cones X_lo_i: M_{k-1}((μ_i − lo_i) y) ⪰ 0 ----
    # Sensitivity: +1 · 𝟙[loc+loc+e_i = α]   (no box-dep)
    #              −lo_i · 𝟙[loc+loc = α]    (box-dep on lo[i])
    for j, i in enumerate(active_loc):
        bar_idx_here = lo_bar_start + j
        # +1 part (from μ_i term)
        alpha_hash_li = _hash_add(base_hash_loc, bases_arr[i], prime)
        alpha_idx_li = _alpha_lookup(
            alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
        mask = alpha_idx_li >= 0
        n_m = int(mask.sum())
        if n_m:
            _append(
                alpha_idx_li[mask],
                np.full(n_m, bar_idx_here, dtype=np.int32),
                ks_l[mask], ls_l[mask],
                np.full(n_m, +1.0, dtype=np.float64),
            )
        # -lo_i part (from -lo_i · 1 term)
        mask0 = alpha_idx_loc0 >= 0
        n_m0 = int(mask0.sum())
        if n_m0:
            # value = -lo[i] · 1  (filled in at submit time)
            _append(
                alpha_idx_loc0[mask0],
                np.full(n_m0, bar_idx_here, dtype=np.int32),
                ks_l[mask0], ls_l[mask0],
                np.zeros(n_m0, dtype=np.float64),  # static = 0
                lo_idx=np.full(n_m0, i, dtype=np.int32),
                lo_coef=np.full(n_m0, -1.0, dtype=np.float64),
            )

    # ---- UPPER BOX cones X_hi_i: M_{k-1}((hi_i − μ_i) y) ⪰ 0 ----
    # Sensitivity: +hi_i · 𝟙[loc+loc = α]    (box-dep on hi[i])
    #              −1   · 𝟙[loc+loc+e_i = α]   (no box-dep)
    for j, i in enumerate(active_loc):
        bar_idx_here = hi_bar_start + j
        # +hi_i part
        mask0 = alpha_idx_loc0 >= 0
        n_m0 = int(mask0.sum())
        if n_m0:
            _append(
                alpha_idx_loc0[mask0],
                np.full(n_m0, bar_idx_here, dtype=np.int32),
                ks_l[mask0], ls_l[mask0],
                np.zeros(n_m0, dtype=np.float64),  # static = 0
                hi_idx=np.full(n_m0, i, dtype=np.int32),
                hi_coef=np.full(n_m0, +1.0, dtype=np.float64),
            )
        # -1 part
        alpha_hash_li = _hash_add(base_hash_loc, bases_arr[i], prime)
        alpha_idx_li = _alpha_lookup(
            alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
        mask = alpha_idx_li >= 0
        n_m = int(mask.sum())
        if n_m:
            _append(
                alpha_idx_li[mask],
                np.full(n_m, bar_idx_here, dtype=np.int32),
                ks_l[mask], ls_l[mask],
                np.full(n_m, -1.0, dtype=np.float64),
            )

    # ---- Window cones X_W: +t · E_W^t − E_W^Q ----
    for w_j, w in enumerate(active_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_bar_start + w_j

        # (a) t-part: coef = +t at α = loc[k]+loc[l].
        mask_t = alpha_idx_loc0 >= 0
        n_m = int(mask_t.sum())
        if n_m:
            _append(
                alpha_idx_loc0[mask_t],
                np.full(n_m, W_bar_idx, dtype=np.int32),
                ks_l[mask_t], ls_l[mask_t],
                np.zeros(n_m, dtype=np.float64),  # static = 0
                tcoefs=np.full(n_m, +1.0, dtype=np.float64),
            )

        # (b) -Q-part: coef = -Σ_{i,j} M_W[i,j] at α = loc[k]+loc[l]+e_i+e_j.
        # Vectorize over (k, l, ij_pair).
        if len(nz_i) > 0:
            # For each ij_pair, compute alpha_hash = base_hash_loc + bases[i] + bases[j]
            # Stack: shape (n_pairs, n_kl).
            n_pairs = len(nz_i)
            shifts = _hash_add(bases_arr[nz_i], bases_arr[nz_j], prime)
            mw_vals = Mw[nz_i, nz_j]  # (n_pairs,)
            for pp in range(n_pairs):
                alpha_hash_pij = _hash_add(base_hash_loc, shifts[pp], prime)
                alpha_idx_pij = _alpha_lookup(
                    alpha_hash_pij, sorted_h, sort_o, old_to_new_arr)
                mask_q = alpha_idx_pij >= 0
                n_q = int(mask_q.sum())
                if n_q == 0:
                    continue
                _append(
                    alpha_idx_pij[mask_q],
                    np.full(n_q, W_bar_idx, dtype=np.int32),
                    ks_l[mask_q], ls_l[mask_q],
                    np.full(n_q, -float(mw_vals[pp]), dtype=np.float64),
                )

    # ---- Concatenate, aggregate duplicates, expand to initial values ----
    if not bar_subi_list:
        n_bar_entries = 0
        all_subi = all_subj = all_subk = all_subl = None
        all_static = all_tcoef = None
        all_lo_idx = all_lo_coef = all_hi_idx = all_hi_coef = None
    else:
        raw_subi = np.concatenate(bar_subi_list)
        raw_subj = np.concatenate(bar_subj_list)
        raw_subk = np.concatenate(bar_subk_list)
        raw_subl = np.concatenate(bar_subl_list)
        raw_static = np.concatenate(bar_val_list)
        raw_tcoef = np.concatenate(bar_tcoef_list)
        raw_lo_idx = np.concatenate(bar_box_lo_idx_list)
        raw_lo_coef = np.concatenate(bar_box_lo_coef_list)
        raw_hi_idx = np.concatenate(bar_box_hi_idx_list)
        raw_hi_coef = np.concatenate(bar_box_hi_coef_list)

        # Aggregate duplicates by (subi, subj, subk, subl). Duplicates
        # arise in window Q-blocks (multiple (i,j) pairs hitting same α).
        # Box dependence is per-cone-block — no entry combines lo and hi
        # dependence, and within any duplicate group all entries share the
        # same lo_idx OR all share the same hi_idx OR all have -1.
        order = np.lexsort([raw_subl, raw_subk, raw_subj, raw_subi])
        s_subi = raw_subi[order]
        s_subj = raw_subj[order]
        s_subk = raw_subk[order]
        s_subl = raw_subl[order]
        s_static = raw_static[order]
        s_tcoef = raw_tcoef[order]
        s_lo_idx = raw_lo_idx[order]
        s_lo_coef = raw_lo_coef[order]
        s_hi_idx = raw_hi_idx[order]
        s_hi_coef = raw_hi_coef[order]

        n_total = s_subi.size
        is_first = np.empty(n_total, dtype=bool)
        is_first[0] = True
        is_first[1:] = ((s_subi[1:] != s_subi[:-1]) |
                        (s_subj[1:] != s_subj[:-1]) |
                        (s_subk[1:] != s_subk[:-1]) |
                        (s_subl[1:] != s_subl[:-1]))
        group_id = np.cumsum(is_first.astype(np.int64)) - 1
        n_groups = int(group_id[-1] + 1) if n_total > 0 else 0

        all_subi = s_subi[is_first]
        all_subj = s_subj[is_first]
        all_subk = s_subk[is_first]
        all_subl = s_subl[is_first]
        all_static = np.bincount(group_id, weights=s_static.astype(np.float64),
                                  minlength=n_groups)
        all_tcoef = np.bincount(group_id, weights=s_tcoef.astype(np.float64),
                                 minlength=n_groups)
        all_lo_coef = np.bincount(group_id,
                                   weights=s_lo_coef.astype(np.float64),
                                   minlength=n_groups)
        all_hi_coef = np.bincount(group_id,
                                   weights=s_hi_coef.astype(np.float64),
                                   minlength=n_groups)
        # For lo_idx and hi_idx, take the max within each group: -1 if all
        # are -1, else the (unique) non-negative index. We use "max" to
        # pick the non-negative index over -1; if there were ever two
        # different non-negative lo_idx in the same group it'd be a bug.
        # Use np.maximum.reduceat with sorted group boundaries.
        first_idx = np.where(is_first)[0]
        # group_max_lo[g] = max(s_lo_idx[first_idx[g]:first_idx[g+1]])
        all_lo_idx = np.maximum.reduceat(s_lo_idx, first_idx).astype(np.int32)
        all_hi_idx = np.maximum.reduceat(s_hi_idx, first_idx).astype(np.int32)

        # Drop true zeros (val == 0 and tcoef == 0 and box coefs == 0).
        keep = ((np.abs(all_static) > 0.0) | (np.abs(all_tcoef) > 0.0)
                | (np.abs(all_lo_coef) > 0.0) | (np.abs(all_hi_coef) > 0.0))
        if not np.all(keep):
            all_subi = all_subi[keep]; all_subj = all_subj[keep]
            all_subk = all_subk[keep]; all_subl = all_subl[keep]
            all_static = all_static[keep]; all_tcoef = all_tcoef[keep]
            all_lo_idx = all_lo_idx[keep]; all_lo_coef = all_lo_coef[keep]
            all_hi_idx = all_hi_idx[keep]; all_hi_coef = all_hi_coef[keep]

        n_bar_entries = int(all_subi.size)
        # Initial submission: static + t·tcoef + lo[lo_idx]·lo_coef + hi[hi_idx]·hi_coef
        init_vals = (all_static + float(t_val) * all_tcoef
                     + np.where(all_lo_idx >= 0,
                                lo[np.maximum(all_lo_idx, 0)] * all_lo_coef,
                                0.0)
                     + np.where(all_hi_idx >= 0,
                                hi[np.maximum(all_hi_idx, 0)] * all_hi_coef,
                                0.0))
        task.putbarablocktriplet(
            all_subi, all_subj, all_subk, all_subl, init_vals)

    task.putobjsense(mosek.objsense.maximize)
    task.putcj(LAMBDA_IDX, 1.0)

    build_time = time.time() - t0

    info = {
        'build_time_s': build_time,
        'bar_sizes': bar_sizes, 'n_bar': n_bar,
        'lo_bar_start': lo_bar_start, 'lo_bar_end': lo_bar_end,
        'hi_bar_start': hi_bar_start, 'hi_bar_end': hi_bar_end,
        'win_bar_start': win_bar_start, 'win_bar_end': win_bar_end,
        'active_loc': list(active_loc),
        'active_windows': list(active_windows),
        'n_scalar': n_scalar, 'n_consist_kept': n_consist,
        'n_cons': n_y, 'n_y': n_y,
        'n_bar_entries': n_bar_entries,
        't_val': float(t_val),
        'lo': np.asarray(lo, dtype=np.float64).copy(),
        'hi': np.asarray(hi, dtype=np.float64).copy(),
        'LAMBDA_IDX': LAMBDA_IDX, 'MU_START': MU_START, 'V_START': V_START,
        'lambda_upper_bound': float(lambda_upper_bound),
        # Cached triplet for update_task_box.
        '_all_subi': all_subi, '_all_subj': all_subj,
        '_all_subk': all_subk, '_all_subl': all_subl,
        '_all_static': all_static, '_all_tcoef': all_tcoef,
        '_all_lo_idx': all_lo_idx, '_all_lo_coef': all_lo_coef,
        '_all_hi_idx': all_hi_idx, '_all_hi_coef': all_hi_coef,
    }

    if verbose:
        print(f"  [box-dual] n_bar={n_bar} n_scalar={n_scalar:,} "
              f"n_cons={n_y:,} n_bar_entries={n_bar_entries:,} "
              f"t={t_val:.4f} build={build_time:.2f}s "
              f"(loc={len(active_loc)} win={len(active_windows)})",
              flush=True)

    return task, info


def update_task_box(task: mosek.Task, info: Dict[str, Any],
                     lo: np.ndarray, hi: np.ndarray, t_val: float) -> None:
    """Re-submit the bar triplet for new (lo, hi, t). Build is unchanged."""
    if info.get('_all_subi') is None or info['_all_subi'].size == 0:
        info['t_val'] = float(t_val)
        info['lo'] = np.asarray(lo, dtype=np.float64).copy()
        info['hi'] = np.asarray(hi, dtype=np.float64).copy()
        return
    lo_arr = np.asarray(lo, dtype=np.float64)
    hi_arr = np.asarray(hi, dtype=np.float64)
    new_vals = (info['_all_static'] + float(t_val) * info['_all_tcoef']
                + np.where(info['_all_lo_idx'] >= 0,
                           lo_arr[np.maximum(info['_all_lo_idx'], 0)]
                           * info['_all_lo_coef'], 0.0)
                + np.where(info['_all_hi_idx'] >= 0,
                           hi_arr[np.maximum(info['_all_hi_idx'], 0)]
                           * info['_all_hi_coef'], 0.0))
    task.putbarablocktriplet(
        info['_all_subi'], info['_all_subj'],
        info['_all_subk'], info['_all_subl'], new_vals)
    info['t_val'] = float(t_val)
    info['lo'] = lo_arr.copy()
    info['hi'] = hi_arr.copy()
    # Force cold IPM restart (warm start can trap in stale verdict).
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


# ---------------------------------------------------------------------
# Cache build (per-worker; persists task across box solves)
# ---------------------------------------------------------------------

def build_sdp_escalation_cache(d: int, windows=None,
                                bandwidth: Optional[int] = None,
                                target: float = 1.281,
                                verbose: bool = False) -> dict:
    """One-time per-worker setup: build precompute, MOSEK env, and the
    initial dual-Farkas task at t=target with placeholder box [0, 1].

    Returns a dict containing:
      - P            : precompute dict
      - env          : mosek.Env instance
      - task         : the dual-Farkas mosek.Task (reuse via update_task_box)
      - info         : task metadata (cached triplet for re-submit)
      - target       : the t baked into the task

    Per-box solves: call `bound_sdp_escalation_int_ge` which uses
    `update_task_box` to re-submit only the bar coefficients.
    """
    if windows is None:
        from interval_bnb.windows import build_windows
        windows = build_windows(d)
    api = _import_dual_sdp_api()
    P = api['_precompute'](d, order=2, verbose=verbose, lazy_ab_eiej=True)
    # Build with a placeholder box [0, 1] — coefficients will be updated
    # per call via update_task_box.
    lo0 = np.zeros(d, dtype=np.float64)
    hi0 = np.ones(d, dtype=np.float64)
    env = mosek.Env()
    task, info = _build_dual_task_box(
        P, lo0, hi0, t_val=float(target), env=env, verbose=verbose,
    )
    return {
        'd': d,
        'order': 2,
        'P': P,
        'env': env,
        'task': task,
        'info': info,
        'target': float(target),
        'windows_id': id(windows),
    }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

_CUSHION_FLOOR: float = 1e-6
_CUSHION_RESIDUAL_FACTOR: float = 100.0


def _safe_cushion(r_prim: float, r_dual: float, gap: float) -> float:
    """Compatibility shim — Farkas approach uses a different rigor argument."""
    rp = abs(float(r_prim)) if np.isfinite(r_prim) else float('inf')
    rd = abs(float(r_dual)) if np.isfinite(r_dual) else float('inf')
    g = abs(float(gap)) if np.isfinite(gap) else float('inf')
    return max(_CUSHION_RESIDUAL_FACTOR * max(rp, rd, g), _CUSHION_FLOOR)


def bound_sdp_escalation_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int], windows, d: int,
    target_num: int, target_den: int,
    cache: Optional[dict] = None,
    *,
    order: int = 2, bandwidth: Optional[int] = None,
    time_limit_s: float = 30.0,
    tol_gap_abs: float = 1e-5, tol_gap_rel: float = 1e-5, tol_feas: float = 1e-5,
    n_threads: int = 48,
    early_stop: bool = True,
    early_stop_feas_frac: float = 0.15,
    early_stop_infeas_frac: float = 0.85,
    return_diagnostic: bool = False,
):
    """True iff the dual Farkas LP at t = target_num/target_den proves
    primal infeasibility (val_B > target). Sound under-approximation.
    Empty box → True (vacuous)."""
    if int(sum(lo_int)) > _SCALE or int(sum(hi_int)) < _SCALE:
        if return_diagnostic:
            return True, {'status': 'EMPTY', 'cert_via': 'vacuous_empty'}
        return True

    target_f = float(target_num) / float(target_den)
    lo = np.asarray([float(li) / _SCALE for li in lo_int], dtype=np.float64)
    hi = np.asarray([float(hv) / _SCALE for hv in hi_int], dtype=np.float64)

    api = _import_dual_sdp_api()
    if cache is None:
        cache = build_sdp_escalation_cache(d, windows, target=target_f)
    elif cache['target'] != target_f:
        # Target changed — rebuild (or just update via update_task_box).
        update_task_box(cache['task'], cache['info'], lo, hi, target_f)
        cache['target'] = target_f
    else:
        update_task_box(cache['task'], cache['info'], lo, hi, target_f)

    task = cache['task']
    info = cache['info']

    # Solver tolerances + time limit.
    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, float(tol_gap_rel))
    task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit_s))
    if n_threads > 0:
        task.putintparam(mosek.iparam.num_threads, int(n_threads))

    try:
        verdict = api['solve_dual_task'](
            task, info, verbose=False,
            early_stop_on_clear_verdict=early_stop,
            early_stop_feas_frac=early_stop_feas_frac,
            early_stop_infeas_frac=early_stop_infeas_frac,
        )
    except Exception as e:
        if return_diagnostic:
            return False, {
                'status': f'EXCEPTION:{type(e).__name__}',
                'error_msg': str(e), 'cert_via': 'solver_failure',
            }
        return False

    # solve_dual_task returns verdict in lower case: 'infeas' / 'feas' /
    # 'uncertain' / 'solver_*'. 'infeas' means primal SDP infeasible at
    # t = target ⟹ val_B > target ⟹ CERT.
    cert = bool(verdict.get('verdict') == 'infeas')

    if return_diagnostic:
        return cert, {
            **verdict,
            'target_f': target_f,
            'cert_via': 'farkas_infeasibility',
        }
    return cert


def bound_sdp_escalation_lb_float(
    lo: np.ndarray, hi: np.ndarray, windows, d: int,
    *,
    cache: Optional[dict] = None,
    order: int = 2, bandwidth: Optional[int] = None,
    target: float = 1.281,
    time_limit_s: float = 30.0,
    tol_gap_abs: float = 1e-5, tol_gap_rel: float = 1e-5, tol_feas: float = 1e-5,
    n_threads: int = 48,
    early_stop: bool = True,
    early_stop_feas_frac: float = 0.15,
    early_stop_infeas_frac: float = 0.85,
) -> dict:
    """Float-side diagnostic: solve dual Farkas at t = target.
    Returns a dict with the verdict and MOSEK info.
    """
    api = _import_dual_sdp_api()
    if cache is None:
        cache = build_sdp_escalation_cache(d, windows, target=target)
    elif cache['target'] != target:
        update_task_box(cache['task'], cache['info'],
                         np.asarray(lo, dtype=np.float64),
                         np.asarray(hi, dtype=np.float64), target)
        cache['target'] = target
    else:
        update_task_box(cache['task'], cache['info'],
                         np.asarray(lo, dtype=np.float64),
                         np.asarray(hi, dtype=np.float64), target)
    task = cache['task']; info = cache['info']
    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, float(tol_gap_rel))
    task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit_s))
    if n_threads > 0:
        task.putintparam(mosek.iparam.num_threads, int(n_threads))
    try:
        t0 = time.time()
        verdict = api['solve_dual_task'](
            task, info, verbose=False,
            early_stop_on_clear_verdict=early_stop,
            early_stop_feas_frac=early_stop_feas_frac,
            early_stop_infeas_frac=early_stop_infeas_frac,
        )
        dt = time.time() - t0
        verdict['solve_time'] = dt
        verdict['target'] = target
        return verdict
    except Exception as e:
        return {
            'status': f'EXCEPTION:{type(e).__name__}',
            'error_msg': str(e),
            'is_feasible_status': False,
        }
