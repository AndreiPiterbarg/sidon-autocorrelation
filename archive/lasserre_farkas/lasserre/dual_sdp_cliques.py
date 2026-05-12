"""Clique-decomposed dual Lasserre SDP (Task API, Farkas form).

================================================================================
WHAT THIS FILE IS
================================================================================

This is the correlative-sparsity analogue of ``lasserre/dual_sdp.py``.  The
monolithic moment PSD cone ``M_k(y) ⪰ 0`` (size ``n_basis × n_basis``) is
replaced by K smaller clique-restricted moment cones ``M_k^{I_c}(y) ⪰ 0``
where the clique bases ``B_c = { α : |α| ≤ k, supp(α) ⊆ I_c }`` partition
the set of monomials of interest.  For d=16 / order=3 / bandwidth=8 this
reduces the dominant bar size from 969×969 to 9 bars of size 165×165 —
roughly a ``(969/165)^4 / 9 ≈ 130×`` reduction in MOSEK IPM Hessian
storage (which scales as ``sym(N)^2 = N^4 / 4`` per bar).

Mathematical validity:
  * Each ``M_k^{I_c}(y) ⪰ 0`` is a necessary condition for the full
    ``M_k(y) ⪰ 0``.
  * The moment vector ``y`` is shared across all cliques; entries of
    different clique cones that reference the same ``α = β + γ`` are
    automatically linked through the shared scalar equality rows.
  * Under the running-intersection property (RIP) of the chordal
    extension of the banded coupling graph — satisfied by the overlapping
    cliques ``I_c = {c, c+1, ..., c+b}`` — the relaxation produces a
    lower bound
          val_L^clique(d, L=3, b)  ≤  val_L^full(d, L=3)  ≤  val(d)
    with equality (modulo IPM tolerance) when b ≥ d−1.

Localizing cones are built on a per-clique basis: variable i is assigned
to the clique whose centre is closest to i (matches the Fusion path in
``lasserre.cliques._add_sparse_localizing_constraints``), then
``L_i(y) = M_{k-1}^{I_c}(e_i · y)`` is a single PSD bar of size
``n_cb_{order-1}``.

Window cones: for each active window W, find a clique I_c that covers
``support(M_W)``.  If one exists, use that clique's order-1 basis; if
none exists, fall back to the full localising basis from ``P['loc_basis']``
(correctness-preserving — never skip a window).

================================================================================
V1 SCOPE
================================================================================

  * NO Z/2 canonicalisation / block-diagonalisation.  Pass a non-
    canonicalised precompute ``P_raw``.
  * Moment cone: K clique bars (no sym / anti split).
  * Localising cones X_i: per-clique, as above.
  * Optional upper-localising cones X'_i (same clique as X_i).
  * Window cones X_W: per-clique when a covering clique exists, else
    fallback to the full order-1 basis (= monolithic localising cone).

Everything else — scalar equality rows, λ / μ / v_α scalar layout,
objective, verdict semantics — is identical to ``build_dual_task``.

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

from lasserre_fusion import _hash_monos, _hash_lookup  # noqa: E402
from lasserre.cliques import _build_banded_cliques, _build_clique_basis  # noqa: E402

# Reuse the aggregation helpers from the monolithic builder — identical
# semantics, no clique-specific logic.
from lasserre.dual_sdp import (  # noqa: E402
    _aggregate_bar_triplet, _aggregate_scalar_triplet,
)


__all__ = [
    'build_dual_task_cliques',
]


# =========================================================================
# Top-level builder
# =========================================================================

def build_dual_task_cliques(
    P: Dict[str, Any],
    t_val: float,
    env: mosek.Env,
    *,
    bandwidth: int,
    include_upper_loc: bool = False,
    active_loc: Optional[List[int]] = None,
    active_windows: Optional[List[int]] = None,
    lambda_upper_bound: float = 1.0,
    verbose: bool = True,
    cache_for_reuse: bool = True,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Build the clique-decomposed Farkas-infeasibility dual SDP.

    Parameters
    ----------
    P                  : precompute dict from ``lasserre_scalable._precompute``.
                         MUST NOT be canonicalised via ``canonicalize_z2``
                         (v1 is Z/2-unaware — the builder will refuse a
                         canonicalised P).
    t_val              : value of the bisection parameter t baked into the
                         window sensitivity matrices.
    env                : MOSEK environment.
    bandwidth          : REQUIRED.  Clique width b ≥ order.  Each clique
                         has d_loc = b+1 consecutive variables.  Setting
                         b = d-1 reproduces the monolithic bound.
    include_upper_loc  : add X'_i cones for (1 − μ_i) ≥ 0.
    active_loc         : which i ∈ {0, ..., d-1} get localising cones
                         (default: all).
    active_windows     : which windows get window cones (default:
                         ``P['nontrivial_windows']``).
    lambda_upper_bound : cap on λ.
    cache_for_reuse    : when False, skip caching the bulk bar-triplet
                         arrays in ``info`` (saves memory).  The clique
                         driver does NOT use ``update_task_t`` — it
                         rebuilds per probe — so this defaults to True
                         only for unit-test inspection.

    Returns
    -------
    task : mosek.Task with the Farkas LP fully encoded (max λ).
    info : dict with bar sizes, per-clique bar IDs, fallback counts, etc.
    """
    d = int(P['d'])
    order = int(P['order'])
    n_y = int(P['n_y'])
    loc_basis = P['loc_basis']
    n_loc = int(P['n_loc'])
    mono_idx = P['idx']
    M_mats = P['M_mats']
    bases_arr = np.asarray(P['bases'], dtype=np.int64)
    sorted_h = np.asarray(P['sorted_h'])
    sort_o = np.asarray(P['sort_o'])
    consist_mono = P['consist_mono']
    consist_idx = np.asarray(P['consist_idx'], dtype=np.int64)
    consist_ei_idx = np.asarray(P['consist_ei_idx'], dtype=np.int64)

    if P.get('old_to_new') is not None:
        raise NotImplementedError(
            "build_dual_task_cliques: v1 does not compose with Z/2 "
            "canonicalisation (old_to_new is present in P).  Call with a "
            "non-canonicalised precompute.")

    if bandwidth < order:
        raise ValueError(
            f"bandwidth ({bandwidth}) must be ≥ order ({order}) — "
            f"otherwise the clique basis degenerates.")
    if bandwidth > d - 1:
        raise ValueError(
            f"bandwidth ({bandwidth}) must be ≤ d-1 ({d - 1}) — "
            f"use the monolithic build_dual_task instead.")

    if n_loc == 0:
        active_loc = []
        active_windows = []
    else:
        if active_loc is None:
            active_loc = list(range(d))
        if active_windows is None:
            active_windows = list(P['nontrivial_windows'])

    task = env.Task()

    if verbose:
        task.set_Stream(mosek.streamtype.log, lambda s: print(s, end=''))

    t0 = time.time()

    # -----------------------------------------------------------------
    # 1. Build cliques + per-clique bases.
    # -----------------------------------------------------------------
    cliques = _build_banded_cliques(d, bandwidth)
    K = len(cliques)

    # Per-clique moment basis (degree ≤ order) + index-into-y array.
    clique_basis_arr: List[np.ndarray] = []
    clique_basis_idx: List[np.ndarray] = []
    n_cb_per_clique: List[int] = []
    for c_idx, clique in enumerate(cliques):
        B_c = _build_clique_basis(clique, order, d)  # (n_cb, d)
        n_cb = int(B_c.shape[0])
        idx_c = np.empty(n_cb, dtype=np.int64)
        for r in range(n_cb):
            key = tuple(int(x) for x in B_c[r].tolist())
            iy = mono_idx.get(key, -1)
            idx_c[r] = int(iy)
        if np.any(idx_c < 0):
            bad = int(np.where(idx_c < 0)[0][0])
            raise ValueError(
                f"clique {c_idx} basis row {bad} (monomial "
                f"{tuple(B_c[bad].tolist())}) missing from P['idx'] — "
                f"precompute is inconsistent.")
        clique_basis_arr.append(B_c)
        clique_basis_idx.append(idx_c)
        n_cb_per_clique.append(n_cb)

    if verbose:
        print(f"  [cliques] d={d} order={order} bandwidth={bandwidth}  "
              f"K={K} cliques, n_cb={n_cb_per_clique}", flush=True)

    # Per-clique localising basis (degree ≤ order−1), only when order ≥ 2.
    clique_loc_basis_arr: List[np.ndarray] = []
    if order >= 2:
        for c_idx, clique in enumerate(cliques):
            L_c = _build_clique_basis(clique, order - 1, d)
            clique_loc_basis_arr.append(L_c)
    else:
        clique_loc_basis_arr = [np.zeros((0, d), dtype=np.int64)
                                for _ in range(K)]

    # bin → clique assignment (matches lasserre.cliques._add_sparse_
    # localizing_constraints): clique whose centre is closest to i.
    bin_to_clique: Dict[int, int] = {}
    for c_idx, clique in enumerate(cliques):
        mid = (clique[0] + clique[-1]) / 2.0
        for i_var in clique:
            dist = abs(i_var - mid)
            if (i_var not in bin_to_clique
                    or dist < bin_to_clique[i_var][1]):
                bin_to_clique[i_var] = (c_idx, dist)
    bin_to_clique = {i: v[0] for i, v in bin_to_clique.items()}

    # -----------------------------------------------------------------
    # 2. Bar-variable layout.
    #
    #   [moment clique bars]   K bars, size n_cb_per_clique[c]
    #   [loc_i]                one bar per i in active_loc,
    #                          size n_cb_loc_per_clique[c_i]
    #   [uloc_i] (optional)    one bar per i in active_loc,
    #                          size n_cb_loc_per_clique[c_i]
    #   [win_W]                one bar per W in active_windows,
    #                          size = covering clique's loc basis, OR
    #                                 fallback n_loc
    # -----------------------------------------------------------------
    bar_sizes: List[int] = []

    # Moment clique bars.
    moment_bar_ids: List[int] = []
    for c_idx in range(K):
        moment_bar_ids.append(len(bar_sizes))
        bar_sizes.append(n_cb_per_clique[c_idx])

    # Localising bars: pick clique per variable, use that clique's
    # loc-basis size.  Record per-i metadata.
    loc_bar_start = len(bar_sizes)
    loc_info: List[Tuple[int, int]] = []  # (i_var, clique_idx)
    for i_var in active_loc:
        c_i = bin_to_clique.get(int(i_var), 0)
        bar_sizes.append(int(clique_loc_basis_arr[c_i].shape[0]))
        loc_info.append((int(i_var), int(c_i)))
    loc_bar_end = len(bar_sizes)

    uloc_bar_start = loc_bar_end
    if include_upper_loc:
        for (i_var, c_i) in loc_info:
            bar_sizes.append(int(clique_loc_basis_arr[c_i].shape[0]))
    uloc_bar_end = len(bar_sizes)

    # Window bars: per-window cover clique discovery, else fallback.
    # We do this eagerly to know the bar sizes before appendbarvars.
    win_bar_start = uloc_bar_end
    # Per window: (cov_clique_idx or -1 for fallback, bar_size)
    win_cover: List[Tuple[int, int]] = []
    n_fallback_windows = 0
    for w in active_windows:
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        if nz_i.size == 0:
            # Trivial window: still allocate a bar of size 0 is illegal;
            # fall back to full loc basis (rare — these are filtered).
            win_cover.append((-1, n_loc))
            bar_sizes.append(n_loc)
            n_fallback_windows += 1
            continue
        active_bins = set(nz_i.tolist()) | set(nz_j.tolist())
        cov = -1
        for c_idx, clique in enumerate(cliques):
            if active_bins.issubset(set(clique)):
                cov = c_idx
                break
        if cov >= 0:
            size = int(clique_loc_basis_arr[cov].shape[0])
            win_cover.append((cov, size))
            bar_sizes.append(size)
        else:
            win_cover.append((-1, n_loc))
            bar_sizes.append(n_loc)
            n_fallback_windows += 1
            if verbose:
                print(f"[cliques] window {w} has no covering clique; "
                      f"falling back to full loc basis (n_loc={n_loc})",
                      flush=True)
    win_bar_end = len(bar_sizes)

    n_bar = len(bar_sizes)
    if n_bar == 0:
        raise RuntimeError("No bar variables allocated — check cliques / "
                           "active_loc / active_windows.")
    task.appendbarvars(bar_sizes)

    # -----------------------------------------------------------------
    # 3. Scalar variables [λ | μ_k (kept consist eqs) | v_α].
    # -----------------------------------------------------------------
    kept_k: List[int] = [
        k for k in range(len(consist_mono)) if int(consist_idx[k]) >= 0
    ]
    n_consist = len(kept_k)

    LAMBDA_IDX = 0
    MU_START = 1
    MU_END = 1 + n_consist
    V_START = MU_END
    V_END = V_START + n_y
    n_scalar = V_END

    task.appendvars(n_scalar)

    task.putvarbound(
        LAMBDA_IDX, mosek.boundkey.ra, 0.0, float(lambda_upper_bound))
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

    # -----------------------------------------------------------------
    # 4. Constraint rows: one per α (stationarity), all = 0.
    # -----------------------------------------------------------------
    task.appendcons(n_y)
    task.putconboundslice(
        0, n_y,
        [mosek.boundkey.fx] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.zeros(n_y, dtype=np.float64),
    )

    # -----------------------------------------------------------------
    # 5. Scalar coefficients (A-matrix).
    # -----------------------------------------------------------------
    scalar_rows: List[int] = []
    scalar_cols: List[int] = []
    scalar_vals: List[float] = []

    alpha_zero = tuple(0 for _ in range(d))
    if alpha_zero not in mono_idx:
        raise RuntimeError(
            "Zero monomial (0,...,0) missing from P['idx'].")
    alpha_zero_row = int(mono_idx[alpha_zero])
    scalar_rows.append(alpha_zero_row)
    scalar_cols.append(LAMBDA_IDX)
    scalar_vals.append(1.0)

    # μ_k consistency rows: identical to the monolithic path.
    for j, k in enumerate(kept_k):
        a_k = int(consist_idx[k])
        child = consist_ei_idx[k]
        coef_by_row: Dict[int, float] = {}
        for i_var in range(d):
            c = int(child[i_var])
            if c >= 0:
                coef_by_row[c] = coef_by_row.get(c, 0.0) + 1.0
        coef_by_row[a_k] = coef_by_row.get(a_k, 0.0) - 1.0
        col_j = MU_START + j
        for row, val in coef_by_row.items():
            if val != 0.0:
                scalar_rows.append(row)
                scalar_cols.append(col_j)
                scalar_vals.append(float(val))

    # v_α slacks: +1 on diagonal.
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

    # -----------------------------------------------------------------
    # 6. Bar-matrix sensitivity coefficients.
    # -----------------------------------------------------------------
    bar_subi_list: List[np.ndarray] = []
    bar_subj_list: List[np.ndarray] = []
    bar_subk_list: List[np.ndarray] = []
    bar_subl_list: List[np.ndarray] = []
    bar_val_list:  List[np.ndarray] = []
    bar_tcoef_list: List[np.ndarray] = []

    def _append(subi, subj, subk, subl, vals, tcoefs=None):
        if subi.size == 0:
            return
        bar_subi_list.append(np.ascontiguousarray(subi, dtype=np.int32))
        bar_subj_list.append(np.ascontiguousarray(subj, dtype=np.int32))
        bar_subk_list.append(np.ascontiguousarray(subk, dtype=np.int32))
        bar_subl_list.append(np.ascontiguousarray(subl, dtype=np.int32))
        bar_val_list.append(np.ascontiguousarray(vals, dtype=np.float64))
        if tcoefs is None:
            bar_tcoef_list.append(np.zeros(subi.size, dtype=np.float64))
        else:
            bar_tcoef_list.append(
                np.ascontiguousarray(tcoefs, dtype=np.float64))

    # ---- Per-clique moment cones ----
    for c_idx in range(K):
        B_c = clique_basis_arr[c_idx]  # (n_cb, d)
        n_cb = n_cb_per_clique[c_idx]
        B_hash = _hash_monos(B_c, bases_arr)
        ks_m, ls_m = np.tril_indices(n_cb)
        alpha_hash_m = B_hash[ks_m] + B_hash[ls_m]
        alpha_idx_m = _hash_lookup(alpha_hash_m, sorted_h, sort_o)
        if np.any(alpha_idx_m < 0):
            raise RuntimeError(
                f"Clique {c_idx} moment sensitivity lookup produced -1.")
        _append(
            alpha_idx_m,
            np.full(ks_m.shape, moment_bar_ids[c_idx], dtype=np.int32),
            ks_m, ls_m,
            np.full(ks_m.shape, +1.0, dtype=np.float64),
        )

    # ---- Per-variable localising cones X_i ----
    # Pre-compute per-clique loc hashes + lower-triangle indices.
    clique_loc_hash: List[np.ndarray] = []
    clique_loc_tri: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for c_idx in range(K):
        L_c = clique_loc_basis_arr[c_idx]
        n_cbl = int(L_c.shape[0])
        if n_cbl == 0:
            clique_loc_hash.append(np.zeros(0, dtype=np.int64))
            clique_loc_tri.append(
                (np.zeros(0, dtype=np.int64),
                 np.zeros(0, dtype=np.int64),
                 np.zeros(0, dtype=np.int64)))
            continue
        Lh = _hash_monos(L_c, bases_arr)
        ks, ls = np.tril_indices(n_cbl)
        base_hash = Lh[ks] + Lh[ls]  # α = loc[k]+loc[l]
        clique_loc_hash.append(Lh)
        clique_loc_tri.append((ks, ls, base_hash))

    # Also pre-compute the full (monolithic) loc-basis hash/tri for the
    # window fallback path.
    if n_loc > 0:
        L_full = np.asarray(loc_basis, dtype=np.int64)
        L_full_hash = _hash_monos(L_full, bases_arr)
        ks_full, ls_full = np.tril_indices(n_loc)
        base_full_hash = L_full_hash[ks_full] + L_full_hash[ls_full]
    else:
        L_full = np.zeros((0, d), dtype=np.int64)
        L_full_hash = np.zeros(0, dtype=np.int64)
        ks_full = np.zeros(0, dtype=np.int64)
        ls_full = np.zeros(0, dtype=np.int64)
        base_full_hash = np.zeros(0, dtype=np.int64)

    for j, (i_var, c_i) in enumerate(loc_info):
        bar_idx_here = loc_bar_start + j
        ks, ls, base_hash = clique_loc_tri[c_i]
        if ks.size == 0:
            continue
        alpha_hash_li = base_hash + bases_arr[i_var]
        alpha_idx_li = _hash_lookup(alpha_hash_li, sorted_h, sort_o)
        mask = alpha_idx_li >= 0
        n_m = int(mask.sum())
        if n_m == 0:
            continue
        _append(
            alpha_idx_li[mask],
            np.full(n_m, bar_idx_here, dtype=np.int32),
            ks[mask], ls[mask],
            np.full(n_m, +1.0, dtype=np.float64),
        )

    # ---- Upper-localising cones X'_i (optional) ----
    if include_upper_loc:
        for j, (i_var, c_i) in enumerate(loc_info):
            bar_idx_here = uloc_bar_start + j
            ks, ls, base_hash = clique_loc_tri[c_i]
            if ks.size == 0:
                continue
            # +1 on α = loc+loc
            alpha_idx_t = _hash_lookup(base_hash, sorted_h, sort_o)
            mask0 = alpha_idx_t >= 0
            if np.any(mask0):
                n_m = int(mask0.sum())
                _append(
                    alpha_idx_t[mask0],
                    np.full(n_m, bar_idx_here, dtype=np.int32),
                    ks[mask0], ls[mask0],
                    np.full(n_m, +1.0, dtype=np.float64),
                )
            # −1 on α = loc+loc+e_i
            alpha_hash_li = base_hash + bases_arr[i_var]
            alpha_idx_li = _hash_lookup(alpha_hash_li, sorted_h, sort_o)
            mask = alpha_idx_li >= 0
            n_m = int(mask.sum())
            if n_m == 0:
                continue
            _append(
                alpha_idx_li[mask],
                np.full(n_m, bar_idx_here, dtype=np.int32),
                ks[mask], ls[mask],
                np.full(n_m, -1.0, dtype=np.float64),
            )

    # ---- Window cones X_W: +t · E_W^t − E_W^Q ----
    for w_j, w in enumerate(active_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_bar_start + w_j

        cov_idx, _cov_size = win_cover[w_j]
        if cov_idx >= 0:
            ks, ls, base_hash = clique_loc_tri[cov_idx]
        else:
            # Fallback to full localising basis.
            ks, ls, base_hash = ks_full, ls_full, base_full_hash

        if ks.size == 0:
            continue

        # (a) t-part (+t on α = loc+loc), stored with val=0 + t_coef=+1.
        alpha_idx_t = _hash_lookup(base_hash, sorted_h, sort_o)
        mask_t = alpha_idx_t >= 0
        n_m = int(mask_t.sum())
        if n_m:
            _append(
                alpha_idx_t[mask_t],
                np.full(n_m, W_bar_idx, dtype=np.int32),
                ks[mask_t], ls[mask_t],
                np.zeros(n_m, dtype=np.float64),
                tcoefs=np.ones(n_m, dtype=np.float64),
            )

        # (b) Q-part: −M_W[ii, jj] on α = loc+loc+e_ii+e_jj, symmetric.
        for ii, jj in zip(nz_i.tolist(), nz_j.tolist()):
            if ii < jj:
                continue
            raw = float(Mw[ii, jj])
            if raw == 0.0:
                continue
            coef = -raw if ii == jj else -2.0 * raw
            alpha_hash_q = (base_hash
                            + bases_arr[ii] + bases_arr[jj])
            alpha_idx_q = _hash_lookup(alpha_hash_q, sorted_h, sort_o)
            mask = alpha_idx_q >= 0
            n_m = int(mask.sum())
            if n_m == 0:
                continue
            _append(
                alpha_idx_q[mask],
                np.full(n_m, W_bar_idx, dtype=np.int32),
                ks[mask], ls[mask],
                np.full(n_m, coef, dtype=np.float64),
            )

    # ---- Concatenate + aggregate + bulk submit ----
    if bar_subi_list:
        all_subi = np.concatenate(bar_subi_list)
        all_subj = np.concatenate(bar_subj_list)
        all_subk = np.concatenate(bar_subk_list)
        all_subl = np.concatenate(bar_subl_list)
        all_val = np.concatenate(bar_val_list)
        all_tcoef = np.concatenate(bar_tcoef_list)

        all_subi, all_subj, all_subk, all_subl, all_val, all_tcoef = \
            _aggregate_bar_triplet(
                all_subi, all_subj, all_subk, all_subl,
                all_val, all_tcoef)
        n_bar_entries = int(all_subi.size)
        if n_bar_entries:
            init_vals = all_val + float(t_val) * all_tcoef
            task.putbarablocktriplet(
                all_subi, all_subj, all_subk, all_subl, init_vals)
    else:
        n_bar_entries = 0
        all_subi = all_subj = all_subk = all_subl = None
        all_val = all_tcoef = None

    # -----------------------------------------------------------------
    # 7. Objective: maximize λ.
    # -----------------------------------------------------------------
    task.putobjsense(mosek.objsense.maximize)
    task.putcj(LAMBDA_IDX, 1.0)

    build_time = time.time() - t0

    info: Dict[str, Any] = {
        'build_time_s': build_time,
        'bar_sizes': bar_sizes,
        'n_bar': n_bar,
        'cliques': [list(c) for c in cliques],
        'bandwidth': int(bandwidth),
        'moment_bar_ids': list(moment_bar_ids),
        'n_moment_bars': K,
        'n_cb_per_clique': list(n_cb_per_clique),
        'loc_bar_start': loc_bar_start,
        'loc_bar_end': loc_bar_end,
        'uloc_bar_start': uloc_bar_start,
        'uloc_bar_end': uloc_bar_end,
        'win_bar_start': win_bar_start,
        'win_bar_end': win_bar_end,
        'active_loc': list(active_loc),
        'active_windows': list(active_windows),
        'n_scalar': n_scalar,
        'n_consist_kept': n_consist,
        'n_cons': n_y,
        'n_y': n_y,
        'n_bar_entries': n_bar_entries,
        'n_fallback_windows': int(n_fallback_windows),
        't_val': float(t_val),
        'LAMBDA_IDX': LAMBDA_IDX,
        'MU_START': MU_START,
        'V_START': V_START,
        'lambda_upper_bound': float(lambda_upper_bound),
        'z2_canonicalized': False,
        'z2_blockdiag': False,
        'include_upper_loc': bool(include_upper_loc),
    }

    if cache_for_reuse and bar_subi_list:
        info['_all_subi'] = all_subi
        info['_all_subj'] = all_subj
        info['_all_subk'] = all_subk
        info['_all_subl'] = all_subl
        info['_all_static'] = all_val
        info['_all_tcoef'] = all_tcoef
    else:
        info['_all_subi'] = None
        info['_all_subj'] = None
        info['_all_subk'] = None
        info['_all_subl'] = None
        info['_all_static'] = None
        info['_all_tcoef'] = None

    if verbose:
        print(f"  [dual-cliques] K={K}  n_bar={n_bar}  "
              f"n_scalar={n_scalar:,}  n_cons={n_y:,}  "
              f"n_bar_entries={n_bar_entries:,}  "
              f"n_fallback_windows={n_fallback_windows}  "
              f"t={t_val:.6f}  build={build_time:.2f}s  "
              f"(upper_loc={include_upper_loc} "
              f"n_loc_active={len(active_loc)} "
              f"n_win_active={len(active_windows)})",
              flush=True)

    return task, info
