"""Z/2-aware per-box Lasserre order-2 dual-Farkas SDP escalation.

This is a drop-in extension of `interval_bnb.bound_sdp_escalation` that
applies Z/2 symmetry block-diagonalisation when the box is sigma-symmetric.

================================================================================
SOUNDNESS
================================================================================

Let sigma be the bin-reversal involution mu -> (mu_{d-1}, ..., mu_0). The
ambient Sidon problem is sigma-equivariant: the window set {M_W} satisfies
{M_{sigma(W)}} = {M_W} as a set (sigma((ell, s)) = (ell, 2d - ell - s) is
also a window), and the simplex constraint sum mu = 1 is sigma-invariant.

A BOX B = {mu : lo <= mu <= hi} is sigma-INVARIANT iff lo[i] = lo[d-1-i]
and hi[i] = hi[d-1-i] for every i (call this `is_box_sigma_symmetric`).

CASE 1: BOX IS sigma-SYMMETRIC
------------------------------
For a sigma-symmetric box B, the SDP relaxation problem
    "find y with M_2(y) >= 0, M_2((mu_i - lo_i) y) >= 0,
            M_2((hi_i - mu_i) y) >= 0,
            t * M_2(y) - Q_W(y) >= 0  for all W,
            sum-to-one consistency, normalisation y_0 = 1"
is sigma-equivariant. So if y is feasible, so is sigma(y) defined by
y_alpha -> y_{sigma(alpha)} (sigma acts on multi-indices by reversal):
    - M_2(y) -> M_2(sigma(y)) = Pi M_2(y) Pi^T (Pi the basis-perm),
      still PSD;
    - M_2((mu_i - lo_i) y) -> M_2((mu_{d-1-i} - lo_{d-1-i}) sigma(y))
      = M_2((mu_{d-1-i} - lo_{d-1-i}) y_after) (using lo_i = lo_{d-1-i}
      since the box is sigma-symmetric), still in the localizing list;
    - same for upper-box and window cones.

Hence (y + sigma(y)) / 2 is also feasible (PSD cone is convex), and is
sigma-INVARIANT (y_alpha = y_{sigma(alpha)}). So the SDP feasible set is
preserved by averaging over sigma-orbits, and we may RESTRICT to
sigma-invariant y without changing optimal value.

Under sigma-invariant y:
  (a) Moment matrix M_2(y) block-diagonalises into M_sym + M_anti, both
      affine in the canonical (sigma-orbit-representative) coordinates
      tilde-y. Lossless reformulation -- same feasible set, smaller PSD
      cones (n_basis^3 -> 2 (n_basis/2)^3 ~= 4x cheaper Cholesky).

  (b) Localizing cones M_2((mu_i - lo_i) y) and M_2((mu_{sigma(i)} - lo_{sigma(i)}) y)
      become equivalent via permutation similarity by Pi. Keep one rep per
      sigma-orbit -- count drops from d to ceil(d/2).

  (c) Window cones t M_2(y) - Q_W(y) and t M_2(y) - Q_{sigma(W)}(y) become
      equivalent via Pi. Keep one rep per sigma-orbit -- count drops from
      |nontrivial_windows| to ~|nontrivial_windows|/2.

This matches the existing Z/2 machinery in `lasserre.dual_sdp` (full-simplex
case). We re-use:
  - `lasserre.z2_elim.canonicalize_z2(P)` to substitute y -> tilde-y
  - `lasserre.z2_blockdiag.build_blockdiag_picks(...)` for the M_sym/M_anti
    affine maps
  - `lasserre.z2_blockdiag.localizing_sigma_reps(d)` for the loc orbit reps
  - `lasserre.z2_blockdiag.window_sigma_reps(d, windows)` for the window
    orbit reps

CASE 2: BOX IS NOT sigma-SYMMETRIC
----------------------------------
The SDP is NOT sigma-equivariant: lo[i] != lo[d-1-i] means M_2((mu_i - lo_i) y)
maps under sigma to M_2((mu_{d-1-i} - lo_i) y), but lo_i is the WRONG endpoint
for index d-1-i in the localizing list. So sigma-averaging is NOT a
feasibility-preserving move, and restricting to sigma-invariant y CUTS the
feasible set -- giving a STRICTLY TIGHTER (smaller) val^Z2(B) >= val(B).

That tightening MIGHT still produce a valid Farkas certificate (val^Z2(B) > t
implies val(B) > t), but at the cost of POTENTIALLY MISSING infeasibility
certificates for boxes B where val(B) > t but val^Z2(B) <= t. So the Z/2
reduction on an asymmetric box is SOUND (no false infeas) but possibly
LOOSER (more false feas).

For the per-box BnB cascade where most residual boxes are asymmetric, we
default to FULL SDP for safety and runtime-detect sigma-symmetric boxes
to apply the speedup.

ALTERNATIVE (NOT IMPLEMENTED HERE): split asymmetric box B into B and
sigma(B), solve each with full SDP. Equivalent runtime, no Z/2 win.

ALTERNATIVE (NOT IMPLEMENTED HERE): build a sigma-symmetric SUPERSET
B' = convex_hull(B u sigma(B)) = {min(lo, sigma(lo)) <= mu <= max(hi, sigma(hi))}
and apply Z/2. This gives a WEAKER bound (B' is larger than B), so a Farkas
cert for B' is a Farkas cert for B (sound). But typically val(B') < val(B)
so this LOSES tightness. Skipped.

================================================================================
"""
from __future__ import annotations

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
from .bound_sdp_escalation import (
    _build_dual_task_box, update_task_box, build_sdp_escalation_cache,
    _import_dual_sdp_api,
)


__all__ = [
    'is_box_sigma_symmetric',
    'symmetrize_box_outer',
    'build_sdp_escalation_cache_z2',
    '_build_dual_task_box_z2',
    'update_task_box_z2',
    'bound_sdp_escalation_z2_int_ge',
    'bound_sdp_escalation_z2_lb_float',
]


# ---------------------------------------------------------------------
# sigma-symmetry detection
# ---------------------------------------------------------------------

def is_box_sigma_symmetric(lo: Sequence[float], hi: Sequence[float],
                            tol: float = 0.0) -> bool:
    """Return True iff the box {lo <= mu <= hi} is invariant under sigma.

    sigma(mu)_i = mu_{d-1-i}. The box is sigma-invariant iff
    lo[i] == lo[d-1-i] and hi[i] == hi[d-1-i] for all i.

    `tol` is a tolerance in float endpoints (default 0 = exact match).
    For integer endpoints, pass `int_lo, int_hi` and tol=0.
    """
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    d = lo.size
    if hi.size != d:
        return False
    for i in range(d):
        j = d - 1 - i
        if abs(lo[i] - lo[j]) > tol:
            return False
        if abs(hi[i] - hi[j]) > tol:
            return False
    return True


def symmetrize_box_outer(lo: np.ndarray, hi: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Outer sigma-symmetrisation: B' = {min(lo, sigma(lo)) <= mu <= max(hi, sigma(hi))}.

    B' is sigma-symmetric and contains both B and sigma(B). Useful only as
    an optional "weaker bound but Z/2-eligible" fallback (NOT used by default
    in this module since it costs tightness).
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    lo_sym = np.minimum(lo, lo[::-1])
    hi_sym = np.maximum(hi, hi[::-1])
    return lo_sym, hi_sym


# ---------------------------------------------------------------------
# Z/2-reduced cache build
# ---------------------------------------------------------------------

def build_sdp_escalation_cache_z2(d: int, windows=None,
                                    target: float = 1.281,
                                    verbose: bool = False) -> dict:
    """Build a Z/2-aware cache containing BOTH:
      - the original full-SDP task (for asymmetric boxes), and
      - a Z/2-reduced task built from the sigma-symmetric placeholder
        box [0, 1] (which IS sigma-symmetric), used for sigma-symmetric
        boxes.

    The Z/2 task uses:
      * canonicalize_z2(P) to substitute y -> tilde-y (no sigma equalities)
      * moment block-diag (M_sym, M_anti)
      * sigma-rep-only localising cones (one per orbit)
      * sigma-rep-only window cones (one per orbit)

    Per-box dispatch in `bound_sdp_escalation_z2_int_ge` chooses which task
    to update_+ solve based on box symmetry.
    """
    if windows is None:
        from interval_bnb.windows import build_windows
        windows = build_windows(d)

    # Full (always-safe) cache.
    full_cache = build_sdp_escalation_cache(
        d, windows, target=target, verbose=verbose)

    # Z/2-reduced cache.
    api = _import_dual_sdp_api()
    from lasserre.z2_elim import canonicalize_z2
    from lasserre.z2_blockdiag import (
        build_blockdiag_picks, localizing_sigma_reps, window_sigma_reps,
    )

    P = api['_precompute'](d, order=2, verbose=verbose, lazy_ab_eiej=True)
    P_canon = canonicalize_z2(P, verbose=False)

    # Build moment block-diag picks.
    blockdiag = build_blockdiag_picks(
        P_canon['basis'], P_canon['idx'], P_canon['n_y'])

    # sigma-reps for localisers and windows.
    loc_fixed, loc_pairs = localizing_sigma_reps(d)
    active_loc_z2 = list(loc_fixed) + [p for (p, _) in loc_pairs]

    win_list = list(P_canon['windows'])
    win_fixed, win_pairs = window_sigma_reps(d, win_list)
    nontrivial_set = set(int(w) for w in P_canon['nontrivial_windows'])
    # Restrict to sigma-reps that ARE nontrivial.
    active_windows_z2 = [w for w in (list(win_fixed) +
                                      [p for (p, _) in win_pairs])
                         if int(w) in nontrivial_set]

    # Build the Z/2-reduced task with placeholder symmetric box [0, 1].
    lo0 = np.zeros(d, dtype=np.float64)
    hi0 = np.ones(d, dtype=np.float64)
    env_z2 = mosek.Env()
    task_z2, info_z2 = _build_dual_task_box_z2(
        P_canon, lo0, hi0, t_val=float(target), env=env_z2,
        z2_blockdiag_map=blockdiag,
        active_loc=active_loc_z2,
        active_windows=active_windows_z2,
        verbose=verbose,
    )

    return {
        'd': d,
        'order': 2,
        'target': float(target),
        'windows_id': id(windows),
        # Full SDP (for asymmetric boxes).
        'P': full_cache['P'],
        'env': full_cache['env'],
        'task': full_cache['task'],
        'info': full_cache['info'],
        # Z/2-reduced SDP (for sigma-symmetric boxes).
        'P_canon': P_canon,
        'env_z2': env_z2,
        'task_z2': task_z2,
        'info_z2': info_z2,
        'blockdiag_z2': blockdiag,
        'active_loc_z2': active_loc_z2,
        'active_windows_z2': active_windows_z2,
        # Sizes for diagnostics.
        'n_loc_full': d,
        'n_loc_z2': len(active_loc_z2),
        'n_win_full': len(P['nontrivial_windows']),
        'n_win_z2': len(active_windows_z2),
        'n_basis_full': int(P['n_basis']),
        'n_basis_sym_z2': int(blockdiag['n_sym']),
        'n_basis_anti_z2': int(blockdiag['n_anti']),
    }


# ---------------------------------------------------------------------
# Z/2 dual-task builder (extends _build_dual_task_box with block-diag moment +
# sigma-rep-restricted localisers/windows). Built on canonicalized P so y is
# tilde-y (n_y_canon < n_y_orig).
# ---------------------------------------------------------------------

def _build_dual_task_box_z2(
    P_canon: Dict[str, Any], lo: np.ndarray, hi: np.ndarray, t_val: float,
    env: mosek.Env, *,
    z2_blockdiag_map: Dict[str, Any],
    active_loc: List[int],
    active_windows: List[int],
    lambda_upper_bound: float = 1.0,
    verbose: bool = False,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Z/2-reduced dual-Farkas LP builder for a sigma-symmetric box.

    Differences from `_build_dual_task_box`:
      - `P_canon` is the canonicalize_z2 output (y -> tilde-y).
      - Moment cone is replaced by two BAR blocks (M_sym, M_anti)
        encoded via T_sym, T_anti from `z2_blockdiag_map`.
      - `active_loc` selects localising indices (orbit reps).
      - `active_windows` selects window positions (orbit reps).

    PRECONDITION: lo and hi must be sigma-symmetric (lo[i] == lo[d-1-i] and
    same for hi). The caller (`bound_sdp_escalation_z2_int_ge`) checks this.
    Without sigma-symmetry, the Z/2-reduced SDP is a STRICT TIGHTENING of the
    original (smaller feasible set), giving sound but potentially looser
    Farkas verdicts.
    """
    api = _import_dual_sdp_api()
    _hash_monos = api['_hash_monos']
    _alpha_lookup = api['_alpha_lookup']
    _aggregate_bar_triplet = api['_aggregate_bar_triplet']
    _aggregate_scalar_triplet = api['_aggregate_scalar_triplet']
    from lasserre.core import _hash_add

    d = int(P_canon['d'])
    n_y = int(P_canon['n_y'])
    basis = P_canon['basis']
    n_basis = int(P_canon['n_basis'])
    loc_basis = P_canon['loc_basis']
    n_loc = int(P_canon['n_loc'])
    mono_idx = P_canon['idx']
    M_mats = P_canon['M_mats']
    bases_arr = np.asarray(P_canon['bases'], dtype=np.int64)
    prime = P_canon.get('prime')
    sorted_h = np.asarray(P_canon['sorted_h'])
    sort_o = np.asarray(P_canon['sort_o'])
    consist_mono = P_canon['consist_mono']
    consist_idx = np.asarray(P_canon['consist_idx'], dtype=np.int64)
    consist_ei_idx = np.asarray(P_canon['consist_ei_idx'], dtype=np.int64)
    old_to_new_arr = P_canon.get('old_to_new')
    if old_to_new_arr is not None:
        old_to_new_arr = np.asarray(old_to_new_arr, dtype=np.int64)

    if old_to_new_arr is None:
        raise ValueError(
            "Z/2 build requires P canonicalized via canonicalize_z2(P).")

    if n_loc == 0:
        raise RuntimeError("Order-2 Lasserre requires n_loc > 0")

    n_sym = int(z2_blockdiag_map['n_sym'])
    n_anti = int(z2_blockdiag_map['n_anti'])

    task = env.Task()
    if verbose:
        task.set_Stream(mosek.streamtype.log, lambda s: print(s, end=''))

    t0 = time.time()

    # ----- Bar-variable layout -----
    # [moment X_0_sym, X_0_anti] + [lower-box X_lo_i for i in active_loc]
    # + [upper-box X_hi_i for i in active_loc] + [window X_W for W in active_windows]
    bar_sizes: List[int] = [n_sym]
    moment_bar_sym_id = 0
    if n_anti > 0:
        bar_sizes.append(n_anti)
        moment_bar_anti_id = 1
        moment_offset = 2
    else:
        moment_bar_anti_id = -1
        moment_offset = 1

    lo_bar_start = len(bar_sizes)
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

    # ----- Scalar variables: [lambda | mu_k | v_alpha] -----
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

    # ----- Constraint rows: one per alpha (stationarity = 0) -----
    task.appendcons(n_y)
    task.putconboundslice(
        0, n_y, [mosek.boundkey.fx] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.zeros(n_y, dtype=np.float64),
    )

    # ----- Scalar coefficients -----
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
    bar_tcoef_list: List[np.ndarray] = []
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

    # ---- Moment cone (sym + anti blocks via T_sym, T_anti) ----
    import scipy.sparse as sp
    T_sym = z2_blockdiag_map['T_sym'].tocoo()
    if T_sym.nnz:
        u = T_sym.row // n_sym
        v = T_sym.row % n_sym
        alpha_col = T_sym.col
        val = T_sym.data
        mask = u >= v
        n_m = int(mask.sum())
        if n_m:
            _append(
                alpha_col[mask].astype(np.int32),
                np.full(n_m, moment_bar_sym_id, dtype=np.int32),
                u[mask].astype(np.int32), v[mask].astype(np.int32),
                val[mask].astype(np.float64),
            )
    if n_anti > 0:
        T_anti = z2_blockdiag_map['T_anti'].tocoo()
        if T_anti.nnz:
            u = T_anti.row // n_anti
            v = T_anti.row % n_anti
            alpha_col = T_anti.col
            val = T_anti.data
            mask = u >= v
            n_m = int(mask.sum())
            if n_m:
                _append(
                    alpha_col[mask].astype(np.int32),
                    np.full(n_m, moment_bar_anti_id, dtype=np.int32),
                    u[mask].astype(np.int32), v[mask].astype(np.int32),
                    val[mask].astype(np.float64),
                )

    # ---- Localising prep ----
    L_arr = np.asarray(loc_basis, dtype=np.int64)
    L_hash = _hash_monos(L_arr, bases_arr, prime)
    ks_l, ls_l = np.tril_indices(n_loc)
    base_hash_loc = _hash_add(L_hash[ks_l], L_hash[ls_l], prime)
    alpha_idx_loc0 = _alpha_lookup(
        base_hash_loc, sorted_h, sort_o, old_to_new_arr)

    # ---- LOWER BOX cones (one per i in active_loc) ----
    for j, i in enumerate(active_loc):
        bar_idx_here = lo_bar_start + j
        # +1 part (mu_i term)
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
        # -lo_i part
        mask0 = alpha_idx_loc0 >= 0
        n_m0 = int(mask0.sum())
        if n_m0:
            _append(
                alpha_idx_loc0[mask0],
                np.full(n_m0, bar_idx_here, dtype=np.int32),
                ks_l[mask0], ls_l[mask0],
                np.zeros(n_m0, dtype=np.float64),
                lo_idx=np.full(n_m0, i, dtype=np.int32),
                lo_coef=np.full(n_m0, -1.0, dtype=np.float64),
            )

    # ---- UPPER BOX cones (one per i in active_loc) ----
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
                np.zeros(n_m0, dtype=np.float64),
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

    # ---- Window cones ----
    for w_j, w in enumerate(active_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_bar_start + w_j

        # (a) t-part
        mask_t = alpha_idx_loc0 >= 0
        n_m = int(mask_t.sum())
        if n_m:
            _append(
                alpha_idx_loc0[mask_t],
                np.full(n_m, W_bar_idx, dtype=np.int32),
                ks_l[mask_t], ls_l[mask_t],
                np.zeros(n_m, dtype=np.float64),
                tcoefs=np.full(n_m, +1.0, dtype=np.float64),
            )

        # (b) -Q-part
        if len(nz_i) > 0:
            n_pairs = len(nz_i)
            shifts = _hash_add(bases_arr[nz_i], bases_arr[nz_j], prime)
            mw_vals = Mw[nz_i, nz_j]
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
        first_idx = np.where(is_first)[0]
        all_lo_idx = np.maximum.reduceat(s_lo_idx, first_idx).astype(np.int32)
        all_hi_idx = np.maximum.reduceat(s_hi_idx, first_idx).astype(np.int32)

        keep = ((np.abs(all_static) > 0.0) | (np.abs(all_tcoef) > 0.0)
                | (np.abs(all_lo_coef) > 0.0) | (np.abs(all_hi_coef) > 0.0))
        if not np.all(keep):
            all_subi = all_subi[keep]; all_subj = all_subj[keep]
            all_subk = all_subk[keep]; all_subl = all_subl[keep]
            all_static = all_static[keep]; all_tcoef = all_tcoef[keep]
            all_lo_idx = all_lo_idx[keep]; all_lo_coef = all_lo_coef[keep]
            all_hi_idx = all_hi_idx[keep]; all_hi_coef = all_hi_coef[keep]

        n_bar_entries = int(all_subi.size)
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
        'is_z2': True,
        'n_sym_z2': n_sym, 'n_anti_z2': n_anti,
        '_all_subi': all_subi, '_all_subj': all_subj,
        '_all_subk': all_subk, '_all_subl': all_subl,
        '_all_static': all_static, '_all_tcoef': all_tcoef,
        '_all_lo_idx': all_lo_idx, '_all_lo_coef': all_lo_coef,
        '_all_hi_idx': all_hi_idx, '_all_hi_coef': all_hi_coef,
    }

    if verbose:
        print(f"  [box-dual-Z2] n_bar={n_bar} (sym={n_sym} anti={n_anti}) "
              f"n_scalar={n_scalar:,} n_cons={n_y:,} "
              f"n_bar_entries={n_bar_entries:,} t={t_val:.4f} "
              f"build={build_time:.2f}s "
              f"(loc={len(active_loc)} win={len(active_windows)})",
              flush=True)

    return task, info


# Z/2 update is identical structurally to update_task_box -- coefficients
# carry the same lo/hi/t triplets, just over a different (canonical) y space.
update_task_box_z2 = update_task_box


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def bound_sdp_escalation_z2_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int], windows, d: int,
    target_num: int, target_den: int,
    cache: Optional[dict] = None,
    *,
    order: int = 2, bandwidth: Optional[int] = None,
    time_limit_s: float = 30.0,
    tol_gap_abs: float = 1e-7, tol_gap_rel: float = 1e-7, tol_feas: float = 1e-7,
    n_threads: int = 1,
    return_diagnostic: bool = False,
    force_z2: bool = False,
    force_full: bool = False,
):
    """Z/2-aware per-box dual-Farkas SDP cert.

    Behaviour:
      - If the integer box is sigma-symmetric (lo_int[i]==lo_int[d-1-i],
        same for hi_int) AND not `force_full`: use the Z/2-reduced task.
      - Otherwise (or `force_full`): use the full SDP task.
      - `force_z2=True` overrides safety: applies Z/2 even on asymmetric
        boxes. Sound (no false infeas) but possibly looser (more false feas).

    Returns: True/False, or (True/False, diagnostic) dict if return_diagnostic.
    """
    # Empty box vacuous cert.
    if int(sum(lo_int)) > _SCALE or int(sum(hi_int)) < _SCALE:
        if return_diagnostic:
            return True, {'status': 'EMPTY', 'cert_via': 'vacuous_empty'}
        return True

    target_f = float(target_num) / float(target_den)
    lo = np.asarray([float(li) / _SCALE for li in lo_int], dtype=np.float64)
    hi = np.asarray([float(hv) / _SCALE for hv in hi_int], dtype=np.float64)

    # Symmetry check on INTEGER endpoints (exact).
    box_sym = is_box_sigma_symmetric(lo_int, hi_int, tol=0)

    if cache is None:
        cache = build_sdp_escalation_cache_z2(d, windows, target=target_f)

    api = _import_dual_sdp_api()

    use_z2 = (box_sym and not force_full) or force_z2
    if use_z2:
        task = cache['task_z2']
        info = cache['info_z2']
    else:
        task = cache['task']
        info = cache['info']

    # Update box and t.
    if cache['target'] != target_f:
        update_task_box(task, info, lo, hi, target_f)
        cache['target'] = target_f
    else:
        update_task_box(task, info, lo, hi, target_f)

    # Solver tolerances.
    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, float(tol_gap_rel))
    task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit_s))
    if n_threads > 0:
        task.putintparam(mosek.iparam.num_threads, int(n_threads))

    try:
        verdict = api['solve_dual_task'](task, info, verbose=False)
    except Exception as e:
        if return_diagnostic:
            return False, {
                'status': f'EXCEPTION:{type(e).__name__}',
                'error_msg': str(e), 'cert_via': 'solver_failure',
                'box_sigma_symmetric': box_sym, 'used_z2': use_z2,
            }
        return False

    cert = bool(verdict.get('verdict') == 'infeas')

    if return_diagnostic:
        return cert, {
            **verdict,
            'target_f': target_f,
            'cert_via': 'farkas_infeasibility',
            'box_sigma_symmetric': box_sym,
            'used_z2': use_z2,
        }
    return cert


def bound_sdp_escalation_z2_lb_float(
    lo: np.ndarray, hi: np.ndarray, windows, d: int,
    *,
    cache: Optional[dict] = None,
    order: int = 2, bandwidth: Optional[int] = None,
    target: float = 1.281,
    time_limit_s: float = 30.0,
    tol_gap_abs: float = 1e-7, tol_gap_rel: float = 1e-7, tol_feas: float = 1e-7,
    n_threads: int = 1,
    force_z2: bool = False,
    force_full: bool = False,
) -> dict:
    """Float diagnostic. Returns verdict dict + Z/2 metadata."""
    api = _import_dual_sdp_api()
    if cache is None:
        cache = build_sdp_escalation_cache_z2(d, windows, target=target)

    box_sym = is_box_sigma_symmetric(lo, hi, tol=1e-12)
    use_z2 = (box_sym and not force_full) or force_z2

    if use_z2:
        task = cache['task_z2']
        info = cache['info_z2']
    else:
        task = cache['task']
        info = cache['info']

    if cache['target'] != target:
        update_task_box(task, info,
                         np.asarray(lo, dtype=np.float64),
                         np.asarray(hi, dtype=np.float64), target)
        cache['target'] = target
    else:
        update_task_box(task, info,
                         np.asarray(lo, dtype=np.float64),
                         np.asarray(hi, dtype=np.float64), target)

    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, float(tol_feas))
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, float(tol_gap_rel))
    task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit_s))
    if n_threads > 0:
        task.putintparam(mosek.iparam.num_threads, int(n_threads))
    try:
        t0 = time.time()
        verdict = api['solve_dual_task'](task, info, verbose=False)
        dt = time.time() - t0
        verdict['solve_time'] = dt
        verdict['target'] = target
        verdict['box_sigma_symmetric'] = box_sym
        verdict['used_z2'] = use_z2
        return verdict
    except Exception as e:
        return {
            'status': f'EXCEPTION:{type(e).__name__}',
            'error_msg': str(e),
            'is_feasible_status': False,
            'box_sigma_symmetric': box_sym, 'used_z2': use_z2,
        }
