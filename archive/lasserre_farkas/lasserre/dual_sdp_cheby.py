"""SOS-dual Lasserre SDP in Chebyshev basis (Task API, Farkas form).

================================================================================
WHAT THIS FILE IS
================================================================================

Analog of ``lasserre/dual_sdp.py`` but with every PSD cone sensitivity,
every scalar coupling, expressed in the SHIFTED-CHEBYSHEV BASIS
T_k*(μ) = T_k(2μ − 1) on [0,1].

Primal variables are the Chebyshev moments c_γ of the probability
measure μ on d atoms.  Related to monomial moments y_α via

    y_α = Σ_γ B[α, γ] · c_γ,

a triangular integer/rational change of basis from ``cheby_basis.build_B_matrix``.
Primal feasibility of y ⇔ feasibility of c — the feasible set and the
optimal bound are UNCHANGED.  Chebyshev entries |T_k*(μ)| ≤ 1 on [0,1]
give a well-conditioned parametrisation (monomial moments y_α scale as
1/d^|α|, hitting MOSEK's feasibility tolerance at large d; Chebyshev
does not).

================================================================================
DUAL STRUCTURE (derived from the Chebyshev primal)
================================================================================

Stationarity row per γ ∈ [0 .. n_y):

    [γ = 0]·λ
    + Σ_k μ_k · (B[α_k, γ] − Σ_i B[α_k + e_i, γ])
    + ⟨X_0, E_0^cheb[γ]⟩
    + Σ_i ⟨X_i, E_i^cheb[γ]⟩
    + Σ_i ⟨X'_i, E'_i^cheb[γ]⟩
    + t · Σ_W ⟨X_W, E_W^t,cheb[γ]⟩ − Σ_W ⟨X_W, E_W^Q,cheb[γ]⟩
    + v_γ  =  0.

Bar-sensitivity matrices are built with the cheby_basis product rules:

    E_0^cheb[γ]_{k,l}     =  (coef of c_γ in T_{basis[k]}* · T_{basis[l]}*)
    E_i^cheb[γ]_{k,l}     =  (coef of c_γ in μ_i · T_{loc[k]}* · T_{loc[l]}*)
    E'_i^cheb[γ]_{k,l}    =  E_t^cheb[γ]_{k,l} − E_i^cheb[γ]_{k,l}
    E_W^t,cheb[γ]_{k,l}   =  (coef of c_γ in T_{loc[k]}* · T_{loc[l]}*)
    E_W^Q,cheb[γ]_{k,l}   =  Σ_{i,j} M_W[i,j] · (coef of c_γ in
                              μ_i μ_j T_{loc[k]}* T_{loc[l]}*)

Each ``coef of c_γ in (...)`` is computed exactly via ``fractions.Fraction``
and converted to float64 only at the MOSEK-input boundary.

================================================================================
V1 SCOPE
================================================================================

  * NO Z/2 canonicalisation / blockdiag (orthogonal reduction, can be
    layered in later after the basis change is locked in).
  * NO pre-elimination (future: port the preelim transform into c-space).
  * Moment cone = single size-n_basis BAR.
  * One localizing BAR per i ∈ active_loc (default range(d)).
  * Optional upper-localizing (1 − μ_i).
  * Window BARs for all ``P['nontrivial_windows']``.

Verdict semantics are IDENTICAL to dual_sdp.py: λ* ≈ 1 ⟹ infeas, λ* ≈ 0 ⟹ feas.

================================================================================
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

import mosek

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'tests'))

from lasserre.dual_sdp import (  # noqa: E402
    _aggregate_bar_triplet, _aggregate_scalar_triplet,
)
from lasserre.cheby_basis import (  # noqa: E402
    compute_b_table, cheb_prod, cheb_mu_prod, cheb_mu_i_mu_j_prod,
    mono_to_cheb,
)


__all__ = [
    'build_dual_task_cheb',
    'update_task_t_cheb',
]


def _build_consistency_coupling_cheb(
    P: Dict[str, Any],
    b_table: List[List[Fraction]],
    mono_idx: Dict[Tuple[int, ...], int],
) -> Tuple[List[int], List[List[Tuple[int, float]]]]:
    """Build the scalar coupling rows for consistency equalities in c-basis.

    The monomial consistency  y_{α_k} − Σ_i y_{α_k + e_i} = 0  becomes, after
    y = B·c,  Σ_γ (B[α_k, γ] − Σ_i B[α_k+e_i, γ]) · c_γ = 0.

    For each kept consistency index k (consist_idx[k] ≥ 0), returns the list
    of (γ_idx, coef) pairs.  Zero entries are dropped.
    """
    d = int(P['d'])
    consist_mono = P['consist_mono']
    consist_idx = np.asarray(P['consist_idx'], dtype=np.int64)
    consist_ei_idx = np.asarray(P['consist_ei_idx'], dtype=np.int64)

    kept_k: List[int] = [
        k for k in range(len(consist_mono)) if int(consist_idx[k]) >= 0
    ]

    # B[α, γ] = mono_to_cheb(α, b_table)[γ].
    # We cache mono_to_cheb by alpha tuple.
    cheb_cache: Dict[Tuple[int, ...], Dict[Tuple[int, ...], Fraction]] = {}

    def _B_row(alpha: Tuple[int, ...]) -> Dict[Tuple[int, ...], Fraction]:
        v = cheb_cache.get(alpha)
        if v is None:
            v = mono_to_cheb(alpha, b_table)
            cheb_cache[alpha] = v
        return v

    mono_list = P['mono_list']
    # Reverse map: canonical row α_row → the monomial tuple (for adding e_i).
    # consist_mono has the PARENT α's.  We also need (α_k + e_i) monomials.
    # consist_ei_idx gives the CANONICAL child indices, but we need the
    # monomial-tuple to apply B.  Fall back to mono_list reverse lookup.

    # For robustness, derive child alpha tuples from α_k + e_i directly
    # — this avoids any subtlety with canonicalize_z2 remapping children
    # to their σ-representatives.
    rows_out: List[List[Tuple[int, float]]] = []
    for k in kept_k:
        alpha_k = tuple(int(x) for x in consist_mono[k])
        # B[α_k, γ] − Σ_i B[α_k + e_i, γ]
        acc: Dict[Tuple[int, ...], Fraction] = {}
        for gamma, c in _B_row(alpha_k).items():
            acc[gamma] = acc.get(gamma, Fraction(0)) + c
        for i in range(d):
            child = list(alpha_k)
            child[i] += 1
            child_t = tuple(child)
            # Only subtract if the child alpha is in the monomial support.
            # If not (degree exceeds 2k budget), the primal doesn't have
            # the corresponding consistency term.  But typically
            # consist_ei_idx[k, i] ≥ 0 iff the child is in support.
            if consist_ei_idx[k, i] < 0:
                continue
            for gamma, c in _B_row(child_t).items():
                acc[gamma] = acc.get(gamma, Fraction(0)) - c
        pairs: List[Tuple[int, float]] = []
        for gamma, c in acc.items():
            if c == 0:
                continue
            if gamma not in mono_idx:
                # degree exceeds support; skip — primal doesn't know c_gamma.
                continue
            pairs.append((int(mono_idx[gamma]), float(c)))
        rows_out.append(pairs)
    return kept_k, rows_out


def build_dual_task_cheb(
    P: Dict[str, Any], *,
    t_val: float,
    env: Optional[mosek.Env] = None,
    include_upper_loc: bool = False,
    active_loc: Optional[List[int]] = None,
    active_windows: Optional[List[int]] = None,
    lambda_upper_bound: float = 1.0,
    verbose: bool = True,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Build the Farkas-infeasibility dual SDP in shifted-Chebyshev basis.

    Parameters match ``lasserre.dual_sdp.build_dual_task`` except that
    z2_blockdiag_map is not supported in V1 (Z/2 is an orthogonal
    reduction; layered in a follow-up).
    """
    if P.get('old_to_new') is not None:
        raise NotImplementedError(
            "dual_sdp_cheby V1 does not support Z/2-canonicalised P; "
            "pass the raw (non-canonicalised) precompute.")

    d = int(P['d'])
    order = int(P['order'])
    n_y = int(P['n_y'])
    basis = P['basis']
    n_basis = int(P['n_basis'])
    loc_basis = P['loc_basis']
    n_loc = int(P['n_loc'])
    mono_idx = P['idx']
    M_mats = P['M_mats']

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

    # Precompute b_table and consistency coupling in c-space.
    b_table = compute_b_table(2 * order)
    kept_k, consist_rows_cheb = _build_consistency_coupling_cheb(
        P, b_table, mono_idx)
    n_consist = len(kept_k)

    # ------------------------------------------------------------------
    # 1. Bar-variable layout (same as monolithic).
    # ------------------------------------------------------------------
    bar_sizes: List[int] = [n_basis]  # moment cone only; no z2 split.
    moment_bar_id = 0

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

    # ------------------------------------------------------------------
    # 2. Scalar variables: [λ | μ_k | v_γ].
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Constraint rows: n_y stationarity equalities (all = 0).
    # ------------------------------------------------------------------
    task.appendcons(n_y)
    task.putconboundslice(
        0, n_y,
        [mosek.boundkey.fx] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.zeros(n_y, dtype=np.float64),
    )

    # ------------------------------------------------------------------
    # 4. Scalar coefficients (A matrix).
    # ------------------------------------------------------------------
    scalar_rows: List[int] = []
    scalar_cols: List[int] = []
    scalar_vals: List[float] = []

    # λ: +1 at γ = 0 (since y_0 = c_0 = 1).
    gamma_zero = tuple(0 for _ in range(d))
    if gamma_zero not in mono_idx:
        raise RuntimeError(
            "Zero multi-index missing from mono_idx — required by "
            "Lasserre normalisation c_0 = 1.")
    gamma_zero_row = int(mono_idx[gamma_zero])
    scalar_rows.append(gamma_zero_row)
    scalar_cols.append(LAMBDA_IDX)
    scalar_vals.append(1.0)

    # μ_k: row γ ↔ col (MU_START + j).
    for j, pairs in enumerate(consist_rows_cheb):
        col_j = MU_START + j
        for gamma_idx, coef in pairs:
            if coef != 0.0:
                scalar_rows.append(gamma_idx)
                scalar_cols.append(col_j)
                scalar_vals.append(float(coef))

    # v_α: one nonneg slack per primal constraint y_α ≥ 0.  In c-basis
    # y_α = Σ_γ B[α, γ] · c_γ, so v_α couples to stationarity row γ with
    # coefficient B[α, γ].  (In the monomial code this matrix is the
    # identity because y_α = 1·y_α; in Chebyshev it's genuinely B^T.)
    mono_list = P['mono_list']
    # Cache mono_to_cheb for each α in mono_list.
    cheb_cache_v: Dict[Tuple[int, ...], Dict[Tuple[int, ...], Fraction]] = {}
    for alpha in mono_list:
        alpha_t = tuple(int(x) for x in alpha)
        row = cheb_cache_v.get(alpha_t)
        if row is None:
            row = mono_to_cheb(alpha_t, b_table)
            cheb_cache_v[alpha_t] = row
        alpha_idx = int(mono_idx[alpha_t])
        col_v = V_START + alpha_idx  # v_α index
        for gamma, coef in row.items():
            if coef == 0:
                continue
            if gamma not in mono_idx:
                continue
            scalar_rows.append(int(mono_idx[gamma]))
            scalar_cols.append(col_v)
            scalar_vals.append(float(coef))

    r_arr, c_arr, v_arr = _aggregate_scalar_triplet(
        np.asarray(scalar_rows, dtype=np.int64),
        np.asarray(scalar_cols, dtype=np.int64),
        np.asarray(scalar_vals, dtype=np.float64),
    )
    if r_arr.size:
        task.putaijlist(r_arr, c_arr, v_arr)

    # ------------------------------------------------------------------
    # 5. Bar-matrix sensitivity coefficients in Chebyshev basis.
    # ------------------------------------------------------------------
    # Collect per-cone triplets, then aggregate + submit per cone.
    cache_arrays: List[Tuple[np.ndarray, ...]] = []

    def _emit_cone(subi_list, subj_val, subk_list, subl_list,
                   val_list, tcoef_list=None):
        if not subi_list:
            return
        subi = np.asarray(subi_list, dtype=np.int32)
        subj = np.full(subi.size, subj_val, dtype=np.int32)
        subk = np.asarray(subk_list, dtype=np.int32)
        subl = np.asarray(subl_list, dtype=np.int32)
        vals = np.asarray(val_list, dtype=np.float64)
        if tcoef_list is None:
            tcoefs = np.zeros(subi.size, dtype=np.float64)
        else:
            tcoefs = np.asarray(tcoef_list, dtype=np.float64)
        # Aggregate (dedupe — cheb_prod may emit repeats within a pair).
        a_i, a_j, a_k, a_l, a_v, a_tc = _aggregate_bar_triplet(
            subi, subj, subk, subl, vals, tcoefs)
        if a_i.size == 0:
            return
        init_vals = a_v + float(t_val) * a_tc
        task.putbarablocktriplet(a_i, a_j, a_k, a_l, init_vals)
        cache_arrays.append((a_i, a_j, a_k, a_l, a_v, a_tc))

    # ---- Moment cone ----
    mom_subi, mom_subk, mom_subl, mom_val = [], [], [], []
    for k in range(n_basis):
        bk = tuple(int(x) for x in basis[k])
        for l in range(k + 1):
            bl = tuple(int(x) for x in basis[l])
            expansion = cheb_prod(bk, bl)
            for gamma, coef in expansion.items():
                if coef == 0:
                    continue
                if gamma not in mono_idx:
                    continue
                mom_subi.append(int(mono_idx[gamma]))
                mom_subk.append(k)
                mom_subl.append(l)
                mom_val.append(float(coef))
    _emit_cone(mom_subi, moment_bar_id, mom_subk, mom_subl, mom_val)

    # ---- Localizing cones ----
    for j, i in enumerate(active_loc):
        bar_idx_here = loc_bar_start + j
        li_subi, li_subk, li_subl, li_val = [], [], [], []
        for k in range(n_loc):
            bk = tuple(int(x) for x in loc_basis[k])
            for l in range(k + 1):
                bl = tuple(int(x) for x in loc_basis[l])
                expansion = cheb_mu_prod(bk, bl, i)
                for gamma, coef in expansion.items():
                    if coef == 0:
                        continue
                    if gamma not in mono_idx:
                        continue
                    li_subi.append(int(mono_idx[gamma]))
                    li_subk.append(k)
                    li_subl.append(l)
                    li_val.append(float(coef))
        _emit_cone(li_subi, bar_idx_here, li_subk, li_subl, li_val)

    # ---- Upper-localizing cones X'_i = (1 − μ_i) ≥ 0 ----
    if include_upper_loc:
        for j, i in enumerate(active_loc):
            bar_idx_here = uloc_bar_start + j
            # +1 term: T_{loc[k]}* · T_{loc[l]}* (cheb_prod)
            # −1 term: μ_i · T_{loc[k]}* · T_{loc[l]}* (cheb_mu_prod)
            u_subi, u_subk, u_subl, u_val = [], [], [], []
            for k in range(n_loc):
                bk = tuple(int(x) for x in loc_basis[k])
                for l in range(k + 1):
                    bl = tuple(int(x) for x in loc_basis[l])
                    for gamma, coef in cheb_prod(bk, bl).items():
                        if coef == 0: continue
                        if gamma not in mono_idx: continue
                        u_subi.append(int(mono_idx[gamma]))
                        u_subk.append(k)
                        u_subl.append(l)
                        u_val.append(+float(coef))
                    for gamma, coef in cheb_mu_prod(bk, bl, i).items():
                        if coef == 0: continue
                        if gamma not in mono_idx: continue
                        u_subi.append(int(mono_idx[gamma]))
                        u_subk.append(k)
                        u_subl.append(l)
                        u_val.append(-float(coef))
            _emit_cone(u_subi, bar_idx_here, u_subk, u_subl, u_val)

    # ---- Window cones: X_W encodes t·M_{k-1}(c) − Q_W^cheb(c) ⪰ 0 ----
    for w_j, w in enumerate(active_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_bar_start + w_j

        w_subi, w_subk, w_subl, w_val, w_tcoef = [], [], [], [], []

        # (a) t-part: +t · T_{loc[k]}* · T_{loc[l]}* (cheb_prod).
        for k in range(n_loc):
            bk = tuple(int(x) for x in loc_basis[k])
            for l in range(k + 1):
                bl = tuple(int(x) for x in loc_basis[l])
                for gamma, coef in cheb_prod(bk, bl).items():
                    if coef == 0: continue
                    if gamma not in mono_idx: continue
                    w_subi.append(int(mono_idx[gamma]))
                    w_subk.append(k)
                    w_subl.append(l)
                    w_val.append(0.0)
                    w_tcoef.append(float(coef))

        # (b) Q-part: −Σ_{i,j} M_W[i,j] · μ_i μ_j · T_{loc[k]}* T_{loc[l]}*.
        # cheb_mu_i_mu_j_prod applies mu_j then mu_i.
        for pair_idx in range(len(nz_i)):
            ii = int(nz_i[pair_idx])
            jj = int(nz_j[pair_idx])
            raw = float(Mw[ii, jj])
            if raw == 0.0:
                continue
            for k in range(n_loc):
                bk = tuple(int(x) for x in loc_basis[k])
                for l in range(k + 1):
                    bl = tuple(int(x) for x in loc_basis[l])
                    for gamma, coef in cheb_mu_i_mu_j_prod(bk, bl, ii, jj).items():
                        if coef == 0: continue
                        if gamma not in mono_idx: continue
                        w_subi.append(int(mono_idx[gamma]))
                        w_subk.append(k)
                        w_subl.append(l)
                        w_val.append(-raw * float(coef))
                        w_tcoef.append(0.0)
        _emit_cone(w_subi, W_bar_idx, w_subk, w_subl, w_val, w_tcoef)

    # ---- Consolidate cache for update_task_t_cheb ----
    if cache_arrays:
        all_subi = np.concatenate([c[0] for c in cache_arrays])
        all_subj = np.concatenate([c[1] for c in cache_arrays])
        all_subk = np.concatenate([c[2] for c in cache_arrays])
        all_subl = np.concatenate([c[3] for c in cache_arrays])
        all_val = np.concatenate([c[4] for c in cache_arrays])
        all_tcoef = np.concatenate([c[5] for c in cache_arrays])
    else:
        all_subi = all_subj = all_subk = all_subl = None
        all_val = all_tcoef = None

    # ------------------------------------------------------------------
    # 6. Objective: maximize λ.
    # ------------------------------------------------------------------
    task.putobjsense(mosek.objsense.maximize)
    task.putcj(LAMBDA_IDX, 1.0)

    build_time = time.time() - t0

    info = {
        'build_time_s': build_time,
        'bar_sizes': bar_sizes,
        'n_bar': n_bar,
        'moment_bar_ids': [moment_bar_id],
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
        't_val': float(t_val),
        'LAMBDA_IDX': LAMBDA_IDX,
        'MU_START': MU_START,
        'V_START': V_START,
        'lambda_upper_bound': float(lambda_upper_bound),
        'z2_canonicalized': False,
        'z2_blockdiag': False,
        'include_upper_loc': bool(include_upper_loc),
        'basis_type': 'chebyshev',
        '_all_subi': all_subi,
        '_all_subj': all_subj,
        '_all_subk': all_subk,
        '_all_subl': all_subl,
        '_all_static': all_val,
        '_all_tcoef': all_tcoef,
        'n_bar_entries': 0 if all_subi is None else int(all_subi.size),
    }

    if verbose:
        print(f"  [dual-cheb] n_bar={n_bar}  n_scalar={n_scalar:,}  "
              f"n_cons={n_y:,}  n_bar_entries={info['n_bar_entries']:,}  "
              f"t={t_val:.6f}  build={build_time:.2f}s", flush=True)

    return task, info


def update_task_t_cheb(task: mosek.Task, info: Dict[str, Any],
                        t_val: float) -> None:
    """Update t-dependent bar entries (same logic as dual_sdp.update_task_t)."""
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
