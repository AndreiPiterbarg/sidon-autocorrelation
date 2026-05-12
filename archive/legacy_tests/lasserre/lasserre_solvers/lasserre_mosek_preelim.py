#!/usr/bin/env python
"""MOSEK-tuned Lasserre solver with consistency + simplex PRE-ELIMINATION.

Companion to ``tests/lasserre_mosek_tuned.py``.  The original file is NOT
modified; this file imports its shared utilities.

Pre-elimination is a LOSSLESS variable substitution of the simplex +
consistency equalities

    y_0 = 1                                    (simplex / normalisation)
    y_α - Σ_i y_{α + e_i} = 0                  (consistency)

through Gauss–Jordan.  Per-pivot, one moment variable is substituted out as
an affine combination of the surviving moment variables.  The reduced SDP
has fewer equality rows feeding MOSEK's Schur complement (per-iteration IPM
cost scales as m^3, so cutting m is the single largest per-iter lever for
d=16 L3).

Every SDP cone (moment matrix PSD, localising PSD, upper-localising PSD,
window PSD) is re-expressed through sparse coefficient matrices composed
with the elimination transform T — the efficient ``y.pick(...)`` pattern
cannot be used after substitution because pivoted coordinates are LINEAR
COMBINATIONS of free coordinates, not mere renamings.

Optional Z/2 equalities (mode = z2_eq / z2_bd / z2_full) are still injected,
AFTER the pre-elim — they operate on ỹ through composition with T.

USAGE
-----

    python tests/lasserre_mosek_preelim.py --d 4 --order 3 --mode tuned
    python tests/lasserre_mosek_preelim.py --d 8 --order 3 --mode z2_bd
    python tests/lasserre_mosek_preelim.py --d 10 --order 3 --mode tuned
"""
from __future__ import annotations

import os

# Thread env vars MUST be set before importing numpy / scipy / mosek.
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_phys_cores_for_env = max(1, (os.cpu_count() or 2) // 2)
os.environ.setdefault('OMP_NUM_THREADS', str(_phys_cores_for_env))

import argparse
import gc
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import mosek
from mosek.fusion import (Domain, Expr, Matrix, Model, ObjectiveSense,
                          SolutionStatus)

from lasserre_scalable import _precompute
from lasserre.preelim import build_preelim_transform, PreElimTransform
from lasserre.z2_symmetry import z2_symmetry_pairs
from lasserre.z2_blockdiag import (build_blockdiag_picks,
                                     localizing_sigma_reps,
                                     window_sigma_reps)
from lasserre.z2_elim import canonicalize_z2
from lasserre.core import (
    ab_eiej_slice_ij as _lazy_ab_eiej_slice_ij,
    get_ab_eiej_ij as _get_ab_eiej_ij,
)

# Parallelism knob for the per-(i,j) substitute_pick loop that dominates
# Phase-1 wall-clock at d>=16 L=3.  Threads (not processes): scipy.sparse
# and NumPy both release the GIL on the big ops (searchsorted, csr matmul,
# reindex), so threads scale without the 342 MB × n_workers spawn cost.
import concurrent.futures as _futures
_IJ_WORKERS = int(os.environ.get(
    'SIDON_IJ_WORKERS', str(max(1, (os.cpu_count() or 2)))))

# Reuse shared utilities from the tuned file — DO NOT modify that file.
from lasserre_mosek_tuned import (
    MosekLogStream,
    SolverWatcher,
    apply_baseline_params,
    apply_tuned_params,
    _export_proof_artefacts,
    val_d_known,
)


# =====================================================================
# Helpers for building MOSEK expressions under pre-elimination
# =====================================================================

def _mosek_sparse(M: sp.spmatrix) -> Matrix:
    """Convert a scipy sparse matrix to a MOSEK Fusion Matrix.sparse.

    Performance: avoids ``.tolist()`` — the numpy→list conversion allocates
    one Python int/float object per nnz and was the dominant cost at large
    d.  MOSEK Fusion accepts ndarray directly (jpype handles the copy into
    the Java int[] / double[] in a single bulk call).  We ensure contiguous
    int32 / float64 — MOSEK's Java API expects 32-bit indices; 64-bit row
    indices would be down-cast internally so we do it here once.
    """
    coo = M.tocoo()
    r = np.ascontiguousarray(coo.row, dtype=np.int32)
    c = np.ascontiguousarray(coo.col, dtype=np.int32)
    v = np.ascontiguousarray(coo.data, dtype=np.float64)
    return Matrix.sparse(int(coo.shape[0]), int(coo.shape[1]), r, c, v)


def _const_term(c: np.ndarray):
    """``Expr.constTerm`` with numpy support and no per-element Python cast."""
    return Expr.constTerm(np.ascontiguousarray(c, dtype=np.float64))


def _affine_expr(C: sp.csr_matrix, c: np.ndarray, y_red):
    """Return the Fusion expression  C·ỹ + c  as a vector-valued expression.

    Handles the all-zero sparse-C edge case.
    """
    C = C.tocsr()
    if C.nnz == 0:
        return _const_term(c)
    return Expr.add(Expr.mul(_mosek_sparse(C), y_red), _const_term(c))


def _scaled_vec_by_param(const_vec: np.ndarray, t_var) -> Any:
    """Return the Fusion expression  t · c  as an (n,) vector expression,
    where c ∈ R^n is a constant and t is a scalar Parameter.

    We express this as (c as column) × (t as 1×1), then flatten — the same
    pattern used elsewhere in the tuned file.
    """
    n = int(const_vec.shape[0])
    c_col = Matrix.dense(n, 1,
                          np.ascontiguousarray(const_vec, dtype=np.float64))
    return Expr.flatten(Expr.mul(c_col, Expr.reshape(t_var, 1, 1)))


# =====================================================================
# Build the pre-eliminated Lasserre model
# =====================================================================

def _build_full_model_preelim(
    P: Dict[str, Any], xf: PreElimTransform, *,
    z2_mode: str = 'off',
    add_upper_loc: bool = True,
    verbose: bool = True,
) -> Tuple[Model, Any, Any, Dict[str, Any]]:
    """Build the full Lasserre model using the pre-elimination transform xf.

    Reduced moment vector ỹ is the MOSEK variable.  Every PSD cone is built
    from sparse coefficient matrices composed with xf.T and xf.offset.

    Returns (model, y_red, t_param, stats_dict).
    """
    d = P['d']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    n_win = P['n_win']
    idx = P['idx']

    t0 = time.time()
    mdl = Model('lasserre_preelim')
    # Free coordinates of ỹ correspond directly to moment values y_α (α ∈
    # free_cols), which are nonnegative.  Use the variable domain — cheap
    # MOSEK bound — rather than a general linear inequality.
    y_red = mdl.variable(xf.n_y_red, Domain.greaterThan(0.0))
    t_var = mdl.parameter('t')

    # ---- y ≥ 0 for PIVOTED coords: T_pivot ỹ + c_pivot ≥ 0 ----
    # The free subset is automatically enforced by the domain above; only
    # the pivoted rows (y_j = c_j + Σ_l T[j,l] ỹ_l) need explicit linear
    # inequalities.  This keeps the row count small.
    pivot_rows = xf.pivot_cols  # original-y indices of pivoted coords
    if pivot_rows.size:
        T_piv = xf.T[pivot_rows, :].tocsr()
        off_piv = xf.offset[pivot_rows]
        if T_piv.nnz:
            mdl.constraint(
                Expr.add(Expr.mul(_mosek_sparse(T_piv), y_red),
                         Expr.constTerm(off_piv.tolist())),
                Domain.greaterThan(0.0))
        else:
            # Pivoted coords that reduce to pure constants: offset must be ≥ 0.
            mdl.constraint(Expr.constTerm(off_piv.tolist()),
                            Domain.greaterThan(0.0))

    # ---- Residual equalities from pre-elim (fill-capped rows) ----
    # These stand as  residual_A · y = residual_b.  Composed with xf:
    #   (residual_A · T) · ỹ = residual_b - residual_A · offset.
    n_resid = xf.residual_A.shape[0]
    if n_resid > 0:
        RA_red, RA_off = xf.substitute_matrix(xf.residual_A)
        rhs = xf.residual_b - RA_off
        RA_mat = _mosek_sparse(RA_red)
        mdl.constraint(Expr.mul(RA_mat, y_red),
                        Domain.equalsTo(rhs.tolist()))

    # ---- Z/2 equality constraints (when required by mode) ----
    n_z2_eq = 0
    if z2_mode in ('equalities', 'blockdiag', 'full'):
        pairs = z2_symmetry_pairs(P)
        n_z2_eq = len(pairs)
        if n_z2_eq:
            eq_rows = []
            eq_cols = []
            eq_vals = []
            for r, (i, j) in enumerate(pairs):
                eq_rows.extend([r, r])
                eq_cols.extend([int(i), int(j)])
                eq_vals.extend([1.0, -1.0])
            A_z2 = sp.csr_matrix((eq_vals, (eq_rows, eq_cols)),
                                  shape=(n_z2_eq, n_y),
                                  dtype=np.float64)
            A_z2_red, off_z2 = xf.substitute_matrix(A_z2)
            # (A_z2 · T) · ỹ = - A_z2 · offset
            rhs_z2 = -off_z2
            mdl.constraint(Expr.mul(_mosek_sparse(A_z2_red), y_red),
                            Domain.equalsTo(rhs_z2.tolist()))

    # ---- σ-reps for z2_full mode ----
    if z2_mode == 'full':
        loc_fixed, loc_pairs = localizing_sigma_reps(d)
        loc_active = list(loc_fixed) + [p for (p, _) in loc_pairs]
        win_fixed, win_pairs = window_sigma_reps(d, P['windows'])
        nontriv = set(P['nontrivial_windows'])
        win_active = [w for w in (list(win_fixed)
                                   + [p for (p, _) in win_pairs])
                       if w in nontriv]
    else:
        loc_active = list(range(d))
        win_active = list(P['nontrivial_windows'])

    # ---- Moment-matrix PSD ----
    if z2_mode in ('blockdiag', 'full'):
        bd = build_blockdiag_picks(P['basis'], idx, n_y)
        T_sym = bd['T_sym']
        T_anti = bd['T_anti']
        n_sym = bd['n_sym']
        n_anti = bd['n_anti']

        # Each selector T_sel (shape (n_sel^2, n_y)) composes with xf.T.
        if T_sym.nnz:
            S_red, s_off = xf.substitute_matrix(T_sym)
            sym_flat = _affine_expr(S_red, s_off, y_red)
            sym_2d = Expr.reshape(sym_flat, n_sym, n_sym)
            mdl.constraint(sym_2d, Domain.inPSDCone(n_sym))

        if n_anti > 0 and T_anti.nnz:
            A_red, a_off = xf.substitute_matrix(T_anti)
            anti_flat = _affine_expr(A_red, a_off, y_red)
            anti_2d = Expr.reshape(anti_flat, n_anti, n_anti)
            mdl.constraint(anti_2d, Domain.inPSDCone(n_anti))
    else:
        C_mom, c_mom = xf.substitute_pick(np.asarray(P['moment_pick']))
        M_flat = _affine_expr(C_mom, c_mom, y_red)
        M_mat = Expr.reshape(M_flat, n_basis, n_basis)
        mdl.constraint(M_mat, Domain.inPSDCone(n_basis))

    # ---- Localising PSD: μ_i ≥ 0 and optionally (1 - μ_i) ≥ 0 ----
    # Precompute t_pick substitution once — reused for upper-loc and window PSD.
    # Cache per-i_var pick subs so the upper-loc loop below does not re-compute
    # them (the original call site computed each one twice).
    C_tpick = c_tpick = None
    loc_subs: Dict[int, Tuple[sp.csr_matrix, np.ndarray]] = {}
    if P['order'] >= 2:
        C_tpick, c_tpick = xf.substitute_pick(np.asarray(P['t_pick']))

        for i_var in loc_active:
            C_loc, c_loc = xf.substitute_pick(
                np.asarray(P['loc_picks'][i_var]))
            loc_subs[i_var] = (C_loc, c_loc)
            L_flat = _affine_expr(C_loc, c_loc, y_red)
            Li = Expr.reshape(L_flat, n_loc, n_loc)
            mdl.constraint(Li, Domain.inPSDCone(n_loc))

        if add_upper_loc:
            for i_var in loc_active:
                C_loc, c_loc = loc_subs[i_var]
                C_diff = (C_tpick - C_loc).tocsr()
                c_diff = c_tpick - c_loc
                U_flat = _affine_expr(C_diff, c_diff, y_red)
                Li = Expr.reshape(U_flat, n_loc, n_loc)
                mdl.constraint(Li, Domain.inPSDCone(n_loc))

    # ---- Scalar window constraint: t ≥ f_W(y) for all W ----
    # F original: sparse (n_win, n_y).  Composed: F_red ỹ + f_off,
    # inequality: t ≥ F_red ỹ + f_off  ⇔  t - F_red ỹ ≥ f_off.
    F_scipy = P['F_scipy']
    F_red, f_off = xf.substitute_matrix(F_scipy)
    F_red_mat = _mosek_sparse(F_red) if F_red.nnz else None
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, Expr.reshape(t_var, 1, 1)))
    if F_red_mat is not None:
        rhs_expr = Expr.sub(t_rep, Expr.mul(F_red_mat, y_red))
    else:
        rhs_expr = t_rep
    mdl.constraint(rhs_expr,
                    Domain.greaterThan(f_off.astype(np.float64).tolist()))

    # ---- Window PSD cones: t · M_{k-1}(y) - Q_W(y) ⪰ 0 ----
    # M_{k-1}(y)_flat = C_tpick ỹ + c_tpick.
    # Q_W(y)_flat     = C_W ỹ + c_W.
    #
    # PERFORMANCE.  Each Q_W is a linear combination of SHARED (i,j)
    # picks:
    #       Q_W[a,b] = Σ_{i,j: Mw[i,j]≠0} Mw[i,j] · y[ab_eiej_idx[a,b,i,j]]
    # i.e.  Q_W = Σ Mw[i,j] · Y_{ij}  where Y_{ij}[a,b] = y[ab_eiej_idx[a,b,i,j]]
    # is INDEPENDENT of W.
    #
    # We precompute  C_ij, c_ij  via xf.substitute_pick ONCE PER (i,j) —
    # i.e. d² pick substitutions total, independent of n_win.  Per-window
    # we merely do a weighted sparse sum in reduced space, skipping both
    # the per-window Cw allocation and the per-window Cw @ T composition.
    #
    # At d=16 L3 this turns ~500 × O(230k × 26)-op sparse mat-muls into
    # 256 O(23k × 26)-op pick substitutions + ~500 × ~10 scaled sparse
    # additions — ~11× fewer arithmetic ops and ~20× fewer scipy calls.
    n_win_psd_added = 0
    if P['order'] >= 2:
        ab_eiej_idx = P['ab_eiej_idx']
        lazy_mode = ab_eiej_idx is None and P.get('AB_loc_hash') is not None
        flat_size = n_loc * n_loc

        C_tpick_mat = _mosek_sparse(C_tpick) if C_tpick.nnz else None
        # t · c_tpick expression (vector of length flat_size) — SHARED across
        # all windows.
        t_const_tpick = _scaled_vec_by_param(c_tpick, t_var)

        # Precompute (i,j) pick substitutions AND cache their COO triples
        # for fast per-window assembly (avoids pairwise sparse additions).
        #
        # Parallelised across (i,j) via a thread pool — each task computes
        # one (n_loc^2,) pick vector and one xf.substitute_pick call.  Both
        # searchsorted (inside the lazy slice) and scipy's sparse indexing
        # release the GIL on the large arrays, so threads give near-linear
        # speed-up without the spawn memory cost of multiprocessing.
        #
        # At d=32 L=3 this turns ~1024 sequential O(n_loc^2 × T_nnz) ops
        # into ~1024/N_CPU parallel batches.  Memory stays at the single
        # AB_loc_hash (~342 MB) rather than N_CPU copies.
        ij_C: Dict[Tuple[int, int], sp.csr_matrix] = {}
        ij_c: Dict[Tuple[int, int], np.ndarray] = {}
        ij_coo: Dict[Tuple[int, int],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if ab_eiej_idx is not None or lazy_mode:
            # Local closure keeps P / ab_eiej_idx on the heap once; workers
            # reuse the same references.
            AB_loc_hash = P.get('AB_loc_hash')
            bases = P.get('bases')
            sorted_h = P.get('sorted_h')
            sort_o = P.get('sort_o')
            prime = P.get('prime')

            def _pick_for_ij(i_: int, j_: int) -> np.ndarray:
                # _get_ab_eiej_ij handles both eager/lazy AND applies the
                # Z/2 old_to_new remap when canonicalize_z2 has run.
                return _get_ab_eiej_ij(
                    P, i_, j_).astype(np.int64, copy=False).ravel()

            def _job(ij: Tuple[int, int]):
                i_, j_ = ij
                pick = _pick_for_ij(i_, j_)
                C_ij, c_ij = xf.substitute_pick(pick)
                if C_ij.nnz or np.any(c_ij != 0.0):
                    coo = C_ij.tocoo()
                    return (i_, j_, C_ij, c_ij,
                            coo.row.astype(np.int64, copy=False),
                            coo.col.astype(np.int64, copy=False),
                            coo.data.astype(np.float64, copy=False))
                return None

            ij_pairs = [(i_, j_) for i_ in range(d) for j_ in range(d)]
            n_workers = max(1, min(_IJ_WORKERS, len(ij_pairs)))
            t_ij_start = time.time()
            if n_workers == 1:
                results = [_job(ij) for ij in ij_pairs]
            else:
                with _futures.ThreadPoolExecutor(
                        max_workers=n_workers) as pool:
                    results = list(pool.map(_job, ij_pairs))
            for r in results:
                if r is None:
                    continue
                i_, j_, C_ij, c_ij, rr, cc, dd = r
                ij_C[(i_, j_)] = C_ij
                ij_c[(i_, j_)] = c_ij
                ij_coo[(i_, j_)] = (rr, cc, dd)
            if verbose:
                kept = len(ij_coo)
                print(f"  ij-pick loop: {len(ij_pairs)} pairs, kept "
                      f"{kept}, {n_workers} threads, "
                      f"{time.time() - t_ij_start:.2f}s "
                      f"(lazy={lazy_mode})", flush=True)

        def _cone_tM_only() -> None:
            """Handle Q_W = 0: cone is simply t · M_{k-1}(y) ⪰ 0."""
            if C_tpick_mat is not None:
                lhs_linear = Expr.mul(t_var, Expr.mul(C_tpick_mat, y_red))
                Lw_flat = Expr.add(lhs_linear, t_const_tpick)
            else:
                Lw_flat = t_const_tpick
            mdl.constraint(Expr.reshape(Lw_flat, n_loc, n_loc),
                            Domain.inPSDCone(n_loc))

        # ij_coo is populated whenever the eager 4D array exists OR lazy_mode
        # is active (AB_loc_hash available).  In either case we still have
        # the full Q_W contribution for every window — the fall-through to
        # _cone_tM_only() below must only fire when Mw is literally zero.
        have_qw_data = (ab_eiej_idx is not None) or lazy_mode
        for w in win_active:
            Mw = P['M_mats'][w]
            nz_i, nz_j = np.nonzero(Mw)

            if len(nz_i) == 0 or not have_qw_data:
                _cone_tM_only()
                n_win_psd_added += 1
                continue

            # Concatenate per-window COO triples weighted by Mw[i,j], then
            # build a single CSR in one shot (avoids pairwise sparse adds).
            row_chunks: List[np.ndarray] = []
            col_chunks: List[np.ndarray] = []
            dat_chunks: List[np.ndarray] = []
            cw_off = np.zeros(flat_size, dtype=np.float64)
            for ii, jj in zip(nz_i.tolist(), nz_j.tolist()):
                key = (int(ii), int(jj))
                if key not in ij_coo:
                    continue
                w_ij = float(Mw[ii, jj])
                if w_ij == 0.0:
                    continue
                rr, cc, dd = ij_coo[key]
                if dd.size:
                    row_chunks.append(rr)
                    col_chunks.append(cc)
                    dat_chunks.append(w_ij * dd)
                cw_off = cw_off + w_ij * ij_c[key]

            if row_chunks:
                Cw_red = sp.csr_matrix(
                    (np.concatenate(dat_chunks),
                     (np.concatenate(row_chunks),
                      np.concatenate(col_chunks))),
                    shape=(flat_size, xf.n_y_red),
                    dtype=np.float64,
                )
                Cw_red.sum_duplicates()
                Cw_red.eliminate_zeros()
            else:
                Cw_red = None

            if Cw_red is None and not np.any(cw_off != 0.0):
                # Entire Q_W was zero in the reduced space — same as Q=0.
                _cone_tM_only()
                n_win_psd_added += 1
                continue

            # Compose the PSD expression:
            #   t·C_tpick ỹ  +  t·c_tpick  -  C_W ỹ  -  c_W
            terms: List[Any] = []
            if C_tpick_mat is not None:
                terms.append(Expr.mul(t_var, Expr.mul(C_tpick_mat, y_red)))
            terms.append(t_const_tpick)
            if Cw_red is not None and Cw_red.nnz:
                terms.append(Expr.neg(Expr.mul(
                    _mosek_sparse(Cw_red), y_red)))
            if np.any(cw_off != 0.0):
                terms.append(_const_term(-cw_off))

            Lw_flat = terms[0]
            for term in terms[1:]:
                Lw_flat = Expr.add(Lw_flat, term)
            mdl.constraint(Expr.reshape(Lw_flat, n_loc, n_loc),
                            Domain.inPSDCone(n_loc))
            n_win_psd_added += 1

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    build_time = time.time() - t0
    n_y_total_orig = P['n_y']
    n_eq_elim = int(len(xf.pivot_cols))
    stats = {
        'build_time_s': build_time,
        'n_y_original': n_y_total_orig,
        'n_y_reduced':  xf.n_y_red,
        'n_y_pivoted':  n_eq_elim,
        'n_resid_eqs':  int(xf.residual_A.shape[0]),
        'n_basis': n_basis,
        'n_loc':   n_loc,
        'n_z2_eq': n_z2_eq,
        'n_win_psd': n_win_psd_added,
        'n_win_active': len(win_active),
        'n_win_original': len(P['nontrivial_windows']),
        'n_loc_active': len(loc_active),
        'n_loc_original': d,
        'z2_mode': z2_mode,
        'add_upper_loc': add_upper_loc,
        'T_nnz': int(xf.T.nnz),
    }
    if verbose:
        print(f"  Build (preelim): n_y {n_y_total_orig:,} -> "
              f"{xf.n_y_red:,} (pivoted {n_eq_elim:,}, "
              f"resid {stats['n_resid_eqs']}), "
              f"n_basis={n_basis}, n_loc={n_loc}, "
              f"n_z2_eq={n_z2_eq}, n_win_psd={n_win_psd_added}, "
              f"build_time={build_time:.2f}s", flush=True)
    return mdl, y_red, t_var, stats


# =====================================================================
# Top-level solve
# =====================================================================

def solve_mosek_preelim(
    d: int, order: int, *,
    mode: str = 'tuned',
    add_upper_loc: bool = True,
    n_bisect: int = 15,
    t_lo: float = 0.5,
    t_hi: Optional[float] = None,
    proof_dir: Optional[str] = None,
    pre_elim_z2: bool = False,
    max_fill_ratio: float = 10.0,
    protect_degrees: Optional[set] = None,
    primary_tol: float = 1e-6,
    order_method: str = 'forceGraphpar',
    force_lindep_off: bool = False,
    watcher_interval_s: float = 15.0,
    lazy_ab_eiej: Optional[bool] = None,
    single_t: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the Lasserre SDP via MOSEK Fusion with consistency-equality
    pre-elimination, bisecting on t.

    mode ∈ {baseline, tuned, z2_eq, z2_bd, z2_full}.  Z/2 equalities (when
    selected) are injected AFTER the consistency pre-elim — they're composed
    through xf in the reduced basis.

    pre_elim_z2: if True, ALSO canonicalise the Z/2 equalities in Python
                  BEFORE building the SDP (orthogonal to this file's
                  consistency pre-elim).  Requires a Z/2 mode.

    Returns dict with lb, status, timings, etc.
    """
    if mode not in ('baseline', 'tuned', 'z2_eq', 'z2_bd', 'z2_full'):
        raise ValueError(f"Unknown mode {mode!r}")

    if pre_elim_z2 and mode in ('baseline', 'tuned'):
        raise ValueError(
            f"pre_elim_z2 requires a Z/2 mode (z2_eq / z2_bd / z2_full); "
            f"got mode={mode!r}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK pre-elim Lasserre: d={d} L{order} mode={mode} "
              f"pre_elim_z2={pre_elim_z2} fill_cap={max_fill_ratio}")
        print(f"{'=' * 60}", flush=True)

    P = _precompute(d, order, verbose=verbose, lazy_ab_eiej=lazy_ab_eiej)

    if pre_elim_z2:
        P = canonicalize_z2(P, verbose=verbose)

    # --- Build the pre-elim transform FIRST ---
    t_pre = time.time()
    xf = build_preelim_transform(
        P, max_fill_ratio=max_fill_ratio,
        protect_degrees=protect_degrees,
        verbose=verbose)
    preelim_time = time.time() - t_pre
    if verbose:
        print(f"  preelim build: {preelim_time:.2f}s", flush=True)

    z2_map = {
        'baseline': 'off', 'tuned': 'off',
        'z2_eq': 'equalities', 'z2_bd': 'blockdiag',
        'z2_full': 'full',
    }
    z2_mode = z2_map[mode]

    t_build_start = time.time()
    mdl, y_red, t_param, build_stats = _build_full_model_preelim(
        P, xf,
        z2_mode=z2_mode,
        add_upper_loc=add_upper_loc,
        verbose=verbose)
    build_time = time.time() - t_build_start
    build_stats['preelim_time_s'] = preelim_time
    build_stats['max_fill_ratio'] = max_fill_ratio
    build_stats['pre_elim_z2'] = pre_elim_z2

    if mode == 'baseline':
        params = apply_baseline_params(mdl, verbose=verbose)
    else:
        lindep = 'off' if force_lindep_off else 'on'
        params = apply_tuned_params(
            mdl, tol=primary_tol,
            order_method=order_method,
            presolve_lindep=lindep,
            verbose=verbose)

    log_stream = MosekLogStream(prefix='[MOSEK] ')
    try:
        mdl.setLogHandler(log_stream)
        mdl.setSolverParam('log', 10)
        mdl.setSolverParam('logIntpnt', 10)
        try:
            mdl.setSolverParam('logIntpntFactor', 10)
        except Exception:
            pass
        try:
            mdl.setSolverParam('logPresolve', 10)
        except Exception:
            pass
        try:
            mdl.setSolverParam('logIntpntFreq', 1)
        except Exception:
            pass
    except Exception as exc:
        if verbose:
            print(f"  log handler attach failed: {exc}", flush=True)

    watcher = SolverWatcher(interval_s=watcher_interval_s, tag='solve')

    val = val_d_known.get(d)
    if t_hi is None:
        t_hi = (val + 0.05) if val else 2.0

    per_solve_times: List[float] = []
    err: Optional[str] = None
    ok = True

    def _classify(ps, ds) -> str:
        if ps in (SolutionStatus.Optimal, SolutionStatus.Feasible):
            return 'feas'
        if ds == SolutionStatus.Certificate:
            return 'infeas'
        return 'uncertain'

    def _make_ladder(primary: float) -> List[Tuple[float, str]]:
        # Drop the loose 1e-3 / 1e-4 rungs.  The pre-elim model has denser
        # PSD-cone dependencies (each entry couples through T to ~d reduced
        # moments), so "feas" verdicts at tolerance ≥ 1e-4 are unreliable
        # near val_L3(d) — the primal violations at that slack dominate the
        # sub-1e-4 precision we're bisecting at.  A "feas" at 1e-3 can
        # falsely collapse hi below a true-infeas t and weaken lb.
        # Retry only at 1e-5 and 1e-6 (and 1e-7) — return "uncertain"
        # otherwise, which preserves bisection soundness (lb = lo advances
        # only on certified infeas).
        candidate_tols = (1e-5, 1e-6, 1e-7)
        rungs: List[Tuple[float, str]] = []
        for tol in candidate_tols:
            if abs(tol - primary) < primary * 0.01:
                continue
            for form in ('primal', 'dual'):
                rungs.append((tol, form))
        return rungs

    RETRY_LADDER: List[Tuple[float, str]] = _make_ladder(primary_tol)

    def _check_feasible(tv: float, retry_on_unknown: bool = True
                         ) -> Tuple[str, float, str]:
        t_param.setValue(tv)
        ts = time.time()
        try:
            log_stream.mark_solve_start()
            watcher.start()
            try:
                mdl.solve()
            finally:
                watcher.stop()
            ps = mdl.getPrimalSolutionStatus()
            ds = mdl.getDualSolutionStatus()
            verdict = _classify(ps, ds)
            stat_str = (f"{str(ps).split('.')[-1]}/"
                        f"{str(ds).split('.')[-1]}")
        except Exception as exc:
            return 'uncertain', time.time() - ts, \
                    f'error:{type(exc).__name__}'

        if verdict == 'uncertain' and retry_on_unknown:
            original_tol = primary_tol
            original_form = 'dual'
            original_iters = 1600
            for retry_tol, retry_form in RETRY_LADDER:
                if verdict != 'uncertain':
                    break
                try:
                    mdl.setSolverParam('intpntSolveForm', retry_form)
                    mdl.setSolverParam('intpntCoTolRelGap', retry_tol)
                    mdl.setSolverParam('intpntCoTolPfeas', retry_tol)
                    mdl.setSolverParam('intpntCoTolDfeas', retry_tol)
                    mdl.setSolverParam('intpntCoTolMuRed', retry_tol)
                    mdl.setSolverParam('intpntMaxIterations', 2400)
                    mdl.solve()
                    ps2 = mdl.getPrimalSolutionStatus()
                    ds2 = mdl.getDualSolutionStatus()
                    v2 = _classify(ps2, ds2)
                    stat_str += (
                        f" -> {retry_form}/{retry_tol:.0e}:"
                        f"{str(ps2).split('.')[-1]}/"
                        f"{str(ds2).split('.')[-1]}"
                    )
                    if v2 != 'uncertain':
                        verdict = v2
                        break
                except Exception:
                    pass
            try:
                mdl.setSolverParam('intpntSolveForm', original_form)
                mdl.setSolverParam('intpntCoTolRelGap', original_tol)
                mdl.setSolverParam('intpntCoTolPfeas', original_tol)
                mdl.setSolverParam('intpntCoTolDfeas', original_tol)
                mdl.setSolverParam('intpntCoTolMuRed', original_tol)
                mdl.setSolverParam('intpntMaxIterations', original_iters)
            except Exception:
                pass
        return verdict, time.time() - ts, stat_str

    proof_records: List[Dict[str, Any]] = []

    def _maybe_export(tag: str, tv: float, verdict: str,
                      stat_str: str) -> None:
        if proof_dir is None:
            return
        if verdict in ('feas', 'infeas'):
            rec = _export_proof_artefacts(
                mdl, y_red, t_param, tv, verdict, stat_str,
                proof_dir, tag,
                d=d, order=order, mode=mode, verbose=verbose)
            proof_records.append(rec)

    # ---- Single-t feasibility mode (parallel-bisection launchers) ----
    # When --single-t X.XX is passed, skip the whole bisection; run ONE
    # _check_feasible(X.XX) and return early with a structured verdict.
    # Used by tests/parallel_bisect.py to fan out N solves across processes.
    if single_t is not None:
        verdict, dt, stat = _check_feasible(float(single_t))
        per_solve_times.append(dt)
        _maybe_export(f'single_t{single_t:.6f}', single_t, verdict, stat)
        try:
            mdl.dispose()
        except Exception:
            pass
        gc.collect()
        # Machine-parseable line — parallel_bisect.py greps for it.
        print(f"SINGLE_T_VERDICT t={float(single_t):.10f} "
              f"verdict={verdict} status={stat} wall_s={dt:.3f}",
              flush=True)
        return {
            'd': d, 'order': order, 'mode': mode,
            'single_t': float(single_t),
            'verdict': verdict,
            'status': stat,
            'wall_s': dt,
            'build_time_s': build_time,
            'preelim_time_s': preelim_time,
            'params': params,
            'build_stats': build_stats,
            'ok': True,
        }

    lo, hi = float(t_lo), float(t_hi)
    v_hi, dt_hi, stat_hi = _check_feasible(hi)
    per_solve_times.append(dt_hi)
    _maybe_export(f'hi_probe_t{hi:.6f}', hi, v_hi, stat_hi)
    if verbose:
        print(f"\n  hi probe: t={hi:.6f} -> {v_hi} {stat_hi} "
              f"({dt_hi:.2f}s)", flush=True)
    tries = 0
    while v_hi != 'feas' and tries < 4:
        hi *= 1.5
        v_hi, dt_hi, stat_hi = _check_feasible(hi)
        per_solve_times.append(dt_hi)
        if verbose:
            print(f"  hi probe: t={hi:.6f} -> {v_hi} {stat_hi} "
                  f"({dt_hi:.2f}s)", flush=True)
        tries += 1

    if v_hi != 'feas':
        err = (f"upper bound t={hi} not feasibly solved "
                f"after {tries + 1} tries")
        ok = False

    history = []
    n_uncertain = 0
    consecutive_uncertain = 0
    if ok:
        pending_offset = 0.0
        for step in range(n_bisect):
            mid = 0.5 * (lo + hi) + pending_offset
            mid = max(lo + 1e-9, min(hi - 1e-9, mid))
            verdict, dt, stat = _check_feasible(mid)
            per_solve_times.append(dt)
            history.append({'step': step, 't': mid, 'status': stat,
                              'verdict': verdict, 'wall_s': dt})
            _maybe_export(f'step{step + 1:02d}_t{mid:.6f}',
                          mid, verdict, stat)
            if verdict == 'feas':
                hi = mid
                pending_offset = 0.0
                consecutive_uncertain = 0
            elif verdict == 'infeas':
                lo = mid
                pending_offset = 0.0
                consecutive_uncertain = 0
            else:
                n_uncertain += 1
                consecutive_uncertain += 1
                width = hi - lo
                if consecutive_uncertain == 1:
                    pending_offset = -0.20 * width
                elif consecutive_uncertain == 2:
                    pending_offset = +0.20 * width
                else:
                    pending_offset = (-1) ** consecutive_uncertain \
                                        * 0.10 * width
            if verbose:
                marker = {'feas': 'feas', 'infeas': 'infeas',
                           'uncertain': '?????'}[verdict]
                print(f"  [{step + 1}/{n_bisect}] t={mid:.8f}  "
                      f"{marker:6s} {stat:40s} ({dt:.2f}s)  "
                      f"[{lo:.6f}, {hi:.6f}]", flush=True)
            if consecutive_uncertain >= 3:
                if verbose:
                    print(f"    3 consecutive uncertain; stopping "
                          f"bisection (lb preserved).", flush=True)
                break

    lb = lo
    gc_pct: Optional[float] = None
    if val and lb is not None and val > 1.0:
        gc_pct = 100.0 * (lb - 1.0) / (val - 1.0)

    try:
        mdl.dispose()
    except Exception:
        pass
    gc.collect()

    total_solve_time = sum(per_solve_times)
    n_solves = len(per_solve_times)
    avg_solve = total_solve_time / n_solves if n_solves else 0.0

    result = {
        'd': d, 'order': order, 'mode': mode,
        'lb': lb, 'val_d': val, 'gc_pct': gc_pct,
        'ok': ok, 'error': err,
        'preelim_time_s': preelim_time,
        'build_time_s': build_time,
        'total_solve_time_s': total_solve_time,
        'n_solves': n_solves,
        'n_uncertain': n_uncertain,
        'avg_solve_time_s': avg_solve,
        'total_time_s': preelim_time + build_time + total_solve_time,
        'per_solve_times_s': per_solve_times,
        'build_stats': build_stats,
        'params': params,
        'history': history,
        'proof_dir': proof_dir,
        'proof_records': [
            {k: v for k, v in rec.items() if k != 'meta'}
            for rec in proof_records
        ],
        'last_solve_iter_trace': list(log_stream.iter_rows),
        'watcher_peak_rss_mb': watcher.peak_rss_mb,
        'watcher_samples': list(watcher.samples),
    }
    if verbose:
        gc_str = f"{gc_pct:.2f}%" if gc_pct is not None else "—"
        print(f"\n  lb={lb:.6f}  gc={gc_str}  "
              f"preelim={preelim_time:.2f}s  "
              f"build={build_time:.2f}s  "
              f"avg_solve={avg_solve:.2f}s  "
              f"total_solve={total_solve_time:.2f}s  "
              f"n_solves={n_solves}", flush=True)
    return result


# =====================================================================
# CLI
# =====================================================================

def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--mode',
                    choices=('baseline', 'tuned', 'z2_eq', 'z2_bd',
                              'z2_full'),
                    default='tuned')
    p.add_argument('--no-upper-loc', action='store_true')
    p.add_argument('--n-bisect', type=int, default=15)
    p.add_argument('--t-lo', type=float, default=0.5)
    p.add_argument('--t-hi', type=float, default=None)
    p.add_argument('--json', type=str, default=None)
    p.add_argument('--proof-dir', type=str, default=None)
    p.add_argument('--pre-elim-z2', action='store_true',
                    help='Additionally substitute out Z/2 equalities in '
                         'Python (orthogonal to consistency pre-elim).  '
                         'Requires a Z/2 mode.')
    p.add_argument('--max-fill-ratio', type=float, default=10.0,
                    help='Fill-control cap for consistency pre-elim.  '
                         'A pivot is rejected if (col_count-1)*(row_nnz-1) '
                         '> ratio * row_nnz.  Default 10.0.')
    p.add_argument('--protect-degrees', type=str, default='1,2',
                    help='Comma-separated list of monomial TOTAL degrees '
                         'protected from pivoting.  Default "1,2" — keeps '
                         'backbone moments free for conditioning.  Pass '
                         '"" to disable protection.')
    p.add_argument('--primary-tol', type=float, default=1e-6)
    p.add_argument('--order-method', type=str, default='forceGraphpar',
                    choices=('free', 'none', 'appminloc',
                              'experimental', 'tryGraphpar',
                              'forceGraphpar'))
    p.add_argument('--watcher-interval', type=float, default=15.0)
    p.add_argument('--lindep-off', action='store_true')
    p.add_argument('--lazy-ab-eiej', dest='lazy_ab_eiej',
                    action='store_true', default=None,
                    help='Force lazy ab_eiej slicing (skips the '
                         '(n_loc, n_loc, d, d) materialisation). '
                         'Needed for d>=24 L=3 where the full array '
                         'exceeds RAM.  Default follows env '
                         'SIDON_LAZY_ABEIEJ (default on).')
    p.add_argument('--no-lazy-ab-eiej', dest='lazy_ab_eiej',
                    action='store_false', default=None,
                    help='Force eager ab_eiej materialisation '
                         '(regression test against old behaviour).')
    p.add_argument('--single-t', type=float, default=None,
                    help='Skip bisection; run one _check_feasible(t) and '
                         'print a SINGLE_T_VERDICT line.  Used by '
                         'tests/parallel_bisect.py to fan out across '
                         'subprocesses.')
    args = p.parse_args()

    pd_str = args.protect_degrees.strip()
    if pd_str:
        protect_degrees = {int(x) for x in pd_str.split(',') if x.strip()}
    else:
        protect_degrees = set()

    r = solve_mosek_preelim(
        args.d, args.order,
        mode=args.mode,
        add_upper_loc=not args.no_upper_loc,
        n_bisect=args.n_bisect,
        t_lo=args.t_lo,
        t_hi=args.t_hi,
        proof_dir=args.proof_dir,
        pre_elim_z2=args.pre_elim_z2,
        max_fill_ratio=args.max_fill_ratio,
        protect_degrees=protect_degrees,
        primary_tol=args.primary_tol,
        order_method=args.order_method,
        force_lindep_off=args.lindep_off,
        watcher_interval_s=args.watcher_interval,
        lazy_ab_eiej=args.lazy_ab_eiej,
        single_t=args.single_t,
        verbose=True)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
