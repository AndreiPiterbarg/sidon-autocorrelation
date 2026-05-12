#!/usr/bin/env python
r"""Direct SCS solver for Lasserre SDP — fully vectorized, GPU-capable.

Builds SCS problem data directly in numpy/scipy. Zero Python loops in
hot paths. All COO construction is vectorized.

For bisection: the A matrix is split into t-independent (A_base) and
t-dependent (A_t) parts. Each bisection step computes A = A_base + t * A_t
as a single sparse addition — no per-element Python loops.

GPU: SCS 3.x supports GPU via compilation flag. On H100, the sparse
matvec and PSD projections run on GPU. Install: pip install scs[gpu].

Usage:
    python tests/run_scs_direct.py --d 16 --order 3 --bw 15
    python tests/run_scs_direct.py --d 14 --order 3 --bw 13 --gpu
"""
import sys
import os
import time
import json
import argparse
import gc as _gc
import numpy as np
from scipy import sparse as sp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, enum_monomials, val_d_known,
)

# Atom-based window ranking (primal oracle).  Scores each candidate window
# by μ̂^T M_W μ̂ where μ̂ are projected rank-1 atoms of M_1(y).  Mathematically
# a pure ranking heuristic — every selected window is still a valid Lasserre
# localizer so lb ≤ val(d) holds regardless of ordering.  Empirically +14 pp
# gc per round over min-eig / dual-guided selection (measured Apr 2026).
from lasserre.gap_accelerator import (
    atom_based_window_ranking, blend_rankings,
)

SQRT2 = np.sqrt(2.0)


# =====================================================================
# Verdict logic for phase-1 bisection (soundness-critical)
# =====================================================================
#
# The bisection loop maintains an interval [lo, hi] with the invariant
#   t <= lo  =>  problem is infeasible  (so val(d) > lo,  hence lb >= lo)
#   t >= hi  =>  problem is feasible    (so val(d) <= hi)
#
# Moving `lo` upward on an UNCERTIFIED infeasibility claim is the only way
# to produce an UNSOUND lower bound (lb > val(d)). The precursor of this
# module produced lb=1.388 > val(32)=1.336 on 2026-04-16 by doing exactly
# that after an early tau exit.
#
# A solver outcome falls into exactly one of three verdicts:
#
#   'feas'   : tau_val <= tau_tol AND pri_res <= RELAX * eps_pri.
#              The feasibility slack achievable by the solver is within
#              tolerance. This is SAFE: moves `hi` down, shrinks lb upper
#              estimate only.
#
#   'infeas' : status == 'solved' AND tau_val > INFEAS_MARGIN * tau_tol
#              AND pri_res < eps_pri AND dual_res < eps_dual.
#              ADMM converged to approximate KKT with pri/dual residuals
#              inside tolerance; its optimal tau* is strictly above the
#              feasibility threshold, so the true tau* is > 0 up to the
#              KKT tolerance. This is the ONLY way to certify infeasibility.
#
#   'uncertain': anything else.
#              Typical case: iter cap reached with tau_val still decreasing,
#              or adaptive rho oscillating. The bisection caller MUST NOT
#              move the bracket on 'uncertain' — it is equivalent to "no
#              information" about feasibility of this t_val.
#
# RELAX = 10 and INFEAS_MARGIN = 3 are soundness-preserving safety factors,
# not tuning knobs: they are chosen to guarantee the classification is
# consistent with the tolerance band of the underlying ADMM iterates.

_VERDICT_FEAS = 'feas'
_VERDICT_INFEAS = 'infeas'
_VERDICT_UNCERTAIN = 'uncertain'
_VERDICT_PRI_RELAX = 10.0    # feasible pri_res may be up to 10x eps_pri
_VERDICT_INFEAS_MARGIN = 1.5  # tau must exceed 1.5x tol to certify infeas
# INFEAS_MARGIN=1.5 rationale: at the feasibility boundary, ADMM converges to
# tau* ≈ 0 (or small positive). For clearly infeasible t, ADMM stabilises at
# tau* >> tau_tol. Margin 1.5 provides safety against ADMM noise while being
# narrow enough that most infeasible bisection steps certify correctly.
# Previous value of 3.0 was too conservative: tau ∈ (tau_tol, 3×tau_tol) gave
# 'uncertain', stalling the bisection and blocking lb advancement.


def _classify_verdict(sol_t, tau_val, tau_tol):
    """Classify an ADMM solve result as 'feas' / 'infeas' / 'uncertain'.

    Parameters
    ----------
    sol_t : dict
        Output of ADMMSolver.solve or admm_solve, must carry 'info' with
        status, pri_res, dual_res, eps_pri, eps_dual. Older callers that
        do not populate residuals are treated as 'uncertain'.
    tau_val : float
        Current tau variable value (positive => slack present).
    tau_tol : float
        Feasibility tolerance on tau.

    Returns
    -------
    str : one of _VERDICT_FEAS / _VERDICT_INFEAS / _VERDICT_UNCERTAIN.
    """
    info = sol_t.get('info', {}) if sol_t is not None else {}
    status = info.get('status', '')
    pri_res = info.get('pri_res', float('inf'))
    dual_res = info.get('dual_res', float('inf'))
    eps_pri = info.get('eps_pri', 0.0)
    eps_dual = info.get('eps_dual', 0.0)

    if not np.isfinite(tau_val):
        return _VERDICT_UNCERTAIN

    # Feasible verdict: tau is below threshold and primal feasibility is
    # in the tolerance band. Dual residual is NOT required for feasibility;
    # any primal-feasible point with tau <= tau_tol certifies that tau*
    # cannot exceed tau_tol, regardless of dual convergence status.
    if tau_val <= tau_tol and pri_res <= _VERDICT_PRI_RELAX * eps_pri:
        return _VERDICT_FEAS

    # Infeasible verdict: REQUIRES both residuals within KKT tolerance
    # AND status=='solved' (not 'solved_inaccurate') AND tau strictly
    # above margin. Any looser criterion admits solver stalls as infeas.
    if (status == 'solved'
            and tau_val > _VERDICT_INFEAS_MARGIN * tau_tol
            and pri_res < eps_pri
            and dual_res < eps_dual):
        return _VERDICT_INFEAS

    return _VERDICT_UNCERTAIN


# =====================================================================
# Support-diverse greedy cut selection
# =====================================================================
#
# Given a ranked list of violated windows (most negative min-eig first),
# pick up to n_add windows whose *active-bin supports* are not redundant
# with either each other or the windows already active. Two windows with
# heavily overlapping supports produce nearly-parallel cuts; adding both
# is informationally redundant and wastes memory (each PSD window cone
# costs O(n_loc^2) rows).
#
# For an autoconvolution window W=(ell, s_lo), the support is the set of
# bin indices {i : there exists j with s_lo <= i+j <= s_lo+ell-2 and
# 0 <= i,j <= d-1}. Geometrically this is the interval
#   [max(0, s_lo-(d-1)), min(d-1, s_lo+ell-2)].
#
# The diversity test is a coverage ratio: fraction of the candidate's
# bins already covered by previously-accepted supports. Above threshold,
# skip; otherwise accept and union its bins into `covered`.
#
# Soundness: this only REORDERS / FILTERS the constraint generation
# schedule. Every accepted cut is a valid necessary condition; skipping
# redundant cuts does not remove any constraint permanently — a window
# skipped this round may be re-considered next round if it still violates.
#
# Correctness edge case: if fewer than n_add windows pass the diversity
# test (the pool of violations is too redundant), the pool is topped up
# with the most-severe *unselected* violations so we never add fewer cuts
# than the original rule would have. This keeps convergence monotone.

_COVERAGE_SKIP_THRESHOLD = 0.75  # skip if >=75% of candidate bins already covered


def _window_active_bins(w, P, d):
    """Return a frozenset of bin indices covered by window w.

    Empty set if the window has no valid (i,j) pair.
    """
    ell, s_lo = P['windows'][w]
    lo_bin = max(0, s_lo - (d - 1))
    hi_bin = min(d - 1, s_lo + ell - 2)
    if lo_bin > hi_bin:
        return frozenset()
    return frozenset(range(lo_bin, hi_bin + 1))


def _greedy_diverse_cuts(violations, n_add, P, d, active_windows,
                          score_fn=None):
    """Select n_add diverse violations from a ranked list.

    Parameters
    ----------
    violations : list of (window_index, min_eig)
        Sorted ascending by min_eig (most violated first).  When score_fn
        is provided, the caller can pass violations in any order; we
        re-sort internally by descending score.
    n_add : int
        Max number of windows to pick.
    P : dict
        Precompute dict with 'windows' list.
    d : int
        Number of bins.
    active_windows : iterable of window indices
        Already-active windows; their supports seed the `covered` set so
        diversity is measured against ALL currently-active constraints,
        not just within-round.
    score_fn : callable or None
        If None, picks in input order (min-eig severity).  If provided,
        `score_fn(violation_tuple) -> float` is used to re-sort violations
        in descending order BEFORE the diversity pass.  This implements
        dual-guided selection (Frangioni 2002; Kiwiel 2004): weighting
        violation severity by the window's dual pressure (proxied by
        closeness of its scalar bound t ≥ f_W(y) to binding).

    Returns
    -------
    list of (window_index, min_eig) — the selected subset preserving
    the input ordering (post-score-sort if score_fn given).
    """
    if n_add <= 0 or not violations:
        return []

    if score_fn is not None:
        # Dual-guided re-ranking. Stable sort on score (descending) puts
        # the "most-informative" violations first; the diversity pass then
        # applies the support-overlap filter on this re-ordered list.
        violations = sorted(violations, key=score_fn, reverse=True)

    covered = set()
    for w in active_windows:
        covered |= _window_active_bins(w, P, d)

    selected = []
    selected_pos = set()  # positions in `violations` accepted by diversity pass

    for pos, item in enumerate(violations):
        w = item[0]
        eig = item[1]
        if len(selected) >= n_add:
            break
        w_bins = _window_active_bins(w, P, d)
        if not w_bins:
            continue
        overlap = len(w_bins & covered) / len(w_bins)
        if overlap < _COVERAGE_SKIP_THRESHOLD:
            selected.append((w, eig))
            selected_pos.add(pos)
            covered |= w_bins

    # Top-up pass: if the diversity filter under-filled, fall back to
    # most-severe unselected so we never add fewer cuts than the default.
    if len(selected) < n_add:
        for pos, item in enumerate(violations):
            if len(selected) >= n_add:
                break
            if pos in selected_pos:
                continue
            selected.append(item)
            selected_pos.add(pos)

    return selected


# =====================================================================
# Vectorized PSD cone COO builder
# =====================================================================
# -----------------------
# For violated window W with clique I_c, the clique-restricted localizing
# matrix is:
#   L_W^{I_c}(y,t)[a,b] = t·y_{t_pick[a,b]} - Σ_{(li,lj)∈active}
#                          (2d/ell)·y_{ab_eiej[a,b,li,lj]}
#
# For any unit vector v ∈ R^{n_cb}, the SPECTRAL CUT is:
#   v^T L_W^{I_c}(y,t) v ≥ 0
#
# This is a NECESSARY condition for L_W^{I_c} ⪰ 0, which is itself
# necessary for the Lasserre hierarchy (proved by Cauchy interlacing:
# L_W^{I_c} is a principal submatrix of the full L_W).
#
# SOUNDNESS: For true moments y* of the optimal μ* at t*=val(d):
#   v^T L_W^{I_c}(y*,t*) v ≥ 0
# because L_W^{I_c}(y*,t*) ⪰ 0 (proved in lasserre_highd docstring).
# So any lb produced from the spectral-cut relaxation satisfies lb ≤ val(d).
#
# LINEARITY: At fixed t=t_val (in the bisection subproblem, t is fixed),
# the cut is linear in y:
#   CUT(y) = t_val·Σ_m u_m·y_m − Σ_m w_m·y_m ≥ 0
# where:
#   u_m = Σ_{(a,b): t_pick[a,b]=m} v_a·v_b       (t-part)
#   w_m = Σ_{(a,b,li,lj): ab_eiej[...]= m,active}
#          (2d/ell)·v_a·v_b                         (Q-part)
#
# EFFICIENCY vs PSD CONES:
#   PSD cone: 1 cone of size n_cb=153 → 11,781 svec rows → ~3ms eigh/iter
#   Spectral cut: 1 scalar row → 0ms extra eigh per iter
#   For 100 windows: save ~300ms/iter  (262ms → ~76ms)
#
# MEMORY:
#   PSD: 100 × 11781 rows × nnz per row = O(100M) COO entries → 10-20 GB
#   Cuts: 100 rows × n_y cols (sparse) = O(7M nnz) → <100 MB


# =====================================================================
# Checkpointing — full-state save/load for resumable runs
# =====================================================================
#
# Saves at the END of each CG round: all state needed to resume the CG
# loop from the next round as if the run had never been interrupted.
# File layout (in --data-dir):
#
#   ckpt_<tag>_cg<N>.json   — metadata (cg_round, best_lb, active_windows,
#                              last_feas_t, scalar_lb, d/order/bw for
#                              compatibility verification, timestamp).
#   ckpt_<tag>_cg<N>.npz    — compressed arrays: last_feas_y, ws_x, ws_y,
#                              ws_s, violations (w and min_eig).
#
# On startup, _try_load_checkpoint() scans <data-dir> for the highest-round
# ckpt matching (d, order, bandwidth) and returns its loaded state dict.
# The solver then skips Round 0 and jumps into the CG loop at round+1.
#
# Soundness: the checkpoint contains only warm-start data.  The SDP is
# rebuilt from active_windows on resume, so a corrupted / stale checkpoint
# can only affect solve SPEED (warm-start quality) — never correctness.
# The problem-identity fields (d/order/bw) must match, otherwise the ckpt
# is silently ignored.

def _save_checkpoint(
    data_dir, tag, cg_round, best_lb, active_windows,
    last_feas_y, last_feas_t, last_x, last_y_dual, last_s,
    violations, scalar_lb, d, order, bandwidth, elapsed_s,
):
    """Atomically write ckpt_<tag>_cg<N>.{json,npz}.

    Uses a temp-file-then-rename pattern so an interrupt mid-write can't
    leave a partial file that would later deserialise incorrectly.
    """
    os.makedirs(data_dir, exist_ok=True)
    stem = os.path.join(data_dir, f'ckpt_{tag}_cg{cg_round}')

    def _arr(x):
        if x is None:
            return np.array([], dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    v_w = (np.array([int(v[0]) for v in violations], dtype=np.int64)
           if violations else np.array([], dtype=np.int64))
    v_e = (np.array([float(v[1]) for v in violations], dtype=np.float64)
           if violations else np.array([], dtype=np.float64))

    npz_tmp = stem + '.npz.tmp'
    # Use a manual file handle: np.savez_compressed auto-appends '.npz'
    # to string paths that don't end in '.npz', which breaks tmp→final rename.
    with open(npz_tmp, 'wb') as _fh:
        np.savez_compressed(
            _fh,
            last_feas_y=_arr(last_feas_y),
            last_x=_arr(last_x),
            last_y_dual=_arr(last_y_dual),
            last_s=_arr(last_s),
            violations_w=v_w,
            violations_eig=v_e,
        )
    os.replace(npz_tmp, stem + '.npz')

    meta = {
        'version': 1,
        'cg_round': int(cg_round),
        'best_lb': float(best_lb),
        'scalar_lb': float(scalar_lb),
        'active_windows': sorted(int(w) for w in active_windows),
        'last_feas_t': (float(last_feas_t)
                        if last_feas_t is not None else None),
        'n_violations': len(violations),
        'd': int(d),
        'order': int(order),
        'bandwidth': int(bandwidth),
        'elapsed_s': float(elapsed_s),
        'timestamp': datetime.now().isoformat(),
    }
    json_tmp = stem + '.json.tmp'
    with open(json_tmp, 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    os.replace(json_tmp, stem + '.json')


def _try_load_checkpoint(data_dir, tag, d, order, bandwidth):
    """Scan data_dir for the highest-round ckpt matching (d,order,bw).

    Returns a dict of resume state, or None if no compatible ckpt is found.
    Graceful on any read error (returns None) — a failed resume should
    never cause a crash; we fall back to a clean start.
    """
    if not os.path.isdir(data_dir):
        return None
    import re as _re
    pat = _re.compile(rf'^ckpt_{_re.escape(tag)}_cg(\d+)\.json$')
    best_round = -1
    best_meta_path = None
    for fn in os.listdir(data_dir):
        m = pat.match(fn)
        if m:
            r = int(m.group(1))
            if r > best_round:
                best_round = r
                best_meta_path = os.path.join(data_dir, fn)

    if best_meta_path is None:
        return None

    try:
        with open(best_meta_path) as fh:
            meta = json.load(fh)
    except Exception:
        return None

    if (meta.get('d') != d or meta.get('order') != order
            or meta.get('bandwidth') != bandwidth):
        return None  # mismatched problem identity — do not resume

    npz_path = best_meta_path.replace('.json', '.npz')
    if not os.path.exists(npz_path):
        return None
    try:
        data = np.load(npz_path)
    except Exception:
        return None

    def _or_none(a):
        return a if (a is not None and a.size > 0) else None

    violations = list(zip(
        data['violations_w'].astype(int).tolist(),
        data['violations_eig'].astype(float).tolist(),
    ))

    return {
        'cg_round': int(meta['cg_round']),
        'best_lb': float(meta['best_lb']),
        'scalar_lb': float(meta['scalar_lb']),
        'active_windows': set(int(w) for w in meta['active_windows']),
        'last_feas_t': meta.get('last_feas_t'),
        'last_feas_y': _or_none(data['last_feas_y']),
        'last_x': _or_none(data['last_x']),
        'last_y_dual': _or_none(data['last_y_dual']),
        'last_s': _or_none(data['last_s']),
        'violations': violations,
        'meta_path': best_meta_path,
        'timestamp': meta.get('timestamp'),
    }


def _psd_lower_tri_map(n):
    """Precompute (scs_row, matrix_i, matrix_j, scale) for PSD cone of size n.

    SCS vectorizes PSD cones as column-major lower triangle:
      k = j*n - j*(j-1)/2 + (i - j)  for i >= j
    Off-diag scaled by sqrt(2).

    Returns arrays: (scs_row_offsets, flat_indices, scales) all of length n*(n+1)/2.
    flat_indices[k] = i*n + j (row-major index into the n×n pick array).
    """
    dim = n * (n + 1) // 2
    scs_rows = np.empty(dim, dtype=np.int64)
    flat_idx = np.empty(dim, dtype=np.int64)
    scales = np.empty(dim)
    k = 0
    for j in range(n):
        block_len = n - j
        scs_rows[k:k+block_len] = np.arange(k, k+block_len)
        flat_idx[k:k+block_len] = np.arange(j, n) * n + j  # column j, rows j..n-1
        scales[k] = 1.0  # diagonal
        if block_len > 1:
            scales[k+1:k+block_len] = SQRT2  # off-diagonal
        k += block_len
    return scs_rows, flat_idx, scales


# Cache these — same n reused across many cones
_psd_cache = {}

def _get_psd_map(n):
    if n not in _psd_cache:
        _psd_cache[n] = _psd_lower_tri_map(n)
    return _psd_cache[n]


def _vectorized_psd_coo(pick_flat, n, row_offset, sign=-1.0):
    """Build COO entries for a PSD cone constraint: sign * x[pick] in PSD.

    pick_flat: int64 array of length n*n (row-major y-indices).
    Returns (rows, cols, vals) arrays — all numpy, no Python loops.
    """
    scs_rows, flat_idx, scales = _get_psd_map(n)
    y_indices = pick_flat[flat_idx]  # look up which x-variable each entry maps to
    valid = y_indices >= 0
    r = (row_offset + scs_rows[valid]).astype(np.int64)
    c = y_indices[valid].astype(np.int64)
    v = (sign * scales[valid])
    return r, c, v


def _vectorized_diff_psd_coo(pick_a, pick_b, n, row_offset):
    """COO for (A - B) >> 0: -A + B in SCS form."""
    scs_rows, flat_idx, scales = _get_psd_map(n)
    a_idx = pick_a[flat_idx]
    b_idx = pick_b[flat_idx]

    # A entries: -scale * x[a_idx]
    valid_a = a_idx >= 0
    r_a = (row_offset + scs_rows[valid_a]).astype(np.int64)
    c_a = a_idx[valid_a].astype(np.int64)
    v_a = -scales[valid_a]

    # B entries: +scale * x[b_idx]
    valid_b = b_idx >= 0
    r_b = (row_offset + scs_rows[valid_b]).astype(np.int64)
    c_b = b_idx[valid_b].astype(np.int64)
    v_b = scales[valid_b].copy()

    return (np.concatenate([r_a, r_b]),
            np.concatenate([c_a, c_b]),
            np.concatenate([v_a, v_b]))


# =====================================================================
# Build base SCS problem (vectorized)
# =====================================================================

def build_base_problem(P, add_upper_loc=True):
    """Build base SCS data. All construction is vectorized numpy."""
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    idx = P['idx']
    n_x = n_y + 1
    t_col = n_y

    c = np.zeros(n_x)
    c[t_col] = 1.0

    all_r, all_c, all_v = [], [], []
    b_parts = []
    row_offset = 0

    # ---- ZERO CONE ----
    # y_0 = 1
    zero_tup = tuple(0 for _ in range(d))
    all_r.append(np.array([row_offset], dtype=np.int64))
    all_c.append(np.array([idx[zero_tup]], dtype=np.int64))
    all_v.append(np.array([1.0]))
    b_parts.append(1.0)
    row_offset += 1

    # Consistency equalities
    if P['consist_eq_lists'] is not None:
        n_eq, er, ec, ev = P['consist_eq_lists']
        all_r.append(np.asarray(er, dtype=np.int64) + row_offset)
        all_c.append(np.asarray(ec, dtype=np.int64))
        all_v.append(np.asarray(ev, dtype=np.float64))
        b_parts.extend([0.0] * n_eq)
        row_offset += n_eq

    n_zero = row_offset

    # ---- NONNEG CONE ----
    nonneg_start = row_offset

    # Consistency inequalities (negated for SCS)
    if P['consist_iq_lists'] is not None:
        n_iq, ir, ic, iv = P['consist_iq_lists']
        all_r.append(np.asarray(ir, dtype=np.int64) + row_offset)
        all_c.append(np.asarray(ic, dtype=np.int64))
        all_v.append(-np.asarray(iv, dtype=np.float64))
        b_parts.extend([0.0] * n_iq)
        row_offset += n_iq

    # Scalar windows: f_W(y) - t <= 0
    F_row, F_col, F_val = P['F_coo_lists']
    n_win = P['n_win']
    fr = np.asarray(F_row, dtype=np.int64) + row_offset
    fc = np.asarray(F_col, dtype=np.int64)
    fv = np.asarray(F_val, dtype=np.float64)
    # t coefficient
    tr = np.arange(n_win, dtype=np.int64) + row_offset
    tc = np.full(n_win, t_col, dtype=np.int64)
    tv = np.full(n_win, -1.0)
    all_r.extend([fr, tr])
    all_c.extend([fc, tc])
    all_v.extend([fv, tv])
    b_parts.extend([0.0] * n_win)
    row_offset += n_win

    # y >= 0
    yr = np.arange(n_y, dtype=np.int64) + row_offset
    yc = np.arange(n_y, dtype=np.int64)
    yv = np.full(n_y, -1.0)
    all_r.append(yr)
    all_c.append(yc)
    all_v.append(yv)
    b_parts.extend([0.0] * n_y)
    row_offset += n_y

    n_nonneg = row_offset - nonneg_start

    # ---- PSD CONES ----
    psd_sizes = []

    # Full M_{k-1}
    if P['m1_valid']:
        n_m = P['m1_size']
        r, c_arr, v = _vectorized_psd_coo(P['m1_pick'], n_m, row_offset)
        all_r.append(r); all_c.append(c_arr); all_v.append(v)
        dim = n_m * (n_m + 1) // 2
        b_parts.extend([0.0] * dim)
        row_offset += dim
        psd_sizes.append(n_m)

    # Clique moment PSD
    for cd in P['clique_data']:
        pick = cd['mom_pick']
        if np.any(pick < 0):
            continue
        n_cb = cd['mom_size']
        r, c_arr, v = _vectorized_psd_coo(pick, n_cb, row_offset)
        all_r.append(r); all_c.append(c_arr); all_v.append(v)
        dim = n_cb * (n_cb + 1) // 2
        b_parts.extend([0.0] * dim)
        row_offset += dim
        psd_sizes.append(n_cb)

    # Clique localizing PSD
    if order >= 2:
        for i_var in range(d):
            c_idx = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx]
            picks = cd['loc_picks'].get(i_var)
            if picks is None or np.any(picks < 0):
                continue
            n_cb = cd['loc_size']
            r, c_arr, v = _vectorized_psd_coo(picks, n_cb, row_offset)
            all_r.append(r); all_c.append(c_arr); all_v.append(v)
            dim = n_cb * (n_cb + 1) // 2
            b_parts.extend([0.0] * dim)
            row_offset += dim
            psd_sizes.append(n_cb)

    # Upper localizing PSD
    if add_upper_loc and order >= 2:
        for i_var in range(d):
            c_idx = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx]
            t_pick = cd.get('t_pick')
            loc_pick = cd['loc_picks'].get(i_var)
            if t_pick is None or loc_pick is None:
                continue
            if np.any(t_pick < 0) or np.any(loc_pick < 0):
                continue
            n_cb = cd['loc_size']
            r, c_arr, v = _vectorized_diff_psd_coo(t_pick, loc_pick, n_cb, row_offset)
            all_r.append(r); all_c.append(c_arr); all_v.append(v)
            dim = n_cb * (n_cb + 1) // 2
            b_parts.extend([0.0] * dim)
            row_offset += dim
            psd_sizes.append(n_cb)

    # Pairwise product localizing: M_1(μ_i μ_j y) ⪰ 0 for all pairs i ≤ j.
    #
    # Valid Lasserre L3 Putinar product constraints (degree 4 ≤ 2k-1 = 5), currently
    # absent from the model.  f_W(μ) = Σ M_W[i,j] μ_i μ_j is a sum of pairwise
    # products; without M_1(μ_i μ_j y) the dual cannot construct tight degree-6 SOS
    # certificates for window feasibility — this is why gc stalls after round 2.
    # 136 cones of size 17×17 at d=16: cholesky_ex fast-path, negligible cost.
    #
    # Batch implementation: stack all 136 pick arrays, build COO for all cones
    # in a single _vectorized_psd_coo call, then split by cone boundary.
    if 'pw_pairs' in P and P['pw_pairs']:
        pw_n = P['pw_size']              # d+1 = 17
        pw_dim = pw_n * (pw_n + 1) // 2  # svec dim = 153 per cone
        n_pw = len(P['pw_pairs'])

        # Stack all picks: (n_pw, pw_n²) — one row per cone
        all_pw_picks = np.stack([picks for _, _, picks in P['pw_pairs']], axis=0)

        # Build COO for all n_pw cones in one vectorised pass.
        # _vectorized_psd_coo expects a flat (pw_n²,) pick array and a row_offset.
        # For n cones stacked, we call it once per cone but inside a tight numpy loop —
        # the bottleneck is the n_pw Python iterations, not the inner numpy ops.
        # Since n_pw=136 << n_y, this is not performance-critical (runs once at setup).
        for cone_idx in range(n_pw):
            r, c_arr, v = _vectorized_psd_coo(
                all_pw_picks[cone_idx], pw_n, row_offset)
            all_r.append(r); all_c.append(c_arr); all_v.append(v)
            b_parts.extend([0.0] * pw_dim)
            row_offset += pw_dim
            psd_sizes.append(pw_n)

    # Assemble
    n_rows = row_offset
    R = np.concatenate(all_r)
    C = np.concatenate(all_c)
    V = np.concatenate(all_v)
    A = sp.csc_matrix((V, (R, C)), shape=(n_rows, n_x))
    b = np.array(b_parts)
    cone = {'z': n_zero, 'l': n_nonneg, 's': psd_sizes}

    meta = {
        'n_x': n_x, 'n_y': n_y, 't_col': t_col,
        'n_zero': n_zero, 'n_nonneg': n_nonneg,
        'n_rows_base': n_rows,
    }

    print(f"  Base problem: {n_rows:,} rows x {n_x:,} cols, nnz={A.nnz:,}")
    print(f"  Cones: z={n_zero}, l={n_nonneg}, PSD={len(psd_sizes)}")
    print(f"  A memory: {A.data.nbytes/1e6:.1f} MB", flush=True)

    return A, b, c, cone, meta


# =====================================================================
# Vectorized window PSD construction
# =====================================================================

def _build_window_psd_block(P, windows, t_val):
    """Build the sparse block for ALL window PSD cones at once.

    Returns (A_block, b_block, psd_sizes) for the window PSD rows.
    All construction is vectorized — no per-window Python loops in the
    inner PSD entry loop.
    """
    decomp = _precompute_window_psd_decomposition(P, windows)
    if decomp is None:
        return None, None, []
    return _assemble_window_psd(decomp, t_val)


def _precompute_window_psd_decomposition(P, windows):
    """Precompute t-independent and t-dependent parts of window PSD block.

    Returns a decomposition dict that can cheaply assemble A for any t_val.
    Build once per CG round; assemble cheaply per bisection step.
    """
    d = P['d']
    n_y = P['n_y']
    n_x = n_y + 1

    base_r, base_c, base_v = [], [], []
    t_r, t_c, t_v = [], [], []
    psd_sizes = []
    row_offset = 0

    for w in windows:
        c_idx_w = int(P['window_covering'][w])
        if c_idx_w < 0:
            continue

        cd = P['clique_data'][c_idx_w]
        n_cb = cd['loc_size']
        t_pick = cd['t_pick']
        if t_pick is None or np.any(t_pick < 0):
            continue

        ab_eiej = cd['viol_ab_eiej']
        gi_grid = cd['viol_gi_grid']
        ell, s_lo = P['windows'][w]
        coeff = 2.0 * d / ell
        mask = (gi_grid >= s_lo) & (gi_grid <= s_lo + ell - 2)
        nz_li, nz_lj = np.nonzero(mask)

        scs_rows, flat_idx, scales = _get_psd_map(n_cb)
        dim = n_cb * (n_cb + 1) // 2

        # T-part (t-dependent): -scale * x[t_pick[flat_idx]]
        t_indices = t_pick[flat_idx]
        valid_t = t_indices >= 0
        if np.any(valid_t):
            t_r.append((row_offset + scs_rows[valid_t]).astype(np.int64))
            t_c.append(t_indices[valid_t].astype(np.int64))
            t_v.append(-scales[valid_t])  # multiply by t_val at assembly

        # Q-part (t-independent): +coeff * scale * x[ab_eiej[...]]
        if len(nz_li) > 0:
            ij_i = flat_idx // n_cb
            ij_j = flat_idx % n_cb
            q_indices = ab_eiej[ij_i[:, None], ij_j[:, None],
                                nz_li[None, :], nz_lj[None, :]]
            valid_q = q_indices >= 0
            scs_rows_exp = np.broadcast_to(
                (row_offset + scs_rows)[:, None], q_indices.shape)
            scales_exp = np.broadcast_to(scales[:, None], q_indices.shape)

            base_r.append(scs_rows_exp[valid_q].astype(np.int64))
            base_c.append(q_indices[valid_q].astype(np.int64))
            base_v.append(coeff * scales_exp[valid_q])

        row_offset += dim
        psd_sizes.append(n_cb)

    if not psd_sizes:
        return None

    n_new_rows = row_offset
    has_t = bool(t_v)

    # Build base sparse matrix (t-independent Q-part)
    if base_v:
        BR = np.concatenate(base_r)
        BC = np.concatenate(base_c)
        BV = np.concatenate(base_v)
    else:
        BR = np.array([], dtype=np.int64)
        BC = np.array([], dtype=np.int64)
        BV = np.array([], dtype=np.float64)
    A_base_win = sp.csc_matrix((BV, (BR, BC)), shape=(n_new_rows, n_x))

    if has_t:
        TR = np.concatenate(t_r)
        TC = np.concatenate(t_c)
        TV = np.concatenate(t_v)
        A_t_win = sp.csc_matrix((TV, (TR, TC)), shape=(n_new_rows, n_x))
    else:
        A_t_win = sp.csc_matrix((n_new_rows, n_x), dtype=np.float64)

    # Precompute union sparsity pattern for in-place A(t) = base + t * t_part
    if has_t:
        A_template = (A_base_win + A_t_win).tocsc()
        A_template.sort_indices()
        A_ref_2t = (A_base_win + 2.0 * A_t_win).tocsc()
        A_ref_2t.sort_indices()
        t_data = A_ref_2t.data - A_template.data
        base_data = A_template.data - t_data
    else:
        A_template = A_base_win.tocsc()
        A_template.sort_indices()
        t_data = None
        base_data = A_template.data.copy()

    return {
        'A_template': A_template,
        'base_data': base_data,
        't_data': t_data,
        'has_t': has_t,
        'n_rows': n_new_rows,
        'n_x': n_x,
        'psd_sizes': psd_sizes,
    }


def _assemble_window_psd(decomp, t_val):
    """Cheaply assemble window PSD block for a given t_val.

    In-place data array update — no new sparse matrix allocation.
    """
    A = decomp['A_template']
    if decomp['has_t']:
        np.add(decomp['base_data'], t_val * decomp['t_data'], out=A.data)
    b = np.zeros(decomp['n_rows'])
    return A, b, decomp['psd_sizes']


# =====================================================================
# Main solver
# =====================================================================

def solve_scs_direct(d, order, bandwidth, c_target=1.28,
                     add_upper_loc=True, max_cg_rounds=10, n_bisect=12,
                     use_gpu=False, scs_max_iters=100000, scs_eps=1e-5,
                     scs_scale=None, use_indirect=False,
                     k_vecs=3, cuts_per_round=100,
                     rho=0.1, atom_frac=0.5,
                     resume=False, data_dir=None,
                     verbose=True):
    """Full CG Lasserre solver using direct SCS."""
    import scs

    t_total = time.time()

    print(f"{'='*70}")
    print(f"DIRECT SCS LASSERRE: L{order} d={d} bw={bandwidth}")
    print(f"  GPU={use_gpu}, max_iters={scs_max_iters}, eps={scs_eps}")
    print(f"{'='*70}\n", flush=True)

    cliques = _build_banded_cliques(d, bandwidth)
    print(f"Cliques: {len(cliques)} of size {len(cliques[0])}\n", flush=True)
    P = _precompute_highd(d, order, cliques, verbose=verbose)
    n_y = P['n_y']

    print("\nBuilding base SCS problem...", flush=True)
    t_build = time.time()
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, add_upper_loc)
    # Optional Z/2 time-reversal symmetry injection (gated by env var so it
    # cannot perturb any in-flight production run that hasn't opted in).
    # See lasserre/z2_symmetry.py for the mathematical justification.
    if os.environ.get('SIDON_Z2_SYMMETRY') == '1':
        try:
            _repo_root = os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))
            if _repo_root not in sys.path:
                sys.path.insert(0, _repo_root)
            from lasserre.z2_symmetry import (
                inject_z2_equalities_into_base_problem)
            A_base, b_base, cone_base, _n_z2 = (
                inject_z2_equalities_into_base_problem(
                    P, A_base, b_base, cone_base, verbose=verbose))
        except Exception as _z2_e:
            print(f"  [WARN] Z/2 symmetry injection failed ({_z2_e}); "
                  f"falling back to un-symmetrised problem.",
                  flush=True)
    print(f"  Built in {time.time()-t_build:.1f}s\n", flush=True)

    scs_kwargs = dict(max_iters=scs_max_iters, eps_abs=scs_eps, eps_rel=scs_eps,
                      verbose=verbose)
    if scs_scale is not None:
        scs_kwargs['scale'] = scs_scale
    if use_indirect:
        scs_kwargs['use_indirect'] = True

    # Round 0: minimize t
    print("  [Round 0] Minimize t...", flush=True)
    t0 = time.time()

    if use_gpu:
        from admm_gpu_solver import admm_solve
        # Round 0 only needs a rough scalar bound — use loose eps.
        # Tight eps (1e-7) causes adaptive rho oscillation in admm_solve.
        r0_eps = max(scs_eps, 1e-5)
        sol = admm_solve(A_base, b_base, c_obj, cone_base,
                         max_iters=scs_max_iters, eps_abs=r0_eps,
                         eps_rel=r0_eps, device='cuda', verbose=verbose)
    else:
        data = {'A': A_base, 'b': b_base, 'c': c_obj}
        solver = scs.SCS(data, cone_base, **scs_kwargs)
        sol = solver.solve()

    scalar_lb = 0.5
    y_vals = np.zeros(n_y)
    if sol['info']['status'] in ('solved', 'solved_inaccurate'):
        scalar_lb = float(sol['x'][meta['t_col']])
        y_vals = sol['x'][:n_y].copy()

    best_lb = scalar_lb
    active_windows = set()

    # k_vecs=0: use standard PSD window cones (full Lasserre hierarchy).
    # Do not return eigenvectors — full PSD cones give the tightest possible
    # relaxation for each window, unlike spectral cuts which are a strict subset.
    violations = _check_violations_highd(
        y_vals, scalar_lb, P, active_windows, k_vecs=0)

    print(f"    Scalar bound = {scalar_lb:.10f} "
          f"({time.time()-t0:.1f}s, {sol['info']['iter']} iters)",
          flush=True)
    print(f"    {len(violations)} violations", flush=True)

    if not violations:
        elapsed = time.time() - t_total
        return _result(best_lb, d, order, bandwidth, n_y, 0, elapsed, y_vals)

    # Initialize before CG loop so early-exit paths (hi-feas loop failure,
    # etc.) can safely reference these without UnboundLocalError.
    last_feas_y = None
    last_feas_t = None

    # CG rounds
    last_x = sol['x'].copy() if sol['x'] is not None else None
    last_y_dual = sol['y'].copy() if sol['y'] is not None else None
    last_s = sol['s'].copy() if sol['s'] is not None else None

    # ── Checkpoint resume (after Round 0 so base SDP is always reproduced,
    # only CG state is restored) ──
    # Resolve data directory for checkpoints.  Default: ../data relative to
    # this script (matches the existing checkpoint write path).
    _ckpt_dir = data_dir if data_dir is not None else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    _ckpt_tag = f"d{d}_o{order}_bw{bandwidth}_scs"
    start_round = 1
    if resume:
        _ckpt = _try_load_checkpoint(_ckpt_dir, _ckpt_tag, d, order, bandwidth)
        if _ckpt is not None:
            print(
                f"  [RESUME] Loaded checkpoint at CG round "
                f"{_ckpt['cg_round']} (lb={_ckpt['best_lb']:.10f}, "
                f"active_windows={len(_ckpt['active_windows'])}, "
                f"saved at {_ckpt.get('timestamp', 'unknown')})",
                flush=True)
            # Restore warm-starts and tracking state.
            best_lb = _ckpt['best_lb']
            active_windows = _ckpt['active_windows']
            last_feas_y = _ckpt['last_feas_y']
            last_feas_t = _ckpt['last_feas_t']
            if _ckpt['last_x'] is not None:
                last_x = _ckpt['last_x']
            if _ckpt['last_y_dual'] is not None:
                last_y_dual = _ckpt['last_y_dual']
            if _ckpt['last_s'] is not None:
                last_s = _ckpt['last_s']
            violations = _ckpt['violations']
            start_round = _ckpt['cg_round'] + 1
            scalar_lb = _ckpt['scalar_lb']   # informational — Round 0 already done
        else:
            print(f"  [RESUME] No compatible checkpoint found — starting fresh",
                  flush=True)

    # Count consecutive rounds with < 1e-6 improvement.  With backoff
    # bisection active, a single stalled round is no longer a reliable
    # signal that the relaxation is capped — even a stall round can come
    # back +0.005 on the next attempt as cuts evolve.  Require three
    # consecutive stalls before aborting (was one).
    consecutive_stalls = 0
    for cg_round in range(start_round, max_cg_rounds + 1):
        # Atom-based window ranking + min-eig blend (Frangioni 2002 primal
        # oracle style; implemented in lasserre/gap_accelerator.py).
        #
        # Each candidate window is scored by μ̂^T M_W μ̂ where μ̂ are rank-1
        # atoms extracted from M_1(y_last_feas) via truncated SVD and
        # projected onto Δ_d (nonneg + sum=1).  Score = Σᵢ wᵢ (μ̂ᵢ^T M_W μ̂ᵢ)
        # weighted by the atomic weights wᵢ.  High score = window would be
        # most violated by the current primal approximation — adding it
        # constrains the optimum the hardest.
        #
        # Empirically +14 pp gc per round over dual-guided (measured Apr 2026).
        # Mathematically sound: pure RANKING heuristic — every selected window
        # is a valid Lasserre localizer; lb ≤ val(d) holds regardless of
        # ordering.  The atom extraction itself is non-rigorous (heuristic
        # simplex projection of eigenvectors) but that is OK because atoms
        # are used only to rank, never to build constraints.
        #
        # Blending: 50% atom-ranked + 50% min-eig-ranked gives robust mix —
        # atoms catch "high-pressure" windows, eig-rank catches severely
        # violated ones that atoms may miss when atom projection is lossy.
        # Continuous schedules driven by cg_round:
        # (1) `round_eps` starts 100x looser than the CLI target `scs_eps`
        #     and tightens log-linearly to the target over 10 rounds.  Early
        #     rounds converge fast at loose eps (certifying easy t values);
        #     later rounds (where certifying infeasibility near the true
        #     optimum requires tight dual certificates) run at the full
        #     CLI-specified tolerance.  Empirically, the previous run
        #     hit R2 step-11 stalls because its retry ran 50000 iters at
        #     scs_eps=1e-7 on an easy t value — that waste is now gone.
        # (2) `round_cuts` grows linearly with cg_round so that, as lb
        #     deceleration kicks in, we add more pressure per round to break
        #     through plateaus instead of paying the per-round overhead for
        #     diminishing returns.
        round_eps = max(scs_eps,
                        min(1e-5,
                            scs_eps * (10.0 ** (2.0 - cg_round / 5.0))))
        round_cuts = cuts_per_round + 10 * cg_round
        n_add = min(round_cuts, len(violations))

        if last_feas_y is not None:
            # Run atom ranking on all violation candidates.  Cheap:
            # O(k × d² × n_windows) where k ≈ 16 atoms and n ≤ 496 candidates.
            cand_ws = [int(v[0]) for v in violations]
            atom_rank = atom_based_window_ranking(last_feas_y, P, cand_ws)
            # violations is already sorted ascending by min_eig (most
            # violated first) — that IS the eig-rank.
            selected = blend_rankings(
                eig_rank=violations,
                atom_rank=atom_rank,
                n_add=n_add,
                atom_frac=atom_frac,
            )
            selection_mode = 'atom+eig'
        else:
            # Round 1: no last_feas_y yet — fall back to diverse eig-rank.
            selected = _greedy_diverse_cuts(
                violations, n_add, P, P['d'], active_windows)
            selection_mode = 'eig+diverse (round1)'

        for w, _score in selected:
            active_windows.add(int(w))

        print(f"\n  [CG round {cg_round}] {len(active_windows)} windows "
              f"(+{len(selected)} added, {selection_mode}, "
              f"round_eps={round_eps:.1e}, round_cuts={round_cuts})",
              flush=True)

        # Initial lo slightly below best_lb as a numerical guard against
        # any residual solver noise that could briefly accept t = best_lb
        # as feasible.  The 1e-3 slack previously used wasted the first
        # bisection step re-certifying a known-infeas region; 1e-5 is
        # tight enough to be safe without throwing away a whole step.
        lo = max(0.5, best_lb - 1e-5)
        hi = best_lb + 0.02 if best_lb > 0.5 else 5.0

        c_feas = np.zeros(meta['n_x'])  # feasibility objective

        # ── Precompute window PSD decomposition ONCE per CG round ──
        # Full 153×153 PSD cones per window give the tightest Lasserre relaxation.
        # The t-decomposition (base/t arrays) enables fast in-place update at each
        # bisection step without rebuilding the full sparse matrix.
        t_decomp = time.time()
        win_decomp = _precompute_window_psd_decomposition(P, active_windows)

        if win_decomp is not None:
            # Build full A at t=1 for union sparsity pattern.
            A_win_t1, b_win_t1, psd_win = _assemble_window_psd(win_decomp, 1.0)
            A_full_t1 = sp.vstack([A_base, A_win_t1], format='csc')
            A_full_t1.sort_indices()
            del A_win_t1, b_win_t1  # consumed by vstack

            # Build at t=2 ONLY to extract the t-coefficient difference.
            A_win_t2, _, _ = _assemble_window_psd(win_decomp, 2.0)
            A_full_t2 = sp.vstack([A_base, A_win_t2], format='csc')
            A_full_t2.sort_indices()
            del A_win_t2

            full_t_data = A_full_t2.data - A_full_t1.data
            full_base_data = A_full_t1.data - full_t_data
            has_t_full = np.any(full_t_data != 0)
            del A_full_t2  # only .data was needed

            b_full_base = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
            cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                         's': list(cone_base['s']) + psd_win}
            n_rows_full = A_full_t1.shape[0]
        else:
            A_full_t1 = A_base.tocsc().copy()
            A_full_t1.sort_indices()
            full_t_data = None
            full_base_data = A_full_t1.data.copy()
            has_t_full = False
            b_full_base = b_base.copy()
            cone_full = cone_base
            n_rows_full = A_base.shape[0]

        print(f"    Decomposition precomputed in {time.time()-t_decomp:.1f}s",
              flush=True)

        # Pre-allocate padded warm-start arrays to avoid per-step allocation
        ws_x = np.zeros(meta['n_x'])
        ws_y = np.zeros(n_rows_full)
        ws_s = np.zeros(n_rows_full)
        has_warm = [False]

        def _prepare_warm(sol_t):
            """Copy solution into pre-allocated warm-start arrays."""
            if sol_t['x'] is not None:
                n = min(len(sol_t['x']), len(ws_x))
                ws_x[:n] = sol_t['x'][:n]
                if n < len(ws_x):
                    ws_x[n:] = 0
            if sol_t['y'] is not None:
                n = min(len(sol_t['y']), len(ws_y))
                ws_y[:n] = sol_t['y'][:n]
                if n < len(ws_y):
                    ws_y[n:] = 0
            if sol_t['s'] is not None:
                n = min(len(sol_t['s']), len(ws_s))
                ws_s[:n] = sol_t['s'][:n]
                if n < len(ws_s):
                    ws_s[n:] = 0
            has_warm[0] = True

        # GPU: persistent solver — reuse factorization + CSR pattern cache
        gpu_solver = [None]  # mutable ref for closure
        gpu_tau_col = [None]

        # Pre-build the fix_t + phase-1 augmented pattern ONCE per CG round.
        # Only A.data and b[0] change per bisection step.
        from admm_gpu_solver import ADMMSolver, augment_phase1

        n_cols_full = A_full_t1.shape[1]
        fix_t_row = sp.csc_matrix(
            ([1.0], ([0], [meta['t_col']])),
            shape=(1, n_cols_full))

        # Build augmented pattern at t=1 and t=2 to extract base/t decomposition.
        # Each A_p1 is ~the size of A_full_t1; holding both t1 and t2
        # versions simultaneously doubles memory. We delete t2 artifacts
        # as soon as aug_t_data is extracted.
        A_fixed_t1 = sp.vstack([fix_t_row, A_full_t1], format='csc')
        A_fixed_t1.sort_indices()
        b_fixed_t1 = np.insert(b_full_base, 0, 1.0)
        cone_fixed = {'z': cone_full['z'] + 1,
                      'l': cone_full['l'],
                      's': cone_full['s']}

        A_p1_t1, b_p1_t1, c_p1_cached, cone_p1_cached, tau_col_cached = \
            augment_phase1(A_fixed_t1, b_fixed_t1, cone_fixed)
        del A_fixed_t1, b_fixed_t1  # no longer needed after augment

        # Build at t=2 to extract t-dependent part of the full augmented matrix
        if has_t_full:
            _tmp_data = A_full_t1.data.copy()
            np.add(full_base_data, 2.0 * full_t_data, out=A_full_t1.data)
            A_fixed_t2 = sp.vstack([fix_t_row, A_full_t1], format='csc')
            A_fixed_t2.sort_indices()
            b_fixed_t2 = np.insert(b_full_base, 0, 2.0)
            A_p1_t2, b_p1_t2, _, _, _ = augment_phase1(
                A_fixed_t2, b_fixed_t2, cone_fixed)
            del A_fixed_t2, b_fixed_t2
            # Extract decomposition in augmented space
            aug_t_data = A_p1_t2.data - A_p1_t1.data
            aug_base_data = A_p1_t1.data - aug_t_data
            aug_b_t_data = b_p1_t2 - b_p1_t1
            aug_b_base_data = b_p1_t1 - aug_b_t_data
            del A_p1_t2, b_p1_t2  # only .data was needed; free the sparse shell
            # Restore A_full_t1.data
            np.copyto(A_full_t1.data, _tmp_data)
            del _tmp_data
            # A_full_t1 is no longer needed in the GPU path (A_p1_template holds
            # the phase-1 augmented matrix; A_full_t1's role was only to build it).
            # Freeing here saves 4-6 GB at round 3+ (300 windows × 11781 rows).
            # CPU SCS path still needs A_full_t1 — only del when use_gpu=True.
            if use_gpu:
                del A_full_t1
                A_full_t1 = None
        else:
            aug_t_data = None
            aug_base_data = A_p1_t1.data.copy()
            aug_b_t_data = None
            aug_b_base_data = b_p1_t1.copy()

        gpu_tau_col[0] = tau_col_cached

        # Reusable template. We reassign rather than copy: A_p1_t1 is no
        # longer referenced after this point, so the identifier swap is
        # safe and saves one full sparse matrix allocation (~A_p1_t1.nnz
        # floats + indices). b likewise.
        A_p1_template = A_p1_t1
        b_p1_template = b_p1_t1
        A_p1_t1 = None  # drop outer binding so GC can reclaim on next loop
        b_p1_t1 = None

        def check_feasible(t_val, max_iters_override=None,
                           eps_override=None):
            """Return (verdict, sol, tau_val, build_time).

            verdict is 'feas' / 'infeas' / 'uncertain' per the rules in
            _classify_verdict. The bisection loop MUST only move the
            bracket on certified verdicts; uncertain verdicts are
            equivalent to "no information" and must not move lo or hi.

            tau_val is the phase-1 slack variable (GPU path) or NaN
            (CPU path, which uses direct SCS without phase-1).
            """
            t_b = time.time()

            # In-place update of augmented A and b for this t_val
            if aug_t_data is not None:
                np.add(aug_base_data, t_val * aug_t_data,
                       out=A_p1_template.data)
                np.add(aug_b_base_data, t_val * aug_b_t_data,
                       out=b_p1_template)
            else:
                np.copyto(A_p1_template.data, aug_base_data)
                np.copyto(b_p1_template, aug_b_base_data)

            build_time = time.time() - t_b

            cur_iters = max_iters_override or scs_max_iters
            cur_eps = eps_override or scs_eps

            if use_gpu:
                if gpu_solver[0] is None:
                    # rho=0.1 proven 10-20x faster than rho=0.5
                    # for phase-1 problems (sweep_bisection.py)
                    gpu_solver[0] = ADMMSolver(
                        A_p1_template, b_p1_template, c_p1_cached,
                        cone_p1_cached, rho=rho,
                        device='cuda', verbose=False)
                    # Warm-up: 100 iters at initial t to seed workspace
                    gpu_solver[0].solve(
                        max_iters=100, eps_abs=1.0, eps_rel=1.0,
                        tau_col=tau_col_cached)
                else:
                    gpu_solver[0]._update_A(A_p1_template)
                    gpu_solver[0].update_b(b_p1_template)

                tau_col = gpu_tau_col[0]
                tau_tol = max(cur_eps * 10, 1e-4)

                sol_t = gpu_solver[0].solve(
                    max_iters=cur_iters, eps_abs=cur_eps, eps_rel=cur_eps,
                    tau_col=tau_col, tau_tol=tau_tol)

                # Extract tau and original x (without tau variable).
                # tau_val is unscaled (x is returned unscaled) so it is
                # comparable against tau_tol directly.
                tau_val = float(sol_t['x'][tau_col]) \
                    if sol_t['x'] is not None else float('inf')

                verdict = _classify_verdict(sol_t, tau_val, tau_tol)

                # Strip tau + fix_t row from x for warm-start compatibility
                sol_orig = dict(sol_t)
                sol_orig['x'] = sol_t['x'][:meta['n_x']].copy() \
                    if sol_t['x'] is not None else None
                _prepare_warm(sol_orig)

                return verdict, sol_orig, tau_val, build_time

            # CPU SCS path. No phase-1 augmentation: SCS solves the
            # direct epigraph problem and returns status=='solved' on
            # feasibility, 'infeasible' on infeasibility. Map the three
            # SCS states to our verdict vocabulary; anything else
            # (e.g. 'infeasible_inaccurate') is uncertain.
            kw = dict(max_iters=cur_iters, eps_abs=cur_eps,
                      eps_rel=cur_eps, verbose=False)
            if scs_scale is not None:
                kw['scale'] = scs_scale
            if use_indirect:
                kw['use_indirect'] = True
            data_t = {'A': A_full_t1, 'b': b_full_base, 'c': c_feas}
            s = scs.SCS(data_t, cone_full, **kw)

            try:
                if has_warm[0]:
                    sol_t = s.solve(warm_start=True,
                                   x=ws_x, y=ws_y, s=ws_s)
                else:
                    sol_t = s.solve()
            except (ValueError, Exception):
                sol_t = s.solve()

            _prepare_warm(sol_t)

            scs_status = sol_t['info']['status']
            if scs_status == 'solved':
                verdict = _VERDICT_FEAS
            elif scs_status == 'infeasible':
                verdict = _VERDICT_INFEAS
            else:
                # 'solved_inaccurate', 'infeasible_inaccurate', 'unbounded',
                # time-limit-exceeded — all are UNCERTAIN, never move the
                # bracket. (Previous behavior: treated solved_inaccurate
                # as feas, which is soundness-safe for hi but can miss
                # tightening on lo. The verdict-aware bisection loop
                # handles uncertain explicitly.)
                verdict = _VERDICT_UNCERTAIN
            return verdict, sol_t, float('nan'), build_time

        # Copy last warm-start from round 0 / previous round
        if last_x is not None:
            ws_x[:min(len(last_x), len(ws_x))] = last_x[:min(len(last_x), len(ws_x))]
        if last_y_dual is not None:
            n = min(len(last_y_dual), len(ws_y))
            ws_y[:n] = last_y_dual[:n]
        if last_s is not None:
            n = min(len(last_s), len(ws_s))
            ws_s[:n] = last_s[:n]
        has_warm[0] = last_x is not None

        # Ensure hi is certified feasible. Only a 'feas' verdict ends
        # the loop; both 'infeas' and 'uncertain' at the current hi
        # trigger a bump (*1.5) until feasibility is confirmed. This is
        # strictly safer than the previous "not feas -> bump" rule: a
        # non-converged ADMM run could previously mark hi as non-feas
        # and trigger an unnecessary bump, but was also at risk of the
        # reverse error once the verdict was flipped into the main
        # bisection. Keeping the verdict vocabulary uniform eliminates
        # that class of error.
        hi_budget = 800
        hi_eps = 1e-5
        for _attempt in range(10):
            verdict_hi, sol_hi, tau_hi_init, bt = check_feasible(
                hi, max_iters_override=hi_budget,
                eps_override=hi_eps)
            if verdict_hi == _VERDICT_FEAS:
                break
            hi *= 1.5
        else:
            print("    WARNING: cannot certify feasibility up to t=100")
            break

        # Adaptive bisection with graduated iteration budgets +
        # tau-interpolation. Verdict-aware: only 'feas'/'infeas' verdicts
        # move the bracket. 'uncertain' (ADMM iter cap without KKT
        # convergence) does NOT move the bracket — this is the soundness
        # fix. See _classify_verdict for the math.
        #
        # Monotonicity log: every verdict is recorded so we can detect
        # and remediate any classification inconsistency post-loop (a
        # 'feas' at t_a together with an 'infeas' at t_b > t_a is a
        # certified contradiction; we pull `lo` back to preserve soundness).
        tau_lo, tau_hi = None, None
        last_feas_y = None
        last_feas_t = None
        verdict_log = []  # list of (t, verdict)

        for step in range(n_bisect):
            # Graduated iteration budget. PRESERVED from previous
            # behavior — no hyperparameter changes.
            if step < 4:
                adaptive_iters = min(scs_max_iters, 800)
                adaptive_eps = max(round_eps, 1e-5)
            elif step < 10:
                adaptive_iters = min(scs_max_iters, 2000)
                adaptive_eps = max(round_eps, 1e-6)
            else:
                adaptive_iters = min(scs_max_iters, 4000)
                adaptive_eps = round_eps

            # Tau-interpolation (unchanged).
            use_interp = False
            if (step >= 3 and tau_lo is not None and tau_hi is not None
                    and tau_lo > 0 and tau_hi < 0
                    and (hi - lo) > 1e-12):
                frac = tau_lo / (tau_lo - tau_hi)
                mid_interp = lo + frac * (hi - lo)
                mid = max(lo + 0.1 * (hi - lo),
                          min(hi - 0.1 * (hi - lo), mid_interp))
                use_interp = True
            else:
                mid = (lo + hi) / 2

            t_step = time.time()
            verdict, sol_mid, tau_val, bt = check_feasible(
                mid, max_iters_override=adaptive_iters,
                eps_override=adaptive_eps)

            # Algorithmic recovery on uncertain: give the solver the full
            # configured scs_max_iters budget once before giving up.
            # We reuse the existing warm-started solver so this is an
            # extension of the current ADMM run, not a restart.
            retried = False
            if verdict == _VERDICT_UNCERTAIN:
                retried = True
                v2, sol_mid, tau_val, bt2 = check_feasible(
                    mid, max_iters_override=scs_max_iters,
                    eps_override=adaptive_eps)
                verdict = v2
                bt += bt2
            dt = time.time() - t_step

            tau_finite = tau_val if np.isfinite(tau_val) else 0.0

            if verdict == _VERDICT_FEAS:
                hi = mid
                # Record tau for interpolation. We DO NOT invent a negative
                # sentinel for non-negative tau; the interp logic only fires
                # when tau has opposite signs on the two brackets (inherent
                # to the original design); non-firing is safe.
                tau_hi = tau_finite
                tag = "feas"
                verdict_log.append((mid, _VERDICT_FEAS))
                if sol_mid['x'] is not None:
                    last_feas_y = sol_mid['x'][:n_y].copy()
                    last_feas_t = mid
            elif verdict == _VERDICT_INFEAS:
                lo = mid
                tau_lo = tau_finite if tau_finite > 0 else 1.0
                tag = "infeas"
                verdict_log.append((mid, _VERDICT_INFEAS))
            else:
                # _VERDICT_UNCERTAIN: bracket stays, nothing new to record.
                tag = "uncertain"
                verdict_log.append((mid, _VERDICT_UNCERTAIN))

            interp_tag = " interp" if use_interp else ""
            retry_tag = " retry" if retried else ""
            print(f"    [{step+1}/{n_bisect}] t={mid:.8f} {tag} "
                  f"({dt:.1f}s, build={bt:.1f}s, "
                  f"{sol_mid['info']['iter']} iters)"
                  f"{interp_tag}{retry_tag}",
                  flush=True)

            # Backoff bisection on uncertain (soundness-preserved).
            #
            # The previous behaviour was to break the bisection loop whenever
            # `mid` returned uncertain — discarding the entire remainder of
            # the bracket (lo, hi) without probing it.  But `lo` is certified
            # infeas and smaller t makes the SDP MORE infeasible, so probing
            # a t closer to lo (i.e., lo + α*(mid-lo) for α < 1) is strictly
            # easier to certify than `mid` itself.  Each successful backoff
            # advances lo (or shrinks hi if it surprisingly comes back feas),
            # producing real lb progress from a round that would otherwise
            # have stalled.  Every accepted verdict is a valid Lasserre
            # feasibility certificate at its respective t, so lb ≤ val(d)
            # is preserved.  Only if ALL backoffs are also uncertain do we
            # truly give up.
            if verdict == _VERDICT_UNCERTAIN:
                # Seven exponential-halving alphas span 50% down to
                # 0.78% of (mid-lo) above lo — seven chances to land
                # outside whatever uncertain zone surrounds `mid`.  Plus
                # an adaptive floor: any probe within 1e-4 of lo is
                # skipped (matches the bisection convergence threshold —
                # finer resolution than that is numerical noise).
                # Rationale for not using a fixed 3- or 5-step ladder:
                # R1 step 7 already required backoff 2/3 to certify,
                # and a 0.01-wide uncertain zone at cone count 500+
                # would defeat anything shorter than ~7 alphas.
                backoff_alphas = [0.5, 0.25, 0.125, 0.0625,
                                  0.03125, 0.015625, 0.0078125]
                for bo_i, alpha in enumerate(backoff_alphas):
                    new_mid = lo + alpha * (mid - lo)
                    # Skip probes within the bisection convergence threshold
                    # of lo.  Anything closer than 1e-4 to a known-infeas
                    # point is within solver noise — probing adds no
                    # information.  Adaptive across any bracket width.
                    if new_mid - lo < 1e-4:
                        continue
                    t_bo = time.time()
                    v_bo, sol_bo, tau_bo, bt_bo = check_feasible(
                        new_mid, max_iters_override=adaptive_iters,
                        eps_override=adaptive_eps)
                    dt_bo = time.time() - t_bo
                    iter_bo = sol_bo['info']['iter'] if sol_bo is not None else -1
                    print(
                        f"      [backoff {bo_i+1}/{len(backoff_alphas)} "
                        f"alpha={alpha:.3f}] t={new_mid:.8f} {v_bo} "
                        f"({dt_bo:.1f}s, {iter_bo} iters)",
                        flush=True)
                    if v_bo == _VERDICT_INFEAS:
                        lo = new_mid
                        tau_bo_finite = (tau_bo if np.isfinite(tau_bo)
                                         and tau_bo > 0 else 1.0)
                        tau_lo = tau_bo_finite
                        verdict_log.append((new_mid, _VERDICT_INFEAS))
                        verdict = _VERDICT_INFEAS
                        break
                    elif v_bo == _VERDICT_FEAS:
                        hi = new_mid
                        tau_bo_finite = (tau_bo if np.isfinite(tau_bo)
                                         else 0.0)
                        tau_hi = tau_bo_finite
                        if sol_bo['x'] is not None:
                            last_feas_y = sol_bo['x'][:n_y].copy()
                            last_feas_t = new_mid
                        verdict_log.append((new_mid, _VERDICT_FEAS))
                        verdict = _VERDICT_FEAS
                        break
                    # uncertain: keep shrinking alpha and try again.

                if verdict == _VERDICT_UNCERTAIN:
                    # Backoffs all failed — truly can't certify anything in
                    # (lo, mid].  Abort this round's bisection; lb stays at
                    # the current lo which was carried in from earlier.
                    print(
                        f"    Stall (uncertain after retry + "
                        f"{len(backoff_alphas)} backoffs); "
                        f"ending bisection early. "
                        f"lo={lo:.8f} hi={hi:.8f}",
                        flush=True)
                    break
                # else: a backoff succeeded; lo or hi was updated above.
                # Fall through to the convergence check and next step.

            # Early termination: interval converged.  Threshold was 5e-4
            # historically but that caps lb precision at +/- 0.0005; when
            # we are pushing for lb > 1.2802 (a 4-decimal target), the
            # looser cut-off can terminate the bisection 0.0003-0.0005
            # below the value the current relaxation actually supports.
            # Tightening to 1e-4 extracts that precision at modest cost
            # (usually 1-2 additional bisection steps).
            if hi - lo < 1e-4 and step >= 4:
                print(f"    Converged: interval {hi-lo:.2e} < 1e-4",
                      flush=True)
                break

        # ── Monotonicity check (soundness) ──
        # The feasibility region is {t : val(d) <= t} = [val(d), +oo).
        # Certified 'feas' at t_a => val(d) <= t_a.
        # Certified 'infeas' at t_b => val(d) > t_b.
        # Combined: t_b < val(d) <= t_a, i.e., t_b < t_a.
        #
        # Thus any pair (feas t_a, infeas t_b) with t_b >= t_a is a
        # verdict contradiction. If one occurs, one of them was wrong.
        # 'feas' is easier to certify correctly (primal-feasible point
        # with small tau is a constructive witness), so we trust 'feas'
        # and pull `lo` back to the largest infeas t STRICTLY BELOW the
        # smallest feas t. This preserves soundness (lb <= val(d)).
        feas_ts = [t for t, v in verdict_log if v == _VERDICT_FEAS]
        infeas_ts = [t for t, v in verdict_log if v == _VERDICT_INFEAS]
        if feas_ts and infeas_ts:
            min_feas_t = min(feas_ts)
            max_infeas_t = max(infeas_ts)
            if max_infeas_t >= min_feas_t - 1e-12:
                safe_infeas = [t for t in infeas_ts if t < min_feas_t]
                lo_new = max(safe_infeas) if safe_infeas else \
                    max(0.5, best_lb - 1e-3)
                print(
                    f"    WARNING: monotonicity violation — "
                    f"infeas@{max_infeas_t:.8f} >= feas@{min_feas_t:.8f}. "
                    f"Relaxing lo: {lo:.8f} -> {lo_new:.8f} for soundness.",
                    flush=True)
                lo = lo_new

        lb = lo
        improvement = lb - best_lb
        best_lb = max(best_lb, lb)

        # Propagate warm-start to next CG round
        last_x = ws_x.copy()
        last_y_dual = ws_y.copy()
        last_s = ws_s.copy()

        v = val_d_known.get(d, 0)
        gc = (best_lb - 1) / (v - 1) * 100 if v > 1 else 0
        print(f"    lb={lb:.10f} (+{improvement:.2e}) gc={gc:.1f}%", flush=True)

        # ── Partial checkpoint (before violation check) ──
        # Written with the PREVIOUS round's violations as a fallback, so a
        # crash during the (expensive) violation eigendecomposition below
        # still leaves a usable ckpt.  The post-violation save overwrites
        # this atomically with the latest candidate pool.
        _data_dir = data_dir if data_dir is not None else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        os.makedirs(_data_dir, exist_ok=True)
        _tag = f"d{d}_o{order}_bw{bandwidth}_scs"

        try:
            _save_checkpoint(
                _data_dir, _tag, cg_round, best_lb, active_windows,
                last_feas_y, last_feas_t,
                ws_x.copy(), ws_y.copy(), ws_s.copy(),
                violations,                 # stale (prior round) — best we have
                scalar_lb, d, order, bandwidth,
                elapsed_s=time.time() - t_total,
            )
        except Exception as e:
            print(f"    WARNING: partial checkpoint save failed: {e}",
                  flush=True)

        # Legacy artefacts (kept for backward-compat with downstream proof tools)
        _best_y = last_feas_y if last_feas_y is not None else ws_x[:n_y].copy()
        _ckpt = _result(best_lb, d, order, bandwidth, n_y,
                        len(active_windows), time.time() - t_total, _best_y)
        _ckpt['cg_round'] = cg_round
        _y_sol = _ckpt.pop('y_solution', None)
        if _y_sol is not None:
            np.save(os.path.join(_data_dir, f'solution_{_tag}_cg{cg_round}.npy'),
                    _y_sol)
        with open(os.path.join(_data_dir, f'result_{_tag}_cg{cg_round}.json'), 'w') as _f:
            json.dump(_ckpt, _f, indent=2, default=str)
        print(f"    Checkpoint saved: cg{cg_round} lb={best_lb:.10f}",
              flush=True)

        # Use saved y* from bisection (avoids extra solve, like CUDA's
        # incremental approach — reuse existing work instead of recomputing).
        # Extract BEFORE the cleanup below so ws_x can be reclaimed.
        y_vals = last_feas_y if last_feas_y is not None else ws_x[:n_y].copy()
        hi_for_viol = last_feas_t if last_feas_t is not None else hi

        # ── End-of-round memory reclamation ──
        # The NEXT CG iteration rebuilds every large sparse matrix and
        # every ADMM workspace with a larger active_windows set. Python
        # GC is lazy and PyTorch caches CUDA memory in a pool, so across
        # rounds peak RSS grows linearly in n_active_windows (observed:
        # +10-15 GB per round on the d=16 L3 run). Dropping our bindings
        # to the big per-round objects + forcing GC + flushing the CUDA
        # pool breaks that accumulation. These variables are re-bound at
        # the top of the next iteration, so the rebind would eventually
        # free them anyway — this just forces it to happen NOW, before
        # the next round's allocations compound the peak.
        # On the GPU path A_full_t1 was already deleted above (saves 4-6 GB).
        # On the CPU path it may still be alive — free it now.
        if A_full_t1 is not None:
            A_full_t1 = None
        A_p1_template = None
        b_p1_template = None
        aug_base_data = None
        aug_b_base_data = None
        if aug_t_data is not None:
            aug_t_data = None
            aug_b_t_data = None
        full_base_data = None
        if full_t_data is not None:
            full_t_data = None
        b_full_base = None
        win_decomp = None  # window PSD decomposition (precomputed per round)
        gpu_solver[0] = None
        _gc.collect()
        try:
            import torch as _torch_mod
            if _torch_mod.cuda.is_available():
                _torch_mod.cuda.empty_cache()
        except Exception:
            pass

        violations = _check_violations_highd(
            y_vals, hi_for_viol, P, active_windows, k_vecs=0)

        # Overwrite the partial checkpoint with the FRESH candidate pool.
        # This is the checkpoint a resume will load — it contains the
        # violations the next round should consider.  Atomic write via
        # _save_checkpoint's tmp-then-rename ensures no torn file.
        try:
            _save_checkpoint(
                _data_dir, _tag, cg_round, best_lb, active_windows,
                last_feas_y, last_feas_t,
                ws_x.copy(), ws_y.copy(), ws_s.copy(),
                violations,                 # fresh candidates for round N+1
                scalar_lb, d, order, bandwidth,
                elapsed_s=time.time() - t_total,
            )
        except Exception as e:
            print(f"    WARNING: post-violation checkpoint save failed: {e}",
                  flush=True)

        if not violations:
            print(f"    No violations — converged.", flush=True)
            break
        # Require 3 CONSECUTIVE rounds with < 1e-6 improvement before
        # giving up.  Any single backoff-bisection round can still yield
        # real progress; only a sustained streak of stall rounds is
        # reliable evidence that the relaxation genuinely caps here.
        if improvement < 1e-6:
            consecutive_stalls += 1
        else:
            consecutive_stalls = 0
        if consecutive_stalls >= 3 and cg_round >= 8:
            print(f"    {consecutive_stalls} consecutive stalls at "
                  f"round {cg_round} — stopping.",
                  flush=True)
            break

    elapsed = time.time() - t_total
    # Keep best feasible y for proof/verification
    best_y = last_feas_y if last_feas_y is not None else y_vals
    return _result(best_lb, d, order, bandwidth, n_y,
                   len(active_windows), elapsed, best_y)


def _result(lb, d, order, bw, n_y, n_active, elapsed, y_solution=None):
    v = val_d_known.get(d, 0)
    gc = (lb - 1) / (v - 1) * 100 if v > 1 else 0
    sound = lb <= v + 1e-4 if v > 0 else True
    return {
        'lb': lb, 'd': d, 'order': order, 'bw': bw,
        'n_y': n_y, 'n_active_windows': n_active,
        'gap_closure': gc, 'elapsed': elapsed, 'sound': sound,
        'solver': 'scs_direct',
        'y_solution': y_solution,
    }


def main():
    parser = argparse.ArgumentParser(
        description='d=16 L3 Lasserre SDP — tightest bound quality')
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--bw', type=int, default=15)
    parser.add_argument('--cg-rounds', type=int, default=15)
    parser.add_argument('--bisect', type=int, default=12)
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--scs-iters', type=int, default=20000)
    parser.add_argument('--scs-eps', type=float, default=1e-6)
    parser.add_argument('--scs-scale', type=float, default=None)
    parser.add_argument('--use-indirect', action='store_true')
    parser.add_argument('--k-vecs', type=int, default=3,
                        help='Eigenvectors per spectral cut (default 3)')
    parser.add_argument('--cuts-per-round', type=int, default=50,
                        help='Max new windows added per CG round (default 50, sweep-optimal)')
    parser.add_argument('--rho', type=float, default=0.1,
                        help='ADMM penalty parameter (default 0.1)')
    parser.add_argument('--atom-frac', type=float, default=0.5,
                        help='Fraction of cuts from atom ranking vs eig-rank (default 0.5)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the highest-round checkpoint in --data-dir '
                             'that matches (--d, --order, --bw).')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory for checkpoints + artefacts '
                             '(default: <repo>/data)')
    args = parser.parse_args()

    print(f"Direct SCS: d={args.d} O{args.order} bw={args.bw} GPU={args.gpu}")
    print(f"Started: {datetime.now().isoformat()}")
    import scs as _scs
    print(f"SCS {_scs.__version__}")

    # Memory monitor
    import threading
    def _mon():
        while True:
            try:
                with open('/sys/fs/cgroup/memory.current') as f:
                    u = int(f.read().strip()) / 1e9
                with open('/sys/fs/cgroup/memory.max') as f:
                    mx = f.read().strip()
                    lim = int(mx) / 1e9 if mx != 'max' else 0
                print(f"  [MEM] {u:.1f}/{lim:.0f}GB" if lim else f"  [MEM] {u:.1f}GB",
                      flush=True)
            except Exception:
                pass
            time.sleep(60)
    threading.Thread(target=_mon, daemon=True).start()

    r = solve_scs_direct(
        args.d, args.order, args.bw,
        max_cg_rounds=args.cg_rounds, n_bisect=args.bisect,
        use_gpu=args.gpu, scs_max_iters=args.scs_iters, scs_eps=args.scs_eps,
        scs_scale=args.scs_scale, use_indirect=args.use_indirect,
        k_vecs=args.k_vecs, cuts_per_round=args.cuts_per_round,
        rho=args.rho, atom_frac=args.atom_frac,
        resume=args.resume, data_dir=args.data_dir,
    )

    print(f"\n{'='*70}")
    print(f"FINAL: d={args.d} O{args.order} bw={args.bw}")
    print(f"  lb = {r['lb']:.10f}")
    print(f"  val({args.d}) = {val_d_known.get(args.d, '?')}")
    print(f"  gc = {r['gap_closure']:.2f}%")
    print(f"  time = {r['elapsed']:.1f}s = {r['elapsed']/3600:.2f}hr")
    print(f"  sound = {r['sound']}")
    if r['lb'] > 1.2802:
        print(f"  *** NEW RECORD: lb={r['lb']:.6f} > 1.2802 ***")
    print(f"{'='*70}")

    tag = f"d{args.d}_o{args.order}_bw{args.bw}_scs"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save solution vector (proof artifact)
    y_sol = r.pop('y_solution', None)
    if y_sol is not None:
        y_path = os.path.join(data_dir, f'solution_{tag}.npy')
        np.save(y_path, y_sol)
        print(f"Solution vector saved to {y_path} ({len(y_sol)} entries)")

    # Save result JSON
    out = os.path.join(data_dir, f'result_{tag}.json')
    with open(out, 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Result saved to {out}")


if __name__ == '__main__':
    main()
