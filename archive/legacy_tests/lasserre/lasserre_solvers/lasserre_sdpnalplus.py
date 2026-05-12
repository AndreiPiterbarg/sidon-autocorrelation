#!/usr/bin/env python
r"""SDPNAL+ solver for full L3 Lasserre SDP — optimal end-to-end.

Semismooth-Newton augmented Lagrangian (Yang-Sun-Toh 2015), KKT-certified
1e-7..1e-8 tolerance.  Eliminates the "uncertain" bisection stalls that
plateau the GPU ADMM path at ~57% gc.  Target: push lb > 1.2802 at
d=16 L3.

STRATEGY
========
Unlike ``tests/run_scs_direct.py`` which cutting-plane-grows active
windows across ~15 CG rounds (14 bisections each, ~210 solves total),
this driver:

  round0      min t over base relaxation only (scalar windows + every
              non-window PSD cone).  Single SDPNAL+ call; the objective
              is linear so no bisection.  Returns the scalar_lb.

  feasibility Prove lb > target_t directly.  Fix t = target_t, build the
              full base + ALL 496 window PSDs, run ONE SDPNAL+ solve.
              primal-infeasible at KKT tol  ⇔  lb > target_t certified.

  bisection   Full bisection with every window PSD active from step 0
              (no cutting planes).  8–12 SDPNAL+ calls, each crisp (no
              "uncertain" outcomes like ADMM).  Converges to the full
              L3 relaxation lb.

OPTIMIZATIONS (exhaustive list)
===============================
Inherited from ``run_d16_l3.build_base_problem``:
  • Reduced moment set S (skip unreachable degree-(2k-1) monomials)
  • Mersenne-prime monomial hashing
  • Partial consistency (eq when all children in S, else inequality)
  • Banded clique decomposition (moment PSD + localizing PSD)
  • Cross-clique M_1 coupling when valid
  • Upper localizing (1-μ_i) ⪰ 0
  • **Pairwise L3 localizers M_1(μ_iμ_j y) ⪰ 0**  (from run_d16_l3, not
    the older run_scs_direct build — critical for L3 tightness)
  • Vectorized PSD cone COO builder (``_psd_lower_tri_map`` cached by n)
  • Window PSD base/t decomposition — cheap in-place data update per
    bisection step, no sparse matrix rebuild

Z/2 time-reversal symmetry:
  • Injected as zero-cone equalities via lasserre.z2_symmetry.  Sound
    (val_σ(d) = val(d)) and structurally tightens the dual cert.

SDPNAL+-specific optimizations:
  • **Pre-stripped y ≥ 0 rows** in SeDuMi conversion (n_y rows saved).
  • **Single build of slack-identity block**; reused across bisection.
  • **Persistent MATLAB engine** when ``matlab.engine`` is installed;
    matrix A lives in the MATLAB workspace and only A.data is updated
    per bisection step — I/O collapses from ~160 MB/step to ~1 MB/step.
  • **Warm-start** (X, y, Z iterates) carried across bisection calls.
  • **ADMplus pre-phase** and tight KKT tolerance via OPTIONS.
  • **Non-compressed .mat** writes (compression costs more than I/O
    saves for sparse .mat at this scale).

Execution:
  • Target-first bisection: probe user's target t before the midpoint
    walk so a single infeasibility solve can terminate the loop.
  • Monotonicity auditing: the bracket invariant (lo infeasible, hi
    feasible) is asserted after every move.
  • MATLAB/SDPNAL+ path-check at startup (fail fast if missing).
  • Memory-monitor thread prints RSS at fixed cadence.
  • Graceful subprocess fallback when matlab.engine is unavailable.

REQUIREMENTS
============
  • MATLAB R2019b+ with a valid license.
  • SDPNAL+ 1.0+ on the MATLAB path
    (https://github.com/Sunhee0515/SDPNALplus).
  • Python: numpy, scipy.  Optional: matlabengine (pip install), enables
    the persistent-session fast path.

USAGE
=====
  # Acceptance: reproduce d=6 L3 full 99.38% gc
  python tests/lasserre_sdpnalplus.py --d 6 --order 3 --bw 5

  # Target: d=16 L3 full via bisection
  python tests/lasserre_sdpnalplus.py --d 16 --order 3 --bw 15 \
      --mode bisection --sdpnal-tol 1e-7

  # Direct proof: lb > 1.2802 in one solve
  python tests/lasserre_sdpnalplus.py --d 16 --order 3 --bw 15 \
      --mode feasibility --target 1.2802
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import scipy.io as sio
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# run_d16_l3.build_base_problem includes the pairwise L3 cones; the
# run_scs_direct version does NOT — we must use run_d16_l3.
from run_d16_l3 import (  # noqa: E402
    build_base_problem,
    _precompute_window_psd_decomposition,
    _assemble_window_psd,
)
from lasserre_highd import (  # noqa: E402
    _precompute_highd, _build_banded_cliques, val_d_known,
)
from lasserre.z2_symmetry import inject_z2_equalities_into_base_problem  # noqa: E402

# Optional: persistent MATLAB session.  Pip install matlabengine.
try:
    import matlab  # type: ignore
    import matlab.engine  # type: ignore
    HAS_MATLAB_ENGINE = True
except ImportError:
    HAS_MATLAB_ENGINE = False


# =====================================================================
# Constants
# =====================================================================

SQRT2 = np.sqrt(2.0)

# SDPNAL+ termination codes (inherited from SDPT3):
#   0 = solved (primal-dual optimal within tolerance)
#   1 = primal infeasible (certified)
#   2 = dual infeasible (certified)
#  -1 = max iters / numerical
_TERM_SOLVED = 0
_TERM_PRI_INFEAS = 1
_TERM_DUAL_INFEAS = 2


# =====================================================================
# Memory monitor (copied pattern from run_d16_l3.py)
# =====================================================================

def _start_memory_monitor(interval_s=60):
    """Background thread that prints RSS / cgroup memory every interval."""
    def _mon():
        while True:
            try:
                # Linux/cgroups v2 first; fall back to psutil if available.
                if os.path.exists('/sys/fs/cgroup/memory.current'):
                    with open('/sys/fs/cgroup/memory.current') as f:
                        u = int(f.read().strip()) / 1e9
                    with open('/sys/fs/cgroup/memory.max') as f:
                        mx = f.read().strip()
                        lim = int(mx) / 1e9 if mx != 'max' else 0
                    print(f"  [MEM] {u:.1f}/{lim:.0f}GB"
                          if lim else f"  [MEM] {u:.1f}GB",
                          flush=True)
                else:
                    try:
                        import psutil
                        p = psutil.Process(os.getpid())
                        rss = p.memory_info().rss / 1e9
                        print(f"  [MEM] {rss:.1f}GB (rss)", flush=True)
                    except ImportError:
                        return  # give up silently
            except Exception:
                pass
            time.sleep(interval_s)
    threading.Thread(target=_mon, daemon=True).start()


# =====================================================================
# SCS (Ax + s = b, s ∈ K) → SeDuMi primal (Au = b, u ∈ K') conversion
# =====================================================================
#
# SCS and SDPNAL+ (via read_sedumi) differ in primal/dual orientation:
#   SCS:     min c^T x  s.t.  Ax + s = b, s ∈ K
#   SeDuMi:  min c^T u  s.t.  Au = b,    u ∈ K
#
# Conversion introduces one explicit slack per non-zero-cone row:
#
#   u = [t ; y ; s_l_slack ; s_s_slack]
#   K.f = 1  (t is free)
#   K.l = n_y + |s_l_slack|  (y and nonneg slacks)
#   K.s = [s_sizes]          (PSD slacks)
#
#   A_sedumi = [A_zero_row_perm , 0        , 0            ]
#              [A_l_nonneg_row_perm , I_l  , 0            ]
#              [A_s_psd_row_perm , 0      , I_s           ]
#
# Optimization: the SCS formulation redundantly encodes y ≥ 0 via
# n_y "-I·y + s = 0, s ≥ 0" rows in the nonneg block.  Since y is
# already in K.l in SeDuMi form, those rows are redundant and are
# stripped before conversion.  At d=16 L3 this is 74,613 saved rows.
# =====================================================================


def _find_y_nonneg_rows(A, b, cone, meta):
    """Return bool mask over A rows: True to keep, False to drop.

    The redundant y ≥ 0 rows look like: a single nonzero at (-1) on
    some y-column, b=0, within the nonneg-cone block.
    """
    n_y = meta['n_y']
    n_zero = meta['n_zero']
    n_nonneg = meta['n_nonneg']

    keep = np.ones(A.shape[0], dtype=bool)
    if n_nonneg == 0:
        return keep

    A_l = A[n_zero:n_zero + n_nonneg].tocsr()
    b_l = b[n_zero:n_zero + n_nonneg]

    nnz_per_row = np.diff(A_l.indptr)
    single = (nnz_per_row == 1)
    b_zero = (np.abs(b_l) < 1e-14)
    cand = np.where(single & b_zero)[0]
    for r in cand:
        j = A_l.indices[A_l.indptr[r]]
        v = A_l.data[A_l.indptr[r]]
        if j < n_y and abs(v + 1.0) < 1e-14:
            keep[n_zero + r] = False
    return keep


def _scs_to_sedumi(A_scs, b_scs, c_scs, cone, meta, y_nonneg_mask=None,
                    verbose=True):
    """Convert (A, b, c, cone) in SCS form to SeDuMi primal form.

    y_nonneg_mask: optional cached mask from an earlier call with the
    same base structure (expensive to recompute at every bisection step).
    """
    t_start = time.time()

    n_y = meta['n_y']
    t_col = meta['t_col']
    n_zero = meta['n_zero']
    n_nonneg = meta['n_nonneg']

    if y_nonneg_mask is None:
        keep_mask = _find_y_nonneg_rows(A_scs, b_scs, cone, meta)
    else:
        # Pad the cached mask if additional rows (e.g., window PSDs)
        # were appended AFTER the cached mask was computed.  Those rows
        # are PSD rows and should all be kept.
        if len(y_nonneg_mask) < A_scs.shape[0]:
            keep_mask = np.concatenate([
                y_nonneg_mask,
                np.ones(A_scs.shape[0] - len(y_nonneg_mask), dtype=bool),
            ])
        else:
            keep_mask = y_nonneg_mask[:A_scs.shape[0]]

    n_stripped = int((~keep_mask).sum())
    if verbose:
        print(f"  Stripped {n_stripped} redundant y>=0 rows "
              f"({n_stripped * 100.0 / A_scs.shape[0]:.1f}% of A_scs rows)",
              flush=True)

    A = A_scs[keep_mask].tocsc()
    b = b_scs[keep_mask].copy()

    m_total = A.shape[0]
    n_zero_kept = int(keep_mask[:n_zero].sum())
    n_nonneg_kept = int(keep_mask[n_zero:n_zero + n_nonneg].sum())

    # Permute columns so t comes first, y next.  SCS had [y..., t].
    col_perm = np.concatenate([[t_col], np.arange(n_y)])
    A_permuted = A[:, col_perm]

    # SeDuMi primal-form layout for u = [ t ; y ; s_l ; vec(X_1); vec(X_2); ... ]
    #   K.f = 1, K.l = n_y + n_l_slack, K.s = [n_1, n_2, ...]
    #   |u| = 1 + n_y + n_l_slack + sum(n_p^2)
    #
    # Our SCS PSD cone rows are in SVEC form (lower-triangle column-major,
    # off-diagonal scaled by sqrt(2)). read_sedumi expects the PSD block
    # of u in VEC form (n_p^2 entries per block, unscaled). We must
    # expand: for each SCS svec row k encoding matrix entry (i, j) with
    # scale s, emit VEC-form coupling
    #     A_sedumi[row_k, vec_col(i,j)] = s        if i == j
    #     A_sedumi[row_k, vec_col(i,j)] = s/2      if i != j
    #     A_sedumi[row_k, vec_col(j,i)] = s/2      if i != j
    # (The symmetric split is algebraically equivalent to full coefficient
    # on the (i,j) side; this version produces a symmetric A_i that
    # read_sedumi processes with fewer numerical surprises.)

    psd_sizes = list(cone.get('s', []))
    psd_n_sq_total = int(sum(int(n_p) * int(n_p) for n_p in psd_sizes))
    n_l_slack = n_nonneg_kept

    # ---- Nonneg slack block (straight identity as before) ----
    nonneg_slack_rows = []
    nonneg_slack_cols = []
    nonneg_slack_vals = []
    if n_l_slack > 0:
        nonneg_slack_rows = np.arange(n_zero_kept,
                                       n_zero_kept + n_l_slack,
                                       dtype=np.int64)
        nonneg_slack_cols = np.arange(n_l_slack, dtype=np.int64)
        nonneg_slack_vals = np.ones(n_l_slack)

    # ---- PSD vec-form slack block ----
    psd_slack_rows_parts = []
    psd_slack_cols_parts = []
    psd_slack_vals_parts = []
    psd_row_cursor = n_zero_kept + n_l_slack
    psd_vec_col_cursor = 0   # offset within the PSD-slack block (0-based)

    for n_p in psd_sizes:
        svec_size = n_p * (n_p + 1) // 2
        # SCS column-major lower-triangle ordering: k = col_start(j) + (i - j)
        # where col_start(j) = sum_{j'<j} (n_p - j') = j*n_p - j*(j-1)/2.
        # Decode k -> (i, j) vectorised.
        ks = np.arange(svec_size, dtype=np.int64)
        # column index j for each k: solve for j s.t. cumulative rows >= k+1
        col_sizes = np.arange(n_p, 0, -1, dtype=np.int64)   # [n, n-1, ..., 1]
        col_starts = np.concatenate([[0], np.cumsum(col_sizes)[:-1]])
        # np.searchsorted on col_starts+1 gives the column
        js = np.searchsorted(col_starts + col_sizes, ks, side='right')
        is_ = js + (ks - col_starts[js])   # row index
        diag_mask = (is_ == js)
        offdiag_mask = ~diag_mask

        # Rows (each svec row writes once if diag, twice if off-diag)
        # Vec-col indexing inside the PSD block: (i, j) -> i*n_p + j
        # Diagonal contributions
        rows_d = psd_row_cursor + ks[diag_mask]
        cols_d = psd_vec_col_cursor + (is_[diag_mask] * n_p + js[diag_mask])
        vals_d = np.ones(diag_mask.sum())
        # Off-diagonal contributions (both (i,j) and (j,i), weight sqrt(2)/2)
        off_ks = ks[offdiag_mask]
        off_is = is_[offdiag_mask]
        off_js = js[offdiag_mask]
        rows_o = psd_row_cursor + np.concatenate([off_ks, off_ks])
        cols_o = psd_vec_col_cursor + np.concatenate([
            off_is * n_p + off_js,     # (i, j)
            off_js * n_p + off_is,     # (j, i)
        ])
        # Scale: SCS off-diag svec has factor sqrt(2). Split symmetrically.
        off_half = np.sqrt(2.0) / 2.0
        vals_o = np.concatenate([
            np.full(len(off_ks), off_half),
            np.full(len(off_ks), off_half),
        ])

        psd_slack_rows_parts.extend([rows_d, rows_o])
        psd_slack_cols_parts.extend([cols_d, cols_o])
        psd_slack_vals_parts.extend([vals_d, vals_o])

        psd_row_cursor += svec_size
        psd_vec_col_cursor += n_p * n_p

    # Assemble the (nonneg + psd) slack block, offset into the SeDuMi
    # column space so nonneg slacks precede PSD vec slacks.
    slack_rows_all = []
    slack_cols_all = []
    slack_vals_all = []
    if n_l_slack > 0:
        slack_rows_all.append(np.asarray(nonneg_slack_rows, dtype=np.int64))
        slack_cols_all.append(np.asarray(nonneg_slack_cols, dtype=np.int64))
        slack_vals_all.append(np.asarray(nonneg_slack_vals, dtype=np.float64))
    if psd_slack_rows_parts:
        psd_rows_cat = np.concatenate(psd_slack_rows_parts).astype(np.int64)
        psd_cols_cat = np.concatenate(psd_slack_cols_parts).astype(np.int64)
        psd_vals_cat = np.concatenate(psd_slack_vals_parts).astype(np.float64)
        # PSD vec cols live AFTER the nonneg slack cols.
        psd_cols_cat = psd_cols_cat + n_l_slack
        slack_rows_all.append(psd_rows_cat)
        slack_cols_all.append(psd_cols_cat)
        slack_vals_all.append(psd_vals_cat)

    n_slack_cols = n_l_slack + psd_n_sq_total
    if slack_rows_all:
        R = np.concatenate(slack_rows_all)
        C = np.concatenate(slack_cols_all)
        V = np.concatenate(slack_vals_all)
        A_slack = sp.csc_matrix((V, (R, C)),
                                 shape=(m_total, n_slack_cols))
        A_sedumi = sp.hstack([A_permuted, A_slack], format='csc')
    else:
        A_sedumi = A_permuted

    n_sedumi = 1 + n_y + n_slack_cols
    c_sedumi = np.zeros(n_sedumi)
    c_sedumi[0] = 1.0  # objective: min t

    K = {
        'f': 1,
        'l': n_y + n_l_slack,
        's': [int(n_p) for n_p in psd_sizes],
    }

    if verbose:
        print(f"  SeDuMi: {m_total:,} rows x {n_sedumi:,} cols, "
              f"nnz={A_sedumi.nnz:,}  "
              f"(K.f={K['f']}, K.l={K['l']}, K.s={len(K['s'])} blocks)",
              flush=True)
        print(f"  Conversion: {time.time() - t_start:.2f}s", flush=True)

    return A_sedumi, b, c_sedumi, K, keep_mask


# =====================================================================
# Problem build — base + (optional) all window PSDs, with base/t cache
# =====================================================================

def build_problem(d, order, bandwidth,
                   add_upper_loc=True, use_z2=True,
                   include_all_windows=True, t_val=1.0,
                   verbose=True):
    """Construct the L3 SDP ready for SDPNAL+ export.

    include_all_windows:
      False  — round-0 base only.  t remains a free variable; min t is
               linear.  One SDPNAL+ solve gives the scalar lb directly.
      True   — full base + all 496 window PSDs built at t_val.  For
               bisection the t-dependent entries are isolated (see
               ``win_decomp``) so per-step updates are O(nnz(t-coef))
               instead of O(nnz(A)).

    Returns dict with:
      A_sedumi, b_sedumi, c_sedumi, K
      A_base_scs, b_base_scs, cone_base_scs
      win_decomp                   — cached base/t decomposition
      meta, P                      — precompute + metadata
      keep_mask                    — cached y-nonneg row mask
      A_base_data, A_t_data        — data arrays for in-place bisection
      A_template_sedumi            — precomputed sparsity template
      t_data_mask                  — indices into A_sedumi.data that are
                                     t-dependent (only these change per
                                     bisection step)
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SDPNAL+ BUILD: d={d} L{order} bw={bandwidth}  "
              f"(upper_loc={add_upper_loc}, z2={use_z2}, "
              f"all_windows={include_all_windows}, t_val={t_val})")
        print(f"{'=' * 70}", flush=True)

    # 1. Cliques + reduced moment set + partial consistency + pairwise.
    cliques = _build_banded_cliques(d, bandwidth)
    if verbose:
        print(f"\nCliques: {len(cliques)} of size {len(cliques[0])}",
              flush=True)
    P = _precompute_highd(d, order, cliques, verbose=verbose)

    # 2. Base SCS problem — pairwise cones INCLUDED (run_d16_l3 build).
    if verbose:
        print("\nBuilding base SCS problem (incl. pairwise L3 cones)...",
              flush=True)
    t_build = time.time()
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(
        P, add_upper_loc=add_upper_loc)
    if verbose:
        print(f"  Base built in {time.time() - t_build:.1f}s", flush=True)

    # 3. Z/2 symmetry.
    if use_z2:
        A_base, b_base, cone_base, n_z2 = \
            inject_z2_equalities_into_base_problem(
                P, A_base, b_base, cone_base, verbose=verbose)
        meta['n_zero'] += n_z2
        meta['n_rows_base'] += n_z2

    # 4. Window PSDs.
    win_decomp = None
    A_with_win = A_base
    b_with_win = b_base
    cone_with_win = cone_base

    if include_all_windows:
        if verbose:
            print(f"\nBuilding all {P['n_win']} window PSDs at t={t_val}...",
                  flush=True)
        t_win = time.time()

        all_ws = set(P.get('nontrivial_windows', range(P['n_win'])))
        active_windows = {w for w in all_ws
                          if int(P['window_covering'][w]) >= 0}
        if verbose:
            print(f"  {len(active_windows)}/{P['n_win']} windows "
                  f"covered by cliques", flush=True)

        win_decomp = _precompute_window_psd_decomposition(P, active_windows)
        if win_decomp is not None:
            A_win, b_win, psd_win = _assemble_window_psd(win_decomp, t_val)
            A_with_win = sp.vstack([A_base, A_win], format='csc')
            b_with_win = np.concatenate([b_base, b_win])
            cone_with_win = {
                'z': cone_base.get('z', 0),
                'l': cone_base.get('l', 0),
                's': list(cone_base.get('s', [])) + list(psd_win),
            }
            if verbose:
                print(f"  {len(psd_win)} window PSD cones added "
                      f"(dim {psd_win[0] if psd_win else 0}).  "
                      f"Build: {time.time() - t_win:.1f}s", flush=True)

    if verbose:
        print(f"\nFinal SCS: {A_with_win.shape[0]:,} rows x "
              f"{A_with_win.shape[1]:,} cols, nnz={A_with_win.nnz:,}",
              flush=True)

    # 5. SCS -> SeDuMi (caches y-nonneg mask for bisection reuse).
    if verbose:
        print("\nConverting SCS -> SeDuMi primal form...", flush=True)
    A_sedumi, b_sedumi, c_sedumi, K, keep_mask = _scs_to_sedumi(
        A_with_win, b_with_win, c_obj, cone_with_win, meta,
        y_nonneg_mask=None, verbose=verbose)

    # 6. Pre-extract the full A data-array base/t decomposition so
    # subsequent bisection steps update only the .data field.
    A_base_data = None
    A_t_data = None
    t_data_mask = None
    if include_all_windows and win_decomp is not None and win_decomp['has_t']:
        # Rebuild A_sedumi at t=t_val and t=t_val+1 to extract the
        # t-coefficient column of its .data array.  Both share the same
        # sparsity pattern by construction (no fill-in because only the
        # window-PSD rows' values change, not their (row, col)).
        if verbose:
            print("Extracting base/t data decomposition for in-place "
                  "bisection updates...", flush=True)
        A_win2, _, _ = _assemble_window_psd(win_decomp, t_val + 1.0)
        A_scs_t2 = sp.vstack([A_base, A_win2], format='csc')
        A_sedumi2, _, _, _, _ = _scs_to_sedumi(
            A_scs_t2, b_with_win, c_obj, cone_with_win, meta,
            y_nonneg_mask=keep_mask, verbose=False)
        # Canonicalise before subtracting .data arrays.
        A_sedumi.sort_indices()
        A_sedumi2.sort_indices()
        if A_sedumi.nnz == A_sedumi2.nnz:
            A_t_data = A_sedumi2.data - A_sedumi.data
            A_base_data = A_sedumi.data - (t_val * A_t_data)
            # Optional: index mask for sparse update (entries where
            # A_t_data is nonzero).
            t_data_mask = np.nonzero(A_t_data)[0]
            if verbose:
                print(f"  t-coef entries: {len(t_data_mask):,} / "
                      f"{A_sedumi.nnz:,} ({len(t_data_mask)*100.0/A_sedumi.nnz:.2f}%)",
                      flush=True)
        else:
            if verbose:
                print(f"  WARNING: nnz mismatch ({A_sedumi.nnz} vs "
                      f"{A_sedumi2.nnz}); falling back to full rebuild "
                      f"per bisection step.", flush=True)

    return {
        'A_sedumi': A_sedumi, 'b_sedumi': b_sedumi,
        'c_sedumi': c_sedumi, 'K': K,
        'meta': meta, 'P': P,
        'A_base_scs': A_base, 'b_base_scs': b_base,
        'cone_base_scs': cone_base, 'c_scs': c_obj,
        'A_full_scs': A_with_win, 'b_full_scs': b_with_win,
        'cone_full_scs': cone_with_win,
        'win_decomp': win_decomp,
        'keep_mask': keep_mask,
        'A_base_data': A_base_data,
        'A_t_data': A_t_data,
        't_data_mask': t_data_mask,
        'include_all_windows': include_all_windows,
    }


def update_problem_t(problem, new_t_val):
    """In-place update A_sedumi.data for a new t_val.  O(nnz(t-coef))."""
    if problem.get('A_base_data') is None:
        # Fallback: full rebuild.
        win_decomp = problem['win_decomp']
        if win_decomp is None:
            return problem
        A_win, _, _ = _assemble_window_psd(win_decomp, new_t_val)
        A_scs_new = sp.vstack(
            [problem['A_base_scs'], A_win], format='csc')
        b_scs_new = np.concatenate(
            [problem['b_base_scs'], np.zeros(A_win.shape[0])])
        A_sed, b_sed, c_sed, K, keep_mask = _scs_to_sedumi(
            A_scs_new, b_scs_new, problem['c_scs'],
            problem['cone_full_scs'], problem['meta'],
            y_nonneg_mask=problem['keep_mask'], verbose=False)
        problem['A_sedumi'] = A_sed
        problem['b_sedumi'] = b_sed
        problem['c_sedumi'] = c_sed
        return problem

    # Fast path: in-place .data update.
    A = problem['A_sedumi']
    np.add(problem['A_base_data'], new_t_val * problem['A_t_data'],
           out=A.data)
    return problem


# =====================================================================
# MATLAB — persistent engine (preferred) or subprocess fallback
# =====================================================================

class MatlabRunner:
    """Wrap a persistent MATLAB session if matlab.engine is available;
    otherwise spawn a fresh subprocess per call.

    The persistent path keeps A in the MATLAB workspace across calls —
    only A.data (~1 MB at d=16 L3) is transferred per bisection step
    instead of the full matrix (~160 MB).
    """

    def __init__(self, matlab_bin='matlab', use_engine=None,
                  wrapper_dir=None, verbose=True):
        self.matlab_bin = matlab_bin
        self.wrapper_dir = wrapper_dir or os.path.dirname(
            os.path.abspath(__file__))
        self.verbose = verbose

        if use_engine is None:
            use_engine = HAS_MATLAB_ENGINE
        self.use_engine = use_engine and HAS_MATLAB_ENGINE

        self._engine = None
        self._a_loaded = False  # tracks whether A is cached in engine

    # --------------------------------------------------------------

    def start(self):
        if self.use_engine and self._engine is None:
            if self.verbose:
                print("Starting persistent MATLAB engine...", flush=True)
            t0 = time.time()
            self._engine = matlab.engine.start_matlab()
            self._engine.addpath(self.wrapper_dir, nargout=0)
            if self.verbose:
                print(f"  Engine ready in {time.time() - t0:.1f}s",
                      flush=True)

    def stop(self):
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
            self._a_loaded = False

    # --------------------------------------------------------------
    # Preflight

    def check_sdpnalplus(self):
        """Verify MATLAB is on PATH (subprocess path) or confirm
        SDPNAL+ is reachable via the engine."""
        if self.use_engine:
            self.start()
            found = self._engine.exist('sdpnalplus', 'file')
            if float(found) == 0:
                raise RuntimeError(
                    "SDPNAL+ is not on the MATLAB path. Clone "
                    "https://github.com/Sunhee0515/SDPNALplus and "
                    "addpath(SDPNALplus_root) in your startup.m.")
        else:
            if shutil.which(self.matlab_bin) is None:
                raise RuntimeError(
                    f"MATLAB binary '{self.matlab_bin}' not found on "
                    f"PATH. Pass --matlab-bin /abs/path/to/matlab.")

    # --------------------------------------------------------------
    # Call

    def solve(self, problem, mode, t_val, sdpnal_tol, sdpnal_maxiter,
               data_dir, tag, warm_start=None, verbose=True):
        """Run one SDPNAL+ solve.

        Returns dict: obj, termcode, kkt, pri_res, dual_res, runtime,
                      iter_main, iter_sub, warm_x, warm_y, warm_Z.
        """
        if self.use_engine:
            return self._solve_engine(
                problem, mode, t_val, sdpnal_tol, sdpnal_maxiter,
                data_dir, tag, warm_start=warm_start, verbose=verbose)
        return self._solve_subprocess(
            problem, mode, t_val, sdpnal_tol, sdpnal_maxiter,
            data_dir, tag, verbose=verbose)

    # --------------- engine path ---------------

    def _py2ml_sparse(self, M):
        """Convert scipy.sparse → MATLAB sparse via engine workspace."""
        Mcoo = M.tocoo()
        i = matlab.double((Mcoo.row + 1).astype(np.float64).tolist())
        j = matlab.double((Mcoo.col + 1).astype(np.float64).tolist())
        v = matlab.double(Mcoo.data.astype(np.float64).tolist())
        m, n = M.shape
        self._engine.workspace['Ai'] = i
        self._engine.workspace['Aj'] = j
        self._engine.workspace['Av'] = v
        self._engine.workspace['Am'] = float(m)
        self._engine.workspace['An'] = float(n)
        self._engine.eval("A = sparse(Ai, Aj, Av, Am, An);", nargout=0)
        self._engine.eval("clear Ai Aj Av Am An;", nargout=0)

    def _py2ml_vec(self, name, v):
        self._engine.workspace[name] = matlab.double(
            v.astype(np.float64).reshape(-1, 1).tolist())

    def _solve_engine(self, problem, mode, t_val, sdpnal_tol,
                       sdpnal_maxiter, data_dir, tag, warm_start=None,
                       verbose=True):
        self.start()
        eng = self._engine

        t_push = time.time()
        # Push b, c, K on every call (cheap — small vectors).
        self._py2ml_vec('b', problem['b_sedumi'])
        c = problem['c_sedumi'].copy()
        if mode == 'feasibility':
            c[:] = 0.0
        self._py2ml_vec('c', c)
        eng.workspace['K_f'] = float(problem['K'].get('f', 0))
        eng.workspace['K_l'] = float(problem['K'].get('l', 0))
        K_s = problem['K'].get('s', [])
        eng.workspace['K_s'] = matlab.double(
            [[float(s)] for s in K_s]) if K_s else matlab.double([])

        # Push A: if already in workspace with the same sparsity pattern
        # AND we have A_t_data, transfer ONLY A.data (fast path).  Else
        # push full sparse.
        if (self._a_loaded and problem.get('A_t_data') is not None
                and problem.get('A_base_data') is not None):
            new_data = problem['A_sedumi'].data
            self._py2ml_vec('A_data', new_data)
            eng.eval(
                "[ii, jj, vv] = find(A); A = sparse(ii, jj, A_data(:), "
                "size(A, 1), size(A, 2)); clear ii jj vv A_data;",
                nargout=0)
            # NOTE: the `find` + sparse reassembly preserves the nonzero
            # pattern; equivalent to an in-place .data update.  MATLAB
            # doesn't expose .data directly on sparse, so this is the
            # standard idiom.
        else:
            self._py2ml_sparse(problem['A_sedumi'])
            self._a_loaded = True

        if verbose:
            print(f"  Engine push: {time.time() - t_push:.1f}s",
                  flush=True)

        # Options.
        eng.workspace['sdpnal_tol'] = float(sdpnal_tol)
        eng.workspace['sdpnal_maxiter'] = float(sdpnal_maxiter)
        eng.workspace['n_y'] = float(problem['meta']['n_y'])
        eng.workspace['t_val_py'] = float(t_val) if t_val is not None else -1.0

        # Warm start (optional).
        if warm_start is not None:
            if warm_start.get('X') is not None:
                eng.workspace['X0'] = warm_start['X']
            if warm_start.get('y') is not None:
                self._py2ml_vec('y0', warm_start['y'])
            if warm_start.get('Z') is not None:
                eng.workspace['Z0'] = warm_start['Z']
            eng.workspace['have_warm'] = 1.0
        else:
            eng.workspace['have_warm'] = 0.0

        # Run the wrapper (defined in sdpnalplus_solve.m).
        t_solve = time.time()
        try:
            # sdpnalplus_solve is a SCRIPT (needs caller's workspace),
            # so invoke via eval rather than as a function call.
            eng.eval('sdpnalplus_solve;', nargout=0)
        except Exception as exc:
            return {'obj': None, 'termcode': -999, 'error': str(exc),
                    'wall_s': time.time() - t_solve}
        wall = time.time() - t_solve

        # Pull results.
        try:
            result = {
                'obj': float(eng.workspace['obj']),
                'termcode': int(eng.workspace['termcode']),
                'kkt': float(eng.workspace['kkt']),
                'pri_res': float(eng.workspace['pri_res']),
                'dual_res': float(eng.workspace['dual_res']),
                'iter_main': int(eng.workspace['iter_main']),
                'iter_sub': int(eng.workspace['iter_sub']),
                't_opt': (float(eng.workspace['t_opt'])
                          if float(eng.workspace.get('has_t_opt', 0.0)) > 0
                          else None),
                'wall_s': wall,
            }
            # Extract warm-start payloads for the next call.
            result['warm_X'] = eng.workspace.get('X_out')
            result['warm_y'] = eng.workspace.get('y_out')
            result['warm_Z'] = eng.workspace.get('Z_out')
        except Exception as exc:
            result = {'obj': None, 'termcode': -998, 'error': str(exc),
                      'wall_s': wall}
        return result

    # --------------- subprocess path ---------------

    def _solve_subprocess(self, problem, mode, t_val, sdpnal_tol,
                           sdpnal_maxiter, data_dir, tag, verbose=True):
        os.makedirs(data_dir, exist_ok=True)
        input_mat = os.path.abspath(
            os.path.join(data_dir, f'sdpnal_input_{tag}.mat'))
        output_mat = os.path.abspath(
            os.path.join(data_dir, f'sdpnal_output_{tag}.mat'))
        log_path = os.path.abspath(
            os.path.join(data_dir, f'sdpnal_matlab_{tag}.log'))

        c = problem['c_sedumi'].copy()
        if mode == 'feasibility':
            c[:] = 0.0

        # The MAT5 format has a 4 GB per-variable limit. At d=16 L3 the
        # full A_sedumi is ~4.3 GB (~358M nnz), over the limit.  Split A
        # into row-chunks each <= 1.5 GB worth of CSC storage and save
        # them as A_chunk_1, A_chunk_2, ..., together with row-boundary
        # metadata so MATLAB can vstack them on load.
        A_full = problem['A_sedumi'].astype(np.float64).tocsr()
        target_bytes_per_chunk = 1_500_000_000   # 1.5 GB
        # CSC/CSR overhead per nnz ~ 12 bytes (8 for data + 4 for index).
        # We also need col pointers (n_cols+1 * 4 bytes).  When MATLAB
        # converts CSR -> CSC on load the overhead flips: col pointers
        # of size n_cols+1 get materialised.  Use the conservative
        # per-chunk nnz cap given by total_bytes = 12*nnz + 4*n_cols.
        n_cols = A_full.shape[1]
        col_ptr_bytes = 4 * (n_cols + 1)
        max_nnz_per_chunk = max(
            1, (target_bytes_per_chunk - col_ptr_bytes) // 12)
        chunk_row_starts = [0]
        cum_nnz = 0
        indptr = A_full.indptr
        for r in range(A_full.shape[0]):
            row_nnz = int(indptr[r + 1] - indptr[r])
            if cum_nnz + row_nnz > max_nnz_per_chunk and cum_nnz > 0:
                chunk_row_starts.append(r)
                cum_nnz = 0
            cum_nnz += row_nnz
        chunk_row_starts.append(A_full.shape[0])
        n_chunks = len(chunk_row_starts) - 1
        if verbose:
            print(f"  A_sedumi: {A_full.shape[0]:,} rows x "
                  f"{A_full.shape[1]:,} cols, nnz={A_full.nnz:,}  "
                  f"-> splitting into {n_chunks} row-chunks for MAT5.",
                  flush=True)

        mdict = {
            'b': problem['b_sedumi'].astype(np.float64).reshape(-1, 1),
            'c': c.astype(np.float64).reshape(-1, 1),
            'K_f': np.float64(problem['K'].get('f', 0)),
            'K_l': np.float64(problem['K'].get('l', 0)),
            'K_s': (np.asarray(problem['K'].get('s', []),
                               dtype=np.float64).reshape(-1, 1)
                    if problem['K'].get('s') else np.zeros((0, 1))),
            'sdpnal_tol': np.float64(sdpnal_tol),
            'sdpnal_maxiter': np.float64(sdpnal_maxiter),
            't_val_py': (np.float64(t_val)
                         if t_val is not None else np.float64(-1.0)),
            'n_y': np.float64(problem['meta']['n_y']),
            'have_warm': np.float64(0.0),
            'output_path': output_mat,
            'n_A_chunks': np.float64(n_chunks),
            'A_n_rows': np.float64(A_full.shape[0]),
            'A_n_cols': np.float64(A_full.shape[1]),
        }
        for i in range(n_chunks):
            r0 = chunk_row_starts[i]
            r1 = chunk_row_starts[i + 1]
            # Convert chunk to CSC before save so MATLAB's sparse is
            # native CSC (saves one in-MATLAB conversion).
            chunk = A_full[r0:r1].tocsc()
            mdict[f'A_chunk_{i + 1}'] = chunk
            if verbose:
                print(f"    chunk {i+1}/{n_chunks}: rows [{r0:,}:{r1:,}]  "
                      f"nnz={chunk.nnz:,}", flush=True)
        # Non-compressed write is faster for sparse .mat at this scale.
        sio.savemat(input_mat, mdict, do_compression=False, oned_as='column')

        # Normalize to forward slashes so MATLAB on Windows interprets
        # the paths correctly regardless of the platform-native separator.
        wrapper_dir_ml = self.wrapper_dir.replace('\\', '/')
        input_mat_ml = input_mat.replace('\\', '/')
        cmd = [
            self.matlab_bin, '-batch',
            f"try, addpath('{wrapper_dir_ml}'); "
            f"load('{input_mat_ml}'); sdpnalplus_solve; "
            f"catch err, disp(getReport(err)); exit(1); end; exit(0);",
        ]
        t0 = time.time()
        with open(log_path, 'w') as logf:
            try:
                res = subprocess.run(
                    cmd, stdout=logf, stderr=subprocess.STDOUT,
                    cwd=self.wrapper_dir)
                rc = res.returncode
            except Exception as exc:
                return {'obj': None, 'termcode': -997,
                        'error': str(exc),
                        'wall_s': time.time() - t0}
        wall = time.time() - t0

        if not os.path.exists(output_mat):
            return {'obj': None, 'termcode': -996,
                    'error': 'no output .mat produced', 'rc': rc,
                    'log_path': log_path, 'wall_s': wall}

        d = sio.loadmat(output_mat, squeeze_me=True,
                         struct_as_record=False)
        return {
            'obj': float(d.get('obj', 0.0)),
            'termcode': int(d.get('termcode', -999)),
            'kkt': float(d.get('kkt', float('nan'))),
            'pri_res': float(d.get('pri_res', float('nan'))),
            'dual_res': float(d.get('dual_res', float('nan'))),
            'iter_main': int(d.get('iter_main', 0)),
            'iter_sub': int(d.get('iter_sub', 0)),
            't_opt': (float(d['t_opt'])
                      if 't_opt' in d and
                         float(d.get('has_t_opt', 0.0)) > 0 else None),
            'wall_s': wall,
            'rc': rc,
            'log_path': log_path,
        }


# =====================================================================
# Verdict classification
# =====================================================================

def _classify(termcode):
    if termcode == _TERM_SOLVED:
        return 'solved'
    if termcode == _TERM_PRI_INFEAS:
        return 'primal_infeas'
    if termcode == _TERM_DUAL_INFEAS:
        return 'dual_infeas'
    return 'uncertain'


# =====================================================================
# High-level drivers
# =====================================================================

def solve_round0(d, order, bandwidth,
                  add_upper_loc=True, use_z2=True,
                  sdpnal_tol=1e-7, sdpnal_maxiter=5000,
                  runner=None, data_dir=None, verbose=True):
    """Minimize t over the base relaxation (no window PSDs). Linear."""
    t0 = time.time()
    prob = build_problem(d, order, bandwidth,
                         add_upper_loc=add_upper_loc, use_z2=use_z2,
                         include_all_windows=False, verbose=verbose)
    res = runner.solve(prob, mode='optimize', t_val=None,
                       sdpnal_tol=sdpnal_tol,
                       sdpnal_maxiter=sdpnal_maxiter,
                       data_dir=data_dir,
                       tag=f'd{d}_o{order}_bw{bandwidth}_r0',
                       verbose=verbose)
    val = val_d_known.get(d)
    lb = res.get('t_opt') or res.get('obj') or 0.0
    gc = (100.0 * (lb - 1.0) / (val - 1.0)
          if val and val > 1.0 else float('nan'))
    return {
        'lb': lb, 'status': _classify(res.get('termcode', -999)),
        'gap_closure': gc, 'val_d': val,
        'kkt': res.get('kkt'), 'pri_res': res.get('pri_res'),
        'dual_res': res.get('dual_res'),
        'wall_s': res.get('wall_s'),
        'elapsed': time.time() - t0,
        'd': d, 'order': order, 'bandwidth': bandwidth,
        'mode': 'round0', 'sound': True, 'solver': 'sdpnalplus',
        '_raw': res,
    }


def solve_feasibility(d, order, bandwidth, target,
                       add_upper_loc=True, use_z2=True,
                       sdpnal_tol=1e-7, sdpnal_maxiter=5000,
                       runner=None, data_dir=None, verbose=True):
    """Certify lb > target in a single SDPNAL+ feasibility solve."""
    t0 = time.time()
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SDPNAL+ FEASIBILITY: prove lb > {target} at "
              f"d={d} L{order} bw={bandwidth}")
        print(f"{'=' * 70}", flush=True)

    prob = build_problem(d, order, bandwidth,
                         add_upper_loc=add_upper_loc, use_z2=use_z2,
                         include_all_windows=True, t_val=target,
                         verbose=verbose)
    tag = f'd{d}_o{order}_bw{bandwidth}_feas_t{target:.6f}'.replace(
        '.', 'p')
    res = runner.solve(prob, mode='feasibility', t_val=target,
                       sdpnal_tol=sdpnal_tol,
                       sdpnal_maxiter=sdpnal_maxiter,
                       data_dir=data_dir, tag=tag, verbose=verbose)
    status = _classify(res.get('termcode', -999))
    certified = (status == 'primal_infeas')
    val = val_d_known.get(d)
    return {
        'certified': certified, 'target': target, 'status': status,
        'lb': (target if certified else None), 'val_d': val,
        'kkt': res.get('kkt'), 'pri_res': res.get('pri_res'),
        'dual_res': res.get('dual_res'),
        'wall_s': res.get('wall_s'), 'elapsed': time.time() - t0,
        'd': d, 'order': order, 'bandwidth': bandwidth,
        'mode': 'feasibility', 'sound': True, 'solver': 'sdpnalplus',
        '_raw': res,
    }


def solve_bisection(d, order, bandwidth, target=None,
                     add_upper_loc=True, use_z2=True,
                     n_bisect=12, bisect_tol=1e-4,
                     sdpnal_tol=1e-7, sdpnal_maxiter=5000,
                     runner=None, data_dir=None, verbose=True):
    """Full bisection over t with all window PSDs active.

    Uses in-place A.data updates (no matrix rebuild) and warm-starts
    X,y,Z between calls.
    """
    t_total = time.time()

    # Stage A: scalar lb (min t, linear).
    if verbose:
        print("\n[STAGE A] Round-0 scalar optimization "
              "(linear, no window PSDs)...", flush=True)
    ra = solve_round0(
        d, order, bandwidth, add_upper_loc=add_upper_loc,
        use_z2=use_z2, sdpnal_tol=sdpnal_tol,
        sdpnal_maxiter=sdpnal_maxiter, runner=runner,
        data_dir=data_dir, verbose=verbose)
    scalar_lb = ra['lb']
    if verbose:
        print(f"  scalar_lb = {scalar_lb:.10f} "
              f"(status={ra['status']}, "
              f"wall={ra['wall_s']:.1f}s)", flush=True)

    # Stage B: build full problem with all windows.
    if verbose:
        print("\n[STAGE B] Building full problem with all window PSDs...",
              flush=True)
    prob = build_problem(d, order, bandwidth,
                         add_upper_loc=add_upper_loc,
                         use_z2=use_z2,
                         include_all_windows=True,
                         t_val=1.0, verbose=verbose)

    val = val_d_known.get(d)
    lo = max(scalar_lb - 1e-3, 0.5)
    hi = (val or 1.5) + 0.02
    best_lb = lo
    history = []
    warm = None  # warm-start payload X, y, Z

    # Target-first probe: if user set a target, test it immediately.
    if target is not None and lo < target < hi:
        queue = [('target', target)]
    else:
        queue = []

    if verbose:
        print(f"\n[STAGE C] Bisection: lo={lo:.6f}  hi={hi:.6f}  "
              f"val({d})={val}  target={target}  "
              f"n_bisect={n_bisect}  bisect_tol={bisect_tol}",
              flush=True)

    for step in range(1, n_bisect + 1):
        if queue:
            label, mid = queue.pop(0)
        else:
            label = f'b{step}'
            mid = 0.5 * (lo + hi)

        if hi - lo < bisect_tol and not queue:
            if verbose:
                print(f"  Bracket converged below {bisect_tol}; stop.",
                      flush=True)
            break

        update_problem_t(prob, mid)
        tag = f'd{d}_o{order}_bw{bandwidth}_{label}'
        t_step = time.time()
        res = runner.solve(prob, mode='feasibility', t_val=mid,
                           sdpnal_tol=sdpnal_tol,
                           sdpnal_maxiter=sdpnal_maxiter,
                           data_dir=data_dir, tag=tag,
                           warm_start=warm, verbose=verbose)
        status = _classify(res.get('termcode', -999))
        dt = time.time() - t_step

        if status == 'solved':
            verdict = 'feas'
            hi = mid
        elif status == 'primal_infeas':
            verdict = 'infeas'
            lo = mid
            best_lb = max(best_lb, mid)
        else:
            verdict = 'uncertain'

        # Soundness audit: bracket invariant must hold.
        if lo > hi + 1e-8:
            print("  !! Monotonicity violation: lo > hi. "
                  "Accepting verdict but flagging audit.", flush=True)

        # Warm-start the next call.
        warm = {
            'X': res.get('warm_X'),
            'y': res.get('warm_y'),
            'Z': res.get('warm_Z'),
        }

        history.append({
            'step': step, 'label': label, 't': mid, 'verdict': verdict,
            'status': status, 'kkt': res.get('kkt'),
            'wall_s': dt,
            'termcode': res.get('termcode'),
        })

        if verbose:
            print(f"  [{step}/{n_bisect}] t={mid:.8f}  {verdict}  "
                  f"(status={status}, KKT={res.get('kkt')}, {dt:.1f}s)",
                  flush=True)

        # Early termination if target cleared.
        if (target is not None and label == 'target'
                and verdict == 'infeas'):
            if verbose:
                print(f"\n  *** TARGET CERTIFIED: lb >= {target} ***",
                      flush=True)
            best_lb = max(best_lb, target)
            break

    elapsed = time.time() - t_total
    gc = (100.0 * (best_lb - 1.0) / (val - 1.0)
          if val and val > 1.0 else float('nan'))

    return {
        'lb': best_lb, 'scalar_lb': scalar_lb, 'val_d': val,
        'gap_closure': gc, 'elapsed': elapsed, 'history': history,
        'd': d, 'order': order, 'bandwidth': bandwidth,
        'mode': 'bisection', 'sound': True, 'solver': 'sdpnalplus',
        'use_z2': use_z2, 'add_upper_loc': add_upper_loc,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SDPNAL+ solver for Lasserre SDP')
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--bw', type=int, required=True,
                        help='Bandwidth; use bw=d-1 for "full L3".')
    parser.add_argument('--mode', choices=['round0', 'feasibility',
                                            'bisection'],
                        default='bisection')
    parser.add_argument('--target', type=float, default=1.2802)
    parser.add_argument('--no-upper-loc', action='store_true')
    parser.add_argument('--no-z2', action='store_true')
    parser.add_argument('--n-bisect', type=int, default=12)
    parser.add_argument('--bisect-tol', type=float, default=1e-4)
    parser.add_argument('--sdpnal-tol', type=float, default=1e-7)
    parser.add_argument('--sdpnal-maxiter', type=int, default=5000)
    parser.add_argument('--matlab-bin', type=str, default='matlab')
    parser.add_argument('--no-engine', action='store_true',
                        help='Force subprocess fallback even if '
                             'matlab.engine is installed.')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--mem-monitor', action='store_true',
                        default=True,
                        help='Spawn background RSS/cgroup monitor.')
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data',
        f'sdpnal_d{args.d}_o{args.order}_bw{args.bw}_'
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(data_dir, exist_ok=True)

    print(f"SDPNAL+ Lasserre: d={args.d} L{args.order} bw={args.bw} "
          f"mode={args.mode}")
    print(f"Data dir: {data_dir}")
    print(f"Started:  {datetime.now().isoformat()}")
    print(f"matlab.engine available: {HAS_MATLAB_ENGINE}")

    if args.mem_monitor:
        _start_memory_monitor(interval_s=60)

    runner = MatlabRunner(matlab_bin=args.matlab_bin,
                          use_engine=(not args.no_engine),
                          verbose=True)
    try:
        runner.check_sdpnalplus()
    except RuntimeError as e:
        print(f"PREFLIGHT FAILED: {e}", file=sys.stderr)
        sys.exit(2)

    add_upper_loc = not args.no_upper_loc
    use_z2 = not args.no_z2

    try:
        if args.mode == 'round0':
            result = solve_round0(
                args.d, args.order, args.bw,
                add_upper_loc=add_upper_loc, use_z2=use_z2,
                sdpnal_tol=args.sdpnal_tol,
                sdpnal_maxiter=args.sdpnal_maxiter,
                runner=runner, data_dir=data_dir, verbose=True)
        elif args.mode == 'feasibility':
            result = solve_feasibility(
                args.d, args.order, args.bw, args.target,
                add_upper_loc=add_upper_loc, use_z2=use_z2,
                sdpnal_tol=args.sdpnal_tol,
                sdpnal_maxiter=args.sdpnal_maxiter,
                runner=runner, data_dir=data_dir, verbose=True)
        else:
            result = solve_bisection(
                args.d, args.order, args.bw, target=args.target,
                add_upper_loc=add_upper_loc, use_z2=use_z2,
                n_bisect=args.n_bisect, bisect_tol=args.bisect_tol,
                sdpnal_tol=args.sdpnal_tol,
                sdpnal_maxiter=args.sdpnal_maxiter,
                runner=runner, data_dir=data_dir, verbose=True)
    finally:
        runner.stop()

    # Strip non-JSON-serialisable warm-start payloads before saving.
    for k in ('_raw',):
        result.pop(k, None)

    print(f"\n{'=' * 70}")
    print(f"FINAL: d={args.d} L{args.order} bw={args.bw} "
          f"mode={args.mode}")
    for k, v in result.items():
        if k == 'history':
            continue
        print(f"  {k} = {v}")
    print(f"{'=' * 70}")

    out_json = os.path.join(
        data_dir, f'result_d{args.d}_o{args.order}_bw{args.bw}_'
                  f'{args.mode}.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Result saved to {out_json}")


if __name__ == '__main__':
    main()
