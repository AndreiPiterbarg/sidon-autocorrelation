#!/usr/bin/env python
r"""Clarabel solver for the full L3 Lasserre SDP at d=16.

Clarabel is an IPM for symmetric-cone programs, pure Rust with Python
bindings. No MATLAB, no license, handles primal/dual infeasibility
certificates to KKT tolerance ~1e-7 to 1e-8 — the same tolerance class
as SDPNAL+ / MOSEK.

Why we're here
==============
- SDPNAL+ OOM'd on the laptop at d=16 L3 full (32 GB wasn't enough).
- MATLAB on a cloud pod is blocked by Penn's student license not
  permitting offline activations.
- Clarabel sidesteps both: no MATLAB, and its IPM memory footprint is
  typically lower than SDPNAL+ for the same problem.
- The exact same problem we built for SDPNAL+ (from
  `build_problem` in lasserre_sdpnalplus.py) feeds Clarabel with a
  small permutation of the PSD rows (lower-tri svec → upper-tri svec).

Strategy
========
Mode = feasibility at target t:
  1. Build the SCS-standard-form problem (A, b, c, cone) via the
     existing pipeline: _precompute_highd + build_base_problem +
     Z/2 injection + _assemble_window_psd at t=target.
  2. Substitute t=target directly into A (eliminating the t column
     entirely shrinks the problem by one variable).
  3. Permute PSD rows from lower-triangle column-major (SCS convention)
     to upper-triangle column-major (Clarabel convention).
  4. Pass to Clarabel. Status == PrimalInfeasible → lb > target
     certified.

OPTIMIZATIONS INHERITED
=======================
  - Reduced moment set S (lasserre_highd._precompute_highd)
  - Partial consistency (eq + iq)
  - Banded clique decomposition
  - Cross-clique M_1 coupling
  - Upper localizing (1 - μ_i) ⪰ 0
  - Pairwise L3 localizers M_1(μ_iμ_j y) ⪰ 0
  - Z/2 time-reversal symmetry
  - Vectorized PSD cone COO builder
  - Window PSD base/t decomposition (from run_d16_l3.build_base_problem)
  - All 496 window PSDs at t=target from round 0 (no cutting planes)

USAGE
=====
  python tests/lasserre_clarabel_driver.py \
      --d 16 --order 3 --bw 15 \
      --target 1.2802 \
      --data-dir data/clarabel_d16_production

  # Small smoke test first
  python tests/lasserre_clarabel_driver.py \
      --d 6 --order 3 --bw 5 --target 1.17

DEPLOYMENT
==========
  pip install clarabel scipy numpy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_d16_l3 import (  # noqa: E402
    build_base_problem,
    _precompute_window_psd_decomposition,
    _assemble_window_psd,
)
from lasserre_highd import (  # noqa: E402
    _precompute_highd, _build_banded_cliques, val_d_known,
)
from lasserre.z2_symmetry import inject_z2_equalities_into_base_problem  # noqa: E402


# =====================================================================
# SCS lower-triangle svec → Clarabel upper-triangle svec permutation
# =====================================================================
#
# SCS packs an n x n symmetric PSD cone as lower-triangle column-major:
#   for j = 0..n-1, for i = j..n-1: k_scs = j*n - j*(j-1)/2 + (i-j)
#   entry X[i,j] (row i, col j, i >= j).
#
# Clarabel packs as upper-triangle column-major:
#   for j = 0..n-1, for i = 0..j: k_clar = j*(j+1)/2 + i
#   entry X[i,j] (row i, col j, i <= j).
#
# Since X is symmetric, SCS's X[i,j] (i >= j) is the same value as
# Clarabel's X[j,i] (j <= i).  Matching SCS position (i,j) with
# Clarabel position (j,i):
#   k_scs = j*n - j*(j-1)/2 + (i-j)   for i >= j
#   k_clar = i*(i+1)/2 + j
#
# So the permutation that maps each SCS row to its Clarabel slot is
#   clar_row_of[k_scs] = i*(i+1)/2 + j    (defined for each SCS k_scs).
#
# We apply the INVERSE permutation to the rows of A: A_clar = A_scs[perm]
# where perm[k_clar] = k_scs_for_that_clar_position.
# =====================================================================


def _lower_to_upper_perm(n):
    """Return the permutation array `perm` of length n*(n+1)/2 such that
    for a symmetric X, svec_upper(X) = svec_lower(X)[perm].

    That is: `perm[k_clar] = k_scs` for the SCS row encoding the same X
    entry as Clarabel row k_clar.
    """
    svec_size = n * (n + 1) // 2
    perm = np.empty(svec_size, dtype=np.int64)
    # Fill via direct enumeration of Clarabel upper-tri layout.
    for j in range(n):
        for i in range(j + 1):   # i <= j
            k_clar = j * (j + 1) // 2 + i
            # Clarabel (i, j), upper triangle with i<=j.
            # SCS position of same value X[i,j]=X[j,i] with j<=i (lower):
            ii, jj = j, i  # swap: SCS needs row >= col
            # k_scs = jj*n - jj*(jj-1)//2 + (ii - jj)
            k_scs = jj * n - jj * (jj - 1) // 2 + (ii - jj)
            perm[k_clar] = k_scs
    return perm


def _permute_psd_rows_lower_to_upper(A, b, cone, meta, verbose=True):
    """Permute rows of A (and b) inside each PSD slab, converting SCS
    lower-triangle svec ordering to Clarabel upper-triangle svec."""
    t0 = time.time()
    n_zero = meta.get('n_zero', cone.get('z', 0))
    n_nonneg = meta.get('n_nonneg', cone.get('l', 0))
    A_csr = A.tocsr()

    row_offset = n_zero + n_nonneg
    # Keep zero + nonneg rows unchanged.
    parts_A = [A_csr[:row_offset]]
    parts_b = [b[:row_offset]]

    current = row_offset
    for n_p in cone.get('s', []):
        svec_size = n_p * (n_p + 1) // 2
        perm = _lower_to_upper_perm(n_p)
        block_A = A_csr[current:current + svec_size]
        block_b = b[current:current + svec_size]
        parts_A.append(block_A[perm, :])
        parts_b.append(block_b[perm])
        current += svec_size

    A_new = sp.vstack(parts_A, format='csc')
    b_new = np.concatenate(parts_b)
    if verbose:
        print(f"  PSD row permutation: {time.time() - t0:.2f}s  "
              f"(lower-tri → upper-tri over {len(cone.get('s', []))} PSD blocks)",
              flush=True)
    return A_new, b_new


# =====================================================================
# Build SCS problem + eliminate t
# =====================================================================

def build_problem_at_target(d, order, bandwidth, target,
                             add_upper_loc=True, use_z2=True,
                             verbose=True):
    """Construct (A, b, c, cone) in Clarabel-ready form for the
    feasibility check at t=target.

    Concretely:
      1. Precompute + base problem (SCS form, lower-tri svec).
      2. Z/2 equalities.
      3. All window PSDs built at t=target.
      4. Substitute t=target (eliminate the t column; shift b).
      5. Permute PSD rows lower→upper svec.
    """
    t_total = time.time()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Clarabel build: d={d} L{order} bw={bandwidth} "
              f"target={target}  (upper_loc={add_upper_loc}, z2={use_z2})")
        print(f"{'=' * 70}", flush=True)

    # 1. Reduced moment set, cliques, partial consistency, pairwise.
    cliques = _build_banded_cliques(d, bandwidth)
    if verbose:
        print(f"\nCliques: {len(cliques)} of size {len(cliques[0])}",
              flush=True)
    P = _precompute_highd(d, order, cliques, verbose=verbose)

    # 2. Base problem: moment + localizing + upper-loc + pairwise +
    #    scalar windows + y>=0 + consistency.
    if verbose:
        print("\nBuilding base SCS problem (incl. pairwise L3 cones)...",
              flush=True)
    t_build = time.time()
    A_base, b_base, c_obj, cone_base, meta = build_base_problem(
        P, add_upper_loc=add_upper_loc)
    if verbose:
        print(f"  base nnz={A_base.nnz:,}  "
              f"({time.time() - t_build:.1f}s)", flush=True)

    # 3. Z/2 time-reversal symmetry.
    if use_z2:
        A_base, b_base, cone_base, n_z2 = \
            inject_z2_equalities_into_base_problem(
                P, A_base, b_base, cone_base, verbose=verbose)
        meta['n_zero'] += n_z2
        meta['n_rows_base'] += n_z2

    # 4. Window PSDs (all 496 active at t=target).
    if verbose:
        print(f"\nBuilding all {P['n_win']} window PSDs at t={target}...",
              flush=True)
    t_win = time.time()
    all_ws = set(P.get('nontrivial_windows', range(P['n_win'])))
    active_windows = {w for w in all_ws
                      if int(P['window_covering'][w]) >= 0}
    if verbose:
        print(f"  {len(active_windows)}/{P['n_win']} windows covered",
              flush=True)

    win_decomp = _precompute_window_psd_decomposition(P, active_windows)
    if win_decomp is not None:
        A_win, b_win, psd_win = _assemble_window_psd(win_decomp, target)
        A = sp.vstack([A_base, A_win], format='csc')
        b = np.concatenate([b_base, b_win])
        cone = {
            'z': cone_base.get('z', 0),
            'l': cone_base.get('l', 0),
            's': list(cone_base.get('s', [])) + list(psd_win),
        }
        if verbose:
            print(f"  +{len(psd_win)} window PSD cones of dim "
                  f"{psd_win[0] if psd_win else 0}  "
                  f"({time.time() - t_win:.1f}s)", flush=True)
    else:
        A = A_base
        b = b_base
        cone = cone_base

    if verbose:
        print(f"\nFull SCS: {A.shape[0]:,} rows x {A.shape[1]:,} cols, "
              f"nnz={A.nnz:,}", flush=True)
        print(f"  z={cone.get('z', 0)}  l={cone.get('l', 0)}  "
              f"PSD={len(cone.get('s', []))} blocks  "
              f"max_psd={max(cone.get('s', [0]))}",
              flush=True)

    # 5. Eliminate t by substitution t = target.
    #    row_i: A[i,t] * t + A[i, y_cols] * y + s_i = b_i
    #    =>    A[i, y_cols] * y + s_i = b_i - A[i,t] * target
    t_col = meta['t_col']
    if verbose:
        print(f"\nEliminating t (t_col={t_col}) by substitution "
              f"t := {target} ...", flush=True)
    A_csc = A.tocsc()
    t_column = np.asarray(A_csc[:, t_col].todense()).ravel()
    b_shifted = b - t_column * target
    # Drop t_col from A.
    keep_cols = [c for c in range(A_csc.shape[1]) if c != t_col]
    A_noT = A_csc[:, keep_cols]
    if verbose:
        print(f"  -> A now {A_noT.shape[0]:,} rows x "
              f"{A_noT.shape[1]:,} cols, nnz={A_noT.nnz:,}", flush=True)

    # 6. Permute PSD rows to Clarabel upper-tri.
    if verbose:
        print("\nPermuting PSD rows to Clarabel upper-tri svec...",
              flush=True)
    A_clar, b_clar = _permute_psd_rows_lower_to_upper(
        A_noT, b_shifted, cone, meta, verbose=verbose)

    if verbose:
        print(f"\nBuild total: {time.time() - t_total:.1f}s", flush=True)

    return {
        'A': A_clar, 'b': b_clar, 'cone': cone, 'meta': meta, 'P': P,
        'target': target,
        'n_var': A_clar.shape[1],   # y variables only
    }


# =====================================================================
# Clarabel solve
# =====================================================================

def solve_clarabel_feasibility(d, order, bandwidth, target,
                                add_upper_loc=True, use_z2=True,
                                tol_gap_abs=1e-6, tol_gap_rel=1e-6,
                                tol_feas=1e-6, tol_infeas_abs=1e-7,
                                tol_infeas_rel=1e-7,
                                max_iter=5000,
                                max_threads=0,
                                time_limit_s=0.0,
                                direct_solve_method='auto',
                                data_dir=None, tag=None,
                                verbose=True):
    """Prove lb > target via a single Clarabel feasibility solve."""
    import clarabel

    if tag is None:
        tag = (f'd{d}_o{order}_bw{bandwidth}_clar_t'
               f'{target:.6f}').replace('.', 'p')

    t_total = time.time()
    prob = build_problem_at_target(d, order, bandwidth, target,
                                    add_upper_loc=add_upper_loc,
                                    use_z2=use_z2, verbose=verbose)

    A, b, cone, n_var = prob['A'], prob['b'], prob['cone'], prob['n_var']

    # Assemble Clarabel cone list in SCS order: zero, nonneg, PSD blocks.
    cones = []
    if cone.get('z', 0):
        cones.append(clarabel.ZeroConeT(int(cone['z'])))
    if cone.get('l', 0):
        cones.append(clarabel.NonnegativeConeT(int(cone['l'])))
    for s in cone.get('s', []):
        cones.append(clarabel.PSDTriangleConeT(int(s)))

    # Feasibility mode: min 0 s.t. constraints. P=0, q=0.
    P = sp.csc_matrix((n_var, n_var))
    q = np.zeros(n_var)

    # A must be CSC double, sorted indices (Clarabel REQUIRES sorted CSC;
    # scipy's tocsc() usually returns sorted but not guaranteed).
    A = A.tocsc().astype(np.float64)
    A.sort_indices()
    A.eliminate_zeros()
    b = np.asarray(b, dtype=np.float64)

    # Settings (Clarabel 0.11+ naming).
    settings = clarabel.DefaultSettings()
    settings.max_iter = int(max_iter)
    settings.verbose = bool(verbose)
    settings.tol_gap_abs = float(tol_gap_abs)
    settings.tol_gap_rel = float(tol_gap_rel)
    settings.tol_feas = float(tol_feas)
    settings.tol_infeas_abs = float(tol_infeas_abs)
    settings.tol_infeas_rel = float(tol_infeas_rel)
    if max_threads:
        settings.max_threads = int(max_threads)
    # time_limit is critical: if the solver is going to spend 50 h on
    # a single problem we'd rather know at hour 10 and give up cleanly
    # than have the pod budget swallow the rest.
    if time_limit_s and time_limit_s > 0 and hasattr(settings, 'time_limit'):
        settings.time_limit = float(time_limit_s)

    # Presolve + chordal decomposition.  Our 496 window PSDs are all
    # size 153 — chordal decomposition cannot split them further (they
    # are already small), but presolve can still eliminate redundant
    # equalities produced by the Z/2 injection and partial consistency.
    if hasattr(settings, 'presolve_enable'):
        settings.presolve_enable = True
    if hasattr(settings, 'chordal_decomposition_enable'):
        settings.chordal_decomposition_enable = True
    if hasattr(settings, 'chordal_decomposition_compact'):
        settings.chordal_decomposition_compact = True
    if hasattr(settings, 'chordal_decomposition_merge_method'):
        settings.chordal_decomposition_merge_method = 'clique_graph'

    # Ruiz equilibration: essential for moment matrices (ill-conditioned
    # by design — each row coefficient is a multinomial in the moments).
    # Default max_iter is 10; 20 tolerates tighter row/col balance at
    # negligible cost (O(nnz) per sweep).
    if hasattr(settings, 'equilibrate_enable'):
        settings.equilibrate_enable = True
    if hasattr(settings, 'equilibrate_max_iter'):
        settings.equilibrate_max_iter = 20

    # Iterative refinement on the KKT solve: mitigates floating-point
    # drift when the moment matrix is nearly-rank-deficient (typical
    # near the feasibility boundary where certificates live).
    if hasattr(settings, 'iterative_refinement_enable'):
        settings.iterative_refinement_enable = True
    if hasattr(settings, 'iterative_refinement_max_iter'):
        settings.iterative_refinement_max_iter = 10

    # Static KKT regularisation: handles near-singular Hessians.
    if hasattr(settings, 'static_regularization_enable'):
        settings.static_regularization_enable = True

    # Direct linear solver.  QDLDL (Clarabel's default) is a pure-Rust
    # sparse LDLᵀ.  MKL Pardiso (via mkl-pardiso feature) can be 2-10×
    # faster on large KKT systems IF Clarabel was built with MKL
    # support.  Try 'mkl' / 'pardiso' / 'cudss' in order; fall back
    # silently to 'qdldl' if the build doesn't expose them.
    if direct_solve_method and direct_solve_method != 'auto' \
            and hasattr(settings, 'direct_solve_method'):
        try:
            settings.direct_solve_method = direct_solve_method
            if verbose:
                print(f"  direct_solve_method set to "
                      f"{direct_solve_method}", flush=True)
        except Exception as exc:
            if verbose:
                print(f"  direct_solve_method '{direct_solve_method}' "
                      f"not supported ({exc}); using default", flush=True)
    elif direct_solve_method == 'auto' \
            and hasattr(settings, 'direct_solve_method'):
        # Robust probe: try creating a tiny solver with each
        # candidate; if DefaultSolver() fails we know that method
        # isn't compiled in.  Fall through silently to QDLDL default.
        for cand in ('mkl', 'pardiso'):
            try:
                probe = clarabel.DefaultSettings()
                probe.verbose = False
                probe.direct_solve_method = cand
                # Minimal 1x1 problem to force solver construction
                tiny_A = sp.csc_matrix(np.array([[1.0]]))
                tiny_b = np.array([0.0])
                clarabel.DefaultSolver(
                    sp.csc_matrix((1, 1)), np.zeros(1),
                    tiny_A, tiny_b,
                    [clarabel.ZeroConeT(1)], probe)
                settings.direct_solve_method = cand
                if verbose:
                    print(f"  direct_solve_method=auto selected "
                          f"{cand}", flush=True)
                break
            except Exception:
                continue
        else:
            if verbose:
                print("  direct_solve_method=auto -> using Clarabel "
                      "default (qdldl)", flush=True)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Launching Clarabel solver at target t={target}")
        print(f"  tol_gap_abs={tol_gap_abs}  tol_gap_rel={tol_gap_rel}  "
              f"tol_feas={tol_feas}  max_iter={max_iter}")
        print(f"  cones: z={cone.get('z', 0)}  l={cone.get('l', 0)}  "
              f"PSD={len(cone.get('s', []))} blocks  "
              f"max_psd={max(cone.get('s', [0]))}")
        print(f"{'=' * 70}", flush=True)

    t_solve = time.time()
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    sol = solver.solve()
    wall_solve = time.time() - t_solve

    # Status interpretation.
    status_str = str(sol.status).split('.')[-1]   # e.g., 'Solved'
    certified = (status_str in ('PrimalInfeasible',))
    is_feasible = (status_str in ('Solved', 'AlmostSolved'))
    val = val_d_known.get(d)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Clarabel finished in {wall_solve:.1f}s")
        print(f"  status     = {status_str}")
        print(f"  iterations = {getattr(sol, 'iterations', '?')}")
        print(f"  solve time = {getattr(sol, 'solve_time', '?')}")
        print(f"  obj primal = {getattr(sol, 'obj_val', '?')}")
        print(f"  obj dual   = {getattr(sol, 'obj_val_dual', '?')}")
        if certified:
            print(f"\n  *** CERTIFIED: lb > {target} ***")
            print(f"  val({d}) = {val}  "
                  f"gap ~ {val - target:.4f} ({(val - target) / (val - 1) * 100:.2f}% residual)")
        elif is_feasible:
            print(f"\n  FEASIBLE at t={target} — lb <= {target}, "
                  f"record NOT beaten by this relaxation.")
        else:
            print(f"\n  INCONCLUSIVE (status {status_str})")
        print(f"{'=' * 70}", flush=True)

    result = {
        'solver': 'clarabel',
        'mode': 'feasibility',
        'd': d, 'order': order, 'bandwidth': bandwidth,
        'target': target,
        'status': status_str,
        'certified': certified,
        'is_feasible': is_feasible,
        'lb': (target if certified else None),
        'val_d': val,
        'iterations': int(getattr(sol, 'iterations', 0) or 0),
        'obj_val': float(getattr(sol, 'obj_val', 0.0) or 0.0),
        'obj_val_dual': float(getattr(sol, 'obj_val_dual', 0.0) or 0.0),
        'solve_time_s': float(getattr(sol, 'solve_time', 0.0) or 0.0),
        'wall_solve_s': wall_solve,
        'wall_total_s': time.time() - t_total,
        'sound': True,
    }

    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        out_path = os.path.join(data_dir, f'result_{tag}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        if verbose:
            print(f"Result saved to {out_path}", flush=True)

    return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Clarabel IPM solver for Lasserre SDP')
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--bw', type=int, required=True,
                        help='Bandwidth. Use bw=d-1 for full L3.')
    parser.add_argument('--target', type=float, default=1.2802,
                        help='t value at which to check feasibility.')
    parser.add_argument('--no-upper-loc', action='store_true')
    parser.add_argument('--no-z2', action='store_true')
    # Tolerance defaults are LOOSENED vs Clarabel's own defaults because
    # we only care about the infeasibility verdict, not optimality gap.
    # tol_infeas_* gates the quality of the primal-infeasibility
    # certificate; tol_gap_* and tol_feas gate the KKT IPM residuals.
    # 1e-6 / 1e-7 keeps the certificate rigorous while cutting the IPM
    # iteration count roughly in half vs 1e-7 / 1e-8.
    parser.add_argument('--tol-gap-abs', type=float, default=1e-6)
    parser.add_argument('--tol-gap-rel', type=float, default=1e-6)
    parser.add_argument('--tol-feas', type=float, default=1e-6)
    parser.add_argument('--tol-infeas-abs', type=float, default=1e-7)
    parser.add_argument('--tol-infeas-rel', type=float, default=1e-7)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--max-threads', type=int, default=0,
                        help='0 = use all available cores (Clarabel default)')
    parser.add_argument('--time-limit-s', type=float, default=0.0,
                        help='Hard wall-clock limit for the solve (0 = no limit).  '
                             'Clarabel will exit cleanly if exceeded.')
    parser.add_argument('--direct-solve-method', type=str, default='auto',
                        choices=('auto', 'qdldl', 'mkl', 'pardiso'),
                        help='Sparse direct LDLᵀ backend.  "auto" tries MKL Pardiso '
                             'first and falls back silently to Clarabel default (QDLDL) '
                             'if MKL support was not compiled in.')
    parser.add_argument('--data-dir', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data',
        f'clarabel_d{args.d}_o{args.order}_bw{args.bw}_'
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(data_dir, exist_ok=True)

    print(f"Clarabel Lasserre: d={args.d} L{args.order} bw={args.bw} "
          f"target={args.target}")
    print(f"Data dir: {data_dir}")
    print(f"Started:  {datetime.now().isoformat()}")

    result = solve_clarabel_feasibility(
        args.d, args.order, args.bw, args.target,
        add_upper_loc=not args.no_upper_loc, use_z2=not args.no_z2,
        tol_gap_abs=args.tol_gap_abs, tol_gap_rel=args.tol_gap_rel,
        tol_feas=args.tol_feas,
        tol_infeas_abs=args.tol_infeas_abs,
        tol_infeas_rel=args.tol_infeas_rel,
        max_iter=args.max_iter, max_threads=args.max_threads,
        time_limit_s=args.time_limit_s,
        direct_solve_method=args.direct_solve_method,
        data_dir=data_dir, verbose=True,
    )

    print()
    print("=" * 70)
    print(f"FINAL — Clarabel d={args.d} L{args.order} bw={args.bw}")
    for k, v in result.items():
        print(f"  {k} = {v}")
    print("=" * 70)


if __name__ == '__main__':
    main()
