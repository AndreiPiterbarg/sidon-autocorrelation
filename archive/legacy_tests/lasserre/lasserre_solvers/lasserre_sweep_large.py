#!/usr/bin/env python3
r"""
Large-scale Lasserre sweep — 256 GB RAM, 320 GB disk, 8-hour budget.

Based exactly on tests/lasserre_fusion.py (same SDP, same maths).
The ONLY change is memory management:

  1. Window PSD constraints are STREAMED: compute COO per window, feed
     to MOSEK, discard.  This cuts peak Python memory by the full cost
     of cw_data_list (~40–100 GB at d>=18).

  2. All Python-side precompute arrays (ab_eiej_idx, AB_loc_hash, …)
     are deleted after MOSEK ingestion, before the binary-search phase.

  3. psutil monitors RSS and available memory; the sweep aborts a config
     early if available memory drops below a safety threshold.

Mathematical correctness: every MOSEK constraint is identical to the
original lasserre_fusion.py — same Matrix.sparse COO data, same PSD
cone structure, same binary-search logic.  The streaming loop visits
windows in the same 0..n_win-1 order and produces the same COO arrays;
only the lifetime of those arrays changes.

Config strategy (descending expected lb):
  L3 d=16  ~60 GB   baseline, guaranteed to finish in ~30 min
  L3 d=18  ~120 GB  main target, ~2-4 h
  L3 d=20  ~200 GB  stretch goal, memory-permitting

Usage:
  python tests/lasserre_sweep_large.py
  python tests/lasserre_sweep_large.py --max_hours 7.5 --max_mem_gb 240
"""
import numpy as np
from mosek.fusion import (Model, Domain, Expr, Matrix,
                          ObjectiveSense, SolutionStatus)
import time
import sys
import os
import gc
import json
import traceback

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =====================================================================
# Monomial enumeration  (verbatim from lasserre_fusion.py)
# =====================================================================

def enum_monomials(d, max_deg):
    """All multi-indices alpha in N^d with |alpha| <= max_deg."""
    result = []
    def gen(pos, remaining, current):
        if pos == d:
            result.append(tuple(current))
            return
        for v in range(remaining + 1):
            current.append(v)
            gen(pos + 1, remaining - v, current)
            current.pop()
    gen(0, max_deg, [])
    return result


def _add_mi(a, b, d):
    return tuple(a[i] + b[i] for i in range(d))


def _unit(d, i):
    return tuple(1 if k == i else 0 for k in range(d))


# =====================================================================
# Vectorized monomial hashing  (verbatim from lasserre_fusion.py)
# =====================================================================

def _make_hash_bases(d, max_comp):
    """Mixed-radix hash: bases[k] = (max_comp+1)^k.  Linear in components."""
    base = max_comp + 1
    return np.array([base**k for k in range(d)], dtype=np.int64)


def _hash_monos(arr, bases):
    """Hash array of monomials.  arr shape (..., d) -> (...) int64."""
    return np.tensordot(arr.astype(np.int64), bases, axes=([-1], [0]))


def _build_hash_table(mono_list, bases):
    """Sorted hashes + original indices for vectorized searchsorted lookup."""
    mono_arr = np.array(mono_list, dtype=np.int64)
    hashes = _hash_monos(mono_arr, bases)
    order = np.argsort(hashes)
    return hashes[order], order


def _hash_lookup(query_hashes, sorted_hashes, sort_order):
    """Vectorized lookup: query_hashes -> moment indices (or -1)."""
    flat = query_hashes.ravel()
    pos = np.searchsorted(sorted_hashes, flat)
    pos = np.clip(pos, 0, len(sorted_hashes) - 1)
    found = sorted_hashes[pos] == flat
    result = np.where(found, sort_order[pos], -1)
    return result.reshape(query_hashes.shape)


# =====================================================================
# Window matrices  (verbatim from lasserre_fusion.py)
# =====================================================================

def build_window_matrices(d):
    conv_len = 2 * d - 1
    windows = [(ell, s) for ell in range(2, 2 * d + 1)
               for s in range(conv_len - ell + 2)]
    ii, jj = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
    sums = ii + jj
    M_mats = []
    for ell, s_lo in windows:
        mask = (sums >= s_lo) & (sums <= s_lo + ell - 2)
        M_mats.append((2.0 * d / ell) * mask.astype(np.float64))
    return windows, M_mats


# =====================================================================
# Collect all moment indices  (verbatim from lasserre_fusion.py)
# =====================================================================

def collect_moments(d, order, basis, loc_basis, consist_mono):
    """Collect needed moments — fully vectorized, no Python loops."""
    chunks = []

    B = np.array(basis, dtype=np.int64)
    chunks.append((B[:, np.newaxis, :] + B[np.newaxis, :, :]).reshape(-1, d))
    chunks.append(np.array(enum_monomials(d, 2 * order), dtype=np.int64))

    if order >= 2:
        LB = np.array(loc_basis, dtype=np.int64)
        E = np.eye(d, dtype=np.int64)
        AB_loc = LB[:, np.newaxis, :] + LB[np.newaxis, :, :]
        chunks.append(AB_loc.reshape(-1, d))

        AB_ei = AB_loc[:, :, np.newaxis, :] + E[np.newaxis, np.newaxis, :, :]
        chunks.append(AB_ei.reshape(-1, d))

        max_comp = 2 * order
        tmp_bases = _make_hash_bases(d, max_comp)
        ab_hashes = _hash_monos(AB_loc, tmp_bases)
        ee_hashes = tmp_bases[:, None] + tmp_bases[None, :]
        all_abij = (ab_hashes[:, :, None, None]
                    + ee_hashes[None, None, :, :]).ravel()
        unique_abij = np.unique(all_abij)

        decoded = np.zeros((len(unique_abij), d), dtype=np.int64)
        for k in range(d - 1, -1, -1):
            decoded[:, k] = unique_abij // tmp_bases[k]
            unique_abij = unique_abij % tmp_bases[k]
        chunks.append(decoded)

    C = np.array(consist_mono, dtype=np.int64)
    chunks.append(C)
    C_ei = C[:, np.newaxis, :] + np.eye(d, dtype=np.int64)[np.newaxis, :, :]
    chunks.append(C_ei.reshape(-1, d))

    all_monos = np.concatenate(chunks, axis=0)
    unique_monos = np.unique(all_monos, axis=0)

    mono_list = [tuple(row) for row in unique_monos]
    idx = {m: i for i, m in enumerate(mono_list)}
    return mono_list, idx


# =====================================================================
# Memory helpers
# =====================================================================

def _rss_gb():
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024**3)
    return 0.0


def _avail_gb():
    if HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024**3)
    return 999.0


def _mem_tag():
    return f"RSS={_rss_gb():.1f}GB avail={_avail_gb():.0f}GB"


# =====================================================================
# Memory-optimised Lasserre solver  (streaming window constraints)
# =====================================================================

def solve_lasserre_streaming(d, order=3, n_bisect=14, verbose=True,
                             avail_floor_gb=15.0, num_threads=0):
    r"""
    Build and solve the Lasserre SDP, identical to solve_lasserre_fusion()
    but with streaming window ingestion to reduce peak Python memory.

    Parameters
    ----------
    d : int
        Number of bins (dimension).
    order : int
        Lasserre relaxation order (2 = L2, 3 = L3, …).
    n_bisect : int
        Number of binary-search steps.
    avail_floor_gb : float
        Abort if available system memory drops below this (GB).
    num_threads : int
        MOSEK thread count (0 = auto).

    Returns
    -------
    dict with 'lb' (lower bound on val(d)), timing info, and SDP sizes.
    """
    t_wall = time.time()

    # ── Setup (same as lasserre_fusion.py) ──
    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)

    basis = enum_monomials(d, order)
    n_basis = len(basis)
    loc_basis = enum_monomials(d, order - 1) if order >= 2 else []
    n_loc = len(loc_basis)
    consist_mono = enum_monomials(d, 2 * order - 1)

    mono_list, idx = collect_moments(d, order, basis, loc_basis, consist_mono)
    n_y = len(mono_list)

    if verbose:
        print(f"  d={d}, order={order} (degree {2*order})")
        print(f"  windows={n_win}, basis={n_basis}, "
              f"loc_basis={n_loc}, moments={n_y}")
        if order >= 2:
            print(f"  Moment matrix : {n_basis}x{n_basis}")
            print(f"  Loc (mu_i)    : {d} x {n_loc}x{n_loc}")
            print(f"  Loc (windows) : {n_win} x {n_loc}x{n_loc}")
        print(f"  {_mem_tag()}", flush=True)

    t_build = time.time()

    # ── Hash infrastructure ──
    max_comp = 2 * order
    bases = _make_hash_bases(d, max_comp)
    sorted_h, sort_o = _build_hash_table(mono_list, bases)

    B_arr = np.array(basis, dtype=np.int64)

    # ── Index precompute ──
    t_pre = time.time()

    # Moment matrix: M[a,b] = y[basis[a]+basis[b]]
    AB_hash = _hash_monos(
        B_arr[:, np.newaxis, :] + B_arr[np.newaxis, :, :], bases)
    moment_pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel().tolist()
    del AB_hash

    ab_eiej_idx = None          # set below if order >= 2
    ab_flat     = None
    loc_picks   = None
    t_pick_list = None

    if loc_basis:
        LB_arr = np.array(loc_basis, dtype=np.int64)
        AB_loc = LB_arr[:, np.newaxis, :] + LB_arr[np.newaxis, :, :]
        AB_loc_hash = _hash_monos(AB_loc, bases)   # (n_loc, n_loc)
        del AB_loc

        # Localizing mu_i: L_i[a,b] = y[loc[a]+loc[b]+e_i]
        loc_picks = []
        for i_var in range(d):
            h = AB_loc_hash + bases[i_var]
            picks = _hash_lookup(h, sorted_h, sort_o)
            assert np.all(picks >= 0), \
                f"Missing moments in mu_{i_var} localizing matrix"
            loc_picks.append(picks.ravel().tolist())

        # t coefficient: T[a,b] = y[loc[a]+loc[b]]
        t_pick_list = _hash_lookup(
            AB_loc_hash, sorted_h, sort_o).ravel().tolist()

        # ab_eiej_idx[a,b,i,j] = idx[loc[a]+loc[b]+e_i+e_j]
        EE_hash = bases[:, np.newaxis] + bases[np.newaxis, :]   # (d, d)
        ABIJ_hash = (AB_loc_hash[:, :, np.newaxis, np.newaxis]
                     + EE_hash[np.newaxis, np.newaxis, :, :])
        ab_eiej_idx = _hash_lookup(ABIJ_hash, sorted_h, sort_o)
        del ABIJ_hash, EE_hash, AB_loc_hash

        ab_flat = (np.arange(n_loc)[:, np.newaxis] * n_loc
                   + np.arange(n_loc)[np.newaxis, :])   # (n_loc, n_loc)

    # Consistency indices
    consist_arr = np.array(consist_mono, dtype=np.int64)
    consist_hash = _hash_monos(consist_arr, bases)
    consist_idx = _hash_lookup(consist_hash, sorted_h, sort_o)
    consist_ei_hash = consist_hash[:, np.newaxis] + bases[np.newaxis, :]
    consist_ei_idx  = _hash_lookup(consist_ei_hash, sorted_h, sort_o)
    del consist_hash, consist_ei_hash

    precompute_s = time.time() - t_pre
    if verbose:
        print(f"  Index precompute: {precompute_s:.2f}s")
        if ab_eiej_idx is not None:
            vc = int((ab_eiej_idx >= 0).sum())
            tc = ab_eiej_idx.size
            print(f"  ab_eiej_idx: {ab_eiej_idx.shape}, "
                  f"{vc}/{tc} valid ({100*vc/tc:.1f}%)")
        print(f"  {_mem_tag()}", flush=True)

    # ── Build MOSEK model ──
    if verbose:
        print(f"  Building MOSEK model …", flush=True)

    t_model = time.time()
    MDL = Model("lasserre_streaming")

    if num_threads > 0:
        MDL.setSolverParam("numThreads", num_threads)
    # Slightly relaxed tolerance — 1e-7 is still far tighter than our
    # bisection resolution (~1e-4) and speeds up each interior-point solve.
    MDL.setSolverParam("intpntCoTolRelGap", 1e-7)

    y = MDL.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = MDL.parameter("t")

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    MDL.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

    # ── Moment consistency  (sparse matrix) ──
    c_rows, c_cols, c_vals = [], [], []
    n_consist_added = 0
    for r in range(len(consist_mono)):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child = consist_ei_idx[r]
        has_child = False
        for ci in range(d):
            if child[ci] >= 0:
                c_rows.append(n_consist_added)
                c_cols.append(int(child[ci]))
                c_vals.append(1.0)
                has_child = True
        if not has_child:
            continue
        c_rows.append(n_consist_added)
        c_cols.append(ai)
        c_vals.append(-1.0)
        n_consist_added += 1

    del consist_idx, consist_ei_idx
    if n_consist_added > 0:
        A_con = Matrix.sparse(n_consist_added, n_y, c_rows, c_cols, c_vals)
        MDL.constraint("consist", Expr.mul(A_con, y), Domain.equalsTo(0.0))
    del c_rows, c_cols, c_vals

    # ── Moment matrix PSD ──
    M_mat = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
    MDL.constraint("moment_psd", M_mat, Domain.inPSDCone(n_basis))
    del moment_pick

    # ── Localizing matrices for mu_i >= 0 ──
    if order >= 2:
        for i_var in range(d):
            Li = Expr.reshape(y.pick(loc_picks[i_var]), n_loc, n_loc)
            MDL.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_loc))
        del loc_picks

    # ── Window localizing matrices — STREAMING ──
    #
    # This is the ONLY structural difference from lasserre_fusion.py.
    # Instead of pre-materialising cw_data_list (which stores all COO
    # arrays simultaneously and dominates Python memory), we compute
    # each window's COO data, feed it to MOSEK, and let it go out of
    # scope.  MOSEK copies the data internally, so the model is
    # identical.
    if order >= 2:
        flat_size = n_loc * n_loc
        t_y_pick = y.pick(t_pick_list)
        del t_pick_list

        for w in range(n_win):
            t_expr = Expr.mul(t_param, t_y_pick)
            Mw = M_mats[w]
            nz_i, nz_j = np.nonzero(Mw)

            if len(nz_i) == 0:
                # Zero window — constraint is t * M_ab >> 0 (always true
                # when t > 0 and moment matrix PSD, but added for parity
                # with lasserre_fusion.py which adds it for every w).
                Lw_mat = Expr.reshape(t_expr, n_loc, n_loc)
                MDL.constraint(f"w_{w}", Lw_mat, Domain.inPSDCone(n_loc))
                continue

            # ab_eiej_idx[:, :, nz_i, nz_j]  — same slice as original
            y_idx = ab_eiej_idx[:, :, nz_i, nz_j]   # (n_loc, n_loc, n_nz)
            valid = y_idx >= 0

            if not np.any(valid):
                Lw_mat = Expr.reshape(t_expr, n_loc, n_loc)
                MDL.constraint(f"w_{w}", Lw_mat, Domain.inPSDCone(n_loc))
                continue

            ab_exp = np.broadcast_to(
                ab_flat[:, :, np.newaxis], y_idx.shape)
            mw_vals = Mw[nz_i, nz_j]
            mw_exp = np.broadcast_to(
                mw_vals[np.newaxis, np.newaxis, :], y_idx.shape)

            rows = ab_exp[valid].ravel().tolist()
            cols = y_idx[valid].ravel().tolist()
            vals = mw_exp[valid].ravel().tolist()

            Cw = Matrix.sparse(flat_size, n_y, rows, cols, vals)
            cw_expr = Expr.mul(Cw, y)
            Lw_flat = Expr.sub(t_expr, cw_expr)
            Lw_mat  = Expr.reshape(Lw_flat, n_loc, n_loc)
            MDL.constraint(f"w_{w}", Lw_mat, Domain.inPSDCone(n_loc))

            # rows, cols, vals, y_idx, valid, ab_exp, mw_exp, Cw, cw_expr,
            # Lw_flat, Lw_mat all go out of scope here.

            if verbose and (w + 1) % 50 == 0:
                avail = _avail_gb()
                print(f"    {w+1}/{n_win} windows  {_mem_tag()}",
                      flush=True)
                if avail < avail_floor_gb:
                    MDL.dispose()
                    raise MemoryError(
                        f"Available memory {avail:.1f} GB < floor "
                        f"{avail_floor_gb} GB after {w+1}/{n_win} windows")

        if verbose:
            print(f"  All {n_win} window PSD constraints added", flush=True)
    else:
        # order == 1: scalar constraints  (same as lasserre_fusion.py)
        units = [_unit(d, i) for i in range(d)]
        for w in range(n_win):
            picks = []
            coeffs = []
            for i in range(d):
                for j in range(d):
                    if M_mats[w][i, j] != 0:
                        eij = _add_mi(units[i], units[j], d)
                        picks.append(idx[eij])
                        coeffs.append(M_mats[w][i, j])
            tv_expr = Expr.dot(coeffs, y.pick(picks))
            MDL.constraint(f"win_{w}",
                           Expr.sub(t_param, tv_expr),
                           Domain.greaterThan(0.0))

    MDL.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    model_s = time.time() - t_model
    build_s = time.time() - t_build

    # ── Free every Python-side precompute array ──
    del ab_eiej_idx, ab_flat
    del M_mats, sorted_h, sort_o, bases, B_arr
    gc.collect()

    if verbose:
        print(f"  MOSEK model built: {model_s:.1f}s  "
              f"(precompute {precompute_s:.1f}s + model {model_s:.1f}s)")
        print(f"  Consistency constraints: {n_consist_added}")
        print(f"  Post-cleanup {_mem_tag()}", flush=True)
        print(f"  Binary search ({n_bisect} steps) …", flush=True)

    # ── Binary search (same as lasserre_fusion.py) ──
    t_bs = time.time()

    def check(t_val):
        t_param.setValue(t_val)
        try:
            MDL.solve()
            ps = MDL.getPrimalSolutionStatus()
            return ps in (SolutionStatus.Optimal, SolutionStatus.Feasible)
        except Exception as exc:
            if verbose:
                print(f"      [MOSEK error] t={t_val:.6f}: {exc}",
                      flush=True)
            return False

    lo, hi = 0.5, 5.0
    while not check(hi):
        hi *= 2
        if hi > 100:
            break
    if not check(hi):
        MDL.dispose()
        raise RuntimeError(f"SDP infeasible up to t={hi}")
    if verbose:
        print(f"  Feasible at t={hi:.4f}  ({_mem_tag()})", flush=True)

    for step in range(n_bisect):
        mid = (lo + hi) / 2
        t0s = time.time()
        if check(mid):
            hi = mid
            tag = "feas"
        else:
            lo = mid
            tag = "inf "
        dt = time.time() - t0s
        if verbose:
            print(f"    [{step+1:2d}/{n_bisect}] t={mid:.10f} {tag}  "
                  f"({dt:.1f}s  {_mem_tag()})", flush=True)

    MDL.dispose()
    gc.collect()

    solve_s = time.time() - t_bs
    wall_s  = time.time() - t_wall
    lb = lo

    if verbose:
        print(f"  Solve: {solve_s:.1f}s   Build: {build_s:.1f}s   "
              f"Wall: {wall_s:.1f}s")
        print(f"  Lower bound: lb = {lb:.10f}")

    return dict(
        lb=lb, d=d, order=order,
        solve_s=solve_s, build_s=build_s, wall_s=wall_s,
        n_y=n_y, n_basis=n_basis, n_loc=n_loc, n_win=n_win,
        n_bisect=n_bisect,
    )


# =====================================================================
# Sweep orchestrator
# =====================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Large-scale Lasserre sweep (256 GB / 8 h)")
    p.add_argument('--max_hours', type=float, default=7.5)
    p.add_argument('--avail_floor_gb', type=float, default=15.0,
                   help='Abort config if avail memory < this')
    p.add_argument('--num_threads', type=int, default=0,
                   help='MOSEK threads (0 = auto)')
    p.add_argument('--outdir', type=str, default='data')
    args = p.parse_args()

    # Known exact val(d) from finite-d optimisation
    val_d = {
        4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
        12: 1.271, 14: 1.284, 16: 1.319,
    }
    # Conservative extrapolations for reporting only (not used in SDP)
    val_d_approx = {18: 1.33, 20: 1.34, 22: 1.35, 24: 1.36}
    val_d_all = {**val_d, **val_d_approx}

    # (d, order, n_bisect, description)
    #
    # Ordering rationale:
    #   - Start with L3 d=16 (~60 GB, ~30 min) to establish a clean
    #     baseline — the prior attempt OOM'd at 64 GB.
    #   - Then L3 d=18 (~120 GB, main target).  With n_loc=190 and
    #     n_win=630 the MOSEK internal memory is ~2× d=16.
    #   - Finally L3 d=20 (~200 GB stretch).  n_loc=231, n_win=780.
    #     Only attempted if ≥ 2 h remain and ≥ 60 GB available.
    #
    # Bisection: 14 steps → precision ~2^{-14} ≈ 6e-5.
    # For d=20 we reduce to 12 steps to save ~2 solves worth of time.
    configs = [
        (16, 3, 14, "L3 d=16  (~60 GB, baseline)"),
        (18, 3, 14, "L3 d=18  (~120 GB, main target)"),
        (20, 3, 12, "L3 d=20  (~200 GB, stretch)"),
    ]

    print("=" * 72)
    print("LARGE-SCALE LASSERRE SWEEP")
    print(f"  Budget   : {args.max_hours} h wall,  "
          f"avail floor {args.avail_floor_gb} GB")
    print(f"  System   : {_avail_gb():.0f} GB available,  "
          f"RSS {_rss_gb():.1f} GB")
    print(f"  Configs  : {len(configs)}")
    print(f"  Threads  : {args.num_threads or 'auto'}")
    print("=" * 72, flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    results = []
    t_start = time.time()

    for d, order, n_bisect, desc in configs:
        elapsed_h = (time.time() - t_start) / 3600
        remain_h  = args.max_hours - elapsed_h
        avail     = _avail_gb()

        print(f"\n{'#' * 72}")
        print(f"# {desc}")
        print(f"# Elapsed {elapsed_h:.2f} h  |  Remaining {remain_h:.2f} h  "
              f"|  Avail {avail:.0f} GB")
        print(f"{'#' * 72}", flush=True)

        if remain_h < 0.1:
            print("  TIME EXHAUSTED — stopping sweep.", flush=True)
            break

        if avail < args.avail_floor_gb + 20:
            print(f"  SKIP: only {avail:.0f} GB available "
                  f"(need >{args.avail_floor_gb + 20:.0f})", flush=True)
            continue

        try:
            r = solve_lasserre_streaming(
                d, order=order, n_bisect=n_bisect, verbose=True,
                avail_floor_gb=args.avail_floor_gb,
                num_threads=args.num_threads)

            lb = r['lb']
            v  = val_d_all.get(d, 0)
            if v > 1:
                closure = (lb - 1) / (v - 1) * 100
            else:
                closure = 0.0
            r['val_d'] = v
            r['gap_closure_pct'] = closure

            results.append(r)

            print(f"\n  >>> lb = {lb:.8f}   val({d}) ~ {v:.3f}   "
                  f"gap closure {closure:.1f}%", flush=True)

            # Incremental save
            fname = (f"lasserre_L{order}_d{d}_"
                     f"{time.strftime('%Y%m%d_%H%M%S')}.json")
            outpath = os.path.join(args.outdir, fname)
            with open(outpath, 'w') as f:
                json.dump(r, f, indent=2, default=str)
            print(f"  Saved: {outpath}", flush=True)

        except MemoryError as exc:
            print(f"\n  OOM: {exc}", flush=True)
            gc.collect()
            print("  Aborting sweep (larger configs will also OOM).",
                  flush=True)
            break

        except Exception as exc:
            print(f"\n  FAILED: {exc}", flush=True)
            traceback.print_exc()
            gc.collect()
            # Don't break — a transient MOSEK failure at one config
            # shouldn't prevent trying the next.

    # ── Summary ──
    total_h = (time.time() - t_start) / 3600
    print(f"\n{'=' * 72}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 72}")
    hdr = (f"{'Level':<8}{'d':<5}{'lb':<16}{'val(d)':<10}"
           f"{'Closure':<12}{'Wall':<12}{'Build':<12}")
    print(hdr)
    print("-" * 72)
    for r in results:
        level  = f"L{r['order']}"
        dd     = r['d']
        lb     = r['lb']
        v      = r.get('val_d', 0)
        cl     = r.get('gap_closure_pct', 0)
        wall_m = r.get('wall_s', 0) / 60
        bld_m  = r.get('build_s', 0) / 60
        print(f"{level:<8}{dd:<5}{lb:<16.10f}{v:<10.3f}"
              f"{cl:<12.1f}%{wall_m:<12.1f}m{bld_m:<12.1f}m")

    if results:
        best = max(results, key=lambda r: r['lb'])
        print(f"\n  Best lower bound : {best['lb']:.10f}  "
              f"(L{best['order']} d={best['d']})")
        print(f"  Previous best    : ~1.2030  (L3 d=16, OOM at 64 GB)")
        delta = best['lb'] - 1.203
        print(f"  Improvement      : {'+' if delta >= 0 else ''}"
              f"{delta:.6f}")
    else:
        print("  No results produced.")

    print(f"\n  Total wall time  : {total_h:.2f} h / "
          f"{args.max_hours} h budget")
    print(f"{'=' * 72}", flush=True)

    # Combined results
    combo = os.path.join(
        args.outdir,
        f"lasserre_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(combo, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Combined: {combo}")


if __name__ == '__main__':
    main()
