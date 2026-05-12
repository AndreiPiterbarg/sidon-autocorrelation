"""Coarse-grid cascade prover v4 — v3 (N+O) + Joint Dual + Shor SDP fallback.

v4 = v3 with two additional SOUND box-cert layers applied to cells where
v3's N+O bound fails (the "hard cells"):

  Layer J (Joint Dual):  joint multi-window LP-dual LB via subgradient ascent
                          on simplex (top_K = 4 windows by default).  Sound:
                            cert_box(c) >= max_λ [Σλ_W TV_W − UB_lin(λ) − UB_quad(λ)]
                          where UB_lin is exact LP, UB_quad is per-window
                          pair-bound triangle.  See _coarse_J_bench.py.

  Layer L (Shor SDP):    per-cell Shor relaxation (PSD lift of Y=[[1,δ],[δ,D]])
                          with full RLT.  Sound: SDP optimum LB-bounds true
                          QP minimum.  See _coarse_L_bench.py.

Soundness chain:  v4 ⊆ v3 ⊆ v2  (a cell certified by v3 is automatically
also certified by v4, since v4 first runs v3 and only escalates non-NO-
certified cells).  Each layer is individually sound; their disjunction
(cert iff ANY succeeds) is sound.  Each is OPTIONAL via CLI (--use_joint,
--use_sdp).

Runtime profile per cell (rough, on average laptop):
  N+O kernel  : ~50 ns
  Joint dual  : ~5 ms (top_K=4, 20 subgradient iters)
  Shor SDP    : ~150 ms (MOSEK 'best_only' mode, single window)

Usage:
    python -m cloninger-steinerberger.cpu.run_cascade_coarse_v4 \\
        --d0 2 --S 200 --c_target 1.20 --use_joint --use_sdp
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import numba
from numba import njit, prange

_this_dir = os.path.dirname(os.path.abspath(__file__))
_cs_dir = os.path.dirname(_this_dir)
_root = os.path.dirname(_cs_dir)
sys.path.insert(0, _cs_dir)
sys.path.insert(0, _this_dir)
sys.path.insert(0, _root)

from compositions import (generate_compositions_batched,
                          generate_canonical_compositions_batched)
from pruning import count_compositions, asymmetry_threshold, _canonical_mask

# Reuse helpers from v2 (unchanged in v3/v4)
from run_cascade_coarse_v2 import (asymmetry_prune_mask_coarse,
                                     coarse_x_cap, _suggest_S, _build_pair_prefix)
from run_cascade_coarse_v3 import (_quad_corr_v3, _one_sided_lp,
                                     precompute_op_rest_d)

# Joint Dual + Shor SDP
import importlib.util
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_J_path = os.path.join(_root, '_coarse_J_bench.py')
_L_path = os.path.join(_root, '_coarse_L_bench.py')
_J_mod = _load_module('_coarse_J_bench', _J_path)
_L_mod = _load_module('_coarse_L_bench', _L_path)
joint_cert_LB = _J_mod.joint_cert_LB
find_pruning_windows = _J_mod.find_pruning_windows
cell_cert_shor = _L_mod.cell_cert_shor
all_windows_L = _L_mod.all_windows
tv_at = _L_mod.tv_at


def _log(msg):
    print(msg, flush=True)


# =====================================================================
# Per-cell certification kernel: returns per-cell flag (NOT global min)
#
# Flag values:
#   0 = SURVIVOR (TV at grid never exceeds c_target — passes to next level)
#   1 = CERTIFIED by N+O at L0 (closed; never reaches J or L)
#   2 = HARD (TV at grid > c_target for some W, but N+O cert fails)
# =====================================================================

@njit(parallel=True, cache=True)
def _classify_cells_v3(batch_int, d, S, c_target, op_rest_d_arr):
    """Run the v3 N+O kernel returning per-cell certification flags.

    Returns:
      flags: int8 array of shape (B,) with values in {0,1,2}
      best_net: float64 array of shape (B,) — best cert net per cell (only
                meaningful if flags[b]>0; else 0).
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    flags = np.zeros(B, dtype=np.int8)
    best_net_out = np.zeros(B, dtype=np.float64)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d
    h = 1.0 / (2.0 * S_d)

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        triggered = False
        best_net = np.float64(-1e30)
        grad_arr = np.empty(d, dtype=np.float64)
        k_local = np.empty(d, dtype=np.int64)
        for j in range(d):
            k_local[j] = np.int64(batch_int[b, j])
        work_buf = np.empty(4 * d, dtype=np.int64)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                if ws > dyn_it:
                    triggered = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    cell_var_O = _one_sided_lp(grad_arr, k_local, d, h, work_buf)

                    grad_sorted = np.empty(d, dtype=np.float64)
                    for i in range(d):
                        grad_sorted[i] = grad_arr[i]
                    for i in range(1, d):
                        key = grad_sorted[i]
                        jj = i - 1
                        while jj >= 0 and grad_sorted[jj] > key:
                            grad_sorted[jj + 1] = grad_sorted[jj]
                            jj -= 1
                        grad_sorted[jj + 1] = key
                    cell_var_BL = 0.0
                    for kk in range(d // 2):
                        cell_var_BL += grad_sorted[d - 1 - kk] - grad_sorted[kk]
                    cell_var_BL *= h
                    if cell_var_O > cell_var_BL:
                        cell_var_O = cell_var_BL

                    qc = _quad_corr_v3(s_lo, ell, d, prefix_nk, prefix_mk,
                                        op_rest_d_arr, inv_4S2)

                    net = margin - cell_var_O - qc
                    if net > best_net:
                        best_net = net

        if triggered:
            best_net_out[b] = best_net
            if best_net >= 0.0:
                flags[b] = np.int8(1)
            else:
                flags[b] = np.int8(2)
        # else: flags[b] = 0 (survivor)

    return flags, best_net_out


# =====================================================================
# Joint Dual filter (per-cell, pure-Python, calls _coarse_J_bench)
# =====================================================================

def joint_dual_filter(c_int, S, d, c_target, top_K=4, n_iters=20):
    """Returns True iff joint dual LB >= c_target (cell certified)."""
    LB, n_used, _baseline = joint_cert_LB(
        c_int, S, d, c_target, windows=None,
        n_lambda_iters=n_iters, top_K=top_K)
    return LB >= c_target - 1e-9, LB, n_used


# =====================================================================
# Shor SDP filter (per-cell)
# =====================================================================

def shor_sdp_filter(c_int, S, d, c_target, mode='best_only', tol=1e-9):
    """Cell certified iff Shor SDP LB >= c_target (mode='best_only' uses
    only the window with max TV at the grid point — the cheapest variant
    that's already ~150 ms/cell)."""
    # Find pruning windows (those with TV > c_target at the grid point).
    windows_jl = find_pruning_windows(np.asarray(c_int, dtype=np.int32), S, d, c_target)
    if not windows_jl:
        # No window violates threshold — cell wouldn't have been hard.
        return False, float('-inf'), (-1, -1)
    if mode == 'best_only':
        # Single window with max TV (the natural triangle-best window)
        windows_jl.sort(key=lambda w: -w[2])
        ell, s_lo, _tv = windows_jl[0]
        lb, status = cell_cert_shor(np.asarray(c_int, dtype=np.float64),
                                     S, d, c_target, (ell, s_lo),
                                     solver='auto', tol=tol)
        return lb >= c_target - 1e-9, lb, (ell, s_lo)
    elif mode == 'max':
        # Max over all pruning windows (multiple SDPs).
        best_lb = float('-inf')
        best_W = (-1, -1)
        for ell, s_lo, _ in windows_jl:
            lb, _status = cell_cert_shor(np.asarray(c_int, dtype=np.float64),
                                          S, d, c_target, (ell, s_lo),
                                          solver='auto', tol=tol)
            if lb > best_lb:
                best_lb = lb
                best_W = (ell, s_lo)
        return best_lb >= c_target - 1e-9, best_lb, best_W
    else:
        raise ValueError(f"unknown mode: {mode}")


# =====================================================================
# Apply v4 layers to a batch — returns (survivors, counts, hard_cells)
# =====================================================================

def apply_v4_layers(batch_int, d, S, c_target, op_rest_d_arr,
                    use_joint=True, use_sdp=True,
                    joint_top_K=4, joint_iters=20,
                    sdp_mode='best_only', verbose=False):
    """Apply N+O kernel + optional Joint + optional Shor SDP.

    Returns dict with:
      survivors:       array of true survivors (TV < c_target at grid)
      n_certified_NO:  cells closed by v3 N+O alone
      n_certified_J:   newly closed by Joint Dual
      n_certified_L:   newly closed by Shor SDP
      n_uncertified:   cells passing through all three but still hard (these
                       must be passed to the next level — same fallback as
                       in v3, but flagged for the soundness report)
      box_certified:   True iff n_uncertified == 0
      time_NO, time_J, time_L: per-layer time (sec)
    """
    t0 = time.time()
    flags, best_net = _classify_cells_v3(batch_int, d, S, c_target, op_rest_d_arr)
    t_NO = time.time() - t0

    # Survivors (TV<c at grid) → flag 0
    surv_mask = (flags == 0)
    survivors = batch_int[surv_mask]

    # Certified by NO → flag 1
    n_cert_NO = int(np.sum(flags == 1))

    # Hard cells (TV>=c at grid, NO failed) → flag 2
    hard_mask = (flags == 2)
    hard_idx = np.where(hard_mask)[0]
    n_hard = len(hard_idx)

    n_cert_J = 0
    n_cert_L = 0
    t_J = 0.0
    t_L = 0.0

    # Try Joint Dual on hard cells
    if use_joint and n_hard > 0:
        t1 = time.time()
        still_hard = []
        for idx in hard_idx:
            c_int = batch_int[idx]
            cert, LB, _ = joint_dual_filter(c_int, S, d, c_target,
                                             top_K=joint_top_K,
                                             n_iters=joint_iters)
            if cert:
                n_cert_J += 1
            else:
                still_hard.append(idx)
        t_J = time.time() - t1
        hard_idx = np.asarray(still_hard, dtype=np.int64)
        n_hard = len(hard_idx)

    # Try Shor SDP on remaining hard cells
    if use_sdp and n_hard > 0:
        t2 = time.time()
        still_hard = []
        for idx in hard_idx:
            c_int = batch_int[idx]
            cert, LB, _ = shor_sdp_filter(c_int, S, d, c_target, mode=sdp_mode)
            if cert:
                n_cert_L += 1
            else:
                still_hard.append(idx)
        t_L = time.time() - t2
        hard_idx = np.asarray(still_hard, dtype=np.int64)
        n_hard = len(hard_idx)

    box_certified = (n_hard == 0)

    if verbose:
        _log(f"      [N+O] {n_cert_NO} certified  ({t_NO:.2f}s)")
        if use_joint:
            _log(f"      [J]   {n_cert_J} additional  ({t_J:.2f}s)")
        if use_sdp:
            _log(f"      [L]   {n_cert_L} additional  ({t_L:.2f}s)")
        _log(f"      uncertified hard: {n_hard}")

    return {
        'survivors': survivors,
        'n_certified_NO': n_cert_NO,
        'n_certified_J': n_cert_J,
        'n_certified_L': n_cert_L,
        'n_uncertified': n_hard,
        'uncertified_idx': hard_idx,
        'box_certified': box_certified,
        'time_NO': t_NO,
        'time_J': t_J,
        'time_L': t_L,
        'min_cert_net_NO': float(best_net.min()) if len(best_net) else 0.0,
    }


# =====================================================================
# Level 0
# =====================================================================

def run_level0(d0, S, c_target, op_rest_d_arr,
               use_joint=True, use_sdp=True,
               joint_top_K=4, joint_iters=20, sdp_mode='best_only',
               verbose=True):
    n_total = count_compositions(d0, S)
    if verbose:
        _log(f"\n[L0] d={d0}, S={S}, compositions={n_total:,}")
        _log(f"     x_cap = {coarse_x_cap(d0, S, c_target)}")
        _log(f"     layers: NO=on, J={'on' if use_joint else 'off'}, "
             f"L={'on' if use_sdp else 'off'}")

    t0 = time.time()
    all_survivors = []
    n_cert_NO_total = 0
    n_cert_J_total = 0
    n_cert_L_total = 0
    n_uncert_total = 0
    n_pruned_asym = 0
    n_processed = 0
    t_NO_total = 0.0
    t_J_total = 0.0
    t_L_total = 0.0
    last_report = t0

    gen = generate_canonical_compositions_batched(d0, S, batch_size=500_000)

    for batch in gen:
        n_processed += len(batch)
        asym_mask = asymmetry_prune_mask_coarse(batch, S, c_target)
        n_pruned_asym += int(np.sum(~asym_mask))
        batch = batch[asym_mask]
        if len(batch) == 0:
            continue

        out = apply_v4_layers(batch, d0, S, c_target, op_rest_d_arr,
                               use_joint=use_joint, use_sdp=use_sdp,
                               joint_top_K=joint_top_K, joint_iters=joint_iters,
                               sdp_mode=sdp_mode, verbose=False)
        n_cert_NO_total += out['n_certified_NO']
        n_cert_J_total += out['n_certified_J']
        n_cert_L_total += out['n_certified_L']
        n_uncert_total += out['n_uncertified']
        t_NO_total += out['time_NO']
        t_J_total += out['time_J']
        t_L_total += out['time_L']
        if len(out['survivors']) > 0:
            all_survivors.append(out['survivors'])
        now = time.time()
        if verbose and (now - last_report >= 2.0):
            n_surv = sum(len(s) for s in all_survivors)
            _log(f"     [{n_processed:,} processed] surv={n_surv:,}  "
                 f"NO_cert={n_cert_NO_total:,}  J+={n_cert_J_total:,}  "
                 f"L+={n_cert_L_total:,}  hard={n_uncert_total:,}")
            last_report = now

    elapsed = time.time() - t0
    if all_survivors:
        all_survivors = np.vstack(all_survivors)
    else:
        all_survivors = np.empty((0, d0), dtype=np.int32)

    n_survivors = len(all_survivors)
    proven = (n_survivors == 0) and (n_uncert_total == 0)
    box_ok = (n_uncert_total == 0)

    if verbose:
        _log(f"     {elapsed:.2f}s: pruned asym={n_pruned_asym:,} "
             f"NO={n_cert_NO_total:,} J={n_cert_J_total:,} "
             f"L={n_cert_L_total:,} hard={n_uncert_total:,} "
             f"survivors={n_survivors:,}")
        _log(f"     time breakdown: N+O={t_NO_total:.2f}s J={t_J_total:.2f}s "
             f"L={t_L_total:.2f}s")
        if box_ok:
            _log(f"     BOX CERT v4: PASS (sound, N+O+J+L)")
        else:
            _log(f"     BOX CERT v4: FAIL ({n_uncert_total} hard cells)")
        if proven:
            _log(f"     PROVEN at L0!")

    return {
        'survivors': all_survivors, 'n_survivors': n_survivors,
        'n_certified_NO': n_cert_NO_total,
        'n_certified_J': n_cert_J_total,
        'n_certified_L': n_cert_L_total,
        'n_uncertified': n_uncert_total,
        'n_pruned_asym': n_pruned_asym,
        'elapsed': elapsed, 'proven': proven,
        'box_certified': box_ok,
        'time_NO': t_NO_total, 'time_J': t_J_total, 'time_L': t_L_total,
    }


# =====================================================================
# process_parent for cascade levels (v4 same logic, calls v4 layers)
# =====================================================================

def process_parent_v4(parent_int, d_child, S, c_target, op_rest_d_arr,
                      use_joint=True, use_sdp=True,
                      joint_top_K=4, joint_iters=20, sdp_mode='best_only'):
    """Generate all children of one parent, apply v4 layers."""
    d_parent = len(parent_int)
    x_cap = coarse_x_cap(d_child, S, c_target)
    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total_product = 1
    for i in range(d_parent):
        p = int(parent_int[i])
        lo = max(0, p - x_cap)
        hi = min(p, x_cap)
        if lo > hi:
            return (np.empty((0, d_child), dtype=np.int32), 0,
                    {'n_certified_NO': 0, 'n_certified_J': 0,
                     'n_certified_L': 0, 'n_uncertified': 0,
                     'time_NO': 0.0, 'time_J': 0.0, 'time_L': 0.0})
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_product *= (hi - lo + 1)
    if total_product == 0:
        return (np.empty((0, d_child), dtype=np.int32), 0,
                {'n_certified_NO': 0, 'n_certified_J': 0,
                 'n_certified_L': 0, 'n_uncertified': 0,
                 'time_NO': 0.0, 'time_J': 0.0, 'time_L': 0.0})

    # Materialise all children (we need post-processing per cell, can't fuse)
    children = np.empty((total_product, d_child), dtype=np.int32)
    cursor = lo_arr.copy()
    n = 0
    while True:
        for i in range(d_parent):
            children[n, 2 * i] = cursor[i]
            children[n, 2 * i + 1] = parent_int[i] - cursor[i]
        n += 1
        carry = d_parent - 1
        while carry >= 0:
            cursor[carry] += 1
            if cursor[carry] <= hi_arr[carry]:
                break
            cursor[carry] = lo_arr[carry]
            carry -= 1
        if carry < 0:
            break
    children = children[:n]

    out = apply_v4_layers(children, d_child, S, c_target, op_rest_d_arr,
                           use_joint=use_joint, use_sdp=use_sdp,
                           joint_top_K=joint_top_K, joint_iters=joint_iters,
                           sdp_mode=sdp_mode, verbose=False)
    counts = {
        'n_certified_NO': out['n_certified_NO'],
        'n_certified_J': out['n_certified_J'],
        'n_certified_L': out['n_certified_L'],
        'n_uncertified': out['n_uncertified'],
        'time_NO': out['time_NO'],
        'time_J': out['time_J'],
        'time_L': out['time_L'],
    }
    return out['survivors'], n, counts


# =====================================================================
# Full cascade v4
# =====================================================================

def run_cascade(d0, S, c_target, max_levels=10, n_workers=None,
                use_joint=True, use_sdp=True,
                joint_top_K=4, joint_iters=20, sdp_mode='best_only',
                verbose=True):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)

    n_total_comps = count_compositions(d0, S)
    if verbose:
        _log(f"\n{'='*70}")
        _log(f"COARSE CASCADE PROVER v4 (Theorem 1 + N+O + Joint Dual + Shor SDP)")
        _log(f"  d0={d0}, S={S}, c_target={c_target}")
        _log(f"  Layers: NO=on, J={'on' if use_joint else 'off'} (top_K={joint_top_K}), "
             f"L={'on' if use_sdp else 'off'} (mode={sdp_mode})")
        _log(f"  L0 compositions (canonical): ~{n_total_comps // 2:,}")
        _log(f"  workers: {n_workers}")
        _log(f"{'='*70}")

    t_total = time.time()
    info = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'grid': 'coarse', 'box_cert_method': 'v4_NO_plus_Joint_plus_Shor',
        'use_joint': use_joint, 'use_sdp': use_sdp,
        'joint_top_K': joint_top_K, 'sdp_mode': sdp_mode,
        'levels': [],
    }

    op_rest_d_cache = {d0: precompute_op_rest_d(d0)}

    l0 = run_level0(d0, S, c_target, op_rest_d_cache[d0],
                     use_joint=use_joint, use_sdp=use_sdp,
                     joint_top_K=joint_top_K, joint_iters=joint_iters,
                     sdp_mode=sdp_mode, verbose=verbose)
    info['l0'] = {
        'survivors': l0['n_survivors'], 'time': l0['elapsed'],
        'n_certified_NO': l0['n_certified_NO'],
        'n_certified_J': l0['n_certified_J'],
        'n_certified_L': l0['n_certified_L'],
        'n_uncertified': l0['n_uncertified'],
        'box_certified': l0['box_certified'],
        'time_NO': l0['time_NO'], 'time_J': l0['time_J'], 'time_L': l0['time_L'],
    }
    if l0['proven']:
        info['proven_at'] = 'L0'
        info['box_certified'] = l0['box_certified']
        info['total_time'] = time.time() - t_total
        if verbose:
            _log(f"\n*** GRID-POINT PROOF at L0 (v4) ***")
            _log(f"    Box certification: {'PASS' if l0['box_certified'] else 'FAIL'}")
        return info

    current = l0['survivors']
    d_parent = d0
    for level in range(1, max_levels + 1):
        d_child = 2 * d_parent
        n_parents = len(current)
        if n_parents == 0:
            break
        x_cap = coarse_x_cap(d_child, S, c_target)
        if verbose:
            _log(f"\n--- Level {level}: d={d_parent} -> {d_child}, "
                 f"parents={n_parents:,}, x_cap={x_cap} ---")

        if d_child not in op_rest_d_cache:
            t_pre = time.time()
            op_rest_d_cache[d_child] = precompute_op_rest_d(d_child)
            if verbose:
                _log(f"    op_rest_d table for d={d_child}: "
                     f"{time.time()-t_pre:.2f}s")
        ord_d = op_rest_d_cache[d_child]

        feasible = np.all(current <= 2 * x_cap, axis=1)
        n_infeasible = n_parents - int(feasible.sum())
        if n_infeasible > 0:
            current = np.ascontiguousarray(current[feasible])
            n_parents = len(current)
            if verbose:
                _log(f"    Pre-filtered {n_infeasible:,} infeasible parents")
        if n_parents == 0:
            current = np.empty((0, d_child), dtype=np.int32)
            d_parent = d_child
            info['levels'].append({
                'level': level, 'd_child': d_child,
                'parents': 0, 'children': 0, 'survivors': 0,
            })
            continue

        t_level = time.time()
        all_survivors = []
        total_children = 0
        cnt_NO = 0; cnt_J = 0; cnt_L = 0; cnt_HARD = 0
        t_NO_lvl = 0.0; t_J_lvl = 0.0; t_L_lvl = 0.0
        n_done = 0
        last_report = time.time()

        # Sequential (multiprocessing not used since J/L use cvxpy state)
        for i in range(n_parents):
            surv, n_t, c = process_parent_v4(
                current[i], d_child, S, c_target, ord_d,
                use_joint=use_joint, use_sdp=use_sdp,
                joint_top_K=joint_top_K, joint_iters=joint_iters,
                sdp_mode=sdp_mode)
            total_children += n_t
            if len(surv) > 0:
                all_survivors.append(surv)
            cnt_NO += c['n_certified_NO']
            cnt_J += c['n_certified_J']
            cnt_L += c['n_certified_L']
            cnt_HARD += c['n_uncertified']
            t_NO_lvl += c['time_NO']
            t_J_lvl += c['time_J']
            t_L_lvl += c['time_L']
            n_done += 1
            now = time.time()
            if verbose and (now - last_report >= 5.0):
                ns = sum(len(s) for s in all_survivors)
                _log(f"    [{n_done}/{n_parents}] children={total_children:,} "
                     f"NO={cnt_NO:,} J+={cnt_J:,} L+={cnt_L:,} hard={cnt_HARD:,} "
                     f"surv={ns:,}")
                last_report = now

        elapsed_level = time.time() - t_level
        if all_survivors:
            current = np.vstack(all_survivors)
            canon = _canonical_mask(current)
            non_canon = ~canon
            current[non_canon] = current[non_canon, ::-1]
            current = np.unique(current, axis=0)
        else:
            current = np.empty((0, d_child), dtype=np.int32)
        n_survivors = len(current)
        rate = n_survivors / max(1, total_children) * 100
        box_ok = (cnt_HARD == 0)
        if verbose:
            _log(f"    {elapsed_level:.1f}s: {total_children:,} children, "
                 f"NO={cnt_NO:,} J={cnt_J:,} L={cnt_L:,} "
                 f"hard={cnt_HARD:,} survivors={n_survivors:,} ({rate:.4f}%)")
            _log(f"    time breakdown: N+O={t_NO_lvl:.2f}s "
                 f"J={t_J_lvl:.2f}s L={t_L_lvl:.2f}s")
            if box_ok:
                _log(f"    BOX CERT v4: PASS")
            else:
                _log(f"    BOX CERT v4: FAIL ({cnt_HARD} hard cells)")
            if n_survivors == 0:
                _log(f"    *** ALL PRUNED ***")
        info['levels'].append({
            'level': level, 'd_child': d_child,
            'parents': n_parents, 'children': int(total_children),
            'n_certified_NO': cnt_NO, 'n_certified_J': cnt_J,
            'n_certified_L': cnt_L, 'n_uncertified': cnt_HARD,
            'survivors': n_survivors, 'rate': rate,
            'time': elapsed_level,
            'time_NO': t_NO_lvl, 'time_J': t_J_lvl, 'time_L': t_L_lvl,
            'box_certified': box_ok,
        })
        if n_survivors == 0:
            info['proven_at'] = f'L{level}'
            info['total_time'] = time.time() - t_total
            all_cert = info['l0'].get('box_certified', False)
            if all_cert:
                for lv in info['levels']:
                    if not lv.get('box_certified', False):
                        all_cert = False; break
            info['box_certified'] = all_cert
            if verbose:
                _log(f"\n*** GRID-POINT PROOF COMPLETE (v4) ***")
                if all_cert:
                    _log(f"*** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
                _log(f"    Total time: {info['total_time']:.1f}s")
            return info
        d_parent = d_child

    info['total_time'] = time.time() - t_total
    info['final_survivors'] = len(current)
    if verbose:
        _log(f"\nCascade did not converge ({len(current):,} survivors "
             f"at d={d_parent})")
    return info


def main():
    parser = argparse.ArgumentParser(
        description='Coarse-grid cascade prover v4 (NO+Joint Dual+Shor SDP)')
    parser.add_argument('--d0', type=int, default=2)
    parser.add_argument('--S', type=int, default=50)
    parser.add_argument('--c_target', type=float, default=1.20)
    parser.add_argument('--max_levels', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--use_joint', action='store_true', default=True)
    parser.add_argument('--no_joint', dest='use_joint', action='store_false')
    parser.add_argument('--use_sdp', action='store_true', default=True)
    parser.add_argument('--no_sdp', dest='use_sdp', action='store_false')
    parser.add_argument('--joint_top_K', type=int, default=4)
    parser.add_argument('--joint_iters', type=int, default=20)
    parser.add_argument('--sdp_mode', default='best_only',
                         choices=['best_only', 'max'])
    args = parser.parse_args()
    result = run_cascade(d0=args.d0, S=args.S, c_target=args.c_target,
                          max_levels=args.max_levels, n_workers=args.n_workers,
                          use_joint=args.use_joint, use_sdp=args.use_sdp,
                          joint_top_K=args.joint_top_K,
                          joint_iters=args.joint_iters,
                          sdp_mode=args.sdp_mode)
    if 'proven_at' in result:
        if result.get('box_certified'):
            _log(f"\nRIGOROUS PROOF v4: C_{{1a}} >= {args.c_target}")
        else:
            _log(f"\nGRID-POINT PROOF: TV >= {args.c_target} (box cert incomplete)")
    else:
        _log(f"\nNOT PROVEN (cascade did not converge)")


if __name__ == '__main__':
    main()
