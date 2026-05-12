"""Cascade compute estimator — SOUND (full canonical enum, not palindromic).

For each (n_half, m, c_target) config:
   L0   : full canonical enumeration via run_level0() with F (kernel) +
          FN + Q + L post-filters.  All compositions, no palindromic
          shortcut.  Sound for general C_{1a}.
   L1+  : sample SAMPLE_N random parents from the previous-level survivors,
          expand each via process_parent_fused (Gray-code over canonical
          children) with F kernel.  Then run FN → Q → L post-filters in
          order, counting avg_children & avg_survivors at each step.
          Estimate total children processed and total survivors via the
          expansion factor.
   Stop : (a) avg_survivors == 0 across the sample → CLOSURE conjectured at
              this level (sample-based; statistical, not certified).
          (b) cumulative est_total_children > COMPUTE_BUDGET → IMPOSSIBLE
              (cascade would not close in plausible wall time).
          (c) wall time per level > LEVEL_TIME_BUDGET → cap and report.

Why sample-based estimation:
   Full L1+ enumeration at d=8 with thousands of L0 survivors is expensive
   (10^7-10^9 children/parent at typical configs), often infeasible.  Sample
   gives unbiased estimate of expansion factor; multiply by total parent count
   to get the certifying cost.  A separate full-cascade run is needed for an
   actual proof — this script is for **planning**.

Pod usage:  process_parent_fused uses Numba parallel internally → saturates
   all 64 cores per call.  We run configs sequentially.

Soundness of F → FN → Q → QN → L chain (each (weak) subset of the prior):
   F  : LP-tight per-window Δ_BB linear bound + #pairs · h² δ² bound.
        Sound under {|δ|_∞ ≤ 1/m, Σδ = 0, a ≥ 0}.   _M1_bench.py:229–335
   FN : F's tight LP linear + N's restricted-spectrum δ² bound,
        |δᵀA_W δ| ≤ op_rest · d · h² (Σδ=0 ⇒ all-ones component
        annihilated).  Per-window min(op_rest·d, ell_int_sum) — sound
        regression vs F.  _FN_bench.py:30–53
   Q  : multi-window joint LP per composition; LP duality
        inf_a max_W TV_W(a) ≥ sup_λ inf_a Σ λ_W TV_W(a).  Closed-form
        on |δ|_∞ ≤ 1/m, Σδ = 0 polytope vertices.  Already enumerates
        ALL (ell, s_lo) windows.   _Q_bench.py:185–303
   QN : Q's LP with each per-window n_pairs_W replaced by
        m_W := min(op_rest(A_W) · d, n_pairs_W).  Strict tightening:
        QN-survivors ⊆ Q-survivors.  Soundness: each m_W is a sound
        per-window |δᵀA_Wδ| bound.  Empirical: kills 25.8 % of Q-residue
        at d=10 (62 of 240, 0/500 random violations).  _QN_bench.py:80–89
   L  : Lasserre/Shor SDP per-composition cell.  Order-1 (Shor) PSD lift
        M_1 with full RLT cuts; only MOSEK status='infeasible' (Farkas
        certificate) counts as a prune.  _L_bench.py:129–222
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from math import comb

import numpy as np

_DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('CASCADE_ROOT', '/home/ubuntu')
if not os.path.isdir(os.path.join(ROOT, 'cloninger-steinerberger')):
    # Pod path missing; fall back to script directory (local Windows / mac dev).
    ROOT = _DEFAULT_ROOT
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))

from pruning import correction, count_compositions
from run_cascade import process_parent_fused, run_level0

# Post-filters: FN (numba prange), Q (Pool), QN-fast (Pool), L (Pool), SP (Pool).
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))
from post_filters import (apply_FN_filter_parallel,
                          apply_Q_filter_parallel,
                          apply_QN_filter_parallel,
                          apply_L_filter_parallel,
                          apply_split_cell_filter_parallel,
                          SPLIT_CELL_MAX_D)

# Bench's per-composition Q for fall-back single-call sampling.
sys.path.insert(0, ROOT)  # for _Q_bench at repo root
from _Q_bench import (_build_windows as _q_build_windows,
                        _enum_balanced_signs as _q_enum_signs,
                        prune_Q_one)

C_UPPER = 1.5029
Q_SAMPLE_K = 100         # legacy (used for sample-Q upper bound); now superseded
N_WORKERS_DEFAULT = 64   # pod is 64 cores

# Hard caps (override via CLI).
DEFAULT_SAMPLE_N      = 30
DEFAULT_LEVEL_TIME    = 240        # 4 min per L1+ sampling level (geometric series should converge fast; if not, kill early)
DEFAULT_PARENT_TIME   = 30         # 30s per parent at any one level
DEFAULT_COMPUTE_BUDGET= 1e15       # cumulative est children kill threshold
DEFAULT_L0_TIMEOUT    = 1800       # 30 min for L0 full enum

# Convergence kill rules.  These fire when a config is unlikely to converge:
#   - At L >= 1, if estimated total survivors at this level > MAX_SURV_HARD (10^9),
#     kill: any deeper level would face >10^16 children to enumerate.
#   - If average expansion alpha = A_{L+1} / A_L stays > 0.5 across two consecutive
#     levels with non-trivial absolute counts, cascade is unlikely to converge.
MAX_SURV_HARD       = 1e9
ALPHA_NONCONVERGE   = 0.5
SURV_NONTRIVIAL     = 1000


def n_full_compositions(d, S):
    """Total compositions of d non-negative integers summing to S."""
    return comb(S + d - 1, d - 1)


# Cache Q windows/sigmas per d_child to avoid rebuild per parent.
_Q_CACHE = {}

def _get_q_setup(d_child):
    if d_child in _Q_CACHE:
        return _Q_CACHE[d_child]
    ws = _q_build_windows(d_child)
    if isinstance(ws, tuple):
        windows, ell_int_sums = ws[0], ws[1]
    else:
        windows = ws
        ell_int_sums = np.array([w[2] for w in windows], dtype=np.int64)
    sigmas = _q_enum_signs(d_child)
    _Q_CACHE[d_child] = (windows, ell_int_sums, sigmas)
    return _Q_CACHE[d_child]


def estimate_q_survivors(F_survivors, n_half_child, m, c_target,
                            K=Q_SAMPLE_K, rng=None):
    """Estimate F+Q survivor count from F-survivors via sampling.

    Sound for monotonicity: Q-survivors ⊆ F-survivors.  We compute either:
      - Exact: when |F| <= K, run Q on ALL F-survivors.
      - Sample: sample K F-survivors uniformly without replacement, run Q
        on each, compute hat-p_keep = (# Q-survivors in sample) / K.
        Point estimate: |F| * hat-p_keep.
        Upper bound (95%, when sample shows 0 survivors): rule-of-three,
        max(1, ceil(3*|F|/K)).

    Returns dict:
       n_FQ_point_est : int      (point estimate of F+Q survivors)
       n_FQ_upper     : int      (conservative upper bound)
       n_F            : int      (exact F count)
       q_kept_in_sample : int    (# of sample that survived Q)
       sample_size    : int      (K or |F|, whichever smaller)
       method         : 'exact' | 'sample' | 'sample_zero'
       wall_sec       : float    (Q wall time)
    """
    n_F = int(len(F_survivors))
    if n_F == 0:
        return {'n_FQ_point_est': 0, 'n_FQ_upper': 0, 'n_F': 0,
                'q_kept_in_sample': 0, 'sample_size': 0,
                'method': 'exact', 'wall_sec': 0.0}

    d_child = int(F_survivors.shape[1])
    windows, ell_int_sums, sigmas = _get_q_setup(d_child)

    if n_F <= K:
        # Exact: run Q on all F-survivors
        sample = F_survivors
        method = 'exact'
    else:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(n_F, K, replace=False)
        sample = F_survivors[idx]
        method = 'sample'

    t0 = time.time()
    q_kept = 0
    for c in sample:
        if not prune_Q_one(c, windows, ell_int_sums, sigmas,
                            n_half_child, m, c_target):
            q_kept += 1
    wall = time.time() - t0
    Ks = len(sample)

    if method == 'exact':
        return {'n_FQ_point_est': q_kept, 'n_FQ_upper': q_kept,
                'n_F': n_F, 'q_kept_in_sample': q_kept,
                'sample_size': Ks, 'method': 'exact',
                'wall_sec': round(wall, 3)}

    p_keep = q_kept / Ks
    point = int(round(p_keep * n_F))
    if q_kept == 0:
        # Rule-of-three upper bound
        upper = max(1, int(np.ceil(3.0 * n_F / Ks)))
        method = 'sample_zero'
    else:
        # Wilson-style 1-sigma upper (rough): p̂ + sqrt(p̂(1-p̂)/Ks)
        sigma = (p_keep * (1 - p_keep) / Ks) ** 0.5
        p_upper = min(1.0, p_keep + 2.0 * sigma)
        upper = int(np.ceil(p_upper * n_F))
    return {'n_FQ_point_est': point, 'n_FQ_upper': upper,
            'n_F': n_F, 'q_kept_in_sample': q_kept,
            'sample_size': Ks, 'method': method,
            'wall_sec': round(wall, 3)}


def estimate_one(n_half, m, c_target, max_levels, sample_n,
                  level_time_sec, compute_budget, l0_timeout,
                  parent_time_sec=DEFAULT_PARENT_TIME,
                  use_split_cell_max_d=0, sp_max_depth=1,
                  max_l_for_split=200,
                  n_workers=N_WORKERS_DEFAULT):
    """Sample-based cascade estimator at one (n_half, m, c_target).
    Returns dict of per-level stats and final verdict."""
    d0 = 2 * n_half
    S0 = 4 * n_half * m
    n_l0_total = n_full_compositions(d0, S0)
    corr = correction(m, n_half)
    out = {
        'n_half': n_half, 'm': m, 'd0': d0, 'S0': S0,
        'c_target': c_target, 'correction': corr,
        'threshold_for_g': c_target + corr,
        'l0_total_compositions': n_l0_total,
        'sample_n': sample_n,
        'compute_budget': compute_budget,
        'levels': [],
    }
    if c_target + corr >= C_UPPER:
        out['verdict'] = 'VACUOUS'
        return out

    # ---- L0: full canonical enumeration with F+Q ----
    t0 = time.time()
    print(f"  L0: enumerating {n_l0_total:,} compositions @ d={d0}", flush=True)
    try:
        # L at L0 (small d) barely helps and eats wall budget — disable.
        r0 = run_level0(n_half, m, c_target, verbose=False, use_F=True,
                          use_Q=True, use_L=False)  # F kernel + parallel Q at L0
    except Exception as e:
        out['verdict'] = f'L0_ERROR: {e}'
        return out
    survivors = r0['survivors']  # actual survivor compositions
    n_l0_surv = int(r0['n_survivors'])

    # Apply L (Shor SDP) and SP (split-cell) post-filters at L0.
    n_l0_after_L = n_l0_surv
    n_l0_after_SP = n_l0_surv
    wall_L0_L = 0.0
    wall_L0_SP = 0.0
    if n_l0_surv > 0:
        try:
            tL = time.time()
            survivors = apply_L_filter_parallel(
                survivors, n_half, m, c_target,
                solver='MOSEK', n_workers=n_workers)
            n_l0_after_L = int(len(survivors))
            wall_L0_L = time.time() - tL
        except Exception as e:
            print(f"  L0 L EXC: {e}", flush=True)
        if (use_split_cell_max_d > 0 and d0 <= use_split_cell_max_d
                and 0 < n_l0_after_L <= max_l_for_split):
            try:
                tSP = time.time()
                survivors = apply_split_cell_filter_parallel(
                    survivors, n_half, m, c_target,
                    n_workers=n_workers,
                    max_d=use_split_cell_max_d,
                    early_terminate=True,
                    max_depth=sp_max_depth)
                n_l0_after_SP = int(len(survivors))
                wall_L0_SP = time.time() - tSP
            except Exception as e:
                print(f"  L0 SP EXC: {e}", flush=True)
                n_l0_after_SP = n_l0_after_L
        else:
            n_l0_after_SP = n_l0_after_L
    n_l0_surv = n_l0_after_SP

    l0_wall = time.time() - t0
    out['levels'].append({
        'level': 0, 'n_compositions': n_l0_total,
        'n_survivors_FFNQ': int(r0['n_survivors']),
        'n_survivors_after_L': n_l0_after_L,
        'n_survivors_after_SP': n_l0_after_SP,
        'n_survivors': n_l0_surv,
        'wall_L_sec': round(wall_L0_L, 2),
        'wall_SP_sec': round(wall_L0_SP, 2),
        'wall_sec': round(l0_wall, 2),
    })
    print(f"  L0 done: F+FN+Q={r0['n_survivors']:,} -> "
          f"+L={n_l0_after_L:,} -> +SP={n_l0_after_SP:,}  "
          f"in {l0_wall:.1f}s (L={wall_L0_L:.1f}s, SP={wall_L0_SP:.1f}s)",
          flush=True)

    cum_children = float(n_l0_total)  # L0 already enumerated everything
    if n_l0_surv == 0:
        out['verdict'] = 'CLOSED_AT_L0'
        out['cum_children_estimated'] = cum_children
        return out

    # ---- L1+ sampling ----
    rng = np.random.default_rng()
    current_survivors = survivors  # the survivor composition pool
    n_curr = n_l0_surv
    d_parent = d0
    level_pool_pruned_to_zero_count = 0  # how many sampled parents had 0 survivors

    for L in range(1, max_levels + 1):
        if n_curr == 0:
            out['verdict'] = f'CLOSED_AT_L{L-1}_DERIVED'
            break
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        sample_size = min(int(sample_n), int(len(current_survivors)))
        if sample_size == 0:
            out['verdict'] = f'CLOSED_AT_L{L-1}_DERIVED'
            break
        if len(current_survivors) > sample_size:
            idx = rng.choice(len(current_survivors), sample_size, replace=False)
            sample = current_survivors[idx]
        else:
            sample = current_survivors

        print(f"\n  L{L}: d_parent={d_parent} -> d_child={d_child}, "
              f"n_curr={n_curr:,}, sampling {len(sample)} parents", flush=True)

        per_parent = []
        sample_total_children = 0
        sample_total_survivors_compositions = []
        t_level = time.time()
        n_parents_completed = 0
        budget_exceeded = False
        # Track the live running expected total survivors so we can early-kill
        running_surv = 0
        slow_parents_skipped = 0

        for i, parent in enumerate(sample):
            elapsed_so_far = time.time() - t_level
            if elapsed_so_far > level_time_sec:
                budget_exceeded = True
                print(f"    [time budget exhausted after {i}/{len(sample)} "
                      f"parents @ {elapsed_so_far:.0f}s]", flush=True)
                break
            tp = time.time()
            try:
                # 1) F kernel only (numba parallel, ~64 cores).  Returns ALL F-survivors.
                surv_F_i, n_children_i = process_parent_fused(
                    parent, m, c_target, n_half_child,
                    use_flat_threshold=False, use_F=True, use_Q=False,
                    skip_sdp_cert=True)
            except Exception as e:
                print(f"    parent {i} EXC: {e}", flush=True)
                continue
            wall_F_p = time.time() - tp
            n_F_i = int(len(surv_F_i))

            # 2) FN post-filter (numba parallel kernel; tightens F).
            tFN = time.time()
            try:
                surv_FN_i = apply_FN_filter_parallel(
                    surv_F_i, n_half_child, m, c_target)
            except Exception as e:
                print(f"    parent {i} FN EXC: {e}", flush=True)
                surv_FN_i = surv_F_i
            wall_FN_p = time.time() - tFN
            n_FN_i = int(len(surv_FN_i))

            # 3) Q post-filter (parallel multiprocessing.Pool).  Exact F+FN+Q.
            tQ = time.time()
            try:
                surv_Q_i = apply_Q_filter_parallel(
                    surv_FN_i, n_half_child, m, c_target,
                    n_workers=n_workers)
            except Exception as e:
                print(f"    parent {i} Q EXC: {e}", flush=True)
                surv_Q_i = surv_FN_i  # fall back: don't trust Q-prune
            wall_Q_p = time.time() - tQ
            n_Q_i = int(len(surv_Q_i))

            # 4) QN-fast post-filter (parallel Pool).  Strict tightening of Q
            #    via per-window m_W = min(op_rest(A_W)·d, n_pairs_W).  Sound;
            #    +25.8% extra prune over Q at d=10 empirically.
            tQN = time.time()
            try:
                surv_QN_i = apply_QN_filter_parallel(
                    surv_Q_i, n_half_child, m, c_target,
                    n_workers=n_workers)
            except Exception as e:
                print(f"    parent {i} QN EXC: {e}", flush=True)
                surv_QN_i = surv_Q_i  # fall back: don't trust QN-prune
            wall_QN_p = time.time() - tQN
            n_QN_i = int(len(surv_QN_i))

            # 5) L post-filter (parallel Pool, MOSEK SDP).  Exact F+FN+Q+QN+L.
            tL = time.time()
            try:
                surv_L_i = apply_L_filter_parallel(
                    surv_QN_i, n_half_child, m, c_target,
                    solver='MOSEK', n_workers=n_workers)
            except Exception as e:
                print(f"    parent {i} L EXC: {e}", flush=True)
                surv_L_i = surv_QN_i
            wall_L_p = time.time() - tL
            n_L_i = int(len(surv_L_i))

            # 6) SP (split-cell) post-filter — gated to d <= use_split_cell_max_d.
            #    Splits each L-survivor into 2^d sub-cells; if all are SDP-
            #    infeasible, parent is split-pruned.  Direct MOSEK + smart
            #    sigma ordering + early termination.  Sound (Farkas certs).
            wall_SP_p = 0.0
            n_SP_i = n_L_i
            surv_SP_i = surv_L_i
            if (use_split_cell_max_d > 0
                    and d_child <= use_split_cell_max_d
                    and 0 < n_L_i <= max_l_for_split):
                tSP = time.time()
                try:
                    surv_SP_i = apply_split_cell_filter_parallel(
                        surv_L_i, n_half_child, m, c_target,
                        n_workers=n_workers,
                        max_d=use_split_cell_max_d,
                        early_terminate=True,
                        max_depth=sp_max_depth)
                except Exception as e:
                    print(f"    parent {i} SP EXC: {e}", flush=True)
                    surv_SP_i = surv_L_i
                wall_SP_p = time.time() - tSP
                n_SP_i = int(len(surv_SP_i))

            wall_p = time.time() - tp
            if wall_p > parent_time_sec * 5:
                slow_parents_skipped += 1
            sample_total_children += int(n_children_i)
            n_surv_i = n_SP_i  # post-F-FN-Q-QN-L-SP count is the level survivor count

            # Pool for the next level = SP-survivors (tightest pruning).
            sample_total_survivors_compositions.append(surv_SP_i)
            n_parents_completed += 1
            if n_SP_i == 0:
                level_pool_pruned_to_zero_count += 1
            per_parent.append({
                'idx': int(i), 'children': int(n_children_i),
                'F_survivors': n_F_i,
                'FN_survivors': n_FN_i,
                'Q_survivors': n_Q_i,
                'QN_survivors': n_QN_i,
                'L_survivors': n_L_i,
                'SP_survivors': n_SP_i,
                'wall_F_sec': round(wall_F_p, 3),
                'wall_FN_sec': round(wall_FN_p, 3),
                'wall_Q_sec': round(wall_Q_p, 3),
                'wall_QN_sec': round(wall_QN_p, 3),
                'wall_L_sec': round(wall_L_p, 3),
                'wall_SP_sec': round(wall_SP_p, 3),
                'wall_total_sec': round(wall_p, 3),
            })
            print(f"    [{i+1}/{len(sample)}] children={n_children_i:,}  "
                  f"F={n_F_i} -> FN={n_FN_i} -> Q={n_Q_i} -> "
                  f"QN={n_QN_i} -> L={n_L_i} -> SP={n_SP_i}  "
                  f"wall={wall_p:.1f}s "
                  f"(F={wall_F_p:.1f} FN={wall_FN_p:.1f} Q={wall_Q_p:.1f} "
                  f"QN={wall_QN_p:.1f} L={wall_L_p:.1f} SP={wall_SP_p:.1f})",
                  flush=True)

        wall_level = time.time() - t_level
        if n_parents_completed == 0:
            out['levels'].append({
                'level': L, 'n_parents_in_pool': n_curr,
                'sample_size': len(sample),
                'n_parents_completed': 0, 'verdict_partial': 'NO_PROGRESS',
                'wall_sec': round(wall_level, 2),
            })
            out['verdict'] = f'STUCK_AT_L{L}'
            break

        avg_children = sample_total_children / n_parents_completed
        avg_F = sum(p['F_survivors']
                    for p in per_parent) / n_parents_completed
        avg_FN = sum(p.get('FN_survivors', p['F_survivors'])
                     for p in per_parent) / n_parents_completed
        avg_Q = sum(p['Q_survivors']
                    for p in per_parent) / n_parents_completed
        avg_QN = sum(p.get('QN_survivors', p['Q_survivors'])
                     for p in per_parent) / n_parents_completed
        avg_L = sum(p['L_survivors']
                    for p in per_parent) / n_parents_completed
        avg_SP = sum(p.get('SP_survivors', p['L_survivors'])
                     for p in per_parent) / n_parents_completed
        avg_survivors = avg_SP  # final cascade-level survivor count is post-SP
        est_total_children = avg_children * n_curr
        est_total_F = int(round(avg_F * n_curr))
        est_total_FN = int(round(avg_FN * n_curr))
        est_total_Q = int(round(avg_Q * n_curr))
        est_total_QN = int(round(avg_QN * n_curr))
        est_total_L = int(round(avg_L * n_curr))
        est_total_SP = int(round(avg_SP * n_curr))
        est_total_survivors = est_total_SP

        print(f"    level summary: "
              f"avg_children={avg_children:,.0f}  "
              f"avg_F={avg_F:.1f}  avg_FN={avg_FN:.1f}  "
              f"avg_Q={avg_Q:.1f}  avg_QN={avg_QN:.1f}  "
              f"avg_L={avg_L:.2f}  avg_SP={avg_SP:.2f}  "
              f"est_total_children={est_total_children:,.2e}  "
              f"est_total_F={est_total_F:,}  est_total_FN={est_total_FN:,}  "
              f"est_total_Q={est_total_Q:,}  est_total_QN={est_total_QN:,}  "
              f"est_total_L={est_total_L:,}  est_total_SP={est_total_SP:,}  "
              f"wall={wall_level:.1f}s", flush=True)

        cum_children += est_total_children
        out['levels'].append({
            'level': L, 'd_child': d_child,
            'n_parents_in_pool': int(n_curr),
            'sample_size': int(len(sample)),
            'n_parents_completed': int(n_parents_completed),
            'avg_children_per_parent': float(avg_children),
            'avg_F_per_parent': float(avg_F),
            'avg_FN_per_parent': float(avg_FN),
            'avg_Q_per_parent': float(avg_Q),
            'avg_QN_per_parent': float(avg_QN),
            'avg_L_per_parent': float(avg_L),
            'avg_SP_per_parent': float(avg_SP),
            'avg_survivors_per_parent': float(avg_survivors),
            'est_total_children': float(est_total_children),
            'est_total_F': int(est_total_F),
            'est_total_FN': int(est_total_FN),
            'est_total_Q': int(est_total_Q),
            'est_total_QN': int(est_total_QN),
            'est_total_L': int(est_total_L),
            'est_total_SP': int(est_total_SP),
            'est_total_survivors': int(est_total_survivors),
            'cum_children_so_far': float(cum_children),
            'wall_sec': round(wall_level, 2),
            'wall_per_parent_sec': round(wall_level / max(1, n_parents_completed), 3),
            'time_budget_hit': bool(budget_exceeded),
            'per_parent': per_parent,
        })

        # Termination logic
        if avg_survivors == 0:
            out['verdict'] = f'CLOSED_AT_L{L}_SAMPLE'
            break
        # Hard kill: estimated absolute survivor count is too big to ever process
        if est_total_survivors > MAX_SURV_HARD:
            out['verdict'] = (f'IMPOSSIBLE_AT_L{L}: '
                              f'est_total_survivors={est_total_survivors:.2e} > '
                              f'MAX={MAX_SURV_HARD:.0e} '
                              f'(would need >{MAX_SURV_HARD*1e7:.0e} children at L{L+1})')
            break
        # Compute budget kill
        if est_total_children > compute_budget:
            out['verdict'] = (f'IMPOSSIBLE_AT_L{L}: '
                              f'est_total_children={est_total_children:.2e} > '
                              f'budget={compute_budget:.2e}')
            break
        if cum_children > compute_budget:
            out['verdict'] = (f'IMPOSSIBLE_CUMULATIVE_L{L}: '
                              f'cum={cum_children:.2e} > '
                              f'budget={compute_budget:.2e}')
            break
        # Convergence diagnostic: if we have at least 2 levels of L1+ data
        # and avg_survivors hasn't dropped substantially, project poor convergence.
        if L >= 2:
            prior_level = next((lev for lev in out['levels']
                                  if lev['level'] == L - 1
                                  and 'avg_survivors_per_parent' in lev), None)
            if prior_level is not None:
                A_prev = prior_level['avg_survivors_per_parent']
                A_curr = avg_survivors
                if A_prev > 1 and A_curr > 0:
                    alpha = A_curr / A_prev
                    out['levels'][-1]['alpha_vs_prev'] = alpha
                    if alpha > ALPHA_NONCONVERGE and A_curr > SURV_NONTRIVIAL:
                        out['verdict'] = (f'NONCONVERGENT_AT_L{L}: '
                                          f'alpha={alpha:.2f} > {ALPHA_NONCONVERGE} '
                                          f'with avg_survivors={A_curr:.0f} > '
                                          f'{SURV_NONTRIVIAL}')
                        break

        # Build pool for next level: combine sampled parents' actual survivors
        if sample_total_survivors_compositions:
            pooled = np.vstack([s for s in sample_total_survivors_compositions
                                  if len(s) > 0]) if any(
                len(s) > 0 for s in sample_total_survivors_compositions) else None
            if pooled is None or len(pooled) == 0:
                out['verdict'] = f'CLOSED_AT_L{L}_SAMPLE'
                break
            current_survivors = pooled
        n_curr = est_total_survivors
        d_parent = d_child

    if 'verdict' not in out:
        out['verdict'] = f'OUT_OF_LEVELS_AT_L{max_levels}'
    out['cum_children_estimated'] = cum_children
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c_target', type=float, default=1.20,
                     help='target lower bound')
    ap.add_argument('--sample_n', type=int, default=DEFAULT_SAMPLE_N)
    ap.add_argument('--level_time_sec', type=float, default=DEFAULT_LEVEL_TIME)
    ap.add_argument('--compute_budget', type=float, default=DEFAULT_COMPUTE_BUDGET)
    ap.add_argument('--l0_timeout', type=float, default=DEFAULT_L0_TIMEOUT)
    ap.add_argument('--max_levels', type=int, default=4)
    ap.add_argument('--use_split_cell_max_d', type=int, default=0,
                     help='Apply split-cell SDP after L when d_child <= this value '
                          '(0 = disabled, default; 10 = enable up through d=10).')
    ap.add_argument('--sp_max_depth', type=int, default=1,
                     help='Split-cell recursion depth: 1 = standard binary, '
                          '2 = recurse into stuck sub-cells (cost (2^d)^depth, '
                          'use sparingly).')
    ap.add_argument('--max_l_for_split', type=int, default=200,
                     help='Skip split-cell when n_L_survivors > this (cost guard).')
    ap.add_argument('--n_workers', type=int, default=N_WORKERS_DEFAULT,
                     help='Parallel workers for Q/QN/L/SP filters.')
    ap.add_argument('--out_dir', type=str, default=None)
    ap.add_argument('--configs', type=str, default=None,
                     help='JSON list of [n_half, m] pairs.  '
                          'Default: [(2,15),(2,20),(3,10),(3,15)].')
    args = ap.parse_args()

    if args.configs:
        configs = json.loads(args.configs)
    else:
        configs = [
            (2, 15),    # d0=4, L0=302K (fast)
            (2, 20),    # d0=4, L0=708K (fast)
            (3, 10),    # d0=6, L0=234M (~30s L0)
            (3, 15),    # d0=6, L0=1.95B (~5 min L0)
        ]

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(ROOT, f'cascade_estimate_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'summary.json')
    summary = {
        'started_utc': ts,
        'c_target': args.c_target,
        'sample_n': args.sample_n,
        'level_time_sec': args.level_time_sec,
        'compute_budget': args.compute_budget,
        'use_split_cell_max_d': args.use_split_cell_max_d,
        'max_l_for_split': args.max_l_for_split,
        'n_workers': args.n_workers,
        'configs': configs,
        'host': (os.uname().nodename if hasattr(os, 'uname')
                 else os.environ.get('COMPUTERNAME', 'unknown')),
        'results': [],
    }

    def save():
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"Run dir: {out_dir}", flush=True)
    save()

    for (n_half, m) in configs:
        print(f"\n{'='*70}\n"
              f"CONFIG: n_half={n_half}, m={m}, c_target={args.c_target}\n"
              f"{'='*70}", flush=True)
        r = estimate_one(n_half, m, args.c_target, args.max_levels,
                          args.sample_n, args.level_time_sec,
                          args.compute_budget, args.l0_timeout,
                          use_split_cell_max_d=args.use_split_cell_max_d,
                          sp_max_depth=args.sp_max_depth,
                          max_l_for_split=args.max_l_for_split,
                          n_workers=args.n_workers)
        summary['results'].append(r)
        save()
        print(f"\n  >>> verdict: {r['verdict']}  "
              f"cum_children_est={r.get('cum_children_estimated', 'n/a'):.2e}\n",
              flush=True)

    save()
    print("\nDONE.", flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == '__main__':
    main()
