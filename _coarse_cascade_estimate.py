"""Coarse-grid cascade compute estimator — SOUND (3-layer N+O → J → L).

Coarse-grid analogue of `_cascade_estimate.py`.  The fine grid runs F → FN →
Q → QN → L → SP at each level; the COARSE grid uses the v4 stack:

  Layer 1 (N+O)   : sound njit kernel — sparsity LP for cell_var, spectral
                    floor for quad_corr.  See run_cascade_coarse_v3.py and
                    _coarse_NO_bench.py.  Sound: each ingredient is taken
                    as `min` against the v2 baseline (no soundness regression).

  Layer 2 (Joint dual K=4) : sound LP-duality bound on
                    `min_δ max_W TV_W(c+δ)`.  Subgradient ascent over the
                    simplex of K windows.  See _coarse_J_bench.py.  Sound by
                    LP weak duality + M1 LP-tight + per-window pair_bound
                    triangle.

  Layer 3 (Shor SDP) : per-cell PSD lift Y = [[1, δᵀ], [δ, D]] ⪰ 0 with full
                    RLT cuts; sound LB on the QP minimum.  See
                    _coarse_L_bench.py.  Cell certified if SDP_LB ≥ c_target.

For each (d0, S, c_target):
   L0   : full canonical enumeration via run_cascade_coarse_v4.run_level0().
          All compositions; sound.
   L1+  : sample SAMPLE_N parents from previous-level survivors, expand each
          via process_parent_v4 (Numba fused gen-and-prune for v3 N+O,
          Python loop for J + Shor SDP layers).  Estimate total cost and
          survivor count via the expansion factor.
   Stop : (a) avg_survivors == 0 across the sample → CLOSURE conjectured at
              this level (sample-based; statistical, not certified).
          (b) cumulative est_total_children > COMPUTE_BUDGET → IMPOSSIBLE.
          (c) wall time per level > LEVEL_TIME_BUDGET → cap and report.
          (d) avg expansion ratio α > 0.5 across two consecutive levels with
              non-trivial survivor counts → NONCONVERGENT.

Soundness chain (each layer's certified set ⊇ previous):
  v2 (triangle baseline) ⊆ v3 (N+O) ⊆ v4 (N+O+J+L)

Speed:
  L1+ parents are processed in parallel via a persistent ProcessPoolExecutor
  (`--n_workers`).  Each worker pre-imports the v4 stack at startup and runs
  Numba in single-thread mode to avoid oversubscription with the parent-level
  pool.  Pool is created once in main() and reused across configs.

Why sample-based estimation:
  Full L1+ enumeration becomes infeasible past d=8-10 even on the coarse
  grid (millions of children per parent).  Sampling gives an unbiased
  estimate; a separate full-cascade run is needed for an actual proof —
  this script is for **planning**.

Usage:
    python _coarse_cascade_estimate.py --c_target 1.20 --sample_n 30 \\
        --max_levels 4
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from math import comb

import numpy as np

# --- Path setup ---
_DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('CASCADE_ROOT', _DEFAULT_ROOT)
if not os.path.isdir(os.path.join(ROOT, 'cloninger-steinerberger')):
    ROOT = _DEFAULT_ROOT
_CS_DIR = os.path.join(ROOT, 'cloninger-steinerberger')
_CPU_DIR = os.path.join(_CS_DIR, 'cpu')
for _p in (_CPU_DIR, _CS_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pruning import count_compositions
from run_cascade_coarse_v4 import (run_level0, process_parent_v4,
                                     precompute_op_rest_d)
from run_cascade_coarse_v2 import coarse_x_cap


# --- Constants and defaults ---

C_UPPER = 1.5029                     # current best upper bound on C_{1a}
N_WORKERS_DEFAULT = max(1, (os.cpu_count() or 4) // 2)

DEFAULT_SAMPLE_N      = 30
DEFAULT_LEVEL_TIME    = 600          # 10 min per L1+ sampling level
DEFAULT_PARENT_TIME   = 60           # 60s per parent
DEFAULT_COMPUTE_BUDGET= 1e15         # cumulative est children kill threshold
DEFAULT_L0_TIMEOUT    = 1800         # 30 min for L0 full enum
DEFAULT_MAX_LEVELS    = 8            # run until 0 survivors or budget hit

# Convergence kill thresholds (mirror fine-grid estimator).
MAX_SURV_HARD       = 1e9
ALPHA_NONCONVERGE   = 0.5
SURV_NONTRIVIAL     = 1000


def n_full_compositions(d, S):
    """Total compositions of d non-negative integers summing to S."""
    return comb(S + d - 1, d - 1)


# =====================================================================
# Worker pool plumbing (parallel L1+ parent expansion).
# =====================================================================

def _worker_init(root_path):
    """Initializer: set sys.path, pin Numba to 1 thread (avoid oversub),
    pre-import the v4 stack so JIT compile happens once per worker."""
    cs_dir = os.path.join(root_path, 'cloninger-steinerberger')
    cpu_dir = os.path.join(cs_dir, 'cpu')
    for p in (cpu_dir, cs_dir, root_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass
    # Pre-import → forces module load + cached-JIT load.
    import run_cascade_coarse_v4  # noqa: F401


def _worker_process_parent(args):
    """Run process_parent_v4 in a worker; returns
    (survivors, n_tested, counts, wall_sec).  Worker measures its own wall
    so per-parent timing reflects compute, not queue wait."""
    parent, d_child, S, c_target, op_rest_d_arr, kw = args
    from run_cascade_coarse_v4 import process_parent_v4
    t0 = time.time()
    survivors, n_tested, counts = process_parent_v4(
        parent, d_child, S, c_target, op_rest_d_arr, **kw)
    return survivors, n_tested, counts, time.time() - t0


def _maybe_make_pool(n_workers):
    """Spawn a ProcessPoolExecutor or return None for sequential fallback."""
    if n_workers <= 1:
        return None
    return ProcessPoolExecutor(
        max_workers=int(n_workers),
        mp_context=mp.get_context('spawn'),
        initializer=_worker_init,
        initargs=(ROOT,),
    )


# =====================================================================
# Main estimator.
# =====================================================================

def estimate_one(d0, S, c_target, max_levels=DEFAULT_MAX_LEVELS,
                 sample_n=DEFAULT_SAMPLE_N,
                 level_time_sec=DEFAULT_LEVEL_TIME,
                 compute_budget=DEFAULT_COMPUTE_BUDGET,
                 l0_timeout=DEFAULT_L0_TIMEOUT,
                 parent_time_sec=DEFAULT_PARENT_TIME,
                 use_joint=True, use_sdp=True,
                 joint_top_K=4, joint_iters=20, sdp_mode='best_only',
                 n_workers=N_WORKERS_DEFAULT, pool=None):
    """Sample-based coarse-cascade estimator for one (d0, S, c_target).

    `pool`: optional persistent ProcessPoolExecutor reused across configs.
    If None and `n_workers > 1`, a temporary pool is spawned and torn down.
    """
    n_l0_total = n_full_compositions(d0, S)
    out = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'l0_total_compositions': n_l0_total,
        'sample_n': sample_n,
        'compute_budget': compute_budget,
        'use_joint': use_joint, 'use_sdp': use_sdp,
        'joint_top_K': joint_top_K,
        'n_workers': int(n_workers),
        'levels': [],
    }
    if c_target >= C_UPPER:
        out['verdict'] = 'VACUOUS'
        return out

    print(f"  Plan:", flush=True)
    print(f"    L0: full enumerate {n_l0_total:,} compositions, "
          f"apply NO+J+L progressively", flush=True)
    print(f"    L1+: at each level, randomly sample {sample_n} parents from "
          f"the previous level's pool,", flush=True)
    print(f"         expand each via process_parent_v4 ({n_workers} workers), "
          f"measure expansion factor and per-layer prunes,", flush=True)
    print(f"         extrapolate total cost = avg_children × pool_size.",
          flush=True)
    print(f"    Stop when: 0 survivors / IMPOSSIBLE budget / NONCONVERGENT "
          f"alpha / max_levels={max_levels}.", flush=True)
    print(f"    Layers: NO=on, "
          f"J={'on' if use_joint else 'off'} (top_K={joint_top_K}, iters={joint_iters}), "
          f"L={'on' if use_sdp else 'off'} (mode={sdp_mode})", flush=True)

    # ---- L0 ----
    print(f"\n  --- L0: precomputing op_rest_d table for d={d0} ---", flush=True)
    t_pre0 = time.time()
    op_rest_d_cache = {d0: precompute_op_rest_d(d0)}
    print(f"  precompute done in {time.time()-t_pre0:.2f}s", flush=True)

    t0 = time.time()
    print(f"\n  --- L0: enumerating {n_l0_total:,} compositions @ d={d0}, S={S} ---",
          flush=True)
    try:
        l0 = run_level0(d0, S, c_target, op_rest_d_cache[d0],
                        use_joint=use_joint, use_sdp=use_sdp,
                        joint_top_K=joint_top_K, joint_iters=joint_iters,
                        sdp_mode=sdp_mode, verbose=False)
    except Exception as e:
        out['verdict'] = f'L0_ERROR: {e}'
        return out
    l0_wall = time.time() - t0
    if l0_wall > l0_timeout:
        out['verdict'] = f'L0_TIMEOUT ({l0_wall:.0f}s > {l0_timeout}s)'

    n_l0_surv      = int(l0['n_survivors'])
    n_cert_NO_l0   = int(l0.get('n_certified_NO', 0))
    n_cert_J_l0    = int(l0.get('n_certified_J', 0))
    n_cert_L_l0    = int(l0.get('n_certified_L', 0))
    n_uncert_l0    = int(l0.get('n_uncertified', n_l0_surv))
    n_pruned_a_l0  = int(l0.get('n_pruned_asym', 0))
    t_NO_l0        = float(l0.get('time_NO', 0.0))
    t_J_l0         = float(l0.get('time_J', 0.0))
    t_L_l0         = float(l0.get('time_L', 0.0))

    out['levels'].append({
        'level': 0, 'd_child': d0,
        'n_compositions': n_l0_total,
        'n_certified_NO': n_cert_NO_l0,
        'n_certified_J':  n_cert_J_l0,
        'n_certified_L':  n_cert_L_l0,
        'n_uncertified':  n_uncert_l0,
        'n_pruned_asym':  n_pruned_a_l0,
        'n_survivors':    n_l0_surv,
        'wall_sec':       round(l0_wall, 2),
        'time_NO':        t_NO_l0,
        'time_J':         t_J_l0,
        'time_L':         t_L_l0,
        'box_certified':  bool(l0.get('box_certified', False)),
        'proven':         bool(l0.get('proven', False)),
    })
    n0_inv = 100.0 / max(1, n_l0_total)
    pct_NO   = n_cert_NO_l0 * n0_inv
    pct_J    = n_cert_J_l0  * n0_inv
    pct_L    = n_cert_L_l0  * n0_inv
    pct_hard = n_uncert_l0  * n0_inv
    print(f"\n  ===== L0 SUMMARY =====", flush=True)
    print(f"  total compositions   : {n_l0_total:,}", flush=True)
    print(f"  pruned by asymmetry  : {n_pruned_a_l0:,}", flush=True)
    print(f"  certified by NO      : {n_cert_NO_l0:,}  ({pct_NO:.2f}%)  in {t_NO_l0:.2f}s", flush=True)
    print(f"  certified by Joint   : {n_cert_J_l0:,}  ({pct_J:.2f}%)  in {t_J_l0:.2f}s", flush=True)
    print(f"  certified by Shor    : {n_cert_L_l0:,}  ({pct_L:.2f}%)  in {t_L_l0:.2f}s", flush=True)
    print(f"  uncertified (hard)   : {n_uncert_l0:,}  ({pct_hard:.2f}%)", flush=True)
    print(f"  survivors -> L1      : {n_l0_surv:,}", flush=True)
    print(f"  L0 wall time         : {l0_wall:.2f}s", flush=True)

    cum_children = float(n_l0_total)
    if n_l0_surv == 0:
        out['verdict'] = 'CLOSED_AT_L0'
        out['cum_children_estimated'] = cum_children
        out['total_wall_sec'] = round(l0_wall, 2)
        return out

    # ---- L1+ sample-based estimation ----
    rng = np.random.default_rng()
    current_survivors = l0['survivors']
    n_curr = n_l0_surv
    d_parent = d0

    # Spawn pool once if not provided; tear down at exit.
    own_pool = False
    if pool is None and n_workers > 1:
        pool = _maybe_make_pool(n_workers)
        own_pool = pool is not None

    try:
        for L in range(1, max_levels + 1):
            if n_curr == 0:
                out['verdict'] = f'CLOSED_AT_L{L-1}_DERIVED'
                break
            d_child = 2 * d_parent

            # Lazy-cache op_rest_d for this d_child.
            ord_d = op_rest_d_cache.get(d_child)
            if ord_d is None:
                t_pre = time.time()
                ord_d = precompute_op_rest_d(d_child)
                op_rest_d_cache[d_child] = ord_d
                print(f"  op_rest_d table for d={d_child}: "
                      f"{time.time()-t_pre:.2f}s", flush=True)

            len_curr = len(current_survivors)
            sample_size = min(int(sample_n), len_curr)
            if sample_size == 0:
                out['verdict'] = f'CLOSED_AT_L{L-1}_DERIVED'
                break
            if len_curr > sample_size:
                idx = rng.choice(len_curr, sample_size, replace=False)
                sample = current_survivors[idx]
                sample_method = 'random'
            else:
                sample = current_survivors
                sample_method = 'full'

            print(f"\n  =================================================", flush=True)
            print(f"  LEVEL L={L}:  d_parent={d_parent}  ->  d_child={d_child}", flush=True)
            print(f"  =================================================", flush=True)
            print(f"  Pool size at L{L-1}    : {n_curr:,} survivors (estimated)", flush=True)
            print(f"  RANDOM SAMPLE         : {len(sample):,} parents  "
                  f"(method={sample_method}, sample_n={sample_n})", flush=True)
            print(f"  -> expand each, measure children + per-layer cert counts",
                  flush=True)
            print(f"  -> extrapolate: avg_children × n_curr = est. total_children",
                  flush=True)

            # Pre-filter infeasible parents (any bin > 2*x_cap).
            x_cap = coarse_x_cap(d_child, S, c_target)
            two_x_cap = 2 * x_cap
            feasible_mask = np.all(sample <= two_x_cap, axis=1)
            n_infeasible = int(feasible_mask.size - feasible_mask.sum())
            if n_infeasible > 0:
                sample = sample[feasible_mask]
                print(f"    pre-filtered {n_infeasible} infeasible parents",
                      flush=True)
                if len(sample) == 0:
                    out['levels'].append({
                        'level': L, 'd_child': d_child,
                        'n_parents_in_pool': n_curr,
                        'sample_size': 0,
                        'n_parents_completed': 0,
                        'verdict_partial': 'ALL_INFEASIBLE',
                        'wall_sec': 0.0,
                    })
                    out['verdict'] = f'CLOSED_AT_L{L}_INFEASIBLE'
                    break

            # Aggregate counters (incremental — no list-comprehension reductions).
            sample_total_children   = 0
            sample_total_NO_cert    = 0
            sample_total_J_cert     = 0
            sample_total_L_cert     = 0
            sample_total_uncert     = 0
            sample_total_survivors  = 0
            sample_total_time_NO    = 0.0
            sample_total_time_J     = 0.0
            sample_total_time_L     = 0.0
            sample_survivors_compositions = []
            per_parent = []
            t_level = time.time()
            n_parents_completed = 0
            budget_exceeded = False
            slow_parents_skipped = 0
            n_sample = len(sample)
            slow_thr = parent_time_sec * 5

            kw = {'use_joint': use_joint, 'use_sdp': use_sdp,
                  'joint_top_K': joint_top_K, 'joint_iters': joint_iters,
                  'sdp_mode': sdp_mode}

            def _record(i, wall_p, survivors, n_tested, counts):
                """Tally one parent's result; capture closure references."""
                nonlocal sample_total_children, sample_total_NO_cert
                nonlocal sample_total_J_cert, sample_total_L_cert
                nonlocal sample_total_uncert, sample_total_survivors
                nonlocal sample_total_time_NO, sample_total_time_J
                nonlocal sample_total_time_L, n_parents_completed
                nonlocal slow_parents_skipped
                if wall_p > slow_thr:
                    slow_parents_skipped += 1
                n_surv_i = int(len(survivors))
                # process_parent_v4 always populates these keys → direct access.
                n_NO = int(counts['n_certified_NO'])
                n_J  = int(counts['n_certified_J'])
                n_L  = int(counts['n_certified_L'])
                n_un = int(counts['n_uncertified'])
                t_NO = float(counts['time_NO'])
                t_J  = float(counts['time_J'])
                t_L  = float(counts['time_L'])
                n_t  = int(n_tested)

                sample_total_children   += n_t
                sample_total_NO_cert    += n_NO
                sample_total_J_cert     += n_J
                sample_total_L_cert     += n_L
                sample_total_uncert     += n_un
                sample_total_survivors  += n_surv_i
                sample_total_time_NO    += t_NO
                sample_total_time_J     += t_J
                sample_total_time_L     += t_L
                if n_surv_i > 0:
                    sample_survivors_compositions.append(survivors)
                n_parents_completed += 1

                per_parent.append({
                    'idx': int(i), 'children': n_t,
                    'NO_cert': n_NO, 'J_cert': n_J, 'L_cert': n_L,
                    'uncert': n_un, 'survivors': n_surv_i,
                    'wall_sec': round(wall_p, 3),
                    'time_NO': round(t_NO, 3),
                    'time_J':  round(t_J, 3),
                    'time_L':  round(t_L, 3),
                })
                ch = max(1, n_t)
                pct_NO_p = 100.0 * n_NO / ch
                pct_J_p  = 100.0 * n_J / ch
                pct_L_p  = 100.0 * n_L / ch
                pct_surv = 100.0 * n_surv_i / ch
                print(f"    parent {i+1:>2}/{n_sample}: children={n_t:>8,}  "
                      f"NO={n_NO:>6,} ({pct_NO_p:5.1f}%, {t_NO:5.1f}s)  "
                      f"+J={n_J:>5,} ({pct_J_p:4.1f}%, {t_J:4.1f}s)  "
                      f"+L={n_L:>4,} ({pct_L_p:4.1f}%, {t_L:4.1f}s)  "
                      f"-> surv={n_surv_i:>5,} ({pct_surv:5.2f}%)  wall={wall_p:.1f}s",
                      flush=True)

            # ---- Parallel parent processing (when pool available). ----
            if pool is not None and n_sample > 1:
                # Submit all parents up-front; the worker measures its own
                # wall time so per-parent prints aren't biased by queue wait.
                fut_to_idx = {}
                for i in range(n_sample):
                    parent_arr = np.ascontiguousarray(sample[i])
                    args = (parent_arr, d_child, S, c_target, ord_d, kw)
                    fut = pool.submit(_worker_process_parent, args)
                    fut_to_idx[fut] = i
                try:
                    for fut in as_completed(fut_to_idx):
                        elapsed_so_far = time.time() - t_level
                        if elapsed_so_far > level_time_sec:
                            budget_exceeded = True
                            cancelled = 0
                            for f in fut_to_idx:
                                if not f.done() and f.cancel():
                                    cancelled += 1
                            print(f"    [time budget exhausted after "
                                  f"{n_parents_completed}/{n_sample} parents "
                                  f"@ {elapsed_so_far:.0f}s; cancelled "
                                  f"{cancelled} pending]", flush=True)
                            break
                        i = fut_to_idx[fut]
                        try:
                            survivors, n_tested, counts, wall_p = fut.result()
                        except Exception as e:
                            print(f"    parent {i} EXC: {e}", flush=True)
                            continue
                        _record(i, wall_p, survivors, n_tested, counts)
                finally:
                    fut_to_idx.clear()
            else:
                # Sequential path (n_workers == 1 or single sample).
                for i in range(n_sample):
                    elapsed_so_far = time.time() - t_level
                    if elapsed_so_far > level_time_sec:
                        budget_exceeded = True
                        print(f"    [time budget exhausted after {i}/{n_sample} "
                              f"parents @ {elapsed_so_far:.0f}s]", flush=True)
                        break
                    tp = time.time()
                    try:
                        survivors, n_tested, counts = process_parent_v4(
                            sample[i], d_child, S, c_target, ord_d, **kw)
                    except Exception as e:
                        print(f"    parent {i} EXC: {e}", flush=True)
                        continue
                    _record(i, time.time() - tp, survivors, n_tested, counts)

            wall_level = time.time() - t_level
            if n_parents_completed == 0:
                out['levels'].append({
                    'level': L, 'd_child': d_child,
                    'n_parents_in_pool': n_curr,
                    'sample_size': n_sample,
                    'n_parents_completed': 0,
                    'verdict_partial': 'NO_PROGRESS',
                    'wall_sec': round(wall_level, 2),
                })
                out['verdict'] = f'STUCK_AT_L{L}'
                break

            inv_n = 1.0 / n_parents_completed
            avg_children    = sample_total_children   * inv_n
            avg_NO          = sample_total_NO_cert    * inv_n
            avg_J           = sample_total_J_cert     * inv_n
            avg_L           = sample_total_L_cert     * inv_n
            avg_uncert      = sample_total_uncert     * inv_n
            avg_survivors   = sample_total_survivors  * inv_n

            est_total_children   = avg_children * n_curr
            est_total_NO         = int(round(avg_NO * n_curr))
            est_total_J          = int(round(avg_J  * n_curr))
            est_total_L          = int(round(avg_L  * n_curr))
            est_total_uncert     = int(round(avg_uncert * n_curr))
            est_total_survivors  = int(round(avg_survivors * n_curr))

            # Prev-level avg_survivors for α (convergence rate).
            prev_avg_surv = None
            for prev_lv in out['levels']:
                if prev_lv.get('level') == L - 1 and 'avg_survivors_per_parent' in prev_lv:
                    prev_avg_surv = prev_lv['avg_survivors_per_parent']
                    break

            inv_avg_ch = 1.0 / max(1.0, avg_children)
            print(f"\n  ----- L{L} LEVEL SUMMARY -----", flush=True)
            print(f"    sample size                  : {n_parents_completed} parents (of {n_sample} drawn, {sample_method})", flush=True)
            print(f"    EXPANSION FACTOR             : {avg_children:,.1f} children/parent", flush=True)
            print(f"    avg NO certs / parent        : {avg_NO:,.1f}  ({100.0*avg_NO*inv_avg_ch:.1f}% of children)", flush=True)
            print(f"    avg Joint certs / parent     : {avg_J:.2f}    ({100.0*avg_J*inv_avg_ch:.2f}% of children)", flush=True)
            print(f"    avg Shor certs / parent      : {avg_L:.2f}    ({100.0*avg_L*inv_avg_ch:.2f}% of children)", flush=True)
            print(f"    avg uncertified / parent     : {avg_uncert:.2f}", flush=True)
            print(f"    avg SURVIVORS / parent       : {avg_survivors:.2f}    "
                  f"(SURVIVAL RATE = {100.0*avg_survivors*inv_avg_ch:.4f}%)", flush=True)
            if prev_avg_surv is not None and prev_avg_surv > 0:
                alpha = avg_survivors / prev_avg_surv
                print(f"    α = surv_curr / surv_prev    : {avg_survivors:.2f} / {prev_avg_surv:.2f} = {alpha:.3f}",
                      flush=True)
            print(f"    EXTRAPOLATED to {n_curr:,} parents:", flush=True)
            print(f"      est total children         : {est_total_children:,.2e}", flush=True)
            print(f"      est total survivors        : {est_total_survivors:,}", flush=True)
            print(f"      cum children so far        : {cum_children + est_total_children:.2e}", flush=True)
            print(f"    Wall time                    : {wall_level:.1f}s "
                  f"(NO={sample_total_time_NO:.1f}s J={sample_total_time_J:.1f}s "
                  f"L={sample_total_time_L:.1f}s; per-parent avg={wall_level/max(1,n_parents_completed):.1f}s)",
                  flush=True)
            if slow_parents_skipped > 0:
                print(f"    [{slow_parents_skipped} slow parents flagged "
                      f"(wall > {slow_thr:.0f}s)]", flush=True)

            cum_children += est_total_children
            out['levels'].append({
                'level': L, 'd_child': d_child,
                'n_parents_in_pool': int(n_curr),
                'sample_size': int(n_sample),
                'n_parents_completed': int(n_parents_completed),
                'avg_children_per_parent': float(avg_children),
                'avg_NO_cert_per_parent': float(avg_NO),
                'avg_J_cert_per_parent':  float(avg_J),
                'avg_L_cert_per_parent':  float(avg_L),
                'avg_uncert_per_parent':  float(avg_uncert),
                'avg_survivors_per_parent': float(avg_survivors),
                'est_total_children': float(est_total_children),
                'est_total_NO_cert': int(est_total_NO),
                'est_total_J_cert':  int(est_total_J),
                'est_total_L_cert':  int(est_total_L),
                'est_total_uncert':  int(est_total_uncert),
                'est_total_survivors': int(est_total_survivors),
                'cum_children_so_far': float(cum_children),
                'wall_sec': round(wall_level, 2),
                'wall_per_parent_sec': round(wall_level
                                              / max(1, n_parents_completed), 3),
                'time_NO_total': round(sample_total_time_NO, 2),
                'time_J_total':  round(sample_total_time_J, 2),
                'time_L_total':  round(sample_total_time_L, 2),
                'time_budget_hit': bool(budget_exceeded),
                'slow_parents_skipped': int(slow_parents_skipped),
                'per_parent': per_parent,
            })

            # ---- Termination logic ----
            if avg_survivors == 0:
                out['verdict'] = f'CLOSED_AT_L{L}_SAMPLE'
                print(f"  >>> VERDICT: CLOSED at L{L} (sample shows 0 survivors)",
                      flush=True)
                break
            if est_total_survivors > MAX_SURV_HARD:
                out['verdict'] = (f'IMPOSSIBLE_AT_L{L}: '
                                  f'est_total_survivors={est_total_survivors:.2e} > '
                                  f'MAX={MAX_SURV_HARD:.0e}')
                print(f"  >>> VERDICT: IMPOSSIBLE — est_survivors exceeds {MAX_SURV_HARD:.0e}",
                      flush=True)
                break
            if est_total_children > compute_budget:
                out['verdict'] = (f'IMPOSSIBLE_AT_L{L}: '
                                  f'est_total_children={est_total_children:.2e} > '
                                  f'budget={compute_budget:.2e}')
                print(f"  >>> VERDICT: IMPOSSIBLE — est_children exceeds budget",
                      flush=True)
                break
            if cum_children > compute_budget:
                out['verdict'] = (f'IMPOSSIBLE_CUMULATIVE_L{L}: '
                                  f'cum={cum_children:.2e} > '
                                  f'budget={compute_budget:.2e}')
                print(f"  >>> VERDICT: IMPOSSIBLE_CUMULATIVE — total children exceeds budget",
                      flush=True)
                break
            if L >= 2:
                prior_level = next((lev for lev in out['levels']
                                    if lev['level'] == L - 1
                                    and 'avg_survivors_per_parent' in lev),
                                    None)
                if prior_level is not None:
                    A_prev = prior_level['avg_survivors_per_parent']
                    A_curr = avg_survivors
                    if A_prev > 1 and A_curr > 0:
                        alpha = A_curr / A_prev
                        out['levels'][-1]['alpha_vs_prev'] = float(alpha)
                        if (alpha > ALPHA_NONCONVERGE
                                and A_curr > SURV_NONTRIVIAL):
                            out['verdict'] = (
                                f'NONCONVERGENT_AT_L{L}: '
                                f'alpha={alpha:.2f} > {ALPHA_NONCONVERGE} '
                                f'with avg_survivors={A_curr:.0f} > '
                                f'{SURV_NONTRIVIAL}')
                            break

            # Build pool for next level: combine sampled parents' actual survivors.
            if sample_survivors_compositions:
                current_survivors = (sample_survivors_compositions[0]
                                     if len(sample_survivors_compositions) == 1
                                     else np.vstack(sample_survivors_compositions))
            else:
                out['verdict'] = f'CLOSED_AT_L{L}_SAMPLE'
                break
            n_curr = est_total_survivors
            d_parent = d_child

        if 'verdict' not in out:
            out['verdict'] = f'OUT_OF_LEVELS_AT_L{max_levels}'
            print(f"  >>> VERDICT: OUT_OF_LEVELS at L{max_levels} "
                  f"(cascade did not converge in {max_levels} levels; "
                  f"increase --max_levels)", flush=True)
        out['cum_children_estimated'] = cum_children
        out['total_wall_sec'] = round(time.time() - t0, 2)

        # ---- Per-config closing summary ----
        print(f"\n  ==== CONFIG SUMMARY (d0={d0}, S={S}, c={c_target}) ====",
              flush=True)
        print(f"  verdict              : {out['verdict']}", flush=True)
        print(f"  total wall time      : {out['total_wall_sec']:.1f}s", flush=True)
        print(f"  cum children est.    : {cum_children:.2e}", flush=True)
        print(f"  levels reached       : {len(out['levels'])}", flush=True)
        for lv in out['levels']:
            L = lv.get('level')
            if L == 0:
                print(f"    L0 (d={lv.get('d_child')}): "
                      f"NO={lv.get('n_certified_NO',0):,} "
                      f"J={lv.get('n_certified_J',0):,} "
                      f"L={lv.get('n_certified_L',0):,} "
                      f"surv={lv.get('n_survivors',0):,} "
                      f"({lv.get('wall_sec',0):.1f}s)", flush=True)
            else:
                print(f"    L{L} (d={lv.get('d_child')}): "
                      f"avg_children={lv.get('avg_children_per_parent', 0):,.0f}  "
                      f"avg_NO={lv.get('avg_NO_cert_per_parent', 0):,.0f}  "
                      f"avg_J={lv.get('avg_J_cert_per_parent', 0):.1f}  "
                      f"avg_L={lv.get('avg_L_cert_per_parent', 0):.2f}  "
                      f"avg_surv={lv.get('avg_survivors_per_parent', 0):.2f}  "
                      f"est_total_surv={lv.get('est_total_survivors', 0):,}  "
                      f"({lv.get('wall_sec', 0):.1f}s)", flush=True)
        return out
    finally:
        if own_pool and pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--sample_n', type=int, default=DEFAULT_SAMPLE_N)
    ap.add_argument('--level_time_sec', type=float,
                     default=DEFAULT_LEVEL_TIME)
    ap.add_argument('--compute_budget', type=float,
                     default=DEFAULT_COMPUTE_BUDGET)
    ap.add_argument('--l0_timeout', type=float, default=DEFAULT_L0_TIMEOUT)
    ap.add_argument('--max_levels', type=int, default=DEFAULT_MAX_LEVELS)
    ap.add_argument('--use_joint', action='store_true', default=True)
    ap.add_argument('--no_joint', dest='use_joint', action='store_false')
    ap.add_argument('--use_sdp', action='store_true', default=True)
    ap.add_argument('--no_sdp', dest='use_sdp', action='store_false')
    ap.add_argument('--joint_top_K', type=int, default=4)
    ap.add_argument('--joint_iters', type=int, default=20)
    ap.add_argument('--sdp_mode', default='best_only',
                     choices=['best_only', 'max'])
    ap.add_argument('--n_workers', type=int, default=N_WORKERS_DEFAULT)
    ap.add_argument('--out_dir', type=str, default=None)
    ap.add_argument('--configs', type=str, default=None,
                     help='JSON list of [d0, S] pairs.  '
                          'Default: [(2,30),(2,60),(2,100),(4,20),(4,30)].')
    args = ap.parse_args()

    if args.configs:
        configs = json.loads(args.configs)
    else:
        configs = [
            (2, 30),    # tiny smoke
            (2, 60),    # small
            (2, 100),   # medium
            (4, 20),    # mid d
            (4, 30),    # mid d, more cells
        ]

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(ROOT,
                                             f'coarse_cascade_estimate_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'summary.json')
    summary = {
        'started_utc': ts,
        'c_target': args.c_target,
        'sample_n': args.sample_n,
        'level_time_sec': args.level_time_sec,
        'compute_budget': args.compute_budget,
        'use_joint': args.use_joint,
        'use_sdp': args.use_sdp,
        'joint_top_K': args.joint_top_K,
        'sdp_mode': args.sdp_mode,
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

    # Spawn the worker pool ONCE and reuse across all configs.  Workers
    # pre-import the v4 stack and warm Numba caches at startup; per-config
    # pool re-creation would cost ~5s each on Windows-spawn.
    pool = _maybe_make_pool(args.n_workers)
    if pool is not None:
        print(f"Spawned worker pool with {args.n_workers} processes "
              f"(persistent across configs)", flush=True)

    try:
        for cfg in configs:
            d0, S = cfg[0], cfg[1]
            c = cfg[2] if len(cfg) > 2 else args.c_target
            print(f"\n{'='*70}\n"
                  f"CONFIG: d0={d0}, S={S}, c_target={c}\n"
                  f"{'='*70}", flush=True)
            try:
                r = estimate_one(d0, S, c, max_levels=args.max_levels,
                                  sample_n=args.sample_n,
                                  level_time_sec=args.level_time_sec,
                                  compute_budget=args.compute_budget,
                                  l0_timeout=args.l0_timeout,
                                  use_joint=args.use_joint,
                                  use_sdp=args.use_sdp,
                                  joint_top_K=args.joint_top_K,
                                  joint_iters=args.joint_iters,
                                  sdp_mode=args.sdp_mode,
                                  n_workers=args.n_workers,
                                  pool=pool)
            except Exception as e:
                r = {'d0': d0, 'S': S, 'c_target': c,
                     'verdict': f'EXC: {e}'}
            summary['results'].append(r)
            save()
            cum = r.get('cum_children_estimated')
            cum_str = f'{cum:.2e}' if cum else 'n/a'
            print(f"\n  >>> verdict: {r.get('verdict')}  "
                  f"cum_children_est={cum_str}\n", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    save()
    print("\nDONE.", flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == '__main__':
    main()
