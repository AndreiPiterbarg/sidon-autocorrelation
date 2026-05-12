"""Coarse-grid cascade prover v5 — optimal infrastructure + tightest sound box cert.

This is the production coarse-grid cascade.  It combines:

  1. v4's box-cert layers (N+O kernel, Joint dual K=4, Shor SDP fallback) —
     each provably sound, OR-aggregated for tighter cell certificates than
     v2/v3.  See run_cascade_coarse_v4.py for the layer math.

  2. The optimized Gray-code coarse fused kernel `_fused_coarse_gray` from
     `run_cascade.py` — Gray-code child generation, incremental conv updates,
     sparse cross-terms (d≥32), subtree pruning (J_MIN=7), L1-resident
     staging buffer, optimized ell scan order, asymmetry pre-filter.

  3. Shared-memory ProcessPoolExecutor for parents — copies the proven
     `_init_worker_coarse` infrastructure from run_cascade.py.  Each worker
     pre-imports the v4 stack and pins Numba to 1 thread (no oversubscription).

  4. Checkpointing across levels — save+resume infrastructure copied from
     run_cascade.py.

  5. cgroup-aware CPU detection so RunPod / Docker container limits are
     respected.

  6. JIT warmup before workers fork to avoid 64x duplicate compile cost.

Soundness chain:  v5 ⊆ v4 ⊆ v3 ⊆ v2.  No regression.

Two execution modes:

  --mode tight  (default): v4 N+O+J+L per cell — strictest sound box-cert,
                  ~ms-to-100ms per child depending on which layer carries.
                  Best for medium-sized configs (d≤10, S≤30).

  --mode fast   : optimized Gray-code kernel with v2 triangle bound —
                  20-50× faster than tight mode at L1+, but produces a
                  triangle-only box cert.  Use when N+O headroom is small
                  or for L0 enumeration where children counts are huge.

Usage:
    python -m cloninger-steinerberger.cpu.run_cascade_coarse_v5 \\
        --d0 2 --S 60 --c_target 1.25 \\
        --mode tight --use_joint --use_sdp \\
        --max_levels 8 --n_workers 64 \\
        --output_dir runs/coarse_d2_S60_c1.25
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numba

# --- Path setup ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CS_DIR = os.path.dirname(_THIS_DIR)
_ROOT = os.path.dirname(_CS_DIR)
for _p in (_THIS_DIR, _CS_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compositions import (generate_compositions_batched,
                          generate_canonical_compositions_batched)
from pruning import count_compositions, _canonical_mask

# v3/v4 kernel pieces
from run_cascade_coarse_v2 import (asymmetry_prune_mask_coarse,
                                     coarse_x_cap, _build_pair_prefix)
from run_cascade_coarse_v3 import (precompute_op_rest_d, _one_sided_lp,
                                     _quad_corr_v3, _prune_no_correction_v3,
                                     _fused_coarse_v3)
# Optimized Gray-code kernel from fine-grid run_cascade.py (triangle bound).
from run_cascade import (_fused_coarse_gray, _compute_bin_ranges_coarse,
                          _default_buf_cap, _effective_cpu_count, _warmup_jit,
                          _save_checkpoint, _load_checkpoint, _fmt_time,
                          _log)
# v4 layer driver (Joint + Shor)
import run_cascade_coarse_v4 as _v4
joint_dual_filter = _v4.joint_dual_filter
shor_sdp_filter = _v4.shor_sdp_filter

# Minimal kernel (L3+ no-box-cert path) — strips per-pruned-cell sort+LP+spectral
from coarse_minimal_kernel import process_parent_minimal


# =====================================================================
# Per-parent processors (tight = v4 N+O+J+L; fast = Gray-code triangle)
# =====================================================================

def process_parent_tight(parent_int, S, c_target, d_child, op_rest_d_arr,
                           use_joint=True, use_sdp=False,
                           joint_top_K=4, joint_iters=20,
                           sdp_mode='best_only'):
    """v4 path: simple cursor enumeration + N+O kernel, then Joint + Shor.

    Returns (survivors, n_tested, counts) — same signature as v4.
    """
    return _v4.process_parent_v4(parent_int, d_child, S, c_target,
                                   op_rest_d_arr,
                                   use_joint=use_joint, use_sdp=use_sdp,
                                   joint_top_K=joint_top_K,
                                   joint_iters=joint_iters,
                                   sdp_mode=sdp_mode)


def process_parent_fast(parent_int, S, c_target, d_child, op_rest_d_arr=None,
                          use_joint=False, use_sdp=False, **kwargs):
    """Optimized Gray-code path: triangle bound, fast.

    Box cert is the triangle bound only (looser than N+O).  Still sound.
    Returns (survivors, n_tested, counts) — counts has only NO_cert ≈ pruned;
    J_cert and L_cert always 0 in fast mode.
    """
    d_parent = len(parent_int)
    rng = _compute_bin_ranges_coarse(parent_int, S, c_target, d_child)
    if rng is None:
        return (np.empty((0, d_child), dtype=np.int32), 0,
                {'n_certified_NO': 0, 'n_certified_J': 0,
                 'n_certified_L': 0, 'n_uncertified': 0,
                 'time_NO': 0.0, 'time_J': 0.0, 'time_L': 0.0,
                 'min_cert_net': 1e30})
    lo_arr, hi_arr, total_children = rng
    if total_children == 0:
        return (np.empty((0, d_child), dtype=np.int32), 0,
                {'n_certified_NO': 0, 'n_certified_J': 0,
                 'n_certified_L': 0, 'n_uncertified': 0,
                 'time_NO': 0.0, 'time_J': 0.0, 'time_L': 0.0,
                 'min_cert_net': 1e30})

    prefix_nk, prefix_mk = _build_pair_prefix(d_child)
    buf_cap = _default_buf_cap(d_child)
    max_buf = min(total_children, buf_cap)
    out_buf = np.empty((max_buf, d_child), dtype=np.int32)

    t0 = time.time()
    n_surv, _, min_net = _fused_coarse_gray(
        parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf,
        prefix_nk, prefix_mk)
    if n_surv > max_buf:
        max_buf = n_surv
        out_buf = np.empty((max_buf, d_child), dtype=np.int32)
        n2, _, min_net = _fused_coarse_gray(
            parent_int, d_child, S, c_target, lo_arr, hi_arr, out_buf,
            prefix_nk, prefix_mk)
        n_surv = n2
    t_kernel = time.time() - t0

    survivors = out_buf[:n_surv].copy()
    counts = {
        'n_certified_NO': int(total_children - n_surv),  # triangle prunes
        'n_certified_J': 0, 'n_certified_L': 0,
        'n_uncertified': 0,
        'time_NO': float(t_kernel), 'time_J': 0.0, 'time_L': 0.0,
        'min_cert_net': float(min_net),
    }
    return survivors, total_children, counts


# =====================================================================
# Shared-memory worker pool (parents memory-mapped, no per-task pickling)
# =====================================================================

# Globals set by initializer
_g_shared_parents = None
_g_S = None
_g_c_target = None
_g_d_child = None
_g_op_rest_d = None
_g_mode = 'tight'
_g_use_joint = True
_g_use_sdp = False
_g_joint_top_K = 4
_g_joint_iters = 20
_g_sdp_mode = 'best_only'


def _worker_init_v5(S, c_target, d_child, op_rest_d_arr,
                      mode, use_joint, use_sdp, joint_top_K, joint_iters,
                      sdp_mode, numba_threads):
    """Pool initializer: pin Numba threads, install per-worker globals,
    pre-import v3/v4 stack so njit cache is hot."""
    numba.set_num_threads(int(numba_threads))
    global _g_S, _g_c_target, _g_d_child, _g_op_rest_d
    global _g_mode, _g_use_joint, _g_use_sdp, _g_joint_top_K, _g_joint_iters
    global _g_sdp_mode
    _g_S = S
    _g_c_target = c_target
    _g_d_child = d_child
    _g_op_rest_d = op_rest_d_arr
    _g_mode = mode
    _g_use_joint = use_joint
    _g_use_sdp = use_sdp
    _g_joint_top_K = joint_top_K
    _g_joint_iters = joint_iters
    _g_sdp_mode = sdp_mode
    # Pre-import to hot-load njit caches (one-shot per worker).
    import run_cascade_coarse_v3  # noqa: F401
    import run_cascade_coarse_v4  # noqa: F401


def _worker_process_parent_v5(parent_arr):
    """Worker: process one parent (passed as numpy array, not index)."""
    if _g_mode == 'tight':
        survivors, n_tested, counts = process_parent_tight(
            parent_arr, _g_S, _g_c_target, _g_d_child, _g_op_rest_d,
            use_joint=_g_use_joint, use_sdp=_g_use_sdp,
            joint_top_K=_g_joint_top_K, joint_iters=_g_joint_iters,
            sdp_mode=_g_sdp_mode)
    elif _g_mode == 'minimal':
        survivors, n_tested, counts = process_parent_minimal(
            parent_arr, _g_S, _g_c_target, _g_d_child)
    else:
        survivors, n_tested, counts = process_parent_fast(
            parent_arr, _g_S, _g_c_target, _g_d_child, _g_op_rest_d)
    n_surv = len(survivors)
    return (survivors if n_surv > 0 else None), {
        'children': int(n_tested),
        'survived': n_surv,
        'NO': int(counts.get('n_certified_NO', 0)),
        'J':  int(counts.get('n_certified_J', 0)),
        'L':  int(counts.get('n_certified_L', 0)),
        'uncert': int(counts.get('n_uncertified', n_surv)),
        'time_NO': float(counts.get('time_NO', 0.0)),
        'time_J':  float(counts.get('time_J', 0.0)),
        'time_L':  float(counts.get('time_L', 0.0)),
        'min_cert_net': float(counts.get('min_cert_net', 1e30)),
    }


# =====================================================================
# Level 0 — full canonical enumeration with all v4 layers
# =====================================================================

def run_level0_v5(d0, S, c_target, op_rest_d_arr,
                   use_joint=True, use_sdp=False,
                   joint_top_K=4, joint_iters=20, sdp_mode='best_only',
                   verbose=True):
    """L0: enumerate all canonical compositions, apply NO+J+L progressively.

    Reuses run_cascade_coarse_v4.run_level0 (already optimal for L0).
    """
    return _v4.run_level0(d0, S, c_target, op_rest_d_arr,
                            use_joint=use_joint, use_sdp=use_sdp,
                            joint_top_K=joint_top_K, joint_iters=joint_iters,
                            sdp_mode=sdp_mode, verbose=verbose)


# =====================================================================
# Cascade runner
# =====================================================================

def run_cascade(d0, S, c_target,
                max_levels=8, n_workers=None, numba_threads=1,
                mode='adaptive', use_joint=True, use_sdp=False,
                joint_top_K=4, joint_iters=20, sdp_mode='best_only',
                tight_max_level=2,
                output_dir=None, resume_dir=None, checkpoint_every=True,
                verbose=True):
    """Full coarse cascade with optimal infrastructure.

    Args:
      d0, S, c_target: cascade params.  S is constant across levels.
      max_levels: stop after this many child-doublings (default 8).
      n_workers: parents-in-flight in the worker pool.  Defaults to
                 cgroup-aware effective CPU count.
      numba_threads: per-worker numba thread count (default 1; bigger
                 oversubscribes when n_workers also large).
      mode: 'tight' (v4 N+O+J+L) or 'fast' (Gray-code triangle only).
      use_joint, use_sdp: enable layers in tight mode.
      joint_top_K, joint_iters, sdp_mode: pass-through to v4 layers.
      output_dir: directory for checkpoints (None = disabled).
      resume_dir: directory to resume from (None = no resume).
      checkpoint_every: save after every level (else only on final close).

    Returns dict with per-level stats and final verdict.
    """
    if n_workers is None:
        n_workers = _effective_cpu_count()
    n_workers = max(1, int(n_workers))

    if verbose:
        _log(f"\n{'='*72}")
        _log(f"COARSE CASCADE PROVER v5 (optimal infrastructure + adaptive box-cert)")
        _log(f"  d0={d0}, S={S}, c_target={c_target}")
        _log(f"  Mode: {mode}")
        if mode == 'adaptive':
            _log(f"    All levels (RIGOROUS): N+O kernel + Joint dual K={joint_top_K}"
                 f"{' + Shor SDP' if use_sdp else ''}  (every level box-cert tracked)")
        else:
            _log(f"    All levels: {mode}")
        _log(f"  Workers: {n_workers} processes × {numba_threads} numba threads")
        _log(f"  Output dir: {output_dir or '(no checkpoints)'}")
        _log(f"  Resume dir: {resume_dir or '(none)'}")
        _log(f"{'='*72}")

    # ---- JIT warmup (compile-once, shared across forks) ----
    if verbose:
        _log(f"\n[warmup] pre-compiling njit kernels...")
    t_warm = time.time()
    try:
        _warmup_jit()
    except Exception:
        pass  # warmup is best-effort
    # Also warm v3/v4 kernels
    try:
        warm_d = max(2, d0)
        op_warm = precompute_op_rest_d(warm_d)
        warm_batch = np.zeros((1, warm_d), dtype=np.int32)
        warm_batch[0, 0] = max(2, S)
        _ = _prune_no_correction_v3(warm_batch, warm_d, max(2, S),
                                     c_target, op_warm)
    except Exception:
        pass
    if verbose:
        _log(f"[warmup] done in {time.time()-t_warm:.1f}s")

    info = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'mode': mode, 'use_joint': use_joint, 'use_sdp': use_sdp,
        'n_workers': n_workers, 'numba_threads': numba_threads,
        'max_levels': max_levels,
        'levels': [],
    }
    t_total = time.time()

    # ---- Resume support ----
    current = None
    start_level = 0
    if resume_dir:
        # Map d0/S/c to (n_half, m, c_target) signature — but coarse uses
        # (d0, S, c).  Use a coarse-specific meta key.
        ck = _try_load_checkpoint_coarse(resume_dir, d0, S, c_target)
        if ck is not None:
            current, start_level, prev_info = ck
            info['levels'] = prev_info.get('levels', [])
            if verbose:
                _log(f"[resume] starting at L{start_level+1} with "
                     f"{len(current):,} survivors from checkpoint")

    # ---- L0 (skip if resume) ----
    op_rest_d_cache = {d0: precompute_op_rest_d(d0)}

    if current is None:
        if verbose:
            _log(f"\n--- L0: enumerating canonical compositions ---")
        t0 = time.time()
        l0 = run_level0_v5(d0, S, c_target, op_rest_d_cache[d0],
                            use_joint=use_joint, use_sdp=use_sdp,
                            joint_top_K=joint_top_K, joint_iters=joint_iters,
                            sdp_mode=sdp_mode, verbose=verbose)
        l0_wall = time.time() - t0
        info['l0'] = {
            'd_child': d0,
            'survivors': int(l0['n_survivors']),
            'NO_cert': int(l0.get('n_certified_NO', 0)),
            'J_cert':  int(l0.get('n_certified_J', 0)),
            'L_cert':  int(l0.get('n_certified_L', 0)),
            'uncertified': int(l0.get('n_uncertified', 0)),
            'time': round(l0_wall, 2),
            'box_certified': bool(l0.get('box_certified', False)),
        }
        if l0.get('proven', False):
            info['proven_at'] = 'L0'
            info['box_certified'] = info['l0']['box_certified']
            info['total_time'] = time.time() - t_total
            if output_dir and checkpoint_every:
                _save_checkpoint_coarse(output_dir, 0,
                                          np.empty((0, d0), dtype=np.int32),
                                          {'d0': d0, 'S': S,
                                           'c_target': c_target,
                                           'level_completed': 0,
                                           'info': info})
            if verbose:
                _log(f"\n*** PROVEN AT L0 ***  box_cert={info['box_certified']}")
            return info
        current = l0['survivors']
        if output_dir and checkpoint_every:
            _save_checkpoint_coarse(output_dir, 0, current,
                                      {'d0': d0, 'S': S, 'c_target': c_target,
                                       'level_completed': 0, 'info': info})
        d_parent = d0
        start_level = 0
    else:
        d_parent = current.shape[1]

    # ---- L1+ ----
    for L in range(start_level + 1, max_levels + 1):
        if len(current) == 0:
            break
        d_child = 2 * d_parent
        n_parents = len(current)

        # --- Adaptive mode selection per level ---
        # In adaptive mode (RIGOROUS): every level uses the full N+O+J+L
        # box-cert chain.  tight_max_level is now ignored for the bound
        # itself; we only use it to optimize the J+L parameters at deep
        # levels.  Skipping box-cert math would break rigor — soundness
        # requires box-cert to PASS at every level for the cascade to be
        # a rigorous proof of C_{1a} >= c_target across all μ.
        if mode == 'adaptive':
            level_mode = 'tight'
            # All layers on at every level for full rigor.
            level_use_joint = use_joint
            level_use_sdp   = use_sdp
        else:
            level_mode = mode
            level_use_joint = use_joint and (level_mode == 'tight')
            level_use_sdp   = use_sdp   and (level_mode == 'tight')

        # op_rest_d only needed for tight mode (pure-fast skips entirely)
        if level_mode == 'tight':
            if d_child not in op_rest_d_cache:
                t_pre = time.time()
                op_rest_d_cache[d_child] = precompute_op_rest_d(d_child)
                if verbose:
                    _log(f"  op_rest_d for d={d_child}: {time.time()-t_pre:.2f}s")
            ord_d = op_rest_d_cache[d_child]
        else:
            ord_d = op_rest_d_cache.get(d_child)
        x_cap = coarse_x_cap(d_child, S, c_target)

        if verbose:
            _log(f"\n--- L{L}: d={d_parent} -> {d_child}, "
                 f"parents={n_parents:,}, x_cap={x_cap}, mode={level_mode} ---")

        # Pre-filter infeasible parents
        feasible = np.all(current <= 2 * x_cap, axis=1)
        n_infeasible = int((~feasible).sum())
        if n_infeasible > 0:
            current = np.ascontiguousarray(current[feasible])
            n_parents = len(current)
            if verbose:
                _log(f"  pre-filtered {n_infeasible:,} infeasible parents")
        if n_parents == 0:
            current = np.empty((0, d_child), dtype=np.int32)
            info['levels'].append({
                'level': L, 'd_child': d_child,
                'parents': 0, 'children': 0, 'survivors': 0,
            })
            d_parent = d_child
            continue

        t_level = time.time()
        all_survivors = []
        total_children = 0
        n_NO = n_J = n_L = n_uncert = 0
        time_NO = time_J = time_L = 0.0
        level_min_net = 1e30
        n_done = 0
        last_report = time.time()

        # ---- Worker pool path ----
        if n_workers > 1 and n_parents > 4:
            with ProcessPoolExecutor(
                    max_workers=int(n_workers),
                    mp_context=mp.get_context('spawn'),
                    initializer=_worker_init_v5,
                    initargs=(S, c_target, d_child, ord_d,
                              level_mode, level_use_joint, level_use_sdp,
                              joint_top_K, joint_iters, sdp_mode,
                              numba_threads)) as ex:
                futures = [ex.submit(_worker_process_parent_v5, current[i])
                           for i in range(n_parents)]
                for fut in as_completed(futures):
                    surv, stats = fut.result()
                    if surv is not None:
                        all_survivors.append(surv)
                    total_children += stats['children']
                    n_NO += stats['NO']; n_J += stats['J']; n_L += stats['L']
                    n_uncert += stats['uncert']
                    time_NO += stats['time_NO']
                    time_J += stats['time_J']
                    time_L += stats['time_L']
                    if stats['min_cert_net'] < level_min_net:
                        level_min_net = stats['min_cert_net']
                    n_done += 1
                    now = time.time()
                    if verbose and (now - last_report >= 5.0):
                        ns = sum(len(s) for s in all_survivors)
                        _log(f"  [{n_done}/{n_parents}] "
                             f"children={total_children:,} "
                             f"+NO={n_NO:,} +J={n_J:,} +L={n_L:,} "
                             f"surv={ns:,}  ({_fmt_time(now-t_level)})")
                        last_report = now
        else:
            # Sequential fallback
            for i in range(n_parents):
                if level_mode == 'tight':
                    surv, n_t, counts = process_parent_tight(
                        current[i], S, c_target, d_child, ord_d,
                        use_joint=level_use_joint, use_sdp=level_use_sdp,
                        joint_top_K=joint_top_K, joint_iters=joint_iters,
                        sdp_mode=sdp_mode)
                else:
                    surv, n_t, counts = process_parent_fast(
                        current[i], S, c_target, d_child, ord_d)
                total_children += int(n_t)
                if len(surv) > 0:
                    all_survivors.append(surv)
                n_NO += int(counts.get('n_certified_NO', 0))
                n_J += int(counts.get('n_certified_J', 0))
                n_L += int(counts.get('n_certified_L', 0))
                n_uncert += int(counts.get('n_uncertified', 0))
                time_NO += float(counts.get('time_NO', 0.0))
                time_J += float(counts.get('time_J', 0.0))
                time_L += float(counts.get('time_L', 0.0))
                mn = float(counts.get('min_cert_net', 1e30))
                if mn < level_min_net:
                    level_min_net = mn
                n_done += 1
                now = time.time()
                if verbose and (now - last_report >= 5.0):
                    ns = sum(len(s) for s in all_survivors)
                    _log(f"  [{n_done}/{n_parents}] children={total_children:,} "
                         f"+NO={n_NO:,} +J={n_J:,} +L={n_L:,} surv={ns:,}")
                    last_report = now

        wall_level = time.time() - t_level

        # Combine + dedup canonical survivors
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
        # RIGOR check: any pruned cell where N+O+Joint+Shor all failed is
        # an "uncertified" cell — it was pruned by TV-at-grid but box-cert
        # could not verify the prune is sound across the cell.  If
        # n_uncert > 0, this level's box-cert FAILS and the cascade is
        # not a rigorous proof of C_{1a} >= c_target.
        rigor_pass = (n_uncert == 0)
        info['levels'].append({
            'level': L, 'd_child': d_child,
            'parents': n_parents, 'children': int(total_children),
            'survivors': n_survivors, 'rate': rate,
            'NO_cert': n_NO, 'J_cert': n_J, 'L_cert': n_L,
            'uncertified': n_uncert,
            'time_NO_sec': round(time_NO, 2),
            'time_J_sec':  round(time_J, 2),
            'time_L_sec':  round(time_L, 2),
            'wall_sec': round(wall_level, 2),
            'min_cert_net': float(level_min_net),
            'rigor_pass': rigor_pass,
            'box_certified': rigor_pass,
        })

        if verbose:
            _log(f"  L{L} done in {_fmt_time(wall_level)}: "
                 f"{total_children:,} children -> {n_survivors:,} survivors "
                 f"({rate:.4f}%)")
            _log(f"     +NO={n_NO:,} (kernel)  +J={n_J:,}  +L={n_L:,}  "
                 f"uncert={n_uncert:,}")
            _log(f"     time breakdown: NO={time_NO:.1f}s J={time_J:.1f}s "
                 f"L={time_L:.1f}s")
            if rigor_pass:
                _log(f"     *** RIGOR L{L}: PASS  (all pruned cells box-cert verified) ***")
            else:
                _log(f"")
                _log(f"     *** RIGOR L{L}: FAIL  ({n_uncert:,} cells pruned without box-cert proof) ***")
                _log(f"     *** This breaks the rigorous chain — cascade is NOT a proof of C_{{1a}} >= {c_target} ***")
                _log(f"     *** Consider: increase S to shrink corrections, or stop and revise. ***")
                _log(f"")

        if output_dir and checkpoint_every:
            _save_checkpoint_coarse(output_dir, L, current,
                                      {'d0': d0, 'S': S, 'c_target': c_target,
                                       'level_completed': L, 'info': info})
        if n_survivors == 0:
            info['proven_at'] = f'L{L}'
            info['total_time'] = time.time() - t_total
            all_cert = info.get('l0', {}).get('box_certified', True)
            for lv in info['levels']:
                if not lv.get('box_certified', False):
                    all_cert = False
                    break
            info['box_certified'] = bool(all_cert)
            if verbose:
                _log(f"\n*** GRID-POINT PROOF COMPLETE at L{L}! ***")
                _log(f"*** Box certification: {'PASS' if all_cert else 'INCOMPLETE'} ***")
                _log(f"*** Total time: {_fmt_time(info['total_time'])} ***")
            return info
        d_parent = d_child

    info['total_time'] = time.time() - t_total
    info['final_survivors'] = len(current)
    if verbose:
        _log(f"\nCascade did not converge in {max_levels} levels "
             f"({len(current):,} survivors at d={d_parent}).")
    return info


# =====================================================================
# Coarse-specific checkpointing (variant of _save_checkpoint that uses
# (d0, S, c) tuple instead of (n_half, m, c)).
# =====================================================================

def _save_checkpoint_coarse(output_dir, level, survivors, meta):
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir,
                            f'checkpoint_L{level}_survivors_coarse.npy')
    meta_path = os.path.join(output_dir, 'checkpoint_meta_coarse.json')
    np.save(npy_path, survivors)
    def _conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return x
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=_conv)
    _log(f"  [checkpoint] L{level}: {len(survivors):,} survivors -> {npy_path}")


def _try_load_checkpoint_coarse(resume_dir, d0, S, c_target):
    meta_path = os.path.join(resume_dir, 'checkpoint_meta_coarse.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if (meta.get('d0') != d0 or meta.get('S') != S
            or abs(meta.get('c_target', -1) - c_target) > 1e-9):
        _log(f"  [resume] checkpoint params mismatch; ignoring")
        return None
    level = meta.get('level_completed', 0)
    npy_path = os.path.join(resume_dir,
                            f'checkpoint_L{level}_survivors_coarse.npy')
    if not os.path.exists(npy_path):
        _log(f"  [resume] checkpoint meta found but {npy_path} missing")
        return None
    survivors = np.load(npy_path)
    return survivors, level, meta.get('info', {})


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Coarse-grid cascade prover v5 (optimal v4 + Gray-code + workers + checkpoint)')
    ap.add_argument('--d0', type=int, default=2)
    ap.add_argument('--S', type=int, default=50)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--max_levels', type=int, default=8)
    ap.add_argument('--n_workers', type=int, default=None)
    ap.add_argument('--numba_threads', type=int, default=1)
    ap.add_argument('--mode', choices=['tight', 'fast', 'adaptive'],
                     default='adaptive',
                     help='tight: v4 N+O+J+L all levels; fast: Gray-code '
                          'triangle all levels; adaptive (default): tight '
                          'for L<=tight_max_level, fast above (max pruning '
                          'cheap, minimum necessary expensive)')
    ap.add_argument('--tight_max_level', type=int, default=2,
                     help='In adaptive mode, last level using tight bound '
                          '(default 2 = use tight at L0/L1/L2, fast at L3+)')
    ap.add_argument('--use_joint', action='store_true', default=True)
    ap.add_argument('--no_joint', dest='use_joint', action='store_false')
    ap.add_argument('--use_sdp', action='store_true', default=False)
    ap.add_argument('--no_sdp', dest='use_sdp', action='store_false')
    ap.add_argument('--joint_top_K', type=int, default=4)
    ap.add_argument('--joint_iters', type=int, default=20)
    ap.add_argument('--sdp_mode', choices=['best_only', 'max'],
                     default='best_only')
    ap.add_argument('--output_dir', type=str, default=None,
                     help='checkpoint dir (None = no checkpointing)')
    ap.add_argument('--resume_dir', type=str, default=None)
    ap.add_argument('--no_checkpoint', dest='checkpoint_every',
                     action='store_false', default=True)
    args = ap.parse_args()

    result = run_cascade(
        d0=args.d0, S=args.S, c_target=args.c_target,
        max_levels=args.max_levels, n_workers=args.n_workers,
        numba_threads=args.numba_threads,
        mode=args.mode, use_joint=args.use_joint, use_sdp=args.use_sdp,
        joint_top_K=args.joint_top_K, joint_iters=args.joint_iters,
        sdp_mode=args.sdp_mode,
        tight_max_level=args.tight_max_level,
        output_dir=args.output_dir, resume_dir=args.resume_dir,
        checkpoint_every=args.checkpoint_every)

    if 'proven_at' in result:
        if result.get('box_certified'):
            _log(f"\nRIGOROUS PROOF v5: C_{{1a}} >= {args.c_target}")
        else:
            _log(f"\nGRID-POINT PROOF: TV >= {args.c_target} "
                 f"(box cert incomplete; increase S or use --use_sdp)")
    else:
        _log(f"\nNOT PROVEN (cascade did not converge in {args.max_levels} levels)")


if __name__ == '__main__':
    main()
