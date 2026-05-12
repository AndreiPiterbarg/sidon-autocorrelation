"""Rigorous coarse cascade v5 — uncert cells become extra parents at L+1.

==================================================================
SOUNDNESS ARGUMENT  (refinement monotonicity, CS Theorem 1)
==================================================================

The coarse cascade discretizes the configuration polytope
        P_d = { μ : Σ μ_i = S,  μ_i ≥ 0,  i ∈ [d] }
into cells of the form
        Cell(c) = { μ : μ ∈ c + (1/(2S)) · [-1,1]^d,  Σ (μ-c) = 0 }
indexed by integer points c ∈ Z^d (the "grid points").  The cascade
chains a sequence of refinements d_L = 2^L · d_0:  each cell at level
L is partitioned into 2^{d_L} children at level L+1 by halving every
bin (parent bin c_i splits into c_{2i}, c_{2i+1} = c_{2i,L+1} +
c_{2i+1,L+1}).  Crucially every point in the parent cell is contained
in EXACTLY ONE child cell — the children TILE the parent.

Refinement monotonicity:  if  ∀ child c' of c,  TV(μ) ≥ c_target
                          ∀ μ ∈ Cell(c'),
then  TV(μ) ≥ c_target  ∀ μ ∈ Cell(c).
                                                        (*)
PROOF.  ∀μ ∈ Cell(c) ∃ unique child c' with μ ∈ Cell(c'); the
hypothesis gives TV(μ) ≥ c_target.  □

The standard cascade prunes a cell iff its box-cert closes
(N+O+Joint+Shor → cert_box(c) ≥ c_target).  If any of those layers
closes, the cell is verified across its WHOLE width, and the cell
contributes nothing to descent.

==================================================================
WHAT BREAKS RIGOR IN v5/v4
==================================================================

The flag-2 "uncertified" path: at level L there are cells where
TV at the GRID POINT exceeds c_target (so triangle-style pruning
kicks in) but the box-cert chain (N+O+Joint+Shor) does not close.
The standard cascade DROPS these cells (treats them as pruned).  This
is UNSOUND because we have not verified TV ≥ c_target across the cell
width — only at the grid point.

==================================================================
THE FIX: ESCALATE UNCERTIFIED CELLS TO L+1
==================================================================

Instead of dropping flag-2 cells, FORWARD them as "extra parents" to
level L+1 alongside the regular survivors (flag-0 cells with
TV at grid < c_target).

At level L+1 the descendants of an uncert flag-2 cell tile its
geometric region.  By (*), if every descendant of that uncert cell
gets cert_box-closed at L+1 (or recursively passed down to L+2 ...
and ultimately closed at some finite depth), the parent cell IS
covered.

Termination criterion:  the cascade is rigorous iff at SOME level
L_final, every level ≤ L_final has either (a) flag-2 = 0 (all hard
cells fully closed) OR (b) every flag-2 cell at L is a parent of cells
at L+1 that ALL terminate in cert.  In practice, we wait until the
recursion bottoms out: total survivors = 0 AND total uncert = 0.

==================================================================
IMPLEMENTATION
==================================================================

1.  apply_v4_layers already returns uncertified_idx — we extract those
    cells from the batch.
2.  Each level forwards two streams:  survivors (flag-0) and
    extra_parents (flag-2).  Both descend via process_parent at L+1.
3.  Track per-level counts:  closed_at_level[L] (cumulative cells
    cert_box closed), still_in_recursion[L] (uncert + survivors
    descending).
4.  Cascade rigorously closes when current = ∅ AND extra = ∅ at some
    level.

Soundness invariant maintained: at every level L, the union of
descendants of all (closed flag-1) ∪ (forwarded flag-0 ∪ flag-2) cells
covers the original L0 configuration polytope.

==================================================================
LIMITATIONS
==================================================================

This trades compute for rigor.  If the box-cert tightness gap is
fundamental (e.g., S too small to close any cell) extra-parents
multiply geometrically and the cascade explodes.  We add a budget
limit (max_total_cells) and report incomplete recursion.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import warnings
import contextlib

import numpy as np

# --- Filter cvxpy / ortools noise on import ---
_orig_stderr = sys.stderr
sys.stderr = io.StringIO()
warnings.filterwarnings('ignore')
os.environ.setdefault('CVXPY_LOG_LEVEL', 'CRITICAL')

# --- Path setup ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
_CS_DIR = os.path.join(_ROOT, 'cloninger-steinerberger')
_CS_CPU = os.path.join(_CS_DIR, 'cpu')
for _p in (_ROOT, _CS_DIR, _CS_CPU):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Imports (v4 stack) ---
from compositions import (generate_compositions_batched,
                          generate_canonical_compositions_batched)
from pruning import count_compositions, _canonical_mask
from run_cascade_coarse_v2 import (asymmetry_prune_mask_coarse, coarse_x_cap)
from run_cascade_coarse_v3 import precompute_op_rest_d
import run_cascade_coarse_v4 as _v4

# Restore stderr after imports
sys.stderr = _orig_stderr


def _log(msg):
    print(msg, flush=True)


# =====================================================================
# apply_v4_layers wrapper that surfaces the uncert cell INTEGERS.
# =====================================================================

def _apply_v4_with_uncert(batch_int, d, S, c_target, op_rest_d_arr,
                           use_joint=True, use_sdp=True,
                           joint_top_K=4, joint_iters=20,
                           sdp_mode='best_only'):
    """v4.apply_v4_layers but returns uncertified composition arrays.

    Returns dict with the standard v4 fields plus:
      'uncert_compositions':  np.ndarray (n_uncert, d) of uncert int cells.
    """
    # Silence stdout/stderr from cvxpy during cell certs
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        out = _v4.apply_v4_layers(
            batch_int, d, S, c_target, op_rest_d_arr,
            use_joint=use_joint, use_sdp=use_sdp,
            joint_top_K=joint_top_K, joint_iters=joint_iters,
            sdp_mode=sdp_mode, verbose=False)
    uncert_idx = out.get('uncertified_idx', np.empty(0, dtype=np.int64))
    if uncert_idx.size > 0:
        uncert_comp = batch_int[uncert_idx]
    else:
        uncert_comp = np.empty((0, d), dtype=batch_int.dtype)
    out['uncert_compositions'] = uncert_comp
    return out


# =====================================================================
# Custom L0 with uncert capture.
# =====================================================================

def _run_level0_rigorous(d0, S, c_target, op_rest_d_arr,
                          use_joint=True, use_sdp=True,
                          joint_top_K=4, joint_iters=20, sdp_mode='best_only',
                          verbose=True):
    """L0 enumeration that ALSO captures uncert cells (so they descend)."""
    n_total = count_compositions(d0, S)
    if verbose:
        _log(f"\n[L0] d={d0}, S={S}, compositions={n_total:,}, "
             f"x_cap={coarse_x_cap(d0, S, c_target)}")

    t0 = time.time()
    survivors_list = []
    uncert_list = []
    n_NO = n_J = n_L = 0
    n_uncert_total = 0

    for batch in generate_canonical_compositions_batched(d0, S, batch_size=500_000):
        asym_mask = asymmetry_prune_mask_coarse(batch, S, c_target)
        batch = batch[asym_mask]
        if len(batch) == 0:
            continue
        out = _apply_v4_with_uncert(batch, d0, S, c_target, op_rest_d_arr,
                                      use_joint=use_joint, use_sdp=use_sdp,
                                      joint_top_K=joint_top_K,
                                      joint_iters=joint_iters,
                                      sdp_mode=sdp_mode)
        n_NO += out['n_certified_NO']
        n_J += out['n_certified_J']
        n_L += out['n_certified_L']
        n_uncert_total += out['n_uncertified']
        if len(out['survivors']) > 0:
            survivors_list.append(out['survivors'])
        if len(out['uncert_compositions']) > 0:
            uncert_list.append(out['uncert_compositions'])
    elapsed = time.time() - t0

    survivors = (np.vstack(survivors_list) if survivors_list
                 else np.empty((0, d0), dtype=np.int32))
    uncerts = (np.vstack(uncert_list) if uncert_list
               else np.empty((0, d0), dtype=np.int32))

    if verbose:
        _log(f"     {elapsed:.2f}s: NO={n_NO:,} J={n_J:,} L={n_L:,} "
             f"surv={len(survivors):,}  uncert={len(uncerts):,}")
    return {
        'survivors': survivors,
        'uncerts': uncerts,
        'n_certified_NO': n_NO, 'n_certified_J': n_J, 'n_certified_L': n_L,
        'n_uncertified': n_uncert_total, 'elapsed': elapsed,
    }


# =====================================================================
# Custom process_parent that returns uncert cells too.
# =====================================================================

def _process_parent_rigorous(parent_int, d_child, S, c_target, op_rest_d_arr,
                               use_joint=True, use_sdp=True,
                               joint_top_K=4, joint_iters=20,
                               sdp_mode='best_only'):
    """v4.process_parent_v4 that also returns uncert child cells."""
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
            return (np.empty((0, d_child), dtype=np.int32),
                    np.empty((0, d_child), dtype=np.int32),
                    0,
                    {'NO': 0, 'J': 0, 'L': 0, 'uncert': 0})
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_product *= (hi - lo + 1)
    if total_product == 0:
        return (np.empty((0, d_child), dtype=np.int32),
                np.empty((0, d_child), dtype=np.int32),
                0,
                {'NO': 0, 'J': 0, 'L': 0, 'uncert': 0})

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

    out = _apply_v4_with_uncert(children, d_child, S, c_target, op_rest_d_arr,
                                  use_joint=use_joint, use_sdp=use_sdp,
                                  joint_top_K=joint_top_K,
                                  joint_iters=joint_iters,
                                  sdp_mode=sdp_mode)
    counts = {
        'NO': out['n_certified_NO'],
        'J':  out['n_certified_J'],
        'L':  out['n_certified_L'],
        'uncert': out['n_uncertified'],
    }
    return out['survivors'], out['uncert_compositions'], n, counts


# =====================================================================
# Main rigor cascade
# =====================================================================

def run_rigorous_cascade(d0, S, c_target,
                         max_levels=8, use_joint=True, use_sdp=True,
                         max_total_cells=2_000_000,
                         verbose=True):
    """Run the rigorous cascade with uncert escalation.

    Soundness:  by refinement monotonicity, every cell forwarded from
    L to L+1 (whether flag-0 survivor or flag-2 uncert) is tiled by
    its 2^d_parent children at L+1.  Cascade closes rigorously iff
    at some level L_final, the set of cells forwarded to L_final+1
    is empty AND no cell at any level was dropped uncertified.

    Args:
      d0, S, c_target: cascade params.
      max_levels: max refinement steps after L0.
      use_joint, use_sdp: enable Joint dual / Shor SDP layers.
      max_total_cells: budget; abort if cumulative L cells exceed.
    """
    info = {
        'd0': d0, 'S': S, 'c_target': c_target,
        'max_levels': max_levels,
        'use_joint': use_joint, 'use_sdp': use_sdp,
        'levels': [],
        'rigorous_closure': False,
        'reason_unclosed': None,
    }
    t_total = time.time()
    op_cache = {d0: precompute_op_rest_d(d0)}

    # ---- L0 ----
    if verbose:
        _log(f"\n=== RIGOR CASCADE  d0={d0}  S={S}  c={c_target}  "
             f"J={use_joint}  L={use_sdp} ===")
    l0 = _run_level0_rigorous(d0, S, c_target, op_cache[d0],
                                use_joint=use_joint, use_sdp=use_sdp,
                                verbose=verbose)
    info['l0'] = {
        'survivors': len(l0['survivors']),
        'uncerts': len(l0['uncerts']),
        'NO': l0['n_certified_NO'], 'J': l0['n_certified_J'],
        'L': l0['n_certified_L'],
        'time': round(l0['elapsed'], 2),
    }
    if verbose:
        _log(f"[L0] surv={len(l0['survivors']):,}  uncert={len(l0['uncerts']):,}  "
             f"({l0['elapsed']:.2f}s)")

    if len(l0['survivors']) == 0 and len(l0['uncerts']) == 0:
        info['rigorous_closure'] = True
        info['proven_at'] = 'L0'
        info['total_time'] = time.time() - t_total
        if verbose:
            _log(f"[L0] *** RIGOROUSLY PROVEN at L0 *** "
                 f"({_v4.cell_cert_shor.__name__ if use_sdp else 'no SDP'})")
        return info

    # Forward both survivors and uncerts as parents at L+1.
    parents = np.vstack([l0['survivors'], l0['uncerts']]) \
              if (len(l0['survivors']) + len(l0['uncerts']) > 0) \
              else np.empty((0, d0), dtype=np.int32)
    d_parent = d0
    cumul_cells = len(parents)

    # ---- L >= 1 ----
    for L in range(1, max_levels + 1):
        if len(parents) == 0:
            info['rigorous_closure'] = True
            info['proven_at'] = f'L{L-1}'
            break
        d_child = 2 * d_parent
        if d_child not in op_cache:
            op_cache[d_child] = precompute_op_rest_d(d_child)
        x_cap = coarse_x_cap(d_child, S, c_target)
        feasible = np.all(parents <= 2 * x_cap, axis=1)
        if not feasible.all():
            parents = np.ascontiguousarray(parents[feasible])
        n_parents = len(parents)
        if verbose:
            _log(f"\n--- L{L}: d={d_parent}->{d_child}  parents={n_parents:,}  "
                 f"x_cap={x_cap} ---")
        if n_parents == 0:
            info['rigorous_closure'] = True
            info['proven_at'] = f'L{L}'
            break

        t_lvl = time.time()
        survivors_list = []
        uncert_list = []
        total_children = 0
        n_NO = n_J = n_L = n_uncert = 0
        last_report = time.time()
        aborted_mid_level = False
        for i in range(n_parents):
            surv, unc, n_t, c = _process_parent_rigorous(
                parents[i], d_child, S, c_target, op_cache[d_child],
                use_joint=use_joint, use_sdp=use_sdp)
            total_children += n_t
            if len(surv) > 0:
                survivors_list.append(surv)
            if len(unc) > 0:
                uncert_list.append(unc)
            n_NO += c['NO']; n_J += c['J']; n_L += c['L']
            n_uncert += c['uncert']
            # Mid-level budget check (avoid runaway L2 explosions)
            running_total = sum(len(s) for s in survivors_list) + \
                            sum(len(u) for u in uncert_list)
            if running_total > max_total_cells:
                if verbose:
                    _log(f"   *** mid-level budget exceeded "
                         f"({running_total:,} > {max_total_cells:,}); "
                         f"aborting at parent {i+1}/{n_parents} ***")
                aborted_mid_level = True
                break
            if verbose and (time.time() - last_report > 8.0):
                ns = sum(len(s) for s in survivors_list)
                nu = sum(len(u) for u in uncert_list)
                _log(f"   [{i+1}/{n_parents}] children={total_children:,} "
                     f"NO={n_NO:,} J={n_J:,} L={n_L:,} "
                     f"surv={ns:,} uncert={nu:,}")
                last_report = time.time()
        wall = time.time() - t_lvl

        survs = (np.vstack(survivors_list) if survivors_list
                 else np.empty((0, d_child), dtype=np.int32))
        uncs = (np.vstack(uncert_list) if uncert_list
                else np.empty((0, d_child), dtype=np.int32))
        # Canonicalise (no dedup of uncert vs survivors — both descend).
        if len(survs) > 0:
            cmask = _canonical_mask(survs)
            survs[~cmask] = survs[~cmask, ::-1]
            survs = np.unique(survs, axis=0)
        if len(uncs) > 0:
            cmask = _canonical_mask(uncs)
            uncs[~cmask] = uncs[~cmask, ::-1]
            uncs = np.unique(uncs, axis=0)

        info['levels'].append({
            'level': L, 'd_child': d_child,
            'parents': n_parents, 'children': int(total_children),
            'NO': int(n_NO), 'J': int(n_J), 'L': int(n_L),
            'uncert': int(n_uncert),
            'survivors_out': len(survs),
            'uncerts_out': len(uncs),
            'wall_sec': round(wall, 2),
        })
        if verbose:
            _log(f"L{L}: {total_children:,} children -> "
                 f"NO={n_NO:,} J={n_J:,} L={n_L:,} uncert={n_uncert:,} "
                 f"surv_next={len(survs):,} unc_next={len(uncs):,} "
                 f"({wall:.1f}s)")

        # Forward both streams; both descend.
        parents = (np.vstack([survs, uncs]) if (len(survs) + len(uncs) > 0)
                   else np.empty((0, d_child), dtype=np.int32))
        cumul_cells += len(parents)
        d_parent = d_child

        if aborted_mid_level or cumul_cells > max_total_cells:
            info['reason_unclosed'] = (
                f'budget exceeded at L{L}: cumul {cumul_cells:,} > '
                f'{max_total_cells:,}'
                + (' (mid-level abort)' if aborted_mid_level else ''))
            if verbose:
                _log(f"   *** budget exceeded; aborting ***")
            break

        if len(parents) == 0:
            info['rigorous_closure'] = True
            info['proven_at'] = f'L{L}'
            break

    info['total_time'] = round(time.time() - t_total, 2)
    info['final_parents'] = len(parents)
    if not info['rigorous_closure'] and info['reason_unclosed'] is None:
        info['reason_unclosed'] = (
            f'max_levels={max_levels} reached with {len(parents):,} parents')

    if verbose:
        _log(f"\n=== DONE ({info['total_time']:.1f}s) ===")
        if info['rigorous_closure']:
            _log(f"*** RIGOROUS PROOF: C_{{1a}} >= {c_target} *** "
                 f"(closed at {info.get('proven_at', '?')})")
        else:
            _log(f"NOT closed: {info['reason_unclosed']}")
    return info


# =====================================================================
# CLI / test driver
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d0', type=int, default=2)
    ap.add_argument('--S', type=int, default=60)
    ap.add_argument('--c_target', type=float, default=1.281)
    ap.add_argument('--max_levels', type=int, default=4)
    ap.add_argument('--no_joint', action='store_true')
    ap.add_argument('--no_sdp', action='store_true')
    ap.add_argument('--max_total_cells', type=int, default=500_000)
    ap.add_argument('--out_json', type=str, default=None)
    args = ap.parse_args()

    info = run_rigorous_cascade(
        d0=args.d0, S=args.S, c_target=args.c_target,
        max_levels=args.max_levels,
        use_joint=not args.no_joint,
        use_sdp=not args.no_sdp,
        max_total_cells=args.max_total_cells,
        verbose=True)
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(info, f, indent=2, default=lambda x:
                       int(x) if isinstance(x, np.integer) else
                       float(x) if isinstance(x, np.floating) else
                       x.tolist() if isinstance(x, np.ndarray) else x)
    return info


if __name__ == '__main__':
    main()
