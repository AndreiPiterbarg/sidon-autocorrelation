"""LP-only deep-split cascade with K=0 SDP fallback, smart axis split, and dedup.

Optimized for 360-core pod with 1.4 TB RAM.

KEY OPTIMIZATIONS
=================
1. LP-shadow-price axis selection: split each parent on the axes whose
   McCormick-face dual marginals (|λ| × width) are largest, NOT just
   widest. Cuts iterations to convergence.
2. Canonical-hash dedup (parallel-safe):
   - Main-process `seen_certified_hashes` set: skip any child whose
     hash matches a previously-certified box (across all iters).
   - Within a single iter's split_pool: skip duplicate child hashes
     produced by different parents.
   - Dedup happens BEFORE dispatch to workers → workers never see
     duplicates → parallel-safe by construction.
3. K=0 SDP fallback on LP-fails (insurance against demon boxes).

ARCHITECTURE
============
Per iteration:
    1. SPLIT every survivor box (smart axes via cached ineqlin, fall
       back to widest if no ineqlin)
    2. Hash-dedup children (within-iter + against seen_certified)
    3. LP-filter (360-way parallel, 1 thread/proc); workers return
       (cert, lp_val, ineqlin) for LP-fails
    4. K=0 SDP fallback on LP-fails (workers parallel)
    5. Save SDP-fails as next iter's survivors (carry their LP ineqlin)
    6. Update seen_certified_hashes with this iter's cert hashes

Stops when survivors empty / max-iters / global mult > --max-mult past iter 3.

CLI
---
    python3 -u _deep_split_cascade.py \
        --input runs/d22_pod_iter5_higherK/iter_001/children_after_lp.npz \
        --d 22 --target 1.2805 \
        --split-depth 4 --max-iters 14 \
        --workers-lp 360 --workers-sdp 90 \
        --output-dir runs/d22_deep_split \
        --max-mult 6.0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cert_pipeline.k_ladder import (
    SurvivorBox, survivors_from_npz, survivors_to_npz,
)
from cert_pipeline.kill_survivors import MathInsufficient
from cert_pipeline.box_journal import canonical_box_hash


# =====================================================================
# Worker globals (set by initializers in each fork)
# =====================================================================

_LP_WINDOWS = None
_LP_D = None
_SDP_CACHE = None
_SDP_WINDOWS = None
_SDP_D = None


def _lp_init(d: int):
    """LP worker init: build windows once."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(_HERE))
    global _LP_WINDOWS, _LP_D
    from interval_bnb.windows import build_windows
    _LP_D = d
    _LP_WINDOWS = build_windows(d)


def _lp_one(args: Tuple[Tuple[int, ...], Tuple[int, ...], float]) -> Tuple[bool, float, Optional[bytes]]:
    """LP one box. Returns (cert, lp_val, ineqlin_bytes_or_None).

    For LP-CERTED boxes: returns (True, lp_val, None) — no ineqlin needed.
    For LP-FAILED boxes: returns (False, lp_val, ineqlin_bytes) — first
       4*d*d float64 entries of ineqlin packed as raw bytes (~15 KB at
       d=22; pickle is fast for raw bytes).
    """
    lo_int, hi_int, target = args
    from interval_bnb.bound_epigraph import bound_epigraph_int_ge_with_marginals
    from interval_bnb.box import SCALE as _SCALE
    lo = np.asarray([float(x) / _SCALE for x in lo_int], dtype=np.float64)
    hi = np.asarray([float(x) / _SCALE for x in hi_int], dtype=np.float64)
    try:
        cert, lp_val, ineqlin = bound_epigraph_int_ge_with_marginals(
            lo, hi, _LP_WINDOWS, _LP_D, target
        )
    except Exception:
        return (False, float('nan'), None)
    if cert:
        return (True, float(lp_val), None)
    if ineqlin is None:
        return (False, float(lp_val), None)
    n_y = _LP_D * _LP_D
    truncated = np.asarray(ineqlin[:4 * n_y], dtype=np.float64)
    return (False, float(lp_val), truncated.tobytes())


def _sdp_init(d: int, target: float):
    """K=0 SDP worker init: build cache once per worker."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(_HERE))
    global _SDP_CACHE, _SDP_WINDOWS, _SDP_D
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
    )
    _SDP_D = d
    _SDP_WINDOWS = build_windows(d)
    _SDP_CACHE = build_sdp_escalation_cache_fast(d, _SDP_WINDOWS, target=target)


def _sdp_one(args: Tuple[Tuple[int, ...], Tuple[int, ...], int, int]) -> bool:
    """K=0 SDP cert. True iff certified."""
    lo_int, hi_int, target_num, target_den = args
    from interval_bnb.bound_sdp_escalation_fast import (
        bound_sdp_escalation_int_ge_fast,
    )
    try:
        cert = bound_sdp_escalation_int_ge_fast(
            list(lo_int), list(hi_int), _SDP_WINDOWS, _SDP_D,
            target_num=target_num, target_den=target_den,
            cache=_SDP_CACHE,
            n_window_psd_cones=0,  # K=0
            n_threads=1,
            time_limit_s=60.0,
            early_stop=True,
        )
        return bool(cert)
    except Exception:
        return False


# =====================================================================
# Splitting (smart axis selection via LP shadow prices)
# =====================================================================

def _split_one_box_smart(
    lo_int: List[int], hi_int: List[int],
    target_depth: int,
    ineqlin: Optional[np.ndarray],
    d: int,
) -> List[Tuple[List[int], List[int]]]:
    """Bisect along `target_depth` chosen axes (each once → 2^target_depth children).

    Axis priority:
      • If ineqlin is provided: rank axes by McCormick-face binding mass × width
        (lp_binding_axis_score). This focuses splits on axes that actually shrink
        the LP gap.
      • Else fall back to widest-axis ranking.

    Only axes with int width >= 2 are splittable (dyadic-2^60 grid).
    Raises MathInsufficient if NO axis is splittable (single grid point).
    """
    from interval_bnb.bound_epigraph import lp_binding_axis_score

    if target_depth <= 0:
        return [(list(lo_int), list(hi_int))]

    widths = np.asarray([hi - lo for hi, lo in zip(hi_int, lo_int)], dtype=np.float64)
    splittable_axes = [i for i in range(d) if (hi_int[i] - lo_int[i]) >= 2]
    if not splittable_axes:
        raise MathInsufficient(
            f'box is a single dyadic grid point. lo_int[0..3]={lo_int[:3]}'
        )

    if ineqlin is not None and len(ineqlin) >= 4 * d * d:
        scores = lp_binding_axis_score(ineqlin, widths.astype(np.float64), d)
        # Restrict to splittable axes; rank descending by score.
        score_pairs = [(scores[i], i) for i in splittable_axes]
        score_pairs.sort(key=lambda x: -x[0])
        pick = [i for (_, i) in score_pairs[:target_depth]]
    else:
        # Widest-axis fallback (same as kill_survivors.split_box_to_depth).
        widths_pairs = sorted(
            ((widths[i], i) for i in splittable_axes), reverse=True
        )
        pick = [i for (_, i) in widths_pairs[:target_depth]]

    children = [(list(lo_int), list(hi_int))]
    for axis in pick:
        new = []
        for (clo, chi) in children:
            mid = (clo[axis] + chi[axis]) // 2
            if mid <= clo[axis] or mid >= chi[axis]:
                new.append((clo, chi))
                continue
            left_lo, left_hi = list(clo), list(chi)
            left_hi[axis] = mid
            right_lo, right_hi = list(clo), list(chi)
            right_lo[axis] = mid
            new.append((left_lo, left_hi))
            new.append((right_lo, right_hi))
        children = new
    return children


# =====================================================================
# Pool dispatch with dedup
# =====================================================================

def split_and_dedup(
    parents: List[SurvivorBox],
    ineqlin_by_hash: Dict[str, np.ndarray],
    split_depth: int,
    seen_certified: set,
    sim_scale: int,
    d: int,
) -> Tuple[List[SurvivorBox], Dict[str, int]]:
    """Split each parent → produce children.

    Dedup pipeline (parallel-safe; main thread only):
      • Drop simplex-empty children.
      • Drop hashes already in seen_certified (already proven).
      • Drop within-iter duplicate hashes (different parents producing
        the same child box).

    Returns (unique_children, stats_dict).
    """
    out: List[SurvivorBox] = []
    seen_this_iter: set = set()
    stats = {
        'free_children': 0,
        'simplex_clipped': 0,
        'cert_already': 0,
        'within_iter_dup': 0,
        'kept': 0,
        'math_insufficient': 0,
    }
    free_mult = 2 ** split_depth
    for s in parents:
        ineq = ineqlin_by_hash.get(s.hash)
        try:
            kids = _split_one_box_smart(
                s.lo_int, s.hi_int, split_depth, ineq, d
            )
        except MathInsufficient as e:
            stats['math_insufficient'] += 1
            continue
        next_iters = s.iters_survived + 1
        next_depth = s.depth + split_depth
        for (clo, chi) in kids:
            stats['free_children'] += 1
            if sum(clo) > sim_scale or sum(chi) < sim_scale:
                stats['simplex_clipped'] += 1
                continue
            chash = canonical_box_hash(clo, chi)
            if chash in seen_certified:
                stats['cert_already'] += 1
                continue
            if chash in seen_this_iter:
                stats['within_iter_dup'] += 1
                continue
            seen_this_iter.add(chash)
            vol = float(np.prod([(h - l) / sim_scale for l, h in zip(clo, chi)]))
            out.append(SurvivorBox(
                hash=chash, lo_int=clo, hi_int=chi,
                depth=next_depth, volume=vol, lp_val=None,
                src=f'split({s.hash})', iters_survived=next_iters,
            ))
            stats['kept'] += 1
    return out, stats


# =====================================================================
# Main driver
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='survivors npz')
    ap.add_argument('--d', type=int, default=22)
    ap.add_argument('--target', type=float, default=1.2805)
    ap.add_argument('--target-num', type=int, default=12805)
    ap.add_argument('--target-den', type=int, default=10000)
    ap.add_argument('--split-depth', type=int, default=4)
    ap.add_argument('--max-iters', type=int, default=15)
    ap.add_argument('--workers-lp', type=int, default=360)
    ap.add_argument('--workers-sdp', type=int, default=90)
    ap.add_argument('--no-sdp', action='store_true',
                    help='skip K=0 SDP fallback (LP-only cascade)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-mult', type=float, default=6.0)
    ap.add_argument('--lp-chunksize', type=int, default=64)
    ap.add_argument('--sdp-chunksize', type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / 'journal.jsonl'

    print(f'[init] loading {args.input}', flush=True)
    pool = survivors_from_npz(args.input)
    if not pool:
        print('[init] EMPTY input — already converged.', flush=True)
        return
    print(f'[init] loaded {len(pool)} boxes (depth range '
          f'{min(s.depth for s in pool)}..{max(s.depth for s in pool)})',
          flush=True)
    print(f'[init] target={args.target} d={args.d} split-depth={args.split_depth}',
          flush=True)
    print(f'[init] workers: LP={args.workers_lp}  K=0 SDP={args.workers_sdp}',
          flush=True)

    from interval_bnb.box import SCALE as _SCALE

    print(f'[init] spawning LP pool ({args.workers_lp} workers)...', flush=True)
    t0 = time.time()
    ctx = mp.get_context('fork')
    lp_pool = ctx.Pool(args.workers_lp, initializer=_lp_init,
                       initargs=(args.d,))
    print(f'[init] LP pool ready in {time.time()-t0:.1f}s', flush=True)

    sdp_pool = None
    if not args.no_sdp:
        print(f'[init] spawning K=0 SDP pool ({args.workers_sdp} workers, '
              f'each builds cache ~1-2s)...', flush=True)
        t0 = time.time()
        sdp_pool = ctx.Pool(args.workers_sdp, initializer=_sdp_init,
                            initargs=(args.d, args.target))
        print(f'[init] K=0 SDP pool ready in {time.time()-t0:.1f}s', flush=True)
    else:
        print('[init] K=0 SDP fallback DISABLED (--no-sdp)', flush=True)

    journal = open(journal_path, 'a')

    # Cross-iter state
    seen_certified: set = set()
    ineqlin_by_hash: Dict[str, np.ndarray] = {}  # only for current pool

    n_y = args.d * args.d
    ineqlin_n = 4 * n_y  # bytes per box = ineqlin_n * 8

    for level in range(1, args.max_iters + 1):
        if not pool:
            print(f'\n[iter {level}] pool empty — CONVERGED at iter {level-1}.',
                  flush=True)
            break
        n_parents = len(pool)
        depth_in = min(s.depth for s in pool)
        depth_out = depth_in + args.split_depth

        print(f'\n{"="*78}', flush=True)
        print(f'[iter {level}] n_parents={n_parents}  depth {depth_in} '
              f'-> {depth_out}  seen_cert_hashes={len(seen_certified)}',
              flush=True)
        print(f'{"="*78}', flush=True)

        # ----- 1. SPLIT + DEDUP -----
        t0 = time.time()
        children, sstats = split_and_dedup(
            pool, ineqlin_by_hash, args.split_depth,
            seen_certified, _SCALE, args.d,
        )
        t_split = time.time() - t0
        n_children = len(children)
        free_mult = 2 ** args.split_depth
        print(f'  [split+dedup] {n_parents} parents x {free_mult} = '
              f'{sstats["free_children"]} raw children -> {n_children} kept',
              flush=True)
        print(f'    dropped: simplex_clip={sstats["simplex_clipped"]} '
              f'already_cert={sstats["cert_already"]} '
              f'within_iter_dup={sstats["within_iter_dup"]} '
              f'math_insufficient_parents={sstats["math_insufficient"]}',
              flush=True)
        print(f'    elapsed {t_split:.1f}s', flush=True)

        # We've consumed the parent ineqlins. Free the dict for next iter.
        ineqlin_by_hash = {}

        # ----- 2. LP FILTER (parallel, returns ineqlin for fails) -----
        t0 = time.time()
        lp_args = [(tuple(c.lo_int), tuple(c.hi_int), args.target)
                   for c in children]
        lp_fails: List[SurvivorBox] = []
        lp_fail_ineq: Dict[str, np.ndarray] = {}
        n_lp_cert = 0
        n_done = 0
        next_progress = max(1, n_children // 20)
        for (cert, lp_val, ineq_bytes), c in zip(
                lp_pool.imap(_lp_one, lp_args, chunksize=args.lp_chunksize),
                children):
            n_done += 1
            if cert:
                n_lp_cert += 1
                seen_certified.add(c.hash)  # track LP-cert hashes
            else:
                c.lp_val = lp_val if np.isfinite(lp_val) else None
                lp_fails.append(c)
                if ineq_bytes is not None:
                    arr = np.frombuffer(ineq_bytes, dtype=np.float64).copy()
                    if arr.size == ineqlin_n:
                        lp_fail_ineq[c.hash] = arr
            if n_done >= next_progress:
                pct = 100 * n_done / n_children
                rate = n_done / max(1e-3, time.time() - t0)
                eta = (n_children - n_done) / max(1e-3, rate)
                print(f'    [LP] {n_done}/{n_children} ({pct:.0f}%) '
                      f'cert={n_lp_cert}  rate={rate:.0f} box/s  eta={eta:.0f}s',
                      flush=True)
                next_progress += max(1, n_children // 20)
        t_lp = time.time() - t0
        print(f'  [LP] DONE  cert={n_lp_cert}/{n_children} '
              f'({100*n_lp_cert/max(1,n_children):.1f}%)  '
              f'lp_fails={len(lp_fails)}  in {t_lp:.0f}s',
              flush=True)

        # ----- 3. K=0 SDP FALLBACK (parallel) -----
        survivors: List[SurvivorBox] = []
        n_sdp_cert = 0
        if args.no_sdp:
            survivors = list(lp_fails)
            t_sdp = 0.0
            print(f'  [K=0 SDP] SKIPPED (--no-sdp); survivors={len(survivors)}',
                  flush=True)
        elif lp_fails:
            t0 = time.time()
            sdp_args = [(tuple(c.lo_int), tuple(c.hi_int),
                          args.target_num, args.target_den)
                         for c in lp_fails]
            n_done = 0
            next_progress = max(1, len(lp_fails) // 20)
            for cert, c in zip(
                    sdp_pool.imap(_sdp_one, sdp_args,
                                  chunksize=args.sdp_chunksize),
                    lp_fails):
                n_done += 1
                if cert:
                    n_sdp_cert += 1
                    seen_certified.add(c.hash)  # track SDP-cert hashes
                else:
                    survivors.append(c)
                if n_done >= next_progress:
                    pct = 100 * n_done / len(lp_fails)
                    rate = n_done / max(1e-3, time.time() - t0)
                    eta = (len(lp_fails) - n_done) / max(1e-3, rate)
                    print(f'    [K=0 SDP] {n_done}/{len(lp_fails)} '
                          f'({pct:.0f}%) cert={n_sdp_cert}  '
                          f'rate={rate:.0f} box/s  eta={eta:.0f}s',
                          flush=True)
                    next_progress += max(1, len(lp_fails) // 20)
            t_sdp = time.time() - t0
            print(f'  [K=0 SDP] DONE  cert={n_sdp_cert}/{len(lp_fails)} '
                  f'({100*n_sdp_cert/max(1,len(lp_fails)):.1f}%)  '
                  f'survivors={len(survivors)}  in {t_sdp:.0f}s',
                  flush=True)
        else:
            t_sdp = 0.0
            print('  [K=0 SDP] skipped (no LP-fails)', flush=True)

        # Carry forward ineqlin only for actual survivors
        ineqlin_by_hash = {s.hash: lp_fail_ineq[s.hash]
                            for s in survivors if s.hash in lp_fail_ineq}

        # ----- 4. SAVE -----
        save_path = out_dir / f'iter_{level:03d}_survivors.npz'
        survivors_to_npz(survivors, str(save_path))
        survivors_to_npz(survivors, str(out_dir / 'pending_survivors.npz'))

        global_mult = len(survivors) / max(1, n_parents)
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

        print(f'  [iter {level} SUMMARY] parents={n_parents}  '
              f'children_kept={n_children}  lp_cert={n_lp_cert}  '
              f'sdp_cert={n_sdp_cert}  survivors={len(survivors)}  '
              f'GLOBAL_MULT={global_mult:.3f}  rss={rss_gb:.1f}GB  '
              f'wall={t_split+t_lp+t_sdp:.0f}s', flush=True)
        print(f'  [iter {level} DEDUP IMPACT] cert_already={sstats["cert_already"]} '
              f'within_iter_dup={sstats["within_iter_dup"]} '
              f'(saved {sstats["cert_already"]+sstats["within_iter_dup"]} LP+SDP calls)',
              flush=True)

        rec = {
            'iter': level, 'depth_in': depth_in, 'depth_out': depth_out,
            'n_parents': n_parents, 'n_children_kept': n_children,
            'n_simplex_clipped': sstats['simplex_clipped'],
            'n_cert_already': sstats['cert_already'],
            'n_within_iter_dup': sstats['within_iter_dup'],
            'n_lp_cert': n_lp_cert, 'n_sdp_cert': n_sdp_cert,
            'n_survivors': len(survivors),
            'global_mult': global_mult,
            't_split_s': t_split, 't_lp_s': t_lp, 't_sdp_s': t_sdp,
            'rss_gb': rss_gb,
            'seen_certified_total': len(seen_certified),
        }
        journal.write(json.dumps(rec) + '\n')
        journal.flush()

        # ----- 5. STOP CONDITIONS -----
        if not survivors:
            print(f'\n[iter {level}] CONVERGED — proof complete '
                  f'(seen_certified={len(seen_certified)} unique boxes).',
                  flush=True)
            break
        if level >= 3 and global_mult > args.max_mult:
            print(f'\n[iter {level}] global_mult={global_mult:.2f} > '
                  f'max-mult={args.max_mult} → ABORT', flush=True)
            break

        pool = survivors

    journal.close()
    lp_pool.close(); lp_pool.join()
    if sdp_pool is not None:
        sdp_pool.close(); sdp_pool.join()
    print('\n[done]', flush=True)


if __name__ == '__main__':
    main()
