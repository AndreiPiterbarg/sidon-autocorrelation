"""Empirical MOSEK threads-per-proc benchmark — validates audit fix #6.

The split-first auto-tune now picks `MIN_THREADS_PER_PROC=4`, scaling
parallelism up by ~4x at low-RAM K stages. The CLAIM is that MOSEK has
steep diminishing returns past ~4 threads on these per-box SDPs, so the
total throughput goes up. This script MEASURES that claim on real boxes
before any pod hours are committed.

USAGE
-----
    python -m cert_pipeline.bench_threads \\
        --d 22 --target 1.2805 \\
        --K 0 \\
        --boxes-from runs/<some_prior_run>/pending_survivors.npz \\
        --n-boxes 8 \\
        --threads 4,8,16,32 \\
        --output bench_threads_d22_K0.json

Or, if no prior survivor pool exists, generate boxes by deeply splitting
the root simplex (each box is small but well-shaped):

    python -m cert_pipeline.bench_threads \\
        --d 22 --target 1.2805 \\
        --K 0 \\
        --boxes-from-root \\
        --root-split-depth 8 \\
        --n-boxes 8 \\
        --threads 4,8,16,32

OUTPUT
------
A JSON with per-(box, threads) wall-clock + verdict, plus aggregate
roll-ups:
    median_wall_per_thread_count = {4: ..., 8: ..., 16: ..., 32: ...}
    proj_throughput_boxes_per_hour = N_par * 3600 / median_wall
    where N_par = floor(192 / threads_per_proc)
The throughput row is what we actually care about — the `threads` value
that maximises it is the right MIN_THREADS_PER_PROC for the pod.

INTERPRETATION
--------------
- If proj_throughput at threads=4 is >= proj_throughput at threads=16,
  the audit fix is validated; commit pod hours.
- If proj_throughput peaks at threads=8 or threads=16, dial
  MIN_THREADS_PER_PROC up correspondingly in k_ladder.py.
- If threads=4 is significantly slower per box than threads=16 (more
  than 2x), the per-thread parallel loss outweighs the n_par gain;
  KEEP MIN_THREADS_PER_PROC=16 and don't relaunch.

Each (box, threads) trial is a fresh subprocess (clean RSS, clean
MOSEK env), serially. Wall-clock time captures everything including
cache build to be conservative — that's what the production pool
actually pays (worker builds cache once, but at the start).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from cert_pipeline.k_ladder import (
    SurvivorBox, survivors_from_npz,
)
from cert_pipeline.kill_survivors import (
    split_box_to_depth, MathInsufficient,
)
from cert_pipeline.box_journal import canonical_box_hash


def _trial_subprocess(d: int, target: float, lo: np.ndarray, hi: np.ndarray,
                       K: int, n_threads: int, time_limit: float,
                       repo_root: str, result_q: mp.Queue) -> None:
    """One trial: solve a box's SDP at given K and thread count, report
    wall-clock + verdict. Runs in a fresh subprocess so MOSEK / RSS /
    cache state are isolated.
    """
    import resource
    sys.path.insert(0, repo_root)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
        bound_sdp_escalation_lb_float_fast,
    )
    windows = build_windows(d)
    t0 = time.time()
    cache = build_sdp_escalation_cache_fast(d, windows, target=target)
    t_cache = time.time() - t0

    t1 = time.time()
    try:
        res = bound_sdp_escalation_lb_float_fast(
            lo, hi, windows, d, cache=cache, target=target,
            n_window_psd_cones=K, time_limit_s=time_limit,
            n_threads=n_threads,
        )
        verdict = res.get('verdict')
        lam = float(res.get('lambda_star', float('nan')))
        error = None
    except Exception as e:
        verdict = f'EXCEPTION:{type(e).__name__}'
        lam = float('nan')
        error = str(e)
    wall = time.time() - t1
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_q.put({
        'wall_solve_s': wall,
        'wall_cache_build_s': t_cache,
        'wall_total_s': wall + t_cache,
        'verdict': verdict,
        'lambda_star': lam,
        'error': error,
        'peak_rss_gb': round(rss_kb / 1024 / 1024, 2),
        'n_threads': n_threads,
    })


def _box_to_arrays(box: SurvivorBox):
    SCALE = 2 ** 60
    lo = np.array([float(x) / SCALE for x in box.lo_int], dtype=np.float64)
    hi = np.array([float(x) / SCALE for x in box.hi_int], dtype=np.float64)
    return lo, hi


def _boxes_from_root(d: int, root_split_depth: int,
                      n_boxes: int) -> List[SurvivorBox]:
    """Generate `n_boxes` SurvivorBoxes by splitting the root simplex
    `root_split_depth` times along the top-`root_split_depth` widest
    axes. At depth 8 in d=22 we get 256 children, of which the simplex
    filter keeps a substantial subset; we take the first `n_boxes`
    that intersect the simplex.
    """
    from interval_bnb.box import Box, SCALE as _SCALE
    init = Box.initial(d, sym_cuts=[(0, d - 1)])
    lo_int0 = list(init.lo_int) if init.lo_int is not None else [0] * d
    hi_int0 = list(init.hi_int) if init.hi_int is not None else [_SCALE] * d
    try:
        kids = split_box_to_depth(lo_int0, hi_int0, root_split_depth)
    except MathInsufficient as e:
        raise RuntimeError(
            f'cannot generate root-split boxes: {e}') from e
    out: List[SurvivorBox] = []
    for (clo, chi) in kids:
        if sum(clo) > _SCALE or sum(chi) < _SCALE:
            continue
        chash = canonical_box_hash(clo, chi)
        vol = float(np.prod([(h - l) / _SCALE for l, h in zip(clo, chi)]))
        out.append(SurvivorBox(
            hash=chash, lo_int=clo, hi_int=chi,
            depth=root_split_depth, volume=vol,
            lp_val=None, src='bench_root_split',
            iters_survived=0,
        ))
        if len(out) >= n_boxes:
            break
    return out


def main():
    ap = argparse.ArgumentParser(
        description='MOSEK threads-per-proc throughput benchmark.')
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--target', type=float, required=True)
    ap.add_argument('--K', type=int, default=0,
                    help='n_window_psd_cones (0 = all-linear, fastest)')
    ap.add_argument('--threads', type=str, default='4,8,16,32',
                    help='Comma-separated thread counts to benchmark')
    ap.add_argument('--n-boxes', type=int, default=8)
    ap.add_argument('--time-limit-s', type=float, default=300.0)
    ap.add_argument('--total-cores', type=int, default=192,
                    help='For projecting n_par = total_cores // threads')
    ap.add_argument('--output', type=str, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--boxes-from', type=str, default='',
                     help='npz with SurvivorBox list (e.g. '
                          'runs/<tag>/pending_survivors.npz)')
    src.add_argument('--boxes-from-root', action='store_true',
                     help='Generate boxes by splitting the root simplex')
    ap.add_argument('--root-split-depth', type=int, default=8,
                    help='Used with --boxes-from-root')
    args = ap.parse_args()

    threads_list = [int(t) for t in args.threads.split(',')]

    # ---- Source the test boxes ----
    if args.boxes_from_root:
        boxes = _boxes_from_root(args.d, args.root_split_depth, args.n_boxes)
        print(f"[bench] generated {len(boxes)} boxes by root-split "
              f"(depth={args.root_split_depth})", flush=True)
    else:
        all_boxes = survivors_from_npz(args.boxes_from)
        # Pick by volume - largest first - so the bench is conservative
        # (matches the new k_ladder calibration policy).
        all_boxes.sort(key=lambda s: -s.volume)
        boxes = all_boxes[:args.n_boxes]
        print(f"[bench] loaded {len(all_boxes)} boxes from {args.boxes_from}, "
              f"taking top {len(boxes)} by volume", flush=True)

    if not boxes:
        print(f"[bench] no boxes to benchmark — abort", flush=True)
        sys.exit(1)

    # ---- Run trials ----
    results: List[Dict[str, Any]] = []
    ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')
    for box_i, box in enumerate(boxes):
        lo, hi = _box_to_arrays(box)
        for n_threads in threads_list:
            print(f"\n[bench] box {box_i+1}/{len(boxes)} "
                  f"hash={box.hash} threads={n_threads}", flush=True)
            result_q = ctx.Queue()
            t0 = time.time()
            p = ctx.Process(target=_trial_subprocess, args=(
                args.d, args.target, lo, hi, args.K, n_threads,
                args.time_limit_s, str(_REPO), result_q,
            ))
            p.start()
            try:
                trial = result_q.get(timeout=args.time_limit_s + 60)
            except Exception:
                trial = {
                    'wall_total_s': time.time() - t0,
                    'verdict': 'TIMEOUT',
                    'error': 'queue_timeout',
                    'n_threads': n_threads,
                }
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            trial['box_hash'] = box.hash
            trial['box_volume'] = box.volume
            trial['box_index'] = box_i
            results.append(trial)
            print(f"  → wall={trial.get('wall_total_s', 0):.1f}s  "
                  f"verdict={trial.get('verdict')}  "
                  f"lam={trial.get('lambda_star', float('nan')):.3f}  "
                  f"rss={trial.get('peak_rss_gb', 0)} GB", flush=True)

    # ---- Aggregate ----
    by_threads: Dict[int, List[float]] = {}
    for r in results:
        nt = int(r.get('n_threads', 0))
        if r.get('verdict') in ('infeas', 'feas'):
            by_threads.setdefault(nt, []).append(float(r.get('wall_total_s', 0)))
    rollup: Dict[str, Any] = {
        'd': args.d, 'target': args.target, 'K': args.K,
        'n_boxes': len(boxes),
        'threads_tried': threads_list,
        'total_cores_assumed': args.total_cores,
        'per_threads': {},
    }
    for nt in threads_list:
        walls = by_threads.get(nt, [])
        if not walls:
            rollup['per_threads'][str(nt)] = {
                'n_successful': 0,
                'note': 'no successful trials at this thread count',
            }
            continue
        median_wall = statistics.median(walls)
        n_par = max(1, args.total_cores // nt)
        proj_throughput = n_par * 3600.0 / median_wall
        rollup['per_threads'][str(nt)] = {
            'n_successful': len(walls),
            'median_wall_s': median_wall,
            'mean_wall_s': statistics.mean(walls),
            'min_wall_s': min(walls),
            'max_wall_s': max(walls),
            'projected_n_par': n_par,
            'projected_throughput_boxes_per_hour': round(proj_throughput, 1),
        }

    # Recommendation
    best_nt = -1
    best_tput = -1.0
    for nt_str, info in rollup['per_threads'].items():
        if 'projected_throughput_boxes_per_hour' not in info:
            continue
        tput = float(info['projected_throughput_boxes_per_hour'])
        if tput > best_tput:
            best_tput = tput
            best_nt = int(nt_str)
    rollup['recommendation'] = {
        'best_threads_per_proc': best_nt,
        'best_throughput_boxes_per_hour': best_tput,
        'note': ('SET MIN_THREADS_PER_PROC = best_threads_per_proc in '
                 'cert_pipeline/k_ladder.py if it differs from 4.'),
    }

    rollup['raw_trials'] = results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rollup, indent=2, default=str))
    print(f"\n{'='*70}")
    print(f"[bench] DONE  output={args.output}")
    print(f"  per-threads roll-up:")
    for nt_str in sorted(rollup['per_threads'].keys(), key=int):
        info = rollup['per_threads'][nt_str]
        if 'projected_throughput_boxes_per_hour' in info:
            print(f"    threads={nt_str:>3}  median_wall={info['median_wall_s']:6.1f}s  "
                  f"n_par={info['projected_n_par']:>3}  "
                  f"tput={info['projected_throughput_boxes_per_hour']:.0f} box/h")
        else:
            print(f"    threads={nt_str:>3}  {info.get('note', '?')}")
    rec = rollup['recommendation']
    print(f"  RECOMMENDATION: MIN_THREADS_PER_PROC = {rec['best_threads_per_proc']} "
          f"({rec['best_throughput_boxes_per_hour']:.0f} boxes/h)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
