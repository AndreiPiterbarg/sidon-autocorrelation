"""K-LADDER SDP ESCALATION with auto-tuned per-stage parallelism.

THE LADDER
----------
For each K in [0, 16, 32, 64, 128]:

  1. CALIBRATION (serial, 2 boxes):
     Pick CALIBRATION_N boxes (default 2) from the survivor pool. Run
     the SDP at this K SEQUENTIALLY (one process at a time) with
     `CALIB_THREADS_PER_PROC` MOSEK threads. Record per-box:
       - peak resident set size (Linux ru_maxrss)
       - wall time
       - verdict (infeas / feas / uncertain / exception)
     Determine `max_rss_gb` = max over the calibration boxes.

  2. PARALLELISM AUTO-TUNE:
     `available_ram_gb` = psutil.virtual_memory().available / 1e9
                          (read AT calibration time — accounts for what
                          else is running on the pod).
     `usable_ram_gb` = available_ram_gb * (1 - HEADROOM_FRAC)
                      (leaves HEADROOM_FRAC=0.25 free for OS+cache).
     `max_rss_with_safety` = max_rss_gb * RSS_SAFETY_FACTOR (1.3×)
     `n_parallel_by_ram` = floor(usable_ram_gb / max_rss_with_safety)
     `n_parallel_by_cores` = floor(TOTAL_CORES / desired_threads_per_proc)
     `n_parallel` = max(1, min(n_parallel_by_ram, n_parallel_by_cores))
     `threads_per_proc` = max(1, floor(TOTAL_CORES / n_parallel))
                         (give remaining cores to each process)

  3. FULL SWEEP (parallel pool):
     Run the SDP at this K on ALL REMAINING SURVIVORS (excluding
     calibration boxes which already have results) using
     `n_parallel` worker processes × `threads_per_proc` MOSEK threads.

  4. FILTER:
     verdict == 'infeas' → CERT (move to certified set, drop from pool)
     verdict != 'infeas' → SURVIVOR (carry to next K stage)

  5. STOP AT K=128:
     If after K=128 there are still survivors, REPORT them as
     `final_survivors` — these need either splitting + re-injection
     into the BnB cascade OR a tighter SDP formulation (e.g. PSD
     window cones for ALL windows, Z/2 symmetry, higher-order Lasserre).

OUTPUT FILE LAYOUT (per stage)
------------------------------
  k_ladder/
    stage_K{K}/
      calibration.json          # per-box calib results, max_rss, derived n_parallel
      sweep_progress.log        # one line per result (jsonl)
      sweep_per_box/<hash>.json # full per-box result with lo_int/hi_int (recoverable)
      sweep_summary.json        # cert/fail counts, wall, etc.
      survivors.npz             # boxes that didn't cert at this K (input to next K)
    final_summary.json          # ladder rollup
    final_survivors.npz         # boxes that didn't cert at K=128 (or empty if all cert)

EVERY box is recoverable from disk at any point. Survivor sets are
persisted as npz so a subsequent run can resume from a specific K.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict, field

try:
    import resource  # Linux/macOS only — used inside worker subprocesses
    # for ru_maxrss; never invoked on Windows because the calibration +
    # sweep paths only fire on the pod.
except ImportError:
    resource = None  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ==========================================================================
# CONFIGURATION
# ==========================================================================

# K values to escalate through. Stops at the LAST entry; survivors after
# this point are NOT processed further (must be split or re-injected).
DEFAULT_K_LADDER: Tuple[int, ...] = (0, 16, 32, 64, 128)

# CALIBRATION
CALIBRATION_N = 2                # boxes per stage to calibrate parallelism
CALIB_THREADS_PER_PROC = 16      # threads used during calibration solves
                                 # (gives a representative RSS measurement;
                                 # sweep may use fewer for higher parallelism)

# RAM HEADROOM POLICY
HEADROOM_FRAC = 0.25             # leave 25% RAM free for OS / cache / other
RSS_SAFETY_FACTOR = 1.3          # add 30% safety margin on measured RSS
                                 # (per-proc RAM may grow beyond calibration peak
                                 # on harder boxes)

# SWEEP THREAD/PROC TRADE-OFF
# MOSEK has steep diminishing returns past ~4-8 threads on these per-box
# SDPs (work is moment-matrix dominated, not factorization dominated). So
# at low-RAM K-stages we'd rather run MORE procs with FEWER threads each.
# The auto-tune lets RAM be the binding constraint up to MAX_PARALLEL_FLOOR
# procs (=cores//MIN_THREADS_PER_PROC); past that, threads_per_proc grows.
MIN_THREADS_PER_PROC = 4         # MOSEK floor — fewer than this is waste
MAX_THREADS_PER_PROC = 48        # MOSEK ceiling — more than this rarely helps
                                 # on the per-box SDP sizes here

# PER-CALL TIME LIMIT
DEFAULT_TIME_LIMIT_S = 600.0


# ==========================================================================
# Survivor representation
# ==========================================================================

@dataclass
class SurvivorBox:
    """A box that needs SDP cert. All-integer endpoints (sound).

    `iters_survived` counts how many split-then-SDP iterations this
    lineage has been through without getting certified. Bumped in
    `kill_survivors.split_survivors`. Used by `run_split_first` to
    abort with `MATH_INSUFFICIENT_AT_d` after a configurable threshold
    — distinguishes math-failure (val_B too close to target) from
    config-failure (parallelism / time limits).
    """
    hash: str
    lo_int: List[int]
    hi_int: List[int]
    depth: int = 0
    volume: float = 0.0
    lp_val: Optional[float] = None
    src: str = ''                # source dump file + index for provenance
    iters_survived: int = 0      # split-then-SDP iters this lineage cleared


def survivors_to_npz(survivors: Sequence[SurvivorBox], path: str) -> None:
    """Persist a survivor list to a single npz file."""
    if not survivors:
        np.savez(path,
                 hash=np.zeros(0, dtype='U16'),
                 lo_int=np.zeros((0, 1), dtype=np.int64),
                 hi_int=np.zeros((0, 1), dtype=np.int64),
                 depth=np.zeros(0, dtype=np.int64),
                 volume=np.zeros(0, dtype=np.float64),
                 lp_val=np.zeros(0, dtype=np.float64),
                 src=np.zeros(0, dtype='U64'),
                 iters_survived=np.zeros(0, dtype=np.int64))
        return
    d = len(survivors[0].lo_int)
    np.savez(
        path,
        hash=np.array([s.hash for s in survivors], dtype='U16'),
        lo_int=np.array([s.lo_int for s in survivors], dtype=object),
        hi_int=np.array([s.hi_int for s in survivors], dtype=object),
        depth=np.array([s.depth for s in survivors], dtype=np.int64),
        volume=np.array([s.volume for s in survivors], dtype=np.float64),
        lp_val=np.array([s.lp_val if s.lp_val is not None else float('nan')
                          for s in survivors], dtype=np.float64),
        src=np.array([s.src for s in survivors], dtype='U64'),
        iters_survived=np.array([s.iters_survived for s in survivors],
                                dtype=np.int64),
    )


def survivors_from_npz(path: str) -> List[SurvivorBox]:
    """Load a survivor list. Backward-compatible: defaults
    `iters_survived=0` if the field is absent (older npz files)."""
    data = np.load(path, allow_pickle=True)
    if data['hash'].size == 0:
        return []
    has_iters = 'iters_survived' in data.files
    out = []
    for i in range(data['hash'].size):
        out.append(SurvivorBox(
            hash=str(data['hash'][i]),
            lo_int=[int(x) for x in data['lo_int'][i]],
            hi_int=[int(x) for x in data['hi_int'][i]],
            depth=int(data['depth'][i]),
            volume=float(data['volume'][i]),
            lp_val=(None if np.isnan(data['lp_val'][i])
                    else float(data['lp_val'][i])),
            src=str(data['src'][i]),
            iters_survived=int(data['iters_survived'][i]) if has_iters else 0,
        ))
    return out


# ==========================================================================
# Worker entry points
# ==========================================================================

def _solve_one_box(d: int, target: float, lo_arr: np.ndarray,
                   hi_arr: np.ndarray, K: int, threads: int,
                   time_limit: float, repo_root: str) -> Dict[str, Any]:
    """Solve ONE box's SDP at this K. Returns full diagnostic dict.

    Used both by the calibration (serial) path and the sweep (worker
    pool) path. Fork-safe: builds its own MOSEK Fusion cache.
    """
    sys.path.insert(0, repo_root)
    # Pin BLAS threads to 1 — MOSEK manages its own.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
        bound_sdp_escalation_lb_float_fast,
    )
    windows = build_windows(d)
    t0_cache = time.time()
    cache = build_sdp_escalation_cache_fast(d, windows, target=target)
    cache_build_s = time.time() - t0_cache

    t0 = time.time()
    try:
        res = bound_sdp_escalation_lb_float_fast(
            lo_arr, hi_arr, windows, d, cache=cache, target=target,
            n_window_psd_cones=K, time_limit_s=time_limit,
            n_threads=threads,
        )
        verdict = res.get('verdict')
        lam = float(res.get('lambda_star', float('nan')))
        status = str(res.get('solsta', ''))
        error = None
    except Exception as e:
        verdict = f'EXCEPTION:{type(e).__name__}'
        lam = float('nan')
        status = 'exception'
        error = str(e)
    wall = time.time() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'K': K, 'wall_s': wall, 'cache_build_s': cache_build_s,
        'verdict': verdict, 'lambda_star': lam, 'status': status,
        'peak_rss_kb': int(rss_kb),
        'peak_rss_gb': round(rss_kb / 1024 / 1024, 2),
        'threads': threads, 'error': error,
    }


def _calibration_subprocess(d: int, target: float, lo_arr: np.ndarray,
                              hi_arr: np.ndarray, K: int, threads: int,
                              time_limit: float, repo_root: str,
                              result_q: mp.Queue) -> None:
    """Subprocess wrapper for one calibration box (so RSS isolated)."""
    res = _solve_one_box(d, target, lo_arr, hi_arr, K, threads,
                          time_limit, repo_root)
    result_q.put(res)


def _sweep_worker(work_q: mp.Queue, result_q: mp.Queue,
                  d: int, target: float, K: int, threads: int,
                  time_limit: float, repo_root: str,
                  worker_id: int) -> None:
    """Pool worker: build cache once, drain work_q. Same as _solve_one_box
    but the MOSEK cache is built once per worker and reused.
    """
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
    try:
        t0c = time.time()
        cache = build_sdp_escalation_cache_fast(d, windows, target=target)
        cache_build_s = time.time() - t0c
    except Exception as e:
        result_q.put({'worker_id': worker_id, 'fatal': True,
                      'error': f'cache_build: {type(e).__name__}: {e}'})
        return
    result_q.put({'worker_id': worker_id, 'ready': True,
                  'cache_build_s': cache_build_s})

    while True:
        try:
            item = work_q.get(timeout=2.0)
        except Exception:
            break
        bhash = item['hash']
        lo_arr = np.asarray(item['lo'], dtype=np.float64)
        hi_arr = np.asarray(item['hi'], dtype=np.float64)
        t0 = time.time()
        try:
            res = bound_sdp_escalation_lb_float_fast(
                lo_arr, hi_arr, windows, d, cache=cache, target=target,
                n_window_psd_cones=K, time_limit_s=time_limit,
                n_threads=threads,
            )
            verdict = res.get('verdict')
            lam = float(res.get('lambda_star', float('nan')))
            status = str(res.get('solsta', ''))
            error = None
        except Exception as e:
            verdict = f'EXCEPTION:{type(e).__name__}'
            lam = float('nan')
            status = 'exception'
            error = str(e)
        wall = time.time() - t0
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result_q.put({
            'worker_id': worker_id,
            'hash': bhash, 'src': item.get('src', ''),
            'lo_int': item['lo_int'], 'hi_int': item['hi_int'],
            'depth': item.get('depth', 0),
            'volume': item.get('volume', 0.0),
            'lp_val': item.get('lp_val'),
            'iters_survived': item.get('iters_survived', 0),
            'K': K, 'wall_s': wall,
            'verdict': verdict, 'lambda_star': lam, 'status': status,
            'peak_rss_kb': int(rss_kb),
            'peak_rss_gb': round(rss_kb / 1024 / 1024, 2),
            'threads': threads, 'error': error,
        })


# ==========================================================================
# Stage runner
# ==========================================================================

@dataclass
class StageResult:
    K: int
    n_input: int                   # survivors entering this stage
    n_calibration: int
    calibration: List[Dict[str, Any]]
    max_rss_gb_observed: float     # from calibration (and sweep if higher)
    n_parallel: int
    threads_per_proc: int
    available_ram_gb_at_start: float
    n_cert: int
    n_fail: int                    # survivors going to next stage
    n_exception: int
    sweep_wall_s: float
    avg_solve_s: float
    survivors: List[SurvivorBox] = field(default_factory=list)


def _available_ram_gb() -> float:
    """Read available RAM (free + reclaimable cache). Fallback to /proc."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024 ** 3)
    # Fallback: parse /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 64.0  # safe-ish default if we can't tell


def _total_cores() -> int:
    return os.cpu_count() or 16


def run_one_stage(d: int, target: float, K: int,
                   survivors: Sequence[SurvivorBox],
                   stage_dir: Path, repo_root: Path, *,
                   time_limit_s: float = DEFAULT_TIME_LIMIT_S,
                   calibration_n: int = CALIBRATION_N,
                   headroom_frac: float = HEADROOM_FRAC,
                   rss_safety_factor: float = RSS_SAFETY_FACTOR,
                   ) -> StageResult:
    """Run one K-stage: calibrate → auto-tune → sweep → filter.

    Calibration boxes are processed serially in subprocess so we get
    isolated peak RSS per box. The sweep then runs on the REMAINING
    boxes (calibration boxes already have results).
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    sweep_per_box_dir = stage_dir / 'sweep_per_box'
    sweep_per_box_dir.mkdir(parents=True, exist_ok=True)
    progress_log = stage_dir / 'sweep_progress.log'

    n_input = len(survivors)
    if n_input == 0:
        empty = StageResult(
            K=K, n_input=0, n_calibration=0, calibration=[],
            max_rss_gb_observed=0.0, n_parallel=0, threads_per_proc=0,
            available_ram_gb_at_start=_available_ram_gb(),
            n_cert=0, n_fail=0, n_exception=0,
            sweep_wall_s=0.0, avg_solve_s=0.0, survivors=[],
        )
        (stage_dir / 'sweep_summary.json').write_text(
            json.dumps(asdict(empty), indent=2))
        return empty

    print(f"\n{'='*70}\nK={K} STAGE — {n_input} input survivors\n{'='*70}",
          flush=True)

    # ---------- 1. CALIBRATION ----------
    # Pick the LARGEST-VOLUME boxes for calibration. These have the
    # widest McCormick gap and tend to be the slowest-to-solve / highest
    # peak-RSS cases — using the first-N boxes (which on a freshly-split
    # pool are deterministically siblings of survivor #1) systematically
    # under-estimates worst-case RSS and lets the auto-tune over-commit
    # parallelism, causing mid-sweep OOMs.
    n_calib = min(calibration_n, n_input)
    survivors_by_vol = sorted(survivors, key=lambda s: -s.volume)
    calib_boxes = list(survivors_by_vol[:n_calib])
    # Track which calibration hashes we picked so the sweep can exclude
    # them (was previously implicit via list slicing).
    calib_hashes = {b.hash for b in calib_boxes}
    calib_results: List[Dict[str, Any]] = []
    print(f"  [calib] running {n_calib} largest-volume boxes serially "
          f"({CALIB_THREADS_PER_PROC} threads each, "
          f"vols={[f'{b.volume:.2e}' for b in calib_boxes]})...", flush=True)
    for i, box in enumerate(calib_boxes):
        t0 = time.time()
        ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')
        result_q = ctx.Queue()
        # Convert int endpoints back to float arrays for the solver.
        SCALE = 2 ** 60
        lo_arr = np.array([float(x) / SCALE for x in box.lo_int],
                           dtype=np.float64)
        hi_arr = np.array([float(x) / SCALE for x in box.hi_int],
                           dtype=np.float64)
        p = ctx.Process(target=_calibration_subprocess, args=(
            d, target, lo_arr, hi_arr, K, CALIB_THREADS_PER_PROC,
            time_limit_s, str(repo_root), result_q,
        ))
        p.start()
        try:
            res = result_q.get(timeout=time_limit_s + 60)
        except Exception:
            res = {'wall_s': time.time() - t0, 'verdict': 'TIMEOUT',
                   'peak_rss_gb': 0.0, 'error': 'queue_timeout'}
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
        res.update({
            'hash': box.hash, 'lp_val': box.lp_val, 'src': box.src,
            'lo_int': box.lo_int, 'hi_int': box.hi_int,
            'depth': box.depth, 'volume': box.volume,
            'iters_survived': box.iters_survived,
        })
        calib_results.append(res)
        print(f"    calib {i+1}/{n_calib}: hash={box.hash} "
              f"verdict={res.get('verdict')} t={res.get('wall_s', 0):.1f}s "
              f"rss={res.get('peak_rss_gb', 0)} GB", flush=True)
    max_rss_gb_calib = max(
        (r.get('peak_rss_gb', 0.0) for r in calib_results), default=0.1)
    avg_calib_t = sum(r.get('wall_s', 0.0) for r in calib_results) / n_calib

    # ---------- 2. PARALLELISM AUTO-TUNE ----------
    # Strategy: maximize concurrent procs subject to RAM headroom, with the
    # cores cap relaxed to cores//MIN_THREADS_PER_PROC (not //CALIB_*). This
    # is what closes the "12-parallel" gap at low-RAM K-stages: at K=0 with
    # ~5 GB/proc on a 192-core, 600 GB pod we get ~48 procs × 4 threads
    # instead of 12 procs × 16 threads (~4× total throughput).
    avail_gb = _available_ram_gb()
    cores = _total_cores()
    usable_ram = avail_gb * (1 - headroom_frac)
    rss_with_safety = max(0.5, max_rss_gb_calib * rss_safety_factor)
    n_par_by_ram = max(1, int(usable_ram // rss_with_safety))
    n_par_by_cores = max(1, cores // MIN_THREADS_PER_PROC)
    n_parallel = max(1, min(n_par_by_ram, n_par_by_cores))
    # Distribute remaining cores; clamp to MAX_THREADS_PER_PROC so we don't
    # overshoot MOSEK's useful range when n_parallel is tiny.
    threads_per_proc = max(MIN_THREADS_PER_PROC,
                            min(MAX_THREADS_PER_PROC, cores // n_parallel))
    print(f"  [auto-tune] avail_ram={avail_gb:.1f} GB  cores={cores}  "
          f"max_rss={max_rss_gb_calib:.2f} GB", flush=True)
    print(f"              usable_ram={usable_ram:.1f} GB  "
          f"rss_with_safety={rss_with_safety:.2f} GB", flush=True)
    print(f"              n_par by_ram={n_par_by_ram} by_cores={n_par_by_cores} "
          f"(MIN_THREADS_PER_PROC={MIN_THREADS_PER_PROC}) "
          f"→ n_parallel={n_parallel}, threads/proc={threads_per_proc}", flush=True)

    # Save calibration summary.
    (stage_dir / 'calibration.json').write_text(json.dumps({
        'K': K, 'n_calibration': n_calib,
        'calibration_results': calib_results,
        'max_rss_gb_calib': max_rss_gb_calib,
        'avg_calib_wall_s': avg_calib_t,
        'available_ram_gb': avail_gb,
        'usable_ram_gb': usable_ram,
        'rss_with_safety_gb': rss_with_safety,
        'n_parallel_by_ram': n_par_by_ram,
        'n_parallel_by_cores': n_par_by_cores,
        'n_parallel_chosen': n_parallel,
        'threads_per_proc': threads_per_proc,
        'headroom_frac': headroom_frac,
        'rss_safety_factor': rss_safety_factor,
        'min_threads_per_proc': MIN_THREADS_PER_PROC,
        'max_threads_per_proc': MAX_THREADS_PER_PROC,
    }, indent=2, default=str))

    # ---------- 3. FULL SWEEP ----------
    # Exclude calibration boxes by HASH (not position) — calibration
    # picks largest-volume boxes which may be anywhere in the pool.
    sweep_boxes = [b for b in survivors if b.hash not in calib_hashes]
    print(f"  [sweep] launching pool: {n_parallel} procs × "
          f"{threads_per_proc} threads, sweep_size={len(sweep_boxes)} boxes",
          flush=True)

    sweep_results: List[Dict[str, Any]] = []
    sweep_wall = 0.0
    if sweep_boxes:
        ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')
        work_q = ctx.Queue()
        result_q = ctx.Queue()
        SCALE = 2 ** 60
        for box in sweep_boxes:
            work_q.put({
                'hash': box.hash,
                'lo_int': box.lo_int, 'hi_int': box.hi_int,
                'lo': [float(x) / SCALE for x in box.lo_int],
                'hi': [float(x) / SCALE for x in box.hi_int],
                'depth': box.depth, 'volume': box.volume,
                'lp_val': box.lp_val, 'src': box.src,
                'iters_survived': box.iters_survived,
            })
        procs = []
        for w in range(n_parallel):
            p = ctx.Process(target=_sweep_worker, args=(
                work_q, result_q, d, target, K, threads_per_proc,
                time_limit_s, str(repo_root), w,
            ))
            p.start()
            procs.append(p)
        n_total = len(sweep_boxes)
        n_collected = 0
        n_cert = 0
        n_fail = 0
        n_exc = 0
        n_ready = 0
        sum_solve_s = 0.0
        t0 = time.time()
        last_log_t = t0
        prog_fh = open(progress_log, 'w', encoding='utf-8')
        try:
            while n_collected < n_total:
                try:
                    r = result_q.get(timeout=time_limit_s + 60)
                except Exception:
                    print(f"  [sweep] result-queue timeout — "
                          f"got {n_collected}/{n_total}", flush=True)
                    break
                if r.get('ready'):
                    n_ready += 1
                    prog_fh.write(json.dumps({
                        'event': 'worker_ready', **r}) + '\n')
                    prog_fh.flush()
                    continue
                if r.get('fatal'):
                    prog_fh.write(json.dumps({
                        'event': 'worker_fatal', **r}) + '\n')
                    prog_fh.flush()
                    continue
                n_collected += 1
                v = r.get('verdict')
                if v == 'infeas':
                    n_cert += 1
                else:
                    n_fail += 1
                    if v and v.startswith('EXCEPTION'):
                        n_exc += 1
                sum_solve_s += float(r.get('wall_s', 0.0))
                sweep_results.append(r)
                # Per-box JSON (recoverable lo_int/hi_int!)
                try:
                    bhash = r['hash']
                    (sweep_per_box_dir / f'{bhash}.json').write_text(
                        json.dumps(r, indent=2, default=str))
                except Exception as e:
                    prog_fh.write(json.dumps({
                        'event': 'per_box_write_fail',
                        'hash': r.get('hash'), 'error': str(e)}) + '\n')
                prog_fh.write(json.dumps({
                    'event': 'box_result',
                    'hash': r.get('hash'),
                    'verdict': v, 'wall_s': r.get('wall_s'),
                    'peak_rss_gb': r.get('peak_rss_gb'),
                    'lambda_star': r.get('lambda_star'),
                    'wall_t': time.time() - t0,
                }) + '\n')
                prog_fh.flush()
                now = time.time()
                if now - last_log_t > 30 or n_collected % 10 == 0:
                    elapsed = now - t0
                    rate = n_collected / max(0.1, elapsed)
                    eta = (n_total - n_collected) / max(0.001, rate)
                    rss_obs = max((rr.get('peak_rss_gb', 0.0)
                                    for rr in sweep_results), default=0.0)
                    print(f"    [sweep K={K}] {n_collected}/{n_total} done  "
                          f"cert={n_cert} fail={n_fail} exc={n_exc}  "
                          f"max_rss_obs={rss_obs:.2f}GB  "
                          f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
                    last_log_t = now
        finally:
            prog_fh.close()
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        sweep_wall = time.time() - t0
    else:
        n_cert = 0
        n_fail = 0
        n_exc = 0

    # ---------- 4. FILTER ----------
    # Combine calibration + sweep results.
    all_results = calib_results + sweep_results
    survivors_out: List[SurvivorBox] = []
    n_cert_total = 0
    n_fail_total = 0
    n_exc_total = 0
    max_rss_observed = max_rss_gb_calib
    for r in all_results:
        v = r.get('verdict')
        rss = r.get('peak_rss_gb', 0.0)
        if rss > max_rss_observed:
            max_rss_observed = rss
        if v == 'infeas':
            n_cert_total += 1
        else:
            n_fail_total += 1
            if v and v.startswith('EXCEPTION'):
                n_exc_total += 1
            # Build SurvivorBox from result for next stage.
            survivors_out.append(SurvivorBox(
                hash=r.get('hash', ''),
                lo_int=r.get('lo_int', []),
                hi_int=r.get('hi_int', []),
                depth=r.get('depth', 0),
                volume=r.get('volume', 0.0),
                lp_val=r.get('lp_val'),
                src=r.get('src', ''),
                iters_survived=r.get('iters_survived', 0),
            ))
    avg_solve_s = (sum(r.get('wall_s', 0.0) for r in all_results)
                   / max(1, len(all_results)))
    survivors_to_npz(survivors_out, str(stage_dir / 'survivors.npz'))

    summary = StageResult(
        K=K, n_input=n_input, n_calibration=n_calib,
        calibration=calib_results,
        max_rss_gb_observed=max_rss_observed,
        n_parallel=n_parallel, threads_per_proc=threads_per_proc,
        available_ram_gb_at_start=avail_gb,
        n_cert=n_cert_total, n_fail=n_fail_total, n_exception=n_exc_total,
        sweep_wall_s=sweep_wall, avg_solve_s=avg_solve_s,
        survivors=survivors_out,
    )
    sd = asdict(summary)
    # Compress: don't store full survivors list in JSON (it's in survivors.npz).
    sd.pop('survivors', None)
    sd['n_survivors_for_next_stage'] = len(survivors_out)
    (stage_dir / 'sweep_summary.json').write_text(json.dumps(sd, indent=2,
                                                              default=str))
    print(f"  [stage K={K}] cert={n_cert_total} fail={n_fail_total} "
          f"exc={n_exc_total}  max_rss={max_rss_observed:.2f} GB  "
          f"sweep_wall={sweep_wall:.0f}s ({sweep_wall/60:.1f} min)",
          flush=True)
    return summary


# ==========================================================================
# Top-level ladder runner
# ==========================================================================

@dataclass
class LadderResult:
    K_ladder: List[int]
    stages: List[StageResult]
    final_survivors: List[SurvivorBox]
    total_input: int
    total_cert: int
    total_wall_s: float


def run_k_ladder(d: int, target: float,
                  initial_survivors: Sequence[SurvivorBox],
                  output_dir: Path, repo_root: Path, *,
                  k_ladder: Sequence[int] = DEFAULT_K_LADDER,
                  time_limit_s: float = DEFAULT_TIME_LIMIT_S,
                  ) -> LadderResult:
    """Run the full K-ladder over the survivor set.

    For each K in `k_ladder`: calibrate → auto-tune parallelism → sweep
    → filter. Survivors carry to next K. Stops at last K.

    Returns a LadderResult with per-stage summaries and the final
    survivor list. `final_survivors` is empty if every box was certified.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: List[StageResult] = []
    survivors = list(initial_survivors)
    total_input = len(survivors)
    total_t0 = time.time()
    for K in k_ladder:
        if not survivors:
            print(f"\n[ladder] all boxes certified before reaching K={K} — "
                  f"halting early", flush=True)
            break
        stage_dir = output_dir / f'stage_K{K}'
        result = run_one_stage(d, target, K, survivors, stage_dir, repo_root,
                                time_limit_s=time_limit_s)
        stages.append(result)
        survivors = result.survivors
    total_wall = time.time() - total_t0
    total_cert = total_input - len(survivors)
    survivors_to_npz(survivors, str(output_dir / 'final_survivors.npz'))
    final = LadderResult(
        K_ladder=list(k_ladder),
        stages=stages,
        final_survivors=survivors,
        total_input=total_input,
        total_cert=total_cert,
        total_wall_s=total_wall,
    )
    # Strip survivor lists from per-stage dicts before JSON dump (in npz).
    fdict = asdict(final)
    for s in fdict['stages']:
        s.pop('survivors', None)
    fdict['n_final_survivors'] = len(survivors)
    (output_dir / 'final_summary.json').write_text(
        json.dumps(fdict, indent=2, default=str))
    print(f"\n{'#'*70}\n# K-LADDER COMPLETE  total={total_input}  "
          f"cert={total_cert}  survivors={len(survivors)}  "
          f"wall={total_wall:.0f}s ({total_wall/60:.1f} min)\n{'#'*70}",
          flush=True)
    return final
