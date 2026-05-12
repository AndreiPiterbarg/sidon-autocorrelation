"""Phase 2 of each iteration: SDP escalation pool over dumped boxes.

PURPOSE
-------
Take the boxes captured by the BnB phase (worker dumps + master_queue
dump) and certify each via the dual-Farkas Lasserre order-2 SDP. Uses
`interval_bnb.bound_sdp_escalation_fast.bound_sdp_escalation_lb_float_fast`
with K=0 (linear epigraph for all windows; fastest variant) and a
fallback to K=16 if K=0 returns `verdict != 'infeas'`.

ARCHITECTURE
------------
* A pool of N_PROCS worker processes is spawned, each with T_THREADS
  MOSEK threads. Each worker builds the static SDP cache ONCE
  (~5-30s) then drains a shared `mp.Queue` of (box_idx, lo, hi, src,
  hash) records. For each:

    1. Filter via cheap epigraph LP. If LP_val >= target → skip
       (cascade should have caught it; we treat as cascade-cert and
       record it that way for accounting).
    2. SDP K=0 with cushion. If verdict='infeas' → SDP_CERT.
    3. Otherwise SDP K=16. If verdict='infeas' → SDP_CERT (fallback).
    4. Otherwise → SDP_FAIL (the box must be split + re-injected in
       the next BnB iteration).

* Each result is emitted to the journal AND written as a per-box
  JSON file `<iter_dir>/sdp/per_box/<hash>.json` containing the full
  diagnostic (status, λ*, residuals, peak RSS, wall time).

* A `sdp/summary.json` rolls up cert/fail counts, total wall, and
  fallback usage.

OUTPUT FILES
------------
  <iter_dir>/sdp/
    pool_progress.log      # appended one line per result, monotonic
    per_box/<hash>.json    # one file per box (full diagnostic)
    summary.json           # cert/fail/fallback rollup
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class SDPPhaseConfig:
    """All knobs governing one SDP-phase iteration."""
    d: int
    target_str: str               # Fraction-style "1.2805"
    n_procs: int = 12
    threads_per_proc: int = 16    # 12 × 16 = 192 cores
    K_first: int = 0              # try K=0 first
    K_fallback: int = 16          # if K=0 fails, retry with this K
    time_limit_s: float = 600.0
    epigraph_lp_skip: bool = True # boxes where LP cert → skip SDP

    # Result file layout — relative to iter_dir
    sdp_subdir: str = 'sdp'


# --------------------------------------------------------------------------
# Worker: builds cache once, then drains work_q
# --------------------------------------------------------------------------

def _sdp_worker(work_q: mp.Queue, result_q: mp.Queue,
                d: int, target: float, threads: int,
                K_first: int, K_fallback: int,
                time_limit: float, repo_root: str,
                worker_id: int):
    """Pool worker. Builds SDP cache, drains work_q, writes per-box JSON."""
    import resource
    sys.path.insert(0, repo_root)
    # Pin numpy/OpenBLAS to single thread per worker — MOSEK manages its
    # own threads, and over-subscription kills performance.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
        bound_sdp_escalation_lb_float_fast,
    )
    from interval_bnb.bound_epigraph import bound_epigraph_lp_float

    windows = build_windows(d)
    t0_cache = time.time()
    try:
        cache = build_sdp_escalation_cache_fast(d, windows, target=target)
        cache_build_s = time.time() - t0_cache
    except Exception as e:
        # Cache build failure — abort this worker; remaining boxes will
        # be picked up by other workers.
        result_q.put({
            'worker_id': worker_id, 'fatal': True,
            'error': f'cache_build: {type(e).__name__}: {e}',
        })
        return

    # Tell the master we're alive and ready.
    result_q.put({'worker_id': worker_id, 'ready': True,
                  'cache_build_s': cache_build_s})

    while True:
        try:
            item = work_q.get(timeout=2.0)
        except Exception:
            break  # queue exhausted
        box_idx = item['box_idx']
        lo = item['lo']
        hi = item['hi']
        src = item['src']
        bhash = item['hash']
        depth = item.get('depth', 0)
        volume = item.get('volume', 0.0)
        lp_val = item.get('lp_val')           # may be precomputed by master
        lo_int = item.get('lo_int')           # integer endpoints for re-feed
        hi_int = item.get('hi_int')

        result_base = {
            'worker_id': worker_id,
            'box_idx': box_idx,
            'hash': bhash,
            'src': src,
            'depth': depth,
            'volume': volume,
            'lp_val': lp_val,
            # CRITICAL for recovery: store integer endpoints so a failed
            # box can be re-fed into another iteration (split + re-inject)
            # OR retried with a higher K (or different cone formulation)
            # WITHOUT going back to the source dump file.
            'lo_int': [int(x) for x in lo_int] if lo_int is not None else None,
            'hi_int': [int(x) for x in hi_int] if hi_int is not None else None,
        }

        # Recompute LP if not provided. (Master normally precomputes it.)
        if lp_val is None:
            try:
                lp_val = bound_epigraph_lp_float(lo, hi, windows, d)
            except Exception:
                lp_val = float('nan')
            result_base['lp_val'] = float(lp_val)

        # If LP already certs at target, return that (no SDP needed).
        if np.isfinite(lp_val) and lp_val >= target:
            result_q.put({
                **result_base,
                'verdict': 'lp_cert',  # cascade-equivalent cert
                'lambda_star': float('nan'),
                'used_K': None,
                'wall_s': 0.0,
                'peak_rss_gb': 0.0,
                'status': 'lp_only',
            })
            continue

        # SDP attempt 1: K=K_first
        t0 = time.time()
        try:
            res1 = bound_sdp_escalation_lb_float_fast(
                lo, hi, windows, d, cache=cache, target=target,
                n_window_psd_cones=K_first, time_limit_s=time_limit,
                n_threads=threads,
            )
            verdict = res1.get('verdict')
            lam = float(res1.get('lambda_star', float('nan')))
            status = str(res1.get('solsta', ''))
            used_K = K_first
            error = None
        except Exception as e:
            verdict = f'EXCEPTION:{type(e).__name__}'
            lam = float('nan')
            status = 'exception'
            used_K = K_first
            error = str(e)

        # Fallback: K=K_fallback if K_first didn't cert
        if verdict != 'infeas' and K_fallback is not None and K_fallback > K_first:
            try:
                res2 = bound_sdp_escalation_lb_float_fast(
                    lo, hi, windows, d, cache=cache, target=target,
                    n_window_psd_cones=K_fallback, time_limit_s=time_limit,
                    n_threads=threads,
                )
                verdict2 = res2.get('verdict')
                lam2 = float(res2.get('lambda_star', float('nan')))
                if verdict2 == 'infeas':
                    verdict = verdict2
                    lam = lam2
                    status = str(res2.get('solsta', ''))
                    used_K = K_fallback
                    error = None
            except Exception:
                pass

        wall = time.time() - t0
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        result_q.put({
            **result_base,
            'verdict': verdict,
            'lambda_star': lam,
            'used_K': used_K,
            'wall_s': wall,
            'peak_rss_gb': round(rss_kb / 1024 / 1024, 2),
            'status': status,
            'error': error,
        })


# --------------------------------------------------------------------------
# Phase runner
# --------------------------------------------------------------------------

@dataclass
class SDPPhaseResult:
    iter_dir: str
    n_total: int
    n_lp_cert: int
    n_sdp_cert: int
    n_sdp_fail: int
    n_fallback_used: int
    n_exception: int
    wall_s: float
    avg_solve_s: float


def run_one_sdp_phase(cfg: SDPPhaseConfig, repo_root: Path,
                       iter_dir: Path,
                       work: List[Dict[str, Any]]) -> SDPPhaseResult:
    """Run SDP pool over the given list of work items.

    Each work item is a dict:
      {'box_idx': int, 'hash': str, 'lo': np.ndarray, 'hi': np.ndarray,
       'src': str, 'depth': int, 'volume': float, 'lp_val': float|None}

    Per-box results are written to <iter_dir>/sdp/per_box/<hash>.json.
    A summary.json + pool_progress.log are written.
    """
    sdp_dir = iter_dir / cfg.sdp_subdir
    per_box_dir = sdp_dir / 'per_box'
    per_box_dir.mkdir(parents=True, exist_ok=True)
    progress_log = sdp_dir / 'pool_progress.log'

    if not work:
        # Nothing to do — write empty summary and return.
        empty = SDPPhaseResult(
            iter_dir=str(iter_dir), n_total=0, n_lp_cert=0,
            n_sdp_cert=0, n_sdp_fail=0, n_fallback_used=0, n_exception=0,
            wall_s=0.0, avg_solve_s=0.0,
        )
        (sdp_dir / 'summary.json').write_text(
            json.dumps(asdict(empty), indent=2))
        return empty

    target_f = float(cfg.target_str)
    print(f"[sdp-phase] launching pool: {cfg.n_procs} procs × "
          f"{cfg.threads_per_proc} threads, K={cfg.K_first}→{cfg.K_fallback}, "
          f"work={len(work)} boxes", flush=True)

    ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')
    work_q = ctx.Queue()
    result_q = ctx.Queue()
    for w in work:
        work_q.put(w)

    procs: List[mp.Process] = []
    for w_id in range(cfg.n_procs):
        p = ctx.Process(target=_sdp_worker, args=(
            work_q, result_q,
            cfg.d, target_f, cfg.threads_per_proc,
            cfg.K_first, cfg.K_fallback, cfg.time_limit_s,
            str(repo_root), w_id,
        ))
        p.start()
        procs.append(p)

    # Drain results.
    n_total = len(work)
    n_collected = 0
    n_lp_cert = 0
    n_sdp_cert = 0
    n_sdp_fail = 0
    n_fb = 0
    n_exc = 0
    sum_solve_s = 0.0
    t0 = time.time()
    last_log_t = t0
    progress_fh = open(progress_log, 'w', encoding='utf-8')

    # Track ready-callbacks separately so we know when workers fully
    # initialized (for accurate cache_build timing).
    n_ready = 0

    try:
        while n_collected < n_total:
            try:
                r = result_q.get(timeout=cfg.time_limit_s + 60)
            except Exception:
                print(f"  [sdp-phase] result queue timeout — got "
                      f"{n_collected}/{n_total}", flush=True)
                break

            # Distinguish ready/fatal pings from box results.
            if r.get('ready'):
                n_ready += 1
                progress_fh.write(json.dumps({
                    'event': 'worker_ready',
                    'worker_id': r['worker_id'],
                    'cache_build_s': r['cache_build_s'],
                    'wall_t': time.time() - t0,
                }) + '\n')
                progress_fh.flush()
                continue
            if r.get('fatal'):
                progress_fh.write(json.dumps({
                    'event': 'worker_fatal',
                    'worker_id': r['worker_id'],
                    'error': r['error'],
                    'wall_t': time.time() - t0,
                }) + '\n')
                progress_fh.flush()
                continue

            # Box result.
            n_collected += 1
            verdict = r.get('verdict')
            if verdict == 'lp_cert':
                n_lp_cert += 1
            elif verdict == 'infeas':
                n_sdp_cert += 1
                if r.get('used_K') == cfg.K_fallback:
                    n_fb += 1
            elif verdict and verdict.startswith('EXCEPTION'):
                n_exc += 1
                n_sdp_fail += 1
            else:
                n_sdp_fail += 1
            sum_solve_s += float(r.get('wall_s', 0))

            # Per-box JSON. Includes lo_int/hi_int (integer endpoints)
            # so a failed box can be retried with higher K or split and
            # re-injected without going back to the source dump file.
            # Strip the numpy `lo`/`hi` float arrays — same content as
            # lo_int/hi_int but not JSON-serializable cleanly.
            try:
                bhash = r['hash']
                clean = {k: v for k, v in r.items()
                         if k not in ('lo', 'hi')}
                (per_box_dir / f'{bhash}.json').write_text(
                    json.dumps(clean, indent=2, default=str))
            except Exception as e:
                progress_fh.write(json.dumps({
                    'event': 'per_box_write_fail',
                    'box_idx': r.get('box_idx'),
                    'error': str(e),
                }) + '\n')

            progress_fh.write(json.dumps({
                'event': 'box_result',
                'box_idx': r.get('box_idx'),
                'hash': r.get('hash'),
                'verdict': verdict,
                'used_K': r.get('used_K'),
                'wall_s': r.get('wall_s'),
                'peak_rss_gb': r.get('peak_rss_gb'),
                'lambda_star': r.get('lambda_star'),
                'wall_t': time.time() - t0,
            }) + '\n')
            progress_fh.flush()

            # Periodic progress log (every 30s OR every 10 results).
            now = time.time()
            if now - last_log_t > 30 or n_collected % 10 == 0:
                elapsed = now - t0
                rate = n_collected / max(0.1, elapsed)
                eta = (n_total - n_collected) / max(0.001, rate)
                print(f"  [sdp-phase] {n_collected}/{n_total} done  "
                      f"lp={n_lp_cert} sdp_cert={n_sdp_cert} "
                      f"fail={n_sdp_fail} fb={n_fb} exc={n_exc}  "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
                last_log_t = now
    finally:
        progress_fh.close()
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    wall = time.time() - t0
    avg_solve = sum_solve_s / max(1, n_collected)
    summary = SDPPhaseResult(
        iter_dir=str(iter_dir),
        n_total=n_total, n_lp_cert=n_lp_cert,
        n_sdp_cert=n_sdp_cert, n_sdp_fail=n_sdp_fail,
        n_fallback_used=n_fb, n_exception=n_exc,
        wall_s=wall, avg_solve_s=avg_solve,
    )
    summary_dict = asdict(summary)
    summary_dict['config'] = asdict(cfg)
    (sdp_dir / 'summary.json').write_text(
        json.dumps(summary_dict, indent=2))
    print(f"[sdp-phase] DONE  lp_cert={n_lp_cert} sdp_cert={n_sdp_cert} "
          f"fail={n_sdp_fail} fb={n_fb} exc={n_exc}  "
          f"wall={wall:.0f}s ({wall/60:.1f} min)", flush=True)
    return summary
