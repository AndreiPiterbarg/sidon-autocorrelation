"""d=22 t=1.2805 BnB orchestrator with stuck-detection + SDP pilot.

Phase 1 — Cascade BnB:
  Launches `parallel_branch_and_bound` (d=22, target=1.2805) as a child
  process. Streams its stdout, parses `[par] t=...s ... in_flight=N ...`
  lines, tracks (timestamp, in_flight) history.

  Tracks `min_in_flight` incrementally. Detects "cascade plateau":
    - We've passed the minimum (in_flight has risen above min by >5%)
    - In_flight has been monotonically non-decreasing for STUCK_WINDOW_S
    - In_flight is small enough to attack with SDP (<= 5 * workers)

  When trigger fires, signals the BnB process (SIGTERM). Workers dump
  their stacks to stuck_boxes_w*.npz via the existing
  INTERVAL_BNB_DUMP_BOXES mechanism.

Phase 2 — SDP pilot on 3 boxes:
  Loads 3 stuck boxes, runs Z/2+K=0 SDP on each in parallel via 3
  subprocesses. Each subprocess uses ~16 threads + ~50 GB RAM.
  Records: per-box wall time, peak RSS, verdict, lambda*.

Phase 3 — Decide for the rest (manual; printed report at end).

Storage budget:
  - in_flight history list: 1 entry/sec for 1 hr = 3600 floats = ~30 KB
  - min_in_flight: 1 int (incremental)
  - stuck box dumps: numpy (N, d) int64 = N*22*8 B (e.g. 1000 boxes = 176 KB)
  - SDP pilot results: 3 dicts, ~1 KB
  Total Python-side memory: <1 MB, plus the BnB subprocess.
"""
from __future__ import annotations

import os
import re
import sys
import time
import json
import signal
import subprocess
import shutil
import multiprocessing as mp
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Config (CLI overridable later if needed)
# ---------------------------------------------------------------------

D = 22
TARGET = '1.2805'
WORKERS = 16                 # BnB cascade workers
TIME_BUDGET_S = 3600         # 1 hour hard cap on phase 1
INIT_SPLIT_DEPTH = 22

# Stuck-detection — running min after warmup; trigger when min hasn't
# decreased for STUCK_WINDOW_S. Accepts oscillation around the floor
# (no strict monotone requirement).
SAMPLE_PERIOD_S = 5.0
STUCK_WINDOW_S = 300         # min must be stable for 5 min
MAX_INFLIGHT_FOR_SDP = 1000  # tolerate larger pile (we'll SDP all of them)
WARMUP_S = 300               # discard pre-warmup ramp from min calc

# SDP pilot
N_PILOT_BOXES = 3
PILOT_THREADS_PER_PROC = 16  # 16 × 3 = 48 threads concurrently
PILOT_TIME_LIMIT_S = 600     # 10 min per box
PILOT_K = 0                  # K=0 selective windows (linear epigraph for all)

# Phase 4 — full SDP pool over remaining LP-failing boxes
PHASE4_N_PROCS = 12          # 12 × 16 = 192 threads (all cores)
PHASE4_TIME_LIMIT_S = 600    # 10 min per box
PHASE4_K = 0                 # same K as pilot
PHASE4_FALLBACK_K = 16       # if K=0 fails, retry with K=16 (one shot)

# Box dump prefix (workers write stuck_boxes_w*.npz here)
DUMP_PREFIX = 'stuck_d22_1p2805'

# Files
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / 'd22_1p2805_orchestrator.log'
HISTORY_FILE = HERE / 'd22_1p2805_inflight_history.json'
PILOT_RESULTS = HERE / 'd22_1p2805_sdp_pilot.json'
PHASE4_RESULTS = HERE / 'd22_1p2805_sdp_phase4.json'


# ---------------------------------------------------------------------
# Phase 1 — BnB with stuck-detection
# ---------------------------------------------------------------------

INFLIGHT_RE = re.compile(r'in_flight=\s*(\d+)')


def _spawn_bnb() -> subprocess.Popen:
    """Spawn the BnB cascade as a child process with dump enabled."""
    env = dict(os.environ)
    # Cascade env vars (matched to run_d22_dump_stuck.py defaults).
    env['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
    env['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
    env['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
    env['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
    env['INTERVAL_BNB_ANCHOR_DEPTH'] = '24'
    env['INTERVAL_BNB_CENTROID_DEPTH'] = '60'
    env['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '26'
    env['INTERVAL_BNB_DUMP_BOXES'] = DUMP_PREFIX
    env['PYTHONUNBUFFERED'] = '1'

    code = f'''
import sys, time
sys.path.insert(0, {repr(str(HERE))})
from interval_bnb.parallel import parallel_branch_and_bound
t0 = time.time()
r = parallel_branch_and_bound(
    d={D}, target_c="{TARGET}",
    workers={WORKERS},
    init_split_depth={INIT_SPLIT_DEPTH},
    donate_threshold_floor=2,
    time_budget_s={TIME_BUDGET_S},
    verbose=True,
)
print(f"\\n=== RESULT: success={{r['success']}} cov={{100*r['coverage_fraction']:.4f}}%"
      f" in_flight={{r['in_flight_final']}} elapsed={{time.time()-t0:.0f}}s ===")
'''
    return subprocess.Popen(
        [sys.executable, '-u', '-c', code],
        env=env, cwd=str(HERE),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
        preexec_fn=os.setsid if sys.platform != 'win32' else None,
    )


def _stuck_trigger(history: list[tuple[float, int]],
                   post_warmup_min: int,
                   post_warmup_min_t: float,
                   t_now: float) -> bool:
    """Trigger when the post-warmup running minimum hasn't decreased for
    STUCK_WINDOW_S seconds AND in_flight ≤ MAX_INFLIGHT_FOR_SDP.

    `post_warmup_min` is the smallest in_flight seen at any time t ≥
    WARMUP_S; `post_warmup_min_t` is the latest t at which that minimum
    was observed. If t_now − post_warmup_min_t ≥ STUCK_WINDOW_S, the
    cascade hasn't been able to push in_flight any lower for 5 min →
    treat as stuck.
    """
    if not history:
        return False
    t_now_actual, cur = history[-1]
    if t_now_actual < WARMUP_S + STUCK_WINDOW_S:
        return False  # not enough post-warmup history yet
    if cur > MAX_INFLIGHT_FOR_SDP:
        return False
    if post_warmup_min == 10**12:
        return False  # no post-warmup samples yet
    return (t_now_actual - post_warmup_min_t) >= STUCK_WINDOW_S


def phase1_run_bnb_until_stuck() -> dict:
    """Runs the BnB cascade, monitors in_flight, triggers dump when stuck.
    Returns metadata dict: {history, min_inflight, trigger_reason, dumps}.
    """
    print(f"=== PHASE 1: launching BnB d={D} t={TARGET} workers={WORKERS} ===",
          flush=True)
    proc = _spawn_bnb()
    history: list[tuple[float, int]] = []  # (elapsed_s, in_flight)
    min_inflight = 10**12        # all-time min (incl warmup)
    min_inflight_t = 0.0
    post_warmup_min = 10**12     # min only counting samples after WARMUP_S
    post_warmup_min_t = 0.0
    trigger_reason = None
    t0 = time.time()

    log_fh = open(LOG_FILE, 'w', encoding='utf-8')
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                # Subprocess exited.
                rc = proc.poll()
                trigger_reason = f'subprocess_exit_rc={rc}'
                break
            log_fh.write(line)
            log_fh.flush()
            elapsed = time.time() - t0
            m = INFLIGHT_RE.search(line)
            if m:
                inflight = int(m.group(1))
                history.append((elapsed, inflight))
                if inflight < min_inflight:
                    min_inflight = inflight
                    min_inflight_t = elapsed
                # Track POST-WARMUP min separately (cleaner trigger)
                if elapsed >= WARMUP_S and inflight < post_warmup_min:
                    post_warmup_min = inflight
                    post_warmup_min_t = elapsed
                # print short tracker (every 6 samples = ~30s)
                if len(history) % 6 == 0:
                    pw_str = (f"pw_min={post_warmup_min}@{post_warmup_min_t:.0f}s "
                              f"({elapsed - post_warmup_min_t:.0f}s ago)"
                              if post_warmup_min < 10**12 else "pw_min=(warmup)")
                    print(f"[track] t={elapsed:6.0f}s in_flight={inflight:>6d} "
                          f"all_min={min_inflight}@{min_inflight_t:.0f}s {pw_str}",
                          flush=True)

                # Stuck trigger?
                if _stuck_trigger(history, post_warmup_min,
                                    post_warmup_min_t, elapsed):
                    trigger_reason = (
                        f'stuck@{elapsed:.0f}s in_flight={inflight} '
                        f'pw_min={post_warmup_min}@{post_warmup_min_t:.0f}s '
                        f'(stable {elapsed - post_warmup_min_t:.0f}s)')
                    print(f"\n!! STUCK trigger: {trigger_reason}", flush=True)
                    break

                if elapsed > TIME_BUDGET_S:
                    trigger_reason = f'time_budget_exceeded@{elapsed:.0f}s'
                    print(f"\n!! Time budget hit", flush=True)
                    break
    finally:
        log_fh.close()

    # Trigger graceful shutdown via SIGINT (the workers' SIGINT handler
    # sets failed_event → workers exit their loop → finally dumps boxes).
    # SIGTERM would abrupt-kill without running finally.
    print("[orchestrator] sending SIGINT to BnB process group (workers dump on exit)",
          flush=True)
    if proc.poll() is None:
        try:
            if sys.platform != 'win32':
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception as e:
            print(f"  SIGINT failed: {e}", flush=True)
        # Wait up to 120s for graceful dump (workers have to flush npz files).
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            print("  graceful timeout — escalating to SIGTERM", flush=True)
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    if sys.platform != 'win32':
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass
                proc.wait(timeout=15)

    # Collect dump files.
    dumps = sorted(HERE.glob(f'{DUMP_PREFIX}_w*.npz'))
    print(f"[orchestrator] {len(dumps)} dump files found", flush=True)

    meta = {
        'history': history,
        'min_inflight': min_inflight,
        'min_inflight_t': min_inflight_t,
        'trigger_reason': trigger_reason,
        'n_dump_files': len(dumps),
        'd': D, 'target': TARGET, 'workers': WORKERS,
    }
    HISTORY_FILE.write_text(json.dumps({
        **{k: v for k, v in meta.items() if k != 'history'},
        'history': [(t, v) for (t, v) in history],
    }, indent=2))
    print(f"[orchestrator] history written to {HISTORY_FILE}", flush=True)
    return meta, dumps


# ---------------------------------------------------------------------
# Phase 2 — SDP pilot on 3 boxes
# ---------------------------------------------------------------------

def _load_stuck_boxes(dumps: list[Path]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Load (lo, hi, src) tuples from worker dumps. Boxes are float64."""
    out = []
    for fp in dumps:
        try:
            data = np.load(fp, allow_pickle=True)
            keys = list(data.files)
            if 'lo' not in keys or 'hi' not in keys:
                continue
            lo_arr = data['lo']
            hi_arr = data['hi']
            if lo_arr.ndim == 2:
                for i in range(lo_arr.shape[0]):
                    out.append((lo_arr[i].astype(np.float64),
                                hi_arr[i].astype(np.float64),
                                f"{fp.name}#{i}"))
            else:
                out.append((lo_arr.astype(np.float64),
                            hi_arr.astype(np.float64), fp.name))
        except Exception as e:
            print(f"  load fail {fp}: {e}", flush=True)
    return out


def _filter_lp_failing(boxes, target_f: float):
    """Filter to LP-failing boxes (LP < target). Returns sorted by LP desc
    (closest to target first — easiest SDP cases)."""
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_epigraph import bound_epigraph_lp_float
    windows = build_windows(D)
    rows = []
    for (lo, hi, src) in boxes:
        if lo.shape[0] != D or lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        try:
            lp = bound_epigraph_lp_float(lo, hi, windows, D)
        except Exception:
            continue
        if lp < target_f:  # only LP-failing
            rows.append((lp, lo, hi, src))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


# Worker entry point for the parallel SDP solve.
def _sdp_worker(box_idx: int, lo: np.ndarray, hi: np.ndarray,
                target: float, threads: int, time_limit: float,
                K: int, result_q: mp.Queue):
    """Run one SDP solve (in subprocess), report back via queue."""
    import resource, sys, time
    sys.path.insert(0, str(HERE))
    from interval_bnb.windows import build_windows
    try:
        from interval_bnb.bound_sdp_escalation_z2 import (
            build_sdp_escalation_cache_z2, bound_sdp_escalation_z2_lb_float,
            is_box_sigma_symmetric,
        )
        z2_avail = True
    except Exception:
        z2_avail = False
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
        bound_sdp_escalation_lb_float_fast,
    )
    windows = build_windows(D)

    # Try Z/2 first if box is σ-symmetric AND z2 module loads.
    used_path = 'k0_fast'
    sym = False
    if z2_avail:
        try:
            sym = is_box_sigma_symmetric(lo, hi)
        except Exception:
            sym = False

    t0 = time.time()
    try:
        if sym and z2_avail:
            cache = build_sdp_escalation_cache_z2(D, windows, target=target)
            res = bound_sdp_escalation_z2_lb_float(
                lo, hi, windows, D, cache=cache, target=target,
                time_limit_s=time_limit, n_threads=threads,
            )
            used_path = 'z2_auto'
        else:
            cache = build_sdp_escalation_cache_fast(D, windows, target=target)
            res = bound_sdp_escalation_lb_float_fast(
                lo, hi, windows, D, cache=cache, target=target,
                n_window_psd_cones=K, time_limit_s=time_limit,
                n_threads=threads,
            )
    except Exception as e:
        result_q.put({
            'box_idx': box_idx, 'wall_s': time.time() - t0,
            'verdict': f'EXCEPTION:{type(e).__name__}', 'error': str(e),
            'used_path': used_path, 'sym': sym,
        })
        return

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_q.put({
        'box_idx': box_idx, 'wall_s': time.time() - t0,
        'verdict': res.get('verdict'),
        'lambda_star': float(res.get('lambda_star', float('nan'))),
        'status': str(res.get('solsta', '')),
        'peak_rss_kb': int(rss_kb),
        'peak_rss_mb': int(rss_kb / 1024),
        'peak_rss_gb': round(rss_kb / 1024 / 1024, 2),
        'used_path': used_path,
        'sym': sym,
    })


def phase2_pilot(dumps: list[Path]) -> list[dict]:
    """Run SDP on the first N_PILOT_BOXES LP-failing boxes in parallel.
    Returns list of result dicts."""
    print(f"\n=== PHASE 2: SDP pilot on {N_PILOT_BOXES} boxes "
          f"({PILOT_THREADS_PER_PROC} threads each) ===", flush=True)
    target_f = float(TARGET)

    print("  loading dump files...", flush=True)
    raw = _load_stuck_boxes(dumps)
    print(f"  loaded {len(raw)} raw boxes", flush=True)
    if not raw:
        print("  NO BOXES — aborting pilot", flush=True)
        return []

    print("  filtering LP-failing boxes...", flush=True)
    rows = _filter_lp_failing(raw, target_f)
    print(f"  {len(rows)} LP-failing boxes available", flush=True)
    if not rows:
        print("  no LP-failing boxes — pilot can't run (all already LP-cert)",
              flush=True)
        return []

    pilot_boxes = rows[:N_PILOT_BOXES]
    for k, (lp, lo, hi, src) in enumerate(pilot_boxes):
        print(f"    pilot #{k}: LP={lp:.6f} src={src} "
              f"hw={float((hi-lo).max()/2):.4f}", flush=True)

    # Launch N_PILOT_BOXES processes in parallel.
    ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')
    result_q = ctx.Queue()
    procs = []
    for k, (lp, lo, hi, src) in enumerate(pilot_boxes):
        p = ctx.Process(target=_sdp_worker, args=(
            k, lo, hi, target_f, PILOT_THREADS_PER_PROC,
            PILOT_TIME_LIMIT_S, PILOT_K, result_q,
        ))
        p.start()
        procs.append(p)

    print(f"  {len(procs)} pilot procs running...", flush=True)
    pilot_t0 = time.time()
    results = []
    for _ in range(len(procs)):
        try:
            results.append(result_q.get(timeout=PILOT_TIME_LIMIT_S + 60))
        except Exception as e:
            print(f"  pilot result missing: {e}", flush=True)
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    pilot_wall = time.time() - pilot_t0

    print(f"  pilot finished in {pilot_wall:.1f}s wall:", flush=True)
    for r in sorted(results, key=lambda x: x.get('box_idx', -1)):
        print(f"    box#{r.get('box_idx')}: verdict={r.get('verdict')} "
              f"t={r.get('wall_s', 0):.1f}s "
              f"rss={r.get('peak_rss_gb', '?')}GB "
              f"path={r.get('used_path')} sym={r.get('sym')} "
              f"lam={r.get('lambda_star', 0):.4f}", flush=True)
    PILOT_RESULTS.write_text(json.dumps({
        'pilot_wall_s': pilot_wall,
        'n_boxes_attempted': len(pilot_boxes),
        'pilot_boxes_meta': [
            {'lp': lp, 'src': src, 'hw': float((hi-lo).max()/2)}
            for (lp, lo, hi, src) in pilot_boxes
        ],
        'results': results,
        'config': {
            'd': D, 'target': TARGET, 'K': PILOT_K,
            'threads_per_proc': PILOT_THREADS_PER_PROC,
            'time_limit_s': PILOT_TIME_LIMIT_S,
        },
    }, indent=2, default=str))
    print(f"  pilot results -> {PILOT_RESULTS}", flush=True)
    return results


# ---------------------------------------------------------------------
# Phase 4 — SDP on ALL remaining LP-failing boxes (12-process pool)
# ---------------------------------------------------------------------

def _phase4_worker(work_q: mp.Queue, result_q: mp.Queue,
                    target: float, threads: int, time_limit: float,
                    K: int, fallback_K: int):
    """Pool worker: build SDP cache once, then drain `work_q`. For each
    box: try K=K first; if not infeas, retry with K=fallback_K (one-shot
    fallback). Report result via `result_q`.
    """
    import resource
    import sys
    sys.path.insert(0, str(HERE))
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
        bound_sdp_escalation_lb_float_fast,
    )
    windows = build_windows(D)
    cache = build_sdp_escalation_cache_fast(D, windows, target=target)
    while True:
        try:
            box_idx, lp, lo, hi, src = work_q.get(timeout=2.0)
        except Exception:
            break  # queue exhausted
        t0 = time.time()
        try:
            res = bound_sdp_escalation_lb_float_fast(
                lo, hi, windows, D, cache=cache, target=target,
                n_window_psd_cones=K, time_limit_s=time_limit,
                n_threads=threads,
            )
            verdict = res.get('verdict')
            lam = float(res.get('lambda_star', float('nan')))
            used_K = K
            # Fallback: if K=K didn't cert, try K=fallback_K once.
            if verdict != 'infeas' and fallback_K is not None and fallback_K > K:
                t1_fb = time.time()
                try:
                    res2 = bound_sdp_escalation_lb_float_fast(
                        lo, hi, windows, D, cache=cache, target=target,
                        n_window_psd_cones=fallback_K,
                        time_limit_s=time_limit,
                        n_threads=threads,
                    )
                    verdict2 = res2.get('verdict')
                    lam2 = float(res2.get('lambda_star', float('nan')))
                    if verdict2 == 'infeas':
                        verdict = verdict2
                        lam = lam2
                        used_K = fallback_K
                except Exception:
                    pass
            err = None
        except Exception as e:
            verdict = f'EXCEPTION:{type(e).__name__}'
            lam = float('nan')
            used_K = K
            err = str(e)
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result_q.put({
            'box_idx': box_idx, 'src': src, 'lp': float(lp),
            'wall_s': time.time() - t0,
            'verdict': verdict, 'lambda_star': lam,
            'used_K': used_K, 'rss_gb': round(rss_kb / 1024 / 1024, 2),
            'error': err,
        })


def phase4_full_sweep(dumps: list[Path],
                       skip_srcs: set[str],
                       n_already_cert: int = 0) -> dict:
    """Run SDP on every LP-failing box (skipping the pilot ones).
    Pool of PHASE4_N_PROCS workers, each builds cache once.
    """
    print(f"\n=== PHASE 4: full SDP sweep "
          f"({PHASE4_N_PROCS} workers × {PHASE4_THREADS_PER_PROC if False else PILOT_THREADS_PER_PROC} threads, K={PHASE4_K} → fallback K={PHASE4_FALLBACK_K}) ===",
          flush=True)
    target_f = float(TARGET)

    raw = _load_stuck_boxes(dumps)
    rows = _filter_lp_failing(raw, target_f)
    # Drop the boxes already used in the pilot.
    work = [r for r in rows if r[3] not in skip_srcs]
    print(f"  total raw={len(raw)}  LP-failing={len(rows)}  "
          f"skipped (pilot)={len(rows)-len(work)}  to process={len(work)}",
          flush=True)
    if not work:
        return {'n_processed': 0, 'results': []}

    ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')
    work_q = ctx.Queue()
    result_q = ctx.Queue()

    for k, (lp, lo, hi, src) in enumerate(work):
        work_q.put((k, lp, lo, hi, src))

    procs = []
    for w in range(PHASE4_N_PROCS):
        p = ctx.Process(target=_phase4_worker, args=(
            work_q, result_q, target_f,
            PILOT_THREADS_PER_PROC, PHASE4_TIME_LIMIT_S,
            PHASE4_K, PHASE4_FALLBACK_K,
        ))
        p.start()
        procs.append(p)

    print(f"  pool started — collecting results...", flush=True)
    pilot_t0 = time.time()
    results = []
    n_total = len(work)
    n_cert = 0
    n_fail = 0
    n_fb = 0
    last_log_t = pilot_t0
    while len(results) < n_total:
        try:
            r = result_q.get(timeout=PHASE4_TIME_LIMIT_S + 60)
        except Exception:
            print(f"  timeout collecting results — got {len(results)}/{n_total}",
                  flush=True)
            break
        results.append(r)
        if r.get('verdict') == 'infeas':
            n_cert += 1
        else:
            n_fail += 1
        if r.get('used_K') == PHASE4_FALLBACK_K:
            n_fb += 1
        # Periodic progress log (every 30s OR every 10 results).
        now = time.time()
        if now - last_log_t > 30 or len(results) % 10 == 0:
            elapsed = now - pilot_t0
            rate = len(results) / max(0.1, elapsed)
            eta = (n_total - len(results)) / max(0.001, rate)
            print(f"  [phase4] {len(results)}/{n_total} done  "
                  f"cert={n_cert} fail={n_fail} fb={n_fb}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s",
                  flush=True)
            last_log_t = now
    pilot_wall = time.time() - pilot_t0

    # Drain any straggler workers.
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    summary = {
        'n_total_lp_failing': len(rows),
        'n_processed': len(results),
        'n_cert': n_cert,
        'n_fail': n_fail,
        'n_fallback_used': n_fb,
        'wall_s': pilot_wall,
        'avg_solve_s': (sum(r.get('wall_s', 0) for r in results)
                        / max(1, len(results))),
        'n_already_lp_cert': n_already_cert,
        'config': {
            'd': D, 'target': TARGET,
            'K': PHASE4_K, 'fallback_K': PHASE4_FALLBACK_K,
            'n_procs': PHASE4_N_PROCS,
            'threads_per_proc': PILOT_THREADS_PER_PROC,
            'time_limit_s': PHASE4_TIME_LIMIT_S,
        },
        'results': results,
    }
    PHASE4_RESULTS.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  PHASE 4 DONE  cert={n_cert}/{len(results)} "
          f"fail={n_fail} fallback_used={n_fb}  wall={pilot_wall:.0f}s "
          f"({pilot_wall/60:.1f} min)", flush=True)
    print(f"  results -> {PHASE4_RESULTS}", flush=True)
    return summary


# ---------------------------------------------------------------------
# Phase 3 — Decide for the rest (printed report)
# ---------------------------------------------------------------------

def phase3_summary(meta, pilot_results, n_dumped):
    print("\n=== PHASE 3: PILOT-BASED EXTRAPOLATION ===", flush=True)
    succ = [r for r in pilot_results if r.get('verdict') == 'infeas']
    fail = [r for r in pilot_results if r.get('verdict') != 'infeas']
    if not succ:
        print("  NO pilot box certified — SDP escalation may not work here.",
              flush=True)
        print("  Recommend: try K=16 fallback, or split boxes further first.",
              flush=True)
        return
    avg_t = sum(r['wall_s'] for r in succ) / len(succ)
    max_rss_gb = max(r.get('peak_rss_gb', 0) for r in succ)
    print(f"  Pilot certs : {len(succ)}/{len(pilot_results)}", flush=True)
    print(f"  Avg solve   : {avg_t:.1f}s per box", flush=True)
    print(f"  Peak RSS    : {max_rss_gb:.1f} GB per process", flush=True)
    n_remaining = max(0, n_dumped - N_PILOT_BOXES)
    print(f"  Remaining   : ~{n_remaining} boxes to escalate", flush=True)
    parallel_by_mem = int(780 / max(1, max_rss_gb))
    parallel_by_cores = int(196 / PILOT_THREADS_PER_PROC)
    par = min(parallel_by_mem, parallel_by_cores)
    print(f"  Parallelism : min({parallel_by_cores} cores, "
          f"{parallel_by_mem} mem) = {par}", flush=True)
    print(f"  Est. wallclock for remaining: "
          f"{n_remaining * avg_t / max(1, par) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"\n#### d={D} t={TARGET} BnB+SDP orchestrator ####", flush=True)
    # Phase 1
    meta, dumps = phase1_run_bnb_until_stuck()
    print(f"\nPhase 1 done. min_inflight={meta['min_inflight']} "
          f"reason={meta['trigger_reason']}", flush=True)

    # Also pick up the master_queue dump (if any).
    master_dump = HERE / f'{DUMP_PREFIX}_master_queue.npz'
    if master_dump.exists():
        dumps = list(dumps) + [master_dump]
        print(f"  + master queue dump: {master_dump.name}", flush=True)
    # Phase 2 — pilot if we have dumped boxes
    if not dumps:
        print("\nNo stuck boxes dumped (cascade may have closed everything).",
              flush=True)
        return
    pilot_results = phase2_pilot(dumps)

    # Phase 3 — extrapolate
    n_dumped = sum(
        np.load(d, allow_pickle=True)['lo'].shape[0] if 'lo' in
        np.load(d, allow_pickle=True).files else 0
        for d in dumps
    )
    phase3_summary(meta, pilot_results, n_dumped)

    # Phase 4 — full SDP sweep over remaining LP-failing boxes
    pilot_srcs = set()
    try:
        with open(PILOT_RESULTS, 'r') as fh:
            pdata = json.loads(fh.read())
            for b in pdata.get('pilot_boxes_meta', []):
                pilot_srcs.add(b.get('src'))
    except Exception:
        pass
    print(f"\n[main] pilot_srcs to skip in phase4: {pilot_srcs}",
          flush=True)
    phase4_summary = phase4_full_sweep(dumps, skip_srcs=pilot_srcs)
    print(f"\n#### END-TO-END SUMMARY ####", flush=True)
    print(f"  Phase 1 cascade: stuck @ {meta.get('trigger_reason')}", flush=True)
    print(f"  Phase 2 pilot  : {sum(1 for r in pilot_results if r.get('verdict')=='infeas')}/{len(pilot_results)} infeas", flush=True)
    print(f"  Phase 4 sweep  : {phase4_summary.get('n_cert')}/{phase4_summary.get('n_processed')} infeas, "
          f"{phase4_summary.get('n_fallback_used')} needed K=fallback, "
          f"wall={phase4_summary.get('wall_s', 0)/60:.1f} min",
          flush=True)
    total_lp_failing = phase4_summary.get('n_total_lp_failing', 0)
    total_cert = (sum(1 for r in pilot_results if r.get('verdict')=='infeas')
                  + phase4_summary.get('n_cert', 0))
    print(f"  TOTAL d=22 t={TARGET}: cert={total_cert}/{total_lp_failing} LP-failing boxes",
          flush=True)
    if phase4_summary.get('n_fail', 0) > 0:
        print(f"  WARN: {phase4_summary['n_fail']} boxes failed even at K={PHASE4_FALLBACK_K} — "
              f"would need further escalation (split or higher K)",
              flush=True)


if __name__ == '__main__':
    main()
