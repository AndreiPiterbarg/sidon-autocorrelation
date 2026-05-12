"""Phase 1 of each iteration: run cascade BnB with INSTANT-DUMP SIGINT.

PURPOSE
-------
Run `interval_bnb.parallel.parallel_branch_and_bound` until the cascade
plateaus (running min in_flight is stable for STUCK_WINDOW_S after a
WARMUP_S grace period). Then deliver SIGINT to the BnB process group.
With `INTERVAL_BNB_INSTANT_DUMP=1` set, every worker's SIGINT handler
will:

  1. Dump its current `local_stack` to `<prefix>_w{wid}.npz`
  2. Call `os._exit(0)` immediately — NO further cascade processing,
     NO Python `finally:` blocks, NO `_publish_stats` flush.

The master also drains the shared mp.Queue to `<prefix>_master_queue.npz`
before issuing SIGINT (so the queue contents are captured even if the
workers exit before they could pull from it).

PUBLICATION-GRADE GUARANTEE
---------------------------
Every box that was `in_flight` at SIGINT-trigger time is captured to one
of {worker dumps} ∪ {master_queue}. No box is lost to "graceful
processing" cascade closures (which would inflate the cascade cert
count without on-disk evidence and create a soundness hole).

Trade-off: workers exit before flushing `_publish_stats`, so the BnB's
final cert_count / closed_volume snapshot lags slightly behind reality.
We compensate by snapshotting `[par]` log lines at every emission.

OUTPUT FILES (per iteration directory)
--------------------------------------
  bnb.log           # full subprocess stdout
  bnb_metrics.jsonl # parsed [par] line per emission (timestamped)
  bnb_summary.json  # trigger reason, in_flight at trigger,
                    # cert_at_trigger, closed_vol_at_trigger,
                    # final cert and volume from last [par] line
  dumps/
    <prefix>_w*.npz
    <prefix>_master_queue.npz
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Regex for parsing the BnB master's `[par] t=...s ...` log lines
# --------------------------------------------------------------------------

# Example line:
# [par] t=  615.1s  nodes=     26000  cert=     12868  in_flight=   529  queue= 137  active=16/16  rate=      0/s  progress=100.00000%  eta=      0s
PAR_LINE_RE = re.compile(
    r'\[par\]\s+'
    r't=\s*([\d.]+)s\s+'
    r'nodes=\s*(\d+)\s+'
    r'cert=\s*(\d+)\s+'
    r'in_flight=\s*(\d+)\s+'
    r'queue=\s*(-?\d+)\s+'
    r'active=\s*(\d+)/(\d+)\s+'
    r'rate=\s*(\d+)/s\s+'
    r'progress=\s*([\d.]+)%'
)

# Final-state line emitted by master's finally block (after all workers
# joined). Used for soundness accounting: cert_at_exit reflects ALL
# cascade closures, including those during the SIGINT-handling delay.
# Format:
# [par] FINAL_STATE kbd_interrupt=1 failed=1 cert=16104 in_flight=243 nodes=33500 closed_volume=2.34e-04 elapsed_s=652.41
FINAL_STATE_RE = re.compile(
    r'\[par\]\s+FINAL_STATE\s+'
    r'kbd_interrupt=(\d+)\s+'
    r'failed=(\d+)\s+'
    r'cert=(\d+)\s+'
    r'in_flight=(\d+)\s+'
    r'nodes=(\d+)\s+'
    r'closed_volume=([\d.eE+-]+)\s+'
    r'elapsed_s=([\d.]+)'
)


# --------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------

@dataclass
class BnBPhaseConfig:
    """All knobs governing one BnB-phase iteration."""

    # Problem
    d: int
    target_str: str               # exact-rational, passed to BnB

    # BnB driver
    workers: int = 16
    init_split_depth: int = 22
    donate_threshold_floor: int = 2
    time_budget_s: int = 7200     # hard cap on a single iter (2h default)

    # Stuck-detector
    sample_period_s: float = 5.0  # match BnB master's [par] cadence
    warmup_s: int = 300           # ignore initial ramp samples
    stuck_window_s: int = 300     # min must be stable for this long
    max_inflight_for_dump: int = 10000  # always dump regardless of size

    # Cascade env vars (BnB cascade tier thresholds)
    topk_joint_depth: int = 14
    topk_joint_k: int = 3
    epigraph_depth: int = 24
    epigraph_filter: float = 0.02
    anchor_depth: int = 24
    centroid_depth: int = 60
    lp_split_depth: int = 26

    # Dump
    dump_prefix: str = 'bnb_dump'
    instant_dump: bool = True     # PUBLICATION-GRADE: instant SIGINT exit

    # Misc
    verbose: bool = True


# --------------------------------------------------------------------------
# Spawn helper
# --------------------------------------------------------------------------

def _spawn_bnb(cfg: BnBPhaseConfig, repo_root: Path) -> subprocess.Popen:
    """Launch the BnB cascade as a subprocess in its own process group."""
    env = dict(os.environ)
    env['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = str(cfg.topk_joint_depth)
    env['INTERVAL_BNB_TOPK_JOINT_K'] = str(cfg.topk_joint_k)
    env['INTERVAL_BNB_EPIGRAPH_DEPTH'] = str(cfg.epigraph_depth)
    env['INTERVAL_BNB_EPIGRAPH_FILTER'] = str(cfg.epigraph_filter)
    env['INTERVAL_BNB_ANCHOR_DEPTH'] = str(cfg.anchor_depth)
    env['INTERVAL_BNB_CENTROID_DEPTH'] = str(cfg.centroid_depth)
    env['INTERVAL_BNB_LP_SPLIT_DEPTH'] = str(cfg.lp_split_depth)
    env['INTERVAL_BNB_DUMP_BOXES'] = cfg.dump_prefix
    env['INTERVAL_BNB_INSTANT_DUMP'] = '1' if cfg.instant_dump else '0'
    env['PYTHONUNBUFFERED'] = '1'

    code = f'''
import sys, time
sys.path.insert(0, {repr(str(repo_root))})
from interval_bnb.parallel import parallel_branch_and_bound
t0 = time.time()
r = parallel_branch_and_bound(
    d={cfg.d}, target_c="{cfg.target_str}",
    workers={cfg.workers},
    init_split_depth={cfg.init_split_depth},
    donate_threshold_floor={cfg.donate_threshold_floor},
    time_budget_s={cfg.time_budget_s},
    verbose={cfg.verbose},
)
print(
    f"\\n=== BNB_RESULT: success={{r['success']}}"
    f" cov={{100*r['coverage_fraction']:.4f}}%"
    f" in_flight={{r['in_flight_final']}}"
    f" nodes={{r['total_nodes']}}"
    f" cert={{r['total_leaves_certified']}}"
    f" elapsed={{time.time()-t0:.0f}}s ==="
)
'''
    return subprocess.Popen(
        [sys.executable, '-u', '-c', code],
        env=env, cwd=str(repo_root),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
        preexec_fn=os.setsid if sys.platform != 'win32' else None,
    )


# --------------------------------------------------------------------------
# Stuck detector
# --------------------------------------------------------------------------

class StuckDetector:
    """Tracks the post-warmup running minimum of in_flight and reports
    "stuck" when min hasn't decreased for `stuck_window_s` seconds.

    Pre-warmup samples are excluded from the min (they're ramp transients
    where the queue is briefly empty).
    """

    def __init__(self, warmup_s: int, stuck_window_s: int,
                 max_inflight: int):
        self.warmup_s = warmup_s
        self.stuck_window_s = stuck_window_s
        self.max_inflight = max_inflight
        self.history: List[Tuple[float, int]] = []  # (t, in_flight)
        self.post_warmup_min: int = 10**12
        self.post_warmup_min_t: float = 0.0
        # All-time min (incl warmup) — diagnostic only
        self.all_time_min: int = 10**12
        self.all_time_min_t: float = 0.0

    def push(self, t: float, in_flight: int) -> None:
        self.history.append((t, in_flight))
        if in_flight < self.all_time_min:
            self.all_time_min = in_flight
            self.all_time_min_t = t
        if t >= self.warmup_s and in_flight < self.post_warmup_min:
            self.post_warmup_min = in_flight
            self.post_warmup_min_t = t

    def stuck(self) -> bool:
        if not self.history:
            return False
        t_now, cur = self.history[-1]
        if t_now < self.warmup_s + self.stuck_window_s:
            return False
        if cur > self.max_inflight:
            return False
        if self.post_warmup_min == 10**12:
            return False
        return (t_now - self.post_warmup_min_t) >= self.stuck_window_s

    def trigger_reason(self) -> str:
        if not self.history:
            return 'no-data'
        t_now, cur = self.history[-1]
        return (f'stuck@{t_now:.0f}s in_flight={cur} '
                f'pw_min={self.post_warmup_min}@{self.post_warmup_min_t:.0f}s '
                f'(stable {t_now - self.post_warmup_min_t:.0f}s)')


# --------------------------------------------------------------------------
# Phase runner
# --------------------------------------------------------------------------

@dataclass
class BnBPhaseResult:
    """Result of one BnB phase execution."""
    iter_dir: str
    trigger_reason: str
    elapsed_s: float
    last_par: Optional[Dict[str, Any]]   # the last parsed [par] line
    par_history: List[Dict[str, Any]]    # all parsed [par] lines
    dump_files: List[str]
    n_dumped_total: int                  # total raw boxes across all dumps
    n_dumped_per_file: Dict[str, int]
    cert_count_at_trigger: int
    closed_volume_at_trigger: float      # placeholder (BnB doesn't report yet)
    # FINAL_STATE — captured AFTER all workers exit. cert_count here
    # includes any boxes the cascade closed during the SIGINT-handling
    # delay window (which the [par] periodic log misses). Critical for
    # soundness accounting:
    #     cert_at_trigger + (final_state_cert − cert_at_trigger) +
    #         dumped_boxes + lost_boxes  ==  total_initial_boxes
    final_state: Optional[Dict[str, Any]] = None


def run_one_bnb_phase(cfg: BnBPhaseConfig, repo_root: Path,
                       iter_dir: Path) -> BnBPhaseResult:
    """Run BnB until stuck-trigger fires, then SIGINT for instant-dump.

    Writes `bnb.log` (raw stdout), `bnb_metrics.jsonl` (parsed [par]
    lines), `bnb_summary.json` (trigger reason + last-known cert).
    Returns a BnBPhaseResult populated from those files.
    """
    iter_dir.mkdir(parents=True, exist_ok=True)
    dumps_dir = iter_dir / 'dumps'
    dumps_dir.mkdir(parents=True, exist_ok=True)
    bnb_log = iter_dir / 'bnb.log'
    metrics_log = iter_dir / 'bnb_metrics.jsonl'

    # Override dump_prefix so files land in dumps_dir.
    # We pass a path-relative prefix; the BnB workers cd into repo_root
    # and write files there, so we need an absolute prefix or post-move.
    # We choose: prefix = absolute path inside dumps_dir.
    abs_prefix = str(dumps_dir / cfg.dump_prefix)
    cfg_local = BnBPhaseConfig(**asdict(cfg))
    cfg_local.dump_prefix = abs_prefix

    print(f"[bnb-phase] launching BnB d={cfg.d} target={cfg.target_str} "
          f"workers={cfg.workers} → {iter_dir}", flush=True)
    proc = _spawn_bnb(cfg_local, repo_root)

    detector = StuckDetector(
        warmup_s=cfg.warmup_s,
        stuck_window_s=cfg.stuck_window_s,
        max_inflight=cfg.max_inflight_for_dump,
    )
    par_history: List[Dict[str, Any]] = []
    last_par: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None
    trigger_reason = 'unknown'
    t0 = time.time()
    log_fh = open(bnb_log, 'w', encoding='utf-8')
    metrics_fh = open(metrics_log, 'w', encoding='utf-8')

    # NOTE: we DON'T break out of readline() on stuck-trigger anymore
    # — we send SIGINT and KEEP READING until the subprocess closes
    # stdout. This lets us capture the FINAL_STATE line printed by the
    # BnB master in its finally block (which contains cert_count after
    # all workers exited and SIGINT-delay closures completed).
    sigint_sent = False
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                rc = proc.poll()
                if not sigint_sent:
                    trigger_reason = f'subprocess_exit_rc={rc}'
                break
            log_fh.write(line)
            log_fh.flush()
            elapsed = time.time() - t0

            # FINAL_STATE detection — the BnB master's last word.
            fs = FINAL_STATE_RE.search(line)
            if fs:
                final_state = {
                    'kbd_interrupt': bool(int(fs.group(1))),
                    'failed': bool(int(fs.group(2))),
                    'cert': int(fs.group(3)),
                    'in_flight': int(fs.group(4)),
                    'nodes': int(fs.group(5)),
                    'closed_volume': float(fs.group(6)),
                    'elapsed_s': float(fs.group(7)),
                    'wall_t': elapsed,
                }
                metrics_fh.write(json.dumps({
                    'event': 'final_state', **final_state}) + '\n')
                metrics_fh.flush()
                print(f"  [bnb-phase] FINAL_STATE captured: "
                      f"cert={final_state['cert']} "
                      f"in_flight={final_state['in_flight']} "
                      f"nodes={final_state['nodes']}", flush=True)
                # Don't break — wait for actual subprocess exit so we
                # know the final-state line was the real terminator.

            # Parse [par] lines for in_flight tracking + metrics journal.
            m = PAR_LINE_RE.search(line)
            if m:
                par = {
                    'wall_t': elapsed,
                    'bnb_t': float(m.group(1)),
                    'nodes': int(m.group(2)),
                    'cert': int(m.group(3)),
                    'in_flight': int(m.group(4)),
                    'queue': int(m.group(5)),
                    'active': int(m.group(6)),
                    'workers': int(m.group(7)),
                    'rate': int(m.group(8)),
                    'progress_pct': float(m.group(9)),
                }
                last_par = par
                par_history.append(par)
                metrics_fh.write(json.dumps(par) + '\n')
                metrics_fh.flush()
                detector.push(elapsed, par['in_flight'])

                if len(par_history) % 6 == 0:
                    pw_age = (elapsed - detector.post_warmup_min_t
                              if detector.post_warmup_min < 10**12 else 0)
                    print(f"  [bnb-phase] t={elapsed:6.0f}s "
                          f"in_flight={par['in_flight']:>6d} "
                          f"cert={par['cert']:>7d} "
                          f"pw_min={detector.post_warmup_min} "
                          f"({pw_age:.0f}s ago)", flush=True)

                if detector.stuck() and not sigint_sent:
                    trigger_reason = detector.trigger_reason()
                    print(f"  [bnb-phase] !! STUCK: {trigger_reason}",
                          flush=True)
                    print(f"  [bnb-phase] sending SIGINT "
                          f"(instant-dump mode = {cfg.instant_dump})",
                          flush=True)
                    if proc.poll() is None:
                        try:
                            if sys.platform != 'win32':
                                os.killpg(os.getpgid(proc.pid),
                                            signal.SIGINT)
                            else:
                                proc.send_signal(signal.SIGINT)
                        except Exception as e:
                            print(f"    SIGINT failed: {e}", flush=True)
                    sigint_sent = True
                    # DO NOT break — keep reading stdout so we capture
                    # the FINAL_STATE line and any [par] lines emitted
                    # during master cleanup.
                    continue
                if elapsed > cfg.time_budget_s and not sigint_sent:
                    trigger_reason = f'time_budget_exceeded@{elapsed:.0f}s'
                    print(f"  [bnb-phase] !! time budget hit — SIGINT",
                          flush=True)
                    if proc.poll() is None:
                        try:
                            if sys.platform != 'win32':
                                os.killpg(os.getpgid(proc.pid),
                                            signal.SIGINT)
                            else:
                                proc.send_signal(signal.SIGINT)
                        except Exception as e:
                            print(f"    SIGINT failed: {e}", flush=True)
                    sigint_sent = True
                    continue
    finally:
        log_fh.close()
        metrics_fh.close()

    # If subprocess didn't exit on its own (rare — master usually returns
    # cleanly after KeyboardInterrupt path), force escalating signals.
    if proc.poll() is None:
        print(f"  [bnb-phase] subprocess still alive — escalating to SIGKILL",
              flush=True)
        try:
            if sys.platform != 'win32':
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass

    # Glob dump files. Note: prefix is the absolute path inside dumps_dir,
    # so we glob with the basename pattern.
    dump_basename = cfg.dump_prefix
    dump_files = sorted(dumps_dir.glob(f'{dump_basename}_*.npz'))
    n_per: Dict[str, int] = {}
    n_total = 0
    for f in dump_files:
        try:
            d = np.load(str(f), allow_pickle=True)
            n = d['lo'].shape[0] if 'lo' in d.files else 0
            n_per[f.name] = n
            n_total += n
        except Exception:
            n_per[f.name] = -1  # error sentinel
    print(f"  [bnb-phase] dump complete: {n_total} boxes across "
          f"{len(dump_files)} files", flush=True)

    # cert_count_at_trigger comes from the last [par] line we saw.
    cert_at_trigger = last_par['cert'] if last_par else 0
    closed_vol_at_trigger = 0.0  # BnB's [par] doesn't expose this; the
                                 # FINAL_STATE line below carries the
                                 # final closed_volume which is what
                                 # matters for soundness.

    # SOUNDNESS-ACCOUNTING ARITHMETIC (using FINAL_STATE if available)
    # ----------------------------------------------------------------
    # Let:
    #   C0 = cert_count at last [par] log (just before SIGINT)
    #   C1 = cert_count at FINAL_STATE (after all workers exited)
    #   D  = dumped boxes (across worker dumps + master_queue dump)
    #   I0 = in_flight at last [par] log
    # Then:
    #   cascade_closures_during_sigint_delay = C1 - C0
    #   total_closures_after_trigger         = (C1 - C0) + D
    #   apparent_loss = I0 - (C1 - C0) - D
    # apparent_loss SHOULD be ~0 (within queue-race noise of ~queue.qsize/2).
    if final_state is not None:
        c1 = final_state['cert']
        cascade_closures_after_trigger = c1 - cert_at_trigger
        i0 = (last_par['in_flight'] if last_par else 0)
        apparent_loss = i0 - cascade_closures_after_trigger - n_total
    else:
        cascade_closures_after_trigger = None
        i0 = (last_par['in_flight'] if last_par else 0)
        apparent_loss = None

    summary = {
        'trigger_reason': trigger_reason,
        'elapsed_s': time.time() - t0,
        'n_par_lines': len(par_history),
        'n_dump_files': len(dump_files),
        'n_dumped_total': n_total,
        'n_dumped_per_file': n_per,
        'last_par': last_par,
        'cert_count_at_trigger': cert_at_trigger,
        'closed_volume_at_trigger': closed_vol_at_trigger,
        'final_state': final_state,
        'soundness_accounting': {
            'cert_at_trigger_C0': cert_at_trigger,
            'cert_at_final_state_C1': (final_state['cert']
                                        if final_state else None),
            'cascade_closures_during_sigint_delay': cascade_closures_after_trigger,
            'in_flight_at_trigger_I0': i0,
            'dumped_boxes_D': n_total,
            'apparent_loss_I0_minus_cascadeDelta_minus_D': apparent_loss,
            'note': ('apparent_loss should be ~0 within queue-race noise; '
                     'large positive values indicate boxes neither captured '
                     'nor cert-closed → genuine soundness gap. '
                     'final_state=null means BnB master died before printing '
                     'FINAL_STATE — accounting cannot be verified.'),
        },
        'detector': {
            'all_time_min': detector.all_time_min,
            'all_time_min_t': detector.all_time_min_t,
            'post_warmup_min': (None if detector.post_warmup_min == 10**12
                                else detector.post_warmup_min),
            'post_warmup_min_t': detector.post_warmup_min_t,
        },
        'config': asdict(cfg),
    }
    (iter_dir / 'bnb_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))

    if final_state is not None:
        print(f"  [bnb-phase] SOUNDNESS: cert_at_trigger={cert_at_trigger} "
              f"cascade_closures_during_sigint_delay={cascade_closures_after_trigger} "
              f"dumped={n_total} apparent_loss={apparent_loss}", flush=True)
    else:
        print(f"  [bnb-phase] SOUNDNESS: NO FINAL_STATE captured — "
              f"cert_at_trigger={cert_at_trigger}, dumped={n_total}, "
              f"in_flight_at_trigger={i0} (apparent loss UNVERIFIED)",
              flush=True)

    return BnBPhaseResult(
        iter_dir=str(iter_dir),
        trigger_reason=trigger_reason,
        elapsed_s=summary['elapsed_s'],
        last_par=last_par,
        par_history=par_history,
        dump_files=[str(f) for f in dump_files],
        n_dumped_total=n_total,
        n_dumped_per_file=n_per,
        cert_count_at_trigger=cert_at_trigger,
        closed_volume_at_trigger=closed_vol_at_trigger,
        final_state=final_state,
    )
