#!/usr/bin/env python
"""Parallel bisection driver for MOSEK Lasserre solves.

MOSEK's Factor-setup phase is serial by design: no parameter parallelizes
it.  But MOSEK itself can run as a subprocess, and many subprocesses at
DIFFERENT t-values run independently.  We fan out N feas-checks per round
across N cores / processes.  Wall-clock per round ≈ one serial solve; the
round bracket resolution is 2^{-ceil(log2 N)} ≈ 1/N, which would take
log2(N) serial bisection steps.  At N=16 one round ≈ 4 serial steps.

Usage:

    # Fire 16 single-t probes across t ∈ [0.8, 1.5], 16 processes at once.
    python tests/parallel_bisect.py --d 16 --order 3 \
           --mode z2_full --pre-elim-z2 \
           --n-parallel 16 --rounds 3 \
           --t-lo 0.8 --t-hi 1.5 \
           --lazy-ab-eiej

Each subprocess prints a `SINGLE_T_VERDICT ...` line; the launcher greps
it, merges verdicts across the probe grid, narrows [lo, hi] to the tightest
interval consistent with the verdicts, and repeats for `--rounds` rounds.

Concurrency safety:
  * MOSEK's license may cap concurrent sessions.  Verify with your licence
    file; typical machine licences allow arbitrary processes on one host.
  * Memory: each subprocess loads its own Python+NumPy+MOSEK state.  At
    d=16 L=3 z2_full ~1-3 GB per process is a reasonable plan; ensure
    N_PARALLEL * RAM_PER_PROC < total RAM.
  * Per-process numThreads is set via env MOSEK_NUM_THREADS_PER_PROC.
    Total concurrent threads ~ N_PARALLEL * threads_per_proc.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


_VERDICT_RE = re.compile(
    r'SINGLE_T_VERDICT\s+t=([\d.eE+-]+)\s+verdict=(\S+)\s+status=(\S+)'
    r'\s+wall_s=([\d.eE+-]+)')


@dataclass
class Probe:
    t: float
    verdict: Optional[str] = None     # 'feas' | 'infeas' | 'uncertain'
    wall_s: Optional[float] = None
    status: Optional[str] = None
    returncode: Optional[int] = None


def _ladder(lo: float, hi: float, n: int) -> List[float]:
    """Return n t-values uniformly spaced in (lo, hi), strictly interior."""
    if n <= 0:
        return []
    if n == 1:
        return [0.5 * (lo + hi)]
    step = (hi - lo) / (n + 1)
    return [lo + step * (k + 1) for k in range(n)]


def _build_cmd(args: argparse.Namespace, t_val: float,
               threads_per_proc: int) -> List[str]:
    """Assemble an lasserre_mosek_preelim.py invocation for one t-value."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'lasserre_mosek_preelim.py')
    cmd = [
        sys.executable, script,
        '--d', str(args.d),
        '--order', str(args.order),
        '--mode', args.mode,
        '--single-t', f'{t_val:.10f}',
        '--n-bisect', '0',
        '--primary-tol', f'{args.primary_tol:.0e}',
        '--max-fill-ratio', str(args.max_fill_ratio),
        '--protect-degrees', args.protect_degrees,
        '--order-method', args.order_method,
        '--watcher-interval', str(args.watcher_interval),
    ]
    if args.pre_elim_z2:
        cmd.append('--pre-elim-z2')
    if args.no_upper_loc:
        cmd.append('--no-upper-loc')
    if args.lindep_off:
        cmd.append('--lindep-off')
    if args.lazy_ab_eiej:
        cmd.append('--lazy-ab-eiej')
    return cmd


def _run_round(args: argparse.Namespace, ts: List[float],
               threads_per_proc: int, log_dir: Optional[str],
               round_idx: int) -> List[Probe]:
    """Fire N subprocesses concurrently; collect SINGLE_T_VERDICT lines."""
    probes = [Probe(t=t) for t in ts]
    procs: List[Tuple[subprocess.Popen, Probe, str]] = []
    env = os.environ.copy()
    # Per-process MOSEK thread cap so total concurrent threads ≈ physical.
    env['OMP_NUM_THREADS'] = str(threads_per_proc)
    env['MKL_NUM_THREADS'] = str(threads_per_proc)
    env['OPENBLAS_NUM_THREADS'] = str(threads_per_proc)
    env['SIDON_IJ_WORKERS'] = str(max(1, threads_per_proc))
    env['MOSEK_NUM_THREADS'] = str(threads_per_proc)

    for probe in probes:
        cmd = _build_cmd(args, probe.t, threads_per_proc)
        if log_dir:
            log_path = os.path.join(
                log_dir,
                f'r{round_idx:02d}_t{probe.t:.6f}.log')
            lf = open(log_path, 'w')
        else:
            log_path = None
            lf = subprocess.PIPE
        p = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            env=env, text=True)
        procs.append((p, probe, log_path))

    t_start = time.time()
    # Wait for all; optionally tail progress
    for p, probe, log_path in procs:
        rc = p.wait()
        probe.returncode = rc
        # Parse the SINGLE_T_VERDICT line from the log
        if log_path:
            try:
                text = open(log_path).read()
                m = _VERDICT_RE.search(text)
                if m:
                    probe.verdict = m.group(2)
                    probe.status = m.group(3)
                    probe.wall_s = float(m.group(4))
            except Exception as e:
                probe.verdict = 'uncertain'
                probe.status = f'logparse:{type(e).__name__}'
        elif p.stdout is not None:
            # shouldn't happen with log_dir set; keep for completeness
            text = p.stdout.read() if not p.stdout.closed else ''
            m = _VERDICT_RE.search(text or '')
            if m:
                probe.verdict = m.group(2)
                probe.status = m.group(3)
                probe.wall_s = float(m.group(4))
        if probe.verdict is None:
            probe.verdict = 'uncertain'
            probe.status = f'rc={rc}'
    t_elapsed = time.time() - t_start
    return probes


def _narrow_bracket(lo: float, hi: float,
                    probes: List[Probe]) -> Tuple[float, float,
                                                    Dict[str, int]]:
    """Update [lo, hi] from a batch of probe verdicts.

    Rule: lb (== lo) is the largest t with verdict=infeas; ub (== hi) is
    the smallest t with verdict=feas.  'uncertain' verdicts are ignored
    for bracket updates but counted in stats.
    """
    stats = {'feas': 0, 'infeas': 0, 'uncertain': 0}
    for p in probes:
        stats[p.verdict] = stats.get(p.verdict, 0) + 1
    infeas_ts = sorted(p.t for p in probes if p.verdict == 'infeas')
    feas_ts   = sorted(p.t for p in probes if p.verdict == 'feas')
    new_lo = max([lo] + infeas_ts)
    new_hi = min([hi] + feas_ts)
    if new_lo >= new_hi:
        # conflicting verdicts — pick the widest containing bracket.
        # Re-expand using only the verdicts consistent with the old bracket.
        new_lo = lo
        new_hi = hi
        for p in probes:
            if p.verdict == 'infeas' and lo < p.t < hi:
                new_lo = max(new_lo, p.t)
        for p in probes:
            if p.verdict == 'feas' and lo < p.t < hi:
                new_hi = min(new_hi, p.t)
    return new_lo, new_hi, stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--mode',
                    choices=('baseline', 'tuned', 'z2_eq', 'z2_bd',
                              'z2_full'),
                    default='z2_full')
    p.add_argument('--pre-elim-z2', action='store_true', default=True)
    p.add_argument('--no-pre-elim-z2', dest='pre_elim_z2',
                    action='store_false')
    p.add_argument('--no-upper-loc', action='store_true')
    p.add_argument('--t-lo', type=float, default=1.0)
    p.add_argument('--t-hi', type=float, default=None,
                    help='Upper bracket. Default = val(d) + 0.05.')
    p.add_argument('--n-parallel', type=int, default=16,
                    help='Parallel t-probes per round.  Each round = one '
                         '(serial) Factor-setup wall-clock.')
    p.add_argument('--rounds', type=int, default=3,
                    help='Number of parallel rounds.  After R rounds the '
                         'bracket shrinks by ~(n_parallel+1)^R.')
    p.add_argument('--threads-per-proc', type=int, default=None,
                    help='MOSEK threads per subprocess.  Default '
                         'max(1, cpu_count // n_parallel).')
    p.add_argument('--primary-tol', type=float, default=1e-6)
    p.add_argument('--max-fill-ratio', type=float, default=10.0)
    p.add_argument('--protect-degrees', type=str, default='1,2')
    p.add_argument('--order-method', type=str, default='forceGraphpar')
    p.add_argument('--lindep-off', action='store_true')
    p.add_argument('--lazy-ab-eiej', action='store_true', default=True)
    p.add_argument('--no-lazy-ab-eiej', dest='lazy_ab_eiej',
                    action='store_false')
    p.add_argument('--watcher-interval', type=float, default=60.0)
    p.add_argument('--log-dir', type=str,
                    default=None,
                    help='Directory for per-probe MOSEK logs (required).')
    p.add_argument('--conv-tol', type=float, default=1e-4,
                    help='Stop when hi - lo < conv_tol.')
    args = p.parse_args()

    if args.log_dir is None:
        args.log_dir = f'data/parallel_bisect_d{args.d}_L{args.order}_{args.mode}'
    os.makedirs(args.log_dir, exist_ok=True)

    if args.t_hi is None:
        from lasserre.core import val_d_known
        args.t_hi = float(val_d_known.get(args.d, 2.0)) + 0.05

    if args.threads_per_proc is None:
        ncpu = os.cpu_count() or 8
        args.threads_per_proc = max(1, ncpu // args.n_parallel)

    print(f"Parallel bisection  d={args.d}  L={args.order}  mode={args.mode}")
    print(f"  N parallel = {args.n_parallel}, rounds = {args.rounds}, "
          f"threads/proc = {args.threads_per_proc}")
    print(f"  initial bracket: [{args.t_lo:.6f}, {args.t_hi:.6f}]")
    print(f"  log dir: {args.log_dir}", flush=True)

    lo = float(args.t_lo)
    hi = float(args.t_hi)
    history: List[Dict] = []
    t0 = time.time()

    for r in range(args.rounds):
        ts = _ladder(lo, hi, args.n_parallel)
        print(f"\n[round {r+1}/{args.rounds}]  bracket=[{lo:.6f},{hi:.6f}]"
              f"  probing t in [{ts[0]:.4f}..{ts[-1]:.4f}] "
              f"({len(ts)} points)", flush=True)
        t_round = time.time()
        probes = _run_round(args, ts, args.threads_per_proc,
                            args.log_dir, r + 1)
        new_lo, new_hi, stats = _narrow_bracket(lo, hi, probes)
        round_wall = time.time() - t_round
        max_wall = max((p.wall_s or 0) for p in probes) if probes else 0
        history.append({
            'round': r + 1,
            'lo_in': lo, 'hi_in': hi,
            'lo_out': new_lo, 'hi_out': new_hi,
            'stats': stats,
            'round_wall_s': round_wall,
            'max_probe_wall_s': max_wall,
            'probes': [{'t': p.t, 'verdict': p.verdict,
                         'wall_s': p.wall_s, 'status': p.status}
                        for p in probes],
        })
        print(f"  verdicts: feas={stats.get('feas',0)}, "
              f"infeas={stats.get('infeas',0)}, "
              f"uncertain={stats.get('uncertain',0)}  "
              f"round_wall={round_wall:.1f}s, "
              f"slowest_probe={max_wall:.1f}s", flush=True)
        print(f"  new bracket: [{new_lo:.6f}, {new_hi:.6f}]  "
              f"width={new_hi-new_lo:.3e}", flush=True)
        lo, hi = new_lo, new_hi
        if hi - lo < args.conv_tol:
            print(f"  converged (width < {args.conv_tol})", flush=True)
            break

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"final  lb={lo:.10f}  ub={hi:.10f}  "
          f"width={hi-lo:.3e}  total_wall={total:.1f}s")
    print(f"{'=' * 60}")

    # Write summary JSON next to logs.
    import json as _j
    summary_path = os.path.join(args.log_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        _j.dump({
            'args': vars(args),
            'final_lo': lo,
            'final_hi': hi,
            'total_wall_s': total,
            'history': history,
        }, f, indent=2, default=str)
    print(f"summary: {summary_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
