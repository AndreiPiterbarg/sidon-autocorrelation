#!/usr/bin/env python
"""Memory-culprit probe for the monolithic Lasserre SOS-dual pipeline.

Runs ONE probe of ``solve_mosek_dual`` (no bisection) and reports peak
RSS plus RSS at each phase — precompute, Z/2 canonicalisation (if any),
task build, param apply, optimize.  Designed to map out which knob
actually controls memory at d=10/12/16.

Phases are delimited by explicit sync points; a background thread
samples /proc/self/status every 100ms and records the peak.

Usage (single run):
    python tests/probe_mem.py --d 12 --order 3 --threads 32 \\
        --t-val 1.28 --json data/probe_d12_t32.json

Usage (sweep, bash):
    for t in 1 4 16 32 64 128 ; do
      python tests/probe_mem.py --d 12 --threads $t \\
         --json data/probe_d12_t${t}.json
    done

Flags you care about (memory levers):
    --threads N        MOSEK num_threads (per-thread scratch is the
                       dominant term at high d — main lever to test)
    --z2-full          canonicalize_z2 + blockdiag moment + σ-rep
                       (theoretical ~2× memory cut)
    --upper-loc        add the (1-μ_i) ≥ 0 cones (doubles loc bars)
    --solve-form       'dual' / 'primal' / 'free' (affects MOSEK
                       internal storage layout)
    --tol              IPM tolerance (loose tol = fewer iters, same mem
                       per iter; checks whether mem grows during IPM)
    --build-only       build + apply_params, DO NOT optimize.  Isolates
                       MOSEK internal IPM-scratch from task-data memory.
    --active-windows K only add the first K windows (from nontrivial
                       list).  Tests how much memory comes from windows.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import mosek


# =====================================================================
# RSS sampler
# =====================================================================

def _rss_bytes() -> int:
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


class PeakSampler:
    def __init__(self, interval_s: float = 0.1):
        self.interval = interval_s
        self.peak = 0
        self._phase_peaks: List[Tuple[str, int]] = []
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._cur_phase = 'init'
        self._phase_peak_cur = 0

    def _loop(self):
        while not self._stop.wait(self.interval):
            r = _rss_bytes()
            if r > self.peak:
                self.peak = r
            if r > self._phase_peak_cur:
                self._phase_peak_cur = r

    def start(self):
        self.peak = _rss_bytes()
        self._phase_peak_cur = self.peak
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def mark(self, name: str):
        """Record peak since last mark, reset phase counter."""
        cur = _rss_bytes()
        if cur > self.peak:
            self.peak = cur
        if cur > self._phase_peak_cur:
            self._phase_peak_cur = cur
        self._phase_peaks.append((self._cur_phase, self._phase_peak_cur))
        self._cur_phase = name
        self._phase_peak_cur = cur

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)
        self.mark('end')
        return {
            'peak_bytes': self.peak,
            'peak_gb': self.peak / (1024 ** 3),
            'phases': [
                {'phase': p, 'peak_bytes': b, 'peak_gb': b / (1024 ** 3)}
                for p, b in self._phase_peaks
            ],
        }


# =====================================================================
# Probe driver
# =====================================================================

def run_probe(args) -> Dict[str, Any]:
    from lasserre_scalable import _precompute
    from lasserre.dual_sdp import build_dual_task, solve_dual_task
    from lasserre.z2_elim import canonicalize_z2
    from lasserre.z2_blockdiag import (
        build_blockdiag_picks, localizing_sigma_reps, window_sigma_reps,
    )
    from lasserre_mosek_tuned import val_d_known

    d = int(args.d)
    order = int(args.order)
    t_val = float(args.t_val) if args.t_val is not None \
        else (val_d_known.get(d, 1.3) + 0.02)

    sampler = PeakSampler()
    sampler.start()
    t0 = time.time()

    print(f'[probe] d={d} order={order} threads={args.threads} '
          f't={t_val:.4f}  z2_full={args.z2_full}  '
          f'upper_loc={args.upper_loc}  solve_form={args.solve_form}  '
          f'tol={args.tol}  build_only={args.build_only}  '
          f'active_windows={args.active_windows}', flush=True)

    phase_times: Dict[str, float] = {}

    # --- Phase: import + mosek Env ---
    sampler.mark('mosek_env')
    env = mosek.Env()

    # --- Phase: precompute ---
    tp0 = time.time()
    sampler.mark('precompute')
    P_raw = _precompute(d, order, verbose=False, lazy_ab_eiej=args.lazy_ab)
    phase_times['precompute_s'] = time.time() - tp0
    print(f'  precompute: {phase_times["precompute_s"]:.2f}s  '
          f'RSS={_rss_bytes() / 1024**3:.2f}GB', flush=True)

    # --- Phase: Z/2 stack (if enabled) ---
    if args.z2_full:
        tp0 = time.time()
        sampler.mark('z2_canonicalize')
        P = canonicalize_z2(P_raw, verbose=False)
        sampler.mark('z2_blockdiag_picks')
        bd = build_blockdiag_picks(P['basis'], P['idx'], P['n_y'])
        loc_fixed, loc_pairs = localizing_sigma_reps(d)
        active_loc = list(loc_fixed) + [p for p, _ in loc_pairs]
        win_fixed, win_pairs = window_sigma_reps(d, P_raw['windows'])
        nontriv = set(P_raw['nontrivial_windows'])
        active_windows = [w for w in
                          (list(win_fixed) + [p for p, _ in win_pairs])
                          if w in nontriv]
        phase_times['z2_s'] = time.time() - tp0
        print(f'  z2_full: n_y {P_raw["n_y"]} -> {P["n_y"]}   '
              f'loc {d} -> {len(active_loc)}   '
              f'win {len(nontriv)} -> {len(active_windows)}   '
              f'({phase_times["z2_s"]:.2f}s, '
              f'RSS={_rss_bytes() / 1024**3:.2f}GB)', flush=True)
    else:
        P = P_raw
        bd = None
        active_loc = None
        active_windows = None

    # Optional: truncate window list for sensitivity test.
    if args.active_windows is not None and args.active_windows > 0:
        src = list(P['nontrivial_windows']) if active_windows is None \
            else list(active_windows)
        active_windows = src[:int(args.active_windows)]
        print(f'  active_windows truncated: {len(active_windows)} '
              f'(of {len(src)})', flush=True)

    # --- Phase: build task ---
    tp0 = time.time()
    sampler.mark('build_task')
    task, info = build_dual_task(
        P, t_val=t_val, env=env,
        include_upper_loc=args.upper_loc,
        z2_blockdiag_map=bd,
        active_loc=active_loc,
        active_windows=active_windows,
        lambda_upper_bound=1.0,
        verbose=False)
    phase_times['build_s'] = time.time() - tp0
    rss_post_build = _rss_bytes()
    print(f'  build_dual_task: {phase_times["build_s"]:.2f}s  '
          f'RSS={rss_post_build / 1024**3:.2f}GB  '
          f'bars={info["n_bar"]}  n_scalar={info["n_scalar"]:,}  '
          f'n_cons={info["n_cons"]:,}  '
          f'n_bar_entries={info["n_bar_entries"]:,}',
          flush=True)

    # --- Phase: apply params ---
    sampler.mark('apply_params')
    sf = args.solve_form.lower()
    if sf == 'dual':
        task.putintparam(mosek.iparam.intpnt_solve_form,
                         mosek.solveform.dual)
    elif sf == 'primal':
        task.putintparam(mosek.iparam.intpnt_solve_form,
                         mosek.solveform.primal)
    else:
        task.putintparam(mosek.iparam.intpnt_solve_form,
                         mosek.solveform.free)
    task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, args.tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, args.tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, args.tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_mu_red, args.tol)
    task.putintparam(mosek.iparam.intpnt_max_iterations, 1600)
    task.putintparam(mosek.iparam.num_threads, int(args.threads))
    try:
        task.putintparam(mosek.iparam.intpnt_order_method,
                         mosek.orderingtype.graphpar)
    except Exception:
        pass

    if args.build_only:
        sampler.mark('skipped_optimize')
        phase_times['optimize_s'] = 0.0
        verdict = 'skipped'
        lam_star = float('nan')
        solsta = 'skipped'
    else:
        # --- Phase: optimize ---
        tp0 = time.time()
        sampler.mark('optimize')
        try:
            task.optimize()
            phase_times['optimize_s'] = time.time() - tp0
            solsta = str(task.getsolsta(mosek.soltype.itr)).split('.')[-1]
            lam_star = float(task.getxxslice(
                mosek.soltype.itr,
                info['LAMBDA_IDX'], info['LAMBDA_IDX'] + 1)[0])
            if lam_star >= 0.75:
                verdict = 'infeas'
            elif lam_star <= 0.25:
                verdict = 'feas'
            else:
                verdict = 'uncertain'
        except Exception as e:
            phase_times['optimize_s'] = time.time() - tp0
            solsta = f'ERR:{type(e).__name__}'
            lam_star = float('nan')
            verdict = 'error'
        print(f'  optimize: {phase_times["optimize_s"]:.2f}s  '
              f'RSS={_rss_bytes() / 1024**3:.2f}GB  '
              f'solsta={solsta}  lam*={lam_star:.4e}  verdict={verdict}',
              flush=True)

    # --- Teardown ---
    try:
        task.__del__()
    except Exception:
        pass
    del task, info, P, P_raw
    gc.collect()

    total_wall = time.time() - t0
    samp = sampler.stop()

    print(f'\n[probe] TOTAL wall={total_wall:.2f}s  '
          f'PEAK={samp["peak_gb"]:.2f}GB', flush=True)
    for p in samp['phases']:
        print(f'  phase {p["phase"]:<24s} peak={p["peak_gb"]:.2f}GB',
              flush=True)

    return {
        'd': d, 'order': order,
        't_val': t_val,
        'threads': int(args.threads),
        'z2_full': bool(args.z2_full),
        'upper_loc': bool(args.upper_loc),
        'solve_form': sf,
        'tol': float(args.tol),
        'build_only': bool(args.build_only),
        'active_windows_limit': args.active_windows,
        'lazy_ab_eiej': bool(args.lazy_ab),
        'phase_times_s': phase_times,
        'total_wall_s': total_wall,
        'peak_rss_bytes': samp['peak_bytes'],
        'peak_rss_gb': samp['peak_gb'],
        'rss_phases': samp['phases'],
        'n_bar': info['n_bar'] if 'info' in dir() else None,
        'verdict': locals().get('verdict', 'n/a'),
        'lambda_star': locals().get('lam_star', float('nan')),
        'solsta': locals().get('solsta', 'n/a'),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--t-val', type=float, default=None,
                   help='Probe t (default: val_d_known[d] + 0.02).')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--z2-full', action='store_true')
    p.add_argument('--upper-loc', action='store_true')
    p.add_argument('--solve-form', type=str, default='dual',
                   choices=('primal', 'dual', 'free'))
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--build-only', action='store_true',
                   help='Skip task.optimize(); isolate build memory.')
    p.add_argument('--active-windows', type=int, default=None,
                   help='Limit to first K nontrivial windows.')
    p.add_argument('--lazy-ab', action='store_true',
                   help='Use lazy ab_eiej_idx (skip 4D array).')
    p.add_argument('--json', type=str, default=None)
    args = p.parse_args()

    r = run_probe(args)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.',
                    exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f'\n  JSON -> {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
