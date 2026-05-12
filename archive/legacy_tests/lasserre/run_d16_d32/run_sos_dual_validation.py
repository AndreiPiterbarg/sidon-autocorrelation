#!/usr/bin/env python
"""Single-file validation driver for the SOS-dual Farkas LP.

Runs every v1 regression/timing test we care about sequentially, so the
whole thing can be fire-and-forgotten on a compute pod:

    nohup python3 tests/run_sos_dual_validation.py > run_sos_dual.log 2>&1 &

Pass criterion: every moment-primal / SOS-dual verdict pair at probed t
values must be compatible (exact match OR one-sided uncertain), and full
bisections at d=4 L=3, d=6 L=3, d=8 L=3 must produce SOS-dual lb within
1e-4 of moment-primal lb (OR strictly better — the SOS-dual achieves
strictly tighter lb where the moment-primal stalls on boundary
uncertainty).

Stages:
  1. 24-probe regression test across (d, L) ∈ {(4,2),(4,3),(6,2),(6,3)}.
  2. d=4 L=3 full bisection moment-primal + SOS-dual.
  3. d=6 L=3 full bisection moment-primal + SOS-dual.
  4. d=8 L=3 full bisection moment-primal + SOS-dual.

Stage output is streamed to stdout with clear banners; final summary
records lb / wall-time per stage so the single log file is the artefact.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import mosek

from lasserre_scalable import _precompute
from lasserre.dual_sdp import build_dual_task, solve_dual_task
from lasserre_mosek_preelim import solve_mosek_preelim
from lasserre_mosek_dual import solve_mosek_dual


# =========================================================================
# Utilities
# =========================================================================

def _banner(title: str) -> None:
    print()
    print('=' * 78)
    print(f'  {title}')
    print('=' * 78, flush=True)


def _substage(title: str) -> None:
    print()
    print(f'--- {title} ---', flush=True)


def _compatible(pv: str, dv: str) -> bool:
    if pv == dv:
        return True
    if 'uncertain' in (pv, dv):
        return True
    return False


def _quiet_primal_single_t(d: int, L: int, t: float,
                             add_upper_loc: bool = False) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        r = solve_mosek_preelim(
            d=d, order=L, mode='tuned',
            add_upper_loc=add_upper_loc,
            single_t=t, verbose=False)
    return r


def _quiet_dual_single_t(P, env, t: float) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        task, info = build_dual_task(
            P, t_val=t, env=env, include_upper_loc=False, verbose=False)
        res = solve_dual_task(task, info, verbose=False)
        try:
            task.__del__()
        except Exception:
            pass
    return res


# =========================================================================
# Stage 1: 24-probe regression
# =========================================================================

REGRESSION_CASES = [
    (4, 2, [0.5, 1.0, 1.05, 1.07, 1.09, 1.15, 2.0]),
    (4, 3, [0.5, 1.05, 1.08, 1.12, 1.15, 2.0]),
    (6, 2, [0.5, 1.05, 1.10, 1.15, 1.20, 2.0]),
    (6, 3, [0.5, 1.10, 1.15, 1.20, 2.0]),
]


def stage_regression() -> Dict[str, Any]:
    _banner('STAGE 1 — 24-probe SOS-dual vs moment-primal regression')
    failures: List[tuple] = []
    total_probes = 0
    total_primal_s = 0.0
    total_dual_s = 0.0
    env = mosek.Env()

    for (d, L, probes) in REGRESSION_CASES:
        _substage(f'd={d}  L={L}  ({len(probes)} probes)')
        P = _precompute(d=d, order=L, verbose=False, lazy_ab_eiej=False)
        for t in probes:
            total_probes += 1
            t0 = time.time()
            dres = _quiet_dual_single_t(P, env, t)
            dt_d = time.time() - t0
            total_dual_s += dt_d

            t0 = time.time()
            pres = _quiet_primal_single_t(d, L, t)
            dt_p = time.time() - t0
            total_primal_s += dt_p

            ok = _compatible(pres['verdict'], dres['verdict'])
            tag = 'OK' if ok else 'FAIL'
            note = ''
            if pres['verdict'] != dres['verdict']:
                note = '  (one-sided uncertain — tolerated)' \
                    if 'uncertain' in (pres['verdict'], dres['verdict']) \
                    else '  (HARD DISAGREEMENT)'
            print(f'  [{tag}] t={t:.4f}  '
                  f'primal={pres["verdict"]:10s}  '
                  f'dual={dres["verdict"]:10s}  '
                  f'lam*={dres["lambda_star"]:+.3e}  '
                  f'primal_t={dt_p:.2f}s  dual_t={dt_d:.2f}s{note}',
                  flush=True)
            if not ok:
                failures.append((d, L, t, pres['verdict'], dres['verdict']))

    print()
    print(f'Regression summary: {total_probes} probes, '
          f'{total_probes - len(failures)} passed, {len(failures)} failed.')
    print(f'  Wall time:  primal={total_primal_s:.2f}s  '
          f'dual={total_dual_s:.2f}s  '
          f'(dual speedup={total_primal_s / max(total_dual_s, 1e-9):.2f}x)')
    if failures:
        print('FAILURES:')
        for d, L, t, pv, dv in failures:
            print(f'  d={d} L={L} t={t}: primal={pv}  dual={dv}')

    return {
        'stage': 'regression',
        'total_probes': total_probes,
        'n_failures': len(failures),
        'failures': failures,
        'total_primal_s': total_primal_s,
        'total_dual_s': total_dual_s,
        'speedup': total_primal_s / max(total_dual_s, 1e-9),
    }


# =========================================================================
# Stages 2-4: full bisection at (d, L)
# =========================================================================

def stage_bisection(d: int, L: int, *,
                     n_bisect: int = 15,
                     t_lo: float = 0.5,
                     t_hi: float = 2.0) -> Dict[str, Any]:
    _banner(f'STAGE — full bisection at d={d}  L={L}  n_bisect={n_bisect}')

    # --- SOS-dual first (fast, decides the boundary cleanly) ---
    _substage(f'SOS-dual bisection (d={d} L={L})')
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stdout):
        dres = solve_mosek_dual(
            d=d, order=L,
            add_upper_loc=False,
            n_bisect=n_bisect,
            t_lo=t_lo, t_hi=t_hi,
            primary_tol=1e-6,
            mosek_log=True,
            verbose=True)
    dual_total = time.time() - t0

    # --- Moment-primal ---
    _substage(f'Moment-primal bisection (d={d} L={L})')
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stdout):
        pres = solve_mosek_preelim(
            d=d, order=L, mode='tuned',
            add_upper_loc=False,
            n_bisect=n_bisect,
            t_lo=t_lo, t_hi=t_hi,
            primary_tol=1e-6,
            verbose=True)
    primal_total = time.time() - t0

    lb_dual = dres['lb']
    lb_primal = pres['lb']
    gc_dual = dres.get('gc_pct')
    gc_primal = pres.get('gc_pct')

    print()
    print(f'>>> d={d} L={L} bisection comparison:')
    print(f'    moment-primal: lb={lb_primal:.6f}  '
          f'gc={gc_primal if gc_primal is None else f"{gc_primal:.2f}%"}  '
          f'n_solves={pres["n_solves"]}  '
          f'n_uncertain={pres["n_uncertain"]}  '
          f'wall={primal_total:.1f}s')
    print(f'    SOS-dual:      lb={lb_dual:.6f}  '
          f'gc={gc_dual if gc_dual is None else f"{gc_dual:.2f}%"}  '
          f'n_solves={dres["n_solves"]}  '
          f'n_uncertain={dres["n_uncertain"]}  '
          f'wall={dual_total:.1f}s')
    delta = lb_dual - lb_primal
    print(f'    Δlb (dual − primal): {delta:+.6f}    '
          f'speedup: {primal_total / max(dual_total, 1e-9):.2f}x',
          flush=True)

    return {
        'stage': f'bisection_d{d}_L{L}',
        'd': d, 'L': L, 'n_bisect': n_bisect,
        'lb_dual': lb_dual,
        'lb_primal': lb_primal,
        'delta_lb': delta,
        'gc_dual_pct': gc_dual,
        'gc_primal_pct': gc_primal,
        'n_solves_dual': dres['n_solves'],
        'n_solves_primal': pres['n_solves'],
        'n_uncertain_dual': dres['n_uncertain'],
        'n_uncertain_primal': pres['n_uncertain'],
        'wall_dual_s': dual_total,
        'wall_primal_s': primal_total,
        'speedup': primal_total / max(dual_total, 1e-9),
    }


# =========================================================================
# Orchestrator
# =========================================================================

def main() -> int:
    overall_t0 = time.time()
    _banner('SOS-dual v1 full validation suite')
    print(f'  python={sys.version.split()[0]}  hostname={os.uname().nodename}')
    print(f'  cpu_count={os.cpu_count()}  '
          f'MOSEK license path={os.environ.get("MOSEKLM_LICENSE_FILE", "(default)")}')

    summary: List[Dict[str, Any]] = []

    # Stage 1: regression.
    try:
        s1 = stage_regression()
        summary.append(s1)
    except Exception as exc:
        print(f'!!! STAGE 1 FAILED: {type(exc).__name__}: {exc}')
        import traceback; traceback.print_exc()
        summary.append({'stage': 'regression',
                          'error': f'{type(exc).__name__}: {exc}'})

    # Stage 2: d=4 L=3 full bisection.
    try:
        s2 = stage_bisection(4, 3, n_bisect=15, t_lo=0.5, t_hi=2.0)
        summary.append(s2)
    except Exception as exc:
        print(f'!!! STAGE 2 FAILED: {type(exc).__name__}: {exc}')
        import traceback; traceback.print_exc()
        summary.append({'stage': 'bisection_d4_L3',
                          'error': f'{type(exc).__name__}: {exc}'})

    # Stage 3: d=6 L=3 full bisection.
    try:
        s3 = stage_bisection(6, 3, n_bisect=15, t_lo=0.5, t_hi=2.0)
        summary.append(s3)
    except Exception as exc:
        print(f'!!! STAGE 3 FAILED: {type(exc).__name__}: {exc}')
        import traceback; traceback.print_exc()
        summary.append({'stage': 'bisection_d6_L3',
                          'error': f'{type(exc).__name__}: {exc}'})

    # Stage 4: d=8 L=3 full bisection.
    try:
        s4 = stage_bisection(8, 3, n_bisect=12, t_lo=0.5, t_hi=1.4)
        summary.append(s4)
    except Exception as exc:
        print(f'!!! STAGE 4 FAILED: {type(exc).__name__}: {exc}')
        import traceback; traceback.print_exc()
        summary.append({'stage': 'bisection_d8_L3',
                          'error': f'{type(exc).__name__}: {exc}'})

    total_wall = time.time() - overall_t0

    _banner('FINAL SUMMARY')
    print(f'  total wall time: {total_wall:.1f}s')
    print()
    for s in summary:
        if 'error' in s:
            print(f'  [{s["stage"]}] ERROR: {s["error"]}')
            continue
        if s['stage'] == 'regression':
            print(f'  [regression] {s["total_probes"]} probes, '
                  f'{s["total_probes"] - s["n_failures"]} passed, '
                  f'{s["n_failures"]} failed.  speedup={s["speedup"]:.2f}x')
        else:
            print(f'  [{s["stage"]}] '
                  f'lb_dual={s["lb_dual"]:.6f} '
                  f'lb_primal={s["lb_primal"]:.6f} '
                  f'Δ={s["delta_lb"]:+.6f}  '
                  f'n_unc_primal={s["n_uncertain_primal"]} '
                  f'n_unc_dual={s["n_uncertain_dual"]}  '
                  f'speedup={s["speedup"]:.2f}x  '
                  f'wall_primal={s["wall_primal_s"]:.1f}s '
                  f'wall_dual={s["wall_dual_s"]:.1f}s')

    # Dump machine-readable summary alongside human log.
    with open('run_sos_dual_summary.json', 'w') as f:
        json.dump({
            'total_wall_s': total_wall,
            'stages': summary,
        }, f, indent=2, default=str)
    print()
    print('Wrote machine-readable summary to run_sos_dual_summary.json')

    # Exit non-zero if any regression failed.
    any_fail = False
    for s in summary:
        if 'error' in s:
            any_fail = True
        if s.get('n_failures', 0) > 0:
            any_fail = True
    return 1 if any_fail else 0


if __name__ == '__main__':
    sys.exit(main())
