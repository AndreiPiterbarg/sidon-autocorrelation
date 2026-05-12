#!/usr/bin/env python3
"""Benchmark sweep: find the fastest SOS-dual config for d=16 L=3.

We optimise full 15-step bisection wall-clock on a 128-core pod.  A d=16
bisection is multi-hour per config, so we screen at cheaper d, fit a
power-law in m = n_cons = C(d+6,6), and project to d=16 per-config.

Design (non-redundant by construction):
  • z2_full=True, upper_loc=True, reuse_task=True always on (algorithmic
    wins proven at d=6; re-measuring without them burns pod time).
  • solve_form=primal default (won locally); one dual control included.
  • Serial num_threads<64 skipped — Schur at d=16 is 74613^2 dense, so
    under-threaded BLAS is strictly worse.
  • Parallel total_threads=128 (use the machine).

For each (config, d) we measure:
  • Py build_s              — time in build_dual_task
  • factor_setup_s          — MOSEK factor setup time (parsed from log)
  • iter0_TIME_s            — MOSEK pre-IPM wall (factor setup + Schur setup)
  • optimizer_total_s       — MOSEK optimize() wall
  • ipm_iters               — number of IPM iterations that ran
  • wall_total_s            — full probe wall (incl Py build)

Across d values we fit T = A · m^p for each *component* of each config.
Projection to d=16 uses the per-config exponent.

Configs (6, all with z2_full=True, upper_loc=True):
  S-primal-64    serial,    primal,  64 threads   (baseline)
  S-primal-128   serial,    primal, 128 threads   (does MOSEK saturate?)
  S-dual-64      serial,    dual,    64 threads   (form control)
  P2-64t         parallel,  primal,  2 workers × 64t = 128
  P4-32t         parallel,  primal,  4 workers × 32t = 128
  P8-16t         parallel,  primal,  8 workers × 16t = 128

For parallel configs we run a single tier; the tier-wall drives projection
via n_tiers(k) = ceil(15 / log2(k+1)).

Typical invocation on the pod (default screens at d=8,10,12 — ~2 h total):
  python3 tests/bench_d16_config.py --configs all --json bench_d16.json

Smaller screen (≈30 min):
  python3 tests/bench_d16_config.py --d-list 8,10 --configs all --json bench_fast.json

Verify winner at d=14 before launching d=16:
  python3 tests/bench_d16_config.py --d-list 14 --configs <winner> --json bench_d14.json
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from lasserre_mosek_dual import solve_mosek_dual, solve_mosek_dual_parallel
from lasserre.core import val_d_known


# ---------------------------------------------------------------------------
# Config catalogue
# ---------------------------------------------------------------------------
CONFIGS: Dict[str, Dict[str, Any]] = {
    'S-primal-64':  {'mode': 'serial',   'solve_form': 'primal', 'threads':  64},
    'S-primal-128': {'mode': 'serial',   'solve_form': 'primal', 'threads': 128},
    'S-primal-192': {'mode': 'serial',   'solve_form': 'primal', 'threads': 192},
    'S-dual-64':    {'mode': 'serial',   'solve_form': 'dual',   'threads':  64},
    'P2-64t':       {'mode': 'parallel', 'solve_form': 'primal',
                     'n_parallel': 2, 'total_threads': 128},
    'P2-96t':       {'mode': 'parallel', 'solve_form': 'primal',
                     'n_parallel': 2, 'total_threads': 192},
    'P3-64t':       {'mode': 'parallel', 'solve_form': 'primal',
                     'n_parallel': 3, 'total_threads': 192},
    'P4-32t':       {'mode': 'parallel', 'solve_form': 'primal',
                     'n_parallel': 4, 'total_threads': 128},
    'P4-48t':       {'mode': 'parallel', 'solve_form': 'primal',
                     'n_parallel': 4, 'total_threads': 192},
    'P6-32t':       {'mode': 'parallel', 'solve_form': 'primal',
                     'n_parallel': 6, 'total_threads': 192},
    'P8-16t':       {'mode': 'parallel', 'solve_form': 'primal',
                     'n_parallel': 8, 'total_threads': 128},
}

TIERS_FOR_FULL: Dict[int, int] = {2: 10, 3: 8, 4: 7, 6: 6, 8: 5}


# ---------------------------------------------------------------------------
# n_cons = C(d+6, 6) closed form (L=3 always)
# ---------------------------------------------------------------------------
def m_of_d(d: int) -> int:
    # C(d+6, 6)
    num = 1
    for i in range(6):
        num *= (d + 6 - i)
    den = 720  # 6!
    return num // den


# ---------------------------------------------------------------------------
# MOSEK log parsers
# ---------------------------------------------------------------------------
_RE_BUILD = re.compile(r'\[dual-task\].*?build=([\d.]+)s')
_RE_FACTOR_SETUP = re.compile(r'Factor\s+-\s+setup time\s*:\s*([\d.]+)')
# MOSEK IPM log line format:  "0  <pfeas>  <dfeas>  <gfeas>  <prstatus>  <pobj>  <dobj>  <mu>  <time>"
_RE_IPM_ITER = re.compile(
    r'^\s*(\d+)\s+[-+.eE\d]+\s+[-+.eE\d]+\s+[-+.eE\d]+\s+'
    r'[-+.eE\d]+\s+[-+.eE\d]+\s+[-+.eE\d]+\s+[-+.eE\d]+\s+([\d.]+)\s*$',
    re.MULTILINE)
_RE_OPTIMIZER_TERM = re.compile(r'Optimizer terminated\.\s*Time:\s*([\d.]+)')


def _parse_serial_log(log_text: str) -> Dict[str, Any]:
    """Extract first-probe timing breakdown from a serial SOS-dual log."""
    build_m = _RE_BUILD.search(log_text)
    fs_m = _RE_FACTOR_SETUP.search(log_text)
    iters = _RE_IPM_ITER.findall(log_text)
    opt_m = _RE_OPTIMIZER_TERM.search(log_text)

    iter0_time = float(iters[0][1]) if iters else None
    last_iter_num = int(iters[-1][0]) if iters else None
    return {
        'build_s':          float(build_m.group(1)) if build_m else None,
        'factor_setup_s':   float(fs_m.group(1)) if fs_m else None,
        'iter0_TIME_s':     iter0_time,
        'ipm_iters':        last_iter_num,
        'optimizer_total_s':float(opt_m.group(1)) if opt_m else None,
        'pre_ipm_s':        (((float(build_m.group(1)) if build_m else 0)
                              + (iter0_time or 0)) if iter0_time else None),
    }


# ---------------------------------------------------------------------------
# Single (config, d) runners
# ---------------------------------------------------------------------------
def _run_serial(d: int, cfg: Dict[str, Any],
                 t_probe: float, n_probes: int) -> Dict[str, Any]:
    per_probe: List[Dict[str, Any]] = []
    for i in range(n_probes):
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            r = solve_mosek_dual(
                d=d, order=3,
                add_upper_loc=True,
                z2_full=True,
                single_t=float(t_probe),
                solve_form=cfg['solve_form'],
                num_threads=cfg['threads'],
                mosek_log=True,           # need the MOSEK log to parse
                verbose=True)
        wall = time.time() - t0
        log_text = buf.getvalue()
        sys.stdout.write(log_text)        # echo so it lands in the run log
        parsed = _parse_serial_log(log_text)
        per_probe.append({
            'probe_index': i,
            'wall_s': wall,
            'verdict': r.get('verdict'),
            'status': r.get('status'),
            'lambda_star': r.get('lambda_star'),
            **parsed,
        })

    def _mean(key: str) -> Optional[float]:
        vals = [p[key] for p in per_probe if p.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    return {
        'mode': 'serial',
        'd': d, 'm': m_of_d(d),
        'n_probes': n_probes,
        't_probe': float(t_probe),
        'wall_mean_s':            _mean('wall_s'),
        'build_mean_s':           _mean('build_s'),
        'factor_setup_mean_s':    _mean('factor_setup_s'),
        'iter0_TIME_mean_s':      _mean('iter0_TIME_s'),
        'optimizer_total_mean_s': _mean('optimizer_total_s'),
        'pre_ipm_mean_s':         _mean('pre_ipm_s'),
        'ipm_iters':              per_probe[-1].get('ipm_iters') if per_probe else None,
        'per_probe': per_probe,
    }


def _run_parallel(d: int, cfg: Dict[str, Any], n_probes: int) -> Dict[str, Any]:
    val = val_d_known.get(d) or 1.5
    lo, hi = val - 0.25, val + 0.05

    tiers: List[Dict[str, Any]] = []
    for i in range(n_probes):
        t0 = time.time()
        r = solve_mosek_dual_parallel(
            d=d, order=3,
            add_upper_loc=True,
            z2_full=True,
            n_tiers=1,
            n_parallel=cfg['n_parallel'],
            t_lo=lo, t_hi=hi,
            solve_form=cfg['solve_form'],
            total_threads=cfg['total_threads'],
            verbose=True)
        wall = time.time() - t0
        tier_walls = [h['wall_s'] for h in r.get('history', [])
                      if h.get('tier') == 0]
        tier_wall = max(tier_walls) if tier_walls else wall / 2
        hi_wall = max(wall - tier_wall, 0.0)
        tiers.append({
            'probe_index': i,
            'wall_total_s': wall,
            'wall_hi_s': hi_wall,
            'wall_tier_s': tier_wall,
            'inner_threads': r.get('inner_threads'),
        })

    def _mean(key: str) -> Optional[float]:
        vals = [t[key] for t in tiers if t.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    return {
        'mode': 'parallel',
        'd': d, 'm': m_of_d(d),
        'n_probes_ran': n_probes,
        'n_parallel': cfg['n_parallel'],
        'total_threads': cfg['total_threads'],
        'inner_threads': tiers[0].get('inner_threads') if tiers else None,
        'wall_mean_s':          _mean('wall_total_s'),
        'wall_hi_mean_s':       _mean('wall_hi_s'),
        'wall_tier_mean_s':     _mean('wall_tier_s'),
        'per_tier': tiers,
    }


# ---------------------------------------------------------------------------
# Scaling fit and projection
# ---------------------------------------------------------------------------
def _fit_power_law(ms: List[int], ys: List[float]) -> Tuple[float, float, float]:
    """Fit log y = log A + p log m on >=2 points.  Returns (A, p, max_rel_resid).

    max_rel_resid = max_i |y_i_pred - y_i| / y_i  (crude fit-quality proxy).
    With 2 points the fit is exact (resid 0 by construction).  With 3+ we
    do least squares and report residuals.
    """
    assert len(ms) == len(ys) and len(ms) >= 2
    xs = [math.log(m) for m in ms]
    ls = [math.log(max(y, 1e-12)) for y in ys]
    n = len(xs)
    xm = sum(xs) / n
    ym = sum(ls) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    sxy = sum((x - xm) * (l - ym) for x, l in zip(xs, ls))
    if sxx < 1e-18:
        return (math.exp(ym), 0.0, 0.0)
    p = sxy / sxx
    logA = ym - p * xm
    A = math.exp(logA)
    preds = [A * (m ** p) for m in ms]
    resids = [abs(pr - y) / max(y, 1e-12) for pr, y in zip(preds, ys)]
    return (A, p, max(resids))


def _project_config(entries: List[Dict[str, Any]], d_target: int,
                      cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Given per-d entries for one config, project to d_target."""
    m_target = m_of_d(d_target)
    # Collect training points for the relevant metric.
    ms = [e['m'] for e in entries]
    if cfg['mode'] == 'serial':
        ys = [e['wall_mean_s'] for e in entries]
        A, p, resid = _fit_power_law(ms, ys)
        t_probe = A * (m_target ** p)
        proj = 15.0 * t_probe
        return {
            'target_d': d_target, 'target_m': m_target,
            'fit_A': A, 'fit_p': p, 'fit_max_rel_resid': resid,
            'proj_probe_s': t_probe,
            'proj_bisection_s': proj,
        }
    # Parallel: tier wall scales with m; hi_probe wall also scales.  Fit each.
    tier_ys = [e['wall_tier_mean_s'] for e in entries]
    hi_ys = [e['wall_hi_mean_s'] for e in entries]
    A_t, p_t, resid_t = _fit_power_law(ms, tier_ys)
    A_h, p_h, resid_h = _fit_power_law(ms, hi_ys)
    t_tier = A_t * (m_target ** p_t)
    t_hi = A_h * (m_target ** p_h)
    k = cfg['n_parallel']
    n_tiers = TIERS_FOR_FULL[k]
    proj = t_hi + n_tiers * t_tier
    return {
        'target_d': d_target, 'target_m': m_target,
        'fit_A_tier': A_t, 'fit_p_tier': p_t, 'fit_max_rel_resid_tier': resid_t,
        'fit_A_hi':   A_h, 'fit_p_hi':   p_h, 'fit_max_rel_resid_hi':   resid_h,
        'proj_tier_s': t_tier,
        'proj_hi_s':   t_hi,
        'proj_bisection_s': proj,
    }


def _recommended_cli(d_target: int, cfg_name: str,
                      cfg: Dict[str, Any]) -> str:
    base = (f"python3 tests/lasserre_mosek_dual.py "
            f"--d {d_target} --order 3 "
            f"--z2-full --upper-loc "
            f"--solve-form {cfg['solve_form']}")
    if cfg['mode'] == 'serial':
        return (f"{base} --num-threads {cfg['threads']} --n-bisect 15 "
                f"--json sos_dual_d{d_target}_L3_{cfg_name}.json "
                f"> sos_dual_d{d_target}_L3_{cfg_name}.log 2>&1")
    k = cfg['n_parallel']
    n_tiers = TIERS_FOR_FULL[k]
    return (f"{base} --parallel {k} --n-tiers {n_tiers} "
            f"--total-threads {cfg['total_threads']} "
            f"--json sos_dual_d{d_target}_L3_{cfg_name}.json "
            f"> sos_dual_d{d_target}_L3_{cfg_name}.log 2>&1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d-list', type=str, default='8,10,12',
                    help='Comma-separated d values to sweep (default: 8,10,12).')
    p.add_argument('--configs', type=str, default='all',
                    help='"all" or comma-separated config names.')
    p.add_argument('--n-probes', type=int, default=1,
                    help='Repeats per (config, d) for noise floor (default 1; '
                         'MOSEK is deterministic so 1 is usually fine).')
    p.add_argument('--t-probe', type=float, default=None,
                    help='Serial single_t value (default: val_d + 0.05).')
    p.add_argument('--d-target', type=int, default=16)
    p.add_argument('--json', type=str, default=None)
    args = p.parse_args()

    ds = [int(x) for x in args.d_list.split(',') if x.strip()]
    if args.configs == 'all':
        names = list(CONFIGS.keys())
    else:
        names = [s.strip() for s in args.configs.split(',') if s.strip()]
        for n in names:
            if n not in CONFIGS:
                print(f"ERROR: unknown config {n!r}. "
                      f"Known: {list(CONFIGS.keys())}", file=sys.stderr)
                return 2

    print(f"\n{'#' * 76}")
    print(f"#  bench_d16_config")
    print(f"#  d_list={ds}   configs={names}   n_probes={args.n_probes}")
    print(f"#  d_target={args.d_target} (m={m_of_d(args.d_target)})")
    print(f"#  cpus={os.cpu_count()}")
    for d in ds:
        print(f"#    d={d}: m={m_of_d(d)}  val={val_d_known.get(d)}")
    print(f"{'#' * 76}\n", flush=True)

    overall_t0 = time.time()
    # results[name][d] -> entry dict
    results: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in names}

    # Outer loop: d (lower d first — fast warmup + fail-fast).
    # Inner loop: config.
    for d in ds:
        val = val_d_known.get(d)
        t_probe = args.t_probe if args.t_probe is not None else (
            (val + 0.05) if val else 1.5)
        print(f"\n{'=' * 76}")
        print(f"  d={d}  m={m_of_d(d)}  t_probe={t_probe:.6f}")
        print(f"{'=' * 76}", flush=True)

        for name in names:
            cfg = CONFIGS[name]
            print(f"\n--- RUN d={d} config={name} cfg={cfg} ---", flush=True)
            try:
                if cfg['mode'] == 'serial':
                    entry = _run_serial(d, cfg, t_probe, args.n_probes)
                else:
                    entry = _run_parallel(d, cfg, args.n_probes)
                entry['config'] = name
                entry['cfg'] = dict(cfg)
                results[name][d] = entry
                key = 'wall_mean_s'
                print(f"  >> {name} d={d}: wall_mean={entry[key]:.2f}s"
                      + (f"  build={entry['build_mean_s']:.2f}s "
                         f"factor_setup={entry['factor_setup_mean_s']:.2f}s "
                         f"iter0={entry['iter0_TIME_mean_s']:.2f}s "
                         f"opt={entry['optimizer_total_mean_s']:.2f}s "
                         f"iters={entry['ipm_iters']}"
                         if cfg['mode'] == 'serial' else
                         f"  tier={entry['wall_tier_mean_s']:.2f}s "
                         f"hi={entry['wall_hi_mean_s']:.2f}s "
                         f"inner_threads={entry['inner_threads']}"),
                      flush=True)
            except Exception as exc:
                results[name][d] = {'config': name, 'cfg': dict(cfg),
                                      'd': d, 'm': m_of_d(d),
                                      'error': f"{type(exc).__name__}: {exc}"}
                print(f"  !! {name} d={d} FAILED: {results[name][d]['error']}",
                      flush=True)
                traceback.print_exc()
            gc.collect()

    total_wall = time.time() - overall_t0

    # -----------------------------------------------------------------
    # Scaling fits + projection
    # -----------------------------------------------------------------
    projections: List[Dict[str, Any]] = []
    for name in names:
        cfg = CONFIGS[name]
        entries = [results[name][d] for d in ds
                    if d in results[name] and 'error' not in results[name][d]]
        if len(entries) < 2:
            projections.append({
                'config': name, 'error': 'need ≥2 d points for fit',
                'n_points': len(entries),
            })
            continue
        proj = _project_config(entries, args.d_target, cfg)
        proj['config'] = name
        proj['n_points'] = len(entries)
        proj['cfg'] = dict(cfg)
        proj['points'] = [{'d': e['d'], 'm': e['m'],
                           'wall_mean_s': e.get('wall_mean_s')}
                          for e in entries]
        projections.append(proj)

    ok = [p for p in projections if 'error' not in p]
    ok.sort(key=lambda r: r['proj_bisection_s'])

    # -----------------------------------------------------------------
    # Print tables
    # -----------------------------------------------------------------
    print(f"\n{'=' * 96}")
    print(f"  OBSERVED wall_mean_s per (config, d)  "
          f"— bench_wall={total_wall:.1f}s")
    print(f"{'=' * 96}")
    header = f"  {'config':<14}"
    for d in ds:
        header += f"  {'d='+str(d):>12}"
    header += f"  {'err?':<6}"
    print(header)
    for name in names:
        row = f"  {name:<14}"
        for d in ds:
            e = results[name].get(d, {})
            if 'error' in e:
                row += f"  {'ERR':>12}"
            elif 'wall_mean_s' in e and e['wall_mean_s'] is not None:
                row += f"  {e['wall_mean_s']:>12.2f}"
            else:
                row += f"  {'—':>12}"
        err = next((e['error'] for e in results[name].values()
                    if 'error' in e), '')
        row += f"  {err[:40]}"
        print(row)

    print(f"\n{'=' * 96}")
    print(f"  PROJECTION to d={args.d_target} (m={m_of_d(args.d_target)})"
          f" — ranked by projected 15-level-equivalent bisection wall")
    print(f"{'=' * 96}")
    print(f"  {'rank':<4} {'config':<14} {'mode':<9} {'pts':<4} "
          f"{'fit_p':>8} {'max_resid':>10} {'proj_probe/tier':>16} "
          f"{'proj_bisection':>16}")
    for i, r in enumerate(ok, 1):
        if r['cfg']['mode'] == 'serial':
            p_fit = r['fit_p']; resid = r['fit_max_rel_resid']
            per = r['proj_probe_s']
        else:
            p_fit = r['fit_p_tier']; resid = r['fit_max_rel_resid_tier']
            per = r['proj_tier_s']
        print(f"  {i:<4} {r['config']:<14} {r['cfg']['mode']:<9} "
              f"{r['n_points']:<4} {p_fit:>8.3f} {resid:>10.2%} "
              f"{per:>16.1f} {r['proj_bisection_s']:>16.1f}")

    if ok:
        winner = ok[0]
        w_cfg = CONFIGS[winner['config']]
        cli = _recommended_cli(args.d_target, winner['config'], w_cfg)
        print(f"\n{'=' * 96}")
        print(f"  RECOMMENDED for d={args.d_target}: "
              f"{winner['config']}  "
              f"(projected {winner['proj_bisection_s']:.0f}s = "
              f"{winner['proj_bisection_s']/60:.1f} min)")
        print(f"{'=' * 96}")
        print(f"\n{cli}\n", flush=True)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'd_list': ds,
                'configs_run': names,
                'd_target': args.d_target,
                'n_probes': args.n_probes,
                'total_bench_wall_s': total_wall,
                'results': results,
                'projections': projections,
                'recommended_cli': (_recommended_cli(
                    args.d_target, ok[0]['config'], CONFIGS[ok[0]['config']])
                    if ok else None),
            }, f, indent=2, default=str)
        print(f"Wrote {args.json}", flush=True)

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_main())
