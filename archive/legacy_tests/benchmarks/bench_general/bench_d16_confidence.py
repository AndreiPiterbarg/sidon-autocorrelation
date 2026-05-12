#!/usr/bin/env python3
"""Confidence-tightening bench: nail the best d=16 config in ~55 min.

Runs ONLY the gaps left by bench_d16_config.py, plus two new d values
chosen to anchor the scaling fit closer to d=16.

Inputs:
  - Existing results from bench_d16.json (if present) — loaded and merged.

New measurements (budget: ~55 min):
  • d=11 (NEW d, m=12,376): all 6 configs, 1 probe each             ≈ 18 min
  • d=12 P4-32t           : single tier (fills last session's crash) ≈  8 min
  • d=13 (NEW d, m=27,132): S-primal-64 + P2-64t (top 2)             ≈ 28 min

This yields, per config, the following anchor points for the fit:
  S-primal-64:  d = 8, 10, 11, 12, 13           5 points
  S-primal-128: d = 8, 10, 11, 12               4 points
  S-dual-64:    d = 8, 10, 11, 12               4 points
  P2-64t:       d = 8, 10, 11, 12, 13           5 points
  P4-32t:       d = 8, 10, 11, 12               4 points
  P8-16t:       d = 8, 10, 11                   3 points

5 points over d=8..13 (m=3k..27k) is strong for a 2-param power-law fit
that extrapolates to d=16 (m=75k, ~2.75× the largest training m).

Skipped (won't change the ranking, don't burn the budget):
  • d=12 P8-16t — P8 already lost at d=10 by a wide margin.
  • d=13 P4-32t — P2 beat P4 at d=12 and the gap widens with d.
  • d=13 P8-16t, S-primal-128, S-dual-64 — not in the top-2.
  • Anything at d=14/d=16 — single probe alone exceeds the budget.

Outputs:
  • Merged wall-time table (all d × all configs).
  • Per-config power-law fit with residuals (fit_p, max_rel_resid).
  • Projected d=16 bisection wall *with 95% CI* from residual scatter.
  • Final RECOMMEND block with exact CLI for the full d=16 bisection.
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

# Re-use the infrastructure we built last round.
from bench_d16_config import (
    CONFIGS, TIERS_FOR_FULL, m_of_d,
    _run_serial, _run_parallel,
    _recommended_cli,
)


# ---------------------------------------------------------------------------
# Plan: exactly what to run, and the d at which to run it.
# ---------------------------------------------------------------------------
# Each entry is (d, config_name).  Kept explicit so the cost is obvious.
PLAN: List[Tuple[int, str]] = [
    # d=11 — new anchor point, cheap (~18 min total)
    (11, 'S-primal-64'),
    (11, 'S-primal-128'),
    (11, 'S-dual-64'),
    (11, 'P2-64t'),
    (11, 'P4-32t'),
    (11, 'P8-16t'),
    # d=12 — fill missing P4-32t (P8 skipped; won't win)
    (12, 'P4-32t'),
    # d=13 — new high-d anchor for the top 2 only
    (13, 'S-primal-64'),
    (13, 'P2-64t'),
]


# ---------------------------------------------------------------------------
# Power-law fit with CI (from residual scatter on log scale)
# ---------------------------------------------------------------------------
def _fit_power_law_with_ci(ms: List[int], ys: List[float]
                             ) -> Dict[str, float]:
    """Fit log y = log A + p log m, return {A, p, max_rel_resid,
    rmse_log, rss}.

    95% CI on the exponent p uses a Student-t factor on the log-scale
    residual std.  With n≥3 points we get a meaningful interval.
    """
    assert len(ms) == len(ys) and len(ms) >= 2
    xs = [math.log(m) for m in ms]
    ls = [math.log(max(y, 1e-12)) for y in ys]
    n = len(xs)
    xm = sum(xs) / n
    ym = sum(ls) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    sxy = sum((x - xm) * (l - ym) for x, l in zip(xs, ls))
    p = sxy / max(sxx, 1e-18)
    logA = ym - p * xm
    A = math.exp(logA)
    preds_log = [logA + p * x for x in xs]
    log_resids = [ls[i] - preds_log[i] for i in range(n)]
    rss = sum(r * r for r in log_resids)
    rmse_log = math.sqrt(rss / max(n - 2, 1))  # residual std
    # Crude CI: ±1.96 * SE(p) assuming Gaussian residuals.
    se_p = rmse_log / math.sqrt(max(sxx, 1e-18))
    preds_lin = [A * m ** p for m in ms]
    rel_resids = [abs(preds_lin[i] - ys[i]) / max(ys[i], 1e-12)
                  for i in range(n)]
    return {
        'A': A, 'p': p,
        'max_rel_resid': max(rel_resids),
        'rmse_log': rmse_log,
        'se_p': se_p,
        'ci95_p_lo': p - 1.96 * se_p,
        'ci95_p_hi': p + 1.96 * se_p,
        'n': n,
    }


def _project_with_ci(fit: Dict[str, float], m_target: int) -> Dict[str, float]:
    """Project y at m_target using fit; 95% CI from propagated log-scale std."""
    p = fit['p']; A = fit['A']
    y_hat = A * (m_target ** p)
    # Log-space variance at m_target: var(log y_hat) ≈ rmse_log^2 (residual-only;
    # ignores parameter correlation, which is fine for a ±50% ballpark CI).
    log_std = fit['rmse_log']
    y_lo = y_hat * math.exp(-1.96 * log_std)
    y_hi = y_hat * math.exp(+1.96 * log_std)
    return {'point': y_hat, 'lo95': y_lo, 'hi95': y_hi}


# ---------------------------------------------------------------------------
# Load prior results (from bench_d16.json, if provided)
# ---------------------------------------------------------------------------
def _load_prior(path: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        j = json.load(f)
    # Old format: {"results": {config_name: {d_str: entry}}}
    raw = j.get('results', {})
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for name, by_d in raw.items():
        out[name] = {}
        for d_key, e in by_d.items():
            try:
                d = int(d_key)
            except (TypeError, ValueError):
                continue
            if isinstance(e, dict) and 'error' not in e:
                out[name][d] = e
    return out


# ---------------------------------------------------------------------------
# Run a single (d, config) step, with timing.
# ---------------------------------------------------------------------------
def _run_step(d: int, name: str) -> Dict[str, Any]:
    cfg = CONFIGS[name]
    val = None
    try:
        from lasserre.core import val_d_known
        val = val_d_known.get(d)
    except Exception:
        pass
    t_probe = (val + 0.05) if val else 1.5
    if cfg['mode'] == 'serial':
        return _run_serial(d, cfg, t_probe, n_probes=1)
    return _run_parallel(d, cfg, n_probes=1)


def _projected_bisection(cfg: Dict[str, Any], fits: Dict[str, Dict[str, float]],
                           m_target: int) -> Dict[str, float]:
    """Pull per-metric fits together into a projected bisection wall."""
    if cfg['mode'] == 'serial':
        pw = _project_with_ci(fits['wall'], m_target)
        return {
            'probe_point': pw['point'], 'probe_lo95': pw['lo95'],
            'probe_hi95': pw['hi95'],
            'bisection_point': 15 * pw['point'],
            'bisection_lo95':  15 * pw['lo95'],
            'bisection_hi95':  15 * pw['hi95'],
        }
    k = cfg['n_parallel']
    n_tiers = TIERS_FOR_FULL[k]
    th = _project_with_ci(fits['tier'], m_target)
    hh = _project_with_ci(fits['hi'], m_target)
    # NOTE: hi and tier errors are correlated; treating independent here is
    # approximate but not wild — both fit the same problem size.
    return {
        'tier_point': th['point'], 'tier_lo95': th['lo95'], 'tier_hi95': th['hi95'],
        'hi_point': hh['point'], 'hi_lo95': hh['lo95'], 'hi_hi95': hh['hi95'],
        'bisection_point': hh['point'] + n_tiers * th['point'],
        'bisection_lo95':  hh['lo95']  + n_tiers * th['lo95'],
        'bisection_hi95':  hh['hi95']  + n_tiers * th['hi95'],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-json', type=str, default='bench_d16.json',
                    help='Existing bench_d16_config.py JSON to merge with.')
    p.add_argument('--d-target', type=int, default=16)
    p.add_argument('--json', type=str, default='bench_d16_confidence.json')
    p.add_argument('--skip-new', action='store_true',
                    help='Do not run new measurements; just re-fit from prior JSON.')
    args = p.parse_args()

    # ---- Load prior data ----
    prior = _load_prior(args.prior_json)
    print(f"\n{'#' * 76}")
    print(f"#  bench_d16_confidence")
    print(f"#  d_target={args.d_target}  m_target={m_of_d(args.d_target)}")
    print(f"#  prior JSON: {args.prior_json}")
    prior_count = sum(len(v) for v in prior.values())
    print(f"#  loaded {prior_count} prior (config, d) entries "
          f"across {len(prior)} configs")
    print(f"#  plan: {len(PLAN)} new (d, config) runs")
    print(f"{'#' * 76}\n", flush=True)

    # ---- Run the plan ----
    # Merge new into prior under same structure.
    merged: Dict[str, Dict[int, Dict[str, Any]]] = {
        n: dict(prior.get(n, {})) for n in CONFIGS
    }
    overall_t0 = time.time()

    if not args.skip_new:
        for i, (d, name) in enumerate(PLAN, 1):
            elapsed = time.time() - overall_t0
            print(f"\n{'=' * 76}")
            print(f"[{i}/{len(PLAN)}] d={d}  config={name}  "
                  f"(elapsed {elapsed/60:.1f} min)")
            print(f"{'=' * 76}", flush=True)
            try:
                entry = _run_step(d, name)
                entry['config'] = name
                entry['cfg'] = dict(CONFIGS[name])
                merged.setdefault(name, {})[d] = entry
                key = 'wall_mean_s'
                if CONFIGS[name]['mode'] == 'serial':
                    print(f"  >> {name} d={d}: wall={entry[key]:.2f}s  "
                          f"build={entry.get('build_mean_s')}s  "
                          f"factor_setup={entry.get('factor_setup_mean_s')}s  "
                          f"iter0={entry.get('iter0_TIME_mean_s')}s",
                          flush=True)
                else:
                    print(f"  >> {name} d={d}: wall={entry[key]:.2f}s  "
                          f"tier={entry.get('wall_tier_mean_s'):.2f}s  "
                          f"hi={entry.get('wall_hi_mean_s'):.2f}s",
                          flush=True)
            except Exception as exc:
                merged.setdefault(name, {})[d] = {
                    'config': name, 'cfg': dict(CONFIGS[name]),
                    'd': d, 'error': f"{type(exc).__name__}: {exc}",
                }
                print(f"  !! FAILED: {merged[name][d]['error']}", flush=True)
                traceback.print_exc()
            gc.collect()

    total_wall = time.time() - overall_t0

    # ---- Fit each config & project ----
    all_ds = sorted({d for by_d in merged.values() for d in by_d})

    summary: List[Dict[str, Any]] = []
    for name in CONFIGS:
        cfg = CONFIGS[name]
        entries = [e for d, e in sorted(merged.get(name, {}).items())
                    if 'error' not in e and 'd' in (e if 'd' in e else {'d': d})
                    and e.get('wall_mean_s') is not None]
        # Filter for entries that belong to this config (safety).
        entries = [e for e in entries if e.get('config') == name
                    or e.get('cfg', {}).get('mode') == cfg['mode']]
        if len(entries) < 2:
            summary.append({'config': name, 'n_points': len(entries),
                              'error': 'need ≥2 points'})
            continue
        ms = [e['m'] if 'm' in e else m_of_d(e['d']) for e in entries]
        if cfg['mode'] == 'serial':
            ys = [e['wall_mean_s'] for e in entries]
            fits = {'wall': _fit_power_law_with_ci(ms, ys)}
        else:
            ts = [e['wall_tier_mean_s'] for e in entries]
            hs = [e['wall_hi_mean_s'] for e in entries]
            fits = {'tier': _fit_power_law_with_ci(ms, ts),
                    'hi':   _fit_power_law_with_ci(ms, hs)}
        proj = _projected_bisection(cfg, fits, m_of_d(args.d_target))
        summary.append({
            'config': name, 'cfg': dict(cfg),
            'n_points': len(entries),
            'd_points': [e['d'] if 'd' in e else None for e in entries],
            'fits': fits, 'projection': proj,
        })

    ok = [s for s in summary if 'error' not in s]
    ok.sort(key=lambda s: s['projection']['bisection_point'])

    # ---- Print merged wall table ----
    print(f"\n{'=' * 100}")
    print(f"  MERGED wall_mean_s per (config, d)  "
          f"— bench wall this run: {total_wall/60:.1f} min")
    print(f"{'=' * 100}")
    header = f"  {'config':<14}"
    for d in all_ds:
        header += f"  {'d='+str(d):>8}"
    header += f"  {'n':>3}"
    print(header)
    for name in CONFIGS:
        row = f"  {name:<14}"
        for d in all_ds:
            e = merged.get(name, {}).get(d, {})
            if not e or 'error' in e or e.get('wall_mean_s') is None:
                row += f"  {'—':>8}"
            else:
                row += f"  {e['wall_mean_s']:>8.1f}"
        row += f"  {len(merged.get(name, {})):>3}"
        print(row)

    # ---- Print projections ----
    print(f"\n{'=' * 100}")
    print(f"  PROJECTED full 15-step-equivalent bisection wall at d={args.d_target}"
          f" (m={m_of_d(args.d_target):,})")
    print(f"{'=' * 100}")
    print(f"  {'rank':<4} {'config':<14} {'n':>2} {'p':>6} "
          f"{'p_95%CI':>15} {'proj_point':>12} {'proj_95%CI_hrs':>20}")
    for i, s in enumerate(ok, 1):
        fits = s['fits']
        main_fit = fits.get('wall') or fits.get('tier')
        p = main_fit['p']
        ci_p = f"[{main_fit['ci95_p_lo']:+.2f},{main_fit['ci95_p_hi']:+.2f}]"
        proj_point_s = s['projection']['bisection_point']
        proj_lo_s = s['projection']['bisection_lo95']
        proj_hi_s = s['projection']['bisection_hi95']
        print(f"  {i:<4} {s['config']:<14} {s['n_points']:>2} "
              f"{p:>6.3f} {ci_p:>15} "
              f"{proj_point_s/3600:>11.2f}h "
              f"[{proj_lo_s/3600:6.2f}, {proj_hi_s/3600:6.2f}]h")

    # ---- Recommendation ----
    if ok:
        winner = ok[0]
        w_cfg = winner['cfg']
        cli = _recommended_cli(args.d_target, winner['config'], w_cfg)
        print(f"\n{'=' * 100}")
        print(f"  RECOMMENDED for d={args.d_target}: {winner['config']}")
        print(f"  Projected: {winner['projection']['bisection_point']/3600:.1f}h"
              f"   95% CI: [{winner['projection']['bisection_lo95']/3600:.1f},"
              f" {winner['projection']['bisection_hi95']/3600:.1f}]h")
        # Margin to second place.
        if len(ok) >= 2:
            gap = (ok[1]['projection']['bisection_point'] -
                   winner['projection']['bisection_point']) / 3600
            print(f"  Gap to #2 ({ok[1]['config']}): "
                  f"{gap:+.2f}h fewer walltime")
        print(f"{'=' * 100}")
        print(f"\n{cli}\n", flush=True)

    # ---- Save merged JSON ----
    out = {
        'd_target': args.d_target,
        'm_target': m_of_d(args.d_target),
        'plan': [{'d': d, 'config': n} for (d, n) in PLAN],
        'bench_wall_this_run_s': total_wall,
        'merged_results': merged,
        'summary': summary,
        'recommended_cli': (
            _recommended_cli(args.d_target, ok[0]['config'], ok[0]['cfg'])
            if ok else None),
    }
    with open(args.json, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {args.json}", flush=True)

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_main())
