#!/usr/bin/env python
r"""Quick cloud-CPU benchmark → accurate wall-time projection for the
full d=16 L3 MOSEK run, WITHOUT ever building the d=16 model.

Why no d=16 build?
------------------
At d=16 L3, just constructing the MOSEK Fusion problem (Python-side
sparse matrix assembly + MOSEK symbolic analysis / presolve / ordering
before the first IPM iteration) takes ~2 h on a typical cloud CPU. That
is itself the thing we want to measure — we cannot afford to do it to
measure it.

So we NEVER build at d=16 here. We:

 1. Measure (build_s, avg_solve_s) at d=4, d=6, d=8 on this very
 machine, using the same MOSEK tuning planned for d=16.
 2. Fit two separate power laws:
 build_s = A_b · n_basis^{p_b}
 avg_solve = A_s · n_basis^{p_s}
 3. Compute d=16's problem size from closed-form combinatorics
 (these are mathematical invariants — no construction required):
 n_y = C(d + 2k, 2k) = C(22, 6) = 74613
 n_basis = C(d + k, k) = C(19, 3) = 969
 n_loc = C(d + k-1, k-1) = C(18, 2) = 153
 4. Extrapolate BOTH build and solve times to d=16; total wall
 time = build_proj + N_solves · per_solve_proj.
 5. Cross-check each exponent against a baked-in historical fit from
 archived d=6/8/10 MOSEK L3 runs. Warn if they diverge by > 0.3.

Why build must be extrapolated, not measured directly
-----------------------------------------------------
The per-solve power-law is well-studied for interior-point SDP
(O(size^4.25) on n_basis for MOSEK L3 empirically). Build time at d=16
is much harder to predict from first principles — dominated by MOSEK
symbolic/presolve which can scale anywhere from n_basis^3 to n_basis^5
depending on ordering algorithm. Hence we measure it on small d and
fit, rather than theorize.

Historical reference (archived in this repo):
 d=6 z2_full n_y=924 n_basis=84 build=0.74 s avg_solve=4.49 s (8 thr)
 d=8 z2_bd n_y=3003 n_basis=165 build=8.44 s avg_solve=80.7 s (8 thr)
 d=10 z2_full n_y=8008 n_basis=286 build=26.82 s avg_solve=831 s (112 thr)

Historical power-law exponents on n_basis:
 build: p ≈ 2.9–3.6 (curves upward at larger d)
 solve: p ≈ 4.25 (very consistent)

Usage
-----
 # DEFAULT (~30–90 min): d=4, d=6, d=8, d=10 — 4 fit points,
 # with d=10 (closest size to d=16) as the dominant anchor.
 python tests/estimate_d16_l3_mosek.py --out data/d16_l3_estimate.json

 # Skip the slow d=10 point (~5 min, 3 fit points)
 python tests/estimate_d16_l3_mosek.py --no-d10

 # Minimal rough estimate (~30 s, 2 fit points)
 python tests/estimate_d16_l3_mosek.py --no-d8 --no-d10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# The solver implementation is selected via --impl at CLI time (see main).
# The default is the baseline tuned solver; --impl z2block routes to the
# Gatermann-Parrilo block-diagonalized variant in lasserre_mosek_z2block.
from lasserre_mosek_tuned import solve_mosek_tuned as solve_mosek_tuned_tuned
from lasserre_mosek_tuned import val_d_known

# solve_mosek_tuned is rebound in main() based on --impl. Initial binding
# is the tuned solver so that module-level imports of solve_mosek_tuned
# keep working for existing callers.
solve_mosek_tuned = solve_mosek_tuned_tuned


# =====================================================================
# Historical reference — archived MOSEK L3 runs on this repo
# =====================================================================
# Source JSON files confirm the numbers; threads column matters for
# cross-machine comparison (exponent = machine-independent; prefactor =
# machine-dependent).
HISTORICAL: List[Dict[str, Any]] = [
 {'d': 6, 'mode': 'z2_full', 'n_y': 924,
 'n_basis': 84, 'build_s': 0.74, 'avg_solve_s': 4.49,
 'threads': 8,
 'source': 'data/mosek_d6_l3_z2full.json'},
 {'d': 8, 'mode': 'z2_bd', 'n_y': 3003,
 'n_basis': 165, 'build_s': 8.44, 'avg_solve_s': 80.73,
 'threads': 8,
 'source': 'data/mosek_d8_l3_z2bd.json'},
 {'d': 10, 'mode': 'z2_full', 'n_y': 8008,
 'n_basis': 286, 'build_s': 26.82, 'avg_solve_s': 831.09,
 'threads': 112,
 'source': 'data/mosek_validation_results/mosek_d10_l3_z2full.json'},
]

# Closed-form d=16 L3 target sizes (exact; no construction needed).
D16_TARGET = {
 'd': 16, 'order': 3,
 'n_y': math.comb(16 + 6, 6), # 74613
 'n_basis': math.comb(16 + 3, 3), # 969
 'n_loc': math.comb(16 + 2, 2), # 153
}


# =====================================================================
# Small-d calibration solves
# =====================================================================

def calibrate_point(d: int, order: int, mode: str, n_bisect: int,
 primary_tol: float, order_method: str,
 verbose: bool) -> Dict[str, Any]:
 """Run a full MOSEK solve for (d, order) with n_bisect steps and
 return measured (build_s, avg_solve_s) on the CURRENT machine.
 """
 print(f"\n[calib] d={d} order={order} mode={mode} "
 f"n_bisect={n_bisect}", flush=True)
 t0 = time.time()
 r = solve_mosek_tuned(
 d, order, mode=mode,
 add_upper_loc=True,
 n_bisect=n_bisect,
 primary_tol=primary_tol,
 order_method=order_method,
 watcher_interval_s=30.0,
 verbose=verbose)
 wall = time.time() - t0

 bs = r.get('build_stats', {}) or {}
 per = r.get('per_solve_times_s', []) or []
 avg = (sum(per) / len(per)) if per else float('nan')
 out = {
 'd': d, 'order': order, 'mode': mode,
 'n_y': int(bs.get('n_y', 0)),
 'n_basis': int(bs.get('n_basis', 0)),
 'n_loc': int(bs.get('n_loc', 0)),
 'n_solves': len(per),
 'build_s': float(r.get('build_time_s', float('nan'))),
 'avg_solve_s': avg,
 'per_solve_s': [float(x) for x in per],
 'wall_s': wall,
 'lb': r.get('lb'), 'val_d': val_d_known.get(d),
 }
 print(f"[calib] d={d}: n_y={out['n_y']} n_basis={out['n_basis']} "
 f"build={out['build_s']:.2f}s "
 f"avg_solve={out['avg_solve_s']:.2f}s "
 f"({out['n_solves']} solves, wall={wall:.1f}s)", flush=True)
 return out


# =====================================================================
# Power-law fit
# =====================================================================

def fit_power_law(sizes: List[float], times: List[float]
 ) -> Dict[str, float]:
 """Fit ``t = a · size^p`` by least squares in log–log space.
 Requires ≥2 points. For 2 points, r2 is exact (=1.0).
 """
 assert len(sizes) == len(times) and len(sizes) >= 2
 xs = [math.log(s) for s in sizes]
 ys = [math.log(t) for t in times]
 n = len(xs)
 mx = sum(xs) / n
 my = sum(ys) / n
 num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
 den = sum((x - mx) ** 2 for x in xs)
 p = num / den if den > 0 else float('nan')
 log_a = my - p * mx
 a = math.exp(log_a)
 if n == 2:
 r2 = 1.0
 else:
 ss_res = sum((y - (log_a + p * x)) ** 2
 for x, y in zip(xs, ys))
 ss_tot = sum((y - my) ** 2 for y in ys)
 r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
 return {'a': a, 'p': p, 'r2': r2}


def project(fit: Dict[str, float], size: float) -> float:
 return fit['a'] * (size ** fit['p'])


def reprefactor(p_fixed: float, sizes: List[float], times: List[float],
 size: float) -> float:
 """Given a fixed exponent p_fixed (e.g. the historical one), find
 the prefactor that best fits the CURRENT machine's points and
 project to `size`. Lets us use historical shape + this-machine
 speed for a robust cross-check.
 """
 mx = sum(math.log(s) for s in sizes) / len(sizes)
 my = sum(math.log(t) for t in times) / len(times)
 log_a = my - p_fixed * mx
 return math.exp(log_a) * (size ** p_fixed)


# =====================================================================
# Reporting
# =====================================================================

def fmt_dur(seconds: float) -> str:
 if not math.isfinite(seconds) or seconds < 0:
 return '?'
 if seconds < 60:
 return f"{seconds:.1f} s"
 if seconds < 3600:
 return f"{seconds/60:.1f} min"
 if seconds < 86400:
 return f"{seconds/3600:.2f} h"
 return f"{seconds/86400:.2f} d"


def report(calib: List[Dict[str, Any]],
 n_solves_lo: int, n_solves_hi: int) -> Dict[str, Any]:
 n_basis_tgt = D16_TARGET['n_basis']
 n_y_tgt = D16_TARGET['n_y']

 print("\n" + "=" * 72)
 print(" PROJECTION: d=16 L3 MOSEK wall time (no d=16 build used)")
 print("=" * 72)

 print("\n-- d=16 problem size (closed-form) --")
 print(f" n_y = C(22,6) = {n_y_tgt}")
 print(f" n_basis = C(19,3) = {n_basis_tgt}")
 print(f" n_loc = C(18,2) = {D16_TARGET['n_loc']}")

 print("\n-- Calibration points (this machine) --")
 for p in calib:
 print(f" d={p['d']:<2} n_y={p['n_y']:<6} n_basis={p['n_basis']:<4}"
 f" build={p['build_s']:>7.2f}s "
 f"avg_solve={p['avg_solve_s']:>8.2f}s "
 f"({p['n_solves']} solves)")

 sizes_basis = [float(p['n_basis']) for p in calib]
 sizes_ny = [float(p['n_y']) for p in calib]
 builds = [float(p['build_s']) for p in calib]
 solves = [float(p['avg_solve_s']) for p in calib]

 # This-machine fits
 fb_basis = fit_power_law(sizes_basis, builds)
 fs_basis = fit_power_law(sizes_basis, solves)
 fb_ny = fit_power_law(sizes_ny, builds)
 fs_ny = fit_power_law(sizes_ny, solves)

 # Historical fits (ref exponents; this-machine prefactor for cross-check)
 h_basis = [float(h['n_basis']) for h in HISTORICAL]
 h_ny = [float(h['n_y']) for h in HISTORICAL]
 h_builds = [float(h['build_s']) for h in HISTORICAL]
 h_solves = [float(h['avg_solve_s']) for h in HISTORICAL]
 hfb_basis = fit_power_law(h_basis, h_builds)
 hfs_basis = fit_power_law(h_basis, h_solves)
 hfb_ny = fit_power_law(h_ny, h_builds)
 hfs_ny = fit_power_law(h_ny, h_solves)

 print("\n-- Power-law fits (this machine, "
 f"{len(calib)} points) --")
 print(f" BUILD: t ~ n_basis^{fb_basis['p']:.2f} "
 f"(a={fb_basis['a']:.2e}, r²={fb_basis['r2']:.3f})")
 print(f" BUILD: t ~ n_y^{fb_ny['p']:.2f} "
 f"(a={fb_ny['a']:.2e}, r²={fb_ny['r2']:.3f})")
 print(f" SOLVE: t ~ n_basis^{fs_basis['p']:.2f} "
 f"(a={fs_basis['a']:.2e}, r²={fs_basis['r2']:.3f})")
 print(f" SOLVE: t ~ n_y^{fs_ny['p']:.2f} "
 f"(a={fs_ny['a']:.2e}, r²={fs_ny['r2']:.3f})")

 print("\n-- Historical reference exponents (d=6,8,10) --")
 print(f" BUILD: n_basis^{hfb_basis['p']:.2f}, "
 f"n_y^{hfb_ny['p']:.2f}")
 print(f" SOLVE: n_basis^{hfs_basis['p']:.2f}, "
 f"n_y^{hfs_ny['p']:.2f}")

 # Projections
 build_proj = {
 'this_basis': project(fb_basis, n_basis_tgt),
 'this_ny': project(fb_ny, n_y_tgt),
 'hist_basis': reprefactor(hfb_basis['p'], sizes_basis,
 builds, n_basis_tgt),
 'hist_ny': reprefactor(hfb_ny['p'], sizes_ny,
 builds, n_y_tgt),
 }
 solve_proj = {
 'this_basis': project(fs_basis, n_basis_tgt),
 'this_ny': project(fs_ny, n_y_tgt),
 'hist_basis': reprefactor(hfs_basis['p'], sizes_basis,
 solves, n_basis_tgt),
 'hist_ny': reprefactor(hfs_ny['p'], sizes_ny,
 solves, n_y_tgt),
 }

 print("\n-- Projected d=16 BUILD time --")
 for k, v in build_proj.items():
 print(f" {k:<12}: {fmt_dur(v)} ({v:.0f} s)")
 print("\n-- Projected d=16 PER-SOLVE time --")
 for k, v in solve_proj.items():
 print(f" {k:<12}: {fmt_dur(v)} ({v:.0f} s)")

 # Aggregate brackets
 b_vals = list(build_proj.values())
 s_vals = list(solve_proj.values())
 b_lo, b_hi = min(b_vals), max(b_vals)
 b_med = sorted(b_vals)[len(b_vals) // 2]
 s_lo, s_hi = min(s_vals), max(s_vals)
 s_med = sorted(s_vals)[len(s_vals) // 2]

 # Total = build + N_solves × per-solve
 # Use matching lo/lo, med/avg, hi/hi to avoid optimistic mix-and-match.
 n_mid = (n_solves_lo + n_solves_hi) / 2
 total_lo = b_lo + s_lo * n_solves_lo
 total_hi = b_hi + s_hi * n_solves_hi
 total_med = b_med + s_med * n_mid

 print("\n" + "=" * 72)
 print(" TOTAL WALL-TIME ESTIMATE")
 print("=" * 72)
 print(f" build : {fmt_dur(b_lo)} – {fmt_dur(b_hi)} "
 f"(med {fmt_dur(b_med)})")
 print(f" per-solve : {fmt_dur(s_lo)} – {fmt_dur(s_hi)} "
 f"(med {fmt_dur(s_med)})")
 print(f" n_solves : {n_solves_lo} – {n_solves_hi}")
 print(f" TOTAL (lo/hi) : {fmt_dur(total_lo)} – {fmt_dur(total_hi)}")
 print(f" TOTAL (median) : {fmt_dur(total_med)}")
 print("=" * 72)

 # Confidence check
 print("\n-- Confidence --")
 dbB = abs(fb_basis['p'] - hfb_basis['p'])
 dsB = abs(fs_basis['p'] - hfs_basis['p'])
 print(f" build exponent drift : |p_this - p_hist| = {dbB:.2f} "
 f"(on n_basis)")
 print(f" solve exponent drift : |p_this - p_hist| = {dsB:.2f} "
 f"(on n_basis)")
 warn = []
 if len(calib) < 3:
 warn.append("fewer than 3 calibration points (add --include-d8)")
 if dbB > 0.5:
 warn.append(f"build exponent diverges from history ({dbB:.2f})")
 if dsB > 0.3:
 warn.append(f"solve exponent diverges from history ({dsB:.2f})")
 if warn:
 for w in warn:
 print(f" {w}")
 else:
 print(" all sanity checks pass.")

 return {
 'd16_sizes': D16_TARGET,
 'calibration': calib,
 'fits_this_machine': {
 'build_basis': fb_basis, 'build_ny': fb_ny,
 'solve_basis': fs_basis, 'solve_ny': fs_ny,
 },
 'fits_historical': {
 'build_basis': hfb_basis, 'build_ny': hfb_ny,
 'solve_basis': hfs_basis, 'solve_ny': hfs_ny,
 },
 'projections': {
 'build_s': build_proj, 'per_solve_s': solve_proj,
 },
 'expected_n_solves': [n_solves_lo, n_solves_hi],
 'aggregate': {
 'build_s': {'lo': b_lo, 'med': b_med, 'hi': b_hi},
 'per_solve_s': {'lo': s_lo, 'med': s_med, 'hi': s_hi},
 'total_wall_s': {'lo': total_lo,
 'med': total_med, 'hi': total_hi},
 },
 }


# =====================================================================
# CLI
# =====================================================================

def main() -> int:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--impl', default='tuned',
 choices=('tuned', 'z2block'),
 help='Solver implementation. "tuned" = the baseline '
 '`lasserre_mosek_tuned` module (supports z2_eq, '
 'z2_bd, z2_full modes). "z2block" = the Gater'
 'mann-Parrilo block-diag variant in '
 '`lasserre_mosek_z2block` — supports all the '
 'same modes PLUS `z2block` (equality-free).')
 ap.add_argument('--mode', default='z2_bd',
 choices=('z2_eq', 'z2_bd', 'z2_full', 'z2block'),
 help='Z/2 mode used for calibration. MATCH the '
 'mode planned for the d=16 production run.')
 ap.add_argument('--primary-tol', type=float, default=1e-6,
 help='MOSEK IPM tolerance (match d=16 prod).')
 ap.add_argument('--order-method', default='forceGraphpar',
 help='MOSEK AMD ordering (match d=16 prod).')
 ap.add_argument('--calib-bisect-small', type=int, default=3,
 help='Bisection steps for d=4 and d=6 calibration.')
 ap.add_argument('--calib-bisect-d8', type=int, default=2,
 help='Bisection steps when --include-d8.')
 ap.add_argument('--calib-bisect-d10', type=int, default=1,
 help='Bisection steps when --include-d10. '
 'Default 1 because each d=10 solve takes '
 '~15 min on 8 threads.')
 ap.add_argument('--include-d8', dest='include_d8',
 action='store_true', default=True,
 help='Include d=8 L3 calibration (default: True). '
 'Adds ~3–5 min but gives a critical 3rd fit '
 'point for stable exponent estimation.')
 ap.add_argument('--no-d8', dest='include_d8', action='store_false',
 help='Skip d=8 calibration.')
 ap.add_argument('--include-d10', dest='include_d10',
 action='store_true', default=True,
 help='Include d=10 L3 calibration (default: True). '
 'Adds ~30–90 min depending on CPU — the '
 'closest fit point to d=16 and the most '
 'predictive single data point.')
 ap.add_argument('--no-d10', dest='include_d10', action='store_false',
 help='Skip d=10 calibration (much faster).')
 ap.add_argument('--expected-solves-lo', type=int, default=8)
 ap.add_argument('--expected-solves-hi', type=int, default=12)
 ap.add_argument('--out', type=str, default=None)
 ap.add_argument('--quiet-mosek', action='store_true')
 args = ap.parse_args()

 # --impl: swap in the z2block solver if requested.
 global solve_mosek_tuned
 if args.impl == 'z2block':
 from lasserre_mosek_z2block import solve_mosek_tuned as _sm_z2block
 solve_mosek_tuned = _sm_z2block
 print(f" impl : z2block (lasserre_mosek_z2block.solve_"
 f"mosek_tuned)", flush=True)
 else:
 print(f" impl : tuned (lasserre_mosek_tuned.solve_"
 f"mosek_tuned)", flush=True)
 if args.mode == 'z2block' and args.impl != 'z2block':
 ap.error("--mode z2block requires --impl z2block "
 "(only the z2block solver supports the equality-free "
 "block-diag encoding).")

 t_start = time.time()
 print("=" * 72)
 print(" d=16 L3 MOSEK wall-time estimator (NO d=16 build)")
 print(f" started: {datetime.now().isoformat(timespec='seconds')}")
 print(f" host : {platform.node()} "
 f"({platform.machine()}, {platform.system()})")
 print(f" cpu : {os.cpu_count()} threads visible")
 print(f" mode : {args.mode} tol={args.primary_tol} "
 f"order={args.order_method}")
 print("=" * 72)

 calib: List[Dict[str, Any]] = []

 for d_calib, n_bisect in ((4, args.calib_bisect_small),
 (6, args.calib_bisect_small)):
 try:
 calib.append(calibrate_point(
 d_calib, 3, args.mode, n_bisect,
 args.primary_tol, args.order_method,
 verbose=not args.quiet_mosek))
 except Exception as exc:
 print(f"[calib] d={d_calib} FAILED: {exc!r}", flush=True)

 if args.include_d8:
 print("\n[calib] d=8 included by default "
 "(~3–5 min; pass --no-d8 to skip).", flush=True)
 try:
 calib.append(calibrate_point(
 8, 3, args.mode, args.calib_bisect_d8,
 args.primary_tol, args.order_method,
 verbose=not args.quiet_mosek))
 except Exception as exc:
 print(f"[calib] d=8 FAILED: {exc!r}", flush=True)

 if args.include_d10:
 print("\n[calib] d=10 included by default "
 "(~30–90 min on 8-thread CPU; pass --no-d10 to skip). "
 "This is the most predictive single data point for d=16 "
 "because it is the closest size to the target.",
 flush=True)
 try:
 calib.append(calibrate_point(
 10, 3, args.mode, args.calib_bisect_d10,
 args.primary_tol, args.order_method,
 verbose=not args.quiet_mosek))
 except Exception as exc:
 print(f"[calib] d=10 FAILED: {exc!r}", flush=True)

 if len(calib) < 2:
 print("\nERROR: need ≥2 successful calibration points to fit "
 "a power law. Aborting.", file=sys.stderr, flush=True)
 return 2

 wall = time.time() - t_start
 print(f"\n[bench] calibration wall time: {fmt_dur(wall)}")

 rpt = report(calib,
 args.expected_solves_lo, args.expected_solves_hi)
 rpt['bench_wall_s'] = wall
 rpt['host'] = platform.node()
 rpt['cpu_count'] = os.cpu_count()
 rpt['timestamp'] = datetime.now().isoformat(timespec='seconds')
 rpt['args'] = vars(args)
 rpt['historical'] = HISTORICAL

 if args.out:
 out_path = Path(args.out)
 out_path.parent.mkdir(parents=True, exist_ok=True)
 with out_path.open('w') as f:
 json.dump(rpt, f, indent=2, default=str)
 print(f"\n[bench] wrote report → {out_path}")

 return 0


if __name__ == '__main__':
 sys.exit(main())
