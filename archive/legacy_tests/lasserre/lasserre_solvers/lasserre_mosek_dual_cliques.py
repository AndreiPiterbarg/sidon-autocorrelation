#!/usr/bin/env python
"""Clique-decomposed SOS-dual driver.

Parallel to ``tests/lasserre_mosek_dual.py`` but calls the clique-aware
builder ``lasserre.dual_sdp_cliques.build_dual_task_cliques`` so the
dominant moment PSD cone is replaced by K smaller clique-restricted
cones (Waki–Kim–Kojima–Muramatsu 2006).  This cuts IPM Hessian storage —
the #1 memory term at d ≥ 12 — by a factor of roughly
``(n_basis_full / n_basis_clique)^4 / K``.

V1 scope:
  * No Z/2 canonicalisation / blockdiag / σ-rep dropping.  Combining
    cliques with Z/2 requires clique-aware orbit analysis that is not
    yet implemented; passing ``--z2-full`` raises NotImplementedError.
  * Rebuilds the task per-bisection probe (no ``update_task_t``).  The
    clique path sidesteps the subset bar-triplet update fragility that
    ``lasserre/dual_sdp.py`` guards against at d ≥ 6.

Adaptive IPM tolerance (``eff_tol``) and graphpar Cholesky ordering
inherit through ``_apply_task_params`` (same implementation as
``tests/lasserre_mosek_dual.py`` — these are cross-cutting improvements).

Usage:
    python tests/lasserre_mosek_dual_cliques.py --d 8 --order 3 \
        --bandwidth 6 --n-bisect 12
"""
from __future__ import annotations

import os

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_phys_cores_for_env = max(1, (os.cpu_count() or 2) // 2)
os.environ.setdefault('OMP_NUM_THREADS', str(_phys_cores_for_env))

import argparse
import concurrent.futures as _cf
import gc
import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import mosek
import numpy as np

from lasserre_scalable import _precompute
from lasserre.dual_sdp import solve_dual_task
from lasserre.dual_sdp_cliques import build_dual_task_cliques
from lasserre_mosek_tuned import val_d_known
from lasserre_mosek_dual import _apply_task_params


# =====================================================================
# Serial bisection
# =====================================================================

def solve_mosek_dual_cliques(
    d: int, order: int, *,
    bandwidth: int,
    add_upper_loc: bool = False,
    z2_full: bool = False,
    n_bisect: int = 15,
    t_lo: float = 0.5,
    t_hi: Optional[float] = None,
    single_t: Optional[float] = None,
    primary_tol: float = 1e-6,
    max_iterations: int = 1600,
    solve_form: str = 'dual',
    num_threads: Optional[int] = None,
    lambda_upper_bound: float = 1.0,
    feas_threshold: float = 0.25,
    infeas_threshold: float = 0.75,
    mosek_log: bool = False,
    verbose: bool = True,
    # --- tolerance scheduling (Task 2 refinements) ---
    use_adaptive_tol: bool = True,
    tol_cap: float = 1e-4,
    tol_rate: float = 0.001,
    refine_on_ambiguous: bool = True,
    refine_band: Tuple[float, float] = (0.15, 0.85),
) -> Dict[str, Any]:
    """Bisect on t using the clique-decomposed SOS-dual Farkas LP.

    At each probe, the task is rebuilt from scratch via
    ``build_dual_task_cliques`` (bulk MOSEK Task-API submission — still
    fast).  This deliberately avoids ``update_task_t`` because the clique
    path has not been validated against subset updates.

    Parameters match ``tests.lasserre_mosek_dual.solve_mosek_dual`` except:

      * ``bandwidth`` is required.  Must satisfy ``order ≤ b ≤ d − 1``.
      * ``z2_full`` is not supported (v1).
      * ``reuse_task`` is deliberately absent.
    """
    if z2_full:
        raise NotImplementedError(
            "Z/2 + cliques not supported in v1 — run without --z2-full.")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK SOS-dual CLIQUES: d={d} L{order}  "
              f"bandwidth={bandwidth}  upper_loc={add_upper_loc}")
        print(f"{'=' * 60}", flush=True)

    P = _precompute(d, order, verbose=verbose, lazy_ab_eiej=False)

    env = mosek.Env()
    params_applied: Optional[Dict[str, Any]] = None

    per_solve_times: List[float] = []
    proof_records: List[Dict[str, Any]] = []
    refine_log: List[Dict[str, Any]] = []

    def _eff_tol(bracket_width: float) -> float:
        if not use_adaptive_tol:
            return float(primary_tol)
        return max(float(primary_tol),
                   min(float(tol_cap),
                       float(tol_rate) * max(bracket_width, 1e-9)))

    def _solve_at_tol(t_val: float, tol: float, *,
                      print_params: bool
                      ) -> Tuple[str, float, float]:
        """Build + solve a fresh task at the given tolerance.  Returns
        (verdict, wall_s, lam_star, status_str)."""
        nonlocal params_applied
        task, info = build_dual_task_cliques(
            P, t_val=t_val, env=env,
            bandwidth=bandwidth,
            include_upper_loc=add_upper_loc,
            active_loc=None,
            active_windows=None,
            lambda_upper_bound=lambda_upper_bound,
            verbose=verbose and mosek_log,
            cache_for_reuse=False,
        )
        if params_applied is None:
            params_applied = _apply_task_params(
                task, tol=tol,
                max_iterations=max_iterations,
                solve_form=solve_form,
                num_threads=num_threads,
                verbose=print_params)
        else:
            _apply_task_params(
                task, tol=tol,
                max_iterations=max_iterations,
                solve_form=solve_form,
                num_threads=num_threads,
                verbose=False)

        if mosek_log:
            task.set_Stream(
                mosek.streamtype.log, lambda s: print(s, end=''))
            task.putintparam(mosek.iparam.log, 10)
            task.putintparam(mosek.iparam.log_intpnt, 10)

        res = solve_dual_task(
            task, info,
            feas_threshold=feas_threshold,
            infeas_threshold=infeas_threshold,
            verbose=False)

        try:
            task.__del__()
        except Exception:
            pass
        return (res['verdict'], float(res['lambda_star']),
                res['status'], float(res.get('wall_s', 0.0)))

    def _is_ambiguous(verdict: str, lam_star: float) -> bool:
        """Refine-trigger: any non-sharp λ* OR uncertain verdict."""
        lo_b, hi_b = refine_band
        # An NaN / failure is "ambiguous".
        if not (lam_star == lam_star):  # NaN check
            return True
        if verdict == 'uncertain':
            return True
        if lo_b * lambda_upper_bound <= lam_star <= hi_b * lambda_upper_bound:
            return True
        return False

    def _probe(t_val: float,
               bracket_width: float) -> Tuple[str, float, str, float]:
        """Solve at eff_tol; if result is ambiguous, refine at primary_tol."""
        ts = time.time()
        first_is_primary = (not use_adaptive_tol) or (
            _eff_tol(bracket_width) <= primary_tol + 1e-18)

        eff_tol = _eff_tol(bracket_width)
        verdict, lam_star, status, _ = _solve_at_tol(
            t_val, tol=eff_tol, print_params=(params_applied is None))

        refined = False
        if (refine_on_ambiguous and (not first_is_primary)
                and _is_ambiguous(verdict, lam_star)):
            refine_log.append({
                't': float(t_val), 'eff_tol': float(eff_tol),
                'lam_loose': float(lam_star),
                'verdict_loose': str(verdict),
                'bracket_width': float(bracket_width),
            })
            if verbose:
                print(f"    [refine] t={t_val:.6f} eff_tol={eff_tol:.2e} "
                      f"lam_loose={lam_star:.3e} verdict_loose={verdict} "
                      f"-> re-solving at primary_tol={primary_tol:.2e}",
                      flush=True)
            verdict, lam_star, status, _ = _solve_at_tol(
                t_val, tol=float(primary_tol), print_params=False)
            refined = True

        total = time.time() - ts
        if verbose and refined:
            print(f"    [refine] post: lam*={lam_star:.3e} verdict={verdict} "
                  f"(probe total {total:.2f}s)", flush=True)

        return (verdict, total, status, float(lam_star))

    # ---- Single-t mode ----
    if single_t is not None:
        verdict, wall, stat, lam = _probe(float(single_t), 0.0)
        per_solve_times.append(wall)
        print(f"SINGLE_T_VERDICT t={float(single_t):.10f} "
              f"verdict={verdict} status={stat} lam_star={lam:.6e} "
              f"wall_s={wall:.3f}", flush=True)
        try:
            del env
        except Exception:
            pass
        gc.collect()
        return {
            'd': d, 'order': order, 'bandwidth': bandwidth,
            'single_t': float(single_t),
            'verdict': verdict, 'status': stat,
            'lambda_star': lam, 'wall_s': wall,
            'ok': True, 'params': params_applied,
        }

    # ---- Bisection ----
    val = val_d_known.get(d)
    if t_hi is None:
        t_hi = (val + 0.05) if val else 2.0
    lo, hi = float(t_lo), float(t_hi)
    initial_width = hi - lo

    v_hi, dt_hi, stat_hi, lam_hi = _probe(hi, initial_width)
    per_solve_times.append(dt_hi)
    history: List[Dict[str, Any]] = []
    proof_records.append({
        'tag': f'hi_probe_t{hi:.6f}', 't': hi, 'verdict': v_hi,
        'status': stat_hi, 'lambda_star': lam_hi, 'wall_s': dt_hi})
    if verbose:
        print(f"\n  hi probe: t={hi:.6f} -> {v_hi} {stat_hi} "
              f"({dt_hi:.2f}s)", flush=True)

    tries = 0
    ok = True
    err: Optional[str] = None
    while v_hi != 'feas' and tries < 4:
        hi *= 1.5
        v_hi, dt_hi, stat_hi, lam_hi = _probe(hi, hi - lo)
        per_solve_times.append(dt_hi)
        proof_records.append({
            'tag': f'hi_probe_t{hi:.6f}', 't': hi, 'verdict': v_hi,
            'status': stat_hi, 'lambda_star': lam_hi, 'wall_s': dt_hi})
        if verbose:
            print(f"  hi probe: t={hi:.6f} -> {v_hi} {stat_hi} "
                  f"({dt_hi:.2f}s)", flush=True)
        tries += 1
    if v_hi != 'feas':
        err = (f"upper bound t={hi} not feasibly solved after "
               f"{tries + 1} tries")
        ok = False

    n_uncertain = 0
    consecutive_uncertain = 0
    if ok:
        for step in range(n_bisect):
            mid = 0.5 * (lo + hi)
            mid = max(lo + 1e-9, min(hi - 1e-9, mid))
            verdict, dt, stat, lam = _probe(mid, hi - lo)
            per_solve_times.append(dt)
            history.append({
                'step': step, 't': mid, 'status': stat,
                'verdict': verdict, 'lambda_star': lam, 'wall_s': dt})
            if verdict == 'feas':
                hi = mid
                consecutive_uncertain = 0
            elif verdict == 'infeas':
                lo = mid
                consecutive_uncertain = 0
            else:
                n_uncertain += 1
                consecutive_uncertain += 1
            if verbose:
                marker = {'feas': 'feas', 'infeas': 'infeas',
                          'uncertain': '?????'}[verdict]
                print(f"  [{step + 1}/{n_bisect}] t={mid:.8f}  "
                      f"{marker:6s} {stat:40s} ({dt:.2f}s)  "
                      f"[{lo:.6f}, {hi:.6f}]", flush=True)
            if consecutive_uncertain >= 3:
                if verbose:
                    print(f"    3 consecutive uncertain; stopping "
                          f"bisection (lb preserved).", flush=True)
                break

    lb = lo
    gc_pct: Optional[float] = None
    if val and lb is not None and val > 1.0:
        gc_pct = 100.0 * (lb - 1.0) / (val - 1.0)

    try:
        del env
    except Exception:
        pass
    gc.collect()

    total_solve_time = sum(per_solve_times)
    n_solves = len(per_solve_times)
    avg_solve = total_solve_time / n_solves if n_solves else 0.0

    result = {
        'd': d, 'order': order,
        'form': 'sos-dual-cliques',
        'bandwidth': bandwidth,
        'add_upper_loc': add_upper_loc,
        'z2_full': False,
        'lb': lb, 'val_d': val, 'gc_pct': gc_pct,
        'ok': ok, 'error': err,
        'total_solve_time_s': total_solve_time,
        'n_solves': n_solves,
        'n_uncertain': n_uncertain,
        'avg_solve_time_s': avg_solve,
        'per_solve_times_s': per_solve_times,
        'params': params_applied,
        'history': history,
        'proof_records': proof_records,
        'bracket_mid': 0.5 * (lo + hi),
        'lo': lo, 'hi': hi,
        'use_adaptive_tol': bool(use_adaptive_tol),
        'tol_cap': float(tol_cap),
        'tol_rate': float(tol_rate),
        'refine_on_ambiguous': bool(refine_on_ambiguous),
        'refine_band': list(refine_band),
        'n_refines': len(refine_log),
        'refine_log': refine_log,
    }
    if verbose:
        gc_str = f"{gc_pct:.2f}%" if gc_pct is not None else "—"
        print(f"\n  lb={lb:.6f}  gc={gc_str}  "
              f"avg_solve={avg_solve:.2f}s  "
              f"total_solve={total_solve_time:.2f}s  "
              f"n_solves={n_solves}", flush=True)
    return result


# =====================================================================
# Parallel fan-out bisection
# =====================================================================

def solve_mosek_dual_cliques_parallel(
    d: int, order: int, *,
    bandwidth: int,
    add_upper_loc: bool = False,
    z2_full: bool = False,
    n_tiers: int = 8,
    n_parallel: int = 4,
    t_lo: float = 0.5,
    t_hi: Optional[float] = None,
    primary_tol: float = 1e-6,
    max_iterations: int = 1600,
    solve_form: str = 'primal',
    total_threads: Optional[int] = None,
    lambda_upper_bound: float = 1.0,
    feas_threshold: float = 0.25,
    infeas_threshold: float = 0.75,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Parallel-fan-out clique bisection (no task reuse).

    Mirrors ``solve_mosek_dual_parallel`` in the monolithic driver but
    calls ``build_dual_task_cliques`` with ``cache_for_reuse=False`` so
    the full aggregated triplet is not retained per worker.
    """
    if z2_full:
        raise NotImplementedError(
            "Z/2 + cliques not supported in v1 — run without --z2-full.")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK SOS-dual CLIQUES parallel: d={d} L{order}  "
              f"bandwidth={bandwidth} n_tiers={n_tiers} "
              f"n_parallel={n_parallel}  upper_loc={add_upper_loc}")
        print(f"{'=' * 60}", flush=True)

    if total_threads is None:
        try:
            import psutil
            phys = psutil.cpu_count(logical=False)
            total_threads = int(phys) if phys else max(
                1, (os.cpu_count() or 2) // 2)
        except Exception:
            total_threads = max(1, (os.cpu_count() or 2) // 2)
    inner_threads = max(1, total_threads // n_parallel)
    if verbose:
        print(f"  total_threads={total_threads}  "
              f"inner_threads/worker={inner_threads}", flush=True)

    P = _precompute(d, order, verbose=verbose, lazy_ab_eiej=False)

    env = mosek.Env()
    env_lock = threading.Lock()

    def _one_probe(t_val: float,
                   bracket_width: float = 0.0
                   ) -> Tuple[float, str, float, float]:
        ts = time.time()
        eff_tol = max(float(primary_tol),
                      min(1e-3, 0.01 * max(bracket_width, 1e-9)))
        with env_lock:
            task, info = build_dual_task_cliques(
                P, t_val=t_val, env=env,
                bandwidth=bandwidth,
                include_upper_loc=add_upper_loc,
                active_loc=None,
                active_windows=None,
                lambda_upper_bound=lambda_upper_bound,
                verbose=False,
                cache_for_reuse=False,
            )
        _apply_task_params(
            task, tol=eff_tol,
            max_iterations=max_iterations,
            solve_form=solve_form,
            num_threads=inner_threads,
            verbose=False)
        res = solve_dual_task(
            task, info,
            feas_threshold=feas_threshold,
            infeas_threshold=infeas_threshold,
            verbose=False)
        try:
            task.__del__()
        except Exception:
            pass
        return (t_val, res['verdict'], time.time() - ts,
                float(res['lambda_star']))

    val = val_d_known.get(d)
    if t_hi is None:
        t_hi = (val + 0.05) if val else 2.0
    lo, hi = float(t_lo), float(t_hi)

    history: List[Dict[str, Any]] = []
    total_wall_t0 = time.time()

    v_hi_tuple = _one_probe(hi, hi - lo)
    if v_hi_tuple[1] != 'feas':
        tries = 0
        while v_hi_tuple[1] != 'feas' and tries < 3:
            hi *= 1.5
            v_hi_tuple = _one_probe(hi, hi - lo)
            tries += 1
    if verbose:
        print(f"  hi probe: t={hi:.6f} -> {v_hi_tuple[1]} "
              f"({v_hi_tuple[2]:.2f}s)", flush=True)

    for tier in range(n_tiers):
        width = hi - lo
        eps = 1e-9
        ts = np.linspace(lo + eps, hi - eps, n_parallel + 2)[1:-1]
        t_start = time.time()
        with _cf.ThreadPoolExecutor(max_workers=n_parallel) as pool:
            results = list(pool.map(
                lambda t: _one_probe(t, width), ts.tolist()))
        tier_wall = time.time() - t_start

        results.sort(key=lambda r: r[0])
        max_infeas_t = lo
        min_feas_t = hi
        for (t_val, verdict, wall_s, lam_s) in results:
            history.append({'tier': tier, 't': t_val, 'verdict': verdict,
                            'lambda_star': lam_s, 'wall_s': wall_s})
            if verdict == 'infeas' and t_val > max_infeas_t:
                max_infeas_t = t_val
            if verdict == 'feas' and t_val < min_feas_t:
                min_feas_t = t_val
        lo = max_infeas_t
        hi = min_feas_t
        if verbose:
            verdict_str = ' '.join(
                f"{r[0]:.4f}:{r[1][0]}" for r in results)
            print(f"  [tier {tier+1}/{n_tiers}] ({tier_wall:.1f}s) "
                  f"{verdict_str}  -> [{lo:.6f}, {hi:.6f}]", flush=True)
        if hi - lo < 1e-9:
            break

    gc.collect()
    total_wall = time.time() - total_wall_t0

    lb = lo
    gc_pct = None
    if val and val > 1.0:
        gc_pct = 100.0 * (lb - 1.0) / (val - 1.0)

    result = {
        'd': d, 'order': order,
        'form': 'sos-dual-cliques-parallel',
        'bandwidth': bandwidth,
        'add_upper_loc': add_upper_loc,
        'z2_full': False,
        'n_tiers': n_tiers,
        'n_parallel': n_parallel,
        'total_threads': total_threads,
        'inner_threads': inner_threads,
        'lb': lb, 'val_d': val, 'gc_pct': gc_pct,
        'total_wall_s': total_wall,
        'history': history,
        'bracket_mid': 0.5 * (lo + hi),
        'lo': lo, 'hi': hi,
        'ok': True,
    }
    if verbose:
        gc_str = f"{gc_pct:.2f}%" if gc_pct is not None else "—"
        print(f"\n  lb={lb:.6f}  gc={gc_str}  "
              f"total_wall={total_wall:.2f}s  "
              f"n_probes={len(history)}", flush=True)
    return result


# =====================================================================
# CLI
# =====================================================================

def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--bandwidth', type=int, required=True,
                   help='Clique bandwidth b.  Must satisfy '
                        'order ≤ b ≤ d−1.')
    p.add_argument('--n-bisect', type=int, default=15)
    p.add_argument('--t-lo', type=float, default=0.5)
    p.add_argument('--t-hi', type=float, default=None)
    p.add_argument('--single-t', type=float, default=None)
    p.add_argument('--primary-tol', type=float, default=1e-6)
    p.add_argument('--max-iterations', type=int, default=1600)
    p.add_argument('--solve-form', type=str, default='dual',
                   choices=('primal', 'dual', 'free'))
    p.add_argument('--num-threads', type=int, default=None)
    p.add_argument('--lambda-ub', type=float, default=1.0)
    p.add_argument('--feas-threshold', type=float, default=0.25)
    p.add_argument('--infeas-threshold', type=float, default=0.75)
    p.add_argument('--upper-loc', action='store_true')
    p.add_argument('--z2-full', action='store_true',
                   help='Not supported in v1.  Raises NotImplementedError.')
    p.add_argument('--parallel', type=int, default=0)
    p.add_argument('--n-tiers', type=int, default=8)
    p.add_argument('--total-threads', type=int, default=None)
    p.add_argument('--mosek-log', action='store_true')
    p.add_argument('--json', type=str, default=None)
    # --- tolerance scheduling ---
    p.add_argument('--no-adaptive-tol', action='store_true',
                   help='Disable adaptive tol; always use --primary-tol.')
    p.add_argument('--tol-cap', type=float, default=1e-4,
                   help='Upper cap on eff_tol (default 1e-4).')
    p.add_argument('--tol-rate', type=float, default=0.001,
                   help='Multiplier on bracket width for eff_tol.')
    p.add_argument('--no-refine', action='store_true',
                   help='Disable refine-on-ambiguous.')
    p.add_argument('--refine-lo', type=float, default=0.15)
    p.add_argument('--refine-hi', type=float, default=0.85)
    args = p.parse_args()

    if args.bandwidth < args.order:
        print(f"error: --bandwidth ({args.bandwidth}) must be ≥ --order "
              f"({args.order}); the clique basis degenerates otherwise.",
              file=sys.stderr)
        return 2
    if args.bandwidth > args.d - 1:
        print(f"error: --bandwidth ({args.bandwidth}) must be ≤ d-1 "
              f"({args.d - 1}); use the monolithic lasserre_mosek_dual.py "
              f"instead.", file=sys.stderr)
        return 2

    if args.parallel > 0:
        r = solve_mosek_dual_cliques_parallel(
            args.d, args.order,
            bandwidth=args.bandwidth,
            add_upper_loc=args.upper_loc,
            z2_full=args.z2_full,
            n_tiers=args.n_tiers,
            n_parallel=args.parallel,
            t_lo=args.t_lo, t_hi=args.t_hi,
            primary_tol=args.primary_tol,
            max_iterations=args.max_iterations,
            solve_form=args.solve_form,
            total_threads=args.total_threads,
            lambda_upper_bound=args.lambda_ub,
            feas_threshold=args.feas_threshold,
            infeas_threshold=args.infeas_threshold,
            verbose=True)
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(r, f, indent=2, default=str)
        return 0

    r = solve_mosek_dual_cliques(
        args.d, args.order,
        bandwidth=args.bandwidth,
        add_upper_loc=args.upper_loc,
        z2_full=args.z2_full,
        n_bisect=args.n_bisect,
        t_lo=args.t_lo, t_hi=args.t_hi,
        single_t=args.single_t,
        primary_tol=args.primary_tol,
        max_iterations=args.max_iterations,
        solve_form=args.solve_form,
        num_threads=args.num_threads,
        lambda_upper_bound=args.lambda_ub,
        feas_threshold=args.feas_threshold,
        infeas_threshold=args.infeas_threshold,
        mosek_log=args.mosek_log,
        use_adaptive_tol=(not args.no_adaptive_tol),
        tol_cap=args.tol_cap,
        tol_rate=args.tol_rate,
        refine_on_ambiguous=(not args.no_refine),
        refine_band=(args.refine_lo, args.refine_hi),
        verbose=True)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
