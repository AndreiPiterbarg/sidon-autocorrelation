#!/usr/bin/env python
"""Bisection driver for the SOS-dual Farkas LP.

Companion to ``tests/lasserre_mosek_preelim.py``.  This file solves the
Lasserre SDP in dual form via ``lasserre/dual_sdp.py::build_dual_task``
(MOSEK Task API, bulk-submission of bar-matrix coefficients) and bisects
on t by re-building the task at each step.

Verdict semantics match the moment-primal driver exactly:

    Farkas LP max λ ≈ 1  ⟹  primal (Lasserre) INFEASIBLE at t
                          ⟹  val_L(d) > t   ⟹  bisection advances lo.
    Farkas LP max λ ≈ 0  ⟹  primal FEASIBLE at t
                          ⟹  val_L(d) ≤ t  ⟹  bisection pulls hi.

V1 scope (see ``lasserre/dual_sdp.py`` docstring):
  - no Z/2 canonicalisation, blockdiag, σ-rep dropping.
  - no upper-localising (1−μ_i) cones.
  - no consistency pre-elimination.

Usage:
    python tests/lasserre_mosek_dual.py --d 4 --order 3 --n-bisect 15
    python tests/lasserre_mosek_dual.py --d 6 --order 3 --single-t 1.17
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

from lasserre_scalable import _precompute
from lasserre.dual_sdp import (build_dual_task, solve_dual_task, update_task_t,
                                 compute_cert_rank)
from lasserre.dual_sdp_preelim import (
    build_dual_task_preelim, update_task_t_preelim,
    audit_window_multipliers, flat_extension_check,
)
from lasserre.dual_sdp_cheby import (
    build_dual_task_cheb, update_task_t_cheb,
)
from lasserre.z2_elim import canonicalize_z2
from lasserre.z2_blockdiag import (build_blockdiag_picks,
                                     localizing_sigma_reps,
                                     window_sigma_reps)
from lasserre_mosek_tuned import val_d_known


# =====================================================================
# Minimal Task-API parameter tuning
# =====================================================================

_ORDER_METHOD_MAP = {
    'graphpar': 'graphpar',
    'forcegraphpar': 'force_graphpar',
    'force_graphpar': 'force_graphpar',
    'trygraphpar': 'try_graphpar',
    'try_graphpar': 'try_graphpar',
    'appminloc': 'appminloc',
    'experimental': 'experimental',
    'none': 'none',
    'free': 'free',
}


def _apply_task_params(task: mosek.Task, *, tol: float = 1e-6,
                        max_iterations: int = 1600,
                        solve_form: str = 'dual',
                        num_threads: Optional[int] = None,
                        intpnt_multi_thread: Optional[bool] = None,
                        intpnt_purify: Optional[str] = None,
                        presolve_lindep_use: Optional[bool] = None,
                        intpnt_order_method: str = 'graphpar',
                        verbose: bool = True) -> Dict[str, Any]:
    """Apply basic MOSEK tuning to a mosek.Task (SOS-dual form).

    New memory-tuning knobs (all default to MOSEK defaults when None):
      intpnt_multi_thread : False → serial Schur factorization
                             (eliminates per-thread replication; A1 lever).
      intpnt_purify       : 'none' → skip final purification step.
      presolve_lindep_use : False → skip linear-dependency scan.
      intpnt_order_method : 'graphpar' | 'force_graphpar' | 'try_graphpar' |
                             'appminloc' | 'experimental' | 'none' | 'free'.
    """
    sf = solve_form.lower()
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
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_mu_red, tol)
    task.putintparam(mosek.iparam.intpnt_max_iterations, int(max_iterations))
    # A4: intpnt_order_method selection.
    om_key = _ORDER_METHOD_MAP.get(intpnt_order_method.lower(), 'graphpar')
    try:
        task.putintparam(mosek.iparam.intpnt_order_method,
                         getattr(mosek.orderingtype, om_key))
    except Exception:
        pass
    # A1: intpnt_multi_thread — off forces serial Cholesky / Schur factor.
    mt_applied = None
    if intpnt_multi_thread is not None:
        try:
            val = mosek.onoffkey.on if intpnt_multi_thread else mosek.onoffkey.off
            task.putintparam(mosek.iparam.intpnt_multi_thread, val)
            mt_applied = bool(intpnt_multi_thread)
        except Exception as e:
            if verbose:
                print(f"  WARN: intpnt_multi_thread set failed: {e}")
    # A5a: intpnt_purify — 'none' saves a few m-vectors at end of solve.
    purify_applied = None
    if intpnt_purify is not None:
        try:
            purify_key = intpnt_purify.lower()
            pk = {'none': 'none', 'primal': 'primal', 'dual': 'dual',
                  'primal_dual': 'primal_dual', 'auto': 'auto'}.get(purify_key)
            if pk is not None:
                task.putintparam(mosek.iparam.intpnt_purify,
                                 getattr(mosek.purify, pk))
                purify_applied = pk
        except Exception as e:
            if verbose:
                print(f"  WARN: intpnt_purify set failed: {e}")
    # A5b: presolve_lindep_use.
    lindep_applied = None
    if presolve_lindep_use is not None:
        try:
            val = (mosek.onoffkey.on if presolve_lindep_use
                   else mosek.onoffkey.off)
            task.putintparam(mosek.iparam.presolve_lindep_use, val)
            lindep_applied = bool(presolve_lindep_use)
        except Exception as e:
            if verbose:
                print(f"  WARN: presolve_lindep_use set failed: {e}")
    if num_threads is None:
        try:
            import psutil
            c = psutil.cpu_count(logical=False)
            num_threads = int(c) if c else max(1, (os.cpu_count() or 2) // 2)
        except Exception:
            num_threads = max(1, (os.cpu_count() or 2) // 2)
    task.putintparam(mosek.iparam.num_threads, int(num_threads))
    applied = {
        'solve_form': sf, 'tol': tol,
        'max_iterations': int(max_iterations),
        'num_threads': int(num_threads),
        'basis': 'never',
        'order_method': om_key,
        'intpnt_multi_thread': mt_applied,
        'intpnt_purify': purify_applied,
        'presolve_lindep_use': lindep_applied,
    }
    if verbose:
        print(f"  Task params: {applied}", flush=True)
    return applied


# =====================================================================
# Bisection top-level
# =====================================================================

def solve_mosek_dual(
    d: int, order: int, *,
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
    intpnt_multi_thread: Optional[bool] = None,
    intpnt_purify: Optional[str] = None,
    presolve_lindep_use: Optional[bool] = None,
    intpnt_order_method: str = 'graphpar',
    lazy_ab_eiej: bool = False,
    use_preelim: bool = False,
    preelim_max_fill_ratio: float = 10.0,
    preelim_protect_degrees: Optional[List[int]] = None,
    min_window_ell: Optional[int] = None,
    early_stop_on_clear_verdict: bool = False,
    early_stop_gap_tol: float = 1e-2,
    early_stop_feas_frac: float = 0.15,
    early_stop_infeas_frac: float = 0.85,
    audit_windows: bool = False,
    audit_tol: float = 1e-6,
    drop_window_indices: Optional[List[int]] = None,
    try_drop_windows: bool = False,
    rank_diag: bool = False,
    fix_lambda: Optional[float] = None,
    flat_extension: bool = False,
    flat_rank_tol: float = 1e-6,
    use_cheby: bool = False,
    lambda_upper_bound: float = 1.0,
    feas_threshold: float = 0.25,
    infeas_threshold: float = 0.75,
    mosek_log: bool = False,
    reuse_task: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Bisect on t using the SOS-dual Farkas LP.

    At each t*, the Farkas LP is rebuilt (bulk-submitted to MOSEK Task API)
    and solved.  λ* ≈ 1 → primal infeasible (advance lo); λ* ≈ 0 → primal
    feasible (pull hi).  Same verdict semantics as the moment-primal driver.

    Parameters
    ----------
    add_upper_loc   : include the (1 − μ_i) ≥ 0 upper-localising cones
                      (mirror of the moment-primal's add_upper_loc flag).
    z2_full         : apply the full Z/2 stack — canonicalize_z2 on the
                      precompute, block-diagonalise the moment cone, and
                      drop σ-orbit duplicates on the localising + window
                      cones.  Mirrors the primal's ``mode=z2_full,
                      pre_elim_z2=True`` combination.
    n_bisect        : bisection steps after the initial hi probe.
    t_lo, t_hi      : initial bracket (t_hi defaults to val_d_known[d] + 0.05).
    single_t        : if set, skip bisection and report one verdict.
    lambda_upper_bound, feas_threshold, infeas_threshold:
        see ``solve_dual_task`` for verdict thresholding.
    mosek_log       : if True, stream MOSEK log to stdout.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK SOS-dual Lasserre: d={d} L{order}  "
              f"upper_loc={add_upper_loc}  z2_full={z2_full}")
        print(f"{'=' * 60}", flush=True)

    P_raw = _precompute(d, order, verbose=verbose, lazy_ab_eiej=lazy_ab_eiej)

    # Apply Z/2 stack if requested — same transformations the primal's
    # z2_full+pre_elim_z2 applies.
    if z2_full:
        P = canonicalize_z2(P_raw, verbose=verbose)
        bd = build_blockdiag_picks(P['basis'], P['idx'], P['n_y'])
        loc_fixed, loc_pairs = localizing_sigma_reps(d)
        active_loc = list(loc_fixed) + [p for p, _ in loc_pairs]
        win_fixed, win_pairs = window_sigma_reps(d, P_raw['windows'])
        nontriv = set(P_raw['nontrivial_windows'])
        active_windows = [
            w for w in (list(win_fixed) + [p for p, _ in win_pairs])
            if w in nontriv
        ]
        if verbose:
            print(f"  z2_full: n_y {P_raw['n_y']} -> {P['n_y']}   "
                  f"moment {P['n_basis']}^2 -> sym({bd['n_sym']}) + "
                  f"anti({bd['n_anti']})   "
                  f"loc {d} -> {len(active_loc)}   "
                  f"win {len(nontriv)} -> {len(active_windows)}", flush=True)
    else:
        P = P_raw
        bd = None
        active_loc = None
        active_windows = None

    # #3 drop-specified-windows lever.  Applied AFTER Z/2 σ-rep dropping.
    if drop_window_indices:
        if active_windows is None:
            active_windows = list(P.get('nontrivial_windows', []))
        drop_set = set(int(w) for w in drop_window_indices)
        before = len(active_windows)
        active_windows = [w for w in active_windows if w not in drop_set]
        if verbose:
            print(f"  [#3-drop] active_windows {before} -> "
                  f"{len(active_windows)} (dropped {before - len(active_windows)})",
                  flush=True)

    # C1 min-window-ell filter: drop narrow-bandwidth windows (ell < min_ell).
    # windows[w] = (ell, s_lo).  Filter ALSO applies when active_windows is
    # None (the full-build path), so we materialise the default list first.
    if min_window_ell is not None and int(min_window_ell) > 0:
        windows_list = P_raw['windows']
        if active_windows is None:
            active_windows = list(P.get('nontrivial_windows', []))
        before = len(active_windows)
        active_windows = [w for w in active_windows
                          if int(windows_list[w][0]) >= int(min_window_ell)]
        if verbose:
            print(f"  [min-window-ell={min_window_ell}] active_windows "
                  f"{before} -> {len(active_windows)} "
                  f"(dropped {before - len(active_windows)} narrow)",
                  flush=True)

    env = mosek.Env()
    params_applied: Optional[Dict[str, Any]] = None

    per_solve_times: List[float] = []
    proof_records: List[Dict[str, Any]] = []

    # Persistent task handle for the reuse path.  Built lazily on first probe.
    persistent: Dict[str, Any] = {'task': None, 'info': None}

    def _eff_tol(bracket_width: float) -> float:
        # Adaptive IPM tolerance: verdict thresholds are 0.25 / 0.75 on
        # λ*, so early bisection probes only need λ* resolved to ~0.1.
        # Loosening tol here saves IPM iterations without changing the
        # verdict.  Floor at primary_tol so an explicit tight request is
        # honoured.  Cap at 1e-4 (not 1e-3) to keep the knife-edge
        # degenerate-probe zone small — pairs with refine-on-ambiguous
        # (see lasserre_mosek_dual_cliques.py) for accuracy preservation.
        return max(float(primary_tol),
                   min(1e-4, 0.001 * max(bracket_width, 1e-9)))

    def _probe(t_val: float, bracket_width: float) -> Tuple[str, float, str, float]:
        ts = time.time()
        nonlocal params_applied
        eff_tol = _eff_tol(bracket_width)
        if reuse_task and persistent['task'] is not None:
            # Reuse path — only re-submit t-dependent bar entries.
            task = persistent['task']
            info = persistent['info']
            if use_cheby:
                update_task_t_cheb(task, info, t_val)
            elif use_preelim:
                update_task_t_preelim(task, info, t_val)
            else:
                update_task_t(task, info, t_val)
            # Re-apply tuning with the current eff_tol so tolerance
            # tracks the shrinking bracket.
            _apply_task_params(
                task, tol=eff_tol,
                max_iterations=max_iterations,
                solve_form=solve_form,
                num_threads=num_threads,
                intpnt_multi_thread=intpnt_multi_thread,
                intpnt_purify=intpnt_purify,
                presolve_lindep_use=presolve_lindep_use,
                intpnt_order_method=intpnt_order_method,
                verbose=False)
        else:
            # Cold build (first probe, or reuse disabled).
            if use_cheby:
                if z2_full:
                    raise NotImplementedError(
                        "--use-cheby V1 does not support --z2-full.")
                task, info = build_dual_task_cheb(
                    P, t_val=t_val, env=env,
                    include_upper_loc=add_upper_loc,
                    active_loc=active_loc,
                    active_windows=active_windows,
                    lambda_upper_bound=lambda_upper_bound,
                    verbose=verbose and mosek_log)
            elif use_preelim:
                # In single-t or non-reuse mode, skip the _all_* triplet
                # cache — saves ~300 MB at d=14 and ~1.6 GB at d=20.
                _cache_for_reuse = bool(reuse_task and single_t is None)
                _protect_deg_set = (set(int(x) for x in preelim_protect_degrees)
                                    if preelim_protect_degrees is not None
                                    else None)
                task, info = build_dual_task_preelim(
                    P, t_val=t_val, env=env,
                    include_upper_loc=add_upper_loc,
                    z2_blockdiag_map=bd,
                    active_loc=active_loc,
                    active_windows=active_windows,
                    lambda_upper_bound=lambda_upper_bound,
                    preelim_max_fill_ratio=preelim_max_fill_ratio,
                    preelim_protect_degrees=_protect_deg_set,
                    cache_for_reuse=_cache_for_reuse,
                    verbose=verbose and mosek_log)
            else:
                task, info = build_dual_task(
                    P, t_val=t_val, env=env,
                    include_upper_loc=add_upper_loc,
                    z2_blockdiag_map=bd,
                    active_loc=active_loc,
                    active_windows=active_windows,
                    lambda_upper_bound=lambda_upper_bound,
                    verbose=verbose and mosek_log)
            if params_applied is None:
                params_applied = _apply_task_params(
                    task, tol=eff_tol,
                    max_iterations=max_iterations,
                    solve_form=solve_form,
                    num_threads=num_threads,
                    intpnt_multi_thread=intpnt_multi_thread,
                    intpnt_purify=intpnt_purify,
                    presolve_lindep_use=presolve_lindep_use,
                    intpnt_order_method=intpnt_order_method,
                    verbose=verbose)
            else:
                _apply_task_params(
                    task, tol=eff_tol,
                    max_iterations=max_iterations,
                    solve_form=solve_form,
                    num_threads=num_threads,
                    intpnt_multi_thread=intpnt_multi_thread,
                    intpnt_purify=intpnt_purify,
                    presolve_lindep_use=presolve_lindep_use,
                    intpnt_order_method=intpnt_order_method,
                    verbose=False)
            if mosek_log:
                task.set_Stream(
                    mosek.streamtype.log, lambda s: print(s, end=''))
                task.putintparam(mosek.iparam.log, 10)
                task.putintparam(mosek.iparam.log_intpnt, 10)
            if reuse_task:
                persistent['task'] = task
                persistent['info'] = info

        # Technique C: fix λ to a chosen value (typically 1.0).  Turns the
        # "max λ s.t. ..." Farkas LP into a pure feasibility problem.  MOSEK
        # terminates at the first interior point, no objective to optimize,
        # no need to reduce the barrier past the feasibility threshold.
        # Gap-preserving: feasibility at λ=1 ⟺ Farkas cert exists ⟺ primal
        # INFEAS.  LP-infeasibility at λ=1 ⟺ no cert ⟺ primal FEAS.
        if fix_lambda is not None:
            task.putvarbound(
                info['LAMBDA_IDX'], mosek.boundkey.fx,
                float(fix_lambda), float(fix_lambda))

        res = solve_dual_task(
            task, info,
            feas_threshold=feas_threshold,
            infeas_threshold=infeas_threshold,
            early_stop_on_clear_verdict=early_stop_on_clear_verdict,
            early_stop_gap_tol=early_stop_gap_tol,
            early_stop_feas_frac=early_stop_feas_frac,
            early_stop_infeas_frac=early_stop_infeas_frac,
            verbose=verbose)
        total = time.time() - ts

        # Technique C verdict remap: when λ is fixed and MOSEK returns
        # prim_infeas_cer, the LP has no feasible point with λ = fix_lambda.
        # For fix_lambda ≈ 1.0 this means NO Farkas cert exists ⟹ the
        # original primal IS feasible ⟹ verdict FEAS.
        if fix_lambda is not None:
            solsta_str = res.get('solsta', '')
            if 'prim_infeas_cer' in solsta_str:
                res['verdict'] = 'feas'
                res['lambda_star'] = 0.0
                res['status'] = (res.get('status', '') +
                                 ' [fix-λ: prim_infeas → FEAS]')

        # Technique B: rank diagnostic of Farkas certificate.
        if rank_diag:
            try:
                rd = compute_cert_rank(task, info, verbose=verbose)
                res['rank_diag'] = rd
            except Exception as exc:
                if verbose:
                    print(f"  [rank-diag] failed: {exc}", flush=True)

        # #3 post-solve window multiplier audit.
        if audit_windows and use_preelim:
            try:
                audit = audit_window_multipliers(
                    task, info, tol=audit_tol, verbose=verbose)
                res['window_audit'] = audit
            except Exception as exc:
                if verbose:
                    print(f"  [window-audit] failed: {exc}", flush=True)

        # #6 post-solve flat-extension check (only meaningful when FEAS).
        if flat_extension and use_preelim and res.get('verdict') == 'feas':
            try:
                fc = flat_extension_check(
                    task, info, P, rank_tol=flat_rank_tol, verbose=verbose)
                res['flat_extension'] = fc
            except Exception as exc:
                if verbose:
                    print(f"  [flat-ext] failed: {exc}", flush=True)

        # Stash the most recent full result so single_t mode can recover
        # auxiliary fields (rank_diag, etc.) without changing the bisection
        # tuple-return contract.
        persistent['last_res'] = res

        if not reuse_task:
            # Cold path: dispose per probe.
            try:
                task.__del__()
            except Exception:
                pass

        return (res['verdict'], total, res['status'],
                float(res['lambda_star']))

    # ---- Single-t mode ----
    if single_t is not None:
        strategy = 'direct'
        attempts: List[Dict[str, Any]] = []

        # Technique A try-first: attempt drop-all-windows first (cheap,
        # one-sided).  If it returns INFEAS the bound is proved (gap
        # preserved, since the reduced primal is a relaxation so its
        # infeasibility implies the full primal is infeasible).  If it
        # returns FEAS/uncertain, dispose and re-run with full windows.
        if try_drop_windows:
            _full_active_windows = (list(active_windows)
                                    if active_windows is not None else None)
            active_windows = []  # override for the first attempt
            if verbose:
                print(f"\n  [try-drop-windows] attempt 1: "
                      f"active_windows=[] (facial reduction)", flush=True)
            verdict_a, wall_a, stat_a, lam_a = _probe(float(single_t), 0.0)
            per_solve_times.append(wall_a)
            last_res_a = persistent.get('last_res', {}) or {}
            attempts.append({
                'strategy': 'drop_all_windows',
                'verdict': verdict_a, 'status': stat_a,
                'lambda_star': lam_a, 'wall_s': wall_a,
            })

            if verdict_a == 'infeas':
                strategy = 'drop_windows_succeeded'
                if verbose:
                    print(f"  [try-drop-windows] attempt 1 -> INFEAS  "
                          f"(bound proved in {wall_a:.2f}s)", flush=True)
                verdict, wall, stat, lam = (verdict_a, wall_a,
                                             stat_a, lam_a)
                last_res = last_res_a
            else:
                strategy = 'drop_windows_failed_fallback_full'
                if verbose:
                    print(f"  [try-drop-windows] attempt 1 -> {verdict_a}  "
                          f"(inconclusive; falling back to full LP)",
                          flush=True)
                # Dispose attempt-1 task; rebuild with full windows.
                if persistent['task'] is not None:
                    try:
                        persistent['task'].__del__()
                    except Exception:
                        pass
                    persistent['task'] = None
                    persistent['info'] = None
                gc.collect()
                active_windows = _full_active_windows
                verdict, wall, stat, lam = _probe(float(single_t), 0.0)
                per_solve_times.append(wall)
                last_res = persistent.get('last_res', {}) or {}
                attempts.append({
                    'strategy': 'full',
                    'verdict': verdict, 'status': stat,
                    'lambda_star': lam, 'wall_s': wall,
                })
        else:
            verdict, wall, stat, lam = _probe(float(single_t), 0.0)
            per_solve_times.append(wall)
            last_res = persistent.get('last_res', {}) or {}

        print(f"SINGLE_T_VERDICT t={float(single_t):.10f} "
              f"verdict={verdict} status={stat} lam_star={lam:.6e} "
              f"wall_s={wall:.3f} strategy={strategy}", flush=True)
        if persistent['task'] is not None:
            try:
                persistent['task'].__del__()
            except Exception:
                pass
            persistent['task'] = None
        try:
            del env
        except Exception:
            pass
        gc.collect()
        total_wall = sum(a['wall_s'] for a in attempts) if attempts else wall
        out = {
            'd': d, 'order': order,
            'single_t': float(single_t),
            'verdict': verdict,
            'status': stat,
            'lambda_star': lam,
            'wall_s': wall,
            'total_wall_s': total_wall,
            'strategy': strategy,
            'ok': True,
            'params': params_applied,
        }
        if attempts:
            out['attempts'] = attempts
        if 'rank_diag' in last_res:
            out['rank_diag'] = last_res['rank_diag']
        return out

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

    # Dispose the persistent task (if any) before tearing down env.
    if persistent['task'] is not None:
        try:
            persistent['task'].__del__()
        except Exception:
            pass
        persistent['task'] = None

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
        'form': 'sos-dual',
        'add_upper_loc': add_upper_loc,
        'z2_full': z2_full,
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
    }
    if verbose:
        gc_str = f"{gc_pct:.2f}%" if gc_pct is not None else "—"
        print(f"\n  lb={lb:.6f}  gc={gc_str}  "
              f"avg_solve={avg_solve:.2f}s  "
              f"total_solve={total_solve_time:.2f}s  "
              f"n_solves={n_solves}", flush=True)
    return result


# =====================================================================
# CLI
# =====================================================================

def solve_mosek_dual_parallel(
    d: int, order: int, *,
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
    """Parallel-fan-out bisection.

    Strategy: at each tier, probe ``n_parallel`` t-values equispaced
    between the current lo/hi via a thread pool; the monotonicity of the
    SDP feasibility in t (infeas → feas as t increases) then collapses
    the bracket to ``(max t reporting infeas, min t reporting feas)``.
    After ``n_tiers`` tiers, the bracket has shrunk by a factor of
    ~``(n_parallel+1)^n_tiers``.

    Each worker builds and solves its own mosek.Task.  MOSEK releases the
    GIL during optimize(), so threads scale.  To avoid CPU over-subscription
    we pass ``max(1, total_threads // n_parallel)`` as ``num_threads`` for
    each inner MOSEK solve.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK SOS-dual parallel: d={d} L{order}  "
              f"n_tiers={n_tiers}  n_parallel={n_parallel}  "
              f"upper_loc={add_upper_loc}  z2_full={z2_full}")
        print(f"{'=' * 60}", flush=True)

    # Auto-detect total CPU budget.
    if total_threads is None:
        try:
            import psutil
            phys = psutil.cpu_count(logical=False)
            total_threads = int(phys) if phys else max(1, (os.cpu_count() or 2) // 2)
        except Exception:
            total_threads = max(1, (os.cpu_count() or 2) // 2)
    inner_threads = max(1, total_threads // n_parallel)
    if verbose:
        print(f"  total_threads={total_threads}  inner_threads/worker={inner_threads}",
              flush=True)

    # One precompute + env shared across workers (read-only; safe).
    P_raw = _precompute(d, order, verbose=verbose, lazy_ab_eiej=False)
    if z2_full:
        P = canonicalize_z2(P_raw, verbose=verbose)
        bd = build_blockdiag_picks(P['basis'], P['idx'], P['n_y'])
        loc_fixed, loc_pairs = localizing_sigma_reps(d)
        active_loc = list(loc_fixed) + [p for p, _ in loc_pairs]
        win_fixed, win_pairs = window_sigma_reps(d, P_raw['windows'])
        nontriv = set(P_raw['nontrivial_windows'])
        active_windows = [
            w for w in (list(win_fixed) + [p for p, _ in win_pairs])
            if w in nontriv
        ]
    else:
        P = P_raw
        bd = None
        active_loc = None
        active_windows = None

    env = mosek.Env()
    env_lock = threading.Lock()

    def _one_probe(t_val: float,
                   bracket_width: float = 0.0
                   ) -> Tuple[float, str, float, float]:
        """Runs in a worker thread.  Returns (t, verdict, wall_s, lam*)."""
        ts = time.time()
        eff_tol = max(float(primary_tol),
                      min(1e-4, 0.001 * max(bracket_width, 1e-9)))
        with env_lock:
            task, info = build_dual_task(
                P, t_val=t_val, env=env,
                include_upper_loc=add_upper_loc,
                z2_blockdiag_map=bd,
                active_loc=active_loc,
                active_windows=active_windows,
                lambda_upper_bound=lambda_upper_bound,
                verbose=False)
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

    # Upper bound probe first.
    v_hi_tuple = _one_probe(hi, hi - lo)
    if v_hi_tuple[1] != 'feas':
        # expand hi like the serial driver would
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
        # n_parallel interior points evenly spaced between lo and hi,
        # excluding endpoints.
        eps = 1e-9
        ts = np.linspace(lo + eps, hi - eps, n_parallel + 2)[1:-1]
        t_start = time.time()
        with _cf.ThreadPoolExecutor(max_workers=n_parallel) as pool:
            results = list(pool.map(
                lambda t: _one_probe(t, width), ts.tolist()))
        tier_wall = time.time() - t_start

        results.sort(key=lambda r: r[0])  # sort by t
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
                f"{r[0]:.4f}:{r[1][0]}"
                for r in results)
            print(f"  [tier {tier+1}/{n_tiers}] ({tier_wall:.1f}s) "
                  f"{verdict_str}  -> [{lo:.6f}, {hi:.6f}]",
                  flush=True)
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
        'form': 'sos-dual-parallel',
        'add_upper_loc': add_upper_loc,
        'z2_full': z2_full,
        'n_tiers': n_tiers,
        'n_parallel': n_parallel,
        'total_threads': total_threads,
        'inner_threads': inner_threads,
        'lb': lb, 'val_d': val, 'gc_pct': gc_pct,
        'total_wall_s': total_wall,
        'history': history,
        'ok': True,
    }
    if verbose:
        gc_str = f"{gc_pct:.2f}%" if gc_pct is not None else "—"
        print(f"\n  lb={lb:.6f}  gc={gc_str}  "
              f"total_wall={total_wall:.2f}s  "
              f"n_probes={len(history)}", flush=True)
    return result


import numpy as np  # noqa: E402  (imported here for solve_mosek_dual_parallel)


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
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
    p.add_argument('--upper-loc', action='store_true',
                    help='Include (1-μ_i) ≥ 0 upper-localising cones.')
    p.add_argument('--z2-full', action='store_true',
                    help='Apply canonicalize_z2 + block-diagonalise moment '
                         'cone + σ-rep drop on loc + window cones.')
    p.add_argument('--parallel', type=int, default=0,
                    help='If > 0, run parallel-fan-out bisection with this '
                         'many workers; otherwise run serial bisection.')
    p.add_argument('--n-tiers', type=int, default=8,
                    help='Number of tiers in parallel bisection.')
    p.add_argument('--total-threads', type=int, default=None,
                    help='Total CPU budget for parallel mode.  Each worker '
                         'gets total/n_parallel MOSEK threads.')
    p.add_argument('--mosek-log', action='store_true')
    p.add_argument('--json', type=str, default=None)
    # Memory-tuning knobs (Group 1 / Group 2).
    p.add_argument('--mosek-multi-thread', type=str, default='default',
                   choices=['default', 'on', 'off'],
                   help="intpnt_multi_thread: OFF forces serial Schur "
                        "factorization, eliminating per-thread replication.")
    p.add_argument('--intpnt-purify', type=str, default='default',
                   choices=['default', 'none', 'primal', 'dual',
                            'primal_dual', 'auto'])
    p.add_argument('--presolve-lindep', type=str, default='default',
                   choices=['default', 'on', 'off'])
    p.add_argument('--order-method', type=str, default='graphpar',
                   choices=['graphpar', 'force_graphpar', 'try_graphpar',
                            'appminloc', 'experimental', 'none', 'free'])
    p.add_argument('--lazy-ab-eiej', action='store_true',
                   help="Skip eager (n_loc,n_loc,d,d) ab_eiej_idx "
                        "materialisation in precompute.")
    p.add_argument('--use-preelim', action='store_true',
                   help="Apply consistency/simplex pre-elim to the dual "
                        "Farkas LP (row-contraction via T).  Reduces "
                        "n_cons by ~30% at d=12-14.  Same bound.")
    p.add_argument('--preelim-max-fill-ratio', type=float, default=10.0,
                   help="Fill-cap for preelim Gauss-Jordan elimination "
                        "(default 10.0, matches preelim.DEFAULT_MAX_FILL_RATIO).")
    p.add_argument('--preelim-protect-degrees', type=str, default=None,
                   help="Comma-separated monomial total degrees to protect "
                        "from pivoting in preelim (default: '1,2').  Pass "
                        "'0' for deg-0-only protection = maximum elimination.")
    p.add_argument('--min-window-ell', type=int, default=None,
                   help="Drop windows with bandwidth ell < min-window-ell "
                        "(C1 sacrifice).  Filters active_windows by the "
                        "first entry of each (ell, s_lo) tuple in P['windows'].")
    p.add_argument('--early-stop-on-clear-verdict', action='store_true',
                   help="B1 sacrifice: install an IPM info-callback that "
                        "terminates once the primal obj (= λ*) is clearly "
                        "FEAS (|λ*| < 0.15·Λ) or INFEAS (λ* > 0.85·Λ) and "
                        "the dual gap is < 1e-2.")
    p.add_argument('--early-stop-gap-tol', type=float, default=1e-2,
                   help="Gap tolerance for early-stop (default 1e-2). "
                        "Loosen to 0.1 to fire earlier (fewer IPM iters).")
    p.add_argument('--early-stop-feas-frac', type=float, default=0.15,
                   help="FEAS threshold fraction of Λ (default 0.15). "
                        "Loosen to e.g. 0.25 to fire earlier.")
    p.add_argument('--early-stop-infeas-frac', type=float, default=0.85,
                   help="INFEAS threshold fraction of Λ (default 0.85). "
                        "Tighten toward 0.75 to fire earlier for INFEAS probes.")
    # #3 window-audit + drop levers.
    p.add_argument('--audit-windows', action='store_true',
                   help="Post-solve: report ||X_W||_F per window bar "
                        "variable.  Windows with norm < --audit-tol are "
                        "slack by complementary slackness (droppable).")
    p.add_argument('--audit-tol', type=float, default=1e-6,
                   help="Frobenius-norm threshold for 'slack' window "
                        "classification (default 1e-6).")
    p.add_argument('--drop-windows', type=str, default='',
                   help="Comma-separated window INDICES to drop before "
                        "building.  Use to test bound preservation after "
                        "an --audit-windows run identifies slack windows.")
    # Technique A try-first: attempt with active_windows=[] first, fall back on FEAS.
    p.add_argument('--try-drop-windows', action='store_true',
                   help="Single-t mode: first attempt with active_windows=[] "
                        "(cheap, one-sided — the reduced LP is a relaxation, "
                        "so INFEAS there lifts the cert to the full problem). "
                        "If INFEAS, bound is proved (gap preserved). "
                        "If FEAS/uncertain, dispose and re-run with full "
                        "windows.  Zero gap risk, potential big speedup at "
                        "large d where X_W → 0 at the optimum.")
    # Technique B: rank diagnostic of the Farkas certificate bar variables.
    p.add_argument('--rank-diag', action='store_true',
                   help="Post-solve: SVD of X_0 and other bar variables; "
                        "report numerical rank at tols {1e-4,1e-6,1e-8,1e-10} "
                        "and top singular-value spectrum.  Gauges "
                        "Burer-Monteiro viability.")
    # Technique C: fix λ to constant (removes the objective; pure feasibility LP).
    p.add_argument('--fix-lambda', type=float, default=None,
                   help="Fix λ ≡ FIX_LAMBDA (typically 1.0).  Converts "
                        "max-λ Farkas LP to feasibility.  At 1.0: LP-feasible "
                        "⟹ cert exists ⟹ INFEAS verdict; LP-infeasible "
                        "(prim_infeas_cer) ⟹ no cert ⟹ FEAS verdict.  "
                        "Gap-preserving.")
    # #6 flat-extension early-exit.
    p.add_argument('--flat-extension', action='store_true',
                   help="Post-solve (FEAS verdicts only): check "
                        "rank(M_L(y*)) == rank(M_{L-1}(y*)) via SVD.  "
                        "Matching ranks ⟹ relaxation tight (val_L(d)=t*).")
    p.add_argument('--flat-rank-tol', type=float, default=1e-6,
                   help="Relative SVD tolerance for rank detection.")
    # #5 Chebyshev basis.
    p.add_argument('--use-cheby', action='store_true',
                   help="Use shifted-Chebyshev basis T_k*(μ) = T_k(2μ-1) "
                        "instead of monomial basis.  Better conditioning "
                        "at large d.  Incompatible with --z2-full and "
                        "--use-preelim in V1.")
    args = p.parse_args()

    def _tri(s):
        return None if s == 'default' else (s == 'on')
    mt_flag = _tri(args.mosek_multi_thread)
    lindep_flag = _tri(args.presolve_lindep)
    purify_flag = None if args.intpnt_purify == 'default' else args.intpnt_purify

    if args.parallel > 0:
        r = solve_mosek_dual_parallel(
            args.d, args.order,
            add_upper_loc=args.upper_loc,
            z2_full=args.z2_full,
            n_tiers=args.n_tiers,
            n_parallel=args.parallel,
            t_lo=args.t_lo,
            t_hi=args.t_hi,
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

    r = solve_mosek_dual(
        args.d, args.order,
        add_upper_loc=args.upper_loc,
        z2_full=args.z2_full,
        n_bisect=args.n_bisect,
        t_lo=args.t_lo,
        t_hi=args.t_hi,
        single_t=args.single_t,
        primary_tol=args.primary_tol,
        max_iterations=args.max_iterations,
        solve_form=args.solve_form,
        num_threads=args.num_threads,
        intpnt_multi_thread=mt_flag,
        intpnt_purify=purify_flag,
        presolve_lindep_use=lindep_flag,
        intpnt_order_method=args.order_method,
        lazy_ab_eiej=args.lazy_ab_eiej,
        use_preelim=args.use_preelim,
        preelim_max_fill_ratio=args.preelim_max_fill_ratio,
        preelim_protect_degrees=(
            [int(x) for x in args.preelim_protect_degrees.split(',') if x.strip()]
            if args.preelim_protect_degrees else None),
        min_window_ell=args.min_window_ell,
        early_stop_on_clear_verdict=args.early_stop_on_clear_verdict,
        early_stop_gap_tol=args.early_stop_gap_tol,
        early_stop_feas_frac=args.early_stop_feas_frac,
        early_stop_infeas_frac=args.early_stop_infeas_frac,
        audit_windows=args.audit_windows,
        audit_tol=args.audit_tol,
        drop_window_indices=(
            [int(x) for x in args.drop_windows.split(',') if x.strip()]
            if args.drop_windows else None),
        try_drop_windows=args.try_drop_windows,
        rank_diag=args.rank_diag,
        fix_lambda=args.fix_lambda,
        flat_extension=args.flat_extension,
        flat_rank_tol=args.flat_rank_tol,
        use_cheby=args.use_cheby,
        lambda_upper_bound=args.lambda_ub,
        feas_threshold=args.feas_threshold,
        infeas_threshold=args.infeas_threshold,
        mosek_log=args.mosek_log,
        verbose=True)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
