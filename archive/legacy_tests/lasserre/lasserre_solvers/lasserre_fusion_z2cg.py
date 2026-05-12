#!/usr/bin/env python
"""Fusion-based Z/2 + constraint generation (moment-primal form).

Combines:
  * canonicalize_z2 on the precompute (n_y halves)
  * Fusion moment PSD cone blockdiag-split into (sym, anti) via
    build_blockdiag_picks (moment Cholesky cost n^3 → 2·(n/2)^3 = n^3/4)
  * σ-rep dropping on localizing cones (d → ⌈d/2⌉) and windows
  * Constraint generation on PSD window cones (scalar windows always on)

This is the moment-primal form (directly writes y ∈ R^{n_y} with all
Lasserre constraints and solves MOSEK Fusion).  Farkas-LP sign issues
from the Task-API path are avoided entirely.
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
from mosek.fusion import (Model, Domain, Expr, Matrix,
                           ObjectiveSense, SolutionStatus)

from lasserre_scalable import _precompute
from lasserre.precompute import _check_window_violations, _add_psd_window
from lasserre.z2_elim import canonicalize_z2
from lasserre.z2_blockdiag import (build_blockdiag_picks,
                                    localizing_sigma_reps,
                                    window_sigma_reps)
from lasserre.core import val_d_known


# =====================================================================
# RSS guard
# =====================================================================

def _rss_bytes() -> int:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


class RSSGuard:
    """Passive RSS monitor.  REPORTS peak and crossings but NEVER aborts.
    Let the OS kernel OOM killer decide when we're out of RAM — never
    preempt it with a soft limit that wastes headroom.
    """

    def __init__(self, limit_gb: float = 0.0, interval_s: float = 2.0):
        self.limit = int(limit_gb * (1024 ** 3)) if limit_gb > 0 else 0
        self.interval = interval_s
        self.peak = 0
        self.over_limit = False  # kept for report; NOT used to abort
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self):
        self.peak = _rss_bytes()

        def _loop():
            while not self._stop.wait(self.interval):
                r = _rss_bytes()
                if r > self.peak:
                    self.peak = r
                # Report-only crossing (no abort).
                if (self.limit > 0 and r > self.limit
                        and not self.over_limit):
                    self.over_limit = True
                    print(f'\n  [rss] crossed {self.limit/1024**3:.1f}GB '
                          f'(now {r/1024**3:.1f}GB) — continuing; '
                          f'OS OOM killer remains the only stop',
                          flush=True)

        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()

    def stop(self) -> int:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)
        return self.peak


# =====================================================================
# Checkpointing (resume across spot preemptions)
# =====================================================================

def _save_checkpoint(path: str, *, active_windows, lo, hi, best_lb,
                      cg_round_next: int, last_feas_t, last_feas_y,
                      seed_t, converged: bool,
                      n_cg_candidates: int, emergency: bool = False,
                      **extra) -> None:
    """Atomic JSON write for CG state; y array saved as separate .npz.

    The state fully determines what the next probe would do, so on
    resume we can rebuild the Fusion model + add `active_windows` PSD
    cones and continue from `cg_round_next`.
    """
    tmp_json = path + '.tmp'
    state = {
        'active_windows': [int(w) for w in active_windows],
        'lo': float(lo), 'hi': float(hi), 'best_lb': float(best_lb),
        'cg_round_next': int(cg_round_next),
        'last_feas_t': float(last_feas_t),
        'seed_t': float(seed_t),
        'converged': bool(converged),
        'n_cg_candidates': int(n_cg_candidates),
        'emergency': bool(emergency),
        'saved_at': time.time(),
    }
    state.update(extra)
    with open(tmp_json, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp_json, path)
    if last_feas_y is not None:
        np.savez(path + '.y.npz', y=np.asarray(last_feas_y))
    print(f'  [ckpt] saved active={len(state["active_windows"])} '
          f'cg_round_next={cg_round_next} lo={lo:.6f} hi={hi:.6f}'
          f'{" EMERGENCY" if emergency else ""}',
          flush=True)


def _load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    y_file = path + '.y.npz'
    if os.path.exists(y_file):
        try:
            state['last_feas_y'] = np.load(y_file)['y']
        except Exception:
            state['last_feas_y'] = None
    else:
        state['last_feas_y'] = None
    return state


# =====================================================================
# AWS spot termination watcher
# =====================================================================

def _spot_termination_imminent() -> bool:
    """Check AWS instance metadata for a pending spot reclaim.
    Returns True iff the endpoint returns HTTP 200 (AWS signals a
    2-minute warning).  On non-AWS hosts the metadata endpoint is
    unreachable and we return False (no warnings expected).
    """
    import urllib.request
    try:
        req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/spot/instance-action')
        with urllib.request.urlopen(req, timeout=0.5) as r:
            return r.status == 200
    except Exception:
        return False


class SpotWatcher:
    """Background poller that flushes a checkpoint + os._exit(0) on
    detection of AWS spot termination.

    os._exit is used (not sys.exit) so the daemon thread can terminate
    the main process even if MOSEK's IPM is holding the GIL.
    """
    def __init__(self, emergency_cb, interval_s: float = 15.0):
        self.emergency_cb = emergency_cb
        self.interval = interval_s
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self):
        def _loop():
            while not self._stop.wait(self.interval):
                if _spot_termination_imminent():
                    print('\n!!! AWS SPOT TERMINATION IMMINENT — '
                          'flushing emergency checkpoint', flush=True)
                    try:
                        self.emergency_cb()
                    except Exception as e:
                        print(f'  emergency ckpt error: {e}',
                              flush=True)
                    print('  goodbye.', flush=True)
                    os._exit(0)
        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)


# =====================================================================
# Base constraints (moment-primal form)
# =====================================================================

def _build_base_constraints_z2(mdl, y, P, bd, active_loc, upper_loc,
                                verbose=True):
    """Add Lasserre constraints to `mdl` using Z/2 blockdiag moment and
    σ-rep localizing cones.

    y is a MOSEK Fusion variable of size P['n_y'] (canonical).
    bd is from build_blockdiag_picks(canonical_P).
    active_loc is the list of σ-representative bins (from
    localizing_sigma_reps).
    """
    d = P['d']
    n_y = P['n_y']
    n_loc = P['n_loc']
    idx = P['idx']

    # y_0 = 1 (canonical index of zero monomial).
    zero = tuple(0 for _ in range(d))
    mdl.constraint('y0', y.index(idx[zero]), Domain.equalsTo(1.0))

    # Consistency (simplex) equations — canonical idx.
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    c_rows, c_cols, c_vals = [], [], []
    n_added = 0
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child = consist_ei_idx[r]
        has_child = False
        for ci in range(d):
            if child[ci] >= 0:
                c_rows.append(n_added)
                c_cols.append(int(child[ci]))
                c_vals.append(1.0)
                has_child = True
        if not has_child:
            continue
        c_rows.append(n_added)
        c_cols.append(ai)
        c_vals.append(-1.0)
        n_added += 1
    if n_added > 0:
        A_con = Matrix.sparse(n_added, n_y, c_rows, c_cols, c_vals)
        mdl.constraint('consist', Expr.mul(A_con, y),
                       Domain.equalsTo(0.0))

    # Moment PSD: SPLIT into sym + anti blocks.
    T_sym = bd['T_sym'].tocoo()
    T_anti = bd['T_anti'].tocoo()
    n_sym = int(bd['n_sym'])
    n_anti = int(bd['n_anti'])

    if T_sym.nnz:
        M_sym_flat = Expr.mul(Matrix.sparse(
            n_sym * n_sym, n_y,
            T_sym.row.tolist(), T_sym.col.tolist(), T_sym.data.tolist()), y)
        M_sym = Expr.reshape(M_sym_flat, n_sym, n_sym)
        mdl.constraint('moment_sym', M_sym, Domain.inPSDCone(n_sym))

    if T_anti.nnz and n_anti > 0:
        M_anti_flat = Expr.mul(Matrix.sparse(
            n_anti * n_anti, n_y,
            T_anti.row.tolist(), T_anti.col.tolist(), T_anti.data.tolist()),
            y)
        M_anti = Expr.reshape(M_anti_flat, n_anti, n_anti)
        mdl.constraint('moment_anti', M_anti, Domain.inPSDCone(n_anti))

    # Localizing cones (σ-reps only): M_{k-1}(μ_i y) ⪰ 0 for i in active_loc.
    if P['order'] >= 2:
        for i_var in active_loc:
            Li = Expr.reshape(
                y.pick(P['loc_picks'][i_var]), n_loc, n_loc)
            mdl.constraint(f'loc_mu_{i_var}', Li,
                           Domain.inPSDCone(n_loc))

        if upper_loc:
            # Upper-loc: M_{k-1}((1-μ_i)y) ⪰ 0 for i in active_loc.
            for i_var in active_loc:
                sub_moment = y.pick(P['t_pick'])
                mu_i = y.pick(P['loc_picks'][i_var])
                diff = Expr.sub(sub_moment, mu_i)
                L_upper = Expr.reshape(diff, n_loc, n_loc)
                mdl.constraint(f'loc_upper_{i_var}', L_upper,
                               Domain.inPSDCone(n_loc))

    if verbose:
        n_psd = (2 if n_anti > 0 else 1) + len(active_loc)
        if upper_loc:
            n_psd += len(active_loc)
        print(f'  base PSD cones: {n_psd} (moment_sym {n_sym}² + '
              f'moment_anti {n_anti}² + {len(active_loc)} loc'
              f'{" + " + str(len(active_loc)) + " upper-loc" if upper_loc else ""})'
              f' + consistency: {n_added} rows', flush=True)


# =====================================================================
# Solver
# =====================================================================

def solve_z2cg_fusion(args) -> Dict[str, Any]:
    t_start = time.time()
    guard = RSSGuard(limit_gb=args.rss_limit_gb)
    guard.start()

    d = int(args.d)
    order = int(args.order)

    print(f'[z2cg-fusion] d={d} order={order}  threads={args.threads}  '
          f'upper_loc={args.upper_loc}  '
          f'cg_rounds={args.cg_rounds}  cg_add={args.cg_add}  '
          f'bisect/round={args.bisect}  tol={args.tol}  '
          f'rss_limit={args.rss_limit_gb}GB', flush=True)

    # Precompute + canonicalize.
    tp0 = time.time()
    P_raw = _precompute(d, order, verbose=True, lazy_ab_eiej=True)
    P = canonicalize_z2(P_raw, verbose=True) if args.use_z2 else P_raw
    if args.use_z2:
        bd = build_blockdiag_picks(P['basis'], P['idx'], P['n_y'])
        loc_fixed, loc_pairs = localizing_sigma_reps(d)
        active_loc = list(loc_fixed) + [p for p, _ in loc_pairs]
        win_fixed, win_pairs = window_sigma_reps(d, P_raw['windows'])
        nontriv_raw = set(P_raw['nontrivial_windows'])
        cg_candidates = [
            w for w in (list(win_fixed) + [p for p, _ in win_pairs])
            if w in nontriv_raw
        ]
    else:
        bd = None
        active_loc = list(range(d))
        cg_candidates = list(P_raw['nontrivial_windows'])
    cg_candidates_set = set(cg_candidates)
    precompute_s = time.time() - tp0
    print(f'  precompute{"+z2" if args.use_z2 else ""}: '
          f'{precompute_s:.1f}s  n_y={P["n_y"]}  '
          f'loc_active={len(active_loc)}  cg_cand={len(cg_candidates)}  '
          f'RSS={_rss_bytes()/1024**3:.2f}GB',
          flush=True)

    # Build model.
    t_build0 = time.time()
    mdl = Model('z2cg_fusion')
    mdl.setSolverParam('intpntCoTolRelGap', args.tol)
    mdl.setSolverParam('intpntCoTolPfeas', args.tol)
    mdl.setSolverParam('intpntCoTolDfeas', args.tol)
    mdl.setSolverParam('intpntCoTolMuRed', args.tol)
    mdl.setSolverParam('numThreads', int(args.threads))
    mdl.setSolverParam('intpntBasis', 'never')
    mdl.setSolverParam('intpntMaxIterations', int(args.max_iters))
    try:
        mdl.setSolverParam('intpntOrderMethod', 'forceGraphpar')
    except Exception:
        try:
            mdl.setSolverParam('intpntOrderMethod', 'tryGraphpar')
        except Exception:
            pass

    y = mdl.variable('y', P['n_y'], Domain.greaterThan(0.0))
    t_param = mdl.parameter('t')

    if args.use_z2:
        _build_base_constraints_z2(mdl, y, P, bd, active_loc,
                                   args.upper_loc, verbose=True)
    else:
        # Fall back to canonical base constraint builder.
        from lasserre.precompute import _build_base_constraints
        _build_base_constraints(mdl, y, P, args.upper_loc, verbose=True)

    # Scalar window constraints (all σ-rep nontrivial windows).
    rows = []
    cols = []
    vals = []
    for w in cg_candidates:
        # f_W(y) = Σ B_W[α] y_α, need these in CANONICAL y.
        # P['F_scipy'] in precompute is for ORIGINAL y.  If canonicalize_z2
        # was applied, F_scipy was remapped (checked: canonicalize_z2
        # rebuilds F_scipy in canonical indexing).
        pass

    # Use P['F_scipy'] directly which is already canonical post-canonicalize.
    F = P['F_scipy'].tocsr()
    # Build constraint: t*1 - f_W(y) >= 0 for W in cg_candidates.
    # Since F is sparse (n_win × n_y), extract rows of W's we want.
    F_sub = F[cg_candidates, :].tocoo()
    if F_sub.nnz > 0:
        F_mosek = Matrix.sparse(
            len(cg_candidates), P['n_y'],
            F_sub.row.tolist(), F_sub.col.tolist(), F_sub.data.tolist())
        f_all = Expr.mul(F_mosek, y)
        ones_col = Matrix.dense(
            len(cg_candidates), 1, [1.0] * len(cg_candidates))
        t_rep = Expr.flatten(Expr.mul(
            ones_col, Expr.reshape(t_param, 1, 1)))
        mdl.constraint('win_scalar',
                       Expr.sub(t_rep, f_all),
                       Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))
    build_s = time.time() - t_build0
    print(f'  model built: {build_s:.1f}s  '
          f'RSS={_rss_bytes()/1024**3:.2f}GB', flush=True)

    active_windows: set = set()
    bisect_history: List[Dict[str, Any]] = []
    cg_history: List[Dict[str, Any]] = []

    def check_feasible(t_val):
        t_param.setValue(t_val)
        try:
            mdl.solve()
            ps = mdl.getPrimalSolutionStatus()
            return ps in (SolutionStatus.Optimal, SolutionStatus.Feasible)
        except Exception as e:
            print(f'    solve err: {e}', flush=True)
            return False

    # ---- Checkpoint / resume ----
    ckpt = _load_checkpoint(args.checkpoint) if args.checkpoint else None
    resume_mode = ckpt is not None

    if resume_mode:
        print(f'\n  *** RESUMING from {args.checkpoint} ***', flush=True)
        print(f'    active_windows={len(ckpt["active_windows"])}  '
              f'cg_round_next={ckpt["cg_round_next"]}  '
              f'bracket=[{ckpt["lo"]:.6f}, {ckpt["hi"]:.6f}]  '
              f'best_lb={ckpt["best_lb"]:.6f}  '
              f'converged={ckpt["converged"]}',
              flush=True)
        seed_t = float(ckpt['seed_t'])
        # Re-add the active PSD windows (Fusion's model state is not
        # persisted; we rebuild the cone structure from the saved list).
        for w in ckpt['active_windows']:
            _add_psd_window(mdl, y, t_param, w, P)
            active_windows.add(int(w))
        print(f'    rebuilt {len(active_windows)} PSD window cones',
              flush=True)
        lo = float(ckpt['lo'])
        hi = float(ckpt['hi'])
        best_lb = float(ckpt['best_lb'])
        converged = bool(ckpt['converged'])
        last_feas_t = float(ckpt['last_feas_t'])
        last_feas_y = ckpt['last_feas_y']
        cg_round_start = int(ckpt['cg_round_next'])
        if converged:
            print(f'    checkpoint says converged; exiting with bound {best_lb:.6f}',
                  flush=True)
    else:
        # Seed: solve at a generous t, find violations.
        seed_t = (val_d_known.get(d, 1.3) + 0.05)
        t_seed_0 = time.time()
        feas0 = check_feasible(seed_t)
        seed_s = time.time() - t_seed_0
        if not feas0:
            seed_t *= 1.3
            feas0 = check_feasible(seed_t)
        print(f'\n  [seed] t={seed_t:.4f}  feas={feas0}  '
              f'wall={seed_s:.1f}s  RSS={_rss_bytes()/1024**3:.2f}GB',
              flush=True)
        if not feas0:
            return {'ok': False, 'reason': 'seed_not_feasible',
                    'peak_rss_gb': guard.peak / 1024**3}
        y_vals = np.array(y.level())

        violations_all = _check_window_violations(
            y_vals, seed_t, P, active_windows, tol=args.violation_tol)
        violations = [(w, e) for (w, e) in violations_all
                       if w in cg_candidates_set]
        print(f'  [seed] {len(violations)} violated σ-rep windows',
              flush=True)

        n_add = min(args.cg_add, len(violations))
        for w, _e in violations[:n_add]:
            _add_psd_window(mdl, y, t_param, w, P)
            active_windows.add(w)
        print(f'  [seed] added {n_add} PSD window cones; '
              f'active={len(active_windows)}', flush=True)

        lo = float(args.t_lo)
        hi = seed_t
        best_lb = lo
        converged = False
        last_feas_t = seed_t
        last_feas_y = y_vals.copy()
        cg_round_start = 0

    # Emergency-checkpoint closure: captures current mutable state from
    # the outer scope via nonlocal look-up (no binding at definition).
    def _emergency_checkpoint():
        if not args.checkpoint:
            return
        try:
            _save_checkpoint(
                args.checkpoint,
                active_windows=sorted(active_windows),
                lo=lo, hi=hi, best_lb=best_lb,
                cg_round_next=cg_round_in_progress,
                last_feas_t=last_feas_t,
                last_feas_y=last_feas_y,
                seed_t=seed_t,
                converged=converged,
                n_cg_candidates=len(cg_candidates),
                emergency=True,
            )
        except Exception as exc:
            print(f'  [ckpt] emergency write failed: {exc}',
                  flush=True)

    watcher = SpotWatcher(_emergency_checkpoint, interval_s=15.0)
    if args.checkpoint:
        watcher.start()

    cg_round_in_progress = cg_round_start  # used by emergency save

    for cg_round in range(cg_round_start, args.cg_rounds):
        cg_round_in_progress = cg_round
        if converged:
            break
        # Reset hi to seed_t at each round (bracket can widen up when
        # constraints are added, since old hi may now be infeasible).
        round_lo, round_hi = lo, seed_t
        print(f'\n  [CG round {cg_round+1}/{args.cg_rounds}]  '
              f'active={len(active_windows)}  '
              f'bracket=[{round_lo:.6f}, {round_hi:.6f}]  '
              f'RSS={_rss_bytes()/1024**3:.1f}GB', flush=True)

        for step in range(args.bisect):
            mid = 0.5 * (round_lo + round_hi)
            mid = max(round_lo + 1e-9, min(round_hi - 1e-9, mid))
            tp0 = time.time()
            feas = check_feasible(mid)
            step_s = time.time() - tp0
            if feas:
                round_hi = mid
                try:
                    last_feas_y = np.array(y.level())
                    last_feas_t = mid
                except Exception:
                    pass
            else:
                round_lo = mid
            bisect_history.append({
                'cg_round': cg_round, 'step': step, 't': mid,
                'feasible': feas, 'wall_s': step_s,
                'rss_gb': _rss_bytes() / 1024**3,
            })
            print(f'    [{step+1}/{args.bisect}] t={mid:.8f}  '
                  f'{"feas  " if feas else "infeas"}  '
                  f'({step_s:.1f}s)  RSS={_rss_bytes()/1024**3:.1f}GB  '
                  f'[{round_lo:.6f}, {round_hi:.6f}]', flush=True)
        lo = max(lo, round_lo)
        hi = round_hi
        best_lb = max(best_lb, lo)

        # Use the most recent feasible y (from bisection) for the
        # violation check.  Never call y.level() on an infeasible solve.
        y_vals = last_feas_y

        violations_all = _check_window_violations(
            y_vals, last_feas_t, P, active_windows, tol=args.violation_tol)
        violations = [(w, e) for (w, e) in violations_all
                       if w in cg_candidates_set]
        print(f'    violations after bisect: {len(violations)}',
              flush=True)
        cg_history.append({
            'cg_round': cg_round, 'lo': lo, 'hi': hi,
            'active_before': len(active_windows),
            'n_violations': len(violations),
            'rss_gb': _rss_bytes() / 1024**3,
        })
        if len(violations) == 0:
            converged = True
            print(f'    CG converged (no violations).', flush=True)
            # Final checkpoint with converged=True so a restart knows
            # to exit cleanly.
            if args.checkpoint:
                _save_checkpoint(
                    args.checkpoint,
                    active_windows=sorted(active_windows),
                    lo=lo, hi=hi, best_lb=best_lb,
                    cg_round_next=cg_round + 1,
                    last_feas_t=last_feas_t,
                    last_feas_y=last_feas_y,
                    seed_t=seed_t,
                    converged=True,
                    n_cg_candidates=len(cg_candidates))
            break
        n_add = min(args.cg_add, len(violations))
        for w, _e in violations[:n_add]:
            _add_psd_window(mdl, y, t_param, w, P)
            active_windows.add(w)
        print(f'    added {n_add} σ-reps; active={len(active_windows)}',
              flush=True)

        # Checkpoint after each round.  If we crash/preempt during the
        # NEXT round's bisection, we resume from this saved state.
        if args.checkpoint:
            _save_checkpoint(
                args.checkpoint,
                active_windows=sorted(active_windows),
                lo=lo, hi=hi, best_lb=best_lb,
                cg_round_next=cg_round + 1,
                last_feas_t=last_feas_t,
                last_feas_y=last_feas_y,
                seed_t=seed_t,
                converged=False,
                n_cg_candidates=len(cg_candidates))

    watcher.stop()
    total_wall = time.time() - t_start
    peak = guard.stop()
    try:
        mdl.dispose()
    except Exception:
        pass
    gc.collect()

    result = {
        'ok': True,
        'converged': converged,
        'rss_over_limit': guard.over_limit,
        'd': d, 'order': order,
        'threads': int(args.threads),
        'tol': args.tol,
        'upper_loc': bool(args.upper_loc),
        'use_z2': bool(args.use_z2),
        'cg_rounds_max': int(args.cg_rounds),
        'cg_rounds_used': len(cg_history),
        'cg_add': int(args.cg_add),
        'bisect_per_round': int(args.bisect),
        'rss_limit_gb': float(args.rss_limit_gb),
        'lb': best_lb, 'lo': lo, 'hi': hi,
        'bracket_mid': 0.5 * (lo + hi),
        'val_d_known': val_d_known.get(d),
        'n_active_windows': len(active_windows),
        'n_cg_candidates': len(cg_candidates),
        'total_wall_s': total_wall,
        'peak_rss_bytes': int(peak),
        'peak_rss_gb': peak / (1024 ** 3),
        'cg_history': cg_history,
        'bisect_history': bisect_history,
    }
    if val_d_known.get(d) and val_d_known[d] > 1.0:
        result['gc_pct'] = 100.0 * (best_lb - 1.0) / (val_d_known[d] - 1.0)

    print(f'\n[z2cg-fusion] DONE  converged={converged}  '
          f'lb={best_lb:.6f}  '
          f'gc={result.get("gc_pct", 0):.2f}%  '
          f'wall={total_wall:.1f}s  peak_RSS={peak/1024**3:.2f}GB  '
          f'windows={len(active_windows)}/{len(cg_candidates)}',
          flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--max-iters', type=int, default=400)
    p.add_argument('--cg-rounds', type=int, default=6)
    p.add_argument('--cg-add', type=int, default=15)
    p.add_argument('--bisect', type=int, default=4)
    p.add_argument('--t-lo', type=float, default=1.0)
    p.add_argument('--violation-tol', type=float, default=1e-6)
    p.add_argument('--upper-loc', action='store_true', default=True)
    p.add_argument('--no-upper-loc', dest='upper_loc',
                   action='store_false')
    p.add_argument('--no-z2', dest='use_z2', action='store_false',
                   default=True)
    p.add_argument('--rss-limit-gb', type=float, default=0.0,
                   help='0 = disable soft RSS limit (OS OOM killer only).')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to checkpoint JSON.  If exists on start, '
                        'resume from it.  After each CG round and on AWS '
                        'spot-preemption warning, save to this path. '
                        'Separate y-vector saved at <path>.y.npz.')
    p.add_argument('--json', type=str, default=None)
    args = p.parse_args()

    r = solve_z2cg_fusion(args)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.',
                    exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f'  JSON -> {args.json}')
    return 0 if r.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
