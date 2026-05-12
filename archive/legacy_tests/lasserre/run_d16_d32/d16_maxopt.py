#!/usr/bin/env python
"""d=16 Lasserre SOS with every composable space optimization.

Stacks:
  * Constraint generation   — start with 0 PSD window cones, add only
                              violated ones (typically ~20-50 instead
                              of ~500).  Window cones dominate bar-entry
                              storage by ~85% at d=16, so this is the
                              single biggest memory saver we have.
  * Upper-localizing cones  — (1 − μ_i) ≥ 0.  d extra small loc cones.
                              Adds ~5% gap closure for modest memory.
  * Lazy ab_eiej_idx        — skip the 4D index array (would be huge
                              at d=16, mostly zeros).
  * Thread control          — cap MOSEK at --threads (default 16).  The
                              per-thread scratch is ~1.5-3 GB at d=16
                              with Z/2; an unbounded run easily OOMs
                              because the pod has 128 cores.
  * Graph-partition Cholesky— sparser Schur factor on banded problems.
  * Relaxed tol (1e-5)      — fewer IPM iterations per probe.
  * Moderate cg rounds      — cap --cg-rounds to avoid runaway window
                              expansion.

What is NOT stacked (with reasons):
  * Z/2 blockdiag          — composition with CG is non-trivial (the
                             σ-rep window dropping depends on the full
                             window set being known up-front, which CG
                             violates).  We run without Z/2 here.
  * Clique decomposition   — empirically useless at d ≤ 12 in our runs.

RSS is sampled every 0.5s in a background thread and peaked out; the
run aborts if RSS exceeds --rss-limit-gb (default 650 on a 755 GB pod).

Usage:
    python3 tests/d16_maxopt.py --d 16 --order 3 --threads 16 \\
        --rss-limit-gb 650 --cg-rounds 6 --cg-add 10 \\
        --bisect 8 --tol 1e-5 --json data/d16_maxopt.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from mosek.fusion import (Model, Domain, Expr, Matrix,
                           ObjectiveSense, SolutionStatus)

from lasserre_scalable import _precompute
from lasserre.precompute import _add_psd_window
from lasserre.precompute import _build_base_constraints
from lasserre.precompute import _check_window_violations
from lasserre.core import val_d_known


# ---------------------------------------------------------------------
# RSS monitoring
# ---------------------------------------------------------------------

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
    """Background sampler with hard RSS limit.

    If RSS exceeds ``limit_bytes`` it sets ``over_limit`` and prints a
    warning.  The caller can abort the MOSEK solve at the next CG
    checkpoint.  We deliberately do NOT self._exit — the running MOSEK
    solve cannot be interrupted cleanly, so an over-limit condition
    should be handled at CG-round boundaries.
    """

    def __init__(self, limit_gb: float, interval_s: float = 0.5):
        self.limit = int(limit_gb * (1024 ** 3))
        self.interval = interval_s
        self.peak = 0
        self.over_limit = False
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self.history: list = []  # (t, rss) samples at "mark" points

    def start(self):
        self.peak = _rss_bytes()

        def _loop():
            while not self._stop.wait(self.interval):
                r = _rss_bytes()
                if r > self.peak:
                    self.peak = r
                if r > self.limit and not self.over_limit:
                    self.over_limit = True
                    print(f'\n  !!! RSS {r / 1024**3:.1f} GB exceeds '
                          f'limit {self.limit / 1024**3:.1f} GB — '
                          f'flagging abort at next CG checkpoint',
                          flush=True)

        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()

    def mark(self, label: str):
        r = _rss_bytes()
        if r > self.peak:
            self.peak = r
        self.history.append({
            't': time.time(), 'label': label,
            'rss_bytes': r, 'rss_gb': r / (1024 ** 3),
            'peak_so_far_gb': self.peak / (1024 ** 3),
        })
        print(f'  [rss] {label:<30s} cur={r/1024**3:.2f}GB  '
              f'peak={self.peak/1024**3:.2f}GB', flush=True)

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)
        return {
            'peak_bytes': self.peak,
            'peak_gb': self.peak / (1024 ** 3),
            'history': self.history,
        }


# ---------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------

def solve_d16_maxopt(args) -> Dict[str, Any]:
    d = int(args.d)
    order = int(args.order)
    guard = RSSGuard(limit_gb=args.rss_limit_gb)
    guard.start()
    t_start = time.time()

    print(f'[d16_maxopt] d={d} order={order}  threads={args.threads}  '
          f'tol={args.tol}  cg_rounds={args.cg_rounds}  '
          f'cg_add={args.cg_add}  bisect={args.bisect}  '
          f'upper_loc={args.upper_loc}  '
          f'rss_limit={args.rss_limit_gb}GB', flush=True)

    # --- precompute ---
    guard.mark('start')
    tp0 = time.time()
    P = _precompute(d, order, verbose=True, lazy_ab_eiej=True)
    precompute_s = time.time() - tp0
    guard.mark(f'post_precompute ({precompute_s:.1f}s)')
    if guard.over_limit:
        return {'ok': False, 'reason': 'rss_exceeded_precompute',
                'peak_gb': guard.peak / 1024**3}

    n_y = int(P['n_y'])
    n_win = int(P['n_win'])
    n_loc = int(P['n_loc'])
    print(f'  sizes: n_y={n_y:,}  n_win={n_win:,}  n_loc={n_loc}  '
          f'n_basis={P["n_basis"]}', flush=True)

    # --- build Fusion model ---
    mdl = Model('d16_maxopt')
    mdl.setSolverParam('intpntCoTolRelGap', args.tol)
    mdl.setSolverParam('intpntCoTolPfeas', args.tol)
    mdl.setSolverParam('intpntCoTolDfeas', args.tol)
    mdl.setSolverParam('intpntCoTolMuRed', args.tol)
    mdl.setSolverParam('numThreads', int(args.threads))
    mdl.setSolverParam('intpntBasis', 'never')
    mdl.setSolverParam('intpntMaxIterations', int(args.max_iters))
    # graphpar reordering (better for banded-sparse Schur complements)
    try:
        mdl.setSolverParam('intpntOrderMethod', 'forceGraphpar')
    except Exception:
        try:
            mdl.setSolverParam('intpntOrderMethod', 'tryGraphpar')
        except Exception:
            pass

    y = mdl.variable('y', n_y, Domain.greaterThan(0.0))
    t_param = mdl.parameter('t')

    guard.mark('pre_base_constraints')
    tp0 = time.time()
    _build_base_constraints(mdl, y, P, args.upper_loc, verbose=True)
    base_build_s = time.time() - tp0
    guard.mark(f'post_base_constraints ({base_build_s:.1f}s)')
    if guard.over_limit:
        return {'ok': False, 'reason': 'rss_exceeded_base_build',
                'peak_gb': guard.peak / 1024**3}

    # Scalar window constraints: t_param >= f_W(y) for all W
    tp0 = time.time()
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col,
                         Expr.reshape(t_param, 1, 1)))
    mdl.constraint('win_scalar', Expr.sub(t_rep, f_all),
                   Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))
    scalar_window_s = time.time() - tp0
    guard.mark(f'post_scalar_window_constraints ({scalar_window_s:.1f}s)')
    if guard.over_limit:
        return {'ok': False, 'reason': 'rss_exceeded_scalar_windows',
                'peak_gb': guard.peak / 1024**3}

    # --- CG loop ---
    active_windows: set = set()

    def check_feasible(t_val: float) -> bool:
        t_param.setValue(t_val)
        try:
            mdl.solve()
            ps = mdl.getPrimalSolutionStatus()
            return ps in (SolutionStatus.Optimal, SolutionStatus.Feasible)
        except Exception as exc:
            print(f'    solve error: {type(exc).__name__}: {exc}',
                  flush=True)
            return False

    # Round 0: seed violations from scalar-only boundary
    guard.mark('pre_round0_solve')
    tp0 = time.time()
    t_seed = 1.5  # above scalar bound 1.0 but below val(d)+margin
    if not check_feasible(t_seed):
        t_seed = 3.0
        check_feasible(t_seed)
    for _ in range(4):
        t_param.setValue(t_seed)
        mdl.solve()
        break  # just one solve at t_seed — CG seeding
    y_vals = np.array(y.level())
    round0_s = time.time() - tp0
    guard.mark(f'post_round0_solve ({round0_s:.1f}s)')

    violations = _check_window_violations(
        y_vals, t_seed, P, active_windows)
    print(f'  [round 0] {len(violations)} violated windows at '
          f't_seed={t_seed:.4f}', flush=True)

    history = [{'phase': 'round0', 't': t_seed, 'n_active': 0,
                'n_violations': len(violations),
                'wall_s': round0_s,
                'rss_gb': _rss_bytes() / 1024**3}]

    def _add_violations(violations):
        n_add = min(args.cg_add, len(violations))
        for w, _eig in violations[:n_add]:
            _add_psd_window(mdl, y, t_param, w, P)
            active_windows.add(w)
        return n_add

    if violations:
        n = _add_violations(violations)
        print(f'    added {n} PSD window cones (active={len(active_windows)})',
              flush=True)
        guard.mark(f'post_seed_windows ({len(active_windows)} active)')

    # Main CG + bisection loop
    lb_lo, lb_hi = 1.0, val_d_known.get(d, 1.5) + 0.05
    best_lb = lb_lo
    bisect_history = []

    for cg_round in range(1, args.cg_rounds + 1):
        if guard.over_limit:
            print(f'\n  RSS limit hit — aborting at CG round {cg_round}',
                  flush=True)
            break
        print(f'\n  [CG round {cg_round}/{args.cg_rounds}] '
              f'{len(active_windows)} PSD windows active  '
              f'bracket=[{lb_lo:.6f}, {lb_hi:.6f}]  '
              f'RSS={_rss_bytes()/1024**3:.1f}GB', flush=True)

        # Quick bisection with current active windows
        cur_hi = lb_hi
        cur_lo = lb_lo
        for step in range(args.bisect):
            if guard.over_limit:
                break
            mid = 0.5 * (cur_lo + cur_hi)
            mid = max(cur_lo + 1e-9, min(cur_hi - 1e-9, mid))
            tp0 = time.time()
            feas = check_feasible(mid)
            step_s = time.time() - tp0
            if feas:
                cur_hi = mid
            else:
                cur_lo = mid
            bisect_history.append({
                'cg_round': cg_round, 'step': step,
                't': mid, 'feasible': feas, 'wall_s': step_s,
                'rss_gb': _rss_bytes() / 1024**3,
            })
            print(f'    [{step+1}/{args.bisect}] t={mid:.8f}  '
                  f'{"feas  " if feas else "infeas"}  '
                  f'({step_s:.1f}s)  RSS={_rss_bytes()/1024**3:.1f}GB  '
                  f'[{cur_lo:.6f}, {cur_hi:.6f}]', flush=True)
        best_lb = max(best_lb, cur_lo)

        # Check violations at current lb
        tp0 = time.time()
        t_param.setValue(cur_lo)
        mdl.solve()
        y_vals = np.array(y.level())
        vchk_s = time.time() - tp0
        violations = _check_window_violations(
            y_vals, cur_lo, P, active_windows)
        print(f'    violations after round: {len(violations)}  '
              f'(vcheck {vchk_s:.1f}s)', flush=True)
        history.append({
            'phase': f'cg_round_{cg_round}', 't': cur_lo,
            'n_active': len(active_windows),
            'n_violations': len(violations), 'wall_s': vchk_s,
            'rss_gb': _rss_bytes() / 1024**3,
        })

        if not violations:
            print(f'    no further violations — CG converged', flush=True)
            break

        n = _add_violations(violations)
        guard.mark(f'post_cg_round_{cg_round} ({len(active_windows)} active)')
        lb_hi = min(lb_hi, cur_hi)  # never relax the upper bracket

    total_wall = time.time() - t_start
    samp = guard.stop()

    mdl.dispose()
    gc.collect()

    result = {
        'ok': True,
        'd': d, 'order': order,
        'threads': int(args.threads),
        'tol': args.tol,
        'upper_loc': bool(args.upper_loc),
        'cg_rounds': int(args.cg_rounds),
        'cg_add': int(args.cg_add),
        'bisect_per_round': int(args.bisect),
        'rss_limit_gb': float(args.rss_limit_gb),
        'lb': best_lb,
        'val_d_known': val_d_known.get(d),
        'n_active_windows': len(active_windows),
        'n_win_total': n_win,
        'total_wall_s': total_wall,
        'peak_rss_bytes': samp['peak_bytes'],
        'peak_rss_gb': samp['peak_gb'],
        'rss_markers': samp['history'],
        'cg_history': history,
        'bisect_history': bisect_history,
        'rss_over_limit': guard.over_limit,
    }

    print(f'\n[d16_maxopt] DONE  lb={best_lb:.6f}  '
          f'wall={total_wall:.1f}s  peak_RSS={samp["peak_gb"]:.2f}GB  '
          f'windows={len(active_windows)}/{n_win}  '
          f'over_limit={guard.over_limit}', flush=True)

    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--max-iters', type=int, default=400)
    p.add_argument('--cg-rounds', type=int, default=6)
    p.add_argument('--cg-add', type=int, default=10)
    p.add_argument('--bisect', type=int, default=6)
    p.add_argument('--upper-loc', action='store_true', default=True)
    p.add_argument('--no-upper-loc', dest='upper_loc',
                   action='store_false')
    p.add_argument('--rss-limit-gb', type=float, default=650.0,
                   help='Abort at next CG checkpoint if RSS exceeds this.')
    p.add_argument('--json', type=str, default=None)
    args = p.parse_args()

    r = solve_d16_maxopt(args)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.',
                    exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f'  JSON -> {args.json}')
    return 0 if r.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
