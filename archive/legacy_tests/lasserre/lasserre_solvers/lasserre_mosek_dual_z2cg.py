#!/usr/bin/env python
"""Z/2 full + constraint generation, composed on the Task-API dual.

Combines two space optimizations that were not previously composed:
  * Z/2 canonicalisation + moment-cone blockdiag + σ-rep dropping on
    localizing and window cones.  Halves n_y and the moment cone, cuts
    loc/win cones by σ-pairing.
  * Constraint generation (CG): start with NO PSD window cones, iterate
    (solve → check violations → add violated σ-reps → resolve) until no
    violations remain.

=====================================================================
CORRECTNESS (NO GAP-CLOSURE LOSS)
=====================================================================

Two invariants guarantee the bound equals the full (un-relaxed,
full-window) Lasserre bound:

  (I1) canonicalize_z2 is a lossless reformulation of the σ-invariant
       Lasserre problem: val(canonical) = val(full).
       Proof: the problem is σ-invariant in bin relabelling, so the
       σ-invariant pseudo-moment y* ∈ Lasserre-feasible iff its canonical
       representative is feasible in the quotient.  See
       lasserre/z2_blockdiag.py docstring.

  (I2) Adding the σ-representative of window w to the active set is
       equivalent to adding both w and σ(w), because under canonical y
       the M_k(q_W · y) and M_k(q_{σ(W)} · y) constraints reference the
       same y-rows (σ(α) maps to the same canonical index as α) and
       therefore encode the same linear inequality on y.  Adding either
       enforces both.

  (I3) CG converges to the full Lasserre bound iff no σ-rep window is
       violated at the terminating iterate.  This is tested after each
       solve via _check_window_violations over the σ-rep candidate set.
       The driver reports 'converged=True' only when the last round
       produced zero violated σ-reps.  If cg_rounds is exhausted without
       convergence, the return dict has 'converged=False' and the caller
       must not interpret the bound as tight.

=====================================================================
OUTPUTS
=====================================================================

Returns a dict with all of:
  * 'lb'              : best lower bound (lo at termination)
  * 'converged'       : True iff final violation-check returned 0
  * 'n_active_windows': number of σ-reps added during CG
  * 'peak_rss_gb'     : peak RSS (sampled in background thread)
  * 'cg_history'      : per-round timing + violation counts
  * 'bisect_history'  : per-probe timing + RSS
  * 'rss_over_limit'  : True iff RSS guard tripped
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

from lasserre_scalable import _precompute
from lasserre.dual_sdp import (
    build_dual_task, _alpha_lookup, _aggregate_scalar_triplet,
)
from lasserre_fusion import _hash_monos
from lasserre.z2_elim import canonicalize_z2
from lasserre.z2_blockdiag import (build_blockdiag_picks,
                                    localizing_sigma_reps,
                                    window_sigma_reps)
from lasserre.precompute import _check_window_violations
from lasserre.core import val_d_known
from lasserre_mosek_dual import _apply_task_params


# =====================================================================
# Scalar window constraint augmentation
# =====================================================================
#
# The primal Lasserre (moment-primal) problem has TWO constraints per
# window W:
#   (scalar)  t ≥ f_W(y)                   -- always active, cheap
#   (PSD)     M_{k-1}((t - q_W) y) ⪰ 0      -- added lazily via CG
#
# build_dual_task encodes only the PSD constraint.  Without the scalar
# constraint, the CG seed (active_windows=[]) has NO window enforcement
# and the Lasserre relaxation degenerates to the simplex moment problem
# (always feasible, λ*=0 everywhere, bound = 1).  That matched
# lasserre_scalable.solve_cg's behavior: it adds scalar constraints
# *for all windows* from round 0, then adds PSD windows lazily on top.
#
# This helper augments an existing task (built by build_dual_task) with
# scalar window constraints: for each W in `scalar_windows`, add a new
# scalar variable α_W ≥ 0, contribute -B_W[α] to the α-row scalar LP,
# and +t_val to α_W's objective coefficient.  Mathematical derivation
# in the z2cg docstring.  Canonical-y aware via _alpha_lookup.
# =====================================================================

def _augment_with_scalar_windows(task: mosek.Task,
                                  info: Dict[str, Any],
                                  P: Dict[str, Any],
                                  scalar_windows: List[int],
                                  t_val: float) -> None:
    """Append scalar-window constraints to an existing Farkas-LP task.

    Canonical-y aware: maps original monomial indices (e_i+e_j) through
    P['old_to_new'] when canonicalize_z2 has been applied.
    """
    if not scalar_windows:
        info['n_scalar_windows'] = 0
        info['sw_start'] = None
        return
    d = int(P['d'])
    M_mats = P['M_mats']
    bases_arr = np.asarray(P['bases'], dtype=np.int64)
    sorted_h = np.asarray(P['sorted_h'])
    sort_o = np.asarray(P['sort_o'])
    old_to_new = P.get('old_to_new')
    if old_to_new is not None:
        old_to_new = np.asarray(old_to_new, dtype=np.int64)

    n_sw = len(scalar_windows)
    sw_start = task.getnumvar()
    task.appendvars(n_sw)
    task.putvarboundslice(
        sw_start, sw_start + n_sw,
        [mosek.boundkey.lo] * n_sw,
        np.zeros(n_sw, dtype=np.float64),
        np.full(n_sw, +np.inf, dtype=np.float64),
    )

    # Canonical indices for all degree-2 monomials e_i + e_j.
    E_arr = np.eye(d, dtype=np.int64)
    ee_hash = _hash_monos(
        E_arr[:, None, :] + E_arr[None, :, :], bases_arr)  # (d, d)
    ee_idx_canonical = _alpha_lookup(
        ee_hash, sorted_h, sort_o, old_to_new)  # (d, d)

    # Build row/col/val triples for the scalar constraint contributions.
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for k, w in enumerate(scalar_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        if nz_i.size == 0:
            continue
        col = sw_start + k
        for ii, jj in zip(nz_i.tolist(), nz_j.tolist()):
            a_idx = int(ee_idx_canonical[ii, jj])
            if a_idx < 0:
                continue
            rows.append(a_idx)
            cols.append(col)
            # In the dual_sdp.py sign convention (row α sums to 0 with
            # slack v_α ≥ 0), a scalar primal constraint σ_W + f_W(y) = t
            # with slack σ_W ≥ 0 contributes +B_W[α] · α_W to row α.
            # Objective gets +t · α_W.  Boundedness of the LP comes from
            # the budget row λ + t·Σα_W ≤ 1 that we append separately.
            vals.append(+float(Mw[ii, jj]))

    if rows:
        r, c, v = _aggregate_scalar_triplet(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(vals, dtype=np.float64))
        task.putaijlist(r, c, v)

    # Objective coefficient: +t_val · α_W for each scalar window.
    # Farkas RHS is t on each scalar-window row, so b^T u = λ + t·Σα_W.
    # Normalization constraint below keeps this bounded.
    task.putclist(
        list(range(sw_start, sw_start + n_sw)),
        [+float(t_val)] * n_sw)

    # Farkas normalization: λ + t · Σ_W α_W ≤ 1.  Without this, the
    # Farkas LP is unbounded in α_W (positive objective coefficient,
    # no row-based upper bound).  With it, primal-feasibility ⟺
    # max (λ + t·Σα_W) = 0, primal-infeasibility ⟺ max = 1.  Matches
    # the existing dual_sdp.py convention where λ ∈ [0, 1] normalizes
    # the certificate magnitude.
    lam_idx = int(info['LAMBDA_IDX'])
    # Add a new constraint row: coefficients +1 on λ, +t on each α_W.
    # RHS ≤ 1.
    norm_row = task.getnumcon()
    task.appendcons(1)
    task.putconbound(norm_row, mosek.boundkey.up, -np.inf, 1.0)
    # Put coefficients into that row.
    norm_cols = [lam_idx] + list(range(sw_start, sw_start + n_sw))
    norm_vals = [1.0] + [float(t_val)] * n_sw
    task.putarow(norm_row, norm_cols, norm_vals)

    info['sw_start'] = int(sw_start)
    info['n_scalar_windows'] = int(n_sw)
    info['sw_t_val'] = float(t_val)


# =====================================================================
# RSS monitor (sampler + hard limit guard)
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
    def __init__(self, limit_gb: float, interval_s: float = 0.5):
        self.limit = int(limit_gb * (1024 ** 3))
        self.interval = interval_s
        self.peak = 0
        self.over_limit = False
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self):
        self.peak = _rss_bytes()

        def _loop():
            while not self._stop.wait(self.interval):
                r = _rss_bytes()
                if r > self.peak:
                    self.peak = r
                if r > self.limit and not self.over_limit:
                    self.over_limit = True
                    print(f'\n  !!! RSS {r/1024**3:.1f}GB exceeds '
                          f'limit {self.limit/1024**3:.1f}GB — flagging',
                          flush=True)

        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()

    def stop(self) -> int:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)
        return self.peak


# =====================================================================
# Driver
# =====================================================================

def solve_z2cg(
    d: int, order: int, *,
    t_lo: float = 1.0,
    t_hi: Optional[float] = None,
    bisect_per_round: int = 4,
    cg_rounds: int = 6,
    cg_add_per_round: int = 20,
    violation_tol: float = 1e-6,
    primary_tol: float = 1e-6,
    loose_tol: float = 1e-5,
    num_threads: int = 8,
    solve_form: str = 'dual',
    lambda_upper_bound: float = 1.0,
    feas_threshold: float = 0.25,
    infeas_threshold: float = 0.75,
    rss_limit_gb: float = 600.0,
    upper_loc: bool = True,
    max_iters: int = 400,
    use_scalar_windows: bool = True,
    use_z2: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Bisect on t with Z/2 full + CG.  Always rebuild per probe.

    Convergence guarantee: if the returned 'converged' is True AND no
    RSS-over-limit abort occurred, the reported lb equals the full
    Lasserre bound up to IPM tolerance primary_tol.  Otherwise (exhausted
    rounds / RSS abort) the bound is a valid LOWER bound on the full
    Lasserre value but may be loose.
    """
    t_start = time.time()
    guard = RSSGuard(limit_gb=rss_limit_gb)
    guard.start()

    if verbose:
        print(f'[z2cg] d={d} order={order}  threads={num_threads}  '
              f'cg_rounds={cg_rounds}  cg_add={cg_add_per_round}  '
              f'bisect/round={bisect_per_round}  '
              f'primary_tol={primary_tol}  rss_limit={rss_limit_gb}GB',
              flush=True)

    # --- precompute + Z/2 canonicalisation ---
    tp0 = time.time()
    # lazy ab_eiej: the 4D array would be huge at d=16; _check_window_
    # violations falls back to get_ab_eiej_slice which handles old_to_new
    # remap internally.
    P_raw = _precompute(d, order, verbose=verbose, lazy_ab_eiej=True)
    if use_z2:
        P = canonicalize_z2(P_raw, verbose=verbose)
        bd = build_blockdiag_picks(P['basis'], P['idx'], P['n_y'])
        loc_fixed, loc_pairs = localizing_sigma_reps(d)
        active_loc = list(loc_fixed) + [p for p, _ in loc_pairs]
        win_fixed, win_pairs = window_sigma_reps(d, P_raw['windows'])
        nontriv = set(P_raw['nontrivial_windows'])
        cg_candidates = [
            w for w in (list(win_fixed) + [p for p, _ in win_pairs])
            if w in nontriv
        ]
    else:
        P = P_raw
        bd = None
        active_loc = None   # build_dual_task default = range(d)
        nontriv = set(P_raw['nontrivial_windows'])
        cg_candidates = list(nontriv)
    cg_candidates_set = set(cg_candidates)
    precompute_s = time.time() - tp0
    if verbose:
        if use_z2:
            print(f'  precompute + z2 canon: {precompute_s:.1f}s  '
                  f'n_y {P_raw["n_y"]} -> {P["n_y"]}  '
                  f'moment {P["n_basis"]}^2 -> sym({bd["n_sym"]})+'
                  f'anti({bd["n_anti"]})  '
                  f'loc {d} -> {len(active_loc)}  '
                  f'win cand {len(cg_candidates)} of {len(nontriv)} nontriv  '
                  f'sw={"on" if use_scalar_windows else "off"}',
                  flush=True)
        else:
            print(f'  precompute only (no-z2): {precompute_s:.1f}s  '
                  f'n_y={P["n_y"]}  n_basis={P["n_basis"]}  '
                  f'loc={d}  win cand {len(cg_candidates)}  '
                  f'sw={"on" if use_scalar_windows else "off"}',
                  flush=True)
    if guard.over_limit:
        return {'ok': False, 'reason': 'rss_over_limit_precompute',
                'peak_rss_gb': guard.peak / 1024**3}

    env = mosek.Env()
    active_windows: List[int] = []
    scalar_windows: List[int] = list(cg_candidates) if use_scalar_windows else []
    cg_history: List[Dict[str, Any]] = []
    bisect_history: List[Dict[str, Any]] = []
    n_rebuilds = 0

    def _probe_and_get_y(t_val: float, tol: float,
                         label: str = '') -> Dict[str, Any]:
        """Build + solve + extract λ* and canonical y.  Returns dict with
        verdict, lam, y_canonical, wall_s, rss_gb.  The task is disposed
        before return (so no accumulation across probes)."""
        nonlocal n_rebuilds
        t0 = time.time()
        task, info = build_dual_task(
            P, t_val=t_val, env=env,
            include_upper_loc=upper_loc,
            z2_blockdiag_map=bd,
            active_loc=active_loc,
            active_windows=list(active_windows),
            lambda_upper_bound=lambda_upper_bound,
            verbose=False)
        # Augment with scalar window constraints — essential for the
        # CG seed to be meaningful.  See module docstring for why.
        _augment_with_scalar_windows(
            task, info, P, scalar_windows, t_val)
        n_rebuilds += 1
        _apply_task_params(
            task, tol=tol,
            max_iterations=max_iters,
            solve_form=solve_form,
            num_threads=num_threads,
            verbose=False)

        try:
            task.optimize()
        except Exception as exc:
            try:
                task.__del__()
            except Exception:
                pass
            return {'verdict': 'error',
                    'lam': float('nan'), 'y': None,
                    'wall_s': time.time() - t0, 'label': label,
                    'error': f'{type(exc).__name__}: {exc}',
                    'rss_gb': _rss_bytes() / 1024**3}

        solsta = task.getsolsta(mosek.soltype.itr)
        try:
            lam = float(task.getxxslice(
                mosek.soltype.itr,
                info['LAMBDA_IDX'], info['LAMBDA_IDX'] + 1)[0])
        except Exception:
            lam = float('nan')
        # Extract y (canonical) from dual of stationarity rows.
        # Empirically MOSEK's getyslice for this MAX LP returns the
        # moment-primal y scaled by some factor (observed y[0] = 1.22
        # at t=1.22 when the correct value is y[0] = 1).  Since the
        # moment-primal problem is HOMOGENEOUS in y (all constraints
        # are homogeneous except y_0 = 1), normalising y by its value
        # at α=0 recovers the correct normalized moment-primal y.
        try:
            y_raw = np.array(task.getyslice(
                mosek.soltype.itr, 0, info['n_cons']), dtype=np.float64)
            alpha_zero = tuple(0 for _ in range(int(P['d'])))
            alpha_zero_row = int(P['idx'][alpha_zero])
            y_at_zero_raw = float(y_raw[alpha_zero_row])
            if abs(y_at_zero_raw) > 1e-12:
                y_canon = y_raw / y_at_zero_raw
            else:
                # Degenerate case — MOSEK returned zero y[0].  This
                # typically happens in infeasible Farkas (λ* = 1)
                # where the dual y isn't a moment-primal witness.  We
                # leave the scale alone; the caller (violation check
                # under CG) should only use y from feasible probes.
                y_canon = y_raw
        except Exception:
            y_canon = None
            y_at_zero_raw = float('nan')

        if solsta == mosek.solsta.optimal:
            if lam >= infeas_threshold * lambda_upper_bound:
                verdict = 'infeas'
            elif lam <= feas_threshold * lambda_upper_bound:
                verdict = 'feas'
            else:
                verdict = 'uncertain'
        elif solsta == mosek.solsta.dual_infeas_cer:
            verdict = 'infeas'
        else:
            verdict = 'uncertain'

        try:
            task.__del__()
        except Exception:
            pass
        gc.collect()

        try:
            y_at_zero_final = float(y_canon[alpha_zero_row]) if y_canon is not None else float('nan')
        except Exception:
            y_at_zero_final = float('nan')
        return {
            'verdict': verdict, 'lam': lam, 'y': y_canon,
            'y_at_zero_raw': locals().get('y_at_zero_raw', float('nan')),
            'y_at_zero_final': y_at_zero_final,
            'solsta': str(solsta).split('.')[-1],
            'wall_s': time.time() - t0, 'label': label,
            'rss_gb': _rss_bytes() / 1024**3,
            't_val': float(t_val), 'tol': float(tol),
        }

    # --- initialise bracket ---
    if t_hi is None:
        t_hi = (val_d_known.get(d, 1.3) + 0.05)
    lo, hi = float(t_lo), float(t_hi)

    # hi probe at primary_tol (accurate y needed for first violation check)
    if verbose:
        print(f'\n  [round 0] seeding: probe at hi={hi:.4f} with no '
              f'PSD windows', flush=True)
    r = _probe_and_get_y(hi, primary_tol, label='round0_hi')
    bisect_history.append(r)
    if verbose:
        print(f'    t={r["t_val"]:.6f}  {r["verdict"]:6s}  '
              f'lam={r["lam"]:+.3e}  wall={r["wall_s"]:.1f}s  '
              f'RSS={r["rss_gb"]:.1f}GB  '
              f'y[0]_raw={r.get("y_at_zero_raw", "?"):+.4e}  '
              f'y[0]_final={r.get("y_at_zero_final", "?"):+.4e}',
              flush=True)
    if guard.over_limit:
        return {'ok': False, 'reason': 'rss_over_limit_round0',
                'peak_rss_gb': guard.peak / 1024**3}

    # Expand hi if first probe wasn't feasible (scalar bound may be > default).
    tries = 0
    while r['verdict'] != 'feas' and tries < 3:
        hi *= 1.3
        if verbose:
            print(f'    hi not feas, expand to {hi:.4f}', flush=True)
        r = _probe_and_get_y(hi, primary_tol, label=f'round0_hi_retry{tries}')
        bisect_history.append(r)
        tries += 1
        if guard.over_limit:
            return {'ok': False, 'reason': 'rss_over_limit_round0_retry',
                    'peak_rss_gb': guard.peak / 1024**3}
    if r['verdict'] != 'feas':
        return {'ok': False, 'reason': 'could_not_find_feasible_hi',
                'peak_rss_gb': guard.peak / 1024**3,
                'bisect_history': bisect_history}

    y_last_feas = r['y']
    t_last_feas = r['t_val']

    converged = False

    # --- CG rounds ---
    for cg_round in range(cg_rounds):
        if guard.over_limit:
            break

        # --- violation check at the most recent FEASIBLE y ---
        # Correctness: we check ALL σ-rep nontrivial windows not yet active.
        # Under canonical y, checking σ-rep is equivalent to checking the
        # full orbit (I2 in the docstring).
        tp0 = time.time()
        if y_last_feas is None:
            violations = []
        else:
            violations_all = _check_window_violations(
                y_last_feas, t_last_feas, P,
                set(active_windows), tol=violation_tol)
            # Only keep candidate σ-reps (guarantees no duplicate orbit work).
            violations = [(w, eig) for (w, eig) in violations_all
                           if w in cg_candidates_set]
        vcheck_s = time.time() - tp0

        if verbose:
            print(f'\n  [CG round {cg_round+1}/{cg_rounds}]  active={len(active_windows)}  '
                  f'violated={len(violations)}  '
                  f'(vcheck {vcheck_s:.1f}s)  '
                  f'RSS={_rss_bytes()/1024**3:.1f}GB', flush=True)
        cg_history.append({
            'cg_round': cg_round,
            'active_before': len(active_windows),
            'n_violated': len(violations),
            'violation_vcheck_s': vcheck_s,
            't_used_for_vcheck': float(t_last_feas),
            'rss_gb': _rss_bytes() / 1024**3,
        })

        if len(violations) == 0:
            converged = True
            if verbose:
                print(f'    CG converged: no violations remain.  '
                      f'Final lb is the full Lasserre value to IPM tol.',
                      flush=True)
            break

        # Add top-k violated (worst min-eig first).  _check_window_violations
        # already sorts ascending by eig.
        n_add = min(cg_add_per_round, len(violations))
        for (w, _eig) in violations[:n_add]:
            active_windows.append(int(w))
        if verbose:
            worst = violations[0][1]
            print(f'    added {n_add} σ-reps '
                  f'(worst eig={worst:.3e}); total active={len(active_windows)}',
                  flush=True)

        # --- bisection with current active_windows ---
        round_lo, round_hi = lo, hi
        for step in range(bisect_per_round):
            if guard.over_limit:
                break
            mid = 0.5 * (round_lo + round_hi)
            mid = max(round_lo + 1e-9, min(round_hi - 1e-9, mid))
            # Use loose tol on mid-bracket probes (verdict is sharp), tight
            # tol on final step so y is accurate for next violation check.
            use_tol = primary_tol if step == bisect_per_round - 1 else loose_tol
            r = _probe_and_get_y(
                mid, use_tol, label=f'round{cg_round+1}_step{step+1}')
            bisect_history.append(r)
            if r['verdict'] == 'feas':
                round_hi = mid
                y_last_feas = r['y']
                t_last_feas = mid
            elif r['verdict'] == 'infeas':
                round_lo = mid
            # uncertain: no bracket move
            if verbose:
                print(f'    [{step+1}/{bisect_per_round}] t={mid:.8f}  '
                      f'{r["verdict"]:6s}  lam={r["lam"]:+.3e}  '
                      f'({r["wall_s"]:.1f}s)  RSS={r["rss_gb"]:.1f}GB  '
                      f'tol={use_tol:.0e}  bracket=[{round_lo:.6f},{round_hi:.6f}]',
                      flush=True)
        lo, hi = round_lo, round_hi

        # Ensure the violation check next round uses a feasible y.
        # If the final probe above was INfeasible, fall back to the most
        # recent feas iterate.
        if r['verdict'] != 'feas' and y_last_feas is None:
            if verbose:
                print('    WARNING: no feasible iterate available after '
                      'round; violation check may be unreliable.',
                      flush=True)

    total_wall = time.time() - t_start
    peak_bytes = guard.stop()

    result = {
        'ok': True,
        'converged': bool(converged),
        'rss_over_limit': bool(guard.over_limit),
        'd': d, 'order': order,
        'threads': int(num_threads),
        'primary_tol': float(primary_tol),
        'loose_tol': float(loose_tol),
        'violation_tol': float(violation_tol),
        'upper_loc': bool(upper_loc),
        'bisect_per_round': int(bisect_per_round),
        'cg_rounds_max': int(cg_rounds),
        'cg_rounds_used': len(cg_history),
        'cg_add_per_round': int(cg_add_per_round),
        'rss_limit_gb': float(rss_limit_gb),
        'lo': float(lo), 'hi': float(hi),
        'lb': float(lo),
        'bracket_mid': 0.5 * (lo + hi),
        'val_d_known': val_d_known.get(d),
        'n_active_windows': len(active_windows),
        'n_cg_candidates': len(cg_candidates),
        'n_rebuilds': int(n_rebuilds),
        'total_wall_s': float(total_wall),
        'peak_rss_bytes': int(peak_bytes),
        'peak_rss_gb': peak_bytes / (1024 ** 3),
        'cg_history': cg_history,
        'bisect_history': bisect_history,
    }
    gc_pct = None
    if val_d_known.get(d) and val_d_known[d] > 1.0:
        gc_pct = 100.0 * (lo - 1.0) / (val_d_known[d] - 1.0)
        result['gc_pct'] = float(gc_pct)

    if verbose:
        print(f'\n[z2cg] DONE  converged={converged}  '
              f'lb={lo:.6f}  '
              f'{"gc=" + format(gc_pct, ".2f") + "%" if gc_pct is not None else ""}  '
              f'wall={total_wall:.1f}s  peak_RSS={peak_bytes/1024**3:.2f}GB  '
              f'windows={len(active_windows)}/{len(cg_candidates)} (σ-reps)  '
              f'rebuilds={n_rebuilds}', flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--t-lo', type=float, default=1.0)
    p.add_argument('--t-hi', type=float, default=None)
    p.add_argument('--bisect', type=int, default=4,
                   help='bisection probes per CG round')
    p.add_argument('--cg-rounds', type=int, default=6)
    p.add_argument('--cg-add', type=int, default=20)
    p.add_argument('--violation-tol', type=float, default=1e-6)
    p.add_argument('--primary-tol', type=float, default=1e-6)
    p.add_argument('--loose-tol', type=float, default=1e-5)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--solve-form', type=str, default='dual',
                   choices=('primal', 'dual', 'free'))
    p.add_argument('--lambda-ub', type=float, default=1.0)
    p.add_argument('--feas-threshold', type=float, default=0.25)
    p.add_argument('--infeas-threshold', type=float, default=0.75)
    p.add_argument('--rss-limit-gb', type=float, default=600.0)
    p.add_argument('--no-upper-loc', dest='upper_loc',
                   action='store_false', default=True)
    p.add_argument('--max-iters', type=int, default=400)
    p.add_argument('--no-scalar-windows', dest='use_scalar_windows',
                   action='store_false', default=True)
    p.add_argument('--no-z2', dest='use_z2', action='store_false',
                   default=True)
    p.add_argument('--json', type=str, default=None)
    args = p.parse_args()

    r = solve_z2cg(
        args.d, args.order,
        t_lo=args.t_lo, t_hi=args.t_hi,
        bisect_per_round=args.bisect,
        cg_rounds=args.cg_rounds, cg_add_per_round=args.cg_add,
        violation_tol=args.violation_tol,
        primary_tol=args.primary_tol,
        loose_tol=args.loose_tol,
        num_threads=args.threads,
        solve_form=args.solve_form,
        lambda_upper_bound=args.lambda_ub,
        feas_threshold=args.feas_threshold,
        infeas_threshold=args.infeas_threshold,
        rss_limit_gb=args.rss_limit_gb,
        upper_loc=args.upper_loc,
        max_iters=args.max_iters,
        use_scalar_windows=args.use_scalar_windows,
        use_z2=args.use_z2,
        verbose=True)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.',
                    exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f'  JSON -> {args.json}')
    return 0 if r.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
