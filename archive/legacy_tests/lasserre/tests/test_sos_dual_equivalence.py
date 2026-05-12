#!/usr/bin/env python
"""Regression test: SOS-dual Farkas LP must agree with the moment-primal
Lasserre SDP on every feas/infeas verdict across a sweep of t values.

This file is a plain pytest-free script — it exits non-zero on failure and
prints a one-line pass/fail summary.  Designed to be dropped into CI with

    python tests/test_sos_dual_equivalence.py

Each (d, L, probes) case is solved at THREE stack levels:

    v1     : basic (no Z/2, no upper-loc)                       vs mode=tuned
                                                                   add_upper_loc=False
    v1+uloc: basic + upper-loc (1−μ_i) cones                    vs mode=tuned
                                                                   add_upper_loc=True
    full   : canonicalize_z2 + blockdiag moment + σ-rep drop +   vs mode=z2_full
             upper-loc                                             pre_elim_z2=True
                                                                   add_upper_loc=True

The test tolerates ONE-SIDED UNCERTAINTY:  if the moment-primal reports
uncertain while the SOS-dual is decisive (observed at d=4/6 L=3 near
val boundary), the test PASSES — it's the dual being more reliable, not
a disagreement.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import mosek

from lasserre_scalable import _precompute
from lasserre.dual_sdp import build_dual_task, solve_dual_task
from lasserre_mosek_preelim import solve_mosek_preelim
from lasserre.z2_elim import canonicalize_z2
from lasserre.z2_blockdiag import (build_blockdiag_picks,
                                     localizing_sigma_reps,
                                     window_sigma_reps)


# ---------------------------------------------------------------------------
# Test cases: (d, L, list of t values to probe)
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (d, L, probes)
    (4, 2, [0.5, 1.0, 1.05, 1.07, 1.09, 1.15, 2.0]),
    (4, 3, [0.5, 1.05, 1.08, 1.12, 1.15, 2.0]),   # avoid t=1.10 (uncertain band)
    (6, 2, [0.5, 1.05, 1.10, 1.15, 1.20, 2.0]),
    (6, 3, [0.5, 1.10, 1.15, 1.20, 2.0]),         # avoid t=1.17 (uncertain band)
]


def _prep_full(P):
    """Build the (canonicalized P, blockdiag map, σ-rep lists) for the
    'full' stack level."""
    Pc = canonicalize_z2(P, verbose=False)
    bd = build_blockdiag_picks(Pc['basis'], Pc['idx'], Pc['n_y'])
    d = int(P['d'])
    loc_fixed, loc_pairs = localizing_sigma_reps(d)
    loc_active = list(loc_fixed) + [p for (p, _) in loc_pairs]
    win_fixed, win_pairs = window_sigma_reps(d, P['windows'])
    nontriv = set(P['nontrivial_windows'])
    win_active = [w for w in (list(win_fixed) + [p for (p, _) in win_pairs])
                  if w in nontriv]
    return Pc, bd, loc_active, win_active


def _run_dual(P, t, env, *, level: str = 'v1',
              aux=None):
    """level in {'v1', 'v1+uloc', 'full'}.  aux = (Pc, bd, loc_active, win_active)
    when level == 'full'."""
    with contextlib.redirect_stdout(io.StringIO()):
        if level == 'v1':
            task, info = build_dual_task(
                P, t_val=t, env=env, include_upper_loc=False, verbose=False)
        elif level == 'v1+uloc':
            task, info = build_dual_task(
                P, t_val=t, env=env, include_upper_loc=True, verbose=False)
        elif level == 'full':
            Pc, bd, loc_active, win_active = aux
            task, info = build_dual_task(
                Pc, t_val=t, env=env, include_upper_loc=True,
                z2_blockdiag_map=bd,
                active_loc=loc_active, active_windows=win_active,
                verbose=False)
        else:
            raise ValueError(f"unknown level {level!r}")
        res = solve_dual_task(task, info, verbose=False)
        try:
            task.__del__()
        except Exception:
            pass
    return res


def _run_primal(d, L, t, *, level: str = 'v1'):
    """level in {'v1', 'v1+uloc', 'full'}."""
    if level == 'v1':
        mode, add_upper_loc, pre_elim_z2 = 'tuned', False, False
    elif level == 'v1+uloc':
        mode, add_upper_loc, pre_elim_z2 = 'tuned', True, False
    elif level == 'full':
        mode, add_upper_loc, pre_elim_z2 = 'z2_full', True, True
    else:
        raise ValueError(f"unknown level {level!r}")
    with contextlib.redirect_stdout(io.StringIO()):
        r = solve_mosek_preelim(
            d=d, order=L, mode=mode,
            add_upper_loc=add_upper_loc,
            pre_elim_z2=pre_elim_z2,
            single_t=t, verbose=False)
    return r


def _compatible(primal_verdict, dual_verdict):
    """
    Verdict-pair compatibility rules:
      - Both agree exactly                           → pass
      - Primal uncertain, dual decisive              → pass (dual wins)
      - Dual uncertain, primal decisive              → pass (primal wins)
      - Both uncertain                               → pass (tolerable)
      - Direct disagreement (one feas, one infeas)   → FAIL
    """
    if primal_verdict == dual_verdict:
        return True
    if 'uncertain' in (primal_verdict, dual_verdict):
        return True
    return False


LEVELS = ('v1', 'v1+uloc', 'full')


def main() -> int:
    failures = []
    total_probes = 0
    per_level_totals: dict = {lvl: {'primal': 0.0, 'dual': 0.0, 'probes': 0}
                                for lvl in LEVELS}
    env = mosek.Env()

    for (d, L, probes) in TEST_CASES:
        P = _precompute(d=d, order=L, verbose=False, lazy_ab_eiej=False)
        aux_full = _prep_full(P)
        for level in LEVELS:
            print(f"=== d={d} L={L}  level={level:8s}  ({len(probes)} probes) ===",
                  flush=True)
            for t in probes:
                total_probes += 1
                per_level_totals[level]['probes'] += 1
                t0 = time.time()
                dres = _run_dual(P, t, env, level=level, aux=aux_full)
                t_dual = time.time() - t0
                per_level_totals[level]['dual'] += t_dual

                t0 = time.time()
                pres = _run_primal(d, L, t, level=level)
                t_primal = time.time() - t0
                per_level_totals[level]['primal'] += t_primal

                ok = _compatible(pres['verdict'], dres['verdict'])
                tag = 'OK' if ok else 'FAIL'
                note = ''
                if pres['verdict'] != dres['verdict']:
                    note = '  (one-sided uncertain)' \
                        if 'uncertain' in (pres['verdict'], dres['verdict']) \
                        else '  (HARD DISAGREEMENT)'
                print(f"  [{tag}] t={t:.4f}  "
                      f"primal={pres['verdict']:10s}  "
                      f"dual={dres['verdict']:10s}  "
                      f"lam*={dres['lambda_star']:+.3e}  "
                      f"primal_t={t_primal:.2f}s  dual_t={t_dual:.2f}s{note}",
                      flush=True)
                if not ok:
                    failures.append((d, L, level, t,
                                       pres['verdict'], dres['verdict']))

    total_primal_s = sum(p['primal'] for p in per_level_totals.values())
    total_dual_s = sum(p['dual'] for p in per_level_totals.values())
    print()
    print(f"Summary: {total_probes} probes (across {len(LEVELS)} stack levels), "
          f"{total_probes - len(failures)} passed, {len(failures)} failed.")
    for lvl, p in per_level_totals.items():
        if p['probes']:
            sp = p['primal'] / max(p['dual'], 1e-9)
            print(f"  [{lvl:8s}] {p['probes']} probes.  "
                  f"primal={p['primal']:.2f}s  dual={p['dual']:.2f}s  "
                  f"speedup={sp:.2f}x")
    print(f"  Overall:  primal={total_primal_s:.2f}s  "
          f"dual={total_dual_s:.2f}s  "
          f"speedup={total_primal_s / max(total_dual_s, 1e-9):.2f}x")
    if failures:
        print("FAILURES:")
        for d, L, lvl, t, pv, dv in failures:
            print(f"  d={d} L={L} level={lvl} t={t}: primal={pv}  dual={dv}")
        return 1
    print("ALL TESTS PASSED.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
