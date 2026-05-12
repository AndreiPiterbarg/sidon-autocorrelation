#!/usr/bin/env python
"""Correctness audit for lasserre/dual_sdp.py.

Runs a battery of isolated checks that stress the signs, the MOSEK
symmetry-factor convention, the Z/2 block-diagonalisation, the scipy
aggregation helper, and the reuse path's idempotency + state-transition
stability.  Used as a pre-publication sanity check.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import mosek

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from lasserre_scalable import _precompute
from lasserre.dual_sdp import (
    build_dual_task, solve_dual_task, update_task_t,
    _aggregate_bar_triplet,
)
from lasserre.z2_elim import canonicalize_z2
from lasserre.z2_blockdiag import (
    build_blockdiag_picks,
    localizing_sigma_reps,
    window_sigma_reps,
    orbit_decomposition,
)
from lasserre_mosek_dual import _apply_task_params


ENV = mosek.Env()


def _dual_probe(P, t, *, uloc=False, bd=None, al=None, aw=None):
    with contextlib.redirect_stdout(io.StringIO()):
        task, info = build_dual_task(
            P, t_val=t, env=ENV,
            include_upper_loc=uloc, z2_blockdiag_map=bd,
            active_loc=al, active_windows=aw, verbose=False)
        res = solve_dual_task(task, info, verbose=False)
        try:
            task.__del__()
        except Exception:
            pass
    return res


def _prep_full(P):
    Pc = canonicalize_z2(P, verbose=False)
    bd = build_blockdiag_picks(Pc['basis'], Pc['idx'], Pc['n_y'])
    lf, lp = localizing_sigma_reps(P['d'])
    al = list(lf) + [p for p, _ in lp]
    wf, wp = window_sigma_reps(P['d'], P['windows'])
    nt = set(P['nontrivial_windows'])
    aw = [w for w in (list(wf) + [p for p, _ in wp]) if w in nt]
    return Pc, bd, al, aw


# -------------------- A1: monotonicity of verdict in t --------------------
def audit_monotonicity():
    print("A1: verdict monotone in t at d=4 L=3 (full stack)")
    P = _precompute(4, 3, verbose=False, lazy_ab_eiej=False)
    Pc, bd, al, aw = _prep_full(P)
    ts = np.linspace(0.3, 1.5, 13)
    verdicts = []
    for t in ts:
        r = _dual_probe(Pc, t, uloc=True, bd=bd, al=al, aw=aw)
        verdicts.append((float(t), r['verdict'], r['lambda_star']))
    for (t, v, lam) in verdicts:
        print(f"   t={t:.4f} -> {v:10s} lam*={lam:+.3e}")
    last_infeas = max((t for t, v, _ in verdicts if v == 'infeas'),
                      default=None)
    first_feas = min((t for t, v, _ in verdicts if v == 'feas'),
                     default=None)
    print(f"   last infeas={last_infeas}  first feas={first_feas}")
    assert last_infeas is None or first_feas is None \
        or last_infeas < first_feas, "verdict NOT monotone in t"
    print("   PASS\n")


# -------------------- A2: block-diag T_sym matches M_sym explicitly --------
def audit_blockdiag_identity():
    print("A2: T_sym @ y == M_sym(y) on random sigma-invariant y (d=4 L=2)")
    P = _precompute(4, 2, verbose=False, lazy_ab_eiej=False)
    Pc = canonicalize_z2(P, verbose=False)
    bd = build_blockdiag_picks(Pc['basis'], Pc['idx'], Pc['n_y'])
    rng = np.random.default_rng(42)
    y_canon = rng.standard_normal(Pc['n_y'])
    fixed, pairs = orbit_decomposition(Pc['basis'])
    n_sym = bd['n_sym']
    sym_cols = [[(f, 1.0)] for f in fixed]
    for (p, q) in pairs:
        sym_cols.append([(p, 1.0/np.sqrt(2.0)), (q, 1.0/np.sqrt(2.0))])
    basis_tup = [tuple(b) for b in Pc['basis']]
    M_sym_ref = np.zeros((n_sym, n_sym))
    for u in range(n_sym):
        for v in range(n_sym):
            acc = 0.0
            for (k1, c1) in sym_cols[u]:
                for (k2, c2) in sym_cols[v]:
                    s = tuple(x + y for x, y in zip(basis_tup[k1],
                                                       basis_tup[k2]))
                    j = Pc['idx'][s]
                    acc += c1 * c2 * y_canon[j]
            M_sym_ref[u, v] = acc
    M_sym_from_T = (bd['T_sym'] @ y_canon).reshape(n_sym, n_sym)
    err = float(np.max(np.abs(M_sym_from_T - M_sym_ref)))
    print(f"   max |T_sym.y - M_sym_ref| = {err:.2e}")
    assert err < 1e-10, f"T_sym mismatch: {err}"
    print("   PASS\n")


# -------------------- A3: aggregation duplicate summing --------------------
def audit_aggregation():
    print("A3: _aggregate_bar_triplet duplicate summing")
    subi = np.array([0, 1, 0, 2, 1], dtype=np.int32)
    subj = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    subk = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    subl = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    val = np.array([1.0, 2.0, 3.0, -5.0, 7.0], dtype=np.float64)
    tc = np.array([0.0, 0.0, 0.5, 0.0, -0.5], dtype=np.float64)
    out_subi, out_subj, out_subk, out_subl, out_val, out_tc = \
        _aggregate_bar_triplet(subi, subj, subk, subl, val, tc)
    # Expected unique keys (0,0,0,0), (1,0,0,0), (2,0,0,0) with summed
    # (val, tc) = (4, 0.5), (9, -0.5), (-5, 0).
    print(f"   result: subi={list(out_subi)} val={list(out_val)} "
          f"tcoef={list(out_tc)}")
    # Order may vary by sort stability; check multisets.
    assert set(zip(out_subi.tolist(), out_val.tolist(), out_tc.tolist())) == {
        (0, 4.0, 0.5),
        (1, 9.0, -0.5),
        (2, -5.0, 0.0),
    }, "aggregation wrong"
    print("   PASS\n")


# -------------------- A4: update_task_t idempotency ------------------------
def audit_idempotency():
    print("A4: update_task_t(t) == build(t) then solve, same verdict & lam*")
    P = _precompute(4, 2, verbose=False, lazy_ab_eiej=False)
    Pc, bd, al, aw = _prep_full(P)
    with contextlib.redirect_stdout(io.StringIO()):
        task, info = build_dual_task(
            Pc, t_val=1.5, env=ENV,
            include_upper_loc=True, z2_blockdiag_map=bd,
            active_loc=al, active_windows=aw, verbose=False)
        _apply_task_params(task, verbose=False)
        r1 = solve_dual_task(task, info, verbose=False)
        update_task_t(task, info, 1.5)  # same t
        r2 = solve_dual_task(task, info, verbose=False)
        try:
            task.__del__()
        except Exception:
            pass
    print(f"   before update: verdict={r1['verdict']} "
          f"lam*={r1['lambda_star']:+.3e}")
    print(f"   after  update: verdict={r2['verdict']} "
          f"lam*={r2['lambda_star']:+.3e}")
    assert r1['verdict'] == r2['verdict'], "idempotent update broke verdict"
    assert abs(r1['lambda_star'] - r2['lambda_star']) < 1e-6, \
        "idempotent update broke lam*"
    print("   PASS\n")


# -------------------- A5: state transitions ------------------------------
def audit_state_transitions():
    print("A5: state transitions across feas/infeas boundary (d=4 L=3 full)")
    P = _precompute(4, 3, verbose=False, lazy_ab_eiej=False)
    Pc, bd, al, aw = _prep_full(P)
    with contextlib.redirect_stdout(io.StringIO()):
        task, info = build_dual_task(
            Pc, t_val=1.5, env=ENV,
            include_upper_loc=True, z2_blockdiag_map=bd,
            active_loc=al, active_windows=aw, verbose=False)
        _apply_task_params(task, verbose=False)
    print(f"   task built.  n_bar_entries={info['n_bar_entries']:,}")
    seq = [1.5, 0.3, 1.5, 1.08, 1.15, 0.5, 2.0, 1.12, 0.9, 2.0]
    reuse_results = []
    for t in seq:
        update_task_t(task, info, t)
        with contextlib.redirect_stdout(io.StringIO()):
            r = solve_dual_task(task, info, verbose=False)
        reuse_results.append((t, r['verdict'], r['lambda_star']))
    try:
        task.__del__()
    except Exception:
        pass
    # Compare against cold builds for each t in seq.
    cold_results = []
    for t in seq:
        r_cold = _dual_probe(Pc, t, uloc=True, bd=bd, al=al, aw=aw)
        cold_results.append((t, r_cold['verdict'], r_cold['lambda_star']))
    mismatches = 0
    for (t, rv, rl), (_, cv, cl) in zip(reuse_results, cold_results):
        tag = 'OK' if rv == cv else 'MISMATCH'
        if tag == 'MISMATCH':
            mismatches += 1
        print(f"   t={t:.4f}: reuse={rv:8s} (lam*={rl:+.3e})  "
              f"cold={cv:8s} (lam*={cl:+.3e})  [{tag}]")
    assert mismatches == 0, f"reuse vs cold: {mismatches} mismatches"
    print("   PASS\n")


# -------------------- A6: edge case — order=1 (no loc cones needed) -------
def audit_order_1():
    print("A6: order=1 has empty loc_basis; builder must not crash")
    P = _precompute(4, 1, verbose=False, lazy_ab_eiej=False)
    # At order=1, n_loc = 1 (just the zero monomial).  Builder should
    # still work.
    try:
        r = _dual_probe(P, 2.0)
        print(f"   at t=2.0: verdict={r['verdict']} "
              f"lam*={r['lambda_star']:+.3e}")
        print("   PASS (builder accepts order=1)")
    except Exception as exc:
        print(f"   FAIL: {exc}")
        raise


# -------------------- A7: edge case — d=2 smallest non-trivial ------------
def audit_d_2():
    print("A7: d=2 order=2 smallest non-trivial problem")
    P = _precompute(2, 2, verbose=False, lazy_ab_eiej=False)
    Pc, bd, al, aw = _prep_full(P)
    for t in [0.3, 0.7, 1.0, 1.2, 2.0]:
        r = _dual_probe(Pc, t, uloc=True, bd=bd, al=al, aw=aw)
        print(f"   d=2 L=2 t={t:.4f}: {r['verdict']:10s} "
              f"lam*={r['lambda_star']:+.3e}")
    print("   PASS (no crash)\n")


# -------------------- A8: primal-dual cross-check at many d/L -------------
def audit_cross_check():
    print("A8: dual verdict matches moment-primal (full stack) at many (d,L)")
    from lasserre_mosek_preelim import solve_mosek_preelim

    cases = [(4, 2), (4, 3), (6, 2), (6, 3)]
    t_probes = [0.5, 1.0, 1.05, 1.1, 1.15, 1.2, 1.5, 2.0]
    for (d, L) in cases:
        P = _precompute(d, L, verbose=False, lazy_ab_eiej=False)
        Pc, bd, al, aw = _prep_full(P)
        mismatches = 0
        for t in t_probes:
            # full-stack dual
            dr = _dual_probe(Pc, t, uloc=True, bd=bd, al=al, aw=aw)
            # full-stack primal
            with contextlib.redirect_stdout(io.StringIO()):
                pr = solve_mosek_preelim(
                    d=d, order=L, mode='z2_full', pre_elim_z2=True,
                    add_upper_loc=True, single_t=t, verbose=False)
            pv, dv = pr['verdict'], dr['verdict']
            ok = (pv == dv) or 'uncertain' in (pv, dv)
            if not ok:
                mismatches += 1
                print(f"   FAIL d={d} L={L} t={t}: primal={pv} dual={dv}")
        print(f"   d={d} L={L}: {len(t_probes)} probes, "
              f"{mismatches} mismatches")
        assert mismatches == 0, f"cross-check failed at d={d} L={L}"
    print("   PASS\n")


def main() -> int:
    print("=" * 72)
    print("  SOS-DUAL CORRECTNESS AUDIT")
    print("=" * 72, flush=True)
    audit_monotonicity()
    audit_blockdiag_identity()
    audit_aggregation()
    audit_idempotency()
    audit_state_transitions()
    audit_order_1()
    audit_d_2()
    audit_cross_check()
    print("=" * 72)
    print("  ALL AUDITS PASSED.")
    print("=" * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
