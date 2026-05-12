#!/usr/bin/env python
r"""Run Lasserre highd config using CVXPY+SCS (fits in 256 GB).

Solves the IDENTICAL SDP as the MOSEK solver but uses SCS backend.
SCS is first-order ADMM — O(nnz) memory instead of O(n_y^2 * PSD_dim).

Usage:
    python tests/run_single_cvxpy.py --d 14 --order 3 --bw 13
    python tests/run_single_cvxpy.py --d 16 --order 3 --bw 12
"""
import sys
import os
import time
import json
import argparse
import numpy as np
from scipy import sparse as sp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, enum_monomials, val_d_known,
)

import cvxpy as cp


def build_and_solve(P, t_fixed=None, active_windows=None,
                    add_upper_loc=True, verbose=True,
                    max_iters=50000, eps=1e-5):
    """Build and solve the Lasserre SDP with CVXPY+SCS.

    If t_fixed is None: minimize t (scalar optimization).
    If t_fixed is a float: check feasibility at that t with window PSD constraints.
    """
    d = P['d']
    order = P['order']
    n_y = P['n_y']
    idx = P['idx']
    if active_windows is None:
        active_windows = set()

    y = cp.Variable(n_y, nonneg=True)
    constraints = []

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    constraints.append(y[idx[zero]] == 1)

    # Consistency from pre-built COO
    if P['consist_eq_lists'] is not None:
        n_eq, er, ec, ev = P['consist_eq_lists']
        A_eq = sp.csr_matrix((ev, (er, ec)), shape=(n_eq, n_y))
        constraints.append(A_eq @ y == 0)

    if P['consist_iq_lists'] is not None:
        n_iq, ir, ic, iv = P['consist_iq_lists']
        A_iq = sp.csr_matrix((iv, (ir, ic)), shape=(n_iq, n_y))
        constraints.append(A_iq @ y >= 0)

    # Full M_{k-1} PSD
    if P['m1_valid']:
        pick = P['m1_pick']
        n = P['m1_size']
        constraints.append(cp.reshape(y[pick], (n, n)) >> 0)

    # Clique moment PSD
    for cd in P['clique_data']:
        pick = cd['mom_pick']
        if np.any(pick < 0):
            continue
        n_cb = cd['mom_size']
        constraints.append(cp.reshape(y[pick], (n_cb, n_cb)) >> 0)

    # Clique localizing PSD
    if order >= 2:
        for i_var in range(d):
            c_idx = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx]
            picks = cd['loc_picks'].get(i_var)
            if picks is None or np.any(picks < 0):
                continue
            n_cb = cd['loc_size']
            constraints.append(cp.reshape(y[picks], (n_cb, n_cb)) >> 0)

    # Upper localizing PSD
    if add_upper_loc and order >= 2:
        for i_var in range(d):
            c_idx = P['bin_to_clique_map'].get(i_var, 0)
            cd = P['clique_data'][c_idx]
            t_pick = cd.get('t_pick')
            loc_pick = cd['loc_picks'].get(i_var)
            if t_pick is None or loc_pick is None:
                continue
            if np.any(t_pick < 0) or np.any(loc_pick < 0):
                continue
            n_cb = cd['loc_size']
            constraints.append(
                cp.reshape(y[t_pick] - y[loc_pick], (n_cb, n_cb)) >> 0)

    if t_fixed is None:
        # Minimize t with scalar window constraints
        t_var = cp.Variable()
        F = P['F_scipy']
        constraints.append(t_var >= F @ y)
        prob = cp.Problem(cp.Minimize(t_var), constraints)
    else:
        # Feasibility check at fixed t with window PSD constraints
        # Scalar windows
        F = P['F_scipy']
        constraints.append(t_fixed >= F @ y)

        # Window PSD: L_W = t*M_{k-1}(y) - Q_W(y) >> 0 (linear in y when t fixed)
        for w in active_windows:
            c_idx_w = int(P['window_covering'][w])
            if c_idx_w < 0:
                continue
            cd = P['clique_data'][c_idx_w]
            n_cb = cd['loc_size']
            t_pick_w = cd['t_pick']
            if t_pick_w is None or np.any(t_pick_w < 0):
                continue

            # T-part: t_fixed * y[t_pick]
            T_flat = t_fixed * y[t_pick_w]

            # Q-part: sum M_W[i,j] * y[ab_eiej[a,b,i,j]]
            ab_eiej = cd['viol_ab_eiej']
            ell, s_lo = P['windows'][w]
            coeff = 2.0 * d / ell
            sums_local = cd['viol_gi_grid']
            mask = (sums_local >= s_lo) & (sums_local <= s_lo + ell - 2)
            nz_li, nz_lj = np.nonzero(mask)

            if len(nz_li) == 0:
                constraints.append(cp.reshape(T_flat, (n_cb, n_cb)) >> 0)
                continue

            # Build Q as sparse: Q_flat[a*n_cb+b] = sum_k coeff * y[ab_eiej[a,b,li_k,lj_k]]
            # This is a linear function of y
            q_rows = []
            q_cols = []
            q_vals = []
            for k_idx in range(len(nz_li)):
                li, lj = int(nz_li[k_idx]), int(nz_lj[k_idx])
                idx_slice = ab_eiej[:, :, li, lj]  # (n_cb, n_cb)
                for a in range(n_cb):
                    for b in range(n_cb):
                        yi = int(idx_slice[a, b])
                        if yi >= 0:
                            q_rows.append(a * n_cb + b)
                            q_cols.append(yi)
                            q_vals.append(coeff)

            if q_vals:
                Q_sp = sp.csr_matrix((q_vals, (q_rows, q_cols)),
                                     shape=(n_cb * n_cb, n_y))
                Q_flat = Q_sp @ y
                L_flat = T_flat - Q_flat
            else:
                L_flat = T_flat

            constraints.append(cp.reshape(L_flat, (n_cb, n_cb)) >> 0)

        prob = cp.Problem(cp.Minimize(0), constraints)

    t0 = time.time()
    try:
        prob.solve(solver=cp.SCS, verbose=verbose, max_iters=max_iters,
                   eps_abs=eps, eps_rel=eps)
    except Exception as e:
        print(f"  SCS error: {e}", flush=True)
        return None, None, None

    elapsed = time.time() - t0
    status = prob.status

    if t_fixed is None:
        t_val = float(t_var.value) if t_var.value is not None else None
    else:
        t_val = t_fixed

    y_vals = np.array(y.value).flatten() if y.value is not None else None

    return t_val, y_vals, {'status': status, 'elapsed': elapsed}


def solve_full(P, add_upper_loc=True, max_cg_rounds=10, n_bisect=8, verbose=True):
    """Full CG solve using CVXPY+SCS."""
    d = P['d']
    n_y = P['n_y']
    t_total = time.time()

    # Round 0: minimize t (scalar only)
    print("  [Round 0] SCS minimize t...", flush=True)
    t_val, y_vals, info = build_and_solve(P, t_fixed=None,
                                          add_upper_loc=add_upper_loc,
                                          verbose=verbose)
    if t_val is None:
        print(f"  Round 0 failed: {info}")
        return {'lb': 0.0, 'd': d, 'order': P['order'], 'n_y': n_y,
                'n_active_windows': 0, 'elapsed': time.time() - t_total}

    scalar_lb = t_val
    best_lb = scalar_lb
    print(f"    Scalar bound = {scalar_lb:.10f} ({info['elapsed']:.1f}s, "
          f"status={info['status']})", flush=True)

    active_windows = set()
    violations = _check_violations_highd(y_vals, scalar_lb, P, active_windows)
    print(f"    {len(violations)} violations", flush=True)

    if not violations:
        return {'lb': best_lb, 'd': d, 'order': P['order'], 'n_y': n_y,
                'n_active_windows': 0, 'elapsed': time.time() - t_total}

    # CG rounds with bisection
    for cg_round in range(1, max_cg_rounds + 1):
        n_add = min(100, len(violations))
        for w, eig in violations[:n_add]:
            active_windows.add(w)
        print(f"\n  [CG round {cg_round}] {len(active_windows)} windows",
              flush=True)

        lo = max(0.5, best_lb - 1e-3)
        hi = best_lb + 0.02 if best_lb > 0.5 else 5.0

        # Check hi is feasible
        _, _, info_hi = build_and_solve(P, t_fixed=hi,
                                       active_windows=active_windows,
                                       verbose=False)
        while info_hi is None or info_hi['status'] not in ['optimal', 'optimal_inaccurate']:
            hi *= 1.5
            if hi > 100:
                break
            _, _, info_hi = build_and_solve(P, t_fixed=hi,
                                           active_windows=active_windows,
                                           verbose=False)

        # Bisection
        for step in range(n_bisect):
            mid = (lo + hi) / 2
            _, _, info_mid = build_and_solve(P, t_fixed=mid,
                                            active_windows=active_windows,
                                            verbose=False)
            if info_mid and info_mid['status'] in ['optimal', 'optimal_inaccurate']:
                hi = mid
                print(f"    [{step+1}/{n_bisect}] t={mid:.6f} feasible", flush=True)
            else:
                lo = mid
                print(f"    [{step+1}/{n_bisect}] t={mid:.6f} infeasible", flush=True)

        lb = lo
        improvement = lb - best_lb
        best_lb = max(best_lb, lb)

        v = val_d_known.get(d, 0)
        gc = (best_lb - 1) / (v - 1) * 100 if v > 1 else 0
        print(f"    lb={lb:.10f} (+{improvement:.2e}) gc={gc:.1f}%", flush=True)

        # Get y* at feasibility boundary for violation checking
        _, y_vals, _ = build_and_solve(P, t_fixed=hi,
                                      active_windows=active_windows,
                                      verbose=False)
        if y_vals is None:
            print(f"    Could not extract y*")
            break

        violations = _check_violations_highd(y_vals, hi, P, active_windows)
        if not violations:
            print(f"    No violations — converged.", flush=True)
            break

        if improvement < 1e-6 and cg_round >= 3:
            print(f"    Improvement < 1e-6 — stopping.", flush=True)
            break

    elapsed = time.time() - t_total
    return {'lb': best_lb, 'd': d, 'order': P['order'], 'n_y': n_y,
            'n_active_windows': len(active_windows), 'elapsed': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--bw', type=int, required=True)
    parser.add_argument('--cg-rounds', type=int, default=10)
    parser.add_argument('--bisect', type=int, default=8)
    args = parser.parse_args()

    print(f"CVXPY+SCS solver: d={args.d} O{args.order} bw={args.bw}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"CVXPY: {cp.__version__}")

    # Memory + cgroup monitoring
    import threading
    def mem_monitor():
        import subprocess as _sp
        while True:
            try:
                with open('/sys/fs/cgroup/memory.current') as f:
                    used = int(f.read().strip()) / 1e9
                with open('/sys/fs/cgroup/memory.max') as f:
                    limit = int(f.read().strip()) / 1e9
                print(f"  [MEM] {used:.1f}/{limit:.0f} GB", flush=True)
            except Exception:
                try:
                    r = _sp.run(['free', '-g'], capture_output=True, text=True, timeout=5)
                    for line in r.stdout.strip().split('\n'):
                        if 'Mem' in line:
                            print(f"  [MEM] {line.strip()}", flush=True)
                except Exception:
                    pass
            time.sleep(30)
    threading.Thread(target=mem_monitor, daemon=True).start()
    print(flush=True)

    cliques = _build_banded_cliques(args.d, args.bw)
    P = _precompute_highd(args.d, args.order, cliques, verbose=True)

    r = solve_full(P, max_cg_rounds=args.cg_rounds, n_bisect=args.bisect,
                   verbose=False)

    vd = val_d_known.get(args.d, 0)
    gc = (r['lb'] - 1) / (vd - 1) * 100 if vd > 1 else 0
    sound = r['lb'] <= vd + 1e-6 if vd > 0 else True

    print(f"\n{'='*70}")
    print(f"FINAL: d={args.d} O{args.order} bw={args.bw}")
    print(f"  lb = {r['lb']:.10f}")
    print(f"  val({args.d}) = {vd}")
    print(f"  gap_closure = {gc:.2f}%")
    print(f"  time = {r['elapsed']:.1f}s = {r['elapsed']/3600:.2f}hr")
    print(f"  sound = {sound}")
    print(f"{'='*70}")

    out = {
        'd': args.d, 'order': args.order, 'bw': args.bw,
        'lb': r['lb'], 'gap_closure': gc, 'n_y': r['n_y'],
        'elapsed': r['elapsed'], 'sound': sound,
        'solver': 'cvxpy_scs',
        'timestamp': datetime.now().isoformat(),
    }
    tag = f"d{args.d}_o{args.order}_bw{args.bw}_scs"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data', f'result_{tag}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
