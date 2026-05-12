"""Smoke test: compare SDP solvers (MOSEK / Clarabel / SCS) on the L-bench
Shor SDP at the per-composition cell level.

Setup:
    Run the L SDP on Q-survivors at (n_half=3, m=10, d=6) with each solver.
    Optional: also run at (n_half=4, m=10, d=8) if time permits.

For each solver report:
    - prunes (= status 'infeasible')
    - median wall time per SDP
    - prune-set agreement vs MOSEK baseline (set difference of pruned indices)

End: rank by median time; flag solvers whose prune sets disagree from MOSEK.
"""
from __future__ import annotations
import os, sys, time, json, traceback
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import (_build_A_matrices, _shor_feasibility, _detect_solver)

import cvxpy as cp


def gather_q_survivors(n_half, m, c_target, batch_size=200_000, max_qs=None):
    """Run F+Q over all palindromic comps; return list of c_int (np.int32 arrays)
    that survive Q (these are the inputs to L)."""
    d = 2 * n_half
    S_half = 2 * n_half * m
    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    survivors = []
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        for idx in f_idx:
            c_int = batch[idx]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                               n_half, m, c_target, margin=1e-9):
                survivors.append(c_int.copy())
                if max_qs is not None and len(survivors) >= max_qs:
                    return survivors, windows
    return survivors, windows


def run_solver_on_survivors(c_list, A_mats, windows, n_half, m, c_target,
                            solver_name, solver_kwargs):
    """Run the Shor SDP on each c in c_list with the named solver. Returns
    (n_pruned, prune_set_indices, total_time, per_solve_times, status_counter)."""
    prune_set = []
    times = []
    statuses = {}
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = 1e-9 * m * m

    for ci, c_int in enumerate(c_list):
        d = len(c_int)
        c = np.asarray(c_int, dtype=np.float64)
        lo = np.maximum(0.0, c - 1.0)
        hi = c + 1.0

        x = cp.Variable(d)
        X = cp.Variable((d, d), symmetric=True)
        ones11 = np.ones((1, 1))
        Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                     [cp.reshape(x, (d, 1), order='C'), X]])
        cons = [Y >> 0, x >= lo, x <= hi, cp.sum(x) == nm]
        for i in range(d):
            cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
            cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
            cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
            cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]
        for i in range(d):
            for j in range(i + 1, d):
                li, lj = lo[i], lo[j]
                ui, uj = hi[i], hi[j]
                cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
                cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
                cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
                cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]
        for A_mat, (ell, _) in zip(A_mats, windows):
            thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
            cons += [cp.trace(A_mat @ X) <= thr]
        prob = cp.Problem(cp.Minimize(0), cons)

        t0 = time.time()
        try:
            prob.solve(solver=solver_name, verbose=False, **solver_kwargs)
            status = prob.status
        except Exception as e:
            status = f'EXC:{type(e).__name__}'
        elapsed = time.time() - t0
        times.append(elapsed)
        statuses[status] = statuses.get(status, 0) + 1
        if status == 'infeasible':
            prune_set.append(ci)
    return len(prune_set), set(prune_set), sum(times), times, statuses


def main():
    t_start = time.time()
    out = {'configs': []}

    # Configs to test
    configs = [
        (3, 10, 1.28),  # d=6, ~172 Q-survivors
    ]
    # Conditionally include d=8 if within budget
    do_d8 = True

    available = set(cp.installed_solvers())
    print(f"Available CVXPY solvers: {sorted(available)}")

    for (n_half, m, c_target) in configs:
        d = 2 * n_half
        print(f"\n=== Config: n_half={n_half}, m={m}, d={d}, c_target={c_target} ===")
        t0 = time.time()
        survivors, windows = gather_q_survivors(n_half, m, c_target)
        t_gather = time.time() - t0
        A_mats = _build_A_matrices(d, windows)
        n_q = len(survivors)
        print(f"  Q-survivors: {n_q} (gathered in {t_gather:.1f}s)")

        cfg_out = {
            'n_half': n_half, 'm': m, 'd': d, 'c_target': c_target,
            'n_q_survivors': n_q,
            'n_windows': len(windows),
            'solvers': {},
        }

        # Solver list and kwargs
        solver_specs = []
        if 'MOSEK' in available:
            solver_specs.append(('MOSEK', 'MOSEK', {
                'mosek_params': {
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-9,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-9,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
                }}))
        if 'CLARABEL' in available:
            solver_specs.append(('CLARABEL', 'CLARABEL',
                                  {'eps_abs': 1e-9, 'eps_rel': 1e-9, 'max_iter': 300}))
        if 'SCS' in available:
            solver_specs.append(('SCS-default', 'SCS',
                                  {'eps': 1e-4, 'max_iters': 20000}))
            solver_specs.append(('SCS-tight', 'SCS',
                                  {'eps': 1e-9, 'max_iters': 50000}))
        if 'CVXOPT' in available:
            solver_specs.append(('CVXOPT', 'CVXOPT', {}))
        if 'SDPA' in available:
            solver_specs.append(('SDPA', 'SDPA', {}))

        # MOSEK first to lock in baseline prune set
        baseline_prune = None
        for label, solver_name, kw in solver_specs:
            print(f"\n  --- {label} ---")
            try:
                n_pr, prune_set, tot, times, statuses = run_solver_on_survivors(
                    survivors, A_mats, windows, n_half, m, c_target,
                    solver_name, kw)
            except Exception as e:
                print(f"    ERROR running {label}: {e}")
                traceback.print_exc()
                cfg_out['solvers'][label] = {'error': str(e)}
                continue
            med = float(np.median(times)) if times else 0.0
            p95 = float(np.percentile(times, 95)) if times else 0.0
            mn  = float(np.min(times)) if times else 0.0
            mx  = float(np.max(times)) if times else 0.0

            if baseline_prune is None:
                baseline_prune = prune_set
                missing = set()
                extra = set()
            else:
                missing = baseline_prune - prune_set  # MOSEK pruned but this didn't
                extra   = prune_set - baseline_prune

            print(f"    pruned: {n_pr}/{n_q}    statuses: {statuses}")
            print(f"    time per SDP: med={med*1000:.1f} ms,"
                  f" min={mn*1000:.1f}, p95={p95*1000:.1f}, max={mx*1000:.1f}")
            print(f"    total: {tot:.2f}s")
            print(f"    vs MOSEK baseline: missing={len(missing)} extra={len(extra)}")

            cfg_out['solvers'][label] = {
                'cvxpy_solver': solver_name,
                'kwargs': {k: (v if isinstance(v, (int, float, str)) else str(v))
                           for k, v in kw.items()},
                'n_pruned': n_pr,
                'total_s': tot,
                'med_ms': med * 1000,
                'min_ms': mn * 1000,
                'p95_ms': p95 * 1000,
                'max_ms': mx * 1000,
                'statuses': statuses,
                'missing_vs_mosek': sorted(list(missing)),
                'extra_vs_mosek': sorted(list(extra)),
                'agrees_with_mosek': len(missing) == 0 and len(extra) == 0,
            }

        out['configs'].append(cfg_out)

        # Bail on d=8 if too slow.
        if (n_half, m, c_target) == (3, 10, 1.28):
            elapsed = time.time() - t_start
            # Allowed: stay within 15min total
            if elapsed > 540:
                print(f"\n  Already used {elapsed:.0f}s; skipping d=8.")
                do_d8 = False

    # Optional: d=8 with smaller cap
    if do_d8 and (time.time() - t_start) < 540:
        n_half, m, c_target = 4, 10, 1.28
        d = 2 * n_half
        print(f"\n=== Config: n_half={n_half}, m={m}, d={d}, c_target={c_target} (capped) ===")
        # Cap to first ~60 Q-survivors for time budget
        cap = 60
        t0 = time.time()
        survivors, windows = gather_q_survivors(n_half, m, c_target, max_qs=cap)
        A_mats = _build_A_matrices(d, windows)
        n_q = len(survivors)
        print(f"  Q-survivors (first {n_q}, capped at {cap}): gathered in"
              f" {time.time()-t0:.1f}s")

        cfg_out = {
            'n_half': n_half, 'm': m, 'd': d, 'c_target': c_target,
            'n_q_survivors_used': n_q, 'cap': cap,
            'n_windows': len(windows),
            'solvers': {},
        }

        solver_specs = []
        if 'MOSEK' in available:
            solver_specs.append(('MOSEK', 'MOSEK', {
                'mosek_params': {
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-9,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-9,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
                }}))
        if 'CLARABEL' in available:
            solver_specs.append(('CLARABEL', 'CLARABEL',
                                  {'eps_abs': 1e-9, 'eps_rel': 1e-9, 'max_iter': 300}))
        if 'SCS' in available:
            solver_specs.append(('SCS-default', 'SCS',
                                  {'eps': 1e-4, 'max_iters': 20000}))
            solver_specs.append(('SCS-tight', 'SCS',
                                  {'eps': 1e-9, 'max_iters': 50000}))

        baseline_prune = None
        for label, solver_name, kw in solver_specs:
            print(f"\n  --- {label} ---")
            try:
                n_pr, prune_set, tot, times, statuses = run_solver_on_survivors(
                    survivors, A_mats, windows, n_half, m, c_target,
                    solver_name, kw)
            except Exception as e:
                print(f"    ERROR running {label}: {e}")
                traceback.print_exc()
                cfg_out['solvers'][label] = {'error': str(e)}
                continue
            med = float(np.median(times)) if times else 0.0
            p95 = float(np.percentile(times, 95)) if times else 0.0
            mn  = float(np.min(times)) if times else 0.0
            mx  = float(np.max(times)) if times else 0.0

            if baseline_prune is None:
                baseline_prune = prune_set
                missing = set()
                extra = set()
            else:
                missing = baseline_prune - prune_set
                extra   = prune_set - baseline_prune

            print(f"    pruned: {n_pr}/{n_q}    statuses: {statuses}")
            print(f"    time per SDP: med={med*1000:.1f} ms,"
                  f" min={mn*1000:.1f}, p95={p95*1000:.1f}, max={mx*1000:.1f}")
            print(f"    vs MOSEK baseline: missing={len(missing)} extra={len(extra)}")

            cfg_out['solvers'][label] = {
                'cvxpy_solver': solver_name,
                'kwargs': {k: (v if isinstance(v, (int, float, str)) else str(v))
                           for k, v in kw.items()},
                'n_pruned': n_pr,
                'total_s': tot,
                'med_ms': med * 1000,
                'min_ms': mn * 1000,
                'p95_ms': p95 * 1000,
                'max_ms': mx * 1000,
                'statuses': statuses,
                'missing_vs_mosek': sorted(list(missing)),
                'extra_vs_mosek': sorted(list(extra)),
                'agrees_with_mosek': len(missing) == 0 and len(extra) == 0,
            }

        out['configs'].append(cfg_out)

    out['total_wall_s'] = time.time() - t_start

    # Final ranking per config
    print("\n\n=== FINAL RANKING ===")
    rankings = {}
    for cfg in out['configs']:
        key = f"d={cfg['d']}"
        rows = []
        for label, info in cfg['solvers'].items():
            if 'error' in info or 'med_ms' not in info:
                continue
            rows.append((label, info['med_ms'], info['n_pruned'],
                         info['agrees_with_mosek']))
        rows.sort(key=lambda r: r[1])
        print(f"\n  Config {key}, n_q={cfg.get('n_q_survivors', cfg.get('n_q_survivors_used'))}:")
        for label, med, n_pr, ok in rows:
            tag = 'OK' if ok else 'DISAGREE'
            print(f"    {label:18s}  med={med:8.1f} ms  pruned={n_pr:4d}  {tag}")
        rankings[key] = [{'label': r[0], 'med_ms': r[1], 'n_pruned': r[2],
                          'agrees': r[3]} for r in rows]
    out['rankings'] = rankings

    out_path = os.path.join(_dir, '_smoke_solver_compare.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(f"Total wall: {out['total_wall_s']:.1f}s")


if __name__ == '__main__':
    main()
