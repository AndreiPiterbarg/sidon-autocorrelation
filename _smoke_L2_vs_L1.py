"""Smoke test: order-2 Lasserre vs order-1 Shor at d=8 (n_half=4, m=10, c=1.28).

Goal: Determine empirically whether order-2 Lasserre (`_L_bench._lasserre2_feasibility`)
prunes strictly more cells than order-1 Shor (`_L_bench._shor_feasibility`) at d=8.

Pipeline:
  - F = `_M1_bench.prune_F` over all palindromic compositions (n_half=4, m=10, full sum 80)
  - Q = `_Q_bench.prune_Q_one` on F-survivors
  - L1 = `_L_bench.prune_L_one(order=1, solver=MOSEK)` on Q-survivors
  - L2 = `_L_bench.prune_L_one(order=2, solver=MOSEK)` on the same Q-survivors

Soundness invariant:  L2 is a tightening of L1 (M_2 >> 0 implies M_1 = principal
minor >> 0; window/box/RLT constraints lifted), so every L1-prune must also be
L2-prune.  A cell that's L1-pruned but NOT L2-pruned would be a soundness bug.

Output:
  - Counts at each stage (F, Q, L1, L2).
  - Cells L2-extra over L1.
  - Cells L1-but-not-L2 (must be 0; otherwise SOUNDNESS BUG).
  - Avg / median / max wall time per SDP at each order.
  - Total wall time.
  - Verdict line.

Capped at MAX_L_CELLS Q-survivors so the smoke completes in a few minutes on Windows.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, _enum_balanced_signs, prune_Q_one
from _L_bench import _build_A_matrices, prune_L_one, _detect_solver

# --------------------------------------------------------------
# Config
# --------------------------------------------------------------
N_HALF = 4
M = 10
C_TARGET = 1.28
D = 2 * N_HALF              # 8
S_HALF = 2 * N_HALF * M     # 40
S_FULL = 4 * N_HALF * M     # 80
BATCH = 200_000
MAX_L_CELLS = 60            # cap Q-survivors that go to L (per task spec)
EPS_MARGIN = 1e-9
TOL = 1e-9
OUT_PATH = os.path.join(_HERE, '_smoke_L2_vs_L1.json')


def main():
    print('=' * 72)
    print(f'Smoke L2 vs L1 at d={D} (n_half={N_HALF}, m={M}, c={C_TARGET})')
    print('=' * 72)

    # 1) Pick solver (MOSEK preferred; fall back to Clarabel).
    solver = _detect_solver('MOSEK')
    if solver == 'NONE':
        print('ERROR: no SDP solver available.')
        sys.exit(1)
    if solver != 'MOSEK':
        print(f'WARN: MOSEK unavailable; falling back to {solver}')
    print(f'Solver: {solver}')

    # 2) Build window/A scaffolding once.
    windows, ell_int_sums = _build_windows(D)
    sigmas = _enum_balanced_signs(D)
    A_mats = _build_A_matrices(D, windows)
    n_win = len(windows)
    n_sig = len(sigmas)
    print(f'd={D}: n_win={n_win}, n_sigma_balanced={n_sig}')

    # 3) Stream palindromic compositions, run F, then Q on survivors.
    #    (The L bench treats palindromic compositions: full = [half | half[::-1]]).
    #    Warm up F kernel.
    warm = np.zeros((1, D), dtype=np.int32)
    warm[0, 0] = 2 * M
    prune_F(warm, N_HALF, M, C_TARGET)

    t_F = t_Q = 0.0
    n_processed = 0
    n_F_surv = 0
    q_survivors = []   # list of np.ndarray of length D

    t0 = time.time()
    for half_batch in generate_compositions_batched(N_HALF, S_HALF, batch_size=BATCH):
        # Reflect to palindromic full
        full = np.empty((len(half_batch), D), dtype=np.int32)
        full[:, :N_HALF] = half_batch
        full[:, N_HALF:] = half_batch[:, ::-1]
        n_processed += len(full)

        tf = time.time()
        sF = prune_F(full, N_HALF, M, C_TARGET)
        t_F += time.time() - tf
        f_idx = np.where(sF)[0]
        n_F_surv += len(f_idx)

        tq = time.time()
        for idx in f_idx:
            c_int = full[idx]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                N_HALF, M, C_TARGET, margin=EPS_MARGIN):
                # Q-survivor
                q_survivors.append(c_int.copy())
        t_Q += time.time() - tq
    t_FQ = time.time() - t0

    n_Q_surv = len(q_survivors)
    print(f'F survivors:  {n_F_surv}    (expected ~1014 from prior runs)')
    print(f'Q survivors:  {n_Q_surv}    (expected ~964 from prior runs)')
    print(f'F+Q wall:     {t_FQ:.2f}s   (F {t_F:.2f}s, Q {t_Q:.2f}s)')

    # 4) Cap and run L1 / L2 on the same set.
    cells = q_survivors[:MAX_L_CELLS]
    n_cells = len(cells)
    print(f'\nRunning SDPs on first {n_cells} Q-survivors '
          f'(cap MAX_L_CELLS={MAX_L_CELLS})')

    l1_pruned = []   # bool list
    l1_status = []
    l1_times = []
    print('\n--- L1 (order-1 Shor SDP) ---')
    t0 = time.time()
    for k, c_int in enumerate(cells):
        ts = time.time()
        pr, st = prune_L_one(c_int, A_mats, windows, N_HALF, M, C_TARGET,
                              solver=solver, order=1, tol=TOL,
                              eps_margin=EPS_MARGIN)
        dt = time.time() - ts
        l1_pruned.append(bool(pr))
        l1_status.append(st)
        l1_times.append(dt)
        if (k + 1) % 10 == 0 or k == n_cells - 1:
            print(f'  L1 {k+1}/{n_cells}: pruned={sum(l1_pruned)}  '
                  f'last={dt*1000:.0f}ms ({st})')
    t_L1 = time.time() - t0

    l2_pruned = []
    l2_status = []
    l2_times = []
    print('\n--- L2 (order-2 Lasserre SDP) ---')
    t0 = time.time()
    for k, c_int in enumerate(cells):
        ts = time.time()
        pr, st = prune_L_one(c_int, A_mats, windows, N_HALF, M, C_TARGET,
                              solver=solver, order=2, tol=TOL,
                              eps_margin=EPS_MARGIN)
        dt = time.time() - ts
        l2_pruned.append(bool(pr))
        l2_status.append(st)
        l2_times.append(dt)
        if (k + 1) % 5 == 0 or k == n_cells - 1:
            print(f'  L2 {k+1}/{n_cells}: pruned={sum(l2_pruned)}  '
                  f'last={dt*1000:.0f}ms ({st})')
    t_L2 = time.time() - t0

    # 4b) Retry L2 with looser tol on cells where L2 had solver exception.
    retry_results = {}
    for k, st in enumerate(l2_status):
        if isinstance(st, str) and st.startswith('EXC:'):
            print(f'\n--- L2 retry on cell {k} (was {st}) at tol=1e-7 ---')
            ts = time.time()
            pr, st2 = prune_L_one(cells[k], A_mats, windows, N_HALF, M, C_TARGET,
                                    solver=solver, order=2, tol=1e-7,
                                    eps_margin=EPS_MARGIN)
            dt = time.time() - ts
            print(f'  retry: pruned={pr} status={st2} took {dt*1000:.0f}ms')
            retry_results[k] = {'pruned': bool(pr), 'status': st2,
                                  'time_ms': dt * 1000}
            if pr:
                l2_pruned[k] = True
                l2_status[k] = st2

    # 5) Compare.
    n_L1 = sum(l1_pruned)
    n_L2 = sum(l2_pruned)
    extra_L2 = sum(1 for a, b in zip(l1_pruned, l2_pruned) if (not a) and b)
    # Distinguish: a real soundness bug requires L2 to have RETURNED a status
    # claiming the cell is L2-feasible (e.g. 'optimal').  A solver exception
    # (timeout, numerical failure) is reported separately and isn't a bug.
    bug_cells = []
    solver_fail_cells = []
    for i, (a, b) in enumerate(zip(l1_pruned, l2_pruned)):
        if a and (not b):
            st = l2_status[i]
            if isinstance(st, str) and st.startswith('EXC:'):
                solver_fail_cells.append(i)
            elif st in ('optimal', 'optimal_inaccurate'):
                bug_cells.append(i)
            else:
                # 'unbounded', 'unknown', etc.  Treat as ambiguous-not-bug.
                solver_fail_cells.append(i)
    soundness_violations = len(bug_cells)
    n_l2_solver_fail = len(solver_fail_cells)

    avg_L1 = float(np.mean(l1_times)) if l1_times else 0.0
    avg_L2 = float(np.mean(l2_times)) if l2_times else 0.0
    med_L1 = float(np.median(l1_times)) if l1_times else 0.0
    med_L2 = float(np.median(l2_times)) if l2_times else 0.0
    max_L1 = float(np.max(l1_times)) if l1_times else 0.0
    max_L2 = float(np.max(l2_times)) if l2_times else 0.0

    print('\n' + '=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'F survivors:     {n_F_surv}')
    print(f'Q survivors:     {n_Q_surv}')
    print(f'Tested cells:    {n_cells}')
    print(f'L1 pruned:       {n_L1} / {n_cells}')
    print(f'L2 pruned:       {n_L2} / {n_cells}')
    print(f'L2 extra over L1: {extra_L2} cells')
    print(f'L1-only with L2 solver failure (EXC/unknown): {n_l2_solver_fail} cells')
    if n_l2_solver_fail:
        print(f'    solver-failed cells (idx): {solver_fail_cells}')
    print(f'L1-only with L2 status optimal (SOUNDNESS BUG): {soundness_violations}')
    if soundness_violations:
        print(f'  *** BUG cells: {bug_cells}')
    print(f'L1 time avg/med/max: {avg_L1*1000:.1f} / {med_L1*1000:.1f} / {max_L1*1000:.1f} ms')
    print(f'L2 time avg/med/max: {avg_L2*1000:.1f} / {med_L2*1000:.1f} / {max_L2*1000:.1f} ms')
    print(f'L1 total wall:    {t_L1:.2f}s')
    print(f'L2 total wall:    {t_L2:.2f}s')
    print(f'Total wall:       {t_FQ + t_L1 + t_L2:.2f}s')

    # 6) Verdict line.
    if soundness_violations > 0:
        verdict = (f'SOUNDNESS_BUG: {soundness_violations} cells L1-pruned '
                   f'but NOT L2-pruned with L2 status optimal '
                   f'(impossible if Lasserre order-2 is a tightening) -- '
                   f'INVESTIGATE')
    elif extra_L2 > 0:
        pct = 100.0 * extra_L2 / n_cells
        verdict = (f'ORDER2_TIGHTER: kills {extra_L2} extra of {n_cells} cells '
                   f'({pct:.1f}%); avg time order1={avg_L1*1000:.0f} ms, '
                   f'order2={avg_L2*1000:.0f} ms')
    else:
        verdict = (f'ORDER2_NO_GAIN: 0 extra prunes over L1 (L1 pruned {n_L1}, '
                   f'L2 pruned {n_L2}, plus {n_l2_solver_fail} L2 solver-fails); '
                   f'avg time order1={avg_L1*1000:.0f} ms, '
                   f'order2={avg_L2*1000:.0f} ms (~{avg_L2/max(avg_L1,1e-9):.1f}x slower)')
    print('\n' + verdict)

    # 7) Save JSON.
    out = {
        'config': {
            'n_half': N_HALF, 'm': M, 'c_target': C_TARGET, 'd': D,
            'max_l_cells': MAX_L_CELLS, 'solver': solver,
            'eps_margin': EPS_MARGIN, 'tol': TOL,
        },
        'pipeline': {
            'n_processed_palindromic': n_processed,
            'F_survivors': n_F_surv,
            'Q_survivors': n_Q_surv,
            't_F_s': t_F, 't_Q_s': t_Q, 't_F_plus_Q_s': t_FQ,
        },
        'tested_cells': n_cells,
        'L1': {
            'pruned': n_L1,
            'pruned_idx': [i for i, p in enumerate(l1_pruned) if p],
            'status_counts': _count_statuses(l1_status),
            'time_avg_ms': avg_L1 * 1000,
            'time_med_ms': med_L1 * 1000,
            'time_max_ms': max_L1 * 1000,
            'time_total_s': t_L1,
        },
        'L2': {
            'pruned': n_L2,
            'pruned_idx': [i for i, p in enumerate(l2_pruned) if p],
            'status_counts': _count_statuses(l2_status),
            'time_avg_ms': avg_L2 * 1000,
            'time_med_ms': med_L2 * 1000,
            'time_max_ms': max_L2 * 1000,
            'time_total_s': t_L2,
        },
        'comparison': {
            'L2_extra_over_L1': extra_L2,
            'L1_only_soundness_violations': soundness_violations,
            'soundness_violation_idx': bug_cells,
            'L2_solver_failures_on_L1_pruned': n_l2_solver_fail,
            'L2_solver_failure_idx': solver_fail_cells,
            'L2_retry_results': retry_results,
        },
        'verdict': verdict,
        'tested_cell_compositions': [c.tolist() for c in cells],
    }
    with open(OUT_PATH, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nWrote {OUT_PATH}')


def _count_statuses(status_list):
    out = {}
    for s in status_list:
        out[s] = out.get(s, 0) + 1
    return out


if __name__ == '__main__':
    main()
