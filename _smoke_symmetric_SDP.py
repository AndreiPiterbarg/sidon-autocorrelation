"""Smoke test: symmetry-reduced SDP vs standard L-prune SDP for palindromic c.

PREMISE
=======
Cascade enumerates palindromic compositions: c[i] = c[d-1-i].
Cell { x : lo <= x <= hi, sum x = 4nm } with lo = max(0,c-1), hi = c+1 is
J-invariant (J = reversal matrix), since lo and hi are palindromic.

The window matrices A_W are NOT individually J-invariant (A_W[i,j] = 1 iff
s_lo <= i+j <= s_hi -- a function of i+j, not symmetric under (i,j) ->
(d-1-i, d-1-j) which sends i+j -> 2(d-1)-(i+j)).  HOWEVER, the SET of
windows is closed under J-pairing: window (ell, s_lo) pairs with
(ell, 2(d-1)-(s_lo+ell-2)).  Both have the SAME threshold (4n*ell*c*m^2).

SYMMETRY THEOREM
================
If X is SDP-feasible (PSD-lifted moment matrix), then X_sym := (X + JXJ)/2
is also feasible (J-paired window thresholds are equal, and PSD/RLT cones
are J-invariant for palindromic lo, hi).  Hence we can WLOG restrict to
J-symmetric X without losing anything.

CONSEQUENCE
===========
The symmetry-reduced SDP gives the SAME ANSWER (feasible/infeasible) as the
original.  Symmetry helps only with COST (smaller PSD block, faster solve).

This smoke test confirms experimentally:
  1. symmetric-restricted SDP and unrestricted SDP have IDENTICAL prune
     decisions on a sample of palindromic Q-survivors.
  2. Time savings are measured.

VERDICT
=======
Since the SDP IS already auto-symmetric (any optimal X can be replaced by
its J-symmetric average), restricting X to be J-symmetric gives NO EXTRA
PRUNING.  Wins are O(1) speed factor only (~2x via smaller block).

Implementation: encode X = JXJ as d(d+1)/2 - <off-diagonal mirrored> linear
equalities X[i,j] = X[d-1-i, d-1-j].  This is the cheapest way to test the
hypothesis.

Run with:
    python _smoke_symmetric_SDP.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, _detect_solver, _make_cell, prune_L_one


def _shor_feasibility_symmetric(c_int, lo, hi, A_mats, windows, n_half, m,
                                 c_target, solver='MOSEK', tol=1e-9,
                                 eps_margin=1e-9):
    """Same as L's Shor SDP, but with the additional constraint X = J X J,
    where J is the reversal matrix.

    Equivalent to enforcing X[i,j] = X[d-1-i, d-1-j] for all i,j.

    Returns (pruned_bool, status_str).
    """
    import cvxpy as cp

    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)

    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])

    cons = [Y >> 0]
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

    # SYMMETRY: x_i = x_{d-1-i}, X[i,j] = X[d-1-i, d-1-j]
    for i in range(d // 2):
        cons += [x[i] == x[d - 1 - i]]
    for i in range(d):
        for j in range(d):
            ip, jp = d - 1 - i, d - 1 - j
            if (ip, jp) != (i, j) and (i < ip or (i == ip and j < jp)):
                cons += [X[i, j] == X[ip, jp]]

    # Diagonal McCormick
    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]

    # RLT off-diagonal
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    # Window constraints
    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        if solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=False,
                        mosek_params={
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                        })
        elif solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=False,
                        eps_abs=tol, eps_rel=tol, max_iter=300)
        elif solver == 'SCS':
            prob.solve(solver='SCS', verbose=False, eps=tol, max_iters=20000)
        else:
            return False, 'NO_SOLVER'
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'

    return prob.status == 'infeasible', prob.status


def main():
    print('=' * 70)
    print('SMOKE: symmetric SDP vs standard L SDP at d=8 (n_half=4, m=10, c=1.28)')
    print('=' * 70)

    n_half, m, c_target = 4, 10, 1.28
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m

    solver = _detect_solver('MOSEK')
    print(f'Solver: {solver}')
    print(f'd={d}, m={m}, c_target={c_target}')

    # Build windows / A matrices
    windows, ell_int_sums = _build_windows(d)
    A_mats = _build_A_matrices(d, windows)
    sigmas = _enum_balanced_signs(d)
    print(f'n_windows = {len(windows)}, n_sigmas = {len(sigmas)}')

    # ----------------------------------------------------------------
    # Verify J-pairing of windows
    # ----------------------------------------------------------------
    win_set = {tuple(w) for w in windows}
    n_paired = 0
    for w in windows:
        ell, s_lo = w
        s_hi = s_lo + ell - 2
        pair = (ell, 2 * (d - 1) - s_hi)
        if pair in win_set:
            n_paired += 1
    print(f'J-pairing check: {n_paired}/{len(windows)} windows have a J-pair '
          f'-> {"OK" if n_paired == len(windows) else "FAIL"}')

    # ----------------------------------------------------------------
    # Find Q-survivors (palindromic compositions surviving Q at given c_target)
    # ----------------------------------------------------------------
    print('\nFinding palindromic Q-survivors...')
    q_survivors = []
    n_processed = 0
    n_F_surv = 0
    t0 = time.time()
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        n_F_surv += int(np.sum(sF))
        for idx in np.where(sF)[0]:
            c_int = batch[idx]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target, margin=1e-9):
                q_survivors.append(c_int.copy())
    print(f'  processed={n_processed:,}, F-surv={n_F_surv:,}, '
          f'Q-surv={len(q_survivors):,}, t={time.time()-t0:.1f}s')

    if len(q_survivors) == 0:
        print('No Q-survivors -> nothing to compare.  Trying lower c_target=1.20')
        return

    # Restrict to first ~30 to keep wall < 10 min
    sample = q_survivors[:30]
    print(f'\nComparing standard SDP vs symmetric SDP on {len(sample)} Q-survivors:')

    n_match = 0
    n_disagree = 0
    n_std_pruned = 0
    n_sym_pruned = 0
    t_std_total = 0.0
    t_sym_total = 0.0
    disagreements = []

    for k, c_int in enumerate(sample):
        # Standard L
        t0 = time.time()
        std_pruned, std_status = prune_L_one(c_int, A_mats, windows, n_half, m,
                                                c_target, solver=solver, order=1)
        t_std = time.time() - t0
        t_std_total += t_std

        # Symmetric L
        lo, hi = _make_cell(c_int, m)
        t0 = time.time()
        sym_pruned, sym_status = _shor_feasibility_symmetric(
            c_int, lo, hi, A_mats, windows, n_half, m, c_target, solver=solver)
        t_sym = time.time() - t0
        t_sym_total += t_sym

        n_std_pruned += int(std_pruned)
        n_sym_pruned += int(sym_pruned)

        if std_pruned == sym_pruned:
            n_match += 1
        else:
            n_disagree += 1
            disagreements.append({
                'c': c_int.tolist(),
                'std': (std_pruned, std_status),
                'sym': (sym_pruned, sym_status),
            })
            if len(disagreements) <= 3:
                print(f'  [{k}] DISAGREE: std=({std_pruned},{std_status}) '
                      f'sym=({sym_pruned},{sym_status}) c={c_int.tolist()}')

    print(f'\n--- RESULTS (n={len(sample)}) ---')
    print(f'  Standard L pruned:  {n_std_pruned}/{len(sample)}')
    print(f'  Symmetric L pruned: {n_sym_pruned}/{len(sample)}')
    print(f'  Match:    {n_match}/{len(sample)}')
    print(f'  Disagree: {n_disagree}/{len(sample)}')
    print(f'  Time std: {t_std_total:.2f}s  ({t_std_total/len(sample)*1000:.1f} ms/cell)')
    print(f'  Time sym: {t_sym_total:.2f}s  ({t_sym_total/len(sample)*1000:.1f} ms/cell)')
    print(f'  Speedup:  {t_std_total / max(1e-6, t_sym_total):.2f}x')

    if n_disagree == 0 and n_std_pruned == n_sym_pruned:
        print('\n*** Symmetric SDP gives IDENTICAL prune count to standard SDP. ***')
        print('  -> Symmetry restriction is "free" -- standard SDP\'s solution is')
        print('     auto-symmetrizable (J X J is also feasible).')
        print('  -> Cost-only optimization, NOT a stronger relaxation.')
    else:
        print('\nUnexpected disagreement -- investigate.')

    print('\n=' * 35)
    if n_sym_pruned > n_std_pruned:
        print('VERDICT: SYMMETRY_PRUNES_X_EXTRA')
    else:
        print('VERDICT: NO_GAIN_SOLUTION_ALREADY_SYMMETRIC')


if __name__ == '__main__':
    main()
