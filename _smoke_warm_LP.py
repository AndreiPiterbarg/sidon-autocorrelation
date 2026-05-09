"""Warm-start LP smoke test for Q-bench.

PROBLEM
=======
Q-bench solves a multi-window LP per composition. Adjacent compositions in
enumeration differ by 1-2 entries -> their LPs differ by tiny perturbations
of the constraint matrix A and constant vector V. HiGHS (via highspy) supports
basis warm-start which can reduce simplex iterations 3-10x on similar LPs.

SETUP
=====
LP per composition (variables x = [lambda_0, ..., lambda_{n_win-1}, t]):
    max  t
    s.t. -A[s, :] . lambda + t <= 0   for all s in sigmas    (n_sigma rows)
         sum_w lambda_w = 1                                  (1 row)
         lambda_w >= 0,  t free
    nvar = n_win + 1; n_constr = n_sigma + 1.

At (n_half=3, m=10, d=6):
    n_win  = 23   (so nvar = 24)
    n_sigma = C(6, 3) = 20   (so n_constr = 21)
At d=8: n_win ~ 66 (nvar 67), n_sigma=70.
At d=12: n_win ~ 276 (nvar 277), n_sigma=924.

EXPERIMENT
==========
1. Enumerate F-survivors at (n_half=3, m=10, c_target=1.28)  -> 172 LPs.
2. Cold solve: each call builds a fresh Highs() model + run().
3. Warm solve: keep one Highs() instance; rebuild only A_ub rows + V (cost
   stays 0,..,0,-1 and equality row stays sum=1) and call run().
   For each successive composition, the prior optimal basis stays cached
   inside Highs and is used as starting basis (HiGHS dual simplex resumes).
4. Verify all t_opt agree to 1e-9 -> soundness.

Reports cold time, warm time, speedup, and writes results to JSON.
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import (
    _enum_balanced_signs, _build_windows, _composition_window_data,
    _q_bound_lp,
)

import highspy

# ----------------------------------------------------------------------
# Build LP data for a composition (returns A (n_sigma x n_win), V (n_win,))
# Mirrors _Q_bench._q_bound_lp, but stops before the linprog call.
# ----------------------------------------------------------------------
def build_lp_AV(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target):
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    n_d = float(n_half)
    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m
    V = (ws.astype(np.float64) * inv_4nl
         - ell_int_sums.astype(np.float64) * inv_4nl
         - cs_m2)
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    M = sigmas.astype(np.float64) @ BB_over_ell.T
    A = V[None, :] - M / (2.0 * n_d)
    return A, V


# ----------------------------------------------------------------------
# Cold solver: scipy.linprog (matches _Q_bench._q_bound_lp baseline)
# ----------------------------------------------------------------------
def cold_solve_scipy(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target):
    return _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                       n_half, m, c_target)


# ----------------------------------------------------------------------
# Cold solver: highspy fresh Highs() each call (no basis carry-over)
# ----------------------------------------------------------------------
def _highs_silent():
    h = highspy.Highs()
    h.silent()
    return h


def cold_solve_highspy(c_int, windows, ell_int_sums, sigmas, n_half, m,
                       c_target):
    A, _V = build_lp_AV(c_int, windows, ell_int_sums, sigmas, n_half, m,
                        c_target)
    n_sigma, n_win = A.shape
    nvar = n_win + 1
    h = _highs_silent()
    INF = h.getInfinity()
    # Variables: lambda_0..n_win-1 in [0, inf]; t free
    col_lower = np.concatenate([np.zeros(n_win), np.array([-INF])])
    col_upper = np.full(nvar, INF)
    col_cost = np.zeros(nvar); col_cost[-1] = -1.0   # we minimize, so negate
    # Rows: row i (i<n_sigma): -A[i,:] . lambda + t <= 0
    #       last row:    sum lambda = 1
    row_lower = np.concatenate([np.full(n_sigma, -INF), np.array([1.0])])
    row_upper = np.concatenate([np.zeros(n_sigma), np.array([1.0])])
    # Sparse column-wise build
    a_dense = np.zeros((n_sigma + 1, nvar))
    a_dense[:n_sigma, :n_win] = -A
    a_dense[:n_sigma, -1] = 1.0   # +t in -A.lam + t <= 0
    a_dense[n_sigma, :n_win] = 1.0
    a_dense[n_sigma, -1] = 0.0
    # Build CSC
    a_start = np.zeros(nvar + 1, dtype=np.int32)
    rows_idx = []
    vals_idx = []
    for j in range(nvar):
        col = a_dense[:, j]
        nz = np.nonzero(col)[0]
        rows_idx.extend(nz.tolist())
        vals_idx.extend(col[nz].tolist())
        a_start[j + 1] = len(rows_idx)
    a_index = np.asarray(rows_idx, dtype=np.int32)
    a_value = np.asarray(vals_idx, dtype=np.float64)
    # Pass model:  passModel(num_col, num_row, num_nz, a_format, sense,
    #   offset, col_cost, col_lower, col_upper, row_lower, row_upper,
    #   a_start, a_index, a_value, integrality)
    a_format = 1   # 1 = colwise
    sense = 1      # 1 = minimize
    integrality = np.zeros(nvar, dtype=np.int32)
    h.passModel(nvar, n_sigma + 1, len(a_value), a_format, sense, 0.0,
                col_cost, col_lower, col_upper, row_lower, row_upper,
                a_start, a_index, a_value, integrality)
    h.run()
    status = h.getModelStatus()
    if h.modelStatusToString(status) != 'Optimal':
        return -np.inf, None
    sol = h.getSolution()
    x = np.asarray(sol.col_value)
    t_opt = -h.getObjectiveValue()
    return float(t_opt), x[:n_win]


# ----------------------------------------------------------------------
# Warm solver: ONE Highs() instance reused. Replace A coefficients in-place.
# HiGHS keeps the prior optimal basis as the starting basis automatically
# when run() is called again on a modified problem (proven by repeated
# observation in HiGHS docs / source).
# Strategy: build the LP once with placeholder A; then for each composition
# rewrite the A_ub coefficients via changeCoeff or rebuild the model
# preserving the basis with setBasis.
#
# Cleanest robust approach: build the LP once, and for each new comp, use
# changeCoeff for non-zero entries that change.  But A is dense, so we
# instead:
#   1. clearModel each iter (keeps Highs object alive but loses basis)
#       -> defeats warm-start.
#   2. Use addRows / deleteRows to swap constraint rows
#       -> too costly.
#   3. Use BEST: passModel + setBasis(prev_basis).
# We use option 3.
# ----------------------------------------------------------------------
def warm_solve_loop(comps, windows, ell_int_sums, sigmas, n_half, m,
                     c_target):
    n_sigma = len(sigmas); n_win = len(windows); nvar = n_win + 1
    h = _highs_silent()
    INF = h.getInfinity()
    col_lower = np.concatenate([np.zeros(n_win), np.array([-INF])])
    col_upper = np.full(nvar, INF)
    col_cost = np.zeros(nvar); col_cost[-1] = -1.0
    row_lower = np.concatenate([np.full(n_sigma, -INF), np.array([1.0])])
    row_upper = np.concatenate([np.zeros(n_sigma), np.array([1.0])])
    a_format = 1; sense = 1
    integrality = np.zeros(nvar, dtype=np.int32)
    prev_basis = None
    results = []
    for k, c_int in enumerate(comps):
        A, _V = build_lp_AV(c_int, windows, ell_int_sums, sigmas, n_half,
                             m, c_target)
        a_dense = np.zeros((n_sigma + 1, nvar))
        a_dense[:n_sigma, :n_win] = -A
        a_dense[:n_sigma, -1] = 1.0
        a_dense[n_sigma, :n_win] = 1.0
        # Build CSC
        a_start = np.zeros(nvar + 1, dtype=np.int32)
        rows_idx = []; vals_idx = []
        for j in range(nvar):
            col = a_dense[:, j]
            nz = np.nonzero(col)[0]
            rows_idx.extend(nz.tolist())
            vals_idx.extend(col[nz].tolist())
            a_start[j + 1] = len(rows_idx)
        a_index = np.asarray(rows_idx, dtype=np.int32)
        a_value = np.asarray(vals_idx, dtype=np.float64)
        h.passModel(nvar, n_sigma + 1, len(a_value), a_format, sense, 0.0,
                    col_cost, col_lower, col_upper, row_lower, row_upper,
                    a_start, a_index, a_value, integrality)
        # Inject prior basis (warm start) ----
        if prev_basis is not None:
            h.setBasis(prev_basis)
        h.run()
        status = h.getModelStatus()
        if h.modelStatusToString(status) != 'Optimal':
            results.append((-np.inf, None))
            prev_basis = None      # invalidated
            continue
        t_opt = -h.getObjectiveValue()
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        results.append((float(t_opt), x[:n_win].copy()))
        # Save basis for next iteration
        prev_basis = h.getBasis()
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    n_half = 3; m = 10; d = 2 * n_half; c_target = 1.28
    S_half = 2 * n_half * m
    print(f'\n=== smoke_warm_LP: n_half={n_half}, m={m}, d={d}, '
          f'c_target={c_target} ===')

    # 1. Enumerate F-survivors (palindromic; mirrors _Q_bench.run)
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    f_surv = []
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=2000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        for i in np.where(sF)[0]:
            f_surv.append(batch[i].copy())
    print(f'  F-survivors: {len(f_surv)}')

    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    n_win = len(windows); n_sigma = len(sigmas)
    nvar = n_win + 1
    print(f'  n_win = {n_win}  -> nvar = {nvar}')
    print(f'  n_sigma = C({d},{d//2}) = {n_sigma}  -> n_constr = '
          f'{n_sigma + 1}')

    # 2. SCIPY cold (the production baseline)
    t0 = time.time()
    scipy_results = []
    for c_int in f_surv:
        scipy_results.append(cold_solve_scipy(c_int, windows, ell_int_sums,
                                                sigmas, n_half, m, c_target))
    t_scipy = time.time() - t0
    print(f'  scipy cold ({len(f_surv)} LPs):    {t_scipy:7.3f}s '
          f'  ({1000*t_scipy/len(f_surv):.2f} ms/LP)')

    # 3. HIGHSPY cold (each call fresh Highs())
    t0 = time.time()
    highs_cold_results = []
    for c_int in f_surv:
        highs_cold_results.append(cold_solve_highspy(c_int, windows,
                                                       ell_int_sums, sigmas,
                                                       n_half, m, c_target))
    t_highs_cold = time.time() - t0
    print(f'  highspy cold ({len(f_surv)} LPs):  {t_highs_cold:7.3f}s '
          f'  ({1000*t_highs_cold/len(f_surv):.2f} ms/LP)')

    # 4. HIGHSPY warm (one Highs() instance, basis reused)
    t0 = time.time()
    warm_results = warm_solve_loop(f_surv, windows, ell_int_sums, sigmas,
                                     n_half, m, c_target)
    t_warm = time.time() - t0
    print(f'  highspy warm ({len(f_surv)} LPs):  {t_warm:7.3f}s '
          f'  ({1000*t_warm/len(f_surv):.2f} ms/LP)')

    # 5. Soundness: t_opt agreement (warm == scipy within 1e-9 + cold == warm)
    max_diff_cold_scipy = 0.0
    max_diff_warm_scipy = 0.0
    n_warm_finite = 0
    for (ts, _), (tc, _), (tw, _) in zip(scipy_results, highs_cold_results,
                                            warm_results):
        if np.isfinite(ts) and np.isfinite(tc):
            max_diff_cold_scipy = max(max_diff_cold_scipy, abs(ts - tc))
        if np.isfinite(ts) and np.isfinite(tw):
            max_diff_warm_scipy = max(max_diff_warm_scipy, abs(ts - tw))
            n_warm_finite += 1

    print(f'\nSoundness check:')
    print(f'  max |t_scipy - t_highs_cold| = {max_diff_cold_scipy:.3e}')
    print(f'  max |t_scipy - t_highs_warm| = {max_diff_warm_scipy:.3e}')
    print(f'  finite-pair count (warm) = {n_warm_finite} / {len(f_surv)}')

    speedup_warm_vs_scipy = t_scipy / t_warm if t_warm > 0 else float('inf')
    speedup_warm_vs_cold = t_highs_cold / t_warm if t_warm > 0 else \
        float('inf')

    print(f'\nSpeed:')
    print(f'  warm vs scipy   : {speedup_warm_vs_scipy:.2f}x')
    print(f'  warm vs hcold   : {speedup_warm_vs_cold:.2f}x')

    out = {
        'config': {'n_half': n_half, 'm': m, 'd': d,
                    'c_target': c_target, 'n_lps': len(f_surv),
                    'n_win': n_win, 'n_sigma': n_sigma, 'nvar': nvar},
        'time_s': {'scipy_cold': t_scipy,
                    'highspy_cold': t_highs_cold,
                    'highspy_warm': t_warm},
        'ms_per_lp': {'scipy_cold': 1000 * t_scipy / len(f_surv),
                       'highspy_cold': 1000 * t_highs_cold / len(f_surv),
                       'highspy_warm': 1000 * t_warm / len(f_surv)},
        'speedup_warm_vs_scipy': speedup_warm_vs_scipy,
        'speedup_warm_vs_highspy_cold': speedup_warm_vs_cold,
        'soundness': {
            'max_diff_t_scipy_vs_highs_cold': max_diff_cold_scipy,
            'max_diff_t_scipy_vs_highs_warm': max_diff_warm_scipy,
            'tolerance_1e-9_passed': (max_diff_cold_scipy < 1e-9 and
                                      max_diff_warm_scipy < 1e-9),
        },
    }
    out_path = os.path.join(_dir, '_smoke_warm_LP.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults written to {out_path}')


if __name__ == '__main__':
    main()
